#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>  // host-side printf of the compile-gated sizing dump
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
// Merger GBL refit inputs:
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BLMaterialMapCollection.h"
#include "RecoTracker/Record/interface/BLMaterialMapRecord.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BLBFieldMapCollection.h"
#include "RecoTracker/Record/interface/BLBFieldMapRecord.h"
#include "RecoTracker/Record/interface/StackedModuleGeometryRecord.h"
#include "RecoTracker/Record/interface/CAGeometryRecord.h"
// Merger-side OT-hit extension inputs. The attach reuses the CA extender geometry (built the same
// way from the tracker ES records) + the OT stub collection, so it can scan the full attachable layer
// set (incl. the odd OT disks the prompt doublet graph excludes) over the merged collection.
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometryHost.h"
#include "HeterogeneousCore/AlpakaCore/interface/MoveToDeviceCache.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include <mutex>

#include "CAHitNtupletGenerator.h"
#include "ExtDerivedTables.h"
#include "ExtenderGeometry.h"
#include "ExtenderGeometryBuild.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

#ifdef CA_SIZING_DUMP
  namespace {
    // Per-event demand dump of a merged track collection, in the same line form the CA producer emits
    // ("[CA Sizing] iter=<stage> <key>=<value> ..."), so one parser reads every stage. The demand is the
    // device-side merged track count; the capacities are the track and hit blocks the whole merger chain
    // was sized from. One 4-byte D2H and one host wait per event, compiled in only under the toggle.
    void dumpMergerSizing(Queue& queue, reco::TracksSoACollection& tracks, const char* stage) {
      const uint32_t capTuples = uint32_t(tracks.view().tracks().metadata().size());
      const uint32_t capHitCont = uint32_t(tracks.view().trackHits().metadata().size());
      auto nTracksHost = cms::alpakatools::make_host_buffer<int32_t>(queue);
      auto nTracksDev = cms::alpakatools::make_device_view(queue, tracks.view().tracks().nTracks());
      alpaka::memcpy(queue, nTracksHost, nTracksDev);
      alpaka::wait(queue);
      printf("[CA Sizing] iter=%s nTracks=%u capTuples=%u capHitCont=%u\n",
             stage,
             uint32_t(*nTracksHost.data()),
             capTuples,
             capHitCont);
    }
  }  // namespace
#endif

  class PixelTracksSoAMerger : public global::EDProducer<> {
    using Algo = CAHitMaskingAndMerger;

  public:
    explicit PixelTracksSoAMerger(const edm::ParameterSet& iConfig);
    ~PixelTracksSoAMerger() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void produce(edm::StreamID streamID, device::Event& iEvent, const device::EventSetup& iSetup) const override;

    // The merger addresses the shared CA geometry's layers block with Phase2OTStubs layer numbers, while
    // the OT phi binner is built with ::pixelTopology::Phase2OTStubs::numberOfLayers histograms. If the
    // ESProducer is configured for a different layer count the two disagree silently, giving a phi binner
    // whose histogram count differs from the layer index range the walk uses and an out-of-range read of
    // the layers block. CAGeometryESProducer leaves this check to its consumers.
    static reco::CAGeometrySoACollection const& checkedGeometry(reco::CAGeometrySoACollection const& geometry) {
      const int nLayersES = int(geometry.view().layers().metadata().size()) - 1;
      constexpr int nLayersExpected = int(::pixelTopology::Phase2OTStubs::numberOfLayers);
      if (nLayersES != nLayersExpected)
        throw cms::Exception("CAGeometryMismatch")
            << "PixelTracksSoAMerger: the shared CA geometry (CAGeometryRecord) describes " << nLayersES
            << " CA layers but the merger's attach is built for " << nLayersExpected
            << " (pixelTopology::Phase2OTStubs::numberOfLayers). The ESProducer's `nLayers` must match.";
      return geometry;
    }

    pixelTrack::Quality const minQuality_;
    double const matchFraction_;

    // Strict cross-arm twin merge: pair opposite-arm twins by trajectory + shared-hit evidence and
    // unite their hit lists onto the winner track. Always on.
    double const twinMergeDeltaEta_;
    double const twinMergeDeltaPhi_;
    int const twinMergeMinSharedHits_;
    // The looser tier-2 merge (always on) also pairs twins that fail the shared-hit requirement but sit
    // inside the tighter trajectory windows below. Strict tier-1 pairing is unchanged by it.
    double const twinMergeTier2DeltaEta_;
    double const twinMergeTier2DeltaPhi_;
    // Covariance-scaled arm-invariant twin compatibility gate (phi/1pT/cotTheta:
    // dp^2 > nSigma2*(cov_i+cov_j) rejects the pair); <=0 disables it.
    double const twinMergeNSigma2_;
    // A twin winner that absorbs its sibling's hits keeps its pre-union state, so the united winners are
    // always refit: ndof is recomputed to the united fit-hit count and the merger-side GBL refit
    // overwrites state, cov, chi2 and ndof.
    // When true the merger sets AttachParams::verbose, so the per-event [CAExtension] summary prints
    // (host-quality preGateSkipped and per-layer-class walk-committed extras counters included).
    // Diagnostic only; never on in production.
    bool const mergerExtendVerbose_;
    // Host-quality pre-gate: a track is offered to the attach walk only if its fit chi2/ndof is below
    // this (the companion nHits and pt predicates sit at their off sentinels).
    double const extHostMaxChi2Ndof_;
    // The derived-selection package. Its measured input rows are compiled-in constants in
    // ExtDerivedTables.h, not configuration; this epsilon is the package's only operating point.
    double const extDerivedEps_;
    double const extFmsBarrel_;
    double const extFmsEndcap_;
    // Attach recall/calibration knobs; see caExtension::AttachParams.
    double const extRecallReachRelax_;
    int const extRecallPixelFirstBudget_;
    double const extPixelGateChi2Cut_;
    // Runtime walk visit budget K, a loop bound only; buffer sizing stays at the compile-time
    // maxWalkLayers. The continuation and cap behaviour it works against is fixed in the eParams block.
    int const extMaxWalkLayers_;
    // The |eta| floor of the far-first disc ordering. The visit order is otherwise nearest-disc-first,
    // which on a forward pixel-only host spends the visit and slot budgets on interior holes instead of
    // the outermost disc crossing, the only content that lengthens the transverse lever arm.
    double const extAttachFarMinAbsEta_;
    // Its window-ambiguity condition: the far crossing commits only where the candidate set that cleared
    // its gate is small enough for the argmin to be a measurement rather than a choice among competitors.
    int const extAttachFarMaxWin_;
    int const extMaxSharedOwners_;
    // Ceiling on the candidate capacity the attach scratch is sized to. The attach is handed the merged
    // track capacity as its host-known candidate bound; this parameter lets a measured maximum take over
    // the sizing of the candidate scratch, the refit scaffold arrays and the per-candidate launch grids.
    unsigned int const extRefitMaxCandidates_;
    // The attach thresholds the walk gates on:
    //   extPreGateMaxChi2_  -> AttachParams::preGateMaxChi2, the base pre-gate reduced-chi2 cut.
    //   extChi2Cut_ / extEndcapChi2Cut_ -> AttachParams::chi2Cut / endcapChi2Cut, the attach-window
    //     base cuts in the barrel and the endcap.
    //   extDispGateSig2_    -> AttachParams::extDispGateSig2, the (|d0|/sigma_d0)^2 threshold above
    //     which a host is treated as displaced.
    double const extPreGateMaxChi2_;
    // Attach pre-gate |eta| ceiling (AttachParams::maxAbsEta): tracks beyond it are not offered to the walk.
    double const extMaxAbsEta_;
    double const extChi2Cut_;
    double const extEndcapChi2Cut_;
    double const extDispGateSig2_;
    // per-input-collection arm label: 0 = prompt-side, 1 = displaced-side. Configured explicitly
    // ("inputArms", one entry per entry of "inputTkSoAs"), never inferred from the module labels.
    std::vector<int> armPerInput_;

    std::vector<device::EDGetToken<reco::TracksSoACollection>> inputTkSoATokenV_;
    std::vector<edm::InputTag> inputTkSoATagV_;

    // Actual-count sizing inputs. Each HP selector publishes, next to its device collection, two
    // plain host unsigned ints: the number of tracks and the number of hits it actually wrote. Their
    // tags are derived from the collection tags -- same module label and process, instance labels
    // "nTracks" / "nKeptHits" -- so no separate configuration can drift out of step with
    // inputTkSoAs. A producer that does not publish them (any non-HP-selector input) simply leaves
    // the products absent, and the sizing below falls back to the capacity sum.
    std::vector<edm::EDGetTokenT<uint32_t>> inputNTracksTokenV_;
    std::vector<edm::EDGetTokenT<uint32_t>> inputNKeptHitsTokenV_;
    // One-shot report of that fallback (produce() is const and concurrent).
    mutable std::once_flag capacitySizingWarnOnce_;

    const device::EDPutToken<reco::TracksSoACollection> outputTkSoAToken_;

    // Merger refit + attach inputs, mirroring the displaced CAHitNtupletAlpaka token set: the shared
    // rechit SoA the merged hit ids index, the raw OT rechit source, plus the BL material map, the
    // (Bz,Br) field map and the bfield ES. The CA-ordered module geometry comes from the EventSetup
    // (CAGeometryRecord) rather than a per-event product: it is the same geometry the CA arms build and
    // its blocks are pure per-run conditions.
    device::ESGetToken<reco::CAGeometrySoACollection, CAGeometryRecord> tokenCAGeom_;
    device::EDGetToken<reco::TrackingRecHitsSoACollection> tokenPixelRecHits_;
    device::EDGetToken<reco::OTRecHitsSoACollection> tokenOTHits_;
    device::ESGetToken<reco::StackedModuleGeometrySoACollection, StackedModuleGeometryRecord> tokenStackedGeomDev_;
    device::ESGetToken<BLMaterialMap, BLMaterialMapRecord> tokenBLMaterialMap_;
    device::ESGetToken<BLBFieldMap, BLBFieldMapRecord> tokenBLBFieldMap_;
    edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tokenField_;

    // Attach-only inputs: the OT stub collection feeds the full-hits OT source; the host tracker
    // geometry/topology + host stacked-sensor geometry feed the once-per-job build of the extender layer
    // surfaces (the same builder the CA runs in globalBeginRun). The device CA geometry / pixel rechits /
    // OT rechits / device stacked geometry / material map / bfield are shared with the refit path.
    device::EDGetToken<reco::StubsSoACollection> tokenOTStubs_;
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tokenTrackerGeom_;
    edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tokenTrackerTopo_;
    edm::ESGetToken<::reco::StackedModuleGeometryHost, StackedModuleGeometryRecord> tokenStackedGeomHost_;
    // Extender layer surfaces (per-CA-layer R/Z envelopes) lifted to device once, on the first event
    // that needs them. Mutable + once-guarded because produce() is const and may be called
    // concurrently by the global producer.
    using ExtLayersCache = cms::alpakatools::MoveToDeviceCache<Device, ::reco_extender::ExtenderGeometryHost>;
    mutable std::once_flag extLayersOnce_;
    mutable std::optional<ExtLayersCache> extLayersCache_;

    // The measured tables are job constants: every row is a compile-time constant and the only run-time
    // input, extDerivedEps_, is fixed at construction, so they are interpolated once per process and
    // uploaded once per device. Per device rather than per producer, because this is a global::EDProducer
    // whose produce() may run concurrently on every device the job owns. One flat allocation holds every
    // row back to back; the walk receives base and offset pointers.
    struct ExtTablesHost {
      std::vector<float> flat;
      int offQhat = -1, offEtaL = -1, offRho = -1, offDV = -1;  // the derived-selection rows
      int offEtaLRaw = -1, offRhoRaw = -1;                      // the per-source-round hole rows
      int offQhat3 = -1, offSigBExc = -1, offRho3 = -1;         // the bend package rows
    };
    // The per-device device copies. Built inside the once-init, which enumerates the platform's devices
    // exactly like CopyToDeviceCacheImpl and synchronises each copy before publishing the cache.
    class ExtTablesCache {
    public:
      explicit ExtTablesCache(ExtTablesHost h) : host_(std::move(h)) {
        if (host_.flat.empty())
          return;
        // Platform/Device/Queue are the accelerator namespace's own aliases (AlpakaInterface config.h).
        for (auto const& dev : cms::alpakatools::devices<Platform>()) {
          Queue q(dev);
          auto buf = cms::alpakatools::make_device_buffer<float[]>(q, host_.flat.size());
          auto hostBuf = cms::alpakatools::make_host_buffer<float[]>(q, host_.flat.size());
          std::copy(host_.flat.begin(), host_.flat.end(), hostBuf.data());
          alpaka::memcpy(q, buf, hostBuf);
          alpaka::wait(q);  // once per job per device: the staging buffer must outlive this copy
          devs_.emplace_back(dev);
          bufs_.emplace_back(std::move(buf));
        }
      }
      ExtTablesHost const& host() const { return host_; }
      // Base pointer of this queue's device copy. A linear scan over the platform devices, rather than
      // assuming the native handle is a dense index.
      const float* base(Device const& dev) const {
        for (std::size_t i = 0; i < devs_.size(); ++i)
          if (devs_[i] == dev)
            return alpaka::getPtrNative(bufs_[i]);  // const-correct accessor for a const Buf
        return nullptr;
      }

    private:
      ExtTablesHost host_;
      std::vector<Device> devs_;
      std::vector<cms::alpakatools::device_buffer<Device, float[]>> bufs_;
    };
    ExtTablesHost buildExtTables() const;  // validation + eps interpolation; pure in the cfi parameters
    mutable std::once_flag extTablesOnce_;
    mutable std::optional<ExtTablesCache> extTablesCache_;

    Algo deviceAlgo_;
  };

  PixelTracksSoAMerger::PixelTracksSoAMerger(const edm::ParameterSet& iConfig)
      : EDProducer(iConfig),
        minQuality_(pixelTrack::qualityByName(iConfig.getParameter<std::string>("minQuality"))),
        matchFraction_(iConfig.getParameter<double>("matchFraction")),
        twinMergeDeltaEta_(iConfig.getParameter<double>("twinMergeDeltaEta")),
        twinMergeDeltaPhi_(iConfig.getParameter<double>("twinMergeDeltaPhi")),
        twinMergeMinSharedHits_(iConfig.getParameter<int>("twinMergeMinSharedHits")),
        twinMergeTier2DeltaEta_(iConfig.getParameter<double>("twinMergeTier2DeltaEta")),
        twinMergeTier2DeltaPhi_(iConfig.getParameter<double>("twinMergeTier2DeltaPhi")),
        twinMergeNSigma2_(iConfig.getParameter<double>("twinMergeNSigma2")),
        mergerExtendVerbose_(iConfig.getParameter<bool>("mergerExtendVerbose")),
        extHostMaxChi2Ndof_(iConfig.getParameter<double>("extHostMaxChi2Ndof")),
        extDerivedEps_(iConfig.getParameter<double>("extDerivedEps")),
        extFmsBarrel_(iConfig.getParameter<double>("extFmsBarrel")),
        extFmsEndcap_(iConfig.getParameter<double>("extFmsEndcap")),
        extRecallReachRelax_(iConfig.getParameter<double>("extRecallReachRelax")),
        extRecallPixelFirstBudget_(iConfig.getParameter<int>("extRecallPixelFirstBudget")),
        extPixelGateChi2Cut_(iConfig.getParameter<double>("extPixelGateChi2Cut")),
        extMaxWalkLayers_(iConfig.getParameter<int>("extMaxWalkLayers")),
        extAttachFarMinAbsEta_(iConfig.getParameter<double>("extAttachFarMinAbsEta")),
        extAttachFarMaxWin_(iConfig.getParameter<int>("extAttachFarMaxWin")),
        extMaxSharedOwners_(iConfig.getParameter<int>("extMaxSharedOwners")),
        extRefitMaxCandidates_(iConfig.getParameter<unsigned int>("extRefitMaxCandidates")),
        extPreGateMaxChi2_(iConfig.getParameter<double>("extPreGateMaxChi2")),
        extMaxAbsEta_(iConfig.getParameter<double>("extMaxAbsEta")),
        extChi2Cut_(iConfig.getParameter<double>("extChi2Cut")),
        extEndcapChi2Cut_(iConfig.getParameter<double>("extEndcapChi2Cut")),
        extDispGateSig2_(iConfig.getParameter<double>("extDispGateSig2")),
        inputTkSoATagV_(iConfig.getParameter<std::vector<edm::InputTag>>("inputTkSoAs")),
        outputTkSoAToken_(produces()) {
    // The arm of each input collection is configured, one entry per input collection: 0 =
    // prompt-side, 1 = displaced-side. It decides which pairs of input tracks the cross-arm twin
    // merge may consider, so a wrong or missing entry silently changes the physics; both error
    // cases below are therefore fatal.
    {
      const auto arms = iConfig.getParameter<std::vector<unsigned int>>("inputArms");
      if (arms.size() != inputTkSoATagV_.size()) {
        throw cms::Exception("PixelTrackConfiguration")
            << "inputArms has " << arms.size() << " entries but inputTkSoAs has " << inputTkSoATagV_.size()
            << "; one arm (0 = prompt, 1 = displaced) must be given per input collection.";
      }
      for (std::size_t i = 0; i < arms.size(); ++i) {
        if (arms[i] > 1) {
          throw cms::Exception("PixelTrackConfiguration")
              << "inputArms[" << i << "] = " << arms[i] << " for input collection '" << inputTkSoATagV_[i].label()
              << "'; the only valid values are 0 (prompt-side) and 1 (displaced-side).";
        }
        armPerInput_.push_back(int(arms[i]));
      }
    }
    for (const auto& it : inputTkSoATagV_) {
      inputTkSoATokenV_.push_back(consumes(it));
      // The producing module's own count of what it wrote (see the member declarations): same module
      // label and process as the collection, fixed instance labels.
      inputNTracksTokenV_.push_back(consumes<uint32_t>(edm::InputTag(it.label(), "nTracks", it.process())));
      inputNKeptHitsTokenV_.push_back(consumes<uint32_t>(edm::InputTag(it.label(), "nKeptHits", it.process())));
    }
    {
      const int nDisp = std::count(armPerInput_.begin(), armPerInput_.end(), 1);
      if (nDisp == 0 || nDisp == int(armPerInput_.size())) {
        edm::LogWarning("PixelTracksSoAMerger")
            << "all input collections are on a single arm (nDisplaced=" << nDisp << "/" << armPerInput_.size()
            << "); no cross-arm twin pairs are possible.";
      }
    }
    // The merger-side twin refit and the merger attach share the same heavy device inputs
    // (CA geometry, pixel rechits, OT rechits, device stacked geometry, material map, bfield + the (Bz,Br)
    // BLBFieldMap the GBL curvature->pT conversion reads, so hltESPBLBFieldMap is a hard process
    // requirement), plus the attach-only OT stub collection and the host tracker geometry/topology +
    // host stacked geometry that feed the once-per-job extender-layer build.
    tokenCAGeom_ = esConsumes();
    tokenPixelRecHits_ = consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"));
    tokenOTHits_ = consumes(iConfig.getParameter<edm::InputTag>("otRecHitsSrc"));
    tokenStackedGeomDev_ = esConsumes();
    tokenBLMaterialMap_ = esConsumes();
    tokenField_ = esConsumes();
    tokenBLBFieldMap_ = esConsumes();
    tokenOTStubs_ = consumes(iConfig.getParameter<edm::InputTag>("otStubsSrc"));
    tokenTrackerGeom_ = esConsumes();
    tokenTrackerTopo_ = esConsumes();
    tokenStackedGeomHost_ = esConsumes();
    if (minQuality_ == pixelTrack::Quality::notQuality) {
      throw cms::Exception("PixelTrackConfiguration")
          << iConfig.getParameter<std::string>("minQuality") + " is not a pixelTrack::Quality";
    }
    if (minQuality_ < pixelTrack::Quality::dup) {
      throw cms::Exception("PixelTrackConfiguration")
          << iConfig.getParameter<std::string>("minQuality") + " not supported";
    }
  }

  void PixelTracksSoAMerger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    // Exactly the two arms the device-side merge supports (produce() throws above two): the prompt
    // and the displaced high-purity selectors, in the roles the deployed cfi gives them.
    desc.add<std::vector<edm::InputTag>>("inputTkSoAs",
                                         {edm::InputTag("hltPhase2PixelTrackTorchHighPuritySelector"),
                                          edm::InputTag("hltPhase2PixelTrackHighPuritySelectorDisplaced")});
    desc.add<std::vector<unsigned int>>("inputArms", {0, 1})
        ->setComment(
            "Arm of each entry of inputTkSoAs: 0 = prompt-side, 1 = displaced-side. Only pairs from "
            "different arms are considered by the cross-arm twin merge. One entry per input collection.");
    desc.add<std::string>("minQuality", "highPurity");
    desc.add<double>("matchFraction", 0.0);
    desc.add<double>("twinMergeDeltaEta", 0.03)->setComment("Max |dEta| between twin members.");
    desc.add<double>("twinMergeDeltaPhi", 0.03)->setComment("Max |dPhi| between twin members.");
    desc.add<int>("twinMergeMinSharedHits", 1)
        ->setComment("Minimum number of common hit ids required to qualify a twin pair.");

    desc.add<double>("twinMergeTier2DeltaEta", 0.01)
        ->setComment(
            "Max |dEta| for a tier-2 (shared-hit-less) twin pair. Set to the median separation of "
            "confirmed twins, so it admits the twin population without opening the window to neighbours.");
    desc.add<double>("twinMergeTier2DeltaPhi", 0.01)
        ->setComment("Max |dPhi| for a tier-2 (shared-hit-less) twin pair; same sizing as the dEta window.");

    desc.add<double>("twinMergeNSigma2", 8.0)
        ->setComment(
            "Covariance-scaled arm-invariant twin compatibility gate: reject a pair when any of phi, "
            "1/pT, cotTheta has dp^2 > nSigma2*(cov_i+cov_j) (tip/zip excluded: beamline-biased across "
            "arms for displaced tracks). <=0 disables the gate. It is a threshold ON the fit covariance, "
            "so it has to be re-derived whenever the fit's covariance convention changes.");

    // The merger-side attach runs one walk over the merged HP-selected collection using the ES FULL
    // geometry, gathering OT hits across the whole attachable layer set INCLUDING the odd OT disks the
    // prompt doublet graph excludes. It reuses the CA's caExtension::* attach entrypoints.
    desc.add<bool>("mergerExtendVerbose", false)
        ->setComment(
            "Diagnostic: print the per-event [CAExtension] attach summary (hosts skipped by the quality "
            "pre-gate, extras attached per layer class). Default false; enable only for debugging.");

    // Host-quality pre-gate: skip the OT attach walk on hosts the HP selector will reject anyway; they
    // are the stray-hit fake source and the bulk of the attach compute. Of the three sentinel-gated
    // predicates only the chi2/ndof one is active (nHits/pt sit at their off sentinels).
    desc.add<double>("extHostMaxChi2Ndof", 1.69)
        ->setComment(
            "mergerExtend host-quality pre-gate: skip attach on hosts with reduced chi2/ndof >= this "
            "(<=0 disables). tracks.chi2() is already the per-ndof GBL chi2, so this is compared to a "
            "chi2/ndof threshold directly. A threshold ON the fit covariance convention.");
    // The derived-selection package. One free number here; its measured input rows are compiled-in
    // constants in ExtDerivedTables.h, being detector measurements rather than choices.
    desc.add<double>("extDerivedEps", 0.55)
        ->setComment(
            "mergerExtend: THE single efficiency epsilon of the derived selection, spent once for the window, "
            "the gate, the rank and the hole prior. It is the probability mass of the correct-hit pull "
            "distribution the window is required to contain, so raising it widens window, gate and rank "
            "quantile together -- more attach yield at more fake exposure. It is the package's only "
            "continuous knob.");
    desc.add<double>("extFmsBarrel", 1.953)
        ->setComment(
            "mergerExtend: MEASURED material-dispersion scale of the MS variance in the BARREL -- the "
            "factor by which the material lattice plus the Highland formula under-state the real "
            "trajectory dispersion, measured on scattering-dominated triples.");
    desc.add<double>("extFmsEndcap", 1.506)
        ->setComment("mergerExtend: the same measured material-dispersion scale in the ENDCAP.");
    // Attach recall/calibration knobs.
    desc.add<double>("extRecallReachRelax", 2.5)
        ->setComment(
            "mergerExtend pixel/inward recall: extra reachability-envelope slack (cm) on PIXEL "
            "layers (CA L<28) for prefer-pixel (nPix<pixHitsTarget) hosts only, admitting in-road pixel layers "
            "that the strict 1.0 cm envelope rejects -- most of the missing displaced pixel content is "
            "already in-road and lost only to that envelope. 0 => identity.");
    desc.add<int>("extRecallPixelFirstBudget", 3)
        ->setComment(
            "mergerExtend pixel/inward recall: reserve up to N of the K=maxWalkLayers visit seats "
            "for pixel-first on prefer-pixel hosts by suppressing OT-disk forcing while a pixel layer is the "
            "nearest unvisited-reachable (recovers the reachable-but-skipped pixel layers). 0 reserves "
            "nothing. Sensible values are around pixHitsTarget (3).");
    desc.add<double>("extPixelGateChi2Cut", 3.0)
        ->setComment(
            "mergerExtend chi2-gate calibration on PIXEL layers (CA L<28), REPLACING the base cut there. "
            "A 2-dof gate at chi2 = 2.0 retains only 1 - exp(-1) = 63.2 % of correct hits by construction; "
            "95 % retention is -2 ln(0.05) = 5.99. OT layers keep chi2Cut/endcapChi2Cut. "
            "<=0 (sentinel) => identity.");
    // The attach thresholds. All of them are thresholds ON the fit's chi2, so they track the fit's
    // covariance convention and must be re-derived if it changes.
    desc.add<double>("extPreGateMaxChi2", 7.5)
        ->setComment(
            "mergerExtend attach pre-gate: a track is offered to the attach walk only if its fit "
            "chi2/ndof (tracks.chi2(), already per-ndof) is below this. Lowering it restricts the "
            "extension to better-measured tracks -- fewer wrong hits attached, fewer hits recovered. "
            "It is a threshold on the fit's chi2 and follows the fit's covariance convention.");
    desc.add<double>("extMaxAbsEta", 4.5)
        ->setComment(
            "mergerExtend attach pre-gate |eta| ceiling: a track whose fitted |cotTheta| exceeds sinh of "
            "this is not extended at all. It bounds both pre-gate passes and the host mask, and it is the "
            "top edge of the forward-eta TOB1-3 pocket band. Raising it buys forward hit content: beyond "
            "the outer-tracker edge the reachable targets are the pixel discs, which attach cleanly, while "
            "outer-tracker attach purity falls off above |eta| ~ 2.5 where the trajectory crosses the last "
            "disc near-tangentially and the true-hit residual spread grows to several cm. Above 2.5 the "
            "ceiling decouples from the duplicate-removal fallback's drop authority, which stays bounded "
            "at |eta| 2.5.");
    desc.add<double>("extChi2Cut", 1.0)
        ->setComment(
            "mergerExtend attach-window BARREL base chi2 cut (per-hit winner chi2 on OT and pixel barrel "
            "layers, before the per-class scales; pixel layers are overridden by extPixelGateChi2Cut when "
            "that is >0). Raising it widens the accept window everywhere in the barrel. It is a threshold "
            "on the fit's chi2 and its natural scale changes with the fit's covariance convention.");
    desc.add<double>("extEndcapChi2Cut", 1.0)
        ->setComment("mergerExtend attach-window ENDCAP base chi2 cut: the endcap counterpart of extChi2Cut.");
    desc.add<double>("extDispGateSig2", 200.0)
        ->setComment(
            "mergerExtend displacement-aware gate: the (|d0|/sigma_d0)^2 threshold above which a track "
            "counts as displaced and tightens its TOB1-3 accept window. The gate it belongs to is always "
            "on in this producer (it is not a cfi parameter). The value is a squared significance, so 200 means about "
            "14 "
            "sigma of displacement; lowering it applies the tighter window to more tracks.");
    desc.add<int>("extMaxWalkLayers", 6)
        ->setComment(
            "mergerExtend: runtime walk visit budget K, i.e. how many nearest reachable layers one track "
            "may visit. Buffer sizing and strides stay at the compile-time maxWalkLayers; only the walk "
            "loop bounds read this value, and it is clamped in-kernel to [1, kChainMaxVisits=8]. Each "
            "extra layer costs walk time in proportion and buys at most one more attached hit.");
    desc.add<double>("extAttachFarMinAbsEta", 2.8)
        ->setComment(
            "mergerExtend far-first disc ordering: the |eta| floor at which it takes effect. Below it the "
            "walk keeps its nearest-disc-first order. The floor is placed where the far content first "
            "becomes geometrically reachable: the fraction of recoverable forward pixel layers that pass "
            "the walk's own envelope and arc test is 0.002 / 0.446 / 0.798 / 0.986 at |eta| 2.4-2.6 / "
            "2.6-2.8 / 2.8-3.0 / 3.0-3.5.");
    desc.add<int>("extAttachFarMaxWin", 1)
        ->setComment(
            "mergerExtend far-first disc ordering: its window-ambiguity condition. On a far crossing (an "
            "endcap pixel disc beyond the track's own outermost |z|) the argmin winner is committed only "
            "if at most this many candidates cleared that crossing's gate; above it the walk declines the "
            "crossing, keeps the slot and carries on into the nearer discs. Candidate multiplicity is the "
            "quantity that best predicts whether the argmin is a measurement or a choice among "
            "competitors, so 1 is the pure end of the trade and larger values buy more far crossings at "
            "steadily lower purity. The far-first ordering it belongs to is always on in this producer (it "
            "is not a cfi parameter).");
    desc.add<int>("extMaxSharedOwners", 2)
        ->setComment(
            "mergerExtend: how many tracks may claim the same attached extra hit (top-N claim slots). "
            "1 makes attachment exclusive; larger values let genuine shared hits reach both tracks at the "
            "cost of correlating their fits.");
    desc.add<unsigned int>("extRefitMaxCandidates", 131072)
        ->setComment(
            "mergerExtend: ceiling on the candidate capacity the attach scratch is sized to. The attach "
            "is handed the merged track capacity as its host-known candidate bound (no candidate-count "
            "readback), and that bound sizes the candidate list and extras arrays, the stage-B refit "
            "scaffold arrays and every per-candidate launch grid. Only the tracks that clear the attach "
            "pre-gate are candidates, so the structural bound is several times the counts actually seen. "
            "The effective bound is min(track capacity, this), and the value should come from a measured "
            "per-event maximum of the candidate count with headroom -- below that maximum, candidates "
            "past the ceiling are dropped by the fill pass. The default is the compiled-in search-volume "
            "cap, i.e. above the track capacity, so it leaves the bound at the track capacity.");

    // Refit / attach inputs (mirroring the displaced CA producer). The CA-ordered module geometry is an
    // EventSetup product.
    desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("hltPhase2PixelRecHitsStubsMerger"))
        ->setComment("Shared pixel+stub rechit SoA the merged track hit ids index. twinMergeFullRefit/mergerExtend.");
    desc.add<edm::InputTag>("otRecHitsSrc", edm::InputTag("hltPixelSeedingOTRecHitsSoA"))
        ->setComment("Raw OT rechit SoA for the tagged (bit30) OT extras. twinMergeFullRefit/mergerExtend.");
    desc.add<edm::InputTag>("otStubsSrc", edm::InputTag("hltOTStubProducer"))
        ->setComment("OT stub SoA for the full-hits OT source (stub-membership mask). mergerExtend only.");

    descriptions.addWithDefaultLabel(desc);
  }

  // The once-per-job build of the measured tables. The rows are compile-time constants; this function
  // does the one piece of arithmetic that depends on a run-time input, the interpolation of the two
  // quantile maps at extDerivedEps_, and packs every row into one flat float payload whose offsets the
  // returned struct records, so a single device allocation carries all of them.
  PixelTracksSoAMerger::ExtTablesHost PixelTracksSoAMerger::buildExtTables() const {
    // The compiled-in rows and the walk must agree on the cell geometry they are keyed on.
    static_assert(extDerivedTables::kQCells == caExtension::kExtQCells, "Q-hat cell count mismatch");
    static_assert(extDerivedTables::kOTLayers == caExtension::kExtOTLayers, "OT layer count mismatch");
    static_assert(extDerivedTables::kMatClasses == caExtension::kExtMatClasses, "module class count mismatch");
    static_assert(extDerivedTables::kSigBClasses == caExtension::kExtSigBClasses, "bend class count mismatch");

    if (!(extDerivedEps_ > 0. && extDerivedEps_ < 1.))
      throw cms::Exception("Configuration")
          << "extDerivedEps = " << extDerivedEps_
          << " is not a probability: it is the mass of the correct-hit pull distribution the attach window "
             "must contain, so it has to lie strictly inside (0,1).";

    ExtTablesHost t;
    constexpr int nEps = extDerivedTables::kNEps;
    auto const& eps = extDerivedTables::kEps;
    // Interpolate the maps at eps once, on the host: linear in eps between grid nodes, clamped to the
    // grid ends, since a measured quantile is never extrapolated.
    const double epsUse = std::min(std::max(extDerivedEps_, eps.front()), eps.back());
    int epsLo = 0;
    while (epsLo + 2 < nEps && eps[epsLo + 1] < epsUse)
      ++epsLo;
    const double epsT = (eps[epsLo + 1] > eps[epsLo]) ? (epsUse - eps[epsLo]) / (eps[epsLo + 1] - eps[epsLo]) : 0.;
    auto interpolateMap = [&](auto const& map) {
      std::vector<float> out(caExtension::kExtQCells);
      for (int c = 0; c < caExtension::kExtQCells; ++c) {
        const double a0 = map[std::size_t(c) * nEps + epsLo];
        const double a1 = map[std::size_t(c) * nEps + epsLo + 1];
        out[c] = float(a0 + epsT * (a1 - a0));
      }
      return out;
    };
    const std::vector<float> qthrHost = interpolateMap(extDerivedTables::kQhat);    // 2-dof
    const std::vector<float> qthr3Host = interpolateMap(extDerivedTables::kQhat3);  // 3-dof (stub candidates)
    // The remaining rows are eps-independent: they go to the device as they were measured.
    auto narrow = [](auto const& row) { return std::vector<float>(row.begin(), row.end()); };

    // Pack: one flat payload, each row recording where it starts.
    auto push = [&t](std::vector<float> const& v, int& off) {
      off = int(t.flat.size());
      t.flat.insert(t.flat.end(), v.begin(), v.end());
    };
    push(qthrHost, t.offQhat);
    push(narrow(extDerivedTables::kEtaL), t.offEtaL);
    push(narrow(extDerivedTables::kRho), t.offRho);
    push(narrow(extDerivedTables::kDV), t.offDV);
    // the per-source-round hole rows, read by the raw-round hole pricing
    push(narrow(extDerivedTables::kEtaLRaw), t.offEtaLRaw);
    push(narrow(extDerivedTables::kRhoRaw), t.offRhoRaw);
    // the stub-bend rows
    push(qthr3Host, t.offQhat3);
    push(narrow(extDerivedTables::kSigBExcess), t.offSigBExc);
    push(narrow(extDerivedTables::kRho3), t.offRho3);
    return t;
  }

  void PixelTracksSoAMerger::produce(edm::StreamID streamID,
                                     device::Event& iEvent,
                                     const device::EventSetup& es) const {
    // get both Pixel and Tracker SoA collections
    auto queue = iEvent.queue();

    std::vector<const reco::TracksSoACollection*> inputTkSoAs;
    for (const auto& it : inputTkSoATokenV_) {
      auto const& aux = iEvent.get(it);
      inputTkSoAs.push_back(&aux);
    }

    // input SoA collections have the same layout
    // each of them is made up of two SoAs:
    // - one that contains the tracks
    // - one that contains the hits associated to the tracks
    // this code merges and copy them into a new SoA collection

    const std::size_t nInputs = inputTkSoAs.size();
    // The gather kernel takes its (up to) two inputs as separate view arguments; the deployed
    // TwoIter chain has exactly two. More inputs would be silently dropped, so refuse loudly.
    if (nInputs > 2)
      throw cms::Exception("Configuration")
          << "PixelTracksSoAMerger: got " << nInputs
          << " input track collections; the device-side merge supports at most 2 (deployed config).";

    // Sizing with no host readback: the merged output, and the downstream allocations keyed off it, is
    // sized from the sum of the two host counts (nTracks, nKeptHits) each input's producer publishes
    // beside its collection. Each count is what the producing compaction kernel committed, and the gather
    // below copies exactly those device-side quantities, so the output capacity equals the input content
    // by construction. An input without published counts reverts the sizing to the capacity sum; the
    // counts are used all-or-nothing, so both terms of the sum share one basis.
    int totTracks = 0;
    int totHits = 0;
    {
      int actualTracks = 0;
      int actualHits = 0;
      bool allCountsPresent = (nInputs > 0);
      for (std::size_t i = 0; i < nInputs && allCountsPresent; ++i) {
        auto const nTracksHandle = iEvent.getHandle(inputNTracksTokenV_[i]);
        auto const nKeptHitsHandle = iEvent.getHandle(inputNKeptHitsTokenV_[i]);
        if (nTracksHandle.isValid() && nKeptHitsHandle.isValid()) {
          actualTracks += int(*nTracksHandle);
          actualHits += int(*nKeptHitsHandle);
        } else {
          allCountsPresent = false;
        }
      }
      if (allCountsPresent) {
        totTracks = actualTracks;
        totHits = actualHits;
      } else {
        for (std::size_t i = 0; i < nInputs; ++i) {
          totTracks += int(inputTkSoAs[i]->view().tracks().metadata().size());
          totHits += int(inputTkSoAs[i]->view().trackHits().metadata().size());
        }
        if (nInputs > 0) {
          std::call_once(capacitySizingWarnOnce_, [&]() {
            edm::LogWarning("PixelTracksSoAMerger")
                << "at least one input track collection publishes no nTracks/nKeptHits counts; sizing "
                   "the merged output and everything downstream of it from the sum of the inputs' SoA "
                   "capacities instead. Correct, but larger than necessary.";
          });
        }
      }
    }
    // Empty-event early-out: with actual counts totTracks can be zero (both inputs empty). Publish
    // the empty collection directly -- its nTracks scalar is memset because allocation does not
    // initialise it -- rather than running the gather/filter/attach/dedup chain on zero-length
    // containers whose launch configurations never occur otherwise.
    if (totTracks == 0) {
      reco::TracksSoACollection emptyTracks(queue, 0, 0);
      auto emptyNTracks_d = cms::alpakatools::make_device_view(queue, emptyTracks.view().tracks().nTracks());
      alpaka::memset(queue, emptyNTracks_d, 0);
      iEvent.emplace(outputTkSoAToken_, std::move(emptyTracks));
      return;
    }

    // outputTemp is a SoA collection with the same layout as the inputs, sized from the host-known bounds
    // above. It is scoped in an optional so it can go back to the caching allocator as soon as
    // makeFilteredTracks has enqueued its last read of it, everything after that point being this
    // module's device high-water mark. The free is stream-ordered, so no wait is introduced.
    std::optional<reco::TracksSoACollection> outputTemp;
    outputTemp.emplace(queue, totTracks, totHits);

    // Allocate the per-track arm-label device buffer, one slot per merged track slot. The gather
    // kernel fills it on device in the dense merged-SoA track order, and stamps -1 over any tail.
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> armBuf;
    int32_t* armPtr = nullptr;
    if (totTracks > 0) {
      armBuf.emplace(cms::alpakatools::make_device_buffer<int32_t[]>(queue, totTracks));
      armPtr = armBuf->data();
    }

    // Device-side gather. Kernel_mergeGather reads each input's nTracks() and last-track hitOffsets() on
    // device and compacts all track and trackHits columns into a dense merged layout, writing the merged
    // nTracks scalar, the per-input shifted hitOffsets and the arm labels on device.
    //
    // An input that selected nothing needs no host gate: the gather adds zero to the cumulative
    // offsets and stamps Quality::bad over any unfilled tail, and the downstream filterTracks kernel
    // skips every slot below minQuality. (Both inputs empty is handled by the early-out above.)
    if (nInputs > 0) {
      // For nInputs == 1, pass the same input twice (the kernel reads nInputs inputs).
      auto const& inp0 = *inputTkSoAs[0];
      auto const& inp1 = (nInputs >= 2) ? *inputTkSoAs[1] : *inputTkSoAs[0];
      const int32_t arm0 = (nInputs >= 1) ? int32_t(armPerInput_[0]) : 0;
      const int32_t arm1 = (nInputs >= 2) ? int32_t(armPerInput_[1]) : 0;
      deviceAlgo_.mergeGather(*outputTemp, inp0, inp1, int(nInputs), armPtr, arm0, arm1, queue);
    }

    // Merger attach configuration, built here so the merged hit-storage headroom below can key off
    // maxExtraHitsPerTrack. The walk scans the full attachable layer set including the odd OT disks,
    // seeding OT hits so those disks are reachable.
    caExtension::AttachParams eParams;
    eParams.enable = true;
    eParams.verbose = mergerExtendVerbose_;
    eParams.useOTRecHits = true;
    eParams.extHostMaxChi2Ndof = float(extHostMaxChi2Ndof_);
    eParams.extHostMinHits = 0;          // off sentinel
    eParams.extHostMinPt = -1.f;         // off sentinel
    eParams.extChi2CutScaleTID = 1.f;    // identity on the TID disks
    eParams.extRawOTVetoTOB456 = false;  // the raw single-cluster round is open (see the note below)
    eParams.extRawOTVetoTID = false;     // idem
    eParams.extDisplacementAwareGate = true;
    eParams.extForwardPocketGate = false;   // off (see the arm plumbing below)
    eParams.extPocketGateArmScoped = true;  // inert with the gate off
    eParams.extMtvAlignedExtraCap = true;
    // Attach recall/calibration knobs.
    eParams.extRecallReachRelax = float(extRecallReachRelax_);
    eParams.extRecallPixelFirstBudget = extRecallPixelFirstBudget_;
    eParams.extCovScalePixel = 1.f;  // identity
    // extChi2CutScaleTOB456, extCovScaleStub and extStubBendGate are not configurable and stay at
    // their AttachParams no-op defaults: the derived selection replaces the quantities they scale
    // with the measured quantile, and the bend chi2 row (extBendPackage) supersedes the standalone veto.
    eParams.extCovScaleRawOT = 1.f;  // identity
    eParams.extPixelGateChi2Cut = float(extPixelGateChi2Cut_);
    // The attach thresholds that track the fit's chi2/covariance convention, from the cfi.
    eParams.preGateMaxChi2 = float(extPreGateMaxChi2_);  // attach pre-gate base reduced-chi2 cut
    eParams.maxAbsEta = float(extMaxAbsEta_);            // attach pre-gate |eta| ceiling
    eParams.chi2Cut = float(extChi2Cut_);                // attach-window barrel base cut
    eParams.endcapChi2Cut = float(extEndcapChi2Cut_);    // attach-window endcap base cut
    eParams.extDispGateSig2 = float(extDispGateSig2_);   // dispgate (|d0|/sigma_d0)^2 threshold
    // Continuation/cap knobs.
    eParams.extCapExemptAnchored = true;
    eParams.extCapBudgetFloor = 0;       // no floor
    eParams.extStateProcessNoise = 1.f;  // the physical Q, not a scale
    eParams.extRecallForcePixelVisit = true;
    // Their refinements, all at identity/off.
    eParams.extCapExemptTOB46Only = false;  // the exemption applies on every layer class, not only TOB4-6
    eParams.extCapExemptMaxChi2 = 0.f;      // off sentinel: the exemption carries no chi2 ceiling
    eParams.extMaxWalkLayers = extMaxWalkLayers_;
    eParams.extAttachFarFirst = true;
    eParams.extAttachFarMinAbsEta = float(extAttachFarMinAbsEta_);
    eParams.extAttachFarMaxWin = extAttachFarMaxWin_;
    eParams.extMaxSharedOwners = extMaxSharedOwners_;
    // Ceiling on the attach scratch sizing: the attach takes the merged track capacity as its candidate
    // bound, and this caps that bound (see the cfi comment). At or above the track capacity it is inert.
    eParams.extRefitMaxCandidates = uint32_t(extRefitMaxCandidates_);
    eParams.extAmbigDeltaChi2 = -1.f;  // off sentinel
    // ---- the measured tables come from the once-per-job, per-device cache ----------
    // The eps interpolation and the device upload are job constants (see buildExtTables()); the
    // pointers are taken below, inside the attach branch that uses them.
    std::call_once(extTablesOnce_, [&]() { extTablesCache_.emplace(buildExtTables()); });
    ExtTablesHost const& extTables = extTablesCache_->host();
    // The raw single-cluster attach round on TOB4-6 and TID/TEDD is open unconditionally, which is sound
    // only because the hole hypothesis is priced with that round's own conditional availability and
    // cluster density. The four pricing components (derived selection, bend rows, derived hole,
    // per-source-round prior) are therefore enabled together, so configuration cannot break the pairing.
    // Merged hit-storage capacity: headroom for the worst-case attached extras (each candidate gains
    // <= maxExtraHitsPerTrack) so the in-place rewrite never overflows and the event never falls back
    // whole. Without the attach, mergedHitCap == totHits.
    int mergedHitCap = totHits;
    if (totTracks > 0)
      mergedHitCap += totTracks * eParams.maxExtraHitsPerTrack;

    // Final merger-side GBL refit mask, one slot per output track (-1 = skip), consumed by
    // refitUnitedTracks -> refitMergedTwins. The attach grows every output track's hit list, so the
    // single final refit must cover all output tracks: refitAllTracks makes filterTracks stamp the
    // identity mask over every surviving output index, and the unused capacity tail stays -1 so the
    // fast-fit scan skips it.
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> unitedMask;
    int32_t* unitedMaskPtr = nullptr;
    if (totTracks > 0) {
      unitedMask.emplace(cms::alpakatools::make_device_buffer<int32_t[]>(queue, totTracks));
      alpaka::memset(queue, *unitedMask, 0xff);  // 0xffffffff = int32_t(-1)
      unitedMaskPtr = unitedMask->data();
    }

    // Forward-eta pocket gate arm plumbing. The gate is not enabled, so the per-track arm-label buffers
    // it would need are never built and both pointers stay null: filterTracks performs no arm scatter and
    // the walk reads no arm label.
    const uint8_t* pocketArmInPtr = nullptr;
    uint8_t* pocketArmIdOutPtr = nullptr;

    auto mergedTracks = deviceAlgo_.makeFilteredTracks(totTracks,
                                                       mergedHitCap,
                                                       *outputTemp,
                                                       minQuality_,
                                                       matchFraction_,
                                                       queue,
                                                       /*twinMerge=*/true,
                                                       armPtr,
                                                       float(twinMergeDeltaEta_),
                                                       float(twinMergeDeltaPhi_),
                                                       twinMergeMinSharedHits_,
                                                       /*twinMergeTier2=*/true,
                                                       float(twinMergeTier2DeltaEta_),
                                                       float(twinMergeTier2DeltaPhi_),
                                                       float(twinMergeNSigma2_),
                                                       /*twinMergeMinSharedFwd=*/1,
                                                       /*twinMergeRefit=*/true,
                                                       /*refitAllTracks=*/true,
                                                       unitedMaskPtr,
                                                       pocketArmInPtr,
                                                       pocketArmIdOutPtr);

    // makeFilteredTracks has enqueued every kernel that reads outputTemp, and mergedTracks is a fresh
    // collection rather than a view onto it, so it is released here. The caching allocator's free is
    // stream-ordered, so this is a lifetime change, not a synchronisation.
    outputTemp.reset();

    // Merger-side attach ("extender"): one attach walk over the merged collection, before the final
    // refit, using the ES full geometry. All the search structures the walk needs but the merger does
    // not already hold are built inside launchMergerAttach. Modifies the merged SoA in place.
    if (totTracks > 0) {
      auto const& geometry = checkedGeometry(es.getData(tokenCAGeom_));
      auto const& pixelRecHits = iEvent.get(tokenPixelRecHits_);
      auto const& otHits = iEvent.get(tokenOTHits_);
      auto const& otStubs = iEvent.get(tokenOTStubs_);
      auto const& stackedGeom = es.getData(tokenStackedGeomDev_);
      // The EventSetup delivers the radiation-length map rho(r,z) the fits integrate material along. The
      // map goes together with the fit-side terms: the total-material Highland logarithm, the composite
      // ionization medium in the dE/dx constants, and the cumulative energy-loss median law, the Landau
      // most-probable value not being additive. A map stating a different amount of charged material
      // without the matching change in those terms trades one error for the cancellation of another.
      const float* rhoMapDevice = es.getData(tokenBLMaterialMap_).data();
      const float bfield = float(1. / es.getData(tokenField_).inverseBzAtOriginInGeV());
      // (Bz,Br) field map, feeding the GBL refits and the extension chain: the fast BL fit publishes q/pT
      // in its own effective field, so the smoothed-prediction pass re-solves the band in that field and
      // the walk rebuilds its running helix with it. The walk's road geometry is the fitted circle,
      // recovered only by inverting the conversion the fit applied.
      const float* bMapDevice = es.getData(tokenBLBFieldMap_).data();

      auto caLayers = geometry.view().layers();
      const uint32_t nRecHits = uint32_t(pixelRecHits.view().trackingHits().metadata().size());
      // The merged track capacity is also the attach's host-known candidate bound, flowing into
      // AttachParams::extRefitMaxCandidates through min(). Candidates are a subset of merged tracks, so a
      // bound equal to the merged track count drops nothing.
      const uint32_t nTracksCap = uint32_t(mergedTracks.view().tracks().metadata().size());
      const uint32_t hitCapacity = uint32_t(mergedTracks.view().trackHits().metadata().size());

      // Extender layer surfaces: build once (first flag-ON event), lift to device, reuse thereafter.
      std::call_once(extLayersOnce_, [&]() {
        auto const& trackerGeom = es.getData(tokenTrackerGeom_);
        auto const& trackerTopo = es.getData(tokenTrackerTopo_);
        auto const& stackedGeomHost = es.getData(tokenStackedGeomHost_);
        extLayersCache_.emplace(::reco_extender::buildExtenderLayerGeometry(trackerGeom, trackerTopo, stackedGeomHost));
      });
      auto const& extLayersDev = extLayersCache_->get(queue);

      // All-open hit mask (the CA's TrackingRecHitsMasking has no merger analog): an empty ConstView,
      // which Kernel_extFindExtras reads (metadata().size() == 0) as "all open" without touching the
      // columns, so no per-event mask buffer is allocated.
      const ::reco::TrackingRecHitsMaskingConstView maskView{};

      // Full-hits OT source (raw OT rechits + stubs + phi binner + masks + per-CA-layer offsets, over
      // the ES caLayers). Kept alive across the attach launch. Null if there are no OT hits this event.
      const caExtension::OTHitsSource* otSrcPtr = nullptr;
      std::optional<caExtension::OTHitsBuffers> otBufs;
      std::optional<caExtension::OTHitsSource> otSrc;
      const uint32_t nOTHits = otHits.nHits();
      if (nOTHits > 0) {
        otBufs.emplace(caExtension::buildOTHitsSource(queue,
                                                      eParams,
                                                      otHits.const_view().otRecHits(),
                                                      otHits.const_view().otHitModules(),
                                                      stackedGeom.const_view(),
                                                      otStubs.const_view().stubs(),
                                                      caLayers,
                                                      nOTHits,
                                                      otStubs.nStubs()));
        otSrc.emplace(otBufs->makeSource());
        otSrcPtr = &otSrc.value();
      }

      // ---- the smoothed-prediction pass, immediately before the walk -------------------
      // One fast-BL band re-solve per walk host, publishing the smoothed-prediction payload the walk
      // consumes instead of the perigee-propagated covariance. The host set is exactly the walk's own
      // pre-gate population, replicated here as a mask so the pass costs nothing on non-hosts.
      std::optional<cms::alpakatools::device_buffer<Device, caExtension::ExtPredCoeff[]>> predBuf;
      std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> hostMaskBuf;
      {
        predBuf.emplace(cms::alpakatools::make_device_buffer<caExtension::ExtPredCoeff[]>(queue, nTracksCap));
        hostMaskBuf.emplace(cms::alpakatools::make_device_buffer<int32_t[]>(queue, nTracksCap));
        // The ExtPredCoeff array is not memset: the host-mask sweep visits every slot of
        // [0, nTracksCap) and clears `valid` there, the only field any consumer reads before the
        // payload is written (both readers test valid > 0.5f first).
        caExtension::launchExtHostMask(
            queue, eParams, mergedTracks.view().tracks(), nTracksCap, hostMaskBuf->data(), predBuf->data());
        // Called directly, not through CAHitMaskingAndMerger: routing it through the CA producer's
        // translation unit would change what the compiler generates for the code already there, since on
        // the serial backend these kernels are ordinary inlined host code.
        caExtension::launchExtPredCoeff(queue,
                                        pixelRecHits.view().trackingHits(),
                                        geometry.view().modules(),
                                        mergedTracks.view().tracks(),
                                        mergedTracks.view().trackHits(),
                                        hostMaskBuf->data(),
                                        otSrcPtr,
                                        bfield,
                                        rhoMapDevice,
                                        bMapDevice,
                                        /*fitCorrections=*/true,  // must match the CA's useFitCorrections
                                        predBuf->data(),
                                        nTracksCap);
        // The measured tables are already on this device: one flat, job-constant payload uploaded once
        // per device by the cache above, each row a base plus an offset into it.
        const float* extTablesBase = extTablesCache_->base(alpaka::getDev(queue));
        eParams.extDerivedSelection = true;
        eParams.extDerivedEps = extDerivedEps_;
        eParams.extDerivedHole = true;
        eParams.extHoleDetectionPrior = true;
        eParams.extPred = predBuf->data();
        eParams.extQhat = extTablesBase + extTables.offQhat;
        eParams.extEtaL = extTablesBase + extTables.offEtaL;
        eParams.extRho = extTablesBase + extTables.offRho;
        // The per-source-round hole rows: the raw round is priced with its own conditional availability
        // and cluster density rather than with the stub rows.
        eParams.extHoleRawRoundPrior = true;
        eParams.extEtaLRaw = extTablesBase + extTables.offEtaLRaw;
        eParams.extRhoRaw = extTablesBase + extTables.offRhoRaw;
        eParams.extDV = extTablesBase + extTables.offDV;
        eParams.extFmsBarrel = float(extFmsBarrel_);
        eParams.extFmsEndcap = float(extFmsEndcap_);
        // The 5th Q-hat |eta| bin, enabled when a road can legitimately be forward. The pre-gate alone
        // does not decide it: the cell is keyed on the walk state's |cot|, taken after the Kalman updates,
        // which drifts past sinh(2.4) on ordinary sub-2.4 hosts. Off, the ladder saturates at the 2.0 row.
        eParams.extFwdEtaBin = (extMaxAbsEta_ > 2.4);
        // the stub-bend package: the third (bend) row of the selection chi2 and its own 3-dof map
        eParams.extRho3 = extTablesBase + extTables.offRho3;
        eParams.extBendPackage = true;
        eParams.extQhat3 = extTablesBase + extTables.offQhat3;
        eParams.extSigBExcess = extTablesBase + extTables.offSigBExc;
      }

      caExtension::launchMergerAttach(queue,
                                      eParams,
                                      bfield,
                                      rhoMapDevice,
                                      mergedTracks.view().tracks(),
                                      mergedTracks.view().trackHits(),
                                      pocketArmIdOutPtr,  // pocket gate: per-track arm (null when off/arm-blind)
                                      pixelRecHits.view().trackingHits(),
                                      pixelRecHits.view().hitModules(),
                                      maskView,
                                      caLayers,
                                      geometry.view().modules(),
                                      extLayersDev.view(),
                                      nRecHits,
                                      nTracksCap,
                                      hitCapacity,
                                      otSrcPtr,
                                      nullptr);
      // No host wait here: otBufs and the other OT-source buffers are function-scope caching-allocator
      // buffers, and the allocator re-issues a freed block only once the event recorded on its queue at
      // free time has completed, so the queued launches that still read them are safe. Per-stream
      // back-pressure comes from the CA producers, which synchronize each event's queue at their acquire
      // boundary.
    }

    // Full GBL refit of the united winners: fetch the geometry + rechit + OT + material-map + bfield
    // inputs (mirroring the displaced CA producer) and overwrite the winners' state/cov/chi2/ndof.
    if (unitedMaskPtr) {
      auto const& geometry = checkedGeometry(es.getData(tokenCAGeom_));
      auto const& pixelRecHits = iEvent.get(tokenPixelRecHits_);
      auto const& otHits = iEvent.get(tokenOTHits_);
      auto const& stackedGeom = es.getData(tokenStackedGeomDev_);
      const float* rhoMapDevice = es.getData(tokenBLMaterialMap_).data();
      const float bfield = float(1. / es.getData(tokenField_).inverseBzAtOriginInGeV());
      // (Bz,Br) field map for the final refit's GBL effective field; this refit sets the published pt.
      const float* bMapDevice = es.getData(tokenBLBFieldMap_).data();
      deviceAlgo_.refitUnitedTracks(mergedTracks,
                                    unitedMaskPtr,
                                    geometry,
                                    pixelRecHits,
                                    &otHits,
                                    &stackedGeom,
                                    rhoMapDevice,
                                    bMapDevice,
                                    bfield,
                                    queue,
                                    /*dropOutlierFromHitList=*/true,
                                    /*outlierCoreProtect=*/true,
                                    /*fieldKernelWeights=*/true,
                                    /*chargeSymmetric=*/true,
                                    /*trajectoryCorrections=*/true,
                                    /*scatteringLogAtTotal=*/true,
                                    /*cumulativeEloss=*/true);
    }

#ifdef CA_SIZING_DUMP
    // The merged collection is complete here: the gather set its track count, and neither the attach
    // (in-place) nor the refit changes it. This is the demand the output capacity -- and everything the
    // merger sizes from it -- has to cover. Placed ahead of both exit paths so it runs once per event.
    dumpMergerSizing(queue, mergedTracks, "merger");
#endif

    // Final post-refit dedup over the refined merged tracks. The pre-attach dedup misses the
    // same-particle pairs whose members shared fewer than two hits before the attach, dominantly the
    // cross-arm prompt and displaced twins in the forward disk region; the attach gives both twins the
    // same OT hits and the final refit sharpens them, so a covariance-scaled co-occurrence match on the
    // refined tracks pairs them and drops the worse member. Returns a fresh compacted SoA.
    // Hit-id key space for the cov-dedup co-occurrence histogram: nDedupPixHits (the unified
    // pixel+strip SoA the merged ids index) + nDedupOTHits (bit30-tagged extras compress to
    // nDedupPixHits+otIdx). Re-fetch the already-consumed products (cheap references).
    const uint32_t nDedupPixHits = uint32_t(iEvent.get(tokenPixelRecHits_).view().trackingHits().metadata().size());
    const uint32_t nDedupOTHits = iEvent.get(tokenOTHits_).nHits();
    // The merge-or-keep-both confirm needs the merger's GBL-refit inputs (the same set
    // refitUnitedTracks takes). They are built here and passed to finalDedup; the confirm itself is
    // not enabled, so the fallback pair simply drops its loser.
    auto const& dedupGeometry = es.getData(tokenCAGeom_);
    auto const& dedupPixelRecHits = iEvent.get(tokenPixelRecHits_);
    auto const& dedupOtHits = iEvent.get(tokenOTHits_);
    auto const& dedupStackedGeom = es.getData(tokenStackedGeomDev_);
    const float* dedupRhoMap = es.getData(tokenBLMaterialMap_).data();
    const float dedupBfield = float(1. / es.getData(tokenField_).inverseBzAtOriginInGeV());
    // (Bz,Br) field map for the dedup-confirm union GBL refit.
    const float* dedupBMap = es.getData(tokenBLBFieldMap_).data();
    caExtension::OTHitsSource confirmOtSrc{};
    const caExtension::OTHitsSource* confirmOtSrcPtr = nullptr;
    if (dedupOtHits.nHits() > 0) {
      confirmOtSrc.otHits = dedupOtHits.const_view().otRecHits();
      confirmOtSrc.otHitModules = dedupOtHits.const_view().otHitModules();
      confirmOtSrc.stackedGeometry = dedupStackedGeom.const_view();
      confirmOtSrc.nOTHits = dedupOtHits.nHits();
      confirmOtSrcPtr = &confirmOtSrc;
    }
    MergerDedupConfirmInputs confirm{dedupPixelRecHits.view().trackingHits(),
                                     dedupGeometry.view().modules(),
                                     confirmOtSrcPtr,
                                     dedupRhoMap,
                                     dedupBMap,
                                     dedupBfield,
                                     // The union-refit merge-or-keep-both confirm is off, so its
                                     // union-hit-loss budget is never consulted.
                                     /*fbMergeConfirm=*/false,
                                     /*fbMergeConfirmDelta=*/1,
                                     // Dedup ranking/guard set: finder mode (count without dropping) and
                                     // the cross-arm corner guard are off; the length key that decides
                                     // which of two duplicates survives is the plain hit count, which
                                     // separates better than (nLayers, nHits) or a cluster-weighted count.
                                     /*fbFinderOnly=*/false,
                                     /*rankClusters=*/false,
                                     /*rankNHits=*/true,
                                     /*guardCrossArm=*/false,
                                     /*guardVertPosMin=*/1.f,
                                     /*guardChi2Margin=*/0.f,
                                     // Fallback tuning: the gate width follows the shared-hit gate (<=0
                                     // sentinel), the drop authority is bounded at |eta| 2.5 and the
                                     // confirm-only box cuts are off.
                                     /*fbNSigma2=*/-1.f,
                                     /*fbDropBound=*/2.5f,
                                     /*fbEnable=*/true,
                                     /*fbSameCharge=*/false,
                                     /*fbAbsFloorDPhi=*/1.e30f,
                                     /*fbAbsFloorDQoP=*/1.e30f,
                                     /*fbAbsFloorDCot=*/1.e30f};
    auto dedupTracks = deviceAlgo_.finalDedupTracks(mergedTracks, nDedupPixHits, nDedupOTHits, queue, &confirm);
    iEvent.emplace(outputTkSoAToken_, std::move(dedupTracks));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PixelTracksSoAMerger);
