#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <iomanip>
#include <map>

#include <TFormula.h>
#include "CommonTools/Utils/interface/FormulaEvaluator.h"

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/OTRecHitsSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/StubsSoACollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/RunningAverage.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/Record/interface/StackedModuleGeometryRecord.h"
#include "RecoTracker/Record/interface/CAGeometryRecord.h"
#include "RecoTracker/PixelSeeding/interface/alpaka/CAGeometrySoACollection.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BLMaterialMapCollection.h"
#include "RecoTracker/Record/interface/BLMaterialMapRecord.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BLBFieldMapCollection.h"
#include "RecoTracker/Record/interface/BLBFieldMapRecord.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometryHost.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometryHost.h"
#include "RecoTracker/PixelSeeding/interface/alpaka/StackedModuleGeometrySoACollection.h"
#include "CAGeometryBuild.h"
#include "CAHitNtupletGenerator.h"

#include "HeterogeneousCore/AlpakaCore/interface/MoveToDeviceCache.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

// #define GPU_DEBUG

namespace reco {
  struct CAGeometryParams {
    // The layer- and layer-pair dependent settings come from the `geometry` PSet, the generic cut
    // scalars from the top level of the module PSet. globalBeginRun builds the CA geometry product
    // from them, broadcasting the two per-layer triplet cuts onto the per-layer-pair SoA columns.
    // The members ending in PerPair, floorDCACuts_ and the three maxStub* vectors are the OT-stub
    // extension: optional and empty when unconfigured, in which case the per-layer or scalar value
    // is broadcast instead, or the cut is disabled.
    CAGeometryParams(edm::ParameterSet const& iConfig)
        : CAGeometryParams(iConfig, iConfig.getParameterSet("geometry")) {}

    CAGeometryParams(edm::ParameterSet const& iConfig, edm::ParameterSet const& geo)
        :  // graph (per layer pair)
          layerPairs_(geo.getParameter<std::vector<unsigned int>>("pairGraph")),
          startingPairs_(geo.getParameter<std::vector<unsigned int>>("startingPairs")),
          skipsLayers_(geo.getParameter<std::vector<unsigned int>>("skipsLayers")),
          // doublet cuts (per layer pair)
          phiCuts_(geo.getParameter<std::vector<int>>("phiCuts")),
          minInner_(geo.getParameter<std::vector<double>>("minInner")),
          maxInner_(geo.getParameter<std::vector<double>>("maxInner")),
          minOuter_(geo.getParameter<std::vector<double>>("minOuter")),
          maxOuter_(geo.getParameter<std::vector<double>>("maxOuter")),
          maxDR_(geo.getParameter<std::vector<double>>("maxDR")),
          minDZ_(geo.getParameter<std::vector<double>>("minDZ")),
          maxDZ_(geo.getParameter<std::vector<double>>("maxDZ")),
          ptCuts_(geo.getParameter<std::vector<double>>("ptCuts")),
          // doublet cuts (scalars, top level)
          cellZ0Cut_(iConfig.getParameter<double>("cellZ0Cut")),
          dzdrFact_(iConfig.getParameter<double>("dzdrFact")),
          minYsizeB1_(iConfig.getParameter<int>("minYsizeB1")),
          minYsizeB2_(iConfig.getParameter<int>("minYsizeB2")),
          maxDYsize12_(iConfig.getParameter<int>("maxDYsize12")),
          maxDYsize_(iConfig.getParameter<int>("maxDYsize")),
          maxDYPred_(iConfig.getParameter<int>("maxDYPred")),
          // triplet cuts (per layer)
          caDCACuts_(geo.getParameter<std::vector<double>>("caDCACuts")),
          caThetaCuts_(geo.getParameter<std::vector<double>>("caThetaCuts")),
          // triplet cuts (scalars, top level)
          ptmin_(iConfig.getParameter<double>("ptmin")),
          hardCurvCut_(iConfig.getParameter<double>("hardCurvCut")),
          maxPhiResid_(iConfig.getParameter<double>("maxPhiResid")),
          sameDPhiSign_(iConfig.getParameter<bool>("sameDPhiSign")),
          // ntuplet cuts and fishbone (per layer)
          startMaxInnerR_(geo.getParameter<std::vector<double>>("startMaxInnerR")),
          maxDCurv_(geo.getParameter<std::vector<double>>("maxDCurv")),
          floorDCurv_(geo.getParameter<std::vector<double>>("floorDCurv")),
          fishboneCuts_(geo.getParameter<std::vector<double>>("fishboneCuts")),
          // OT-stub extension (all optional, per layer pair)
          caDCACutsPerPair_(optionalVDouble(geo, "caDCACutsPerPair")),
          caThetaCutsPerPair_(optionalVDouble(geo, "caThetaCutsPerPair")),
          cellZ0CutPerPair_(optionalVDouble(geo, "cellZ0CutPerPair")),
          floorDCACuts_(optionalVDouble(geo, "floorDCACuts")),
          maxStubCurvSigma_(optionalVDouble(geo, "maxStubCurvSigma")),
          maxStubGeomCurvSigma_(optionalVDouble(geo, "maxStubGeomCurvSigma")),
          maxStubInnerDoubletDCurv_(optionalVDouble(geo, "maxStubInnerDoubletDCurv")) {
      // Set when some starting pair has an inner layer other than BPix1.
      startNoBPix1_ = false;
      for (const unsigned int& i : startingPairs_) {
        if (layerPairs_[2 * i] > 0) {
          startNoBPix1_ = true;
          break;
        }
      }

      // Size validation. Everything per layer pair must agree with the graph, everything per layer
      // must agree with the layer count, and every starting-pair id must address an existing pair.
      const size_t nPairs = layerPairs_.size() / 2;
      const size_t nLayers = caThetaCuts_.size();
      auto checkPairs = [&](std::vector<double> const& v, char const* name) {
        if (!v.empty() && v.size() != nPairs)
          throw cms::Exception("Configuration") << "CAHitNtupletAlpaka: geometry." << name << " has " << v.size()
                                                << " entries but the CA graph has " << nPairs << " layer pairs.";
      };
      if (layerPairs_.size() % 2 != 0)
        throw cms::Exception("Configuration")
            << "CAHitNtupletAlpaka: geometry.pairGraph has an odd number of entries (" << layerPairs_.size() << ").";
      if (skipsLayers_.size() != nPairs)
        throw cms::Exception("Configuration") << "CAHitNtupletAlpaka: geometry.skipsLayers has " << skipsLayers_.size()
                                              << " entries but the CA graph has " << nPairs << " layer pairs.";
      if (phiCuts_.size() != nPairs)
        throw cms::Exception("Configuration") << "CAHitNtupletAlpaka: geometry.phiCuts has " << phiCuts_.size()
                                              << " entries but the CA graph has " << nPairs << " layer pairs.";
      for (auto const* p : {&minInner_, &maxInner_, &minOuter_, &maxOuter_, &maxDR_, &minDZ_, &maxDZ_, &ptCuts_})
        if (p->size() != nPairs)
          throw cms::Exception("Configuration")
              << "CAHitNtupletAlpaka: a per-layer-pair vector of the geometry PSet has " << p->size()
              << " entries but the CA graph has " << nPairs << " layer pairs.";
      for (auto const* p : {&caDCACuts_, &startMaxInnerR_, &maxDCurv_, &floorDCurv_, &fishboneCuts_})
        if (p->size() != nLayers)
          throw cms::Exception("Configuration")
              << "CAHitNtupletAlpaka: a per-layer vector of the geometry PSet has " << p->size()
              << " entries but caThetaCuts declares " << nLayers << " layers.";
      checkPairs(caDCACutsPerPair_, "caDCACutsPerPair");
      checkPairs(caThetaCutsPerPair_, "caThetaCutsPerPair");
      checkPairs(cellZ0CutPerPair_, "cellZ0CutPerPair");
      checkPairs(floorDCACuts_, "floorDCACuts");
      checkPairs(maxStubCurvSigma_, "maxStubCurvSigma");
      checkPairs(maxStubGeomCurvSigma_, "maxStubGeomCurvSigma");
      checkPairs(maxStubInnerDoubletDCurv_, "maxStubInnerDoubletDCurv");
      for (const unsigned int& i : startingPairs_)
        if (i >= nPairs)
          throw cms::Exception("Configuration") << "CAHitNtupletAlpaka: geometry.startingPairs contains the pair id "
                                                << i << " but the CA graph only has " << nPairs << " layer pairs.";
      for (size_t i = 0; i < layerPairs_.size(); ++i)
        if (layerPairs_[i] >= nLayers)
          throw cms::Exception("Configuration")
              << "CAHitNtupletAlpaka: geometry.pairGraph refers to the layer " << layerPairs_[i]
              << " but caThetaCuts only declares " << nLayers << " layers.";

#ifdef GPU_DEBUG
      std::cout << "\n========== CAGeometryParams CONSTRUCTOR ==========" << std::endl;
      std::cout << "Reading geometry from Python ParameterSet..." << std::endl;
      std::cout << "  caThetaCuts size: " << caThetaCuts_.size() << std::endl;
      std::cout << "  caDCACuts size: " << caDCACuts_.size() << std::endl;
      std::cout << "  pairGraph size: " << layerPairs_.size() << " (= " << nPairs << " pairs)" << std::endl;
      std::cout << "  startingPairs number: " << startingPairs_.size() << std::endl;
      std::cout << "  phiCuts size: " << phiCuts_.size() << std::endl;
      std::cout << "  First 5 phiCuts values: ";
      for (size_t i = 0; i < std::min(size_t(5), phiCuts_.size()); ++i) {
        std::cout << phiCuts_[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "==================================================\n" << std::endl;
#endif
    }

    // Reads an optional vdouble of the geometry PSet, returning an empty vector when it is not set.
    static std::vector<double> optionalVDouble(edm::ParameterSet const& pset, char const* name) {
      return pset.existsAs<std::vector<double>>(name) ? pset.getParameter<std::vector<double>>(name)
                                                      : std::vector<double>{};
    }

    // graph (per layer pair)
    const std::vector<unsigned int> layerPairs_;
    const std::vector<unsigned int> startingPairs_;
    const std::vector<unsigned int> skipsLayers_;

    // doublet cuts (per layer pair)
    const std::vector<int> phiCuts_;
    const std::vector<double> minInner_;
    const std::vector<double> maxInner_;
    const std::vector<double> minOuter_;
    const std::vector<double> maxOuter_;
    const std::vector<double> maxDR_;
    const std::vector<double> minDZ_;
    const std::vector<double> maxDZ_;
    const std::vector<double> ptCuts_;

    // doublet cuts (scalars)
    const double cellZ0Cut_;
    const double dzdrFact_;
    const int minYsizeB1_;
    const int minYsizeB2_;
    const int maxDYsize12_;
    const int maxDYsize_;
    const int maxDYPred_;

    // triplet cuts (per layer)
    const std::vector<double> caDCACuts_;
    const std::vector<double> caThetaCuts_;

    // triplet cuts (scalars)
    const double ptmin_;
    const double hardCurvCut_;
    const double maxPhiResid_;
    const bool sameDPhiSign_;

    // ntuplet cuts and fishbone (per layer)
    const std::vector<double> startMaxInnerR_;
    const std::vector<double> maxDCurv_;
    const std::vector<double> floorDCurv_;
    const std::vector<double> fishboneCuts_;

    // OT-stub extension (per layer pair, empty when not configured)
    const std::vector<double> caDCACutsPerPair_;
    const std::vector<double> caThetaCutsPerPair_;
    const std::vector<double> cellZ0CutPerPair_;
    const std::vector<double> floorDCACuts_;
    const std::vector<double> maxStubCurvSigma_;
    const std::vector<double> maxStubGeomCurvSigma_;
    const std::vector<double> maxStubInnerDoubletDCurv_;

    bool startNoBPix1_;

    // BeginRun handles for the per-run geometry build. Consumed only by the topologies that build
    // their own geometry; Phase2OTStubs takes it from CAGeometryRecord instead and leaves these
    // unset.
    mutable edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tokenGeometry_;
    mutable edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tokenTopology_;
  };

}  // namespace reco

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Per-run device-resident CA geometry SoA of one CA producer. Phase2OTStubs holds the
  // per-iteration configuration blocks only (graph plus the three cut blocks) and takes the geometry
  // blocks from the shared CAGeometryRecord product; every other topology holds all six blocks.
  struct CARunGeometry {
    using CAGeometryCache = cms::alpakatools::MoveToDeviceCache<Device, ::reco::CAGeometryHost>;
    CARunGeometry(::reco::CAGeometryHost&& geometry) : geometry_(std::move(geometry)) {}
    CAGeometryCache geometry_;
  };

  template <typename TrackerTraits>
  class CAHitNtupletAlpaka
      : public stream::SynchronizingEDProducer<edm::GlobalCache<::reco::CAGeometryParams>, edm::RunCache<CARunGeometry>> {
    using HitsConstView = ::reco::TrackingRecHitConstView;
    using HitsOnDevice = reco::TrackingRecHitsSoACollection;
    using HitsOnHost = ::reco::TrackingRecHitHost;

    using MapToHit = reco::TrackingRecHitsMaskingCollection;
    using MapToHitConstView = ::reco::TrackingRecHitsMaskingConstView;

    using TkSoAHost = ::reco::TracksHost;
    using TkSoADevice = reco::TracksSoACollection;

    using Algo = CAHitNtupletGenerator<TrackerTraits>;

  public:
    explicit CAHitNtupletAlpaka(const edm::ParameterSet& iConfig, const ::reco::CAGeometryParams* iCache);
    ~CAHitNtupletAlpaka() override = default;

    // acquire() launches the whole CA build (kernels plus one async readback of the offsets); the
    // framework schedules produce only after this event's queue has drained, waiting in its own
    // async callback rather than on a TBB thread. produce() launches the fit passes and
    // classification on the landed offsets and puts the product.
    void acquire(device::Event const& iEvent, device::EventSetup const& es) override;
    void produce(device::Event& iEvent, device::EventSetup const& es) override;

    // The generator accumulates capped-container overflow counts per stream; report them once at
    // stream end. The module label distinguishes several CA instances of the same job.
    void endStream() override { deviceAlgo_.reportOverflows(moduleDescription().moduleLabel()); }

    static void globalEndJob(::reco::CAGeometryParams const*) { /* Do nothing */ };
    static void globalEndRun(edm::Run const& iRun,
                             edm::EventSetup const&,
                             RunContext const* iContext) { /* Do nothing */ };

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    static std::shared_ptr<CARunGeometry> globalBeginRun(edm::Run const& iRun,
                                                         edm::EventSetup const& iSetup,
                                                         GlobalCache const* iCache) {
      // The CAGeometryParams constructor validates the size consistency of the whole geometry
      // PSet; the asserts below restate only the invariants this function relies on directly.
      int n_layers = iCache->caThetaCuts_.size();
      int n_pairs = iCache->layerPairs_.size() / 2;

#ifdef GPU_DEBUG
      std::cout << "No. Layers to be used = " << n_layers << std::endl;
      std::cout << "No. Pairs to be used = " << n_pairs << std::endl;
#endif

      assert(int(n_pairs) == int(iCache->maxDR_.size()));
      assert(int(*std::max_element(iCache->layerPairs_.begin(), iCache->layerPairs_.end())) < n_layers);
      assert(iCache->startingPairs_.empty() ||
             int(*std::max_element(iCache->startingPairs_.begin(), iCache->startingPairs_.end())) < n_pairs);
      // Every per-layer vector has caThetaCuts.size() entries, the n_layers used below.
      assert(n_layers == int(iCache->fishboneCuts_.size()));

      // Phase2OTStubs: the configuration blocks only, {0, n_pairs, n_pairs, n_pairs, n_layers, 0}.
      // The geometry blocks (layers + modules) come from the shared CAGeometryRecord product, so the
      // tracker walk runs once per IOV and the module table is resident once per device instead of
      // once per consumer. Every other topology: the self-contained per-run build, all six blocks in
      // one product, {n_layers + 1, n_pairs, n_pairs, n_pairs, n_layers, n_modules}, and the two
      // geometry handles the generator takes are the same object.
      ::reco::CAGeometryHost product = [&]() {
        if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
          return ::reco::CAGeometryHost{cms::alpakatools::host(), 0, n_pairs, n_pairs, n_pairs, n_layers, 0};
        } else {
          auto const& trackerGeometry = iSetup.getData(iCache->tokenGeometry_);
          auto const& trackerTopology = iSetup.getData(iCache->tokenTopology_);
          // stackedGeometry is a Phase2OTStubs-only input, and that topology does not build here.
          return ::reco::buildCAGeometryHost<TrackerTraits>(
              trackerGeometry, trackerTopology, /* stackedGeometry = */ nullptr, n_layers, n_pairs, n_layers);
        }
      }();

#ifdef GPU_DEBUG
      std::cout << "Full CA LayerStart: " << n_layers << " layers with " << product.view().modules().metadata().size()
                << " modules in total (0 modules => the geometry blocks come from CAGeometryRecord)." << std::endl;
#endif

      auto graphSoA = product.view().graph();
      auto doubletCutsSoA = product.view().doubletCuts();
      auto tripletCutsSoA = product.view().tripletCuts();
      auto ntupletCutsSoA = product.view().ntupletCuts();

      for (int i = 0; i < n_layers; ++i) {
        ntupletCutsSoA.startMaxInnerR()[i] = iCache->startMaxInnerR_[i];
        ntupletCutsSoA.maxDCurv()[i] = iCache->maxDCurv_[i];
        ntupletCutsSoA.floorDCurv()[i] = iCache->floorDCurv_[i];
        // fishboneCut lives with the per-layer cuts, not with the per-layer geometry: the prompt
        // and displaced OT-stub iterations share one layer table but use different thresholds.
        ntupletCutsSoA.fishboneCut()[i] = iCache->fishboneCuts_[i];
      }

      // Per-layer-pair fill. The two triplet cuts the configuration expresses per layer are
      // broadcast onto the pairs, each pair taking the value of its own inner layer, which is the
      // indexing the kernels read back:
      //   * TripletCuts::accept reads maxDCA/floorDCA at the inner cell's pair (L1,L2), whose inner
      //     layer is the triplet's innermost layer L1 -> caDCACuts[L1];
      //   * it reads maxRZTolerance at the outer cell's pair (L2,L3), whose inner layer is the
      //     triplet's middle layer L2 -> caThetaCuts[L2].
      // The scalar cellZ0Cut is broadcast unchanged onto every pair. The "...PerPair" vectors of the
      // OT-stub extension replace the corresponding broadcast when configured.
      for (int i = 0; i < n_pairs; ++i) {
        const uint32_t innerLayer = iCache->layerPairs_[2 * i];
        // The layer-pair graph is per-iteration configuration (the prompt and displaced iterations
        // use different graphs), so it is filled here for every topology.
        graphSoA.layerPair()[i] = {{uint32_t(iCache->layerPairs_[2 * i]), uint32_t(iCache->layerPairs_[2 * i + 1])}};
        graphSoA.skipsLayers()[i] = uint16_t(bool(iCache->skipsLayers_[i]));
        graphSoA.startingPair()[i] = false;
        doubletCutsSoA.maxDPhi()[i] = iCache->phiCuts_[i];
        doubletCutsSoA.minInner()[i] = iCache->minInner_[i];
        doubletCutsSoA.maxInner()[i] = iCache->maxInner_[i];
        doubletCutsSoA.minOuter()[i] = iCache->minOuter_[i];
        doubletCutsSoA.maxOuter()[i] = iCache->maxOuter_[i];
        doubletCutsSoA.maxDZ()[i] = iCache->maxDZ_[i];
        doubletCutsSoA.minDZ()[i] = iCache->minDZ_[i];
        doubletCutsSoA.maxDR()[i] = iCache->maxDR_[i];
        // convert ptCut in curvature radius in cm
        // 1 GeV track has 1 GeV/c / (e * 3.8T) ~ 87 cm radius in a 3.8T field
        const float minRadius = iCache->ptCuts_[i] * 87.78f;
        // Use minRadius^2/4 in the CA to avoid sqrt
        const float minRadius2T4 = 4.f * minRadius * minRadius;
        doubletCutsSoA.minPt()[i] = minRadius2T4;
        doubletCutsSoA.maxZ0()[i] =
            (!iCache->cellZ0CutPerPair_.empty()) ? iCache->cellZ0CutPerPair_[i] : iCache->cellZ0Cut_;
        tripletCutsSoA.maxRZTolerance()[i] =
            (!iCache->caThetaCutsPerPair_.empty()) ? iCache->caThetaCutsPerPair_[i] : iCache->caThetaCuts_[innerLayer];
        tripletCutsSoA.maxDCA()[i] =
            (!iCache->caDCACutsPerPair_.empty()) ? iCache->caDCACutsPerPair_[i] : iCache->caDCACuts_[innerLayer];
        // OT-stub extension cuts: from the configuration when present, otherwise the disabled
        // sentinel -1.
        tripletCutsSoA.floorDCA()[i] =
            (!iCache->floorDCACuts_.empty()) ? static_cast<float>(iCache->floorDCACuts_[i]) : -1.0f;
        doubletCutsSoA.maxStubCurvSigma()[i] =
            (!iCache->maxStubCurvSigma_.empty()) ? static_cast<float>(iCache->maxStubCurvSigma_[i]) : -1.0f;
        tripletCutsSoA.maxStubGeomCurvSigma()[i] =
            (!iCache->maxStubGeomCurvSigma_.empty()) ? static_cast<float>(iCache->maxStubGeomCurvSigma_[i]) : -1.0f;
        tripletCutsSoA.maxStubInnerDoubletDCurv()[i] = (!iCache->maxStubInnerDoubletDCurv_.empty())
                                                           ? static_cast<float>(iCache->maxStubInnerDoubletDCurv_[i])
                                                           : -1.0f;
      }

      // `startingPairs` is a list of pair ids, stored as one flag per pair in the SoA.
      for (const unsigned int& i : iCache->startingPairs_)
        graphSoA.startingPair()[i] = true;

      doubletCutsSoA.dzdrFact() = iCache->dzdrFact_;
      doubletCutsSoA.minInnerSizeB1() = iCache->minYsizeB1_;
      doubletCutsSoA.minInnerSizeB2() = iCache->minYsizeB2_;
      doubletCutsSoA.maxDSizeB1() = iCache->maxDYsize12_;
      doubletCutsSoA.maxDSize() = iCache->maxDYsize_;
      doubletCutsSoA.maxDSizePred() = iCache->maxDYPred_;

      tripletCutsSoA.ptmin() = iCache->ptmin_;
      tripletCutsSoA.maxCurv() = iCache->hardCurvCut_;
      tripletCutsSoA.maxPhiResid() = iCache->maxPhiResid_;
      tripletCutsSoA.sameDPhiSign() = iCache->sameDPhiSign_;

#ifdef GPU_DEBUG
      // The `layers` block is empty for the topology whose geometry comes from CAGeometryRecord
      // (Phase2OTStubs): print a dash instead of indexing it.
      auto layersSoA = product.view().layers();
      const bool hasLayerBlock = layersSoA.metadata().size() > 0;
      std::cout << "\n========== CA GEOMETRY FROM PYTHON CONFIG ==========" << std::endl;
      std::cout << "Number of layers: " << n_layers << std::endl;
      std::cout << "Number of layer pairs: " << n_pairs << std::endl;

      std::cout << "\n--- Layer Pair Geometry (all pairs) ---" << std::endl;
      std::cout << "Pair | Inner | Outer | phiCut | minIn | maxIn | minOut | maxOut | maxDR | minDZ | maxDZ | minPt | "
                   "stubSigma | start"
                << std::endl;
      std::cout << "-----|-------|-------|--------|-------|-------|--------|--------|-------|-------|-------|-------|--"
                   "---------|------"
                << std::endl;
      for (int i = 0; i < n_pairs; ++i) {
        float stubSig = (!iCache->maxStubCurvSigma_.empty()) ? iCache->maxStubCurvSigma_[i] : -1.0;
        std::cout << std::setw(4) << i << " | " << std::setw(5) << iCache->layerPairs_[2 * i] << " | " << std::setw(5)
                  << iCache->layerPairs_[2 * i + 1] << " | " << std::setw(6) << iCache->phiCuts_[i] << " | "
                  << std::setw(5) << iCache->minInner_[i] << " | " << std::setw(5) << iCache->maxInner_[i] << " | "
                  << std::setw(6) << iCache->minOuter_[i] << " | " << std::setw(6) << iCache->maxOuter_[i] << " | "
                  << std::setw(5) << iCache->maxDR_[i] << " | " << std::setw(5) << iCache->minDZ_[i] << " | "
                  << std::setw(5) << iCache->maxDZ_[i] << " | " << std::setw(5) << iCache->ptCuts_[i] << " | "
                  << std::setw(9) << stubSig << " | " << (graphSoA.startingPair()[i] ? "Y" : "N") << std::endl;
      }

      std::cout << "\n--- Layer Cuts (all layers) ---" << std::endl;
      std::cout << "Layer | isBarrel | caThetaCut | caDCACut | startMaxInnerR | fishboneCut" << std::endl;
      std::cout << "------|----------|------------|----------|----------------|------------" << std::endl;
      // caThetaCuts/caDCACuts are the per-layer configuration vectors (size n_layers); the SoA
      // columns they are broadcast onto are per layer pair.
      for (int i = 0; i < n_layers; ++i) {
        std::cout << std::setw(5) << i << " | " << std::setw(8)
                  << (hasLayerBlock ? (layersSoA.isBarrel()[i] ? "Y" : "N") : "-") << " | " << std::setw(10)
                  << iCache->caThetaCuts_[i] << " | " << std::setw(8) << iCache->caDCACuts_[i] << " | " << std::setw(14)
                  << iCache->startMaxInnerR_[i] << " | " << std::setw(11) << iCache->fishboneCuts_[i] << std::endl;
      }
      std::cout << "====================================================\n" << std::endl;
#endif

      return std::make_shared<CARunGeometry>(std::move(product));
    }

    static std::unique_ptr<::reco::CAGeometryParams> initializeGlobalCache(edm::ParameterSet const& iConfig) {
      return std::make_unique<::reco::CAGeometryParams>(iConfig);
    }

  private:
    // Resolve the geometry handle (the `layers` and `modules` blocks) for this event. Phase2OTStubs
    // takes it from the EventSetup, one build per IOV shared with the other OT-stub CA iteration and
    // with the merger; every other topology returns the run-cache product itself, so the generator
    // gets the same object on both handles. The ESProducer's `nLayers` and this producer's
    // `geometry` PSet are configured independently, and a mismatch would silently index the wrong
    // layerStarts, hence the per-event check.
    typename Algo::CAGeometryOnDevice const& geometryHandle(device::EventSetup const& es,
                                                            typename Algo::CAGeometryOnDevice const& runCacheProduct) {
      if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
        auto const& esGeometry = es.getData(tokenCAGeom_);
        const int nLayersES = int(esGeometry.view().layers().metadata().size()) - 1;
        const int nLayersCfg = int(globalCache()->caThetaCuts_.size());
        if (nLayersES != nLayersCfg)
          throw cms::Exception("CAGeometryMismatch")
              << "CAHitNtupletAlpaka: the shared CA geometry (CAGeometryRecord) describes " << nLayersES
              << " CA layers but this producer is configured for " << nLayersCfg
              << ". The ESProducer's `nLayers` and the `geometry` PSet of every CA producer that shares it must "
                 "describe the same layer table.";
        return esGeometry;
      } else {
        return runCacheProduct;
      }
    }

    // The CAGeometryRecord product built once per IOV by CAGeometryESProducer<Phase2OTStubs> and
    // read by both OT-stub CA iterations and by PixelTracksSoAMerger. Left unset for every other
    // topology, which builds its own geometry per run.
    device::ESGetToken<reco::CAGeometrySoACollection, CAGeometryRecord> tokenCAGeom_;

    const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tokenField_;
    // The BL-fit material map and the (Bz,Br) r-z field map. The fit reads them only under
    // useFitCorrections, so they are consumed only then: the other topologies run in menus that do
    // not provide these records.
    bool useFitCorrections_ = false;
    device::ESGetToken<BLMaterialMap, BLMaterialMapRecord> tokenBLMaterialMap_;
    device::ESGetToken<BLBFieldMap, BLBFieldMapRecord> tokenBLBFieldMap_;
    const device::EDGetToken<HitsOnDevice> tokenHit_;
    const device::EDPutToken<TkSoADevice> tokenTrack_;
#ifdef CA_TRIPLET_DUMP
    // Per-built-triplet training-dataset product (one row per built triplet, valid rows = view().nValid()).
    // Only registered/emitted in CA_TRIPLET_DUMP builds; production builds carry nothing.
    const device::EDPutToken<TripletDumpSoACollection> tokenTripletDump_;
#endif
    // Optional per-iteration hit mask. An empty "hitMask" InputTag means no masking: nothing is
    // consumed and the kernels receive an empty view.
    const bool hasHitMask_;
    device::EDGetToken<MapToHit> tokenHitMask_;

    const ::reco::FormulaEvaluator maxNumberOfDoublets_;
    const ::reco::FormulaEvaluator maxNumberOfTuples_;
    const uint32_t minNumberOfDoublets_;
    const uint32_t minNumberOfTuples_;

    Algo deviceAlgo_;

    // Seam state (per stream): the pending CA build acquire hands to produce, and the
    // no-BPix1 early-out flag (no CA ran; produce emplaces an empty collection).
    std::optional<typename Algo::PendingTuples> pending_;
    bool noBPix1_ = false;
  };

  template <typename TrackerTraits>
  CAHitNtupletAlpaka<TrackerTraits>::CAHitNtupletAlpaka(const edm::ParameterSet& iConfig,
                                                        const ::reco::CAGeometryParams* iCache)
      : SynchronizingEDProducer(iConfig),
        tokenField_(esConsumes()),
        tokenHit_(consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"))),
        tokenTrack_(produces()),
#ifdef CA_TRIPLET_DUMP
        tokenTripletDump_(produces()),
#endif
        hasHitMask_(not iConfig.getParameter<edm::InputTag>("hitMask").label().empty()),
        maxNumberOfDoublets_(iConfig.getParameter<std::string>("maxNumberOfDoublets")),
        maxNumberOfTuples_(iConfig.getParameter<std::string>("maxNumberOfTuples")),
        minNumberOfDoublets_(iConfig.getParameter<uint32_t>("minNumberOfDoublets")),
        minNumberOfTuples_(iConfig.getParameter<uint32_t>("minNumberOfTuples")),
        deviceAlgo_(iConfig) {
    useFitCorrections_ = iConfig.getParameter<bool>("useFitCorrections");
    if (useFitCorrections_) {
      tokenBLMaterialMap_ = esConsumes();
      tokenBLBFieldMap_ = esConsumes();
    }
    if (hasHitMask_) {
      tokenHitMask_ = device::EDGetToken<MapToHit>(consumes(iConfig.getParameter<edm::InputTag>("hitMask")));
    }
    if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
      // Shared geometry: consume the CAGeometryRecord product instead of walking the tracker
      // geometry per run.
      tokenCAGeom_ = esConsumes();
    } else {
      iCache->tokenGeometry_ = esConsumes<edm::Transition::BeginRun>();
      iCache->tokenTopology_ = esConsumes<edm::Transition::BeginRun>();
    }
  }

  template <typename TrackerTraits>
  void CAHitNtupletAlpaka<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("siPixelRecHitsPreSplittingAlpaka"));
    desc.add<edm::InputTag>("hitMask", edm::InputTag(""))
        ->setComment(
            "Optional per-iteration hit mask (a reco::TrackingRecHitsMaskingCollection). Empty (the default) "
            "means no masking: no product is consumed and every hit is available to the CA. Set it only for the "
            "iterations that must skip the hits an earlier iteration already used.");
    desc.add<std::string>(
        "iterationName",
        std::string("promptHighPt"));  // This is just an example, it has to be changed for each tracking iteration

    Algo::fillPSetDescription(desc);
    descriptions.addWithDefaultLabel(desc);
  }

  template <typename TrackerTraits>
  void CAHitNtupletAlpaka<TrackerTraits>::acquire(device::Event const& iEvent, device::EventSetup const& es) {
    // Release any state left from a previous event whose produce() was skipped, so its buffers are
    // destroyed before new work is enqueued rather than at an arbitrary later point, when their
    // producing queue may already be recycled.
    pending_.reset();
    auto bf = 1. / es.getData(tokenField_).inverseBzAtOriginInGeV();
    // BL-fit material map and (Bz,Br)/Bz(0,0) r-z field map, device-resident EventSetup conditions
    // copied once per IOV. Read only under useFitCorrections; null otherwise, which selects the
    // fit's scalar-field, flat-material path.
    const float* rhoMapDevice = useFitCorrections_ ? es.getData(tokenBLMaterialMap_).data() : nullptr;
    const float* bMapDevice = useFitCorrections_ ? es.getData(tokenBLBFieldMap_).data() : nullptr;

    // Two handles onto the CA geometry SoA: the per-iteration configuration (graph and cuts) from
    // the run cache, and the geometry (layers and modules) from wherever this topology takes it. For
    // every topology but Phase2OTStubs they are the same object.
    auto const& caCuts = runCache()->geometry_.get(iEvent.queue());
    auto const& caGeometry = geometryHandle(es, caCuts);
    auto const& hits = iEvent.get(tokenHit_);

    /// Don't bother if no hits on BPix1 and no good graph for that
    /// (so no staring pair without BPix1 as first layer).
    /// TODO: this could be extended to a more general check for
    /// no hits on any of the starting layers.
    noBPix1_ = !(globalCache()->startNoBPix1_ or hits.offsetBPIX2() > 0);
    if (noBPix1_) {
      pending_.reset();
      return;  // produce emplaces the empty collection
    }

    std::array<double, 1> nHitsV = {{double(hits.nHits())}};
    std::array<double, 1> emptyV;

    // Floor the hit-dependent buffer sizes. The sizing formulas are fitted to collision occupancy,
    // where the doublet count grows quadratically; extrapolated downwards they undershoot the demand
    // of a quiet event, where nearly every doublet is a genuine track segment. The floors are
    // configuration parameters sized to that demand with a margin, and the remaining tail is
    // truncated and counted by the overflow sentinel. A floor of zero would also collapse the
    // capacity-bound launch block counts to an invalid 0-block configuration.
    uint32_t const maxTuples = std::max<uint32_t>(maxNumberOfTuples_.evaluate(nHitsV, emptyV), minNumberOfTuples_);
    uint32_t const maxDoublets =
        std::max<uint32_t>(maxNumberOfDoublets_.evaluate(nHitsV, emptyV), minNumberOfDoublets_);

#ifdef CA_PIPELINE_COUNTERS
    printf("[CA Pipeline] Event: run=%u lumi=%u event=%llu nHits=%u\n",
           iEvent.id().run(),
           iEvent.id().luminosityBlock(),
           (unsigned long long)iEvent.id().event(),
           hits.nHits());
#endif

    // Optional mask: with no mask module configured the view stays default-constructed (null column,
    // zero rows) and the doublet kernels skip every mask lookup.
    MapToHitConstView maskView;
    if (hasHitMask_) {
      maskView = iEvent.get(tokenHitMask_).view();
      // The doublet kernels index the mask with merged-hit global indices, guarding only on the
      // view being non-empty, so a `hitMask` tag pointing at a mask built for a different hit
      // collection would read out of range on device.
      if (maskView.metadata().size() != int(hits.nHits()))
        throw cms::Exception("CAHitMaskMismatch")
            << "CAHitNtupletAlpaka: the configured `hitMask` has " << maskView.metadata().size()
            << " rows but the hit collection has " << hits.nHits()
            << ". The mask must be the one built for this hit collection.";
    }

    // The whole CA build (hit prep, doublets, connect, ntuplets) plus one async D2H of the
    // tuple-multiplicity offsets. No blocking wait: produce runs only after this queue has drained.
    pending_ = deviceAlgo_.beginTuplesAsync(
        hits, caGeometry, caCuts, bf, maxDoublets, maxTuples, maskView, iEvent.queue(), rhoMapDevice, bMapDevice);
  }

  template <typename TrackerTraits>
  void CAHitNtupletAlpaka<TrackerTraits>::produce(device::Event& iEvent, device::EventSetup const& es) {
    if (noBPix1_) {
      edm::LogWarning("CAHitNtupletAlpaka") << "No hit on BPix1 and all the starting pairs have BPix1 as inner "
                                            << "layer.\nIt's useless to run the CA. Returning with 0 tracks!";
      auto& queue = iEvent.queue();
      reco::TracksSoACollection tracks(queue, 0, 0);
      auto ntracks_d = cms::alpakatools::make_device_view(queue, tracks.view().tracks().nTracks());
      alpaka::memset(queue, ntracks_d, 0);
      iEvent.emplace(tokenTrack_, std::move(tracks));
#ifdef CA_TRIPLET_DUMP
      // No CA was run -> no triplets; still emit an empty (0-row) dump so the product is present.
      iEvent.emplace(tokenTripletDump_, TripletDumpSoACollection(queue, 0u));
#endif
      return;
    }

    // Fit passes and classification, consuming the offsets that landed across the seam with no
    // readback or wait. The fit passes read the `modules` block only; the cut blocks are not touched
    // downstream of the build.
    auto const& caCuts = runCache()->geometry_.get(iEvent.queue());
    auto const& caGeometry = geometryHandle(es, caCuts);
    auto const& hits = iEvent.get(tokenHit_);
    iEvent.emplace(tokenTrack_, deviceAlgo_.finishTuplesAsync(std::move(*pending_), hits, caGeometry, iEvent.queue()));
    pending_.reset();

#ifdef CA_TRIPLET_DUMP
    // The per-built-triplet dump is sized to the full triplet capacity; valid rows are
    // [0, view().nValid()). An event too sparse to allocate it emplaces an empty collection, so the
    // product is always present.
    if (deviceAlgo_.tripletDumpBuffer().has_value()) {
      iEvent.emplace(tokenTripletDump_, std::move(*deviceAlgo_.tripletDumpBuffer()));
      deviceAlgo_.tripletDumpBuffer().reset();
    } else {
      iEvent.emplace(tokenTripletDump_, TripletDumpSoACollection(iEvent.queue(), 0u));
    }
#endif
  }

  using CAHitNtupletAlpakaPhase1 = CAHitNtupletAlpaka<pixelTopology::Phase1>;
  using CAHitNtupletAlpakaHIonPhase1 = CAHitNtupletAlpaka<pixelTopology::HIonPhase1>;
  using CAHitNtupletAlpakaPhase2 = CAHitNtupletAlpaka<pixelTopology::Phase2>;
  using CAHitNtupletAlpakaPhase2OT = CAHitNtupletAlpaka<pixelTopology::Phase2OT>;
  using CAHitNtupletAlpakaPhase2OTStubs = CAHitNtupletAlpaka<pixelTopology::Phase2OTStubs>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"

DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaPhase1);
DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaHIonPhase1);
DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaPhase2);
DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaPhase2OT);
DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaPhase2OTStubs);
