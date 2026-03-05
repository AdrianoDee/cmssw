// Phase2OTValidateCADoubletCuts - DQM validation of CA doublet cut variables
//
// Monitors distributions of all CA cut variables for hit pairs in compatible
// detector layers, separated by Genuine/Fake truth classification,
// Flat/Tilted sensor type (OT barrel), and layer pair category.

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/approx_atan2.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsSoA.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

#include "TH1F.h"

class Phase2OTValidateCADoubletCuts : public DQMEDAnalyzer {
public:
  explicit Phase2OTValidateCADoubletCuts(const edm::ParameterSet& iConfig);
  ~Phase2OTValidateCADoubletCuts() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  // --- Histogram bundle for one (truth, pair, tilt) combination ---
  struct PairHistos {
    MonitorElement* h_z0 = nullptr;
    MonitorElement* h_dz = nullptr;
    MonitorElement* h_innerCoord = nullptr;
    MonitorElement* h_outerCoord = nullptr;
    MonitorElement* h_dphi = nullptr;
    MonitorElement* h_dr = nullptr;
    MonitorElement* h_kappaSig = nullptr;
  };

  void bookPairHistos(DQMStore::IBooker& iBooker, PairHistos& h, bool bookKappa);

  // --- Layer pair descriptor ---
  struct PairDesc {
    int pairIdx;             // index in layerPairsPhase2OTStubs
    uint8_t innerCA;         // inner CA layer
    uint8_t outerCA;         // outer CA layer
    std::string category;    // folder: OT_Barrel, OT_Barrel_to_FwdDisk, etc.
    std::string name;        // subfolder: L1_L2, D1F_D2F, BPix3_L1, etc.
    bool isBarrelBarrel;     // both OT barrel → tilt sub-classification
    bool innerIsPixel;       // inner hit is pixel (IT-OT pairs)
    bool hasKappaCut;        // stub-stub kappa significance applicable
  };

  void buildPairTable();

  // --- Tilt categories (OT barrel-barrel only) ---
  enum TiltCat { FF = 0, FT = 1, TT = 2, NoTilt = -1 };
  static const char* tiltName(TiltCat t) {
    static const char* n[] = {"FF", "FT", "TT"};
    return (t >= 0) ? n[t] : "";
  }

  // --- Truth categories ---
  enum TruthCat { Genuine = 0, Fake = 1 };
  static const char* truthName(TruthCat t) { return t == Genuine ? "Genuine" : "Fake"; }

  // --- Stub classification (reused from Phase2OTValidateVectorHitStyleStub) ---
  enum StubClass { StubGenuine = 0, StubCombinatoric = 1, StubUnknown = 2, StubPHitOnly = 3 };

  struct StubTruthInfo {
    StubClass classification;
    edm::Ptr<TrackingParticle> principalTP;
  };

  // --- Pixel hit info (for IT-OT pairs) ---
  struct PixelHitInfo {
    float rGlobal;
    float zGlobal;
    int16_t iphi;
    edm::Ptr<TrackingParticle> principalTP;
  };

  // Key for histogram map: (truthCat, pairDescIndex, tiltCat)
  using HistoKey = std::tuple<int, int, int>;

  // --- Data members ---
  std::string topFolder_;

  edm::EDGetTokenT<reco::StubsHost> stubsToken_;
  edm::EDGetTokenT<reco::OTRecHitsHost> otRecHitsToken_;
  edm::EDGetTokenT<Phase2TrackerRecHit1DCollectionNew> recHitCollectionToken_;
  edm::EDGetTokenT<SiPixelRecHitCollection> pixelRecHitToken_;
  edm::EDGetTokenT<std::vector<TrackingParticle>> tpToken_;

  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;

  TrackerHitAssociator::Config hitAssocConfig_;

  std::vector<PairDesc> pairDescs_;
  std::map<HistoKey, PairHistos> histos_;
};

// ============================================================
// Build the pair descriptor table from the layerPairsPhase2OTStubs topology
// ============================================================
void Phase2OTValidateCADoubletCuts::buildPairTable() {
  auto otLayerName = [](uint8_t caLayer) -> std::string {
    if (caLayer >= 28 && caLayer <= 33)
      return "L" + std::to_string(caLayer - 27);
    if (caLayer >= 39 && caLayer <= 43)
      return "D" + std::to_string(caLayer - 38) + "F";
    if (caLayer >= 34 && caLayer <= 38)
      return "D" + std::to_string(caLayer - 33) + "B";
    return "?";
  };

  auto pixelLayerName = [](uint8_t caLayer) -> std::string {
    if (caLayer <= 3)
      return "BPix" + std::to_string(caLayer + 1);
    if (caLayer >= 4 && caLayer <= 15)
      return "FPix_D" + std::to_string(caLayer - 3);
    if (caLayer >= 16 && caLayer <= 27)
      return "BPix_D" + std::to_string(caLayer - 15);
    return "?";
  };

  auto isOTBarrel = [](uint8_t caLayer) { return caLayer >= 28 && caLayer <= 33; };
  auto isOTFwdDisk = [](uint8_t caLayer) { return caLayer >= 39 && caLayer <= 43; };
  auto isOTBwdDisk = [](uint8_t caLayer) { return caLayer >= 34 && caLayer <= 38; };
  auto isPixel = [](uint8_t caLayer) { return caLayer <= 27; };

  // Iterate over pairs that involve at least one OT layer (indices 57 onward)
  for (int p = 57; p < phase2PixelTopology::nPairsPhase2OTStubs; ++p) {
    uint8_t innerCA = phase2PixelTopology::layerPairsPhase2OTStubs[2 * p];
    uint8_t outerCA = phase2PixelTopology::layerPairsPhase2OTStubs[2 * p + 1];

    PairDesc desc;
    desc.pairIdx = p;
    desc.innerCA = innerCA;
    desc.outerCA = outerCA;
    desc.innerIsPixel = isPixel(innerCA);
    desc.isBarrelBarrel = isOTBarrel(innerCA) && isOTBarrel(outerCA);
    desc.hasKappaCut = (phase2PixelTopology::stubSigmaCutsPhase2OTStubs[p] > 0.f);

    // Determine category and name
    if (desc.isBarrelBarrel) {
      desc.category = "OT_Barrel";
      desc.name = otLayerName(innerCA) + "_" + otLayerName(outerCA);
    } else if (isOTBarrel(innerCA) && isOTFwdDisk(outerCA)) {
      desc.category = "OT_Barrel_to_FwdDisk";
      desc.name = otLayerName(innerCA) + "_" + otLayerName(outerCA);
    } else if (isOTBarrel(innerCA) && isOTBwdDisk(outerCA)) {
      desc.category = "OT_Barrel_to_BwdDisk";
      desc.name = otLayerName(innerCA) + "_" + otLayerName(outerCA);
    } else if (isOTFwdDisk(innerCA) && isOTFwdDisk(outerCA)) {
      desc.category = "OT_FwdDisk_Chain";
      desc.name = otLayerName(innerCA) + "_" + otLayerName(outerCA);
    } else if (isOTBwdDisk(innerCA) && isOTBwdDisk(outerCA)) {
      desc.category = "OT_BwdDisk_Chain";
      desc.name = otLayerName(innerCA) + "_" + otLayerName(outerCA);
    } else if (isPixel(innerCA) && isOTBarrel(outerCA)) {
      desc.category = "IT_to_OT_Barrel";
      desc.name = pixelLayerName(innerCA) + "_" + otLayerName(outerCA);
    } else if (isPixel(innerCA) && isOTFwdDisk(outerCA)) {
      desc.category = "IT_to_OT_FwdDisk";
      desc.name = pixelLayerName(innerCA) + "_" + otLayerName(outerCA);
    } else if (isPixel(innerCA) && isOTBwdDisk(outerCA)) {
      desc.category = "IT_to_OT_BwdDisk";
      desc.name = pixelLayerName(innerCA) + "_" + otLayerName(outerCA);
    } else {
      continue;  // skip unexpected combinations
    }

    pairDescs_.push_back(desc);
  }
}

// ============================================================
// Constructor
// ============================================================
Phase2OTValidateCADoubletCuts::Phase2OTValidateCADoubletCuts(const edm::ParameterSet& iConfig)
    : topFolder_(iConfig.getParameter<std::string>("TopFolderName")),
      stubsToken_(consumes<reco::StubsHost>(iConfig.getParameter<edm::InputTag>("stubsSrc"))),
      otRecHitsToken_(consumes<reco::OTRecHitsHost>(iConfig.getParameter<edm::InputTag>("otRecHitsSrc"))),
      recHitCollectionToken_(
          consumes<Phase2TrackerRecHit1DCollectionNew>(iConfig.getParameter<edm::InputTag>("recHitSrc"))),
      pixelRecHitToken_(consumes<SiPixelRecHitCollection>(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"))),
      tpToken_(consumes<std::vector<TrackingParticle>>(iConfig.getParameter<edm::InputTag>("trackingParticleSrc"))),
      geomToken_(esConsumes()),
      topoToken_(esConsumes()),
      hitAssocConfig_(iConfig.getParameter<edm::ParameterSet>("hitAssociatorConfig"), consumesCollector()) {
  buildPairTable();
}

// ============================================================
// Book a set of histograms for one (truth, pair, tilt) combination
// ============================================================
void Phase2OTValidateCADoubletCuts::bookPairHistos(DQMStore::IBooker& iBooker, PairHistos& h, bool bookKappa) {
  h.h_z0 = iBooker.book1D("h_z0", "z_{0} compatibility;|z_{i}r_{o} - r_{i}z_{o}|/dr [cm];Pairs", 200, 0, 60);
  h.h_dz = iBooker.book1D("h_dz", "Signed #Deltaz;z_{o} - z_{i} [cm];Pairs", 200, -300, 300);
  h.h_innerCoord =
      iBooker.book1D("h_innerCoord", "Inner coordinate;z_{i} or r_{i} [cm];Pairs", 200, -300, 300);
  h.h_outerCoord =
      iBooker.book1D("h_outerCoord", "Outer coordinate;z_{o} or r_{o} [cm];Pairs", 200, -300, 300);
  h.h_dphi = iBooker.book1D("h_dphi", "Integer #Delta#phi;|#Delta#phi| [int16 units];Pairs", 200, 0, 4000);
  h.h_dr = iBooker.book1D("h_dr", "Radial separation;r_{o} - r_{i} [cm];Pairs", 200, -10, 200);
  if (bookKappa)
    h.h_kappaSig = iBooker.book1D("h_kappaSig", "#kappa significance;|#kappa_{i} - #kappa_{o}|/#sigma;Pairs", 200, 0, 20);
}

// ============================================================
// bookHistograms
// ============================================================
void Phase2OTValidateCADoubletCuts::bookHistograms(DQMStore::IBooker& iBooker,
                                                    edm::Run const&,
                                                    edm::EventSetup const&) {
  for (int truth = 0; truth <= 1; ++truth) {
    for (size_t ip = 0; ip < pairDescs_.size(); ++ip) {
      const auto& pd = pairDescs_[ip];

      if (pd.isBarrelBarrel) {
        // Book with tilt sub-folders
        for (int t = FF; t <= TT; ++t) {
          std::string folder = topFolder_ + "/" + truthName(static_cast<TruthCat>(truth)) + "/" + pd.category + "/" +
                               pd.name + "/" + tiltName(static_cast<TiltCat>(t));
          iBooker.setCurrentFolder(folder);
          HistoKey key(truth, ip, t);
          bookPairHistos(iBooker, histos_[key], pd.hasKappaCut);
        }
      } else {
        // No tilt sub-classification
        std::string folder =
            topFolder_ + "/" + truthName(static_cast<TruthCat>(truth)) + "/" + pd.category + "/" + pd.name;
        iBooker.setCurrentFolder(folder);
        HistoKey key(truth, ip, NoTilt);
        bookPairHistos(iBooker, histos_[key], pd.hasKappaCut);
      }
    }
  }
}

// ============================================================
// analyze
// ============================================================
void Phase2OTValidateCADoubletCuts::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // --- Get products ---
  const auto stubsHandle = iEvent.getHandle(stubsToken_);
  if (!stubsHandle.isValid())
    return;
  const auto otHitsHandle = iEvent.getHandle(otRecHitsToken_);
  if (!otHitsHandle.isValid())
    return;
  const auto recHitsHandle = iEvent.getHandle(recHitCollectionToken_);
  if (!recHitsHandle.isValid())
    return;
  const auto pixelRecHitsHandle = iEvent.getHandle(pixelRecHitToken_);
  if (!pixelRecHitsHandle.isValid())
    return;

  edm::Handle<std::vector<TrackingParticle>> tpHandle;
  iEvent.getByToken(tpToken_, tpHandle);
  if (!tpHandle.isValid())
    return;

  const auto& stubs = *stubsHandle;
  const auto& otHits = *otHitsHandle;
  const auto& recHits = *recHitsHandle;
  const auto& trackerGeom = iSetup.getData(geomToken_);
  const auto& trackerTopo = iSetup.getData(topoToken_);

  auto stubsView = stubs.view().stubs();
  auto hitsView = otHits.view().otRecHits();
  uint32_t nStubs = stubs.nStubs();

  // --- Create TrackerHitAssociator ---
  TrackerHitAssociator hitAssociator(iEvent, hitAssocConfig_);

  // --- Build flat vector of original OT RecHit pointers ---
  std::vector<const Phase2TrackerRecHit1D*> origRecHits;
  origRecHits.reserve(nStubs * 2);
  for (const auto& detSet : recHits) {
    for (const auto& recHit : detSet) {
      origRecHits.push_back(&recHit);
    }
  }

  // --- Build (simTrackId, eventId) -> TrackingParticle map ---
  std::map<SimHitIdpr, edm::Ptr<TrackingParticle>> simTrackToTP;
  for (size_t iTP = 0; iTP < tpHandle->size(); ++iTP) {
    edm::Ptr<TrackingParticle> tpPtr(tpHandle, iTP);
    for (const auto& g4Track : tpPtr->g4Tracks()) {
      simTrackToTP[{g4Track.trackId(), g4Track.eventId()}] = tpPtr;
    }
  }

  // --- Classify each stub ---
  std::vector<StubTruthInfo> stubTruth(nStubs);

  // Hoist temporary containers out of loop to avoid per-iteration heap allocations
  std::vector<SimHitIdpr> ids;
  std::set<SimHitIdpr> innerIds, outerIds;
  std::set<edm::Ptr<TrackingParticle>> innerTPs, outerTPs;
  std::vector<edm::Ptr<TrackingParticle>> commonTPs;

  for (uint32_t iStub = 0; iStub < nStubs; ++iStub) {
    uint8_t stubType = stubsView[iStub].stubType();

    if (stubType == reco::StubType::PHitOnly) {
      stubTruth[iStub] = {StubPHitOnly, edm::Ptr<TrackingParticle>()};
      continue;
    }

    uint32_t innerSoAIdx = stubsView[iStub].innerHitIdx();
    uint32_t outerSoAIdx = stubsView[iStub].outerHitIdx();
    uint32_t innerOrigIdx = hitsView[innerSoAIdx].origRecHitIdx();
    uint32_t outerOrigIdx = hitsView[outerSoAIdx].origRecHitIdx();

    innerIds.clear();
    outerIds.clear();
    if (innerOrigIdx < origRecHits.size()) {
      ids.clear();
      hitAssociator.associatePhase2TrackerRecHit(origRecHits[innerOrigIdx], ids);
      innerIds.insert(ids.begin(), ids.end());
    }
    if (outerOrigIdx < origRecHits.size()) {
      ids.clear();
      hitAssociator.associatePhase2TrackerRecHit(origRecHits[outerOrigIdx], ids);
      outerIds.insert(ids.begin(), ids.end());
    }

    innerTPs.clear();
    outerTPs.clear();
    for (const auto& id : innerIds) {
      auto it = simTrackToTP.find(id);
      if (it != simTrackToTP.end())
        innerTPs.insert(it->second);
    }
    for (const auto& id : outerIds) {
      auto it = simTrackToTP.find(id);
      if (it != simTrackToTP.end())
        outerTPs.insert(it->second);
    }

    commonTPs.clear();
    std::set_intersection(
        innerTPs.begin(), innerTPs.end(), outerTPs.begin(), outerTPs.end(), std::back_inserter(commonTPs));

    if (innerIds.empty() && outerIds.empty()) {
      stubTruth[iStub] = {StubUnknown, edm::Ptr<TrackingParticle>()};
    } else if (!commonTPs.empty()) {
      stubTruth[iStub] = {StubGenuine, commonTPs.front()};
    } else {
      stubTruth[iStub] = {StubCombinatoric, edm::Ptr<TrackingParticle>()};
    }
  }

  // --- Build per-CA-layer stub index lists ---
  constexpr int nCALayers = phase2PixelTopology::nLayersPhase2OTStubs;
  std::vector<std::vector<uint32_t>> stubsByLayer(nCALayers);

  for (uint32_t iStub = 0; iStub < nStubs; ++iStub) {
    uint8_t stubType = stubsView[iStub].stubType();
    if (stubType == reco::StubType::PHitOnly)
      continue;

    uint8_t flags = stubsView[iStub].flags();
    if (!reco::StubFlags::isValid(flags))
      continue;

    uint8_t otLayer = reco::StubFlags::layer(flags);
    bool isBarrel = reco::StubFlags::isBarrel(flags);
    float z = stubsView[iStub].zGlobal();

    int caLayer;
    if (isBarrel)
      caLayer = 28 + (otLayer - 1);
    else if (z > 0)
      caLayer = 39 + (otLayer - 1);
    else
      caLayer = 34 + (otLayer - 1);

    if (caLayer >= 0 && caLayer < nCALayers)
      stubsByLayer[caLayer].push_back(iStub);
  }

  // --- Build per-CA-layer pixel hit lists (for IT-OT pairs) ---
  std::vector<std::vector<uint32_t>> pixelHitsByLayer(nCALayers);
  std::vector<PixelHitInfo> pixelHits;
  pixelHits.reserve(nStubs);

  for (const auto& detSet : *pixelRecHitsHandle) {
    DetId detId = detSet.detId();
    const auto* detUnit = trackerGeom.idToDetUnit(detId);
    if (!detUnit)
      continue;

    for (const auto& pixHit : detSet) {
      auto gp = detUnit->toGlobal(pixHit.localPosition());
      float r = gp.perp();
      float z = gp.z();
      float phi = gp.barePhi();
      int16_t iphi = phi2short(phi);

      // Truth-match pixel hit
      edm::Ptr<TrackingParticle> tp;
      std::vector<SimHitIdpr> ids = hitAssociator.associateHitId(pixHit);
      for (const auto& id : ids) {
        auto it = simTrackToTP.find(id);
        if (it != simTrackToTP.end()) {
          tp = it->second;
          break;
        }
      }

      uint32_t pixIdx = pixelHits.size();
      pixelHits.push_back({r, z, iphi, tp});

      // Determine CA layer for this pixel hit
      int caLayer = -1;
      if (detId.subdetId() == 1) {  // PixelSubdetector::PixelBarrel
        int layer = trackerTopo.pxbLayer(detId);
        if (layer >= 1 && layer <= 4)
          caLayer = layer - 1;
      } else if (detId.subdetId() == 2) {  // PixelSubdetector::PixelEndcap
        int disk = trackerTopo.pxfDisk(detId);
        if (z > 0)
          caLayer = 3 + disk;  // 4-15
        else
          caLayer = 15 + disk;  // 16-27
      }
      if (caLayer >= 0 && caLayer < nCALayers)
        pixelHitsByLayer[caLayer].push_back(pixIdx);
    }
  }

  // --- Thread-local histogram accumulation to avoid MonitorElement spinlock contention ---
  // DQMEDAnalyzer is a stream module: multiple streams share the same MonitorElement objects.
  // Accumulate into local TH1F (zero locking), then merge after each pair descriptor's N×M loop.
  struct LocalPairHistos {
    TH1F h_z0{"", "", 200, 0, 60};
    TH1F h_dz{"", "", 200, -300, 300};
    TH1F h_innerCoord{"", "", 200, -300, 300};
    TH1F h_outerCoord{"", "", 200, -300, 300};
    TH1F h_dphi{"", "", 200, 0, 4000};
    TH1F h_dr{"", "", 200, -10, 200};
    TH1F h_kappaSig{"", "", 200, 0, 20};
    LocalPairHistos() {
      h_z0.SetDirectory(nullptr);
      h_dz.SetDirectory(nullptr);
      h_innerCoord.SetDirectory(nullptr);
      h_outerCoord.SetDirectory(nullptr);
      h_dphi.SetDirectory(nullptr);
      h_dr.SetDirectory(nullptr);
      h_kappaSig.SetDirectory(nullptr);
    }
    void reset() {
      h_z0.Reset();
      h_dz.Reset();
      h_innerCoord.Reset();
      h_outerCoord.Reset();
      h_dphi.Reset();
      h_dr.Reset();
      h_kappaSig.Reset();
    }
  };

  // 2 truth categories (Genuine, Fake) × 3 tilt categories (FF, FT, TT / NoTilt mapped to [0])
  LocalPairHistos localHistos[2][3];

  auto mergeLocalIntoME = [](TH1F& local, MonitorElement* me) {
    if (!me)
      return;
    for (int bin = 0; bin <= local.GetNbinsX() + 1; ++bin) {
      double content = local.GetBinContent(bin);
      if (content > 0)
        me->Fill(local.GetBinCenter(bin), content);
    }
  };

  // --- Cached stub data for contiguous memory access in inner loops ---
  struct CachedOTStub {
    float r, z, dPhiDr, dPhiDrError;
    int16_t iphi;
    uint8_t flags;
    StubClass classification;
    edm::Ptr<TrackingParticle> principalTP;
  };
  std::vector<CachedOTStub> outerCache;

  // --- Iterate over all pair descriptors and fill histograms ---
  for (size_t ip = 0; ip < pairDescs_.size(); ++ip) {
    const auto& pd = pairDescs_[ip];

    // Pre-resolve histogram pointers to avoid map lookups in inner loop
    PairHistos* resolvedHistos[2][3] = {};
    if (pd.isBarrelBarrel) {
      for (int tr = 0; tr < 2; ++tr)
        for (int t = FF; t <= TT; ++t) {
          auto it = histos_.find(HistoKey(tr, static_cast<int>(ip), t));
          if (it != histos_.end())
            resolvedHistos[tr][t] = &it->second;
        }
    } else {
      for (int tr = 0; tr < 2; ++tr) {
        auto it = histos_.find(HistoKey(tr, static_cast<int>(ip), NoTilt));
        if (it != histos_.end())
          resolvedHistos[tr][0] = &it->second;
      }
    }

    // Pre-compute whether the outer layer is barrel (loop-invariant)
    bool outerIsBarrel = phase2PixelTopology::isBarrelPhase2OTStubs[pd.outerCA];

    // Reset local histograms for this pair descriptor
    for (int tr = 0; tr < 2; ++tr)
      for (int t = 0; t < 3; ++t)
        localHistos[tr][t].reset();

    if (pd.innerIsPixel) {
      // IT-OT pair: inner = pixel hit, outer = stub
      const auto& innerHits = pixelHitsByLayer[pd.innerCA];
      const auto& outerStubs = stubsByLayer[pd.outerCA];

      // Cache outer stub data for contiguous memory access
      outerCache.clear();
      outerCache.reserve(outerStubs.size());
      for (uint32_t oStub : outerStubs) {
        outerCache.push_back({stubsView[oStub].rGlobal(),
                              stubsView[oStub].zGlobal(),
                              0.f,
                              0.f,
                              stubsView[oStub].iphi(),
                              0,
                              stubTruth[oStub].classification,
                              stubTruth[oStub].principalTP});
      }

      for (uint32_t pixIdx : innerHits) {
        const auto& pix = pixelHits[pixIdx];
        float ri = pix.rGlobal;
        float zi = pix.zGlobal;
        int16_t mep = pix.iphi;

        for (size_t oc = 0; oc < outerCache.size(); ++oc) {
          const auto& os = outerCache[oc];
          float ro = os.r;
          float zo = os.z;
          float dr = ro - ri;
          float dz = zo - zi;
          float z0 = (dr > 0) ? std::abs(zi * ro - ri * zo) / dr : 0;
          int16_t mop = os.iphi;
          uint16_t idphi = std::min(std::abs(int16_t(mop - mep)), std::abs(int16_t(mep - mop)));

          // Truth classification for the pair
          bool pairGenuine = false;
          if (pix.principalTP.isNonnull() && os.classification == StubGenuine && os.principalTP.isNonnull()) {
            pairGenuine = (pix.principalTP == os.principalTP);
          }

          int truthIdx = pairGenuine ? Genuine : Fake;

          // Inner coordinate: z for barrel outer, r for endcap outer
          float valInner = outerIsBarrel ? zi : ri;
          float valOuter = outerIsBarrel ? zo : ro;

          auto& lh = localHistos[truthIdx][0];
          lh.h_z0.Fill(z0);
          lh.h_dz.Fill(dz);
          lh.h_innerCoord.Fill(valInner);
          lh.h_outerCoord.Fill(valOuter);
          lh.h_dphi.Fill(idphi);
          lh.h_dr.Fill(dr);
        }
      }
    } else {
      // OT-OT pair: both stubs
      const auto& innerStubs = stubsByLayer[pd.innerCA];
      const auto& outerStubs = stubsByLayer[pd.outerCA];

      // Cache outer stub data for contiguous memory access
      outerCache.clear();
      outerCache.reserve(outerStubs.size());
      for (uint32_t oStub : outerStubs) {
        outerCache.push_back({stubsView[oStub].rGlobal(),
                              stubsView[oStub].zGlobal(),
                              stubsView[oStub].dPhiDr(),
                              stubsView[oStub].dPhiDrError(),
                              stubsView[oStub].iphi(),
                              stubsView[oStub].flags(),
                              stubTruth[oStub].classification,
                              stubTruth[oStub].principalTP});
      }

      for (uint32_t iStub : innerStubs) {
        float ri = stubsView[iStub].rGlobal();
        float zi = stubsView[iStub].zGlobal();
        int16_t mep = stubsView[iStub].iphi();
        uint8_t iFlags = stubsView[iStub].flags();
        bool iFlat = reco::StubFlags::isFlat(iFlags);

        // Hoist inner stub truth info outside inner loop
        StubClass iClass = stubTruth[iStub].classification;
        const auto& iTP = stubTruth[iStub].principalTP;
        bool iIsGenuineWithTP = (iClass == StubGenuine && iTP.isNonnull());

        // Hoist inner stub kappa pre-computation outside inner loop
        float k_i = 0.f, sk_i = 0.f;
        bool hasInnerKappa = false;
        if (pd.hasKappaCut) {
          float d_i = stubsView[iStub].dPhiDr(), s_i = stubsView[iStub].dPhiDrError();
          float den_i = 1.f + ri * ri * d_i * d_i;
          float sqrt_den_i = std::sqrt(den_i);
          k_i = d_i / sqrt_den_i;
          sk_i = s_i / (den_i * sqrt_den_i);
          hasInnerKappa = true;
        }

        for (size_t oc = 0; oc < outerCache.size(); ++oc) {
          const auto& os = outerCache[oc];
          float ro = os.r;
          float zo = os.z;
          float dr = ro - ri;
          float dz = zo - zi;
          float z0 = (dr > 0) ? std::abs(zi * ro - ri * zo) / dr : 0;
          int16_t mop = os.iphi;
          uint16_t idphi = std::min(std::abs(int16_t(mop - mep)), std::abs(int16_t(mep - mop)));

          // Kappa significance (stub-stub pairs only)
          float kappaSig = -1.f;
          if (hasInnerKappa && dr > 0) {
            float d_o = os.dPhiDr, s_o = os.dPhiDrError;
            float den_o = 1.f + ro * ro * d_o * d_o;
            float sqrt_den_o = std::sqrt(den_o);
            float k_o = d_o / sqrt_den_o;
            float sk_o = s_o / (den_o * sqrt_den_o);

            float combined_err2 = sk_i * sk_i + sk_o * sk_o;
            if (combined_err2 > 0)
              kappaSig = std::abs(k_i - k_o) / std::sqrt(combined_err2);
          }

          // Truth classification for the pair
          bool pairGenuine = false;
          if (iIsGenuineWithTP && os.classification == StubGenuine && os.principalTP.isNonnull()) {
            pairGenuine = (iTP == os.principalTP);
          }

          int truthIdx = pairGenuine ? Genuine : Fake;

          // Determine tilt category (OT barrel-barrel only)
          int tiltLookup;
          if (pd.isBarrelBarrel) {
            bool oFlat = reco::StubFlags::isFlat(os.flags);
            if (iFlat && oFlat)
              tiltLookup = FF;
            else if (iFlat != oFlat)
              tiltLookup = FT;
            else
              tiltLookup = TT;
          } else {
            tiltLookup = 0;
          }

          // Inner/outer coordinate: z for barrel pair, r for endcap pair
          float valInner = outerIsBarrel ? zi : ri;
          float valOuter = outerIsBarrel ? zo : ro;

          auto& lh = localHistos[truthIdx][tiltLookup];
          lh.h_z0.Fill(z0);
          lh.h_dz.Fill(dz);
          lh.h_innerCoord.Fill(valInner);
          lh.h_outerCoord.Fill(valOuter);
          lh.h_dphi.Fill(idphi);
          lh.h_dr.Fill(dr);
          if (kappaSig >= 0)
            lh.h_kappaSig.Fill(kappaSig);
        }
      }
    }

    // Merge local histograms into MonitorElements (batched, ~200 Fill() calls instead of millions)
    for (int tr = 0; tr < 2; ++tr)
      for (int t = 0; t < 3; ++t)
        if (resolvedHistos[tr][t]) {
          auto& lh = localHistos[tr][t];
          auto* rh = resolvedHistos[tr][t];
          mergeLocalIntoME(lh.h_z0, rh->h_z0);
          mergeLocalIntoME(lh.h_dz, rh->h_dz);
          mergeLocalIntoME(lh.h_innerCoord, rh->h_innerCoord);
          mergeLocalIntoME(lh.h_outerCoord, rh->h_outerCoord);
          mergeLocalIntoME(lh.h_dphi, rh->h_dphi);
          mergeLocalIntoME(lh.h_dr, rh->h_dr);
          mergeLocalIntoME(lh.h_kappaSig, rh->h_kappaSig);
        }
  }
}

// ============================================================
// fillDescriptions
// ============================================================
void Phase2OTValidateCADoubletCuts::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("TopFolderName", "HLT/StubCACutsCompat");

  desc.add<edm::InputTag>("stubsSrc", edm::InputTag("hltOTStubProducer"));
  desc.add<edm::InputTag>("otRecHitsSrc", edm::InputTag("hltPixelSeedingOTRecHitsSoA"));
  desc.add<edm::InputTag>("recHitSrc", edm::InputTag("hltSiPhase2RecHits"));
  desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("hltSiPixelRecHits"));
  desc.add<edm::InputTag>("trackingParticleSrc", edm::InputTag("mix", "MergedTrackTruth"));

  edm::ParameterSetDescription hitAssocDesc;
  hitAssocDesc.add<bool>("associatePixel", true);
  hitAssocDesc.add<bool>("associateStrip", true);
  hitAssocDesc.add<bool>("usePhase2Tracker", true);
  hitAssocDesc.add<bool>("associateRecoTracks", false);
  hitAssocDesc.add<bool>("associateHitbySimTrack", false);
  hitAssocDesc.add<edm::InputTag>("phase2TrackerSimLinkSrc", edm::InputTag("simSiPixelDigis", "Tracker"));
  hitAssocDesc.add<edm::InputTag>("pixelSimLinkSrc", edm::InputTag("simSiPixelDigis", "Pixel"));
  hitAssocDesc.add<edm::InputTag>("stripSimLinkSrc", edm::InputTag("simSiStripDigis"));
  hitAssocDesc.add<std::vector<std::string>>(
      "ROUList",
      {"TrackerHitsPixelBarrelLowTof",
       "TrackerHitsPixelBarrelHighTof",
       "TrackerHitsPixelEndcapLowTof",
       "TrackerHitsPixelEndcapHighTof",
       "TrackerHitsTIBLowTof",
       "TrackerHitsTIBHighTof",
       "TrackerHitsTIDLowTof",
       "TrackerHitsTIDHighTof",
       "TrackerHitsTOBLowTof",
       "TrackerHitsTOBHighTof",
       "TrackerHitsTECLowTof",
       "TrackerHitsTECHighTof"});
  desc.add<edm::ParameterSetDescription>("hitAssociatorConfig", hitAssocDesc);

  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(Phase2OTValidateCADoubletCuts);
