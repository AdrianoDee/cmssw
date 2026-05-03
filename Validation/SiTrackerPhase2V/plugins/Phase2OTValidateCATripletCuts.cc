// Phase2OTValidateCATripletCuts - DQM validation of CA triplet cut variables
//
// Monitors distributions of theta alignment and DCA quantities for triplets
// formed by compatible doublet pairs sharing a middle hit, separated by
// Genuine/Fake truth classification, layer triplet category, and SS-stub
// multiplicity.  Designed to inform thetaCut/dcaCut relaxation factors
// for SS-stub triplets.

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
#include "RecoTracker/PixelSeeding/interface/CircleEq.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

#include "TH1F.h"

class Phase2OTValidateCATripletCuts : public DQMEDAnalyzer {
public:
  explicit Phase2OTValidateCATripletCuts(const edm::ParameterSet& iConfig);
  ~Phase2OTValidateCATripletCuts() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  // --- Histogram bundle for one (truth, triplet, nSS) combination ---
  struct TripletHistos {
    MonitorElement* h_thetaNorm = nullptr;
    MonitorElement* h_dcaNorm = nullptr;
    MonitorElement* h_curvature = nullptr;
    MonitorElement* h_geomStubKappaSig = nullptr;
    MonitorElement* h_geomKappa = nullptr;
    MonitorElement* h_phiResidual = nullptr;
  };

  void bookTripletHistos(DQMStore::IBooker& iBooker, TripletHistos& h);

  // --- Triplet descriptor ---
  struct TripletDesc {
    int innerPairIdx;
    int outerPairIdx;
    uint8_t innerCA;
    uint8_t middleCA;
    uint8_t outerCA;
    std::string category;
    std::string name;
  };

  void buildTripletTable();

  // --- Truth categories ---
  enum TruthCat { Genuine = 0, Fake = 1 };
  static const char* truthName(TruthCat t) { return t == Genuine ? "Genuine" : "Fake"; }

  // --- nSS categories ---
  enum NSSCat { nSS0 = 0, nSS1 = 1, nSSge2 = 2 };
  static const char* nSSName(NSSCat n) {
    static const char* names[] = {"nSS0", "nSS1", "nSSge2"};
    return names[n];
  }

  // --- Stub classification (reused from doublet validator) ---
  enum StubClass { StubGenuine = 0, StubCombinatoric = 1, StubUnknown = 2, StubPHitOnly = 3 };

  struct StubTruthInfo {
    StubClass classification;
    edm::Ptr<TrackingParticle> principalTP;
  };

  // --- Pixel hit info (for IT-OT triplets) ---
  struct PixelHitInfo {
    float xGlobal;
    float yGlobal;
    float rGlobal;
    float zGlobal;
    int16_t iphi;
    edm::Ptr<TrackingParticle> principalTP;
  };

  // Key for histogram map: (truthCat, tripletDescIndex, nSSCat)
  using HistoKey = std::tuple<int, int, int>;

  // --- Data members ---
  std::string topFolder_;
  float ptmin_;

  edm::EDGetTokenT<reco::StubsHost> stubsToken_;
  edm::EDGetTokenT<reco::OTRecHitsHost> otRecHitsToken_;
  edm::EDGetTokenT<Phase2TrackerRecHit1DCollectionNew> recHitCollectionToken_;
  edm::EDGetTokenT<SiPixelRecHitCollection> pixelRecHitToken_;
  edm::EDGetTokenT<std::vector<TrackingParticle>> tpToken_;

  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;

  TrackerHitAssociator::Config hitAssocConfig_;

  std::vector<TripletDesc> tripletDescs_;
  std::map<HistoKey, TripletHistos> histos_;
};

// ============================================================
// Build the triplet descriptor table
// ============================================================
void Phase2OTValidateCATripletCuts::buildTripletTable() {
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

  auto layerName = [&](uint8_t caLayer) -> std::string {
    return (caLayer <= 27) ? pixelLayerName(caLayer) : otLayerName(caLayer);
  };

  auto isOTBarrel = [](uint8_t caLayer) { return caLayer >= 28 && caLayer <= 33; };
  auto isOTFwdDisk = [](uint8_t caLayer) { return caLayer >= 39 && caLayer <= 43; };
  auto isOTBwdDisk = [](uint8_t caLayer) { return caLayer >= 34 && caLayer <= 38; };
  auto isOT = [&](uint8_t caLayer) { return isOTBarrel(caLayer) || isOTFwdDisk(caLayer) || isOTBwdDisk(caLayer); };
  auto isPixel = [](uint8_t caLayer) { return caLayer <= 27; };

  auto classify = [&](uint8_t inner, uint8_t middle, uint8_t outer) -> std::string {
    bool anyPixel = isPixel(inner) || isPixel(middle) || isPixel(outer);
    if (!anyPixel) {
      // All OT
      if (isOTBarrel(inner) && isOTBarrel(middle) && isOTBarrel(outer))
        return "OT_Barrel";
      if (isOTFwdDisk(inner) && isOTFwdDisk(middle) && isOTFwdDisk(outer))
        return "OT_FwdDisk";
      if (isOTBwdDisk(inner) && isOTBwdDisk(middle) && isOTBwdDisk(outer))
        return "OT_BwdDisk";
      if (isOTBarrel(inner) && isOTBarrel(middle) && isOTFwdDisk(outer))
        return "OT_Barrel_to_FwdDisk";
      if (isOTBarrel(inner) && isOTBarrel(middle) && isOTBwdDisk(outer))
        return "OT_Barrel_to_BwdDisk";
      if (isOTBarrel(inner) && isOTFwdDisk(middle) && isOTFwdDisk(outer))
        return "OT_Barrel_to_FwdDisk";
      if (isOTBarrel(inner) && isOTBwdDisk(middle) && isOTBwdDisk(outer))
        return "OT_Barrel_to_BwdDisk";
      return "OT_Mixed";
    }
    // At least one pixel layer
    if (isOTBarrel(outer) || (isOTBarrel(middle) && !isPixel(outer)))
      return "IT_to_OT_Barrel";
    if (isOTFwdDisk(outer) || (isOTFwdDisk(middle) && !isPixel(outer)))
      return "IT_to_OT_FwdDisk";
    if (isOTBwdDisk(outer) || (isOTBwdDisk(middle) && !isPixel(outer)))
      return "IT_to_OT_BwdDisk";
    return "IT_to_OT_Mixed";
  };

  // Iterate over all pairs of layer pairs where A.outer == B.inner
  // and at least one of the three layers is OT (CA index >= 28)
  constexpr int nPairs = phase2PixelTopology::nPairsPhase2OTStubs;
  const auto* lp = phase2PixelTopology::layerPairsPhase2OTStubs;

  for (int a = 0; a < nPairs; ++a) {
    uint8_t aInner = lp[2 * a];
    uint8_t aOuter = lp[2 * a + 1];
    for (int b = 0; b < nPairs; ++b) {
      uint8_t bInner = lp[2 * b];
      uint8_t bOuter = lp[2 * b + 1];
      if (aOuter != bInner)
        continue;

      // At least one OT layer required
      if (!isOT(aInner) && !isOT(aOuter) && !isOT(bOuter))
        continue;

      TripletDesc desc;
      desc.innerPairIdx = a;
      desc.outerPairIdx = b;
      desc.innerCA = aInner;
      desc.middleCA = aOuter;
      desc.outerCA = bOuter;
      desc.category = classify(aInner, aOuter, bOuter);
      desc.name = layerName(aInner) + "_" + layerName(aOuter) + "_" + layerName(bOuter);

      tripletDescs_.push_back(desc);
    }
  }
}

// ============================================================
// Constructor
// ============================================================
Phase2OTValidateCATripletCuts::Phase2OTValidateCATripletCuts(const edm::ParameterSet& iConfig)
    : topFolder_(iConfig.getParameter<std::string>("TopFolderName")),
      ptmin_(iConfig.getParameter<double>("ptmin")),
      stubsToken_(consumes<reco::StubsHost>(iConfig.getParameter<edm::InputTag>("stubsSrc"))),
      otRecHitsToken_(consumes<reco::OTRecHitsHost>(iConfig.getParameter<edm::InputTag>("otRecHitsSrc"))),
      recHitCollectionToken_(
          consumes<Phase2TrackerRecHit1DCollectionNew>(iConfig.getParameter<edm::InputTag>("recHitSrc"))),
      pixelRecHitToken_(consumes<SiPixelRecHitCollection>(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"))),
      tpToken_(consumes<std::vector<TrackingParticle>>(iConfig.getParameter<edm::InputTag>("trackingParticleSrc"))),
      geomToken_(esConsumes()),
      topoToken_(esConsumes()),
      hitAssocConfig_(iConfig.getParameter<edm::ParameterSet>("hitAssociatorConfig"), consumesCollector()) {
  buildTripletTable();
}

// ============================================================
// Book a set of histograms for one (truth, triplet, nSS) combination
// ============================================================
void Phase2OTValidateCATripletCuts::bookTripletHistos(DQMStore::IBooker& iBooker, TripletHistos& h) {
  h.h_thetaNorm = iBooker.book1D(
      "h_thetaNorm",
      "#theta alignment normalized;#theta_{norm} (compare to thetaCut);Triplets",
      2000,
      0,
      0.5);
  h.h_dcaNorm = iBooker.book1D("h_dcaNorm", "DCA to origin;DCA [cm] (compare to dcaCut);Triplets", 2000, 0, 10.0);
  h.h_curvature = iBooker.book1D("h_curvature", "Circle curvature;|C| [1/cm];Triplets", 200, 0, 0.05);
  h.h_geomStubKappaSig = iBooker.book1D(
      "h_geomStubKappaSig",
      "Geometric-vs-stub #kappa significance;|#kappa_{geom} - #kappa_{stub}| / #sigma;Triplets",
      500,
      0,
      50.0);
  h.h_geomKappa =
      iBooker.book1D("h_geomKappa", "Geometric curvature from inner-outer;#kappa_{geom} [1/cm];Triplets", 400, -0.02, 0.02);
  h.h_phiResidual = iBooker.book1D(
      "h_phiResidual",
      "#phi residual at middle hit;#phi_{mid} - #phi_{pred} [rad];Triplets",
      400,
      -0.2,
      0.2);
}

// ============================================================
// bookHistograms
// ============================================================
void Phase2OTValidateCATripletCuts::bookHistograms(DQMStore::IBooker& iBooker,
                                                    edm::Run const&,
                                                    edm::EventSetup const&) {
  for (int truth = 0; truth <= 1; ++truth) {
    for (size_t it = 0; it < tripletDescs_.size(); ++it) {
      const auto& td = tripletDescs_[it];
      for (int nss = nSS0; nss <= nSSge2; ++nss) {
        std::string folder = topFolder_ + "/" + truthName(static_cast<TruthCat>(truth)) + "/" + td.category + "/" +
                             td.name + "/" + nSSName(static_cast<NSSCat>(nss));
        iBooker.setCurrentFolder(folder);
        HistoKey key(truth, it, nss);
        bookTripletHistos(iBooker, histos_[key]);
      }
    }
  }
}

// ============================================================
// analyze
// ============================================================
void Phase2OTValidateCATripletCuts::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
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

  // --- Build per-CA-layer pixel hit lists (for IT-OT triplets) ---
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
      float x = gp.x();
      float y = gp.y();
      float r = gp.perp();
      float z = gp.z();
      float phi = gp.barePhi();
      int16_t iphi = phi2short(phi);

      edm::Ptr<TrackingParticle> tp;
      std::vector<SimHitIdpr> pixIds = hitAssociator.associateHitId(pixHit);
      for (const auto& id : pixIds) {
        auto it = simTrackToTP.find(id);
        if (it != simTrackToTP.end()) {
          tp = it->second;
          break;
        }
      }

      uint32_t pixIdx = pixelHits.size();
      pixelHits.push_back({x, y, r, z, iphi, tp});

      int caLayer = -1;
      if (detId.subdetId() == 1) {
        int layer = trackerTopo.pxbLayer(detId);
        if (layer >= 1 && layer <= 4)
          caLayer = layer - 1;
      } else if (detId.subdetId() == 2) {
        int disk = trackerTopo.pxfDisk(detId);
        if (z > 0)
          caLayer = 3 + disk;
        else
          caLayer = 15 + disk;
      }
      if (caLayer >= 0 && caLayer < nCALayers)
        pixelHitsByLayer[caLayer].push_back(pixIdx);
    }
  }

  // --- Cached hit data for contiguous memory access ---
  struct CachedHit {
    float x, y, r, z;
    int16_t iphi;
    uint8_t stubType;  // reco::StubType value, or 0xFF for pixel hits
    float dPhiDr;       // stub direction measurement (0 for pixel hits)
    float dPhiDrError;  // stub direction error (0 for pixel hits)
    StubClass classification;
    edm::Ptr<TrackingParticle> principalTP;
  };

  // Helper: build CachedHit list for a given CA layer
  auto buildLayerHits = [&](uint8_t caLayer, std::vector<CachedHit>& out) {
    out.clear();
    bool isOT = (caLayer >= 28);
    if (isOT) {
      for (uint32_t iStub : stubsByLayer[caLayer]) {
        out.push_back({stubsView[iStub].xGlobal(),
                       stubsView[iStub].yGlobal(),
                       stubsView[iStub].rGlobal(),
                       stubsView[iStub].zGlobal(),
                       stubsView[iStub].iphi(),
                       stubsView[iStub].stubType(),
                       stubsView[iStub].dPhiDr(),
                       stubsView[iStub].dPhiDrError(),
                       stubTruth[iStub].classification,
                       stubTruth[iStub].principalTP});
      }
    } else {
      for (uint32_t pixIdx : pixelHitsByLayer[caLayer]) {
        const auto& pix = pixelHits[pixIdx];
        out.push_back({pix.xGlobal,
                       pix.yGlobal,
                       pix.rGlobal,
                       pix.zGlobal,
                       pix.iphi,
                       0xFF,
                       0.f,
                       0.f,
                       pix.principalTP.isNonnull() ? StubGenuine : StubUnknown,
                       pix.principalTP});
      }
    }
  };

  // --- Thread-local histogram accumulation to avoid MonitorElement spinlock contention ---
  struct LocalTripletHistos {
    TH1F h_thetaNorm{"", "", 2000, 0, 0.5};
    TH1F h_dcaNorm{"", "", 2000, 0, 10.0};
    TH1F h_curvature{"", "", 200, 0, 0.05};
    TH1F h_geomStubKappaSig{"", "", 500, 0, 50.0};
    TH1F h_geomKappa{"", "", 400, -0.02, 0.02};
    TH1F h_phiResidual{"", "", 400, -0.2, 0.2};
    LocalTripletHistos() {
      h_thetaNorm.SetDirectory(nullptr);
      h_dcaNorm.SetDirectory(nullptr);
      h_curvature.SetDirectory(nullptr);
      h_geomStubKappaSig.SetDirectory(nullptr);
      h_geomKappa.SetDirectory(nullptr);
      h_phiResidual.SetDirectory(nullptr);
    }
    void reset() {
      h_thetaNorm.Reset();
      h_dcaNorm.Reset();
      h_curvature.Reset();
      h_geomStubKappaSig.Reset();
      h_geomKappa.Reset();
      h_phiResidual.Reset();
    }
  };

  // 2 truth categories × 3 nSS categories
  LocalTripletHistos localHistos[2][3];

  auto mergeLocalIntoME = [](TH1F& local, MonitorElement* me) {
    if (!me)
      return;
    for (int bin = 0; bin <= local.GetNbinsX() + 1; ++bin) {
      double content = local.GetBinContent(bin);
      if (content > 0)
        me->Fill(local.GetBinCenter(bin), content);
    }
  };

  // Reusable hit cache vectors
  std::vector<CachedHit> innerHits, middleHits, outerHits;

  // --- Iterate over all triplet descriptors ---
  for (size_t it = 0; it < tripletDescs_.size(); ++it) {
    const auto& td = tripletDescs_[it];

    // Pre-resolve histogram pointers
    TripletHistos* resolvedHistos[2][3] = {};
    for (int tr = 0; tr < 2; ++tr)
      for (int nss = 0; nss < 3; ++nss) {
        auto hit = histos_.find(HistoKey(tr, static_cast<int>(it), nss));
        if (hit != histos_.end())
          resolvedHistos[tr][nss] = &hit->second;
      }

    // Reset local histograms
    for (int tr = 0; tr < 2; ++tr)
      for (int nss = 0; nss < 3; ++nss)
        localHistos[tr][nss].reset();

    // Build hit caches for the 3 layers
    buildLayerHits(td.innerCA, innerHits);
    buildLayerHits(td.middleCA, middleHits);
    buildLayerHits(td.outerCA, outerHits);

    if (middleHits.empty() || innerHits.empty() || outerHits.empty())
      continue;

    int16_t innerPhiCut = phase2PixelTopology::maxDPhiPhase2OTStubs[td.innerPairIdx];
    int16_t outerPhiCut = phase2PixelTopology::maxDPhiPhase2OTStubs[td.outerPairIdx];

    // Build inner doublets indexed by middle hit
    // innerDoublets[m] = list of inner hit indices passing phi cut
    std::vector<std::vector<uint32_t>> innerDoublets(middleHits.size());
    for (uint32_t mi = 0; mi < middleHits.size(); ++mi) {
      int16_t mPhi = middleHits[mi].iphi;
      for (uint32_t ii = 0; ii < innerHits.size(); ++ii) {
        int16_t iPhi = innerHits[ii].iphi;
        uint16_t idphi = std::min(std::abs(int16_t(mPhi - iPhi)), std::abs(int16_t(iPhi - mPhi)));
        if (idphi < innerPhiCut)
          innerDoublets[mi].push_back(ii);
      }
    }

    // Build outer doublets indexed by middle hit
    std::vector<std::vector<uint32_t>> outerDoublets(middleHits.size());
    for (uint32_t mi = 0; mi < middleHits.size(); ++mi) {
      int16_t mPhi = middleHits[mi].iphi;
      for (uint32_t oi = 0; oi < outerHits.size(); ++oi) {
        int16_t oPhi = outerHits[oi].iphi;
        uint16_t idphi = std::min(std::abs(int16_t(mPhi - oPhi)), std::abs(int16_t(oPhi - mPhi)));
        if (idphi < outerPhiCut)
          outerDoublets[mi].push_back(oi);
      }
    }

    // Enumerate triplets
    for (uint32_t mi = 0; mi < middleHits.size(); ++mi) {
      if (innerDoublets[mi].empty() || outerDoublets[mi].empty())
        continue;

      const auto& mHit = middleHits[mi];

      for (uint32_t ii : innerDoublets[mi]) {
        const auto& iHit = innerHits[ii];

        for (uint32_t oi : outerDoublets[mi]) {
          const auto& oHit = outerHits[oi];

          // --- Compute theta alignment (from areAlignedRZ) ---
          float r1 = iHit.r, z1 = iHit.z;
          float ri = mHit.r, zi = mHit.z;
          float ro = oHit.r, zo = oHit.z;

          float radius_diff = std::abs(r1 - ro);
          float distance_13_squared = radius_diff * radius_diff + (z1 - zo) * (z1 - zo);
          float pMin = ptmin_ * std::sqrt(distance_13_squared);
          float tan_val = std::abs(z1 * (ri - ro) + zi * (ro - r1) + zo * (r1 - ri));

          float thetaNorm = 0.f;
          if (radius_diff > 0 && distance_13_squared > 0)
            thetaNorm = (tan_val * pMin) / (distance_13_squared * radius_diff);

          // --- Compute DCA (from CircleEq) ---
          CircleEq<float> eq(iHit.x, iHit.y, mHit.x, mHit.y, oHit.x, oHit.y);
          float curvature = std::abs(eq.curvature());
          float dca = std::abs(eq.dca0());
          float dcaNorm = (curvature > 0) ? dca / curvature : 0.f;

          // --- Count SS stubs ---
          int nSS = 0;
          if (iHit.stubType == reco::StubType::SS)
            ++nSS;
          if (mHit.stubType == reco::StubType::SS)
            ++nSS;
          if (oHit.stubType == reco::StubType::SS)
            ++nSS;

          int nSSIdx;
          if (nSS == 0)
            nSSIdx = nSS0;
          else if (nSS == 1)
            nSSIdx = nSS1;
          else
            nSSIdx = nSSge2;

          // --- Geometric-vs-stub kappa cross-check ---
          // Requires at least 2 stub hits with dPhiDr available
          float geomStubKappaSig = -1.f;
          float kappa_geom = 0.f;
          float phiResidual = 0.f;
          bool hasGeomStubKappa = false;

          {
            // Collect stub kappas (skip pixel hits where stubType == 0xFF)
            struct StubKappa {
              float kappa;
              float sigma;
            };
            StubKappa stubKappas[3];
            int nStubKappas = 0;
            const CachedHit* hits[3] = {&iHit, &mHit, &oHit};
            for (int ih = 0; ih < 3; ++ih) {
              if (hits[ih]->stubType != 0xFF && hits[ih]->dPhiDrError > 0.f) {
                float d = hits[ih]->dPhiDr;
                float s = hits[ih]->dPhiDrError;
                float r = hits[ih]->r;
                float den = 1.f + r * r * d * d;
                float sqrt_den = std::sqrt(den);
                stubKappas[nStubKappas] = {d / sqrt_den, s / (den * sqrt_den)};
                ++nStubKappas;
              }
            }

            if (nStubKappas >= 2) {
              // Weighted average of stub kappas
              float w_sum = 0.f, wk_sum = 0.f;
              for (int ik = 0; ik < nStubKappas; ++ik) {
                float w = 1.f / (stubKappas[ik].sigma * stubKappas[ik].sigma);
                w_sum += w;
                wk_sum += w * stubKappas[ik].kappa;
              }
              float kappa_stub_avg = wk_sum / w_sum;
              float sigma_stub_avg = 1.f / std::sqrt(w_sum);

              // Geometric curvature from inner-outer phi/r span
              float dphi_13 = short2phi(int16_t(oHit.iphi - iHit.iphi));
              float dr_13 = oHit.r - iHit.r;

              if (std::abs(dr_13) > 1e-6f) {
                float dphidr_geom = dphi_13 / dr_13;
                float r_mid = 0.5f * (iHit.r + oHit.r);
                float den_g = 1.f + r_mid * r_mid * dphidr_geom * dphidr_geom;
                float sqrt_den_g = std::sqrt(den_g);
                kappa_geom = dphidr_geom / sqrt_den_g;

                // Geometric kappa error from phi resolution (~500 urad for OT stubs)
                constexpr float sigma_phi = 5e-4f;
                float sigma_kappa_geom = sigma_phi / std::abs(dr_13);
                float sk_geom = sigma_kappa_geom / (den_g * sqrt_den_g);

                // Combined significance
                float combined_err = std::sqrt(sk_geom * sk_geom + sigma_stub_avg * sigma_stub_avg);
                if (combined_err > 0.f) {
                  geomStubKappaSig = std::abs(kappa_geom - kappa_stub_avg) / combined_err;
                  hasGeomStubKappa = true;
                }

                // Phi residual at middle hit
                // Predict phi at middle radius using inner phi + stub kappa
                float phi_inner = short2phi(iHit.iphi);
                float phi_predicted = phi_inner + kappa_stub_avg * (mHit.r - iHit.r);
                float phi_middle = short2phi(mHit.iphi);
                phiResidual = phi_middle - phi_predicted;
                // Wrap to [-pi, pi]
                while (phiResidual > M_PI)
                  phiResidual -= 2.f * M_PI;
                while (phiResidual < -M_PI)
                  phiResidual += 2.f * M_PI;
              }
            }
          }

          // --- Truth match ---
          // Triplet is genuine if all 3 hits share a common TrackingParticle
          bool tripletGenuine = false;
          if (iHit.principalTP.isNonnull() && mHit.principalTP.isNonnull() && oHit.principalTP.isNonnull()) {
            tripletGenuine = (iHit.principalTP == mHit.principalTP && mHit.principalTP == oHit.principalTP);
          }

          int truthIdx = tripletGenuine ? Genuine : Fake;

          // --- Fill local histograms ---
          auto& lh = localHistos[truthIdx][nSSIdx];
          lh.h_thetaNorm.Fill(thetaNorm);
          lh.h_dcaNorm.Fill(dcaNorm);
          lh.h_curvature.Fill(curvature);
          if (hasGeomStubKappa) {
            lh.h_geomStubKappaSig.Fill(geomStubKappaSig);
            lh.h_geomKappa.Fill(kappa_geom);
            lh.h_phiResidual.Fill(phiResidual);
          }
        }
      }
    }

    // Merge local histograms into MonitorElements
    for (int tr = 0; tr < 2; ++tr)
      for (int nss = 0; nss < 3; ++nss)
        if (resolvedHistos[tr][nss]) {
          auto& lh = localHistos[tr][nss];
          auto* rh = resolvedHistos[tr][nss];
          mergeLocalIntoME(lh.h_thetaNorm, rh->h_thetaNorm);
          mergeLocalIntoME(lh.h_dcaNorm, rh->h_dcaNorm);
          mergeLocalIntoME(lh.h_curvature, rh->h_curvature);
          mergeLocalIntoME(lh.h_geomStubKappaSig, rh->h_geomStubKappaSig);
          mergeLocalIntoME(lh.h_geomKappa, rh->h_geomKappa);
          mergeLocalIntoME(lh.h_phiResidual, rh->h_phiResidual);
        }
  }
}

// ============================================================
// fillDescriptions
// ============================================================
void Phase2OTValidateCATripletCuts::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("TopFolderName", "HLT/StubCATripletCuts");
  desc.add<double>("ptmin", 0.9);

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
DEFINE_FWK_MODULE(Phase2OTValidateCATripletCuts);
