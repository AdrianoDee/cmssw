// TrueStubNtupleAnalyzer - Per-TrackingParticle ntuple for CA connection cut tuning
//
// Creates a ROOT TTree with one entry per TrackingParticle. Each entry contains:
//   - TP kinematics (pt, eta, phi, vertex, charge, pdgId)
//   - Vector of pixel hits: global position, CA layer ID, local errors
//   - Vector of OT stubs: global position, CA layer ID, dPhiDr, bend, ptEst, etc.
//
// The companion Python script (analyzeCAcuts.py) reads the ntuple, forms doublets
// from consecutive-layer hits/stubs within each TP, computes CA cut variables,
// and produces per-layer-pair distribution plots for cut tuning.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

#include <TTree.h>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsSoA.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometryHost.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometrySoA.h"
#include "RecoTracker/Record/interface/StackedModuleGeometryRecord.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

class TrueStubNtupleAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit TrueStubNtupleAnalyzer(const edm::ParameterSet& iConfig);
  ~TrueStubNtupleAnalyzer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  // Compute pixel CA layer ID from TrackerTopology + position
  // Barrel: layer 1-4 → CA 0-3
  // Forward endcap (z>0): disk 1-12 → CA 4-15
  // Backward endcap (z<0): disk 1-12 → CA 16-27
  static uint8_t pixelCALayerId(const TrackerTopology& topo, DetId detId, float globalZ) {
    if (detId.subdetId() == PixelSubdetector::PixelBarrel) {
      return static_cast<uint8_t>(topo.pxbLayer(detId) - 1);  // 1-based → 0-based
    }
    // Endcap
    uint32_t disk = topo.pxfDisk(detId);
    if (globalZ > 0) {
      return static_cast<uint8_t>(3 + disk);  // disk 1 → CA 4, disk 12 → CA 15
    }
    return static_cast<uint8_t>(15 + disk);  // disk 1 → CA 16, disk 12 → CA 27
  }

  // Compute OT stub CA layer ID from flags (same as OTStubNtupleAnalyzer)
  // Barrel: layer 1-6 → CA 28-33
  // Backward endcap: disk 1-5 → CA 34-38
  // Forward endcap: disk 1-5 → CA 39-43
  static uint8_t otCALayerId(bool isBarrel, bool isFwdEndcap, uint8_t layer) {
    if (isBarrel) {
      return 27 + layer;
    }
    return isFwdEndcap ? (38 + layer) : (33 + layer);
  }

  // Input tokens
  edm::EDGetTokenT<Phase2TrackerRecHit1DCollectionNew> otRecHitCollectionToken_;
  edm::EDGetTokenT<SiPixelRecHitCollection> pixelRecHitCollectionToken_;
  edm::EDGetTokenT<std::vector<TrackingParticle>> tpToken_;
  edm::EDGetTokenT<reco::OTRecHitsHost> otRecHitsSoAToken_;
  edm::EDGetTokenT<reco::StubsHost> stubsToken_;

  // EventSetup tokens
  edm::ESGetToken<reco::StackedModuleGeometryHost, StackedModuleGeometryRecord> geomToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomToken_;

  // TrackerHitAssociator config
  TrackerHitAssociator::Config hitAssocConfig_;

  // TP selection cuts
  double tpMinPt_;
  double tpMaxEta_;
  double tpMaxVtxZ_;
  double tpMaxD0_;
  double tpMaxLxy_;
  int tpMinNLayers_;

  // TTree
  TTree* tree_;

  // === Per-TP scalar branches ===
  UInt_t run_, lumi_;
  ULong64_t event_;
  Float_t tp_pt_, tp_eta_, tp_phi_;
  Float_t tp_vx_, tp_vy_, tp_vz_, tp_d0_, tp_lxy_;
  Int_t tp_pdgId_, tp_charge_;
  UInt_t tp_nPixelHits_, tp_nStubs_;

  // === Per-pixel-hit vector branches ===
  std::vector<float> pix_x_, pix_y_, pix_z_, pix_r_, pix_phi_;
  std::vector<uint8_t> pix_caLayerId_;
  std::vector<float> pix_xerrLocal_, pix_yerrLocal_;
  std::vector<int16_t> pix_clusterSizeY_;

  // === Per-OT-stub vector branches ===
  std::vector<float> stub_x_, stub_y_, stub_z_, stub_r_, stub_phi_;
  std::vector<int16_t> stub_iphi_;
  std::vector<float> stub_dPhiDr_, stub_dPhiDrError_;
  std::vector<float> stub_bend_, stub_ptEst_;
  std::vector<uint8_t> stub_caLayerId_, stub_layer_, stub_stubType_;
  std::vector<bool> stub_isBarrel_, stub_isFlat_, stub_isFwdEndcap_;

  // Inner/outer hit positions
  std::vector<float> inner_x_, inner_y_, inner_z_, inner_r_;
  std::vector<float> outer_x_, outer_y_, outer_z_, outer_r_;

  // Inner hit global error covariance (for kappa significance)
  std::vector<float> inner_xxErr_, inner_xyErr_, inner_yyErr_;
  std::vector<float> inner_xzErr_, inner_yzErr_, inner_zzErr_;

  // Outer hit global error covariance
  std::vector<float> outer_xxErr_, outer_xyErr_, outer_yyErr_;
  std::vector<float> outer_xzErr_, outer_yzErr_, outer_zzErr_;
};

TrueStubNtupleAnalyzer::TrueStubNtupleAnalyzer(const edm::ParameterSet& iConfig)
    : otRecHitCollectionToken_(
          consumes<Phase2TrackerRecHit1DCollectionNew>(iConfig.getParameter<edm::InputTag>("otRecHitSrc"))),
      pixelRecHitCollectionToken_(
          consumes<SiPixelRecHitCollection>(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"))),
      tpToken_(consumes<std::vector<TrackingParticle>>(iConfig.getParameter<edm::InputTag>("trackingParticleSrc"))),
      otRecHitsSoAToken_(consumes<reco::OTRecHitsHost>(iConfig.getParameter<edm::InputTag>("otRecHitsSoASrc"))),
      stubsToken_(consumes<reco::StubsHost>(iConfig.getParameter<edm::InputTag>("stubsSrc"))),
      geomToken_(esConsumes<reco::StackedModuleGeometryHost, StackedModuleGeometryRecord>()),
      topoToken_(esConsumes()),
      trackerGeomToken_(esConsumes()),
      hitAssocConfig_(iConfig.getParameter<edm::ParameterSet>("hitAssociatorConfig"), consumesCollector()),
      tpMinPt_(iConfig.getParameter<double>("TP_minPt")),
      tpMaxEta_(iConfig.getParameter<double>("TP_maxEta")),
      tpMaxVtxZ_(iConfig.getParameter<double>("TP_maxVtxZ")),
      tpMaxD0_(iConfig.getParameter<double>("TP_maxD0")),
      tpMaxLxy_(iConfig.getParameter<double>("TP_maxLxy")),
      tpMinNLayers_(iConfig.getParameter<int>("TP_minNLayers")) {
  usesResource("TFileService");
}

void TrueStubNtupleAnalyzer::beginJob() {
  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("stubs", "Per-TP ntuple for CA cut tuning");

  // Event identification
  tree_->Branch("run", &run_);
  tree_->Branch("lumi", &lumi_);
  tree_->Branch("event", &event_);

  // TP kinematics
  tree_->Branch("tp_pt", &tp_pt_);
  tree_->Branch("tp_eta", &tp_eta_);
  tree_->Branch("tp_phi", &tp_phi_);
  tree_->Branch("tp_vx", &tp_vx_);
  tree_->Branch("tp_vy", &tp_vy_);
  tree_->Branch("tp_vz", &tp_vz_);
  tree_->Branch("tp_d0", &tp_d0_);
  tree_->Branch("tp_lxy", &tp_lxy_);
  tree_->Branch("tp_pdgId", &tp_pdgId_);
  tree_->Branch("tp_charge", &tp_charge_);
  tree_->Branch("tp_nPixelHits", &tp_nPixelHits_);
  tree_->Branch("tp_nStubs", &tp_nStubs_);

  // Pixel hit vectors
  tree_->Branch("pix_x", &pix_x_);
  tree_->Branch("pix_y", &pix_y_);
  tree_->Branch("pix_z", &pix_z_);
  tree_->Branch("pix_r", &pix_r_);
  tree_->Branch("pix_phi", &pix_phi_);
  tree_->Branch("pix_caLayerId", &pix_caLayerId_);
  tree_->Branch("pix_xerrLocal", &pix_xerrLocal_);
  tree_->Branch("pix_yerrLocal", &pix_yerrLocal_);
  tree_->Branch("pix_clusterSizeY", &pix_clusterSizeY_);

  // OT stub vectors
  tree_->Branch("stub_x", &stub_x_);
  tree_->Branch("stub_y", &stub_y_);
  tree_->Branch("stub_z", &stub_z_);
  tree_->Branch("stub_r", &stub_r_);
  tree_->Branch("stub_phi", &stub_phi_);
  tree_->Branch("stub_iphi", &stub_iphi_);
  tree_->Branch("stub_dPhiDr", &stub_dPhiDr_);
  tree_->Branch("stub_dPhiDrError", &stub_dPhiDrError_);
  tree_->Branch("stub_bend", &stub_bend_);
  tree_->Branch("stub_ptEst", &stub_ptEst_);
  tree_->Branch("stub_caLayerId", &stub_caLayerId_);
  tree_->Branch("stub_layer", &stub_layer_);
  tree_->Branch("stub_stubType", &stub_stubType_);
  tree_->Branch("stub_isBarrel", &stub_isBarrel_);
  tree_->Branch("stub_isFlat", &stub_isFlat_);
  tree_->Branch("stub_isFwdEndcap", &stub_isFwdEndcap_);

  // Inner/outer hit positions
  tree_->Branch("inner_x", &inner_x_);
  tree_->Branch("inner_y", &inner_y_);
  tree_->Branch("inner_z", &inner_z_);
  tree_->Branch("inner_r", &inner_r_);
  tree_->Branch("outer_x", &outer_x_);
  tree_->Branch("outer_y", &outer_y_);
  tree_->Branch("outer_z", &outer_z_);
  tree_->Branch("outer_r", &outer_r_);

  // Inner hit global error covariance
  tree_->Branch("inner_xxErr", &inner_xxErr_);
  tree_->Branch("inner_xyErr", &inner_xyErr_);
  tree_->Branch("inner_yyErr", &inner_yyErr_);
  tree_->Branch("inner_xzErr", &inner_xzErr_);
  tree_->Branch("inner_yzErr", &inner_yzErr_);
  tree_->Branch("inner_zzErr", &inner_zzErr_);

  // Outer hit global error covariance
  tree_->Branch("outer_xxErr", &outer_xxErr_);
  tree_->Branch("outer_xyErr", &outer_xyErr_);
  tree_->Branch("outer_yyErr", &outer_yyErr_);
  tree_->Branch("outer_xzErr", &outer_xzErr_);
  tree_->Branch("outer_yzErr", &outer_yzErr_);
  tree_->Branch("outer_zzErr", &outer_zzErr_);
}

void TrueStubNtupleAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Get inputs
  const auto& otRecHitCollection = iEvent.get(otRecHitCollectionToken_);
  const auto& pixelRecHitCollection = iEvent.get(pixelRecHitCollectionToken_);
  edm::Handle<std::vector<TrackingParticle>> tpHandle;
  iEvent.getByToken(tpToken_, tpHandle);
  const auto& otHitsSoA = iEvent.get(otRecHitsSoAToken_);
  const auto& stubsHost = iEvent.get(stubsToken_);

  // EventSetup
  const auto& stackedGeom = iSetup.getData(geomToken_);
  const auto& topo = iSetup.getData(topoToken_);
  const auto& trackerGeom = iSetup.getData(trackerGeomToken_);

  // SoA views
  auto otHitsView = otHitsSoA.view().otRecHits();
  auto stackedGeomView = stackedGeom.view();
  auto stubsView = stubsHost.const_view().stubs();
  uint32_t nOTHits = otHitsSoA.nHits();
  uint32_t nStubs = stubsView.metadata().size();
  uint32_t nStackedModules = stackedGeomView.metadata().size();

  // Event identification
  run_ = iEvent.id().run();
  lumi_ = iEvent.luminosityBlock();
  event_ = iEvent.id().event();

  // Create TrackerHitAssociator
  TrackerHitAssociator hitAssociator(iEvent, hitAssocConfig_);

  // === Step 1: Build (simTrackId, eventId) → TrackingParticle* map ===
  std::map<SimHitIdpr, const TrackingParticle*> simTrackToTP;
  for (size_t iTP = 0; iTP < tpHandle->size(); ++iTP) {
    const auto& tp = (*tpHandle)[iTP];
    for (const auto& g4Track : tp.g4Tracks()) {
      simTrackToTP[{g4Track.trackId(), g4Track.eventId()}] = &tp;
    }
  }

  // === Step 2: Build origRecHitIdx → OTRecHitsSoA index map ===
  std::unordered_map<uint32_t, uint32_t> origIdxToSoAIdx;
  origIdxToSoAIdx.reserve(nOTHits);
  for (uint32_t iHit = 0; iHit < nOTHits; ++iHit) {
    origIdxToSoAIdx[otHitsView[iHit].origRecHitIdx()] = iHit;
  }

  // === Step 3: Associate OT RecHits with TPs (origRecHitIdx → set<TP*>) ===
  std::unordered_map<uint32_t, std::set<const TrackingParticle*>> origIdxToTPs;
  {
    uint32_t flatIdx = 0;
    for (const auto& detSet : otRecHitCollection) {
      for (const auto& recHit : detSet) {
        std::vector<SimHitIdpr> ids;
        hitAssociator.associatePhase2TrackerRecHit(&recHit, ids);
        for (const auto& id : ids) {
          auto it = simTrackToTP.find(id);
          if (it != simTrackToTP.end()) {
            origIdxToTPs[flatIdx].insert(it->second);
          }
        }
        flatIdx++;
      }
    }
  }

  // === Step 4: Group stubs by TP ===
  // For each stub, find which TPs it belongs to via innerHitIdx → OTRecHitsSoA → origRecHitIdx → TPs
  std::map<const TrackingParticle*, std::vector<uint32_t>> tpToStubIndices;
  for (uint32_t iStub = 0; iStub < nStubs; ++iStub) {
    uint32_t innerHitIdx = stubsView[iStub].innerHitIdx();
    if (innerHitIdx >= nOTHits)
      continue;
    uint32_t origIdx = otHitsView[innerHitIdx].origRecHitIdx();
    auto tpIt = origIdxToTPs.find(origIdx);
    if (tpIt != origIdxToTPs.end()) {
      for (const auto* tp : tpIt->second) {
        tpToStubIndices[tp].push_back(iStub);
      }
    }
  }

  // === Step 5: Associate pixel RecHits with TPs ===
  struct PixelHitInfo {
    float x, y, z, r, phi;
    uint8_t caLayerId;
    float xerrLocal, yerrLocal;
    int16_t clusterSizeY;
  };
  std::map<const TrackingParticle*, std::vector<PixelHitInfo>> tpToPixelHits;
  {
    for (const auto& detSet : pixelRecHitCollection) {
      DetId detId(detSet.detId());
      // Only pixel subdetectors
      if (detId.subdetId() != PixelSubdetector::PixelBarrel && detId.subdetId() != PixelSubdetector::PixelEndcap)
        continue;

      const auto* det = trackerGeom.idToDet(detId);
      if (!det)
        continue;

      for (const auto& recHit : detSet) {
        std::vector<SimHitIdpr> ids;
        hitAssociator.associateHitId(recHit, ids);

        if (ids.empty())
          continue;

        // Compute global position
        auto globalPos = det->toGlobal(recHit.localPosition());
        float gx = globalPos.x();
        float gy = globalPos.y();
        float gz = globalPos.z();
        float gr = std::sqrt(gx * gx + gy * gy);
        float gphi = std::atan2(gy, gx);

        uint8_t caLayer = pixelCALayerId(topo, detId, gz);

        // Local errors
        auto const& le = recHit.localPositionError();
        float xerr = le.xx();
        float yerr = le.yy();

        // Cluster y-size
        int16_t csY = 0;
        if (recHit.cluster().isNonnull()) {
          csY = static_cast<int16_t>(recHit.cluster()->sizeY());
        }

        PixelHitInfo info{gx, gy, gz, gr, gphi, caLayer, xerr, yerr, csY};

        for (const auto& id : ids) {
          auto it = simTrackToTP.find(id);
          if (it != simTrackToTP.end()) {
            tpToPixelHits[it->second].push_back(info);
          }
        }
      }
    }
  }

  // === Step 6: Apply TP selection and fill TTree ===
  for (size_t iTP = 0; iTP < tpHandle->size(); ++iTP) {
    const auto& tp = (*tpHandle)[iTP];

    // Selection cuts
    if (tp.charge() == 0)
      continue;
    if (tp.eventId().bunchCrossing() != 0)
      continue;
    if (tp.pt() < tpMinPt_)
      continue;
    if (std::abs(tp.eta()) > tpMaxEta_)
      continue;
    if (std::abs(tp.vz()) > tpMaxVtxZ_)
      continue;
    if (std::abs(tp.d0()) > tpMaxD0_)
      continue;
    float lxy = std::sqrt(tp.vx() * tp.vx() + tp.vy() * tp.vy());
    if (lxy > tpMaxLxy_)
      continue;

    const TrackingParticle* tpPtr = &tp;

    // Get stubs for this TP
    auto stubIt = tpToStubIndices.find(tpPtr);
    uint32_t nTPStubs = (stubIt != tpToStubIndices.end()) ? stubIt->second.size() : 0;

    // Get pixel hits for this TP
    auto pixIt = tpToPixelHits.find(tpPtr);
    uint32_t nTPPixelHits = (pixIt != tpToPixelHits.end()) ? pixIt->second.size() : 0;

    // Require minimum number of OT layers (stubs)
    if (static_cast<int>(nTPStubs) < tpMinNLayers_)
      continue;

    // Fill TP scalars
    tp_pt_ = tp.pt();
    tp_eta_ = tp.eta();
    tp_phi_ = tp.phi();
    tp_vx_ = tp.vx();
    tp_vy_ = tp.vy();
    tp_vz_ = tp.vz();
    tp_d0_ = tp.d0();
    tp_lxy_ = lxy;
    tp_pdgId_ = tp.pdgId();
    tp_charge_ = tp.charge();
    tp_nPixelHits_ = nTPPixelHits;
    tp_nStubs_ = nTPStubs;

    // Clear vectors
    pix_x_.clear();
    pix_y_.clear();
    pix_z_.clear();
    pix_r_.clear();
    pix_phi_.clear();
    pix_caLayerId_.clear();
    pix_xerrLocal_.clear();
    pix_yerrLocal_.clear();
    pix_clusterSizeY_.clear();

    stub_x_.clear();
    stub_y_.clear();
    stub_z_.clear();
    stub_r_.clear();
    stub_phi_.clear();
    stub_iphi_.clear();
    stub_dPhiDr_.clear();
    stub_dPhiDrError_.clear();
    stub_bend_.clear();
    stub_ptEst_.clear();
    stub_caLayerId_.clear();
    stub_layer_.clear();
    stub_stubType_.clear();
    stub_isBarrel_.clear();
    stub_isFlat_.clear();
    stub_isFwdEndcap_.clear();

    inner_x_.clear();
    inner_y_.clear();
    inner_z_.clear();
    inner_r_.clear();
    outer_x_.clear();
    outer_y_.clear();
    outer_z_.clear();
    outer_r_.clear();

    inner_xxErr_.clear();
    inner_xyErr_.clear();
    inner_yyErr_.clear();
    inner_xzErr_.clear();
    inner_yzErr_.clear();
    inner_zzErr_.clear();
    outer_xxErr_.clear();
    outer_xyErr_.clear();
    outer_yyErr_.clear();
    outer_xzErr_.clear();
    outer_yzErr_.clear();
    outer_zzErr_.clear();

    // Fill pixel hit vectors
    if (pixIt != tpToPixelHits.end()) {
      for (const auto& ph : pixIt->second) {
        pix_x_.push_back(ph.x);
        pix_y_.push_back(ph.y);
        pix_z_.push_back(ph.z);
        pix_r_.push_back(ph.r);
        pix_phi_.push_back(ph.phi);
        pix_caLayerId_.push_back(ph.caLayerId);
        pix_xerrLocal_.push_back(ph.xerrLocal);
        pix_yerrLocal_.push_back(ph.yerrLocal);
        pix_clusterSizeY_.push_back(ph.clusterSizeY);
      }
    }

    // Fill OT stub vectors
    if (stubIt != tpToStubIndices.end()) {
      constexpr uint32_t nPixelModules = 4000;

      for (uint32_t iStub : stubIt->second) {
        auto const& stub = stubsView[iStub];

        // Stub global position
        stub_x_.push_back(stub.xGlobal());
        stub_y_.push_back(stub.yGlobal());
        stub_z_.push_back(stub.zGlobal());
        stub_r_.push_back(stub.rGlobal());
        stub_phi_.push_back(std::atan2(stub.yGlobal(), stub.xGlobal()));
        stub_iphi_.push_back(stub.iphi());

        // Stub direction/curvature
        stub_dPhiDr_.push_back(stub.dPhiDr());
        stub_dPhiDrError_.push_back(stub.dPhiDrError());
        stub_bend_.push_back(stub.bend());
        stub_ptEst_.push_back(stub.ptEst());

        // Module/layer identification
        uint8_t flags = stub.flags();
        bool isBarrel = reco::StubFlags::isBarrel(flags);
        bool isFlat = reco::StubFlags::isFlat(flags);
        uint8_t layer = reco::StubFlags::layer(flags);
        stub_layer_.push_back(layer);
        stub_isBarrel_.push_back(isBarrel);
        stub_isFlat_.push_back(isFlat);
        stub_stubType_.push_back(static_cast<uint8_t>(stub.stubType()));

        // Determine fwdEndcap from geometry
        uint32_t detIdx = stub.detectorIndex();
        bool isFwdEndcap = false;
        if (detIdx >= nPixelModules) {
          uint32_t geomIdx = detIdx - nPixelModules;
          if (geomIdx < nStackedModules) {
            isFwdEndcap = stackedGeomView[geomIdx].isFwdEndcap();
          }
        }
        stub_isFwdEndcap_.push_back(isFwdEndcap);

        // CA layer ID
        stub_caLayerId_.push_back(otCALayerId(isBarrel, isFwdEndcap, layer));

        // Inner hit position and errors
        uint32_t innerIdx = stub.innerHitIdx();
        if (innerIdx < nOTHits) {
          auto const& ih = otHitsView[innerIdx];
          inner_x_.push_back(ih.xGlobal());
          inner_y_.push_back(ih.yGlobal());
          inner_z_.push_back(ih.zGlobal());
          inner_r_.push_back(ih.rGlobal());
          inner_xxErr_.push_back(ih.xGlobalErr());
          inner_xyErr_.push_back(ih.xyGlobalErr());
          inner_yyErr_.push_back(ih.yGlobalErr());
          inner_xzErr_.push_back(ih.xzGlobalErr());
          inner_yzErr_.push_back(ih.yzGlobalErr());
          inner_zzErr_.push_back(ih.zGlobalErr());
        } else {
          inner_x_.push_back(0);
          inner_y_.push_back(0);
          inner_z_.push_back(0);
          inner_r_.push_back(0);
          inner_xxErr_.push_back(0);
          inner_xyErr_.push_back(0);
          inner_yyErr_.push_back(0);
          inner_xzErr_.push_back(0);
          inner_yzErr_.push_back(0);
          inner_zzErr_.push_back(0);
        }

        // Outer hit position and errors
        bool hasOuter = (stub.stubType() != reco::StubType::PHitOnly);
        uint32_t outerIdx = stub.outerHitIdx();
        if (hasOuter && outerIdx < nOTHits) {
          auto const& oh = otHitsView[outerIdx];
          outer_x_.push_back(oh.xGlobal());
          outer_y_.push_back(oh.yGlobal());
          outer_z_.push_back(oh.zGlobal());
          outer_r_.push_back(oh.rGlobal());
          outer_xxErr_.push_back(oh.xGlobalErr());
          outer_xyErr_.push_back(oh.xyGlobalErr());
          outer_yyErr_.push_back(oh.yGlobalErr());
          outer_xzErr_.push_back(oh.xzGlobalErr());
          outer_yzErr_.push_back(oh.yzGlobalErr());
          outer_zzErr_.push_back(oh.zGlobalErr());
        } else {
          outer_x_.push_back(0);
          outer_y_.push_back(0);
          outer_z_.push_back(0);
          outer_r_.push_back(0);
          outer_xxErr_.push_back(0);
          outer_xyErr_.push_back(0);
          outer_yyErr_.push_back(0);
          outer_xzErr_.push_back(0);
          outer_yzErr_.push_back(0);
          outer_zzErr_.push_back(0);
        }
      }
    }

    tree_->Fill();
  }
}

void TrueStubNtupleAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("otRecHitSrc", edm::InputTag("siPhase2RecHits"))
      ->setComment("Phase2TrackerRecHit1DCollectionNew for OT truth matching");
  desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("siPixelRecHits"))
      ->setComment("SiPixelRecHitCollection for pixel truth matching");
  desc.add<edm::InputTag>("trackingParticleSrc", edm::InputTag("mix", "MergedTrackTruth"))
      ->setComment("TrackingParticle collection");
  desc.add<edm::InputTag>("otRecHitsSoASrc", edm::InputTag("pixelSeedingOTRecHitsSoA"))
      ->setComment("OTRecHitsSoA collection for origRecHitIdx mapping");
  desc.add<edm::InputTag>("stubsSrc", edm::InputTag("otStubProducerVectorHitStyle"))
      ->setComment("StubsHost collection (from regular or TrueStub pipeline)");

  // TP selection
  desc.add<double>("TP_minPt", 0.9)->setComment("Minimum TP pT [GeV]");
  desc.add<double>("TP_maxEta", 2.6)->setComment("Maximum |eta|");
  desc.add<double>("TP_maxVtxZ", 30.0)->setComment("Maximum |vz| [cm]");
  desc.add<double>("TP_maxD0", 1.0)->setComment("Maximum |d0| [cm]");
  desc.add<double>("TP_maxLxy", 1.0)->setComment("Maximum Lxy [cm]");
  desc.add<int>("TP_minNLayers", 4)->setComment("Minimum number of OT stubs");

  // TrackerHitAssociator config
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

DEFINE_FWK_MODULE(TrueStubNtupleAnalyzer);
