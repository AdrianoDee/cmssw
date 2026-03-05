// OTStubNtupleAnalyzer - Creates a flat ROOT TTree with detailed stub information
// for analysis of dPhiDr, dPhiDz, and other compatibility quantities.

#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsHost.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometryHost.h"
#include "RecoTracker/Record/interface/StackedModuleGeometryRecord.h"

#include <TTree.h>
#include <cmath>

class OTStubNtupleAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit OTStubNtupleAnalyzer(edm::ParameterSet const& iConfig);
  ~OTStubNtupleAnalyzer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

  // Input tokens
  edm::EDGetTokenT<reco::OTRecHitsHost> recHitsToken_;
  edm::EDGetTokenT<reco::StubsHost> stubsToken_;
  edm::ESGetToken<::reco::StackedModuleGeometryHost, StackedModuleGeometryRecord> geomToken_;

  // TTree and branches
  TTree* tree_;

  // Event identification
  UInt_t run_;
  UInt_t lumi_;
  ULong64_t event_;
  UInt_t nStubsInEvent_;
  UInt_t stubIndex_;

  // Module/Layer identification
  UInt_t detectorIndex_;
  UInt_t stackDetId_;
  UChar_t layer_;
  UChar_t caLayerId_;
  Bool_t isBarrel_;
  Bool_t isEndcap_;
  Bool_t isTilted_;
  Bool_t isFlat_;
  Bool_t isFlipped_;
  Bool_t isFwdEndcap_;
  UChar_t stubType_;
  Bool_t isPS_;

  // Geometry parameters
  Float_t tiltAngle_;
  Float_t sinTilt_;
  Float_t cosTilt_;
  Float_t sensorSeparation_;
  Float_t meanModuleRadius_;
  Float_t maxBend_;

  // Module direction vectors (for debugging)
  Float_t globalLowUpNormX_;
  Float_t globalLowUpNormY_;
  Float_t globalLowUpNormZ_;
  Float_t localXInGlobalX_;
  Float_t localXInGlobalY_;
  Float_t localXInGlobalZ_;

  // Stub global position
  Float_t stub_x_;
  Float_t stub_y_;
  Float_t stub_z_;
  Float_t stub_r_;
  Float_t stub_phi_;
  Float_t stub_eta_;

  // Stub local position
  Float_t stub_xLocal_;
  Float_t stub_yLocal_;
  Float_t stub_xerrLocal_;
  Float_t stub_yerrLocal_;

  // Inner hit global position
  Float_t inner_x_;
  Float_t inner_y_;
  Float_t inner_z_;
  Float_t inner_r_;
  Float_t inner_phi_;

  // Outer hit global position
  Float_t outer_x_;
  Float_t outer_y_;
  Float_t outer_z_;
  Float_t outer_r_;
  Float_t outer_phi_;

  // Inner hit local position
  Float_t inner_xLocal_;
  Float_t inner_yLocal_;
  Float_t inner_xerrLocal_;
  Float_t inner_yerrLocal_;

  // Outer hit local position
  Float_t outer_xLocal_;
  Float_t outer_yLocal_;
  Float_t outer_xerrLocal_;
  Float_t outer_yerrLocal_;

  // Sensor separation (computed)
  Float_t dr_sensor_;
  Float_t dz_sensor_;
  Float_t d3_sensor_;
  Float_t dr_effective_;  // Effective dr using L1 trigger formula

  // Raw differences
  Float_t dphi_raw_;
  Float_t dphi_parallax_;   // Parallax contribution to dphi
  Float_t dphi_corrected_;  // dphi with parallax subtracted
  Float_t dr_;
  Float_t dz_;

  // Computed direction quantities
  Float_t dPhiDr_;
  Float_t dPhiDrError_;
  Float_t dPhiDz_;
  Float_t dPhiDzError_;

  // Eta consistency (z/r ratio)
  Float_t eta_inner_;
  Float_t eta_outer_;
  Float_t eta_diff_;

  // Width/Parallax
  Float_t width_;
  Float_t width_raw_;
  Float_t parallaxCorr_;

  // Bend and pT
  Float_t bend_;
  Float_t ptEst_;

  // Hit indices and stub identification
  UInt_t innerHitIdx_;
  UInt_t outerHitIdx_;
  UInt_t pHitGroupId_;

  // Inner hit global error covariance (variance, from OTRecHitsSoA)
  Float_t inner_xxGlobalErr_;
  Float_t inner_xyGlobalErr_;
  Float_t inner_yyGlobalErr_;
  Float_t inner_xzGlobalErr_;
  Float_t inner_yzGlobalErr_;
  Float_t inner_zzGlobalErr_;

  // Outer hit global error covariance (variance, from OTRecHitsSoA)
  Float_t outer_xxGlobalErr_;
  Float_t outer_xyGlobalErr_;
  Float_t outer_yyGlobalErr_;
  Float_t outer_xzGlobalErr_;
  Float_t outer_yzGlobalErr_;
  Float_t outer_zzGlobalErr_;

  // Flag for valid outer hit (false for PHitOnly stubs)
  Bool_t hasOuterHit_;
};

OTStubNtupleAnalyzer::OTStubNtupleAnalyzer(edm::ParameterSet const& iConfig)
    : recHitsToken_(consumes(iConfig.getParameter<edm::InputTag>("recHits"))),
      stubsToken_(consumes(iConfig.getParameter<edm::InputTag>("stubs"))),
      geomToken_(esConsumes()) {
  usesResource("TFileService");
}

void OTStubNtupleAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("recHits", edm::InputTag("pixelSeedingOTRecHitsSoAConverter"))
      ->setComment("Input OT RecHits SoA collection");
  desc.add<edm::InputTag>("stubs", edm::InputTag("otStubProducerVectorHitStyle"))
      ->setComment("Input stubs collection");
  descriptions.addWithDefaultLabel(desc);
}

void OTStubNtupleAnalyzer::beginJob() {
  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("stubs", "OT Stub Ntuple for Compatibility Analysis");

  // Event identification
  tree_->Branch("run", &run_, "run/i");
  tree_->Branch("lumi", &lumi_, "lumi/i");
  tree_->Branch("event", &event_, "event/l");
  tree_->Branch("nStubsInEvent", &nStubsInEvent_, "nStubsInEvent/i");
  tree_->Branch("stubIndex", &stubIndex_, "stubIndex/i");

  // Module/Layer identification
  tree_->Branch("detectorIndex", &detectorIndex_, "detectorIndex/i");
  tree_->Branch("stackDetId", &stackDetId_, "stackDetId/i");
  tree_->Branch("layer", &layer_, "layer/b");
  tree_->Branch("caLayerId", &caLayerId_, "caLayerId/b");
  tree_->Branch("isBarrel", &isBarrel_, "isBarrel/O");
  tree_->Branch("isEndcap", &isEndcap_, "isEndcap/O");
  tree_->Branch("isTilted", &isTilted_, "isTilted/O");
  tree_->Branch("isFlat", &isFlat_, "isFlat/O");
  tree_->Branch("isFlipped", &isFlipped_, "isFlipped/O");
  tree_->Branch("isFwdEndcap", &isFwdEndcap_, "isFwdEndcap/O");
  tree_->Branch("stubType", &stubType_, "stubType/b");
  tree_->Branch("isPS", &isPS_, "isPS/O");

  // Geometry parameters
  tree_->Branch("tiltAngle", &tiltAngle_, "tiltAngle/F");
  tree_->Branch("sinTilt", &sinTilt_, "sinTilt/F");
  tree_->Branch("cosTilt", &cosTilt_, "cosTilt/F");
  tree_->Branch("sensorSeparation", &sensorSeparation_, "sensorSeparation/F");
  tree_->Branch("meanModuleRadius", &meanModuleRadius_, "meanModuleRadius/F");
  tree_->Branch("maxBend", &maxBend_, "maxBend/F");

  // Module direction vectors
  tree_->Branch("globalLowUpNormX", &globalLowUpNormX_, "globalLowUpNormX/F");
  tree_->Branch("globalLowUpNormY", &globalLowUpNormY_, "globalLowUpNormY/F");
  tree_->Branch("globalLowUpNormZ", &globalLowUpNormZ_, "globalLowUpNormZ/F");
  tree_->Branch("localXInGlobalX", &localXInGlobalX_, "localXInGlobalX/F");
  tree_->Branch("localXInGlobalY", &localXInGlobalY_, "localXInGlobalY/F");
  tree_->Branch("localXInGlobalZ", &localXInGlobalZ_, "localXInGlobalZ/F");

  // Stub global position
  tree_->Branch("stub_x", &stub_x_, "stub_x/F");
  tree_->Branch("stub_y", &stub_y_, "stub_y/F");
  tree_->Branch("stub_z", &stub_z_, "stub_z/F");
  tree_->Branch("stub_r", &stub_r_, "stub_r/F");
  tree_->Branch("stub_phi", &stub_phi_, "stub_phi/F");
  tree_->Branch("stub_eta", &stub_eta_, "stub_eta/F");

  // Stub local position
  tree_->Branch("stub_xLocal", &stub_xLocal_, "stub_xLocal/F");
  tree_->Branch("stub_yLocal", &stub_yLocal_, "stub_yLocal/F");
  tree_->Branch("stub_xerrLocal", &stub_xerrLocal_, "stub_xerrLocal/F");
  tree_->Branch("stub_yerrLocal", &stub_yerrLocal_, "stub_yerrLocal/F");

  // Inner hit global position
  tree_->Branch("inner_x", &inner_x_, "inner_x/F");
  tree_->Branch("inner_y", &inner_y_, "inner_y/F");
  tree_->Branch("inner_z", &inner_z_, "inner_z/F");
  tree_->Branch("inner_r", &inner_r_, "inner_r/F");
  tree_->Branch("inner_phi", &inner_phi_, "inner_phi/F");

  // Outer hit global position
  tree_->Branch("outer_x", &outer_x_, "outer_x/F");
  tree_->Branch("outer_y", &outer_y_, "outer_y/F");
  tree_->Branch("outer_z", &outer_z_, "outer_z/F");
  tree_->Branch("outer_r", &outer_r_, "outer_r/F");
  tree_->Branch("outer_phi", &outer_phi_, "outer_phi/F");

  // Inner hit local position
  tree_->Branch("inner_xLocal", &inner_xLocal_, "inner_xLocal/F");
  tree_->Branch("inner_yLocal", &inner_yLocal_, "inner_yLocal/F");
  tree_->Branch("inner_xerrLocal", &inner_xerrLocal_, "inner_xerrLocal/F");
  tree_->Branch("inner_yerrLocal", &inner_yerrLocal_, "inner_yerrLocal/F");

  // Outer hit local position
  tree_->Branch("outer_xLocal", &outer_xLocal_, "outer_xLocal/F");
  tree_->Branch("outer_yLocal", &outer_yLocal_, "outer_yLocal/F");
  tree_->Branch("outer_xerrLocal", &outer_xerrLocal_, "outer_xerrLocal/F");
  tree_->Branch("outer_yerrLocal", &outer_yerrLocal_, "outer_yerrLocal/F");

  // Sensor separation (computed)
  tree_->Branch("dr_sensor", &dr_sensor_, "dr_sensor/F");
  tree_->Branch("dz_sensor", &dz_sensor_, "dz_sensor/F");
  tree_->Branch("d3_sensor", &d3_sensor_, "d3_sensor/F");
  tree_->Branch("dr_effective", &dr_effective_, "dr_effective/F");

  // Raw differences
  tree_->Branch("dphi_raw", &dphi_raw_, "dphi_raw/F");
  tree_->Branch("dphi_parallax", &dphi_parallax_, "dphi_parallax/F");
  tree_->Branch("dphi_corrected", &dphi_corrected_, "dphi_corrected/F");
  tree_->Branch("dr", &dr_, "dr/F");
  tree_->Branch("dz", &dz_, "dz/F");

  // Computed direction quantities
  tree_->Branch("dPhiDr", &dPhiDr_, "dPhiDr/F");
  tree_->Branch("dPhiDrError", &dPhiDrError_, "dPhiDrError/F");
  tree_->Branch("dPhiDz", &dPhiDz_, "dPhiDz/F");
  tree_->Branch("dPhiDzError", &dPhiDzError_, "dPhiDzError/F");

  // Eta consistency
  tree_->Branch("eta_inner", &eta_inner_, "eta_inner/F");
  tree_->Branch("eta_outer", &eta_outer_, "eta_outer/F");
  tree_->Branch("eta_diff", &eta_diff_, "eta_diff/F");

  // Width/Parallax
  tree_->Branch("width", &width_, "width/F");
  tree_->Branch("width_raw", &width_raw_, "width_raw/F");
  tree_->Branch("parallaxCorr", &parallaxCorr_, "parallaxCorr/F");

  // Bend and pT
  tree_->Branch("bend", &bend_, "bend/F");
  tree_->Branch("ptEst", &ptEst_, "ptEst/F");

  // Hit indices and stub identification
  tree_->Branch("innerHitIdx", &innerHitIdx_, "innerHitIdx/i");
  tree_->Branch("outerHitIdx", &outerHitIdx_, "outerHitIdx/i");
  tree_->Branch("pHitGroupId", &pHitGroupId_, "pHitGroupId/i");
  tree_->Branch("hasOuterHit", &hasOuterHit_, "hasOuterHit/O");

  // Inner hit global error covariance (variance)
  tree_->Branch("inner_xxGlobalErr", &inner_xxGlobalErr_, "inner_xxGlobalErr/F");
  tree_->Branch("inner_xyGlobalErr", &inner_xyGlobalErr_, "inner_xyGlobalErr/F");
  tree_->Branch("inner_yyGlobalErr", &inner_yyGlobalErr_, "inner_yyGlobalErr/F");
  tree_->Branch("inner_xzGlobalErr", &inner_xzGlobalErr_, "inner_xzGlobalErr/F");
  tree_->Branch("inner_yzGlobalErr", &inner_yzGlobalErr_, "inner_yzGlobalErr/F");
  tree_->Branch("inner_zzGlobalErr", &inner_zzGlobalErr_, "inner_zzGlobalErr/F");

  // Outer hit global error covariance (variance)
  tree_->Branch("outer_xxGlobalErr", &outer_xxGlobalErr_, "outer_xxGlobalErr/F");
  tree_->Branch("outer_xyGlobalErr", &outer_xyGlobalErr_, "outer_xyGlobalErr/F");
  tree_->Branch("outer_yyGlobalErr", &outer_yyGlobalErr_, "outer_yyGlobalErr/F");
  tree_->Branch("outer_xzGlobalErr", &outer_xzGlobalErr_, "outer_xzGlobalErr/F");
  tree_->Branch("outer_yzGlobalErr", &outer_yzGlobalErr_, "outer_yzGlobalErr/F");
  tree_->Branch("outer_zzGlobalErr", &outer_zzGlobalErr_, "outer_zzGlobalErr/F");
}

void OTStubNtupleAnalyzer::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  auto const& recHitsHost = iEvent.get(recHitsToken_);
  auto const& stubsHost = iEvent.get(stubsToken_);
  auto const& geomHost = iSetup.getData(geomToken_);

  auto const& hitsView = recHitsHost.const_view().otRecHits();
  auto const& stubsView = stubsHost.const_view().stubs();
  auto const& geomView = geomHost.const_view();

  uint32_t nStubs = stubsView.metadata().size();
  uint32_t nModules = recHitsHost.nModules();

  // Event identification
  run_ = iEvent.id().run();
  lumi_ = iEvent.luminosityBlock();
  event_ = iEvent.id().event();
  nStubsInEvent_ = nStubs;

  // CA module offset (pixel modules come first)
  constexpr uint32_t nPixelModules = 4000;

  for (uint32_t iStub = 0; iStub < nStubs; ++iStub) {
    stubIndex_ = iStub;

    // Get stub data
    auto const& stub = stubsView[iStub];

    // Get hit indices
    innerHitIdx_ = stub.innerHitIdx();
    outerHitIdx_ = stub.outerHitIdx();
    pHitGroupId_ = stub.pHitGroupId();

    // PHitOnly stubs have no valid outer hit (outerHitIdx == UINT32_MAX)
    hasOuterHit_ = (stub.stubType() != reco::StubType::PHitOnly);

    // Get inner hit (always valid)
    auto const& innerHit = hitsView[innerHitIdx_];

    // Detector/module index
    detectorIndex_ = stub.detectorIndex();
    uint32_t geomIdx = detectorIndex_ - nPixelModules;

    // Get geometry data (only if valid)
    if (geomIdx < nModules) {
      auto const& geom = geomView[geomIdx];
      stackDetId_ = geom.stackedDetId();
      isFlipped_ = geom.isFlipped();
      isFwdEndcap_ = geom.isFwdEndcap();
      tiltAngle_ = geom.tiltAngle();
      sinTilt_ = geom.sinTilt();
      cosTilt_ = geom.cosTilt();
      sensorSeparation_ = geom.sensorSeparation();
      meanModuleRadius_ = geom.meanRadius();
      maxBend_ = geom.maxBend();

      // Module direction vectors
      globalLowUpNormX_ = geom.globalLowUpNormX();
      globalLowUpNormY_ = geom.globalLowUpNormY();
      globalLowUpNormZ_ = geom.globalLowUpNormZ();
      localXInGlobalX_ = geom.localXInGlobalX();
      localXInGlobalY_ = geom.localXInGlobalY();
      localXInGlobalZ_ = geom.localXInGlobalZ();
    } else {
      stackDetId_ = 0;
      isFlipped_ = false;
      isFwdEndcap_ = false;
      tiltAngle_ = 0;
      sinTilt_ = 0;
      cosTilt_ = 1;
      sensorSeparation_ = 0;
      meanModuleRadius_ = 0;
      maxBend_ = 0;
      globalLowUpNormX_ = 0;
      globalLowUpNormY_ = 0;
      globalLowUpNormZ_ = 0;
      localXInGlobalX_ = 0;
      localXInGlobalY_ = 0;
      localXInGlobalZ_ = 0;
    }

    // Extract layer from flags
    uint8_t flags = stub.flags();
    layer_ = reco::StubFlags::layer(flags);
    isBarrel_ = reco::StubFlags::isBarrel(flags);
    isEndcap_ = !isBarrel_;
    isFlat_ = reco::StubFlags::isFlat(flags);
    isTilted_ = isBarrel_ && !isFlat_;

    // CA layer ID calculation
    // layer_ is 1-based (from TrackerTopology::layer()):
    //   Barrel layers 1-6 map to CA layers 28-33
    //   Endcap backward disks 1-5 map to CA layers 34-38
    //   Endcap forward disks 1-5 map to CA layers 39-43
    if (isBarrel_) {
      caLayerId_ = 27 + layer_;
    } else {
      caLayerId_ = isFwdEndcap_ ? (38 + layer_) : (33 + layer_);
    }

    // Stub type
    stubType_ = static_cast<uint8_t>(stub.stubType());
    isPS_ = (stub.stubType() == reco::StubType::PS);

    // Stub global position
    stub_x_ = stub.xGlobal();
    stub_y_ = stub.yGlobal();
    stub_z_ = stub.zGlobal();
    stub_r_ = stub.rGlobal();
    stub_phi_ = std::atan2(stub_y_, stub_x_);

    // Compute eta from theta
    float theta = std::atan2(stub_r_, std::abs(stub_z_));
    stub_eta_ = -std::log(std::tan(theta / 2.0f));
    if (stub_z_ < 0)
      stub_eta_ = -stub_eta_;

    // Stub local position
    stub_xLocal_ = stub.xLocal();
    stub_yLocal_ = stub.yLocal();
    stub_xerrLocal_ = stub.xerrLocal();
    stub_yerrLocal_ = stub.yerrLocal();

    // Inner hit positions
    inner_x_ = innerHit.xGlobal();
    inner_y_ = innerHit.yGlobal();
    inner_z_ = innerHit.zGlobal();
    inner_r_ = innerHit.rGlobal();
    inner_phi_ = std::atan2(inner_y_, inner_x_);

    inner_xLocal_ = innerHit.xLocal();
    inner_yLocal_ = innerHit.yLocal();
    inner_xerrLocal_ = innerHit.xerrLocal();
    inner_yerrLocal_ = innerHit.yerrLocal();

    // Inner hit global error covariance
    inner_xxGlobalErr_ = innerHit.xGlobalErr();
    inner_xyGlobalErr_ = innerHit.xyGlobalErr();
    inner_yyGlobalErr_ = innerHit.yGlobalErr();
    inner_xzGlobalErr_ = innerHit.xzGlobalErr();
    inner_yzGlobalErr_ = innerHit.yzGlobalErr();
    inner_zzGlobalErr_ = innerHit.zGlobalErr();

    // Outer hit positions (only valid for PS and SS stubs, not PHitOnly)
    if (hasOuterHit_) {
      auto const& outerHit = hitsView[outerHitIdx_];
      outer_x_ = outerHit.xGlobal();
      outer_y_ = outerHit.yGlobal();
      outer_z_ = outerHit.zGlobal();
      outer_r_ = outerHit.rGlobal();
      outer_phi_ = std::atan2(outer_y_, outer_x_);

      outer_xLocal_ = outerHit.xLocal();
      outer_yLocal_ = outerHit.yLocal();
      outer_xerrLocal_ = outerHit.xerrLocal();
      outer_yerrLocal_ = outerHit.yerrLocal();

      // Outer hit global error covariance
      outer_xxGlobalErr_ = outerHit.xGlobalErr();
      outer_xyGlobalErr_ = outerHit.xyGlobalErr();
      outer_yyGlobalErr_ = outerHit.yGlobalErr();
      outer_xzGlobalErr_ = outerHit.xzGlobalErr();
      outer_yzGlobalErr_ = outerHit.yzGlobalErr();
      outer_zzGlobalErr_ = outerHit.zGlobalErr();
    } else {
      outer_x_ = 0;
      outer_y_ = 0;
      outer_z_ = 0;
      outer_r_ = 0;
      outer_phi_ = 0;
      outer_xLocal_ = 0;
      outer_yLocal_ = 0;
      outer_xerrLocal_ = 0;
      outer_yerrLocal_ = 0;
      outer_xxGlobalErr_ = 0;
      outer_xyGlobalErr_ = 0;
      outer_yyGlobalErr_ = 0;
      outer_xzGlobalErr_ = 0;
      outer_yzGlobalErr_ = 0;
      outer_zzGlobalErr_ = 0;
    }

    // Compute sensor separations
    dr_sensor_ = outer_r_ - inner_r_;
    dz_sensor_ = outer_z_ - inner_z_;
    float dx = outer_x_ - inner_x_;
    float dy = outer_y_ - inner_y_;
    float dzsq = dz_sensor_ * dz_sensor_;
    d3_sensor_ = std::sqrt(dx * dx + dy * dy + dzsq);

    // Compute effective dr: project sensor separation onto radial direction
    // dr_effective = separation / (cosTilt + sinTilt * z/r)
    // Note: tiltAngle = atan2(dz, dr) measured from radial axis, so sinTilt > 0 for +z tilt
    float separation_cm = sensorSeparation_ * 0.1f;  // mm to cm
    float denominator = cosTilt_ + sinTilt_ * stub_z_ / stub_r_;
    if (std::abs(denominator) > 1e-6f) {
      dr_effective_ = separation_cm / denominator;
    } else {
      // Fallback to geometric dr if denominator is too small
      dr_effective_ = dr_sensor_;
    }

    // Raw differences (same as dr_sensor, dz_sensor for clarity)
    dr_ = dr_sensor_;
    dz_ = dz_sensor_;

    // dphi with wraparound handling
    dphi_raw_ = outer_phi_ - inner_phi_;
    if (dphi_raw_ > M_PI)
      dphi_raw_ -= 2.0f * M_PI;
    if (dphi_raw_ < -M_PI)
      dphi_raw_ += 2.0f * M_PI;

    // Compute dphi parallax correction using vector-based method (same as kernel)
    // Only apply for tilted barrel and endcap modules (not flat barrel, where local-x = phi => pC = 0)
    bool applyDphiParallax = !isFlat_ || !isBarrel_;
    if (applyDphiParallax) {
      // Use PHYSICAL inner hit position (same as kernel)
      // For dphi, globalLowUpNorm is used WITHOUT flip (it's already physical inner->outer)
      float gx_inner = isFlipped_ ? outer_x_ : inner_x_;
      float gy_inner = isFlipped_ ? outer_y_ : inner_y_;
      float gz_inner = isFlipped_ ? outer_z_ : inner_z_;
      float gR_inner = isFlipped_ ? outer_r_ : inner_r_;

      // Replicate the vector-based computeParallaxCorrection from the kernel
      float zLocX_dphi = globalLowUpNormX_;
      float zLocY_dphi = globalLowUpNormY_;
      float zLocZ_dphi = globalLowUpNormZ_;

      float x0X_dphi = localXInGlobalX_;
      float x0Y_dphi = localXInGlobalY_;
      float x0Z_dphi = localXInGlobalZ_;

      constexpr float eps_dphi = 1e-6f;
      float zN2 = zLocX_dphi * zLocX_dphi + zLocY_dphi * zLocY_dphi + zLocZ_dphi * zLocZ_dphi;
      float xN2 = x0X_dphi * x0X_dphi + x0Y_dphi * x0Y_dphi + x0Z_dphi * x0Z_dphi;

      if (zN2 > eps_dphi && xN2 > eps_dphi && gR_inner > eps_dphi) {
        if (std::abs(zN2 - 1.0f) > 1e-3f) {
          float inv = 1.0f / std::sqrt(zN2);
          zLocX_dphi *= inv;
          zLocY_dphi *= inv;
          zLocZ_dphi *= inv;
        }
        if (std::abs(xN2 - 1.0f) > 1e-3f) {
          float inv = 1.0f / std::sqrt(xN2);
          x0X_dphi *= inv;
          x0Y_dphi *= inv;
          x0Z_dphi *= inv;
        }
        float xDotZ = x0X_dphi * zLocX_dphi + x0Y_dphi * zLocY_dphi + x0Z_dphi * zLocZ_dphi;
        float xPX = x0X_dphi - xDotZ * zLocX_dphi;
        float xPY = x0Y_dphi - xDotZ * zLocY_dphi;
        float xPZ = x0Z_dphi - xDotZ * zLocZ_dphi;
        float xPN2 = xPX * xPX + xPY * xPY + xPZ * xPZ;
        if (xPN2 > eps_dphi) {
          float invN = 1.0f / std::sqrt(xPN2);
          float xLX = xPX * invN;
          float xLY = xPY * invN;
          float xLZ = xPZ * invN;
          float lVx_d = gx_inner * xLX + gy_inner * xLY + gz_inner * xLZ;
          float lVz_d = gx_inner * zLocX_dphi + gy_inner * zLocY_dphi + gz_inner * zLocZ_dphi;
          if (std::abs(lVz_d) > eps_dphi) {
            float pC_dphi = (lVx_d / lVz_d) * separation_cm;
            dphi_parallax_ = pC_dphi / gR_inner;
          } else {
            dphi_parallax_ = 0.0f;
          }
        } else {
          dphi_parallax_ = 0.0f;
        }
      } else {
        dphi_parallax_ = 0.0f;
      }
    } else {
      // Flat barrel: no dphi parallax correction (local-x = phi => pC = 0)
      dphi_parallax_ = 0.0f;
    }
    dphi_corrected_ = dphi_raw_ - dphi_parallax_;

    // dPhiDr (from stub, already computed with parallax correction)
    dPhiDr_ = stub.dPhiDr();
    dPhiDrError_ = stub.dPhiDrError();

    // dPhiDz (proposed for endcap analysis) - use raw dphi (not parallax-corrected)
    // The parallax correction was designed for barrel stub width computation, not for dPhiDz.
    // For endcap modules, dz is large and meaningful; for barrel modules, dz is small.
    if (std::abs(dz_) > 1e-6f && hasOuterHit_) {
      dPhiDz_ = dphi_raw_ / dz_;
      // Error estimation: use global phi variance σ²(φ) = (sin²φ·Cxx - 2·sinφ·cosφ·Cxy + cos²φ·Cyy) / r²
      // This correctly accounts for the local-y error contribution which leaks into phi
      // for modules where the strip direction is not exactly radial (important for endcap).
      float phi_i = std::atan2(inner_y_, inner_x_);
      float sinphi_i = std::sin(phi_i);
      float cosphi_i = std::cos(phi_i);
      float ri2 = inner_r_ * inner_r_;
      float sig2phi_i = (ri2 > 0.0f) ? (sinphi_i * sinphi_i * inner_xxGlobalErr_ -
                                         2.0f * sinphi_i * cosphi_i * inner_xyGlobalErr_ +
                                         cosphi_i * cosphi_i * inner_yyGlobalErr_) /
                                            ri2
                                      : 0.0f;
      float phi_o = std::atan2(outer_y_, outer_x_);
      float sinphi_o = std::sin(phi_o);
      float cosphi_o = std::cos(phi_o);
      float ro2 = outer_r_ * outer_r_;
      float sig2phi_o = (ro2 > 0.0f) ? (sinphi_o * sinphi_o * outer_xxGlobalErr_ -
                                         2.0f * sinphi_o * cosphi_o * outer_xyGlobalErr_ +
                                         cosphi_o * cosphi_o * outer_yyGlobalErr_) /
                                            ro2
                                      : 0.0f;
      float dphi_err = std::sqrt(sig2phi_i + sig2phi_o);
      dPhiDzError_ = dphi_err / std::abs(dz_);
    } else {
      dPhiDz_ = 0.0f;
      dPhiDzError_ = 0.0f;
    }

    // Eta consistency (for r-z line check)
    eta_inner_ = (inner_r_ > 1e-6f) ? inner_z_ / inner_r_ : 0.0f;
    eta_outer_ = (outer_r_ > 1e-6f) ? outer_z_ / outer_r_ : 0.0f;
    eta_diff_ = eta_outer_ - eta_inner_;

    // Width calculations
    // Use the same lower/upper convention as the kernel: width = lower - upper
    // When isFlipped=false: lower=inner, upper=outer
    // When isFlipped=true:  lower=outer, upper=inner
    float lx_lower = isFlipped_ ? outer_xLocal_ : inner_xLocal_;
    float lx_upper = isFlipped_ ? inner_xLocal_ : outer_xLocal_;
    width_raw_ = lx_lower - lx_upper;
    // Use the bend from stub which is based on parallax-corrected width
    width_ = stub.bend() * stub_r_;  // Reverse the bend = width/r formula

    // Compute parallax correction using same logic as OTStubFormationVectorHitStyleKernels
    // The parallax correction accounts for the track angle when computing stub width
    // For software lower sensor: if not flipped, lower=inner; if flipped, lower=outer
    float gx_lower = isFlipped_ ? outer_x_ : inner_x_;
    float gy_lower = isFlipped_ ? outer_y_ : inner_y_;
    float gz_lower = isFlipped_ ? outer_z_ : inner_z_;

    // zLoc direction: along the stack from LOWER -> UPPER (global, normalized)
    // globalLowUpNorm_ is stored as PHYSICAL inner->outer direction.
    // For width calculation, we need the TOPOLOGICAL lower->upper direction.
    // For flipped modules: physical inner->outer = upper->lower (topological),
    // so we need to flip the sign to get lower->upper.
    float flipSign = isFlipped_ ? -1.0f : 1.0f;
    float zLocX = flipSign * globalLowUpNormX_;
    float zLocY = flipSign * globalLowUpNormY_;
    float zLocZ = flipSign * globalLowUpNormZ_;

    // xAxis (global) is the direction of LOWER local-x
    float x0X = localXInGlobalX_;
    float x0Y = localXInGlobalY_;
    float x0Z = localXInGlobalZ_;

    // Compute parallax correction
    constexpr float eps = 1e-6f;
    float zLocN2 = zLocX * zLocX + zLocY * zLocY + zLocZ * zLocZ;
    float x0N2 = x0X * x0X + x0Y * x0Y + x0Z * x0Z;

    if (zLocN2 < eps || x0N2 < eps) {
      parallaxCorr_ = 0.0f;
    } else {
      // Normalize if needed
      if (std::abs(zLocN2 - 1.0f) > 1e-3f) {
        float invN = 1.0f / std::sqrt(zLocN2);
        zLocX *= invN;
        zLocY *= invN;
        zLocZ *= invN;
      }
      if (std::abs(x0N2 - 1.0f) > 1e-3f) {
        float invN = 1.0f / std::sqrt(x0N2);
        x0X *= invN;
        x0Y *= invN;
        x0Z *= invN;
      }

      // xPerp = x0 - (x0*zLoc) zLoc (project localX into plane orthogonal to zLoc)
      float x0DotZ = x0X * zLocX + x0Y * zLocY + x0Z * zLocZ;
      float xPerpX = x0X - x0DotZ * zLocX;
      float xPerpY = x0Y - x0DotZ * zLocY;
      float xPerpZ = x0Z - x0DotZ * zLocZ;

      float xPerpN2 = xPerpX * xPerpX + xPerpY * xPerpY + xPerpZ * xPerpZ;
      if (xPerpN2 < eps) {
        parallaxCorr_ = 0.0f;
      } else {
        float invXPerpN = 1.0f / std::sqrt(xPerpN2);
        float xLocX = xPerpX * invXPerpN;
        float xLocY = xPerpY * invXPerpN;
        float xLocZ = xPerpZ * invXPerpN;

        // Local components: lVx = gV * xLoc, lVz = gV * zLoc
        float lVx = gx_lower * xLocX + gy_lower * xLocY + gz_lower * xLocZ;
        float lVz = gx_lower * zLocX + gy_lower * zLocY + gz_lower * zLocZ;

        if (std::abs(lVz) < eps) {
          parallaxCorr_ = 0.0f;
        } else {
          // Normalized slope and correction
          float slope_x = lVx / lVz;
          // sensorSeparation_ is in mm, convert to cm
          parallaxCorr_ = slope_x * sensorSeparation_ * 0.1f;

          // Sanity check
          if (!std::isfinite(parallaxCorr_)) {
            parallaxCorr_ = 0.0f;
          }
        }
      }
    }

    // Bend and pT
    bend_ = stub.bend();
    ptEst_ = stub.ptEst();

    // Fill the tree
    tree_->Fill();
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(OTStubNtupleAnalyzer);
