// Phase2HitTruthTableProducer (masking_proof, 2026-11-11): truth of every hit of the CA global hit index space of
// the Phase2/Phase2OT topologies, for the triplet-gate training (train_triplet_dnn.py reads HitTruth_tpKey/tpPt/tpEta
// at the triplet's hit indices h1/h2/h3). Port of RecoTracker/PixelSeeding/test/models/HitTruthTableProducer.cc,
// which serves the vector-hit space, to the space of Phase2OT: pixel rechits first, then the outer-tracker rechits
// of Phase2OTRecHitsSoAConverter (the P sensors of the PS modules of the OT barrel).
//   pixel hit: index = pixel moduleStart[geomDet index] + cluster.originalId()            (as the original producer)
//   OT hit:    index = OT moduleStart[module offset] + position in its det set, the module offset being the order of
//              the module among the PS P-sensor TOB modules in TrackerGeometry::detUnits() (as the converter)
// Truth: TrackerHitAssociator (associateHitId for pixel, associatePhase2TrackerRecHit for OT) -> SimTrack ->
// TrackingParticle; the smallest selected TP sharing the hit. Every mapped hit is checked against the SoA in the
// transverse plane, after removing the offset common to the event (the SoA positions are in the beamspot frame).
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsHost.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

class Phase2HitTruthTableProducer : public edm::stream::EDProducer<> {
public:
  explicit Phase2HitTruthTableProducer(const edm::ParameterSet& iConfig);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  const std::string tableName_;
  const edm::EDGetTokenT<reco::TrackingRecHitHost> pixelHitsSoAToken_;
  const edm::EDGetTokenT<reco::TrackingRecHitHost> otHitsSoAToken_;
  const edm::EDGetTokenT<SiPixelRecHitCollection> pixelRecHitToken_;
  const edm::EDGetTokenT<Phase2TrackerRecHit1DCollectionNew> otRecHitToken_;
  const edm::EDGetTokenT<std::vector<TrackingParticle>> tpToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  TrackerHitAssociator::Config hitAssocConfig_;
  const bool tpSignalOnly_, tpChargedOnly_;
  const double tpMinPt_, tpMaxEta_, tpMaxTip_, tpMaxVtxZ_, maxPosDiff_;
};

Phase2HitTruthTableProducer::Phase2HitTruthTableProducer(const edm::ParameterSet& iConfig)
    : tableName_(iConfig.getParameter<std::string>("tableName")),
      pixelHitsSoAToken_(consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitSoASrc"))),
      otHitsSoAToken_(consumes(iConfig.getParameter<edm::InputTag>("otRecHitSoASrc"))),
      pixelRecHitToken_(consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"))),
      otRecHitToken_(consumes(iConfig.getParameter<edm::InputTag>("otRecHitSrc"))),
      tpToken_(consumes(iConfig.getParameter<edm::InputTag>("trackingParticleSrc"))),
      geomToken_(esConsumes()),
      hitAssocConfig_(iConfig.getParameter<edm::ParameterSet>("hitAssociatorConfig"), consumesCollector()),
      tpSignalOnly_(iConfig.getParameter<bool>("TP_signalOnly")),
      tpChargedOnly_(iConfig.getParameter<bool>("TP_chargedOnly")),
      tpMinPt_(iConfig.getParameter<double>("TP_minPt")),
      tpMaxEta_(iConfig.getParameter<double>("TP_maxEta")),
      tpMaxTip_(iConfig.getParameter<double>("TP_maxTip")),
      tpMaxVtxZ_(iConfig.getParameter<double>("TP_maxVtxZ")),
      maxPosDiff_(iConfig.getParameter<double>("maxPositionMismatch")) {
  produces<nanoaod::FlatTable>(tableName_);
}

void Phase2HitTruthTableProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& pixSoA = iEvent.get(pixelHitsSoAToken_);
  const auto& otSoA = iEvent.get(otHitsSoAToken_);
  const auto& pixHits = iEvent.get(pixelRecHitToken_);
  const auto& otHits = iEvent.get(otRecHitToken_);
  const auto& tps = iEvent.get(tpToken_);
  const TrackerGeometry& geom = iSetup.getData(geomToken_);

  const auto pixView = pixSoA.const_view().trackingHits();
  const auto otView = otSoA.const_view().trackingHits();
  const auto pixModules = pixSoA.const_view().hitModules();
  const auto otModules = otSoA.const_view().hitModules();
  const uint32_t nPix = pixSoA.nHits(), nOT = otSoA.nHits(), nTot = nPix + nOT;

  // TrackingParticle selection and SimTrack -> TP
  std::map<SimHitIdpr, uint32_t> simTrackToTP;
  std::vector<bool> tpSelected(tps.size(), false);
  for (uint32_t i = 0; i < tps.size(); ++i) {
    const auto& tp = tps[i];
    for (const auto& g4 : tp.g4Tracks())
      simTrackToTP[{g4.trackId(), g4.eventId()}] = i;
    tpSelected[i] = !(tpChargedOnly_ && tp.charge() == 0) &&
                    !(tpSignalOnly_ && (tp.eventId().bunchCrossing() != 0 || tp.eventId().event() != 0)) &&
                    tp.pt() >= tpMinPt_ && std::abs(tp.eta()) <= tpMaxEta_ && std::abs(tp.d0()) <= tpMaxTip_ &&
                    std::abs(tp.vz()) <= tpMaxVtxZ_;
  }
  TrackerHitAssociator assoc(iEvent, hitAssocConfig_);

  std::vector<int> tpKey(nTot, -1);
  std::vector<float> tpPt(nTot, std::numeric_limits<float>::quiet_NaN()), tpEta(tpPt);
  auto label = [&](uint32_t idx, std::vector<SimHitIdpr> const& ids) {
    std::set<uint32_t> s;
    for (auto const& id : ids) {
      auto it = simTrackToTP.find(id);
      if (it != simTrackToTP.end() && tpSelected[it->second])
        s.insert(it->second);
    }
    if (!s.empty()) {
      tpKey[idx] = int(*s.begin());
      tpPt[idx] = tps[*s.begin()].pt();
      tpEta[idx] = tps[*s.begin()].eta();
    }
  };
  // position check: the SoA is in the beamspot frame, so the per-event offset (the first mapped hit's) is removed
  bool haveOffset = false;
  float offX = 0.f, offY = 0.f;
  uint32_t nChecked = 0, nMismatch = 0, nUnmapped = 0;
  auto check = [&](GlobalPoint const& gp, float sx, float sy) {
    const float dx = gp.x() - sx, dy = gp.y() - sy;
    if (!haveOffset) {
      offX = dx, offY = dy, haveOffset = true;
    }
    ++nChecked;
    if (std::abs(dx - offX) > maxPosDiff_ || std::abs(dy - offY) > maxPosDiff_)
      ++nMismatch;
  };

  // pixel hits
  for (auto const& detSet : pixHits) {
    const GeomDet* det = geom.idToDetUnit(DetId(detSet.detId()));
    if (!det)
      continue;
    const uint32_t base = pixModules.moduleStart()[det->index()];
    for (auto const& rh : detSet) {
      const uint32_t idx = base + rh.cluster()->originalId();
      if (idx >= nPix) {
        ++nUnmapped;
        continue;
      }
      check(det->toGlobal(rh.localPosition()), pixView[idx].xGlobal(), pixView[idx].yGlobal());
      label(idx, assoc.associateHitId(rh));
    }
  }
  // OT hits: module offset = order of the module among the PS P-sensor TOB modules of TrackerGeometry::detUnits()
  std::unordered_map<uint32_t, uint32_t> otOffset;
  for (auto const* du : geom.detUnits()) {
    const DetId id(du->geographicalId());
    if (geom.getDetectorType(id) == TrackerGeometry::ModuleType::Ph2PSP && id.subdetId() == StripSubdetector::TOB)
      otOffset.emplace(id.rawId(), uint32_t(otOffset.size()));
  }
  for (auto const& detSet : otHits) {
    auto it = otOffset.find(detSet.detId());
    if (it == otOffset.end())
      continue;  // not a module of the CA
    const GeomDet* det = geom.idToDet(DetId(detSet.detId()));
    // moduleStart of the OT SoA counts from the first OT hit in the global space (it starts at nPix)
    uint32_t idx = otModules[it->second].moduleStart();
    for (auto const& rh : detSet) {
      if (idx < nPix || idx >= nTot) {
        ++nUnmapped;
        ++idx;
        continue;
      }
      check(det->toGlobal(rh.localPosition()), otView[idx - nPix].xGlobal(), otView[idx - nPix].yGlobal());
      std::vector<SimHitIdpr> ids;
      assoc.associatePhase2TrackerRecHit(&rh, ids);
      label(idx, ids);
      ++idx;
    }
  }
  if (nMismatch > 0 || nUnmapped > 0)
    edm::LogWarning("Phase2HitTruthTableProducer")
        << "hit index mapping: " << nMismatch << " position mismatches and " << nUnmapped << " unmapped hits out of "
        << nChecked << " checked (pixel " << nPix << ", OT " << nOT << ")";
  else
    LogDebug("Phase2HitTruthTableProducer") << "hit index mapping checked on " << nChecked << " hits";

  auto table = std::make_unique<nanoaod::FlatTable>(nTot, tableName_, false, false);
  table->setDoc("truth of the CA hits (pixel rechits, then OT rechits of the SoA converter)");
  table->addColumn<int>("tpKey", tpKey, "matched TrackingParticle key (-1 if none)");
  table->addColumn<float>("tpPt", tpPt, "matched TP pT [GeV] (NaN if none)");
  table->addColumn<float>("tpEta", tpEta, "matched TP eta (NaN if none)");
  iEvent.put(std::move(table), tableName_);
}

void Phase2HitTruthTableProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("tableName", "HitTruth");
  desc.add<edm::InputTag>("pixelRecHitSoASrc", edm::InputTag("siPixelRecHitsPreSplittingAlpaka"));
  desc.add<edm::InputTag>("otRecHitSoASrc", edm::InputTag("phase2OTRecHitsSoAConverter"));
  desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("siPixelRecHits"))
      ->setComment("legacy pixel rechits converted from the SoA (cluster originalId = position in the SoA module)");
  desc.add<edm::InputTag>("otRecHitSrc", edm::InputTag("siPhase2RecHits"));
  desc.add<edm::InputTag>("trackingParticleSrc", edm::InputTag("mix", "MergedTrackTruth"));
  desc.add<bool>("TP_signalOnly", false);
  desc.add<bool>("TP_chargedOnly", true);
  desc.add<double>("TP_minPt", 0.0);
  desc.add<double>("TP_maxEta", 4.5);
  desc.add<double>("TP_maxTip", 60.0);
  desc.add<double>("TP_maxVtxZ", 60.0);
  desc.add<double>("maxPositionMismatch", 1e-3)->setComment("cm, transverse, after the per-event beamspot offset");
  edm::ParameterSetDescription hitAssocDesc;
  hitAssocDesc.add<bool>("associatePixel", true);
  hitAssocDesc.add<bool>("associateStrip", true);
  hitAssocDesc.add<bool>("usePhase2Tracker", true);
  hitAssocDesc.add<bool>("associateRecoTracks", false);
  hitAssocDesc.add<bool>("associateHitbySimTrack", false);
  hitAssocDesc.add<edm::InputTag>("phase2TrackerSimLinkSrc", edm::InputTag("simSiPixelDigis", "Tracker"));
  hitAssocDesc.add<edm::InputTag>("pixelSimLinkSrc", edm::InputTag("simSiPixelDigis", "Pixel"));
  hitAssocDesc.add<edm::InputTag>("stripSimLinkSrc", edm::InputTag("simSiStripDigis"));
  hitAssocDesc.add<std::vector<std::string>>("ROUList",
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

DEFINE_FWK_MODULE(Phase2HitTruthTableProducer);
