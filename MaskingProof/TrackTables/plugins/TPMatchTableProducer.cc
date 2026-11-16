// TPMatchTableProducer (masking_proof, 2026-11-12): one row per efficiency TrackingParticle (the RefVector of a
// TrackingParticleRefSelector configured like the MultiTrackValidator's efficiency selection), with its kinematics, its
// number of pixel / tracker layers (trackingParticleNumberOfLayersProducer, as in the MTV) and, for every configured
// track collection, whether the collection found it (a SimToReco association of the MTV's hit associator). Used to
// break the v7 vs legacy efficiency gap down by TP class (pixel layers, eta, pT) and by legacy iteration.
#include <memory>
#include <string>
#include <vector>

#include "DataFormats/Common/interface/RefToBaseVector.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"

class TPMatchTableProducer : public edm::global::EDProducer<> {
public:
  explicit TPMatchTableProducer(const edm::ParameterSet& iConfig)
      : name_(iConfig.getParameter<std::string>("name")),
        tpToken_(consumes(iConfig.getParameter<edm::InputTag>("tpSrc"))),
        pixLayToken_(consumes(iConfig.getParameter<edm::InputTag>("pixelLayers"))),
        trkLayToken_(consumes(iConfig.getParameter<edm::InputTag>("trackerLayers"))),
        assocToken_(consumes(iConfig.getParameter<edm::InputTag>("associator"))),
        names_(iConfig.getParameter<std::vector<std::string>>("names")) {
    for (auto const& tag : iConfig.getParameter<std::vector<edm::InputTag>>("tracks"))
      trackTokens_.push_back(consumes<edm::View<reco::Track>>(tag));
    if (names_.size() != trackTokens_.size())
      throw cms::Exception("Configuration") << "TPMatchTableProducer: names and tracks differ in length";
    produces<nanoaod::FlatTable>();
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("name", "TP");
    desc.add<edm::InputTag>("tpSrc", edm::InputTag("tpMatchSelectedTPs"))
        ->setComment("RefVector of the efficiency TrackingParticles (TrackingParticleRefSelector)");
    desc.add<edm::InputTag>("pixelLayers", edm::InputTag("trackingParticleNumberOfLayersProducer", "pixelLayers"));
    desc.add<edm::InputTag>("trackerLayers", edm::InputTag("trackingParticleNumberOfLayersProducer", "trackerLayers"));
    desc.add<edm::InputTag>("associator", edm::InputTag("quickTrackAssociatorByHits"));
    desc.add<std::vector<std::string>>("names", {})->setComment("column suffix of each collection: found_<name>");
    desc.add<std::vector<edm::InputTag>>("tracks", {})->setComment("track collections (reco::Track views)");
    descriptions.addWithDefaultLabel(desc);
  }

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const override {
    const auto& tps = iEvent.get(tpToken_);
    const auto& pixLay = iEvent.get(pixLayToken_);
    const auto& trkLay = iEvent.get(trkLayToken_);
    const auto& assoc = iEvent.get(assocToken_);
    const unsigned n = tps.size();

    std::vector<float> pt(n), eta(n), phi(n), vxy(n), vz(n);
    std::vector<int> pdgId(n), nPixLay(n), nTrkLay(n);
    std::vector<uint8_t> signal(n);
    for (unsigned i = 0; i < n; ++i) {
      const auto& tp = *tps[i];
      pt[i] = tp.pt();
      eta[i] = tp.eta();
      phi[i] = tp.phi();
      vxy[i] = tp.vertex().rho();
      vz[i] = tp.vz();
      pdgId[i] = tp.pdgId();
      nPixLay[i] = pixLay[tps[i]];
      nTrkLay[i] = trkLay[tps[i]];
      signal[i] = (tp.eventId().bunchCrossing() == 0 && tp.eventId().event() == 0) ? 1 : 0;
    }

    auto table = std::make_unique<nanoaod::FlatTable>(n, name_, false, false);
    table->setDoc("efficiency TrackingParticles and the collections that found them");
    table->addColumn<float>("pt", pt, "TP pT");
    table->addColumn<float>("eta", eta, "TP eta");
    table->addColumn<float>("phi", phi, "TP phi");
    table->addColumn<float>("vxy", vxy, "TP production vertex rho");
    table->addColumn<float>("vz", vz, "TP production vertex z");
    table->addColumn<int>("pdgId", pdgId, "TP pdgId");
    table->addColumn<int>("nPixelLayers", nPixLay, "pixel layers with sim hits");
    table->addColumn<int>("nTrackerLayers", nTrkLay, "tracker layers with sim hits");
    table->addColumn<uint8_t>("signal", signal, "1 for the signal event (in-time, event 0)");

    for (unsigned c = 0; c < trackTokens_.size(); ++c) {
      edm::Handle<edm::View<reco::Track>> h;
      iEvent.getByToken(trackTokens_[c], h);
      edm::RefToBaseVector<reco::Track> trks;
      for (unsigned j = 0; j < h->size(); ++j)
        trks.push_back(h->refAt(j));
      const reco::SimToRecoCollection s2r = assoc.associateSimToReco(trks, tps);
      std::vector<uint8_t> found(n, 0);
      for (unsigned i = 0; i < n; ++i) {
        auto it = s2r.find(tps[i]);
        found[i] = (it != s2r.end() && !it->val.empty()) ? 1 : 0;
      }
      table->addColumn<uint8_t>("found_" + names_[c], found, "found by " + names_[c]);
    }
    iEvent.put(std::move(table));
  }

  const std::string name_;
  const edm::EDGetTokenT<TrackingParticleRefVector> tpToken_;
  const edm::EDGetTokenT<edm::ValueMap<unsigned int>> pixLayToken_;
  const edm::EDGetTokenT<edm::ValueMap<unsigned int>> trkLayToken_;
  const edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> assocToken_;
  const std::vector<std::string> names_;
  std::vector<edm::EDGetTokenT<edm::View<reco::Track>>> trackTokens_;
};

DEFINE_FWK_MODULE(TPMatchTableProducer);
