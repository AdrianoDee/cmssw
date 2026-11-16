// SimPixelTrackMatcher (masking_proof, 2026-11-13): splits the SimPixelTracks (PR #49194) of the true TrackingParticles by
// whether the CA reconstructed them, and writes their hit pattern.
//  - product "lost": the SimPixelTracks whose TrackingParticle no collection of `lostIfNoneOf` found (MTV hit associator):
//    input of a second SimPixelTrackAnalyzer, so the CA cut variables and loss categories can be compared between all TPs
//    and the TPs the CA never builds
//  - flat table: one row per SimPixelTrack with the TP kinematics, the hit pattern (bitmask of the CA layer IDs of its
//    RecHits, number of layers and RecHits, first two layers) and a found_<name> flag per collection of `tracks`
#include <memory>
#include <string>
#include <vector>

#include "DataFormats/Common/interface/RefToBaseVector.h"
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
#include "SimDataFormats/TrackingAnalysis/interface/SimPixelTrack.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

class SimPixelTrackMatcher : public edm::global::EDProducer<> {
public:
  explicit SimPixelTrackMatcher(const edm::ParameterSet& iConfig)
      : name_(iConfig.getParameter<std::string>("name")),
        sptToken_(consumes(iConfig.getParameter<edm::InputTag>("simPixelTrackSrc"))),
        assocToken_(consumes(iConfig.getParameter<edm::InputTag>("associator"))),
        names_(iConfig.getParameter<std::vector<std::string>>("names")),
        lostIfNoneOf_(iConfig.getParameter<std::vector<std::string>>("lostIfNoneOf")),
        winPtMin_(iConfig.getParameter<double>("windowPtMin")),
        winPtMax_(iConfig.getParameter<double>("windowPtMax")),
        lostToken_(produces<SimPixelTrackCollection>("lost")),
        winToken_(produces<SimPixelTrackCollection>("window")),
        lostWinToken_(produces<SimPixelTrackCollection>("lostWindow")) {
    for (auto const& tag : iConfig.getParameter<std::vector<edm::InputTag>>("tracks"))
      trackTokens_.push_back(consumes<edm::View<reco::Track>>(tag));
    if (names_.size() != trackTokens_.size())
      throw cms::Exception("Configuration") << "SimPixelTrackMatcher: names and tracks differ in length";
    for (auto const& n : lostIfNoneOf_) {
      auto it = std::find(names_.begin(), names_.end(), n);
      if (it == names_.end())
        throw cms::Exception("Configuration") << "SimPixelTrackMatcher: lostIfNoneOf name " << n << " not in names";
      lostIdx_.push_back(it - names_.begin());
    }
    produces<nanoaod::FlatTable>();
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("name", "SPT");
    desc.add<edm::InputTag>("simPixelTrackSrc", edm::InputTag("simPixelTrackProducerPhase2"));
    desc.add<edm::InputTag>("associator", edm::InputTag("quickTrackAssociatorByHits"));
    desc.add<std::vector<std::string>>("names", {});
    desc.add<std::vector<edm::InputTag>>("tracks", {});
    desc.add<std::vector<std::string>>("lostIfNoneOf", {})
        ->setComment("names of the collections defining 'reconstructed': lost = found by none of them");
    desc.add<double>("windowPtMin", 0.)->setComment("pT window of the 'window' and 'lostWindow' products");
    desc.add<double>("windowPtMax", 1e9);
    descriptions.addWithDefaultLabel(desc);
  }

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const override {
    const auto& spts = iEvent.get(sptToken_);
    const auto& assoc = iEvent.get(assocToken_);
    const unsigned n = spts.size();
    TrackingParticleRefVector tps;
    for (auto const& s : spts)
      tps.push_back(s.trackingParticle());

    std::vector<std::vector<uint8_t>> found(trackTokens_.size(), std::vector<uint8_t>(n, 0));
    for (unsigned c = 0; c < trackTokens_.size(); ++c) {
      edm::Handle<edm::View<reco::Track>> h;
      iEvent.getByToken(trackTokens_[c], h);
      edm::RefToBaseVector<reco::Track> trks;
      for (unsigned j = 0; j < h->size(); ++j)
        trks.push_back(h->refAt(j));
      const reco::SimToRecoCollection s2r = assoc.associateSimToReco(trks, tps);
      for (unsigned i = 0; i < n; ++i) {
        auto it = s2r.find(tps[i]);
        found[c][i] = (it != s2r.end() && !it->val.empty()) ? 1 : 0;
      }
    }

    auto lost = std::make_unique<SimPixelTrackCollection>();
    auto win = std::make_unique<SimPixelTrackCollection>();
    auto lostWin = std::make_unique<SimPixelTrackCollection>();
    std::vector<float> pt(n), eta(n), phi(n), vxy(n);
    std::vector<int> nLayers(n), nRecHits(n), layer0(n), layer1(n);
    std::vector<uint32_t> layerMask(n);
    std::vector<uint8_t> signal(n), isLost(n);
    for (unsigned i = 0; i < n; ++i) {
      const auto& s = spts[i];
      const auto& tp = *s.trackingParticle();
      pt[i] = tp.pt();
      eta[i] = tp.eta();
      phi[i] = tp.phi();
      vxy[i] = tp.vertex().rho();
      signal[i] = (tp.eventId().bunchCrossing() == 0 && tp.eventId().event() == 0) ? 1 : 0;
      nLayers[i] = s.numLayers();
      nRecHits[i] = s.numRecHits();
      uint32_t mask = 0;
      for (auto l : s.layerIds())
        if (l < 32)
          mask |= (1u << l);
      layerMask[i] = mask;
      const auto ids = s.layerIds();
      layer0[i] = ids.empty() ? -1 : ids[0];
      layer1[i] = ids.size() < 2 ? -1 : ids[1];
      bool any = false;
      for (auto c : lostIdx_)
        any = any || found[c][i];
      isLost[i] = any ? 0 : 1;
      if (!any)
        lost->push_back(s);
      if (tp.pt() >= winPtMin_ && tp.pt() < winPtMax_) {
        win->push_back(s);
        if (!any)
          lostWin->push_back(s);
      }
    }

    auto table = std::make_unique<nanoaod::FlatTable>(n, name_, false, false);
    table->setDoc("SimPixelTracks: TP kinematics, CA-layer hit pattern, which collections found the TP");
    table->addColumn<float>("pt", pt, "TP pT");
    table->addColumn<float>("eta", eta, "TP eta");
    table->addColumn<float>("phi", phi, "TP phi");
    table->addColumn<float>("vxy", vxy, "TP production vertex rho");
    table->addColumn<uint8_t>("signal", signal, "signal event TP");
    table->addColumn<int>("nLayers", nLayers, "CA layers with RecHits");
    table->addColumn<int>("nRecHits", nRecHits, "RecHits");
    table->addColumn<uint32_t>("layerMask", layerMask, "bit l set if a RecHit is on CA layer l (l < 32)");
    table->addColumn<int>("layer0", layer0, "CA layer of the first RecHit");
    table->addColumn<int>("layer1", layer1, "CA layer of the second RecHit");
    table->addColumn<uint8_t>("lost", isLost, "found by none of lostIfNoneOf");
    for (unsigned c = 0; c < trackTokens_.size(); ++c)
      table->addColumn<uint8_t>("found_" + names_[c], found[c], "found by " + names_[c]);
    iEvent.put(std::move(table));
    iEvent.put(lostToken_, std::move(lost));
    iEvent.put(winToken_, std::move(win));
    iEvent.put(lostWinToken_, std::move(lostWin));
  }

  const std::string name_;
  const edm::EDGetTokenT<SimPixelTrackCollection> sptToken_;
  const edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> assocToken_;
  const std::vector<std::string> names_;
  const std::vector<std::string> lostIfNoneOf_;
  std::vector<unsigned> lostIdx_;
  std::vector<edm::EDGetTokenT<edm::View<reco::Track>>> trackTokens_;
  const double winPtMin_, winPtMax_;
  const edm::EDPutTokenT<SimPixelTrackCollection> lostToken_, winToken_, lostWinToken_;
};

DEFINE_FWK_MODULE(SimPixelTrackMatcher);
