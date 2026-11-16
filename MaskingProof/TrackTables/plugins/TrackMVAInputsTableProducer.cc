// TrackMVAInputsTableProducer (masking_proof, 2026-11-15): one row per reco::Track of a collection (e.g. initialStepTracks)
// with the 16 inputs of the Prompt track MVA, computed exactly as in RecoTracker/FinalTrackSelectors/plugins/
// DefaultTrackMVAClassifier.cc (gbrVals_[0..15] order), plus truth labels from a TrackToTrackingParticleAssociator
// (RecoToSim, the MTV hit associator): matched to any TP, matched to an efficiency TP (the given TP RefVector), duplicate
// (another track of the collection is matched to the same TP with a higher association quality), and the value of an
// optional MultiTrackSelector ValueMap<int> (-1 = rejected, else the quality mask). Training sample for a GBRForest
// (TrackMVAClassifierPrompt, GBRForestFileName) replacing the cut-based initialStep high-purity selection.
#include <cmath>
#include <map>
#include <memory>
#include <vector>

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoTracker/FinalTrackSelectors/interface/getBestVertex.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

class TrackMVAInputsTableProducer : public edm::global::EDProducer<> {
public:
  explicit TrackMVAInputsTableProducer(const edm::ParameterSet& cfg)
      : name_(cfg.getParameter<std::string>("name")),
        tracksToken_(consumes(cfg.getParameter<edm::InputTag>("src"))),
        viewToken_(consumes(cfg.getParameter<edm::InputTag>("src"))),
        beamSpotToken_(consumes(cfg.getParameter<edm::InputTag>("beamspot"))),
        verticesToken_(consumes(cfg.getParameter<edm::InputTag>("vertices"))),
        tpToken_(consumes(cfg.getParameter<edm::InputTag>("trackingParticles"))),
        effTPToken_(consumes(cfg.getParameter<edm::InputTag>("efficiencyTPs"))),
        assocToken_(consumes(cfg.getParameter<edm::InputTag>("associator"))) {
    auto const sel = cfg.getParameter<edm::InputTag>("selector");
    if (!sel.label().empty())
      selToken_ = consumes(sel);
    produces<nanoaod::FlatTable>();
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("name", "TrkMVA");
    desc.add<edm::InputTag>("src", edm::InputTag("initialStepTracks"));
    desc.add<edm::InputTag>("beamspot", edm::InputTag("offlineBeamSpot"));
    desc.add<edm::InputTag>("vertices", edm::InputTag("firstStepPrimaryVertices"));
    desc.add<edm::InputTag>("trackingParticles", edm::InputTag("mix", "MergedTrackTruth"));
    desc.add<edm::InputTag>("efficiencyTPs", edm::InputTag("tpMatchSelectedTPs"));
    desc.add<edm::InputTag>("associator", edm::InputTag("quickTrackAssociatorByHits"));
    desc.add<edm::InputTag>("selector", edm::InputTag("initialStepSelector", "initialStep"));
    descriptions.addWithDefaultLabel(desc);
  }

  void produce(edm::StreamID, edm::Event& ev, edm::EventSetup const&) const override {
    auto const& tracksH = ev.getHandle(tracksToken_);
    auto const& tracks = *tracksH;
    auto const& bs = ev.get(beamSpotToken_);
    auto const& vertices = ev.get(verticesToken_);
    auto const& tpH = ev.getHandle(tpToken_);
    auto const& effTPs = ev.get(effTPToken_);
    auto const& assoc = ev.get(assocToken_);
    edm::Handle<edm::ValueMap<int>> selH;
    if (!selToken_.isUninitialized())
      ev.getByToken(selToken_, selH);

    auto const& viewH = ev.getHandle(viewToken_);  // the associator works on a View of the same collection
    reco::RecoToSimCollection r2s = assoc.associateRecoToSim(viewH, tpH);

    std::map<unsigned, bool> isEff;
    for (auto const& tp : effTPs)
      isEff[tp.key()] = true;

    size_t const n = tracks.size();
    std::vector<float> f[16];
    for (auto& v : f)
      v.resize(n);
    std::vector<int> matched(n, 0), matchedEff(n, 0), dup(n, 0), sel(n, -2), algo(n), oriAlgo(n), tpKey(n, -1);
    std::vector<float> quality(n, 0.f), tpPt(n, -1.f);
    std::map<int, std::pair<double, size_t>> best;  // TP key -> (quality, track index) of the best-matched track
    for (size_t i = 0; i < n; ++i) {
      auto const& trk = tracks[i];
      fill(trk, bs, vertices, f, i);
      algo[i] = trk.algo();
      oriAlgo[i] = trk.originalAlgo();
      edm::RefToBase<reco::Track> ref(viewH, i);
      if (selH.isValid())
        sel[i] = (*selH)[reco::TrackRef(tracksH, i)];
      auto it = r2s.find(ref);
      if (it != r2s.end() && !it->val.empty()) {
        auto const& tp = it->val.front();
        matched[i] = 1;
        tpKey[i] = tp.first.key();
        quality[i] = tp.second;
        tpPt[i] = tp.first->pt();
        matchedEff[i] = isEff.count(tp.first.key()) ? 1 : 0;
        auto b = best.find(tpKey[i]);
        if (b == best.end() || tp.second > b->second.first)
          best[tpKey[i]] = {tp.second, i};
      }
    }
    for (size_t i = 0; i < n; ++i)
      if (matched[i] && best[tpKey[i]].second != i)
        dup[i] = 1;

    static const char* names[16] = {"pt",      "lostmidfrac", "minlost",    "nhits",   "relpterr",  "eta",
                                    "chi2n_no1dmod", "chi2n", "nlayerslost", "nlayers3D", "nlayers", "ndof",
                                    "absd0PV", "absdzPV",     "absdz",      "absd0"};
    auto table = std::make_unique<nanoaod::FlatTable>(n, name_, false);
    for (int k = 0; k < 16; ++k)
      table->addColumn<float>(names[k], f[k], std::string("Prompt track MVA input ") + std::to_string(k));
    table->addColumn<int>("matched", matched, "matched to any TrackingParticle (RecoToSim)");
    table->addColumn<int>("matchedEff", matchedEff, "matched TP passes the efficiency selection");
    table->addColumn<int>("duplicate", dup, "another track of the collection is better matched to the same TP");
    table->addColumn<float>("assocQuality", quality, "RecoToSim association quality");
    table->addColumn<float>("tpPt", tpPt, "matched TP pT (-1 if none)");
    table->addColumn<int>("selector", sel, "cut-based selector ValueMap value (-1 rejected, -2 absent)");
    table->addColumn<int>("algo", algo, "track algo");
    table->addColumn<int>("originalAlgo", oriAlgo, "track original algo");
    ev.put(std::move(table));
  }

private:
  static void fill(reco::Track const& trk,
                   reco::BeamSpot const& beamSpot,
                   reco::VertexCollection const& vertices,
                   std::vector<float>* f,
                   size_t i) {
    // verbatim from DefaultTrackMVAClassifier.cc (mva<PROMPT=true>::operator())
    float chi2n = trk.normalizedChi2();
    float const chi2n_no1Dmod = chi2n;
    int count1dhits = 0;
    for (auto ith = trk.recHitsBegin(); ith != trk.recHitsEnd(); ++ith) {
      if ((*ith)->dimension() == 1)
        ++count1dhits;
    }
    if (count1dhits > 0) {
      float chi2 = trk.chi2();
      float ndof = trk.ndof();
      chi2n = (chi2 + count1dhits) / float(ndof + count1dhits);
    }
    int lostIn = trk.hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
    int lostOut = trk.hitPattern().numberOfLostHits(reco::HitPattern::MISSING_OUTER_HITS);
    Point bestVertex = getBestVertex(trk, vertices);
    f[0][i] = trk.pt();
    f[1][i] = static_cast<float>(trk.numberOfLostHits()) / static_cast<float>(trk.numberOfValidHits() + trk.numberOfLostHits());
    f[2][i] = std::min(lostIn, lostOut);
    f[3][i] = trk.numberOfValidHits();
    f[4][i] = float(trk.ptError()) / std::max(float(trk.pt()), 0.000001f);
    f[5][i] = trk.eta();
    f[6][i] = chi2n_no1Dmod;
    f[7][i] = chi2n;
    f[8][i] = trk.hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::TRACK_HITS);
    f[9][i] = trk.hitPattern().pixelLayersWithMeasurement() + trk.hitPattern().numberOfValidStripLayersWithMonoAndStereo();
    f[10][i] = trk.hitPattern().trackerLayersWithMeasurement();
    f[11][i] = trk.ndof();
    f[12][i] = std::abs(trk.dxy(bestVertex));
    f[13][i] = std::abs(trk.dz(bestVertex));
    f[14][i] = std::abs(trk.dz(beamSpot.position()));
    f[15][i] = std::abs(trk.dxy(beamSpot.position()));
  }

  std::string const name_;
  edm::EDGetTokenT<reco::TrackCollection> const tracksToken_;
  edm::EDGetTokenT<edm::View<reco::Track>> const viewToken_;
  edm::EDGetTokenT<reco::BeamSpot> const beamSpotToken_;
  edm::EDGetTokenT<reco::VertexCollection> const verticesToken_;
  edm::EDGetTokenT<TrackingParticleCollection> const tpToken_;
  edm::EDGetTokenT<TrackingParticleRefVector> const effTPToken_;
  edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> const assocToken_;
  edm::EDGetTokenT<edm::ValueMap<int>> selToken_;
};

DEFINE_FWK_MODULE(TrackMVAInputsTableProducer);
