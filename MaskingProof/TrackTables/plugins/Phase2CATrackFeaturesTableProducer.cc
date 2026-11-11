// Phase2CATrackFeaturesTableProducer (masking_proof, 2026-11-11): the track classifier's 12-feature vector of every
// CA track of a Phase2/Phase2OT topology (pixel + outer-tracker rechit MultiView, no vector hits), as a nano flat
// table. The features come from caTrackFeatures::fill on the same hit index space Kernel_classifyTracks uses (pixel
// rechits first, then the OT rechits of the SoA converter), so the training dataset and the in-kernel evaluation
// agree bit for bit. Columns: the caTrackFeatures::Features names of RecoTracker/PixelSeeding/test/models/
// CATrackFeaturesTableProducer.cc (the FEATS of train_disp_nano.py), plus phi (join key vs the truth table), pt,
// eta, quality and fillOk.
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/SoATemplate/interface/SoAConstMultiView.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsHost.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoTracker/PixelSeeding/interface/CATrackFeatures.h"

// the hit view of the Phase2/Phase2OT CA (caStructures::HitsMultiView of RecoTracker/PixelSeeding/plugins/alpaka/CAStructures.h)
using Phase2HitsMultiView = SoAConstMultiView<reco::TrackingRecHitConstView, 2>;

class Phase2CATrackFeaturesTableProducer : public edm::global::EDProducer<> {
public:
  explicit Phase2CATrackFeaturesTableProducer(const edm::ParameterSet& iConfig)
      : name_(iConfig.getParameter<std::string>("name")),
        tracksToken_(consumes<reco::TracksHost>(iConfig.getParameter<edm::InputTag>("trackSrc"))),
        pixelHitsToken_(consumes<reco::TrackingRecHitHost>(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"))),
        otHitsToken_(consumes<reco::TrackingRecHitHost>(iConfig.getParameter<edm::InputTag>("otRecHitSrc"))) {
    produces<nanoaod::FlatTable>();
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("name", "TrkPromptCA")->setComment("flat table name (the trainer reads <name>_<feature>)");
    desc.add<edm::InputTag>("trackSrc", edm::InputTag("initialStepCALowPtQuads"))->setComment("CA track SoA (host copy)");
    desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("siPixelRecHitsPreSplittingAlpaka"))
        ->setComment("pixel rechit SoA (host copy): the first block of the CA hit index space");
    desc.add<edm::InputTag>("otRecHitSrc", edm::InputTag("phase2OTRecHitsSoAConverter"))
        ->setComment("outer-tracker rechit SoA (host copy): the second block of the CA hit index space");
    descriptions.addWithDefaultLabel(desc);
  }

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const override {
    const auto& tracksHost = iEvent.get(tracksToken_);
    const auto& pixelHits = iEvent.get(pixelHitsToken_);
    const auto& otHits = iEvent.get(otHitsToken_);

    // the hit index space of Kernel_classifyTracks on Phase2OT: pixel rechits, then the OT rechits
    std::vector<reco::TrackingRecHitHost const*> colls{&pixelHits, &otHits};
    const Phase2HitsMultiView hh(
        colls, [](reco::TrackingRecHitHost const* c) -> auto { return c->const_view().trackingHits(); });
    const int nHitsTot = hh.size();

    const auto tracks = tracksHost.const_view().tracks();
    const auto trackHits = tracksHost.const_view().trackHits();
    const int nTracks = tracks.nTracks();

    static const char* kNames[caTrackFeatures::kNFeat] = {"fitChi2",
                                                          "psFrac",
                                                          "r0",
                                                          "nPS",
                                                          "nh",
                                                          "spanZ",
                                                          "nVectorHits",
                                                          "nl",
                                                          "logChi2VectorHit",
                                                          "kErr",
                                                          "dcaEst",
                                                          "nBarrel"};
    constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
    std::vector<std::vector<float>> cols(caTrackFeatures::kNFeat, std::vector<float>(nTracks, kNaN));
    std::vector<float> phi(nTracks), pt(nTracks), eta(nTracks);
    std::vector<int> quality(nTracks);
    std::vector<uint8_t> ok(nTracks, 0);
    std::vector<uint32_t> ids;

    for (int i = 0; i < nTracks; ++i) {
      const uint32_t begin = (i == 0) ? 0u : tracks[i - 1].hitOffsets();
      const uint32_t end = tracks[i].hitOffsets();
      ids.clear();
      for (uint32_t j = begin; j < end; ++j)
        ids.push_back(trackHits[j].id());
      caTrackFeatures::Features feat;
      const bool good = caTrackFeatures::fill(
          ids.begin(), ids.end(), hh, nHitsTot, float(tracks[i].nLayers()), tracks[i].chi2(), feat, nullptr);
      if (good) {
        const auto a = feat.asArray();
        for (int f = 0; f < caTrackFeatures::kNFeat; ++f)
          cols[f][i] = a[f];
      }
      ok[i] = good ? 1 : 0;
      phi[i] = tracks[i].state()(0);
      pt[i] = tracks[i].pt();
      eta[i] = tracks[i].eta();
      quality[i] = int(tracks[i].quality());
    }

    auto table = std::make_unique<nanoaod::FlatTable>(nTracks, name_, false, false);
    table->setDoc("CA track classifier features (caTrackFeatures::fill, pixel + OT rechit index space)");
    for (int f = 0; f < caTrackFeatures::kNFeat; ++f)
      table->addColumn<float>(kNames[f], cols[f], "CA track classifier feature");
    table->addColumn<float>("phi", phi, "track phi (state(0)), join key vs the truth table");
    table->addColumn<float>("pt", pt, "track pt");
    table->addColumn<float>("eta", eta, "track eta");
    table->addColumn<int>("quality", quality, "pixelTrack::Quality of the CA output");
    table->addColumn<uint8_t>("fillOk", ok, "1 if caTrackFeatures::fill succeeded");
    iEvent.put(std::move(table));
  }

  const std::string name_;
  const edm::EDGetTokenT<reco::TracksHost> tracksToken_;
  const edm::EDGetTokenT<reco::TrackingRecHitHost> pixelHitsToken_;
  const edm::EDGetTokenT<reco::TrackingRecHitHost> otHitsToken_;
};

DEFINE_FWK_MODULE(Phase2CATrackFeaturesTableProducer);
