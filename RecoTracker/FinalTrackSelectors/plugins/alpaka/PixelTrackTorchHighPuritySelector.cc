/**
 * PixelTrackTorchHighPuritySelector
 * =================================
 *
 * GPU/Accelerator module performing HighPurity pixel-track selection composed of:
 *
 *   1. CA-based quality preselection
 *   2. Feature extraction
 *   3. TorchScript DNN inference
 *   4. Score-based filtering
 *   5. Track/hit compaction and output production
 *
 * ------------------------------------------------------------------
 * Pipeline Overview
 * ------------------------------------------------------------------
 *
 *   Input:
 *       TracksSoA (pixel tracks + hit associations)
 *       TrackingRecHitsSoA
 *
 *   Transformations:
 *
 *       TracksSoA
 *          │
 *          v
 *       CA preselection
 *          │  Produces compacted preselected track index list 
 *          v
 *       Feature extraction
 *          │  Produces fixed-size features tensors
 *          v
 *       Torch inference
 *          │  Produces per-track classification score
 *          v
 *       Score filtering
 *          │  Filters tracks based on their classification scores
 *          v
 *       Output TrackSoA compaction
 *
 * ------------------------------------------------------------------
 * Torch Inference
 * ------------------------------------------------------------------
 *
 * The Torch model expects fixed-size tensors:
 *
 *     Track tensor:  [maxPreselectedTracks, N_track_features]
 *     Hit tensor:    [maxPreselectedTracks, MaxHitsPerTrack, N_hit_features]
 *
 * Padding slots are filled with NaNs.
 * ------------------------------------------------------------------
*/

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/FixedQueueEDProducer.h"

#include <deque>
#include <nvtx3/nvtx3.hpp>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"

#include "RecoTracker/FinalTrackSelectors/interface/PixelTrackFeaturesSoA.h"
#include "RecoTracker/FinalTrackSelectors/plugins/alpaka/PixelTrackFeaturesDeviceCollection.h"
#include "RecoTracker/FinalTrackSelectors/plugins/alpaka/PixelTrackTorchHighPuritySelectorKernels.h"

#include "PhysicsTools/PyTorchAlpaka/interface/TensorCollection.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"

// #define PIXEL_TRACK_HP_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  struct BatchIO {
    cms::torch::alpakatools::TensorCollection<Queue> inputs;
    cms::torch::alpakatools::TensorCollection<Queue> outputs;
  };

  class PixelTrackTorchHighPuritySelector : public stream::FixedQueueEDProducer<> {
    using TkSoADevice = reco::TracksSoACollection;
    using HitsOnDevice = reco::TrackingRecHitsSoACollection;
    using TrackHitSoA = ::reco::TrackHitSoA;

  public:
    explicit PixelTrackTorchHighPuritySelector(const edm::ParameterSet&);
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    void produce(device::Event&, const device::EventSetup&) override;

    const device::EDGetToken<TkSoADevice> pixelTrackToken_;
    const int maxNumberOfTracks_;
    const int maxPreselectedTracks_;
    const int minNumberOfHits_;
    const int avgHitsPerTrack_;
    const pixelTrack::Quality minimumTrackQuality_;
    const double scoreThreshold_;
    torch::AlpakaModel model_;
    const device::EDPutToken<TkSoADevice> tokenTrackOut_;
    const int batchSize_;
    bool isFirstEvent_;
  };

  PixelTrackTorchHighPuritySelector::PixelTrackTorchHighPuritySelector(const edm::ParameterSet& iConfig)
      : FixedQueueEDProducer(iConfig),
        pixelTrackToken_(consumes(iConfig.getParameter<edm::InputTag>("pixelTrackSrc"))),
        maxNumberOfTracks_(iConfig.getParameter<int>("maxNumberOfTracks")),
        maxPreselectedTracks_(iConfig.getParameter<int>("maxPreselectedTracks")),
        minNumberOfHits_(iConfig.getParameter<int>("minNumberOfHits")),
        avgHitsPerTrack_(iConfig.getParameter<int>("avgHitsPerTrack")),
        minimumTrackQuality_(pixelTrack::qualityByName(iConfig.getParameter<std::string>("minimumTrackQuality"))),
        scoreThreshold_(iConfig.getParameter<double>("scoreThreshold")),
        model_(iConfig.getParameter<edm::FileInPath>("model").fullPath()),
        tokenTrackOut_(produces()),
        batchSize_(iConfig.getParameter<int>("batchSize")) {
    if (minimumTrackQuality_ == pixelTrack::Quality::notQuality) {
      throw cms::Exception("PixelTrackConfiguration")
          << iConfig.getParameter<std::string>("minimumTrackQuality") + " is not a pixelTrack::Quality";
    }
    if (minimumTrackQuality_ < pixelTrack::Quality::dup) {
      throw cms::Exception("PixelTrackConfiguration")
          << iConfig.getParameter<std::string>("minimumTrackQuality") + " not supported";
    }
    if (maxPreselectedTracks_ > maxNumberOfTracks_) {
      throw cms::Exception("PixelTrackConfiguration") << "maxPreselectedTracks must be <= maxNumberOfTracks";
    }
    isFirstEvent_ = true;
    //model_.to(::torch::kHalf);
  }

  void PixelTrackTorchHighPuritySelector::produce(device::Event& iEvent, const device::EventSetup&) {
    /* 
    Processing steps:
      1. CA-based preselection of tracks
      2. Feature extraction (track + hit SoA)
      3. DNN inference
      4. Score-based filtering
      5. Track compaction and output production
*/
    nvtx3::scoped_range range{"PixelTrackTorchHP Producer"};
    // Retrieve tokens
    auto& queue = iEvent.queue();
    const auto& tracks = iEvent.get(pixelTrackToken_).view();

    // Instantiate the necessary objects in memory
    //  - Temporary storage for filtering
    auto d_nPreselectedTracks = cms::alpakatools::make_device_buffer<int>(queue);
    auto d_nSelectedTracks = cms::alpakatools::make_device_buffer<int>(queue);
    auto d_preselectedTrackIndices = cms::alpakatools::make_device_buffer<int[]>(queue, maxNumberOfTracks_);
    auto d_selectedTrackIndices = cms::alpakatools::make_device_buffer<int[]>(queue, maxPreselectedTracks_);
    auto d_nKeptHits = cms::alpakatools::make_device_buffer<int[]>(queue, maxPreselectedTracks_);
    auto d_preselectionOffsets = cms::alpakatools::make_device_buffer<int[]>(queue, maxNumberOfTracks_);

    alpaka::memset(queue, d_nPreselectedTracks, 0);
    alpaka::memset(queue, d_nSelectedTracks, 0);
    alpaka::memset(queue, d_nKeptHits, 0);
    alpaka::memset(queue, d_preselectedTrackIndices, 0xFF);
    alpaka::memset(queue, d_selectedTrackIndices, 0xFF);
    alpaka::memset(queue, d_preselectionOffsets, 0);

    //  - Features and scores containers
    PixelTrackFeaturesOnDevice trackFeatures(queue, maxPreselectedTracks_);
    PixelTrackScoresOnDevice trackScoresOnDevice(queue, maxPreselectedTracks_);

    // Optional debug definitions
#ifdef PIXEL_TRACK_HP_DEBUG
    auto h_nPreselectedTracks = cms::alpakatools::make_host_buffer<int>(queue);
    auto h_nSelectedTracks = cms::alpakatools::make_host_buffer<int>(queue);
    auto nPreselectedTracks = 0;
    auto nSelectedTracks = 0;
    // Helper to copy the number of kept tracks back to host (debug only)
    auto fetchnPreselectedTracks = [&]() {
      alpaka::memcpy(queue, h_nPreselectedTracks, d_nPreselectedTracks);
      alpaka::wait(queue);
      return *h_nPreselectedTracks;
    };
    auto fetchnSelectedTracks = [&]() {
      alpaka::memcpy(queue, h_nSelectedTracks, d_nSelectedTracks);
      alpaka::wait(queue);
      return *h_nSelectedTracks;
    };
#endif

    // 1. CA-based preselection of tracks
    //  Launch first kernel to look which tracks need to be filtered out
    //  based on quality criteria from the CA
    {
      nvtx3::scoped_range range1{"CAPreselection"};
      launchCAPreselection(queue,
                           maxNumberOfTracks_,
                           minNumberOfHits_,
                           minimumTrackQuality_,
                           tracks.tracks(),
                           alpaka::getPtrNative(d_preselectedTrackIndices),
                           alpaka::getPtrNative(d_preselectionOffsets),
                           alpaka::getPtrNative(d_nPreselectedTracks));
    }

#ifdef PIXEL_TRACK_HP_DEBUG
    nPreselectedTracks = fetchnPreselectedTracks();
    std::cout << "PixelTrackTorchHighPuritySelector::Prefiltered tracks=" << nPreselectedTracks << "\n";
#endif

    // 2. Feature extraction (track + hit SoA)
    {
      nvtx3::scoped_range range2{"FeaturesExtractor"};
      launchFeaturesExtractor(queue,
                              maxPreselectedTracks_,
                              tracks.tracks(),
                              alpaka::getPtrNative(d_preselectedTrackIndices),
                              alpaka::getPtrNative(d_nPreselectedTracks),
                              trackFeatures.view(),
                              alpaka::getPtrNative(d_nKeptHits));
    }

    // 3. DNN inference
    //  Prepare TensorCollection inputs and outputs for the model
    auto track_record = trackFeatures.view().records();
    auto score_record = trackScoresOnDevice.view().records();
    const auto n_batches = maxPreselectedTracks_ / batchSize_;
    std::deque<BatchIO> batches;

    // - Tensor collections for DNN inference

    for (auto i_batch = 0; i_batch < n_batches; ++i_batch) {
      nvtx3::scoped_range range3{"DNNInference"};
      batches.emplace_back(
          BatchIO{cms::torch::alpakatools::TensorCollection<Queue>(batchSize_, maxPreselectedTracks_),
                  cms::torch::alpakatools::TensorCollection<Queue>(batchSize_, maxPreselectedTracks_)});

      auto& batch = batches.back();
      // Order must match the TorchScript model input schema
      batch.inputs.add<PixelTrackFeaturesSoA>("track_features",
                                              i_batch,
                                              track_record.chi2(),
                                              track_record.dzError(),
                                              track_record.dxyError(),
                                              track_record.eta(),
                                              track_record.nHits(),
                                              track_record.phi(),
                                              track_record.phiError(),
                                              track_record.pt(),
                                              track_record.qOverPtError(),
                                              track_record.dzBS(),
                                              track_record.dxyBS(),
                                              track_record.nLayers(),
                                              track_record.cotThetaError(),
                                              track_record.covCotThetaDz(),
                                              track_record.covDxyQOverPt(),
                                              track_record.covPhiDxy(),
                                              track_record.covPhiQOverPt());

      batch.outputs.add<PixelTrackScoresSoA>("track_scores", i_batch, score_record.score());

      model_.forward(queue, batch.inputs, batch.outputs);
    }

    // 4. Score-based filtering
    {
      nvtx3::scoped_range range4{"ScoreFilter"};
      launchScoreFilter(queue,
                        maxPreselectedTracks_,
                        scoreThreshold_,
                        trackScoresOnDevice.view(),
                        alpaka::getPtrNative(d_preselectedTrackIndices),
                        alpaka::getPtrNative(d_nPreselectedTracks),
                        alpaka::getPtrNative(d_selectedTrackIndices),
                        alpaka::getPtrNative(d_nSelectedTracks),
                        alpaka::getPtrNative(d_nKeptHits));
    }

#ifdef PIXEL_TRACK_HP_DEBUG
    nSelectedTracks = fetchnSelectedTracks();
    std::cout << "PixelTrackTorchHighPuritySelector::Filtered tracks=" << nSelectedTracks << "\n";
#endif
    {
      nvtx3::scoped_range range5{"OutputCompaction"};
      auto tracks_out = launchProduceOutputTracks(queue,
                                                  maxPreselectedTracks_,
                                                  avgHitsPerTrack_,
                                                  tracks.tracks(),
                                                  tracks.trackHits(),
                                                  alpaka::getPtrNative(d_selectedTrackIndices),
                                                  alpaka::getPtrNative(d_nSelectedTracks),
                                                  alpaka::getPtrNative(d_nKeptHits));
      iEvent.emplace(tokenTrackOut_, std::move(tracks_out));
    }
  }

  void PixelTrackTorchHighPuritySelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("pixelTrackSrc", {"hltPhase2PixelTracksSoA"});
    desc.add<int>("maxNumberOfTracks", 100000);
    desc.add<int>("maxPreselectedTracks", 10000);
    desc.add<int>("minNumberOfHits", 0);
    desc.add<int>("avgHitsPerTrack", 8);
    desc.add<std::string>("minimumTrackQuality", "tight");
    desc.add<edm::FileInPath>("model");
    desc.add<double>("scoreThreshold", 0.5);
    desc.add<int>("batchSize", 10);
    descriptions.addWithDefaultLabel(desc);
  }
};  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PixelTrackTorchHighPuritySelector);
