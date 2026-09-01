#include <alpaka/alpaka.hpp>

#include <numeric>

#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
// #include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

// #include "CAHitNtupletGenerator.h"
#include "CAHitMaskingAndMerger.h"

// #define GPU_DEBUG
// #define NTRACKS_DEBUG
// #define DUPLICATE_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PixelTracksSoAMerger : public stream::SynchronizingEDProducer<> {
    using Algo = CAHitMaskingAndMerger;

  public:
    explicit PixelTracksSoAMerger(const edm::ParameterSet& iConfig);
    // ~PixelTracksSoAMerger() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    // void produce(edm::StreamID streamID, device::Event& iEvent, const device::EventSetup& iSetup) const override;
    void acquire(device::Event const& iEvent, device::EventSetup const& iSetup) override;
    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override;
    // void beginRun(edm::Run const&, edm::EventSetup const& iSetup) override;

    pixelTrack::Quality const minQuality_;
    double const matchFraction_;

    std::vector<device::EDGetToken<reco::TracksSoACollection>> inputTkSoATokenV_;
    std::vector<edm::InputTag> inputTkSoATagV_;

    ::reco::InputTracks allTrackView_;
    int nCollections_ = 0;

    const device::EDPutToken<reco::TracksSoACollection> outputTkSoAToken_;
    
    Algo deviceAlgo_;
    uint32_t nTracks_ = 0;
    uint32_t nHits_ = 0;

    std::optional<reco::TracksSoACollection> tracks_d_;

  };

  PixelTracksSoAMerger::PixelTracksSoAMerger(const edm::ParameterSet& iConfig)
      : SynchronizingEDProducer(iConfig),
        minQuality_(pixelTrack::qualityByName(iConfig.getParameter<std::string>("minQuality"))),
        matchFraction_(iConfig.getParameter<double>("matchFraction")),
        inputTkSoATagV_(iConfig.getParameter<std::vector<edm::InputTag>>("inputTkSoAs")),
        outputTkSoAToken_(produces()) {
    for (const auto& it : inputTkSoATagV_) {
      inputTkSoATokenV_.push_back(consumes(it));
    }

    assert(inputTkSoATagV_.size() <= ::reco::maxTrackSoACollections);
    nCollections_ = inputTkSoATagV_.size();
    allTrackView_.nInputs = inputTkSoATagV_.size();

    if (minQuality_ == pixelTrack::Quality::notQuality) {
      throw cms::Exception("PixelTrackConfiguration")
          << iConfig.getParameter<std::string>("minQuality") + " is not a pixelTrack::Quality";
    }
    if (minQuality_ < pixelTrack::Quality::dup) {
      throw cms::Exception("PixelTrackConfiguration")
          << iConfig.getParameter<std::string>("minQuality") + " not supported";
    }
  }

  void PixelTracksSoAMerger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<std::vector<edm::InputTag>>(
        "inputTkSoAs", {edm::InputTag("pixelTracksHighPtAlpaka"), edm::InputTag("pixelTracksLowPtAlpaka")});
    desc.add<std::string>("minQuality", "highPurity");
    desc.add<double>("matchFraction", 0.0);

    descriptions.addWithDefaultLabel(desc);
  }

  namespace {
    // This utility unrolls the SoA columns (tuples) at compile time, calling the provided functor 'f'
    // once for each element. The index is passed as a std::integral_constant so it
    // is available at compile time.
    template <typename F, std::size_t... Is>
    void unrollColumns(F&& f, std::index_sequence<Is...>) {
      (f(std::integral_constant<std::size_t, Is>{}), ...);
    }
    // User-facing wrapper to deduce the size of the tuple and create the index sequence
    // Usage: mergeSoAColumns<NumberOfColumns>([&](auto columnIndex) { ... });
    template <std::size_t N, typename F>
    void mergeSoAColumns(F&& f) {
      unrollColumns(std::forward<F>(f), std::make_index_sequence<N>{});
    }
  }  // namespace

  void PixelTracksSoAMerger::acquire(device::Event const& iEvent, device::EventSetup const& iSetup) {

    // get both Pixel and Tracker SoA collections
    auto queue = iEvent.queue();

    std::vector<const reco::TracksSoACollection*> inputTkSoAs;
    inputTkSoAs.resize(inputTkSoATokenV_.size());
    // for (const auto& it : inputTkSoATokenV_) {
    //   auto const& aux = iEvent.get(it);
    //   inputTkSoAs.push_back(&aux);
    // }

    int maxTracks = 0;
    for (int i = 0; i < nCollections_; ++i) {

      auto const& aux = iEvent.get(inputTkSoATokenV_[i]);
      inputTkSoAs[i] = &aux;
      allTrackView_.views[i] = aux.view().tracks();
      maxTracks +=  aux.view().tracks().metadata().size(); 
      allTrackView_.hitViews[i] = aux.view().trackHits();
    }

    deviceAlgo_.countGoodTracks(queue, allTrackView_, maxTracks, minQuality_);
    //nTracks_ = nGoodTracks;
    //nHits_ = nGoodHits;

  }

  void PixelTracksSoAMerger::produce(/*edm::StreamID streamID,*/
                                     device::Event& iEvent,
                                     const device::EventSetup& es) {

  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PixelTracksSoAMerger);
