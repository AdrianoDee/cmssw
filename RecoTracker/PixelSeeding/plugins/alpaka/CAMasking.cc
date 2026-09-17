#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsMaskingSoACollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"

#include "CAMaskingMerging.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class CAMasking : public global::EDProducer<> {

    using HitsOnDevice = reco::TrackingRecHitsSoACollection;
    using MapToHit = reco::TrackingRecHitsMaskingSoACollection;

    using HitsConstView = ::reco::TrackingRecHitConstView;
    using MapToHitConstView = MapToHit::ConstView;


  public:
    explicit CAMasking(const edm::ParameterSet& iConfig);
    ~CAMasking() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void produce(edm::StreamID streamID, device::Event& iEvent, const device::EventSetup& iSetup) const override;

      pixelTrack::Quality const minQuality_;
    const bool useOldMask_;
    const bool useHits_;

    // Need one of the two input tokens: either the old mask or the hits on device
    device::EDGetToken<MapToHit> inputRecHitsMaskToken_;
    device::EDGetToken<HitsOnDevice> inputHitsOnDeviceToken_;

    const device::EDGetToken<reco::TracksSoACollection> inputTrackSoAToken_;

    const device::EDPutToken<reco::TrackingRecHitsMaskingSoACollection> outputRecHitsMaskToken_;

  };

  CAMasking::CAMasking(const edm::ParameterSet& iConfig)
      : EDProducer(iConfig),
        minQuality_(pixelTrack::qualityByName(iConfig.getParameter<std::string>("minQuality"))),
        useOldMask_(not iConfig.getParameter<edm::InputTag>("oldMask").label().empty()),
        useHits_(not iConfig.getParameter<edm::InputTag>("hitSoA").label().empty()),
        inputTrackSoAToken_(consumes(iConfig.getParameter<edm::InputTag>("trackSoA"))),
        outputRecHitsMaskToken_(produces()) {
    if (minQuality_ == pixelTrack::Quality::notQuality) {
      throw cms::Exception("PixelTrackConfiguration")
          << iConfig.getParameter<std::string>("minQuality") + " is not a pixelTrack::Quality";
    }
    if (minQuality_ < pixelTrack::Quality::dup) {
      throw cms::Exception("PixelTrackConfiguration")
          << iConfig.getParameter<std::string>("minQuality") + " not supported";
    }

    if (useOldMask_) {
      inputRecHitsMaskToken_ = device::EDGetToken<MapToHit>(consumes(iConfig.getParameter<edm::InputTag>("oldMask")));
    }
    if (useHits_) {
      inputHitsOnDeviceToken_ = device::EDGetToken<HitsOnDevice>(consumes(iConfig.getParameter<edm::InputTag>("hitSoA")));
    }
    if (not useOldMask_ and not useHits_) {
      throw cms::Exception("PixelTrackConfiguration")
          << "Either recHitsMaskSoASrc or hitsOnDeviceSrc must be provided";
    }
    if (useOldMask_ and useHits_) {
      throw cms::Exception("PixelTrackConfiguration")
          << "Only one of recHitsMaskSoASrc or hitsOnDeviceSrc should be provided";
    }
  }

  void CAMasking::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>(
        "oldMask",
        edm::InputTag("")); 
    desc.add<edm::InputTag>(
        "hitSoA",
        edm::InputTag("")); 
    desc.add<edm::InputTag>("trackSoA",
                            edm::InputTag("pixelTracksHighPtAlpaka"));  // has to be changed for each iteration
    desc.add<std::string>("minQuality", "highPurity");

    descriptions.addWithDefaultLabel(desc);
  }

  void CAMasking::produce(edm::StreamID streamID,
                                      device::Event& iEvent,
                                      const device::EventSetup& es) const {
    // get both Pixel and Tracker SoA collections
    auto queue = iEvent.queue();
    const auto& inpTkColl = iEvent.get(inputTrackSoAToken_);

    MapToHitConstView maskView;
    HitsConstView hitsView;
    
    int maskSize = 0;
    if (useOldMask_) {
      maskView = iEvent.get(inputRecHitsMaskToken_).view();
      maskSize = maskView.metadata().size();
    }
    if (useHits_){
      hitsView = iEvent.get(inputHitsOnDeviceToken_).view().trackingHits();
      maskSize = hitsView.metadata().size();
    }


    reco::TrackingRecHitsMaskingSoACollection outMask(queue, maskSize);
    if (useOldMask_) {
      auto outMaskColumn =
          cms::alpakatools::make_device_view(queue, outMask.view().recHitMask(), maskSize);
      auto inMaskColumn =
          cms::alpakatools::make_device_view(queue, maskView.recHitMask(), maskSize);
      alpaka::memcpy(queue, outMaskColumn, inMaskColumn);
    } else {
      auto outMaskColumn = cms::alpakatools::make_device_view(queue, outMask.view().recHitMask(), maskSize);
      alpaka::memset(queue, outMaskColumn, 0);
    }
                                    
    caMasking::makeMaskingAsync(queue, outMask, inpTkColl, minQuality_);

    iEvent.emplace(outputRecHitsMaskToken_, std::move(outMask));
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(CAMasking);
