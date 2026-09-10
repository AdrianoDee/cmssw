#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "CAHitNtupletGenerator.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PixelTracksMaskingSoA : public global::EDProducer<> {
    using Algo = CAHitMaskingAndMerger;

  public:
    explicit PixelTracksMaskingSoA(const edm::ParameterSet& iConfig);
    ~PixelTracksMaskingSoA() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void produce(edm::StreamID streamID, device::Event& iEvent, const device::EventSetup& iSetup) const override;

    uint32_t const iterationIndex_;
    pixelTrack::Quality const minQuality_;
    bool const applyMasking_;
    // When true, hits attached by the extension stage are masked for the next iteration as well;
    // otherwise only the hits the CA itself found are masked.
    bool const maskAttachedHits_;

    const device::EDGetToken<reco::TrackingRecHitsMaskingCollection> inputRecHitsMaskToken_;
    const device::EDGetToken<reco::TracksSoACollection> inputTrackSoAToken_;

    const device::EDPutToken<reco::TrackingRecHitsMaskingCollection> outputRecHitsMaskToken_;

    Algo deviceAlgo_;
  };

  PixelTracksMaskingSoA::PixelTracksMaskingSoA(const edm::ParameterSet& iConfig)
      : EDProducer(iConfig),
        iterationIndex_(iConfig.getParameter<uint32_t>("iterationIndex")),
        minQuality_(pixelTrack::qualityByName(iConfig.getParameter<std::string>("minQuality"))),
        applyMasking_(iConfig.getParameter<bool>("applyMasking")),
        maskAttachedHits_(iConfig.getParameter<bool>("maskAttachedHits")),
        inputRecHitsMaskToken_(consumes(iConfig.getParameter<edm::InputTag>("recHitsMaskSoASrc"))),
        inputTrackSoAToken_(consumes(iConfig.getParameter<edm::InputTag>("tracksSoASrc"))),
        outputRecHitsMaskToken_(produces()) {
    if (minQuality_ == pixelTrack::Quality::notQuality) {
      throw cms::Exception("PixelTrackConfiguration")
          << iConfig.getParameter<std::string>("minQuality") + " is not a pixelTrack::Quality";
    }
    if (minQuality_ < pixelTrack::Quality::dup) {
      throw cms::Exception("PixelTrackConfiguration")
          << iConfig.getParameter<std::string>("minQuality") + " not supported";
    }
  }

  void PixelTracksMaskingSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>(
        "recHitsMaskSoASrc",
        edm::InputTag("siPixelRecHitsExtendedPreSplittingAlpaka"));  // has to be changed for each iteration
    desc.add<edm::InputTag>("tracksSoASrc",
                            edm::InputTag("pixelTracksHighPtAlpaka"));  // has to be changed for each iteration
    desc.add<std::string>("minQuality", "highPurity");
    desc.add<uint32_t>("iterationIndex", 0);
    desc.add<bool>("maskAttachedHits", false)
        ->setComment("also mask hits attached by the in-fit extension (default: original-hits policy)");
    desc.add<bool>("applyMasking", true)
        ->setComment(
            "If false, pass through the input mask unchanged (no hits masked) so the next iteration sees all hits; the "
            "module stays in the chain.");

    descriptions.addWithDefaultLabel(desc);
  }

  void PixelTracksMaskingSoA::produce(edm::StreamID streamID,
                                      device::Event& iEvent,
                                      const device::EventSetup& es) const {
    // Reading the mask layout first puts the module on the queue that produced it, which orders its
    // reads before the early release of that product. Keep this get() first.
    const auto& inpMaskColl = iEvent.get(inputRecHitsMaskToken_);
    const auto& inpTkColl = iEvent.get(inputTrackSoAToken_);

    iEvent.emplace(
        outputRecHitsMaskToken_,
        deviceAlgo_.makeMaskingAsync(
            inpMaskColl, inpTkColl, minQuality_, iterationIndex_, iEvent.queue(), applyMasking_, maskAttachedHits_));
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PixelTracksMaskingSoA);
