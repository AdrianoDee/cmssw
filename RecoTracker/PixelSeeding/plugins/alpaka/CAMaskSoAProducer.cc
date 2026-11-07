#include <string>
#include <vector>
#include <utility>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsMaskingSoACollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/PixelSeeding/interface/CAMaskingUtilities.h"

#include "CAMaskKernels.h"

// #define GPU_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class CAMaskSoAProducer : public global::EDProducer<> {
    using HitsOnDevice = reco::TrackingRecHitsSoACollection;
    using MapToHit = reco::TrackingRecHitsMaskingSoACollection;

    using HitsConstView = ::reco::TrackingRecHitConstView;
    using MapToHitConstView = MapToHit::ConstView;

  public:
    explicit CAMaskSoAProducer(const edm::ParameterSet& iConfig);
    ~CAMaskSoAProducer() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void produce(edm::StreamID streamID, device::Event& iEvent, const device::EventSetup& iSetup) const override;

    pixelTrack::Quality const minQuality_;
    const bool useOldMask_ = false;
    const bool useHits_ = false;

    // Mask only the hits compatible with the fitted helix of the track (needs the hits)
    const bool maskOnlyCompatibleHits_ = false;
    const uint32_t minHitsForResidualCheck_;
    ::caHitCompatibility::HitResidualCuts residualCuts_;
    // consumed only if maskOnlyCompatibleHits_
    edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tokenField_;
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tokenGeometry_;

    // Need at least one of the two inputs: the old mask and/or the hits on device
    device::EDGetToken<MapToHit> inputRecHitsMaskToken_;
    std::vector<device::EDGetToken<HitsOnDevice>> inputHitsOnDeviceToken_;

    const device::EDGetToken<reco::TracksSoACollection> inputTrackSoAToken_;

    const device::EDPutToken<reco::TrackingRecHitsMaskingSoACollection> outputRecHitsMaskToken_;
  };

  CAMaskSoAProducer::CAMaskSoAProducer(const edm::ParameterSet& iConfig)
      : EDProducer(iConfig),
        minQuality_(pixelTrack::qualityByName(iConfig.getParameter<std::string>("minQuality"))),
        useOldMask_(not iConfig.getParameter<edm::InputTag>("oldMask").label().empty()),
        useHits_(not iConfig.getParameter<std::vector<edm::InputTag>>("hitSoAs").empty()),
        maskOnlyCompatibleHits_(iConfig.getParameter<bool>("maskOnlyCompatibleHits")),
        minHitsForResidualCheck_(iConfig.getParameter<uint32_t>("minHitsForResidualCheck")),
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

    if (not useOldMask_ and not useHits_) {
      throw cms::Exception("PixelTrackConfiguration") << "At least one of oldMask or hitSoAs must be provided";
    }

    if (useOldMask_) {
      inputRecHitsMaskToken_ = device::EDGetToken<MapToHit>(consumes(iConfig.getParameter<edm::InputTag>("oldMask")));
    }
    if (useHits_) {
      for (const auto& it : iConfig.getParameter<std::vector<edm::InputTag>>("hitSoAs")) {
        inputHitsOnDeviceToken_.push_back(consumes(it));
      }
    }

    if (maskOnlyCompatibleHits_) {
      // the residuals need the hit positions
      if (not useHits_) {
        throw cms::Exception("PixelTrackConfiguration")
            << "maskOnlyCompatibleHits = True needs the hit collections used by the CA in hitSoAs";
      }
      if (inputHitsOnDeviceToken_.size() > 2) {
        throw cms::Exception("PixelTrackConfiguration")
            << "maskOnlyCompatibleHits = True supports at most two hit collections in hitSoAs (pixel, outer tracker), "
            << inputHitsOnDeviceToken_.size() << " given";
      }
      tokenField_ = esConsumes();
      tokenGeometry_ = esConsumes();
    }

    namespace chc = ::caHitCompatibility;
    residualCuts_.maxTransverse[chc::kPixelBarrel] = iConfig.getParameter<double>("maxResidualPixelBarrelRPhi");
    residualCuts_.maxLongitudinal[chc::kPixelBarrel] = iConfig.getParameter<double>("maxResidualPixelBarrelZ");
    residualCuts_.maxTransverse[chc::kPixelEndcap] = iConfig.getParameter<double>("maxResidualPixelEndcapRPhi");
    residualCuts_.maxLongitudinal[chc::kPixelEndcap] = iConfig.getParameter<double>("maxResidualPixelEndcapR");
    residualCuts_.maxTransverse[chc::kOuterTracker] = iConfig.getParameter<double>("maxResidualOTRPhi");
    residualCuts_.maxLongitudinal[chc::kOuterTracker] = iConfig.getParameter<double>("maxResidualOTZ");
  }

  void CAMaskSoAProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("oldMask", edm::InputTag(""))
        ->setComment("Mask of the previous iterations, copied and updated; if empty, start from an empty mask");
    desc.add<std::vector<edm::InputTag>>("hitSoAs", {})
        ->setComment(
            "Hit collections used by the CA (pixel hits first, then outer tracker hits): they size the mask when "
            "oldMask is empty (if both are given the sizes must match) and are needed by maskOnlyCompatibleHits");
    desc.add<edm::InputTag>("trackSoA",
                            edm::InputTag("pixelTracksHighPtAlpaka"));  // has to be changed for each iteration
    desc.add<std::string>("minQuality", "highPurity");

    // Mask only the hits compatible with the track (as the chi2 cut of the legacy cluster removal)
    desc.add<bool>("maskOnlyCompatibleHits", false)
        ->setComment(
            "If true, mask only the hits whose residuals w.r.t. the fitted helix of the track are below the cuts "
            "below, leaving the incompatible ones to the next iteration. Needs hitSoAs (pixel hits first, then outer "
            "tracker hits, as given to the CA). If false, all the hits of the selected tracks are masked.");
    desc.add<uint32_t>("minHitsForResidualCheck", 4)
        ->setComment(
            "Tracks with fewer hits have all their hits masked: the fit of a triplet passes exactly through its "
            "hits.");
    desc.add<double>("maxResidualPixelBarrelRPhi", 0.01)
        ->setComment("Max |transverse residual| [cm] (distance from the circle) for pixel barrel hits, <= 0: no cut");
    desc.add<double>("maxResidualPixelBarrelZ", 0.02)
        ->setComment("Max |z_hit - z_track(s_hit)| [cm] for pixel barrel hits, <= 0: no cut");
    desc.add<double>("maxResidualPixelEndcapRPhi", 0.01)
        ->setComment("Max |transverse residual| [cm] (distance from the circle) for pixel endcap hits, <= 0: no cut");
    desc.add<double>("maxResidualPixelEndcapR", 0.02)
        ->setComment("Max |r_hit - r_track(z_hit)| [cm] for pixel endcap hits, <= 0: no cut");
    desc.add<double>("maxResidualOTRPhi", 0.02)
        ->setComment(
            "Max |transverse residual| [cm] (distance from the circle) for outer tracker (barrel) hits, <= 0: no cut");
    desc.add<double>("maxResidualOTZ", 0.3)
        ->setComment("Max |z_hit - z_track(s_hit)| [cm] for outer tracker (barrel) hits, <= 0: no cut");

    descriptions.addWithDefaultLabel(desc);
  }

  void CAMaskSoAProducer::produce(edm::StreamID streamID, device::Event& iEvent, const device::EventSetup& es) const {
    // get both Pixel and Tracker SoA collections
    auto& queue = iEvent.queue();
    const auto& inpTkColl = iEvent.get(inputTrackSoAToken_);

    MapToHitConstView maskView;
    caMasking::HitsMultiView hitsMultiView;

    int oldMaskSize = 0;
    if (useOldMask_) {
      maskView = iEvent.get(inputRecHitsMaskToken_).view();
      oldMaskSize = maskView.metadata().size();
    }
    int nHits = 0;
    uint32_t nPixelHits = 0;
    for (std::size_t i = 0; i < inputHitsOnDeviceToken_.size(); ++i) {
      HitsConstView hitsView = iEvent.get(inputHitsOnDeviceToken_[i]).view().trackingHits();
      int const size = hitsView.metadata().size();
      if (i == 0)
        nPixelHits = size;  // the first collection holds the pixel hits
      nHits += size;
      if (maskOnlyCompatibleHits_)
        hitsMultiView.addView(hitsView, size);
    }
    if (useOldMask_ and useHits_ and nHits != oldMaskSize) {
      throw cms::Exception("CAHitMaskMismatch")
          << "CAMaskSoAProducer: the oldMask has " << oldMaskSize << " rows but the hitSoAs have " << nHits
          << " hits in total. The old mask must be the one built for these hit collections.";
    }
    int const maskSize = useOldMask_ ? oldMaskSize : nHits;

#ifdef GPU_DEBUG
    if (useOldMask_)
      printf("CAMaskSoAProducer::maskSize (from oldMask): %d\n", maskSize);
    else
      printf("CAMaskSoAProducer::maskSize (from hitSoAs): %d\n", maskSize);
#endif

    reco::TrackingRecHitsMaskingSoACollection outMask(queue, maskSize);
    if (useOldMask_) {
      auto outMaskColumn = cms::alpakatools::make_device_view(queue, outMask.view().recHitMask());
      auto inMaskColumn = cms::alpakatools::make_device_view(queue, maskView.recHitMask());
      alpaka::memcpy(queue, outMaskColumn, inMaskColumn);
    } else {
      auto outMaskColumn = cms::alpakatools::make_device_view(queue, outMask.view().recHitMask());
      alpaka::memset(queue, outMaskColumn, 0);
    }

    if (maskOnlyCompatibleHits_) {
      caMasking::CompatibleHitsParams params;
      params.cuts = residualCuts_;
      params.bField = float(1. / es.getData(tokenField_).inverseBzAtOriginInGeV());
      params.minHits = minHitsForResidualCheck_;
      params.nPixelHits = nPixelHits;
      // pixel modules are numbered in the order of TrackerGeometry::dets(), barrel first
      params.nPixelBarrelModules = es.getData(tokenGeometry_).detsPXB().size();
      caMasking::makeCompatibleHitsMaskingAsync(queue, outMask, inpTkColl, minQuality_, hitsMultiView, params);
    } else {
      caMasking::makeMaskingAsync(queue, outMask, inpTkColl, minQuality_);
    }

    iEvent.emplace(outputRecHitsMaskToken_, std::move(outMask));
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(CAMaskSoAProducer);
