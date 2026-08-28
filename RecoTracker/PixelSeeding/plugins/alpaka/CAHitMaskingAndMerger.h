#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAHitMaskingAndMerger_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAHitMaskingAndMerger_h

#include <alpaka/alpaka.hpp>


// #include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
// #include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/TracksDevice.h"
// #include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsMaskingSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsMaskingSoACollection.h"
#include "FWCore/Framework/interface/EventSetup.h"
// #include "FWCore/ParameterSet/interface/ParameterSet.h"
// #include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
// #include "RecoTracker/PixelSeeding/interface/alpaka/CAGeometrySoACollection.h"

// #include "CACell.h"
#include "CAHitMaskingAndMergerKernels.h"
// #include "HelixFit.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class CAHitMaskingAndMerger {
  public:
    using MapToHit = reco::TrackingRecHitsMaskingSoACollection;
    using TkSoADevice = reco::TracksSoACollection;

  public:
    CAHitMaskingAndMerger() = default;
    ~CAHitMaskingAndMerger() = default;

    CAHitMaskingAndMerger(const CAHitMaskingAndMerger&) = delete;
    CAHitMaskingAndMerger(CAHitMaskingAndMerger&&) = delete;
    CAHitMaskingAndMerger& operator=(const CAHitMaskingAndMerger&) = delete;
    CAHitMaskingAndMerger& operator=(CAHitMaskingAndMerger&&) = delete;

    MapToHit makeMaskingAsync(Queue& queue,
                              MapToHit const& mask_d,
                              TkSoADevice const& tracks_d,
                              pixelTrack::Quality minQuality,
                              uint32_t iterationIndex) const;

    void updateHitOffsets(Queue& queue, int tksBeg, int tksEnd, int nHits, TkSoADevice& tracks_d) const;

    TkSoADevice makeFilteredTracks(Queue& queue,
                                   int nTracks,
                                   int nHits,
                                   TkSoADevice const& inpTracks,
                                   pixelTrack::Quality minQuality,
                                   double matchFraction) const;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAHitMaskingAndMerger_h
