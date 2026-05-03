#ifndef RecoTracker_FinalTrackSelectors_alpaka_PixelTrackFeaturesDeviceCollection_h
#define RecoTracker_FinalTrackSelectors_alpaka_PixelTrackFeaturesDeviceCollection_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "RecoTracker/FinalTrackSelectors/interface/PixelTrackFeaturesSoA.h"
#include "RecoTracker/FinalTrackSelectors/interface/PixelRecHitFeaturesSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using PixelTrackScoresOnDevice = PortableCollection<PixelTrackScoresSoA>;
  using PixelRecHitFeaturesOnDevice = PortableCollection<RecHitFeatures::PixelRecHitFeaturesSoA>;
  using PixelTrackFeaturesOnDevice = PortableCollection<PixelTrackFeaturesSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
