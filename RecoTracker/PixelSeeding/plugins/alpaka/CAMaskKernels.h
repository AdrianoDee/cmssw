#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAMaskKernels_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAMaskKernels_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsMaskingSoACollection.h"
#include "RecoTracker/PixelSeeding/interface/CAMaskingUtilities.h"

#include "CAStructures.h"

// #define GPU_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE::caMasking {

  using MapToHit = reco::TrackingRecHitsMaskingSoACollection;
  using MapToHitView = MapToHit::View;
  using MapToHitConstView = MapToHit::ConstView;
  using TkSoADevice = reco::TracksSoACollection;

  using HitsMultiView = caStructures::HitsMultiView;

  // Parameters of the hit compatibility check, see CAMaskSoAProducer::fillDescriptions
  struct CompatibleHitsParams {
    ::caHitCompatibility::HitResidualCuts cuts;
    float bField;                  // 1 / inverseBzAtOriginInGeV
    uint32_t minHits;              // tracks with fewer hits have all their hits masked
    uint32_t nPixelHits;           // hits with a global index >= nPixelHits are outer-tracker hits
    uint32_t nPixelBarrelModules;  // pixel hits with detectorIndex < nPixelBarrelModules are barrel hits
  };

  // Mask all the hits of the tracks with quality >= minQuality
  void makeMaskingAsync(Queue& queue, MapToHit& outMask, TkSoADevice const& tracks_d, pixelTrack::Quality minQuality);

  // Mask only the hits compatible with the fitted helix of the tracks with quality >= minQuality
  void makeCompatibleHitsMaskingAsync(Queue& queue,
                                      MapToHit& outMask,
                                      TkSoADevice const& tracks_d,
                                      pixelTrack::Quality minQuality,
                                      HitsMultiView const& hits,
                                      CompatibleHitsParams const& params);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::caMasking

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAMaskKernels_h
