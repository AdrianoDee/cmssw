#ifndef RecoLocalTracker_SiPixelRecHits_PixelRecHitMaskingKernel_h
#define RecoLocalTracker_SiPixelRecHits_PixelRecHitMaskingKernel_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace pixelgpudetails {
    using namespace cms::alpakatools;
    using namespace ALPAKA_ACCELERATOR_NAMESPACE::reco;

    TrackingRecHitsMaskingSoACollection makeHitsMaskingAsync(uint32_t const nHits, Queue& queue);
    
  }  // namespace pixelgpudetails
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalTracker_SiPixelRecHits_PixelRecHitMaskingKernel_h
