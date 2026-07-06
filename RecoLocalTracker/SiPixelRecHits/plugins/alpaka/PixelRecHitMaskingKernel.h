// #ifndef RecoLocalTracker_SiPixelRecHits_PixelRecHitKernel_h
// #define RecoLocalTracker_SiPixelRecHits_PixelRecHitKernel_h

// #include <cstdint>

// #include <alpaka/alpaka.hpp>

// #include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
// #include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// namespace ALPAKA_ACCELERATOR_NAMESPACE {
//   namespace pixelgpudetails {
//     using namespace cms::alpakatools;
//     using namespace ALPAKA_ACCELERATOR_NAMESPACE::reco;
//     class PixelRecHitMaskingKernel {
//     public:
//       PixelRecHitMaskingKernel() = default;
//       ~PixelRecHitMaskingKernel() = default;

//       PixelRecHitMaskingKernel(const PixelRecHitMaskingKernel&) = delete;
//       PixelRecHitMaskingKernel(PixelRecHitMaskingKernel&&) = delete;
//       PixelRecHitMaskingKernel& operator=(const PixelRecHitMaskingKernel&) = delete;
//       PixelRecHitMaskingKernel& operator=(PixelRecHitMaskingKernel&&) = delete;

//       reco::TrackingRecHitsMaskingSoACollection makeHitsMaskingAsync(uint32_t const nHits, Queue queue) const;
//     };
//   }  // namespace pixelgpudetails
// }  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// #endif  // RecoLocalTracker_SiPixelRecHits_PixelRecHitKernel_h
