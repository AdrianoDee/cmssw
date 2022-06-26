#ifndef RecoLocalTracker_SiPixelRecHits_plugins_SiPixelRecHitSoAProducerAlgo_h
#define RecoLocalTracker_SiPixelRecHits_plugins_SiPixelRecHitSoAProducerAlgo_h

#include <cstdint>

#include <cuda_runtime.h>

#include "CUDADataFormats/BeamSpot/interface/BeamSpotCUDA.h"
#include "CUDADataFormats/SiPixelCluster/interface/SiPixelClustersCUDA.h"
#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"

namespace pixelhitconverter {

  template <typename TTraits>
  class SiPixelRecHitSoAProducerAlgo {
  public:
    SiPixelRecHitSoAProducerAlgo() = default;
    ~SiPixelRecHitSoAProducerAlgo() = default;

    SiPixelRecHitSoAProducerAlgo(const SiPixelRecHitSoAProducerAlgo&) = delete;
    SiPixelRecHitSoAProducerAlgo(SiPixelRecHitSoAProducerAlgo&&) = delete;
    SiPixelRecHitSoAProducerAlgo& operator=(const SiPixelRecHitSoAProducerAlgo&) = delete;
    SiPixelRecHitSoAProducerAlgo& operator=(SiPixelRecHitSoAProducerAlgo&&) = delete;

    TrackingRecHit2DGPU convertHitsAsync(float* store32,
                                      uint16_t* store16,
                                      uint32_t* moduleStart,
                                      int nHits,
                                      int startBPIX2,
                                      bool isPhase2,
                                      cudaStream_t stream) const;

    TrackingRecHit2DCPU convertHits(float* store32,
                                      uint16_t* store16,
                                      uint32_t* moduleStart,
                                      int nHits,
                                      int startBPIX2,
                                      bool isPhase2) const;


  };

  using SiPixelRecHitSoAProducerOnGPU = SiPixelRecHitSoAProducerAlgo<cms::cudacompat::GPUTraits>;
  using SiPixelRecHitSoAProducerOnCPU = SiPixelRecHitSoAProducerAlgo<cms::cudacompat::CPUTraits>;
}  // namespace pixelgpudetails

#endif  // RecoLocalTracker_SiPixelRecHits_plugins_SiPixelRecHitSoAProducerAlgo_h
