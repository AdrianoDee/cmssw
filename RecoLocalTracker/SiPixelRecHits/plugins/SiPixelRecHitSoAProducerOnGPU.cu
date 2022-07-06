#include <cstdint>

#include <cuda_runtime.h>

#include "CUDADataFormats/BeamSpot/interface/BeamSpotCUDA.h"
#include "CUDADataFormats/SiPixelCluster/interface/SiPixelClustersCUDA.h"
#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"


#include <array>
#include <cassert>
#include <functional>
#include <vector>

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"

#include "SiPixelRecHitSoAProducerOnGPU.h"

#define GPU_DEBUG

namespace {



  __global__ void setHitsLayerStart(uint32_t const* __restrict__ hitsModuleStart,
                                    uint32_t* hitsLayerStart) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr auto m = 10;

    constexpr int L[11] = {0, 96, 320, 672, 1184, 1296, 1408,  1520, 1632, 1744, 1856};

    assert(0 == hitsModuleStart[0]);

    if (i <= m) {
      hitsLayerStart[i] = hitsModuleStart[L[i]];
#ifdef GPU_DEBUG
      printf("LayerStart %d/%d at module %d: %d\n", i, m, L[i], hitsLayerStart[i]);
#endif
    }
  }
}  // namespace


namespace pixelhitconverter {

  template<>
  TrackingRecHit2DGPU SiPixelRecHitSoAProducerOnGPU::convertHitsAsync(float* store32,
                                    uint16_t* store16,
                                    uint32_t* moduleStart,
                                    cudaStream_t stream) const
    {

  printf("%d \n",__LINE__);
  TrackingRecHit2DGPU hits(store32,store16, moduleStart, nHits_, moduleStart[startBPIX2_], isPhase2_, stream);
  printf("%d \n",__LINE__);
  printf("%d \n",__LINE__);
  if (nHits_) {
    printf("%d \n",__LINE__);
    setHitsLayerStart<<<1, 32, 0, stream>>>(moduleStart, hits.hitsLayerStart());
    cudaCheck(cudaGetLastError());
    constexpr auto nLayers = 10;
    cms::cuda::fillManyFromVector(hits.phiBinner(),
                                  nLayers,
                                  hits.iphi(),
                                  hits.hitsLayerStart(),
                                  nHits_,
                                  256,
                                  hits.phiBinnerStorage(),
                                  stream);
    cudaCheck(cudaGetLastError());

#ifdef GPU_DEBUG
    cudaCheck(cudaDeviceSynchronize());
#endif
  }

  return hits;
}
}
