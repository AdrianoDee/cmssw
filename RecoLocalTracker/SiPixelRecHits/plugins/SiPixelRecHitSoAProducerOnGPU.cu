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


namespace {

  constexpr int L[11] = {0, 96, 320, 672, 1184, 1296, 1408,  1520, 1632, 1744, 1856};

  __global__ void setHitsLayerStart(uint32_t const* __restrict__ hitsModuleStart,
                                    uint32_t* hitsLayerStart) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr auto m = 10;

    assert(0 == hitsModuleStart[0]);

    if (i <= m) {
      hitsLayerStart[i] = hitsModuleStart[L[i]];
#ifdef GPU_DEBUG
      printf("LayerStart %d/%d at module %d: %d\n", i, m, cpeParams->layerGeometry().layerStart[i], hitsLayerStart[i]);
#endif
    }
  }
}  // namespace


namespace pixelhitconverter {

  template<>
  TrackingRecHit2DGPU SiPixelRecHitSoAProducerOnGPU::convertHitsAsync(float* store32,
                                    uint16_t* store16,
                                    uint32_t* moduleStart,
                                    int nHits,
                                    int startBPIX2,
                                    bool isPhase2,
                                    cudaStream_t stream) const
    {

  TrackingRecHit2DGPU hits(store32,store16, moduleStart, nHits, moduleStart[startBPIX2], isPhase2, stream);


  if (nHits) {
    setHitsLayerStart<<<1, 32, 0, stream>>>(moduleStart, hits.hitsLayerStart());
    cudaCheck(cudaGetLastError());
    constexpr auto nLayers = 10;
    cms::cuda::fillManyFromVector(hits.phiBinner(),
                                  nLayers,
                                  hits.iphi(),
                                  hits.hitsLayerStart(),
                                  nHits,
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
