// C++ headers
#include <algorithm>
#include <numeric>

// CUDA runtime
#include <cuda_runtime.h>

// CMSSW headers
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "RecoLocalTracker/SiPixelClusterizer/plugins/SiPixelRawToClusterGPUKernel.h"
#include "RecoLocalTracker/SiPixelClusterizer/plugins/gpuClusteringConstants.h"

#include "PixelRecHits.h"
#include "gpuPixelRecHits.h"

#define GPU_DEBUG 1
namespace {
  __global__ void setHitsLayerStart(uint32_t const* __restrict__ hitsModuleStart,
                                    pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                    uint32_t* hitsLayerStart,int16_t* p) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;

    assert(0 == hitsModuleStart[0]);

/*    for(int j = 0; j<10000;j++)
    {
	printf("%d %d - ",j,p[i]);
    }*/
    if (i < 29) {
      hitsLayerStart[i] = hitsModuleStart[cpeParams->layerGeometry().layerStart[i]];
#ifdef GPU_DEBUG
      printf("LayerStart %d %d: %d %d\n", i, cpeParams->layerGeometry().layerStart[i], hitsLayerStart[i],p[i]);
#endif
    }
  }
}  // namespace

namespace pixelgpudetails {

  TrackingRecHit2DCUDA PixelRecHitGPUKernel::makeHitsAsync(SiPixelDigisCUDA const& digis_d,
                                                           SiPixelClustersCUDA const& clusters_d,
                                                           BeamSpotCUDA const& bs_d,
                                                           pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                                           cudaStream_t stream) const {
    auto nHits = clusters_d.nClusters();
    TrackingRecHit2DCUDA hits_d(nHits, cpeParams, clusters_d.clusModuleStart(), stream);

    int threadsPerBlock = 128;
    int blocks = digis_d.nModules();  // active modules (with digis)

#ifdef GPU_DEBUG
    std::cout << "launching getHits kernel for " << blocks << " blocks. Inputs: " << std::endl
	      << " > "<< digis_d.nDigis() << " digis " <<std::endl
 	      << " > "<< nHits << " clusters " << std::endl;
#endif
    if (blocks)  // protect from empty events
      //gpuPixelRecHits::getHits<<<blocks, threadsPerBlock, 0, stream>>>(
   gpuPixelRecHits::getHits<<<1, 1, 0, stream>>>(    
      cpeParams, bs_d.data(), digis_d.view(), digis_d.nDigis(), clusters_d.view(), hits_d.view());
    cudaCheck(cudaGetLastError());
#ifdef GPU_DEBUG
    cudaDeviceSynchronize();
    cudaCheck(cudaGetLastError());
#endif
std::cout << nHits << std::endl;

    // assuming full warp of threads is better than a smaller number...
    if (nHits) {
      setHitsLayerStart<<<1, 32, 0, stream>>>(clusters_d.clusModuleStart(), cpeParams, hits_d.hitsLayerStart(),hits_d.iphi());
      cudaCheck(cudaGetLastError());
    }

    if (nHits) {
      cms::cuda::fillManyFromVector(hits_d.phiBinner(), 28, hits_d.iphi(), hits_d.hitsLayerStart(), nHits, 256, stream);
      cudaCheck(cudaGetLastError());
    }
   #ifdef GPU_DEBUG
    cudaDeviceSynchronize();
    cudaCheck(cudaGetLastError());
#endif
//for(int i = 0; i<1000;i++)
//{
std::cout << sizeof(hits_d.iphi())/(sizeof(uint16_t)) << std::endl;
std::cout << sizeof(hits_d.hitsLayerStart())/(sizeof(uint32_t)) << std::endl;

//std::cout << hits_d.iphi()[0] << std::endl;
//std::cout << hits_d.iphi()[5] << std::endl;
 //} 
  return hits_d;
  }

}  // namespace pixelgpudetails
