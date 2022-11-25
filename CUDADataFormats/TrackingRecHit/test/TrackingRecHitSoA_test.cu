#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitsUtilities.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitSoADevice.h"

namespace testTrackingRecHit2DNew {

  __global__ void fill(trackingRecHitSoA::HitSoAView soa) {
    // assert(soa);

    int i = threadIdx.x;
    int j = blockIdx.x;
    if (i == 0 and j == 0) {
      soa.offsetBPIX2() = 22;
      soa[10].xLocal() = 1.11;
    }

    soa[i].iphi() = i % 10;
    soa.hitsLayerStart()[j] = j;
    //k = soa.test().a;
    __syncthreads();
  }

  __global__ void show(trackingRecHitSoA::HitSoAView soa) {
    // assert(soa);

    int i = threadIdx.x;
    int j = blockIdx.x;

    if (i == 0 and j == 0) {
      printf("nbins = %d \n", soa.phiBinner().nbins());
      printf("mMaxModules = %d \n", soa.nMaxModules());
      printf("offsetBPIX %d ->%d \n", i, soa.offsetBPIX2());
      printf("nHits %d ->%d \n", i, soa.nHits());
      printf("hitsModuleStart %d ->%d \n", i, soa.hitsModuleStart().at(28));
    }

    if (i < soa.nHits())
      printf("iPhi %d ->%d \n", i, soa[i].iphi());

    if (j * blockDim.x + i < soa.phiBinner().nbins())
      printf(">bin size %d ->%d \n", j * blockDim.x + i, soa.phiBinner().size(j * blockDim.x + i));
    __syncthreads();
  }

  void run(trackingRecHit::TrackingRecHitSoADevice& hits, cudaStream_t stream) {
    // assert(soa);
    printf("RUN!\n");
    int k = 0;
    fill<<<10, 100, 0, stream>>>(hits.view());
    printf("k = %d\n", k);

    cudaCheck(cudaDeviceSynchronize());

    cms::cuda::fillManyFromVector(hits.phiBinner(),
                                  10,
                                  hits.view().iphi(),
                                  hits.view().hitsLayerStart().data(),
                                  2000,
                                  256,
                                  hits.view().phiBinnerStorage(),
                                  stream);
    cudaCheck(cudaDeviceSynchronize());
    show<<<10, 1000, 0, stream>>>(hits.view());
    cudaCheck(cudaDeviceSynchronize());
  }

}  // namespace testTrackingRecHit2DNew
