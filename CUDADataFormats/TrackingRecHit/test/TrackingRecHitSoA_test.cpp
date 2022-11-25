#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitSoAHost.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitSoADevice.h"

#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/allocate_device.h"

#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

namespace testTrackingRecHit2DNew {

  void run(trackingRecHit::TrackingRecHitSoADevice& hits, cudaStream_t stream);

}

int main() {
  cms::cudatest::requireDevices();

  cudaStream_t stream;
  cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamDefault));

  // inner scope to deallocate memory before destroying the stream
  {
    uint32_t nHits = 2000;
    int32_t offset = 100;
    uint32_t moduleStart[1856];

    for (size_t i = 0; i < 1856; i++) {
      moduleStart[i] = i * 2;
    }

    trackingRecHit::TrackingRecHitSoADevice tkhit(nHits, false, offset, nullptr, &moduleStart[0], stream);

    testTrackingRecHit2DNew::run(tkhit, stream);
    printf("tkhit hits %d \n", tkhit.nHits());

    auto test = tkhit.localCoordToHostAsync(stream);
    printf("test[9] %.2f\n", test[9]);

    printf("nModules %d \n", tkhit.nModules());

    // auto mods = tkhit.hitsModuleStartToHostAsync(stream);
    // auto ret = cms::cuda::make_host_unique<uint32_t[]>(tkhit.nModules() + 1, stream);
    // uint32_t* ret;
    // cudaCheck();
    // cudaMemcpyAsync(ret, tkhit.view().hitsModuleStart().data(), sizeof(uint32_t) * (tkhit.nModules() + 1), cudaMemcpyDeviceToHost, stream);
    auto ret = tkhit.hitsModuleStartToHostAsync(stream);
    // size_t skipSize = int(trackingRecHitSoA::columnsSizes * nHits);
    // cudaCheck(cudaMemcpyAsync(ret,
    //                           tkhit.const_buffer().get() + skipSize,
    //                           sizeof(uint32_t) * (1856 + 1),
    //                           cudaMemcpyDeviceToHost,
    //                           ctx.stream()));

    printf("mods[9] %d\n", ret[9]);
  }

  cudaCheck(cudaStreamDestroy(stream));

  return 0;
}
