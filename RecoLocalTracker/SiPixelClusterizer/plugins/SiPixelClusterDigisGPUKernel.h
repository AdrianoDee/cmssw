// #ifndef RecoLocalTracker_SiPixelClusterizer_plugins_SiPixelClusterDigisGPUKernel_h
// #define RecoLocalTracker_SiPixelClusterizer_plugins_SiPixelClusterDigisGPUKernel_h
//
// #include <algorithm>
// #include <cuda_runtime.h>
//
// #include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
// #include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigiErrorsCUDA.h"
// #include "CUDADataFormats/SiPixelCluster/interface/SiPixelClustersCUDA.h"
// #include "FWCore/Utilities/interface/typedefs.h"
// #include "HeterogeneousCore/CUDAUtilities/interface/SimpleVector.h"
// #include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
// #include "HeterogeneousCore/CUDAUtilities/interface/host_noncached_unique_ptr.h"
// #include "DataFormats/SiPixelDigi/interface/PixelErrors.h"
//
// namespace pixelgpudetails {
//
//   class SiPixelClusterDigisGPUKernel {
//
//   public:
//
//     SiPixelClusterDigisGPUKernel() = default;
//     ~SiPixelClusterDigisGPUKernel() = default;
//
//     SiPixelClusterDigisGPUKernel(const SiPixelClusterDigisGPUKernel&) = delete;
//     SiPixelClusterDigisGPUKernel(SiPixelClusterDigisGPUKernel&&) = delete;
//     SiPixelClusterDigisGPUKernel& operator=(const SiPixelClusterDigisGPUKernel&) = delete;
//     SiPixelClusterDigisGPUKernel& operator=(SiPixelClusterDigisGPUKernel&&) = delete;
//
//     void makeDigiClustersAsync(const SiPixelDigisCUDA* digis_h,
//                            const uint32_t nDigis,
//                            cudaStream_t stream);
//
//     std::pair<SiPixelDigisCUDA, SiPixelClustersCUDA> getResults() {
//       digis_d.setNModulesDigis(nModules_Clusters_h[0], nDigis);
//       clusters_d.setNClusters(nModules_Clusters_h[1]);
//       // need to explicitly deallocate while the associated CUDA
//       // stream is still alive
//       //
//       // technically the statement above is not true anymore now that
//       // the CUDA streams are cached within the cms::cuda::StreamCache, but it is
//       // still better to release as early as possible
//       nModules_Clusters_h.reset();
//       return std::make_pair(std::move(digis_d), std::move(clusters_d));
//     }
//
//   private:
//     uint32_t nDigis = 0;
//
//     // Data to be put in the event
//     cms::cuda::host::unique_ptr<uint32_t[]> nModules_Clusters_h;
//     SiPixelClustersCUDA clusters_d;
//     SiPixelDigisCUDA digis_d;
//   };
//
//
// }  // namespace pixelgpudetails
//
// #endif  // RecoLocalTracker_SiPixelClusterizer_plugins_SiPixelClusterDigisGPUKernel_h
