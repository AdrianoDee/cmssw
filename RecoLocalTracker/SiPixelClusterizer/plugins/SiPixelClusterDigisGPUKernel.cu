// // C++ includes
// #include <cassert>
// #include <chrono>
// #include <cstdio>
// #include <cstdlib>
// #include <fstream>
// #include <iomanip>
// #include <iostream>
// #include <string>
//
// // CUDA includes
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <thrust/device_vector.h>
// #include <thrust/execution_policy.h>
// #include <thrust/host_vector.h>
// #include <thrust/sort.h>
// #include <thrust/unique.h>
//
// // cub includes
// #include <cub/cub.cuh>
//
// // CMSSW includes
// #include "CUDADataFormats/SiPixelCluster/interface/gpuClusteringConstants.h"
// #include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
// #include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
// #include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
// #include "RecoLocalTracker/SiPixelClusterizer/plugins/gpuCalibPixel.h"
// #include "RecoLocalTracker/SiPixelClusterizer/plugins/gpuClusterChargeCut.h"
// #include "RecoLocalTracker/SiPixelClusterizer/plugins/gpuClustering.h"
//
// // local includes
// #include "SiPixelClusterDigisGPUKernel.h"
//
// namespace pixelgpudetails {
//
//   __global__ void fillHitsModuleStartPhase2(uint32_t const *__restrict__ cluStart, uint32_t *__restrict__ moduleStart) {
//     assert(gpuClustering::MaxNumModules < 2048*2);
//     assert(1 == gridDim.x);
//     assert(0 == blockIdx.x);
//
//     int first = threadIdx.x;
//
//     // limit to MaxHitsInModule;
//     for (int i = first, iend = gpuClustering::MaxNumModules; i < iend; i += blockDim.x) {
//       moduleStart[i + 1] = std::min(gpuClustering::maxHitsInModule(), cluStart[i]);
//     }
//
//     __shared__ uint32_t ws[32];
//     cms::cuda::blockPrefixScan(moduleStart + 1, moduleStart + 1, 2048, ws);
//     cms::cuda::blockPrefixScan(moduleStart + 2049, moduleStart + 2049, gpuClustering::MaxNumModules - 2048, ws);
//
//     for (int i = first + 2049, iend = gpuClustering::MaxNumModules + 1; i < iend; i += blockDim.x) {
//       moduleStart[i] += moduleStart[2048];
//     }
//     __syncthreads();
//
//     // avoid overflow
//     constexpr auto MAX_HITS = gpuClustering::MaxNumClusters;
//     for (int i = first, iend = gpuClustering::MaxNumModules + 1; i < iend; i += blockDim.x) {
//       if (moduleStart[i] > MAX_HITS)
//         moduleStart[i] = MAX_HITS;
//     }
//   }
//
//   // Interface to outside
//   void SiPixelClusterDigisGPUKernel::makeDigiClustersAsync(const SiPixelDigisCUDA* digis_h,
//                                                        const uint32_t nDigis,
//                                                        cudaStream_t stream) {
//
//     digis_d = SiPixelDigisCUDA(nDigis,stream);
//
//     cudaMemcpyAsync(digis_d.moduleInd(), digis_h->moduleInd(), sizeof(uint16_t), cudaMemcpyDefault, stream);
//     cudaMemcpyAsync(digis_d.xx(), digis_h->xx(), sizeof(uint16_t), cudaMemcpyDefault, stream);
//     cudaMemcpyAsync(digis_d.yy(), digis_h->yy(), sizeof(uint16_t), cudaMemcpyDefault, stream);
//     cudaMemcpyAsync(digis_d.adc(), digis_h->adc(), sizeof(uint16_t), cudaMemcpyDefault, stream);
//     cudaMemcpyAsync(digis_d.pdigi(), digis_h->pdigi(), sizeof(uint32_t), cudaMemcpyDefault, stream);
//
//     // cudaMemcpyAsync(
//     //     &(nModules_Clusters_h[0]), clusters_d.moduleStart(), sizeof(uint32_t), cudaMemcpyDefault, stream)
//     // cudaMemcpyAsync(
//     //     &(nModules_Clusters_h[0]), clusters_d.moduleStart(), sizeof(uint32_t), cudaMemcpyDefault, stream)
//     // cudaMemcpyAsync(
//     //     &(nModules_Clusters_h[0]), clusters_d.moduleStart(), sizeof(uint32_t), cudaMemcpyDefault, stream)
//
//     clusters_d = SiPixelClustersCUDA(gpuClustering::MaxNumModules, stream);
//
//     nModules_Clusters_h = cms::cuda::make_host_unique<uint32_t[]>(1, stream);
//
//     using namespace gpuClustering;
//     int threadsPerBlock = 256;
//     int blocks = (std::max(int(nDigis), int(gpuClustering::MaxNumModules)) + threadsPerBlock - 1) / threadsPerBlock;
//
//     gpuCalibPixel::calibDigisPhase2<<<blocks, threadsPerBlock, 0, stream>>>(digis_d.moduleInd(),
//                                                                       digis_d.c_xx(),
//                                                                       digis_d.c_yy(),
//                                                                       digis_d.adc(),
//                                                                       nDigis,
//                                                                       clusters_d.moduleStart(),
//                                                                       clusters_d.clusInModule(),
//                                                                       clusters_d.clusModuleStart());
//       cudaCheck(cudaGetLastError());
// #ifdef GPU_DEBUG
//       cudaDeviceSynchronize();
//       cudaCheck(cudaGetLastError());
// #endif
//
// #ifdef GPU_DEBUG
//       std::cout << "CUDA countModules kernel launch with " << blocks << " blocks of " << threadsPerBlock
//                 << " threads\n";
// #endif
//
//       countModules<<<blocks, threadsPerBlock, 0, stream>>>(
//           digis_d.c_moduleInd(), clusters_d.moduleStart(), digis_d.clus(), nDigis);
//       cudaCheck(cudaGetLastError());
//
//       // read the number of modules into a data member, used by getProduct())
//       cudaCheck(cudaMemcpyAsync(
//           &(nModules_Clusters_h[0]), clusters_d.moduleStart(), sizeof(uint32_t), cudaMemcpyDefault, stream));
//
//       threadsPerBlock = 256;
//       blocks = MaxNumModules;
// #ifdef GPU_DEBUG
//       std::cout << "CUDA findClus kernel launch with " << blocks << " blocks of " << threadsPerBlock << " threads\n";
// #endif
//       findClus<<<blocks, threadsPerBlock, 0, stream>>>(digis_d.c_moduleInd(),
//                                                        digis_d.c_xx(),
//                                                        digis_d.c_yy(),
//                                                        clusters_d.c_moduleStart(),
//                                                        clusters_d.clusInModule(),
//                                                        clusters_d.moduleId(),
//                                                        digis_d.clus(),
//                                                        nDigis);
//       cudaCheck(cudaGetLastError());
// #ifdef GPU_DEBUG
//       cudaDeviceSynchronize();
//       cudaCheck(cudaGetLastError());
// #endif
//
//       // MUST be ONE block
//       fillHitsModuleStartPhase2<<<1, 2048, 0, stream>>>(clusters_d.c_clusInModule(), clusters_d.clusModuleStart());
//
//       // last element holds the number of all clusters
//       cudaCheck(cudaMemcpyAsync(&(nModules_Clusters_h[1]),
//                                 clusters_d.clusModuleStart() + gpuClustering::MaxNumModules,
//                                 sizeof(uint32_t),
//                                 cudaMemcpyDefault,
//                                 stream));
//
// #ifdef GPU_DEBUG
//       cudaDeviceSynchronize();
//       cudaCheck(cudaGetLastError());
// #endif
//
// }  // end clusterizer
//
// }  // namespace pixelgpudetails
