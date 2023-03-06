#ifndef plugin_SiPixelClusterizer_gpuClusterChargeCut_h
#define plugin_SiPixelClusterizer_gpuClusterChargeCut_h

#include <cstdint>
#include <cstdio>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaUtilities/interface/prefixScan.h"
#include "DataFormats/SiPixelClusterSoA/interface/gpuClusteringConstants.h"
#include "../SiPixelClusterThresholds.h"

namespace gpuClustering {

  template <typename TrackerTraits>
  struct clusterChargeCut {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        const TAcc& acc,
        SiPixelClusterThresholds clusterThresholds, // charge cut on cluster in electrons (for layer 1 and for other layers)
        uint16_t* __restrict__ id,                 // module id of each pixel (modified if bad cluster)
        uint16_t const* __restrict__ adc,          //  charge of each pixel
        uint32_t const* __restrict__ moduleStart,  // index of the first pixel of each module
        uint32_t* __restrict__ nClustersInModule,  // modified: number of clusters found in each module
        uint32_t const* __restrict__ moduleId,     // module id of each module
        int32_t* __restrict__ clusterId,           // modified: cluster id of each pixel
        const uint32_t numElements) const {
      constexpr int startBPIX2 = TrackerTraits::layerStart[1];
      [[maybe_unused]] constexpr int nMaxModules = TrackerTraits::numberOfModules;

      const uint32_t blockIdx(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
      if (blockIdx >= moduleStart[0])
        return;

      auto firstPixel = moduleStart[1 + blockIdx];
      auto thisModuleId = id[firstPixel];
      ALPAKA_ASSERT_OFFLOAD(thisModuleId < MaxNumModules);
      ALPAKA_ASSERT_OFFLOAD(thisModuleId == moduleId[blockIdx]);

      auto nclus = nClustersInModule[thisModuleId];
      if (nclus == 0)
        return;

      const uint32_t threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
      if (threadIdxLocal == 0 && nclus > MaxNumClustersPerModules)
        printf("Warning too many clusters in module %d in block %d: %d > %d\n",
               thisModuleId,
               blockIdx,
               nclus,
               MaxNumClustersPerModules);

      // Stride = block size.
      const uint32_t blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);

      // Get thread / CPU element indices in block.
      const auto& [firstElementIdxNoStride, endElementIdxNoStride] =
          cms::alpakatools::element_index_range_in_block(acc, firstPixel);

      if (nclus > MaxNumClustersPerModules) {
        uint32_t firstElementIdx = firstElementIdxNoStride;
        uint32_t endElementIdx = endElementIdxNoStride;
        // remove excess  FIXME find a way to cut charge first....
        for (uint32_t i = firstElementIdx; i < numElements; ++i) {
          if (not cms::alpakatools::next_valid_element_index_strided(
                  i, firstElementIdx, endElementIdx, blockDimension, numElements))
            break;
          if (id[i] == InvId)
            continue;  // not valid
          if (id[i] != thisModuleId)
            break;  // end of module
          if (clusterId[i] >= MaxNumClustersPerModules) {
            id[i] = InvId;
            clusterId[i] = InvId;
          }
        }
        nclus = MaxNumClustersPerModules;
      }

#ifdef GPU_DEBUG
      if (thisModuleId % 100 == 1)
        if (threadIdxLocal == 0)
          printf("start clusterizer for module %d in block %d\n", thisModuleId, blockIdx);
#endif

      auto& charge = alpaka::declareSharedVar<int32_t[MaxNumClustersPerModules], __COUNTER__>(acc);
      auto& ok = alpaka::declareSharedVar<uint8_t[MaxNumClustersPerModules], __COUNTER__>(acc);
      auto& newclusId = alpaka::declareSharedVar<uint16_t[MaxNumClustersPerModules], __COUNTER__>(acc);

      ALPAKA_ASSERT_OFFLOAD(nclus <= MaxNumClustersPerModules);
      cms::alpakatools::for_each_element_in_block_strided(acc, nclus, [&](uint32_t i) { charge[i] = 0; });
      alpaka::syncBlockThreads(acc);

      uint32_t firstElementIdx = firstElementIdxNoStride;
      uint32_t endElementIdx = endElementIdxNoStride;
      for (uint32_t i = firstElementIdx; i < numElements; ++i) {
        if (not cms::alpakatools::next_valid_element_index_strided(
                i, firstElementIdx, endElementIdx, blockDimension, numElements))
          break;
        if (id[i] == InvId)
          continue;  // not valid
        if (id[i] != thisModuleId)
          break;  // end of module
        alpaka::atomicAdd(acc, &charge[clusterId[i]], static_cast<int32_t>(adc[i]), alpaka::hierarchy::Threads{});
      }
      alpaka::syncBlockThreads(acc);

      auto chargeCut = clusterThresholds.getThresholdForLayerOnCondition(thisModuleId < startBPIX2);
      cms::alpakatools::for_each_element_in_block_strided(
          acc, nclus, [&](uint32_t i) { newclusId[i] = ok[i] = charge[i] > chargeCut ? 1 : 0; });
      alpaka::syncBlockThreads(acc);

      // renumber
      auto& ws = alpaka::declareSharedVar<uint16_t[32], __COUNTER__>(acc);
      cms::alpakatools::blockPrefixScan(acc, newclusId, nclus, ws);

      ALPAKA_ASSERT_OFFLOAD(nclus >= newclusId[nclus - 1]);

      if (nclus == newclusId[nclus - 1])
        return;

      nClustersInModule[thisModuleId] = newclusId[nclus - 1];
      alpaka::syncBlockThreads(acc);

      // mark bad cluster again
      cms::alpakatools::for_each_element_in_block_strided(acc, nclus, [&](uint32_t i) {
        if (0 == ok[i])
          newclusId[i] = InvId + 1;
      });
      alpaka::syncBlockThreads(acc);

      // reassign id
      firstElementIdx = firstElementIdxNoStride;
      endElementIdx = endElementIdxNoStride;
      for (uint32_t i = firstElementIdx; i < numElements; ++i) {
        if (not cms::alpakatools::next_valid_element_index_strided(
                i, firstElementIdx, endElementIdx, blockDimension, numElements))
          break;
        if (id[i] == InvId)
          continue;  // not valid
        if (id[i] != thisModuleId)
          break;  // end of module
        clusterId[i] = newclusId[clusterId[i]] - 1;
        if (clusterId[i] == InvId)
          id[i] = InvId;
      }

      //done
    }
  };

}  // namespace gpuClustering

#endif  // plugin_SiPixelClusterizer_gpuClusterChargeCut_h



// #ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuClusterChargeCut_h
// #define RecoLocalTracker_SiPixelClusterizer_plugins_gpuClusterChargeCut_h

// #include <cstdint>
// #include <cstdio>

// #include "CUDADataFormats/SiPixelCluster/interface/gpuClusteringConstants.h"
// #include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
// #include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
// #include "HeterogeneousCore/CUDAUtilities/interface/prefixScan.h"

// // local include(s)
// #include "SiPixelClusterThresholds.h"

// namespace gpuClustering {

//   template <typename TrackerTraits>
//   __global__ void clusterChargeCut(
//       SiPixelClusterThresholds
//           clusterThresholds,             // charge cut on cluster in electrons (for layer 1 and for other layers)
//       uint16_t* __restrict__ id,         // module id of each pixel (modified if bad cluster)
//       uint16_t const* __restrict__ adc,  //  charge of each pixel
//       uint32_t const* __restrict__ moduleStart,  // index of the first pixel of each module
//       uint32_t* __restrict__ nClustersInModule,  // modified: number of clusters found in each module
//       uint32_t const* __restrict__ moduleId,     // module id of each module
//       int32_t* __restrict__ clusterId,           // modified: cluster id of each pixel
//       uint32_t numElements) {
//     __shared__ int32_t charge[maxNumClustersPerModules];
//     __shared__ uint8_t ok[maxNumClustersPerModules];
//     __shared__ uint16_t newclusId[maxNumClustersPerModules];

//     constexpr int startBPIX2 = TrackerTraits::layerStart[1];
//     [[maybe_unused]] constexpr int nMaxModules = TrackerTraits::numberOfModules;

//     assert(nMaxModules < maxNumModules);
//     assert(startBPIX2 < nMaxModules);

//     auto firstModule = blockIdx.x;
//     auto endModule = moduleStart[0];
//     for (auto module = firstModule; module < endModule; module += gridDim.x) {
//       auto firstPixel = moduleStart[1 + module];
//       auto thisModuleId = id[firstPixel];
//       while (thisModuleId == invalidModuleId and firstPixel < numElements) {
//         // skip invalid or duplicate pixels
//         ++firstPixel;
//         thisModuleId = id[firstPixel];
//       }
//       if (firstPixel >= numElements) {
//         // reached the end of the input while skipping the invalid pixels, nothing left to do
//         break;
//       }
//       if (thisModuleId != moduleId[module]) {
//         // reached the end of the module while skipping the invalid pixels, skip this module
//         continue;
//       }
//       assert(thisModuleId < nMaxModules);

//       auto nclus = nClustersInModule[thisModuleId];
//       if (nclus == 0)
//         continue;

//       if (threadIdx.x == 0 && nclus > maxNumClustersPerModules)
//         printf("Warning too many clusters in module %d in block %d: %d > %d\n",
//                thisModuleId,
//                blockIdx.x,
//                nclus,
//                maxNumClustersPerModules);

//       auto first = firstPixel + threadIdx.x;

//       if (nclus > maxNumClustersPerModules) {
//         // remove excess  FIXME find a way to cut charge first....
//         for (auto i = first; i < numElements; i += blockDim.x) {
//           if (id[i] == invalidModuleId)
//             continue;  // not valid
//           if (id[i] != thisModuleId)
//             break;  // end of module
//           if (clusterId[i] >= maxNumClustersPerModules) {
//             id[i] = invalidModuleId;
//             clusterId[i] = invalidModuleId;
//           }
//         }
//         nclus = maxNumClustersPerModules;
//       }

// #ifdef GPU_DEBUG
//       if (thisModuleId % 100 == 1)
//         if (threadIdx.x == 0)
//           printf("start cluster charge cut for module %d in block %d\n", thisModuleId, blockIdx.x);
// #endif

//       assert(nclus <= maxNumClustersPerModules);
//       for (auto i = threadIdx.x; i < nclus; i += blockDim.x) {
//         charge[i] = 0;
//       }
//       __syncthreads();

//       for (auto i = first; i < numElements; i += blockDim.x) {
//         if (id[i] == invalidModuleId)
//           continue;  // not valid
//         if (id[i] != thisModuleId)
//           break;  // end of module
//         atomicAdd(&charge[clusterId[i]], adc[i]);
//       }
//       __syncthreads();

//       auto chargeCut = clusterThresholds.getThresholdForLayerOnCondition(thisModuleId < startBPIX2);

//       bool good = true;
//       for (auto i = threadIdx.x; i < nclus; i += blockDim.x) {
//         newclusId[i] = ok[i] = charge[i] >= chargeCut ? 1 : 0;
//         if (0 == ok[i])
//           good = false;
//       }

//       // if all clusters above threshold do nothing
//       if (__syncthreads_and(good))
//         continue;

//       // renumber
//       __shared__ uint16_t ws[32];
//       cms::cuda::blockPrefixScan(newclusId, nclus, ws);

//       assert(nclus > newclusId[nclus - 1]);

//       nClustersInModule[thisModuleId] = newclusId[nclus - 1];

//       // reassign id
//       for (auto i = first; i < numElements; i += blockDim.x) {
//         if (id[i] == invalidModuleId)
//           continue;  // not valid
//         if (id[i] != thisModuleId)
//           break;  // end of module
//         if (0 == ok[clusterId[i]])
//           clusterId[i] = id[i] = invalidModuleId;
//         else
//           clusterId[i] = newclusId[clusterId[i]] - 1;
//       }

//       //done
//       __syncthreads();
//     }  // loop on modules
//   }

// }  // namespace gpuClustering

// #endif  // RecoLocalTracker_SiPixelClusterizer_plugins_gpuClusterChargeCut_h
