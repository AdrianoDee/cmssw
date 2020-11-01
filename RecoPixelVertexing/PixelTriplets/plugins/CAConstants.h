#ifndef RecoPixelVertexing_PixelTriplets_plugins_CAConstants_h
#define RecoPixelVertexing_PixelTriplets_plugins_CAConstants_h

#include <cstdint>

#include <cuda_runtime.h>

#include "CUDADataFormats/SiPixelCluster/interface/gpuClusteringConstants.h"
#include "HeterogeneousCore/CUDAUtilities/interface/SimpleVector.h"
#include "HeterogeneousCore/CUDAUtilities/interface/VecArray.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"

// #define ONLY_PHICUT

namespace CAConstants {

  // constants
#ifndef ONLY_PHICUT
#ifdef GPU_SMALL_EVENTS
  constexpr uint32_t maxNumberOfTuples() { return 3 * 1024; }
#else
  //default
  constexpr uint32_t maxNumberOfTuples() { return 4 * 512 * 1024; }
#endif
#else
  constexpr uint32_t maxNumberOfTuples() { return 128 * 1024; }
#endif
  constexpr uint32_t maxNumberOfQuadruplets() { return maxNumberOfTuples(); }
#ifndef ONLY_PHICUT
#ifndef GPU_SMALL_EVENTS
  //default
  constexpr uint32_t maxNumberOfDoublets() { return 8 * 1024 * 1024; }
  constexpr uint32_t maxCellsPerHit() { return 128; }
#else
  constexpr uint32_t maxNumberOfDoublets() { return 128 * 1024; }
  constexpr uint32_t maxCellsPerHit() { return 128 / 2; }
#endif
#else
  constexpr uint32_t maxNumberOfDoublets() { return 512 * 1024; }
  constexpr uint32_t maxCellsPerHit() { return 16 * 128; }
#endif
  constexpr uint32_t maxNumOfActiveDoublets() { return maxNumberOfDoublets() / 128;} 


  constexpr uint32_t maxNumberOfLayerPairs() { return 70; }
  constexpr uint32_t maxNumberOfLayers() { return 28; }
  constexpr uint32_t maxTuples() { return maxNumberOfTuples(); }

  // types
  using hindex_type = uint16_t;  // FIXME from siPixelRecHitsHeterogeneousProduct
  using tindex_type = uint32_t;  //  for tuples

#ifndef ONLY_PHICUT
  using CellNeighbors = cms::cuda::VecArray<uint32_t, 24>;
  using CellTracks = cms::cuda::VecArray<tindex_type, 48>;
#else
  using CellNeighbors = cms::cuda::VecArray<uint32_t, 64>;
  using CellTracks = cms::cuda::VecArray<tindex_type, 128>;
#endif

  using CellNeighborsVector = cms::cuda::SimpleVector<CellNeighbors>;
  using CellTracksVector = cms::cuda::SimpleVector<CellTracks>;

  using OuterHitOfCell = cms::cuda::VecArray<uint32_t, maxCellsPerHit()>;
  using TuplesContainer = cms::cuda::OneToManyAssoc<hindex_type, maxTuples(), 12 * maxTuples()>;
  using HitToTuple =
      cms::cuda::OneToManyAssoc<tindex_type, pixelGPUConstants::maxNumberOfHits, 12 * maxTuples()>;  // 3.5 should be enough
  using TupleMultiplicity = cms::cuda::OneToManyAssoc<tindex_type, 24, maxTuples()>;

}  // namespace CAConstants

#endif  // RecoPixelVertexing_PixelTriplets_plugins_CAConstants_h
