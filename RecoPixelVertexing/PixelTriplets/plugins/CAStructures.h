#ifndef CUDADataFormats_TrackerGeometry_CAStructures_h
#define CUDADataFormats_TrackerGeometry_CAStructures_h

#include "HeterogeneousCore/CUDAUtilities/interface/SimpleVector.h"
#include "HeterogeneousCore/CUDAUtilities/interface/VecArray.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"

namespace caStructures {

  // types
  using hindex_type = uint32_t;  // FIXME from siPixelRecHitsHeterogeneousProduct
  using tindex_type = uint32_t;  // for tuples
  using cindex_type = uint32_t;  // for cells

  template<typename Tracker>
  using CellNeighborsT = cms::cuda::VecArray<cindex_type, Tracker::maxCellNeighbors>;

  template<typename Tracker>
  using CellTracksT = cms::cuda::VecArray<tindex_type, Tracker::maxCellTracks>;

  template<typename Tracker>
  using CellNeighborsVectorT = cms::cuda::SimpleVector<CellNeighborsT<Tracker>>;

  template<typename Tracker>
  using CellTracksVectorT = cms::cuda::SimpleVector<CellTracksT<Tracker>>;

  template<typename Tracker>
  using OuterHitOfCellContainerT = cms::cuda::VecArray<uint32_t, Tracker::maxCellsPerHit>;

  template <typename Tracker>
  using TupleMultiplicityT = cms::cuda::OneToManyAssoc<tindex_type, Tracker::maxHitsOnTrack + 1, Tracker::maxNumberOfTuples>;

  template <typename Tracker>
  using HitToTupleT = cms::cuda::OneToManyAssoc<tindex_type, -1, Tracker::maxHitsForContainers>;  // 3.5 should be enough

  template<typename Tracker>
  using TuplesContainerT = cms::cuda::OneToManyAssoc<hindex_type, Tracker::maxNumberOfTuples, Tracker::maxHitsForContainers>;

  template<typename Tracker>
  struct OuterHitOfCellT {
    OuterHitOfCellContainerT<Tracker>* container;
    int32_t offset;
    constexpr auto& operator[](int i) { return container[i - offset]; }
    constexpr auto const& operator[](int i) const { return container[i - offset]; }
  };

}

#endif
