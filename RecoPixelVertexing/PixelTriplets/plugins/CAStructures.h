#ifndef RecoPixelVertexing_PixelTriplets_plugins_alpaka_CAStructures_h
#define RecoPixelVertexing_PixelTriplets_plugins_alpaka_CAStructures_h

#include "HeterogeneousCore/AlpakaUtilities/interface/SimpleVector.h"
#include "HeterogeneousCore/AlpakaUtilities/interface/VecArray.h"
#include "HeterogeneousCore/AlpakaUtilities/interface/HistoContainer.h"

namespace caStructures {

  // types
  // using typename TrackerTraits::hindex_type = uint32_t;  // FIXME from siPixelRecHitsHeterogeneousProduct
  // using typename TrackerTraits::tindex_type = uint32_t;  // for tuples
  // using typename TrackerTraits::cindex_type = uint32_t;  // for cells

  template <typename TrackerTraits>
  using CellNeighborsT =
      cms::alpakatools::VecArray<typename TrackerTraits::cindex_type, TrackerTraits::maxCellNeighbors>;

  template <typename TrackerTraits>
  using CellTracksT = cms::alpakatools::VecArray<typename TrackerTraits::tindex_type, TrackerTraits::maxCellTracks>;

  template <typename TrackerTraits>
  using CellNeighborsVectorT = cms::alpakatools::SimpleVector<CellNeighborsT<TrackerTraits>>;

  template <typename TrackerTraits>
  using CellTracksVectorT = cms::alpakatools::SimpleVector<CellTracksT<TrackerTraits>>;

  template <typename TrackerTraits>
  using OuterHitOfCellContainerT = cms::alpakatools::VecArray<uint32_t, TrackerTraits::maxCellsPerHit>;

  template <typename TrackerTraits>
  using TupleMultiplicityT = cms::alpakatools::OneToManyAssoc<typename TrackerTraits::tindex_type,
                                                              TrackerTraits::maxHitsOnTrack + 1,
                                                              TrackerTraits::maxNumberOfTuples>;

  template <typename TrackerTraits>
  using HitToTupleT = cms::alpakatools::OneToManyAssoc<typename TrackerTraits::tindex_type,
                                                       TrackerTraits::maxNumberOfHits,
                                                       TrackerTraits::maxHitsForContainers>;  // 3.5 should be enough

  template <typename TrackerTraits>
  using TuplesContainerT = cms::alpakatools::OneToManyAssoc<typename TrackerTraits::hindex_type,
                                                            TrackerTraits::maxNumberOfTuples,
                                                            TrackerTraits::maxHitsForContainers>;

  template <typename TrackerTraits>
  struct OuterHitOfCellT {
    OuterHitOfCellContainerT<TrackerTraits>* container;
    int32_t offset;
    constexpr auto& operator[](int i) { return container[i - offset]; }
    constexpr auto const& operator[](int i) const { return container[i - offset]; }
  };

}  // namespace caStructures

#endif
