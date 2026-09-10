#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAStructures_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAStructures_h

#include <cmath>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "HeterogeneousCore/AlpakaInterface/interface/SimpleVector.h"
#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"

namespace caStructures {

  using Quality = ::pixelTrack::Quality;

  //Configuration params common to all topologies, for the algorithms
  struct AlgoParams {
    // Container sizes
    float avgHitsPerTrack_;
    float avgCellsPerHit_;
    float avgCellsPerCell_;
    float avgTracksPerCell_;

    // Algorithm Parameters
    // minHitsPerNtuplet_ is compared against the number of layers in the ntuplet
    // (CACell::find_ntuplets); the name is that of the configuration parameter set by the menus.
    uint16_t minHitsPerNtuplet_;
    uint16_t minHitsForSharingCut_;

    // Flags
    bool useRiemannFit_;
    // Enables the broken-line fit corrections (material partition and rigid-node guard, Karimaki Fisher
    // basis, pion 1/beta, trapezoid quadrature, 3x3 covariance blend; see BrokenLine.h). Read only by
    // the CA main fit. Positional struct: keep this slot in the makeCommonParams initializer order.
    bool useFitCorrections_;
    bool fitNas4_;
    bool earlyFishbone_;
    bool lateFishbone_;
    bool onlySameLayersFishbone_;
    bool doStats_;
    bool doSharedHitCut_;
    bool dupPassThrough_;
    bool useSimpleTripletCleaner_;
    bool doTripletCleaner_;
    bool doFastDuplicateRemover_;
    bool doEarlyDuplicateRemover_;

    // Inline per-triplet DNN gate (Phase2OTStubs only): an MLP with compile-time weights
    // (CATripletDNN.h) scores every accepted triplet in Kernel_connect, pixel-only ones included
    // (nStubs==0, sentinel stub features), and rejects those below tripletDNNThreshold_
    // (negative => the threshold compiled into the weight header).
    bool useTripletDNN_;
    float tripletDNNThreshold_;

    // Track classifier (Phase2OTStubs only): an MLP with compile-time weights (CATrackDNN.h) scores
    // each fitted candidate in Kernel_classifyTracks; its score replaces the chi2-based
    // strict->tight promotion (trackDNNThreshold_ negative => the threshold in the weight header).
    bool useTrackDNN_;
    float trackDNNThreshold_;

    // Device-memory allocation strategy (see CAHitNtupletGeneratorKernels)
    bool delayAllocations_;    // Defer cell-derived + hit->track buffers until their real size is known
    bool countDoubletsFirst_;  // Run a count-only doublet pass to size simpleCells/hitToCellStorage exactly

    // Duplicate gate width, in units of the fitted covariance: two tracks are duplicates when every
    // fitted parameter p satisfies dp^2 <= fastDupNSigma2_*(cov_i+cov_j) (Kernel_fastDuplicateRemover
    // and Kernel_rejectDuplicate). The Phase-1 specializations use their own nSigma2Phase1 constant.
    float fastDupNSigma2_;

    // Tracking iteration this configuration belongs to; keeps the diagnostics separated when
    // several CA instances run in the same job.
    ::pixelTrack::Iteration iterationName_;
  };

  // Hits data formats
  using HitsView = ::reco::TrackingRecHitView;
  using HitModulesConstView = ::reco::HitModuleSoAConstView;
  using HitsConstView = ::reco::TrackingRecHitConstView;
  using MapToHitConstView = ::reco::TrackingRecHitsMaskingConstView;

  using MapToHitConstView = ::reco::TrackingRecHitsMaskingConstView;

  // Tracks data formats
  using TkSoAView = ::reco::TrackSoAView;
  using TkHitsSoAView = ::reco::TrackHitSoAView;
  using TkSoABlocksView = ::reco::TrackBlocksView;

  // Indices for hits, tracks and cells
  using hindex_type = uint32_t;
  using tindex_type = uint32_t;
  using cindex_type = uint32_t;

  // cellNeighbors storage encoding: bit 31 of each stored neighbor cell index
  // distinguishes layer-skipping (1) from non-layer-skipping (0) neighbors,
  // so the histogram needs a single bin per cell instead of two.
  inline constexpr uint32_t kSkipsLayerFlag = 0x80000000u;
  inline constexpr uint32_t kCellIndexMask = 0x7FFFFFFFu;

  using GenericContainer = cms::alpakatools::
      OneToManyAssocRandomAccess<hindex_type, cms::alpakatools::kDynamicSize, cms::alpakatools::kDynamicSize>;
  using GenericContainerStorage = typename GenericContainer::value_type;
  using GenericContainerOffsets = typename GenericContainer::Counter;
  using GenericContainerView = typename GenericContainer::View;

  using SequentialContainer = cms::alpakatools::
      OneToManyAssocSequential<hindex_type, cms::alpakatools::kDynamicSize, cms::alpakatools::kDynamicSize>;
  using SequentialContainerStorage = typename SequentialContainer::value_type;
  using SequentialContainerOffsets = typename SequentialContainer::Counter;
  using SequentialContainerView = typename SequentialContainer::View;

  template <typename TrackerTraits>
  using PhiBinnerT = cms::alpakatools::HistoContainer<int16_t,
                                                      256,
                                                      cms::alpakatools::kDynamicSize,
                                                      8 * sizeof(int16_t),
                                                      hindex_type,
                                                      TrackerTraits::numberOfLayers>;

  template <typename TrackerTraits>
  using CellNeighborsT =
      cms::alpakatools::VecArray<typename TrackerTraits::cindex_type, TrackerTraits::maxCellNeighbors>;

  template <typename TrackerTraits>
  using CellTracksT = cms::alpakatools::VecArray<tindex_type, TrackerTraits::maxCellTracks>;

  template <typename TrackerTraits>
  using CellNeighborsVectorT = cms::alpakatools::SimpleVector<CellNeighborsT<TrackerTraits>>;

  template <typename TrackerTraits>
  using CellTracksVectorT = cms::alpakatools::SimpleVector<CellTracksT<TrackerTraits>>;

  template <typename TrackerTraits>
  using OuterHitOfCellContainerT = cms::alpakatools::VecArray<uint32_t, TrackerTraits::maxCellsPerHit>;

  template <typename TrackerTraits>
  using TupleMultiplicityT = cms::alpakatools::
      OneToManyAssocRandomAccess<tindex_type, TrackerTraits::maxHitsOnTrack + 1, TrackerTraits::maxNumberOfTuples>;

  template <typename TrackerTraits>
  using HitContainerT =
      cms::alpakatools::OneToManyAssocSequential<uint32_t,
                                                 TrackerTraits::maxNumberOfTuples + 1,
                                                 TrackerTraits::avgHitsPerTrack * TrackerTraits::maxNumberOfTuples>;

  template <typename TrackerTraits>
  using HitToTupleT =
      cms::alpakatools::OneToManyAssocRandomAccess<tindex_type,
                                                   cms::alpakatools::kDynamicSize,
                                                   TrackerTraits::maxNumberOfTuples *
                                                       TrackerTraits::avgHitsPerTrack>;  // 3.5 should be enough

  template <typename TrackerTraits>
  using TuplesContainerT = cms::alpakatools::OneToManyAssocRandomAccess<typename TrackerTraits::hindex_type,
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

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAStructures_h
