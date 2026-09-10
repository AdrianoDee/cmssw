#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernels_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernels_h

// #define GPU_DEBUG
// #define DUMP_GPU_TK_TUPLES
// Per-stage rejection accounting for the CA funnel; see CAPipelineCounters.h.
// #define CA_PIPELINE_COUNTERS

#include <cstdint>
#include <string>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsMaskingSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/AtomicPairCounter.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"
#include "RecoTracker/PixelSeeding/interface/alpaka/CAPairSoACollection.h"
// The layout is defined unconditionally; the device buffer is allocated only under CA_TRIPLET_DUMP.
#include "RecoTracker/PixelSeeding/interface/alpaka/TripletDumpSoACollection.h"
#include "CATripletDumpMacro.h"
#include "CASizingDumpMacro.h"

#include "CACell.h"
#include "CAPipelineCounters.h"
#include "CAPixelDoublets.h"
#include "CAStructures.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace ::caStructures;

  // Defined in HelixFit.h; finalDedup only ever takes it by pointer.
  struct MergerDedupConfirmInputs;

  namespace caHitNtupletGenerator {

    // Re-export pipeline counter enum from global namespace
    using namespace ::caHitNtupletGenerator;

    //Counters
    struct Counters {
      unsigned long long nEvents;
      unsigned long long nHits;
      unsigned long long nCells;
      unsigned long long nTuples;
      unsigned long long nFitTracks;
      unsigned long long nLooseTracks;
      unsigned long long nGoodTracks;
      unsigned long long nUsedHits;
      unsigned long long nDupHits;
      unsigned long long nFishCells;
      unsigned long long nKilledCells;
      unsigned long long nEmptyCells;
      unsigned long long nZeroTrackCells;
      // Overflow accounting for lossy truncation. The first four count events in which the
      // container ended at or past its capacity; the last two count individual dropped entries.
      unsigned long long nTupleOverflow;       // hitContainer: tuple slots at/past capacity
      unsigned long long nCellOverflow;        // simpleCells/doublet store: nCells at maxDoublets
      unsigned long long nTripletOverflow;     // cellToCell (cellNeighbors): nTrips at maxTriplets
      unsigned long long nCellTrackOverflow;   // cellToTrack (cellTracks): nCellTracks at capacity
      unsigned long long nHitToTupleOverflow;  // hitToTuple: entries dropped, key past nOnes
      unsigned long long nHitToCellOverflow;   // hitToCell: entries dropped, key past nOnes
    };

    // Per-track hit rows to reserve for maxTuples tuples at the configured average hits per track.
    // Shared by the internal hit containers and the output trackHits SoA, so the two capacities are
    // equal by construction. Rounded up, and formed in double because at maxTuples ~ 1e6 a float
    // product would round away more than the fraction being kept.
    inline uint32_t nHitRowsForTuples(uint32_t maxTuples, float avgHitsPerTrack) {
      const double rows = double(maxTuples) * double(avgHitsPerTrack);
      uint32_t n = (rows > 0.) ? uint32_t(rows) : 0u;
      if (double(n) < rows)
        ++n;
      return (n > 0u) ? n : 1u;
    }

    //Full list of params = algo params + quality cuts
    //Generic template
    template <typename TrackerTraits, typename Enable = void>
    struct ParamsT {};

    template <typename TrackerTraits>
    struct ParamsT<TrackerTraits, pixelTopology::isPhase1Topology<TrackerTraits>> {
      using TT = TrackerTraits;
      using QualityCuts = ::pixelTrack::QualityCutsT<TT>;  //track quality cuts

      ParamsT(AlgoParams const& commonCuts,
              QualityCuts const& qualityCuts,
              std::string const& tripletDumpIteration = std::string())
          : algoParams_(commonCuts), tripletDumpIteration_(tripletDumpIteration), qualityCuts_(qualityCuts) {}

      const AlgoParams algoParams_;
      // Host side only: iteration whose built triplets are dumped, empty for all of them.
      const std::string tripletDumpIteration_;
      const QualityCuts qualityCuts_{// polynomial coefficients for the pT-dependent chi2 cut
                                     {0.68177776, 0.74609577, -0.08035491, 0.00315399},
                                     // max pT used to determine the chi2 cut
                                     10.,
                                     // chi2 scale factor: 30 for broken line fit, 45 for Riemann fit
                                     30.,
                                     // regional cuts for triplets
                                     {
                                         0.3,  // |Tip| < 0.3 cm
                                         0.5,  // pT > 0.5 GeV
                                         12.0  // |Zip| < 12.0 cm
                                     },
                                     // regional cuts for quadruplets
                                     {
                                         0.5,  // |Tip| < 0.5 cm
                                         0.3,  // pT > 0.3 GeV
                                         12.0  // |Zip| < 12.0 cm
                                     }};

    };  // Params Phase1

    template <typename TrackerTraits>
    struct ParamsT<TrackerTraits, pixelTopology::isPhase2Topology<TrackerTraits>> : public AlgoParams {
      using TT = TrackerTraits;
      using QualityCuts = ::pixelTrack::QualityCutsT<TT>;

      ParamsT(AlgoParams const& commonCuts,
              QualityCuts const& qualityCuts,
              std::string const& tripletDumpIteration = std::string())
          : algoParams_(commonCuts), tripletDumpIteration_(tripletDumpIteration), qualityCuts_(qualityCuts) {}

      // quality cuts
      const AlgoParams algoParams_;
      // Host side only: iteration whose built triplets are dumped, empty for all of them.
      const std::string tripletDumpIteration_;
      const QualityCuts qualityCuts_{5.0f, /*chi2*/ 0.9f, /* pT in Gev*/ 0.4f, /*zip in cm*/ 12.0f /*tip in cm*/};

    };  // Params Phase1

  }  // namespace caHitNtupletGenerator
  template <typename TTTraits>
  class CAHitNtupletGeneratorKernels {
  public:
    using TrackerTraits = TTTraits;

    using SimpleCell = CACell<TrackerTraits>;
    using Params = caHitNtupletGenerator::ParamsT<TrackerTraits>;
    using Counters = caHitNtupletGenerator::Counters;
    // Track qualities
    using Quality = ::pixelTrack::Quality;
    using QualityCuts = ::pixelTrack::QualityCutsT<TrackerTraits>;

    // Histograms

    using PhiBinner = caStructures::PhiBinnerT<TrackerTraits>;  //the traits here define the number of layer/histograms
    using PhiBinnerStorageType = typename PhiBinner::value_type;
    using PhiBinnerView = typename PhiBinner::View;

    using HitToTuple = caStructures::GenericContainer;
    using HitContainer = caStructures::SequentialContainer;
    using TupleMultiplicity = caStructures::GenericContainer;
    using HitToCell = caStructures::GenericContainer;
    using CellToCell = caStructures::GenericContainer;
    using CellToTrack = caStructures::GenericContainer;

    using GenericContainer = caStructures::GenericContainer;
    using GenericContainerStorage = typename GenericContainer::value_type;
    using GenericContainerView = typename GenericContainer::View;
    using DeviceGenericContainerBuffer = std::optional<cms::alpakatools::device_buffer<Device, GenericContainer>>;
    using DeviceGenericStorageBuffer =
        std::optional<cms::alpakatools::device_buffer<Device, GenericContainerStorage[]>>;
    using DeviceGenericOffsetsBuffer =
        std::optional<cms::alpakatools::device_buffer<Device, GenericContainerOffsets[]>>;

    using SequentialContainer = caStructures::SequentialContainer;
    using SequentialContainerStorage = typename SequentialContainer::value_type;
    using SequentialContainerView = typename SequentialContainer::View;
    using DeviceSequentialContainerBuffer = std::optional<cms::alpakatools::device_buffer<Device, SequentialContainer>>;
    using DeviceSequentialStorageBuffer =
        std::optional<cms::alpakatools::device_buffer<Device, SequentialContainerStorage[]>>;
    using DeviceSequentialOffsetsBuffer =
        std::optional<cms::alpakatools::device_buffer<Device, SequentialContainerOffsets[]>>;

    // nOTHits extends the hit->tuple domain to nHits + nOTHits, so tagged OT extras bin at
    // nHits + otIdx and the duplicate cleaning sees them; 0 restricts the domain to nHits.
    CAHitNtupletGeneratorKernels(Params const& params,
                                 uint32_t nHits,
                                 uint32_t nOTHits,
                                 uint32_t offsetBPIX2,
                                 uint32_t nDoublets,
                                 uint32_t nTracks,
                                 uint16_t nLayers,
                                 Queue& queue);
    ~CAHitNtupletGeneratorKernels() = default;

    TupleMultiplicity const* tupleMultiplicity() const { return device_tupleMultiplicity_->data(); }
    // Prefix-scanned per-N-bin offsets of the tuple-multiplicity assoc: off[b] is the number of
    // fitted tuples with fewer than b selected hits, so bin [lo,hi] holds off[hi+1]-off[lo] tuples.
    // maxHitsOnTrack+2 valid entries; one D2H of it gives the host each bin's population.
    GenericContainerOffsets const* tupleMultiplicityOffsets() const { return device_tupleMultiplicityOffsets_->data(); }
    HitContainer const* hitContainer() const { return device_hitContainer_->data(); }
    HitToCell const* hitToCell() const { return device_hitToCell_->data(); }

    uint32_t* pipelineCountersPtr() {
#ifdef CA_PIPELINE_COUNTERS
      return device_pipelineCounters_->data();
#else
      return nullptr;
#endif
    }
    HitToTuple const* hitToTuple() const { return device_hitToTuple_->data(); }
    CellToCell const* cellToCell() const { return device_cellToNeighbors_->data(); }
    CellToTrack const* cellToTrack() const { return device_cellToTracks_->data(); }

#ifdef CA_TRIPLET_DUMP
    // Lets the generator move the dump buffer out before this object is destroyed.
    std::optional<TripletDumpSoACollection>& tripletDumpBuffer() { return device_tripletDump_; }
#endif

    void prepareHits(const HitsConstView& hh,
                     const HitModulesConstView& mm,
                     const ::reco::CALayersSoAConstView& ll,
                     Queue& queue);

    // Per-stream overflow accumulator (8 uint32, layout in Kernel_overflowSentinel), owned by
    // CAHitNtupletGenerator; when set, classifyTuples launches the sentinel into it.
    uint32_t* ovfAccum_ = nullptr;

    void launchKernels(const HitsConstView& hh,
                       uint32_t offsetBPIX2,
                       uint16_t nLayers,
                       TkSoABlocksView& view,
                       const ::reco::CAGraphSoAConstView& cc,
                       const ::reco::CATripletCutsSoAConstView& tripletCuts,
                       const ::reco::CANtupletCutsSoAConstView& ntupletCuts,
                       Queue& queue);

    // otView is the raw OT-rechit position view, null for merged hits only: the classifier feature
    // walk resolves tagged OT extras through it, and the hit->tuple pass bins them at nHits + otIdx.
    void classifyTuples(const HitsConstView& hh,
                        TkSoAView& track_view,
                        Queue& queue,
                        ::reco::OTRecHitsConstView const* otView = nullptr);

    // Returns the doublet count read back by the count-only pass, or 0 when that pass did not run.
    // It is an upper bound on the fill pass, which skips the hit->cell capacity term, so the caller
    // can size allocateAfterDoublets from it without a second readback.
    uint32_t buildDoublets(const HitsConstView& hh,
                           const ::reco::CAGraphSoAConstView& cc,
                           const ::reco::CALayersSoAConstView& ll,
                           const ::reco::CADoubletCutsSoAConstView& doubletCuts,
                           uint32_t offsetBPIX2,
                           const MapToHitConstView& maskView,
                           Queue& queue);

    // Allocate the cell-derived buffers (cell->cell, cell->track, triplet and track-cell SoA) from
    // the actual number of doublets rather than the maxDoublets cap. Called after buildDoublets.
    void allocateAfterDoublets(uint32_t nCells, Queue& queue);

    // Element count of the cells+offsets arena in SimpleCell units, 0 when the arena is not worth
    // it for this event's cap.
    static uint32_t cellArenaExtent(uint32_t cellBound, uint32_t maxDoublets);

    // Allocate the hit->track storage from the actual number of hits in tracks, known after
    // launchKernels' finalizeBulk. Called before classifyTuples.
    void allocateAfterNtuplets(uint32_t nHitsInTracks, Queue& queue);

    // Return the build-only scratch to the caching allocator at the end of launchKernels.
    // Stream-ordered and therefore sync-free: the allocator re-issues a block only once the event
    // it recorded on the queue has completed, so already enqueued kernels keep valid pointers.
    // A no-op in the diagnostic builds, whose reports read these buffers after launchKernels.
    void releaseBuildScratch();

    // Read back the number of doublets to the host. One sync.
    uint32_t readbackNCells(Queue& queue);

    // Enqueue an async D2H of the 5-word extraStorage counter block into a caller-owned pinned
    // buffer. No wait is issued: the caller must synchronize its queue before reading. Word [0] is
    // the tuple AtomicPairCounter (low half = tuple count, high half = hits-in-tracks total).
    void enqueueCountsReadback(cms::alpakatools::host_buffer<cms::alpakatools::AtomicPairCounter::DoubleWord[]>& dst,
                               Queue& queue);

    // Read the whole 5-word extraStorage back to the host in one sync. Word [0] is the
    // AtomicPairCounter (first = tuple count, second = hits-in-tracks total), word [1] is unused,
    // words [2..4] are nCells, nTriplets and nCellTracks.
    void readbackAllCounts(Queue& queue,
                           uint32_t& nTracks,
                           uint32_t& nHitsInTracks,
                           uint32_t& nCells,
                           uint32_t& nTriplets,
                           uint32_t& nCellTracks);

    // SoA element counts chosen in allocateAfterDoublets: the cell->cell and cell->track edge-list
    // capacities.
    uint32_t tripletsN() const { return tripletsN_; }
    uint32_t tracksCellsN() const { return tracksCellsN_; }

    static void printCounters();

  private:
    // params
    Params const& m_params;
    std::optional<cms::alpakatools::device_buffer<Device, Counters>> counters_;

    // Hits->Track
    DeviceGenericContainerBuffer device_hitToTuple_;
    DeviceGenericStorageBuffer device_hitToTupleStorage_;
    DeviceGenericOffsetsBuffer device_hitToTupleOffsets_;
    GenericContainerView device_hitToTupleView_;

    // (Outer) Hits-> Cells
    DeviceGenericContainerBuffer device_hitToCell_;
    DeviceGenericStorageBuffer device_hitToCellStorage_;
    DeviceGenericOffsetsBuffer device_hitToCellOffsets_;
    GenericContainerView device_hitToCellView_;

    // Hits Phi Binner
    std::optional<cms::alpakatools::device_buffer<Device, PhiBinner>> device_hitPhiHist_;
    std::optional<cms::alpakatools::device_buffer<Device, PhiBinnerStorageType[]>> device_phiBinnerStorage_;
    PhiBinnerView device_hitPhiView_;
    std::optional<cms::alpakatools::device_buffer<Device, hindex_type[]>> device_layerStarts_;

    // Scratch int32 mirror of the track quality used by the duplicate-removal kernels to make them
    // order/backend independent. The cell-parallel fast remover accumulates demotions here via atomicMin
    // and copies them back; the track-parallel hit-based removers instead use it as a frozen read-only
    // quality snapshot while each thread writes its own track's quality directly. See the helper kernels
    // Kernel_snapshotQuality / Kernel_applyQuality in CAHitNtupletGeneratorKernelsImpl.h
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> device_qualityScratch_;

    // Cells-> Neighbor Cells
    DeviceGenericContainerBuffer device_cellToNeighbors_;
    DeviceGenericStorageBuffer device_cellToNeighborsStorage_;
    DeviceGenericOffsetsBuffer device_cellToNeighborsOffsets_;
    GenericContainerView device_cellToNeighborsView_;

    // Cells-> Tracks
    DeviceGenericContainerBuffer device_cellToTracks_;
    DeviceGenericStorageBuffer device_cellToTracksStorage_;
    DeviceGenericOffsetsBuffer device_cellToTracksOffsets_;
    GenericContainerView device_cellToTracksView_;

    // Tracks->Hits
    DeviceSequentialContainerBuffer device_hitContainer_;
    DeviceGenericStorageBuffer device_hitContainerStorage_;
    DeviceSequentialOffsetsBuffer device_hitContainerOffsets_;
    SequentialContainerView device_hitContainerView_;

    // No.Hits -> Track (Multiplicity)
    DeviceGenericContainerBuffer device_tupleMultiplicity_;
    DeviceGenericStorageBuffer device_tupleMultiplicityStorage_;
    DeviceGenericOffsetsBuffer device_tupleMultiplicityOffsets_;
    GenericContainerView device_tupleMultiplicityView_;

    std::optional<cms::alpakatools::device_buffer<Device, SimpleCell[]>> device_simpleCells_;

    // One allocation for two same-lifetime cell-scale buffers: the cells (12 B each) and the
    // cell->track offsets (4 B per cell + 1). The caching allocator rounds every request up to a
    // power of two, so one shared buffer saves a bin whenever 12*D + 4*D still fits the bin of
    // 12*D alone; otherwise the two are allocated separately. In the shared layout
    // device_simpleCells_ is the arena: its first maxDoublets entries are the cells and its
    // 256 B aligned tail the offsets, whose pointer is passed on below. Null: allocate normally.
    caStructures::GenericContainerOffsets* arenaCellToTracksOffsets_ = nullptr;

    // Second layout (see chooseCellLayout): the three cell-indexed uint32 arrays, the hit->cell
    // storage and the two cell-keyed offsets, share one buffer instead, since 3 x 4*D packs into
    // one bin whenever 12*D still fits it. Here releaseBuildScratch() keeps the first two alive
    // together with the arena.
    std::optional<cms::alpakatools::device_buffer<Device, caStructures::GenericContainerStorage[]>>
        device_cellIndexArena_;
    caStructures::GenericContainerStorage* arenaHitToCellStorage_ = nullptr;
    caStructures::GenericContainerOffsets* arenaCellToNeighborsOffsets_ = nullptr;

    // Which of the three packings the constructor chose for this event.
    enum class CellLayout { kSeparate, kCellsWithTrackOffsets, kThreeCellIndexArrays };
    static CellLayout chooseCellLayout(uint32_t cellBound, uint32_t maxDoublets);

    std::optional<cms::alpakatools::device_buffer<Device, cms::alpakatools::AtomicPairCounter::DoubleWord[]>>
        device_extraStorage_;
    cms::alpakatools::AtomicPairCounter* device_hitTuple_apc_;
    std::optional<cms::alpakatools::device_view<Device, uint32_t>> device_nCells_;
    std::optional<cms::alpakatools::device_view<Device, uint32_t>> device_nTriplets_;
    std::optional<cms::alpakatools::device_view<Device, uint32_t>> device_nCellTracks_;
    // {offset bound, first dropped tuple} of the content-overflow repair.
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> device_tupleClampBound_;

    std::optional<CAPairSoACollection> deviceTriplets_;
    std::optional<CAPairSoACollection> deviceTracksCells_;

    // Per-built-triplet capture, allocated (sized like deviceTriplets_) and written by
    // Kernel_connect only under CA_TRIPLET_DUMP, then emitted as the 'Triplet' nano table.
    std::optional<TripletDumpSoACollection> device_tripletDump_;

#ifdef CA_PIPELINE_COUNTERS
    // Pipeline stage counters for diagnostic funnel
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> device_pipelineCounters_;
#endif

    // this could be inferred from the above buffers
    // but seems cleaner to have a dedicate variable
    uint32_t maxNumberOfDoublets_;

    // Work-division bound for the cell-loop kernels: the exact cell count when it was read back,
    // the doublet capacity otherwise. Any extent >= the true count visits the same indices in the
    // same order, so both give identical output. Capacity checks keep using maxNumberOfDoublets_.
    uint32_t launchCells_ = 0;

    // OT-hit domain size for the hit->tuple container; 0 means merged hits only.
    uint32_t nOTHits_ = 0;

    // SoA element counts chosen in allocateAfterDoublets, kept for the allocation report.
    uint32_t tripletsN_ = 0;
    uint32_t tracksCellsN_ = 0;
  };

//   class CAHitMaskingAndMergerKernels {
//   public:
//     CAHitMaskingAndMergerKernels() = default;
//     ~CAHitMaskingAndMergerKernels() = default;

//     CAHitMaskingAndMergerKernels(const CAHitMaskingAndMergerKernels&) = delete;
//     CAHitMaskingAndMergerKernels(CAHitMaskingAndMergerKernels&&) = delete;
//     CAHitMaskingAndMergerKernels& operator=(const CAHitMaskingAndMergerKernels&) = delete;
//     CAHitMaskingAndMergerKernels& operator=(CAHitMaskingAndMergerKernels&&) = delete;

//     void updateMasking(::reco::TrackingRecHitsMaskingView& mask_view,
//                        const ::reco::TrackSoAConstView& trackd_view,
//                        const ::reco::TrackHitSoAConstView& trackhitd_view,
//                        const pixelTrack::Quality minQuality,
//                        uint32_t const& iterationIndex,
//                        Queue& queue,
//                        bool maskAttachedHits = false);

//     void filterTracks(::reco::TrackSoAView& track_view,
//                       ::reco::TrackHitSoAView& trackHit_view,
//                       const ::reco::TrackSoAConstView& inpTrack_view,
//                       const ::reco::TrackHitSoAConstView& inpTrackHit_view,
//                       const pixelTrack::Quality minQuality,
//                       const double matchFraction,
//                       Queue& queue,
//                       const int32_t* loserOf = nullptr,
//                       const int32_t* isLoser = nullptr,
//                       const bool twinMergeRefit = false,
//                       const bool refitAllTracks = false,
//                       int32_t* unitedMaskOut = nullptr,
//                       // Per-track arm carried through the compaction, input order in and merged-SoA
//                       // order out; both null when no arm is tracked.
//                       const uint8_t* pocketArmIn = nullptr,
//                       uint8_t* pocketArmIdOut = nullptr);

//     // Strict cross-arm twin merge. Fills bestTwin/loserOf/isLoser device arrays
//     // (all sized nTracks). isLoser is zero-initialised here before use.
//     void twinMerge(const ::reco::TrackSoAConstView& inpTrack_view,
//                    const ::reco::TrackHitSoAConstView& inpTrackHit_view,
//                    const int32_t* armOfTrack,
//                    const float twinDEta,
//                    const float twinDPhi,
//                    const int twinMinShared,
//                    const bool twinTier2,
//                    const float twinDEta2,
//                    const float twinDPhi2,
//                    const float twinNSigma2,
//                    const int twinMinSharedFwd,
//                    const pixelTrack::Quality minQuality,
//                    int32_t* bestTwin,
//                    int32_t* loserOf,
//                    int32_t* isLoser,
//                    int const& nTracks,
//                    Queue& queue);

//     // Mark duplicate losers over the refined merged tracks by shared-hit co-occurrence pairing and
//     // a covariance-scaled 3-parameter gate (nSigma2 from kDedupNSigma2Default unless overridden),
//     // plus the 0-shared forward fallback, then compact into out_view. nHits and nOTHits size the
//     // hit-id key space [0, nHits+nOTHits). confirm carries the GBL-refit inputs for the
//     // merge-or-keep-both check on fallback pairs; null makes the fallback drop its loser outright.
//     void finalDedup(::reco::TrackSoAView& out_view,
//                     ::reco::TrackHitSoAView& outHit_view,
//                     const ::reco::TrackSoAConstView& tracks_view,
//                     const ::reco::TrackHitSoAConstView& trackHit_view,
//                     int const& nTracksCap,
//                     uint32_t nHits,
//                     uint32_t nOTHits,
//                     Queue& queue,
//                     const MergerDedupConfirmInputs* confirm = nullptr);
//     // Device-side gather/compact: reads each input's nTracks() and hitOffsets() on device (no host
//     // readback) and packs the track and trackHits columns of all inputs into a dense merged layout,
//     // writing the merged nTracks, the shifted hitOffsets and the per-track arm labels on device.
//     // nInputs must be <= 2.
//     void mergeGather(::reco::TrackSoAView& outTrack_view,
//                      ::reco::TrackHitSoAView& outHit_view,
//                      const ::reco::TrackSoAConstView& inp0Track_view,
//                      const ::reco::TrackHitSoAConstView& inp0Hit_view,
//                      const ::reco::TrackSoAConstView& inp1Track_view,
//                      const ::reco::TrackHitSoAConstView& inp1Hit_view,
//                      int nInputs,
//                      int32_t* armBuf,
//                      const int32_t arm0,
//                      const int32_t arm1,
//                      Queue& queue);
//   };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernels_h
