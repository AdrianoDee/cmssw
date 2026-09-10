// C++ headers
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

// Alpaka headers
#include <alpaka/alpaka.hpp>

// CMSSW headers
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

// local headers
#include "CAFishbone.h"
#include "CAHitNtupletGeneratorKernels.h"
#include "CAHitNtupletGeneratorKernelsImpl.h"
#include "HelixFit.h"

//#define GPU_DEBUG
// #define NTUPLE_DEBUG
//#define CA_STATS

namespace ALPAKA_ACCELERATOR_NAMESPACE {

// The sizing accumulator serves both the CA_STATS report and the GPU_DEBUG allocation report.
#if defined(CA_STATS) || defined(GPU_DEBUG)
  // Thread- and event-safe accumulator for sizing-parameter recommendations.
  // Fixed-avg parameters (size = max_keys * avg):
  //   index 0 -> avgHitsPerTrack
  //   index 1 -> avgCellsPerHit
  //   index 2 -> avgCellsPerCell
  //   index 3 -> avgTracksPerCell
  // Scaling-with-nHits parameters (size = slope * nHits):
  //   index 0 -> maxNumberOfDoublets (slope = nCells  / nHits)
  //   index 1 -> maxNumberOfTuples   (slope = nTuples / nHits)
  // maxNHits is tracked too, so f(nHits) can be evaluated at the observed worst case.
  namespace {
    struct SizingAccumulator {
      std::mutex mtx;
      uint64_t n = 0;
      double maxReq[4] = {0., 0., 0., 0.};
      double sumReq[4] = {0., 0., 0., 0.};
      double maxPhy[4] = {0., 0., 0., 0.};
      double sumPhy[4] = {0., 0., 0., 0.};
      double maxRatio[2] = {0., 0.};
      double sumRatio[2] = {0., 0.};
      uint32_t maxNHits = 0u;
      uint64_t allocN = 0;
      uint64_t allocBytesSum = 0;
      uint64_t allocBytesMax = 0;
    };
    // One slot per pixelTrack::Iteration so several CA instances running
    // in the same job do not pollute each other's recommendations.
    inline SizingAccumulator &sizingAccumulator(::pixelTrack::Iteration it) {
      static std::array<SizingAccumulator, ::pixelTrack::iterationSize> s;
      return s[static_cast<uint8_t>(it)];
    }
  }  // namespace
#endif  // CA_STATS || GPU_DEBUG

  // Inclusive prefix scan of a OneToManyAssoc's offsets through the multi-level iterativePrefixScan.
  // The generic OneToManyAssoc::launchFinalize uses one multiBlockPrefixScan launch whose dynamic
  // shared memory grows with the number of 1024-entry blocks and so exceeds the 48 kB launch limit
  // for the large CA containers. Same numbers; on the serial backend a plain sequential scan.
  template <typename TView>
  inline void finalizeAssocOffsets(TView const &view, Queue &queue) {
    cms::alpakatools::iterativePrefixScan<Acc1D>(view.offStorage, view.offStorage, view.offSize, queue);
  }

  // Sizing rule of the cells + cell->track-offsets arena (see the member declaration in the header).
  // Returns the arena's extent in SimpleCell units, or 0 when one allocation is not cheaper than two.
  template <typename TrackerTraits>
  uint32_t CAHitNtupletGeneratorKernels<TrackerTraits>::cellArenaExtent(uint32_t cellBound, uint32_t maxDoublets) {
    // The allocator's bin: the smallest power of two >= bytes, floored at its 256 B minimum bin
    // (binGrowth 2, minBin 8, maxBin 30).
    auto bin = [](std::size_t bytes) {
      std::size_t b = 256;
      while (b < bytes)
        b *= 2;
      return b;
    };
    constexpr std::size_t kCell = sizeof(SimpleCell);
    // 64 cells = 64*12 = 768 B, a multiple of 256, so the offsets region starts 256 B aligned
    // whatever the cap is.
    const std::size_t cells = ((std::size_t(cellBound) + 63u) / 64u) * 64u;
    const std::size_t offsetBytes = (std::size_t(maxDoublets) + 1u) * sizeof(GenericContainerOffsets);
    const std::size_t extra = (offsetBytes + kCell - 1u) / kCell;
    const std::size_t separate = bin(std::size_t(cellBound) * kCell) + bin(offsetBytes);
    const std::size_t together = bin((cells + extra) * kCell);
    return (together < separate) ? uint32_t(cells + extra) : 0u;
  }

  // Which of the three packings of the four cell-scale buffers is cheapest for this event's bounds.
  // The allocator's bins are powers of two, so the winner depends on where 12*cellBound falls inside
  // its bin; all three are evaluated and the smallest total is taken. The minimum is over the bytes
  // allocated, not the bytes resident at every instant: kThreeCellIndexArrays shares a buffer with
  // the cell->track offsets, which live to the end of the event, so releaseBuildScratch() cannot
  // hand those two back after launchKernels.
  template <typename TrackerTraits>
  typename CAHitNtupletGeneratorKernels<TrackerTraits>::CellLayout
  CAHitNtupletGeneratorKernels<TrackerTraits>::chooseCellLayout(uint32_t cellBound, uint32_t maxDoublets) {
    auto bin = [](std::size_t bytes) {
      std::size_t b = 256;
      while (b < bytes)
        b *= 2;
      return b;
    };
    auto pad = [](std::size_t b) { return (b + 255u) / 256u * 256u; };
    const std::size_t cells = std::size_t(cellBound) * sizeof(SimpleCell);
    const std::size_t store = std::size_t(std::max(cellBound, 1u)) * sizeof(GenericContainerStorage);
    const std::size_t off = (std::size_t(maxDoublets) + 1u) * sizeof(GenericContainerOffsets);
    const std::size_t sep = bin(cells) + bin(store) + 2u * bin(off);
    const std::size_t a = bin(pad(cells) + off) + bin(store) + bin(off);
    const std::size_t b3 = bin(cells) + bin(pad(store) + pad(off) + off);
    if (b3 < sep && b3 <= a)
      return CellLayout::kThreeCellIndexArrays;
    if (a < sep)
      return CellLayout::kCellsWithTrackOffsets;
    return CellLayout::kSeparate;
  }

  template <typename TrackerTraits>
  CAHitNtupletGeneratorKernels<TrackerTraits>::CAHitNtupletGeneratorKernels(Params const &params,
                                                                            uint32_t nHits,
                                                                            uint32_t nOTHits,
                                                                            uint32_t offsetBPIX2,
                                                                            uint32_t maxDoublets,
                                                                            uint32_t maxTuples,
                                                                            uint16_t nLayers,
                                                                            Queue &queue)
      : m_params(params), nOTHits_(nOTHits) {
    //////////////////////////////////////////////////////////
    // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
    //////////////////////////////////////////////////////////

    counters_ = cms::alpakatools::make_device_buffer<Counters>(queue);
    // Here we define the OneToMany maps and the histograms
    // allocating the buffers and defining the views.
    // For each map/histo, we need:
    // - a buffer for the offsets sized as the number of ones + 1
    //   (with the last bin holding the total number of ones)
    // - a buffer fot the content/storage itself sized as the number of many

    auto const &algoParams = m_params.algoParams_;
    // Allocation strategy (see fillDescriptions). When both delayAllocations_ and countDoubletsFirst_
    // are false every buffer is allocated here with no device->host synchronization
    // Defer cell-derived + hit->track buffers
    const bool delay = algoParams.delayAllocations_;
    // defer simpleCells/hitToCellStorage and allocate to the actual number of doublets produced
    const bool countFirst = algoParams.countDoubletsFirst_;

    // Cell bound, distinct from maxDoublets: it bounds the cell array and the hit->cell association
    // (one entry per cell), while maxDoublets also sets the content capacities and the key-space
    // extent computed in allocateAfterDoublets.
    // It is capped at an allocator bin edge: at the largest maxDoublets the 12 B per cell array would
    // be served from the next power-of-two bin although it fits the smaller one in almost every event,
    // and the allocator never gives a block back, so one such event would pin the larger block for the
    // rest of the job. The content capacities are untouched, so the ratios the truncation guards
    // compare against do not move.
    static constexpr uint32_t kMaxDoubletsForCellBin = (64u * 1024u * 1024u) / uint32_t(sizeof(SimpleCell));
    const uint32_t cellBound = std::min(maxDoublets, kMaxDoubletsForCellBin);
    // Packing of the four cell-scale buffers, evaluated per event (see chooseCellLayout). Only the
    // up-front allocation mode packs: the exact-allocation paths size these buffers from counts that do
    // not exist yet at this point.
    const CellLayout cellLayout =
        (!delay && !countFirst) ? chooseCellLayout(cellBound, maxDoublets) : CellLayout::kSeparate;
    if (cellLayout == CellLayout::kThreeCellIndexArrays) {
      // hit->cell storage, then the two cell-keyed offsets, each 256 B aligned inside one buffer.
      const std::size_t nStore = ((std::size_t(std::max(cellBound, 1u)) + 63u) / 64u) * 64u;
      const std::size_t nOff = ((std::size_t(maxDoublets) + 1u + 63u) / 64u) * 64u;
      device_cellIndexArena_ = cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(
          queue, uint32_t(nStore + nOff + std::size_t(maxDoublets) + 1u));
      arenaHitToCellStorage_ = device_cellIndexArena_->data();
      arenaCellToNeighborsOffsets_ =
          reinterpret_cast<GenericContainerOffsets *>(device_cellIndexArena_->data() + nStore);
      arenaCellToTracksOffsets_ =
          reinterpret_cast<GenericContainerOffsets *>(device_cellIndexArena_->data() + nStore + nOff);
    }
    uint32_t outerHits =
        nHits - offsetBPIX2;  // the number of hits that may be used as outer hits for a cell (so not on bpix1)

    // Sizes the per-track hit storage (hitContainer/hitToTuple). Same expression as the output
    // trackHits SoA in CAHitNtupletGenerator::beginTuplesAsync, so the two capacities match.
    uint32_t nHitsToTracks = caHitNtupletGenerator::nHitRowsForTuples(maxTuples, algoParams.avgHitsPerTrack_);

#ifdef GPU_DEBUG
    std::cout << "Allocation for tuple building with: " << std::endl;
    std::cout << "- nHits          = " << nHits << std::endl;
    std::cout << "- outerHits      = " << outerHits << std::endl;
    std::cout << "- maxDoublets    = " << maxDoublets << std::endl;
    std::cout << "- maxTracks      = " << maxTuples << std::endl;
    std::cout << "- nHitsToTracks  = " << nHitsToTracks << std::endl;
#endif

    // Hits -> Track
    // The handle and the per-hit offsets (key space) are always allocated here. The storage holds one
    // entry per hit-in-track: with delayAllocations_ it is sized from the actual hits-in-tracks count
    // in allocateAfterNtuplets, otherwise here at the nHitsToTracks safety cap.
    // A non-zero nOTHits extends the hit->tuple domain to nHits + nOTHits, so tagged OT extras bin at
    // nHits + otIdx and the duplicate cleaning accounts for them.
    const uint32_t hitToTupleKeys = nHits + nOTHits_;
    device_hitToTuple_ = cms::alpakatools::make_device_buffer<GenericContainer>(queue);
    device_hitToTupleOffsets_ =
        cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, hitToTupleKeys + 1);
    if (!delay) {
      device_hitToTupleStorage_ = cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, nHitsToTracks);
      device_hitToTupleView_ = {device_hitToTuple_->data(),
                                device_hitToTupleOffsets_->data(),
                                device_hitToTupleStorage_->data(),
                                hitToTupleKeys + 1,
                                nHitsToTracks};

      HitToTuple::template launchZero<Acc1D>(device_hitToTupleView_, queue);
    }

    // (Outer) Hits-> Cells
    // The storage holds exactly one entry per cell: with countDoubletsFirst it is sized to the exact
    // nCells in buildDoublets, otherwise to the maxDoublets cap here, the same cap that bounds the
    // cell array itself and that the doublet finder enforces when it appends. A second, independent
    // constant would let this container truncate while maxDoublets does not: one entry per cell, one
    // bound. The handle and per-outer-hit offsets (key space) are always allocated.
    device_hitToCell_ = cms::alpakatools::make_device_buffer<GenericContainer>(queue);
    device_hitToCellOffsets_ = cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, outerHits + 1);
    if (!countFirst) {
      uint32_t nHitsToCells = std::max(cellBound, 1u);
      GenericContainerStorage *hitToCellStorage = arenaHitToCellStorage_;
      if (hitToCellStorage == nullptr) {
        device_hitToCellStorage_ = cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, nHitsToCells);
        hitToCellStorage = device_hitToCellStorage_->data();
      }
      device_hitToCellView_ = {
          device_hitToCell_->data(), device_hitToCellOffsets_->data(), hitToCellStorage, outerHits + 1, nHitsToCells};

      HitToCell::template launchZero<Acc1D>(device_hitToCellView_, queue);
    }

    // Hits Phi Histograms: one histogram per layer
    device_hitPhiHist_ = cms::alpakatools::make_device_buffer<PhiBinner>(queue);
    device_phiBinnerStorage_ = cms::alpakatools::make_device_buffer<hindex_type[]>(queue, nHits);
    device_hitPhiView_ = {
        device_hitPhiHist_->data(), nullptr, device_phiBinnerStorage_->data(), cms::alpakatools::kDynamicSize, nHits};
    // This will hold where each layer starts in the hit soa
    device_layerStarts_ = cms::alpakatools::make_device_buffer<hindex_type[]>(queue, nLayers + 1);

    // Scratch quality mirror used by the (order-independent) duplicate-removal kernels
    device_qualityScratch_ = cms::alpakatools::make_device_buffer<int32_t[]>(queue, maxTuples);

    // Cell array, allocated ahead of allocateAfterDoublets because in the up-front allocation mode it
    // doubles as the arena carrying the cell->track offsets. With countDoubletsFirst it is instead
    // sized to the exact nCells in buildDoublets.
    if (!countFirst) {
      const uint32_t arenaExtent =
          (cellLayout == CellLayout::kCellsWithTrackOffsets) ? cellArenaExtent(cellBound, maxDoublets) : 0u;
      device_simpleCells_ =
          cms::alpakatools::make_device_buffer<SimpleCell[]>(queue, arenaExtent ? arenaExtent : cellBound);
      if (arenaExtent) {
        const std::size_t cellsPadded = ((std::size_t(cellBound) + 63u) / 64u) * 64u;
        arenaCellToTracksOffsets_ =
            reinterpret_cast<GenericContainerOffsets *>(device_simpleCells_->data() + cellsPadded);
      }
    }

    // Cell -> neighbour cells and cell -> tracks (and the triplet/track-cell SoA) are sized from the
    // number of doublets: with delayAllocations_ from the actual nCells in allocateAfterDoublets,
    // otherwise here at the maxDoublets safety cap through the same function.
    if (!delay) {
      allocateAfterDoublets(maxDoublets, queue);
    }

    // Track -> Hits
    // - This is a OneToManyAssocSequential since each bin is filled
    //   in one go: all the hits forming a track are pushed together.
    device_hitContainer_ = cms::alpakatools::make_device_buffer<SequentialContainer>(queue);
    device_hitContainerStorage_ =
        cms::alpakatools::make_device_buffer<SequentialContainerStorage[]>(queue, nHitsToTracks);
    device_hitContainerOffsets_ =
        cms::alpakatools::make_device_buffer<SequentialContainerOffsets[]>(queue, maxTuples + 1);
    device_hitContainerView_ = {device_hitContainer_->data(),
                                device_hitContainerOffsets_->data(),
                                device_hitContainerStorage_->data(),
                                maxTuples + 1u,
                                nHitsToTracks};

    HitContainer::template launchZero<Acc1D>(device_hitContainerView_, queue);
    device_tupleClampBound_ = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, 2u);

    // No.Hits -> Track (track multiplicity)
    device_tupleMultiplicity_ = cms::alpakatools::make_device_buffer<GenericContainer>(queue);
    device_tupleMultiplicityStorage_ =
        cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, maxTuples);
    device_tupleMultiplicityOffsets_ =
        cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, TrackerTraits::maxHitsOnTrack + 2);
    device_tupleMultiplicityView_ = {
        device_tupleMultiplicity_->data(),
        device_tupleMultiplicityOffsets_->data(),
        device_tupleMultiplicityStorage_->data(),
        // this has to be +2 instead of +1 because you want all values from 0 to maxHitsOnTrack to be valid keys
        // (N+1 values) + the extra +1 for the Container definition
        TrackerTraits::maxHitsOnTrack + 2u,
        maxTuples};
    TupleMultiplicity::template launchZero<Acc1D>(device_tupleMultiplicityView_, queue);

    // device_simpleCells_ is allocated above, ahead of allocateAfterDoublets, so that it can carry the
    // cell->track offsets in its tail.
    device_extraStorage_ =
        cms::alpakatools::make_device_buffer<cms::alpakatools::AtomicPairCounter::DoubleWord[]>(queue, 5u);
    device_hitTuple_apc_ = reinterpret_cast<cms::alpakatools::AtomicPairCounter *>(device_extraStorage_->data());
    device_nCells_ =
        cms::alpakatools::make_device_view(queue, *reinterpret_cast<uint32_t *>(device_extraStorage_->data() + 2));
    device_nTriplets_ =
        cms::alpakatools::make_device_view(queue, *reinterpret_cast<uint32_t *>(device_extraStorage_->data() + 3));
    device_nCellTracks_ =
        cms::alpakatools::make_device_view(queue, *reinterpret_cast<uint32_t *>(device_extraStorage_->data() + 4));

    // deviceTriplets_ and deviceTracksCells_ are sized from nCells in allocateAfterDoublets

#ifdef CA_PIPELINE_COUNTERS
    // Pipeline stage counters for diagnostic funnel
    device_pipelineCounters_ =
        cms::alpakatools::make_device_buffer<uint32_t[]>(queue, caHitNtupletGenerator::kTotalCounters);
    alpaka::memset(queue, *device_pipelineCounters_, 0);
#endif

    //TODO: if doStats?
    alpaka::memset(queue, *counters_, 0);

    alpaka::memset(queue, *device_nCells_, 0);
    alpaka::memset(queue, *device_nTriplets_, 0);
    alpaka::memset(queue, *device_nCellTracks_, 0);

    maxNumberOfDoublets_ = cellBound;
    launchCells_ = cellBound;

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Allocations for CAHitNtupletGeneratorKernels: done!" << std::endl;
#endif
  }

  template <typename TrackerTraits>
  uint32_t CAHitNtupletGeneratorKernels<TrackerTraits>::readbackNCells(Queue &queue) {
    using DW = cms::alpakatools::AtomicPairCounter::DoubleWord;
    auto h_extra = cms::alpakatools::make_host_buffer<DW[]>(queue, 5u);
    alpaka::memcpy(queue, h_extra, *this->device_extraStorage_);
    alpaka::wait(queue);  // Necessary wait: the value is returned to the host caller, which uses it to
                          // size the next launches.
    // device_nCells_ aliases extraStorage word [2].
    return static_cast<uint32_t>(alpaka::getPtrNative(h_extra)[2]);
  }

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::enqueueCountsReadback(
      cms::alpakatools::host_buffer<cms::alpakatools::AtomicPairCounter::DoubleWord[]> &dst, Queue &queue) {
    // Async copy only -- the caller reads dst after its queue has been synchronized (the CA
    // producer's acquire->produce seam), so no host wait is needed here.
    alpaka::memcpy(queue, dst, *this->device_extraStorage_);
  }
  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::readbackAllCounts(Queue &queue,
                                                                      uint32_t &nTracks,
                                                                      uint32_t &nHitsInTracks,
                                                                      uint32_t &nCells,
                                                                      uint32_t &nTriplets,
                                                                      uint32_t &nCellTracks) {
    using DW = cms::alpakatools::AtomicPairCounter::DoubleWord;
    auto h_extra = cms::alpakatools::make_host_buffer<DW[]>(queue, 5u);
    alpaka::memcpy(queue, h_extra, *this->device_extraStorage_);
    alpaka::wait(queue);
    // Word [0] is the AtomicPairCounter: its low half is `first` (the tuple count) and its high
    // half is `second` (the hits-in-tracks total), the same halves Kernel_overflowSentinel reads.
    // Taken by position, not by size: the two are only ordered while every tuple has >= 2 hits.
    const uint64_t apc_raw = static_cast<uint64_t>(alpaka::getPtrNative(h_extra)[0]);
    nTracks = static_cast<uint32_t>(apc_raw & 0xFFFFFFFFull);
    nHitsInTracks = static_cast<uint32_t>(apc_raw >> 32);
    // word [2] = nCells, [3] = nTriplets, [4] = nCellTracks
    nCells = static_cast<uint32_t>(alpaka::getPtrNative(h_extra)[2]);
    nTriplets = static_cast<uint32_t>(alpaka::getPtrNative(h_extra)[3]);
    nCellTracks = static_cast<uint32_t>(alpaka::getPtrNative(h_extra)[4]);
  }

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::allocateAfterDoublets(uint32_t nCells, Queue &queue) {
    auto const &algoParams = m_params.algoParams_;

    // Association-content sizes scale with the number of doublets actually found: keep the per-key
    // average multipliers but base them on nCells instead of the much larger maxDoublets cap.
    // Quiet-event floor: those averages are fitted at high occupancy, where most doublets are
    // combinatorial junk with no valid neighbour, so cells per cell is ~0.1; in a clean event nearly
    // every doublet has a genuine neighbour and the ratio rises towards 1 while nCells, and hence the
    // derived capacity, falls. The surplus would be dropped silently by the capacity guards, in an
    // order decided by who wins the atomic. The floor is the clean-event bound of at most
    // kCleanEdgesPerCell edges per cell, tracked linearly up to kQuietCellsCap and constant above it.
    constexpr uint32_t kQuietCellsCap = 512u;
    constexpr uint32_t kCleanEdgesPerCell = 64u;
    const uint32_t quietFloor = std::min(nCells, kQuietCellsCap) * kCleanEdgesPerCell;
    const uint32_t nCellsToCells = std::max({uint32_t(nCells * algoParams.avgCellsPerCell_), quietFloor, 1u});
    const uint32_t nCellsToTracks = std::max({uint32_t(nCells * algoParams.avgTracksPerCell_), quietFloor, 1u});
    const uint32_t nBins = nCells + 1u;  // one offset bin per cell (+1 for the total)

    // The two extents above have different natures. nBins, the offsets extent of both associators, is
    // a key-space extent: one word per cell plus the total, and every index tested against it is a
    // cell id, hence < nCells by construction. nCellsToCells and nCellsToTracks are content
    // capacities: the truncation guards compare against them, so entries past them are dropped
    // silently, and their per-cell ratios are calibrated against the basis in force here.
    // This function is called either once at construction with the maxDoublets cap, when no doublet
    // count exists yet, or after the doublet build with the exact count.

    // Cell -> Neighbor Cells. One bin per cell; bit 31 of each stored neighbor index
    // encodes layer-skipping (valid since nCells is well below 2^31).
    device_cellToNeighbors_ = cms::alpakatools::make_device_buffer<GenericContainer>(queue);
    device_cellToNeighborsStorage_ =
        cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, nCellsToCells);
    GenericContainerOffsets *cellToNeighborsOffsets = arenaCellToNeighborsOffsets_;
    if (cellToNeighborsOffsets == nullptr) {
      device_cellToNeighborsOffsets_ = cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, nBins);
      cellToNeighborsOffsets = device_cellToNeighborsOffsets_->data();
    }
    device_cellToNeighborsView_ = {device_cellToNeighbors_->data(),
                                   cellToNeighborsOffsets,
                                   device_cellToNeighborsStorage_->data(),
                                   nBins,
                                   nCellsToCells};
    CellToCell::template launchZero<Acc1D>(device_cellToNeighborsView_, queue);

    // Cell -> Tracks.
    device_cellToTracks_ = cms::alpakatools::make_device_buffer<GenericContainer>(queue);
    device_cellToTracksStorage_ =
        cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, nCellsToTracks);
    // The offsets live in the tail of the cell arena when the constructor set it up (up-front
    // allocation mode); otherwise they get their own allocation. Same extent, same contents either way.
    GenericContainerOffsets *cellToTracksOffsets = arenaCellToTracksOffsets_;
    if (cellToTracksOffsets == nullptr) {
      device_cellToTracksOffsets_ = cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, nBins);
      cellToTracksOffsets = device_cellToTracksOffsets_->data();
    }
    device_cellToTracksView_ = {
        device_cellToTracks_->data(), cellToTracksOffsets, device_cellToTracksStorage_->data(), nBins, nCellsToTracks};
    CellToTrack::template launchZero<Acc1D>(device_cellToTracksView_, queue);

    // Triplet (cell->cell) and track-cell (cell->track) SoA edge lists.
    tripletsN_ = nCellsToCells;
    tracksCellsN_ = nCellsToTracks;
    deviceTriplets_ = CAPairSoACollection(queue, tripletsN_);
    deviceTracksCells_ = CAPairSoACollection(queue, tracksCellsN_);
#ifdef CA_TRIPLET_DUMP
    device_tripletDump_ = TripletDumpSoACollection(queue, tripletsN_);
#endif
  }

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::allocateAfterNtuplets(uint32_t nHitsInTracks, Queue &queue) {
    // Hit -> track storage holds exactly nHitsInTracks entries, the sum over surviving tracks of their
    // hit multiplicity. With delayAllocations_ it is sized here from the real count; otherwise it was
    // allocated in the constructor, nHitsInTracks is 0 and this is a no-op.
    if (m_params.algoParams_.delayAllocations_) {
      const uint32_t nStorage = std::max(nHitsInTracks, 1u);
      const uint32_t nHits = static_cast<uint32_t>(alpaka::getExtents(*device_hitToTupleOffsets_)[0u]) - 1u;
      device_hitToTupleStorage_ = cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, nStorage);
      device_hitToTupleView_ = {device_hitToTuple_->data(),
                                device_hitToTupleOffsets_->data(),
                                device_hitToTupleStorage_->data(),
                                nHits + 1u,
                                nStorage};
      HitToTuple::template launchZero<Acc1D>(device_hitToTupleView_, queue);
    }

#ifdef GPU_DEBUG
    // Per-buffer device allocation report. Runs here, after the last (deferred)
    // allocation, so every buffer's real extent is available
    {
      const auto coutFlags = std::cout.flags();
      const auto coutPrec = std::cout.precision();

      std::size_t total = 0;
      // Array buffers report their real element count; scalar "handle" buffers (the
      // OneToMany container objects) report 1 element when allocated.
      auto extentOf = [](auto const &optBuf) -> std::size_t {
        return optBuf ? static_cast<std::size_t>(alpaka::getExtents(*optBuf)[0u]) : 0u;
      };
      auto oneIf = [](auto const &optBuf) -> std::size_t { return optBuf ? 1u : 0u; };
      auto report = [&total](const char *name, std::size_t n_elem, std::size_t elem_size) {
        const std::size_t bytes = n_elem * elem_size;
        total += bytes;
        std::cout << "  " << std::left << std::setw(46) << name << " : " << std::right << std::setw(10) << n_elem
                  << " x " << std::setw(4) << elem_size << " B = " << std::setw(12) << bytes << " B  (" << std::fixed
                  << std::setprecision(3) << std::setw(9) << (bytes / 1024.0) << " KiB)" << std::endl;
      };
      auto reportSoA = [&total](const char *name, std::size_t n_elem, std::size_t bytes) {
        total += bytes;
        std::cout << "  " << std::left << std::setw(46) << name << " : " << std::right << std::setw(10) << n_elem
                  << " elements,  SoA layout = " << std::setw(12) << bytes << " B  (" << std::fixed
                  << std::setprecision(3) << std::setw(9) << (bytes / 1024.0) << " KiB)" << std::endl;
      };

      const auto allocIterTag = ::pixelTrack::iterationName[static_cast<uint8_t>(m_params.algoParams_.iterationName_)];
      std::cout << "================== Device allocation report [" << allocIterTag
                << "] ==================" << std::endl;
      report("counters_", oneIf(counters_), sizeof(Counters));
      report("device_hitToTuple_", oneIf(device_hitToTuple_), sizeof(GenericContainer));
      report("device_hitToTupleStorage_", extentOf(device_hitToTupleStorage_), sizeof(GenericContainerStorage));
      report("device_hitToTupleOffsets_", extentOf(device_hitToTupleOffsets_), sizeof(GenericContainerOffsets));
      report("device_hitToCell_", oneIf(device_hitToCell_), sizeof(GenericContainer));
      report("device_hitToCellStorage_", extentOf(device_hitToCellStorage_), sizeof(GenericContainerStorage));
      report("device_hitToCellOffsets_", extentOf(device_hitToCellOffsets_), sizeof(GenericContainerOffsets));
      report("device_hitPhiHist_", oneIf(device_hitPhiHist_), sizeof(PhiBinner));
      report("device_phiBinnerStorage_", extentOf(device_phiBinnerStorage_), sizeof(PhiBinnerStorageType));
      report("device_layerStarts_", extentOf(device_layerStarts_), sizeof(hindex_type));
      report("device_qualityScratch_", extentOf(device_qualityScratch_), sizeof(int32_t));
      report("device_cellToNeighbors_", oneIf(device_cellToNeighbors_), sizeof(GenericContainer));
      report(
          "device_cellToNeighborsStorage_", extentOf(device_cellToNeighborsStorage_), sizeof(GenericContainerStorage));
      report(
          "device_cellToNeighborsOffsets_", extentOf(device_cellToNeighborsOffsets_), sizeof(GenericContainerOffsets));
      report("device_cellToTracks_", oneIf(device_cellToTracks_), sizeof(GenericContainer));
      report("device_cellToTracksStorage_", extentOf(device_cellToTracksStorage_), sizeof(GenericContainerStorage));
      report("device_cellToTracksOffsets_", extentOf(device_cellToTracksOffsets_), sizeof(GenericContainerOffsets));
      report("device_hitContainer_", oneIf(device_hitContainer_), sizeof(SequentialContainer));
      report("device_hitContainerStorage_", extentOf(device_hitContainerStorage_), sizeof(SequentialContainerStorage));
      report("device_hitContainerOffsets_", extentOf(device_hitContainerOffsets_), sizeof(SequentialContainerOffsets));
      report("device_tupleMultiplicity_", oneIf(device_tupleMultiplicity_), sizeof(GenericContainer));
      report("device_tupleMultiplicityStorage_",
             extentOf(device_tupleMultiplicityStorage_),
             sizeof(GenericContainerStorage));
      report("device_tupleMultiplicityOffsets_",
             extentOf(device_tupleMultiplicityOffsets_),
             sizeof(GenericContainerOffsets));
      report("device_simpleCells_", extentOf(device_simpleCells_), sizeof(SimpleCell));
      report("device_extraStorage_",
             extentOf(device_extraStorage_),
             sizeof(cms::alpakatools::AtomicPairCounter::DoubleWord));
#ifdef CA_PIPELINE_COUNTERS
      report("device_pipelineCounters_", extentOf(device_pipelineCounters_), sizeof(uint32_t));
#endif

      const std::size_t tripletsBytes = CAPairSoACollection::Layout::computeDataSize(tripletsN_);
      const std::size_t tracksCellsBytes = CAPairSoACollection::Layout::computeDataSize(tracksCellsN_);
      reportSoA("deviceTriplets_ (SoA)", tripletsN_, tripletsBytes);
      reportSoA("deviceTracksCells_ (SoA)", tracksCellsN_, tracksCellsBytes);

      std::cout << "  --------------------------------------------------------------" << std::endl;
      std::cout << "  " << std::left << std::setw(46) << "THIS EVENT'S TOTAL"
                << " : " << std::right << std::setw(12) << total << " B  (" << std::fixed << std::setprecision(3)
                << std::setw(9) << (total / (1024.0 * 1024.0)) << " MiB)" << std::endl;

      uint64_t snapAllocN = 0;
      uint64_t snapAllocSum = 0;
      uint64_t snapAllocMax = 0;
      {
        auto &agg = sizingAccumulator(this->m_params.algoParams_.iterationName_);
        std::lock_guard<std::mutex> lock(agg.mtx);
        ++agg.allocN;
        agg.allocBytesSum += static_cast<uint64_t>(total);
        if (static_cast<uint64_t>(total) > agg.allocBytesMax)
          agg.allocBytesMax = static_cast<uint64_t>(total);
        snapAllocN = agg.allocN;
        snapAllocSum = agg.allocBytesSum;
        snapAllocMax = agg.allocBytesMax;
      }
      const double meanBytes = double(snapAllocSum) / double(snapAllocN);
      std::cout << "  " << std::left << std::setw(46) << "AGGREGATE (mean over events)"
                << " : " << std::right << std::setw(12) << static_cast<uint64_t>(meanBytes) << " B  (" << std::fixed
                << std::setprecision(3) << std::setw(9) << (meanBytes / (1024.0 * 1024.0)) << " MiB)  over "
                << snapAllocN << " event" << (snapAllocN == 1u ? "" : "s") << std::endl;
      std::cout << "  " << std::left << std::setw(46) << "AGGREGATE (max over events)"
                << " : " << std::right << std::setw(12) << snapAllocMax << " B  (" << std::fixed << std::setprecision(3)
                << std::setw(9) << (snapAllocMax / (1024.0 * 1024.0)) << " MiB)" << std::endl;
      std::cout << "==============================================================" << std::endl;

      std::cout.flags(coutFlags);
      std::cout.precision(coutPrec);
    }
#endif
  }

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::prepareHits(const HitsConstView &hh,
                                                                const HitModulesConstView &mm,
                                                                const reco::CALayersSoAConstView &ll,
                                                                Queue &queue) {
    using namespace caHitNtupletGeneratorKernels;

    const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(1, ll.metadata().size() - 1);
    alpaka::exec<Acc1D>(queue, workDiv1D, SetHitsLayerStart{}, mm, ll, this->device_layerStarts_->data());

    cms::alpakatools::fillManyFromVector<Acc1D>(device_hitPhiHist_->data(),
                                                device_hitPhiView_,
                                                TrackerTraits::numberOfLayers,  // could be ll.metadata().size() - 1
                                                hh.iphi().data(),
                                                this->device_layerStarts_->data(),
                                                hh.metadata().size(),
                                                (uint32_t)256,
                                                queue);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "CAHitNtupletGeneratorKernels -> Hits prepared (layer starts and histo) -> DONE!" << std::endl;
#endif
  }

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::launchKernels(const HitsConstView &hh,
                                                                  uint32_t offsetBPIX2,
                                                                  uint16_t nLayers,
                                                                  TkSoABlocksView &view,
                                                                  const reco::CAGraphSoAConstView &cc,
                                                                  const reco::CATripletCutsSoAConstView &tripletCuts,
                                                                  const reco::CANtupletCutsSoAConstView &ntupletCuts,
                                                                  Queue &queue) {
    using namespace caPixelDoublets;
    using namespace caHitNtupletGeneratorKernels;

    auto tracks_view = view.tracks();
    auto tracks_hits_view = view.trackHits();

    uint32_t nhits = hh.metadata().size();
    auto const launchCells = this->launchCells_;
    auto const maxTuples = tracks_view.metadata().size();
#ifdef NTUPLE_DEBUG
    std::cout << "start tuple building. N hits " << nhits << std::endl;
    if (nhits < 2)
      std::cout << "too few hits " << nhits << std::endl;
#endif

    //
    // applying combinatoric cleaning such as fishbone at this stage is too expensive
    //

    const auto nthTot = 64;
    const auto stride = 4;
    auto blockSize = nthTot / stride;
    // The cell and tuple kernels here and below launch over the full container capacity. Their loops
    // are bounded by the true counts, so coverage never relies on the grid-stride wrap-around, and
    // with the extent at least the capacity each index is visited by exactly one thread, which keeps
    // the serial backend's visit order, and so its bit-exact output, independent of the capacity.
    // blockSize grows in multiples of 16, keeping blockSize*stride <= 1024, until the grid fits
    // CUDA's 65535-block limit.
    auto numberOfBlocks = cms::alpakatools::divide_up_by(launchCells, blockSize);
    while (numberOfBlocks >= 65536 && blockSize * stride < 1024) {
      blockSize += 16;
      numberOfBlocks = cms::alpakatools::divide_up_by(launchCells, blockSize);
    }
    assert(numberOfBlocks < 65536);
    assert(blockSize > 0 && 0 == blockSize % 16);
    const Vec2D blks{numberOfBlocks, 1u};
    const Vec2D thrs{blockSize, stride};
    const auto kernelConnectWorkDiv = cms::alpakatools::make_workdiv<Acc2D>(blks, thrs);

#ifdef CA_TRIPLET_DUMP
    // Iteration label and selection for the dump lines: this module's pixelTrack::Iteration is stamped
    // on every line, and a non-empty tripletDumpIteration makes the other iterations skip the dump
    // (dumpIteration = -1).
    const std::string &dumpIterSel = this->m_params.tripletDumpIteration_;
    const auto thisIter = this->m_params.algoParams_.iterationName_;
    const int dumpIteration =
        (dumpIterSel.empty() || ::pixelTrack::iterationName[uint8_t(thisIter)] == std::string_view(dumpIterSel))
            ? int(uint8_t(thisIter))
            : -1;
#endif
    // Per-iteration compile-time weight bank for the in-kernel triplet DNN: the prompt iteration
    // (promptHighPt) uses the prompt bank, every other iteration the displaced (default) bank.
    // The CA plugin is one compiled kernel shared by both iterations, so this is a runtime arg.
    const DnnBank tripletBank = (this->m_params.algoParams_.iterationName_ == ::pixelTrack::Iteration::promptHighPt)
                                    ? DnnBank::kPrompt
                                    : DnnBank::kDisplaced;
    alpaka::exec<Acc2D>(queue,
                        kernelConnectWorkDiv,
                        Kernel_connect<TrackerTraits>{},
                        this->device_hitTuple_apc_,  // needed only to be reset, ready for next kernel
                        hh,
                        cc,
                        tripletCuts,
                        this->m_params.algoParams_.useTripletDNN_,
                        this->m_params.algoParams_.tripletDNNThreshold_,
                        tripletBank,
#ifdef CA_TRIPLET_DUMP
                        this->device_tripletDump_->view(),
                        dumpIteration,
#endif
                        this->deviceTriplets_->view(),
                        this->device_simpleCells_->data(),
                        this->device_nCells_->data(),
                        this->device_nTriplets_->data(),
                        this->device_hitToCell_->data(),
                        this->device_cellToNeighbors_->data(),
                        this->pipelineCountersPtr());

    finalizeAssocOffsets(this->device_cellToNeighborsView_, queue);

#ifdef CA_TRIPLET_DUMP
    // Stamp the valid-row count into the dump SoA scalar, the number of rows Kernel_connect wrote.
    // Device-to-device copy on the same queue; the host consumer reads view().nValid() to know how
    // many of the full-capacity rows are valid.
    alpaka::memcpy(queue,
                   cms::alpakatools::make_device_view(queue, this->device_tripletDump_->view().nValid()),
                   cms::alpakatools::make_device_view(queue, *this->device_nTriplets_->data()));
#endif

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_connect -> Done!" << std::endl;
#endif

    auto threadsPerBlock = 1024;
    // Floor at one block: with useExactAllocations the launch cell count can be a handful, and
    // lrint(nCells * avgCellsPerCell) then rounds to 0, i.e. a zero-block launch that fills nothing.
    auto blocks = std::max<cms::alpakatools::Idx>(
        1u,
        cms::alpakatools::divide_up_by(std::lrint(launchCells * m_params.algoParams_.avgCellsPerCell_),
                                       threadsPerBlock));
    auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_fillGenericPair<caStructures::CAPairSoAConstView, GenericContainer>{},
                        this->deviceTriplets_->view(),
                        this->device_nTriplets_->data(),
                        this->device_cellToNeighbors_->data());

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "cellToNeighbors -> Filled!" << std::endl;
#endif

    // do not run the fishbone if there are hits only in BPIX1
    if (this->m_params.algoParams_.earlyFishbone_ and nhits > offsetBPIX2) {
      const auto nthTot = 128;
      const auto stride = 16;
      const auto blockSize = nthTot / stride;
      const auto numberOfBlocks = cms::alpakatools::divide_up_by(nhits - offsetBPIX2, blockSize);
      const Vec2D blks{numberOfBlocks, 1u};
      const Vec2D thrs{blockSize, stride};
      const auto fishboneWorkDiv = cms::alpakatools::make_workdiv<Acc2D>(blks, thrs);
      alpaka::exec<Acc2D>(queue,
                          fishboneWorkDiv,
                          CAFishbone<TrackerTraits>{},
                          hh,
                          ntupletCuts,
                          cc,
                          this->device_simpleCells_->data(),
                          this->device_hitToCell_->data(),
                          this->device_cellToTracks_->data(),
                          nhits - offsetBPIX2,
                          false,
                          this->pipelineCountersPtr(),
                          this->m_params.algoParams_.onlySameLayersFishbone_);
#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Early fishbone -> Done!" << std::endl;
#endif
    }

#ifdef CA_PIPELINE_COUNTERS
    // Count cell status after the fishbone kill phase
    {
      auto cellBlocks = cms::alpakatools::divide_up_by(launchCells, 256u);
      auto cellWorkDiv = cms::alpakatools::make_workdiv<Acc1D>(cellBlocks, 256u);
      alpaka::exec<Acc1D>(queue,
                          cellWorkDiv,
                          Kernel_pipelineCellStatus<TrackerTraits>{},
                          this->device_simpleCells_->data(),
                          this->device_nCells_->data(),
                          this->pipelineCountersPtr());
    }
#endif

    blockSize = 64;
    numberOfBlocks = cms::alpakatools::divide_up_by(launchCells, blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_find_ntuplets<TrackerTraits>{},
                        hh,
                        cc,
                        ntupletCuts,
                        tracks_view,
                        this->device_hitContainer_->data(),
                        this->device_cellToNeighbors_->data(),
                        this->device_cellToTracks_->data(),
                        this->deviceTracksCells_->view(),
                        this->device_simpleCells_->data(),
                        this->device_nCellTracks_->data(),
                        this->device_nTriplets_->data(),
                        this->device_nCells_->data(),
                        this->device_hitTuple_apc_,
                        this->m_params.algoParams_);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_find_ntuplets -> Done!" << std::endl;
#endif

#ifdef CA_PIPELINE_COUNTERS
    // Copy *nCellTracks into the pipeline counter array
    {
      auto workDiv1x1 = cms::alpakatools::make_workdiv<Acc1D>(1u, 1u);
      alpaka::exec<Acc1D>(queue,
                          workDiv1x1,
                          Kernel_pipelineCopyCellTrackCount{},
                          this->device_nCellTracks_->data(),
                          this->pipelineCountersPtr());
    }
#endif

    finalizeAssocOffsets(this->device_cellToTracksView_, queue);

    // This pass fills the cell->TRACK edge list, so its grid is sized from avgTracksPerCell_.
    // The loop inside is a grid-stride uniform_elements over the true edge count, so the block
    // count is a throughput choice only and can never truncate work.
    blocks = std::max<cms::alpakatools::Idx>(
        1u,
        cms::alpakatools::divide_up_by(std::lrint(launchCells * m_params.algoParams_.avgTracksPerCell_),
                                       threadsPerBlock));
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_fillGenericPair<caStructures::CAPairSoAConstView, GenericContainer>{},
                        this->deviceTracksCells_->view(),
                        this->device_nCellTracks_->data(),
                        this->device_cellToTracks_->data());

    if (this->m_params.algoParams_.doStats_)
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_mark_used<TrackerTraits>{},
                          this->device_simpleCells_->data(),
                          this->device_cellToTracks_->data(),
                          this->device_nCells_->data());

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

    blockSize = 128;
    numberOfBlocks = cms::alpakatools::divide_up_by(maxTuples + 1, blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        typename HitContainer::finalizeBulk{},
                        this->device_hitTuple_apc_,
                        this->device_hitContainer_->data());

    // Repair the CSR after a hit-container content overflow (see Kernel_findTupleContentOverflow):
    // overflowed tuples become empty instead of describing unwritten content and nTracks is cut
    // in front of them. The memset value 0xFFFFFFFF means "no clamp" and is what the no-overflow
    // case leaves in place.
    alpaka::memset(queue, *this->device_tupleClampBound_, 0xFF);
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_findTupleContentOverflow{},
                        this->device_hitContainer_->data(),
                        this->device_tupleClampBound_->data());
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_clampTupleContentOverflow{},
                        this->device_hitContainer_->data(),
                        this->device_tupleClampBound_->data());

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

#ifdef CA_PIPELINE_COUNTERS
    // Pipeline counter: classify n-tuplets by OT hit content.
    // Must run AFTER finalizeBulk so that foundNtuplets offsets are valid
    // for size()/begin()/end() iteration.
    {
      auto ntupBlocks = cms::alpakatools::divide_up_by(maxTuples, 128u);
      auto ntupWorkDiv = cms::alpakatools::make_workdiv<Acc1D>(ntupBlocks, 128u);
      alpaka::exec<Acc1D>(queue,
                          ntupWorkDiv,
                          Kernel_pipelineNtupletCount<TrackerTraits>{},
                          hh,
                          this->device_hitContainer_->data(),
                          this->device_hitTuple_apc_,
                          maxTuples,
                          this->pipelineCountersPtr());
    }
#endif

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_fillHitDetIndices<TrackerTraits>{},
                        tracks_view,
                        tracks_hits_view,
                        this->device_hitContainer_->data(),
                        hh,
                        this->device_hitTuple_apc_,
                        this->device_tupleClampBound_->data());

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_fillHitDetIndices   -> done!" << std::endl;
#endif

    // remove duplicates (tracks that share a doublet).
    if (this->m_params.algoParams_.doEarlyDuplicateRemover_) {
      if constexpr (std::is_same_v<pixelTopology::Phase1, TrackerTraits>) {
        // for Phase-1, the workdivision is simpler and therefore needs a separate call here
        blockSize = 64;
        numberOfBlocks = cms::alpakatools::divide_up_by(launchCells, blockSize);
        workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

        alpaka::exec<Acc1D>(queue,
                            workDiv1D,
                            Kernel_earlyDuplicateRemoverPhase1{},
                            this->device_simpleCells_->data(),
                            this->device_nCells_->data(),
                            this->device_cellToTracks_->data(),
                            tracks_view,
                            this->m_params.algoParams_.dupPassThrough_);
      } else {
        // Warp-cooperative work division: one cell per Y-thread, one warp per cell.
        const uint32_t stride = alpaka::getPreferredWarpSize(alpaka::getDev(queue));
        const uint32_t cellsPerBlock = 8;  // 8 cells * 32 threads = 256 threads/block
        // gridDim.y is capped at 65535 by CUDA. If maxDoublets needs more, uniform_elements_y
        // strides each Y-thread across the remaining cells.
        auto earlyDupRemoverNumBlocks = cms::alpakatools::divide_up_by(launchCells, cellsPerBlock);
        // Cap at Y-dim limit
        earlyDupRemoverNumBlocks = std::min<uint32_t>(earlyDupRemoverNumBlocks, 65535u);
        const Vec2D earlyDupRemoverBlocks{earlyDupRemoverNumBlocks, 1u};
        const Vec2D earlyDupRemoverThreads{cellsPerBlock, stride};
        const auto earlyDupRemoverWorkDiv =
            cms::alpakatools::make_workdiv<Acc2D>(earlyDupRemoverBlocks, earlyDupRemoverThreads);

        alpaka::exec<Acc2D>(queue,
                            earlyDupRemoverWorkDiv,
                            Kernel_earlyDuplicateRemover<TrackerTraits>{},
                            this->device_simpleCells_->data(),
                            this->device_nCells_->data(),
                            this->device_cellToTracks_->data(),
                            tracks_view,
                            this->m_params.algoParams_.dupPassThrough_);
      }

#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Kernel_earlyDuplicateRemover   -> done!" << std::endl;
#endif
    }

    blockSize = 128;
    numberOfBlocks = cms::alpakatools::divide_up_by(maxTuples, blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_countMultiplicity<TrackerTraits>{},
                        hh,
                        tracks_view,
                        this->device_hitContainer_->data(),
                        this->device_tupleMultiplicity_->data());
    finalizeAssocOffsets(this->device_tupleMultiplicityView_, queue);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_countMultiplicity   -> done!" << std::endl;
#endif

    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_fillMultiplicity<TrackerTraits>{},
                        hh,
                        tracks_view,
                        this->device_hitContainer_->data(),
                        this->device_tupleMultiplicity_->data());
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_fillMultiplicity -> done!" << std::endl;
#endif
    // do not run the fishbone if there are hits only in BPIX1
    if (this->m_params.algoParams_.lateFishbone_ and nhits > offsetBPIX2) {
      const auto nthTot = 128;
      const auto stride = 16;
      const auto blockSize = nthTot / stride;
      const auto numberOfBlocks = cms::alpakatools::divide_up_by(nhits - offsetBPIX2, blockSize);
      const Vec2D blks{numberOfBlocks, 1u};
      const Vec2D thrs{blockSize, stride};
      const auto workDiv2D = cms::alpakatools::make_workdiv<Acc2D>(blks, thrs);

      alpaka::exec<Acc2D>(queue,
                          workDiv2D,
                          CAFishbone<TrackerTraits>{},
                          hh,
                          ntupletCuts,
                          cc,
                          this->device_simpleCells_->data(),
                          this->device_hitToCell_->data(),
                          this->device_cellToTracks_->data(),
                          nhits - offsetBPIX2,
                          true,
                          this->pipelineCountersPtr(),
                          this->m_params.algoParams_.onlySameLayersFishbone_);
    }

#ifdef GPU_DEBUG
    std::cout << "lateFishbone -> done!" << std::endl;
    alpaka::wait(queue);
#endif

    // Every build-only buffer has now had its last reader enqueued. Handing them back here returns tens
    // of MiB to the caching allocator half an event early, so another stream can reuse the bin instead
    // of growing the pool.
    this->releaseBuildScratch();
  }

  // No new synchronisation: CachingAllocator::free is stream-ordered, it records an event on the queue
  // the block was allocated against and only re-issues the block once that event has completed, so the
  // kernels already enqueued keep valid pointers. For each buffer released here the last reader is a
  // kernel enqueued in launchKernels or earlier. The cell->tracks container is NOT released: the
  // duplicate remover reads it in classifyTuples. Under doStats_ the triplet and track-cell SoA stay
  // alive too, since Kernel_checkOverflows reads them on device.
  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::releaseBuildScratch() {
#if defined(GPU_DEBUG) || defined(CA_STATS)
    // Diagnostic builds keep everything to the end of the event: their reports read these buffers'
    // real extents after launchKernels has returned.
    return;
#else
    device_hitToCellStorage_.reset();
    device_hitToCellOffsets_.reset();
    device_hitToCell_.reset();

    device_cellToNeighborsStorage_.reset();
    device_cellToNeighborsOffsets_.reset();
    device_cellToNeighbors_.reset();

    device_phiBinnerStorage_.reset();
    device_hitPhiHist_.reset();

    if (!m_params.algoParams_.doStats_) {
      deviceTriplets_.reset();
      deviceTracksCells_.reset();
    }
#endif
  }

  template <typename TrackerTraits>

  void CAHitNtupletGeneratorKernels<TrackerTraits>::buildDoublets(const HitsConstView &hh,
                                                                  const ::reco::CAGraphSoAConstView &cc,
                                                                  const ::reco::CALayersSoAConstView &ll,
                                                                  uint32_t offsetBPIX2,
                                                                  const MapToHitConstView &maskPtr,
                                                                  Queue &queue) {

    using namespace caPixelDoublets;
    using namespace caHitNtupletGeneratorKernels;

    auto nhits = hh.metadata().size();
    const auto maxDoublets = this->maxNumberOfDoublets_;
#ifdef NTUPLE_DEBUG
    std::cout << "building Doublets out of " << nhits << " Hits" << std::endl;
#endif

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

    if (0 == nhits)
      return 0u;  // protect against empty events

    const int stride = 4;
    int threadsPerBlock = TrackerTraits::getDoubletsFromHistoMaxBlockSize / stride;
    int blocks = (4 * nhits + threadsPerBlock - 1) / threadsPerBlock;
    const Vec2D blks{blocks, 1u};
    const Vec2D thrs{threadsPerBlock, stride};
    const auto workDiv2D = cms::alpakatools::make_workdiv<Acc2D>(blks, thrs);

#ifdef GPU_DEBUG
    std::cout << "nActualPairs = " << cc.metadata().size() << std::endl;
    std::cout << blocks << " - " << threadsPerBlock << " - " << stride << std::endl;
#endif
    // The exact doublet count, when countDoubletsFirst_ paid a readback for it. Returned to the
    // caller so allocateAfterDoublets can be sized from it without a second D2H of the same word.
    uint32_t nCellsCounted = 0u;

    if (this->m_params.algoParams_.countDoubletsFirst_) {
      // countDoubletsFirst_ -> do a dry-run of GetDoubletsFromHisto to learn the exact nCells
      // without writing cells or the hit->cell association, then size device_simpleCells_ and device_hitToCellStorage_
      // to that count before the real fill pass
      alpaka::memset(queue, *this->device_nCells_, 0);
      alpaka::exec<Acc2D>(queue,
                          workDiv2D,
                          GetDoubletsFromHisto<TrackerTraits, /*CountOnly=*/true>{},
                          maxDoublets,
                          static_cast<SimpleCell *>(nullptr),
                          this->device_nCells_->data(),
                          hh,
                          cc,
                          ll,
                          doubletCuts,
                          this->device_layerStarts_->data(),
                          this->device_hitPhiHist_->data(),
                          this->device_hitToCell_->data(),
                          this->pipelineCountersPtr(),
                          maskPtr);

      const uint32_t nCellsFound = this->readbackNCells(queue);
      nCellsCounted = nCellsFound;
      const uint32_t nCellsStorage = std::max(nCellsFound, 1u);
      this->launchCells_ = nCellsStorage;
      const uint32_t outerHits = static_cast<uint32_t>(alpaka::getExtents(*this->device_hitToCellOffsets_)[0u]) - 1u;

      this->device_simpleCells_ = cms::alpakatools::make_device_buffer<SimpleCell[]>(queue, nCellsStorage);
      this->device_hitToCellStorage_ =
          cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, nCellsStorage);
      this->device_hitToCellView_ = {this->device_hitToCell_->data(),
                                     this->device_hitToCellOffsets_->data(),
                                     this->device_hitToCellStorage_->data(),
                                     outerHits + 1u,
                                     nCellsStorage};
      HitToCell::template launchZero<Acc1D>(this->device_hitToCellView_, queue);

      alpaka::memset(queue, *this->device_nCells_, 0);  // reset for the fill pass
    }

    // Fill the cells and the hit->cell association
    alpaka::exec<Acc2D>(queue,
                        workDiv2D,
                        GetDoubletsFromHisto<TrackerTraits>{},
                        maxDoublets,
                        this->device_simpleCells_->data(),
                        this->device_nCells_->data(),
                        hh,
                        cc,
                        ll,
                        doubletCuts,
                        this->device_layerStarts_->data(),
                        this->device_hitPhiHist_->data(),
                        this->device_hitToCell_->data(),
                        this->pipelineCountersPtr(),
                        maskPtr);

    finalizeAssocOffsets(this->device_hitToCellView_, queue);

    threadsPerBlock = 512;
    blocks = cms::alpakatools::divide_up_by(this->launchCells_, threadsPerBlock);
    auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "GetDoubletsFromHisto   -> done!" << std::endl;
#endif

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        FillDoubletsHisto<TrackerTraits>{},
                        this->device_simpleCells_->data(),
                        this->device_nCells_->data(),
                        offsetBPIX2,
                        this->device_hitToCell_->data(),
                        this->counters_->data());

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "FillDoubletsHisto   -> done!" << std::endl;
#endif

    return nCellsCounted;
  }

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::classifyTuples(const HitsConstView &hh,
                                                                   TkSoAView &tracks_view,
                                                                   Queue &queue,
                                                                   ::reco::OTRecHitsConstView const *otView) {
    using namespace caHitNtupletGeneratorKernels;
    const pixelTrack::Iteration iterationName = this->m_params.algoParams_.iterationName_;

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Starting CAHitNtupletGeneratorKernels<TrackerTraits>::classifyTuples" << std::endl;
#endif

    const uint32_t nhits = hh.metadata().size();

    auto blockSize = 64;
    const uint32_t launchCells = this->launchCells_;
    const uint32_t maxTuples = tracks_view.metadata().size();

    // Order/backend-independent duplicate removal via the int32 quality scratch (see the kernel headers)
    // snapshotQuality() freezes quality() into the scratch; applyQuality() copies it back. The fast
    // remover needs both; the track-parallel cleaners are single-writer and need only the snapshot
    auto qScratch = this->device_qualityScratch_->data();
    auto qScratchWorkDiv =
        cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(maxTuples, blockSize), blockSize);
    auto snapshotQuality = [&]() {
      alpaka::exec<Acc1D>(
          queue, qScratchWorkDiv, Kernel_snapshotQuality{}, tracks_view, this->device_hitContainer_->data(), qScratch);
    };
    auto applyQuality = [&]() {
      alpaka::exec<Acc1D>(
          queue, qScratchWorkDiv, Kernel_applyQuality{}, tracks_view, this->device_hitContainer_->data(), qScratch);
    };

    // classify tracks based on kinematics
    auto numberOfBlocks = cms::alpakatools::divide_up_by(maxTuples, blockSize);
    auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    // Per-iteration compile-time weight bank for the in-kernel loose->tight DNN (see Kernel_connect).
    const DnnBank trackBank =
        (iterationName == ::pixelTrack::Iteration::promptHighPt) ? DnnBank::kPrompt : DnnBank::kDisplaced;
    // OT-rechit view + count for the OT-aware feature walk. When no OT source was threaded
    // (merged-hits-only), pass an empty view + 0 so the classifier never touches it.
    const ::reco::OTRecHitsConstView otHitsView = (otView != nullptr) ? *otView : ::reco::OTRecHitsConstView{};
    const uint32_t nOTForFeat = (otView != nullptr) ? this->nOTHits_ : 0u;
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_classifyTracks<TrackerTraits>{},
                        tracks_view,
                        this->device_hitContainer_->data(),
                        hh,
                        this->m_params.qualityCuts_,
                        this->m_params.algoParams_.useTrackDNN_,
                        this->m_params.algoParams_.trackDNNThreshold_,
                        trackBank,
                        otHitsView,
                        nOTForFeat);
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_classifyTracks -> done!" << std::endl;
#endif

    alpaka::exec<Acc1D>(
        queue, workDiv1D, Kernel_assignIteration{}, tracks_view, this->device_hitContainer_->data(), iterationName);
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_assignIteration -> done!" << std::endl;
#endif

    if (this->m_params.algoParams_.lateFishbone_) {
      // apply fishbone cleaning to good tracks
      numberOfBlocks = cms::alpakatools::divide_up_by(launchCells, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_fishboneCleaner<TrackerTraits>{},
                          this->device_simpleCells_->data(),
                          this->device_nCells_->data(),
                          this->device_cellToTracks_->data(),
                          tracks_view);
    }
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_fishboneCleaner   -> done!" << std::endl;
#endif
    if (this->m_params.algoParams_.doFastDuplicateRemover_) {
      // mark duplicates (tracks that share a doublet)
      // Two-tier work division: one cell per thread, with the whole warp ganging up on the rare cells
      // whose track list is longer than kDupCoopMinTracks. The block size must stay a multiple of the
      // warp size: the kernel issues full-mask warp collectives and its grid-stride loop is
      // lane-aligned only if the grid stride is.
      blockSize = 64;
      numberOfBlocks = cms::alpakatools::divide_up_by(launchCells, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      snapshotQuality();
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_fastDuplicateRemover<TrackerTraits>{},
                          this->device_simpleCells_->data(),
                          this->device_nCells_->data(),
                          this->device_cellToTracks_->data(),
                          tracks_view,
                          qScratch,
                          this->m_params.algoParams_.dupPassThrough_,
                          this->m_params.algoParams_.fastDupNSigma2_);
      applyQuality();
#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Kernel_fastDuplicateRemover   -> done!" << std::endl;
#endif
    }
    if (this->m_params.algoParams_.doSharedHitCut_ || this->m_params.algoParams_.doStats_) {
      // fill hit->track "map"
      numberOfBlocks = cms::alpakatools::divide_up_by(maxTuples, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_countHitInTracks<TrackerTraits>{},
                          tracks_view,
                          this->device_hitContainer_->data(),
                          this->device_hitToTuple_->data(),
                          nhits);  // tagged OT extras bin at nhits + otIdx

      finalizeAssocOffsets(this->device_hitToTupleView_, queue);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_fillHitInTracks<TrackerTraits>{},
                          tracks_view,
                          this->device_hitContainer_->data(),
                          this->device_hitToTuple_->data(),
                          nhits,  // tagged OT extras bin at nhits + otIdx
                          this->counters_->data());

#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Kernel_countHitInTracks   -> done!" << std::endl;
#endif
    }

    if (this->m_params.algoParams_.doSharedHitCut_) {
      // mark duplicates (tracks that share at least one hit)
      numberOfBlocks = cms::alpakatools::divide_up_by(maxTuples, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      // The shared-hit removers are track-parallel and single-writer: snapshotQuality() freezes the
      // quality into the scratch, the kernel reads the scratch and writes only each track's own
      // quality directly (no atomics, no copy-back)
      snapshotQuality();
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_rejectDuplicate<TrackerTraits>{},
                          tracks_view,
                          this->m_params.algoParams_.dupPassThrough_,
                          this->device_hitContainer_->data(),
                          qScratch,
                          this->device_hitToTuple_->data(),
                          this->m_params.algoParams_.fastDupNSigma2_);
#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Kernel_rejectDuplicate   -> done!" << std::endl;
#endif
      // Only run if the cut is doing something
      if (this->m_params.algoParams_.minHitsForSharingCut_ > 1) {
        snapshotQuality();
        alpaka::exec<Acc1D>(queue,
                            workDiv1D,
                            Kernel_sharedHitCleaner<TrackerTraits>{},
                            hh,
                            this->device_layerStarts_->data(),
                            tracks_view,
                            this->m_params.algoParams_.minHitsForSharingCut_,
                            this->m_params.algoParams_.dupPassThrough_,
                            this->device_hitContainer_->data(),
                            qScratch,
                            this->device_hitToTuple_->data());
#ifdef GPU_DEBUG
        alpaka::wait(queue);
        std::cout << "Kernel_sharedHitCleaner   -> done!" << std::endl;
#endif
      }

      if (this->m_params.algoParams_.doTripletCleaner_ && (this->m_params.algoParams_.minHitsPerNtuplet_ <= 3)) {
        if (this->m_params.algoParams_.useSimpleTripletCleaner_) {
          numberOfBlocks =
              cms::alpakatools::divide_up_by(int(nhits * this->m_params.algoParams_.avgHitsPerTrack_) + 1, blockSize);
          workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
          snapshotQuality();
          alpaka::exec<Acc1D>(queue,
                              workDiv1D,
                              Kernel_simpleTripletCleaner<TrackerTraits>{},
                              tracks_view,
                              this->m_params.algoParams_.dupPassThrough_,
                              this->device_hitContainer_->data(),
                              qScratch,
                              this->device_hitToTuple_->data());
#ifdef GPU_DEBUG
          alpaka::wait(queue);
          std::cout << "Kernel_simpleTripletCleaner   -> done!" << std::endl;
#endif
        } else {
          numberOfBlocks =
              cms::alpakatools::divide_up_by(int(nhits * this->m_params.algoParams_.avgHitsPerTrack_) + 1, blockSize);
          workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
          snapshotQuality();
          alpaka::exec<Acc1D>(queue,
                              workDiv1D,
                              Kernel_tripletCleaner<TrackerTraits>{},
                              tracks_view,
                              this->m_params.algoParams_.dupPassThrough_,
                              this->device_hitContainer_->data(),
                              qScratch,
                              this->device_hitToTuple_->data());
#ifdef GPU_DEBUG
          alpaka::wait(queue);
          std::cout << "Kernel_tripletCleaner   -> done!" << std::endl;
#endif
        }
      }
    }

    if (this->m_params.algoParams_.doStats_) {
      numberOfBlocks = cms::alpakatools::divide_up_by(std::max(nhits, launchCells), blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_checkOverflows<TrackerTraits>{},
                          tracks_view,
                          this->device_hitContainer_->data(),
                          this->device_tupleMultiplicity_->data(),
                          this->device_hitToTuple_->data(),
                          this->device_hitTuple_apc_,
                          this->device_simpleCells_->data(),
                          this->device_nCells_->data(),
                          this->device_nTriplets_->data(),
                          this->device_nCellTracks_->data(),
                          this->deviceTriplets_->view(),
                          this->deviceTracksCells_->view(),
                          nhits,
                          this->maxNumberOfDoublets_,
                          this->m_params.algoParams_,
                          this->counters_->data());
    }

    // Always-on overflow sentinel: doStats-independent surface for the capacity guards (see the
    // kernel comment). The accumulator is per-stream persistent, owned by CAHitNtupletGenerator;
    // null when no owner armed it (non-CA users of this class).
    if (this->ovfAccum_ != nullptr) {
      auto sentinelDiv = cms::alpakatools::make_workdiv<Acc1D>(1, 1);
      alpaka::exec<Acc1D>(queue,
                          sentinelDiv,
                          Kernel_overflowSentinel{},
                          this->device_hitTuple_apc_,
                          this->device_nCells_->data(),
                          this->device_nTriplets_->data(),
                          this->device_nCellTracks_->data(),
                          uint32_t(tracks_view.metadata().size()),
                          this->maxNumberOfDoublets_,
                          // The two CA pair-SoA capacities, as host-side element counts: the same
                          // numbers the views' metadata().size() returned, and they survive the
                          // build-only buffers being released at the end of launchKernels.
                          this->tripletsN_,
                          this->tracksCellsN_,
                          // hitContainer content slots (allocated in the ctor, extent host-known).
                          // This is the binding per-track-hit bound: the output trackHits SoA is
                          // sized from the same expression, so it is never the smaller of the two.
                          uint32_t(alpaka::getExtents(*this->device_hitContainerStorage_)[0u]),
                          // hitToTuple content slots. Under delayed allocation the storage is sized to
                          // exactly the demand scalar this check compares against, so the comparison
                          // carries no information and 0xFFFFFFFF disables it.
                          (this->m_params.algoParams_.delayAllocations_ || !this->device_hitToTupleStorage_)
                              ? 0xFFFFFFFFu
                              : uint32_t(alpaka::getExtents(*this->device_hitToTupleStorage_)[0u]),
                          this->ovfAccum_);
    }

#ifdef CA_STATS
    alpaka::wait(queue);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(1, 1);
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_printSizes{},
                        hh,
                        tracks_view,
                        this->device_nCells_->data(),
                        this->device_nTriplets_->data(),
                        this->device_nCellTracks_->data());

    alpaka::wait(queue);

    // Sizing recommendations. Each avg parameter sizes a buffer as cap = max_keys * avg, which is safe
    // iff actual_fill <= cap, i.e. avg >= actual_fill / max_keys. The fills are:
    //   avgHitsPerTrack  -> Sum_t nHits(t) (= APC.n),                max_keys = maxTuples
    //   avgCellsPerHit   -> nCells (one outer hit per cell),         max_keys = outerHits
    //   avgCellsPerCell  -> nTriplets (one neighbour link each),     max_keys = maxDoublets
    //   avgTracksPerCell -> nCellTrackPairs,                         max_keys = maxDoublets
    // The counters live in device_extraStorage_: [0,1] = AtomicPairCounter (m = nTracksFound,
    // n = nHitsInTracks), [2] = nCells, [3] = nTriplets, [4] = nCellTrackPairs.
    const auto iterTag = ::pixelTrack::iterationName[static_cast<uint8_t>(this->m_params.algoParams_.iterationName_)];
    std::cout << "========== CA Tracking Summary [" << iterTag << "] ==========" << std::endl;
    {
      using DW = cms::alpakatools::AtomicPairCounter::DoubleWord;
      auto h_extra = cms::alpakatools::make_host_buffer<DW[]>(queue, 5u);
      alpaka::memcpy(queue, h_extra, *this->device_extraStorage_);
      alpaka::wait(queue);

      DW const *const extra = alpaka::getPtrNative(h_extra);

      // Decode the AtomicPairCounter that lives in extra[0]. The struct
      // packs (m, n) into a single 64-bit word. Extract apc_lo and apc_hi
      // as the two 32-bit halves and then use the smaller as the number of tracks
      // and the larger as the number of hits in tracks, since every track has at least one hit.
      const uint64_t apc_raw = static_cast<uint64_t>(extra[0]);
      const uint32_t apc_lo = static_cast<uint32_t>(apc_raw & 0xFFFFFFFFull);
      const uint32_t apc_hi = static_cast<uint32_t>(apc_raw >> 32);
      const uint32_t nTracksFnd = std::min(apc_lo, apc_hi);
      const uint32_t nHitsInTrk = std::max(apc_lo, apc_hi);
      const uint32_t nCellsF = static_cast<uint32_t>(extra[2]);
      const uint32_t nTripletsF = static_cast<uint32_t>(extra[3]);
      const uint32_t nCTPairsF = static_cast<uint32_t>(extra[4]);

      // outerHits is not a parameter of classifyTuples and is not stored
      // on the class. Recover it from device_hitToCellOffsets_
      const uint32_t outerHitsR = static_cast<uint32_t>(alpaka::getExtents(*this->device_hitToCellOffsets_)[0u]) - 1u;
      auto const &a = this->m_params.algoParams_;

      // Required avg: this event would not have overflowed if avg >= it. Each ratio must be referenced
      // to the same basis the buffer is allocated against, and for the two cell-keyed buffers that
      // basis is nCells or maxDoublets depending on the allocation mode. Scaling the current avg by
      // the buffer's occupancy, avg_req = avg_cur * nTriplets / tripletsN_, is correct in both.
      const float req_HpT = (maxTuples > 0) ? float(nHitsInTrk) / float(maxTuples) : 0.f;
      const float req_CpH = (outerHitsR > 0) ? float(nCellsF) / float(outerHitsR) : 0.f;
      const float req_CpC = (tripletsN_ > 0u) ? a.avgCellsPerCell_ * float(nTripletsF) / float(tripletsN_) : 0.f;
      const float req_TpC = (tracksCellsN_ > 0u) ? a.avgTracksPerCell_ * float(nCTPairsF) / float(tracksCellsN_) : 0.f;

      // Physical mean per *actual* key (informative, NOT the sizing bound):
      const float phy_HpT = (nTracksFnd > 0) ? float(nHitsInTrk) / float(nTracksFnd) : 0.f;
      const float phy_CpH = req_CpH;  // each cell has exactly one outer hit
      const float phy_CpC = (nCellsF > 0) ? float(nTripletsF) / float(nCellsF) : 0.f;
      const float phy_TpC = (nCellsF > 0) ? float(nCTPairsF) / float(nCellsF) : 0.f;

      // Safety factor to provide some headroom
      constexpr float kSafety = 1.25f;

      // Pack into arrays so per-event and aggregate code can share a loop.
      const char *paramNames[4] = {"avgHitsPerTrack", "avgCellsPerHit", "avgCellsPerCell", "avgTracksPerCell"};
      const float currents[4] = {a.avgHitsPerTrack_, a.avgCellsPerHit_, a.avgCellsPerCell_, a.avgTracksPerCell_};
      const float req[4] = {req_HpT, req_CpH, req_CpC, req_TpC};
      const float phy[4] = {phy_HpT, phy_CpH, phy_CpC, phy_TpC};

      // Parameters that scale with nHits
      //   ratio[0] = nCells  / nHits  -> maxNumberOfDoublets
      //   ratio[1] = nTuples / nHits  -> maxNumberOfTuples
      const char *scaleNames[2] = {"maxNumberOfDoublets", "maxNumberOfTuples"};
      const uint32_t scaleCurrent[2] = {this->maxNumberOfDoublets_, maxTuples};
      const uint32_t scaleCount[2] = {nCellsF, nTracksFnd};
      const double ratio[2] = {(nhits > 0) ? double(nCellsF) / double(nhits) : 0.,
                               (nhits > 0) ? double(nTracksFnd) / double(nhits) : 0.};

      // Update the cross-event accumulator and take a snapshot under the
      // lock; print outside the lock to keep the critical section small.
      uint64_t snapN = 0;
      double snapMaxReq[4] = {0., 0., 0., 0.};
      double snapMeanReq[4] = {0., 0., 0., 0.};
      double snapMaxPhy[4] = {0., 0., 0., 0.};
      double snapMaxRatio[2] = {0., 0.};
      double snapMeanRatio[2] = {0., 0.};
      uint32_t snapMaxNHits = 0u;
      {
        auto &agg = sizingAccumulator(this->m_params.algoParams_.iterationName_);
        std::lock_guard<std::mutex> lock(agg.mtx);
        ++agg.n;
        for (int i = 0; i < 4; ++i) {
          if (req[i] > agg.maxReq[i])
            agg.maxReq[i] = req[i];
          if (phy[i] > agg.maxPhy[i])
            agg.maxPhy[i] = phy[i];
          agg.sumReq[i] += req[i];
          agg.sumPhy[i] += phy[i];
        }
        for (int i = 0; i < 2; ++i) {
          if (ratio[i] > agg.maxRatio[i])
            agg.maxRatio[i] = ratio[i];
          agg.sumRatio[i] += ratio[i];
        }
        if (nhits > agg.maxNHits)
          agg.maxNHits = nhits;

        snapN = agg.n;
        const double invN = 1.0 / double(snapN);
        for (int i = 0; i < 4; ++i) {
          snapMaxReq[i] = agg.maxReq[i];
          snapMeanReq[i] = agg.sumReq[i] * invN;
          snapMaxPhy[i] = agg.maxPhy[i];
        }
        for (int i = 0; i < 2; ++i) {
          snapMaxRatio[i] = agg.maxRatio[i];
          snapMeanRatio[i] = agg.sumRatio[i] * invN;
        }
        snapMaxNHits = agg.maxNHits;
      }

      const auto coutFlags = std::cout.flags();
      const auto coutPrec = std::cout.precision();

      std::cout << "============= Sizing-parameter recommendation ["
                << ::pixelTrack::iterationName[static_cast<uint8_t>(this->m_params.algoParams_.iterationName_)]
                << "] =============" << std::endl;
      std::cout << "  Fills :  nHits=" << nhits << "  outerHits=" << outerHitsR << "  nTracksFound=" << nTracksFnd
                << "  nHitsInTracks=" << nHitsInTrk << std::endl;
      std::cout << "           nCells=" << nCellsF << "  nTriplets=" << nTripletsF << "  nCellTrackPairs=" << nCTPairsF
                << std::endl;
      std::cout << "  Caps  :  maxDoublets=" << this->maxNumberOfDoublets_ << "  maxTuples=" << maxTuples << std::endl;

      // Per-event rows.
      std::cout << "  -- this event --" << std::endl;
      for (int i = 0; i < 4; ++i) {
        std::cout << "    " << std::left << std::setw(18) << paramNames[i] << " current=" << std::fixed
                  << std::setprecision(3) << std::setw(8) << currents[i] << "  phys(fill/key)=" << std::setw(8)
                  << phy[i] << "  min(fill/cap)=" << std::setw(8) << req[i] << "  recommend>= " << std::setw(8)
                  << (req[i] * kSafety) << "  headroom=" << std::setw(6)
                  << ((req[i] > 0.f) ? (currents[i] / req[i]) : 0.f) << "x" << std::endl;
      }

      // Aggregated-across-events rows
      std::cout << "  -- aggregated over " << snapN << " event" << (snapN == 1u ? "" : "s") << std::endl;
      for (int i = 0; i < 4; ++i) {
        std::cout << "    " << std::left << std::setw(18) << paramNames[i] << " current=" << std::fixed
                  << std::setprecision(3) << std::setw(8) << currents[i] << "  max(req)=" << std::setw(8)
                  << snapMaxReq[i] << "  mean(req)=" << std::setw(8) << snapMeanReq[i] << "  max(phys)=" << std::setw(8)
                  << snapMaxPhy[i] << "  recommend>= " << std::setw(8) << (snapMaxReq[i] * kSafety) << std::endl;
      }

      // Parameters that scale with nHits (doublets and tuples).
      // Safe sizing is f(nHits) = slope * nHits, where slope = max(count/nHits) across events.
      // This guarantees the observed worst-case event would have fit while
      // safety covers unobserved variation. Evaluate f at this event's nHits
      // and at the observed max nHits
      std::cout << "  -- scaling parameters f(nHits) over " << snapN << " event" << (snapN == 1u ? "" : "s") << " --"
                << std::endl;
      for (int i = 0; i < 2; ++i) {
        const double recSlope = snapMaxRatio[i] * kSafety;
        const uint64_t recAtThis = static_cast<uint64_t>(std::ceil(recSlope * double(nhits)));
        const uint64_t recAtMax = static_cast<uint64_t>(std::ceil(recSlope * double(snapMaxNHits)));
        std::cout << "    " << std::left << std::setw(20) << scaleNames[i] << " current=" << std::right << std::setw(10)
                  << scaleCurrent[i] << "  this: " << std::setw(8) << scaleCount[i] << "/" << nhits << "=" << std::fixed
                  << std::setprecision(4) << std::setw(7) << ratio[i] << "  agg max=" << std::setw(7) << snapMaxRatio[i]
                  << "  mean=" << std::setw(7) << snapMeanRatio[i] << std::endl
                  << "    " << std::setw(20) << "" << " recommend>= " << std::setprecision(4) << recSlope << " * nHits"
                  << "  (= " << recAtThis << " here,  " << recAtMax << " at observed max nHits=" << snapMaxNHits << ")"
                  << std::endl;
      }

      std::cout << "  Note: 'min'/'req' = the avg value below which the buffer would overflow." << std::endl;
      std::cout << "        Per-event 'recommend' = req * " << kSafety << " (per-event safety margin)." << std::endl;
      std::cout << "        Aggregate 'recommend' = max(req across events) * " << kSafety << std::endl;
      std::cout << "        Run a representative event sample, then read the aggregate row" << std::endl;
      std::cout << "        from the last event's report." << std::endl;
      std::cout << "===========================================================" << std::endl;

      std::cout.flags(coutFlags);
      std::cout.precision(coutPrec);
    }
#endif
    if (this->m_params.algoParams_.doStats_) {
      // counters (add flag???)

      numberOfBlocks =
          cms::alpakatools::divide_up_by(int(nhits * this->m_params.algoParams_.avgHitsPerTrack_) + 1, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_doStatsForHitInTracks<TrackerTraits>{},
                          this->device_hitToTuple_->data(),
                          this->counters_->data());

      numberOfBlocks = cms::alpakatools::divide_up_by(maxTuples, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_doStatsForTracks<TrackerTraits>{},
                          tracks_view,
                          this->device_hitContainer_->data(),
                          this->counters_->data());

      auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(1, 1);
      alpaka::exec<Acc1D>(queue, workDiv1D, Kernel_printCounters{}, this->counters_->data());
    }
#ifdef CA_PIPELINE_COUNTERS
    // Pipeline stage counters: count final quality distribution, copy to host, and print funnel
    {
      // Count final track quality distribution after all processing
      auto numberOfBlocksQ = cms::alpakatools::divide_up_by(maxTuples, blockSize);
      auto workDivQ = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocksQ, blockSize);
      alpaka::exec<Acc1D>(queue,
                          workDivQ,
                          Kernel_countFinalQuality<TrackerTraits>{},
                          tracks_view,
                          this->device_hitContainer_->data(),
                          hh,
                          device_pipelineCounters_->data());
      alpaka::wait(queue);
      auto host_counters = cms::alpakatools::make_host_buffer<uint32_t[]>(caHitNtupletGenerator::kTotalCounters);
      alpaka::memcpy(queue, host_counters, *device_pipelineCounters_);
      alpaka::wait(queue);
      auto const *c = host_counters.data();
      using PC = caHitNtupletGenerator::PipelineCounter;
      printf("[CA Pipeline] Doublets: total=%u pix-pix=%u pix-OT=%u OT-OT=%u\n",
             c[PC::kDoubletsTotal],
             c[PC::kDoubletsPixPix],
             c[PC::kDoubletsPixOT],
             c[PC::kDoubletsOTOT]);
      printf("[CA Pipeline]   OT barrel: L28-29=%u(FF=%u FT=%u TT=%u) L29-30=%u(FF=%u FT=%u TT=%u)\n",
             c[PC::kDoubletsL28L29],
             c[PC::kDoubletsL28L29_FF],
             c[PC::kDoubletsL28L29_FT],
             c[PC::kDoubletsL28L29_TT],
             c[PC::kDoubletsL29L30],
             c[PC::kDoubletsL29L30_FF],
             c[PC::kDoubletsL29L30_FT],
             c[PC::kDoubletsL29L30_TT]);
      printf(
          "[CA Pipeline]            L30-31=%u(FF=%u FT=%u TT=%u) L31-32=%u(FF=%u FT=%u TT=%u) L32-33=%u(FF=%u FT=%u "
          "TT=%u)\n",
          c[PC::kDoubletsL30L31],
          c[PC::kDoubletsL30L31_FF],
          c[PC::kDoubletsL30L31_FT],
          c[PC::kDoubletsL30L31_TT],
          c[PC::kDoubletsL31L32],
          c[PC::kDoubletsL31L32_FF],
          c[PC::kDoubletsL31L32_FT],
          c[PC::kDoubletsL31L32_TT],
          c[PC::kDoubletsL32L33],
          c[PC::kDoubletsL32L33_FF],
          c[PC::kDoubletsL32L33_FT],
          c[PC::kDoubletsL32L33_TT]);
      printf("[CA Pipeline]   OT brl->disk1(z<0): L28=%u L29=%u L30=%u L31=%u L32=%u L33=%u\n",
             c[PC::kDoubletsL28D1B],
             c[PC::kDoubletsL29D1B],
             c[PC::kDoubletsL30D1B],
             c[PC::kDoubletsL31D1B],
             c[PC::kDoubletsL32D1B],
             c[PC::kDoubletsL33D1B]);
      printf("[CA Pipeline]   OT brl->disk1(z>0): L28=%u L29=%u L30=%u L31=%u L32=%u L33=%u\n",
             c[PC::kDoubletsL28D1F],
             c[PC::kDoubletsL29D1F],
             c[PC::kDoubletsL30D1F],
             c[PC::kDoubletsL31D1F],
             c[PC::kDoubletsL32D1F],
             c[PC::kDoubletsL33D1F]);
      printf("[CA Pipeline]   OT disk chain (z<0): D1-D2=%u D2-D3=%u D3-D4=%u D4-D5=%u\n",
             c[PC::kDoubletsD1BD2B],
             c[PC::kDoubletsD2BD3B],
             c[PC::kDoubletsD3BD4B],
             c[PC::kDoubletsD4BD5B]);
      printf("[CA Pipeline]   OT disk chain (z>0): D1-D2=%u D2-D3=%u D3-D4=%u D4-D5=%u other=%u\n",
             c[PC::kDoubletsD1FD2F],
             c[PC::kDoubletsD2FD3F],
             c[PC::kDoubletsD3FD4F],
             c[PC::kDoubletsD4FD5F],
             c[PC::kDoubletsOTOther]);
      // Per-cut doublet rejection counters: Total, OTEarly (L28-29), OTLate (L30-32)
      {
        using namespace caHitNtupletGenerator;
        static const char *groupNames[] = {"Total", "OTEarly(L28-29)", "OTLate(L30-32)"};
        for (int g = 0; g < 3; ++g) {
          int base = PC::kDblRejBase + g * kNCuts;
          printf(
              "[CA Pipeline] DoubletCuts %s: invalidHit=%u innerCoord=%u clusterCut=%u invalidMod=%u "
              "outerCoord=%u dzRange=%u z0=%u phi=%u zSize=%u pt=%u stubSigma=%u pixStub=%u\n",
              groupNames[g],
              c[base + kCutInvalidHit],
              c[base + kCutInnerCoord],
              c[base + kCutClusterCut],
              c[base + kCutInvalidModule],
              c[base + kCutOuterCoord],
              c[base + kCutDzRange],
              c[base + kCutZ0],
              c[base + kCutPhi],
              c[base + kCutZSize],
              c[base + kCutPt],
              c[base + kCutStubSigma],
              c[base + kCutPixStub]);
        }
      }
      printf("[CA Pipeline] Triplets: total=%u ppp=%u ppO=%u pOO=%u OOO=%u\n",
             c[PC::kTripletsTotal],
             c[PC::kTripletsPixPixPix],
             c[PC::kTripletsPixPixOT],
             c[PC::kTripletsPixOTOT],
             c[PC::kTripletsOTOTOT]);
      printf("[CA Pipeline]   OOO: barrel=%u brl->z<0=%u brl->z>0=%u z<0=%u z>0=%u other=%u\n",
             c[PC::kTripletsOOO_barrel],
             c[PC::kTripletsOOO_brlToBwd],
             c[PC::kTripletsOOO_brlToFwd],
             c[PC::kTripletsOOO_bwd],
             c[PC::kTripletsOOO_fwd],
             c[PC::kTripletsOOO_other]);
      printf("[CA Pipeline]   phiMiddle rejected: %u\n", c[PC::kTripletPhiMiddleRej]);
      printf("[CA Pipeline]   chainPhiResid rejected (early): %u\n", c[PC::kTripletChainPhiResidRej]);
      // Per-cut triplet rejection breakdown (Phase2OTStubs): how many triplet
      // candidates each TripletCuts::accept() stage rejects, in application order.
      {
        using namespace caHitNtupletGenerator;
        const int trpBase = int(kTrpRejBase);
        printf(
            "[CA Pipeline] TripletCuts rejected: alignedRZ=%u alignedXY=%u beamspotDCA=%u phiCompat=%u "
            "sameSignDPhi=%u stubGeomCurv=%u stubInnerDCurv=%u\n",
            c[trpBase + kCutAlignedRZ],
            c[trpBase + kCutAlignedXY],
            c[trpBase + kCutBeamspotCompatibleXY],
            c[trpBase + kCutPhiCompatible],
            c[trpBase + kCutSameSignDPhi],
            c[trpBase + kCutStubsCurvCompatibleWithTriplet],
            c[trpBase + kCutStubsCompatibleWithInnerDoublet]);
      }
      printf("[CA Pipeline] Fishbone killed: %u\n", c[PC::kFishboneKilled]);
      printf("[CA Pipeline] Cell status: used_in_triplet=%u killed_total=%u alive=%u\n",
             c[PC::kCellsUsedInTriplet],
             c[PC::kCellsKilledTotal],
             c[PC::kCellsAlive]);
      printf("[CA Pipeline] Cell-track pairs: %u (avgTracksPerCell=%.3f)\n",
             c[PC::kCellTrackPairs],
             c[PC::kDoubletsTotal] > 0 ? float(c[PC::kCellTrackPairs]) / float(c[PC::kDoubletsTotal]) : 0.f);
      printf("[CA Pipeline] N-tuplets: total=%u with_OT=%u with_3+OT=%u\n",
             c[PC::kNtupletsTotal],
             c[PC::kNtupletsWithOT],
             c[PC::kNtupletsOT3Plus]);
      printf("[CA Pipeline] Quality: total=%u bad=%u edup=%u dup=%u loose=%u strict=%u tight=%u HP=%u\n",
             c[PC::kQualTotal],
             c[PC::kQualBad],
             c[PC::kQualEdup],
             c[PC::kQualDup],
             c[PC::kQualLoose],
             c[PC::kQualStrict],
             c[PC::kQualTight],
             c[PC::kQualHP]);
      printf("[CA Pipeline]   with OT: strict_OT=%u tight_OT=%u HP_OT=%u\n",
             c[PC::kQualStrictWithOT],
             c[PC::kQualTightWithOT],
             c[PC::kQualHPWithOT]);
      printf("[CA Pipeline]   nhits3-4: strict=%u tight=%u HP=%u chi2_boundary=%u\n",
             c[PC::kQualStrict34],
             c[PC::kQualTight34],
             c[PC::kQualHP34],
             c[PC::kChi2Boundary34]);
      printf("[CA Pipeline]   nhits5: strict=%u tight=%u HP=%u chi2_boundary=%u\n",
             c[PC::kQualStrict5],
             c[PC::kQualTight5],
             c[PC::kQualHP5],
             c[PC::kChi2Boundary5]);
      printf("[CA Pipeline]   nhits6+: strict=%u tight=%u HP=%u chi2_boundary=%u\n",
             c[PC::kQualStrict6p],
             c[PC::kQualTight6p],
             c[PC::kQualHP6p],
             c[PC::kChi2Boundary6p]);
      printf("[CA Pipeline]   fishbone: 0fb=%u 1fb=%u 2+fb=%u\n",
             c[PC::kTracksFishbone0],
             c[PC::kTracksFishbone1],
             c[PC::kTracksFishbone2p]);
      // Reset counters for next event
      alpaka::memset(queue, *device_pipelineCounters_, 0);
    }
#endif  // CA_PIPELINE_COUNTERS

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

#ifdef DUMP_GPU_TK_TUPLES
    static std::atomic<int> iev(0);
    static std::mutex lock;
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(1u, 32u);
    {
      std::lock_guard<std::mutex> guard(lock);
      ++iev;
      for (uint32_t k = 0; k < 20000; k += 500) {
        alpaka::exec<Acc1D>(queue,
                            workDiv1D,
                            Kernel_print_found_ntuplets<TrackerTraits>{},
                            hh,
                            tracks_view,
                            this->device_hitContainer_->data(),
                            this->device_hitToTuple_->data(),
                            k,
                            k + 500,
                            iev);
        alpaka::wait(queue);
      }
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_print_found_ntuplets<TrackerTraits>{},
                          hh,
                          tracks_view,
                          this->device_hitToTuple_->data(),
                          20000,
                          1000000,
                          iev);

      alpaka::wait(queue);
    }
#endif
  }

//   void CAHitMaskingAndMergerKernels::updateMasking(::reco::TrackingRecHitsMaskingView &mask_view,
//                                                    const ::reco::TrackSoAConstView &trackd_view,
//                                                    const ::reco::TrackHitSoAConstView &trackhitd_view,
//                                                    const pixelTrack::Quality minQuality,
//                                                    uint32_t const &iterationIndex,
//                                                    Queue &queue,
//                                                    bool maskAttachedHits) {
//     using namespace caHitNtupletGeneratorKernels;

// #ifdef GPU_DEBUG
//     alpaka::wait(queue);
//     std::cout << "Starting CAHitMaskingAndMergerKernels::updateMasking" << std::endl;
// #endif

//     int threadsPerBlock = 128;
//     // max(1, ...) guards against a 0-block launch (invalid CUDA launch configuration)
//     // when the track collection is empty (e.g. an empty event in a no-PU sample).
//     int blocks = std::max(1, int((trackd_view.metadata().size() + threadsPerBlock - 1) / threadsPerBlock));
//     const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);
//     // Parallel updateMasking: fills the already-sized ceil(nTracks/128) grid one thread per track
//     // (pure idempotent same-value writes; see Kernel_updateMaskingParallel in Impl.h).
//     alpaka::exec<Acc1D>(queue,
//                         workDiv1D,
//                         Kernel_updateMaskingParallel{},
//                         mask_view,
//                         trackd_view,
//                         trackhitd_view,
//                         minQuality,
//                         iterationIndex,
//                         maskAttachedHits);
// #ifdef GPU_DEBUG
//     alpaka::wait(queue);
//     std::cout << "Kernel_updateMasking -> done!" << std::endl;
// #endif
//   }

//   void CAHitMaskingAndMergerKernels::filterTracks(::reco::TrackSoAView &track_view,
//                                                   ::reco::TrackHitSoAView &trackHit_view,
//                                                   const ::reco::TrackSoAConstView &inpTrack_view,
//                                                   const ::reco::TrackHitSoAConstView &inpTrackHit_view,
//                                                   const pixelTrack::Quality minQuality,
//                                                   const double matchFraction,
//                                                   Queue &queue,
//                                                   const int32_t *loserOf,
//                                                   const int32_t *isLoser,
//                                                   const bool twinMergeRefit,
//                                                   const bool refitAllTracks,
//                                                   int32_t *unitedMaskOut,
//                                                   const uint8_t *pocketArmIn,
//                                                   uint8_t *pocketArmIdOut) {
//     using namespace caHitNtupletGeneratorKernels;

// #ifdef GPU_DEBUG
//     alpaka::wait(queue);
//     std::cout << "Starting CAHitMaskingAndMergerKernels::filterTracks" << std::endl;
// #endif

//     // Parallel Mark -> prefix-sum -> Scatter filter. nIn == 0 (empty event) leaves the output count to the
//     // caller, which never reaches this point with an empty input (PixelTracksSoAMerger early-returns).
//     const int32_t nIn = int32_t(inpTrack_view.metadata().size());

//     if (nIn > 0) {
//       const int threadsPerBlock = 128;
//       const int blocks = cms::alpakatools::divide_up_by(nIn, threadsPerBlock);
//       const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

//       // Scratch (freed stream-ordered by the caching allocator after the launches below complete).
//       // keep[]/outHitCnt[] are fully written by the Mark kernel over [0,nIn) -> no memset needed.
//       auto keep = cms::alpakatools::make_device_buffer<int32_t[]>(queue, nIn);
//       auto outHitCnt = cms::alpakatools::make_device_buffer<int32_t[]>(queue, nIn);
//       auto tkOff = cms::alpakatools::make_device_buffer<int32_t[]>(queue, nIn);
//       auto hitOff = cms::alpakatools::make_device_buffer<int32_t[]>(queue, nIn);

//       alpaka::exec<Acc1D>(queue,
//                           workDiv1D,
//                           Kernel_filterTracksMark{},
//                           inpTrack_view,
//                           inpTrackHit_view,
//                           minQuality,
//                           matchFraction,
//                           loserOf,
//                           isLoser,
//                           keep.data(),
//                           outHitCnt.data());

//       cms::alpakatools::iterativePrefixScan<Acc1D>(keep.data(), tkOff.data(), uint32_t(nIn), queue);
//       cms::alpakatools::iterativePrefixScan<Acc1D>(outHitCnt.data(), hitOff.data(), uint32_t(nIn), queue);

//       alpaka::exec<Acc1D>(queue,
//                           workDiv1D,
//                           Kernel_filterTracksScatter{},
//                           track_view,
//                           trackHit_view,
//                           inpTrack_view,
//                           inpTrackHit_view,
//                           loserOf,
//                           twinMergeRefit,
//                           refitAllTracks,
//                           unitedMaskOut,
//                           keep.data(),
//                           outHitCnt.data(),
//                           tkOff.data(),
//                           hitOff.data(),
//                           nIn,
//                           pocketArmIn,
//                           pocketArmIdOut);
//     }
// #ifdef GPU_DEBUG
//     alpaka::wait(queue);
//     std::cout << "filterTracks -> done!" << std::endl;
// #endif
//   }

//   void CAHitMaskingAndMergerKernels::twinMerge(const ::reco::TrackSoAConstView &inpTrack_view,
//                                                const ::reco::TrackHitSoAConstView &inpTrackHit_view,
//                                                const int32_t *armOfTrack,
//                                                const float twinDEta,
//                                                const float twinDPhi,
//                                                const int twinMinShared,
//                                                const bool twinTier2,
//                                                const float twinDEta2,
//                                                const float twinDPhi2,
//                                                const float twinNSigma2,
//                                                const int twinMinSharedFwd,
//                                                const pixelTrack::Quality minQuality,
//                                                int32_t *bestTwin,
//                                                int32_t *loserOf,
//                                                int32_t *isLoser,
//                                                int const &nTracks,
//                                                Queue &queue) {
//     using namespace caHitNtupletGeneratorKernels;
//     if (nTracks <= 0)
//       return;
//     // isLoser must already be zero-initialised by the caller: it is written cross-thread in
//     // Kernel_twinConfirm. makeFilteredTracks memsets it before calling this.

//     const int threadsPerBlock = 128;
//     const int blocks = cms::alpakatools::divide_up_by(nTracks, threadsPerBlock);
//     const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

//     // phi -> track pre-filter binner over this collection, so twinFindBest visits only the phi bins
//     // overlapping the twinDPhi window instead of all N tracks; the kernel's whole-ring guard makes it
//     // equivalent to an exhaustive scan, and the trackBinKey clamp keeps every key in [0,kTwinPhiBins)
//     // so the binner cannot overflow. nItems = -1 makes the binner iterate the device-side nTracks()
//     // instead of the capacity, so the tail, whose eta/phi are uninitialised, is never binned; a
//     // candidate there would be rejected at Kernel_twinFindBest's quality gate anyway.
//     const int32_t nBin = int32_t(inpTrack_view.metadata().size());
//     const uint32_t nKeys = uint32_t(kTwinPhiBins);
//     auto phiBinnerBuf = cms::alpakatools::make_device_buffer<GenericContainer>(queue);
//     auto phiOffBuf = cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, nKeys + 1);
//     auto phiStoreBuf = cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, uint32_t(nBin));
//     auto phiOvf = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, 1);
//     alpaka::memset(queue, phiOvf, 0);
//     GenericContainerView view{phiBinnerBuf.data(), phiOffBuf.data(), phiStoreBuf.data(), nKeys + 1, uint32_t(nBin)};
//     GenericContainer::template launchZero<Acc1D>(view, queue);
//     alpaka::exec<Acc1D>(queue,
//                         workDiv1D,
//                         Kernel_trackBinCount{},
//                         inpTrack_view,
//                         int32_t(-1),
//                         phiBinnerBuf.data(),
//                         kTwinPhiBins,
//                         1,
//                         0.f,
//                         nKeys,
//                         phiOvf.data());
//     finalizeAssocOffsets(view, queue);
//     alpaka::exec<Acc1D>(queue,
//                         workDiv1D,
//                         Kernel_trackBinFill{},
//                         inpTrack_view,
//                         int32_t(-1),
//                         phiBinnerBuf.data(),
//                         kTwinPhiBins,
//                         1,
//                         0.f,
//                         nKeys,
//                         phiOvf.data());

//     alpaka::exec<Acc1D>(queue,
//                         workDiv1D,
//                         Kernel_twinFindBest{},
//                         inpTrack_view,
//                         inpTrackHit_view,
//                         armOfTrack,
//                         minQuality,
//                         twinDEta,
//                         twinDPhi,
//                         twinMinShared,
//                         twinTier2,
//                         twinDEta2,
//                         twinDPhi2,
//                         twinNSigma2,
//                         twinMinSharedFwd,
//                         phiBinnerBuf.data(),
//                         kTwinPhiBins,
//                         bestTwin);
//     // twinFindBest and twinConfirm cannot be fused: twinConfirm thread i reads bestTwin[j] with
//     // j = bestTwin[i], an arbitrary opposite-arm track index, so it needs the whole bestTwin[]
//     // finalized, a grid-wide producer to consumer barrier. The inter-kernel queue boundary provides
//     // it; alpaka::syncBlockThreads inside a fused kernel is only block-local.
//     alpaka::exec<Acc1D>(queue, workDiv1D, Kernel_twinConfirm{}, inpTrack_view, bestTwin, loserOf, isLoser);
// #ifdef GPU_DEBUG
//     alpaka::wait(queue);
//     std::cout << "CAHitMaskingAndMergerKernels::twinMerge -> done!" << std::endl;
// #endif
//   }

//   void CAHitMaskingAndMergerKernels::finalDedup(::reco::TrackSoAView &out_view,
//                                                 ::reco::TrackHitSoAView &outHit_view,
//                                                 const ::reco::TrackSoAConstView &tracks_view,
//                                                 const ::reco::TrackHitSoAConstView &trackHit_view,
//                                                 int const &nTracksCap,
//                                                 uint32_t nHits,
//                                                 uint32_t nOTHits,
//                                                 Queue &queue,
//                                                 const MergerDedupConfirmInputs *confirm) {
//     using namespace caHitNtupletGeneratorKernels;
//     if (nTracksCap <= 0)
//       return;

//     // drop[] flags (1 = this refined track is the duplicate loser). Zero-init: the mark kernel only
//     // ever sets 0/1 over [0,nTracks), but the compaction reads it, so clear the whole capacity.
//     auto drop = cms::alpakatools::make_device_buffer<uint8_t[]>(queue, nTracksCap);
//     alpaka::memset(queue, drop, 0);

//     // The dedup kernels can tally a per-reason breakdown of the dropped losers into an 18-word device
//     // buffer. It is left unarmed: reading it means a device-to-host copy the host consumes at once,
//     // i.e. a full stream drain every event, and nothing downstream uses the numbers. A null pointer
//     // switches those tallies off at the source.
//     uint32_t *const diagPtr = nullptr;

//     const int threadsPerBlock = 128;
//     const int blocks = cms::alpakatools::divide_up_by(nTracksCap, threadsPerBlock);
//     const auto markDiv = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

//     // Cov-dedup: shared-hit co-occurrence pairing plus a covariance-scaled 3-parameter gate.
//     // Hit-id -> refined-track co-occurrence histogram over the key space [0, nHits + nOTHits): merged
//     // pixel/strip ids bin on the id, bit30-tagged OT extras compress to nHits + otIdx. The content is
//     // the trackHit CSR capacity, so the fill never overruns, and the count-and-clamp guard flags any
//     // out-of-range key in ovf[0] instead of writing out of bounds.
//     const uint32_t nKeys = nHits + nOTHits;
//     const uint32_t nContent = uint32_t(trackHit_view.metadata().size());  // input trackHit CSR capacity
//     auto ovf = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, 1);
//     alpaka::memset(queue, ovf, 0);

//     auto hitAssoc = cms::alpakatools::make_device_buffer<GenericContainer>(queue);
//     auto hitOff = cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, nKeys + 1);
//     auto hitStore = cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, nContent);
//     GenericContainerView hitView{hitAssoc.data(), hitOff.data(), hitStore.data(), nKeys + 1, nContent};
//     GenericContainer::template launchZero<Acc1D>(hitView, queue);
//     alpaka::exec<Acc1D>(
//         queue, markDiv, Kernel_dedupHitCount{}, tracks_view, trackHit_view, hitAssoc.data(), nHits, nKeys, ovf.data());
//     finalizeAssocOffsets(hitView, queue);
//     alpaka::exec<Acc1D>(
//         queue, markDiv, Kernel_dedupHitFill{}, tracks_view, trackHit_view, hitAssoc.data(), nHits, nKeys, ovf.data());

//     // (iii) 0-shared forward fallback eta-phi track binner (always built; the mark kernel drops the
//     // in-bound 0-shared forward losers and diag-counts the out-of-bound ones).
//     const uint32_t nFbKeys = uint32_t(kDedupFbPhiBins) * uint32_t(kDedupFbEtaSlabs);
//     auto fbAssoc = cms::alpakatools::make_device_buffer<GenericContainer>(queue);
//     auto fbOff = cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, nFbKeys + 1);
//     auto fbStore = cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, uint32_t(nTracksCap));
//     GenericContainerView fbView{fbAssoc.data(), fbOff.data(), fbStore.data(), nFbKeys + 1, uint32_t(nTracksCap)};
//     GenericContainer::template launchZero<Acc1D>(fbView, queue);
//     // nItems = -1 => the binner iterates the device-side nTracks() (the refined collection's real
//     // count), matching Kernel_dedupCovMark's range so the tail capacity slots are never binned.
//     alpaka::exec<Acc1D>(queue,
//                         markDiv,
//                         Kernel_trackBinCount{},
//                         tracks_view,
//                         int32_t(-1),
//                         fbAssoc.data(),
//                         kDedupFbPhiBins,
//                         kDedupFbEtaSlabs,
//                         kDedupFbEtaMax,
//                         nFbKeys,
//                         ovf.data());
//     finalizeAssocOffsets(fbView, queue);
//     alpaka::exec<Acc1D>(queue,
//                         markDiv,
//                         Kernel_trackBinFill{},
//                         tracks_view,
//                         int32_t(-1),
//                         fbAssoc.data(),
//                         kDedupFbPhiBins,
//                         kDedupFbEtaSlabs,
//                         kDedupFbEtaMax,
//                         nFbKeys,
//                         ovf.data());

//     // Cov-gate width and fallback neighbourhood reach, as passed to the dedup kernels:
//     //   s_scanNSigma2     = kDedupNSigma2Default (25) -> shared-hit-path cov-gate width, and the
//     //                       default for the fallback gate unless mergerFbNSigma2 overrides it.
//     //   s_scanFbEtaReach  = 1 -> fallback eta-slab reach de in [-r, r]; the loop's own slab-range
//     //                       clamp is the real bound, so the guard below only rejects absurd values.
//     //   s_scanFbPhiReach  = 1 -> fallback phi-bin reach dp2 in [-r, r], wrap kept; 2r+1 <=
//     //                       kDedupFbPhiBins so the wrapped window visits each bin at most once.
//     // The three are held runtime-opaque on purpose: they reach the kernels as arguments, and
//     // constant-folding them on a single-TU backend would unroll the neighbourhood walk into
//     // compile-time bounds, changing the float accumulation order. Do not make them constexpr.
//     static const float s_scanNSigma2 = [] {
//       volatile float v = kDedupNSigma2Default;
//       return float(v);
//     }();
//     static const int s_scanFbEtaReach = [] {
//       volatile int v = 1;
//       int r = v;
//       if (r < 0)
//         r = 0;
//       if (r > kDedupFbEtaSlabs)
//         r = kDedupFbEtaSlabs;
//       return r;
//     }();
//     static const int s_scanFbPhiReach = [] {
//       volatile int v = 1;
//       int r = v;
//       if (r < 0)
//         r = 0;
//       if (r > kDedupFbPhiBins / 2)
//         r = kDedupFbPhiBins / 2;
//       return r;
//     }();
//     // Fallback-dedup parameters come straight from the merger's confirm struct (or the compile-time
//     // defaults when no struct is passed). Plain values only: no NaN/-1 sentinel selection, which is
//     // not portable across the backends' translation units.
//     const bool fbConfirmOn = (confirm != nullptr) && confirm->enable;
//     const int fbDelta = (confirm != nullptr) ? confirm->delta : 1;
//     const float fbNSigma2 = (confirm != nullptr && confirm->fbNSigma2 > 0.f) ? confirm->fbNSigma2 : s_scanNSigma2;
//     const float fbDropBound = (confirm != nullptr) ? confirm->fbDropBound : kDedupFbDropAbsEtaMax;
//     const int fbEnable = (confirm != nullptr) ? (confirm->fbEnable ? 1 : 0) : 1;
//     const int fbSameCharge = (confirm != nullptr) ? (confirm->fbSameCharge ? 1 : 0) : 0;
//     const float fbAbsFloorDPhi = (confirm != nullptr) ? confirm->fbAbsFloorDPhi : 1.e30f;
//     const float fbAbsFloorDQoP = (confirm != nullptr) ? confirm->fbAbsFloorDQoP : 1.e30f;
//     const float fbAbsFloorDCot = (confirm != nullptr) ? confirm->fbAbsFloorDCot : 1.e30f;
//     // finderOnly needs no hit view; rankClusters/guardCrossArm are force-off unless a valid hit view
//     // is available (confirm present).
//     const int fbFinderOnly = (confirm != nullptr && confirm->finderOnly) ? 1 : 0;
//     const int rankClustersReq = (confirm != nullptr && confirm->rankClusters) ? 1 : 0;
//     // nHits-only ranking. Gated like rankClusters (confirm-present) for a uniform pattern, though
//     // nHits reads only the track SoA (::reco::nHits) and never dereferences the hit view.
//     const int rankNHitsReq = (confirm != nullptr && confirm->rankNHits) ? 1 : 0;
//     const int guardCrossArmReq = (confirm != nullptr && confirm->guardCrossArm) ? 1 : 0;
//     const int fbRankClusters = (confirm != nullptr) ? rankClustersReq : 0;
//     const int fbRankNHits = (confirm != nullptr) ? rankNHitsReq : 0;
//     const int fbGuardCrossArm = (confirm != nullptr) ? guardCrossArmReq : 0;
//     const float fbGuardVertPosMin = (confirm != nullptr) ? confirm->guardVertPosMin : 1.0f;
//     const float fbGuardChi2Margin = (confirm != nullptr) ? confirm->guardChi2Margin : 0.0f;
//     // Hit view for the cluster count / pixel-core arm proxy: the confirm struct's hv (empty when absent;
//     // never dereferenced then, since rankClusters/guardCrossArm are forced off).
//     const ::reco::TrackingRecHitConstView dedupHitView =
//         (confirm != nullptr) ? confirm->hv : ::reco::TrackingRecHitConstView{};

//     // Contested-pair list, allocated only when confirm is on. Two uint32 per slot {i,j}, slots default
//     // to 0xffffffff (unfilled); contestedCount is the atomic append cursor and contestedOverflow
//     // counts the pairs that did not fit, which are kept both.
//     std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> contestedPairsBuf;
//     std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> contestedCountBuf;
//     std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> contestedOvfBuf;
//     uint32_t *contestedPairsPtr = nullptr;
//     uint32_t *contestedCountPtr = nullptr;
//     uint32_t *contestedOvfPtr = nullptr;
//     const uint32_t contestedCap = fbConfirmOn ? kDedupConfirmMaxPairs : 0u;
//     if (fbConfirmOn) {
//       contestedPairsBuf.emplace(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, 2u * contestedCap));
//       contestedCountBuf.emplace(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, 1));
//       contestedOvfBuf.emplace(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, 1));
//       alpaka::memset(queue, *contestedPairsBuf, 0xff);  // 0xffffffff -> unfilled slot tag
//       alpaka::memset(queue, *contestedCountBuf, 0);
//       alpaka::memset(queue, *contestedOvfBuf, 0);
//       contestedPairsPtr = contestedPairsBuf->data();
//       contestedCountPtr = contestedCountBuf->data();
//       contestedOvfPtr = contestedOvfBuf->data();
//     }

//     alpaka::exec<Acc1D>(queue,
//                         markDiv,
//                         Kernel_dedupCovMark{},
//                         tracks_view,
//                         trackHit_view,
//                         hitAssoc.data(),
//                         fbAssoc.data(),
//                         nHits,
//                         nKeys,
//                         drop.data(),
//                         diagPtr,
//                         s_scanNSigma2,
//                         s_scanFbEtaReach,
//                         s_scanFbPhiReach,
//                         fbNSigma2,
//                         fbDropBound,
//                         fbEnable,
//                         // merge-or-keep-both capture; with these off the fallback drops directly:
//                         fbConfirmOn ? 1 : 0,
//                         fbSameCharge,
//                         fbAbsFloorDPhi,
//                         fbAbsFloorDQoP,
//                         fbAbsFloorDCot,
//                         contestedPairsPtr,
//                         contestedCountPtr,
//                         contestedCap,
//                         contestedOvfPtr,
//                         // Dedup ranking and guard parameters:
//                         dedupHitView,
//                         fbFinderOnly,
//                         fbRankClusters,
//                         fbRankNHits,
//                         fbGuardCrossArm,
//                         fbGuardVertPosMin,
//                         fbGuardChi2Margin);

//     // Surface any count-and-clamp overflow; never fatal, since clamped writes were skipped and
//     // unregistered contested pairs are kept both. The two counters are consumed on device by a
//     // one-thread reporter kernel: reading them back would serialize the host against everything
//     // queued ahead of the copy, for a diagnostic.
//     const auto reportDiv = cms::alpakatools::make_workdiv<Acc1D>(1, 1);
//     alpaka::exec<Acc1D>(queue, reportDiv, Kernel_dedupOverflowReport{}, ovf.data(), contestedOvfPtr);

//     // Union refit and verdict: build the de-duplicated unions of the captured contested pairs, GBL-refit
//     // them and adjust drop[] (keep-both leaves drop[i] == 0) before the compaction consumes drop[].
//     // Runs only when the confirm is on.
//     if (fbConfirmOn) {
//       HelixFit<pixelTopology::Phase2OTStubs> fitter(confirm->bfield);
//       fitter.setMaterialMap(confirm->rhoMap);
//       fitter.setBFieldMap(confirm->bFieldMap);  // (Bz,Br) r-z map; null => the scalar bfield
//       fitter.setBField(confirm->bfield);
//       fitter.setOutlierReject(true);  // observe the GBL single-hard outlier drop -> the delta measure
//       fitter.refitDedupUnions(confirm->hv,
//                               confirm->cm,
//                               tracks_view,
//                               trackHit_view,
//                               contestedPairsPtr,
//                               contestedCap,
//                               confirm->otSource,
//                               drop.data(),
//                               fbDelta,
//                               diagPtr,
//                               queue);
//     }

//     // Parallel Counts -> prefix-sum -> Scatter compaction. WHICH tracks are dropped is decided in
//     // Kernel_dedupCovMark.
//     {
//       // keep and hitCnt share one 2*nTracksCap allocation with two pointers into it: one allocate/free
//       // pair and one memset instead of two of each.
//       auto keepAndHitCnt = cms::alpakatools::make_device_buffer<int32_t[]>(queue, 2 * std::size_t(nTracksCap));
//       int32_t *keep = keepAndHitCnt.data();
//       int32_t *hitCnt = keepAndHitCnt.data() + nTracksCap;
//       auto tkOff = cms::alpakatools::make_device_buffer<int32_t[]>(queue, nTracksCap);
//       auto hitOff = cms::alpakatools::make_device_buffer<int32_t[]>(queue, nTracksCap);
//       // Trailing entries [nTracks, nTracksCap) are not written by Counts -> zero them so the scans
//       // stay constant past the last real track (tkOff[cap-1] == total kept).
//       alpaka::memset(queue, keepAndHitCnt, 0);

//       alpaka::exec<Acc1D>(queue, markDiv, Kernel_finalDedupCounts{}, tracks_view, drop.data(), keep, hitCnt);

//       cms::alpakatools::iterativePrefixScan<Acc1D>(keep, tkOff.data(), uint32_t(nTracksCap), queue);
//       cms::alpakatools::iterativePrefixScan<Acc1D>(hitCnt, hitOff.data(), uint32_t(nTracksCap), queue);

//       alpaka::exec<Acc1D>(queue,
//                           markDiv,
//                           Kernel_finalDedupScatter{},
//                           out_view,
//                           outHit_view,
//                           tracks_view,
//                           trackHit_view,
//                           keep,
//                           tkOff.data(),
//                           hitOff.data(),
//                           nTracksCap);
//     }

//     // No host wait: the dedup histograms, drop flags and compaction scratch are function-scope caching
//     // allocator buffers, and the allocator only re-hands a freed block once the event recorded on its
//     // queue at free time has completed, so the queued launches that still read them are safe.
//   }

//   /* This will make sense when we will be able to run this once per job in Alpaka

//   template <typename TrackerTraits>
//   void CAHitNtupletGeneratorKernels<TrackerTraits>::printCounters() {
//     auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(1,1);
//     alpaka::exec<Acc1D>(queue_, workDiv1D, Kernel_printCounters{}, this->counters_->data());
//   }
//   */

//   void CAHitMaskingAndMergerKernels::mergeGather(::reco::TrackSoAView &outTrack_view,
//                                                  ::reco::TrackHitSoAView &outHit_view,
//                                                  const ::reco::TrackSoAConstView &inp0Track_view,
//                                                  const ::reco::TrackHitSoAConstView &inp0Hit_view,
//                                                  const ::reco::TrackSoAConstView &inp1Track_view,
//                                                  const ::reco::TrackHitSoAConstView &inp1Hit_view,
//                                                  int nInputs,
//                                                  int32_t *armBuf,
//                                                  const int32_t arm0,
//                                                  const int32_t arm1,
//                                                  Queue &queue) {
//     using namespace caHitNtupletGeneratorKernels;

//     // Grid-stride kernel: every phase is thread-independent, so a multi-block grid is safe (each thread
//     // recomputes the per-input offsets from the device-side scalars, and the merged nTracks scalar is
//     // written by grid thread 0 and never read inside the kernel). The grid is sized from the dominant
//     // copy range, the hit capacity, clamped so tiny events do not launch empty blocks.
//     const uint32_t threadsPerBlock = 256;
//     const uint32_t hitCap = uint32_t(outHit_view.metadata().size());
//     const uint32_t blocks = std::clamp(cms::alpakatools::divide_up_by(std::max(hitCap, 1u), threadsPerBlock), 1u, 128u);
//     const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);
//     alpaka::exec<Acc1D>(queue,
//                         workDiv1D,
//                         Kernel_mergeGather{},
//                         outTrack_view,
//                         outHit_view,
//                         inp0Track_view,
//                         inp0Hit_view,
//                         inp1Track_view,
//                         inp1Hit_view,
//                         nInputs,
//                         armBuf,
//                         arm0,
//                         arm1);
//   }

  template class CAHitNtupletGeneratorKernels<pixelTopology::Phase1>;
  template class CAHitNtupletGeneratorKernels<pixelTopology::Phase2>;
  template class CAHitNtupletGeneratorKernels<pixelTopology::Phase2OT>;
  template class CAHitNtupletGeneratorKernels<pixelTopology::Phase2OTStubs>;
  template class CAHitNtupletGeneratorKernels<pixelTopology::HIonPhase1>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
