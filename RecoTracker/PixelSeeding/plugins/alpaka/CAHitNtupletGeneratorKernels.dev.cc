// C++ headers
#include <cstdio>
#ifdef DUMP_GPU_TK_TUPLES
#include <mutex>
#endif

// Alpaka headers
#include <alpaka/alpaka.hpp>

// CMSSW headers
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

// local headers
#include "CAFishbone.h"
#include "CAHitNtupletGeneratorKernels.h"
#include "CAHitNtupletGeneratorKernelsImpl.h"

// #define GPU_DEBUG
// #define NTUPLE_DEBUG
#define CA_STATS

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename TrackerTraits>
  CAHitNtupletGeneratorKernels<TrackerTraits>::CAHitNtupletGeneratorKernels(Params const &params,
                                                                            uint32_t nHits,
                                                                            uint32_t offsetBPIX2,
                                                                            uint32_t maxDoublets,
                                                                            uint32_t maxTuples,
                                                                            uint16_t nLayers,
                                                                            Queue &queue)
      : m_params(params) {
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
    uint32_t outerHits =
        nHits - offsetBPIX2;  // the number of hits that may be used as outer hits for a cell (so not on bpix1)

    // These hold the max number of associations needed
    uint32_t nHitsToTracks = std::max(uint32_t(maxTuples * algoParams.avgHitsPerTrack_), 1u);
    uint32_t nHitsToCells = std::max(uint32_t(outerHits * algoParams.avgCellsPerHit_), 1u);
    uint32_t nCellsToCells = std::max(uint32_t(maxDoublets * algoParams.avgCellsPerCell_), 1u);
    uint32_t nCellsToTracks = std::max(uint32_t(maxDoublets * algoParams.avgTracksPerCell_), 1u);

#ifdef GPU_DEBUG
    std::cout << "Allocation for tuple building with: " << std::endl;
    std::cout << "- nHits          = " << nHits << std::endl;
    std::cout << "- outerHits      = " << outerHits << std::endl;
    std::cout << "- maxDoublets    = " << maxDoublets << std::endl;
    std::cout << "- maxTracks      = " << maxTuples << std::endl;

    std::cout << "- nCellsToCells  = " << nCellsToCells << std::endl;
    std::cout << "- nHitsToCells   = " << nHitsToCells << std::endl;
    std::cout << "- nCellsToTracks = " << nCellsToTracks << std::endl;
    std::cout << "- nHitsToTracks  = " << nHitsToTracks << std::endl;
#endif

    // Hits -> Track
    device_hitToTuple_ = cms::alpakatools::make_device_buffer<GenericContainer>(queue);
    device_hitToTupleStorage_ = cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, nHitsToTracks);
    device_hitToTupleOffsets_ = cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, nHits + 1);
    device_hitToTupleView_ = {device_hitToTuple_->data(),
                              device_hitToTupleOffsets_->data(),
                              device_hitToTupleStorage_->data(),
                              nHits + 1,
                              nHitsToTracks};

    HitToTuple::template launchZero<Acc1D>(device_hitToTupleView_, queue);

    // (Outer) Hits-> Cells
    device_hitToCell_ = cms::alpakatools::make_device_buffer<GenericContainer>(queue);
    device_hitToCellStorage_ = cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, nHitsToCells);
    device_hitToCellOffsets_ = cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, outerHits + 1);
    device_hitToCellView_ = {device_hitToCell_->data(),
                             device_hitToCellOffsets_->data(),
                             device_hitToCellStorage_->data(),
                             outerHits + 1,
                             nHitsToCells};

    HitToCell::template launchZero<Acc1D>(device_hitToCellView_, queue);

    // Hits Phi Histograms: one histogram per layer
    device_hitPhiHist_ = cms::alpakatools::make_device_buffer<PhiBinner>(queue);
    device_phiBinnerStorage_ = cms::alpakatools::make_device_buffer<hindex_type[]>(queue, nHits);
    device_hitPhiView_ = {
        device_hitPhiHist_->data(), nullptr, device_phiBinnerStorage_->data(), cms::alpakatools::kDynamicSize, nHits};
    // This will hold where each layer starts in the hit soa
    device_layerStarts_ = cms::alpakatools::make_device_buffer<hindex_type[]>(queue, nLayers + 1);

    // Cell -> (Neighbor Cells, Curvature)
    // It takes 2*Ndoublets keys as for each doublet two container bins are stored:
    //   1. neighboring doublets (non-layer-skipping ones) at index = 2*iDoublet
    //   2. neighboring doublets (layer-skipping ones) at index = 2*iDoublet+1
    device_cellToNeighbors_ = cms::alpakatools::make_device_buffer<NeighborCellContainer>(queue);
    device_cellToNeighborsStorage_ =
        cms::alpakatools::make_device_buffer<NeighborCellContainerStorage[]>(queue, nCellsToCells);
    device_cellToNeighborsOffsets_ =
        cms::alpakatools::make_device_buffer<NeighborCellContainerOffsets[]>(queue, 2 * maxDoublets + 1);
    device_cellToNeighborsView_ = {device_cellToNeighbors_->data(),
                                   device_cellToNeighborsOffsets_->data(),
                                   device_cellToNeighborsStorage_->data(),
                                   2u * maxDoublets + 1u,
                                   nCellsToCells};

    CellToCell::template launchZero<Acc1D>(device_cellToNeighborsView_, queue);

    // Cell -> Tracks
    device_cellToTracks_ = cms::alpakatools::make_device_buffer<GenericContainer>(queue);
    device_cellToTracksStorage_ =
        cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, nCellsToTracks);
    device_cellToTracksOffsets_ =
        cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, maxDoublets + 1);
    device_cellToTracksView_ = {device_cellToTracks_->data(),
                                device_cellToTracksOffsets_->data(),
                                device_cellToTracksStorage_->data(),
                                maxDoublets + 1u,
                                nCellsToTracks};

    CellToTrack::template launchZero<Acc1D>(device_cellToTracksView_, queue);

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

    // Structures and Counters Storage
    device_simpleCells_ = cms::alpakatools::make_device_buffer<SimpleCell[]>(queue, maxDoublets);

    device_extraStorage_ =
        cms::alpakatools::make_device_buffer<cms::alpakatools::AtomicPairCounter::DoubleWord[]>(queue, 5u);
    device_hitTuple_apc_ = reinterpret_cast<cms::alpakatools::AtomicPairCounter *>(device_extraStorage_->data());
    device_nCells_ =
        cms::alpakatools::make_device_view(queue, *reinterpret_cast<uint32_t *>(device_extraStorage_->data() + 2));
    device_nTriplets_ =
        cms::alpakatools::make_device_view(queue, *reinterpret_cast<uint32_t *>(device_extraStorage_->data() + 3));
    device_nCellTracks_ =
        cms::alpakatools::make_device_view(queue, *reinterpret_cast<uint32_t *>(device_extraStorage_->data() + 4));

    deviceTriplets_ = CACellPairSoACollection(queue, std::lrint(maxDoublets * algoParams.avgCellsPerCell_));
    deviceTracksCells_ = CAPairSoACollection(queue, nCellsToTracks);

#ifdef CA_PIPELINE_COUNTERS
    // Pipeline stage counters for diagnostic funnel
    device_pipelineCounters_ =
        cms::alpakatools::make_device_buffer<uint32_t[]>(queue, caHitNtupletGenerator::kNCounters);
    alpaka::memset(queue, *device_pipelineCounters_, 0);
#endif

    //TODO: if doStats?
    alpaka::memset(queue, *counters_, 0);

    alpaka::memset(queue, *device_nCells_, 0);
    alpaka::memset(queue, *device_nTriplets_, 0);
    alpaka::memset(queue, *device_nCellTracks_, 0);

    maxNumberOfDoublets_ = maxDoublets;

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Allocations for CAHitNtupletGeneratorKernels: done!" << std::endl;
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
                                                                  const reco::CALayersSoAConstView &ll,
                                                                  const reco::CAGraphSoAConstView &cc,
                                                                  const reco::CATripletCutsSoAConstView &tripletCuts,
                                                                  const reco::CANtupletCutsSoAConstView &ntupletCuts,
                                                                  Queue &queue) {
    using namespace caPixelDoublets;
    using namespace caHitNtupletGeneratorKernels;

    auto tracks_view = view.tracks();
    auto tracks_hits_view = view.trackHits();

    uint32_t nhits = hh.metadata().size();
    auto const maxDoublets = this->maxNumberOfDoublets_;
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
    auto numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxDoublets / 4, blockSize);
    const auto rescale = numberOfBlocks / 65536;
    blockSize *= (rescale + 1);
    numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxDoublets / 4, blockSize);
    assert(numberOfBlocks < 65536);
    assert(blockSize > 0 && 0 == blockSize % 16);
    const Vec2D blks{numberOfBlocks, 1u};
    const Vec2D thrs{blockSize, stride};
    const auto kernelConnectWorkDiv = cms::alpakatools::make_workdiv<Acc2D>(blks, thrs);

    alpaka::exec<Acc2D>(queue,
                        kernelConnectWorkDiv,
                        Kernel_connect<TrackerTraits>{},
                        this->device_hitTuple_apc_,  // needed only to be reset, ready for next kernel
                        hh,
                        cc,
                        tripletCuts,
                        this->deviceTriplets_->view(),
                        this->device_simpleCells_->data(),
                        this->device_nCells_->data(),
                        this->device_nTriplets_->data(),
                        this->device_hitToCell_->data(),
                        this->device_cellToNeighbors_->data(),
                        this->pipelineCountersPtr());

    CellToCell::template launchFinalize<Acc1D>(this->device_cellToNeighborsView_, queue);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_connect -> Done!" << std::endl;
#endif

    auto threadsPerBlock = 1024;
    auto blocks = cms::alpakatools::divide_up_by(std::lrint(maxDoublets * m_params.algoParams_.avgCellsPerCell_),
                                                 threadsPerBlock);
    auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_fillGenericPair<caStructures::CACellPairSoAConstView, NeighborCellContainer>{},
                        this->deviceTriplets_->view(),
                        this->device_nTriplets_->data(),
                        this->device_cellToNeighbors_->data());

#ifdef CA_SORT_CONTAINERS
    alpaka::exec<Acc1D>(queue, workDiv1D, Kernel_sortHistoBins{}, this->device_cellToNeighbors_->data());
#endif

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "cellToNeighbors -> Filled!" << std::endl;
#endif

    // Deep reachability filter: require OT barrel L1 cells to have chains reaching L4+.
    // This enforces that L1 stubs are confirmed by multiple pure outer OT layers
    // before fishbone and n-tuplet building.
    // Only active for Phase2OTStubs topology; transparent for all other topologies.
    if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
      auto reachBlocks = cms::alpakatools::divide_up_by(maxDoublets, 256u);
      auto reachWorkDiv = cms::alpakatools::make_workdiv<Acc1D>(reachBlocks, 256u);
      alpaka::exec<Acc1D>(queue,
                          reachWorkDiv,
                          Kernel_reachabilityFilter<TrackerTraits>{},
                          this->device_simpleCells_->data(),
                          this->device_nCells_->data(),
                          this->device_cellToNeighbors_->data(),
                          this->m_params.algoParams_.reachTargetLayer_,
                          this->m_params.algoParams_.reachMinHops_,
                          this->pipelineCountersPtr());
#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Reachability filter -> Done!" << std::endl;
#endif
    }

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
                          ll,
                          this->device_simpleCells_->data(),
                          this->device_nCells_->data(),
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
    // Count cell status after all kill phases (reachability + fishbone)
    {
      auto cellBlocks = cms::alpakatools::divide_up_by(maxDoublets, 256u);
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
    numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxDoublets / 4, blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_find_ntuplets<TrackerTraits>{},
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

    // Orphan chain recovery: find n-tuplets from cells with no inner connection (Phase2OTStubs only)
    if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
      if (this->m_params.algoParams_.doOrphanRecovery_) {
        alpaka::exec<Acc1D>(queue,
                            workDiv1D,
                            Kernel_find_orphan_ntuplets<TrackerTraits>{},
                            cc,
                            ntupletCuts,
                            tracks_view,
                            this->device_hitContainer_->data(),
                            this->device_cellToNeighbors_->data(),
                            this->device_cellToTracks_->data(),
                            this->deviceTracksCells_->view(),
                            this->device_simpleCells_->data(),
                            this->device_nCellTracks_->data(),
                            this->device_nCells_->data(),
                            this->device_hitTuple_apc_,
                            this->m_params.algoParams_);

#ifdef GPU_DEBUG
        alpaka::wait(queue);
        std::cout << "Kernel_find_orphan_ntuplets -> Done!" << std::endl;
#endif
      }
    }

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

    CellToTracks::template launchFinalize<Acc1D>(this->device_cellToTracksView_, queue);

    blocks = cms::alpakatools::divide_up_by(std::lrint(maxDoublets * m_params.algoParams_.avgCellsPerCell_),
                                            threadsPerBlock);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_fillGenericPair<caStructures::CAPairSoAConstView, GenericContainer>{},
                        this->deviceTracksCells_->view(),
                        this->device_nCellTracks_->data(),
                        this->device_cellToTracks_->data());

    // Sort tracks within each cell's bin for deterministic duplicate removal
#ifdef CA_SORT_CONTAINERS
    alpaka::exec<Acc1D>(queue, workDiv1D, Kernel_sortHistoBins{}, this->device_cellToTracks_->data());
#endif

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

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

#ifdef CA_PIPELINE_COUNTERS
    // Pipeline counter: classify n-tuplets by OT hit content.
    // Must run AFTER finalizeBulk so that foundNtuplets offsets are valid
    // for size()/begin()/end() iteration.
    {
      auto ntupBlocks = cms::alpakatools::divide_up_by(3 * maxTuples / 4, 128u);
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
                        this->device_hitTuple_apc_);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_fillHitDetIndices   -> done!" << std::endl;
#endif

    // remove duplicates (tracks that share a doublet)
    if (this->m_params.algoParams_.doEarlyDuplicateRemoval_) {
      numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxDoublets / 4, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_earlyDuplicateRemover<TrackerTraits>{},
                          this->device_simpleCells_->data(),
                          this->device_nCells_->data(),
                          this->device_cellToTracks_->data(),
                          tracks_view,
                          this->m_params.algoParams_.dupPassThrough_);

#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Kernel_earlyDuplicateRemover   -> done!" << std::endl;
#endif
    }

    blockSize = 128;
    numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxTuples / 4, blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_countMultiplicity<TrackerTraits>{},
                        tracks_view,
                        this->device_hitContainer_->data(),
                        this->device_tupleMultiplicity_->data());
    GenericContainer::template launchFinalize<Acc1D>(this->device_tupleMultiplicityView_, queue);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_countMultiplicity   -> done!" << std::endl;
#endif

    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_fillMultiplicity<TrackerTraits>{},
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
                          ll,
                          this->device_simpleCells_->data(),
                          this->device_nCells_->data(),
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
  }

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::buildDoublets(const HitsConstView &hh,
                                                                  const ::reco::CAGraphSoAConstView &cc,
                                                                  const ::reco::CALayersSoAConstView &ll,
                                                                  const ::reco::CADoubletCutsSoAConstView &doubletCuts,
                                                                  uint32_t offsetBPIX2,
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
      return;  // protect against empty events

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
                        this->pipelineCountersPtr());

    HitToCell::template launchFinalize<Acc1D>(this->device_hitToCellView_, queue);

    threadsPerBlock = 512;
    blocks = cms::alpakatools::divide_up_by(maxDoublets, threadsPerBlock);
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
                        this->device_hitToCell_->data());

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "FillDoubletsHisto   -> done!" << std::endl;
#endif
  }

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::classifyTuples(const HitsConstView &hh,
                                                                   TkSoAView &tracks_view,
                                                                   Queue &queue) {
    using namespace caHitNtupletGeneratorKernels;

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Starting CAHitNtupletGeneratorKernels<TrackerTraits>::classifyTuples" << std::endl;
#endif

    uint32_t nhits = hh.metadata().size();

    auto blockSize = 64;
    auto const maxDoublets = this->maxNumberOfDoublets_;
    auto const maxTuples = tracks_view.metadata().size();
    // classify tracks based on kinematics
    auto numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxTuples / 4, blockSize);
    auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_classifyTracks<TrackerTraits>{},
                        tracks_view,
                        this->device_hitContainer_->data(),
                        this->m_params.qualityCuts_);
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_classifyTracks -> done!" << std::endl;
#endif

    if (this->m_params.algoParams_.lateFishbone_) {
      // apply fishbone cleaning to good tracks
      numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxDoublets / 4, blockSize);
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
    if (!(this->m_params.algoParams_.disableFastDuplicateRemover_)) {
      // mark duplicates (tracks that share a doublet)
      numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxDoublets / 4, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_fastDuplicateRemover<TrackerTraits>{},
                          this->device_simpleCells_->data(),
                          this->device_nCells_->data(),
                          this->device_cellToTracks_->data(),
                          tracks_view,
                          this->m_params.algoParams_.dupPassThrough_);
#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Kernel_fastDuplicateRemover   -> done!" << std::endl;
#endif
    }
    if (this->m_params.algoParams_.doSharedHitCut_ || this->m_params.algoParams_.doStats_) {
      // fill hit->track "map"
      numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxTuples / 4, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_countHitInTracks<TrackerTraits>{},
                          tracks_view,
                          this->device_hitContainer_->data(),
                          this->device_hitToTuple_->data());

      GenericContainer::template launchFinalize<Acc1D>(this->device_hitToTupleView_, queue);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_fillHitInTracks<TrackerTraits>{},
                          tracks_view,
                          this->device_hitContainer_->data(),
                          this->device_hitToTuple_->data());

      // Sort tracks within each hit's bin for deterministic shared-hit duplicate removal
#ifdef CA_SORT_CONTAINERS
      alpaka::exec<Acc1D>(queue, workDiv1D, Kernel_sortHistoBins{}, this->device_hitToTuple_->data());
#endif
#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Kernel_countHitInTracks   -> done!" << std::endl;
#endif
    }

    if (this->m_params.algoParams_.doSharedHitCut_) {
      // mark duplicates (tracks that share at least one hit)
      numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxTuples / 4, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_rejectDuplicate<TrackerTraits>{},
                          tracks_view,
                          this->m_params.algoParams_.dupPassThrough_,
                          this->device_hitToTuple_->data());
#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Kernel_rejectDuplicate   -> done!" << std::endl;
#endif

      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_sharedHitCleaner<TrackerTraits>{},
                          hh,
                          this->device_layerStarts_->data(),
                          tracks_view,
                          this->m_params.algoParams_.minHitsForSharingCut_,
                          this->m_params.algoParams_.dupPassThrough_,
                          this->device_hitToTuple_->data());
#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Kernel_sharedHitCleaner   -> done!" << std::endl;
#endif
      if (!(this->m_params.algoParams_.disableTripletCleaner_) && (this->m_params.algoParams_.minLayersPerNtuplet_ > 3)) {
        if (this->m_params.algoParams_.useSimpleTripletCleaner_) {
          numberOfBlocks =
              cms::alpakatools::divide_up_by(int(nhits * this->m_params.algoParams_.avgHitsPerTrack_) + 1, blockSize);
          workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
          alpaka::exec<Acc1D>(queue,
                              workDiv1D,
                              Kernel_simpleTripletCleaner<TrackerTraits>{},
                              tracks_view,
                              this->m_params.algoParams_.dupPassThrough_,
                              this->device_hitToTuple_->data());
#ifdef GPU_DEBUG
          alpaka::wait(queue);
          std::cout << "Kernel_simpleTripletCleaner   -> done!" << std::endl;
#endif
        } else {
          numberOfBlocks =
              cms::alpakatools::divide_up_by(int(nhits * this->m_params.algoParams_.avgHitsPerTrack_) + 1, blockSize);
          workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
          alpaka::exec<Acc1D>(queue,
                              workDiv1D,
                              Kernel_tripletCleaner<TrackerTraits>{},
                              tracks_view,
                              this->m_params.algoParams_.dupPassThrough_,
                              this->device_hitToTuple_->data());
#ifdef GPU_DEBUG
          alpaka::wait(queue);
          std::cout << "Kernel_tripletCleaner   -> done!" << std::endl;
#endif
        }
      }
    }

    if (this->m_params.algoParams_.doStats_) {
      numberOfBlocks = cms::alpakatools::divide_up_by(std::max(nhits, maxDoublets), blockSize);
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
    // std::cout << "========== CA Tracking Summary ==========" << std::endl;
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

      numberOfBlocks = cms::alpakatools::divide_up_by(3 * maxTuples / 4, blockSize);
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
      auto numberOfBlocksQ = cms::alpakatools::divide_up_by(3 * maxTuples / 4, blockSize);
      auto workDivQ = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocksQ, blockSize);
      alpaka::exec<Acc1D>(queue,
                          workDivQ,
                          Kernel_countFinalQuality<TrackerTraits>{},
                          tracks_view,
                          this->device_hitContainer_->data(),
                          hh,
                          device_pipelineCounters_->data());
      alpaka::wait(queue);
      auto host_counters = cms::alpakatools::make_host_buffer<uint32_t[]>(caHitNtupletGenerator::kNCounters);
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
      printf("[CA Pipeline]   OT brl->BWD: L28-D1=%u L29-D1=%u L30-D1=%u L31-D1=%u L32-D1=%u L33-D1=%u L33-D2=%u\n",
             c[PC::kDoubletsL28D1B],
             c[PC::kDoubletsL29D1B],
             c[PC::kDoubletsL30D1B],
             c[PC::kDoubletsL31D1B],
             c[PC::kDoubletsL32D1B],
             c[PC::kDoubletsL33D1B],
             c[PC::kDoubletsL33D2B]);
      printf("[CA Pipeline]   OT brl->FWD: L28-D1=%u L29-D1=%u L30-D1=%u L31-D1=%u L32-D1=%u L33-D1=%u L33-D2=%u\n",
             c[PC::kDoubletsL28D1F],
             c[PC::kDoubletsL29D1F],
             c[PC::kDoubletsL30D1F],
             c[PC::kDoubletsL31D1F],
             c[PC::kDoubletsL32D1F],
             c[PC::kDoubletsL33D1F],
             c[PC::kDoubletsL33D2F]);
      printf("[CA Pipeline]   OT BWD: D1-D2=%u D2-D3=%u D3-D4=%u D4-D5=%u\n",
             c[PC::kDoubletsD1BD2B],
             c[PC::kDoubletsD2BD3B],
             c[PC::kDoubletsD3BD4B],
             c[PC::kDoubletsD4BD5B]);
      printf("[CA Pipeline]   OT FWD: D1-D2=%u D2-D3=%u D3-D4=%u D4-D5=%u other=%u\n",
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
      printf("[CA Pipeline]   OOO: barrel=%u brl->BWD=%u brl->FWD=%u BWD=%u FWD=%u other=%u\n",
             c[PC::kTripletsOOO_barrel],
             c[PC::kTripletsOOO_brlToBwd],
             c[PC::kTripletsOOO_brlToFwd],
             c[PC::kTripletsOOO_bwd],
             c[PC::kTripletsOOO_fwd],
             c[PC::kTripletsOOO_other]);
      printf("[CA Pipeline]   phiMiddle rejected: %u\n", c[PC::kTripletPhiMiddleRej]);
      printf("[CA Pipeline]   chainPhiResid rejected (early): %u\n", c[PC::kTripletChainPhiResidRej]);
      printf("[CA Pipeline] Reachability: checked=%u passed=%u killed=%u\n",
             c[PC::kReachCellsChecked],
             c[PC::kReachCellsPassed],
             c[PC::kReachabilityKilled]);
      printf("[CA Pipeline]   kill reason: no_neighbors=%u all_neigh_killed=%u chain_short=%u\n",
             c[PC::kReachNoNeighbors],
             c[PC::kReachAllNeighKilled],
             c[PC::kReachChainShort]);
      printf("[CA Pipeline]   kill type: pix-OT=%u OT-OT=%u\n", c[PC::kReachKilledPixOT], c[PC::kReachKilledOTOT]);
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

  /* This will make sense when we will be able to run this once per job in Alpaka

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::printCounters() {
    auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(1,1);
    alpaka::exec<Acc1D>(queue_, workDiv1D, Kernel_printCounters{}, this->counters_->data());
  }
  */

  template class CAHitNtupletGeneratorKernels<pixelTopology::Phase1>;
  template class CAHitNtupletGeneratorKernels<pixelTopology::Phase2>;
  template class CAHitNtupletGeneratorKernels<pixelTopology::Phase2OT>;
  template class CAHitNtupletGeneratorKernels<pixelTopology::Phase2OTStubs>;
  template class CAHitNtupletGeneratorKernels<pixelTopology::HIonPhase1>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
