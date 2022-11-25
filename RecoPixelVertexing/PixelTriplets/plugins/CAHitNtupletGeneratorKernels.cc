#include "RecoPixelVertexing/PixelTriplets/plugins/CAHitNtupletGeneratorKernelsImpl.h"

#include <mutex>

namespace {
  // cuda atomics are NOT atomics on CPU so protect stat update with a mutex
  // waiting for a more general solution (incuding multiple devices) to be proposed and implemented
  std::mutex lock_stat;
}  // namespace

template <>
void CAHitNtupletGeneratorKernelsCPU::printCounters(Counters const *counters) {
  kernel_printCounters(counters);
}

template <>
void CAHitNtupletGeneratorKernelsCPU::buildDoublets(HitsConstView hh, int32_t offsetBPIX2, cudaStream_t stream) {
  uint32_t nhits = hh.metadata().size();

#ifdef NTUPLE_DEBUG
  std::cout << "building Doublets out of " << nhits << " Hits. BPIX2 offset is " << offsetBPIX2 << std::endl;
#endif

  // use "nhits" to heuristically dimension the workspace
  std::cout << __LINE__ << std::endl;
  // no need to use the Traits allocations, since we know this is being compiled for the CPU
  //device_isOuterHitOfCell_ = Traits::template make_unique<GPUCACell::OuterHitOfCell[]>(std::max(1U, nhits), stream);
  device_isOuterHitOfCell_ = std::make_unique<GPUCACell::OuterHitOfCellContainer[]>(std::max(1U, nhits));
  assert(device_isOuterHitOfCell_.get());
  isOuterHitOfCell_ = GPUCACell::OuterHitOfCell{device_isOuterHitOfCell_.get(), offsetBPIX2};

  auto cellStorageSize = caConstants::maxNumOfActiveDoublets * sizeof(GPUCACell::CellNeighbors) +
                         caConstants::maxNumOfActiveDoublets * sizeof(GPUCACell::CellTracks);
  // no need to use the Traits allocations, since we know this is being compiled for the CPU
  //cellStorage_ = Traits::template make_unique<unsigned char[]>(cellStorageSize, stream);
  cellStorage_ = std::make_unique<unsigned char[]>(cellStorageSize);
  device_theCellNeighborsContainer_ = (GPUCACell::CellNeighbors *)cellStorage_.get();
  device_theCellTracksContainer_ = (GPUCACell::CellTracks *)(cellStorage_.get() + caConstants::maxNumOfActiveDoublets *
                                                                                      sizeof(GPUCACell::CellNeighbors));

  std::cout << __LINE__ << std::endl;
  gpuPixelDoublets::initDoublets(isOuterHitOfCell_,
                                 nhits,
                                 device_theCellNeighbors_.get(),
                                 device_theCellNeighborsContainer_,
                                 device_theCellTracks_.get(),
                                 device_theCellTracksContainer_);

  // no need to use the Traits allocations, since we know this is being compiled for the CPU
  //device_theCells_ = Traits::template make_unique<GPUCACell[]>(params_.maxNumberOfDoublets_, stream);
  std::cout << __LINE__ << std::endl;
  device_theCells_ = std::make_unique<GPUCACell[]>(params_.maxNumberOfDoublets_);
  if (0 == nhits)
    return;  // protect against empty events
  std::cout << __LINE__ << std::endl;
  // take all layer pairs into account
  auto nActualPairs = gpuPixelDoublets::nPairs;
  if (not params_.includeJumpingForwardDoublets_) {
    // exclude forward "jumping" layer pairs
    nActualPairs = gpuPixelDoublets::nPairsForTriplets;
  }
  if (params_.minHitsPerNtuplet_ > 3) {
    // for quadruplets, exclude all "jumping" layer pairs
    nActualPairs = gpuPixelDoublets::nPairsForQuadruplets;
  }
  std::cout << __LINE__ << std::endl;
  assert(nActualPairs <= gpuPixelDoublets::nPairs);
  std::cout << __LINE__ << std::endl;
  gpuPixelDoublets::getDoubletsFromHisto(device_theCells_.get(),
                                         device_nCells_,
                                         device_theCellNeighbors_.get(),
                                         device_theCellTracks_.get(),
                                         hh,
                                         isOuterHitOfCell_,
                                         nActualPairs,
                                         params_.idealConditions_,
                                         params_.doClusterCut_,
                                         params_.doZ0Cut_,
                                         params_.doPtCut_,
                                         params_.maxNumberOfDoublets_);
  std::cout << __LINE__ << std::endl;
}

template <>
void CAHitNtupletGeneratorKernelsCPU::launchKernels(HitsConstView hh, TkSoAView tracks_view, cudaStream_t cudaStream) {
  // zero tuples
  cms::cuda::launchZero(&tracks_view.hitIndices(), cudaStream);

  uint32_t nhits = hh.metadata().size();

  // std::cout << "N hits " << nhits << std::endl;
  // if (nhits<2) std::cout << "too few hits " << nhits << std::endl;

  //
  // applying conbinatoric cleaning such as fishbone at this stage is too expensive
  //

  kernel_connect(device_hitTuple_apc_,
                 device_hitToTuple_apc_,  // needed only to be reset, ready for next kernel
                 hh,
                 device_theCells_.get(),
                 device_nCells_,
                 device_theCellNeighbors_.get(),
                 isOuterHitOfCell_,
                 params_.hardCurvCut_,
                 params_.ptmin_,
                 params_.CAThetaCutBarrel_,
                 params_.CAThetaCutForward_,
                 params_.dcaCutInnerTriplet_,
                 params_.dcaCutOuterTriplet_);

  if (nhits > 1 && params_.earlyFishbone_) {
    gpuPixelDoublets::fishbone(hh, device_theCells_.get(), device_nCells_, isOuterHitOfCell_, nhits, false);
  }

  kernel_find_ntuplets(hh,
                       device_theCells_.get(),
                       device_nCells_,
                       device_theCellTracks_.get(),
                       tracks_view,
                       device_hitTuple_apc_,
                       params_.minHitsPerNtuplet_);
  if (params_.doStats_)
    kernel_mark_used(device_theCells_.get(), device_nCells_);

  cms::cuda::finalizeBulk(device_hitTuple_apc_, &tracks_view.hitIndices());

  kernel_fillHitDetIndices(tracks_view, hh);
  kernel_fillNLayers(tracks_view, device_hitTuple_apc_);

  // remove duplicates (tracks that share a doublet)
  kernel_earlyDuplicateRemover(device_theCells_.get(), device_nCells_, tracks_view, params_.dupPassThrough_);
  kernel_countMultiplicity(tracks_view, device_tupleMultiplicity_.get());
  cms::cuda::launchFinalize(device_tupleMultiplicity_.get(), cudaStream);
  kernel_fillMultiplicity(tracks_view, device_tupleMultiplicity_.get());

  if (nhits > 1 && params_.lateFishbone_) {
    gpuPixelDoublets::fishbone(hh, device_theCells_.get(), device_nCells_, isOuterHitOfCell_, nhits, true);
  }
}

template <>
void CAHitNtupletGeneratorKernelsCPU::classifyTuples(HitsConstView hh, TkSoAView tracks_view, cudaStream_t cudaStream) {
  int32_t nhits = hh.metadata().size();

  auto *quality_d = tracks_view.quality();
  // classify tracks based on kinematics
  kernel_classifyTracks(tracks_view, quality_d, params_.cuts_);

  if (params_.lateFishbone_) {
    // apply fishbone cleaning to good tracks
    kernel_fishboneCleaner(device_theCells_.get(), device_nCells_, quality_d);
  }

  // remove duplicates (tracks that share a doublet)
  kernel_fastDuplicateRemover(device_theCells_.get(), device_nCells_, tracks_view, params_.dupPassThrough_);

  // fill hit->track "map"
  if (params_.doSharedHitCut_ || params_.doStats_) {
    kernel_countHitInTracks(tracks_view, device_hitToTuple_.get());
    cms::cuda::launchFinalize(hitToTupleView_, cudaStream);
    kernel_fillHitInTracks(tracks_view, device_hitToTuple_.get());
  }
  // remove duplicates (tracks that share at least one hit)
  if (params_.doSharedHitCut_) {
    kernel_rejectDuplicate(
        tracks_view, params_.minHitsForSharingCut_, params_.dupPassThrough_, device_hitToTuple_.get());
    kernel_sharedHitCleaner(
        hh, tracks_view, params_.minHitsForSharingCut_, params_.dupPassThrough_, device_hitToTuple_.get());
    if (params_.useSimpleTripletCleaner_) {
      kernel_simpleTripletCleaner(
          tracks_view, params_.minHitsForSharingCut_, params_.dupPassThrough_, device_hitToTuple_.get());
    } else {
      kernel_tripletCleaner(
          tracks_view, params_.minHitsForSharingCut_, params_.dupPassThrough_, device_hitToTuple_.get());
    }
  }

  if (params_.doStats_) {
    std::lock_guard guard(lock_stat);
    kernel_checkOverflows(tracks_view,
                          device_tupleMultiplicity_.get(),
                          device_hitToTuple_.get(),
                          device_hitTuple_apc_,
                          device_theCells_.get(),
                          device_nCells_,
                          device_theCellNeighbors_.get(),
                          device_theCellTracks_.get(),
                          isOuterHitOfCell_,
                          nhits,
                          params_.maxNumberOfDoublets_,
                          counters_);
  }
  if (params_.doStats_) {
    // counters (add flag???)
    std::lock_guard guard(lock_stat);
    kernel_doStatsForHitInTracks(device_hitToTuple_.get(), counters_);
    kernel_doStatsForTracks(tracks_view, quality_d, counters_);
  }

#ifdef DUMP_GPU_TK_TUPLES
  static std::atomic<int> iev(0);
  static std::mutex lock;
  {
    std::lock_guard<std::mutex> guard(lock);
    ++iev;
    kernel_print_found_ntuplets(hh, tracks_view, device_hitToTuple_.get(), 0, 1000000, iev);
  }
#endif
}
