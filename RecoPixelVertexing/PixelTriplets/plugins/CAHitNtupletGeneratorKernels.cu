#include "RecoPixelVertexing/PixelTriplets/plugins/CAHitNtupletGeneratorKernelsImpl.h"
#include <mutex>

template <>
void CAHitNtupletGeneratorKernelsGPU::launchKernels(HitsConstView hh, TkSoAView tracks_view, cudaStream_t cudaStream) {
  // these are pointer on GPU!
  auto *quality_d = tracks_view.quality();

  // zero tuples
  cms::cuda::launchZero(&(tracks_view.hitIndices()), cudaStream);

  int32_t nhits = hh.metadata().size();

#ifdef NTUPLE_DEBUG
  std::cout << "start tuple building. N hits " << nhits << std::endl;
  if (nhits < 2)
    std::cout << "too few hits " << nhits << std::endl;
#endif

  //
  // applying conbinatoric cleaning such as fishbone at this stage is too expensive
  //

  auto nthTot = 64;
  auto stride = 4;
  auto blockSize = nthTot / stride;
  auto numberOfBlocks = nDoubletBlocks(blockSize);
  auto rescale = numberOfBlocks / 65536;
  blockSize *= (rescale + 1);
  numberOfBlocks = nDoubletBlocks(blockSize);
  assert(numberOfBlocks < 65536);
  assert(blockSize > 0 && 0 == blockSize % 16);
  dim3 blks(1, numberOfBlocks, 1);
  dim3 thrs(stride, blockSize, 1);

  kernel_connect<<<blks, thrs, 0, cudaStream>>>(
      device_hitTuple_apc_,
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
  cudaCheck(cudaGetLastError());

  // do not run the fishbone if there are hits only in BPIX1
  if (nhits > isOuterHitOfCell_.offset && params_.earlyFishbone_) {
    auto nthTot = 128;
    auto stride = 16;
    auto blockSize = nthTot / stride;
    auto numberOfBlocks = (nhits - isOuterHitOfCell_.offset + blockSize - 1) / blockSize;
    dim3 blks(1, numberOfBlocks, 1);
    dim3 thrs(stride, blockSize, 1);
    gpuPixelDoublets::fishbone<<<blks, thrs, 0, cudaStream>>>(
        hh, device_theCells_.get(), device_nCells_, isOuterHitOfCell_, nhits, false);
    cudaCheck(cudaGetLastError());
  }

  blockSize = 64;
  numberOfBlocks = (3 * params_.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
  kernel_find_ntuplets<<<numberOfBlocks, blockSize, 0, cudaStream>>>(hh,
                                                                     device_theCells_.get(),
                                                                     device_nCells_,
                                                                     device_theCellTracks_.get(),
                                                                     tracks_view,
                                                                     device_hitTuple_apc_,
                                                                     params_.minHitsPerNtuplet_);
  cudaCheck(cudaGetLastError());

  if (params_.doStats_)
    kernel_mark_used<<<numberOfBlocks, blockSize, 0, cudaStream>>>(device_theCells_.get(), device_nCells_);
  cudaCheck(cudaGetLastError());

#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif

  blockSize = 128;
  numberOfBlocks = (HitContainer::ctNOnes() + blockSize - 1) / blockSize;
  cms::cuda::finalizeBulk<<<numberOfBlocks, blockSize, 0, cudaStream>>>(device_hitTuple_apc_,
                                                                        &tracks_view.hitIndices());

  kernel_fillHitDetIndices<<<numberOfBlocks, blockSize, 0, cudaStream>>>(tracks_view, hh);
  cudaCheck(cudaGetLastError());
  kernel_fillNLayers<<<numberOfBlocks, blockSize, 0, cudaStream>>>(tracks_view, device_hitTuple_apc_);
  cudaCheck(cudaGetLastError());

  // remove duplicates (tracks that share a doublet)
  numberOfBlocks = nDoubletBlocks(blockSize);
  kernel_earlyDuplicateRemover<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
      device_theCells_.get(), device_nCells_, tracks_view, params_.dupPassThrough_);
  cudaCheck(cudaGetLastError());

  blockSize = 128;
  numberOfBlocks = (3 * caConstants::maxTuples / 4 + blockSize - 1) / blockSize;
  kernel_countMultiplicity<<<numberOfBlocks, blockSize, 0, cudaStream>>>(tracks_view, device_tupleMultiplicity_.get());
  cms::cuda::launchFinalize(device_tupleMultiplicity_.get(), cudaStream);
  kernel_fillMultiplicity<<<numberOfBlocks, blockSize, 0, cudaStream>>>(tracks_view, device_tupleMultiplicity_.get());
  cudaCheck(cudaGetLastError());

  // do not run the fishbone if there are hits only in BPIX1
  if (nhits > isOuterHitOfCell_.offset && params_.lateFishbone_) {
    auto nthTot = 128;
    auto stride = 16;
    auto blockSize = nthTot / stride;
    auto numberOfBlocks = (nhits - isOuterHitOfCell_.offset + blockSize - 1) / blockSize;
    dim3 blks(1, numberOfBlocks, 1);
    dim3 thrs(stride, blockSize, 1);
    gpuPixelDoublets::fishbone<<<blks, thrs, 0, cudaStream>>>(
        hh, device_theCells_.get(), device_nCells_, isOuterHitOfCell_, nhits, true);
    cudaCheck(cudaGetLastError());
  }

#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif

  // free space asap
  // device_isOuterHitOfCell_.reset();
}

template <>
void CAHitNtupletGeneratorKernelsGPU::buildDoublets(HitsConstView hh, int32_t offsetBPIX2, cudaStream_t stream) {
  int32_t nhits = hh.metadata().size();

  isOuterHitOfCell_ = GPUCACell::OuterHitOfCell{device_isOuterHitOfCell_.get(), offsetBPIX2};

#ifdef NTUPLE_DEBUG
  std::cout << "building Doublets out of " << nhits << " Hits" << std::endl;
#endif

#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif
  std::cout << __LINE__ << std::endl;
  // in principle we can use "nhits" to heuristically dimension the workspace...
  device_isOuterHitOfCell_ =
      cms::cuda::make_device_unique<GPUCACell::OuterHitOfCellContainer[]>(std::max(1, nhits - offsetBPIX2), stream);
  assert(device_isOuterHitOfCell_.get());
  std::cout << __LINE__ << std::endl;
  isOuterHitOfCell_ = GPUCACell::OuterHitOfCell{device_isOuterHitOfCell_.get(), offsetBPIX2};
  std::cout << __LINE__ << std::endl;
  cellStorage_ = cms::cuda::make_device_unique<unsigned char[]>(
      caConstants::maxNumOfActiveDoublets * sizeof(GPUCACell::CellNeighbors) +
          caConstants::maxNumOfActiveDoublets * sizeof(GPUCACell::CellTracks),
      stream);
  device_theCellNeighborsContainer_ = (GPUCACell::CellNeighbors *)cellStorage_.get();
  device_theCellTracksContainer_ = (GPUCACell::CellTracks *)(cellStorage_.get() + caConstants::maxNumOfActiveDoublets *
                                                                                      sizeof(GPUCACell::CellNeighbors));
  std::cout << __LINE__ << std::endl;
  {
    int threadsPerBlock = 128;
    // at least one block!
    int blocks = (std::max(1, nhits - offsetBPIX2 + threadsPerBlock - 1) / threadsPerBlock);
    gpuPixelDoublets::initDoublets<<<blocks, threadsPerBlock, 0, stream>>>(isOuterHitOfCell_,
                                                                           nhits,
                                                                           device_theCellNeighbors_.get(),
                                                                           device_theCellNeighborsContainer_,
                                                                           device_theCellTracks_.get(),
                                                                           device_theCellTracksContainer_);
    cudaCheck(cudaGetLastError());
  }
  std::cout << __LINE__ << std::endl;
  device_theCells_ = cms::cuda::make_device_unique<GPUCACell[]>(params_.maxNumberOfDoublets_, stream);
  std::cout << __LINE__ << std::endl;
#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif

  if (0 == nhits)
    return;  // protect against empty events

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
  int stride = 4;
  int threadsPerBlock = gpuPixelDoublets::getDoubletsFromHistoMaxBlockSize / stride;
  int blocks = (4 * nhits + threadsPerBlock - 1) / threadsPerBlock;
  dim3 blks(1, blocks, 1);
  dim3 thrs(stride, threadsPerBlock, 1);
  gpuPixelDoublets::getDoubletsFromHisto<<<blks, thrs, 0, stream>>>(device_theCells_.get(),
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
  cudaCheck(cudaGetLastError());

#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif
}

template <>
void CAHitNtupletGeneratorKernelsGPU::classifyTuples(HitsConstView hh, TkSoAView tracks_view, cudaStream_t cudaStream) {
  // these are pointer on GPU!
  auto *quality_d = tracks_view.quality();

  int32_t nhits = hh.metadata().size();
  std::cout << "PSOFOS 1" << std::endl;
  auto blockSize = 64;

  // classify tracks based on kinematics
  auto numberOfBlocks = nQuadrupletBlocks(blockSize);
  kernel_classifyTracks<<<numberOfBlocks, blockSize, 0, cudaStream>>>(tracks_view, quality_d, params_.cuts_);
  std::cout << "PSOFOS 2" << std::endl;
  cudaCheck(cudaGetLastError());

  if (params_.lateFishbone_) {
    // apply fishbone cleaning to good tracks
    numberOfBlocks = nDoubletBlocks(blockSize);
    kernel_fishboneCleaner<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
        device_theCells_.get(), device_nCells_, quality_d);
    cudaCheck(cudaGetLastError());
    std::cout << "PSOFOS 2.1" << std::endl;
  }
  std::cout << "PSOFOS 2.2" << std::endl;
  // mark duplicates (tracks that share a doublet)
  numberOfBlocks = nDoubletBlocks(blockSize);
  std::cout << "PSOFOS 2.3" << std::endl;
  kernel_fastDuplicateRemover<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
      device_theCells_.get(), device_nCells_, tracks_view, params_.dupPassThrough_);
  std::cout << "PSOFOS 2.4" << std::endl;
  cudaCheck(cudaGetLastError());
  std::cout << "PSOFOS 3" << std::endl;

#ifdef GPU_DEBUG
  cudaCheck(cudaDeviceSynchronize());
#endif
  std::cout << "PSOFOS 3 new" << std::endl;
  if (params_.doSharedHitCut_ || params_.doStats_) {
    std::cout << "PSOFOS 3.1" << std::endl;
    // fill hit->track "map"
    assert(hitToTupleView_.offSize > nhits);
    std::cout << "PSOFOS 3.2" << std::endl;
    numberOfBlocks = nQuadrupletBlocks(blockSize);
    std::cout << "PSOFOS 3.3" << std::endl;
    kernel_countHitInTracks<<<numberOfBlocks, blockSize, 0, cudaStream>>>(tracks_view, device_hitToTuple_.get());
    cudaCheck(cudaGetLastError());
    std::cout << "PSOFOS 3.4" << std::endl;
    assert((hitToTupleView_.assoc == device_hitToTuple_.get()) &&
           (hitToTupleView_.offStorage == device_hitToTupleStorage_.get()) && (hitToTupleView_.offSize > 0));
    cms::cuda::launchFinalize(hitToTupleView_, cudaStream);
    cudaCheck(cudaGetLastError());
    std::cout << "PSOFOS 3.5" << std::endl;
    kernel_fillHitInTracks<<<numberOfBlocks, blockSize, 0, cudaStream>>>(tracks_view, device_hitToTuple_.get());
    cudaCheck(cudaGetLastError());
    std::cout << "PSOFOS 3.6" << std::endl;
#ifdef GPU_DEBUG
    cudaCheck(cudaDeviceSynchronize());
#endif
  }
  std::cout << "PSOFOS 4" << std::endl;
  if (params_.doSharedHitCut_) {
    // mark duplicates (tracks that share at least one hit)
    numberOfBlocks = (hitToTupleView_.offSize + blockSize - 1) / blockSize;

    kernel_rejectDuplicate<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
        tracks_view, params_.minHitsForSharingCut_, params_.dupPassThrough_, device_hitToTuple_.get());
    std::cout << "PSOFOS 4.1" << std::endl;
    kernel_sharedHitCleaner<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
        hh, tracks_view, params_.minHitsForSharingCut_, params_.dupPassThrough_, device_hitToTuple_.get());
    std::cout << "PSOFOS 4.2" << std::endl;
    if (params_.useSimpleTripletCleaner_) {
      kernel_simpleTripletCleaner<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
          tracks_view, params_.minHitsForSharingCut_, params_.dupPassThrough_, device_hitToTuple_.get());
      std::cout << "PSOFOS 4.3" << std::endl;
    } else {
      kernel_tripletCleaner<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
          tracks_view, params_.minHitsForSharingCut_, params_.dupPassThrough_, device_hitToTuple_.get());
      std::cout << "PSOFOS 4.4" << std::endl;
    }
    cudaCheck(cudaGetLastError());
#ifdef GPU_DEBUG
    cudaCheck(cudaDeviceSynchronize());
#endif
  }
  std::cout << "PSOFOS 5" << std::endl;
  if (params_.doStats_) {
    numberOfBlocks = (std::max(nhits, int(params_.maxNumberOfDoublets_)) + blockSize - 1) / blockSize;
    kernel_checkOverflows<<<numberOfBlocks, blockSize, 0, cudaStream>>>(tracks_view,
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
    cudaCheck(cudaGetLastError());
    std::cout << "PSOFOS 5.1" << std::endl;
  }
  std::cout << "PSOFOS 6" << std::endl;
  if (params_.doStats_) {
    // counters (add flag???)
    numberOfBlocks = (hitToTupleView_.offSize + blockSize - 1) / blockSize;
    kernel_doStatsForHitInTracks<<<numberOfBlocks, blockSize, 0, cudaStream>>>(device_hitToTuple_.get(), counters_);
    cudaCheck(cudaGetLastError());
    std::cout << "PSOFOS 6.1" << std::endl;
    numberOfBlocks = (3 * caConstants::maxNumberOfQuadruplets / 4 + blockSize - 1) / blockSize;
    kernel_doStatsForTracks<<<numberOfBlocks, blockSize, 0, cudaStream>>>(tracks_view, quality_d, counters_);
    cudaCheck(cudaGetLastError());
    std::cout << "PSOFOS 6.2" << std::endl;
  }
#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif

#ifdef DUMP_GPU_TK_TUPLES
  static std::atomic<int> iev(0);
  static std::mutex lock;
  {
    std::lock_guard<std::mutex> guard(lock);
    ++iev;
    for (int k = 0; k < 20000; k += 500) {
      kernel_print_found_ntuplets<<<1, 32, 0, cudaStream>>>(hh, tracks_view, device_hitToTuple_.get(), k, k + 500, iev);
      cudaDeviceSynchronize();
      std::cout << "PSOFOS 7" << std::endl;
    }
    kernel_print_found_ntuplets<<<1, 32, 0, cudaStream>>>(
        hh, tracks_view, device_hitToTuple_.get(), 20000, 1000000, iev);
    cudaDeviceSynchronize();
    std::cout << "PSOFOS 8" << std::endl;
    // cudaStreamSynchronize(cudaStream);
  }
#endif
}

template <>
void CAHitNtupletGeneratorKernelsGPU::printCounters(Counters const *counters) {
  kernel_printCounters<<<1, 1>>>(counters);
}
