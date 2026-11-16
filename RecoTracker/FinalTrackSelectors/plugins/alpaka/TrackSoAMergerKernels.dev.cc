#include <algorithm>
#include <utility>

// Alpaka headers
#include <alpaka/alpaka.hpp>

// CMSSW headers
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

// local headers
#include "TrackSoAMergerKernels.h"
#include "TrackSoAMergerKernelsImpl.h"

// #define GPU_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  TrackSoAMergerKernels::TrackSoAMergerKernels(Queue &queue, int maxTracks, Params const &params) : params_(params) {
    counters_d_ = reco::TrackMergerCounterSoACollection(queue, maxTracks);
    totCounters_ = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, 2u);

    totTracks_ = cms::alpakatools::make_device_view(queue, *reinterpret_cast<uint32_t *>(totCounters_->data()));
    totHits_ = cms::alpakatools::make_device_view(queue, *reinterpret_cast<uint32_t *>(totCounters_->data() + 1));

    alpaka::memset(queue, *totTracks_, 0);
    alpaka::memset(queue, *totHits_, 0);
  }

  reco::TracksSoACollection TrackSoAMergerKernels::makeMergedTracks(Queue &queue,
                                                                    const TracksMultiView &tracks,
                                                                    const TrackHitsMultiView &hits) {
    countGoodTracks(queue, tracks);
    fillGoodTracks(queue, tracks, hits);
    filterTracks(queue);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "finished filtering track SoAs on GPU" << std::endl;
#endif

    return std::move(*tracks_d_);
  }

  void TrackSoAMergerKernels::countGoodTracks(Queue &queue, const TracksMultiView &tracks) {
    using namespace trackSoAMergerKernels;

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Starting TrackSoAMergerKernels::countGoodTracks" << std::endl;
#endif

    if (tracks.size() != 0) {
      const auto threadsPerBlock = 128u;
      const auto blocks = cms::alpakatools::divide_up_by(tracks.size(), threadsPerBlock);
      const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_countGoodTracks{},
                          tracks,
                          params_.minQuality,
                          counters_d_->view(),
                          totTracks_->data(),
                          totHits_->data());
    }

    auto totCountersHost = cms::alpakatools::make_host_buffer<uint32_t[]>(queue, 2);
    alpaka::memcpy(queue, totCountersHost, *totCounters_);
    alpaka::wait(queue);

#ifdef GPU_DEBUG
    std::cout << "Kernel_countGoodTracks -> done!" << std::endl;
    std::cout << "Total good tracks: " << totCountersHost[0] << ", total hits: " << totCountersHost[1] << std::endl;
#endif

    tracks_d_ = reco::TracksSoACollection(queue, totCountersHost[0], totCountersHost[1]);
    duplicate_d_ = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, totCountersHost[0]);
    alpaka::memset(queue, *duplicate_d_, 0);
    
    // with no good tracks Kernel_fillGoodTracks does not run and would leave nTracks unset
    if (totCountersHost[0] == 0) {
      auto nTracks_d = cms::alpakatools::make_device_view(queue, tracks_d_->view().tracks().nTracks());
      alpaka::memset(queue, nTracks_d, 0);
    }

    if (totCountersHost[0] > 0) {
      // Build the hit offsets for the selected tracks:
      // tracks().hitOffsets() holds the ending offset of each track hits vector,
      // i.e. the inclusive prefix sum of the number of hits per track.
      // The unified prefixScan dispatches between multiBlockPrefixScan and iterativePrefixScan
      // depending on the size (and runs a serial scan on the CPU backends).
      cms::alpakatools::prefixScan<Acc1D>(queue,
                                          counters_d_->view().hitsInTrack().data(),
                                          tracks_d_->view().tracks().hitOffsets().data(),
                                          totCountersHost[0]);
    }
  }

  void TrackSoAMergerKernels::fillGoodTracks(Queue &queue,
                                             const TracksMultiView &tracks,
                                             const TrackHitsMultiView &hits) {
    using namespace trackSoAMergerKernels;

    if (tracks_d_->view().tracks().metadata().size() > 0) {
#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Starting TrackSoAMergerKernels::fillGoodTracks" << std::endl;
#endif
      const auto threadsPerBlock = 128u;
      const auto blocks = cms::alpakatools::divide_up_by(tracks_d_->view().tracks().metadata().size(), threadsPerBlock);
      const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_fillGoodTracks{},
                          tracks,
                          hits,
                          counters_d_->view(),
                          tracks_d_->view().tracks(),
                          tracks_d_->view().trackHits());
    }
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_fillGoodTracks -> done!" << std::endl;
#endif
  }

  void TrackSoAMergerKernels::filterTracks(Queue &queue) {
    using namespace trackSoAMergerKernels;

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Starting TrackSoAMergerKernels::filterTracks" << std::endl;
#endif

    auto const nTracks = tracks_d_->view().tracks().metadata().size();
    if (nTracks < 2) {
      return;
    }

    if (params_.doSameHitsDuplicates or params_.doParamDuplicates) {
      constexpr uint32_t tracksPerBlock = 8;
      constexpr uint32_t comparisonsPerBlock = 32;

      // alpaka maps the LAST Vec2D component to CUDA x: blocksX ends up on grid y (max 65535), as blocksY does on x;
      // both loops in Kernel_trackDuplicates are grid-stride, so capping both extents is safe
      auto const blocksX = std::min(cms::alpakatools::divide_up_by(nTracks, tracksPerBlock), 65535u);

      auto const blocksY = std::min(cms::alpakatools::divide_up_by(nTracks, comparisonsPerBlock), 65535u);

      Vec2D const blocks{blocksX, blocksY};
      Vec2D const threads{tracksPerBlock, comparisonsPerBlock};

      auto const workDiv2D = cms::alpakatools::make_workdiv<Acc2D>(blocks, threads);

#ifdef GPU_DEBUG
      alpaka::wait(queue);
      printf("filterTracks: nTracks %d, blocksX %d, tracksPerBlock %d\n", nTracks, blocksX, tracksPerBlock);
#endif

      alpaka::exec<Acc2D>(queue,
                          workDiv2D,
                          Kernel_trackDuplicates{},
                          tracks_d_->view().tracks(),
                          tracks_d_->view().trackHits(),
                          params_.doSameHitsDuplicates,
                          params_.doParamDuplicates,
                          params_.matchFraction,
                          params_.dupMinHits,
                          params_.dupNSigma2,
                          params_.dupMaxDeltaR2,
                          params_.dupPtDifference,
                          duplicate_d_->data());

      constexpr uint32_t threadsPerBlock = 128;
      auto const blocks1D = cms::alpakatools::divide_up_by(nTracks, threadsPerBlock);
      auto const workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocks1D, threadsPerBlock);

      alpaka::exec<Acc1D>(
          queue, workDiv, Kernel_applyTrackDuplicates{}, tracks_d_->view().tracks(), duplicate_d_->data());
#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Kernel_trackDuplicates + Kernel_applyTrackDuplicates -> done!" << std::endl;
#endif
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
