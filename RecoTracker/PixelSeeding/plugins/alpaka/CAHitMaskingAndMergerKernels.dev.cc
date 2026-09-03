// C++ headers
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
#include "CAHitMaskingAndMergerKernels.h"
#include "CAHitMaskingAndMergerKernelsImpl.h"

// #define GPU_DEBUG
// #define NTUPLE_DEBUG
//#define CA_STATS

namespace ALPAKA_ACCELERATOR_NAMESPACE {

    CAHitMaskingAndMergerKernels::CAHitMaskingAndMergerKernels(uint32_t maxTracks, Queue &queue)
    {
        iterGood_ = CAPairSoACollection(queue, maxTracks);
        totCounters_ = cms::alpakatools::make_device_buffer<uint32_t []>(queue, 2u);
        iterGoodHits_ = cms::alpakatools::make_device_buffer<uint32_t []>(queue,maxTracks);

        totTracks_ = cms::alpakatools::make_device_view(queue, *reinterpret_cast<uint32_t *>(totCounters_->data()));
        totHits_ = cms::alpakatools::make_device_view(queue, *reinterpret_cast<uint32_t *>(totCounters_->data() + 1));

        // alpaka::memset(queue, iterGoodHits_, 0);
        alpaka::memset(queue, *totTracks_, 0);
        alpaka::memset(queue, *totHits_, 0);
    }

  void CAHitMaskingAndMergerKernels::updateMasking(Queue &queue,
                                                   ::reco::TrackingRecHitsMaskingView &mask_view,
                                                   const ::reco::TrackSoAConstView &trackd_view,
                                                   const ::reco::TrackHitSoAConstView &trackhitd_view,
                                                   pixelTrack::Quality minQuality,
                                                   uint32_t iterationIndex) {
    using namespace caHitMaskingAndMergerKernels;

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Starting CAHitMaskingAndMergerKernels::updateMasking" << std::endl;
#endif

    int threadsPerBlock = 128;
    int blocks = cms::alpakatools::divide_up_by(trackd_view.metadata().size(), threadsPerBlock);
    const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

    alpaka::exec<Acc1D>(
        queue, workDiv1D, Kernel_updateMasking{}, mask_view, trackd_view, trackhitd_view, minQuality, iterationIndex);
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_updateMasking -> done!" << std::endl;
#endif
  }

   void CAHitMaskingAndMergerKernels::countGoodTracks(Queue &queue,
                                                                    ::reco::InputTracks const &allTracks,
                                                                    pixelTrack::Quality minQuality) {
    using namespace caHitMaskingAndMergerKernels;

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Starting CAHitMaskingAndMergerKernels::countGoodTracks" << std::endl;
#endif

    uint32_t totalTrackCapacity = 0;
    for (int collectionIndex = 0; collectionIndex < allTracks.nInputs; ++collectionIndex) {
      totalTrackCapacity += allTracks.views[collectionIndex].metadata().size();
    }

    auto countsDevice = cms::alpakatools::make_device_buffer<int[]>(queue, 2);
    alpaka::memset(queue, countsDevice, 0);

    if (totalTrackCapacity != 0) {
      const auto threadsPerBlock = 128u;
      const auto blocks = cms::alpakatools::divide_up_by(totalTrackCapacity, threadsPerBlock);
      const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_countGoodTracks{},
                          allTracks,
                          minQuality,
                          totTracks_->data(),
                          totHits_->data(),
                          iterGoodHits_->data(),
                          iterGood_->view());
    }

    auto totCountersHost  = cms::alpakatools::make_host_buffer<uint32_t[]>(queue, 2);
    // auto totCounterDeviceV = cms::alpakatools::make_device_view<uint32_t>(totCounters_->data(), 2);
    alpaka::memcpy(queue, totCountersHost, *totCounters_);
    alpaka::wait(queue);

#ifdef GPU_DEBUG
    std::cout << "Kernel_countGoodTracks -> done!" << std::endl;
    std::cout << "Total good tracks: " << totCountersHost[0] << ", total hits: " << totCountersHost[1] << std::endl;
#endif

    tracks_d_ = reco::TracksSoACollection(queue, totCountersHost[0], totCountersHost[1]);

    if (totCountersHost[0] > 0) {
      constexpr auto threadsPrefixScan = 256u;
      auto blocksPrefixScan = (totCountersHost[0] + threadsPrefixScan - 1) / threadsPrefixScan;
      auto workDivPrefixScan = cms::alpakatools::make_workdiv<Acc1D>(blocksPrefixScan, threadsPrefixScan);
      auto bCounter = cms::alpakatools::make_device_buffer<int32_t>(queue);
      alpaka::memset(queue, bCounter, 0);

      // Launch to build the hit offsets for the selected tracks
      // tracks().hitOffsets() holds the ending offset of each track hits vector
      // so here we simply do the prefix sum of the number of hits per track
      alpaka::exec<Acc1D>(queue,
                          workDivPrefixScan,
                          cms::alpakatools::multiBlockPrefixScan<uint32_t>(),
                          iterGoodHits_->data(),
                          tracks_d_->view().tracks().hitOffsets().data(),
                          totCountersHost[0],
                          blocksPrefixScan,
                          bCounter.data(),
                          alpaka::getPreferredWarpSize(alpaka::getDev(queue)));
    }

  }

  void CAHitMaskingAndMergerKernels::fillGoodTracks(Queue &queue,
                                                    ::reco::InputTracks const &allTracks) {
    using namespace caHitMaskingAndMergerKernels;
    
    if (tracks_d_->view().tracks().metadata().size() > 0) {

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Starting CAHitMaskingAndMergerKernels::fillGoodTracks" << std::endl;
#endif
      const auto threadsPerBlock = 128u;
      const auto blocks = cms::alpakatools::divide_up_by(tracks_d_->view().tracks().metadata().size(), threadsPerBlock);
      const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_fillGoodTracks{},
                          allTracks,
                          iterGood_->view(),
                          iterGoodHits_->data(),
                          tracks_d_->view().tracks(),
                          tracks_d_->view().trackHits());
    }
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_fillGoodTracks -> done!" << std::endl;
#endif

}

  void CAHitMaskingAndMergerKernels::filterTracks(Queue &queue,
                                                  double matchFraction,
                                                  int minHitsForDuplicate) {
    using namespace caHitMaskingAndMergerKernels;

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Starting CAHitMaskingAndMergerKernels::filterTracks" << std::endl;
#endif

    auto const nTracks = tracks_d_->view().tracks().metadata().size();
    if (nTracks < 2) {
      return;
    }

    constexpr uint32_t tracksPerBlock = 8;
    constexpr uint32_t comparisonsPerTrack = 32;

    auto const blocksX = cms::alpakatools::divide_up_by(nTracks, tracksPerBlock);
    auto const blocksY = std::min(
        cms::alpakatools::divide_up_by(nTracks, comparisonsPerTrack),
        65535u);

    Vec2D const blocks{blocksX, blocksY};
    Vec2D const threads{tracksPerBlock, comparisonsPerTrack};
    auto const workDiv2D = cms::alpakatools::make_workdiv<Acc2D>(blocks, threads);

    printf("filterTracks: nTracks %d, blocksX %d, blocksY %d tracksPerBlock %d\n", nTracks, blocksX, blocksY, tracksPerBlock);
    alpaka::exec<Acc2D>(queue,
                        workDiv2D,
                        Kernel_sameHitsDuplicates{},
                        tracks_d_->view().tracks(),
                        tracks_d_->view().trackHits(),
                        matchFraction,
                        minHitsForDuplicate);
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_sameHitsDuplicates -> done!" << std::endl;
#endif

    alpaka::exec<Acc2D>(queue,
                        workDiv2D,
                        Kernel_trackParameterDuplicates{},
                        tracks_d_->view().tracks(),
                        25.f,
                        1e-4f,
                        0.15f);
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_trackParameterDuplicates -> done!" << std::endl;
#endif

  }


}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
