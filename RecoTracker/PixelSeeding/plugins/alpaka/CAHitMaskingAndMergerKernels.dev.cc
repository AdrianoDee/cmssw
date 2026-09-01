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

#define GPU_DEBUG
// #define NTUPLE_DEBUG
//#define CA_STATS

namespace ALPAKA_ACCELERATOR_NAMESPACE {

    CAHitMaskingAndMergerKernels::CAHitMaskingAndMergerKernels(uint32_t maxTracks, Queue &queue)
    {
        iterGood_ = CAPairSoACollection(queue, maxTracks);
        totCounters_ = cms::alpakatools::make_device_buffer<uint32_t []>(queue, 2u);
        iterGoodHits_ = cms::alpakatools::make_device_buffer<uint32_t []>(queue,maxTracks + 1u);

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
#endif
    std::cout << "Total good tracks: " << totCountersHost[0] << ", total hits: " << totCountersHost[1] << std::endl;

    tracks_d_ = reco::TracksSoACollection(queue, totCountersHost[0], totCountersHost[1]);

        constexpr auto threadsPrefixScan = 256u;
        auto blocksPrefixScan = (totCountersHost[0] + threadsPrefixScan - 1) / threadsPrefixScan;
        auto workDivPrefixScan = cms::alpakatools::make_workdiv<Acc1D>(blocksPrefixScan, threadsPrefixScan);
        auto bCounter = cms::alpakatools::make_device_buffer<int32_t>(queue);
        alpaka::memset(queue, bCounter, 0);

        // Launch to build the hit offsets for the selected tracks
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

  void CAHitMaskingAndMergerKernels::updateHitOffsets(
      Queue &queue, int tksBeg, int tksEnd, int nHits, ::reco::TrackSoAView &trackd_view) {
    using namespace caHitMaskingAndMergerKernels;

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Starting CAHitMaskingAndMergerKernels::updateHitOffsets" << std::endl;
#endif

    int threadsPerBlock = 1;
    int blocks = 1;
    const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);
    alpaka::exec<Acc1D>(queue, workDiv1D, Kernel_updateHitOffsets{}, tksBeg, tksEnd, nHits, trackd_view);
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_updateHitOffsets -> done!" << std::endl;
#endif
  }

  void CAHitMaskingAndMergerKernels::fillGoodTracks(Queue &queue,
                                                    ::reco::InputTracks const &allTracks) {
    using namespace caHitMaskingAndMergerKernels;
    
    if (tracks_d_.view().metadata().size() != 0) {
      const auto threadsPerBlock = 128u;
      const auto blocks = cms::alpakatools::divide_up_by(tracks_d_.view().metadata().size(), threadsPerBlock);
      const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_fillGoodTracks{},
                          allTracks,
                          minQuality,
                          totTracks_->data(),
                          totHits_->data(),
                          iterGoodHits_->data(),
                          iterGood_->view());
    }
}

  void CAHitMaskingAndMergerKernels::filterTracks(Queue &queue,
                                                  ::reco::TrackSoAView &track_view,
                                                  ::reco::TrackHitSoAView &trackHit_view,
                                                //   const ::reco::TrackSoAConstView &inpTrack_view,
                                                //   const ::reco::TrackHitSoAConstView &inpTrackHit_view,
                                                  pixelTrack::Quality minQuality,
                                                  double matchFraction) {
    using namespace caHitMaskingAndMergerKernels;

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Starting CAHitMaskingAndMergerKernels::filterTracks" << std::endl;
#endif

// 
    const auto threadsPerBlock = 128u;
    auto blockSize = track_view.metadata().size() / 128u;
    auto numberOfBlocks = cms::alpakatools::divide_up_by(track_view.metadata().size() * track_view.metadata().size()/2, blockSize);
    const auto rescale = numberOfBlocks / 65536;
    blockSize *= (rescale + 1);

    Vec2D const blocks{numberOfBlocks, 1u};  
    Vec2D const threads{blockSize, threadsPerBlock};
    const auto workDiv2D = cms::alpakatools::make_workdiv<Acc2D>(blocks, threads);
    alpaka::exec<Acc2D>(queue,
                        workDiv2D,
                        Kernel_filterAndMark{},
                        track_view,
                        trackHit_view,
                        // inpTrack_view,
                        // inpTrackHit_view,
                        minQuality,
                        matchFraction);
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_filterAndMark -> done!" << std::endl;
#endif
  }

  /* This will make sense when we will be able to run this once per job in Alpaka

  template <typename TrackerTraits>
  void CAHitNtupletGeneratorKernels<TrackerTraits>::printCounters() {
    auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(1,1);
    alpaka::exec<Acc1D>(queue_, workDiv1D, Kernel_printCounters{}, this->counters_->data());
  }
  */

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
