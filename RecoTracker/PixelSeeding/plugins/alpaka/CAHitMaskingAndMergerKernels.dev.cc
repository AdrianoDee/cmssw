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

//#define GPU_DEBUG
// #define NTUPLE_DEBUG
//#define CA_STATS

namespace ALPAKA_ACCELERATOR_NAMESPACE {

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

  void CAHitMaskingAndMergerKernels::filterTracks(Queue &queue,
                                                  ::reco::TrackSoAView &track_view,
                                                  ::reco::TrackHitSoAView &trackHit_view,
                                                  const ::reco::TrackSoAConstView &inpTrack_view,
                                                  const ::reco::TrackHitSoAConstView &inpTrackHit_view,
                                                  pixelTrack::Quality minQuality,
                                                  double matchFraction) {
    using namespace caHitMaskingAndMergerKernels;

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Starting CAHitMaskingAndMergerKernels::filterTracks" << std::endl;
#endif

    int threadsPerBlock = 128;
    int blocks = inpTrack_view.metadata().size();
    const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);
    alpaka::exec<Acc1D>(queue,
                        workDiv1D,
                        Kernel_filterTracks{},
                        track_view,
                        trackHit_view,
                        inpTrack_view,
                        inpTrackHit_view,
                        minQuality,
                        matchFraction);
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_filterTracks -> done!" << std::endl;
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
