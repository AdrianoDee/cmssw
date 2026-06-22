#include <array>
#include <iostream>

#include <alpaka/alpaka.hpp>

#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/OneToManyAssoc.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "CAITExtend/CAITCandidateCollector.h"
#include "CAITExtend/CAITExtendDoublets.h"
#include "CAITExtend/CAITExtendKernels.h"
#include "CAITExtend/CAITExtendKernelsImpl.h"
#include "CAITExtend/CAITExtendLayout.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::caITExtend {

  template <typename TrackerTraits>
  Kernels<TrackerTraits>::Kernels(ExtensionConfig const& cfg,
                                  uint32_t nTracks,
                                  uint32_t offsetStubs,
                                  Queue& queue)
      : cfg_(cfg), nTracks_(nTracks), offsetStubs_(offsetStubs) {
    using namespace caStructures;
    const auto nSlots = uint64_t(nTracks) * cfg.maxLayersPerTrack;

    slots_ = cms::alpakatools::make_device_buffer<CandidateSlot[]>(queue, nSlots);
    slotLayer_ = cms::alpakatools::make_device_buffer<uint8_t[]>(queue, nSlots);
    chains_ = cms::alpakatools::make_device_buffer<ExtendedChain[]>(queue, nTracks);
    counters_ = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, kNExtCounters);

    alpaka::memset(queue, *counters_, 0);
    alpaka::memset(queue, *slots_, 0);
    alpaka::memset(queue, *slotLayer_, 0xff);
    alpaka::memset(queue, *chains_, 0);
  }

  template <typename TrackerTraits>
  void Kernels<TrackerTraits>::buildChains(TkSoAConstView tracks,
                                           HitsConstView hits,
                                           PhiBinner const* phiBinner,
                                           float bField,
                                           Queue& queue) {
    CollectorParams cp{cfg_.nSigmaPhi,
                       cfg_.nSigmaZ,
                       cfg_.floorDPhi,
                       cfg_.floorDZ,
                       cfg_.maxDPhi,
                       cfg_.maxDZ,
                       bField,
                       cfg_.materialDensity,
                       cfg_.maxLayersPerTrack,
                       offsetStubs_,
                       cfg_.sourceIteration,
                       cfg_.minQuality};
    CollectorBuffers cb{slots_->data(), slotLayer_->data()};

    const auto threadsPerBlock = 32u;
    const auto blocks = nTracks_;
    const auto work = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);
    alpaka::exec<Acc1D>(queue,
                        work,
                        kernelCollectITCandidates<TrackerTraits, LayoutTraits>{},
                        tracks,
                        hits,
                        phiBinner,
                        cp,
                        cb,
                        nTracks_,
                        counters_->data());

    ChainWalkParams wp{bField,
                       cfg_.materialDensity,
                       cfg_.kappaSigmaCut,
                       cfg_.scoreFloor2Phi,
                       cfg_.scoreFloor2Z,
                       cfg_.maxLayersPerTrack,
                       cfg_.maxNewLayers,
                       cfg_.refitMinNewHits,
                       cfg_.sourceIteration,
                       cfg_.minQuality};
    const auto wlen = cms::alpakatools::divide_up_by(nTracks_, 128u);
    const auto walk = cms::alpakatools::make_workdiv<Acc1D>(wlen, 128u);
    alpaka::exec<Acc1D>(queue,
                        walk,
                        kernelITBuildChain<TrackerTraits, LayoutTraits>{},
                        tracks,
                        hits,
                        cb,
                        wp,
                        chains_->data(),
                        nTracks_,
                        counters_->data());
  }

  template <typename TrackerTraits>
  void Kernels<TrackerTraits>::runRefit(HelixFit<TrackerTraits>& fitter,
                                        TkSoAView tracks,
                                        TkHitSoAView trackHits,
                                         TkHitSoAConstView origTrackHits,
                                         HitsConstView hits,
                                         ::reco::CAModulesConstView modules,
                                         ::reco::CAMaterialSoAConstView material,
                                         ::reco::OTRecHitsConstView otRecHits,
                                         ::reco::StubsConstView stubs,
                                        uint32_t nHits,
                                        uint32_t maxNumberOfTuples,
                                        uint16_t maxHitsOnTrack,
                                        Queue& queue) {
    using namespace caStructures;

    // Allocate refit-side buffers.
    const uint32_t contentCap = uint32_t(nTracks_) * uint32_t(origAvgHitsPerTrack_ + cfg_.maxNewLayers);
    extTuples_ = cms::alpakatools::make_device_buffer<SequentialContainer>(queue);
    extTuplesStorage_ = cms::alpakatools::make_device_buffer<SequentialContainerStorage[]>(queue, contentCap);
    extTuplesOffsets_ = cms::alpakatools::make_device_buffer<SequentialContainerOffsets[]>(queue, nTracks_ + 1);

    // Account for IT-extended tracks: nHits can grow up to maxHitsOnTrack +
    // maxNewLayers.  Multiplicity buckets [0..multNOnes-2] must cover that
    // range, so allocate maxHitsOnTrack + maxNewLayers + 2 slots.
    const uint32_t multNOnes = uint32_t(maxHitsOnTrack) + uint32_t(cfg_.maxNewLayers) + 2;
    extMult_ = cms::alpakatools::make_device_buffer<GenericContainer>(queue);
    extMultStorage_ = cms::alpakatools::make_device_buffer<GenericContainerStorage[]>(queue, nTracks_);
    extMultOffsets_ = cms::alpakatools::make_device_buffer<GenericContainerOffsets[]>(queue, multNOnes);

    typename SequentialContainer::View tuplesView{extTuples_->data(),
                                                  extTuplesOffsets_->data(),
                                                  extTuplesStorage_->data(),
                                                  nTracks_ + 1,
                                                  contentCap};
    typename GenericContainer::View multView{
        extMult_->data(), extMultOffsets_->data(), extMultStorage_->data(), multNOnes, nTracks_};

    // Initialise Tuples: sets the SequentialContainer's internal off/content
    // pointers from the view AND zeroes the offsets array.
    SequentialContainer::template launchZero<Acc1D>(tuplesView, queue);
    const auto blocks = cms::alpakatools::divide_up_by(nTracks_, 128u);
    const auto work = cms::alpakatools::make_workdiv<Acc1D>(blocks, 128u);
    alpaka::exec<Acc1D>(queue,
                        work,
                        kernelComputeNewSizes<TrackerTraits>{},
                        TkSoAConstView{tracks},
                        chains_->data(),
                        extTuplesOffsets_->data(),
                        cfg_.refitMinNewHits,
                        cfg_.sourceIteration,
                        nTracks_);
    cms::alpakatools::iterativePrefixScan<Acc1D>(
        extTuplesOffsets_->data(), extTuplesOffsets_->data(), nTracks_ + 1, queue);

    alpaka::exec<Acc1D>(queue,
                        work,
                        kernelWriteExtendedContent<TrackerTraits>{},
                        TkSoAConstView{tracks},
                        origTrackHits,
                        chains_->data(),
                        extTuplesOffsets_->data(),
                        extTuplesStorage_->data(),
                        cfg_.refitMinNewHits,
                        cfg_.sourceIteration,
                        nTracks_);

    // Build TupleMultiplicity (count + finalize + fill).
    GenericContainer::template launchZero<Acc1D>(multView, queue);
    alpaka::exec<Acc1D>(queue,
                        work,
                        kernelCountMultiplicity<TrackerTraits>{},
                        extTuplesOffsets_->data(),
                        extMult_->data(),
                        nTracks_);
    GenericContainer::template launchFinalize<Acc1D>(multView, queue);
    alpaka::exec<Acc1D>(queue,
                        work,
                        kernelFillMultiplicity<TrackerTraits>{},
                        extTuplesOffsets_->data(),
                        extMult_->data(),
                        nTracks_);

    // Refit through the existing HelixFit instance.
    fitter.deallocate();
    fitter.allocate(extMult_->data(), tracks, extTuples_->data());
    fitter.launchBrokenLineKernels(
        hits, modules, material, nHits, maxNumberOfTuples, queue, otRecHits, stubs, offsetStubs_);

    // Rebuild TrackHitSoA + update hitOffsets + stamp iteration.
    alpaka::exec<Acc1D>(queue,
                        work,
                        kernelFillExtendedTrackHits<TrackerTraits>{},
                        tracks,
                        trackHits,
                        chains_->data(),
                        extTuplesOffsets_->data(),
                        extTuplesStorage_->data(),
                        hits,
                        cfg_.sourceIteration,
                        cfg_.extendedIteration,
                        cfg_.refitMinNewHits,
                        nTracks_,
                        counters_->data());
  }

  // Host-side counter readback + log.  Forces a sync via alpaka::wait.
  template <typename TrackerTraits>
  void Kernels<TrackerTraits>::logCounters(Queue& queue) const {
    std::array<uint32_t, kNExtCounters> host{};
    auto hostView = cms::alpakatools::make_host_view(host.data(), kNExtCounters);
    alpaka::memcpy(queue, hostView, *counters_);
    alpaka::wait(queue);
    const auto totalValid = host[10] + host[11] + host[12];
    std::cout << "[ITExtend]"
              << " | funnel: valid=" << totalValid
              << " rejIter=" << host[10]
              << " rejQual=" << host[11]
              << " passed=" << host[12]
              << " (=extended+fallback=" << (host[3] + host[4]) << ")"
              << " | crossings: anyDisk=" << host[13]
              << " anyBarrel=" << host[14]
              << " | chain: extended=" << host[3]
              << " fallback=" << host[4]
              << " kappaRej=" << host[2]
              << " | hist: ext1=" << host[5]
              << " ext2=" << host[6]
              << " ext3=" << host[7]
              << " ext4=" << host[8]
              << " sumHits=" << host[9]
              << " | cand=" << host[0]
              << " dropped=" << host[1]
              << " | writeback: trackHitsTotal=" << host[15]
              << " newHitsWritten=" << host[16]
              << " (expect=" << host[9] << ")"
              << std::endl;
  }

  // Explicit instantiations -- one per TrackerTraits that CAHitNtupletGenerator
  // is instantiated for.  Only Phase2OTStubs has a meaningful runtime path
  // (cfg.enable defaults to false elsewhere); the others compile to dead code
  // but the symbols are still required by the linker because they appear in
  // CAHitNtupletGenerator<TT>::makeTuplesAsync's template body.
  template class Kernels<pixelTopology::Phase1>;
  template class Kernels<pixelTopology::Phase2>;
  template class Kernels<pixelTopology::Phase2OT>;
  template class Kernels<pixelTopology::Phase2OTStubs>;
  template class Kernels<pixelTopology::HIonPhase1>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::caITExtend
