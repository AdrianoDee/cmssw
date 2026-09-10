// Orchestration translation unit of the split BrokenLine fit build. The heavy per-N device kernels and
// the launcher helpers live in BrokenLineFitKernels.h and are explicitly instantiated in the
// BrokenLineFit_*.dev.cc translation units; here they are extern-template calls only. This file keeps
// the launcher orchestration (HelixFit members), the debug dump and twin-refit kernels, and the HelixFit
// explicit instantiations. The split is a build-time division: the launches keep the same order,
// arguments and grids they would have in a single translation unit.
#include "BrokenLineFitKernels.h"
#include "FWCore/Utilities/interface/isFinite.h"  // bit-pattern finiteness test for the failed-refit guard
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"  // parallel hit compaction (Counts->scan->Scatter)
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"  // dedup union-refit scratch output SoA

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Debug dump of the first N fitted tracks (BL output), gated by the HelixFit::setVerboseDump switch.
  // Converts the internal 5-parameter state (phi0, d0, 1/pT, cotTheta, z0) into pT [GeV] and eta.
  class Kernel_BLFitDump {
  public:
    // tagId is an integer rendered inline so different fit invocations can be told apart in the log. Set it via
    // HelixFit::setVerboseDump(true) and the per-call argument. Passing only integers/floats keeps the kernel
    // portable (no host-string pointers crossing the host/device boundary).
    Kernel_BLFitDump(uint32_t n, uint32_t tagId) : nToPrint_(n), tagId_(tagId) {}
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, ::reco::TrackSoAView tracks) const {
      if (alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0] != 0)
        return;
      const uint32_t total = uint32_t(std::max(0, tracks.nTracks()));
      const uint32_t cap = (nToPrint_ < total) ? nToPrint_ : total;
      printf("[BL-fit tag=%u] post-fit dump: nTracks=%u, showing first %u\n", tagId_, total, cap);
      for (uint32_t i = 0; i < cap; ++i) {
        const float phi0 = tracks[i].state()(0);
        const float d0 = tracks[i].state()(1);
        // state(2) is 1/pT in 1/GeV (CMSSW BL convention; see TracksSoA.h).
        // pT[GeV] = 1 / |state(2)|.  Sign of state(2) carries the charge.
        const float invPt = tracks[i].state()(2);
        const float cotTheta = tracks[i].state()(3);
        const float z0 = tracks[i].state()(4);
        const float absInvPt = std::abs(invPt);
        const float pT = (absInvPt > 1.e-9f) ? (1.f / absInvPt) : 1.e9f;
        const float charge = (invPt >= 0.f) ? +1.f : -1.f;
        const float eta = std::asinh(cotTheta);
        const float chi2 = tracks[i].chi2();
        const int q = int(tracks[i].quality());
        // Reported track uncertainties: covariance() is the packed Vector15f; idx 0 = sigma^2(phi),
        // idx 5 = sigma^2(d0)=sigma^2(dxy) (see DataFormats/TrackSoA/interface/TracksSoA.h).
        const float sdxy = 1.e4f * std::sqrt(std::abs(float(tracks[i].covariance()(5))));  // um
        const float sphi = std::sqrt(std::abs(float(tracks[i].covariance()(0))));          // rad
        // longitudinal: cov(14)=var(Zip=z0)=var(dz), cov(12)=var(cotTheta) (see TracksSoA copyFromCircle).
        const float sdz = 1.e4f * std::sqrt(std::abs(float(tracks[i].covariance()(14))));  // um, sigma(dz)
        const float scot = std::sqrt(std::abs(float(tracks[i].covariance()(12))));         // sigma(cotTheta)
        printf(
            "[BL-fit tag=%u] i=%u q=%d chi2=%.3f phi0=%.4f d0=%.5f invPt=%.6f cotTheta=%.4f z0=%.4f "
            "-> pT=%.3f GeV q=%+0.0f eta=%.4f  sigma(dxy)=%.2f um sigma(phi)=%.5f rad  sigma(dz)=%.2f um "
            "sigma(cotTheta)=%.6f\n",
            tagId_,
            i,
            q,
            chi2,
            phi0,
            d0,
            invPt,
            cotTheta,
            z0,
            pT,
            charge,
            eta,
            sdxy,
            sphi,
            sdz,
            scot);
      }
    }

  private:
    uint32_t nToPrint_;
    uint32_t tagId_;
  };

  // The CA main fit: one full sweep of the N-binned BLFastFit and BLFit kernels, the factorized fast
  // BrokenLine fit (circle + line), for every CA iteration and every topology. It has one linearization,
  // so it is a single sweep with no phase split and no re-linearization reference buffer.
  template <typename TrackerTraits>
  void HelixFit<TrackerTraits>::launchBrokenLineKernels(const ::reco::TrackingRecHitConstView& hv,
                                                        const ::reco::CAModulesConstView& cm,
                                                        uint32_t hitsInFit,
                                                        uint32_t maxNumberOfTuples,
                                                        Queue& queue) {
    ALPAKA_ASSERT_ACC(tuples_);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Starting HelixFit<TrackerTraits>::launchBrokenLineKernels" << std::endl;
#endif

    // The launch grid sizes off maxNumberOfTuples, the host-known per-event tuple capacity (the
    // hit-count-scaled cap the CA sizes its tuple/multiplicity containers to; not a device readback). The
    // exact per-N-bin population lives only device-side (tupleMultiplicity); a tighter trim would need a
    // further D2H sync, which this path does not take. Grid sizing cannot move a result: the grid-stride
    // loop plus the invalid-tkid break make the fitted-lane set independent of the grid size.
    //  Fit internals
    auto tkidDevice =
        cms::alpakatools::make_device_buffer<typename caStructures::tindex_type[]>(queue, maxNumberOfConcurrentFits_);
    constexpr auto maxN = TrackerTraits::maxHitsOnTrackForFullFit;
    auto hitsDevice = cms::alpakatools::make_device_buffer<double[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix3xNd<maxN>) / sizeof(double));
    auto hits_geDevice = cms::alpakatools::make_device_buffer<float[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix6xNf<maxN>) / sizeof(float));
    auto fast_fit_resultsDevice = cms::alpakatools::make_device_buffer<double[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(riemannFit::Vector4d) / sizeof(double));
    // Per-fit solver scratch of the factorized fit: prepared-data + shared band block + helper vectors
    // per lane, in an [element][lane] layout whose stride is the concurrent-fit count, so it is allocated
    // at maxNumberOfConcurrentFits_ lanes (like the input buffers). The refit-ladder allocation
    // (refitExtended) is GBL-only and keeps its own quota.
    constexpr int kScratchPerFit = brokenline::kLegacyFitScratchDoubles<int(maxN)>;
    auto gblScratchDevice = cms::alpakatools::make_device_buffer<double[]>(
        queue, std::size_t(maxNumberOfConcurrentFits_) * std::size_t(kScratchPerFit));
    // Dynamic-partition bookkeeping of the fused ladder (see the block comment at the head of
    // BrokenLineFitKernels.h). One uint32 allocation holding, in order: the per-bin round cursor
    // [0, kNBins), the per-bin {firstLane, nLanes, tupleBase} ranges, and the per-block
    // {bin, firstLane, endLane} dispatch table (a few kB, caching-allocator backed). No memset:
    // Kernel_BLMainLaneRanges writes every word it later reads (the cursor on the first round, the ranges
    // and the block map on every round).
    constexpr uint32_t kNBins = kMainNBins<TrackerTraits>;
    constexpr uint32_t kFusedBlk = kMainFusedBlocks<TrackerTraits>;
    constexpr uint32_t kPartWords = kNBins + kMainRangeStride * kNBins + kMainBlockMapStride * kFusedBlk;
    auto partDevice = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, kPartWords);
    uint32_t* pCursor = partDevice.data();
    uint32_t* pRange = pCursor + kNBins;
    uint32_t* pBlockMap = pRange + kMainRangeStride * kNBins;
    // Fused grid: kMainFusedBlocks blocks of kFitBlock, the width that covers every partition the
    // per-event populations can produce. Fixed and host-known; the device table says which blocks are
    // live and on which bin.
    const WorkDiv1D workDivFused = cms::alpakatools::make_workdiv<Acc1D>(kFusedBlk, kFitBlock);

    // Launch state of the fused ladder (see BrokenLineFitKernels.h). The fused fast-fit and fit
    // launchers are explicitly instantiated one phase per TU so nvcc compiles them in parallel.
    const BLMainLaunchCtx<TrackerTraits> ctx{queue,
                                             tuples_,
                                             tupleMultiplicity_,
                                             hv,
                                             cm,
                                             outputSoa_,
                                             bField_,
                                             rhoMap_,
                                             bMap_,
                                             tkidDevice.data(),
                                             hitsDevice.data(),
                                             hits_geDevice.data(),
                                             fast_fit_resultsDevice.data(),
                                             gblScratchDevice.data(),
                                             fitCorrections_,
                                             pRange,
                                             pBlockMap,
                                             workDivFused};

    // Round-count bound of the fused ladder, driven by the tuple-multiplicity per-N-bin offsets: stop
    // at the runtime fitted population instead of the hit-scaled tuple cap. A round beyond it is exactly
    // the on-device no-op the launched kernel would have performed (each thread's tuple_idx >= totTK on
    // the first lane -> writes the invalid sentinel and breaks -> no SoA write), so this cannot move a
    // result.
    constexpr uint32_t kNOff = TrackerTraits::maxHitsOnTrack + 2u;  // == the offsets buffer length
    uint32_t offHost[kNOff] = {0};
    uint32_t chunkBound = maxNumberOfTuples;  // no offsets available: iterate the full hit-scaled cap
    if (hostTupleMultiplicityOffsets_ != nullptr) {
      // The caller (the producer's acquire) already read the offsets back with one async D2H and the
      // framework's acquire->produce boundary guarantees it has landed -- consuming the host values
      // here costs no memcpy and no wait. off[b] is the same array Kernel_BLFastFit reads for
      // totTK = off[nHitsH+1]-off[nHitsL], so the host populations are exactly what the on-device
      // break condition sees; nothing is re-derived here.
      for (uint32_t b = 0; b < kNOff; ++b)
        offHost[b] = hostTupleMultiplicityOffsets_[b];
      // Total fitted population off[maxHitsOnTrack]-off[3] (tuples with selected-hit count in
      // [3, maxHitsOnTrack-1]). In practice it is well below the lane stride, so one round drains it.
      const uint32_t lo = offHost[3];
      const uint32_t hi = offHost[TrackerTraits::maxHitsOnTrack];
      chunkBound = (hi > lo) ? (hi - lo) : 0u;
    }
    {
      // The fused ladder. Per round: 1 range kernel + 1 fused fast fit + 1 fused fit, instead of the
      // per-bin ladder's 2 * kMainNBins launches per chunk. One round seats maxNumberOfConcurrentFits_
      // tuples and the demand can never exceed the population chunkBound counts, so
      // ceil(chunkBound / maxNumberOfConcurrentFits_) rounds always drain it and no tuple is left unfit.
      // chunkBound == 0 (no fitted population) runs zero rounds.
      const uint32_t nRounds = cms::alpakatools::divide_up_by(chunkBound, maxNumberOfConcurrentFits_);
      for (uint32_t round = 0; round < nRounds; ++round) {
        // Dynamic partition: prefix-sum the per-bin populations, read straight off tupleMultiplicity with
        // no count kernel, readback or sync, into lane ranges and the block -> bin dispatch table.
        alpaka::exec<Acc1D>(queue,
                            cms::alpakatools::make_workdiv<Acc1D>(1u, 1u),
                            Kernel_BLMainLaneRanges<TrackerTraits>{},
                            tupleMultiplicity_,
                            pCursor,
                            pRange,
                            pBlockMap,
                            maxNumberOfConcurrentFits_,
                            /*firstRound=*/round == 0u);
        runMainFusedFast<TrackerTraits>(ctx);
        runMainFusedFit<TrackerTraits>(ctx);
      }
    }

    if (verboseDump_) {
      // tag=1 -> post-fit dump of the main-fit launch
      alpaka::exec<Acc1D>(
          queue, cms::alpakatools::make_workdiv<Acc1D>(1u, 1u), Kernel_BLFitDump{verboseDumpN_, 1u}, outputSoa_);
    }
  }

  // Extended-N refit launcher. Rolling N=3..kRefitMaxN over the accepted-extended tuple population
  // (compacted per N-bin via the fast-fit kernel's grid atomic), running the phase-split GBL fit +
  // SoA writeback (overwrites state/cov/chi2/pt/eta/ndof of the extended tracks; hit lists / nLayers /
  // hitOffsets are left as the rewrite wrote them). Every N up to kRefitMaxN stays under the per-thread
  // frame ceiling, so no launch costs a local-memory reservation. Only compiled/launched for
  // Phase2OTStubs (the sole topology the OT extension runs on); a no-op for every other traits set.
  template <typename TrackerTraits>
  void HelixFit<TrackerTraits>::refitExtended(const ::reco::TrackingRecHitConstView& hv,
                                              const ::reco::CAModulesConstView& cm,
                                              caStructures::SequentialContainer const* hitContainer,
                                              const int32_t* acceptedByTuple,
                                              const caExtension::OTHitsSource* otSource,
                                              uint32_t maxNumberOfTuples,
                                              Queue& queue) {
    if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
      ALPAKA_ASSERT_ACC(tuples_);
      const caExtension::OTHitsSource otSrcVal =
          (otSource != nullptr && otSource->nOTHits > 0u) ? *otSource : caExtension::OTHitsSource{};
      constexpr auto maxN = int(kRefitMaxN);
      // Refit lane count = its own stride (kRefitStride, not the global one); every buffer below
      // sizes/strides off it. The fused ladder partitions these nt lanes across the ten N-bins, with the
      // split computed per event on the device (see the block comment in BrokenLineFitKernels.h), so nt
      // is how many tracks one pass seats -- what does not fit is seated by the next round.
      constexpr uint32_t nt = kRefitStride;

      // Own buffer set: the Eigen Map stride is kRefitStride (the refit kernels are instantiated at that
      // stride), sized to maxN. Caching-allocator backed -> transient, recycled across events/streams.
      auto tkidDevice = cms::alpakatools::make_device_buffer<typename caStructures::tindex_type[]>(queue, nt);
      auto hitsDevice = cms::alpakatools::make_device_buffer<double[]>(
          queue, nt * sizeof(riemannFit::Matrix3xNd<maxN>) / sizeof(double));
      auto hits_geDevice = cms::alpakatools::make_device_buffer<float[]>(
          queue, nt * sizeof(riemannFit::Matrix6xNf<maxN>) / sizeof(float));
      auto fast_fit_resultsDevice =
          cms::alpakatools::make_device_buffer<double[]>(queue, nt * sizeof(riemannFit::Vector4d) / sizeof(double));
      // GBL node workspace, per-lane stride = the widest node chain a lane can build: the exact two-thin
      // split's 2N+1 at the largest refit N (the arrival-node layout's N+2 fits inside it), so one
      // allocation serves both node layouts.
      constexpr int kNodesPerFit = generalBrokenLine::kGblSplitNodes<maxN>;
      auto gnodesDevice = cms::alpakatools::make_device_buffer<generalBrokenLine::GblNodeData[]>(
          queue, std::size_t(nt) * std::size_t(kNodesPerFit));
      // The gFullDelta term is absent: it overlays the head of the band region.
      constexpr int kScratchPerFit = generalBrokenLine::kGblScratchDoubles<kNodesPerFit - 1> + 3 * kNodesPerFit;
      auto gblScratchDevice =
          cms::alpakatools::make_device_buffer<double[]>(queue, std::size_t(nt) * std::size_t(kScratchPerFit));
      // Dynamic-partition bookkeeping. One allocation of 2 * kRefitNBins uint32: [0, kRefitNBins) are
      // the per-bin demands Kernel_BLRefitBinCount fills, [kRefitNBins, 2*kRefitNBins) the per-bin
      // claim counters the scans atomically advance -- one array, so one memset covers both.
      auto counterDevice = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, 2u * kRefitNBins);
      // Per-bin {firstLane, nLanes} of the current round, and the per-block {bin, firstLane, endLane}
      // dispatch table of the fused grid. Both written on the device by Kernel_BLRefitLaneRanges.
      auto rangeDevice =
          cms::alpakatools::make_device_buffer<uint32_t[]>(queue, std::size_t(kRefitRangeStride) * kRefitNBins);
      auto blockMapDevice =
          cms::alpakatools::make_device_buffer<uint32_t[]>(queue, std::size_t(kRefitBlockMapStride) * kFusedBlocks);
      // Per-track "already seated in an earlier round" flag (one byte per track).
      auto servedDevice =
          cms::alpakatools::make_device_buffer<uint8_t[]>(queue, std::max<uint32_t>(1u, maxNumberOfTuples));
      // Per-lane phase buffer of the phase-split GBL fit ladder (kBLPhaseDoubles doubles/lane; see
      // the Kernel_BLFitPhase* kernels). Caching-allocator backed like the other refit scratch.
      auto phaseDevice =
          cms::alpakatools::make_device_buffer<double[]>(queue, std::size_t(nt) * std::size_t(kBLPhaseDoubles));
      // Per-lane fit-hit-id table (fitHitId[lane*N + i]), wired only when the caller asked for the
      // fit-rejected hit to be removed from the emitted list; otherwise pFitHitId stays null and
      // Kernel_BLFastFitRefit skips the write.
      const bool dropOutlierActive = (dropOutlierHitId_ != nullptr);
      auto fitHitIdDevice = cms::alpakatools::make_device_buffer<uint32_t[]>(
          queue, dropOutlierActive ? std::size_t(nt) * std::size_t(maxN) : std::size_t(1));
      uint32_t* const pFitHitId = dropOutlierActive ? fitHitIdDevice.data() : nullptr;
      // Core-protected outlier: per-lane pixel-core flag table (fitHitIsCore[lane*N + i]), allocated only
      // when the caller asked for core protection. Independent of the fit-hit-id/drop-id table above (it
      // carries the core flag, not the drop id); null when off, in which case Kernel_BLFastFitRefit skips
      // the write and the outlier scan drops the worst node whatever it is.
      const bool coreProtectActive = refitOutlierCoreProtect_;
      auto fitHitIsCoreDevice = cms::alpakatools::make_device_buffer<uint8_t[]>(
          queue, coreProtectActive ? std::size_t(nt) * std::size_t(maxN) : std::size_t(1));
      uint8_t* const pFitHitIsCore = coreProtectActive ? fitHitIsCoreDevice.data() : nullptr;

      constexpr uint32_t blockSize = 64;  // scan/compaction work division
      // Fit block dimension kFitBlock, a launch dimension only: the fit kernels are grid-stride loops
      // over the same nt lanes and break on the invalid-tkid sentinel, so the fitted-lane set and every
      // arithmetic operation are independent of it. It applies to the fused fit work division only;
      // workDivScan keeps blockSize for Kernel_BLFastFitRefit's atomic lane claiming. A small block suits
      // this solver, which is L1/LSU-pipe bound rather than FP64-latency bound.
      // Fused grid: kFusedBlocks = nt/kFitBlock + kRefitNBins blocks, the width that covers every
      // partition the per-event count can produce, a bin of n_b lanes needing ceil(n_b/kFitBlock) blocks.
      const WorkDiv1D workDivFused = cms::alpakatools::make_workdiv<Acc1D>(kFusedBlocks, kFitBlock);
      const uint32_t numberOfBlocksScan =
          cms::alpakatools::divide_up_by(std::max<uint32_t>(1u, maxNumberOfTuples), blockSize);
      const WorkDiv1D workDivScan = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocksScan, blockSize);

      const uint32_t maxFitSel = pixelTopology::Phase2OTStubs::maxHitsOnTrack - 1;

      // Every N-bin scan launches unconditionally, with no host-side knowledge of the per-bin
      // populations: a bin with no tuple in its N-range compacts nothing and the fit kernels break on the
      // invalid-tkid sentinel, so the launch is a device-side no-op. N=3..kRefitMaxN-1 are exact bins;
      // the final N=kRefitMaxN bin absorbs the tail [kRefitMaxN, maxHitsOnTrack-1] (uniformly
      // sub-sampled). The launch state travels in a context POD (see BrokenLineFitKernels.h) so the
      // per-N launchers can be instantiated in the refit N-range TUs.
      const BLRefitLaunchCtx<TrackerTraits> rctx{queue,
                                                 hitContainer,
                                                 tupleMultiplicity_,
                                                 hv,
                                                 cm,
                                                 outputSoa_,
                                                 acceptedByTuple,
                                                 otSrcVal,
                                                 bField_,
                                                 rhoMap_,
                                                 bMap_,
                                                 outlierReject_,
                                                 maxNumberOfTuples,
                                                 workDivScan,
                                                 tkidDevice.data(),
                                                 hitsDevice.data(),
                                                 hits_geDevice.data(),
                                                 fast_fit_resultsDevice.data(),
                                                 gnodesDevice.data(),
                                                 gblScratchDevice.data(),
                                                 phaseDevice.data(),
                                                 pFitHitId,
                                                 dropOutlierHitId_,
                                                 pFitHitIsCore,
                                                 coreProtectActive,
                                                 fieldKernelWeights_,
                                                 chargeSymmetric_,
                                                 trajectoryCorrections_,
                                                 scatteringLogAtTotal_,
                                                 cumulativeEloss_,
                                                 counterDevice.data(),
                                                 rangeDevice.data(),
                                                 blockMapDevice.data(),
                                                 servedDevice.data(),
                                                 workDivFused};

      // The ladder runs as one fused launch per (phase, iteration) over all ten N-bins, on a lane
      // partition computed per event on the device from the actual populations (see the block comment
      // at the head of BrokenLineFitKernels.h): per round, 1 count + 1 range + 10 scans + 6 fit + 1
      // outlier launch.
      auto launchScan = [&](auto Ntag, uint32_t nHitsL, uint32_t nHitsH) {
        constexpr int Nv = decltype(Ntag)::value;
        runRefitScanBin<Nv, TrackerTraits>(rctx, nHitsL, nHitsH);
      };

      {
        // Fit-consistent curvature->pT conversion field (Kernel_BLFitPhaseSolve::fieldKernelWeights_): it
        // needs the map, so it is on only with both. The re-linearization's prep then takes the field
        // from the phase slot the solve published instead of re-averaging it over the hits.
        const bool fieldKernelOn = rctx.fieldKernelWeights && (rctx.bMap != nullptr);
        // One round seats at most nt = kRefitStride lanes, and seats min(remaining demand, nt) tracks;
        // ceil(maxNumberOfTuples / nt) rounds always drain the demand. A round with nothing left costs
        // only its launch overhead.
        const uint32_t nRounds = cms::alpakatools::divide_up_by(std::max<uint32_t>(1u, maxNumberOfTuples), nt);
        alpaka::memset(queue, servedDevice, 0);  // one memset per array, for the whole call
        for (uint32_t round = 0; round < nRounds; ++round) {
          alpaka::memset(queue, counterDevice, 0);
          alpaka::memset(queue, tkidDevice, 0xff);
          // Dynamic partition: count the round's per-bin demand, then prefix-sum it into lane ranges and
          // the block -> bin dispatch table. Device-only, no readback, no sync.
          alpaka::exec<Acc1D>(queue,
                              workDivScan,
                              Kernel_BLRefitBinCount{},
                              hitContainer,
                              hv,
                              acceptedByTuple,
                              servedDevice.data(),
                              counterDevice.data(),
                              maxFitSel,
                              maxNumberOfTuples,
                              /*hasStubs=*/true);
          alpaka::exec<Acc1D>(queue,
                              cms::alpakatools::make_workdiv<Acc1D>(1u, 1u),
                              Kernel_BLRefitLaneRanges{},
                              counterDevice.data(),
                              rangeDevice.data(),
                              blockMapDevice.data(),
                              nt);
          // The scans stay per-bin: each carries the grid-scope atomic slot claim that must complete
          // before any phase of that bin reads ptkids. They seat their tracks in disjoint lane ranges so
          // all ten survive to the fused fit.
          launchScan(std::integral_constant<int, 3>{}, 3u, 3u);
          riemannFit::rolling_fits<4, int(kRefitMaxN), 1>(
              [&](auto i) { launchScan(i, uint32_t(decltype(i)::value), uint32_t(decltype(i)::value)); });
          launchScan(std::integral_constant<int, int(kRefitMaxN)>{}, kRefitMaxN, maxFitSel);
          for (int iter = 0; iter < 2; ++iter) {
            const bool finalIter = (iter == 1);
            runRefitFusedPrep<TrackerTraits>(rctx, finalIter, fieldKernelOn);
            runRefitFusedSolve<TrackerTraits>(rctx, finalIter);
            runRefitFusedOut<TrackerTraits>(rctx);
          }
          // N >= 5 only: the fused outlier kernel leaves bins 0 and 1 (N = 3, 4) untouched.
          if (rctx.outlierReject)
            runRefitFusedOutlier<TrackerTraits>(rctx);
        }
      }

      if (verboseDump_)
        alpaka::exec<Acc1D>(
            queue, cms::alpakatools::make_workdiv<Acc1D>(1u, 1u), Kernel_BLFitDump{verboseDumpN_, 2u}, outputSoa_);
    } else {
      // The OT extension does not run for non-Phase2OTStubs traits: nothing to refit.
      (void)hv;
      (void)cm;
      (void)hitContainer;
      (void)acceptedByTuple;
      (void)otSource;
      (void)maxNumberOfTuples;
      (void)queue;
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Merger-side twin refit support: build a SequentialContainer over the merged track CSR so the
  // refitExtended machinery can be reused on the twin-united winners. off[0] = 0, off[t+1] = the merged
  // track's cumulative hit-end; content aliases the merged trackHits id column, bit30 OT tags preserved,
  // so no hit copy is needed.
  // ---------------------------------------------------------------------------------------------
  class Kernel_fillTwinRefitOffsets {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  caStructures::SequentialContainerOffsets* __restrict__ off,
                                  ::reco::TrackSoAConstView tracks,
                                  uint32_t nTracks) const {
      for (uint32_t t : cms::alpakatools::uniform_elements(acc, nTracks)) {
        if (t == 0)
          off[0] = 0;
        off[t + 1] = tracks[t].hitOffsets();  // cumulative CSR end of merged track t
      }
    }
  };

  // Wire the container header to its already-filled offsets and aliased content, without zeroing the
  // offsets. Single grid thread.
  class Kernel_initTwinRefitContainer {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, caStructures::SequentialContainerView view) const {
      if (alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0] == 0) {
        view.assoc->psws = 0;
        view.assoc->initStorage(view);
      }
    }
  };

  // Trajectory-order + physically-dedup the refit input of each united winner. Two union pathologies
  // break the GBL fit (which assumes the inside-out, one-measurement-per-crossing lists the CA provides),
  // showing downstream as huge chi2 / NaN covariance:
  //   (1) order: the merger's track filter appends the loser's non-shared
  //       hits after the winner's, so the chain zig-zags in radius. Fix: stable insertion sort by 3D
  //       radius x^2+y^2+z^2 (monotonic along an outgoing near-prompt trajectory in both barrel and
  //       endcap; transverse r alone scrambles same-ring disk hits).
  //   (2) same physical hit under two ids: one arm attached a raw OT rechit (bit30-tagged id), the other
  //       carries the same cluster as a merged-SoA stub id. Identical positions => zero-length GBL
  //       segment => singular normal matrix. The dedup walk only merges consecutive merged-pixel pairs,
  //       never stub/OT ids. Fix: after sorting, collapse a consecutive pair closer than kDedupDsMin
  //       (3D here) when at least one member is a stub/OT id; compact the span and pad the tail with an
  //       untagged out-of-range sentinel (the dedup walk break-terminates on it).
  // content is a copy, so the output SoA hit list is untouched. Only unitedMask[t] >= 0 spans are
  // processed.
  class Kernel_sortUnitedRefitHits {
  public:
    caExtension::OTHitsSource otSource_{};

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  uint32_t* __restrict__ content,
                                  ::reco::TrackSoAConstView tracks,
                                  ::reco::TrackingRecHitConstView hh,
                                  const int32_t* __restrict__ unitedMask,
                                  uint32_t nTracks) const {
      const uint32_t nTot = uint32_t(hh.metadata().size());
      constexpr float kFar = 3.40282347e+38f;
      constexpr uint32_t kSentinel = 0x3FFFFFFFu;  // untagged, >= any real nTot -> walk break
      // Position + class of an id. Returns false for unresolvable ids (guards; sorted last).
      // isPixel = merged-SoA non-stub hit (the only class the walk itself dedups).
      auto posOf = [&](uint32_t id, float& x, float& y, float& z, bool& isPixel) -> bool {
        if (caOTHitTag::isOTId(id)) {
          const uint32_t o = caOTHitTag::otIdx(id);
          if (o >= otSource_.nOTHits)
            return false;
          x = otSource_.otHits[o].xGlobal();
          y = otSource_.otHits[o].yGlobal();
          z = otSource_.otHits[o].zGlobal();
          isPixel = false;
          return true;
        }
        if (id >= nTot)
          return false;
        x = hh[id].xGlobal();
        y = hh[id].yGlobal();
        z = hh[id].zGlobal();
        isPixel = !::reco::isStub(hh, int32_t(id));
        return true;
      };
      auto keyOf = [&](uint32_t id) -> float {
        float x, y, z;
        bool p;
        if (!posOf(id, x, y, z, p))
          return kFar;
        return x * x + y * y + z * z;
      };
      for (uint32_t t : cms::alpakatools::uniform_elements(acc, nTracks)) {
        if (unitedMask[t] < 0)
          continue;
        const uint32_t b = (t == 0) ? 0u : tracks[t - 1].hitOffsets();
        const uint32_t e = tracks[t].hitOffsets();
        // (1) stable insertion sort by 3D radius (spans <= the twin-merge hit cap, ~32)
        for (uint32_t i = b + 1; i < e; ++i) {
          const uint32_t v = content[i];
          const float kv = keyOf(v);
          uint32_t j = i;
          while (j > b && keyOf(content[j - 1]) > kv) {
            content[j] = content[j - 1];
            --j;
          }
          content[j] = v;
        }
        // (2) collapse same-physical-hit duplicates against the previous kept entry, compact, pad.
        uint32_t w = b;
        float px = 0.f, py = 0.f, pz = 0.f;
        bool pPix = false, havePrev = false;
        for (uint32_t i = b; i < e; ++i) {
          const uint32_t id = content[i];
          float x, y, z;
          bool isPix;
          const bool ok = posOf(id, x, y, z, isPix);
          if (ok && havePrev && !(isPix && pPix)) {
            const float dx = x - px, dy = y - py, dz = z - pz;
            if (dx * dx + dy * dy + dz * dz < caFitHitSel::kDedupDsMin2)
              continue;  // same physical measurement through a second id: drop
          }
          content[w++] = id;
          if (ok) {
            px = x;
            py = y;
            pz = z;
            pPix = isPix;
            havePrev = true;
          }
        }
        for (uint32_t i = w; i < e; ++i)
          content[i] = kSentinel;
      }
    }
  };

  // Failed-refit guard buffers: per united winner, the pre-refit fitted quantities the refit
  // overwrites -- state(5) + covariance(15) + chi2 + pt + eta = 22 floats (+ ndof separately).
  constexpr uint32_t kTwinSnapStride = 22;

  class Kernel_twinRefitSnapshot {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::TrackSoAConstView tracks,
                                  const int32_t* __restrict__ unitedMask,
                                  float* __restrict__ snap,
                                  int8_t* __restrict__ snapNdof,
                                  uint32_t nTracks) const {
      for (uint32_t t : cms::alpakatools::uniform_elements(acc, nTracks)) {
        if (unitedMask[t] < 0)
          continue;
        float* s = snap + std::size_t(t) * kTwinSnapStride;
        for (uint32_t k = 0; k < 5; ++k)
          s[k] = tracks[t].state()(k);
        for (uint32_t k = 0; k < 15; ++k)
          s[5 + k] = tracks[t].covariance()(k);
        s[20] = tracks[t].chi2();
        s[21] = tracks[t].pt();
        // eta needs no slot: the restore recomputes it as asinh(state(3)) -- the same formula
        // that produced the pre-refit value, from the restored cotTheta.
        snapNdof[t] = tracks[t].ndof();
      }
    }
  };

  class Kernel_twinRefitGuard {
  public:
    // dropHitId (nullable) is the per-track fit-rejected-hit id filled by the outlier phase. When this
    // guard restores a non-finite refit it also reverts the drop (kNoDrop) so the post-refit compaction
    // keeps the restored track's full pre-refit hit list -- params and hit content stay consistent.
    //
    // This kernel is the last thing that touches the merged tracks' fit results (after refitExtended,
    // before the final cross-arm dedup and its compaction), so it also enforces the merger's fit-failure
    // rule -- no track whose fit failed is good -- in two steps:
    //   1. restore. A track whose refit came out non-finite keeps its pre-refit row: that row is the CA
    //      fit that already succeeded and was already classified, so the track ends on a good fit.
    //   2. demote. If the row is still non-finite after step 1 (bad snapshot, or a track that was never
    //      refit and arrived non-finite), it is stamped Quality::bad -- the value Kernel_mergeGather
    //      stamps on the unused tail -- so every downstream quality gate drops it.
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::TrackSoAView tracks,
                                  const int32_t* __restrict__ unitedMask,
                                  const float* __restrict__ snap,
                                  const int8_t* __restrict__ snapNdof,
                                  uint32_t* __restrict__ dropHitId,
                                  uint32_t nTracks) const {
      constexpr uint32_t kNoDrop = 0xFFFFFFFFu;
      // Bit-pattern finiteness (exponent field), not an FP comparison: a `(v == v) && |v| <= FLT_MAX`
      // bracket is foldable to `true` under -Ofast / -ffinite-math-only, i.e. exactly in the regime where
      // a failed fit still has to be recognised. Same test as the classifier's guard in
      // Kernel_classifyTracks.
      auto rowFinite = [&](uint32_t i) {
        bool ok = edm::isFinite(tracks[i].chi2()) && edm::isFinite(tracks[i].pt()) && edm::isFinite(tracks[i].eta());
        for (uint32_t k = 0; ok && k < 5; ++k)
          ok = edm::isFinite(tracks[i].state()(k));
        for (uint32_t k = 0; ok && k < 15; ++k)
          ok = edm::isFinite(tracks[i].covariance()(k));
        return ok;
      };
      // Live rows only. [nTracks(), capacity) is allocator garbage whose quality() Kernel_mergeGather
      // already stamped bad, so it must be neither read nor judged here.
      const uint32_t nLive = uint32_t(std::max(0, tracks.nTracks()));
      for (uint32_t t : cms::alpakatools::uniform_elements(acc, nTracks)) {
        if (t >= nLive)
          continue;
        bool ok = rowFinite(t);
        if (!ok && unitedMask[t] >= 0) {
          if (dropHitId != nullptr)
            dropHitId[t] = kNoDrop;  // reverted refit -> do not drop its outlier from the hit list
          // Non-finite refit outcome: restore the pre-refit row (state/cov/chi2/pt + ndof; eta was
          // only rewritten together with the state, restore it from the snapshot state's cotTheta).
          const float* s = snap + std::size_t(t) * kTwinSnapStride;
          for (uint32_t k = 0; k < 5; ++k)
            tracks[t].state()(k) = s[k];
          for (uint32_t k = 0; k < 15; ++k)
            tracks[t].covariance()(k) = s[5 + k];
          tracks[t].chi2() = s[20];
          tracks[t].pt() = s[21];
          tracks[t].eta() = alpaka::math::asinh(acc, s[3]);  // state(3) = cotTheta
          tracks[t].ndof() = snapNdof[t];
          ok = rowFinite(t);  // re-test the RESTORED row: finite by construction, verified anyway
        }
        // Step 2: the track ends with a failed fit -> it is not good, whatever it was promoted to.
        if (!ok && tracks[t].quality() >= ::pixelTrack::Quality::loose)
          tracks[t].quality() = ::pixelTrack::Quality::bad;
      }
    }
  };

  // Dropped-hit compaction on the standard Counts -> inclusive-scan -> Scatter family. Which hit id each
  // track loses, dropHitId[t], is decided by the fit outlier stage; these kernels only remove it from the
  // emitted list.
  //
  // Semantics: "remove at most one hit with id == dropHitId[t], the first match, keeping every other hit
  // in its original order". (i) Counts sets hitCnt[t] = span(t) - (dropHitId[t] present ? 1 : 0); a
  // set-but-absent drop removes nothing. (ii) newOff = inclusive prefix scan of hitCnt is the write
  // cursor after track t, so tracks[t].hitOffsets() = newOff[t]. (iii) Scatter forward-copies the kept
  // hits, in order, into [newOff[t-1], newOff[t]). (iv) hitCnt is memset to 0 over the scan capacity so
  // tail slots [nTracks, cap) stay 0 and the unused tail's hitOffsets is left untouched. The compaction
  // only removes, so newOff <= origOff everywhere; the Scatter reads from a stable id/detId/attached
  // snapshot, which keeps parallel per-track writes from clobbering another track's not-yet-read source.

  // Snapshot the emitted hit columns before the in-place compacted rewrite.
  class Kernel_e3SnapshotHits {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::TrackHitSoAConstView trackHits,
                                  uint32_t* __restrict__ snapId,
                                  uint32_t* __restrict__ snapDetId,
                                  uint8_t* __restrict__ snapAttached,
                                  uint32_t nHitsTot) const {
      for (uint32_t k : cms::alpakatools::uniform_elements(acc, nHitsTot)) {
        snapId[k] = trackHits[k].id();
        snapDetId[k] = trackHits[k].detId();
        snapAttached[k] = trackHits[k].attached();
      }
    }
  };

  // Counts: per-track kept-hit count and snapshot of the original per-track hit-end (hitOffsets is rewritten
  // by the Scatter, so the source spans must be read from origOff). hitCnt is memset to 0 over the scan
  // capacity by the launcher.
  class Kernel_e3CompactCounts {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::TrackSoAConstView tracks,
                                  ::reco::TrackHitSoAConstView trackHits,
                                  const uint32_t* __restrict__ dropHitId,
                                  int32_t* __restrict__ hitCnt,
                                  uint32_t* __restrict__ origOff,
                                  uint32_t nTracksCap) const {
      (void)nTracksCap;
      constexpr uint32_t kNoDrop = 0xFFFFFFFFu;
      const uint32_t nTracks = uint32_t(std::max(0, tracks.nTracks()));
      for (uint32_t t : cms::alpakatools::uniform_elements(acc, nTracks)) {
        const uint32_t rBeg = (t == 0u) ? 0u : tracks[t - 1].hitOffsets();
        const uint32_t rEnd = tracks[t].hitOffsets();
        origOff[t] = rEnd;
        const uint32_t drop = dropHitId[t];
        bool present = false;
        if (drop != kNoDrop) {
          for (uint32_t r = rBeg; r < rEnd; ++r) {
            if (trackHits[r].id() == drop) {
              present = true;
              break;  // remove exactly one hit: the first match
            }
          }
        }
        hitCnt[t] = int32_t((rEnd - rBeg) - (present ? 1u : 0u));
      }
    }
  };

  // Scatter: forward-copy each track's kept hits from the snapshot into its compacted span [newOff[t-1],
  // newOff[t]) (newOff = inclusive scan of hitCnt), skipping the first id==drop match, then set hitOffsets.
  class Kernel_e3CompactScatter {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::TrackSoAView tracks,
                                  ::reco::TrackHitSoAView trackHits,
                                  const uint32_t* __restrict__ snapId,
                                  const uint32_t* __restrict__ snapDetId,
                                  const uint8_t* __restrict__ snapAttached,
                                  const uint32_t* __restrict__ origOff,
                                  const uint32_t* __restrict__ dropHitId,
                                  const int32_t* __restrict__ newOff,
                                  uint32_t nTracksCap) const {
      (void)nTracksCap;
      constexpr uint32_t kNoDrop = 0xFFFFFFFFu;
      const uint32_t nTracks = uint32_t(std::max(0, tracks.nTracks()));
      for (uint32_t t : cms::alpakatools::uniform_elements(acc, nTracks)) {
        const uint32_t rBeg = (t == 0u) ? 0u : origOff[t - 1];
        const uint32_t rEnd = origOff[t];
        uint32_t w = (t == 0u) ? 0u : uint32_t(newOff[t - 1]);
        uint32_t drop = dropHitId[t];
        for (uint32_t r = rBeg; r < rEnd; ++r) {
          const uint32_t id = snapId[r];
          if (drop != kNoDrop && id == drop) {
            drop = kNoDrop;  // drop exactly the first match
            continue;
          }
          trackHits[w].id() = id;
          trackHits[w].detId() = snapDetId[r];
          trackHits[w].attached() = snapAttached[r];
          ++w;
        }
        tracks[t].hitOffsets() = uint32_t(newOff[t]);  // == w by construction
      }
    }
  };

  template <typename TrackerTraits>
  void HelixFit<TrackerTraits>::refitMergedTwins(const ::reco::TrackingRecHitConstView& hv,
                                                 const ::reco::CAModulesConstView& cm,
                                                 OutputSoAView mergedTracks,
                                                 OutputHitSoAView mergedHits,
                                                 const int32_t* unitedMask,
                                                 const caExtension::OTHitsSource* otSource,
                                                 uint32_t nTracksCap,
                                                 Queue& queue) {
    if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
      if (nTracksCap == 0)
        return;
      // Build the merged-CSR SequentialContainer, transient and caching-allocator backed, so the frees
      // are stream-ordered after the refit kernels that read them. content is a copy of the merged
      // trackHits id column, bit30 OT tags preserved: the united winners' spans are trajectory-ordered
      // below, while the output SoA hit list keeps its own converter-facing order.
      const uint32_t nHitsTot = uint32_t(mergedHits.metadata().size());
      auto contHeader = cms::alpakatools::make_device_buffer<caStructures::SequentialContainer>(queue);
      auto offBuf = cms::alpakatools::make_device_buffer<caStructures::SequentialContainerOffsets[]>(
          queue, std::size_t(nTracksCap) + 1u);
      auto contentBuf = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, std::max(nHitsTot, 1u));
      if (nHitsTot > 0) {
        auto src = cms::alpakatools::make_device_view(alpaka::getDev(queue), mergedHits.id().data(), nHitsTot);
        auto dst = cms::alpakatools::make_device_view(alpaka::getDev(queue), contentBuf.data(), nHitsTot);
        alpaka::memcpy(queue, dst, src);
      }
      caStructures::SequentialContainerView view{
          contHeader.data(), offBuf.data(), contentBuf.data(), nTracksCap + 1u, std::max(nHitsTot, 1u)};
      constexpr uint32_t blockSize = 128;
      const uint32_t nBlocks = cms::alpakatools::divide_up_by(nTracksCap, blockSize);
      alpaka::exec<Acc1D>(queue,
                          cms::alpakatools::make_workdiv<Acc1D>(nBlocks, blockSize),
                          Kernel_fillTwinRefitOffsets{},
                          offBuf.data(),
                          mergedTracks,
                          nTracksCap);
      // Trajectory-order the united winners' refit spans (see Kernel_sortUnitedRefitHits).
      const caExtension::OTHitsSource otSrcSort =
          (otSource != nullptr && otSource->nOTHits > 0u) ? *otSource : caExtension::OTHitsSource{};
      alpaka::exec<Acc1D>(queue,
                          cms::alpakatools::make_workdiv<Acc1D>(nBlocks, blockSize),
                          Kernel_sortUnitedRefitHits{otSrcSort},
                          contentBuf.data(),
                          mergedTracks,
                          hv,
                          unitedMask,
                          nTracksCap);
      alpaka::exec<Acc1D>(queue, cms::alpakatools::make_workdiv<Acc1D>(1u, 1u), Kernel_initTwinRefitContainer{}, view);
      // Failed-refit guard: snapshot the united winners' fitted quantities, run the refit, then
      // restore any row whose refit came out non-finite (a rare pathological union can leave the
      // GBL normal matrix singular even after the sort/dedup conditioning). A failed refit keeps
      // the pre-refit state instead of poisoning downstream cov consumers (vertexing, MTV pulls)
      // with NaN.
      auto snapBuf = cms::alpakatools::make_device_buffer<float[]>(queue, std::size_t(nTracksCap) * kTwinSnapStride);
      auto snapNdof = cms::alpakatools::make_device_buffer<int8_t[]>(queue, nTracksCap);
      alpaka::exec<Acc1D>(queue,
                          cms::alpakatools::make_workdiv<Acc1D>(nBlocks, blockSize),
                          Kernel_twinRefitSnapshot{},
                          mergedTracks,
                          unitedMask,
                          snapBuf.data(),
                          snapNdof.data(),
                          nTracksCap);
      // Per-merged-track dropped-hit-id buffer (allocated only when the fit-rejected hit is to be removed
      // from the emitted list; kNoDrop-initialised so untouched / non-refit tracks are no-ops).
      // refitExtended forwards dropOutlierHitId_ to the outlier kernels; the compaction below consumes it.
      // Otherwise dropOutlierHitId_ stays null: no allocation, no writes, no compaction launch.
      const bool dropOutlierOn = dropOutlierFromHitList_;
      auto dropHitIdDevice = cms::alpakatools::make_device_buffer<uint32_t[]>(
          queue, dropOutlierOn ? std::size_t(nTracksCap) : std::size_t(1));
      if (dropOutlierOn) {
        alpaka::memset(queue, dropHitIdDevice, 0xff);  // 0xffffffff == kNoDrop
        dropOutlierHitId_ = dropHitIdDevice.data();
      }
      // Point the refit output at the merged SoA + the freshly built container, then reuse the
      // extended refit (fast fit + phase-split GBL ladder). tupleMultiplicity_ is unused by refit.
      tuples_ = contHeader.data();
      outputSoa_ = mergedTracks;
      refitExtended(hv, cm, contHeader.data(), unitedMask, otSource, nTracksCap, queue);
      alpaka::exec<Acc1D>(queue,
                          cms::alpakatools::make_workdiv<Acc1D>(nBlocks, blockSize),
                          Kernel_twinRefitGuard{},
                          mergedTracks,
                          unitedMask,
                          snapBuf.data(),
                          snapNdof.data(),
                          dropOutlierHitId_,  // reverted refits clear their drop (null when nothing is removed)
                          nTracksCap);
      // Remove the fit-rejected outliers from the emitted hit list, via the parallel
      // Counts->prefix-scan->Scatter rebuild. dropOutlierHitId_ is reset afterwards so the transient buffer
      // does not dangle past this call.
      if (dropOutlierOn) {
        // Transient scratch (caching-allocator backed, stream-ordered frees after the kernels below).
        auto dropSnapId = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, std::max(nHitsTot, 1u));
        auto dropSnapDetId = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, std::max(nHitsTot, 1u));
        auto dropSnapAttached = cms::alpakatools::make_device_buffer<uint8_t[]>(queue, std::max(nHitsTot, 1u));
        auto dropOrigOff = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nTracksCap);
        auto dropHitCnt = cms::alpakatools::make_device_buffer<int32_t[]>(queue, nTracksCap);
        auto dropNewOff = cms::alpakatools::make_device_buffer<int32_t[]>(queue, nTracksCap);
        auto dropPfxCounter = cms::alpakatools::make_device_buffer<int32_t>(queue);
        alpaka::memset(queue, dropHitCnt, 0);      // tail [nTracks, cap) stays 0 -> constant scan past the last track
        alpaka::memset(queue, dropPfxCounter, 0);  // multiBlockPrefixScan inter-block counter
        if (nHitsTot > 0)
          alpaka::exec<Acc1D>(
              queue,
              cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(nHitsTot, blockSize), blockSize),
              Kernel_e3SnapshotHits{},
              mergedHits,
              dropSnapId.data(),
              dropSnapDetId.data(),
              dropSnapAttached.data(),
              nHitsTot);
        alpaka::exec<Acc1D>(queue,
                            cms::alpakatools::make_workdiv<Acc1D>(nBlocks, blockSize),
                            Kernel_e3CompactCounts{},
                            mergedTracks,
                            mergedHits,
                            dropOutlierHitId_,
                            dropHitCnt.data(),
                            dropOrigOff.data(),
                            nTracksCap);
        alpaka::exec<Acc1D>(queue,
                            cms::alpakatools::make_workdiv<Acc1D>(nBlocks, blockSize),
                            cms::alpakatools::multiBlockPrefixScan<int32_t>(),
                            dropHitCnt.data(),
                            dropNewOff.data(),
                            nTracksCap,
                            int(nBlocks),
                            dropPfxCounter.data(),
                            alpaka::getPreferredWarpSize(alpaka::getDev(queue)));
        alpaka::exec<Acc1D>(queue,
                            cms::alpakatools::make_workdiv<Acc1D>(nBlocks, blockSize),
                            Kernel_e3CompactScatter{},
                            mergedTracks,
                            mergedHits,
                            dropSnapId.data(),
                            dropSnapDetId.data(),
                            dropSnapAttached.data(),
                            dropOrigOff.data(),
                            dropOutlierHitId_,
                            dropNewOff.data(),
                            nTracksCap);
        dropOutlierHitId_ = nullptr;
      }
    } else {
      (void)hv;
      (void)cm;
      (void)mergedTracks;
      (void)mergedHits;
      (void)unitedMask;
      (void)otSource;
      (void)nTracksCap;
      (void)queue;
    }
  }

  // ===================================================================================================
  // Dedup merge-or-keep-both union refit + verdict. Adapts CMS
  // DuplicateTrackMerger's disjoint-track branch to the GPU merger: for each contested 0-shared fallback
  // pair {loser i, partner j}, build the de-duplicated union of the two tracks' hit lists, GBL-refit the
  // union (reusing the refitExtended ladder over a synthetic fixed-stride hit container + a
  // scratch output SoA), and DROP the loser only if the merged fit confirms same-particle; else keep both.
  // All of this runs only when the merger passes a non-null contested list; without one the dedup takes
  // its ordinary drop path and none of these kernels launch.
  // ===================================================================================================

  // Fill the synthetic union container's cumulative offsets: off[q] = q*stride (fixed-stride CSR).
  class Kernel_fillUnionOffsets {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  caStructures::SequentialContainerOffsets* __restrict__ off,
                                  uint32_t nRows,
                                  uint32_t stride) const {
      for (uint32_t q : cms::alpakatools::uniform_elements(acc, nRows + 1u))
        off[q] = q * stride;
    }
  };

  // Build one union hit list per contested pair p into content[p*stride .. p*stride+stride). Concatenate
  // the two tracks' id spans (skipping exact-id repeats -- 0-shared pairs normally share none), then apply
  // the SAME trajectory-order + same-physical-hit collapse conditioning as Kernel_sortUnitedRefitHits so
  // the GBL fit sees an inside-out, one-measurement-per-crossing list. Unused tail slots are padded with
  // the untagged sentinel (refitDedupWalk break-terminates on it). acceptedMask[p] = 0 => fit this slot;
  // -1 => skip (unfilled slot, or a raw union larger than the stride -- kept both, NEVER truncated).
  class Kernel_buildDedupUnions {
  public:
    caExtension::OTHitsSource otSource_{};

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  uint32_t* __restrict__ content,
                                  int32_t* __restrict__ acceptedMask,
                                  const uint32_t* __restrict__ contestedPairs,
                                  ::reco::TrackSoAConstView tracks,
                                  ::reco::TrackHitSoAConstView trackHits,
                                  ::reco::TrackingRecHitConstView hh,
                                  uint32_t nPairsCap,
                                  uint32_t stride,
                                  uint32_t* __restrict__ diag) const {
      constexpr uint32_t kUnfilled = 0xFFFFFFFFu;  // empty pair-list slot tag (matches contestedPairs memset)
      constexpr uint32_t kSentinel = 0x3FFFFFFFu;  // untagged, >= any nTot -> refitDedupWalk break
      const uint32_t nTot = uint32_t(hh.metadata().size());
      constexpr float kFar = 3.40282347e+38f;
      // Position + class of an id (raw OT rechit vs merged-SoA hit). Mirrors Kernel_sortUnitedRefitHits.
      auto posOf = [&](uint32_t id, float& x, float& y, float& z, bool& isPixel) -> bool {
        if (caOTHitTag::isOTId(id)) {
          const uint32_t o = caOTHitTag::otIdx(id);
          if (o >= otSource_.nOTHits)
            return false;
          x = otSource_.otHits[o].xGlobal();
          y = otSource_.otHits[o].yGlobal();
          z = otSource_.otHits[o].zGlobal();
          isPixel = false;
          return true;
        }
        if (id >= nTot)
          return false;
        x = hh[id].xGlobal();
        y = hh[id].yGlobal();
        z = hh[id].zGlobal();
        isPixel = !::reco::isStub(hh, int32_t(id));
        return true;
      };
      auto keyOf = [&](uint32_t id) -> float {
        float x, y, z;
        bool p;
        if (!posOf(id, x, y, z, p))
          return kFar;
        return x * x + y * y + z * z;
      };
      for (uint32_t p : cms::alpakatools::uniform_elements(acc, nPairsCap)) {
        const uint32_t base = p * stride;
        const uint32_t i = contestedPairs[2u * p + 0u];
        if (i == kUnfilled) {  // no contested pair in this slot
          acceptedMask[p] = -1;
          for (uint32_t s = base; s < base + stride; ++s)
            content[s] = kSentinel;
          continue;
        }
        const uint32_t j = contestedPairs[2u * p + 1u];
        const uint32_t iBeg = (i == 0u) ? 0u : tracks[i - 1].hitOffsets();
        const uint32_t iEnd = tracks[i].hitOffsets();
        const uint32_t jBeg = (j == 0u) ? 0u : tracks[j - 1].hitOffsets();
        const uint32_t jEnd = tracks[j].hitOffsets();
        uint32_t w = base;
        bool overflow = false;
        for (uint32_t a = iBeg; a < iEnd; ++a) {
          if (w - base >= stride) {
            overflow = true;
            break;
          }
          content[w++] = trackHits[a].id();
        }
        for (uint32_t b = jBeg; b < jEnd && !overflow; ++b) {
          const uint32_t id = trackHits[b].id();
          bool dup = false;
          for (uint32_t s = base; s < w; ++s)
            if (content[s] == id) {
              dup = true;
              break;
            }
          if (dup)
            continue;
          if (w - base >= stride) {
            overflow = true;
            break;
          }
          content[w++] = id;
        }
        if (overflow) {  // raw union exceeds the stride: keep both (over-size), never truncate + fit
          acceptedMask[p] = -1;
          for (uint32_t s = base; s < base + stride; ++s)
            content[s] = kSentinel;
          if (diag)
            alpaka::atomicAdd(acc, &diag[14], 1u, alpaka::hierarchy::Blocks{});  // over-size
          continue;
        }
        // (1) stable insertion sort by 3D radius (monotonic along an outgoing trajectory).
        for (uint32_t s = base + 1u; s < w; ++s) {
          const uint32_t v = content[s];
          const float kv = keyOf(v);
          uint32_t q = s;
          while (q > base && keyOf(content[q - 1]) > kv) {
            content[q] = content[q - 1];
            --q;
          }
          content[q] = v;
        }
        // (2) collapse same-physical-hit duplicates (two ids, one position) unless both are pixel.
        uint32_t wk = base;
        float px = 0.f, py = 0.f, pz = 0.f;
        bool pPix = false, havePrev = false;
        for (uint32_t s = base; s < w; ++s) {
          const uint32_t id = content[s];
          float x, y, z;
          bool isPix;
          const bool ok = posOf(id, x, y, z, isPix);
          if (ok && havePrev && !(isPix && pPix)) {
            const float dx = x - px, dy = y - py, dz = z - pz;
            if (dx * dx + dy * dy + dz * dz < caFitHitSel::kDedupDsMin2)
              continue;
          }
          content[wk++] = id;
          if (ok) {
            px = x;
            py = y;
            pz = z;
            pPix = isPix;
            havePrev = true;
          }
        }
        for (uint32_t s = wk; s < base + stride; ++s)
          content[s] = kSentinel;
        acceptedMask[p] = 0;  // fit this union
      }
    }
  };

  // Verdict: DROP the fallback loser i (drop[i]=1) iff the merged union fit CONFIRMS same-particle:
  //   converged (finite state) AND union-hits-outlier-dropped <= delta AND merged chi2/ndof <= the worse
  //   of the two separate fits' chi2/ndof (a relative criterion using the fits' own statistics; chi2()
  //   already stores gchi2/ndof). Unions whose fit-selected multiplicity exceeds kRefitMaxN are tail-bin
  //   sub-sampled by the ladder, so their fit is NOT trusted -> keep both (over-size). Otherwise keep both.
  //   Only sets drop (never clears) -- captured pairs entered the verdict with drop[i]==0.
  class Kernel_dedupUnionVerdict {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::TrackSoAConstView merged,             // scratch union-fit output
                                  const uint32_t* __restrict__ dropHitId,       // per-slot dropped id / kNoDrop
                                  ::reco::TrackSoAConstView tracks,             // refined input SoA (chi2/ndof)
                                  const uint32_t* __restrict__ contestedPairs,  // {i, j} per slot
                                  const int32_t* __restrict__ acceptedMask,     // -1 => build over-size/unfilled
                                  caStructures::SequentialContainer const* __restrict__ unionContainer,
                                  ::reco::TrackingRecHitConstView hh,
                                  uint8_t* __restrict__ drop,
                                  int delta,
                                  uint32_t nPairsCap,
                                  uint32_t kRefitMaxN,
                                  uint32_t* __restrict__ diag) const {
      constexpr uint32_t kUnfilled = 0xFFFFFFFFu;
      constexpr uint32_t kNoDrop = 0xFFFFFFFFu;
      // Bit-pattern finiteness, for the same reason as Kernel_twinRefitGuard's rowFinite above: a
      // `(v == v) && |v| <= FLT_MAX` bracket folds to `true` under the -ffinite-math-only this TU is
      // built with (RecoTracker/PixelSeeding/plugins/BuildFile.xml uses ofast-flag), i.e. exactly in the
      // regime where a diverged union fit still has to be recognised -- a `-inf` merged chi2 would pass
      // the bracket and then satisfy `mchi2 <= worse`, dropping a real track.
      auto finite = [](float v) { return edm::isFinite(v); };
      const bool hasStubsRt = (static_cast<int32_t>(hh.offsetStubs()) >= 0);
      for (uint32_t p : cms::alpakatools::uniform_elements(acc, nPairsCap)) {
        const uint32_t i = contestedPairs[2u * p + 0u];
        if (i == kUnfilled)
          continue;  // no pair
        if (acceptedMask[p] < 0)
          continue;  // build over-size / unfilled: already kept both (counted in the build kernel)
        const uint32_t j = contestedPairs[2u * p + 1u];
        // Fit-selected union multiplicity (what the ladder bins by). > kRefitMaxN would be tail-bin
        // sub-sampled -> not a faithful union fit -> keep both, never truncate.
        const uint32_t nSel = refitDedupWalk(unionContainer, p, hh, hasStubsRt, /*k=*/-1);
        if (nSel < 3u || nSel > kRefitMaxN) {
          if (diag)
            alpaka::atomicAdd(acc, &diag[14], 1u, alpaka::hierarchy::Blocks{});  // over-size / degenerate
          continue;
        }
        const float mchi2 = merged[p].chi2();  // normalized gchi2/ndof
        if (!(finite(mchi2) && finite(merged[p].pt()) && finite(merged[p].eta()))) {
          if (diag)
            alpaka::atomicAdd(acc, &diag[15], 1u, alpaka::hierarchy::Blocks{});  // non-finite -> keep both
          continue;
        }
        const int dropped = (dropHitId[p] != kNoDrop) ? 1 : 0;  // GBL single-hard-drop caps this at 0/1
        if (dropped > delta) {
          if (diag)
            alpaka::atomicAdd(acc, &diag[13], 1u, alpaka::hierarchy::Blocks{});  // keep both
          continue;
        }
        const float ci = tracks[i].chi2();
        const float cj = tracks[j].chi2();
        const float worse = (ci > cj) ? ci : cj;
        if (mchi2 <= worse) {
          drop[i] = 1;  // CONFIRMED same particle -> drop the fallback loser
          if (diag)
            alpaka::atomicAdd(acc, &diag[12], 1u, alpaka::hierarchy::Blocks{});  // confirmed drop
        } else if (diag) {
          alpaka::atomicAdd(acc, &diag[13], 1u, alpaka::hierarchy::Blocks{});  // keep both
        }
      }
    }
  };

  template <typename TrackerTraits>
  void HelixFit<TrackerTraits>::refitDedupUnions(const HitConstView& hv,
                                                 const ::reco::CAModulesConstView& cm,
                                                 const ::reco::TrackSoAConstView& tracks,
                                                 const ::reco::TrackHitSoAConstView& trackHits,
                                                 const uint32_t* contestedPairs,
                                                 uint32_t nPairsCap,
                                                 const caExtension::OTHitsSource* otSource,
                                                 uint8_t* drop,
                                                 int delta,
                                                 uint32_t* diag,
                                                 Queue& queue) {
    if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
      if (nPairsCap == 0u)
        return;
      // Union content stride: hold the raw concatenation of two merged tracks' hit lists (each bounded by
      // maxHitsOnTrack). A raw union exceeding this is kept both (over-size), never truncated.
      constexpr uint32_t kUnionStride = 2u * uint32_t(pixelTopology::Phase2OTStubs::maxHitsOnTrack);
      const caExtension::OTHitsSource otSrcVal =
          (otSource != nullptr && otSource->nOTHits > 0u) ? *otSource : caExtension::OTHitsSource{};

      // Synthetic fixed-stride union hit container (slot p = union of contested pair p). Caching-allocator
      // backed -> transient, stream-ordered frees after the kernels that read it (like refitMergedTwins).
      auto contHeader = cms::alpakatools::make_device_buffer<caStructures::SequentialContainer>(queue);
      auto offBuf = cms::alpakatools::make_device_buffer<caStructures::SequentialContainerOffsets[]>(
          queue, std::size_t(nPairsCap) + 1u);
      auto contentBuf =
          cms::alpakatools::make_device_buffer<uint32_t[]>(queue, std::size_t(nPairsCap) * std::size_t(kUnionStride));
      auto acceptBuf = cms::alpakatools::make_device_buffer<int32_t[]>(queue, nPairsCap);
      alpaka::memset(queue, acceptBuf, 0xff);  // -1 default; the build sets 0 for fittable slots

      constexpr uint32_t blockSize = 128;
      const uint32_t nBlocks = cms::alpakatools::divide_up_by(nPairsCap, blockSize);
      const uint32_t nBlocksOff = cms::alpakatools::divide_up_by(nPairsCap + 1u, blockSize);

      alpaka::exec<Acc1D>(queue,
                          cms::alpakatools::make_workdiv<Acc1D>(nBlocksOff, blockSize),
                          Kernel_fillUnionOffsets{},
                          offBuf.data(),
                          nPairsCap,
                          kUnionStride);
      alpaka::exec<Acc1D>(queue,
                          cms::alpakatools::make_workdiv<Acc1D>(nBlocks, blockSize),
                          Kernel_buildDedupUnions{otSrcVal},
                          contentBuf.data(),
                          acceptBuf.data(),
                          contestedPairs,
                          tracks,
                          trackHits,
                          hv,
                          nPairsCap,
                          kUnionStride,
                          diag);
      caStructures::SequentialContainerView view{contHeader.data(),
                                                 offBuf.data(),
                                                 contentBuf.data(),
                                                 nPairsCap + 1u,
                                                 uint32_t(std::size_t(nPairsCap) * std::size_t(kUnionStride))};
      alpaka::exec<Acc1D>(queue, cms::alpakatools::make_workdiv<Acc1D>(1u, 1u), Kernel_initTwinRefitContainer{}, view);

      // Scratch output SoA: the union fits write HERE, never the real tracks. The trackHits block is unused
      // by the refit (positions come from hv/otSource), so it is sized minimally.
      reco::TracksSoACollection scratch(queue, int(nPairsCap), int(nPairsCap));

      // Dropped-hit-id output over the union population (kNoDrop-initialised; the outlier phase fills the
      // dropped id per slot -> the delta-budget measure). NO hit list is modified: only refitExtended runs,
      // never the refitMergedTwins compaction.
      auto dropHitIdBuf = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nPairsCap);
      alpaka::memset(queue, dropHitIdBuf, 0xff);  // 0xffffffff == kNoDrop

      // Point the refit at the union container + scratch SoA and reuse the extended GBL ladder. Observe the
      // outlier drop (outlierReject on) so the delta-budget measure is populated for N>=5 unions.
      tuples_ = contHeader.data();
      outputSoa_ = scratch.view().tracks();
      dropOutlierHitId_ = dropHitIdBuf.data();
      refitExtended(hv, cm, contHeader.data(), acceptBuf.data(), otSource, nPairsCap, queue);
      dropOutlierHitId_ = nullptr;

      alpaka::exec<Acc1D>(queue,
                          cms::alpakatools::make_workdiv<Acc1D>(nBlocks, blockSize),
                          Kernel_dedupUnionVerdict{},
                          scratch.const_view().tracks(),
                          dropHitIdBuf.data(),
                          tracks,
                          contestedPairs,
                          acceptBuf.data(),
                          contHeader.data(),
                          hv,
                          drop,
                          delta,
                          nPairsCap,
                          uint32_t(kRefitMaxN),
                          diag);
      // No host wait: the union container / scratch SoA / drop-id buffers are function-scope
      // caching-allocator buffers, and the allocator only re-hands a freed block once the event
      // recorded on its queue at free time has completed, so the queued launches that still read
      // them are safe. Per-stream in-flight device work is bounded by the CA producers, which
      // synchronize each event's queue at their acquire boundary.
    } else {
      (void)hv;
      (void)cm;
      (void)tracks;
      (void)trackHits;
      (void)contestedPairs;
      (void)nPairsCap;
      (void)otSource;
      (void)drop;
      (void)delta;
      (void)diag;
      (void)queue;
    }
  }

  template class HelixFit<pixelTopology::Phase1>;
  template class HelixFit<pixelTopology::Phase2>;
  template class HelixFit<pixelTopology::Phase2OT>;
  template class HelixFit<pixelTopology::Phase2OTStubs>;
  template class HelixFit<pixelTopology::HIonPhase1>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
