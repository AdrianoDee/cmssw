#ifndef RecoTracker_PixelSeeding_plugins_alpaka_BrokenLineFitKernels_h
#define RecoTracker_PixelSeeding_plugins_alpaka_BrokenLineFitKernels_h

// Shared BL-fit device kernels and per-N launcher helpers. Every heavy per-N kernel is `extern template`
// here and explicitly instantiated in exactly one N-range TU (BrokenLineFit_*.dev.cc), so nvcc compiles
// disjoint kernel subsets in parallel.

// #define BROKENLINE_DEBUG
// #define GPU_DEBUG
// #define FIT_DEBUG
#include <cstdint>

#if defined(FIT_DEBUG) || defined(GPU_DEBUG)
#include <iostream>
#endif

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BrokenLine.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/GeneralBrokenLine.h"  // unfactorized GBL fit
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"
#include "RecoTracker/PixelTrackFitting/interface/BLBFieldMap.h"  // (Bz,Br) r-z map for the merger GBL beff

#include "CAFitHitSelection.h"
#include "CAExtensionKernels.h"                           // OT hit source of the extended-N refit
#include "RecoTracker/PixelSeeding/interface/OTHitTag.h"  // tagged OT-id encoding
#include "HelixFit.h"

using OutputSoAView = reco::TrackSoAView;
using TupleMultiplicity = caStructures::GenericContainer;
using Tuples = caStructures::SequentialContainer;
namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Block dimension of the fit work divisions (launch dimension only: the kernels are grid-stride loops
  // that break on the invalid-tkid sentinel). The fused refit ladder expresses its bin/lane partition in
  // units of it.
  inline constexpr uint32_t kFitBlock = 32u;

  // Fused refit ladder: ten N-bins share one lane-strided buffer set, partitioned per event on the device
  // into disjoint lane ranges (Kernel_BLRefitBinCount counts the per-bin demands, Kernel_BLRefitLaneRanges
  // prefix-sums them and writes the block->bin dispatch table).
  // Disjointness: hits/hits_ge/fast_fit are column-major with stride kRefitStride; pphase is lane-major at
  // the fixed quota kBLPhaseDoubles; pgnodes/pgblScratch are lane-major with a per-lane quota that is
  // non-decreasing in N (kRefitQuotaMonotone), so the ordered ranges stay disjoint.
  // One pass seats kRefitStride lanes; the per-track "served" flag carries the rest to later rounds.
  inline constexpr uint32_t kRefitNBins = 10u;  // N = 3 .. kRefitMaxN, the top bin absorbing the tail
  inline constexpr int kRefitMinN = 3;

  // Device tables of one round. pRange holds {firstLane, nLanes} per bin; pBlockMap holds
  // {bin, firstLane, endLane} per fused-grid block, with bin == kRefitNBins marking an idle block.
  inline constexpr uint32_t kRefitRangeStride = 2u;
  inline constexpr uint32_t kRefitBlockMapStride = 3u;

  // Fused-grid width covers every partition the count can produce: a bin holding n_b lanes needs
  // ceil(n_b/kFitBlock) blocks, and sum_b <= kRefitStride/kFitBlock + kRefitNBins (a block carries
  // exactly one bin, so the switch over compile-time N never diverges inside a warp).
  inline constexpr uint32_t kFusedBlocks =
      HelixFit<::pixelTopology::Phase2OTStubs>::kRefitStride / kFitBlock + kRefitNBins;

  // Per-lane pgnodes/pgblScratch quotas, the expressions the phase kernels build their per-lane pointers
  // with; referenced by the disjointness static_assert below.
  template <int N, typename TrackerTraits>
  inline constexpr int kRefitSlotNodes = generalBrokenLine::kGblSplitNodes<(
      int(N) > int(TrackerTraits::maxHitsOnTrackForFullFit) ? int(N) : int(TrackerTraits::maxHitsOnTrackForFullFit))>;
  template <int N>
  inline constexpr int kRefitScratchStride =
      generalBrokenLine::kGblScratchDoubles<generalBrokenLine::kGblSplitNodes<N> - 1> +
      3 * generalBrokenLine::kGblSplitNodes<N>;
  template <typename TrackerTraits, int N>
  inline constexpr bool kRefitQuotaMonotone =
      kRefitQuotaMonotone<TrackerTraits, N - 1> &&
      (kRefitSlotNodes<N - 1, TrackerTraits> <= kRefitSlotNodes<N, TrackerTraits>) &&
      (kRefitScratchStride<N - 1> <= kRefitScratchStride<N>);
  template <typename TrackerTraits>
  inline constexpr bool kRefitQuotaMonotone<TrackerTraits, kRefitMinN> = true;
  static_assert(
      kRefitQuotaMonotone<::pixelTopology::Phase2OTStubs, int(HelixFit<::pixelTopology::Phase2OTStubs>::kRefitMaxN)>,
      "the ordered lane ranges only stay disjoint in pgnodes/pgblScratch while the per-lane "
      "quotas are non-decreasing in N");
  static_assert(kRefitNBins == HelixFit<::pixelTopology::Phase2OTStubs>::kRefitMaxN - uint32_t(kRefitMinN) + 1u,
                "the bin ladder must cover N = kRefitMinN .. kRefitMaxN");

  // Per-lane quota of the fit-hit-id / pixel-core-flag tables. The scan works in bin-local slots while the
  // fused outlier works in absolute lanes, so the quota is the ladder-wide kRefitMaxN for every bin.
  inline constexpr uint32_t kRefitFitIdQuota = HelixFit<::pixelTopology::Phase2OTStubs>::kRefitMaxN;

  // Selected-hit multiplicity -> bin: the exact bins N = kRefitMinN .. kRefitMaxN-1, the top bin absorbing
  // the tail [kRefitMaxN, maxFitSel]. Returns kRefitNBins for a track the ladder does not fit.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t refitBinOfNSel(uint32_t nSel, uint32_t maxFitSel) {
    constexpr uint32_t kMaxN = HelixFit<::pixelTopology::Phase2OTStubs>::kRefitMaxN;
    if (nSel < uint32_t(kRefitMinN) || nSel > maxFitSel)
      return kRefitNBins;
    const uint32_t nEff = (nSel < kMaxN) ? nSel : kMaxN;
    return nEff - uint32_t(kRefitMinN);
  }

  // Fused main-fit ladder: the same dynamic partition applied to the CA fit. The bins are disjoint in
  // tuples (begin(nHitsL)..end(nHitsH) are contiguous non-overlapping slices of tupleMultiplicity), so two
  // bins never write the same SoA row. Bin b's demand is end(nHitsH) - begin(nHitsL), so no count kernel
  // is needed: Kernel_BLMainLaneRanges prefix-sums the demands into ranges and writes the block->bin table.
  // Lane disjointness is unconditional: every per-lane buffer is column-major with the fixed inter-element
  // stride riemannFit::stride, so lane l occupies exactly the elements {l + e*stride} whatever N it
  // carries. One pass seats riemannFit::maxNumberOfConcurrentFits lanes in bin order; a per-bin cursor
  // (lane l of bin b is that bin's (tupleBase_b + l - base_b)-th tuple) carries the rest to later rounds.
  inline constexpr int kMainMinN = 3;
  template <typename TrackerTraits>
  inline constexpr uint32_t kMainNBins = uint32_t(TrackerTraits::maxHitsOnTrackForFullFit) - uint32_t(kMainMinN) + 1u;

  // Device tables of one round. pRange holds {firstLane, nLanes, tupleBase} per bin; pBlockMap holds
  // {bin, firstLane, endLane} per fused-grid block, with bin == kMainNBins marking an idle block.
  inline constexpr uint32_t kMainRangeStride = 3u;
  inline constexpr uint32_t kMainBlockMapStride = 3u;

  // Fused-grid width. A bin holding n_b lanes needs ceil(n_b / kFitBlock) blocks, and
  //   sum_b ceil(n_b / kFitBlock) <= maxNumberOfConcurrentFits / kFitBlock + kMainNBins,
  // so this width covers every partition the populations can produce. The grid is fixed and host-known;
  // the device table decides which blocks are live and on which bin. A block carries exactly one bin, so
  // the switch over compile-time N never diverges inside a warp.
  template <typename TrackerTraits>
  inline constexpr uint32_t kMainFusedBlocks =
      riemannFit::maxNumberOfConcurrentFits / kFitBlock + kMainNBins<TrackerTraits>;

  // The main ladder's binning: the exact bins N = kMainMinN .. maxHitsOnTrackForFullFit - 1, with the top
  // bin N = maxHitsOnTrackForFullFit absorbing the tail [maxHitsOnTrackForFullFit, maxHitsOnTrack - 1].
  template <typename TrackerTraits>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr uint32_t mainBinNHitsL(uint32_t b) {
    return uint32_t(kMainMinN) + b;
  }
  template <typename TrackerTraits>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr uint32_t mainBinNHitsH(uint32_t b) {
    return (b + 1u == kMainNBins<TrackerTraits>) ? (uint32_t(TrackerTraits::maxHitsOnTrack) - 1u)
                                                 : (uint32_t(kMainMinN) + b);
  }

  // Out-of-line boundary of the per-lane fit bodies. This package is built with -Ofast, so two inlined
  // copies of the same source may reassociate differently and a bordered-band solve amplifies that into a
  // different chi2; pinned out of line, every caller runs one compiled function. `noclone` keeps
  // -fipa-cp-clone from specialising a copy on one call site's constants.
#if defined(__CUDACC__)
#define BL_REFIT_NOINLINE __noinline__
#elif defined(__clang__)
#define BL_REFIT_NOINLINE __attribute__((noinline))
#elif defined(__GNUC__)
#define BL_REFIT_NOINLINE __attribute__((noinline, noclone))
#else
#define BL_REFIT_NOINLINE
#endif

  template <int N>
  class Kernel_BLFastFit {
  public:
    // Out-of-line per-lane body of the fast fit. No cross-lane or cross-block state: the lane index only
    // forms addresses.
    ALPAKA_FN_ACC BL_REFIT_NOINLINE static void lane(Acc1D const& acc,
                                                     uint32_t local_idx,
                                                     uint32_t tuple_idx,
                                                     Tuples const* __restrict__ foundNtuplets,
                                                     TupleMultiplicity const* __restrict__ tupleMultiplicity,
                                                     ::reco::TrackingRecHitConstView const& hh,
                                                     ::reco::CAModulesConstView const& cm,
                                                     typename caStructures::tindex_type* __restrict__ ptkids,
                                                     double* __restrict__ phits,
                                                     float* __restrict__ phits_ge,
                                                     double* __restrict__ pfast_fit,
                                                     uint32_t nHitsL,
                                                     uint32_t nHitsH,
                                                     bool hasStubs) {
      constexpr uint32_t hitsInFit = N;
      // get it from the ntuple container (one to one to helix)
      auto tkid = *(tupleMultiplicity->begin(nHitsL) + tuple_idx);
      ALPAKA_ASSERT_ACC(tkid < foundNtuplets->nOnes());

      ptkids[local_idx] = tkid;

      auto nHits = foundNtuplets->size(tkid);

      // multiplicity binning and assertions use the selected hit count (nSel), computed below.
      riemannFit::Map3xNd<N> hits(phits + local_idx);
      riemannFit::Map4d fast_fit(pfast_fit + local_idx);
      riemannFit::Map6xNf<N> hits_ge(phits_ge + local_idx);

      // Prepare data structure
      auto const* hitId = foundNtuplets->begin(tkid);

      // Hits the fit uses: kMode filter plus same-layer pixel overlap dedup (OT stubs are never merged),
      // through the caFitHitSel::dedupWalk that count/fillMultiplicity use.
      const bool hasStubsRt = hasStubs && (static_cast<int32_t>(hh.offsetStubs()) >= 0);
      uint32_t nSel = caFitHitSel::dedupWalk(foundNtuplets, tkid, hh, hasStubsRt, /*k=*/-1);
      ALPAKA_ASSERT_ACC(nSel >= nHitsL);
      ALPAKA_ASSERT_ACC(nSel <= nHitsH);

      // Select hitsInFit hits uniformly from the deduped selected hits.
      uint32_t selectedHits[N];
      uint32_t nSelected = 0;
      {
        float incr = std::max(1.f, float(nSel) / float(hitsInFit));
        float fn = 0;
        for (uint32_t i = 0; i < hitsInFit; ++i) {
          int k = int(fn + 0.5f);  // round -> k-th kept hit
          if (hitsInFit - 1 == i)
            k = int(nSel) - 1;  // force last kept hit (max lever arm)
          // map the k-th kept hit to its position j in [0, nHits) via the shared dedup walk
          uint32_t j = caFitHitSel::dedupWalk(foundNtuplets, tkid, hh, hasStubsRt, k);
          ALPAKA_ASSERT_ACC(j < nHits);
          selectedHits[nSelected++] = j;
          fn += incr;
        }
      }
      ALPAKA_ASSERT_ACC(nSelected == hitsInFit);

      for (uint32_t i = 0; i < hitsInFit; ++i) {
        int j = selectedHits[i];
        auto hit = hitId[j];
        float ge[6];

        // The fit point of a stub sits on the sensor in CAModulesSoA::innerSensorFrame (pixel-side sensor
        // for PS stubs, physically inner sensor for SS stubs, == detFrame for pixels); the global error is
        // built here from the local one through that frame.
        auto frame = cm.innerSensorFrame(hh.detectorIndex(hit));
        float xerrFit = hh[hit].xerrLocal();
        // Stub transverse-error calibration, fit input only: scale the local-x variance by the squared
        // pull width so the fit weights match the measured resolution.
        if (hasStubsRt && reco::isStub(hh, int32_t(hit))) {
          const bool is2S = hh[hit].yerrLocal() > 0.1f;                                // strip vs macro-pixel
          const bool isBarrelHit = alpaka::math::abs(acc, hh[hit].zGlobal()) < 118.f;  // OT barrel vs TEDD
          const float f = is2S ? (isBarrelHit ? 0.4624f : 0.8464f)                     // sigma 0.68 / 0.92
                               : (isBarrelHit ? 0.64f : 0.9025f);                      // sigma 0.80 / 0.95
          xerrFit *= f;
        }
        // The strip-length variance (degenerate/unmeasured for a 2S stub) enters the fit as measured.
        float yerrFit = hh[hit].yerrLocal();
        frame.toGlobal(xerrFit, 0, yerrFit, ge);

        // Fill position - for stubs: use the global position of the lower hit that is stored in the hit SoA
        hits.col(i) << hh[hit].xGlobal(), hh[hit].yGlobal(), hh[hit].zGlobal();
        hits_ge.col(i) << ge[0], ge[1], ge[2], ge[3], ge[4], ge[5];
      }
      brokenline::fastFit(acc, hits, fast_fit);

#ifdef BROKENLINE_DEBUG
      // any NaN value should cause the track to be rejected at a later stage
      ALPAKA_ASSERT_ACC(not alpaka::math::isnan(acc, fast_fit(0)));
      ALPAKA_ASSERT_ACC(not alpaka::math::isnan(acc, fast_fit(1)));
      ALPAKA_ASSERT_ACC(not alpaka::math::isnan(acc, fast_fit(2)));
      ALPAKA_ASSERT_ACC(not alpaka::math::isnan(acc, fast_fit(3)));
#endif
    }

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  Tuples const* __restrict__ foundNtuplets,
                                  TupleMultiplicity const* __restrict__ tupleMultiplicity,
                                  ::reco::TrackingRecHitConstView hh,
                                  ::reco::CAModulesConstView cm,
                                  typename caStructures::tindex_type* __restrict__ ptkids,
                                  double* __restrict__ phits,
                                  float* __restrict__ phits_ge,
                                  double* __restrict__ pfast_fit,
                                  uint32_t nHitsL,
                                  uint32_t nHitsH,
                                  int32_t offset,
                                  bool hasStubs) const {
      constexpr uint32_t hitsInFit = N;
      constexpr auto invalidTkId = std::numeric_limits<typename caStructures::tindex_type>::max();

      ALPAKA_ASSERT_ACC(hitsInFit <= nHitsL);
      ALPAKA_ASSERT_ACC(nHitsL <= nHitsH);
      ALPAKA_ASSERT_ACC(phits);
      ALPAKA_ASSERT_ACC(pfast_fit);
      ALPAKA_ASSERT_ACC(foundNtuplets);
      ALPAKA_ASSERT_ACC(tupleMultiplicity);

      // look in bin for this hit multiplicity
      int totTK = tupleMultiplicity->end(nHitsH) - tupleMultiplicity->begin(nHitsL);
      ALPAKA_ASSERT_ACC(totTK <= int(tupleMultiplicity->size()));
      ALPAKA_ASSERT_ACC(totTK >= 0);

#ifdef BROKENLINE_DEBUG
      if (cms::alpakatools::once_per_grid(acc)) {
        printf("%d total Ntuple\n", tupleMultiplicity->size());
        printf("%d Ntuple of size %d/%d for %d hits to fit\n", totTK, nHitsL, nHitsH, hitsInFit);
      }
#endif
      const auto nt = riemannFit::maxNumberOfConcurrentFits;
      for (auto local_idx : cms::alpakatools::uniform_elements(acc, nt)) {
        auto tuple_idx = local_idx + offset;
        if ((int)tuple_idx >= totTK) {
          ptkids[local_idx] = invalidTkId;
          break;
        }
        Kernel_BLFastFit<N>::lane(acc,
                                  local_idx,
                                  uint32_t(tuple_idx),
                                  foundNtuplets,
                                  tupleMultiplicity,
                                  hh,
                                  cm,
                                  ptkids,
                                  phits,
                                  phits_ge,
                                  pfast_fit,
                                  nHitsL,
                                  nHitsH,
                                  hasStubs);
      }
    }
  };

  // Per-track effective field for the GBL curvature->pT conversion. The fit solves a geometric curvature of
  // the transverse projection, whose bending field is B_bend = Bz - Br*tanLambda*cos(alpha) (see
  // BLBFieldMap.h). The hit-average of B_bend/Bz(0,0) scales the origin scalar, with tanLambda*cos(alpha)
  // = (cx*y - cy*x)/(|R|*r) carrying q^2 = 1, so the correction is charge-free. Null bMap -> origin field.
  template <typename TAcc, typename M3xN, typename V4>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE double blEffectiveBField(
      const TAcc& acc, const M3xN& hits, int n, const V4& fast_fit, double bField, const float* bMap) {
    if (bMap == nullptr)
      return bField;
    const double cx = fast_fit(0);
    const double cy = fast_fit(1);
    const double absR = alpaka::math::abs(acc, fast_fit(2));
    const double slopeDen = fast_fit(3) * absR;
    double sSum = 0.;
    for (int i = 0; i < n; ++i) {
      const double x = hits(0, i);
      const double y = hits(1, i);
      const double r = alpaka::math::sqrt(acc, x * x + y * y);
      const double den = slopeDen * r;
      const double tanLambdaCosAlpha = (den != 0.) ? -(cx * y - cy * x) / den : 0.;
      sSum += blBFieldMap::bBendAt(bMap, r, hits(2, i), tanLambdaCosAlpha);
    }
    return bField * (sSum / double(n));
  }

  // Fit-consistent effective bending field: the B_bend samples of blEffectiveBField weighted by the fit's
  // own curvature-information kernel instead of by 1/n, since the published momentum is
  // bFieldEff/|curvature|. The GBL normal system eliminates the 1x1 q/p border with a scalar Schur
  // complement, so a residual along the first (bending) offset at measurement node j enters the curvature
  // with weight g = w[u-1]*P(0,0) + w[u]*P(1,0), u = 1 + 2j. With b = B_bend/Bz(0,0), the deviation
  // beta = b - 1 and its running integrals J1/J2,
  //     B_eff = bField * (1 + SUM_i g_i (s_i J1_i - J2_i) / SUM_i g_i s_i^2/2) ,
  // exactly bField for a constant field. `nodes`/`nNodes` are the solve's measured nodes in hit order,
  // `infl` the influence vector (see kGblInfluenceOffset); degenerate input returns `fallback`.
  template <typename TAcc, typename M3xN, typename V4>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE double blKernelWeightedBField(const TAcc& acc,
                                                               const M3xN& hits,
                                                               int n,
                                                               const V4& fast_fit,
                                                               int qCharge,
                                                               double bField,
                                                               const float* bMap,
                                                               const generalBrokenLine::GblNodeData* nodes,
                                                               int nNodes,
                                                               const double* infl,
                                                               double fallback) {
    if (bMap == nullptr)
      return fallback;
    const double cx = fast_fit(0);
    const double cy = fast_fit(1);
    const double cNorm = alpaka::math::sqrt(acc, cx * cx + cy * cy);
    const double absR = alpaka::math::abs(acc, fast_fit(2));
    if (!(cNorm > 0.) || !(absR > 0.))
      return fallback;
    // reference direction of the arc-length origin, exactly as prepareGblFitData builds it
    const double ex = -fast_fit(2) * cx / cNorm;
    const double ey = -fast_fit(2) * cy / cNorm;
    const double slopeDen = fast_fit(3) * absR;
    // J1/J2 are the two integrals of the field profile's deviation from unity, from the first measured
    // node up to the current one, over the cells that reach midway to each neighbour.
    double s0 = 0., sPrev = 0., bPrev = 0., j1 = 0., j2 = 0., num = 0., den = 0.;
    int iHit = -1;
    for (int j = 0; j < nNodes; ++j) {
      if (!nodes[j].hasMeas)
        continue;
      ++iHit;
      if (iHit >= n)
        break;
      const double x = hits(0, iHit);
      const double y = hits(1, iHit);
      const double dx = x - cx;
      const double dy = y - cy;
      double s = double(qCharge) * fast_fit(2) * alpaka::math::atan2(acc, dx * ey - dy * ex, dx * ex + dy * ey);
      const double r = alpaka::math::sqrt(acc, x * x + y * y);
      const double denAlpha = slopeDen * r;
      const double tanLambdaCosAlpha = (denAlpha != 0.) ? -(cx * y - cy * x) / denAlpha : 0.;
      const double bDev = blBFieldMap::bBendAt(bMap, r, hits(2, iHit), tanLambdaCosAlpha) - 1.;
      if (iHit == 0) {
        s0 = s;  // the integration origin: utilde and s^2/2 are then the same functional of the same arc
        s = 0.;
      } else {
        s -= s0;
        const double mid = 0.5 * (sPrev + s);  // cell boundary between this node and the previous one
        j1 += bPrev * (mid - sPrev);
        j2 += bPrev * (mid * mid - sPrev * sPrev) * 0.5;
        j1 += bDev * (s - mid);
        j2 += bDev * (s * s - mid * mid) * 0.5;
      }
      const int u = 1 + 2 * j;
      const double g = infl[u - 1] * nodes[j].measPrec(0, 0) + infl[u] * nodes[j].measPrec(1, 0);
      num += g * (s * j1 - j2);
      den += g * s * s * 0.5;
      sPrev = s;
      bPrev = bDev;
    }
    if (den == 0.)
      return fallback;
    const double scale = 1. + num / den;  // beta == 0 => num == 0 => scale == 1 => exactly bField
    if (alpaka::math::isnan(acc, scale) || alpaka::math::isinf(acc, scale))
      return fallback;
    return bField * scale;
  }
  // The CA main fit: the factorized Blobel circle+line fit, run by every CA iteration of every topology.
  // Stride is the launch's concurrent-fit count (the fit-buffer lane stride), riemannFit::stride by
  // default.
  template <int N, typename TrackerTraits, uint32_t Stride = riemannFit::stride>
  struct Kernel_BLFit {
  public:
    // Out-of-line per-lane body of the fit: fits lane `local_idx` of the shared stride-wide buffers and
    // writes the one SoA row ptkids[local_idx] names.
    ALPAKA_FN_ACC BL_REFIT_NOINLINE void lane(Acc1D const& acc,
                                              uint32_t local_idx,
                                              TupleMultiplicity const* __restrict__ tupleMultiplicity,
                                              double bField,
                                              OutputSoAView& results_view,
                                              typename caStructures::tindex_type const* __restrict__ ptkids,
                                              double* __restrict__ phits,
                                              float* __restrict__ phits_ge,
                                              double* __restrict__ pfast_fit,
                                              double* __restrict__ pscratch) const {
      auto tkid = ptkids[local_idx];

      ALPAKA_ASSERT_ACC(tkid < tupleMultiplicity->capacity());

      riemannFit::Map3xNdS<N, Stride> hits(phits + local_idx);
      riemannFit::Map4dS<Stride> fast_fit(pfast_fit + local_idx);
      riemannFit::Map6xNfS<N, Stride> hits_ge(phits_ge + local_idx);

      // Factorized Blobel fit: circle + line. All O(N) state (the prepared-data vectors, the shared band
      // block and the helper vectors) lives in this lane's slice of pscratch.
      static_assert(int(N) <= int(TrackerTraits::maxHitsOnTrackForFullFit),
                    "legacy-fit scratch quota is sized at maxHitsOnTrackForFullFit");
      // [element][lane] layout: element e of lane local_idx is at pscratch[e*Stride + local_idx], so a
      // warp reads consecutive lanes at consecutive addresses. Stride = this launch's lane count.
      brokenline::PreparedBrokenLineDataMap<N, Stride> data(pscratch + local_idx);
      brokenline::LegacyFitWorkspaceMap<N, Stride> fitWs(
          pscratch + local_idx + std::size_t(brokenline::kPreparedDataDoubles<N>) * std::size_t(Stride));
      brokenline::karimaki_circle_fit circle;
      riemannFit::LineFit line;
      // Per-track effective bending field: the hit-average of B_bend(r,z)/Bz(0,0), charge-free, equal to
      // bField where the map is flat. It carries the |z| falloff of Bz and the endcap radial component.
      // With fitCorrections_ off or bMap_ null this is the scalar bField.
      const double bFieldEff = fitCorrections_ ? blEffectiveBField(acc, hits, int(N), fast_fit, bField, bMap_) : bField;
      // bFieldEff enters as the momentum p = bFieldEff * radius * sqrt(1+slope^2) the Highland variance is
      // divided by, as the 1/bFieldEff of copyFromCircle's geometric-curvature -> q/pT conversion, and as
      // pt = bFieldEff/|curvature|.
      brokenline::prepareBrokenLineData(
          acc, hits, fast_fit, bFieldEff, rhoMap_, data, fitWs, fitCorrections_, /*elossGaps=*/fitCorrections_);
      brokenline::lineFit(acc, hits_ge, fast_fit, bFieldEff, data, line, fitWs, fitCorrections_);
      // Ionization energy loss as a per-track curvature growth per unit material column,
      //     elossCurv = kappa_0 * (dE/dX)_eff / p ,   (dE/dX)_eff = dE(X_tot)/X_tot ,
      // with dE() the Landau law the GBL refit charges per node (elossTypicalColumn) and p the fit's own
      // bFieldEff*R*sqrt(1+slope^2). Anchored on the total column: exact at the outermost node, low by a
      // few percent in the middle. Zero with fitCorrections_ off.
      double elossCurv = 0.;
      if (fitCorrections_) {
        const double slopeK = -double(data.qCharge) / fast_fit(3);
        const double xTot = data.innerXX0 + fitWs.gapXX0(int(N) - 2);  // beamline -> outermost node [X/X0]
        const double pTot =
            alpaka::math::sqrt(acc, riemannFit::sqr(bFieldEff * fast_fit(2)) * (1. + riemannFit::sqr(slopeK)));
        if (xTot > 0. && pTot > 0. && fast_fit(2) > 0.)
          elossCurv = (generalBrokenLine::elossTypicalColumn(acc, pTot, xTot) / xTot) / (pTot * fast_fit(2));
      }
      brokenline::circleFit(acc, hits, hits_ge, fast_fit, bFieldEff, data, circle, fitWs, fitCorrections_, elossCurv);
      reco::copyFromCircle(results_view, circle.par, circle.cov, line.par, line.cov, 1.f / float(bFieldEff), tkid);
      results_view[tkid].pt() = float(bFieldEff) / float(std::abs(circle.par(2)));
      results_view[tkid].eta() = alpaka::math::asinh(acc, line.par(0));
      results_view[tkid].chi2() = (circle.chi2 + line.chi2) / (2 * N - 5);
      results_view[tkid].ndof() = int8_t(2 * N - 5);
    }

    // Device pointer to the uploaded Geant4 material density grid (blMaterialMap, kSize floats).
    const float* __restrict__ rhoMap_ = nullptr;
    // Device pointer to the (Bz,Br) r-z field map (blBFieldMap). Consumed only under fitCorrections_: the
    // scattering variance is only correct if its momentum comes from the field that bent the track. Null
    // => scalar bField.
    const float* __restrict__ bMap_ = nullptr;
    // Fit correctness package (producer parameter useFitCorrections; see the head of BrokenLine.h): thin
    // scatterer per gap, rigid-node guard, Karimaki-Fisher covariance blend, pion 1/beta Highland form,
    // trapezoid material quadrature, full 3x3 blend, per-track effective bending field and ionization
    // energy-loss offset. Default true for Phase2OTStubs.
    bool fitCorrections_ = false;

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  TupleMultiplicity const* __restrict__ tupleMultiplicity,
                                  double bField,
                                  OutputSoAView results_view,
                                  typename caStructures::tindex_type const* __restrict__ ptkids,
                                  double* __restrict__ phits,
                                  float* __restrict__ phits_ge,
                                  double* __restrict__ pfast_fit,
                                  // Per-lane fit scratch, held off the kernel stack frame (the frame
                                  // drives the driver's per-thread local-memory reservation).
                                  double* __restrict__ pscratch) const {
      ALPAKA_ASSERT_ACC(results_view.pt().data());
      ALPAKA_ASSERT_ACC(results_view.eta().data());
      ALPAKA_ASSERT_ACC(results_view.chi2().data());
      ALPAKA_ASSERT_ACC(pfast_fit);

      constexpr auto invalidTkId = std::numeric_limits<typename caStructures::tindex_type>::max();

      // same as above...
      // look in bin for this hit multiplicity
      const auto nt = Stride;  // lane count = this launch's buffer stride
      for (auto local_idx : cms::alpakatools::uniform_elements(acc, nt)) {
        if (invalidTkId == ptkids[local_idx])
          break;
        lane(acc, local_idx, tupleMultiplicity, bField, results_view, ptkids, phits, phits_ge, pfast_fit, pscratch);
      }
    }
  };

  // Dynamic partition of the main fit ladder. Single-threaded, device-only: hands lanes out in bin order
  // and writes the block->bin dispatch table (one bin per block, idle tail at kMainNBins). Nothing is
  // memset: the kernel writes every word it reads.
  template <typename TrackerTraits>
  class Kernel_BLMainLaneRanges {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  TupleMultiplicity const* __restrict__ tupleMultiplicity,
                                  uint32_t* __restrict__ pCursor,
                                  uint32_t* __restrict__ pRange,
                                  uint32_t* __restrict__ pBlockMap,
                                  uint32_t laneTotal,
                                  bool firstRound) const {
      if (alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0] != 0)
        return;
      constexpr uint32_t kNBins = kMainNBins<TrackerTraits>;
      uint32_t base = 0u;
      for (uint32_t b = 0; b < kNBins; ++b) {
        // The bin's population, off the same container slice the per-bin fast fit reads.
        const int32_t tot = int32_t(tupleMultiplicity->end(mainBinNHitsH<TrackerTraits>(b)) -
                                    tupleMultiplicity->begin(mainBinNHitsL<TrackerTraits>(b)));
        const uint32_t have = firstRound ? 0u : pCursor[b];  // seated in earlier rounds
        const uint32_t want = (tot > 0 && uint32_t(tot) > have) ? (uint32_t(tot) - have) : 0u;
        const uint32_t room = (base < laneTotal) ? (laneTotal - base) : 0u;
        const uint32_t got = (want < room) ? want : room;
        pRange[kMainRangeStride * b + 0] = base;
        pRange[kMainRangeStride * b + 1] = got;
        pRange[kMainRangeStride * b + 2] = have;  // first tuple of the bin this round seats
        pCursor[b] = have + got;
        base += got;
      }
      uint32_t blk = 0u;
      for (uint32_t b = 0; b < kNBins; ++b) {
        const uint32_t lo = pRange[kMainRangeStride * b + 0];
        const uint32_t hi = lo + pRange[kMainRangeStride * b + 1];
        for (uint32_t l = lo; l < hi; l += kFitBlock) {
          ALPAKA_ASSERT_ACC(blk < kMainFusedBlocks<TrackerTraits>);  // sum_b ceil(n_b/kFitBlock) <= width
          const uint32_t e = (l + kFitBlock < hi) ? (l + kFitBlock) : hi;
          pBlockMap[kMainBlockMapStride * blk + 0] = b;
          pBlockMap[kMainBlockMapStride * blk + 1] = l;
          pBlockMap[kMainBlockMapStride * blk + 2] = e;
          ++blk;
        }
      }
      for (; blk < kMainFusedBlocks<TrackerTraits>; ++blk) {
        pBlockMap[kMainBlockMapStride * blk + 0] = kNBins;  // idle
        pBlockMap[kMainBlockMapStride * blk + 1] = 0u;
        pBlockMap[kMainBlockMapStride * blk + 2] = 0u;
      }
    }
  };

  // N-independent config of one fused main-fit launch: the per-bin kernels' members + N-independent args.
  // One object serves every bin; the trampoline rebuilds the bin's kernel inside the callee's frame.
  struct BLMainFusedCfg {
    const float* __restrict__ rhoMap = nullptr;
    const float* __restrict__ bMap = nullptr;
    bool fitCorrections = false;
    bool hasStubs = false;
  };

  // Out-of-line trampoline: rebuilds the bin's kernel object from the shared config inside the callee's
  // frame and calls the pinned lane body. The fast fit needs none: Kernel_BLFastFit<N> carries no members,
  // so the fused switch calls its static `lane` directly.
  template <int N, typename TrackerTraits, uint32_t Stride>
  ALPAKA_FN_ACC BL_REFIT_NOINLINE void blMainFitLaneOutOfLine(
      Acc1D const& acc,
      BLMainFusedCfg const& cfg,
      uint32_t lane,
      TupleMultiplicity const* __restrict__ tupleMultiplicity,
      double bField,
      OutputSoAView& results_view,
      typename caStructures::tindex_type const* __restrict__ ptkids,
      double* __restrict__ phits,
      float* __restrict__ phits_ge,
      double* __restrict__ pfast_fit,
      double* __restrict__ pscratch) {
    const Kernel_BLFit<N, TrackerTraits, Stride> k{cfg.rhoMap, cfg.bMap, cfg.fitCorrections};
    k.lane(acc, lane, tupleMultiplicity, bField, results_view, ptkids, phits, phits_ge, pfast_fit, pscratch);
  }

  // Fused main-fit kernels. The block index picks the bin (and its compile-time N) out of the device table
  // Kernel_BLMainLaneRanges wrote for this round; the elements of that block are the bin's lanes. One block
  // carries one bin, so the switch over N cannot diverge inside a warp. No lane body reads its lane index
  // beyond its own addresses, so the arithmetic does not depend on where a tuple lands.
  template <typename TrackerTraits>
  struct Kernel_BLFastFitFused {
    // The switch dispatches bins 0..7 by hand: a traits set carrying more bins would fall to `default:`
    // and lose its tuples silently, so it must be a build error.
    static_assert(kMainNBins<TrackerTraits> <= 8u,
                  "the fused main ladder dispatches bins 0..7; add cases when a traits set carries more");
    // The fast fit uses the default-stride maps (riemannFit::stride == maxNumberOfConcurrentFits), the
    // same lane space the fit kernel asserts on.
    BLMainFusedCfg cfg_;

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  Tuples const* __restrict__ foundNtuplets,
                                  TupleMultiplicity const* __restrict__ tupleMultiplicity,
                                  ::reco::TrackingRecHitConstView hh,
                                  ::reco::CAModulesConstView cm,
                                  typename caStructures::tindex_type* __restrict__ ptkids,
                                  double* __restrict__ phits,
                                  float* __restrict__ phits_ge,
                                  double* __restrict__ pfast_fit,
                                  const uint32_t* __restrict__ pRange,
                                  const uint32_t* __restrict__ pBlockMap) const {
      ALPAKA_ASSERT_ACC(foundNtuplets);
      ALPAKA_ASSERT_ACC(tupleMultiplicity);
      ALPAKA_ASSERT_ACC(phits);
      ALPAKA_ASSERT_ACC(pfast_fit);
      constexpr uint32_t kNBins = kMainNBins<TrackerTraits>;
      constexpr uint32_t kTotal = kMainFusedBlocks<TrackerTraits> * kFitBlock;
      for (uint32_t blk : cms::alpakatools::uniform_groups(acc, kTotal)) {
        const uint32_t bin = pBlockMap[kMainBlockMapStride * blk];
        if (bin >= kNBins)
          continue;  // idle block: this round's partition did not need it
        ALPAKA_ASSERT_ACC(mainBinNHitsL<TrackerTraits>(bin) <= mainBinNHitsH<TrackerTraits>(bin));
        const uint32_t laneLo = pBlockMap[kMainBlockMapStride * blk + 1];
        const uint32_t laneHi = pBlockMap[kMainBlockMapStride * blk + 2];
        // Where this bin starts in the lane space, and which of its tuples this round seats first.
        const uint32_t binBase = pRange[kMainRangeStride * bin];
        const uint32_t tupleBase = pRange[kMainRangeStride * bin + 2];
        for (auto el : cms::alpakatools::uniform_group_elements(acc, blk, kTotal)) {
          const uint32_t local_idx = laneLo + uint32_t(el.local);
          if (local_idx >= laneHi)
            break;
          const uint32_t tuple_idx = tupleBase + (local_idx - binBase);
#define BL_MAIN_FUSED_FAST_CASE(BIN)                                               \
  case (BIN):                                                                      \
    if constexpr (uint32_t(BIN) < kMainNBins<TrackerTraits>) {                     \
      Kernel_BLFastFit<(BIN) + kMainMinN>::lane(acc,                               \
                                                local_idx,                         \
                                                tuple_idx,                         \
                                                foundNtuplets,                     \
                                                tupleMultiplicity,                 \
                                                hh,                                \
                                                cm,                                \
                                                ptkids,                            \
                                                phits,                             \
                                                phits_ge,                          \
                                                pfast_fit,                         \
                                                mainBinNHitsL<TrackerTraits>(BIN), \
                                                mainBinNHitsH<TrackerTraits>(BIN), \
                                                cfg_.hasStubs);                    \
    }                                                                              \
    break
          switch (bin) {
            BL_MAIN_FUSED_FAST_CASE(0);
            BL_MAIN_FUSED_FAST_CASE(1);
            BL_MAIN_FUSED_FAST_CASE(2);
            BL_MAIN_FUSED_FAST_CASE(3);
            BL_MAIN_FUSED_FAST_CASE(4);
            BL_MAIN_FUSED_FAST_CASE(5);
            BL_MAIN_FUSED_FAST_CASE(6);
            BL_MAIN_FUSED_FAST_CASE(7);
            default:
              break;
          }
#undef BL_MAIN_FUSED_FAST_CASE
        }
      }
    }
  };

  template <typename TrackerTraits, uint32_t Stride = riemannFit::stride>
  struct Kernel_BLFitFused {
    static_assert(Stride == riemannFit::maxNumberOfConcurrentFits,
                  "the fused main ladder spans the main fit's lane buffers");
    static_assert(kMainNBins<TrackerTraits> <= 8u,
                  "the fused main ladder dispatches bins 0..7; add cases when a traits set carries more");
    BLMainFusedCfg cfg_;

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  TupleMultiplicity const* __restrict__ tupleMultiplicity,
                                  double bField,
                                  OutputSoAView results_view,
                                  typename caStructures::tindex_type const* __restrict__ ptkids,
                                  double* __restrict__ phits,
                                  float* __restrict__ phits_ge,
                                  double* __restrict__ pfast_fit,
                                  double* __restrict__ pscratch,
                                  const uint32_t* __restrict__ pBlockMap) const {
      ALPAKA_ASSERT_ACC(results_view.pt().data());
      ALPAKA_ASSERT_ACC(results_view.eta().data());
      ALPAKA_ASSERT_ACC(results_view.chi2().data());
      ALPAKA_ASSERT_ACC(pfast_fit);
      constexpr auto invalidTkId = std::numeric_limits<typename caStructures::tindex_type>::max();
      constexpr uint32_t kNBins = kMainNBins<TrackerTraits>;
      constexpr uint32_t kTotal = kMainFusedBlocks<TrackerTraits> * kFitBlock;
      for (uint32_t blk : cms::alpakatools::uniform_groups(acc, kTotal)) {
        const uint32_t bin = pBlockMap[kMainBlockMapStride * blk];
        if (bin >= kNBins)
          continue;  // idle block
        const uint32_t laneLo = pBlockMap[kMainBlockMapStride * blk + 1];
        const uint32_t laneHi = pBlockMap[kMainBlockMapStride * blk + 2];
        for (auto el : cms::alpakatools::uniform_group_elements(acc, blk, kTotal)) {
          const uint32_t local_idx = laneLo + uint32_t(el.local);
          if (local_idx >= laneHi)
            break;
          // Safety net: a bin's range is its exact count and the fused fast fit filled every lane of it.
          if (invalidTkId == ptkids[local_idx])
            continue;
#define BL_MAIN_FUSED_FIT_CASE(BIN)                                                                                     \
  case (BIN):                                                                                                           \
    if constexpr (uint32_t(BIN) < kMainNBins<TrackerTraits>) {                                                          \
      blMainFitLaneOutOfLine<(BIN) + kMainMinN, TrackerTraits, Stride>(                                                 \
          acc, cfg_, local_idx, tupleMultiplicity, bField, results_view, ptkids, phits, phits_ge, pfast_fit, pscratch); \
    }                                                                                                                   \
    break
          switch (bin) {
            BL_MAIN_FUSED_FIT_CASE(0);
            BL_MAIN_FUSED_FIT_CASE(1);
            BL_MAIN_FUSED_FIT_CASE(2);
            BL_MAIN_FUSED_FIT_CASE(3);
            BL_MAIN_FUSED_FIT_CASE(4);
            BL_MAIN_FUSED_FIT_CASE(5);
            BL_MAIN_FUSED_FIT_CASE(6);
            BL_MAIN_FUSED_FIT_CASE(7);
            default:
              break;
          }
#undef BL_MAIN_FUSED_FIT_CASE
        }
      }
    }
  };

  // Phase-buffer lane stride (doubles): the per-lane cross-launch scratch of the Kernel_BLFitPhase*
  // pipeline, indexed by the named slots below. Slots [64..104] hold the iteration-invariant tail:
  //   [64..64+N-1] matXX0     the material march (function of hit positions alone)
  //   [64+N]       innerXX0   function of hit 0, at the end of the window prepareGblFitData's matCached
  //                           contract reads (matCached[n]); the window is sized for N <= 12
  //   [77] [78]    innerD1/W1 segmentXX0Moments, function of hit 0 alone
  //   [79]         bFieldEff  effective field of this linearization, invariant across its phases
  constexpr int kBLPhaseJacBack = 0;     // [0..24]  hit0 -> PCA backward jacobian (hits-only node layout)
  constexpr int kBLPhaseUsedInner = 25;  // 1 if the inner-node layout was built for this lane
  constexpr int kBLPhaseQCharge = 26;
  constexpr int kBLPhaseSTrans0 = 27;  // sTransverse(0), the arc length of the first node
  constexpr int kBLPhaseInnerXX0 = 28;
  constexpr int kBLPhaseGChi2 = 29;    // chi2 of the solve
  constexpr int kBLPhaseCorrPca = 30;  // [30..34] the five perigee corrections
  constexpr int kBLPhaseCovPca = 35;   // [35..59] the 5x5 perigee covariance, row-major
  constexpr int kBLPhaseNextRef = 60;  // [60..63] the reference the next linearization starts from
  // The matXX0 window is sized for the largest N any instantiation carries (kRefitMaxN = 12).
  constexpr int kBLPhaseMatMax = 12;
  constexpr int kBLPhaseMatOff = 64;
  static_assert(kBLPhaseNextRef + 4 == kBLPhaseMatOff, "the base layout must end exactly where the tail begins");
  constexpr int kBLPhaseInnerD1 = kBLPhaseMatOff + kBLPhaseMatMax + 1;  // past the widest matCached[n]
  constexpr int kBLPhaseInnerW1 = kBLPhaseInnerD1 + 1;
  constexpr int kBLPhaseBFieldEff = kBLPhaseInnerW1 + 1;
  //   [80 .. 80+N-1]   gapD1   } the per-gap two-thin split of the exact-split material model
  //   [92 .. 92+N-1]   gapW1   } (segmentXX0GapSplit; iteration-invariant, cached like matXX0)
  //   [104]            splitLayout  1 if prepareGblDataSplit built this lane's 2N+1 node chain
  constexpr int kBLPhaseGapD1 = kBLPhaseBFieldEff + 1;
  constexpr int kBLPhaseGapW1 = kBLPhaseGapD1 + kBLPhaseMatMax;
  constexpr int kBLPhaseSplit = kBLPhaseGapW1 + kBLPhaseMatMax;
  constexpr int kBLPhaseDoubles = kBLPhaseSplit + 1;  // 105

  // Ionization-loss enable flag for the GBL node builders (passed as applyELossCorrection); the magnitude
  // comes from the Landau laws in GeneralBrokenLine.h.
  constexpr bool kApplyELossCorrection = true;

  // Phase-split GBL fit pipeline. One kernel carrying prep+solve+extract+outlier spills a per-thread frame
  // the driver reserves for every max-resident thread; per-phase kernels keep the live set to a few hundred
  // bytes with no spill. Per-fit intermediates cross launches through a per-lane phase buffer
  // (kBLPhaseDoubles doubles, slot layout above), each a fully-materialized double so the store/load
  // boundary cannot change any rounding.
  //
  // Phase (i): per-node preparation. Loads the linearization reference before any reference-dependent
  // step, rebuilds arc lengths, material and gnodes, and persists the iteration-invariant scalars and
  // jacBack into the phase buffer.
  template <int N, typename TrackerTraits, uint32_t Stride = riemannFit::stride>
  struct Kernel_BLFitPhasePrep {
  public:
    const float* __restrict__ rhoMap_ = nullptr;
    // Refit iteration 1: take the linearization reference from this lane's phase buffer (written by the
    // iteration-0 extract) instead of the fast fit the lane arrived with.
    bool iterFromPhase_ = false;
    // Normalized (Bz,Br) r-z field map. When set, the momentum feeding the MS weights / dE/dx uses a
    // per-track hit-averaged effective field; when null the scalar bField is used instead.
    const float* __restrict__ bMap_ = nullptr;
    // This launch re-linearizes a fit whose first linearization ran on the same lane of the same phase
    // buffer, so the iteration-invariant material rows are read from there instead of marching the map.
    bool matFromPhase_ = false;
    // Take the effective field from the phase buffer instead of averaging B_bend over the hits; set only
    // on a re-linearization whose predecessor's solve published a fit-consistent field there.
    bool fieldFromPhase_ = false;
    // Charge-symmetric corrections (see the head of BrokenLine.h): the node-0 -> PCA step in gblHelixAtPca
    // uses a signed arc and the node prep adds the bending-field profile offset. The second term needs
    // bMap_ and is inert where the map is null.
    bool chargeSymmetric_ = false;
    // Reference-trajectory corrections (see the head of BrokenLine.h): both node builders seed the node-0
    // path length from the reference helix, take the arc->azimuth sign from it and form the field's B_r
    // lambda row (which needs bMap_). The node layout is the same either way.
    bool trajectoryCorrections_ = false;
    // On, Highland's log is evaluated at the track's total declared material and the variance apportioned
    // to the gaps by thickness; off, each gap evaluates the log at its own thickness.
    bool scatteringLogAtTotal_ = false;
    // On, the typical energy loss is the single-column Landau law at the accumulated thickness and each
    // node is charged its increment of it; off, each lump is charged its own Landau MPV.
    bool elossCumulative_ = false;

    // One lane of this phase, pinned out of line. The invalid-tkid sentinel is tested by the caller.
    ALPAKA_FN_ACC BL_REFIT_NOINLINE void lane(Acc1D const& acc,
                                              uint32_t local_idx,
                                              double bField,
                                              [[maybe_unused]]
                                              typename caStructures::tindex_type const* __restrict__ ptkids,
                                              double* __restrict__ phits,
                                              float* __restrict__ phits_ge,
                                              double* __restrict__ pfast_fit,
                                              generalBrokenLine::GblNodeData* __restrict__ pgnodes,
                                              double* __restrict__ pphase) const {
      riemannFit::Map3xNdS<N, Stride> hits(phits + local_idx);
      riemannFit::Map4dS<Stride> fast_fit(pfast_fit + local_idx);
      riemannFit::Map6xNfS<N, Stride> hits_ge(phits_ge + local_idx);
      double* phase = pphase + std::size_t(local_idx) * std::size_t(kBLPhaseDoubles);
      // Reference override before any reference-dependent preparation.
      if (iterFromPhase_) {
        fast_fit(0) = phase[kBLPhaseNextRef + 0];
        fast_fit(1) = phase[kBLPhaseNextRef + 1];
        fast_fit(2) = phase[kBLPhaseNextRef + 2];
        fast_fit(3) = phase[kBLPhaseNextRef + 3];
      }
      // Per-track effective field; with bMap_ null this is the scalar bField.
      const double bFieldEff =
          fieldFromPhase_ ? phase[kBLPhaseBFieldEff] : blEffectiveBField(acc, hits, int(N), fast_fit, bField, bMap_);
      // Publish this linearization's effective field for the out and outlier phases. Written on every
      // prep, so each phase sees its own iteration's value.
      if (!fieldFromPhase_)
        phase[kBLPhaseBFieldEff] = bFieldEff;
      static_assert(int(N) <= kBLPhaseMatMax, "the phase buffer's matXX0 window must cover N");
      brokenline::PreparedGblData<N> data;
      // Per-gap two-thin partition of the exact-split material model, written by prepareGblFitData
      // (uninitialized here: it writes every slot before anything reads one).
      double gapD1[N], gapW1[N];
      // On a re-linearization the material rows are loaded from the phase buffer (see
      // prepareGblFitData's matCached contract); otherwise they are marched from the map and stored.
      brokenline::prepareGblFitData(acc,
                                    hits,
                                    fast_fit,
                                    bFieldEff,
                                    rhoMap_,
                                    data,
                                    matFromPhase_ ? (phase + kBLPhaseMatOff) : nullptr,
                                    gapD1,
                                    gapW1);
      constexpr int kSlotNodes = generalBrokenLine::kGblSplitNodes<(
          int(N) > int(TrackerTraits::maxHitsOnTrackForFullFit) ? int(N)
                                                                : int(TrackerTraits::maxHitsOnTrackForFullFit))>;
      generalBrokenLine::GblNodeData* gnodes = pgnodes + std::size_t(local_idx) * std::size_t(kSlotNodes);
      double innerD1 = 0., innerW1 = 0.;
      if (matFromPhase_) {
        // Function of hit 0 alone, so it rides the same cache; the iteration-0 prep left 0/0 wherever
        // innerXX0 was not positive, which is what the else branch produces.
        innerD1 = phase[kBLPhaseInnerD1];
        innerW1 = phase[kBLPhaseInnerW1];
      } else {
        // one quadrature rule for the whole trajectory: the beamline->hit0 moments come from the
        // same trapezoid march as the gaps (its W is the innerXX0 already stored).
        const double rHit0 = alpaka::math::sqrt(acc, hits(0, 0) * hits(0, 0) + hits(1, 0) * hits(1, 0));
        brokenline::segmentXX0GapSplit(acc, rhoMap_, 0., 0., rHit0, hits(2, 0), innerD1, innerW1);
      }
      if (matFromPhase_) {
        for (int i = 0; i < int(N); ++i) {
          gapD1[i] = phase[kBLPhaseGapD1 + i];
          gapW1[i] = phase[kBLPhaseGapW1 + i];
        }
      }
      if (!matFromPhase_) {  // publish the invariant rows for the next linearization
        for (int i = 0; i < int(N); ++i)
          phase[kBLPhaseMatOff + i] = data.matXX0(i);
        phase[kBLPhaseMatOff + int(N)] = data.innerXX0;  // matCached[n], see the layout comment
        phase[kBLPhaseInnerD1] = innerD1;
        phase[kBLPhaseInnerW1] = innerW1;
        for (int i = 0; i < int(N); ++i) {
          phase[kBLPhaseGapD1 + i] = gapD1[i];
          phase[kBLPhaseGapW1 + i] = gapW1[i];
        }
      }
      // hit0->PCA backward transport: only the hits-only node layout uses it, so identity here keeps the
      // phase buffer defined for the layouts that extract at node 0.
      generalBrokenLine::Matrix5d jacBack = generalBrokenLine::Matrix5d::Identity();
      bool usedInner = false;
      bool usedSplit = false;
      if constexpr (int(N) >= 3) {
        // The 2N+1 layout refuses a track whose gaps cannot carry an interior scatterer; that track takes
        // the arrival-node layout below, which is well posed for any gap.
        usedSplit = generalBrokenLine::prepareGblDataSplit<Acc1D, N>(acc,
                                                                     hits,
                                                                     hits_ge,
                                                                     fast_fit,
                                                                     bFieldEff,
                                                                     data.qCharge,
                                                                     data.sTransverse,
                                                                     data.sTotal,
                                                                     data.matXX0,
                                                                     gapD1,
                                                                     gapW1,
                                                                     data.innerXX0,
                                                                     innerD1,
                                                                     innerW1,
                                                                     gnodes,
                                                                     /*msScale=*/1.0,
                                                                     kApplyELossCorrection,
                                                                     chargeSymmetric_ ? bMap_ : nullptr,
                                                                     bField,
                                                                     trajectoryCorrections_,
                                                                     scatteringLogAtTotal_,
                                                                     elossCumulative_);
      }
      if (!usedSplit)
        generalBrokenLine::prepareGblData<Acc1D, N>(acc,
                                                    hits,
                                                    hits_ge,
                                                    fast_fit,
                                                    bFieldEff,
                                                    data.qCharge,
                                                    data.sTransverse,
                                                    data.sTotal,
                                                    data.matXX0,
                                                    data.innerXX0,
                                                    gnodes,
                                                    /*msScale=*/1.0,
                                                    kApplyELossCorrection,
                                                    &jacBack,
                                                    innerD1,
                                                    innerW1,
                                                    &usedInner,
                                                    chargeSymmetric_ ? bMap_ : nullptr,
                                                    bField,
                                                    trajectoryCorrections_,
                                                    scatteringLogAtTotal_,
                                                    elossCumulative_);
#ifdef BL_LAYER_DUMP
      // Fit-input trace, one block per track per linearization: the reference fast fit, the field, the
      // charge and the material of this pass, plus every hit and its ge error row. Run single-threaded on
      // the serial backend so a track's lines stay contiguous in stdout.
      const auto tkid = ptkids[local_idx];  // the dump is this phase's only consumer of the tuple id
      printf("BLDUMP_TRK %u N %d bfield %.17g ff %.17g %.17g %.17g %.17g q %d innerXX0 %.17g\n",
             (unsigned)tkid,
             int(N),
             bField,
             fast_fit(0),
             fast_fit(1),
             fast_fit(2),
             fast_fit(3),
             data.qCharge,
             double(data.innerXX0));
      for (int i = 0; i < int(N); ++i)
        printf("BLDUMP_HIT %u %d %.17g %.17g %.17g %.9g %.9g %.9g %.9g %.9g %.9g %.17g\n",
               (unsigned)tkid,
               i,
               hits(0, i),
               hits(1, i),
               hits(2, i),
               hits_ge(0, i),
               hits_ge(1, i),
               hits_ge(2, i),
               hits_ge(3, i),
               hits_ge(4, i),
               hits_ge(5, i),
               i < int(N) - 1 ? double(data.matXX0(i)) : 0.0);
#endif
      phase[kBLPhaseSplit] = usedSplit ? 1. : 0.;
      for (int r = 0; r < 5; ++r)
        for (int c = 0; c < 5; ++c)
          phase[kBLPhaseJacBack + 5 * r + c] = jacBack(r, c);
      phase[kBLPhaseUsedInner] = usedInner ? 1. : 0.;
      phase[kBLPhaseQCharge] = double(data.qCharge);
      phase[kBLPhaseSTrans0] = double(data.sTransverse(0));
      phase[kBLPhaseInnerXX0] = double(data.innerXX0);
    }
  };

  // Phase (ii): bordered-band solve. gblFitPca over the prepared nodes; persists the curvilinear
  // corrections/covariance at the PCA + the native chi2 (and, on the final linearization, the
  // smoothed-offset pulls into the scratch tail for the outlier phase).
  template <int N, typename TrackerTraits, uint32_t Stride = riemannFit::stride>
  struct Kernel_BLFitPhaseSolve {
  public:
    const bool outlierReject_ = true;
    // True on the fit's LAST linearization (refit iteration 1): request the pulls
    // the outlier phase consumes.
    const bool finalIter_ = true;
    // Normalized (Bz,Br) r-z field map, forwarded only so this phase can re-sample B_bend at the nodes;
    // when null the effective field in the phase buffer is left alone.
    const float* __restrict__ bMap_ = nullptr;
    // On, replace the hit-count-averaged effective field in the phase buffer by the fit-consistent one
    // this solve's own influence vector defines (blKernelWeightedBField).
    bool fieldKernelWeights_ = false;
    // One lane of this phase, pinned out of line.
    ALPAKA_FN_ACC BL_REFIT_NOINLINE void lane(Acc1D const& acc,
                                              uint32_t local_idx,
                                              double bField,
                                              double* __restrict__ phits,
                                              double* __restrict__ pfast_fit,
                                              generalBrokenLine::GblNodeData* __restrict__ pgnodes,
                                              double* __restrict__ pscratch,
                                              double* __restrict__ pphase) const {
      riemannFit::Map4dS<Stride> fast_fit(pfast_fit + local_idx);
      constexpr int kSlotNodes = generalBrokenLine::kGblSplitNodes<(
          int(N) > int(TrackerTraits::maxHitsOnTrackForFullFit) ? int(N)
                                                                : int(TrackerTraits::maxHitsOnTrackForFullFit))>;
      generalBrokenLine::GblNodeData* gnodes = pgnodes + std::size_t(local_idx) * std::size_t(kSlotNodes);
      // Sized for the widest node chain this lane can build (the exact split's 2N+1 nodes, else the
      // arrival-node layout's N+2), so the band region and the pull region never overlap in either layout.
      constexpr int kSplitN = generalBrokenLine::kGblSplitNodes<N> - 1;
      constexpr int kBandDoubles = generalBrokenLine::kGblScratchDoubles<kSplitN>;
      // Scratch overlay: gFullDelta lives in the head of the band region (= Mb); gNodeVar stays
      // outside it. See GeneralBrokenLine.h for the lifetime proof.
      constexpr int kNodeVarDoubles = 3 * (kSplitN + 1);
      constexpr int kPullsDoubles = kNodeVarDoubles;
      constexpr int kScratchStride = kBandDoubles + kPullsDoubles;
      // structural guard: the N-1 branch has the smallest Mb of the three layouts.
      static_assert((2 * kSplitN + 3) <= 2 * int(N) * (generalBrokenLine::kGblBand + 1),
                    "scratch overlay: fullDelta does not fit inside Mb");
      double* gblScratch = pscratch + std::size_t(local_idx) * std::size_t(kScratchStride);
      double* gFullDelta = gblScratch;
      double* gNodeVar = gblScratch + kBandDoubles;
      double* phase = pphase + std::size_t(local_idx) * std::size_t(kBLPhaseDoubles);
      const bool usedInner = phase[kBLPhaseUsedInner] != 0.;
      const bool usedSplit = phase[kBLPhaseSplit] != 0.;
      const int qCharge = int(phase[kBLPhaseQCharge]);
      const bool wantPulls = outlierReject_ && finalIter_ && N >= 5;
      generalBrokenLine::Vector5d corrPca, gcorr;
      generalBrokenLine::Matrix5d covPca, gcov;
      double gchi2 = 0.;
      if (usedSplit) {
        // 2N+1 nodes; node 0 is the PCA, so the extraction is exact.
        covPca = generalBrokenLine::gblFitPca<Acc1D, kSplitN>(acc,
                                                              gnodes,
                                                              gblScratch,
                                                              &corrPca,
                                                              wantPulls ? gFullDelta : nullptr,
                                                              &gchi2,
                                                              wantPulls ? gNodeVar : nullptr,
                                                              /*fullDeltaInScratch=*/true);
      } else if (usedInner) {
        covPca = generalBrokenLine::gblFitPca<Acc1D, N + 1>(acc,
                                                            gnodes,
                                                            gblScratch,
                                                            &corrPca,
                                                            wantPulls ? gFullDelta : nullptr,
                                                            &gchi2,
                                                            wantPulls ? gNodeVar : nullptr,
                                                            /*fullDeltaInScratch=*/true);
      } else {
        const double th2Inner = gnodes[1].hasScat ? 1.0 / gnodes[1].scatPrec(0, 0) : 0.0;
        gnodes[1].hasScat = false;
        gcov = generalBrokenLine::gblFitPca<Acc1D, N - 1>(acc,
                                                          gnodes + 1,
                                                          gblScratch,
                                                          &gcorr,
                                                          wantPulls ? gFullDelta : nullptr,
                                                          &gchi2,
                                                          wantPulls ? gNodeVar : nullptr,
                                                          /*fullDeltaInScratch=*/true);
        generalBrokenLine::Matrix5d jacBack;
        for (int r = 0; r < 5; ++r)
          for (int c = 0; c < 5; ++c)
            jacBack(r, c) = phase[kBLPhaseJacBack + 5 * r + c];
        const double slopeQ = -double(qCharge) / fast_fit(3);
        gcov(1, 1) += th2Inner;
        gcov(2, 2) += th2Inner * (1.0 + slopeQ * slopeQ);
        corrPca = jacBack * gcorr;
        covPca = jacBack * gcov * jacBack.transpose();
      }
      phase[kBLPhaseGChi2] = gchi2;
      for (int a = 0; a < 5; ++a)
        phase[kBLPhaseCorrPca + a] = corrPca(a);
      for (int r = 0; r < 5; ++r)
        for (int c = 0; c < 5; ++c)
          phase[kBLPhaseCovPca + 5 * r + c] = covPca(r, c);
      // Fit-consistent conversion field: the influence vector this solve built is the only place the fit's
      // own curvature weights exist, so the effective field is re-derived here and republished into the
      // slot the extraction, outlier and next prep read. The solve's own arithmetic is untouched.
      if (fieldKernelWeights_ && bMap_ != nullptr) {
        constexpr int kInflInner = generalBrokenLine::kGblInfluenceOffset<N + 1>;
        constexpr int kInflOuter = generalBrokenLine::kGblInfluenceOffset<N - 1>;
        constexpr int kInflSplit = generalBrokenLine::kGblInfluenceOffset<kSplitN>;
        const double* infl = gblScratch + (usedSplit ? kInflSplit : (usedInner ? kInflInner : kInflOuter));
        const generalBrokenLine::GblNodeData* fitNodes = (usedSplit || usedInner) ? gnodes : gnodes + 1;
        const int nFitNodes = usedSplit ? (kSplitN + 1) : (usedInner ? (int(N) + 2) : int(N));
        riemannFit::Map3xNdS<N, Stride> hits(phits + local_idx);
        phase[kBLPhaseBFieldEff] = blKernelWeightedBField(
            acc, hits, int(N), fast_fit, qCharge, bField, bMap_, fitNodes, nFitNodes, infl, phase[kBLPhaseBFieldEff]);
      }
    }
  };

  // Phase (iii): extraction + output. Perigee transform at the PCA (gblHelixAtPca), the
  // re-linearization reference hand-off into the phase buffer, and the SoA writeback (pre-outlier
  // values; the outlier phase conditionally overwrites).
  template <int N, typename TrackerTraits, uint32_t Stride = riemannFit::stride>
  struct Kernel_BLFitPhaseOut {
  public:
    // Normalized (Bz,Br) r-z field map. When set, the curvature->pT conversion + the PCA perigee use a
    // per-track hit-averaged effective field; when null they use the scalar bField.
    const float* __restrict__ bMap_ = nullptr;
    // Read this linearization's effective field from the phase buffer instead of recomputing it; set only
    // where the prep of the same iteration is guaranteed to have run on this lane.
    bool bFieldFromPhase_ = false;
    // Charge-symmetric corrections package (see Kernel_BLFitPhasePrep::chargeSymmetric_).
    bool chargeSymmetric_ = false;

    // One lane of this phase, pinned out of line.
    ALPAKA_FN_ACC BL_REFIT_NOINLINE void lane(Acc1D const& acc,
                                              uint32_t local_idx,
                                              double bField,
                                              OutputSoAView results_view,
                                              typename caStructures::tindex_type const* __restrict__ ptkids,
                                              double* __restrict__ phits,
                                              double* __restrict__ pfast_fit,
                                              double* __restrict__ pphase) const {
      auto tkid = ptkids[local_idx];
      riemannFit::Map3xNdS<N, Stride> hits(phits + local_idx);
      riemannFit::Map4dS<Stride> fast_fit(pfast_fit + local_idx);
      double* phase = pphase + std::size_t(local_idx) * std::size_t(kBLPhaseDoubles);
      const int qCharge = int(phase[kBLPhaseQCharge]);
      const double sTrans0 = phase[kBLPhaseSTrans0];
      const double gchi2 = phase[kBLPhaseGChi2];
      // With bFieldFromPhase_ the prep of this linearization evaluated the field on exactly this fast_fit
      // and these hits, so recomputing it here could only reproduce it.
      const double bFieldEff =
          bFieldFromPhase_ ? phase[kBLPhaseBFieldEff] : blEffectiveBField(acc, hits, int(N), fast_fit, bField, bMap_);
      generalBrokenLine::Vector5d corrPca;
      generalBrokenLine::Matrix5d covPca;
      for (int a = 0; a < 5; ++a)
        corrPca(a) = phase[kBLPhaseCorrPca + a];
      for (int r = 0; r < 5; ++r)
        for (int c = 0; c < 5; ++c)
          covPca(r, c) = phase[kBLPhaseCovPca + 5 * r + c];
      generalBrokenLine::Vector5d hp;
      generalBrokenLine::Matrix5d hc;
      Eigen::Vector4d nextRef;
      generalBrokenLine::gblHelixAtPca(
          acc, fast_fit, qCharge, bFieldEff, sTrans0, hits(2, 0), corrPca, covPca, hp, hc, &nextRef, chargeSymmetric_);
      phase[kBLPhaseNextRef + 0] = nextRef(0);
      phase[kBLPhaseNextRef + 1] = nextRef(1);
      phase[kBLPhaseNextRef + 2] = nextRef(2);
      phase[kBLPhaseNextRef + 3] = nextRef(3);
      const float ptVal = float(bFieldEff / alpaka::math::abs(acc, hp(2)));
      hp(2) /= bFieldEff;
      for (int a = 0; a < 5; ++a) {
        hc(2, a) /= bFieldEff;
        hc(a, 2) /= bFieldEff;
      }
      reco::copyFromDense(results_view, hp, hc, tkid);
      results_view[tkid].pt() = ptVal;
      results_view[tkid].eta() = alpaka::math::asinh(acc, hp(3));
      const int ndof = 2 * N - 5;  // no outlier drop yet (the outlier phase overwrites on a drop)
      results_view[tkid].chi2() = float(gchi2 / (ndof > 0 ? ndof : 1));
      results_view[tkid].ndof() = int8_t(ndof > 0 ? ndof : 1);
#ifdef BL_LAYER_DUMP
      // The fitted state (SoA convention: hp(2) = 1/pt) and the native chi2.
      printf("BLDUMP_FIT %u hp %.17g %.17g %.17g %.17g %.17g chi2 %.17g\n",
             (unsigned)tkid,
             hp(0),
             hp(1),
             hp(2),
             hp(3),
             hp(4),
             gchi2);
#endif
    }
  };

  // Phase (iv): outlier rejection (final linearization only, N >= 5; the launcher skips the
  // launch otherwise). Smoothed-residual pulls from the solve's gFullDelta/gNodeVar, drop the
  // single worst hit above the cut, re-solve ONCE and overwrite the SoA output.
  template <int N, typename TrackerTraits, uint32_t Stride = riemannFit::stride>
  struct Kernel_BLFitPhaseOutlier {
  public:
    const bool outlierReject_ = true;
    // When both are set and a hit is dropped, the dropped hit's raw id is written to dropHitId_[tkid] from
    // the per-lane fitHitId_ table. Both null: nothing is written.
    const uint32_t* __restrict__ fitHitId_ = nullptr;
    uint32_t* __restrict__ dropHitId_ = nullptr;
    // Core-protected outlier: with coreProtect_ and fitHitIsCore_ set, the worst-pull scan skips original
    // pixel-core nodes (fitHitIsCore_[lane*kRefitFitIdQuota + i] != 0), so the drop can only land on an
    // appended extra.
    const uint8_t* __restrict__ fitHitIsCore_ = nullptr;
    bool coreProtect_ = false;
    // Normalized (Bz,Br) r-z field map. When set, the re-solve's curvature->pT conversion + PCA perigee
    // use a per-track hit-averaged effective field; when null (as on the CA path) they use the scalar
    // bField.
    const float* __restrict__ bMap_ = nullptr;
    // See Kernel_BLFitPhaseOut::bFieldFromPhase_. This phase runs after the final out phase, so the value
    // in the buffer is the final linearization's.
    bool bFieldFromPhase_ = false;
    // Charge-symmetric corrections package (see Kernel_BLFitPhasePrep::chargeSymmetric_).
    bool chargeSymmetric_ = false;
    // One lane of this phase, pinned out of line. The invalid-tkid sentinel is tested by the caller.
    ALPAKA_FN_ACC BL_REFIT_NOINLINE void lane(Acc1D const& acc,
                                              uint32_t local_idx,
                                              double bField,
                                              OutputSoAView results_view,
                                              typename caStructures::tindex_type const* __restrict__ ptkids,
                                              double* __restrict__ phits,
                                              double* __restrict__ pfast_fit,
                                              generalBrokenLine::GblNodeData* __restrict__ pgnodes,
                                              double* __restrict__ pscratch,
                                              double* __restrict__ pphase) const {
      auto tkid = ptkids[local_idx];
      riemannFit::Map3xNdS<N, Stride> hits(phits + local_idx);
      riemannFit::Map4dS<Stride> fast_fit(pfast_fit + local_idx);
      constexpr int kSlotNodes = generalBrokenLine::kGblSplitNodes<(
          int(N) > int(TrackerTraits::maxHitsOnTrackForFullFit) ? int(N)
                                                                : int(TrackerTraits::maxHitsOnTrackForFullFit))>;
      generalBrokenLine::GblNodeData* gnodes = pgnodes + std::size_t(local_idx) * std::size_t(kSlotNodes);
      // Sized for the widest node chain this lane can build (the exact split's 2N+1 nodes, else the
      // arrival-node layout's N+2), so the band region and the pull region never overlap in either layout.
      constexpr int kSplitN = generalBrokenLine::kGblSplitNodes<N> - 1;
      constexpr int kBandDoubles = generalBrokenLine::kGblScratchDoubles<kSplitN>;
      // Scratch overlay: gFullDelta lives in the head of the band region (= Mb); gNodeVar stays
      // outside it. See GeneralBrokenLine.h for the lifetime proof.
      constexpr int kNodeVarDoubles = 3 * (kSplitN + 1);
      constexpr int kPullsDoubles = kNodeVarDoubles;
      constexpr int kScratchStride = kBandDoubles + kPullsDoubles;
      // structural guard: the N-1 branch has the smallest Mb of the three layouts.
      static_assert((2 * kSplitN + 3) <= 2 * int(N) * (generalBrokenLine::kGblBand + 1),
                    "scratch overlay: fullDelta does not fit inside Mb");
      double* gblScratch = pscratch + std::size_t(local_idx) * std::size_t(kScratchStride);
      double* gFullDelta = gblScratch;
      double* gNodeVar = gblScratch + kBandDoubles;
      double* phase = pphase + std::size_t(local_idx) * std::size_t(kBLPhaseDoubles);
      const bool usedInner = phase[kBLPhaseUsedInner] != 0.;
      const int qCharge = int(phase[kBLPhaseQCharge]);
      const double sTrans0 = phase[kBLPhaseSTrans0];
      const double innerXX0 = phase[kBLPhaseInnerXX0];
      // This linearization's effective field (see Kernel_BLFitPhaseOut::lane for the phase-buffer read).
      const double bFieldEff =
          bFieldFromPhase_ ? phase[kBLPhaseBFieldEff] : blEffectiveBField(acc, hits, int(N), fast_fit, bField, bMap_);
      // The hits-only layout carrying upstream inner material cannot be re-solved faithfully, so it is
      // excluded here.
      const bool canResolve = outlierReject_ && (phase[kBLPhaseSplit] != 0. || usedInner || !(innerXX0 > 0.));
      if (!canResolve)
        return;
      const bool usedSplit = phase[kBLPhaseSplit] != 0.;
      generalBrokenLine::GblNodeData* fitNodes = (usedSplit || usedInner) ? gnodes : gnodes + 1;
      const int nFitNodes = usedSplit ? (kSplitN + 1) : (usedInner ? (N + 2) : N);
      auto uIdxOf = [](int k) { return 1 + 2 * k; };
      double worst = 0.;
      int worstNode = -1;
      // Fit-hit index of the current measurement node (number of measured nodes before k), used to look up
      // its core flag; the same mapping the drop-id emit below uses.
      int diScan = 0;
      const bool coreProtectOn = coreProtect_ && fitHitIsCore_ != nullptr;
      // Worst pull among protected (core) nodes; it drives the abstain rule below.
      double worstCore = 0.;
      for (int k = 0; k < nFitNodes; ++k) {
        if (!fitNodes[k].hasMeas)
          continue;
        const int diThis = diScan++;
        const bool isCoreNode =
            coreProtectOn &&
            fitHitIsCore_[std::size_t(local_idx) * std::size_t(kRefitFitIdQuota) + std::size_t(diThis)] != 0;
        // A core node never becomes a drop candidate: it only feeds the worstCore shadow that decides
        // whether the stage abstains.
        const double ru = fitNodes[k].measResidual(0) - gFullDelta[uIdxOf(k)];
        const double rv = fitNodes[k].measResidual(1) - gFullDelta[uIdxOf(k) + 1];
        const generalBrokenLine::Matrix2d V = generalBrokenLine::inv2(fitNodes[k].measPrec);
        const double s00 = V(0, 0) - gNodeVar[3 * k];
        const double s01 = V(0, 1) - gNodeVar[3 * k + 1];
        const double s11 = V(1, 1) - gNodeVar[3 * k + 2];
        const double det = s00 * s11 - s01 * s01;
        if (s00 <= 0. || s11 <= 0. || det <= 0.)
          continue;  // numerically non-PD residual covariance: no reliable pull
        const double pull2 = (ru * (s11 * ru - s01 * rv) + rv * (s00 * rv - s01 * ru)) / det;
        if (isCoreNode) {
          if (pull2 > worstCore)  // shadow only: never a drop candidate
            worstCore = pull2;
          continue;
        }
        if (pull2 > worst) {
          worst = pull2;
          worstNode = k;
        }
      }
      constexpr double kOutlierChi2Cut = 13.8;  // ~99.9% of the chi2 distribution for 2 dof
      // Abstain when the largest-pull measured node is a protected core node: the evidence points at a hit
      // the stage may not delete, and dropping the next-worst instead would be a different hypothesis.
      const bool abstain = coreProtectOn && worstCore > worst;
      if (!abstain && worstNode >= 0 && worst > kOutlierChi2Cut) {
        fitNodes[worstNode].hasMeas = false;
        // The dropped measurement node maps to fit hit i = number of measured nodes before it; publish its
        // raw id, fitHitId_[lane*kRefitFitIdQuota + i], so the merger drops it from the emitted hit list.
        if (fitHitId_ != nullptr && dropHitId_ != nullptr) {
          int di = 0;
          for (int k2 = 0; k2 < worstNode; ++k2)
            if (fitNodes[k2].hasMeas)
              ++di;
          dropHitId_[tkid] = fitHitId_[std::size_t(local_idx) * std::size_t(kRefitFitIdQuota) + std::size_t(di)];
        }
        double gchi2 = 0.;
        generalBrokenLine::Vector5d corrPca2;
        generalBrokenLine::Matrix5d covPca2;
        if (usedSplit) {
          covPca2 = generalBrokenLine::gblFitPca<Acc1D, kSplitN>(acc, gnodes, gblScratch, &corrPca2, nullptr, &gchi2);
        } else if (usedInner) {
          covPca2 = generalBrokenLine::gblFitPca<Acc1D, N + 1>(acc, gnodes, gblScratch, &corrPca2, nullptr, &gchi2);
        } else {  // canResolve guarantees no upstream re-add is needed here
          covPca2 = generalBrokenLine::gblFitPca<Acc1D, N - 1>(acc, gnodes + 1, gblScratch, &corrPca2, nullptr, &gchi2);
          generalBrokenLine::Matrix5d jacBack;
          for (int r = 0; r < 5; ++r)
            for (int c = 0; c < 5; ++c)
              jacBack(r, c) = phase[kBLPhaseJacBack + 5 * r + c];
          corrPca2 = jacBack * corrPca2;
          covPca2 = jacBack * covPca2 * jacBack.transpose();
        }
        generalBrokenLine::Vector5d hp;
        generalBrokenLine::Matrix5d hc;
        generalBrokenLine::gblHelixAtPca(
            acc, fast_fit, qCharge, bFieldEff, sTrans0, hits(2, 0), corrPca2, covPca2, hp, hc, nullptr, chargeSymmetric_);
        const float ptVal = float(bFieldEff / alpaka::math::abs(acc, hp(2)));
        hp(2) /= bFieldEff;
        for (int a = 0; a < 5; ++a) {
          hc(2, a) /= bFieldEff;
          hc(a, 2) /= bFieldEff;
        }
        reco::copyFromDense(results_view, hp, hc, tkid);
        results_view[tkid].pt() = ptVal;
        results_view[tkid].eta() = alpaka::math::asinh(acc, hp(3));
        const int ndof = 2 * (N - 1) - 5;  // one measurement dropped
        results_view[tkid].chi2() = float(gchi2 / (ndof > 0 ? ndof : 1));
        results_view[tkid].ndof() = int8_t(ndof > 0 ? ndof : 1);
      }
    }
  };

  // Fused phase ladder. Per bin, one (phase, iteration) would be ten serialised launches with a few
  // hundred live lanes each, while the solver (one thread marching a bordered-band double solve) is
  // latency-bound; with the lane partition all ten bins run in one launch and wall time is the slowest
  // bin, not their sum. Bin b runs the out-of-line Kernel_BLFitPhase*<b+kRefitMinN>::lane through the
  // trampolines below; no lane body reads its lane index beyond its own addresses.
  //
  // N-independent config of one fused launch: the members the per-N phase kernels carry, so one object
  // serves all ten cases; the trampoline rebuilds the bin's kernel from it inside the callee's frame.
  struct BLRefitFusedCfg {
    const float* __restrict__ rhoMap = nullptr;
    const float* __restrict__ bMap = nullptr;
    const uint32_t* __restrict__ fitHitId = nullptr;
    uint32_t* __restrict__ dropHitId = nullptr;
    const uint8_t* __restrict__ fitHitIsCore = nullptr;
    bool iterFromPhase = false;
    bool matFromPhase = false;
    bool fieldFromPhase = false;
    bool bFieldFromPhase = false;
    bool chargeSymmetric = false;
    bool trajectoryCorrections = false;
    bool scatteringLogAtTotal = false;
    bool elossCumulative = false;
    bool outlierReject = true;
    bool finalIter = true;
    bool fieldKernelWeights = false;
    bool coreProtect = false;
  };

  template <int N, typename TrackerTraits, uint32_t Stride>
  ALPAKA_FN_ACC BL_REFIT_NOINLINE void blFitPhasePrepLaneOutOfLine(
      Acc1D const& acc,
      BLRefitFusedCfg const& cfg,
      uint32_t lane,
      double bField,
      typename caStructures::tindex_type const* __restrict__ ptkids,
      double* __restrict__ phits,
      float* __restrict__ phits_ge,
      double* __restrict__ pfast_fit,
      generalBrokenLine::GblNodeData* __restrict__ pgnodes,
      double* __restrict__ pphase) {
    const Kernel_BLFitPhasePrep<N, TrackerTraits, Stride> k{cfg.rhoMap,
                                                            cfg.iterFromPhase,
                                                            cfg.bMap,
                                                            cfg.matFromPhase,
                                                            cfg.fieldFromPhase,
                                                            cfg.chargeSymmetric,
                                                            cfg.trajectoryCorrections,
                                                            cfg.scatteringLogAtTotal,
                                                            cfg.elossCumulative};
    k.lane(acc, lane, bField, ptkids, phits, phits_ge, pfast_fit, pgnodes, pphase);
  }

  template <int N, typename TrackerTraits, uint32_t Stride>
  ALPAKA_FN_ACC BL_REFIT_NOINLINE void blFitPhaseSolveLaneOutOfLine(Acc1D const& acc,
                                                                    BLRefitFusedCfg const& cfg,
                                                                    uint32_t lane,
                                                                    double bField,
                                                                    double* __restrict__ phits,
                                                                    double* __restrict__ pfast_fit,
                                                                    generalBrokenLine::GblNodeData* __restrict__ pgnodes,
                                                                    double* __restrict__ pscratch,
                                                                    double* __restrict__ pphase) {
    const Kernel_BLFitPhaseSolve<N, TrackerTraits, Stride> k{
        cfg.outlierReject, cfg.finalIter, cfg.bMap, cfg.fieldKernelWeights};
    k.lane(acc, lane, bField, phits, pfast_fit, pgnodes, pscratch, pphase);
  }

  template <int N, typename TrackerTraits, uint32_t Stride>
  ALPAKA_FN_ACC BL_REFIT_NOINLINE void blFitPhaseOutLaneOutOfLine(
      Acc1D const& acc,
      BLRefitFusedCfg const& cfg,
      uint32_t lane,
      double bField,
      OutputSoAView results_view,
      typename caStructures::tindex_type const* __restrict__ ptkids,
      double* __restrict__ phits,
      double* __restrict__ pfast_fit,
      double* __restrict__ pphase) {
    const Kernel_BLFitPhaseOut<N, TrackerTraits, Stride> k{cfg.bMap, cfg.bFieldFromPhase, cfg.chargeSymmetric};
    k.lane(acc, lane, bField, results_view, ptkids, phits, pfast_fit, pphase);
  }

  template <int N, typename TrackerTraits, uint32_t Stride>
  ALPAKA_FN_ACC BL_REFIT_NOINLINE void blFitPhaseOutlierLaneOutOfLine(
      Acc1D const& acc,
      BLRefitFusedCfg const& cfg,
      uint32_t lane,
      double bField,
      OutputSoAView results_view,
      typename caStructures::tindex_type const* __restrict__ ptkids,
      double* __restrict__ phits,
      double* __restrict__ pfast_fit,
      generalBrokenLine::GblNodeData* __restrict__ pgnodes,
      double* __restrict__ pscratch,
      double* __restrict__ pphase) {
    const Kernel_BLFitPhaseOutlier<N, TrackerTraits, Stride> k{cfg.outlierReject,
                                                               cfg.fitHitId,
                                                               cfg.dropHitId,
                                                               cfg.fitHitIsCore,
                                                               cfg.coreProtect,
                                                               cfg.bMap,
                                                               cfg.bFieldFromPhase,
                                                               cfg.chargeSymmetric};
    k.lane(acc, lane, bField, results_view, ptkids, phits, pfast_fit, pgnodes, pscratch, pphase);
  }

  // Fused kernels. The block index picks the bin (and its compile-time N) out of the device table
  // Kernel_BLRefitLaneRanges wrote; the block's elements are the bin's lanes. One bin per block, so the
  // switch over N cannot diverge inside a warp.
  template <typename TrackerTraits, uint32_t Stride>
  struct Kernel_BLFitPhasePrepFused {
    static_assert(Stride == HelixFit<TrackerTraits>::kRefitStride, "the fused ladder spans the refit lane buffer");
    BLRefitFusedCfg cfg_;

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  double bField,
                                  typename caStructures::tindex_type const* __restrict__ ptkids,
                                  double* __restrict__ phits,
                                  float* __restrict__ phits_ge,
                                  double* __restrict__ pfast_fit,
                                  generalBrokenLine::GblNodeData* __restrict__ pgnodes,
                                  double* __restrict__ pphase,
                                  const uint32_t* __restrict__ pBlockMap) const {
      constexpr auto invalidTkId = std::numeric_limits<typename caStructures::tindex_type>::max();
      constexpr uint32_t kTotal = kFusedBlocks * kFitBlock;
      for (uint32_t blk : cms::alpakatools::uniform_groups(acc, kTotal)) {
        const uint32_t bin = pBlockMap[kRefitBlockMapStride * blk];
        if (bin >= kRefitNBins)
          continue;  // idle block: this round's partition did not need it
        const uint32_t laneLo = pBlockMap[kRefitBlockMapStride * blk + 1];
        const uint32_t laneHi = pBlockMap[kRefitBlockMapStride * blk + 2];
        for (auto el : cms::alpakatools::uniform_group_elements(acc, blk, kTotal)) {
          const uint32_t local_idx = laneLo + uint32_t(el.local);
          if (local_idx >= laneHi)
            break;
          // Safety net: a bin's range is its exact count, so the scan fills every lane of it.
          if (invalidTkId == ptkids[local_idx])
            continue;
#define BL_REFIT_FUSED_PREP_CASE(BIN)                                                       \
  case (BIN):                                                                               \
    blFitPhasePrepLaneOutOfLine<(BIN) + kRefitMinN, TrackerTraits, Stride>(                 \
        acc, cfg_, local_idx, bField, ptkids, phits, phits_ge, pfast_fit, pgnodes, pphase); \
    break
          switch (bin) {
            BL_REFIT_FUSED_PREP_CASE(0);
            BL_REFIT_FUSED_PREP_CASE(1);
            BL_REFIT_FUSED_PREP_CASE(2);
            BL_REFIT_FUSED_PREP_CASE(3);
            BL_REFIT_FUSED_PREP_CASE(4);
            BL_REFIT_FUSED_PREP_CASE(5);
            BL_REFIT_FUSED_PREP_CASE(6);
            BL_REFIT_FUSED_PREP_CASE(7);
            BL_REFIT_FUSED_PREP_CASE(8);
            BL_REFIT_FUSED_PREP_CASE(9);
            default:
              break;
          }
#undef BL_REFIT_FUSED_PREP_CASE
        }
      }
    }
  };

  template <typename TrackerTraits, uint32_t Stride>
  struct Kernel_BLFitPhaseSolveFused {
    static_assert(Stride == HelixFit<TrackerTraits>::kRefitStride, "the fused ladder spans the refit lane buffer");
    BLRefitFusedCfg cfg_;

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  double bField,
                                  typename caStructures::tindex_type const* __restrict__ ptkids,
                                  double* __restrict__ phits,
                                  double* __restrict__ pfast_fit,
                                  generalBrokenLine::GblNodeData* __restrict__ pgnodes,
                                  double* __restrict__ pscratch,
                                  double* __restrict__ pphase,
                                  const uint32_t* __restrict__ pBlockMap) const {
      constexpr auto invalidTkId = std::numeric_limits<typename caStructures::tindex_type>::max();
      constexpr uint32_t kTotal = kFusedBlocks * kFitBlock;
      for (uint32_t blk : cms::alpakatools::uniform_groups(acc, kTotal)) {
        const uint32_t bin = pBlockMap[kRefitBlockMapStride * blk];
        if (bin >= kRefitNBins)
          continue;
        const uint32_t laneLo = pBlockMap[kRefitBlockMapStride * blk + 1];
        const uint32_t laneHi = pBlockMap[kRefitBlockMapStride * blk + 2];
        for (auto el : cms::alpakatools::uniform_group_elements(acc, blk, kTotal)) {
          const uint32_t local_idx = laneLo + uint32_t(el.local);
          if (local_idx >= laneHi)
            break;
          if (invalidTkId == ptkids[local_idx])
            continue;
#define BL_REFIT_FUSED_SOLVE_CASE(BIN)                                              \
  case (BIN):                                                                       \
    blFitPhaseSolveLaneOutOfLine<(BIN) + kRefitMinN, TrackerTraits, Stride>(        \
        acc, cfg_, local_idx, bField, phits, pfast_fit, pgnodes, pscratch, pphase); \
    break
          switch (bin) {
            BL_REFIT_FUSED_SOLVE_CASE(0);
            BL_REFIT_FUSED_SOLVE_CASE(1);
            BL_REFIT_FUSED_SOLVE_CASE(2);
            BL_REFIT_FUSED_SOLVE_CASE(3);
            BL_REFIT_FUSED_SOLVE_CASE(4);
            BL_REFIT_FUSED_SOLVE_CASE(5);
            BL_REFIT_FUSED_SOLVE_CASE(6);
            BL_REFIT_FUSED_SOLVE_CASE(7);
            BL_REFIT_FUSED_SOLVE_CASE(8);
            BL_REFIT_FUSED_SOLVE_CASE(9);
            default:
              break;
          }
#undef BL_REFIT_FUSED_SOLVE_CASE
        }
      }
    }
  };

  template <typename TrackerTraits, uint32_t Stride>
  struct Kernel_BLFitPhaseOutFused {
    static_assert(Stride == HelixFit<TrackerTraits>::kRefitStride, "the fused ladder spans the refit lane buffer");
    BLRefitFusedCfg cfg_;

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  double bField,
                                  OutputSoAView results_view,
                                  typename caStructures::tindex_type const* __restrict__ ptkids,
                                  double* __restrict__ phits,
                                  double* __restrict__ pfast_fit,
                                  double* __restrict__ pphase,
                                  const uint32_t* __restrict__ pBlockMap) const {
      constexpr auto invalidTkId = std::numeric_limits<typename caStructures::tindex_type>::max();
      constexpr uint32_t kTotal = kFusedBlocks * kFitBlock;
      for (uint32_t blk : cms::alpakatools::uniform_groups(acc, kTotal)) {
        const uint32_t bin = pBlockMap[kRefitBlockMapStride * blk];
        if (bin >= kRefitNBins)
          continue;
        const uint32_t laneLo = pBlockMap[kRefitBlockMapStride * blk + 1];
        const uint32_t laneHi = pBlockMap[kRefitBlockMapStride * blk + 2];
        for (auto el : cms::alpakatools::uniform_group_elements(acc, blk, kTotal)) {
          const uint32_t local_idx = laneLo + uint32_t(el.local);
          if (local_idx >= laneHi)
            break;
          if (invalidTkId == ptkids[local_idx])
            continue;
#define BL_REFIT_FUSED_OUT_CASE(BIN)                                                   \
  case (BIN):                                                                          \
    blFitPhaseOutLaneOutOfLine<(BIN) + kRefitMinN, TrackerTraits, Stride>(             \
        acc, cfg_, local_idx, bField, results_view, ptkids, phits, pfast_fit, pphase); \
    break
          switch (bin) {
            BL_REFIT_FUSED_OUT_CASE(0);
            BL_REFIT_FUSED_OUT_CASE(1);
            BL_REFIT_FUSED_OUT_CASE(2);
            BL_REFIT_FUSED_OUT_CASE(3);
            BL_REFIT_FUSED_OUT_CASE(4);
            BL_REFIT_FUSED_OUT_CASE(5);
            BL_REFIT_FUSED_OUT_CASE(6);
            BL_REFIT_FUSED_OUT_CASE(7);
            BL_REFIT_FUSED_OUT_CASE(8);
            BL_REFIT_FUSED_OUT_CASE(9);
            default:
              break;
          }
#undef BL_REFIT_FUSED_OUT_CASE
        }
      }
    }
  };

  // The outlier phase runs for N >= 5 only: bins 0 and 1 fall through to `default`, their solve never
  // filled the pulls it would read.
  template <typename TrackerTraits, uint32_t Stride>
  struct Kernel_BLFitPhaseOutlierFused {
    static_assert(Stride == HelixFit<TrackerTraits>::kRefitStride, "the fused ladder spans the refit lane buffer");
    BLRefitFusedCfg cfg_;

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  double bField,
                                  OutputSoAView results_view,
                                  typename caStructures::tindex_type const* __restrict__ ptkids,
                                  double* __restrict__ phits,
                                  double* __restrict__ pfast_fit,
                                  generalBrokenLine::GblNodeData* __restrict__ pgnodes,
                                  double* __restrict__ pscratch,
                                  double* __restrict__ pphase,
                                  const uint32_t* __restrict__ pBlockMap) const {
      constexpr auto invalidTkId = std::numeric_limits<typename caStructures::tindex_type>::max();
      constexpr uint32_t kTotal = kFusedBlocks * kFitBlock;
      for (uint32_t blk : cms::alpakatools::uniform_groups(acc, kTotal)) {
        const uint32_t bin = pBlockMap[kRefitBlockMapStride * blk];
        if (bin >= kRefitNBins)
          continue;
        const uint32_t laneLo = pBlockMap[kRefitBlockMapStride * blk + 1];
        const uint32_t laneHi = pBlockMap[kRefitBlockMapStride * blk + 2];
        for (auto el : cms::alpakatools::uniform_group_elements(acc, blk, kTotal)) {
          const uint32_t local_idx = laneLo + uint32_t(el.local);
          if (local_idx >= laneHi)
            break;
          if (invalidTkId == ptkids[local_idx])
            continue;
#define BL_REFIT_FUSED_OUTLIER_CASE(BIN)                                                                  \
  case (BIN):                                                                                             \
    blFitPhaseOutlierLaneOutOfLine<(BIN) + kRefitMinN, TrackerTraits, Stride>(                            \
        acc, cfg_, local_idx, bField, results_view, ptkids, phits, pfast_fit, pgnodes, pscratch, pphase); \
    break
          switch (bin) {
            BL_REFIT_FUSED_OUTLIER_CASE(2);
            BL_REFIT_FUSED_OUTLIER_CASE(3);
            BL_REFIT_FUSED_OUTLIER_CASE(4);
            BL_REFIT_FUSED_OUTLIER_CASE(5);
            BL_REFIT_FUSED_OUTLIER_CASE(6);
            BL_REFIT_FUSED_OUTLIER_CASE(7);
            BL_REFIT_FUSED_OUTLIER_CASE(8);
            BL_REFIT_FUSED_OUTLIER_CASE(9);
            default:
              break;  // bins 0 and 1 are N = 3, 4: no outlier phase
          }
#undef BL_REFIT_FUSED_OUTLIER_CASE
        }
      }
    }
  };

#undef BL_REFIT_NOINLINE

  // Extended-N refit. Each accepted-extended track's rewritten hit container holds its originals plus the
  // attached extras (tagged OT-rechit ids carry bit30); a full GBL refit, whose OT lever arm shrinks the
  // longitudinal and pT covariance, overwrites the pre-refit state, covariance and chi2. The fit kernels
  // are hit-source agnostic: only the hit load dispatches merged vs OT.

  // Fit-hit selection over the rewritten container (caFitHitSel::dedupWalk), including tagged OT-rechit
  // ids (bit30), which index the raw OT source and count as fit hits. Same kMode filter and pixel-overlap
  // dedup. k < 0 counts; k >= 0 returns the k-th.
  template <typename TupleCont, typename HitsView>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t
  refitDedupWalk(TupleCont const* __restrict__ hitContainer, uint32_t it, HitsView hh, bool hasStubs, int k) {
    auto const* hitId = hitContainer->begin(it);
    auto const nhits = hitContainer->size(it);
    auto const nTot = hh.metadata().size();
    uint32_t nkept = 0;
    int lastKeptJ = -1;
    for (uint32_t j = 0; j < uint32_t(nhits); ++j) {
      auto const h = hitId[j];
      const bool ot = caOTHitTag::isOTId(h);
      if (!ot && h >= static_cast<uint32_t>(nTot))
        break;  // content-overflow guard (untagged out-of-range id)
      // OT extras count as stubs for the kMode filter; merged ids consult the SoA stub flag.
      const bool hitIsStub = ot ? true : reco::isStub(hh, int32_t(h));
      if (!caFitHitSel::useHit(hitIsStub, hasStubs))
        continue;
      // Merge only two consecutive kept merged-pixel hits, never an OT extra or a stub.
      if (hasStubs && lastKeptJ >= 0 && !ot && !reco::isStub(hh, int32_t(h))) {
        auto const hp = hitId[lastKeptJ];
        if (!caOTHitTag::isOTId(hp) && !reco::isStub(hh, int32_t(hp))) {
          float const dx = float(hh[h].xGlobal()) - float(hh[hp].xGlobal());
          float const dy = float(hh[h].yGlobal()) - float(hh[hp].yGlobal());
          if (dx * dx + dy * dy < caFitHitSel::kDedupDsMin2)
            continue;
        }
      }
      if (int(nkept) == k)
        return j;
      ++nkept;
      lastKeptJ = int(j);
    }
    return nkept;
  }

  // Dynamic partition, part 1: count. nSel does not depend on N, so one sweep produces all ten demands.
  // It walks the same population, gates and refitDedupWalk as the per-bin scans, binned with
  // refitBinOfNSel, so count and claim cannot disagree. The counters are the demand of this round.
  class Kernel_BLRefitBinCount {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  Tuples const* __restrict__ hitContainer,
                                  ::reco::TrackingRecHitConstView hh,
                                  const int32_t* __restrict__ acceptedByTuple,
                                  const uint8_t* __restrict__ pServed,
                                  uint32_t* __restrict__ pCounts,
                                  uint32_t maxFitSel,
                                  uint32_t nTracksCap,
                                  bool hasStubs) const {
      const bool hasStubsRt = hasStubs && (static_cast<int32_t>(hh.offsetStubs()) >= 0);
      for (auto tkid : cms::alpakatools::uniform_elements(acc, nTracksCap)) {
        if (acceptedByTuple[tkid] < 0)
          continue;
        if (pServed != nullptr && pServed[tkid] != 0u)
          continue;  // seated in an earlier round
        const uint32_t nSel = refitDedupWalk(hitContainer, tkid, hh, hasStubsRt, /*k=*/-1);
        const uint32_t bin = refitBinOfNSel(nSel, maxFitSel);
        if (bin >= kRefitNBins)
          continue;  // outside the ladder: no bin fits this track
        alpaka::atomicAdd(acc, pCounts + bin, 1u, alpaka::hierarchy::Grids{});
      }
    }
  };

  // Dynamic partition, part 2: ranges. Single-threaded, device-only: hands kRefitStride lanes out in bin
  // order and writes the block->bin dispatch table (one bin per block, idle tail at kRefitNBins). The
  // per-track served flag makes the round bound exact.
  class Kernel_BLRefitLaneRanges {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const uint32_t* __restrict__ pCounts,
                                  uint32_t* __restrict__ pRange,
                                  uint32_t* __restrict__ pBlockMap,
                                  uint32_t laneTotal) const {
      if (alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0] != 0)
        return;
      uint32_t base = 0u;
      for (uint32_t b = 0; b < kRefitNBins; ++b) {
        const uint32_t want = pCounts[b];
        const uint32_t room = (base < laneTotal) ? (laneTotal - base) : 0u;
        const uint32_t got = (want < room) ? want : room;
        pRange[kRefitRangeStride * b + 0] = base;
        pRange[kRefitRangeStride * b + 1] = got;
        base += got;
      }
      uint32_t blk = 0u;
      for (uint32_t b = 0; b < kRefitNBins; ++b) {
        const uint32_t lo = pRange[kRefitRangeStride * b + 0];
        const uint32_t hi = lo + pRange[kRefitRangeStride * b + 1];
        for (uint32_t l = lo; l < hi; l += kFitBlock) {
          ALPAKA_ASSERT_ACC(blk < kFusedBlocks);  // sum_b ceil(n_b/kFitBlock) <= kFusedBlocks
          const uint32_t e = (l + kFitBlock < hi) ? (l + kFitBlock) : hi;
          pBlockMap[kRefitBlockMapStride * blk + 0] = b;
          pBlockMap[kRefitBlockMapStride * blk + 1] = l;
          pBlockMap[kRefitBlockMapStride * blk + 2] = e;
          ++blk;
        }
      }
      for (; blk < kFusedBlocks; ++blk) {
        pBlockMap[kRefitBlockMapStride * blk + 0] = kRefitNBins;  // idle
        pBlockMap[kRefitBlockMapStride * blk + 1] = 0u;
        pBlockMap[kRefitBlockMapStride * blk + 2] = 0u;
      }
    }
  };

  // Prepares the BLFit input buffers for the extended-N refit: keeps accepted-extended tuples whose
  // selected multiplicity falls in [nHitsL,nHitsH] and dense-compacts them into ptkids/buffers through a
  // grid atomic (ptkids pre-set to the invalid sentinel, preserving the break-on-invalid tail). Tagged ids
  // load from the raw OT SoA, the rest from the merged SoA through innerSensorFrame.
  template <int N, uint32_t Stride = riemannFit::stride>
  class Kernel_BLFastFitRefit {
  public:
    // OT source by value (as the extension kernels take it): default (nOTHits==0) => no tagged ids.
    caExtension::OTHitsSource otSource_{};
    // Per-lane raw hit id of each fit slot (fitHitId_[lane*kRefitFitIdQuota + i]). The outlier phase maps
    // its dropped measurement node back to this id, so the merger can remove the fit-rejected hit from the
    // emitted TrackHitSoA list. Null means no such record is kept.
    uint32_t* __restrict__ fitHitId_ = nullptr;
    // Core-protected outlier: per-lane flag of whether each fit slot is an original pixel-core hit (id
    // below offsetStubs, not a bit30 OT tag), read by the outlier phase to restrict the drop to appended
    // nodes. Null means no such record is kept.
    uint8_t* __restrict__ fitHitIsCore_ = nullptr;
    // Device-resident lane range of this bin for this round, {firstLane, nLanes} at
    // pLaneRange_[kRefitRangeStride*(N-kRefitMinN)], written by Kernel_BLRefitLaneRanges. Null means base 0
    // and the whole buffer stride. Decides where a lane lands, never a value.
    const uint32_t* __restrict__ pLaneRange_ = nullptr;
    // Per-track "already seated in an earlier round" flag, nTracksCap bytes: set the moment a track claims
    // a lane, so consecutive rounds partition the population exactly. Null means single-round behaviour
    // with no gate and no marking.
    uint8_t* __restrict__ pServed_ = nullptr;

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  Tuples const* __restrict__ hitContainer,
                                  ::reco::TrackingRecHitConstView hh,
                                  ::reco::CAModulesConstView cm,
                                  const int32_t* __restrict__ acceptedByTuple,
                                  typename caStructures::tindex_type* __restrict__ ptkids,
                                  double* __restrict__ phits,
                                  float* __restrict__ phits_ge,
                                  double* __restrict__ pfast_fit,
                                  uint32_t* __restrict__ pSlot,
                                  uint32_t nHitsL,
                                  uint32_t nHitsH,
                                  uint32_t nTracksCap,
                                  bool hasStubs) const {
      constexpr uint32_t hitsInFit = N;
      constexpr uint32_t kBin = uint32_t(N) - uint32_t(kRefitMinN);
      // Where this bin's lanes are in the shared stride-wide buffer, and how many of them there are.
      // Without a range table this is the whole buffer from lane 0.
      const uint32_t laneBase = (pLaneRange_ != nullptr) ? pLaneRange_[kRefitRangeStride * kBin] : 0u;
      const uint32_t nt = (pLaneRange_ != nullptr) ? pLaneRange_[kRefitRangeStride * kBin + 1] : Stride;
      const bool hasStubsRt = hasStubs && (static_cast<int32_t>(hh.offsetStubs()) >= 0);
      for (auto tkid : cms::alpakatools::uniform_elements(acc, nTracksCap)) {
        if (acceptedByTuple[tkid] < 0)
          continue;  // only accepted-extended tuples get refit
        if (pServed_ != nullptr && pServed_[tkid] != 0u)
          continue;  // already refit in an earlier round
        const uint32_t nSel = refitDedupWalk(hitContainer, tkid, hh, hasStubsRt, /*k=*/-1);
        if (nSel < nHitsL || nSel > nHitsH)
          continue;  // not this N-bin
        const uint32_t slot = alpaka::atomicAdd(acc, pSlot, 1u, alpaka::hierarchy::Grids{});
        if (slot >= nt) {
          // With pServed_ set this is the expected round overflow and the track is left for the next
          // round. Without a range table the slot cap binds and the track keeps its unrefit CA state;
          // exactly one thread per launch observes slot == nt, so the report is one line per launch.
          if (pServed_ == nullptr && slot == nt)
            printf(
                "[refit slot cap] BOUND: N-bin %u..%u demanded > kRefitStride %u; tracks beyond the cap keep "
                "their unrefit CA state.\n",
                nHitsL,
                nHitsH,
                nt);
          continue;
        }
        // The lane this claim owns in the shared buffer (laneBase is 0 without a range table).
        const uint32_t lane = laneBase + slot;
        if (pServed_ != nullptr)
          pServed_[tkid] = 1u;  // seated: no later round may claim it again
        ptkids[lane] = caStructures::tindex_type(tkid);

        auto const* hitId = hitContainer->begin(tkid);
        riemannFit::Map3xNdS<N, Stride> hits(phits + lane);
        riemannFit::Map4dS<Stride> fast_fit(pfast_fit + lane);
        riemannFit::Map6xNfS<N, Stride> hits_ge(phits_ge + lane);

        // Uniform sampling of hitsInFit hits from the deduped selected set; the last kept hit is forced
        // for maximum lever arm.
        uint32_t selectedHits[N];
        {
          float incr = std::max(1.f, float(nSel) / float(hitsInFit));
          float fn = 0;
          for (uint32_t i = 0; i < hitsInFit; ++i) {
            int kk = int(fn + 0.5f);
            if (hitsInFit - 1 == i)
              kk = int(nSel) - 1;
            selectedHits[i] = refitDedupWalk(hitContainer, tkid, hh, hasStubsRt, kk);
            fn += incr;
          }
        }

        for (uint32_t i = 0; i < hitsInFit; ++i) {
          const uint32_t hid = hitId[selectedHits[i]];
          // Record this fit slot's raw hit id (bit30 OT tag preserved) so the outlier phase can name the
          // hit it drops. lane is the lane the fit and outlier kernels iterate (ptkids[lane]).
          if (fitHitId_ != nullptr)
            fitHitId_[std::size_t(lane) * std::size_t(kRefitFitIdQuota) + i] = hid;
          // Mark original pixel-core fit hits (id < offsetStubs and not a bit30 OT tag). Appended extras,
          // raw OT and merged OT stubs, stay droppable. Same lane*kRefitFitIdQuota + i indexing as
          // fitHitId_.
          if (fitHitIsCore_ != nullptr) {
            const bool isCore = !caOTHitTag::isOTId(hid) && (!hasStubsRt || int32_t(hid) < int32_t(hh.offsetStubs()));
            fitHitIsCore_[std::size_t(lane) * std::size_t(kRefitFitIdQuota) + i] = isCore ? uint8_t(1) : uint8_t(0);
          }
          float ge[6];
          float px, py, pz;
          if (caOTHitTag::isOTId(hid)) {
            // Raw OT rechit: lower or upper sensor frame by position in the stack.
            const uint32_t o = caOTHitTag::otIdx(hid);
            const uint32_t geom = uint32_t(otSource_.otHits[o].detectorIndex()) - ::phase2PixelTopology::nModulesPix;
            const bool isUpper = (o >= otSource_.otHitModules.upperSensorStart()[geom]);
            const auto& frame = isUpper ? otSource_.stackedGeometry.upperSensorFrame()[geom]
                                        : otSource_.stackedGeometry.lowerSensorFrame()[geom];
            float xerrFit = otSource_.otHits[o].xerrLocal();
            const bool is2S = otSource_.otHits[o].yerrLocal() > 0.1f;
            const bool isBarrelHit = alpaka::math::abs(acc, otSource_.otHits[o].zGlobal()) < 118.f;
            const float f = is2S ? (isBarrelHit ? 0.4624f : 0.8464f) : (isBarrelHit ? 0.64f : 0.9025f);
            xerrFit *= f;
            float yerrFit = otSource_.otHits[o].yerrLocal();
            frame.toGlobal(xerrFit, 0.f, yerrFit, ge);
            px = otSource_.otHits[o].xGlobal();
            py = otSource_.otHits[o].yGlobal();
            pz = otSource_.otHits[o].zGlobal();
          } else {
            // Merged SoA hit: the same load as Kernel_BLFastFit (innerSensorFrame + stub xerr scale).
            auto frame = cm.innerSensorFrame(hh.detectorIndex(hid));
            float xerrFit = hh[hid].xerrLocal();
            if (reco::isStub(hh, int32_t(hid))) {
              const bool is2S = hh[hid].yerrLocal() > 0.1f;
              const bool isBarrelHit = alpaka::math::abs(acc, hh[hid].zGlobal()) < 118.f;
              const float f = is2S ? (isBarrelHit ? 0.4624f : 0.8464f) : (isBarrelHit ? 0.64f : 0.9025f);
              xerrFit *= f;
            }
            float yerrFit = hh[hid].yerrLocal();
            frame.toGlobal(xerrFit, 0.f, yerrFit, ge);
            px = hh[hid].xGlobal();
            py = hh[hid].yGlobal();
            pz = hh[hid].zGlobal();
          }
          hits.col(i) << px, py, pz;
          hits_ge.col(i) << ge[0], ge[1], ge[2], ge[3], ge[4], ge[5];
        }
        brokenline::fastFit(acc, hits, fast_fit);
      }
    }
  };

  // Per-N launcher helpers (extern template). Each bundles the fast-fit and fit launches for one
  // multiplicity N; the launch state travels in a small context POD, so the helper is a plain function
  // template that can be `extern template`-declared here and instantiated in a disjoint-N TU.

  // Main-fit launch context: the launchBrokenLineKernels locals and HelixFit members the launches need.
  template <typename TrackerTraits>
  struct BLMainLaunchCtx {
    Queue& queue;
    Tuples const* tuples;
    TupleMultiplicity const* tupleMultiplicity;
    ::reco::TrackingRecHitConstView hv;
    ::reco::CAModulesConstView cm;
    OutputSoAView outputSoa;
    double bField;
    const float* rhoMap;
    const float* bMap;  // normalized (Bz,Br) r-z field map; null (or fitCorrections off) => the scalar bField
    typename caStructures::tindex_type* tkids;
    double* phits;
    float* phits_ge;
    double* pfast_fit;
    double* pgblScratch;
    bool fitCorrections;  // fit correctness package (see Kernel_BLFit::fitCorrections_)
    // Dynamic-partition state, written on the device by Kernel_BLMainLaneRanges once per round.
    const uint32_t* pRange;     // per-bin {firstLane, nLanes, tupleBase}, kMainRangeStride * kMainNBins
    const uint32_t* pBlockMap;  // per-block {bin, firstLane, endLane}, kMainBlockMapStride * kMainFusedBlocks
    WorkDiv1D workDivFused;     // kMainFusedBlocks x kFitBlock, fixed and host-known
  };

  // Fused main-ladder launchers: one launch per phase, all N-bins, on the fixed fused grid. Declared
  // `extern template` and instantiated one phase per TU, each pulling every compile-time N of its phase
  // into that TU.
  template <typename TrackerTraits>
  void runMainFusedFast(BLMainLaunchCtx<TrackerTraits> const& c) {
    BLMainFusedCfg cfg{};
    cfg.hasStubs = std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>;
    alpaka::exec<Acc1D>(c.queue,
                        c.workDivFused,
                        Kernel_BLFastFitFused<TrackerTraits>{cfg},
                        c.tuples,
                        c.tupleMultiplicity,
                        c.hv,
                        c.cm,
                        c.tkids,
                        c.phits,
                        c.phits_ge,
                        c.pfast_fit,
                        c.pRange,
                        c.pBlockMap);
  }

  template <typename TrackerTraits>
  void runMainFusedFit(BLMainLaunchCtx<TrackerTraits> const& c) {
    BLMainFusedCfg cfg{};
    cfg.rhoMap = c.rhoMap;
    cfg.bMap = c.bMap;
    cfg.fitCorrections = c.fitCorrections;
    alpaka::exec<Acc1D>(c.queue,
                        c.workDivFused,
                        Kernel_BLFitFused<TrackerTraits>{cfg},
                        c.tupleMultiplicity,
                        c.bField,
                        c.outputSoa,
                        c.tkids,
                        c.phits,
                        c.phits_ge,
                        c.pfast_fit,
                        c.pgblScratch,
                        c.pBlockMap);
  }

  // Extended-N refit launch context: the refitExtended locals and HelixFit members the launches need.
  template <typename TrackerTraits>
  struct BLRefitLaunchCtx {
    Queue& queue;
    Tuples const* hitContainer;
    TupleMultiplicity const* tupleMultiplicity;
    ::reco::TrackingRecHitConstView hv;
    ::reco::CAModulesConstView cm;
    OutputSoAView outputSoa;
    const int32_t* acceptedByTuple;
    caExtension::OTHitsSource otSrc;
    double bField;
    const float* rhoMap;
    const float* bMap;  // normalized (Bz,Br) r-z field map; null => the scalar bField everywhere
    bool outlierReject;
    uint32_t maxNumberOfTuples;
    WorkDiv1D workDivScan;
    typename caStructures::tindex_type* tkids;
    double* phits;
    float* phits_ge;
    double* pfast_fit;
    generalBrokenLine::GblNodeData* pgnodes;
    double* pgblScratch;
    double* pphase;
    // Per-lane fit-hit-id table (written by Kernel_BLFastFitRefit) and the per-merged-track dropped-hit-id
    // output (read by the merger's post-refit hit-list compaction). Both null leaves the fit-rejected hit
    // in the emitted list and writes neither.
    uint32_t* pFitHitId;
    uint32_t* pDropHitId;
    // Core-protected outlier: per-lane pixel-core flag table (written by Kernel_BLFastFitRefit) and its
    // enable. With pFitHitIsCore null or coreProtect false the outlier scan considers every node.
    uint8_t* pFitHitIsCore;
    bool coreProtect;
    // Fit-consistent curvature->pT conversion field (see Kernel_BLFitPhaseSolve::fieldKernelWeights_).
    // Needs the field map, so it is inert wherever bMap is null.
    bool fieldKernelWeights = false;
    // Charge-symmetric corrections (see Kernel_BLFitPhasePrep::chargeSymmetric_). Its arc-sign half is
    // map-independent; its field-profile half needs bMap and is inert wherever the map is null.
    bool chargeSymmetric = false;
    // Reference-trajectory corrections (see Kernel_BLFitPhasePrep::trajectoryCorrections_).
    bool trajectoryCorrections = false;
    // Highland's log at the track's total declared material rather than gap by gap (see
    // Kernel_BLFitPhasePrep::scatteringLogAtTotal_).
    bool scatteringLogAtTotal = false;
    // Cumulative-column typical-loss law rather than the per-lump Landau MPV (see
    // Kernel_BLFitPhasePrep::elossCumulative_).
    bool cumulativeEloss = false;
    // Dynamic-partition state of the fused ladder.
    // Per-bin demand of the current round (kRefitNBins uint32), filled by Kernel_BLRefitBinCount.
    uint32_t* pCounts = nullptr;
    // Per-bin {firstLane, nLanes} of the current round (kRefitRangeStride * kRefitNBins uint32).
    uint32_t* pRange = nullptr;
    // Per-block {bin, firstLane, endLane} dispatch table (kRefitBlockMapStride * kFusedBlocks uint32).
    uint32_t* pBlockMap = nullptr;
    // Per-track "already seated in an earlier round" flag (nTracksCap bytes).
    uint8_t* pServed = nullptr;
    // Work division of the fused phase launches: kFusedBlocks x kFitBlock, fixed and host-known.
    WorkDiv1D workDivFused;
  };

  // Fused-ladder launchers: one (phase, iteration) per call, all ten N-bins in one launch. Declared
  // `extern template` and instantiated one phase per TU.
  // Phase-buffer caches: the material rows and the bFieldEff slot are carried across the two
  // linearizations. The material rows travel between two launches of the same kernel object, so the cached
  // doubles cannot differ from a recompute; bFieldEff travels from prep to out/outlier, whose different
  // inlining contexts of blEffectiveBField may contract FMAs differently and move a result by 1 ULP.

  // One bin's fast-fit compaction scan for the fused ladder. It keeps its own launch for the grid-scope
  // atomic slot claim that must complete before any phase reads ptkids; it claims against the bin's own
  // counter and seats tracks in the bin's own lane range.
  template <int N, typename TrackerTraits>
  void runRefitScanBin(BLRefitLaunchCtx<TrackerTraits> const& c, uint32_t nHitsL, uint32_t nHitsH) {
    constexpr uint32_t kRefitStride = HelixFit<TrackerTraits>::kRefitStride;
    constexpr uint32_t kBin = uint32_t(N) - uint32_t(kRefitMinN);
    constexpr bool hasStubs = true;  // Phase2OTStubs
    Kernel_BLFastFitRefit<N, kRefitStride> scan{c.otSrc, c.pFitHitId, c.pFitHitIsCore, c.pRange, c.pServed};
    alpaka::exec<Acc1D>(c.queue,
                        c.workDivScan,
                        scan,
                        c.hitContainer,
                        c.hv,
                        c.cm,
                        c.acceptedByTuple,
                        c.tkids,
                        c.phits,
                        c.phits_ge,
                        c.pfast_fit,
                        c.pCounts + kRefitNBins + kBin,  // the claim counters live past the demands
                        nHitsL,
                        nHitsH,
                        c.maxNumberOfTuples,
                        hasStubs);
  }

  template <typename TrackerTraits>
  void runRefitFusedPrep(BLRefitLaunchCtx<TrackerTraits> const& c, bool finalIter, bool fieldKernelOn) {
    constexpr uint32_t kRefitStride = HelixFit<TrackerTraits>::kRefitStride;
    BLRefitFusedCfg cfg{};
    cfg.rhoMap = c.rhoMap;
    cfg.bMap = c.bMap;
    // iterFromPhase / matFromPhase / fieldFromPhase are all finalIter: the bin's one scan fixed the
    // lane->track map for both iterations and the hits never move, so iteration 1 finds its own
    // iteration-0 reference, material rows and field in its own lane of c.pphase. fieldFromPhase also
    // needs the fit-consistent conversion field to be on (fieldKernelOn).
    cfg.iterFromPhase = finalIter;
    cfg.matFromPhase = finalIter;
    cfg.fieldFromPhase = finalIter && fieldKernelOn;
    cfg.chargeSymmetric = c.chargeSymmetric;
    cfg.trajectoryCorrections = c.trajectoryCorrections;
    cfg.scatteringLogAtTotal = c.scatteringLogAtTotal;
    cfg.elossCumulative = c.cumulativeEloss;
    alpaka::exec<Acc1D>(c.queue,
                        c.workDivFused,
                        Kernel_BLFitPhasePrepFused<TrackerTraits, kRefitStride>{cfg},
                        c.bField,
                        c.tkids,
                        c.phits,
                        c.phits_ge,
                        c.pfast_fit,
                        c.pgnodes,
                        c.pphase,
                        c.pBlockMap);
  }

  template <typename TrackerTraits>
  void runRefitFusedSolve(BLRefitLaunchCtx<TrackerTraits> const& c, bool finalIter) {
    constexpr uint32_t kRefitStride = HelixFit<TrackerTraits>::kRefitStride;
    BLRefitFusedCfg cfg{};
    cfg.bMap = c.bMap;
    cfg.outlierReject = c.outlierReject;
    cfg.finalIter = finalIter;
    cfg.fieldKernelWeights = c.fieldKernelWeights;
    alpaka::exec<Acc1D>(c.queue,
                        c.workDivFused,
                        Kernel_BLFitPhaseSolveFused<TrackerTraits, kRefitStride>{cfg},
                        c.bField,
                        c.tkids,
                        c.phits,
                        c.pfast_fit,
                        c.pgnodes,
                        c.pgblScratch,
                        c.pphase,
                        c.pBlockMap);
  }

  template <typename TrackerTraits>
  void runRefitFusedOut(BLRefitLaunchCtx<TrackerTraits> const& c) {
    constexpr uint32_t kRefitStride = HelixFit<TrackerTraits>::kRefitStride;
    BLRefitFusedCfg cfg{};
    cfg.bMap = c.bMap;
    // This iteration's prep ran on this lane and published the effective field.
    cfg.bFieldFromPhase = true;
    cfg.chargeSymmetric = c.chargeSymmetric;
    alpaka::exec<Acc1D>(c.queue,
                        c.workDivFused,
                        Kernel_BLFitPhaseOutFused<TrackerTraits, kRefitStride>{cfg},
                        c.bField,
                        c.outputSoa,
                        c.tkids,
                        c.phits,
                        c.pfast_fit,
                        c.pphase,
                        c.pBlockMap);
  }

  template <typename TrackerTraits>
  void runRefitFusedOutlier(BLRefitLaunchCtx<TrackerTraits> const& c) {
    constexpr uint32_t kRefitStride = HelixFit<TrackerTraits>::kRefitStride;
    BLRefitFusedCfg cfg{};
    cfg.bMap = c.bMap;
    cfg.fitHitId = c.pFitHitId;
    cfg.dropHitId = c.pDropHitId;
    cfg.fitHitIsCore = c.pFitHitIsCore;
    cfg.outlierReject = c.outlierReject;
    cfg.coreProtect = c.coreProtect;
    cfg.bFieldFromPhase = true;  // the final prep ran on this lane
    cfg.chargeSymmetric = c.chargeSymmetric;
    alpaka::exec<Acc1D>(c.queue,
                        c.workDivFused,
                        Kernel_BLFitPhaseOutlierFused<TrackerTraits, kRefitStride>{cfg},
                        c.bField,
                        c.outputSoa,
                        c.tkids,
                        c.phits,
                        c.pfast_fit,
                        c.pgnodes,
                        c.pgblScratch,
                        c.pphase,
                        c.pBlockMap);
  }

  // Explicit-instantiation signatures (`extern` in the orchestration TU, bare in each
  // BrokenLineFit_*.dev.cc to instantiate its disjoint subset).
// One signature per phase and traits set, each pulling every compile-time N of that phase into the TU
// that instantiates it.
#define BLFIT_MAIN_FUSED_FAST_SIG(T) \
  template void runMainFusedFast<::pixelTopology::T>(BLMainLaunchCtx<::pixelTopology::T> const&)
#define BLFIT_MAIN_FUSED_FIT_SIG(T) \
  template void runMainFusedFit<::pixelTopology::T>(BLMainLaunchCtx<::pixelTopology::T> const&)
#define BLFIT_REFIT_SCAN_SIG(N)                                     \
  template void runRefitScanBin<N, ::pixelTopology::Phase2OTStubs>( \
      BLRefitLaunchCtx<::pixelTopology::Phase2OTStubs> const&, uint32_t, uint32_t)
// One signature per phase of the refit ladder, one TU per phase.
#define BLFIT_REFIT_FUSED_PREP_SIG()                               \
  template void runRefitFusedPrep<::pixelTopology::Phase2OTStubs>( \
      BLRefitLaunchCtx<::pixelTopology::Phase2OTStubs> const&, bool, bool)
#define BLFIT_REFIT_FUSED_SOLVE_SIG()                               \
  template void runRefitFusedSolve<::pixelTopology::Phase2OTStubs>( \
      BLRefitLaunchCtx<::pixelTopology::Phase2OTStubs> const&, bool)
#define BLFIT_REFIT_FUSED_OUT_SIG()                               \
  template void runRefitFusedOut<::pixelTopology::Phase2OTStubs>( \
      BLRefitLaunchCtx<::pixelTopology::Phase2OTStubs> const&)
#define BLFIT_REFIT_FUSED_OUTLIER_SIG()                               \
  template void runRefitFusedOutlier<::pixelTopology::Phase2OTStubs>( \
      BLRefitLaunchCtx<::pixelTopology::Phase2OTStubs> const&)

  // Main fit: one instantiation per (phase, traits). The stubs traits (N = 3..10) get a TU each; the four
  // non-stubs traits (N = 3..6) share one.
  extern BLFIT_MAIN_FUSED_FAST_SIG(Phase1);
  extern BLFIT_MAIN_FUSED_FAST_SIG(Phase2);
  extern BLFIT_MAIN_FUSED_FAST_SIG(Phase2OT);
  extern BLFIT_MAIN_FUSED_FAST_SIG(HIonPhase1);
  extern BLFIT_MAIN_FUSED_FAST_SIG(Phase2OTStubs);
  extern BLFIT_MAIN_FUSED_FIT_SIG(Phase1);
  extern BLFIT_MAIN_FUSED_FIT_SIG(Phase2);
  extern BLFIT_MAIN_FUSED_FIT_SIG(Phase2OT);
  extern BLFIT_MAIN_FUSED_FIT_SIG(HIonPhase1);
  extern BLFIT_MAIN_FUSED_FIT_SIG(Phase2OTStubs);

  // Extended-N refit: N = 3 .. kRefitMaxN (12), Phase2OTStubs only. The per-bin fast-fit scans keep their
  // own launch and are instantiated in the N-range TUs BrokenLineFit_refitLo/Hi.
  extern BLFIT_REFIT_SCAN_SIG(3);
  extern BLFIT_REFIT_SCAN_SIG(4);
  extern BLFIT_REFIT_SCAN_SIG(5);
  extern BLFIT_REFIT_SCAN_SIG(6);
  extern BLFIT_REFIT_SCAN_SIG(7);
  extern BLFIT_REFIT_SCAN_SIG(8);
  extern BLFIT_REFIT_SCAN_SIG(9);
  extern BLFIT_REFIT_SCAN_SIG(10);
  extern BLFIT_REFIT_SCAN_SIG(11);
  extern BLFIT_REFIT_SCAN_SIG(12);
  // One fused phase per TU: BrokenLineFit_refitFusedPrep/Solve/Out/Outlier.dev.cc.
  extern BLFIT_REFIT_FUSED_PREP_SIG();
  extern BLFIT_REFIT_FUSED_SOLVE_SIG();
  extern BLFIT_REFIT_FUSED_OUT_SIG();
  extern BLFIT_REFIT_FUSED_OUTLIER_SIG();

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_BrokenLineFitKernels_h
