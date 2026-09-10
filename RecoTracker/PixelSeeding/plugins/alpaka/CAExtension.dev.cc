// OT hit attach, the CKF seed-region-rebuild equivalent. The walk runs once per event, in the
// merger: launchMergerAttach is called from PixelTracksSoAMerger::produce over the concatenated,
// HP-selected and twin-merged collection, before the merger's final GBL refit; the CA iterations
// themselves never attach. A pre-gate and compaction stage selects which merged tracks enter the
// search, each accepted extra records its gate chi2 so that shared attached hits are arbitrated
// across tracks by an atomic best-chi2 claim per hit, and the layer geometry comes from the
// producer's run cache.

#include <cassert>
#include <cstdint>
#include <cstdio>  // host-side printf of the compile-gated diagnostic and sizing dumps
#include <cstring>
#include <cmath>
#include <atomic>
#include <optional>  // candidate dump: transient device buffers held only in an instrumented build
#include <sstream>   // candidate dump: chunked host-side LogInfo record formatting
#include <vector>

#include <alpaka/alpaka.hpp>

#include "FWCore/MessageLogger/interface/MessageLogger.h"  // candidate dump: host readback LogInfo

#include "DataFormats/Math/interface/approx_atan2.h"
#include "DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BrokenLine.h"

#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"

#include "CAExtensionKernels.h"
#include "CAFitHitSelection.h"
#include "ExtenderHelixHelpers.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::caExtension {

  // Defining EXT_CAND_DUMP compiles in the per-(candidate, visited layer) trace of
  // Kernel_extFindExtras: in-window candidate counts per source round, every scored candidate's id
  // and gate chi2, the committed winner and the layer outcome, read back on the host as fixed-format
  // LogInfo lines whose hit ids join to an all-hit truth table. Diagnostic only: it writes nothing
  // any track reads, and undefined every dump branch folds away with no buffer allocated.
#ifdef EXT_CAND_DUMP
  inline constexpr bool kExtCandDump = true;
#else
  inline constexpr bool kExtCandDump = false;
#endif

  using Quality = ::pixelTrack::Quality;

  // Kernel_extFindExtras runs one block per candidate with this many cooperating lanes (threads on
  // GPU, elements-per-thread on the serial backend). It sets both the launch block size and the
  // in-kernel per-lane shared arrays / hit-partition modulus, so the two must agree -- keep this the
  // single source of truth.
  constexpr uint32_t kExtFindLanes = 128u;
  // The lane partition sets only which lane scores a hit; the (chi2, hitId) argmin is
  // partition-independent, so the count is free to change within two limits, asserted below: it must
  // be a power of two, since every reduce halves the stride and would otherwise drop lanes, and at
  // most 256, since the per-lane shared arrays are sized by it and the serial backend's block shared
  // arena is 47 KiB (roughly a quarter of it in use at 128 lanes).
  static_assert(kExtFindLanes > 0u && (kExtFindLanes & (kExtFindLanes - 1u)) == 0u,
                "kExtFindLanes must be a power of two: the shared-memory reduces halve the stride");
  static_assert(kExtFindLanes <= 256u, "kExtFindLanes > 256 overflows the serial backend's 47 KiB block shared arena");

  // Highland multiple-scattering variance over the (r0,z0)->(r1,z1) segment.
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float multScattVar(TAcc const& acc,
                                                    float bf,
                                                    HelixState const& h,
                                                    float arcS,
                                                    const float* rho,
                                                    float r0,
                                                    float z0,
                                                    float r1,
                                                    float z1) {
    constexpr float kMeVToGeV = 13.6e-3f;
    const float absRho = alpaka::math::abs(acc, h.rho);
    float pT = bf * absRho;
    pT = alpaka::math::min(acc, 20.f, pT);
    const float pT2 = pT * pT;
    const float slope2 = h.cotTheta * h.cotTheta;
    float ll;
    float factor;
    if (rho != nullptr) {
      ll = float(brokenline::segmentXX0(acc, rho, double(r0), double(z0), double(r1), double(z1)));
      factor = kMeVToGeV * kMeVToGeV;  // full Highland (geometry factor 1)
    } else {
      constexpr float inv_X0 = 0.06f / 16.f;  // constant-density fallback when no material map is given
      ll = alpaka::math::abs(acc, arcS) * inv_X0;
      factor = 0.7f * kMeVToGeV * kMeVToGeV;
    }
    if (ll < 1e-9f)
      return 0.f;
    const float logTerm = 1.f + 0.038f * alpaka::math::log(acc, ll);
    return factor / (pT2 * (1.f + slope2)) * ll * logTerm * logTerm;
  }

  // Derived road model, inert unless extDerivedSelection. The Highland coefficient per unit X/X0, in
  // the same theta0 form the fast BL builds under useFitCorrections (full momentum, pion 1/beta, no
  // momentum cap), so that road and fit charge scattering the same way; multScattVar above uses
  // beta == 1 and clamps pT at 20 GeV and serves the fixed-cut gate.
  //   c = (13.6 MeV / (p beta))^2 (1 + 0.038 ln W)^2      [rad^2 per unit X/X0]
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float extHighlandC(TAcc const& acc, float pTot, float radLen) {
    constexpr float kMeVToGeV = 13.6e-3f;
    constexpr float kMPi2 = 0.13957f * 0.13957f;
    if (!(radLen > 1e-9f) || !(pTot > 1e-6f))
      return 0.f;
    const float p2 = pTot * pTot;
    const float p2beta2 = p2 * p2 / (p2 + kMPi2);  // (p beta)^2 = p^4/(p^2 + m_pi^2)
    const float lg = 1.f + 0.038f * alpaka::math::log(acc, radLen);
    return kMeVToGeV * kMeVToGeV / p2beta2 * lg * lg;
  }

  // Module class of a target hit, for the measured target-side additive variance dV.
  // The endcap PS/2S split uses the standard geometric rule (TEDD r < 50 cm == PS).
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int extMatClass(int L, bool isBarrel, float rh) {
    if (L < 28)
      return isBarrel ? kExtMatClsPXB : kExtMatClsPXD;
    if (L <= 33)
      return (L <= 30) ? kExtMatClsTBPS : kExtMatClsTB2S;  // TOB1-3 = PS, TOB4-6 = 2S
    return (rh < 50.f) ? kExtMatClsTEDDPS : kExtMatClsTEDD2S;
  }

  // Maps a hit's CA module index to its extender layer via binary search over layerStarts.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int hitLayer(uint32_t moduleIdx, ::reco_extender::ExtenderLayersConstView layers) {
    const int nLayers = layers.metadata().size() - 1;
    int lo = 0;
    int hi = nLayers;
    while (lo < hi) {
      const int mid = (lo + hi) >> 1;
      if (layers.layerStarts()[mid + 1] <= moduleIdx)
        lo = mid + 1;
      else
        hi = mid;
    }
    return lo;
  }

  // Pack (gate chi2, tuple id) into one word so an atomicMin resolves the best claimant per hit:
  // for non-negative floats the IEEE bit pattern is order-preserving, so the smaller chi2 wins and
  // the tuple id in the low half breaks exact ties deterministically.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE uint64_t packClaim(float chi2, uint32_t tupleId) {
    const float nonNeg = chi2 < 0.f ? 0.f : chi2;
    uint32_t bits;
    static_assert(sizeof(bits) == sizeof(nonNeg));
    memcpy(&bits, &nonNeg, sizeof(bits));  // IEEE bit pattern is order-preserving for non-negative floats
    return (uint64_t(bits) << 32) | uint64_t(tupleId);
  }

  // Inverse of packClaim's chi2 half: recover the gate chi2 from a packed claim's high word (the claimant
  // tuple id lives in the low 32 bits). Only meaningful for a real claim (caller must exclude the
  // kUnclaimed sentinel 0xff..ff, whose high word is a NaN bit pattern). Used by the ambiguity gate
  // to compare the top-2 claimants' gate chi2 on a contested hit.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float unpackClaimChi2(uint64_t claim) {
    const uint32_t bits = uint32_t(claim >> 32);
    float chi2;
    static_assert(sizeof(chi2) == sizeof(bits));
    memcpy(&chi2, &bits, sizeof(chi2));
    return chi2;
  }

  // The walk's pre-gate predicate, as a per-tuple mask for the option-D prediction pass.
  // A strict superset of what Kernel_extPreGate accepts: it applies the same quality, finiteness,
  // chi2/ndof, pT and |eta| tests, but of the three host-quality predicates only extHostMaxChi2Ndof
  // (Kernel_extPreGate also applies extHostMinHits and extHostMinPt). Benign -- the mask only selects
  // which tracks get an ExtPredCoeff payload, and a payload nothing reads costs only its slot.
  struct Kernel_extHostMask {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::TrackSoAConstView tracks,
                                  const float preGateMaxChi2,
                                  const float preGateMinPt,
                                  const float maxAbsEta,
                                  const float extHostMaxChi2Ndof,
                                  const uint32_t nTracksCap,
                                  int32_t* __restrict__ hostMask,
                                  // Optional smoothed-prediction payload. This sweep visits every slot of
                                  // [0, nTracksCap), so it also clears the payload's `valid` flag there; only
                                  // `valid` is consulted before the payload is written (every reader tests
                                  // pc.valid > 0.5f first), so clearing that one float is sufficient.
                                  ExtPredCoeff* __restrict__ pred = nullptr) const {
      const uint32_t nT = alpaka::math::min(acc, nTracksCap, uint32_t(alpaka::math::max(acc, 0, tracks.nTracks())));
      const float cotMax = alpaka::math::sinh(acc, maxAbsEta);
      for (auto i : cms::alpakatools::uniform_elements(acc, nTracksCap)) {
        hostMask[i] = -1;
        if (pred != nullptr)
          pred[i].valid = 0.f;
        if (i >= nT)
          continue;
        if (tracks[i].quality() == ::pixelTrack::Quality::edup)
          continue;
        bool finite = alpaka::math::isfinite(acc, tracks[i].chi2());
        for (int a = 0; a < 5; ++a)
          finite = finite && alpaka::math::isfinite(acc, tracks[i].state()(a));
        if (!finite)
          continue;
        if (!(tracks[i].chi2() <= preGateMaxChi2))
          continue;
        if (!(tracks[i].pt() >= preGateMinPt))
          continue;
        if (!(alpaka::math::abs(acc, tracks[i].state()(3)) <= cotMax))
          continue;
        if (extHostMaxChi2Ndof > 0.f && !(tracks[i].chi2() < extHostMaxChi2Ndof))
          continue;
        hostMask[i] = 0;
      }
    }
  };

  // Pre-gate and compaction: select the tuples that enter the attach search. Never-fitted tuples are
  // identified by their zeroed passBuf row (the fitted-circle radius cannot be exactly zero). The
  // surviving tuple ids are compacted into candList so the extras buffers scale with the candidate
  // cap rather than the tuple capacity. The same predicate and atomic compaction run twice: a count
  // pass (candList == nullptr) computes nCands and the candidate stats on the device, then a fill
  // pass writes candList without re-counting the stats. The compaction is atomic in both passes, so
  // the slot order is run-to-run arbitrary; the per-hit arbitration downstream is order-independent.
  struct Kernel_extPreGate {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::TrackSoAConstView tracks,
                                  const double* __restrict__ passBuf,
                                  uint32_t maxNumberOfTuples,
                                  float preGateMaxChi2,
                                  float preGateMinPt,
                                  float maxAbsCotTheta,
                                  float extHostMaxChi2Ndof,
                                  int extHostMinHits,
                                  float extHostMinPt,
                                  // nullptr => no candidate restriction; else a tuple enters
                                  // only if acceptedMask[tuple] >= 0 (a previous pass's extended set).
                                  const int32_t* __restrict__ acceptedMask,
                                  uint32_t cap,
                                  uint32_t* __restrict__ candList,  // nullptr => count-only pass
                                  uint32_t* __restrict__ nCands,
                                  uint32_t* __restrict__ stats) const {
      const bool countOnly = (candList == nullptr);
      const uint32_t nT = alpaka::math::min(acc, maxNumberOfTuples, uint32_t(std::max(0, tracks.nTracks())));
      for (auto i : cms::alpakatools::uniform_elements(acc, nT)) {
        if (tracks[i].quality() == Quality::edup)
          continue;
        // never fitted (see header comment). passBuf == nullptr means the caller guarantees every row
        // is fitted, so there is no never-fitted row to skip.
        if (passBuf != nullptr && passBuf[kPassBufStride * i + 2] == 0.)
          continue;
        bool finite = alpaka::math::isfinite(acc, tracks[i].chi2());
        for (int a = 0; a < 5; ++a)
          finite = finite && alpaka::math::isfinite(acc, tracks[i].state()(a));
        if (!finite)
          continue;
        if (tracks[i].chi2() > preGateMaxChi2)
          continue;
        if (tracks[i].pt() < preGateMinPt)
          continue;
        // The |eta| pre-gate (|cotTheta| > sinh(maxAbsEta)). Counted, because nothing else in stats[]
        // sees the population it removes.
        if (alpaka::math::abs(acc, tracks[i].state()(3)) > maxAbsCotTheta) {
          if (countOnly)
            alpaka::atomicAdd(acc, &stats[kStatPreGateEtaSkipped], 1u, alpaka::hierarchy::Grids{});
          continue;
        }
        // host-quality pre-gate: skip hosts the HP selector will reject anyway -- they are both the
        // dominant stray-hit source and most of the attach work. Each predicate is independently
        // sentinel-gated; a host failing any active predicate is skipped and counted into
        // kStatPreGateSkipped (count pass only, to match the kStatCandidates single-count convention).
        // tracks[i].chi2() is already reduced chi2/ndof (BLFit stores gchi2/ndof) so it is compared to a
        // chi2/ndof threshold directly, no re-norm.
        {
          bool preGateSkip = false;
          if (extHostMaxChi2Ndof > 0.f && !(tracks[i].chi2() < extHostMaxChi2Ndof))
            preGateSkip = true;
          if (extHostMinHits > 0 && ::reco::nHits(tracks, i) < extHostMinHits)
            preGateSkip = true;
          if (extHostMinPt > 0.f && !(tracks[i].pt() >= extHostMinPt))
            preGateSkip = true;
          if (preGateSkip) {
            if (countOnly)
              alpaka::atomicAdd(acc, &stats[kStatPreGateSkipped], 1u, alpaka::hierarchy::Grids{});
            continue;
          }
        }
        // Candidate restriction: only the caller-supplied set. A follow-on attach pass passes the
        // previous pass's acceptedByTuple (>=0 iff the tuple got an accepted extension). Null => the
        // predicate is never evaluated and every pre-gate survivor enters.
        if (acceptedMask != nullptr && acceptedMask[i] < 0)
          continue;
        const uint32_t pos = alpaka::atomicAdd(acc, nCands, 1u, alpaka::hierarchy::Grids{});
        if (pos >= cap) {
          if (countOnly)
            alpaka::atomicAdd(acc, &stats[kStatCandOverflow], 1u, alpaka::hierarchy::Grids{});
          continue;
        }
        if (countOnly)
          alpaka::atomicAdd(acc, &stats[kStatCandidates], 1u, alpaka::hierarchy::Grids{});
        else
          candList[pos] = i;
      }
    }
  };

  // Per-candidate layer walk and windowed hit search. Candidates come from candList (slot j -> tuple
  // id), hit lists are read through hitOffsets/TrackHitSoA, and each accepted extra records its gate
  // chi2 for the arbitration pass. One block per candidate, kExtFindLanes lanes cooperating; the two
  // hot loops are split across the lanes with shared-memory argmin reductions:
  //   - candidate ordering: each lane computes, for every uncovered layer, a cheap geometric
  //     nearest-approach proxy distance to the layer's (r,z) envelope; the walk picks at each step
  //     the min-(proxy dist, then L) unvisited confirmed-reachable layer. Unreachable layers do not
  //     count toward the K = maxWalkLayers budget. The proxy approximates the crossing distance, so
  //     in corner cases the visit order can differ from a full-predict ordering, the physics not;
  //   - per-layer hit scan: the in-window hits are round-robin partitioned across lanes, each lane
  //     keeps its local best (min chi2, ties by min hitId) and a lane-0 reduction takes the global
  //     best. Both reductions are argmin only, over a strict total order, so the accepted hit does
  //     not depend on the lane partition and no floating-point summation is introduced.
  // Lane 0 owns the sequential state, the other lanes only run the two parallel maps. On the serial
  // backend the lanes are elements of a single thread, so serial validation exercises the same logic.
  struct Kernel_extFindExtras {
    const float* rhoMap_ = nullptr;
    const ExtPhiBinner* phiBinner_ = nullptr;
    // Raw OT-rechit source, held by value (the kernel object is copied to the device; a host
    // pointer could not be dereferenced there, but the source's views/device-pointers copy fine).
    // otSource_.nOTHits == 0 => merged-hits-only walk. When populated, the per-layer hit scan runs a
    // second bin-loop over the OT phi binner after the merged one.
    OTHitsSource otSource_{};
    // Endcap gate-variance decomposition (runtime-gated on params.verbose): accumulate the permil
    // split of sigSec2 (hit/MS/predSec/align) per considered disk hit into the stats buffer. false
    // skips the global atomics, so the hot loop pays nothing.
    bool secFracDiag_ = false;
    // Candidate-level dump (record structs in the header). With candDump_ false the buffers below are
    // null and no accumulator or record write executes. When true, lane 0 writes one ExtCandLayerRec
    // per (candidate, visited layer) into candLayerBuf_, and candDumpOvf_[0] count-and-clamps a
    // visit-index overrun. candHdrBuf_ is allocated and sentinel-filled by the launcher but the walk
    // writes no header record, so the host readback emits no H lines.
    bool candDump_ = false;
    ExtCandLayerRec* candLayerBuf_ = nullptr;  // [maxCandidates * maxWalkLayers]
    ExtCandHdrRec* candHdrBuf_ = nullptr;      // [maxCandidates]
    uint32_t* candDumpOvf_ = nullptr;          // [2] count-and-clamp overflow guard: [0]=visit-index, [1]=member
    // Per-road-candidate member dump. nullptr => off (the candDump_ path is off too). When set, each
    // scored road candidate (both rounds) writes one ExtCandMemberRec into
    // candMemberBuf_[(j*maxWalkLayers + vi)*kExtDumpMaxMembers + m], m from the block shared counter
    // shDumpMemN (reset at layer select); m >= kExtDumpMaxMembers bumps candDumpOvf_[1].
    ExtCandMemberRec* candMemberBuf_ = nullptr;  // [maxCandidates * maxWalkLayers * kExtDumpMaxMembers]

    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        const int maxExtraHitsPerTrack,
        const int maxWalkLayers,
        const float bf,
        const float chi2Cut,
        const float endcapChi2Cut,
        const float typePriorityBiasCm,
        const int pixHitsTarget,
        const float maxRPhiResidCm,
        const float maxSecResidCm,
        const float endcapMaxSecResidCm,
        const float alignSigmaPhiCm,
        const float alignSigmaSecCm,
        const float extChi2CutScaleTOB456,    // TOB4-6 barrel gate widen
        const float extChi2CutScaleTID,       // TID endcap gate scale
        const bool extRawOTVetoTOB456,        // skip raw-OT round on TOB4-6 (31-33)
        const bool extRawOTVetoTID,           // skip raw-OT round on TID (34-53)
        const bool extDisplacementAwareGate,  // dispgate: tighten TOB1-3 accept on displaced hosts
        const float extDispGateSig2,          // dispgate: (|d0|/sigma_d0)^2 displaced-host threshold
        const bool extForwardPocketGate,      // pocket gate: forward-eta TOB1-3 accept tightening
        const bool extPocketGateArmScoped,    // pocket gate: displaced-arm-only when true (needs armId)
        const float maxAbsCotTheta,           // sinh(maxAbsEta): the pre-gate ceiling, = the pocket band top
        const bool extMtvAlignedExtraCap,     // MTV-aligned per-track extra-cluster cap (accept-time)
        const float extRecallReachRelax,      // pixel-layer reachability envelope slack (cm)
        const int extRecallPixelFirstBudget,  // reserve up to N K-seats for pixel-first
        const float extCovScalePixel,         // pixel propagated-cov scale (merged L<28)
        const float extCovScaleStub,          // stub propagated-cov scale (merged L>=28)
        const float extCovScaleRawOT,         // raw-OT propagated-cov scale (round 1 + partner)
        const float extPixelGateChi2Cut,      // pixel honest-calibration chi2 cut (<=0 = off)
        const float extStubBendGate,          // 2S stub-bend nSigma veto (<=0 = off)
        const bool extCapExemptAnchored,      // anchored (prior OT accept) tracks bypass the cap
        const int extCapBudgetFloor,          // raise the per-track cluster budget to >= floor
        const float extStateProcessNoise,     // per-gap Highland MS injected into P (direction)
        const bool extRecallForcePixelVisit,  // prefer-pixel hosts visit pixel regardless of envelope
        const bool extCapExemptTOB46Only,     // scope the anchored exemption to TOB4-6 (CA 31-33)
        const float extCapExemptMaxChi2,      // exempt only if candidate gate chi2 < this (<=0 off)
        const int extMaxWalkLayers,           // runtime visit budget (loop bound; sizing stays maxWalkLayers)
        const bool extAttachFarFirst,         // far-first disc ordering on OT-less forward pixel hosts
        const float extAttachFarMinAbsEta,    // its |eta| floor; below it the nearest-first order stands
        const int extAttachFarMaxWin,         // its window-ambiguity condition: decline a far commit whose
                                              // gate-passing candidate set exceeds this (<=0 => no condition)
        // the derived-selection package (one switch; all inert when off)
        const bool extDerivedSelection,            // master switch (OT layers only)
        const float extDerivedEps,                 // the one free number: window == gate == rank == hole prior
        const bool extDerivedHole,                 // the hole "attach nothing" hypothesis in the argmin
        const bool extHoleDetectionPrior,          // the hole window-mass repair: numerator eta_L, not eta_L*eps
        const ExtPredCoeff* __restrict__ extPred,  // per-slot option-D payload (null = package off)
        const float* __restrict__ extQthr,         // [kExtQCells] Q-hat(eps) per (class x |eta| x visit) cell
        const float* __restrict__ extEtaL,         // [kExtOTLayers] measured per-layer stub availability
        const float* __restrict__ extRho,          // [kExtOTLayers] measured per-layer stub areal density [cm^-2]
        const bool extFwdEtaBin,                   // arm the 5th Q-hat |eta| bin
        const float* __restrict__ extEtaLRaw,      // [kExtOTLayers] raw round conditional availability (null=off)
        const float* __restrict__ extRhoRaw,       // [kExtOTLayers] raw-cluster areal density [cm^-2] (null=off)
        const float* __restrict__ extDV,           // [kExtMatClasses] measured target-side additive variance [cm^2]
        const float extFmsBarrel,                  // measured material-dispersion scale (barrel)
        const float extFmsEndcap,                  // measured material-dispersion scale (endcap)
        // the stub-bend package (one switch; all inert when off)
        const bool extBendPackage,                // the bend as the third chi2 row + the hole's basis
        const float* __restrict__ extQhat3,       // [kExtQCells] Q-hat_3(eps): the 3-dof map, stubs only
        const float* __restrict__ extSigBExcess,  // [kExtSigBClasses] measured honest/floor bend-error excess
        const float* __restrict__ extRho3,        // [kExtOTLayers] measured 3-dof stub density [cm^-1 rad^-1]
        const ::reco::TrackSoAConstView tracks,
        const ::reco::TrackHitSoAConstView trackHits,
        const uint8_t* __restrict__ armId,  // pocket gate: per-track arm (0=prompt,1=disp); null=off/arm-blind
        const ::reco::TrackingRecHitConstView hits,
        const ::reco::HitModuleSoAConstView hitModules,
        const ::reco::TrackingRecHitsMaskingConstView hitMask,
        const ::reco_extender::ExtenderLayersConstView caLayers,
        const ::reco::CAModulesConstView caModules,
        const uint32_t* __restrict__ candList,
        const uint32_t* __restrict__ nCands,
        const uint32_t maxCandidates,
        uint32_t* __restrict__ extrasIds,
        float* __restrict__ extrasChi2,
        int32_t* __restrict__ nExtras,
        uint32_t* __restrict__ stats) const {  // per-event counter buffer (kStat*/kDiag* indices, CAExtensionKernels.h)
      const uint32_t nC = alpaka::math::min(acc, *nCands, maxCandidates);
      const int nLayers = caLayers.metadata().size() - 1;
      // An empty hit mask view (metadata().size() == 0) means "all open": there is no masked hit to
      // skip, and the per-hit mask read is bypassed. A non-empty view is honoured per hit.
      const bool hitMaskArmed = (hitMask.metadata().size() != 0);
      // Walk-state shared arrays are compile-bounded; reads/writes are clamped to the caps so a runtime
      // K/slot budget larger than the compile-time one cannot overrun them.
      constexpr int kChainMaxVisits = 8;  // >= maxWalkLayers (the K visit budget)
      // The runtime walk budget (loop bound) is the requested extMaxWalkLayers clamped to
      // [1, kChainMaxVisits] -- the shared chain/hole arrays are sized at the compile-time
      // kChainMaxVisits, so vi = shWalkSteps-1 must stay < kChainMaxVisits. Buffer sizing/strides keep
      // the compile-time maxWalkLayers below; only the walk loop bounds read extWalkBudget. The
      // static_assert keeps headroom over the shipped default budget.
      static_assert(kChainMaxVisits >= 6, "kChainMaxVisits must cover the shipped maxWalkLayers=6 walk budget");
      const int extWalkBudget =
          (extMaxWalkLayers < 1) ? 1 : ((extMaxWalkLayers > kChainMaxVisits) ? kChainMaxVisits : extMaxWalkLayers);
      constexpr int kMaxOrigHits = 32;
      // Shared bound (reco::kMaxCALayers): the producers of the layers block throw above it, so the
      // walk's shared per-layer arrays and the uint64_t coveredMask can be sized from it here.
      constexpr int kMaxLayersList = ::reco::kMaxCALayers;
      constexpr float kReachSlackCm = 1.0f;
      // displacement-aware inner-OT gate constants (see AttachParams::extDisplacementAwareGate).
      // A host is "genuinely displaced" when |d0|/sigma_d0 >= sqrt(extDispGateSig2), i.e.
      // tip*tip >= extDispGateSig2 * V(tip). On such hosts the TOB1-3 (CA 28-30) per-hit accept window
      // is scaled by kDispGateTOB13Scale (tightened) so the innermost-barrel stray attaches that
      // dominate the displaced-tail impurity are rejected; prompt hosts and all outer layers keep
      // scale 1. The (|d0|/sigma_d0)^2 threshold itself is the runtime extDispGateSig2 argument.
      constexpr float kDispGateTOB13Scale = 0.4f;  // TOB1-3 accept-window scale on a displaced host
      // Forward-eta TOB1-3 pocket gate constants (see AttachParams::extForwardPocketGate). A host is
      // forward when its fitted |cotTheta| falls in [kPocketCotThetaLo, kPocketCotThetaHi), i.e.
      // |eta| from 1.5 up to the pre-gate ceiling, expressed as sinh of the eta edges so that the hot
      // setup path needs no asinh. On such hosts, when they are not already displacement-gated, the
      // TOB1-3 (CA 28-30) per-hit accept window is scaled by kPocketTOB13Scale; every other host and
      // outer layer keeps scale 1.
      constexpr float kPocketCotThetaLo = 2.129279f;  // sinh(1.5)
      // Upper edge of the pocket: the runtime |eta| pre-gate ceiling. Tying it to the ceiling is what
      // keeps the band non-empty when the ceiling moves -- a literal sinh(2.4) here would put every host
      // admitted above 2.4 outside the band and switch the TOB1-3 tightening off for exactly the hosts
      // it exists to tighten.
      const float kPocketCotThetaHi = maxAbsCotTheta;
      constexpr float kPocketTOB13Scale = 0.4f;  // TOB1-3 accept-window scale on a forward-eta host
      // MTV-aligned extra cap constants (see AttachParams::extMtvAlignedExtraCap). MTV
      // efficiency/fake need n_core/(n_core+n_extra) > 0.75 (QuickTrackAssociatorByHitsImpl). With
      // 0.75 == 1 - 1/kMtvSharedFracDen the worst case, every extra wrong, survives iff
      // n_extra < n_core/kMtvSharedFracDen, i.e. the per-track cap on appended extra clusters is
      // floor((n_core_clusters - 1)/kMtvSharedFracDen). kExtNoClusterCap is the no-cap sentinel.
      constexpr int kMtvSharedFracDen = 3;          // == 1/(1-0.75); the only constant of the cap
      constexpr int kExtNoClusterCap = 0x3fffffff;  // cap off => never binds
      // Per-(candidate,layer) linearization (see the coefficient block below): fall back to the exact
      // per-hit predict when the 2nd-order Taylor term 0.5*|d2phi/dr2|*W^2 (W = the layer half-extent)
      // exceeds this, or when the coefficient solve is near-tangential / has a vanishing denominator.
      constexpr float kTaylor2ndMaxRad = 1.0e-3f;
      constexpr float kLinCondEps = 1.0e-4f;
      const uint32_t nLanes = kExtFindLanes;  // lanes cooperating on one candidate (== launch block size)

      // shared per-block state (reused across grid-stride candidates)
      auto& sh = alpaka::declareSharedVar<RunningHelix, __COUNTER__>(acc);
      // The per-host bending field the running helix's geometry is expressed in (see the setup).
      auto& shBf = alpaka::declareSharedVar<float, __COUNTER__>(acc);
      auto& shCoveredMask = alpaka::declareSharedVar<uint64_t, __COUNTER__>(acc);
      auto& shPreferPixel = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      auto& shHostDisplaced = alpaka::declareSharedVar<int, __COUNTER__>(acc);      // dispgate: displaced host flag
      auto& shHostForwardPocket = alpaka::declareSharedVar<int, __COUNTER__>(acc);  // pocket: forward-eta host
      auto& shNOrigSafe = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      auto& shOrigR = alpaka::declareSharedVar<float[kMaxOrigHits], __COUNTER__>(acc);
      auto& shOrigZ = alpaka::declareSharedVar<float[kMaxOrigHits], __COUNTER__>(acc);
      auto& shReachable = alpaka::declareSharedVar<int[kMaxLayersList], __COUNTER__>(acc);
      auto& shReachDist = alpaka::declareSharedVar<float[kMaxLayersList], __COUNTER__>(acc);
      auto& shVisited = alpaka::declareSharedVar<int[kMaxLayersList], __COUNTER__>(acc);
      auto& shNExtra = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      // MTV-aligned extra cap (extMtvAlignedExtraCap): shExtraClusterCap = floor((n_core_cl-1)/den), computed
      // once in setup; shNExtraClusters accumulates the appended extras' cluster count, seeded with any tagged
      // extras already in the list so the cap bounds the total over successive attach passes. Cap off =>
      // shExtraClusterCap is the no-cap sentinel.
      auto& shExtraClusterCap = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      auto& shNExtraClusters = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      auto& shLastArcS = alpaka::declareSharedVar<float, __COUNTER__>(acc);
      auto& shLastR = alpaka::declareSharedVar<float, __COUNTER__>(acc);
      auto& shLastZ = alpaka::declareSharedVar<float, __COUNTER__>(acc);
      // the option-D band state and the per-visit derived road (all shared, no stack)
      // shDerCloc is the symmetric 3x3 local covariance of (u [cm], u' [rad], kappa [1/cm]) at the
      // current anchor -- seeded from the merger-side option-D payload at the last fitted node and
      // re-anchored at every accept by an exact 3x3 KF fold (propagate with F(ds), add the traversed
      // gap's process noise from its own material moments, update with the accepted r-phi measurement).
      // Packed [c00, c01, c02, c11, c12, c22].
      auto& shDerCloc = alpaka::declareSharedVar<float[6], __COUNTER__>(acc);
      auto& shDerOn = alpaka::declareSharedVar<int, __COUNTER__>(acc);         // package usable for this host
      auto& shDerAnchorS = alpaka::declareSharedVar<float, __COUNTER__>(acc);  // transverse arc of the anchor
      auto& shDerAnchorR = alpaka::declareSharedVar<float, __COUNTER__>(acc);
      auto& shDerAnchorZ = alpaka::declareSharedVar<float, __COUNTER__>(acc);
      auto& shDerQgap = alpaka::declareSharedVar<float, __COUNTER__>(acc);  // last fitted gap kink var; 0 after
      // Deterministic ionization energy-loss state, carried alongside the band: the band above is the
      // road's width, these four are its centre. They are the analogue of shDerCloc, seeded from the
      // same per-host payload at the same anchor (node n-1), propagated by the same F(ds) -- here a
      // deterministic second-order polynomial rather than a covariance -- and re-anchored at every
      // accept. The Kalman update does not reset them: the running helix estimates the
      // constant-kappa_0 reference trajectory, so the true track's offset from it keeps accumulating.
      // All four zero leaves every expression below adding an exact float 0.
      auto& shEU = alpaka::declareSharedVar<float, __COUNTER__>(acc);   // u at the anchor [cm]
      auto& shEUp = alpaka::declareSharedVar<float, __COUNTER__>(acc);  // du/ds at the anchor [rad]
      auto& shEDk = alpaka::declareSharedVar<float, __COUNTER__>(acc);  // dkappa at the anchor [1/cm]
      auto& shEK = alpaka::declareSharedVar<float, __COUNTER__>(acc);   // dkappa per column [1/(cm X0)]
      // Per-layer-visit derived road (written by lane 0 with the linearization, read by every lane).
      auto& shDerLayOn = alpaka::declareSharedVar<int, __COUNTER__>(acc);       // 1 = derived gate active on this layer
      auto& shDerSigR2 = alpaka::declareSharedVar<float, __COUNTER__>(acc);     // layer part of sigma_R^2 [cm^2]
      auto& shDerSigS2 = alpaka::declareSharedVar<float, __COUNTER__>(acc);     // layer MS part of sigma_S^2 [cm^2]
      auto& shDerQ = alpaka::declareSharedVar<float, __COUNTER__>(acc);         // Q-hat(eps) of this cell
      auto& shDerWinR = alpaka::declareSharedVar<float, __COUNTER__>(acc);      // r-phi window half-width [cm]
      auto& shDerCeilR = alpaka::declareSharedVar<float, __COUNTER__>(acc);     // r-phi runaway ceiling [cm], envelope
      auto& shDerCeilS = alpaka::declareSharedVar<float, __COUNTER__>(acc);     // secondary runaway ceiling [cm]
      auto& shDerHoleK = alpaka::declareSharedVar<float, __COUNTER__>(acc);     // 2 dof, stub round
      auto& shDerHoleKRaw = alpaka::declareSharedVar<float, __COUNTER__>(acc);  // 2 dof, raw-OT round
      auto& shDerHoleK3 = alpaka::declareSharedVar<float, __COUNTER__>(acc);    // 3 dof: rho_3, (2 pi)^{3/2}
      // The bend row's per-layer-visit constants. The track-side prediction and everything
      // in R_bb except the hit's own sigma_b are properties of the layer crossing, so they are formed
      // once here, exactly like the r-phi/secondary road above.
      auto& shDerBendOn = alpaka::declareSharedVar<int, __COUNTER__>(acc);    // 1 = the third row is live
      auto& shDerPredB = alpaka::declareSharedVar<float, __COUNTER__>(acc);   // dphi/dr of the track [1/cm]
      auto& shDerRbbTrk = alpaka::declareSharedVar<float, __COUNTER__>(acc);  // H_b C H_b^T + Q_MS,bb [1/cm^2]
      auto& shDerQ3 = alpaka::declareSharedVar<float, __COUNTER__>(acc);      // Q-hat_3(eps) of this cell
      auto& shDerC = alpaka::declareSharedVar<float, __COUNTER__>(acc);       // Highland c of this gap
      auto& shDerW = alpaka::declareSharedVar<float, __COUNTER__>(acc);       // gap moments to the crossing
      auto& shDerS1 = alpaka::declareSharedVar<float, __COUNTER__>(acc);
      auto& shDerS2 = alpaka::declareSharedVar<float, __COUNTER__>(acc);
      // The energy-loss road centre of this layer visit, already projected onto the two gate rows:
      // a phi shift [rad] and a secondary shift [cm], both added to the prediction (never to a
      // width). Zero whenever the payload carries no eloss constants or the material march did not
      // run, which is what keeps the corrections-off path unchanged.
      auto& shDerElossPhi = alpaka::declareSharedVar<float, __COUNTER__>(acc);
      auto& shDerElossSec = alpaka::declareSharedVar<float, __COUNTER__>(acc);
      auto& shCurrentL = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      auto& shWalkDone = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      auto& shWalkSteps = alpaka::declareSharedVar<int, __COUNTER__>(acc);     // reachable layers walked so far
      auto& shOTDiskVisits = alpaka::declareSharedVar<int, __COUNTER__>(acc);  // OT-disk layers walked so far
      auto& shPixVisits = alpaka::declareSharedVar<int, __COUNTER__>(acc);     // pixel layers walked so far
      auto& shConfirmDone = alpaka::declareSharedVar<int, __COUNTER__>(acc);   // 1 once all candidate layers confirmed
      auto& shMergedHit = alpaka::declareSharedVar<int, __COUNTER__>(acc);     // round 0 (merged) attached here?
      auto& shHasOT = alpaka::declareSharedVar<int, __COUNTER__>(acc);  // >=1 OT extra committed on this candidate
      // Count of accepted OT-layer (CA >= 28) extras so far this walk, seeded with any tagged OT extras
      // already in the hit list. shNOTExtraAcc >= 1 == the track is "anchored" (a prior OT accept exists),
      // which is the class-aware condition for the cluster-cap exemption. Written unconditionally; read
      // only under extCapExemptAnchored.
      auto& shNOTExtraAcc = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      // Exact full-5x5 shadow covariance (verbose-gated) carried alongside the block-diagonal
      // RunningHelix; updated at each accept with the same measurement rows the walk uses (a covariance
      // update is state/innovation-independent). Diagnostic only -- it never feeds any gate.
      auto& shC = alpaka::declareSharedVar<float[15], __COUNTER__>(acc);
      auto& shNDiskAcc = alpaka::declareSharedVar<int, __COUNTER__>(acc);  // prior disk (endcap) accepts on this cand
      auto& laneChi2 = alpaka::declareSharedVar<float[kExtFindLanes], __COUNTER__>(acc);
      auto& laneHit = alpaka::declareSharedVar<int32_t[kExtFindLanes], __COUNTER__>(acc);
      auto& laneDPhi = alpaka::declareSharedVar<float[kExtFindLanes], __COUNTER__>(acc);
      auto& laneDSec = alpaka::declareSharedVar<float[kExtFindLanes], __COUNTER__>(acc);
      auto& laneSigPhi2 = alpaka::declareSharedVar<float[kExtFindLanes], __COUNTER__>(acc);
      auto& laneSigSec2 = alpaka::declareSharedVar<float[kExtFindLanes], __COUNTER__>(acc);
      auto& laneRh = alpaka::declareSharedVar<float[kExtFindLanes], __COUNTER__>(acc);
      auto& laneZh = alpaka::declareSharedVar<float[kExtFindLanes], __COUNTER__>(acc);
      auto& laneArcS = alpaka::declareSharedVar<float[kExtFindLanes], __COUNTER__>(acc);
      // The full derived gate variances [cm^2] of each lane's best hit, so the hole hypothesis can price
      // the winner's own window volume |R| = sigma_R^2 sigma_S^2 after the reduce. (The endcap line-block
      // Jacobian row H = (dr/dcot, dr/dzip) of the same winner is staged in laneJrCot/laneJrZip below.)
      auto& laneDerSigR2 = alpaka::declareSharedVar<float[kExtFindLanes], __COUNTER__>(acc);
      auto& laneDerSigS2 = alpaka::declareSharedVar<float[kExtFindLanes], __COUNTER__>(acc);
      // The winner's R_bb, so the hole hypothesis can price the 3-dof window volume |R|.
      auto& laneDerRbb = alpaka::declareSharedVar<float[kExtFindLanes], __COUNTER__>(acc);
      auto& laneJrCot = alpaka::declareSharedVar<float[kExtFindLanes], __COUNTER__>(acc);
      auto& laneJrZip = alpaka::declareSharedVar<float[kExtFindLanes], __COUNTER__>(acc);
      // Region-balanced select staging -- per-lane best OT-disk layer + OT-disk tally.
      auto& laneChi2Disk = alpaka::declareSharedVar<float[kExtFindLanes], __COUNTER__>(acc);
      auto& laneHitDisk = alpaka::declareSharedVar<int32_t[kExtFindLanes], __COUNTER__>(acc);
      auto& laneNDisk = alpaka::declareSharedVar<int[kExtFindLanes], __COUNTER__>(acc);
      // Winning original-lane index, carried through the hit-scan argmin tree reduce so lane 0 can index
      // the winner's KF payload (laneDPhi / laneSigPhi2 / ...) after the reduction.
      auto& laneWin = alpaka::declareSharedVar<int32_t[kExtFindLanes], __COUNTER__>(acc);
      // Per-(candidate,layer) crossing-quantity linearization coefficients (lane 0 -> shared, one set
      // per scanned layer). shUseLin selects the Taylor path (1) or the exact per-hit predict (0).
      auto& shLinRef = alpaka::declareSharedVar<float, __COUNTER__>(acc);    // expansion point (R barrel / Z endcap)
      auto& shLinPhi0 = alpaka::declareSharedVar<float, __COUNTER__>(acc);   // phi at ref
      auto& shLinDphi1 = alpaka::declareSharedVar<float, __COUNTER__>(acc);  // dphi/du
      auto& shLinDphi2 = alpaka::declareSharedVar<float, __COUNTER__>(acc);  // d2phi/du2
      auto& shLinSec0 = alpaka::declareSharedVar<float, __COUNTER__>(acc);   // secondary (z barrel / r endcap) at ref
      auto& shLinDsec1 = alpaka::declareSharedVar<float, __COUNTER__>(acc);  // dsec/du
      auto& shLinDsec2 = alpaka::declareSharedVar<float, __COUNTER__>(acc);  // d2sec/du2
      auto& shLinArc0 = alpaka::declareSharedVar<float, __COUNTER__>(acc);   // arcS at ref
      auto& shLinDarc1 = alpaka::declareSharedVar<float, __COUNTER__>(acc);  // darcS/du
      auto& shLinArgLo = alpaka::declareSharedVar<float, __COUNTER__>(acc);  // barrel radial validity band lo
      auto& shLinArgHi = alpaka::declareSharedVar<float, __COUNTER__>(acc);  // barrel radial validity band hi
      auto& shUseLin = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      // 1 iff the per-layer reference predict p0 was valid, so shLinDphi1 (the track's local
      // dphi/dr) is a real derivative and the 2S stub-bend term has a track curvature expectation to compare
      // against. Written by lane 0 in the linearization block; read only under extStubBendGate > 0.
      auto& shLinValid = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      // Candidate-dump per-(candidate,layer) accumulators, always declared but written only under the
      // runtime candDump_ guard: reset at layer select, accumulated by the scan lanes via block
      // atomics, recorded by lane 0 at the record-write site. shDumpOccM/O = genuine (post-mask)
      // window candidates; shDumpPassM/O = gate passers; shDumpBestM = min gate chi2*1000 over
      // considered hits; shDumpWin*/Round = committed winner; shDumpCapDropped = winner dropped by the
      // MTV cluster cap; shDumpVeto = raw-OT round veto-skipped; shDumpPartner = stack-partner extra.
      auto& shDumpOccM = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
      auto& shDumpOccO = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
      auto& shDumpPassM = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
      auto& shDumpPassO = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
      auto& shDumpBestM = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
      auto& shDumpWinHit = alpaka::declareSharedVar<int32_t, __COUNTER__>(acc);
      auto& shDumpWinChi2 = alpaka::declareSharedVar<float, __COUNTER__>(acc);
      auto& shDumpWinRound = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      auto& shDumpCapDropped = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      auto& shDumpVeto = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      auto& shDumpPartner = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      // Per-(candidate,layer) member write cursor. Reset at layer select (under candDump_), atomicAdd'd
      // once per scored road candidate in both rounds to allocate its slot m in the member buffer.
      // Always declared (a shared allocation only); written and read only under candDump_.
      auto& shDumpMemN = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
      // Layer-outcome accumulator for the walk instrument's hole counter: 1 iff the visited layer
      // carried >=1 gate-passing candidate, or-ed across both source rounds' argmin-winner presence.
      auto& shChainHasPass = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      // occupancy-gated hole counter (always declared, written only under candDump_). A "hole" is a
      // visited-reachable layer whose outcome was rejGate (occupancy present -- shHoleOcc>0 -- but no
      // gate-passer -- shChainHasPass==0), not rejEmpty (empty road = dead-module / no-crossing carve-out)
      // and not rejCap (a passer existed). shHoleRun = cumulative holes so far; shHoleConsec = consecutive
      // holes since the last accept (reset on any gate-passer). shHoleOcc = per-layer genuine-window
      // occupancy count (reset at layer select).
      auto& shHoleRun = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      auto& shHoleConsec = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      auto& shHoleOcc = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
      // Far-first window-ambiguity state; each stays at its setup value unless extAttachFarFirst arms
      // the host.
      //   shFarArmed  = 1 iff this host's disc ordering was re-keyed far-first (A1 & A2 & A3 below).
      //   shFarZFloor = the largest |z| among the host's own core hits, read off the hits the reach
      //                 proxy already cached. A visited endcap pixel layer is a far crossing iff
      //                 |Z_L| exceeds it, i.e. iff the crossing lengthens the track.
      //   shFarLayer  = 1 iff the layer being scanned is such a far crossing and the condition is on.
      //   shFarPass   = the count of gate-passing merged candidates on it, the set the argmin sees.
      auto& shFarArmed = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      auto& shFarZFloor = alpaka::declareSharedVar<float, __COUNTER__>(acc);
      auto& shFarLayer = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      auto& shFarPass = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
      // Per-lane best (min-gate-chi2) road candidate over all considered hits of both rounds; it
      // supplies the id and the base-gate-pass flag that accompany bestFailChi2. Same lane-array
      // reduction as the walk argmin: each lane keeps its slot's running min-chi2 (tie by min id) and
      // lane 0 reduces across lanes at the record write. Written only under candDump_, so no atomic
      // is needed. Chi2 sentinel 3.4e38, id sentinel -1; laneBestAnyHit carries the bit30-tagged id.
      // Sized to one element outside an EXT_CAND_DUMP build, so the production kernel does not pay
      // 1.5 kB of block shared memory on a one-block-per-candidate kernel.
      constexpr uint32_t kBestAnySlots = kExtCandDump ? kExtFindLanes : 1u;
      auto& laneBestAnyChi2 = alpaka::declareSharedVar<float[kBestAnySlots], __COUNTER__>(acc);
      auto& laneBestAnyHit = alpaka::declareSharedVar<int32_t[kBestAnySlots], __COUNTER__>(acc);
      auto& laneBestAnyPass = alpaka::declareSharedVar<int[kBestAnySlots], __COUNTER__>(acc);

      // grid-stride over candidates: one group (block) per candidate, all lanes see the same groups
      for (uint32_t j : cms::alpakatools::uniform_groups(acc, nC * nLanes)) {
        const uint32_t i = candList[j];

        // setup (lane 0): running helix, covered-layer mask, cached orig (r,z), walk state
        for (auto element : cms::alpakatools::uniform_group_elements(acc, j, nC * nLanes)) {
          if (element.local != 0u)
            continue;
          // The field this walk's geometry is built in. makeRunningHelix inverts the fit's
          // curvature->pT conversion, rho_geom = 1/(q/pT * B), and every crossing, Taylor coefficient
          // and the d(phi)/d(q/pT) Jacobian below use that rho. The fit publishes q/pT as (geometric
          // curvature)/bFieldEff, with bFieldEff = blEffectiveBField over the fitted hits, so only
          // that same bFieldEff returns the fitted circle: the origin scalar would rescale the radius
          // by B_eff/B(0,0), applying to the road the forward field bias the fit removes. The
          // pre-attach pass publishes it per host; an invalid payload falls back to the scalar.
          shBf = bf;
          if (extPred != nullptr) {
            const ExtPredCoeff pcB = extPred[i];
            if (pcB.valid > 0.5f && pcB.bFieldEff > 0.f)
              shBf = pcB.bFieldEff;
          }
          sh = makeRunningHelix(acc, tracks, int(i), shBf);
          const auto hitBegin = (i == 0) ? 0u : tracks[i - 1].hitOffsets();
          const auto hitEnd = tracks[i].hitOffsets();
          const int nOrig = int(hitEnd - hitBegin);
          uint64_t coveredMask = 0;
          int nPixHitsOrig = 0;
          // MTV-aligned extra cap: count the host's genuine-core clusters (pixel + P-hit-only stub = 1, a
          // full 2-hit stub = 2 via reco::isStub -- matching the converter's cluster expansion) and,
          // separately, the tagged OT extras already in the list (each 1 cluster). Excluding those extras
          // from n_core and seeding shNExtraClusters with them makes the cap bound the total appended
          // clusters over successive attach passes. Cap off => neither accumulator is ever read.
          int nCoreClusters = 0;
          int nPriorExtraClusters = 0;
          for (auto idx = hitBegin; idx < hitEnd; ++idx) {
            const auto hitId = trackHits[idx].id();
            // A hit list that a previous attach pass extended already contains tagged OT extras (bit30)
            // which index the raw OT source, not the merged SoA -- dispatch like WriteFinal's detOf (both
            // sources share the CA module numbering). On a first pass no id here is tagged.
            const bool ot = isOTId(hitId);
            const uint32_t detIdx =
                ot ? uint32_t(otSource_.otHits[otIdx(hitId)].detectorIndex()) : hits[hitId].detectorIndex();
            const int layer = hitLayer(detIdx, caLayers);
            if (layer >= 0 && layer < 64) {
              coveredMask |= (uint64_t(1) << layer);
              if (layer < 28)
                ++nPixHitsOrig;
            }
            if (ot)
              ++nPriorExtraClusters;  // already-appended extra: 1 cluster, and not part of the core
            else
              nCoreClusters += ::reco::isStub(hits, int32_t(hitId)) ? 2 : 1;  // 2-hit stub -> 2 clusters
          }
          shCoveredMask = coveredMask;
          shPreferPixel = (nPixHitsOrig < pixHitsTarget) ? 1 : 0;
          // displacement-aware inner-OT gate: flag a "genuinely displaced" host from its fitted transverse
          // impact-parameter significance |d0|/sigma_d0 = |state(1)|/sqrt(V(tip)), V(tip) = covariance(5).
          // Computed once per candidate from the original (pre-walk) host params; broadcast to every lane by
          // the existing post-setup syncBlockThreads. Gate off or prompt host => 0 => TOB1-3 scale stays 1.
          {
            const float d0 = tracks[i].state()(1);
            const float vTip = tracks[i].covariance()(5);
            shHostDisplaced = (extDisplacementAwareGate && vTip > 0.f && d0 * d0 >= extDispGateSig2 * vTip) ? 1 : 0;
          }
          // forward-eta TOB1-3 pocket gate: flag a "forward" host from its fitted |cotTheta| = |state(3)| in
          // the band [kPocketCotThetaLo, kPocketCotThetaHi). Made exclusive with the displacement gate above
          // (&& !shHostDisplaced) so a host already tightened by dispgate does not also fold in the pocket
          // scale (no 0.4*0.4 stacking on hosts that are both). Arm-scoped additionally requires the merged-
          // track arm armId[i]==1 (displaced); arm-blind (or null armId) drops that term. Broadcast to every
          // lane by the existing post-setup syncBlockThreads. Gate off => 0 => TOB1-3 scale stays 1.
          {
            const float absCot = alpaka::math::abs(acc, tracks[i].state()(3));
            const bool inBand = (absCot >= kPocketCotThetaLo && absCot < kPocketCotThetaHi);
            const bool armOk = (!extPocketGateArmScoped) || (armId != nullptr && armId[i] == uint8_t(1));
            shHostForwardPocket = (extForwardPocketGate && inBand && !shHostDisplaced && armOk) ? 1 : 0;
          }
          const int nOrigSafe = nOrig < kMaxOrigHits ? nOrig : kMaxOrigHits;
          shNOrigSafe = nOrigSafe;
          // The original (r,z) hit array is cached cooperatively across all lanes (below), so the
          // lane-0 setup does not fill it here.
          shNExtra = 0;
          // MTV-aligned extra cap: form the per-track cluster budget (cap off => sentinel => never binds)
          // and seed the appended-cluster counter with the extras already in the list so it caps the total
          // over successive passes. extCapBudgetFloor raises the budget to at least that value, which is
          // what keeps a short OT-only core from getting a budget of 0. Applied only inside the cap-on
          // branch, so the no-cap sentinel is never lowered; floor 0 leaves the budget unchanged.
          int capBudget = (nCoreClusters - 1) / kMtvSharedFracDen;
          if (extCapBudgetFloor > 0 && capBudget < extCapBudgetFloor)
            capBudget = extCapBudgetFloor;
          shExtraClusterCap = extMtvAlignedExtraCap ? capBudget : kExtNoClusterCap;
          shNExtraClusters = extMtvAlignedExtraCap ? nPriorExtraClusters : 0;
          shWalkSteps = 0;
          shNDiskAcc = 0;  // prior endcap accepts, for the shadow-covariance decomposition
          if (secFracDiag_) {
            // Seed the exact shadow cov from the fitted 5x5 (BL guarantees the circle<->line cross
            // entries 3,4,7,8,10,11 are exactly zero, matching the RunningHelix block-diagonal start),
            for (int m = 0; m < 15; ++m)
              shC[m] = tracks[i].covariance()(m);
          }
          shOTDiskVisits = 0;
          shPixVisits = 0;  // reset the per-candidate pixel-layer visit tally
          // Far-first window-ambiguity state: unarmed until the eager-confirm round says otherwise.
          shFarArmed = 0;
          shFarZFloor = 0.f;
          shFarLayer = 0;
          shFarPass = 0u;
          // Hole counter: reset the per-candidate cumulative/consecutive occupancy-gated hole run.
          // Written and read only under candDump_, and never read into track output.
          if (candDump_) {
            shHoleRun = 0;
            shHoleConsec = 0;
          }
          shConfirmDone = 0;
          shHasOT = 0;
          // Seed the anchor counter with the tagged OT extras already in the list (each one an OT accept),
          // so a track anchored by an earlier attach pass stays anchored -- consistent with the cap
          // seeding shNExtraClusters from the same extras. On a first pass nPriorExtraClusters == 0.
          shNOTExtraAcc = nPriorExtraClusters;
          shLastArcS = 0.f;
          shLastR = alpaka::math::abs(acc, sh.helix().tip);
          shLastZ = sh.helix().zip;
          // Seed the band state for this host. The merger-side pre-attach pass published, per
          // candidate slot, the 3x3 local covariance of (u, u', kappa) at the host's last fitted node
          // plus that node's arc and (r,z) and the last fitted gap's exit-direction kink variance:
          // everything the derived road needs is anchored there, not at the PCA. The fixed-cut road
          // instead anchors every material integral at the perigee (shLastArcS = 0 above), which
          // mis-places the integral and, carrying a 3-D scattering angle on a transverse lever,
          // makes it too narrow by cosh(eta).
          shDerOn = 0;
          shDerAnchorS = 0.f;
          shDerAnchorR = shLastR;
          shDerAnchorZ = shLastZ;
          shDerQgap = 0.f;
          shEU = 0.f;
          shEUp = 0.f;
          shEDk = 0.f;
          shEK = 0.f;
          for (int q = 0; q < 6; ++q)
            shDerCloc[q] = 0.f;
          if (extDerivedSelection && extPred != nullptr && extQthr != nullptr && extEtaL != nullptr &&
              extRho != nullptr && extDV != nullptr) {
            const ExtPredCoeff pc = extPred[i];  // indexed by tuple id (the producer's own indexing)
            if (pc.valid > 0.5f && pc.c00 > 0.f && pc.c11 > 0.f) {
              shDerCloc[0] = pc.c00;
              shDerCloc[1] = pc.c01;
              shDerCloc[2] = pc.c02;
              shDerCloc[3] = pc.c11;
              shDerCloc[4] = pc.c12;
              shDerCloc[5] = pc.c22;
              shDerAnchorS = pc.anchorS;
              shDerAnchorR = pc.anchorR;
              shDerAnchorZ = pc.anchorZ;
              shDerQgap = pc.qgapCoef > 0.f ? pc.qgapCoef : 0.f;
              // The deterministic energy-loss centre, at the same anchor as the band. Guarded on a
              // strictly positive growth rate, which is what the fit's own gate produces: with
              // fitCorrections off, or a degenerate column/momentum, the producer leaves all four at
              // 0 and every expression below reduces to the uncorrected form.
              if (pc.elossK > 0.f) {
                shEU = pc.elossU;
                shEUp = pc.elossUp;
                shEDk = pc.elossDkAnchor;
                shEK = pc.elossK;
              }
              shDerOn = 1;
            }
            alpaka::atomicAdd(acc, &stats[shDerOn ? kStatDerHostOn : kStatDerHostOff], 1u, alpaka::hierarchy::Grids{});
          }
        }
        // Warp-coop cache of the original (r,z) hit array: each lane owns k = lane, lane+nLanes, ...
        // over the [0, nOrigSafe) set. nOrigSafe / hitBegin are recomputed per lane from the same
        // read-only track SoA (cheap, redundant, no cross-lane dependency), so no extra sync is needed
        // -- the existing syncBlockThreads below publishes the stores to every reader (the candidate-
        // layer proxy loop). Independent per-hit writes of identical values; serial backend (warpSize
        // == 1) fills every k in 0,1,2,... order.
        for (auto element : cms::alpakatools::uniform_group_elements(acc, j, nC * nLanes)) {
          const uint32_t lane = element.local;
          const auto hitBegin = (i == 0) ? 0u : tracks[i - 1].hitOffsets();
          const auto hitEnd = tracks[i].hitOffsets();
          const int nOrig = int(hitEnd - hitBegin);
          const int nOrigSafe = nOrig < kMaxOrigHits ? nOrig : kMaxOrigHits;
          for (int k = int(lane); k < nOrigSafe; k += int(nLanes)) {
            const auto hitId = trackHits[hitBegin + k].id();
            if (isOTId(hitId)) {  // tagged extra from an earlier pass -> fetch (r,z) from the OT source
              const uint32_t o = otIdx(hitId);
              const float xg = otSource_.otHits[o].xGlobal();
              const float yg = otSource_.otHits[o].yGlobal();
              shOrigR[k] = alpaka::math::sqrt(acc, xg * xg + yg * yg);
              shOrigZ[k] = otSource_.otHits[o].zGlobal();
            } else {
              shOrigR[k] = hits[hitId].rGlobal();
              shOrigZ[k] = hits[hitId].zGlobal();
            }
          }
        }
        alpaka::syncBlockThreads(acc);

        // Candidate-layer list, parallel over layers: for each uncovered layer a nearest-approach
        // distance from the candidate's cached original hits to the layer's (r,z) envelope, using
        // only the nominal R/Z and halfExtentR/Z rather than predictOnBarrel/Endcap, which the walk
        // runs on demand. shReachable[L] flags a candidate layer, the actual reachability being
        // decided in the walk from the full predict; shReachDist[L] is the ordering proxy, a
        // min-over-original-hits reduction plus the type-priority bias. Each lane owns layers
        // L = lane, lane+nLanes, ...
        for (auto element : cms::alpakatools::uniform_group_elements(acc, j, nC * nLanes)) {
          const uint32_t lane = element.local;
          for (int L = int(lane); L < nLayers; L += int(nLanes)) {
            shVisited[L] = 0;
            int cand = 0;
            float dist = 0.f;
            if (!(shCoveredMask & (uint64_t(1) << L))) {
              const bool isBarrel = caLayers.isBarrel()[L];
              const float R = caLayers.layerR()[L];
              const float Z = caLayers.layerZ()[L];
              const float hR = caLayers.halfExtentR()[L];
              const float hZ = caLayers.halfExtentZ()[L];
              float d = 1e9f;
              for (int k = 0; k < shNOrigSafe; ++k) {
                const float rk = shOrigR[k];
                const float zk = shOrigZ[k];
                float dd;
                if (isBarrel) {
                  // nearest approach to the barrel cylinder R (z spans [Z-hZ, Z+hZ]): radial gap
                  // |R-r_ref| combined with the z-envelope overlap gap (0 inside the envelope).
                  const float dr = R - rk;
                  const float zGap = alpaka::math::max(acc, alpaka::math::abs(acc, zk - Z) - hZ, 0.f);
                  dd = alpaka::math::sqrt(acc, dr * dr + zGap * zGap);
                } else {
                  // nearest approach to the endcap disk Z (r spans [R-hR, R+hR]): z gap |Z-z_ref|
                  // combined with the r-envelope overlap gap (0 inside the envelope).
                  const float dz = Z - zk;
                  const float rGap = alpaka::math::max(acc, alpaka::math::abs(acc, rk - R) - hR, 0.f);
                  dd = alpaka::math::sqrt(acc, dz * dz + rGap * rGap);
                }
                if (dd < d)
                  d = dd;
              }
              const bool isPixelLayer = (L < 28);
              const float typeBias = (isPixelLayer == (shPreferPixel != 0)) ? -typePriorityBiasCm : 0.f;
              cand = 1;
              dist = d + typeBias;
            }
            shReachable[L] = cand;
            shReachDist[L] = dist;
          }
        }
        alpaka::syncBlockThreads(acc);

        // Walk the candidate layers in (proxy dist, then L) order: the full predict is confirmed in
        // one lane-parallel batch the first time the confirmed set is dry, every candidate layer at
        // once with each lane owning L = lane, lane+nLanes, ... Since a visit is chosen by the
        // select-min over confirmed layers, confirming them all at once changes neither the visit
        // order nor the physics. Layers whose predict fails the envelope/arc test are consumed
        // without spending the K = maxWalkLayers budget, and after the eager round each step is a
        // lane-parallel select plus one window scan. shReachable states: 0 = covered or confirmed
        // unreachable, 1 = candidate, 2 = confirmed reachable. shWalkDone: 0 = scan shCurrentL,
        // 1 = walk finished, 2 = eager-confirm pending.
        while (true) {
          // warp-parallel select: each lane keeps the local-best confirmed-reachable unvisited layer
          // over its stride L = lane, lane+nLanes, ... (min proxy dist, ties by smaller L via the
          // ascending strided scan + strict <), staged into laneHit/laneChi2 (both free here: the hit
          // scan below rewrites them before its reduction reads them).
          for (auto element : cms::alpakatools::uniform_group_elements(acc, j, nC * nLanes)) {
            const uint32_t lane = element.local;
            int bestL = -1;
            float bestD = 0.f;
            // Also stage the per-lane best OT-disk layer (L >= 28, i.e. not a pixel layer, and
            // !isBarrel) and the per-lane OT-disk tally, so the reduction can enforce the
            // region-balanced budget without a second layer sweep. Pixel layers (barrel and disks) are
            // deliberately excluded from both the reserve accounting and the forced selection: the
            // crowding this fixes is strictly the OT TOB-vs-TID contention, and counting pixel layers
            // here starves the pixel attach on stub tracks instead.
            int bestLD = -1;
            float bestDD = 0.f;
            int nDiskLane = 0;
            for (int L = int(lane); L < nLayers; L += int(nLanes)) {
              if (shReachable[L] == 2 && !shVisited[L]) {
                if (bestL < 0 || shReachDist[L] < bestD) {
                  bestD = shReachDist[L];
                  bestL = L;
                }
                if (L >= 28 && !caLayers.isBarrel()[L]) {  // OT disk (28 = first OT CA layer)
                  ++nDiskLane;
                  if (bestLD < 0 || shReachDist[L] < bestDD) {
                    bestDD = shReachDist[L];
                    bestLD = L;
                  }
                }
              }
            }
            laneHit[lane] = bestL;
            laneChi2[lane] = bestD;
            laneHitDisk[lane] = bestLD;
            laneChi2Disk[lane] = bestDD;
            laneNDisk[lane] = nDiskLane;
          }
          alpaka::syncBlockThreads(acc);

          // In-place shared-memory tree reduction across all lanes. The comparator is a strict total
          // order (dist, then layer L, and each candidate layer lives in exactly one lane's stride),
          // so min is associative and commutative and any reduction order yields the same winner on
          // every backend. Three reductions share one log2(nLanes)-step tree: primary (laneChi2=dist,
          // laneHit=L), OT-disk (laneChi2Disk, laneHitDisk) and the nDiskUnvis integer sum
          // (laneNDisk). Invalid entries carry laneHit/laneHitDisk == -1.
          for (uint32_t stride = nLanes >> 1; stride > 0u; stride >>= 1) {
            for (auto element : cms::alpakatools::uniform_group_elements(acc, j, nC * nLanes)) {
              const uint32_t lane = element.local;
              if (lane < stride) {
                const uint32_t o = lane + stride;
                const bool takeP = (laneHit[lane] < 0) ||
                                   (laneHit[o] >= 0 && (laneChi2[o] < laneChi2[lane] ||
                                                        (laneChi2[o] == laneChi2[lane] && laneHit[o] < laneHit[lane])));
                if (takeP) {
                  laneChi2[lane] = laneChi2[o];
                  laneHit[lane] = laneHit[o];
                }
                const bool takeD = (laneHitDisk[lane] < 0) ||
                                   (laneHitDisk[o] >= 0 &&
                                    (laneChi2Disk[o] < laneChi2Disk[lane] ||
                                     (laneChi2Disk[o] == laneChi2Disk[lane] && laneHitDisk[o] < laneHitDisk[lane])));
                if (takeD) {
                  laneChi2Disk[lane] = laneChi2Disk[o];
                  laneHitDisk[lane] = laneHitDisk[o];
                }
                laneNDisk[lane] += laneNDisk[o];
              }
            }
            alpaka::syncBlockThreads(acc);
          }

          // lane 0: read the reduced global bests (gL / gLD / nDiskUnvis) from slot 0, budget-check,
          // then decide: visit the winner, eager-confirm all candidates once, or finish.
          for (auto element : cms::alpakatools::uniform_group_elements(acc, j, nC * nLanes)) {
            if (element.local != 0u)
              continue;
            const int gL = laneHit[0];       // -1 when no confirmed-reachable unvisited layer
            const int gLD = laneHitDisk[0];  // -1 when no unvisited OT disk
            const int nDiskUnvis = laneNDisk[0];
            // Region-balanced walk budget, reserved-seats form: up to ceil(K/2) of the
            // K = maxWalkLayers visit budget is reserved for OT disks. With
            //   nDiskReach = confirmed-reachable OT disks, constant after the eager-confirm round,
            //   reserve    = min(nDiskReach, ceil(K/2)),
            //   owed       = reserve - (OT disks already visited), clamped at 0,
            //   remaining  = K - shWalkSteps,
            // the selection is forced to the nearest unvisited OT disk as soon as remaining is no
            // larger than owed. The reachDist ordering places all reachable barrel before any disk,
            // so on transition tracks the free budget goes to the near barrel layers and the owed
            // tail recovers the disks, trading the outermost TOB visits. Since accepted extras follow
            // visit order, this also avoids spending every maxExtraHitsPerTrack slot on barrel.
            // OT-barrel-only and OT-disk-only tracks are unaffected, and pixel-layer visits spend
            // only the free budget.
            const int diskVisits = shOTDiskVisits;
            const int nDiskReach = nDiskUnvis + diskVisits;
            const int halfBudget = (extWalkBudget + 1) / 2;  // loop-bound uses the runtime budget
            const int diskReserve = (nDiskReach < halfBudget) ? nDiskReach : halfBudget;
            const int owed = (diskReserve > diskVisits) ? (diskReserve - diskVisits) : 0;
            const int remaining = extWalkBudget - shWalkSteps;  // loop-bound uses the runtime budget
            // Pixel-first visit budget: on prefer-pixel (OT-only-core) hosts, reserve up to
            // extRecallPixelFirstBudget of the K seats for pixel by suppressing OT-disk forcing while the
            // nearest unvisited-reachable layer is a pixel layer (gL < 28, already the -typePriorityBiasCm
            // winner) and fewer than the reserve many pixel layers have been visited. This keeps the
            // reserved OT-disk seats from crowding out reachable pixel layers the walk would otherwise
            // skip. Scoped to shPreferPixel so stub tracks (the population the pixel exclusion at
            // select-staging protects) are untouched. Budget 0 => never suppresses.
            const bool pixReserveActive = (extRecallPixelFirstBudget > 0) && (shPreferPixel != 0) &&
                                          (gL >= 0 && gL < 28) && (shPixVisits < extRecallPixelFirstBudget);
            const bool forceDisk = (gLD >= 0) && (remaining <= owed) && !pixReserveActive;
            const int selL = forceDisk ? gLD : gL;
            // MTV-aligned extra cap: stop once the cluster budget is exhausted, since not even a
            // one-cluster extra could be appended. With the cap off shExtraClusterCap is the sentinel
            // and this term is never true. An anchored track (>=1 accepted OT extra on a prior walk
            // layer) is exempt from the cluster-budget term, since a short OT-only core reaches the
            // cap right after its first anchor and would terminate before reaching TOB5/6; the
            // maxExtraHitsPerTrack and maxWalkLayers caps still bind. With extCapExemptTOB46Only the
            // exemption fires only when the next selected layer is itself TOB4-6 (CA 31-33). No
            // candidate gate chi2 exists at this decision, so extCapExemptMaxChi2 applies at the
            // per-accept site below instead.
            const bool capStopExempt =
                extCapExemptAnchored && shNOTExtraAcc >= 1 && (!extCapExemptTOB46Only || (selL >= 31 && selL <= 33));
            if (shNExtra >= maxExtraHitsPerTrack || shWalkSteps >= extWalkBudget ||
                (!capStopExempt && shNExtraClusters >= shExtraClusterCap)) {
              shWalkDone = 1;
              if (shNExtra >= maxExtraHitsPerTrack) {  // this walk ended on the extras-slot budget
                alpaka::atomicAdd(acc, &stats[kDiagSlotExhaust], 1u, alpaka::hierarchy::Grids{});
                if (shHasOT != 0)
                  alpaka::atomicAdd(acc, &stats[kDiagSlotExhaustOT], 1u, alpaka::hierarchy::Grids{});
              }
            } else if (selL >= 0) {
              shCurrentL = selL;
              shVisited[selL] = 1;
              // far-first window-ambiguity condition: is this visit one of the far crossings the
              // re-key promoted? Endcap pixel layer, armed host, and beyond the host's own outermost
              // |z| -- the interior discs are the fallback and keep the unconditioned commit rule.
              // shFarArmed is 0 for every host the ordering does not touch and for every host when the
              // ordering is off, so both the scan-side count and the commit-side decline below are then
              // unreachable. Lane-0 store; the syncBlockThreads after this block publishes it to the
              // scan lanes.
              shFarPass = 0u;
              shFarLayer = (shFarArmed && extAttachFarMaxWin > 0 && selL < 28 && !caLayers.isBarrel()[selL] &&
                            alpaka::math::abs(acc, caLayers.layerZ()[selL]) > shFarZFloor)
                               ? 1
                               : 0;
              if (candDump_) {  // dump: reset this layer's per-(candidate,layer) accumulators
                shDumpOccM = 0u;
                shDumpOccO = 0u;
                shDumpPassM = 0u;
                shDumpPassO = 0u;
                shDumpBestM = 0xFFFFFFFFu;
                shDumpWinHit = -1;
                shDumpWinChi2 = -1.f;
                shDumpWinRound = -1;
                shDumpCapDropped = 0;
                shDumpVeto = 0;
                shDumpPartner = 0;
                shDumpMemN = 0u;  // reset this layer's member write cursor
                // Reset every lane's best-any (min-chi2 over all considered) slot to the sentinel
                // for this layer (both rounds accumulate into it). Lane-0 store; a syncBlockThreads follows
                // the layer-select block before the scan, so all lanes see the reset.
                for (uint32_t l2 = 0; l2 < nLanes; ++l2) {
                  laneBestAnyChi2[l2] = 3.4e38f;
                  laneBestAnyHit[l2] = -1;
                  laneBestAnyPass[l2] = 0;
                }
              }
              // Reset this layer's gate-passer + occupancy accumulators (or-ed / summed across
              // rounds). Under candDump_ so the hole counter can consume them below.
              if (candDump_) {
                shChainHasPass = 0;
                shHoleOcc = 0u;
              }
              shWalkDone = 0;
              ++shWalkSteps;  // only reachable-visited layers count toward the K budget
              if (selL >= 28 && !caLayers.isBarrel()[selL])
                ++shOTDiskVisits;  // OT-disk visits consumed from the reserve
              if (selL < 28)
                ++shPixVisits;  // pixel visits consumed from the pixel-first reserve
            } else if (!shConfirmDone) {
              shWalkDone = 2;     // confirmed set dry, candidates still unexamined -> eager-confirm all
              shConfirmDone = 1;  // one round leaves no state-1 layer; the walk never stages again
            } else {
              shWalkDone = 1;  // no confirmed unvisited layer and no candidate left -> walk finished
            }
          }
          alpaka::syncBlockThreads(acc);
          if (shWalkDone == 1)
            break;
          if (shWalkDone == 2) {
            // eager batch confirm (parallel over lanes): confirm every state-1 (candidate) layer in
            // <=2 strided rounds, each lane owning L = lane, lane+nLanes, ... one on-demand full
            // predict per layer, tested against the envelope/arc criteria (kReachSlackCm) and
            // evaluated on the current running helix -- which at this first dry point is still the
            // initial fitted helix, no hit having been attached yet. Sets state 2 (reachable) or 0
            // (consumed, never counted against K).
            for (auto element : cms::alpakatools::uniform_group_elements(acc, j, nC * nLanes)) {
              const uint32_t lane = element.local;
              for (int L = int(lane); L < nLayers; L += int(nLanes)) {
                if (shReachable[L] != 1)
                  continue;
                const bool isBarrel = caLayers.isBarrel()[L];
                const float R = caLayers.layerR()[L];
                const float Z = caLayers.layerZ()[L];
                const Prediction pr =
                    isBarrel ? predictOnBarrel(acc, sh.helix(), R) : predictOnEndcap(acc, sh.helix(), Z);
                bool ok = pr.valid;
                if (ok) {
                  // Pixel/inward recall: add extRecallReachRelax cm of extra envelope slack on
                  // pixel layers (CA L<28) for prefer-pixel (OT-only-core) hosts only, so the in-road pixel
                  // layers whose OT-only inward extrapolation drifts just past the strict kReachSlackCm=1.0cm
                  // envelope become reachable (their hits sit well inside the road). Non-pixel layers
                  // and non-prefer-pixel hosts keep kReachSlackCm exactly. 0 => no extra slack.
                  const float reachSlack =
                      kReachSlackCm +
                      ((extRecallReachRelax > 0.f && shPreferPixel != 0 && L < 28) ? extRecallReachRelax : 0.f);
                  // Force-pixel-visit: for prefer-pixel hosts, a pixel layer (CA L<28) with a valid
                  // crossing (ok == pr.valid here) is confirmed reachable regardless of the envelope test --
                  // the fragile OT-only inward extrapolation only mis-centers such a crossing, and the
                  // per-layer road/gate scan then decides. A valid crossing is still required: a host
                  // whose predict is invalid is state-limited and not recoverable here.
                  // Off / non-pixel / non-prefer-pixel => the envelope test runs.
                  const bool forcePixJ3 = (extRecallForcePixelVisit && shPreferPixel != 0 && L < 28);
                  if (!forcePixJ3) {
                    if (isBarrel)
                      ok = !(alpaka::math::abs(acc, pr.secondary - Z) > caLayers.halfExtentZ()[L] + reachSlack);
                    else
                      ok = !(alpaka::math::abs(acc, pr.secondary - R) > caLayers.halfExtentR()[L] + reachSlack);
                  }
                }
                // A near-tangential crossing is not rejected here: with dc = |circle centre|, a
                // crossing at radius r_x has c = (r_x^2 + dc^2 - rho^2)/(2 dc r_x), and as |c| -> 1 the
                // phi-window width diverges as 1/sqrt(1 - c^2). What bounds that amplification for
                // low-pT curlers and displaced large-tip tangencies is the phi-window clamp and the
                // runaway ceilings in the scan below, not this reachability test.
                shReachable[L] = ok ? 2 : 0;  // unreachable: consumed, never counted toward K
              }
            }
            alpaka::syncBlockThreads(acc);
            // Far-first disc ordering (see AttachParams::extAttachFarFirst). The select-min below
            // ranks layers by shReachDist[], a nearest-first proxy; on a forward pixel-only host that
            // is min_k |Z_L - z_k|, so the walk takes the next disc out and spends its budgets inward,
            // where the transverse lever arm does not grow. Re-keying the endcap pixel layers to
            // -|layerZ[L]| visits the farthest reachable crossing first: the crossing radius is
            // monotone in |z| along a single-sided trajectory, so this prefers the outermost crossing
            // without a predicted radius, which does not exist yet at ordering time. This is the only
            // point at which shReachable[] is final, so the no-OT-opportunity test A3 is evaluated
            // once with no extra sync. shReachDist[] feeds no road, gate, covariance or output, so
            // re-keying it changes the visit order only; barrel and OT layers keep the nearest-first
            // key, leaving the OT-disk reserve untouched.
            if (extAttachFarFirst) {
              for (auto element : cms::alpakatools::uniform_group_elements(acc, j, nC * nLanes)) {
                if (element.local != 0u)
                  continue;
                // A1: the host's own fitted |cotTheta|, the same object the host mask and the pre-gate
                // compare against sinh(maxAbsEta). Everything below the floor keeps the nearest-first
                // order untouched.
                const float absCot = alpaka::math::abs(acc, tracks[i].state()(3));
                const bool a1 = absCot >= alpaka::math::sinh(acc, extAttachFarMinAbsEta);
                // A2: the host core carries no OT/stub hit. Stub-carrying forward tracks lose from a
                // re-ordered pixel visit, so they are excluded by construction rather than by tuning.
                const bool a2 = (shCoveredMask >> 28) == 0;
                // A3: no OT layer is confirmed reachable. Where TEDD content is still in reach it
                // carries a longer lever arm than any pixel disc and the nearest-first order collects it
                // already.
                bool otReach = false;
                for (int L = 28; L < nLayers; ++L) {
                  if (shReachable[L] == 2) {
                    otReach = true;
                    break;
                  }
                }
                if (a1 && a2 && !otReach) {
                  for (int L = 0; L < nLayers && L < 28; ++L) {
                    if (shReachable[L] == 2 && !caLayers.isBarrel()[L])
                      shReachDist[L] = -alpaka::math::abs(acc, caLayers.layerZ()[L]);
                  }
                  // The far-crossing floor for the window-ambiguity condition: the outermost |z| the host
                  // already occupies. A disc beyond it extends the track (this is the "extends?" property
                  // the ordering exists to buy); a disc inside it is an interior hole and keeps the
                  // unconditioned commit rule, so it can still take the budget seat a declined far
                  // crossing gives back. Read from the same cached original hits the reach proxy used, so it costs
                  // one lane-0 sweep over at most kMaxOrigHits floats and no new state.
                  float zFloor = 0.f;
                  for (int k = 0; k < shNOrigSafe; ++k) {
                    const float az = alpaka::math::abs(acc, shOrigZ[k]);
                    if (az > zFloor)
                      zFloor = az;
                  }
                  shFarZFloor = zFloor;
                  shFarArmed = 1;
                  alpaka::atomicAdd(acc, &stats[kStatAttachFarArmed], 1u, alpaka::hierarchy::Grids{});
                }
              }
              alpaka::syncBlockThreads(acc);  // publish the re-keyed order to every selecting lane
            }
            continue;  // re-select among the newly confirmed layers
          }

          // Per-(candidate,layer) linearization of the crossing quantities. The per-hit scan below
          // crosses the layer at every in-window hit's own radius rh (barrel) or z (endcap), where
          // evaluating predictOn{Barrel,Endcap} exactly would cost about a dozen transcendentals per
          // hit over 40-80 hits per window. The crossing quantities (phi, secondary, arcS) are
          // instead expanded about the layer reference R0/Z0 to a Taylor polynomial (phi and
          // secondary 2nd order, arcS 1st) whose coefficients are the analytic derivatives of the
          // same closed form, so the per-hit work collapses to a few FMAs. A per-layer fallback
          // (shUseLin=0) keeps the exact predict for near-tangential or Taylor-stressed layers.
          //
          // Barrel (u = rh - R0): with circle centre (xc,yc), |centre| dc, signed radius rho and
          //   c(R) = (R^2 + dc^2 - rho^2)/(2 dc R)  [= cos of the O-vertex crossing angle], the crossing
          //   azimuth is theta(R) = phi_c + sBranch*acos(c) so
          //     c'   = 1/(2dc) - (dc^2-rho^2)/(2 dc R^2),   c'' = (dc^2-rho^2)/(dc R^3)
          //     dphi/dr  = -sBranch * c'/sqrt(1-c^2)
          //     d2phi/dr2= -sBranch * ( c''/sqrt(1-c^2) + c c'^2/(1-c^2)^{3/2} )
          //   The arc derivative follows from ds/dR = rho / (xc sin theta - yc cos theta) (radial vs
          //   circle-tangent projection), and dz/dr = cotTheta * ds/dr, d2z/dr2 = cotTheta * d2s/dr2 with
          //   d2s/dr2 = -rho*(xc cos theta + yc sin theta)*(dphi/dr) / (xc sin theta - yc cos theta)^2.
          // Endcap (u = zh - Z0): arcS = (z - zip)/cotTheta is exactly linear (darcS/dz = 1/cotTheta),
          //   and alphaH(z) = alphaOrigin - arcS/rho is linear too (a1 = dalphaH/dz = -1/(rho cotTheta)).
          //   With x = xc + |rho| cos alphaH, y = yc + |rho| sin alphaH (so x',x'' etc. follow the chain
          //   rule in a1), phi = atan2(y,x), r = sqrt(x^2+y^2):
          //     dphi/dz = (x y' - y x')/r^2,  d2phi/dz2 = (x y'' - y x'' - (dphi/dz)*2(x x'+y y'))/r^2
          //     dr/dz   = (x x' + y y')/r,    d2r/dz2   = (x'^2+x x''+y'^2+y y'')/r - (x x'+y y')^2/r^3
          for (auto element : cms::alpakatools::uniform_group_elements(acc, j, nC * nLanes)) {
            if (element.local != 0u)
              continue;
            const int L = shCurrentL;
            const bool isBarrel = caLayers.isBarrel()[L];
            const float R0 = isBarrel ? caLayers.layerR()[L] : caLayers.layerZ()[L];
            const float W = isBarrel ? caLayers.halfExtentR()[L] : caLayers.halfExtentZ()[L];
            const HelixState hh = sh.helix();
            const Prediction p0 = isBarrel ? predictOnBarrel(acc, hh, R0) : predictOnEndcap(acc, hh, R0);
            shLinRef = R0;
            shLinPhi0 = p0.phi;
            shLinSec0 = p0.secondary;
            shLinArc0 = p0.arcS;
            shLinDphi1 = 0.f;
            shLinDphi2 = 0.f;
            shLinDsec1 = 0.f;
            shLinDsec2 = 0.f;
            shLinDarc1 = 0.f;
            shLinArgLo = -1.0e30f;
            shLinArgHi = 1.0e30f;
            int useLin = 0;
            if (p0.valid) {
              const float xc = hh.xc, yc = hh.yc, rho = hh.rho;
              const float absRho = alpaka::math::abs(acc, rho);
              if (isBarrel) {
                const float dc2 = xc * xc + yc * yc;
                const float dc = alpaka::math::sqrt(acc, dc2);
                const float Aa = dc2 - rho * rho;
                float c0 = (R0 * R0 + Aa) / (2.f * dc * R0);
                c0 = alpaka::math::min(acc, 1.f, alpaka::math::max(acc, -1.f, c0));
                const float sq2 = alpaka::math::max(acc, 1.f - c0 * c0, kLinCondEps * kLinCondEps);
                const float sq = alpaka::math::sqrt(acc, sq2);
                const float cp1 = 1.f / (2.f * dc) - Aa / (2.f * dc * R0 * R0);
                const float cpp = Aa / (dc * R0 * R0 * R0);
                const float phic = alpaka::math::atan2(acc, yc, xc);
                const float sBranch = (foldPi(p0.phi - phic) >= 0.f) ? 1.f : -1.f;
                const float dphi1 = -sBranch * cp1 / sq;
                const float dphi2 = -sBranch * (cpp / sq + c0 * cp1 * cp1 / (sq * sq2));
                const float st = alpaka::math::sin(acc, p0.phi);
                const float ct = alpaka::math::cos(acc, p0.phi);
                const float Q0 = xc * st - yc * ct;
                const float dsdr = rho / Q0;
                const float Qp = (xc * ct + yc * st) * dphi1;
                const float d2sdr2 = -rho * Qp / (Q0 * Q0);
                shLinDphi1 = dphi1;
                shLinDphi2 = dphi2;
                shLinDsec1 = hh.cotTheta * dsdr;
                shLinDsec2 = hh.cotTheta * d2sdr2;
                shLinDarc1 = dsdr;
                shLinArgLo = dc - absRho - 1.0e-3f;  // predictOnBarrel radial validity band
                shLinArgHi = dc + absRho + 1.0e-3f;
                const float t2 = 0.5f * alpaka::math::abs(acc, dphi2) * W * W;
                useLin = ((1.f - alpaka::math::abs(acc, c0)) > kLinCondEps &&
                          alpaka::math::abs(acc, Q0) > kLinCondEps && t2 < kTaylor2ndMaxRad)
                             ? 1
                             : 0;
              } else {
                const float cot = hh.cotTheta;
                const float a1 = -1.f / (rho * cot);  // dalphaH/dz (constant in z)
                const float alphaH = hh.alphaOrigin - p0.arcS / rho;
                const float sA = alpaka::math::sin(acc, alphaH);
                const float cA = alpaka::math::cos(acc, alphaH);
                const float x = xc + absRho * cA;
                const float y = yc + absRho * sA;
                const float r2 = alpaka::math::max(acc, x * x + y * y, 1.0e-6f);
                const float rr = alpaka::math::sqrt(acc, r2);
                const float xp = -absRho * sA * a1;
                const float yp = absRho * cA * a1;
                const float xpp = -absRho * cA * a1 * a1;
                const float ypp = -absRho * sA * a1 * a1;
                const float Nn = x * yp - y * xp;
                const float dphi1 = Nn / r2;
                const float Np = x * ypp - y * xpp;
                const float Dp = 2.f * (x * xp + y * yp);
                const float dphi2 = (Np - dphi1 * Dp) / r2;
                const float Mm = x * xp + y * yp;
                const float drdz = Mm / rr;
                const float Mp = xp * xp + x * xpp + yp * yp + y * ypp;
                shLinDphi1 = dphi1;
                shLinDphi2 = dphi2;
                shLinDsec1 = drdz;
                shLinDsec2 = Mp / rr - Mm * Mm / (rr * r2);
                shLinDarc1 = 1.f / cot;  // arcS = (z - zip)/cotTheta is exactly linear
                const float t2 = 0.5f * alpaka::math::abs(acc, dphi2) * W * W;
                useLin = (t2 < kTaylor2ndMaxRad) ? 1 : 0;  // NaN-safe: NaN < thr is false -> fall back
              }
            }
            shUseLin = useLin;
            shLinValid = p0.valid ? 1 : 0;  // shLinDphi1 is a real derivative iff p0 was valid

            // the derived road, once per layer visit
            // Everything expensive in the derived road model is a property of the layer crossing, not
            // of the individual candidate hit: the band prediction, the incidence brackets, the material
            // integral and the quantile threshold. Computing them here -- once, on the reference
            // crossing p0 -- instead of per hit is what makes the derived window cheaper than the
            // fixed-cut one: the per-hit `segmentXX0` material march, which dominates the walk's cost,
            // collapses to one march per visit.
            shDerLayOn = 0;
            shDerSigR2 = 0.f;
            shDerSigS2 = 0.f;
            shDerQ = 0.f;
            shDerWinR = 0.f;
            shDerCeilR = 0.f;
            shDerCeilS = 0.f;
            shDerHoleK = -1e30f;
            shDerHoleKRaw = -1e30f;
            shDerHoleK3 = -1e30f;
            shDerBendOn = 0;
            shDerPredB = 0.f;
            shDerRbbTrk = 0.f;
            shDerQ3 = 0.f;
            shDerC = 0.f;
            shDerW = 0.f;
            shDerS1 = 0.f;
            shDerS2 = 0.f;
            shDerElossPhi = 0.f;
            shDerElossSec = 0.f;
            // The derived road covers the OT layers. On the pixel discs the walk enumerates, reaches
            // and searches as before, but there is no measured Q-hat class and no eta_L/rho row for
            // them, so the derived road stays off there and the fixed-cut gate is used.
            if (shDerOn && L >= 28 && p0.valid) {
              // Runaway ceilings, derived from this layer's module envelope. On the derived path an
              // absolute cm cap is only a ceiling on pathological states, set from geometry rather
              // than resolution and far outside the eps-window of a healthy road; the fixed cm caps
              // used off this path would bind most roads here and act as the selector. Both ceilings
              // come from the layer's module-envelope half-extents plus the same kReachSlackCm the
              // reachability test grants the prediction: for the secondary (barrel z, endcap r)
              // prediction and hit both lie within h_sec of nominal, so |dSec| <= 2 h_sec + slack is a
              // hard geometric bound; in R-phi the envelope bounds nothing azimuthally and the only
              // transverse scale it supplies is the crossing surface's thickness 2 h_prop, so
              // ceilR = 2 h_prop + slack, which still caps the phi-bin scan. Both binding rates are
              // counted (kStatDerCapR / kStatDerCapS): well under 1 % means the ceiling is a guard,
              // more means it is doing the selection and the delivered efficiency is not eps.
              const float hSecEnv = isBarrel ? caLayers.halfExtentZ()[L] : caLayers.halfExtentR()[L];
              const float hPropEnv = isBarrel ? caLayers.halfExtentR()[L] : caLayers.halfExtentZ()[L];
              shDerCeilS = 2.f * hSecEnv + kReachSlackCm;
              shDerCeilR = 2.f * hPropEnv + kReachSlackCm;
              const float ds = p0.arcS - shDerAnchorS;  // transverse arc from the anchor to the crossing
              // the band prediction: V_u(ds) = g^T C_loc g, g = [1, ds, -ds^2/2]
              const float g1 = ds;
              const float g2 = -0.5f * ds * ds;
              const float Vband = shDerCloc[0] + g1 * g1 * shDerCloc[3] + g2 * g2 * shDerCloc[5] +
                                  2.f * (g1 * shDerCloc[1] + g2 * shDerCloc[2] + g1 * g2 * shDerCloc[4]);
              // the crossing geometry: transverse incidence psi and dip lambda
              const float rCross = isBarrel ? R0 : p0.secondary;
              const float zCross = isBarrel ? p0.secondary : R0;
              const float rSafe = alpaka::math::max(acc, rCross, 1.f);
              const float xCr = rSafe * alpaka::math::cos(acc, p0.phi);
              const float yCr = rSafe * alpaka::math::sin(acc, p0.phi);
              // cos(psi) = |P x C| / (|rho| r): the transverse track direction against the radial one.
              // Clamped away from 0 so the barrel 1/cos^2 bracket stays finite on a grazing crossing
              // (those layers are already guarded by the reachability test).
              float cpsi = alpaka::math::abs(acc, xCr * hh.yc - yCr * hh.xc) /
                           alpaka::math::max(acc, alpaka::math::abs(acc, hh.rho) * rSafe, 1e-6f);
              cpsi = alpaka::math::min(acc, 1.f, alpaka::math::max(acc, 1e-3f, cpsi));
              const float c2 = cpsi * cpsi;
              const float s2 = 1.f - c2;
              const float cot = hh.cotTheta;
              const float cot2 = cot * cot;
              const float sinlam2 = alpaka::math::max(acc, cot2 / (1.f + cot2), 1e-9f);
              const float coslam2 = 1.f / (1.f + cot2);
              // conv^2 converts the trajectory's lateral offset to the target surface; G_rphi/G_sec are
              // the exact transverse-plane projection brackets -- zero free parameters, and >= 1
              // identically, so the exact projection can only widen the road.
              const float conv2 = isBarrel ? (1.f / c2) : c2;
              const float Grphi = isBarrel ? (1.f / c2) : (c2 + s2 / sinlam2);
              const float Gsec = isBarrel ? (1.f / coslam2 + cot2 * s2 / c2) : (s2 + c2 / sinlam2);
              // the material moments of the gap anchor -> crossing (one map march per visit)
              double d1 = 0., w1 = 0.;
              const float Wm = float(brokenline::segmentXX0Moments(
                  acc, rhoMap_, double(shDerAnchorR), double(shDerAnchorZ), double(rSafe), double(zCross), d1, w1));
              // segmentXX0Moments publishes the Kleinwort two-thin-scatterer pair; the raw moments come
              // back exactly: S1 = w1 W d1, S2 = w1 W d1^2 (d measured from the arrival end = the target).
              const float S1m = float(w1) * Wm * float(d1);
              const float S2m = S1m * float(d1);
              const float pT = alpaka::math::abs(acc, shBf * hh.rho);
              const float pTot = pT * alpaka::math::sqrt(acc, 1.f + cot2);
              const float cH = extHighlandC(acc, pTot, Wm);
              const float Qa = cH * S2m;                                 // 3-D lever
              const float Qgap = shDerQgap * ds * ds;                    // last fitted gap
              const float fms = isBarrel ? extFmsBarrel : extFmsEndcap;  // material-dispersion scale
              // --- the endcap line rows (a sizeable share of the forward variance; exactly 0 in the
              // barrel by the closed form). At fixed z the crossing arc carries the (z0, cot) variance,
              // s_T = (z - z0)/cot, and an arc shift moves the crossing azimuthally by sin(psi) per unit
              // arc, so Var_line(u) = (sin psi / cot)^2 * Var_z(s_T) -- and Var_z(s_T) is exactly the
              // propagated barrel secondary variance the walk already forms.
              float Vline = 0.f;
              if (!isBarrel) {
                const float sArc = p0.arcS;
                const float varZ = sh.vZip + sArc * sArc * sh.vCot + 2.f * sArc * sh.cCotZip;
                const float sp =
                    alpaka::math::sqrt(acc, s2) / alpaka::math::max(acc, alpaka::math::abs(acc, cot), 1e-3f);
                Vline = alpaka::math::max(acc, sp * sp * varZ, 0.f);
              }
              shDerSigR2 = alpaka::math::max(acc, Vband * conv2 + (Qa + Qgap) * Grphi * fms + Vline, 0.f);
              shDerSigS2 = alpaka::math::max(acc, Qa * Gsec * fms, 0.f);
              shDerC = cH;
              shDerW = Wm;
              shDerS1 = S1m;
              shDerS2 = S2m;
              // Ionization energy-loss road centre: everything above sizes the road, this moves it.
              // The walk propagates a circle of the curvature the fast fit published, kappa_0 at the
              // production vertex, while the real track's unsigned curvature grows along the path, so
              // at an OT layer it sits inside that circle by the double integral of the growth.
              // Nothing downstream absorbs that offset: the fit subtracts its own dE/dx term, and the
              // road's width cannot, a centre offset delta inflating the empirical 2-dof quantile by
              // ~1 + (delta/sigma)^2. In the band's local frame (u = radial offset from the reference
              // circle, outward positive; u'' = -dkappa) the offset at this crossing is exactly
              //     uE = shEU + shEUp*ds - 0.5*( shEDk*ds^2 + shEK*S2m )
              // the last term being int_0^ds (ds-s') dkappa_material(s') ds' = S2/2 with S2 the second
              // moment of the traversed material about the arrival end -- the march that just ran. No
              // linearity assumption on the material profile is made.
              // Projection onto the two gate rows, first order, with the crossing geometry the width
              // brackets use: n = (P - C)/|rho| is the outward normal, b = n.phihat (|b| = cos psi),
              // a = n.rhat and a^2 + b^2 = 1.
              //   barrel (fixed r): the displacement u*n plus the slide along the track back to the
              //     cylinder give r*dphi = u/b exactly; the slide moves z by dz = -u*a*(ds/dr)*cotTheta,
              //     and (ds/dr)*cotTheta is shLinDsec1.
              //   endcap (fixed z): no slide, so r*dphi = u*b and dr = u*a directly.
              // Both shifts are added to the prediction, never to a variance. The phi shift is stored
              // in radians at the reference crossing rather than per hit.
              if (shEK > 0.f) {
                // Metric: `ds` and u' are transverse (ds = p0.arcS - shDerAnchorS) while
                // segmentXX0Moments returns S1/S2 with a 3-D lever. The eloss recursion needs the
                // lever in the same transverse metric as ds, since dkappa is a transverse curvature
                // and u a transverse offset, so one cos(lambda) per lever power survives and S2
                // carries lever^2 -> coslam2. The MS process noise below instead takes S2 with no
                // conversion, the scattering angle converting too and the two cos(lambda) cancelling.
                const float uE = shEU + shEUp * ds - 0.5f * (shEDk * ds * ds + shEK * S2m * coslam2);
                const float absRhoS = alpaka::math::max(acc, alpaka::math::abs(acc, hh.rho), 1e-6f);
                const float bN = (hh.xc * yCr - hh.yc * xCr) / (rSafe * absRhoS);          // n.phihat, signed cos(psi)
                const float aN = (rSafe - (hh.xc * xCr + hh.yc * yCr) / rSafe) / absRhoS;  // n.rhat
                // Guard the grazing crossing the same way the width does (cpsi is clamped at 1e-3):
                // 1/b is the barrel amplification and diverges there, and those layers are already
                // outside the reachability test's healthy region.
                const float bSgn = (bN >= 0.f) ? 1.f : -1.f;
                const float bSafe = bSgn * alpaka::math::max(acc, alpaka::math::abs(acc, bN), 1e-3f);
                const float dRPhiE = isBarrel ? (uE / bSafe) : (uE * bN);
                const float dSecE = isBarrel ? (-uE * aN * shLinDsec1) : (uE * aN);
                if (alpaka::math::isfinite(acc, dRPhiE) && alpaka::math::isfinite(acc, dSecE)) {
                  shDerElossPhi = dRPhiE / rSafe;
                  shDerElossSec = dSecE;
                }
              }
              // the single-eps threshold: one measured quantile, spent once
              const float absCotHost = alpaka::math::abs(acc, cot);
              const int cell = extQhatCell(L, absCotHost, shWalkSteps - 1, extFwdEtaBin);
              const float Q = (cell >= 0) ? extQthr[cell] : 0.f;
              if (Q > 0.f && alpaka::math::isfinite(acc, shDerSigR2) && alpaka::math::isfinite(acc, shDerSigS2)) {
                shDerQ = Q;
                // The r-phi window is the bounding half-width of the same chi2 ball -- a derived
                // consequence of the one eps, not a second one. The per-hit term is bounded by the
                // safe sensor upper bound so the window can be sized before the hits are read.
                constexpr float kMaxSigPhiHitCmW = 0.1f;
                // The window is sized before the hits are read, so the target-side additive variance is
                // taken at its safe upper bound: TB2S, the largest of the classes on the OT layers this
                // block covers. The bound is safe in the direction that matters: it can only admit hits
                // the per-hit gate then judges, never reject one.
                const float dvMax = extDV[kExtMatClsTB2S];
                const float sHi =
                    shDerSigR2 + kMaxSigPhiHitCmW * kMaxSigPhiHitCmW + alignSigmaPhiCm * alignSigmaPhiCm + dvMax;
                shDerWinR = alpaka::math::sqrt(acc, Q * sHi);
                shDerLayOn = 1;
                alpaka::atomicAdd(acc, &stats[kStatDerLayers], 1u, alpaka::hierarchy::Grids{});
                // The hole hypothesis's constant part:
                // chi2_hole = 2 ln[ P / ((1 - eta_L eps) nu) ],  nu = rho (2 pi)^{d/2} |R|^{1/2},
                // with the numerator P = eta_L (the detection prior, extHoleDetectionPrior on) or
                // eta_L*eps (the plain form, off -- see just below and the params header)
                // at d = 2 (the two-row statistic (r-phi, secondary), i.e. without the bend row),
                // so |R|^{1/2} = sigma_R sigma_S and the per-candidate part is -ln(sigma_R^2 sigma_S^2).
                // eta_L / rho: the layer-keyed rows are [L - 28], with no |eta| axis, so a TID row is
                // dominated by whichever |eta| population supplies most of its residuals (see
                // AttachParams::extEtaL).
                const float etaL = extEtaL[L - 28];
                const float rhoL = extRho[L - 28];
                const float etaLc = alpaka::math::min(acc, 0.9999f, alpaka::math::max(acc, 1e-4f, etaL));
                const float ne = etaLc * extDerivedEps;
                // the hit hypothesis's prior. The plain form eta_L*eps double-counts the window,
                // because the Gaussian density this prior multiplies already integrates to eps over that
                // same window (see AttachParams::extHoleDetectionPrior for the derivation and the pda
                // reference). The corrected form is the detection probability eta_L alone; the gate mass
                // belongs only in the no-detection term (1 - eta_L*eps), which is unchanged either way.
                // The correction is exactly + 2 ln(1/eps) of chi2_hole and introduces no new number.
                const float pdNum = extHoleDetectionPrior ? etaLc : ne;
                // Armed at d >= 3 only (kExtDerHoleMinDim): the hole's price is set by the window
                // volume, and at d = 2 that volume is an order of magnitude larger than the d = 3
                // one, dropping chi2_hole below the median gate threshold. Without extBendPackage
                // the hole is therefore compiled but inert and extDerivedHole has no effect.
                // The bend row's per-layer constants:
                // p_b = (dPhiDr_hit - dPhiDr_track) / sqrt(R_bb),
                //     R_bb = sigma_b^2 + H_b C H_b^T + Q_MS,bb.
                // track side: the closed-form dphi/dr of the fitted helix at this surface -- the same
                // derivative the walk already linearises, in analytic rather than finite-difference
                // form (see extBendPredDPhiDr).
                // H_b: its partials wrt (phi0, d0, 1/pT, cot, z0), from the same crossing expression
                // (ExtenderHelixHelpers.h::extBendPredDPhiDrGrad). Only four are computed: dphi/dr at
                // the crossing is a rotation invariant, so its phi0 partial vanishes in both regions,
                // and in the barrel it depends on neither cot nor z0.
                // Q_MS,bb: the MS angular variance at the crossing converted to dPhiDr units,
                // tan(alpha) = r dPhiDr => d(dPhiDr)/d(alpha) = (1 + tan^2 alpha)/r.
                if (extBendPackage) {
                  const float predB = extBendPredDPhiDr(acc, hh, isBarrel, R0);
                  if (predB != 0.f) {
                    float Hb1 = 0.f, Hb2 = 0.f, Hb3 = 0.f, Hb4 = 0.f;
                    const bool hbOk = extBendPredDPhiDrGrad(acc, hh, isBarrel, R0, shBf, predB, Hb1, Hb2, Hb3, Hb4);
                    if (hbOk) {
                      // H_b0 (the phi0 partial) is identically zero, so every cross term it appears in
                      // drops out of H_b C H_b^T analytically and the quadratic form runs over the four
                      // surviving partials only.
                      const float HCH = Hb1 * Hb1 * sh.vTip + Hb2 * Hb2 * sh.vPt + 2.f * Hb1 * Hb2 * sh.cTipPt +
                                        Hb3 * Hb3 * sh.vCot + Hb4 * Hb4 * sh.vZip + 2.f * Hb3 * Hb4 * sh.cCotZip;
                      const float varAng = cH * Wm;  // theta_0^2 accumulated to the crossing [rad^2]
                      const float tana = rSafe * predB;
                      const float jAng = (1.f + tana * tana) / rSafe;
                      const float Qmsb = varAng * jAng * jAng * fms;
                      const float rbbTrk = alpaka::math::max(acc, HCH, 0.f) + alpaka::math::max(acc, Qmsb, 0.f);
                      // The third row is the stub bend. The enclosing block is OT-only and extQhatCell
                      // returns -1 off it, so the `L >= 28` conjunct below is implied; it is kept as an
                      // explicit guard.
                      const float Q3 = (cell >= 0 && L >= 28 && extQhat3 != nullptr) ? extQhat3[cell] : 0.f;
                      if (alpaka::math::isfinite(acc, rbbTrk) && Q3 > 0.f) {
                        shDerBendOn = 1;
                        shDerPredB = predB;
                        shDerRbbTrk = rbbTrk;
                        shDerQ3 = Q3;
                        // The phi-scan is sized before the hits are read, so it must cover the wider
                        // of the two statistics; the per-candidate gate then uses its own.
                        if (Q3 > Q)
                          shDerWinR = alpaka::math::sqrt(acc, Q3 * sHi);
                      }
                    }
                  }
                }
                // The hole arms when the statistic reaches 3 dof (see kExtDerMeasDim*):
                // nu = rho_d (2 pi)^{d/2} sqrt(|R_d|), and the density must live in the measurement
                // space the volume is taken in. |R_3| = sigma_R^2 sigma_S^2 R_bb has units cm^2 rad^2,
                // so its square root is a cm x rad and the density that makes nu dimensionless is a
                // per-area-per-bend one: rho_3 = rho_A / (2 b99), with b99 the per-layer 99th-percentile
                // bend half-range [rad/cm]. Using the areal rho_A against a 3-dof volume is dimensionally
                // wrong and shifts chi2_hole by 2 ln(2 b99), several units. Both constants are formed here
                // and the commit site picks the one matching the winner's own dimension: a stub winner
                // carries a bend row (d = 3), a raw-OT winner does not (d = 2).
                if (shDerBendOn && L >= 28 && extDerivedHole && ne > 0.f && ne < 1.f) {
                  constexpr float kTwoPi32 = 15.7496099f;  // (2 pi)^{3/2}
                  const float rho3L = (L >= 28) ? extRho3[L - 28] : 0.f;
                  if (rho3L > 0.f)
                    shDerHoleK3 = 2.f * alpaka::math::log(acc, pdNum / ((1.f - ne) * rho3L * kTwoPi32));
                }
                // The d = 2 branch is for a raw-OT winner inside the bend package, never a mode of its
                // own: without extBendPackage the hole must stay fully inert. The guard is explicit here
                // because the dimension is a runtime quantity, and a missing guard silently arms the
                // 2-dof hole -- which, priced against a 3-dof volume, declines a large fraction of the
                // argmin winners and costs hit content.
                if (extBendPackage && extDerivedHole && rhoL > 0.f && ne > 0.f && ne < 1.f) {
                  constexpr float kTwoPi = 6.2831853f;
                  shDerHoleK = 2.f * alpaka::math::log(acc, pdNum / ((1.f - ne) * rhoL * kTwoPi));
                  // The raw-OT round's own price (extHoleRawRoundPrior). Round 1 runs only where
                  // round 0 attached nothing, so the hypothesis it arbitrates is not stub
                  // availability but the conditional availability of a usable raw cluster given that
                  // no stub was formed, eta_cond = (eta_rawOT - eta_stub)/(1 - eta_stub), and the
                  // background it competes against is raw clusters rather than stubs. Pricing a raw
                  // winner with the stub rows would make the hole several chi2 units too expensive.
                  // Both replacement rows are measured like eta_L; no new free parameter, same eps.
                  const float etaLR =
                      (extEtaLRaw != nullptr && L >= 28)
                          ? alpaka::math::min(acc, 0.9999f, alpaka::math::max(acc, 1e-4f, extEtaLRaw[L - 28]))
                          : etaLc;
                  const float rhoR = (extRhoRaw != nullptr && L >= 28) ? extRhoRaw[L - 28] : rhoL;
                  const float neR = etaLR * extDerivedEps;
                  const float pdNumR = extHoleDetectionPrior ? etaLR : neR;
                  if (rhoR > 0.f && neR > 0.f && neR < 1.f)
                    shDerHoleKRaw = 2.f * alpaka::math::log(acc, pdNumR / ((1.f - neR) * rhoR * kTwoPi));
                }
              }
              // The road-centre shift is computed above, before the Q-hat lookup that arms shDerLayOn,
              // so a layer whose derived road fails to arm (Q <= 0 or a non-finite sigma, the
              // pathological state the fallback exists for) would otherwise carry a shifted prediction
              // into the fixed-cut fallback. Keep the two paths cleanly separated: derived road on =>
              // corrected prediction + derived gate; off => uncorrected prediction + fixed-cut gate.
              if (!shDerLayOn) {
                shDerElossPhi = 0.f;
                shDerElossSec = 0.f;
              }
            }
          }
          alpaka::syncBlockThreads(acc);

          // Additive attach policy: two rounds per layer visit sharing the same lane arrays and
          // syncBlockThreads pattern. Round 0 scans the merged hit source; round 1 scans the raw OT
          // source only where round 0 attached nothing and the OT source is active, so OT hits add on
          // stub-less layers instead of replacing a merged attachment. The lane arrays are reused for
          // round 1 after round 0's lane-0 reduce has consumed them, separated by a syncBlockThreads.
          // With no OT source round 1 never runs.
          for (int round = 0; round < 2; ++round) {
            // The stack-partner scan lives inside round 1, so it is skipped whenever round 1 is.
            if (round == 1 && (otSource_.nOTHits == 0u || shMergedHit != 0))
              break;  // no OT source, or the merged round already attached here
            // source-scoped raw-OT veto: skip the raw-OT round on the impure layer classes (TOB4-6
            // CA 31-33 / TID CA 34-53). The merged-stub round 0 already ran on this layer and is
            // untouched; only the low-purity raw-OT source is withheld here.
            if (round == 1 && ((extRawOTVetoTOB456 && shCurrentL >= 31 && shCurrentL <= 33) ||
                               (extRawOTVetoTID && shCurrentL >= 34 && shCurrentL <= 53))) {
              if (candDump_)  // dump: record that the raw-OT round was withheld on this layer
                shDumpVeto = 1;
              break;
            }

            // parallel hit scan on shCurrentL: each lane scans a round-robin subset of the phi window
            // and keeps its local best (min chi2, ties by min hitId). Round 0 reads the merged binner /
            // SoA; round 1 reads the OT binner / SoA (same window, same gate, same KF update).
            for (auto element : cms::alpakatools::uniform_group_elements(acc, j, nC * nLanes)) {
              const uint32_t lane = element.local;
              const int L = shCurrentL;
              const bool isBarrel = caLayers.isBarrel()[L];
              const float R = caLayers.layerR()[L];
              const float Z = caLayers.layerZ()[L];

              // Scale the TOB4-6 (CA 31-33) barrel chi2 cut by extChi2CutScaleTOB456. Multiplied into
              // baseChi2Cut, so it governs both the merged round (the bestChi2 seed) and the raw-OT
              // round (hitCut). Scale 1.0 leaves the cut at its base value.
              const float scaleTOB456 = (L >= 31 && L <= 33) ? extChi2CutScaleTOB456 : 1.f;
              // Independent TID endcap (CA 34-53) scale, on a layer range disjoint from TOB4-6.
              const float scaleTID = (L >= 34 && L <= 53) ? extChi2CutScaleTID : 1.f;
              // displacement-aware inner-OT gate: on a genuinely-displaced host (shHostDisplaced, computed
              // in setup), tighten the TOB1-3 (CA 28-30) accept window by kDispGateTOB13Scale. Multiplied
              // into baseChi2Cut so it governs both the merged round and the raw-OT round on those layers.
              // Prompt hosts / gate off => shHostDisplaced == 0 => scale 1. Outer layers (31-53) and
              // pixel layers are never affected.
              const float scaleTOB13 = (shHostDisplaced && L >= 28 && L <= 30) ? kDispGateTOB13Scale : 1.f;
              // forward-eta pocket gate: independent TOB1-3 tightening on a forward-eta host (shHostForwardPocket,
              // computed in setup). Mutually exclusive with dispgate by construction (shHostForwardPocket requires
              // !shHostDisplaced), so at most one of scaleTOB13/scalePocket is < 1 -- the product folds in
              // exactly one. Gate off / non-forward host / L outside 28-30 => scale 1.
              const float scalePocket = (shHostForwardPocket && L >= 28 && L <= 30) ? kPocketTOB13Scale : 1.f;
              // On pixel layers (CA L<28) extPixelGateChi2Cut replaces the base cut: a 2-dof gate at
              // chi2 = 2.0 is the 1-e^-1 = 63 % efficiency quantile, while the 95 % quantile is
              // -2*ln(0.05) ~= 6.0. OT layers keep chi2Cut/endcapChi2Cut; the OT-scoped scales are all
              // 1 on L<28, so on pixel this is exactly the override. <= 0 (sentinel) => pixel keeps
              // chi2Cut/endcapChi2Cut.
              const float pixBaseCut =
                  (extPixelGateChi2Cut > 0.f && L < 28) ? extPixelGateChi2Cut : (isBarrel ? chi2Cut : endcapChi2Cut);
              const float baseChi2Cut = pixBaseCut * scaleTOB456 * scaleTID * scaleTOB13 * scalePocket;

              // On the derived path the admission threshold is the measured quantile -- one number
              // replacing chi2Cut/endcapChi2Cut, the TOB4-6/TID region scales, the pixel gate cut and
              // the three covariance scales. (Pixel layers keep baseChi2Cut: every table is measured on
              // OT roads.) With two statistics live on the same layer -- 3-dof for stubs, 2-dof for
              // raw-OT and non-stub merged hits -- the argmin seed cannot double as the cut, so it seeds
              // at the wider map, no gate-passing candidate is pre-empted, and each round applies its
              // own threshold explicitly (the raw-OT round does so via hitCut).
              float bestChi2 =
                  shDerLayOn ? (shDerBendOn ? alpaka::math::max(acc, shDerQ, shDerQ3) : shDerQ) : baseChi2Cut;
              float bestDPhi = 0.f, bestDSec = 0.f;
              float bestSigPhi2Hit = 0.f, bestSigSec2Hit = 0.f;
              float bestDerSigR2 = 0.f, bestDerSigS2 = 0.f;  // full gate variances [cm^2]
              float bestDerRbb = 0.f;                        // the winner's R_bb (0 = no bend row)
              float bestRh = 0.f, bestZh = 0.f, bestArcS = 0.f;
              float bestJrCot = 0.f, bestJrZip = 0.f;  // endcap line-block H of the winner
              int32_t bestHit = -1;
              // candDump_ only: per-lane, per-round best over all considered road candidates
              // (passers and failers), min chi2 then min id. Combined into laneBestAny* after the scan.
              float bestAnyChi2 = 3.4e38f;
              int32_t bestAnyId = -1;
              int bestAnyPass = 0;

              const Prediction pred =
                  isBarrel ? predictOnBarrel(acc, sh.helix(), R) : predictOnEndcap(acc, sh.helix(), Z);
              if (pred.valid) {
                // Phi window over the CA per-layer phi histogram (identical derivation to the serial
                // version; every lane computes it redundantly from the shared helix + layer, so all
                // lanes agree on the bin range and hit-counter partition).
                constexpr float kMaxSigPhiHitCm = 0.1f;  // safe upper bound on any sensor's r-phi sigma (cm)
                // The derived window half-width sizes the phi-bin scan, which is also where its compute
                // saving comes from: the fixed-cut scan is always the 0.5 cm cap wide, while the derived
                // road is narrower on most visits and wider only where it must be. Bounded above by the
                // runaway ceiling so a pathological road cannot hold the launch.
                const float capRPhiEffMax = shDerLayOn ? alpaka::math::min(acc, shDerWinR, shDerCeilR)
                                                       : alpaka::math::max(acc, maxRPhiResidCm, 5.f * kMaxSigPhiHitCm);
                const float rMinLayer = alpaka::math::max(acc, R - caLayers.halfExtentR()[L], 1.f);
                float phiExtA = pred.phi, phiExtB = pred.phi;
                if (isBarrel) {
                  const Prediction pA = predictOnBarrel(acc, sh.helix(), rMinLayer);
                  const Prediction pB = predictOnBarrel(acc, sh.helix(), R + caLayers.halfExtentR()[L]);
                  phiExtA = pA.valid ? pA.phi : pred.phi;
                  phiExtB = pB.valid ? pB.phi : pred.phi;
                } else {
                  const Prediction pA = predictOnEndcap(acc, sh.helix(), Z - caLayers.halfExtentZ()[L]);
                  const Prediction pB = predictOnEndcap(acc, sh.helix(), Z + caLayers.halfExtentZ()[L]);
                  phiExtA = pA.valid ? pA.phi : pred.phi;
                  phiExtB = pB.valid ? pB.phi : pred.phi;
                }
                const float deltaMax = alpaka::math::max(acc,
                                                         alpaka::math::abs(acc, foldPi(phiExtA - pred.phi)),
                                                         alpaka::math::abs(acc, foldPi(phiExtB - pred.phi)));
                const float dPhiWin = capRPhiEffMax / rMinLayer + deltaMax;
                // the total phi half-window, before it becomes the iphi bin window (guard bins added
                // below).
                const float halfWin = dPhiWin;
                const float kToShort = 32768.f / kExtenderPi;  // short units per radian (matches hit iphi scale)
                constexpr int kGuardShorts = 3 * 256;          // ~3 phi bins for iphi quantization + atan approx
                const int iphicut = int(alpaka::math::round(acc, halfWin * kToShort)) + kGuardShorts;
                // The phi-bin window is centred on the energy-loss-corrected prediction: the window is
                // no wider than the road, so a mis-centred scan would drop the very hits the corrected
                // gate accepts. shDerElossPhi is 0 off the corrected path (adding an exact float zero).
                const int16_t mep = int16_t(alpaka::math::round(acc, foldPi(pred.phi + shDerElossPhi) * kToShort));
                const uint32_t hoff = ExtPhiBinner::histOff(uint32_t(L));
                const auto kl = ExtPhiBinner::bin(int16_t(mep - iphicut));
                auto khh = ExtPhiBinner::bin(int16_t(mep + iphicut));
                khh = (khh + 1) % ExtPhiBinner::nbins();
                uint32_t hitCounter = 0;  // running window-hit index; hit g is owned by lane (g % nLanes)
                // Round 0 only: the merged-source scan. In round 1 the loop condition is false at entry
                // (body never runs) so hitCounter stays 0 for the fresh OT pass below; a different lane
                // ownership partition than the merged pass, but the argmin (min chi2, tie by id) result is
                // partition-independent, so physics is unchanged.
                for (auto kk = kl; round == 0 && kk != khh; kk = (kk + 1) % ExtPhiBinner::nbins()) {
                  auto const* __restrict__ pbeg = phiBinner_->begin(kk + hoff);
                  auto const* __restrict__ pend = phiBinner_->end(kk + hoff);
                  // Per-bin strided ownership: each lane owns the hits g = lane (mod nLanes) and jumps
                  // straight to them, instead of traversing every window hit and masking off the ones it
                  // does not own. A uniform traversal would execute the gate body one owner lane at a
                  // time -- a warp-serial scan at ~1 active thread per warp -- whereas with the stride up
                  // to nLanes gates execute in the same warp instructions. The partition sets only which
                  // lane scores a hit; the (chi2, hitId) argmin over all of them is partition-independent.
                  const uint32_t nBin = uint32_t(pend - pbeg);
                  const uint32_t o0 = (nLanes + lane - (hitCounter % nLanes)) % nLanes;  // first owned in-bin offset
                  hitCounter += nBin;
                  for (uint32_t o = o0; o < nBin; o += nLanes) {
                    const uint32_t hitId = pbeg[o];
                    if (hitMaskArmed && hitMask[hitId].recHitMask() != 0u)
                      continue;
                    if (candDump_)  // dump: genuine (post-mask) merged candidate in the phi window
                      alpaka::atomicAdd(acc, &shDumpOccM, 1u, alpaka::hierarchy::Blocks{});
                    if (candDump_)  // hole counter: layer occupancy present
                      alpaka::atomicAdd(acc, &shHoleOcc, 1u, alpaka::hierarchy::Blocks{});
                    const float xh = hits[hitId].xGlobal();
                    const float yh = hits[hitId].yGlobal();
                    const float zh = hits[hitId].zGlobal();
                    const float rh = hits[hitId].rGlobal();
                    const float phiH = alpaka::math::atan2(acc, yh, xh);

                    // Per-hit crossing prediction: the linearized Taylor path (few FMAs) when this layer's
                    // coefficients were well-conditioned (shUseLin), else the exact per-hit closed form.
                    // Both feed the identical downstream gate through p2.phi/p2.secondary/p2.arcS.
                    Prediction p2;
                    if (shUseLin) {
                      const float u = isBarrel ? (rh - shLinRef) : (zh - shLinRef);
                      const float arcLin = shLinArc0 + shLinDarc1 * u;
                      bool okLin = (arcLin > 0.f && arcLin <= kExtenderMaxArcLengthCm);
                      if (isBarrel)
                        okLin = okLin && (rh >= shLinArgLo && rh <= shLinArgHi);
                      if (!okLin)
                        continue;
                      p2.phi = foldPi(shLinPhi0 + shLinDphi1 * u + 0.5f * shLinDphi2 * u * u);
                      p2.secondary = shLinSec0 + shLinDsec1 * u + 0.5f * shLinDsec2 * u * u;
                      p2.arcS = arcLin;
                      p2.valid = true;
                    } else {
                      p2 = isBarrel ? predictOnBarrel(acc, sh.helix(), rh) : predictOnEndcap(acc, sh.helix(), zh);
                      if (!p2.valid)
                        continue;
                    }

                    // Cheap residual pre-filter: skip the full gate for hits that provably fail the
                    // r-phi absolute-residual cap |dPhi*rh| < capRPhiEff, with
                    // capRPhiEff = max(maxRPhiResidCm, 5*sigPhiHit*rh). sigPhiHit would need the
                    // module-frame projection, not computed yet here, so sigPhiHit*rh is bounded from
                    // cheap inputs: with v=(yh,-xh) the tangential direction and G the global (x,y)
                    // hit covariance, frame.toGlobal is an orthonormal rotation of diag(xerr, yerr)
                    // (both variances), so v^T G v <= max(xerr, yerr)*rh^2 and
                    // sigPhiHit*rh <= sqrt(max(|xerr|,|yerr|)) plus the 1e-6 variance floor's
                    // sqrt(1e-6)*rh. Hence capRPhiEffUpperPF >= capRPhiEff for every hit and skipping
                    // a hit past this bound can never remove one the full gate would have accepted.
                    // The 1.001 guards against frame-projection float roundoff, and the per-hit
                    // prediction p2.phi is used, not the layer-nominal pred.phi.
                    const float dPhiPF = foldPi(p2.phi + shDerElossPhi - phiH);
                    const float xerrPF = hits[hitId].xerrLocal();
                    const float yerrPF = hits[hitId].yerrLocal();
                    const float sigPhiHitRhUB =
                        1.001f *
                        alpaka::math::max(
                            acc,
                            alpaka::math::sqrt(
                                acc,
                                alpaka::math::max(acc, alpaka::math::abs(acc, xerrPF), alpaka::math::abs(acc, yerrPF))),
                            alpaka::math::sqrt(acc, 1e-6f) * rh);
                    const float capRPhiEffUpperPF = shDerLayOn
                                                        ? alpaka::math::min(acc, shDerWinR, shDerCeilR)
                                                        : alpaka::math::max(acc, maxRPhiResidCm, 5.f * sigPhiHitRhUB);
                    if (alpaka::math::abs(acc, dPhiPF) * rh > capRPhiEffUpperPF)
                      continue;  // provably fails absResidualOk

                    const float secH = isBarrel ? zh : rh;
                    // road centre: the deterministic dE/dx offset of this crossing, formed once per
                    // visit in the lane-0 block and added to the prediction (both rows). It is a known
                    // bias, not an uncertainty, so it belongs here and not in any sigma. Exactly 0 with
                    // the fit corrections off.
                    const float dPhi = foldPi(p2.phi + shDerElossPhi - phiH);
                    const float dSec = p2.secondary + shDerElossSec - secH;
                    const float rh_safe = alpaka::math::max(acc, rh, 1.f);

                    // Hit resolution projected to (r*dphi, sec) through the module frame.
                    const float xerr = hits[hitId].xerrLocal();
                    const float yerr = hits[hitId].yerrLocal();
                    const auto detIdx = hits[hitId].detectorIndex();
                    const auto frame = caModules[detIdx].innerSensorFrame();
                    float ge[6];
                    frame.toGlobal(xerr, 0.f, yerr, ge);
                    const float rh2 = rh_safe * rh_safe;
                    const float gxx = ge[0], gxy = ge[1], gyy = ge[2], gzz = ge[5];
                    const float sigPhi2HitRaw = (yh * yh * gxx - 2.f * xh * yh * gxy + xh * xh * gyy) / (rh2 * rh2);
                    const float sigSec2HitRaw =
                        isBarrel ? gzz : (xh * xh * gxx + 2.f * xh * yh * gxy + yh * yh * gyy) / rh2;
                    const float sigPhi2Hit = alpaka::math::max(acc, sigPhi2HitRaw, 1e-6f);
                    const float sigSec2Hit = alpaka::math::max(acc, sigSec2HitRaw, 1e-4f);
                    const float sigPhiHit = alpaka::math::sqrt(acc, sigPhi2Hit);
                    const float sigSecHit = alpaka::math::sqrt(acc, sigSec2Hit);

                    // Incremental multiple scattering since the last accepted point.
                    const HelixState hForMs = sh.helix();
                    const float dArcSinceLast = alpaka::math::max(acc, p2.arcS - shLastArcS, 0.f);
                    const float msVar =
                        multScattVar(acc, shBf, hForMs, dArcSinceLast, rhoMap_, shLastR, shLastZ, rh_safe, zh);

                    // Propagated fit covariance at the prediction.
                    const float s = p2.arcS;
                    const float Jtip = -1.f / rh_safe;
                    const float JinvPt = -shBf * s * s / (2.f * rh_safe);
                    const float Jcot = s;
                    const float predPhiDiag = sh.vPhi + Jtip * Jtip * sh.vTip + JinvPt * JinvPt * sh.vPt;
                    const float predSecDiag_barrel = sh.vZip + Jcot * Jcot * sh.vCot;
                    const float predPhiCross =
                        2.f * Jtip * sh.cPhiTip + 2.f * JinvPt * sh.cPhiPt + 2.f * Jtip * JinvPt * sh.cTipPt;
                    const float predSecCross_barrel = 2.f * Jcot * sh.cCotZip;
                    const float predPhiVar = alpaka::math::max(acc, predPhiDiag + predPhiCross, 1e-12f);

                    float predSecVar;
                    float selJrCot = 0.f, selJrZip = 0.f;  // endcap dr/dcot, dr/dzip (barrel: unused)
                    if (isBarrel) {
                      predSecVar = alpaka::math::max(acc, predSecDiag_barrel + predSecCross_barrel, 1e-12f);
                    } else {
                      // Exact r-Jacobian at fixed z via forward-mode ad: rWithGrad5 runs the same
                      // closed-form predictOnEndcap r(params) in vector-dual arithmetic once, carrying
                      // all five partials dr/dparam simultaneously (transcendentals evaluated once, not
                      // five times). Analytic differentiation also avoids the 1/eps amplification a
                      // finite-difference Jacobian would apply to FMA-fusion roundoff.
                      float Jr[5];
                      rWithGrad5(acc, sh.phi0, sh.tip, sh.invPt, sh.cotTheta, sh.zip, zh, shBf, Jr);
                      const float Jr_phi = Jr[0];
                      const float Jr_tip = Jr[1];
                      const float Jr_pt = Jr[2];
                      const float Jr_cot = Jr[3];
                      const float Jr_zip = Jr[4];
                      selJrCot = Jr_cot;  // line-block H row for an r-at-fixed-z measurement
                      selJrZip = Jr_zip;
                      const float predSecCircleDiag =
                          Jr_phi * Jr_phi * sh.vPhi + Jr_tip * Jr_tip * sh.vTip + Jr_pt * Jr_pt * sh.vPt;
                      const float predSecCircleCross = 2.f * Jr_phi * Jr_tip * sh.cPhiTip +
                                                       2.f * Jr_phi * Jr_pt * sh.cPhiPt +
                                                       2.f * Jr_tip * Jr_pt * sh.cTipPt;
                      const float predSecLineDiag = Jr_cot * Jr_cot * sh.vCot + Jr_zip * Jr_zip * sh.vZip;
                      const float predSecLineCross = 2.f * Jr_cot * Jr_zip * sh.cCotZip;
                      predSecVar = alpaka::math::max(
                          acc, predSecCircleDiag + predSecCircleCross + predSecLineDiag + predSecLineCross, 1e-12f);
                    }

                    const float alignSigmaPhi2 = (alignSigmaPhiCm * alignSigmaPhiCm) / (rh_safe * rh_safe);
                    const float alignSigmaSec2 = alignSigmaSecCm * alignSigmaSecCm;
                    const float msPhi2 = msVar * dArcSinceLast * dArcSinceLast / (rh_safe * rh_safe);
                    // Covariance recalibration: scale the propagated fit covariance (predPhiVar,
                    // predSecVar) entering the 2-dof gate by the per-layer-class factor -- pixel (merged
                    // round, CA L<28) vs stub (merged round, CA L>=28). The pixel propagated covariance
                    // under-estimates the tail, so an unscaled gate rejects a large share of the pixel
                    // hits the CKF keeps. Applied only to the propagated part; the intrinsic hit + MS +
                    // align terms are untouched, and bestSigPhi2Hit below stays the raw hit+MS sigma.
                    const float covScale = (L < 28) ? extCovScalePixel : extCovScaleStub;
                    const float sigPhi2 = sigPhi2Hit + msPhi2 + predPhiVar * covScale + alignSigmaPhi2;
                    const float sigSec2 =
                        sigSec2Hit + msVar * dArcSinceLast * dArcSinceLast + predSecVar * covScale + alignSigmaSec2;
                    // The derived variances are their own consts and are never assigned into
                    // sigPhi2 / sigSec2: making those mutable would cost the compiler the FMA
                    // contraction it takes on the expressions above, moving the fixed-cut arm's
                    // arithmetic at the 1e-6 level even on hits the derived arm never scores.
                    // sigma_R and sigma_S are built in centimetres, the frame the tables are measured
                    // in; the r-phi one is converted back to rad^2 so it is consumed like sigPhi2.
                    float sigPhi2Der = 0.f, sigSec2Der = 0.f;
                    if (shDerLayOn) {
                      const float dVc = extDV[extMatClass(L, isBarrel, rh_safe)];
                      const float sR2cm = shDerSigR2 + sigPhi2Hit * rh2 + alignSigmaPhiCm * alignSigmaPhiCm + dVc;
                      const float sS2cm = shDerSigS2 + predSecVar + sigSec2Hit + alignSigmaSec2 + dVc;
                      sigPhi2Der = alpaka::math::max(acc, sR2cm / rh2, 1e-12f);
                      sigSec2Der = alpaka::math::max(acc, sS2cm, 1e-12f);
                    }
                    // Third row: a stub carries an independent local direction measurement, entered
                    // here as a row of the same chi2 with the track-side prediction uncertainty and
                    // the scattering in the denominator, cut and ranked at the same eps, unlike the
                    // standalone nsigma veto of extStubBendGate whose denominator is the measurement
                    // error alone. sigma_b is the two-cluster precision error times the per-class
                    // excess (extSigBExcess), the two being inseparable since in endcap PS the coarser
                    // dPhiDrError is only right because its inflation cancels that excess. Non-stub
                    // candidates keep the 2-dof statistic. The position<->bend correlation is not
                    // carried: the rows entered here are dominated by their propagated and MS shares,
                    // so it is small, and neglecting it leaves E[X^2] at d.
                    float pb2 = 0.f;
                    float rbbCand = 0.f;
                    bool bendRow = false;
                    if (shDerBendOn && ::reco::isStub(hits, int32_t(hitId))) {
                      const float sPrec = hits[hitId].dPhiDrErrorPrec();
                      if (sPrec > 0.f) {
                        const float sb = sPrec * extSigBExcess[extSigBClass(hits[hitId].stubFlags())];
                        const float rbb = sb * sb + shDerRbbTrk;
                        if (rbb > 0.f) {
                          const float pb = (hits[hitId].dPhiDr() - shDerPredB) / alpaka::math::sqrt(acc, rbb);
                          pb2 = pb * pb;
                          rbbCand = rbb;
                          bendRow = true;
                        }
                      }
                    }
                    const float qDer = bendRow ? shDerQ3 : shDerQ;

                    const float chi2 = shDerLayOn ? ((dPhi * dPhi) / sigPhi2Der + (dSec * dSec) / sigSec2Der + pb2)
                                                  : ((dPhi * dPhi) / sigPhi2 + (dSec * dSec) / sigSec2);
                    if (candDump_)  // dump: min gate chi2 over considered merged hits (best fail)
                      alpaka::atomicMin(acc,
                                        &shDumpBestM,
                                        chi2 >= 4.29e6f ? 0xFFFFFFFEu : uint32_t(chi2 * 1000.f + 0.5f),
                                        alpaka::hierarchy::Blocks{});
                    // Permil decomposition of the endcap gate variance (merged round).
                    if (secFracDiag_ && !isBarrel) {
                      const float pml = 1000.f / sigSec2;
                      const float msSec = msVar * dArcSinceLast * dArcSinceLast;
                      alpaka::atomicAdd(
                          acc, &stats[kDiagSecFracHitM], uint32_t(sigSec2Hit * pml + 0.5f), alpaka::hierarchy::Grids{});
                      alpaka::atomicAdd(
                          acc, &stats[kDiagSecFracMsM], uint32_t(msSec * pml + 0.5f), alpaka::hierarchy::Grids{});
                      alpaka::atomicAdd(acc,
                                        &stats[kDiagSecFracPredM],
                                        uint32_t(predSecVar * pml + 0.5f),
                                        alpaka::hierarchy::Grids{});
                      alpaka::atomicAdd(acc,
                                        &stats[kDiagSecFracAlignM],
                                        uint32_t(alignSigmaSec2 * pml + 0.5f),
                                        alpaka::hierarchy::Grids{});
                      alpaka::atomicAdd(acc, &stats[kDiagSecNM], 1u, alpaka::hierarchy::Grids{});
                      // Split the block-diagonal predSecVar into circle/line blocks and project the
                      // exact full-5x5 shadow covariance -> predSecVar_full (same sigSec2 denominator, so
                      // the two are directly comparable), bucketed by whether a prior disk hit was
                      // already folded in.
                      float JrF[5];
                      rWithGrad5(acc, sh.phi0, sh.tip, sh.invPt, sh.cotTheta, sh.zip, zh, shBf, JrF);
                      const float circP = JrF[0] * JrF[0] * sh.vPhi + JrF[1] * JrF[1] * sh.vTip +
                                          JrF[2] * JrF[2] * sh.vPt + 2.f * JrF[0] * JrF[1] * sh.cPhiTip +
                                          2.f * JrF[0] * JrF[2] * sh.cPhiPt + 2.f * JrF[1] * JrF[2] * sh.cTipPt;
                      const float lineP =
                          JrF[3] * JrF[3] * sh.vCot + JrF[4] * JrF[4] * sh.vZip + 2.f * JrF[3] * JrF[4] * sh.cCotZip;
                      const float predFull = alpaka::math::max(acc, projectCov(shC, JrF), 0.f);
                      alpaka::atomicAdd(acc,
                                        &stats[kDiagPredCircleM],
                                        uint32_t(alpaka::math::max(acc, circP, 0.f) * pml + 0.5f),
                                        alpaka::hierarchy::Grids{});
                      alpaka::atomicAdd(acc,
                                        &stats[kDiagPredLineM],
                                        uint32_t(alpaka::math::max(acc, lineP, 0.f) * pml + 0.5f),
                                        alpaka::hierarchy::Grids{});
                      alpaka::atomicAdd(
                          acc, &stats[kDiagPredFullM], uint32_t(predFull * pml + 0.5f), alpaka::hierarchy::Grids{});
                      const int slN = (shNDiskAcc == 0) ? kDiagNDisk0M : kDiagNDisk1M;
                      const int slBd = (shNDiskAcc == 0) ? kDiagPredBd0M : kDiagPredBd1M;
                      const int slFull = (shNDiskAcc == 0) ? kDiagPredFull0M : kDiagPredFull1M;
                      alpaka::atomicAdd(acc, &stats[slN], 1u, alpaka::hierarchy::Grids{});
                      alpaka::atomicAdd(
                          acc, &stats[slBd], uint32_t(predSecVar * pml + 0.5f), alpaka::hierarchy::Grids{});
                      alpaka::atomicAdd(
                          acc, &stats[slFull], uint32_t(predFull * pml + 0.5f), alpaka::hierarchy::Grids{});
                    }

                    // Absolute residual caps on top of the chi2 gate.
                    const float dPhiR = dPhi * rh;
                    constexpr float kSigmaCapMult = 5.f;
                    const float sigPhiHitCm = sigPhiHit * rh;
                    const float capRPhiEff = alpaka::math::max(acc, maxRPhiResidCm, kSigmaCapMult * sigPhiHitCm);
                    const float capSecEff = isBarrel ? alpaka::math::max(acc, maxSecResidCm, kSigmaCapMult * sigSecHit)
                                                     : endcapMaxSecResidCm;
                    // on the derived path the cm caps act as runaway ceilings only
                    // The selector here is the measured quantile ball; the ceilings are the per-layer
                    // module-envelope bounds computed once per visit (shDerCeilR / shDerCeilS, derived in the
                    // per-layer block), and every time one of them rejects a hit the ball admitted, that is
                    // counted -- a ceiling that binds often means the delivered efficiency is not the stated
                    // eps. The bounding half-widths sqrt(Q*sigma) follow from the same single eps and are not a
                    // second efficiency choice.
                    bool ballOk = true;
                    bool ceilOk = true;
                    if (shDerLayOn) {
                      // The bounding box is the box of this candidate's own ball.
                      const float wR = alpaka::math::sqrt(acc, qDer * sigPhi2Der) * rh;
                      const float wS = alpaka::math::sqrt(acc, qDer * sigSec2Der);
                      ballOk = (alpaka::math::abs(acc, dPhiR) < wR) && (alpaka::math::abs(acc, dSec) < wS);
                      const bool okR = alpaka::math::abs(acc, dPhiR) < shDerCeilR;
                      const bool okS = alpaka::math::abs(acc, dSec) < shDerCeilS;
                      ceilOk = okR && okS;
                      if (ballOk && !okR)
                        alpaka::atomicAdd(acc, &stats[kStatDerCapR], 1u, alpaka::hierarchy::Grids{});
                      if (ballOk && !okS)
                        alpaka::atomicAdd(acc, &stats[kStatDerCapS], 1u, alpaka::hierarchy::Grids{});
                    }
                    const bool absResidualOk = shDerLayOn ? (ballOk && ceilOk)
                                                          : ((alpaka::math::abs(acc, dPhiR) < capRPhiEff) &&
                                                             (alpaka::math::abs(acc, dSec) < capSecEff));

                    if (candDump_ && chi2 < baseChi2Cut && absResidualOk)  // dump: merged gate passer
                      alpaka::atomicAdd(acc, &shDumpPassM, 1u, alpaka::hierarchy::Blocks{});
                    // far-first window-ambiguity condition: on a far crossing of an armed host, count
                    // the candidates that pass this gate -- the exact set the argmin reduce below
                    // arbitrates, which is why a count of 1 makes an argmin error impossible. The same
                    // predicate the dump line above uses, so the two counts agree wherever both are on.
                    // shFarLayer is 0 on every layer the ordering does not touch, so the branch is then
                    // never taken.
                    if (shFarLayer && chi2 < baseChi2Cut && absResidualOk)
                      alpaka::atomicAdd(acc, &shFarPass, 1u, alpaka::hierarchy::Blocks{});
                    // This lane's best over all considered merged hits (min chi2, tie min id).
                    if (candDump_ && ((chi2 < bestAnyChi2) || (chi2 == bestAnyChi2 && int32_t(hitId) < bestAnyId))) {
                      bestAnyChi2 = chi2;
                      bestAnyId = int32_t(hitId);
                      bestAnyPass = ((chi2 < baseChi2Cut) && absResidualOk) ? 1 : 0;
                    }
                    // Record this merged road candidate (id + gate chi2 + pass) for the offline join.
                    // One member per scored road candidate; slot m from the block cursor;
                    // vi = shWalkSteps-1 (stable during the scan). candMemberBuf_ null => no write.
                    if (candDump_ && candMemberBuf_ != nullptr) {
                      const int viM = shWalkSteps - 1;
                      if (viM >= 0 && uint32_t(viM) < uint32_t(maxWalkLayers) && j < maxCandidates) {
                        const uint32_t m = alpaka::atomicAdd(acc, &shDumpMemN, 1u, alpaka::hierarchy::Blocks{});
                        if (m < kExtDumpMaxMembers) {
                          ExtCandMemberRec mr;
                          mr.hitId = int32_t(hitId);
                          mr.chi2 = chi2;
                          mr.round = int16_t(0);
                          mr.pass = int16_t((chi2 < baseChi2Cut && absResidualOk) ? 1 : 0);
                          candMemberBuf_[(uint32_t(j) * uint32_t(maxWalkLayers) + uint32_t(viM)) * kExtDumpMaxMembers +
                                         m] = mr;
                        } else {
                          alpaka::atomicAdd(acc, &candDumpOvf_[1], 1u, alpaka::hierarchy::Grids{});
                        }
                      }
                    }
                    // 2S stub-bend acceptance term: veto a merged stub whose measured local bend
                    // dPhiDr is incompatible with the track's local curvature expectation by more than
                    // extStubBendGate nSigma, on the same kappa significance the CA build's
                    // pixel-to-stub consistency cut uses, kappa = dPhiDr / sqrt(1 + r^2 dPhiDr^2) and
                    // sigma(kappa) = dPhiDrError / (den^1.5). kappa_track comes from the linearized
                    // crossing derivative (barrel shLinDphi1; endcap shLinDphi1/shLinDsec1). Correct
                    // 2S winners sit at a bend significance of order 1, wrong ones an order of
                    // magnitude higher; PS stubs carry a large dPhiDrError, so the term is small or
                    // skipped there. Merged round 0 only, and a sentinel <= 0 disables it. It is also
                    // not applied where the bend row is live: that row prices the same measurement,
                    // and the veto divides by the measurement error alone, which alongside the
                    // corrected sigma_b would cost a large fraction of the correct attachments.
                    bool bendOk = true;
                    if (!bendRow && extStubBendGate > 0.f && shLinValid && ::reco::isStub(hits, int32_t(hitId))) {
                      const float dS = hits[hitId].dPhiDr();
                      const float sS = hits[hitId].dPhiDrError();
                      if (sS > 0.f) {
                        const float dphidrTrk =
                            isBarrel ? shLinDphi1 : (shLinDsec1 != 0.f ? shLinDphi1 / shLinDsec1 : 0.f);
                        const float denS = 1.f + rh * rh * dS * dS;
                        const float sqrtDenS = alpaka::math::sqrt(acc, denS);
                        const float kS = dS / sqrtDenS;
                        const float skS = sS / (denS * sqrtDenS);
                        const float denT = 1.f + rh * rh * dphidrTrk * dphidrTrk;
                        const float kT = dphidrTrk / alpaka::math::sqrt(acc, denT);
                        const float bendSig = alpaka::math::abs(acc, kT - kS) / skS;
                        bendOk = (bendSig <= extStubBendGate);
                      }
                    }
                    const bool isBetter = (chi2 < bestChi2) || (chi2 == bestChi2 && int32_t(hitId) < bestHit);
                    // The merged round's admission threshold is explicit, because the argmin seed above
                    // is the wider of the two maps. Off the derived path this term is identically true
                    // and the baseChi2Cut seed does the cutting.
                    const bool gateOk = shDerLayOn ? (chi2 < qDer) : true;
                    if (isBetter && gateOk && absResidualOk && bendOk) {
                      bestChi2 = chi2;
                      bestHit = int32_t(hitId);
                      bestDPhi = dPhi;
                      bestDSec = dSec;
                      bestSigPhi2Hit = sigPhi2Hit + msPhi2;
                      bestSigSec2Hit = sigSec2Hit + msVar * dArcSinceLast * dArcSinceLast;
                      bestDerSigR2 = sigPhi2Der * rh2;  // full derived r-phi variance [cm^2]
                      bestDerSigS2 = sigSec2Der;
                      bestDerRbb = rbbCand;  // 0 when the candidate carries no bend row
                      bestRh = rh;
                      bestZh = zh;
                      bestArcS = s;
                      bestJrCot = selJrCot;
                      bestJrZip = selJrZip;
                    }
                  }
                }

                // Round 1: raw OT rechits, on the same layer visit, phi window and per-layer
                // linearization coefficients as the merged scan, reached only when round 0 attached
                // nothing here and the OT source is active, so an OT hit adds on a stub-less layer
                // instead of competing with a merged attachment. hitCounter restarts at 0, which only
                // re-partitions lane ownership. The gate is the merged path's, with the same
                // toGlobal(variance) convention.
                if (round == 1) {
                  const auto& otHits = otSource_.otHits;
                  for (auto kk = kl; kk != khh; kk = (kk + 1) % ExtPhiBinner::nbins()) {
                    auto const* __restrict__ pbeg = otSource_.phiBinner->begin(kk + hoff);
                    auto const* __restrict__ pend = otSource_.phiBinner->end(kk + hoff);
                    const uint32_t nBin = uint32_t(pend - pbeg);
                    const uint32_t o0 = (nLanes + lane - (hitCounter % nLanes)) % nLanes;  // first owned in-bin offset
                    hitCounter += nBin;
                    for (uint32_t oo = o0; oo < nBin; oo += nLanes) {
                      const uint32_t o = pbeg[oo];
                      if (otSource_.usedInStub[o] || (otSource_.ownership != nullptr && otSource_.ownership[o] != 0u)) {
                        continue;  // stub member or already owned
                      }
                      if (candDump_)  // dump: genuine (post-mask) raw-OT candidate in the phi window
                        alpaka::atomicAdd(acc, &shDumpOccO, 1u, alpaka::hierarchy::Blocks{});
                      if (candDump_)  // hole counter: layer occupancy present
                        alpaka::atomicAdd(acc, &shHoleOcc, 1u, alpaka::hierarchy::Blocks{});
                      const float xh = otHits[o].xGlobal();
                      const float yh = otHits[o].yGlobal();
                      const float zh = otHits[o].zGlobal();
                      const float rh = alpaka::math::sqrt(acc, xh * xh + yh * yh);  // rGlobal not stored in OT SoA
                      const float phiH = alpaka::math::atan2(acc, yh, xh);

                      // Per-hit crossing prediction: the same linearized Taylor path / exact fallback as merged.
                      Prediction p2;
                      if (shUseLin) {
                        const float u = isBarrel ? (rh - shLinRef) : (zh - shLinRef);
                        const float arcLin = shLinArc0 + shLinDarc1 * u;
                        bool okLin = (arcLin > 0.f && arcLin <= kExtenderMaxArcLengthCm);
                        if (isBarrel)
                          okLin = okLin && (rh >= shLinArgLo && rh <= shLinArgHi);
                        if (!okLin)
                          continue;
                        p2.phi = foldPi(shLinPhi0 + shLinDphi1 * u + 0.5f * shLinDphi2 * u * u);
                        p2.secondary = shLinSec0 + shLinDsec1 * u + 0.5f * shLinDsec2 * u * u;
                        p2.arcS = arcLin;
                        p2.valid = true;
                      } else {
                        p2 = isBarrel ? predictOnBarrel(acc, sh.helix(), rh) : predictOnEndcap(acc, sh.helix(), zh);
                        if (!p2.valid)
                          continue;
                      }

                      // Cheap r-phi residual pre-filter (same bound as merged; OT xerr/yerr).
                      const float dPhiPF = foldPi(p2.phi + shDerElossPhi - phiH);
                      const float xerrPF = otHits[o].xerrLocal();
                      const float yerrPF = otHits[o].yerrLocal();
                      const float sigPhiHitRhUB =
                          1.001f *
                          alpaka::math::max(acc,
                                            alpaka::math::sqrt(acc,
                                                               alpaka::math::max(acc,
                                                                                 alpaka::math::abs(acc, xerrPF),
                                                                                 alpaka::math::abs(acc, yerrPF))),
                                            alpaka::math::sqrt(acc, 1e-6f) * rh);
                      const float capRPhiEffUpperPF = shDerLayOn
                                                          ? alpaka::math::min(acc, shDerWinR, shDerCeilR)
                                                          : alpaka::math::max(acc, maxRPhiResidCm, 5.f * sigPhiHitRhUB);
                      if (alpaka::math::abs(acc, dPhiPF) * rh > capRPhiEffUpperPF)
                        continue;  // provably fails absResidualOk

                      const float secH = isBarrel ? zh : rh;
                      // Same road centre as the merged round (same layer visit, same crossing).
                      const float dPhi = foldPi(p2.phi + shDerElossPhi - phiH);
                      const float dSec = p2.secondary + shDerElossSec - secH;
                      const float rh_safe = alpaka::math::max(acc, rh, 1.f);

                      // Hit resolution projected to (r*dphi, sec) through the OT sensor frame (lower/upper per
                      // the hit's position within the stack). same toGlobal(variance) convention as merged
                      // (xerr/yerrLocal are variances, passed unsquared).
                      const float xerr = otHits[o].xerrLocal();
                      const float yerr = otHits[o].yerrLocal();
                      const uint32_t geomIdx = uint32_t(otHits[o].detectorIndex()) - ::phase2PixelTopology::nModulesPix;
                      const bool isUpper = (o >= otSource_.otHitModules.upperSensorStart()[geomIdx]);
                      const auto& frame = isUpper ? otSource_.stackedGeometry.upperSensorFrame()[geomIdx]
                                                  : otSource_.stackedGeometry.lowerSensorFrame()[geomIdx];
                      float ge[6];
                      frame.toGlobal(xerr, 0.f, yerr, ge);
                      const float rh2 = rh_safe * rh_safe;
                      const float gxx = ge[0], gxy = ge[1], gyy = ge[2], gzz = ge[5];
                      const float sigPhi2HitRaw = (yh * yh * gxx - 2.f * xh * yh * gxy + xh * xh * gyy) / (rh2 * rh2);
                      const float sigSec2HitRaw =
                          isBarrel ? gzz : (xh * xh * gxx + 2.f * xh * yh * gxy + yh * yh * gyy) / rh2;
                      const float sigPhi2Hit = alpaka::math::max(acc, sigPhi2HitRaw, 1e-6f);
                      const float sigSec2Hit = alpaka::math::max(acc, sigSec2HitRaw, 1e-4f);
                      const float sigPhiHit = alpaka::math::sqrt(acc, sigPhi2Hit);
                      const float sigSecHit = alpaka::math::sqrt(acc, sigSec2Hit);

                      // Incremental multiple scattering since the last accepted point.
                      const HelixState hForMs = sh.helix();
                      const float dArcSinceLast = alpaka::math::max(acc, p2.arcS - shLastArcS, 0.f);
                      const float msVar =
                          multScattVar(acc, shBf, hForMs, dArcSinceLast, rhoMap_, shLastR, shLastZ, rh_safe, zh);

                      // Propagated fit covariance at the prediction.
                      const float s = p2.arcS;
                      const float Jtip = -1.f / rh_safe;
                      const float JinvPt = -shBf * s * s / (2.f * rh_safe);
                      const float Jcot = s;
                      const float predPhiDiag = sh.vPhi + Jtip * Jtip * sh.vTip + JinvPt * JinvPt * sh.vPt;
                      const float predSecDiag_barrel = sh.vZip + Jcot * Jcot * sh.vCot;
                      const float predPhiCross =
                          2.f * Jtip * sh.cPhiTip + 2.f * JinvPt * sh.cPhiPt + 2.f * Jtip * JinvPt * sh.cTipPt;
                      const float predSecCross_barrel = 2.f * Jcot * sh.cCotZip;
                      const float predPhiVar = alpaka::math::max(acc, predPhiDiag + predPhiCross, 1e-12f);

                      float predSecVar;
                      float selJrCot = 0.f, selJrZip = 0.f;
                      if (isBarrel) {
                        predSecVar = alpaka::math::max(acc, predSecDiag_barrel + predSecCross_barrel, 1e-12f);
                      } else {
                        float Jr[5];
                        rWithGrad5(acc, sh.phi0, sh.tip, sh.invPt, sh.cotTheta, sh.zip, zh, shBf, Jr);
                        const float Jr_phi = Jr[0];
                        const float Jr_tip = Jr[1];
                        const float Jr_pt = Jr[2];
                        const float Jr_cot = Jr[3];
                        const float Jr_zip = Jr[4];
                        selJrCot = Jr_cot;
                        selJrZip = Jr_zip;
                        const float predSecCircleDiag =
                            Jr_phi * Jr_phi * sh.vPhi + Jr_tip * Jr_tip * sh.vTip + Jr_pt * Jr_pt * sh.vPt;
                        const float predSecCircleCross = 2.f * Jr_phi * Jr_tip * sh.cPhiTip +
                                                         2.f * Jr_phi * Jr_pt * sh.cPhiPt +
                                                         2.f * Jr_tip * Jr_pt * sh.cTipPt;
                        const float predSecLineDiag = Jr_cot * Jr_cot * sh.vCot + Jr_zip * Jr_zip * sh.vZip;
                        const float predSecLineCross = 2.f * Jr_cot * Jr_zip * sh.cCotZip;
                        predSecVar = alpaka::math::max(
                            acc, predSecCircleDiag + predSecCircleCross + predSecLineDiag + predSecLineCross, 1e-12f);
                      }

                      const float alignSigmaPhi2 = (alignSigmaPhiCm * alignSigmaPhiCm) / (rh_safe * rh_safe);
                      const float alignSigmaSec2 = alignSigmaSecCm * alignSigmaSecCm;
                      const float msPhi2 = msVar * dArcSinceLast * dArcSinceLast / (rh_safe * rh_safe);
                      // Covariance recalibration (raw-OT round 1): scale the propagated cov by the
                      // raw-OT class factor. Widening the OT classes does not pay, so this scale exists
                      // for symmetry with the pixel/stub ones; at 1.0 the propagated cov is unscaled.
                      const float covScale = extCovScaleRawOT;
                      const float sigPhi2 = sigPhi2Hit + msPhi2 + predPhiVar * covScale + alignSigmaPhi2;
                      const float sigSec2 =
                          sigSec2Hit + msVar * dArcSinceLast * dArcSinceLast + predSecVar * covScale + alignSigmaSec2;
                      // the derived road, as a separate pair of consts (same reason as in the merged round:
                      // a mutable destination would change the FMA contraction of the fixed-cut expressions above)
                      // sigma_R and sigma_S are built in centimetres (the frame the tables are measured in); the
                      // r-phi one is converted back to rad^2 so it is consumed exactly like sigPhi2.
                      float sigPhi2Der = 0.f, sigSec2Der = 0.f;
                      if (shDerLayOn) {
                        const float dVc = extDV[extMatClass(L, isBarrel, rh_safe)];
                        const float sR2cm = shDerSigR2 + sigPhi2Hit * rh2 + alignSigmaPhiCm * alignSigmaPhiCm + dVc;
                        const float sS2cm = shDerSigS2 + predSecVar + sigSec2Hit + alignSigmaSec2 + dVc;
                        sigPhi2Der = alpaka::math::max(acc, sR2cm / rh2, 1e-12f);
                        sigSec2Der = alpaka::math::max(acc, sS2cm, 1e-12f);
                      }

                      const float chi2 = shDerLayOn ? ((dPhi * dPhi) / sigPhi2Der + (dSec * dSec) / sigSec2Der)
                                                    : ((dPhi * dPhi) / sigPhi2 + (dSec * dSec) / sigSec2);
                      if (candDump_)  // dump: min gate chi2 over considered raw-OT hits (best fail)
                        alpaka::atomicMin(acc,
                                          &shDumpBestM,
                                          chi2 >= 4.29e6f ? 0xFFFFFFFEu : uint32_t(chi2 * 1000.f + 0.5f),
                                          alpaka::hierarchy::Blocks{});
                      // Permil decomposition of the endcap gate variance (raw-OT round).
                      if (secFracDiag_ && !isBarrel) {
                        const float pml = 1000.f / sigSec2;
                        const float msSec = msVar * dArcSinceLast * dArcSinceLast;
                        alpaka::atomicAdd(acc,
                                          &stats[kDiagSecFracHitOT],
                                          uint32_t(sigSec2Hit * pml + 0.5f),
                                          alpaka::hierarchy::Grids{});
                        alpaka::atomicAdd(
                            acc, &stats[kDiagSecFracMsOT], uint32_t(msSec * pml + 0.5f), alpaka::hierarchy::Grids{});
                        alpaka::atomicAdd(acc,
                                          &stats[kDiagSecFracPredOT],
                                          uint32_t(predSecVar * pml + 0.5f),
                                          alpaka::hierarchy::Grids{});
                        alpaka::atomicAdd(acc,
                                          &stats[kDiagSecFracAlignOT],
                                          uint32_t(alignSigmaSec2 * pml + 0.5f),
                                          alpaka::hierarchy::Grids{});
                        alpaka::atomicAdd(acc, &stats[kDiagSecNOT], 1u, alpaka::hierarchy::Grids{});
                      }

                      const float dPhiR = dPhi * rh;
                      constexpr float kSigmaCapMult = 5.f;
                      const float sigPhiHitCm = sigPhiHit * rh;
                      const float capRPhiEff = alpaka::math::max(acc, maxRPhiResidCm, kSigmaCapMult * sigPhiHitCm);
                      const float capSecEff = isBarrel
                                                  ? alpaka::math::max(acc, maxSecResidCm, kSigmaCapMult * sigSecHit)
                                                  : endcapMaxSecResidCm;
                      // on the derived path the cm caps act as runaway ceilings only (see the merged round):
                      // the quantile ball selects, the module-envelope ceilings only guard, and every ceiling
                      // rejection of a ball-passer is counted.
                      bool ballOk = true;
                      bool ceilOk = true;
                      if (shDerLayOn) {
                        const float wR = alpaka::math::sqrt(acc, shDerQ * sigPhi2Der) * rh;
                        const float wS = alpaka::math::sqrt(acc, shDerQ * sigSec2Der);
                        ballOk = (alpaka::math::abs(acc, dPhiR) < wR) && (alpaka::math::abs(acc, dSec) < wS);
                        const bool okR = alpaka::math::abs(acc, dPhiR) < shDerCeilR;
                        const bool okS = alpaka::math::abs(acc, dSec) < shDerCeilS;
                        ceilOk = okR && okS;
                        if (ballOk && !okR)
                          alpaka::atomicAdd(acc, &stats[kStatDerCapR], 1u, alpaka::hierarchy::Grids{});
                        if (ballOk && !okS)
                          alpaka::atomicAdd(acc, &stats[kStatDerCapS], 1u, alpaka::hierarchy::Grids{});
                      }
                      const bool absResidualOk = shDerLayOn ? (ballOk && ceilOk)
                                                            : ((alpaka::math::abs(acc, dPhiR) < capRPhiEff) &&
                                                               (alpaka::math::abs(acc, dSec) < capSecEff));

                      // Tagged OT candidate id for the argmin (bit30). Signed-int tie-break => an OT hit
                      // sorts after any merged hit at equal chi2, so merged wins ties deterministically.
                      const int32_t candId = int32_t(kOTHitTag | o);
                      const float hitCut = shDerLayOn ? shDerQ : baseChi2Cut;
                      if (candDump_ && chi2 < hitCut && absResidualOk)  // dump: raw-OT gate passer
                        alpaka::atomicAdd(acc, &shDumpPassO, 1u, alpaka::hierarchy::Blocks{});
                      // This lane's best over all considered raw-OT hits (min chi2, tie min id;
                      // candId carries the bit30 OT tag, so at equal chi2 a merged id sorts before an OT id).
                      if (candDump_ && ((chi2 < bestAnyChi2) || (chi2 == bestAnyChi2 && candId < bestAnyId))) {
                        bestAnyChi2 = chi2;
                        bestAnyId = candId;
                        bestAnyPass = ((chi2 < hitCut) && absResidualOk) ? 1 : 0;
                      }
                      // Record this raw-OT road candidate (bit30-tagged id + gate chi2 + pass) into the
                      // same per-(candidate,layer) member block as the merged round (shared cursor).
                      if (candDump_ && candMemberBuf_ != nullptr) {
                        const int viM = shWalkSteps - 1;
                        if (viM >= 0 && uint32_t(viM) < uint32_t(maxWalkLayers) && j < maxCandidates) {
                          const uint32_t m = alpaka::atomicAdd(acc, &shDumpMemN, 1u, alpaka::hierarchy::Blocks{});
                          if (m < kExtDumpMaxMembers) {
                            ExtCandMemberRec mr;
                            mr.hitId = candId;  // bit30-tagged raw-OT id
                            mr.chi2 = chi2;
                            mr.round = int16_t(1);
                            mr.pass = int16_t((chi2 < hitCut && absResidualOk) ? 1 : 0);
                            candMemberBuf_[(uint32_t(j) * uint32_t(maxWalkLayers) + uint32_t(viM)) * kExtDumpMaxMembers +
                                           m] = mr;
                          } else {
                            alpaka::atomicAdd(acc, &candDumpOvf_[1], 1u, alpaka::hierarchy::Grids{});
                          }
                        }
                      }
                      const bool isBetter = (chi2 < bestChi2) || (chi2 == bestChi2 && candId < bestHit);
                      if (isBetter && (chi2 < hitCut) && absResidualOk) {
                        bestChi2 = chi2;
                        bestHit = candId;
                        bestDPhi = dPhi;
                        bestDSec = dSec;
                        bestSigPhi2Hit = sigPhi2Hit + msPhi2;
                        bestSigSec2Hit = sigSec2Hit + msVar * dArcSinceLast * dArcSinceLast;
                        bestDerSigR2 = sigPhi2Der * rh2;
                        bestDerSigS2 = sigSec2Der;
                        bestDerRbb = 0.f;  // a raw-OT rechit is not a stub -- no bend row
                        bestRh = rh;
                        bestZh = zh;
                        bestArcS = s;
                        bestJrCot = selJrCot;
                        bestJrZip = selJrZip;
                      }
                    }
                  }
                }
              }
              laneChi2[lane] = bestChi2;
              laneHit[lane] = bestHit;
              // Min-combine this round's best-any into the lane's persistent slot (initialized to
              // the sentinel at layer-select; round 0 then round 1 accumulate). Each lane owns its slot -> no
              // race, no atomic. Tie by min id (an invalid stored slot, id -1, always loses to a valid id).
              if (candDump_ && bestAnyId >= 0 &&
                  ((bestAnyChi2 < laneBestAnyChi2[lane]) ||
                   (bestAnyChi2 == laneBestAnyChi2[lane] &&
                    (laneBestAnyHit[lane] < 0 || bestAnyId < laneBestAnyHit[lane])))) {
                laneBestAnyChi2[lane] = bestAnyChi2;
                laneBestAnyHit[lane] = bestAnyId;
                laneBestAnyPass[lane] = bestAnyPass;
              }
              laneWin[lane] = int(lane);  // seed the argmin tree's carried lane index
              laneDPhi[lane] = bestDPhi;
              laneDSec[lane] = bestDSec;
              laneSigPhi2[lane] = bestSigPhi2Hit;
              laneSigSec2[lane] = bestSigSec2Hit;
              laneDerSigR2[lane] = bestDerSigR2;
              laneDerSigS2[lane] = bestDerSigS2;
              laneDerRbb[lane] = bestDerRbb;
              laneRh[lane] = bestRh;
              laneZh[lane] = bestZh;
              laneArcS[lane] = bestArcS;
              laneJrCot[lane] = bestJrCot;
              laneJrZip[lane] = bestJrZip;
            }
            alpaka::syncBlockThreads(acc);

            // In-place shared-memory tree reduction over the per-lane bests. The key (chi2, then
            // hitId) is a strict total order, each lane's best id being distinct since merged hitIds
            // and bit30-tagged OT candIds never collide, so any reduction order yields the same
            // winner on every backend. laneWin carries the winner's original lane so lane 0 can index
            // its KF payload; the reduce touches only laneChi2/laneHit/laneWin. No threshold needs
            // pre-loading: an accepted lane always has laneChi2 < baseChi2Cut, isBetter requiring
            // chi2 < bestChi2 with bestChi2 seeded at the cut.
            for (uint32_t stride = nLanes >> 1; stride > 0u; stride >>= 1) {
              for (auto element : cms::alpakatools::uniform_group_elements(acc, j, nC * nLanes)) {
                const uint32_t lane = element.local;
                if (lane < stride) {
                  const uint32_t o = lane + stride;
                  const bool takeO =
                      (laneHit[lane] < 0) ||
                      (laneHit[o] >= 0 &&
                       (laneChi2[o] < laneChi2[lane] || (laneChi2[o] == laneChi2[lane] && laneHit[o] < laneHit[lane])));
                  if (takeO) {
                    laneChi2[lane] = laneChi2[o];
                    laneHit[lane] = laneHit[o];
                    laneWin[lane] = laneWin[o];
                  }
                }
              }
              alpaka::syncBlockThreads(acc);
            }

            // lane 0 records the extra + applies the Kalman update to the shared running helix, reading
            // the reduced global best (gHit / gChi2 / gLane) from slot 0.
            for (auto element : cms::alpakatools::uniform_group_elements(acc, j, nC * nLanes)) {
              if (element.local != 0u)
                continue;
              const int L = shCurrentL;
              const bool isBarrel = caLayers.isBarrel()[L];
              // The per-layer-class gate scales again, here only for the stack-partner threshold below;
              // the primary gate was applied per lane in the scan and the argmin tree needs no cut.
              const float scaleTOB456 = (L >= 31 && L <= 33) ? extChi2CutScaleTOB456 : 1.f;
              const float scaleTID = (L >= 34 && L <= 53) ? extChi2CutScaleTID : 1.f;  // TID endcap
              // dispgate: same TOB1-3 window tightening as the lane scan, so the partner threshold below
              // stays consistent with the primary gate on a displaced host. 1.f when off / prompt / L>30.
              const float scaleTOB13 = (shHostDisplaced && L >= 28 && L <= 30) ? kDispGateTOB13Scale : 1.f;
              // pocket gate: same forward-eta TOB1-3 tightening as the lane scan, kept consistent with the
              // primary gate. Exclusive with scaleTOB13 (see setup). 1.f when off / non-forward / L>30.
              const float scalePocket = (shHostForwardPocket && L >= 28 && L <= 30) ? kPocketTOB13Scale : 1.f;
              int32_t gHit = laneHit[0];        // -1 when no lane passed the gate
              const float gChi2 = laneChi2[0];  // winner chi2 (used only when gHit >= 0)
              // A gate-passer existed on this layer iff the argmin winner exists (laneHit[0] >= 0, read
              // before any cap mutation below). or-ed across both rounds, so shChainHasPass equals the
              // dump's nPassM + nPassOT > 0. Independent of the cap and the veto, because a continuation
              // is about a passer existing, not about it being committed. Maintained under candDump_ so
              // the hole counter (rejGate == occupancy && !pass) can consume it.
              if (candDump_ && laneHit[0] >= 0)
                shChainHasPass = 1;
              // Far-first window-ambiguity condition (AttachParams::extAttachFarMaxWin). The
              // far-first ordering gathers the outermost crossing whatever its purity, and here the
              // walk may decline one whose window it cannot arbitrate: shFarPass is the number of
              // candidates that cleared this crossing's gate, and above the limit the argmin is a
              // choice among competing hits rather than a measurement. Declining is exactly "no hit
              // on this layer": the layer is already charged to the K visit budget, the extra slot is
              // not spent and the walk carries on into the nearer interior discs, which keep the
              // unconditioned commit rule. Round 0 only, the merged round owning the pixel content.
              if (shFarLayer && round == 0 && gHit >= 0 && shFarPass > uint32_t(extAttachFarMaxWin)) {
                gHit = -1;
                alpaka::atomicAdd(acc, &stats[kStatAttachFarDecline], 1u, alpaka::hierarchy::Grids{});
              }
              // The hole hypothesis, "attach nothing on this layer", competes in the argmin: the
              // winner is committed only if it beats it. Its price comes from the measured per-layer
              // stub availability eta_L, the measured stub areal density rho and the window volume,
              //     chi2_hole = 2 ln[ P / ((1 - eta_L eps) nu) ],  nu = rho (2 pi) sqrt(|R|),
              // with the numerator P set at the layer site above and |R| = sigma_R^2 sigma_S^2 for
              // this 2-coordinate statistic. It is a competing hypothesis rather than a cut on the
              // hit, and introduces no tuned constant: the same eps and two measured detector tables.
              if (shDerLayOn && gHit >= 0 && (shDerHoleK > -1e29f || shDerHoleKRaw > -1e29f || shDerHoleK3 > -1e29f)) {
                alpaka::atomicAdd(acc, &stats[kStatDerHoleCand], 1u, alpaka::hierarchy::Grids{});
                // raw-channel monitor: the round-1 projection of the same counter. The raw round's own
                // decline rate is what says whether the hole actually arbitrates that channel, and it
                // must be readable without a truth dump.
                if (round == 1)
                  alpaka::atomicAdd(acc, &stats[kStatDerHoleCandRaw], 1u, alpaka::hierarchy::Grids{});
                const int wLane = laneWin[0];
                const float sR2 = (wLane >= 0) ? laneDerSigR2[wLane] : 0.f;
                const float sS2 = (wLane >= 0) ? laneDerSigS2[wLane] : 0.f;
                // |R| is the volume of the statistic the winner was actually judged with,
                // and the density is the one that lives in that same space -- for a stub winner the
                // 3-dof volume sigma_R^2 sigma_S^2 R_bb against rho_3 [cm^-1 rad^-1], for a raw-OT
                // winner the 2-dof branch, sigma_R^2 sigma_S^2 against rho_A [cm^-2]. Mixing
                // the two is a units error, not a rounding one.
                const float rbbW = (wLane >= 0) ? laneDerRbb[wLane] : 0.f;
                const bool win3 = (rbbW > 0.f);
                // The 2-dof price is keyed by source round: a round-1 winner is a raw cluster on a layer
                // whose stub round already came up empty, so it is priced with the raw round's own
                // conditional availability and cluster density (shDerHoleKRaw); a round-0 2-dof winner
                // (pixel / P-hit-only merged) keeps the stub rows. extHoleRawRoundPrior off leaves
                // shDerHoleKRaw at its sentinel and the single stub row prices both rounds.
                const float hole2 = (round == 1 && shDerHoleKRaw > -1e29f) ? shDerHoleKRaw : shDerHoleK;
                const float holeK = win3 ? shDerHoleK3 : hole2;
                if (sR2 > 0.f && sS2 > 0.f && holeK > -1e29f) {
                  const float detR = win3 ? (sR2 * sS2 * rbbW) : (sR2 * sS2);
                  const float chi2Hole = holeK - alpaka::math::log(acc, detR);
                  if (gChi2 >= chi2Hole) {
                    gHit = -1;  // the hole wins: this layer contributes no measurement
                    alpaka::atomicAdd(acc, &stats[kStatDerHoleFire], 1u, alpaka::hierarchy::Grids{});
                    if (round == 1)
                      alpaka::atomicAdd(acc, &stats[kStatDerHoleFireRaw], 1u, alpaka::hierarchy::Grids{});
                  }
                }
              }
              // MTV-aligned extra cap: this winner resolves to gExtraClusters output clusters (a merged
              // 2-hit stub -> 2 via reco::isStub; a raw-OT round-1 winner / pixel / P-hit-only -> 1). If
              // appending it would breach the per-track cluster budget, drop it (set gHit = -1) --
              // identical to "no hit on this layer", so the walk keeps the shorter, higher-purity track.
              // In round 0, dropping also leaves shMergedHit == 0, letting round 1 try a 1-cluster OT hit
              // that may still fit the budget. Cap off => shExtraClusterCap sentinel => never fires.
              int gExtraClusters = 0;
              if (gHit >= 0) {
                gExtraClusters = isOTId(uint32_t(gHit)) ? 1 : (::reco::isStub(hits, gHit) ? 2 : 1);
                // An OT-layer (CA >= 28) candidate on an anchored track (>= 1 prior accepted OT
                // extra) bypasses the cluster-cap drop, having passed the per-hit gate from the
                // anchored running state. shNOTExtraAcc counts prior-layer OT accepts only, so the
                // test is that a prior OT accept exists; unanchored tracks and pixel candidates keep
                // the guard. extCapExemptTOB46Only narrows the exempted set from L >= 28 to L in
                // 31-33, and extCapExemptMaxChi2 additionally requires the winner's gate chi2 to be
                // below it, so a high-chi2 tail is re-capped even inside TOB4-6.
                const bool capExemptLayerOk = extCapExemptTOB46Only ? (L >= 31 && L <= 33) : (L >= 28);
                const bool capExemptChi2Ok = (extCapExemptMaxChi2 <= 0.f) || (gChi2 < extCapExemptMaxChi2);
                const bool capAcceptExempt =
                    extCapExemptAnchored && shNOTExtraAcc >= 1 && capExemptLayerOk && capExemptChi2Ok;
                if (!capAcceptExempt && shNExtraClusters + gExtraClusters > shExtraClusterCap) {
                  // The over-budget winner is dropped here, in arrival order; the dump still records it
                  // with outcome rejCap.
                  gHit = -1;
                  if (candDump_)  // dump: gate-passing winner over the MTV cluster budget
                    shDumpCapDropped = 1;
                }
              }
              const int gLane = (gHit >= 0) ? laneWin[0] : -1;
              if (round == 0)
                shMergedHit = (gHit >= 0) ? 1 : 0;  // gate round 1: OT scans only where the merged round missed
              if (gHit >= 0) {
                // Capture the just-traversed gap's Highland msVar now (pre measurement-update
                // helix + the old last-accept point r/z -- exactly the quantity the scan folded into this
                // layer's window) for the post-downdate P += Q injection after recomputeHelix below. 0 when off.
                float msVarGapJ2 = 0.f;
                if (extStateProcessNoise > 0.f) {
                  const HelixState hMsGapJ2 =
                      sh.helix();  // pre-update state (the measurement downdate has not run yet)
                  const float dArcGapJ2 = alpaka::math::max(acc, laneArcS[gLane] - shLastArcS, 0.f);
                  msVarGapJ2 = multScattVar(acc,
                                            shBf,
                                            hMsGapJ2,
                                            dArcGapJ2,
                                            rhoMap_,
                                            shLastR,
                                            shLastZ,
                                            alpaka::math::max(acc, laneRh[gLane], 1.f),
                                            laneZh[gLane]);
                }
                extrasIds[j * maxExtraHitsPerTrack + shNExtra] = uint32_t(gHit);
                extrasChi2[j * maxExtraHitsPerTrack + shNExtra] = gChi2;
                ++shNExtra;
                if (candDump_) {  // dump: capture this layer's committed winner (round 0 or 1)
                  shDumpWinHit = gHit;
                  shDumpWinChi2 = gChi2;
                  shDumpWinRound = round;
                }
                shNExtraClusters += gExtraClusters;  // MTV-aligned cap: track appended clusters
                // Tally OT-layer (CA >= 28) accepts -- the anchor count the cluster-cap exemption reads
                // on subsequent layers. Written here, after the per-accept guard, so at that guard it
                // holds only prior-layer OT accepts. Read only under extCapExemptAnchored.
                if (L >= 28)
                  ++shNOTExtraAcc;
                // Diagnostic: split walk-committed extras by impurity layer class (TOB1-3 / TOB4-6 /
                // TID). Same cheap-atomic pattern as kDiagWalk*; read back only in the verbose summary.
                if (L >= 28 && L <= 30)
                  alpaka::atomicAdd(acc, &stats[kStatExtraTOB13], 1u, alpaka::hierarchy::Grids{});
                else if (L >= 31 && L <= 33)
                  alpaka::atomicAdd(acc, &stats[kStatExtraTOB456], 1u, alpaka::hierarchy::Grids{});
                else if (L >= 34 && L <= 53)
                  alpaka::atomicAdd(acc, &stats[kStatExtraTID], 1u, alpaka::hierarchy::Grids{});
                // raw-channel monitor: the same split restricted to round 1, i.e. the single-cluster
                // channel's own per-class yield -- the class counters above are round-blind and the round
                // counters below are class-blind, so neither alone reads it. Round-1 commits are a small
                // subset (round 1 runs only where round 0 attached nothing), so the merged round's hot
                // path is untouched.
                if (round == 1) {
                  if (L >= 28 && L <= 30)
                    alpaka::atomicAdd(acc, &stats[kStatRawTOB13], 1u, alpaka::hierarchy::Grids{});
                  else if (L >= 31 && L <= 33)
                    alpaka::atomicAdd(acc, &stats[kStatRawTOB456], 1u, alpaka::hierarchy::Grids{});
                  else if (L >= 34 && L <= 53)
                    alpaka::atomicAdd(acc, &stats[kStatRawTID], 1u, alpaka::hierarchy::Grids{});
                }
                // Split walk-committed extras by source (round 0 = merged, round 1 = raw OT).
                if (round == 1) {
                  alpaka::atomicAdd(acc, &stats[kDiagWalkOT], 1u, alpaka::hierarchy::Grids{});
                  shHasOT = 1;
                } else {
                  alpaka::atomicAdd(acc, &stats[kDiagWalkMerged], 1u, alpaka::hierarchy::Grids{});
                }
                if (secFracDiag_) {
                  // Fold this accept into the exact full-5x5 shadow cov, linearized on the pre-update
                  // state -- the same point at which the walk captured its residual and Jacobians.
                  const float rhs = alpaka::math::max(acc, laneRh[gLane], 1.f);
                  const float ss = laneArcS[gLane];
                  const float Hp[5] = {1.f, -1.f / rhs, -shBf * ss * ss / (2.f * rhs), 0.f, 0.f};
                  float Hs[5];
                  if (isBarrel) {
                    Hs[0] = 0.f;
                    Hs[1] = 0.f;
                    Hs[2] = 0.f;
                    Hs[3] = ss;
                    Hs[4] = 1.f;
                  } else {
                    rWithGrad5(acc, sh.phi0, sh.tip, sh.invPt, sh.cotTheta, sh.zip, laneZh[gLane], shBf, Hs);
                  }
                  const float R00 = laneSigPhi2[gLane] + (alignSigmaPhiCm * alignSigmaPhiCm) / (rhs * rhs);
                  const float R11 = laneSigSec2[gLane] + alignSigmaSecCm * alignSigmaSecCm;
                  updateShadowCov(acc, shC, Hp, Hs, R00, R11);
                  if (!isBarrel)
                    ++shNDiskAcc;
                }
                sh.updateCircleFromPhi(acc, laneDPhi[gLane], laneSigPhi2[gLane], laneRh[gLane], shBf, laneArcS[gLane]);
                // Line (cotTheta,zip) KF update on both regions: the barrel uses the exact H=(s,1) of
                // z=zip+s*cot, the endcap the r-at-fixed-z dual H=(dr/dcot, dr/dzip), captured with
                // the residual on the pre-update helix. Innovation sign: laneDSec = pred - meas and
                // the mean update x += K (meas - pred) is applied inside the helpers (state -=
                // K*dSec), so both line calls pass the raw +dSec residual, as the circle call does.
                if (isBarrel)
                  sh.updateLineFromSec(acc, laneDSec[gLane], laneSigSec2[gLane], laneArcS[gLane]);
                else
                  sh.updateLineFromSecH(acc, laneDSec[gLane], laneSigSec2[gLane], laneJrCot[gLane], laneJrZip[gLane]);
                sh.recomputeHelix(acc, shBf);
                // State process noise (P += Q): after the measurement downdate above, the direction
                // terms of P are re-inflated by the just-traversed gap's Highland scattering variance
                // (captured pre-update as msVarGapJ2). Circle block: vPhi += scale*msVar. Line block:
                // vCot += scale*msVar*(1+cot^2)^2, since d(cot)/d(theta) = -(1+cot^2) maps the
                // polar-angle scattering variance through that Jacobian squared. Curvature vPt is
                // untouched, scattering being elastic. Injected post-downdate so this layer's window
                // is not double-counted; it widens the next layer's propagated predPhiVar/predSecVar,
                // which keeps an anchored track from being gate-killed by an over-confident P.
                if (extStateProcessNoise > 0.f) {
                  const float cot2p1 = 1.f + sh.cotTheta * sh.cotTheta;
                  sh.vPhi += extStateProcessNoise * msVarGapJ2;
                  sh.vCot += extStateProcessNoise * msVarGapJ2 * cot2p1 * cot2p1;
                }
                // re-anchor the option-D band state at this accept
                // Exact 3x3 fold of (u, u', kappa): propagate to the accept's arc, add the traversed
                // gap's process noise from its own material moments -- the exact (W, S1, S2) split
                // rather than one equivalent kink, which would leave the late-visit road far too narrow
                // -- then update with the accepted r-phi measurement and move the anchor. Process noise
                // goes into the state per gap, never into the innovation per visit; that is also why the
                // material march runs once per visit.
                if (shDerOn) {
                  const float dsA = laneArcS[gLane] - shDerAnchorS;
                  float c00 = shDerCloc[0], c01 = shDerCloc[1], c02 = shDerCloc[2];
                  float c11 = shDerCloc[3], c12 = shDerCloc[4], c22 = shDerCloc[5];
                  // The last fitted gap's exit-direction variance is structurally invisible to the fit
                  // (varBeta(n-1) == 0); it enters the road explicitly until the first accept, and from
                  // here on it lives in the band like any other direction uncertainty.
                  if (shDerQgap > 0.f) {
                    c11 += shDerQgap;
                    shDerQgap = 0.f;
                  }
                  // F = [[1, ds, -ds^2/2], [0, 1, -ds], [0, 0, 1]]  (the same g the road consumes)
                  const float f02 = -0.5f * dsA * dsA;
                  const float n00 = c00 + 2.f * dsA * c01 + 2.f * f02 * c02 + dsA * dsA * c11 + 2.f * dsA * f02 * c12 +
                                    f02 * f02 * c22;
                  const float n01 = c01 + dsA * c11 + f02 * c12 - dsA * (c02 + dsA * c12 + f02 * c22);
                  const float n02 = c02 + dsA * c12 + f02 * c22;
                  const float n11 = c11 - 2.f * dsA * c12 + dsA * dsA * c22;
                  const float n12 = c12 - dsA * c22;
                  const float n22 = c22;
                  // Process noise of the traversed gap: the offset carries the 3-D lever,
                  // the angle rows carry 1/cos(lambda) per power of angle.
                  const float cotA = sh.cotTheta;
                  const float invCosL = alpaka::math::sqrt(acc, 1.f + cotA * cotA);
                  c00 = n00 + shDerC * shDerS2;
                  c01 = n01 + shDerC * shDerS1 * invCosL;
                  c02 = n02;
                  c11 = n11 + shDerC * shDerW * invCosL * invCosL;
                  c12 = n12;
                  c22 = n22;
                  // KF update with the accepted r-phi measurement, H = [1, 0, 0], R = its own error.
                  const float Rm = alpaka::math::max(
                      acc,
                      laneSigPhi2[gLane] * laneRh[gLane] * laneRh[gLane] + alignSigmaPhiCm * alignSigmaPhiCm,
                      1e-10f);
                  const float Sm = c00 + Rm;
                  if (Sm > 0.f) {
                    const float k0 = c00 / Sm, k1 = c01 / Sm, k2 = c02 / Sm;
                    shDerCloc[0] = c00 - k0 * c00;
                    shDerCloc[1] = c01 - k1 * c00;
                    shDerCloc[2] = c02 - k2 * c00;
                    shDerCloc[3] = c11 - k1 * c01;
                    shDerCloc[4] = c12 - k2 * c01;
                    shDerCloc[5] = c22 - k2 * c02;
                  } else {
                    shDerCloc[0] = c00;
                    shDerCloc[1] = c01;
                    shDerCloc[2] = c02;
                    shDerCloc[3] = c11;
                    shDerCloc[4] = c12;
                    shDerCloc[5] = c22;
                  }
                  // The energy-loss centre re-anchors with the band by the same recursion: the same
                  // F(ds) as the covariance above, applied to the deterministic triple rather than to
                  // a matrix, and the same gap moments (shDerW/S1/S2) the process noise just used, so
                  // the two states never disagree about how much material was traversed.
                  //   u      <- u + u'*ds - (dkappa*ds^2 + K*S2)/2    [an identity, not a truncation]
                  //   u'     <- u' - (dkappa*ds + K*S1)
                  //   dkappa <- dkappa + K*W
                  // It is not reset by the measurement update: the residual fed to the filter above
                  // is the corrected one, so the running helix tracks the constant-kappa_0 reference
                  // trajectory the fit published while these three carry the true track's growing
                  // offset from it.
                  if (shEK > 0.f) {
                    // Same transverse-metric conversion as the per-visit evaluation above: dsA is a
                    // transverse arc, shDerS1/S2 carry 3-D lever powers 1 and 2, so they take one and
                    // two powers of cos(lambda) respectively. shDerW is a bare material column with no
                    // lever in it and is metric-free, so dkappa's fold is unchanged.
                    const float cosLA = 1.f / invCosL;
                    const float uN = shEU + shEUp * dsA - 0.5f * (shEDk * dsA * dsA + shEK * shDerS2 * cosLA * cosLA);
                    const float upN = shEUp - (shEDk * dsA + shEK * shDerS1 * cosLA);
                    shEDk += shEK * shDerW;
                    shEU = uN;
                    shEUp = upN;
                  }
                  shDerAnchorS = laneArcS[gLane];
                  shDerAnchorR = alpaka::math::max(acc, laneRh[gLane], 1.f);
                  shDerAnchorZ = laneZh[gLane];
                }
                shLastArcS = laneArcS[gLane];
                shLastR = laneRh[gLane];
                shLastZ = laneZh[gLane];
                // Both-sensors attach on stub-less layers. The round-1 winner sits on one sensor of
                // an OT stack and the killed doublet's partner rechit typically lives on the other
                // sensor of the same module, so that partner sensor range is scanned against the
                // updated helix with the same OT gate math (fresh predict at the hit's rh/zh, same
                // chi2Cut and absResidualOk) and the best passing hit is accepted as a second extra
                // on this layer. Lane-0-serial over a few hits per module, round 1 only.
                // The partner is a second raw-OT extra of one cluster, so it needs room for one more
                // cluster on top of the primary already counted in shNExtraClusters, and being a
                // same-module OT extra on an anchored track it bypasses the cluster-cap check.
                // extCapExemptTOB46Only gates it on L in 31-33 as at every other exemption site;
                // extCapExemptMaxChi2 is not applied, since the partner's own gate chi2 does not
                // exist yet here and it rides a primary accept that was itself chi2-gated.
                const bool capPartnerExempt =
                    extCapExemptAnchored && shNOTExtraAcc >= 1 && (!extCapExemptTOB46Only || (L >= 31 && L <= 33));
                if (round == 1 && shNExtra < maxExtraHitsPerTrack &&
                    (capPartnerExempt || shNExtraClusters < shExtraClusterCap)) {
                  const auto& otHits = otSource_.otHits;
                  const uint32_t oWin = otIdx(uint32_t(gHit));
                  const uint32_t geomIdx = uint32_t(otHits[oWin].detectorIndex()) - ::phase2PixelTopology::nModulesPix;
                  const uint32_t upStart = otSource_.otHitModules.upperSensorStart()[geomIdx];
                  const bool winUpper = (oWin >= upStart);
                  // partner = the other sensor range of the same module (ranges from otHitModules)
                  const uint32_t pBeg = winUpper ? otSource_.otHitModules.moduleStart()[geomIdx] : upStart;
                  const uint32_t pEnd = winUpper ? upStart : otSource_.otHitModules.moduleStart()[geomIdx + 1u];
                  // The partner threshold carries the same per-layer-class scaling (scaleTOB456,
                  // scaleTID, scaleTOB13, scalePocket) as the per-lane gate.
                  const float bestPChi2Dep =
                      (isBarrel ? chi2Cut : endcapChi2Cut) * scaleTOB456 * scaleTID * scaleTOB13 * scalePocket;
                  // The partner rides the same single measured quantile as the primary gate.
                  float bestPChi2 = shDerLayOn ? shDerQ : bestPChi2Dep;
                  int32_t bestPHit = -1;
                  float bestPDPhi = 0.f, bestPDSec = 0.f, bestPSigPhi2 = 0.f, bestPSigSec2 = 0.f;
                  float bestPRh = 0.f, bestPZh = 0.f, bestPArcS = 0.f;
                  float bestPJrCot = 0.f, bestPJrZip = 0.f;
                  for (uint32_t p = pBeg; p < pEnd; ++p) {
                    if (otSource_.usedInStub[p] || (otSource_.ownership != nullptr && otSource_.ownership[p] != 0u))
                      continue;  // stub member or already owned
                    const float xh = otHits[p].xGlobal();
                    const float yh = otHits[p].yGlobal();
                    const float zh = otHits[p].zGlobal();
                    const float rh = alpaka::math::sqrt(acc, xh * xh + yh * yh);  // rGlobal not stored in OT SoA
                    const float phiH = alpaka::math::atan2(acc, yh, xh);
                    // fresh exact prediction on the updated helix (the per-layer linearization was built
                    // on the pre-winner helix and is now stale, so predict directly at rh/zh).
                    const Prediction p2 =
                        isBarrel ? predictOnBarrel(acc, sh.helix(), rh) : predictOnEndcap(acc, sh.helix(), zh);
                    if (!p2.valid)
                      continue;
                    const float secH = isBarrel ? zh : rh;
                    // The partner sits on the same module as the just-accepted winner, so the road
                    // centre of this layer visit is the offset that applies to it too.
                    const float dPhi = foldPi(p2.phi + shDerElossPhi - phiH);
                    const float dSec = p2.secondary + shDerElossSec - secH;
                    const float rh_safe = alpaka::math::max(acc, rh, 1.f);
                    // OT sensor frame (lower/upper per this partner hit's position within the stack).
                    const float xerr = otHits[p].xerrLocal();
                    const float yerr = otHits[p].yerrLocal();
                    const bool isUpper = (p >= upStart);
                    const auto& frame = isUpper ? otSource_.stackedGeometry.upperSensorFrame()[geomIdx]
                                                : otSource_.stackedGeometry.lowerSensorFrame()[geomIdx];
                    float ge[6];
                    frame.toGlobal(xerr, 0.f, yerr, ge);
                    const float rh2 = rh_safe * rh_safe;
                    const float gxx = ge[0], gxy = ge[1], gyy = ge[2], gzz = ge[5];
                    const float sigPhi2HitRaw = (yh * yh * gxx - 2.f * xh * yh * gxy + xh * xh * gyy) / (rh2 * rh2);
                    const float sigSec2HitRaw =
                        isBarrel ? gzz : (xh * xh * gxx + 2.f * xh * yh * gxy + yh * yh * gyy) / rh2;
                    const float sigPhi2Hit = alpaka::math::max(acc, sigPhi2HitRaw, 1e-6f);
                    const float sigSec2Hit = alpaka::math::max(acc, sigSec2HitRaw, 1e-4f);
                    const float sigPhiHit = alpaka::math::sqrt(acc, sigPhi2Hit);
                    const float sigSecHit = alpaka::math::sqrt(acc, sigSec2Hit);
                    // Incremental multiple scattering since the last accepted point (the winner).
                    const HelixState hForMs = sh.helix();
                    const float dArcSinceLast = alpaka::math::max(acc, p2.arcS - shLastArcS, 0.f);
                    const float msVar =
                        multScattVar(acc, shBf, hForMs, dArcSinceLast, rhoMap_, shLastR, shLastZ, rh_safe, zh);
                    const float s = p2.arcS;
                    const float Jtip = -1.f / rh_safe;
                    const float JinvPt = -shBf * s * s / (2.f * rh_safe);
                    const float Jcot = s;
                    const float predPhiDiag = sh.vPhi + Jtip * Jtip * sh.vTip + JinvPt * JinvPt * sh.vPt;
                    const float predSecDiag_barrel = sh.vZip + Jcot * Jcot * sh.vCot;
                    const float predPhiCross =
                        2.f * Jtip * sh.cPhiTip + 2.f * JinvPt * sh.cPhiPt + 2.f * Jtip * JinvPt * sh.cTipPt;
                    const float predSecCross_barrel = 2.f * Jcot * sh.cCotZip;
                    const float predPhiVar = alpaka::math::max(acc, predPhiDiag + predPhiCross, 1e-12f);
                    float predSecVar;
                    float selJrCot = 0.f, selJrZip = 0.f;
                    if (isBarrel) {
                      predSecVar = alpaka::math::max(acc, predSecDiag_barrel + predSecCross_barrel, 1e-12f);
                    } else {
                      float Jr[5];
                      rWithGrad5(acc, sh.phi0, sh.tip, sh.invPt, sh.cotTheta, sh.zip, zh, shBf, Jr);
                      selJrCot = Jr[3];
                      selJrZip = Jr[4];
                      const float predSecCircleDiag =
                          Jr[0] * Jr[0] * sh.vPhi + Jr[1] * Jr[1] * sh.vTip + Jr[2] * Jr[2] * sh.vPt;
                      const float predSecCircleCross = 2.f * Jr[0] * Jr[1] * sh.cPhiTip +
                                                       2.f * Jr[0] * Jr[2] * sh.cPhiPt +
                                                       2.f * Jr[1] * Jr[2] * sh.cTipPt;
                      const float predSecLineDiag = Jr[3] * Jr[3] * sh.vCot + Jr[4] * Jr[4] * sh.vZip;
                      const float predSecLineCross = 2.f * Jr[3] * Jr[4] * sh.cCotZip;
                      predSecVar = alpaka::math::max(
                          acc, predSecCircleDiag + predSecCircleCross + predSecLineDiag + predSecLineCross, 1e-12f);
                    }
                    const float alignSigmaPhi2 = (alignSigmaPhiCm * alignSigmaPhiCm) / (rh_safe * rh_safe);
                    const float alignSigmaSec2 = alignSigmaSecCm * alignSigmaSecCm;
                    const float msPhi2 = msVar * dArcSinceLast * dArcSinceLast / (rh_safe * rh_safe);
                    // Covariance recalibration (raw-OT stack-partner scan): the same raw-OT class scale
                    // as round 1; at 1.0 the propagated cov is unscaled.
                    const float covScale = extCovScaleRawOT;
                    const float sigPhi2 = sigPhi2Hit + msPhi2 + predPhiVar * covScale + alignSigmaPhi2;
                    const float sigSec2 =
                        sigSec2Hit + msVar * dArcSinceLast * dArcSinceLast + predSecVar * covScale + alignSigmaSec2;
                    // the derived road, as a separate pair of consts (same reason as in the merged round:
                    // a mutable destination would change the FMA contraction of the fixed-cut expressions above)
                    // sigma_R and sigma_S are built in centimetres (the frame the tables are measured in); the
                    // r-phi one is converted back to rad^2 so it is consumed exactly like sigPhi2.
                    float sigPhi2Der = 0.f, sigSec2Der = 0.f;
                    if (shDerLayOn) {
                      const float dVc = extDV[extMatClass(L, isBarrel, rh_safe)];
                      const float sR2cm = shDerSigR2 + sigPhi2Hit * rh2 + alignSigmaPhiCm * alignSigmaPhiCm + dVc;
                      const float sS2cm = shDerSigS2 + predSecVar + sigSec2Hit + alignSigmaSec2 + dVc;
                      sigPhi2Der = alpaka::math::max(acc, sR2cm / rh2, 1e-12f);
                      sigSec2Der = alpaka::math::max(acc, sS2cm, 1e-12f);
                    }

                    const float chi2 = shDerLayOn ? ((dPhi * dPhi) / sigPhi2Der + (dSec * dSec) / sigSec2Der)
                                                  : ((dPhi * dPhi) / sigPhi2 + (dSec * dSec) / sigSec2);
                    const float dPhiR = dPhi * rh;
                    constexpr float kSigmaCapMult = 5.f;
                    const float sigPhiHitCm = sigPhiHit * rh;
                    const float capRPhiEff = alpaka::math::max(acc, maxRPhiResidCm, kSigmaCapMult * sigPhiHitCm);
                    const float capSecEff = isBarrel ? alpaka::math::max(acc, maxSecResidCm, kSigmaCapMult * sigSecHit)
                                                     : endcapMaxSecResidCm;
                    // on the derived path the cm caps act as runaway ceilings only (see the merged round):
                    // the quantile ball selects, the module-envelope ceilings only guard, and every ceiling
                    // rejection of a ball-passer is counted.
                    bool ballOk = true;
                    bool ceilOk = true;
                    if (shDerLayOn) {
                      const float wR = alpaka::math::sqrt(acc, shDerQ * sigPhi2Der) * rh;
                      const float wS = alpaka::math::sqrt(acc, shDerQ * sigSec2Der);
                      ballOk = (alpaka::math::abs(acc, dPhiR) < wR) && (alpaka::math::abs(acc, dSec) < wS);
                      const bool okR = alpaka::math::abs(acc, dPhiR) < shDerCeilR;
                      const bool okS = alpaka::math::abs(acc, dSec) < shDerCeilS;
                      ceilOk = okR && okS;
                      if (ballOk && !okR)
                        alpaka::atomicAdd(acc, &stats[kStatDerCapR], 1u, alpaka::hierarchy::Grids{});
                      if (ballOk && !okS)
                        alpaka::atomicAdd(acc, &stats[kStatDerCapS], 1u, alpaka::hierarchy::Grids{});
                    }
                    const bool absResidualOk = shDerLayOn ? (ballOk && ceilOk)
                                                          : ((alpaka::math::abs(acc, dPhiR) < capRPhiEff) &&
                                                             (alpaka::math::abs(acc, dSec) < capSecEff));
                    const int32_t candId = int32_t(kOTHitTag | p);
                    const bool isBetter = (chi2 < bestPChi2) || (chi2 == bestPChi2 && candId < bestPHit);
                    if (isBetter && absResidualOk) {
                      bestPChi2 = chi2;
                      bestPHit = candId;
                      bestPDPhi = dPhi;
                      bestPDSec = dSec;
                      bestPSigPhi2 = sigPhi2Hit + msPhi2;
                      bestPSigSec2 = sigSec2Hit + msVar * dArcSinceLast * dArcSinceLast;
                      bestPRh = rh;
                      bestPZh = zh;
                      bestPArcS = s;
                      bestPJrCot = selJrCot;
                      bestPJrZip = selJrZip;
                    }
                  }
                  if (bestPHit >= 0) {
                    extrasIds[j * maxExtraHitsPerTrack + shNExtra] = uint32_t(bestPHit);
                    extrasChi2[j * maxExtraHitsPerTrack + shNExtra] = bestPChi2;
                    ++shNExtra;
                    ++shNExtraClusters;  // MTV-aligned cap: partner raw-OT extra == 1 cluster
                    if (secFracDiag_) {
                      // Fold the partner accept into the shadow cov: sh already carries the primary
                      // accept, so the partner row is linearized on this post-primary state.
                      const float rhs = alpaka::math::max(acc, bestPRh, 1.f);
                      const float ss = bestPArcS;
                      const float Hp[5] = {1.f, -1.f / rhs, -shBf * ss * ss / (2.f * rhs), 0.f, 0.f};
                      float Hs[5];
                      if (isBarrel) {
                        Hs[0] = 0.f;
                        Hs[1] = 0.f;
                        Hs[2] = 0.f;
                        Hs[3] = ss;
                        Hs[4] = 1.f;
                      } else {
                        rWithGrad5(acc, sh.phi0, sh.tip, sh.invPt, sh.cotTheta, sh.zip, bestPZh, shBf, Hs);
                      }
                      const float R00 = bestPSigPhi2 + (alignSigmaPhiCm * alignSigmaPhiCm) / (rhs * rhs);
                      const float R11 = bestPSigSec2 + alignSigmaSecCm * alignSigmaSecCm;
                      updateShadowCov(acc, shC, Hp, Hs, R00, R11);
                      if (!isBarrel)
                        ++shNDiskAcc;
                    }
                    sh.updateCircleFromPhi(acc, bestPDPhi, bestPSigPhi2, bestPRh, shBf, bestPArcS);
                    // An endcap partner attach also folds its r-innovation into the line block, with
                    // the same +dSec convention as the primary accept above (the mean-update sign lives
                    // inside updateLineFromSecH).
                    if (isBarrel)
                      sh.updateLineFromSec(acc, bestPDSec, bestPSigSec2, bestPArcS);
                    else
                      sh.updateLineFromSecH(acc, bestPDSec, bestPSigSec2, bestPJrCot, bestPJrZip);
                    sh.recomputeHelix(acc, shBf);
                    shLastArcS = bestPArcS;
                    shLastR = bestPRh;
                    shLastZ = bestPZh;
                    alpaka::atomicAdd(acc, &stats[kStatOTPartner], 1u, alpaka::hierarchy::Grids{});
                    if (candDump_)  // dump: a stack-partner second raw-OT extra was attached here
                      shDumpPartner = 1;
                  }
                }
              }
            }
            alpaka::syncBlockThreads(acc);
          }  // for round (0 = merged scan, 1 = OT-only additive scan)
          // Hole counter: update the cumulative/consecutive occupancy-gated hole run from this
          // layer's outcome (lane-0; shHoleOcc summed by scan lanes is visible after the round-loop sync
          // above, shChainHasPass written by lane-0). A hole == rejGate (occupancy present, no gate-passer);
          // rejEmpty (no occupancy) and rejCap (a passer existed) are not holes. A gate-passer breaks the
          // consecutive run. Runs before the dump record write so holeRunToHere reflects this layer,
          // and only under candDump_.
          if (candDump_) {
            for (auto element : cms::alpakatools::uniform_group_elements(acc, j, nC * nLanes)) {
              if (element.local != 0u)
                continue;
              const bool hole = (shHoleOcc > 0u) && (shChainHasPass == 0);
              if (hole) {
                ++shHoleRun;
                ++shHoleConsec;
              } else if (shChainHasPass != 0) {
                shHoleConsec = 0;  // a gate-passer was found: reset the consecutive-hole run
              }
            }
          }
          // Candidate dump: record this (candidate, visited layer) trace for all layer families.
          // Lane 0 writes one ExtCandLayerRec at
          // the fixed index j * maxWalkLayers + vi (vi = 0-based visit index). candDump_ is uniform
          // across the block, so the trailing sync is reached by every lane.
          if (candDump_) {
            for (auto element : cms::alpakatools::uniform_group_elements(acc, j, nC * nLanes)) {
              if (element.local != 0u)
                continue;
              const int vi = shWalkSteps - 1;  // this scanned layer's 0-based visit index
              if (vi >= 0 && uint32_t(vi) < uint32_t(maxWalkLayers) && j < maxCandidates) {
                ExtCandLayerRec r;
                r.trackId = int32_t(i);
                r.layerId = shCurrentL;
                r.nWinMerged = int32_t(shDumpOccM);
                r.nWinOT = int32_t(shDumpOccO);
                r.nPassMerged = int32_t(shDumpPassM);
                r.nPassOT = int32_t(shDumpPassO);
                r.winnerHitId = shDumpWinHit;
                r.round = shDumpWinRound;
                r.winnerChi2 = shDumpWinChi2;
                r.bestFailChi2 = (shDumpBestM == 0xFFFFFFFFu) ? -1.f : float(shDumpBestM) * 1e-3f;
                // Lane-0 serial reduction over the per-lane best-any slots (min chi2, tie by min id)
                // -> the overall-best (min-chi2) road candidate over all considered hits in both rounds.
                // The winner tree-reduce above does not touch laneBestAny*. bAnyHit carries the
                // bit30-tagged id (-1 = none considered), bAnyPass its base-gate pass; bestFailChi2 (from
                // shDumpBestM) is that same candidate's chi2 by construction.
                float bAnyChi2 = 3.4e38f;
                int32_t bAnyHit = -1;
                int bAnyPass = 0;
                for (uint32_t l = 0; l < nLanes; ++l) {
                  const int32_t h = laneBestAnyHit[l];
                  if (h < 0)
                    continue;
                  const float c = laneBestAnyChi2[l];
                  if (bAnyHit < 0 || c < bAnyChi2 || (c == bAnyChi2 && h < bAnyHit)) {
                    bAnyChi2 = c;
                    bAnyHit = h;
                    bAnyPass = laneBestAnyPass[l];
                  }
                }
                r.bestHitId = bAnyHit;        // already bit30-tagged if raw-OT; -1 = none considered
                r.holeRunToHere = shHoleRun;  // cumulative occupancy-gated holes through this layer
                int32_t flags = 0;
                if (shDumpVeto)
                  flags |= kExtDumpFlagVetoSkip;
                if (shDumpPartner)
                  flags |= kExtDumpFlagPartner;
                if (bAnyHit >= 0 && bAnyPass)
                  flags |= kExtDumpFlagBestPass;  // the min-chi2 road candidate cleared the base gate
                r.flags = flags;
                if (shDumpWinHit >= 0)
                  r.outcome = kExtDumpAccept;
                else if (shDumpCapDropped)
                  r.outcome = kExtDumpRejCap;
                else if (shDumpOccM + shDumpOccO == 0u)
                  r.outcome = kExtDumpRejEmpty;
                else if (shDumpPassM + shDumpPassO == 0u)
                  r.outcome = kExtDumpRejGate;
                else
                  r.outcome = kExtDumpRejOther;
                candLayerBuf_[uint32_t(j) * uint32_t(maxWalkLayers) + uint32_t(vi)] = r;
              } else if (candDumpOvf_ != nullptr) {  // count-and-clamp: visit index out of the sized stride
                alpaka::atomicAdd(acc, &candDumpOvf_[0], 1u, alpaka::hierarchy::Grids{});
              }
            }
            alpaka::syncBlockThreads(acc);
          }
        }  // while (walk)

        for (auto element : cms::alpakatools::uniform_group_elements(acc, j, nC * nLanes)) {
          if (element.local == 0u) {
            nExtras[j] = shNExtra;
          }
        }
        alpaka::syncBlockThreads(acc);  // finish this candidate before shared memory is reused
      }  // for (candidate j)
    }
  };

  // Cross-track arbitration, init phase: set the atomicMin identity on exactly the claim
  // rows the claim/resolve pair will touch. Same loop shape, same (j, k) index space and the same
  // claimIdx expression as Kernel_extClaimExtras below, so the set of rows initialised here is exactly
  // the set of rows read there and in Kernel_extResolveExtras. Two candidates proposing the same hit
  // write the same value to the same row, so the duplicate writes are harmless.
  struct Kernel_extInitClaims {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const int maxExtraHitsPerTrack,
                                  const int maxSharedOwners,
                                  const uint32_t* __restrict__ nCands,
                                  const uint32_t maxCandidates,
                                  const uint32_t nHits,  // OT claims live at nHits + otIdx (0 => merged-only)
                                  const uint32_t* __restrict__ extrasIds,
                                  const int32_t* __restrict__ nExtras,
                                  uint64_t* __restrict__ hitClaims) const {
      const int nOwners = maxSharedOwners > 1 ? maxSharedOwners : 1;
      const uint64_t kUnclaimed = ~uint64_t(0);  // 0xff..ff, the unclaimed identity
      const uint32_t nC = alpaka::math::min(acc, *nCands, maxCandidates);
      for (auto idx : cms::alpakatools::uniform_elements(acc, nC * uint32_t(maxExtraHitsPerTrack))) {
        const uint32_t j = idx / maxExtraHitsPerTrack;
        const int k = int(idx % maxExtraHitsPerTrack);
        if (k >= nExtras[j])
          continue;
        const uint32_t id = extrasIds[j * maxExtraHitsPerTrack + k];
        const uint32_t claimIdx = isOTId(id) ? nHits + otIdx(id) : id;
        for (int s = 0; s < nOwners; ++s)
          hitClaims[std::size_t(claimIdx) * nOwners + s] = kUnclaimed;
      }
    }
  };

  // Cross-track arbitration, claim phase: every accepted extra bids for its hit with its gate chi2.
  struct Kernel_extClaimExtras {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const int maxExtraHitsPerTrack,
                                  const int maxSharedOwners,  // N sorted claim slots per hit (1 => exclusive)
                                  const uint32_t* __restrict__ candList,
                                  const uint32_t* __restrict__ nCands,
                                  const uint32_t maxCandidates,
                                  const uint32_t nHits,  // OT claims live at nHits + otIdx (0 => merged-only)
                                  const uint32_t* __restrict__ extrasIds,
                                  const float* __restrict__ extrasChi2,
                                  const int32_t* __restrict__ nExtras,
                                  uint64_t* __restrict__ hitClaims) const {
      const int nOwners = maxSharedOwners > 1 ? maxSharedOwners : 1;
      const uint32_t nC = alpaka::math::min(acc, *nCands, maxCandidates);
      for (auto idx : cms::alpakatools::uniform_elements(acc, nC * uint32_t(maxExtraHitsPerTrack))) {
        const uint32_t j = idx / maxExtraHitsPerTrack;
        const int k = int(idx % maxExtraHitsPerTrack);
        if (k >= nExtras[j])
          continue;
        const uint32_t id = extrasIds[j * maxExtraHitsPerTrack + k];
        const uint32_t claimIdx = isOTId(id) ? nHits + otIdx(id) : id;
        const uint64_t claim = packClaim(extrasChi2[j * maxExtraHitsPerTrack + k], candList[j]);
        // atomicMin insertion cascade into the hit's N sorted slots: each slot keeps the min of its
        // old value and the incoming carry (atomic, no lost update), the max cascades to the next
        // slot; a carry falling past slot N-1 is dropped (largest claim evicted). The final N-slot
        // set is the N smallest packed claims regardless of interleaving (multiset conservation over
        // each atomicMin). N == 1 collapses to a single atomicMin per hit.
        uint64_t carry = claim;
        const uint64_t kUnclaimed = ~uint64_t(0);  // 0xff..ff, the identity Kernel_extInitClaims wrote
        for (int s = 0; s < nOwners; ++s) {
          const uint64_t old = alpaka::atomicMin(
              acc, &hitClaims[std::size_t(claimIdx) * nOwners + s], carry, alpaka::hierarchy::Grids{});
          carry = old > carry ? old : carry;  // evicted (larger) value cascades to the next slot
          if (carry == kUnclaimed)
            break;  // an empty slot absorbed the carry; nothing more to insert
        }
      }
    }
  };

  // Cross-track arbitration, resolve phase: keep only the extras whose claim won; compact each
  // candidate's list in place and accumulate the per-event counters.
  struct Kernel_extResolveExtras {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const int maxExtraHitsPerTrack,
                                  const int maxSharedOwners,      // N sorted claim slots per hit (1 => exclusive)
                                  const float extAmbigDeltaChi2,  // ambiguity gate (<=0 => off)
                                  const uint32_t* __restrict__ candList,
                                  const uint32_t* __restrict__ nCands,
                                  const uint32_t maxCandidates,
                                  const uint32_t nHits,  // OT claims live at nHits + otIdx (0 => merged-only)
                                  uint32_t* __restrict__ extrasIds,
                                  float* __restrict__ extrasChi2,
                                  int32_t* __restrict__ nExtras,
                                  const uint64_t* __restrict__ hitClaims,
                                  uint32_t* __restrict__ stats) const {
      const int nOwners = maxSharedOwners > 1 ? maxSharedOwners : 1;
      const uint32_t nC = alpaka::math::min(acc, *nCands, maxCandidates);
      for (auto j : cms::alpakatools::uniform_elements(acc, nC)) {
        const uint32_t tupleId = candList[j];
        const int n = nExtras[j];
        int kept = 0;
        for (int k = 0; k < n; ++k) {
          const uint32_t hitId = extrasIds[j * maxExtraHitsPerTrack + k];
          const uint32_t claimIdx = isOTId(hitId) ? nHits + otIdx(hitId) : hitId;
          // Keep the extra iff this tuple owns any of the hit's N claim slots (N == 1 => the single
          // atomicMin winner, i.e. exclusive ownership).
          bool won = false;
          for (int s = 0; s < nOwners; ++s)
            won = won || (uint32_t(hitClaims[std::size_t(claimIdx) * nOwners + s] & 0xffffffffu) == tupleId);
          // ambiguity gate: when this hit is contested by >=2 distinct claimants whose top-2 gate
          // chi2 sit within extAmbigDeltaChi2, the hit is an ambiguous near-tie -> keep it only for the
          // single best claimant (slot 0, smallest gate chi2) and veto it from every worse claimant. Needs
          // the 2nd slot (nOwners >= 2). <= 0 disables it and leaves `won` untouched.
          bool ambigVeto = false;
          if (extAmbigDeltaChi2 > 0.f && won && nOwners >= 2) {
            const uint64_t c0 = hitClaims[std::size_t(claimIdx) * nOwners + 0];
            const uint64_t c1 = hitClaims[std::size_t(claimIdx) * nOwners + 1];
            const uint64_t kUnclaimed = ~uint64_t(0);
            if (c1 != kUnclaimed) {  // a real 2nd claimant exists => the hit is contested
              const float chi0 = unpackClaimChi2(c0);
              const float chi1 = unpackClaimChi2(c1);
              if ((chi1 - chi0) < extAmbigDeltaChi2 && uint32_t(c0 & 0xffffffffu) != tupleId) {
                won = false;  // ambiguous contested hit: this tuple is not the best claimant -> drop
                ambigVeto = true;
              }
            }
          }
          if (won) {
            extrasIds[j * maxExtraHitsPerTrack + kept] = hitId;
            extrasChi2[j * maxExtraHitsPerTrack + kept] = extrasChi2[j * maxExtraHitsPerTrack + k];
            // Keep the chain score aligned with extrasIds through the arbitration compaction.
            ++kept;
          } else if (ambigVeto) {
            alpaka::atomicAdd(acc, &stats[kStatAmbigVetoed], 1u, alpaka::hierarchy::Grids{});
          } else {
            // split arbitration losses by source (merged vs raw OT)
            alpaka::atomicAdd(
                acc, &stats[isOTId(hitId) ? kDiagArbLostOT : kDiagArbLostMerged], 1u, alpaka::hierarchy::Grids{});
          }
        }
        if (kept < n)
          alpaka::atomicAdd(acc, &stats[kStatArbLost], uint32_t(n - kept), alpaka::hierarchy::Grids{});
        nExtras[j] = kept;
        if (kept > 0) {
          alpaka::atomicAdd(acc, &stats[kStatExtended], 1u, alpaka::hierarchy::Grids{});
          alpaka::atomicAdd(acc, &stats[kStatTotalExtras], uint32_t(kept), alpaka::hierarchy::Grids{});
        }
        const int bucket = kept < kStatHistBuckets ? kept : kStatHistBuckets - 1;
        alpaka::atomicAdd(acc, &stats[kStatHistFirst + bucket], 1u, alpaka::hierarchy::Grids{});
      }
    }
  };

  // One-line per-event attach summary (single thread), gated by AttachParams::verbose.
  struct Kernel_extPrintSummary {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, const uint32_t* __restrict__ stats) const {
      if (alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0] != 0)
        return;
      printf("[CAExtension] cand=%u overflow=%u extended=%u extras=%u arbLost=%u hist:",
             stats[kStatCandidates],
             stats[kStatCandOverflow],
             stats[kStatExtended],
             stats[kStatTotalExtras],
             stats[kStatArbLost]);
      for (int b = 0; b < kStatHistBuckets; ++b)
        printf(" %u", stats[kStatHistFirst + b]);
      printf("\n");
      // Host-quality pre-gate skips, per-layer-class walk-committed extras, ambiguity-gate vetoes.
      printf("[CAExtension] preGateSkipped=%u etaSkipped=%u extras[TOB1-3=%u TOB4-6=%u TID=%u] ambigVetoed=%u\n",
             stats[kStatPreGateSkipped],
             stats[kStatPreGateEtaSkipped],
             stats[kStatExtraTOB13],
             stats[kStatExtraTOB456],
             stats[kStatExtraTID],
             stats[kStatAmbigVetoed]);
      // Far-first disc ordering: hosts whose ordering was re-keyed, and far commits its window-ambiguity
      // condition declined. Both stay 0 unless extAttachFarFirst is on; declined stays 0 additionally when
      // extAttachFarMaxWin <= 0. The ratio is the only production-readable measure of what the condition
      // does -- a declined layer is indistinguishable from "no gate-passer" in every other counter.
      printf(
          "[CAExtension] farFirst: armed=%u declined=%u\n", stats[kStatAttachFarArmed], stats[kStatAttachFarDecline]);
      // The derived selection's own alarms. capR/capS count the ball-passers the module-envelope
      // runaway ceilings rejected; divided by derLayers (the layer visits the derived gate armed on)
      // that is the ceilings' binding rate -- << 1 % means the ceiling is a guard, anything more means
      // it is silently doing the selection and the delivered efficiency is not the stated eps. The
      // hole* counters stay 0 while the hypothesis is gated to d >= 3 (kExtDerHoleMinDim). All zero
      // unless the derived package is armed, and printed only under verbose.
      printf("[CAExtension] derived: layers=%u capR=%u capS=%u holeCand=%u holeFire=%u hostOn=%u hostOff=%u\n",
             stats[kStatDerLayers],
             stats[kStatDerCapR],
             stats[kStatDerCapS],
             stats[kStatDerHoleCand],
             stats[kStatDerHoleFire],
             stats[kStatDerHostOn],
             stats[kStatDerHostOff]);
      // raw-channel monitor: the round-1 projections of the two lines above -- the single-cluster
      // channel's per-class yield, and the rate at which the per-source-round hole pricing declines it.
      // rawExtras against extras[] above gives the merged round's own split; holeRaw against
      // holeCand/holeFire gives the raw round's own hole-test and decline rates.
      printf("[CAExtension] rawRound: extras[TOB1-3=%u TOB4-6=%u TID=%u] holeCand=%u holeFire=%u\n",
             stats[kStatRawTOB13],
             stats[kStatRawTOB456],
             stats[kStatRawTID],
             stats[kStatDerHoleCandRaw],
             stats[kStatDerHoleFireRaw]);
    }
  };

  AttachBuffers allocateAttachBuffers(Queue& queue,
                                      const AttachParams& params,
                                      uint32_t nHits,
                                      uint32_t nOTHits,
                                      uint32_t maxNumberOfTuples,
                                      ::reco::TrackSoAConstView tracks,
                                      const double* passBuf,
                                      const int32_t* candidateMask,
                                      uint32_t knownCandCapacity) {
    // candCapacity-independent buffers, needed by (or unaffected by) the count pass
    auto nCands = cms::alpakatools::make_device_buffer<uint32_t>(queue);
    auto stats = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, kStatSize);
    // Per-hit arbitration claim buffer: merged hits occupy [0, nHits); OT extras (when active) occupy
    // [nHits, nHits + nOTHits). nOTHits == 0 keeps this sized to nHits exactly.
    // N-way sharing (extMaxSharedOwners): each hit keeps N sorted claim slots at
    // [claimIdx*N, claimIdx*N+N); N == 1 reduces to one slot per hit (a single atomicMin winner).
    const auto nOwners = std::size_t(std::max(1, params.extMaxSharedOwners));
    auto hitClaims =
        cms::alpakatools::make_device_buffer<uint64_t[]>(queue, std::max<std::size_t>(1, (nHits + nOTHits) * nOwners));
    alpaka::memset(queue, nCands, 0);
    alpaka::memset(queue, stats, 0);
    // hitClaims is not memset here: the 0xFF "unclaimed" identity is needed only on the rows the
    // arbitration touches (at most nCands * maxExtraHitsPerTrack, a tiny fraction of the
    // nHits + nOTHits rows this buffer spans). Kernel_extInitClaims writes exactly those rows in
    // launchAttach, immediately before Kernel_extClaimExtras, over the same (candidate, slot) index
    // space the claim and resolve kernels read; rows outside that set are never read.

    // count-only pre-gate pass: nCands (+ candidate/overflow counters), no candList writes
    constexpr auto threadsPerBlock = 256u;
    const auto blocksTuples = cms::alpakatools::divide_up_by(maxNumberOfTuples, threadsPerBlock);
    const auto workDivTuples = cms::alpakatools::make_workdiv<Acc1D>(blocksTuples, threadsPerBlock);
    alpaka::exec<Acc1D>(queue,
                        workDivTuples,
                        Kernel_extPreGate{},
                        tracks,
                        passBuf,
                        maxNumberOfTuples,
                        params.preGateMaxChi2,
                        params.preGateMinPt,
                        std::sinh(params.maxAbsEta),
                        params.extHostMaxChi2Ndof,
                        params.extHostMinHits,
                        params.extHostMinPt,
                        candidateMask,  // restrict the count to the caller's set (null = no restriction)
                        params.maxCandidates,
                        /*candList=*/static_cast<uint32_t*>(nullptr),
                        nCands.data(),
                        stats.data());

    // Size the scratch from the caller's host-known candidate bound. knownCandCapacity > 0 is part
    // of the contract: the count pass above still fills the stats and overflow counters but its
    // result is not read back, so nothing here blocks the host. It need only be an upper bound, the
    // fill pass capping at candCapacity so that no candidate is dropped. extRefitMaxCandidates is an
    // optional ceiling on it, for callers that can only offer the whole track capacity as a bound;
    // the scratch, the refit scaffold and the per-candidate grids all shrink with it.
    assert(knownCandCapacity > 0);
    const uint32_t candBound = std::min(knownCandCapacity, params.extRefitMaxCandidates);
    const uint32_t candCapacity = std::min(std::max(candBound, 16u), params.maxCandidates);

    // Reset the counter so the fill pass in launchAttach re-runs the same predicate from scratch.
    alpaka::memset(queue, nCands, 0);

    const auto nSlots = std::size_t(candCapacity) * std::size_t(params.maxExtraHitsPerTrack);
    AttachBuffers bufs{cms::alpakatools::make_device_buffer<uint32_t[]>(queue, std::max<std::size_t>(1, candCapacity)),
                       std::move(nCands),
                       cms::alpakatools::make_device_buffer<uint32_t[]>(queue, std::max<std::size_t>(1, nSlots)),
                       cms::alpakatools::make_device_buffer<float[]>(queue, std::max<std::size_t>(1, nSlots)),
                       cms::alpakatools::make_device_buffer<int32_t[]>(queue, std::max<std::size_t>(1, candCapacity)),
                       std::move(hitClaims),
                       std::move(stats)};
    bufs.candCapacity = candCapacity;
    // bufs.nExtras needs no memset: Kernel_extFindExtras stores nExtras[j] = shNExtra unconditionally
    // for every candidate group j of its uniform_groups(acc, nC * nLanes) loop, and it is the first
    // kernel of launchAttach to touch the buffer. Every reader clamps to the same
    // nC = min(*nCands, maxCandidates), so the [nC, candCapacity) tail is never read.
    return bufs;
  }

  // Full-hits OT source. These kernels fill the per-event buffers that describe the raw OT rechit
  // collection -- iphi, the used-in-stub mask, the per-CA-layer row offsets and the phi binner -- which
  // the walk's round 1, its stack-partner scan and the final rewrite then read through OTHitsSource.

  // iphi per OT rechit, using the same convention as the merged SoA (OTRecHitsSoA.h documents
  // iphi = unsafe_atan2s<7>(yGlobal, xGlobal)).
  struct Kernel_otFillIphi {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::OTRecHitsConstView otHits,
                                  int16_t* __restrict__ iphi,
                                  uint32_t nOTHits) const {
      for (auto i : cms::alpakatools::uniform_elements(acc, nOTHits)) {
        const float xg = otHits[i].xGlobal();
        const float yg = otHits[i].yGlobal();
        iphi[i] = unsafe_atan2s<7>(yg, xg);
      }
    }
  };

  // Mark every OT rechit that is a member of some stub (used[lowerHitIdx] = used[upperHitIdx] = 1).
  // Guards the UINT32_MAX sentinel (P-hit-only stubs) and the nOTHits bound.
  struct Kernel_otMarkUsedInStub {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::StubsConstView stubs,
                                  uint8_t* __restrict__ used,
                                  uint32_t nStubs,
                                  uint32_t nOTHits) const {
      for (auto s : cms::alpakatools::uniform_elements(acc, nStubs)) {
        const uint32_t lo = stubs[s].lowerHitIdx();
        const uint32_t up = stubs[s].upperHitIdx();
        if (lo != UINT32_MAX && lo < nOTHits)
          used[lo] = uint8_t(1);
        if (up != UINT32_MAX && up < nOTHits)
          used[up] = uint8_t(1);
      }
    }
  };

  // Per-CA-layer OT-row offsets (mirror of SetHitsLayerStart, but over the OT stacked-module start
  // array). For an OT layer i the start is otHitModules.moduleStart[ll.layerStarts()[i] - nPixelModules]
  // (detectorIndex numbering = nPixelModules + geom index in CA order). Pixel layers (layerStarts <
  // nPixelModules) precede all OT layers, so they get an empty range anchored at the first OT row (0).
  // The final entry (i == nLayers, layerStarts == total modules) resolves to the nOTHits sentinel that
  // fillManyFromVector needs as the upper bound.
  struct Kernel_otSetLayerStart {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::OTHitModuleConstView otHitModules,
                                  ::reco::CALayersSoAConstView ll,
                                  uint32_t nPixelModules,
                                  uint32_t nOTHits,
                                  uint32_t* __restrict__ otHitsLayerStart) const {
      // OTHitModules block is sized nOTModules + 1 (cumulative-sum convention).
      const uint32_t nOTModules = uint32_t(otHitModules.metadata().size()) - 1u;
      for (auto i : cms::alpakatools::uniform_elements(acc, uint32_t(ll.metadata().size()))) {
        const int caMod = ll.layerStarts()[i];
        if (caMod < int(nPixelModules)) {
          otHitsLayerStart[i] = 0u;  // pixel layer: empty OT range at the first OT row
        } else {
          const uint32_t geom = uint32_t(caMod) - nPixelModules;
          otHitsLayerStart[i] = (geom < nOTModules) ? otHitModules.moduleStart()[geom] : nOTHits;
        }
      }
    }
  };

  OTHitsBuffers buildOTHitsSource(Queue& queue,
                                  const AttachParams& params,
                                  ::reco::OTRecHitsConstView otHits,
                                  ::reco::OTHitModuleConstView otHitModules,
                                  ::reco::StackedModuleGeometryConstView stackedGeometry,
                                  ::reco::StubsConstView stubs,
                                  ::reco::CALayersSoAConstView caLayers,
                                  uint32_t nOTHits,
                                  uint32_t nStubs) {
    using namespace cms::alpakatools;
    constexpr uint32_t nPixelModules = ::phase2PixelTopology::nModulesPix;  // 4000; OT detIdx = 4000 + geom
    const uint32_t nLayersP1 = uint32_t(caLayers.metadata().size());        // nLayers + 1

    // Per-event buffers (transient, caching-allocator backed).
    auto iphi = make_device_buffer<int16_t[]>(queue, nOTHits);
    auto usedInStub = make_device_buffer<uint8_t[]>(queue, nOTHits);
    auto layerStart = make_device_buffer<uint32_t[]>(queue, nLayersP1);
    auto phiHist = make_device_buffer<ExtPhiBinner>(queue);
    auto phiStorage = make_device_buffer<ExtPhiBinner::value_type[]>(queue, nOTHits);

    // usedInStub is filled sparsely, so it is zeroed every event. The per-hit `ownership` array is not
    // allocated: nothing writes it, so OTHitsSource::ownership stays null and the walk's two vetoes
    // are guarded on the pointer (a null array masks nothing).
    alpaka::memset(queue, usedInStub, 0);

    constexpr uint32_t kThreads = 256u;
    {
      const auto blocks = divide_up_by(nOTHits, kThreads);
      alpaka::exec<Acc1D>(
          queue, make_workdiv<Acc1D>(blocks, kThreads), Kernel_otFillIphi{}, otHits, iphi.data(), nOTHits);
    }
    if (nStubs > 0) {
      const auto blocks = divide_up_by(nStubs, kThreads);
      alpaka::exec<Acc1D>(queue,
                          make_workdiv<Acc1D>(blocks, kThreads),
                          Kernel_otMarkUsedInStub{},
                          stubs,
                          usedInStub.data(),
                          nStubs,
                          nOTHits);
    }
    {
      const auto blocks = divide_up_by(nLayersP1, kThreads);
      alpaka::exec<Acc1D>(queue,
                          make_workdiv<Acc1D>(blocks, kThreads),
                          Kernel_otSetLayerStart{},
                          otHitModules,
                          caLayers,
                          nPixelModules,
                          nOTHits,
                          layerStart.data());
    }
    // OT phi binner: same 256-bin, per-CA-layer partition as the merged-hit binner (exactly the
    // CAHitNtupletGeneratorKernels.dev.cc prepareHits() pattern), sized to nOTHits.
    ExtPhiBinner::View phiView{phiHist.data(), nullptr, phiStorage.data(), cms::alpakatools::kDynamicSize, nOTHits};
    fillManyFromVector<Acc1D>(phiHist.data(),
                              phiView,
                              ::pixelTopology::Phase2OTStubs::numberOfLayers,
                              iphi.data(),
                              layerStart.data(),
                              nOTHits,
                              (uint32_t)256,
                              queue);

#ifdef EXT_OTFILL_DIAG
    // Compile-gated one-shot OT-fill sanity dump (first event only): nOTHits, used-in-stub fraction,
    // and per-CA-layer OT bin totals (layerStart diffs == hits binned per layer). Not compiled on a
    // plain build; enable with -DEXT_OTFILL_DIAG to check the OT source wiring and sizing.
    static std::atomic<bool> printedOT{false};
    bool expected = false;
    if (printedOT.compare_exchange_strong(expected, true)) {
      std::vector<uint32_t> hLayer(nLayersP1);
      std::vector<uint8_t> hUsed(nOTHits);
      auto hLayerView = make_host_view(hLayer.data(), nLayersP1);
      auto hUsedView = make_host_view(hUsed.data(), nOTHits);
      alpaka::memcpy(queue, hLayerView, make_device_view(queue, layerStart.data(), nLayersP1));
      alpaka::memcpy(queue, hUsedView, make_device_view(queue, usedInStub.data(), nOTHits));
      alpaka::wait(queue);
      std::size_t nUsed = 0;
      for (uint32_t k = 0; k < nOTHits; ++k)
        nUsed += (hUsed[k] != 0);
      const double frac = nOTHits ? double(nUsed) / double(nOTHits) : 0.;
      printf("[CAExtension][OTfill] nOTHits=%u nStubs=%u usedInStub=%zu (frac=%.4f) perLayerOT:",
             nOTHits,
             nStubs,
             nUsed,
             frac);
      for (uint32_t i = 0; i + 1 < nLayersP1; ++i) {
        const uint32_t cnt = hLayer[i + 1] - hLayer[i];
        if (cnt > 0)
          printf(" L%u:%u", i, cnt);
      }
      printf("\n");
    }
#endif

    return OTHitsBuffers{std::move(iphi),
                         std::move(usedInStub),
                         std::move(layerStart),
                         std::move(phiHist),
                         std::move(phiStorage),
                         otHits,
                         stubs,
                         otHitModules,
                         stackedGeometry,
                         nOTHits};
  }

  void launchExtHostMask(Queue& queue,
                         const AttachParams& params,
                         ::reco::TrackSoAConstView tracks,
                         uint32_t nTracksCap,
                         int32_t* hostMask,
                         ExtPredCoeff* pred) {
    if (nTracksCap == 0u || hostMask == nullptr)
      return;
    constexpr uint32_t bs = 128;
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(nTracksCap, bs), bs),
                        Kernel_extHostMask{},
                        tracks,
                        params.preGateMaxChi2,
                        params.preGateMinPt,
                        params.maxAbsEta,
                        params.extHostMaxChi2Ndof,
                        nTracksCap,
                        hostMask,
                        pred);
  }

  void launchAttach(Queue& queue,
                    const AttachParams& params,
                    float bf,
                    const float* rhoMap,
                    const ExtPhiBinner* phiBinner,
                    ::reco::TrackSoAConstView tracks,
                    ::reco::TrackHitSoAConstView trackHits,
                    const uint8_t* armId,  // pocket gate: per-track arm (0=prompt,1=disp); null=off/arm-blind
                    ::reco::TrackingRecHitConstView hits,
                    ::reco::HitModuleSoAConstView hitModules,
                    ::reco::TrackingRecHitsMaskingConstView hitMask,
                    ::reco_extender::ExtenderLayersConstView extLayers,
                    ::reco::CAModulesConstView caModules,
                    const double* passBuf,
                    uint32_t maxNumberOfTuples,
                    uint32_t nHits,
                    AttachBuffers& bufs,
                    const OTHitsSource* otSource,
                    const int32_t* candidateMask) {
    // The full-hits OT source is consumed by the walk's second bin-loop (Kernel_extFindExtras round 1)
    // and its stack-partner scan. Held by value so the kernel object copies it to the device. A default
    // source (nOTHits == 0) => the walk scans merged hits only.
    const bool otActive = (otSource != nullptr && otSource->nOTHits > 0u);
    const OTHitsSource otSrcVal = otActive ? *otSource : OTHitsSource{};
    // Runtime candidate capacity the scratch was sized to (allocateAttachBuffers ran the count pass).
    const uint32_t candCapacity = bufs.candCapacity;

    // The walk's candidate-level instrument (see EXT_CAND_DUMP at the top of this file). Compiled in
    // only when the macro is defined; in a production build kExtCandDump is a compile-time false, so
    // nothing below is allocated, no readback or sync happens, and the walk carries no dump code.
    const bool candDumpOn = kExtCandDump && candCapacity > 0u;
    const uint32_t candDumpStride = uint32_t(params.maxWalkLayers);
    std::optional<cms::alpakatools::device_buffer<Device, ExtCandLayerRec[]>> dumpLayer;
    std::optional<cms::alpakatools::device_buffer<Device, ExtCandHdrRec[]>> dumpHdr;
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> dumpOvf;
    // Per-road-candidate member buffer, sized candCapacity*maxWalkLayers*kExtDumpMaxMembers.
    std::optional<cms::alpakatools::device_buffer<Device, ExtCandMemberRec[]>> dumpMember;
    ExtCandLayerRec* dumpLayerPtr = nullptr;
    ExtCandHdrRec* dumpHdrPtr = nullptr;
    uint32_t* dumpOvfPtr = nullptr;
    ExtCandMemberRec* dumpMemberPtr = nullptr;
    if (candDumpOn) {
      dumpLayer.emplace(
          cms::alpakatools::make_device_buffer<ExtCandLayerRec[]>(queue, std::size_t(candCapacity) * candDumpStride));
      dumpHdr.emplace(cms::alpakatools::make_device_buffer<ExtCandHdrRec[]>(queue, candCapacity));
      dumpOvf.emplace(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, 2));  // [0]=visit-index, [1]=member
      dumpMember.emplace(cms::alpakatools::make_device_buffer<ExtCandMemberRec[]>(
          queue, std::size_t(candCapacity) * candDumpStride * std::size_t(kExtDumpMaxMembers)));
      alpaka::memset(queue, *dumpLayer, 0xFF);  // int32 trackId = -1 sentinel marks unwritten slots
      alpaka::memset(queue, *dumpHdr, 0xFF);
      alpaka::memset(queue, *dumpOvf, 0);
      alpaka::memset(queue, *dumpMember, 0xFF);  // int32 hitId = -1 sentinel marks unwritten member slots
      dumpLayerPtr = dumpLayer->data();
      dumpHdrPtr = dumpHdr->data();
      dumpOvfPtr = dumpOvf->data();
      dumpMemberPtr = dumpMember->data();
    }

    constexpr auto threadsPerBlock = 256u;
    const auto blocksTuples = cms::alpakatools::divide_up_by(maxNumberOfTuples, threadsPerBlock);
    const auto workDivTuples = cms::alpakatools::make_workdiv<Acc1D>(blocksTuples, threadsPerBlock);
    const auto blocksCands = cms::alpakatools::divide_up_by(candCapacity, threadsPerBlock);
    const auto workDivCands = cms::alpakatools::make_workdiv<Acc1D>(blocksCands, threadsPerBlock);
    const auto blocksSlots =
        cms::alpakatools::divide_up_by(candCapacity * uint32_t(params.maxExtraHitsPerTrack), threadsPerBlock);
    const auto workDivSlots = cms::alpakatools::make_workdiv<Acc1D>(blocksSlots, threadsPerBlock);

    // fill pass: re-run the pre-gate predicate, now writing candList (cap = candCapacity). nCands was
    // reset to 0 by the allocator; stats were already tallied by the count pass and are not touched.
    alpaka::exec<Acc1D>(queue,
                        workDivTuples,
                        Kernel_extPreGate{},
                        tracks,
                        passBuf,
                        maxNumberOfTuples,
                        params.preGateMaxChi2,
                        params.preGateMinPt,
                        std::sinh(params.maxAbsEta),
                        params.extHostMaxChi2Ndof,
                        params.extHostMinHits,
                        params.extHostMinPt,
                        candidateMask,  // same restriction as the count pass (null = no restriction)
                        candCapacity,
                        bufs.candList.data(),
                        bufs.nCands.data(),
                        bufs.stats.data());

    // Kernel_extFindExtras is block-per-candidate: one block (kExtFindLanes lanes) per candidate.
    // Cap the grid and grid-stride the rest (uniform_groups); with runtime sizing candCapacity is
    // already a tight bound on the candidate count, so the grid is essentially exact.
    constexpr uint32_t kExtFindMaxBlocks = 16384u;
    const auto blocksFind = std::min<uint32_t>(candCapacity, kExtFindMaxBlocks);
    const auto workDivFind = cms::alpakatools::make_workdiv<Acc1D>(blocksFind, kExtFindLanes);
    alpaka::exec<Acc1D>(
        queue,
        workDivFind,
        Kernel_extFindExtras{
            rhoMap, phiBinner, otSrcVal, params.verbose, candDumpOn, dumpLayerPtr, dumpHdrPtr, dumpOvfPtr, dumpMemberPtr},
        params.maxExtraHitsPerTrack,
        params.maxWalkLayers,
        bf,
        params.chi2Cut,
        params.endcapChi2Cut,
        params.typePriorityBiasCm,
        params.pixHitsTarget,
        params.maxRPhiResidCm,
        params.maxSecResidCm,
        params.endcapMaxSecResidCm,
        params.alignSigmaPhiCm,
        params.alignSigmaSecCm,
        params.extChi2CutScaleTOB456,      // TOB4-6 barrel gate widen
        params.extChi2CutScaleTID,         // TID endcap gate scale
        params.extRawOTVetoTOB456,         // raw-OT veto on TOB4-6
        params.extRawOTVetoTID,            // raw-OT veto on TID
        params.extDisplacementAwareGate,   // dispgate: displacement-aware TOB1-3 accept tightening
        params.extDispGateSig2,            // dispgate: (|d0|/sigma_d0)^2 displaced-host threshold
        params.extForwardPocketGate,       // pocket gate: forward-eta TOB1-3 accept tightening
        params.extPocketGateArmScoped,     // pocket gate: displaced-arm-only when true
        std::sinh(params.maxAbsEta),       // pre-gate ceiling = the pocket band's top edge
        params.extMtvAlignedExtraCap,      // MTV-aligned per-track extra-cluster cap
        params.extRecallReachRelax,        // pixel reachability envelope slack
        params.extRecallPixelFirstBudget,  // pixel-first K-seat reserve
        params.extCovScalePixel,           // pixel propagated-cov scale
        params.extCovScaleStub,            // stub propagated-cov scale
        params.extCovScaleRawOT,           // raw-OT propagated-cov scale
        params.extPixelGateChi2Cut,        // pixel honest-calibration chi2 cut
        params.extStubBendGate,            // 2S stub-bend nSigma veto
        params.extCapExemptAnchored,       // anchored cluster-cap exemption
        params.extCapBudgetFloor,          // cluster-budget floor
        params.extStateProcessNoise,       // state process noise (P += Q)
        params.extRecallForcePixelVisit,   // force pixel visit on prefer-pixel
        params.extCapExemptTOB46Only,      // scope the anchored exemption to TOB4-6
        params.extCapExemptMaxChi2,        // candidate gate-chi2 cap on the exemption
        params.extMaxWalkLayers,           // runtime visit budget (loop bound)
        params.extAttachFarFirst,          // far-first disc ordering (OT-less forward pixel)
        params.extAttachFarMinAbsEta,      // its |eta| floor
        params.extAttachFarMaxWin,         // its window-ambiguity condition (<=0 = none)
        params.extDerivedSelection,        //  master switch
        float(params.extDerivedEps),       //  the one free number
        params.extDerivedHole,             //  the hole hypothesis
        params.extHoleDetectionPrior,      // the hole window-mass repair (numerator eta_L)
        params.extPred,                    //  per-slot option-D payload
        params.extQhat,                    //  Q-hat(eps) per cell
        params.extEtaL,                    //  measured per-layer stub availability
        params.extRho,                     //  measured per-layer stub density
        params.extFwdEtaBin,               //  the 5th Q-hat |eta| bin
        params.extEtaLRaw,                 // raw round conditional availability (null=off)
        params.extRhoRaw,                  // raw-cluster areal density (null=off)
        params.extDV,                      //  measured target-side dV
        params.extFmsBarrel,               //  material-dispersion scale (barrel)
        params.extFmsEndcap,               //  material-dispersion scale (endcap)
        params.extBendPackage,             //  the bend row + the hole's basis
        params.extQhat3,                   //  Q-hat_3 (stub candidates)
        params.extSigBExcess,              //  measured per-class bend-error excess
        params.extRho3,                    //  measured 3-dof stub density
        tracks,
        trackHits,
        armId,  // pocket gate: per-track arm (null when off/arm-blind)
        hits,
        hitModules,
        hitMask,
        extLayers,
        caModules,
        bufs.candList.data(),
        bufs.nCands.data(),
        candCapacity,
        bufs.extrasIds.data(),
        bufs.extrasChi2.data(),
        bufs.nExtras.data(),
        bufs.stats.data());  // per-event counter buffer (kStat*/kDiag* indices)

    // Seed the atomicMin identity on exactly the rows the next two kernels read.
    alpaka::exec<Acc1D>(queue,
                        workDivSlots,
                        Kernel_extInitClaims{},
                        params.maxExtraHitsPerTrack,
                        params.extMaxSharedOwners,
                        bufs.nCands.data(),
                        candCapacity,
                        otActive ? nHits : 0u,
                        bufs.extrasIds.data(),
                        bufs.nExtras.data(),
                        bufs.hitClaims.data());

    alpaka::exec<Acc1D>(queue,
                        workDivSlots,
                        Kernel_extClaimExtras{},
                        params.maxExtraHitsPerTrack,
                        params.extMaxSharedOwners,
                        bufs.candList.data(),
                        bufs.nCands.data(),
                        candCapacity,
                        otActive ? nHits : 0u,
                        bufs.extrasIds.data(),
                        bufs.extrasChi2.data(),
                        bufs.nExtras.data(),
                        bufs.hitClaims.data());

    alpaka::exec<Acc1D>(queue,
                        workDivCands,
                        Kernel_extResolveExtras{},
                        params.maxExtraHitsPerTrack,
                        params.extMaxSharedOwners,
                        params.extAmbigDeltaChi2,  // ambiguity gate (<=0 => off)
                        bufs.candList.data(),
                        bufs.nCands.data(),
                        candCapacity,
                        otActive ? nHits : 0u,
                        bufs.extrasIds.data(),
                        bufs.extrasChi2.data(),
                        bufs.nExtras.data(),
                        bufs.hitClaims.data(),
                        bufs.stats.data());

    if (params.verbose) {
      const auto oneThread = cms::alpakatools::make_workdiv<Acc1D>(1u, 1u);
      alpaka::exec<Acc1D>(queue, oneThread, Kernel_extPrintSummary{}, bufs.stats.data());
    }

    // Candidate dump: host readback (D2H + wait) then the fixed-format edm::LogInfo lines the offline
    // parser consumes. Records with trackId < 0 are unwritten slots.
    if (candDumpOn) {
      const std::size_t nLay = std::size_t(candCapacity) * candDumpStride;
      const std::size_t nMem = nLay * std::size_t(kExtDumpMaxMembers);  // road-candidate members
      std::vector<ExtCandLayerRec> hLay(nLay);
      std::vector<ExtCandHdrRec> hHdr(candCapacity);
      std::vector<ExtCandMemberRec> hMem(nMem);
      uint32_t hOvf[2] = {0u, 0u};  // [0] = visit-index overrun, [1] = member-slot overflow
      alpaka::memcpy(queue, cms::alpakatools::make_host_view(hLay.data(), nLay), *dumpLayer);
      alpaka::memcpy(queue, cms::alpakatools::make_host_view(hHdr.data(), candCapacity), *dumpHdr);
      alpaka::memcpy(queue, cms::alpakatools::make_host_view(hMem.data(), nMem), *dumpMember);
      alpaka::memcpy(queue, cms::alpakatools::make_host_view(hOvf, 2u), *dumpOvf);
      // Host wait: hLay/hHdr/hMem/hOvf are plain host storage that the census loop below reads
      // immediately. Only reached in an instrumented build (candDumpOn).
      alpaka::wait(queue);

      // Aggregate census (mirrors the finalDedup single-line summary).
      uint32_t nCand = 0, oAcc = 0, oEmpty = 0, oGate = 0, oCap = 0, oOther = 0, nVeto = 0, nRec = 0;
      for (uint32_t s = 0; s < candCapacity; ++s) {
        if (hHdr[s].trackId < 0)
          continue;
        ++nCand;
        for (uint32_t vi = 0; vi < candDumpStride; ++vi) {
          const ExtCandLayerRec& r = hLay[std::size_t(s) * candDumpStride + vi];
          if (r.trackId < 0)
            continue;
          ++nRec;
          switch (r.outcome) {
            case kExtDumpAccept:
              ++oAcc;
              break;
            case kExtDumpRejEmpty:
              ++oEmpty;
              break;
            case kExtDumpRejGate:
              ++oGate;
              break;
            case kExtDumpRejCap:
              ++oCap;
              break;
            default:
              ++oOther;
              break;
          }
          if (r.flags & kExtDumpFlagVetoSkip)
            ++nVeto;
        }
      }
      edm::LogInfo("CAExtensionCandDump")
          << "[extCandDump] SUMMARY nCand=" << nCand << " nRec=" << nRec << " overflow=" << hOvf[0]
          << " memberOverflow=" << hOvf[1] << " outcome{accept=" << oAcc << " rejEmpty=" << oEmpty
          << " rejGate=" << oGate << " rejCap=" << oCap << " rejOther=" << oOther << "} vetoSkips=" << nVeto
          << " stride=" << candDumpStride << " memberCap=" << kExtDumpMaxMembers;

      // Per-record detail: one H line per candidate + one L line per visited layer, batched into modest
      // LogInfo messages (bounds per-message overhead / category throttling on high-volume runs). Fixed
      // field order; every line independently greppable by the "[extCandDump]" prefix regardless of
      // message boundaries.
      std::ostringstream oss;
      uint32_t inChunk = 0;
      constexpr uint32_t kChunkCands = 32u;  // candidates per LogInfo message (bounds message size)
      auto flush = [&]() {
        const std::string msg = oss.str();
        if (!msg.empty())
          edm::LogInfo("CAExtensionCandDump") << msg;
        oss.str("");
        oss.clear();
        inChunk = 0;
      };
      for (uint32_t s = 0; s < candCapacity; ++s) {
        const ExtCandHdrRec& h = hHdr[s];
        if (h.trackId < 0)
          continue;
        oss << "[extCandDump] H slot=" << s << " trk=" << h.trackId << " nVis=" << h.nVisited
            << " hostFlags=" << h.hostFlags << std::hex << " covered=0x" << h.coveredMask << " reachable=0x"
            << h.reachableMask << " visited=0x" << h.visitedMask << std::dec << "\n";
        for (uint32_t vi = 0; vi < candDumpStride; ++vi) {
          const ExtCandLayerRec& r = hLay[std::size_t(s) * candDumpStride + vi];
          if (r.trackId < 0)
            continue;
          oss << "[extCandDump] L slot=" << s << " trk=" << r.trackId << " vi=" << vi << " layer=" << r.layerId
              << " nWinM=" << r.nWinMerged << " nWinOT=" << r.nWinOT << " nPassM=" << r.nPassMerged
              << " nPassOT=" << r.nPassOT << " win=" << r.winnerHitId << " winChi2=" << r.winnerChi2
              << " bestFail=" << r.bestFailChi2 << " round=" << r.round << " outcome=" << r.outcome
              << " flags=" << r.flags << " bestHit=" << r.bestHitId
              << " bestPass=" << ((r.flags & kExtDumpFlagBestPass) ? 1 : 0) << " holeRun=" << r.holeRunToHere << "\n";
          // One M line per road candidate scored on this (slot, layer). hit is the join key to the
          // HitTruth all-hit table (bit30 = raw-OT). Written in atomic-cursor order, not sorted.
          const std::size_t memBase = (std::size_t(s) * candDumpStride + vi) * std::size_t(kExtDumpMaxMembers);
          for (uint32_t m = 0; m < kExtDumpMaxMembers; ++m) {
            const ExtCandMemberRec& mm = hMem[memBase + m];
            if (mm.hitId < 0)
              continue;  // unwritten member slot (0xFF sentinel)
            oss << "[extCandDump] M slot=" << s << " vi=" << vi << " hit=" << mm.hitId << " chi2=" << mm.chi2
                << " round=" << int(mm.round) << " pass=" << int(mm.pass) << "\n";
          }
        }
        if (++inChunk >= kChunkCands)
          flush();
      }
      flush();
    }
  }

  // The post-walk stage: compact the extended candidates, rebuild their arc-sorted hit lists, and
  // rewrite the accepted extensions into the track SoA in place.

  using SeqContainer = caStructures::SequentialContainer;

  // Upper bound on originals + extras per merged list. Sizes the extension container's content
  // capacity (see `contentCap` below) and the per-candidate `sizes[]` bound.
  constexpr int kMaxMergedHits = 32;
  // Scratch bound for the per-track hit-list rebuild in Kernel_extFillContainer. The originals count
  // is the post-twin-union hit count, which the merger caps at kTwinMaxMergedHits (== kMaxMergedHits),
  // and the extras count is bounded by AttachParams::maxExtraHitsPerTrack, so the two stack arrays
  // must hold kMaxMergedHits + maxExtraHitsPerTrack entries: an overrun would be guarded only by an
  // ALPAKA_ASSERT_ACC, which is a no-op in the production (ndebug) build. This constant sizes only the
  // two stack arrays; `contentCap` and `sizes[]` keep using kMaxMergedHits.
  constexpr int kMaxExtraHitsScratch = 8;  // >= AttachParams::maxExtraHitsPerTrack; raise if a cfi does
  constexpr int kMaxRebuildHits = kMaxMergedHits + kMaxExtraHitsScratch;
  // CA-layer lookup (the CA's own partition, for the nLayers recount of accepted extensions).
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int caLayerOf(uint32_t moduleIdx, ::reco::CALayersSoAConstView layers) {
    const int nLayers = layers.metadata().size() - 1;
    int lo = 0;
    int hi = nLayers;
    while (lo < hi) {
      const int mid = (lo + hi) >> 1;
      if (layers.layerStarts()[mid + 1] <= moduleIdx)
        lo = mid + 1;
      else
        hi = mid;
    }
    return lo;
  }

  // Compact the arbitration-surviving extended candidates into the dense extIdx space the rest of the
  // post-walk stage indexes by, carrying the tuple id and attach slot along.
  struct Kernel_extCompact {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const uint32_t* __restrict__ candList,
                                  const uint32_t* __restrict__ nCands,
                                  const uint32_t maxCandidates,
                                  const int32_t* __restrict__ nExtras,
                                  uint32_t* __restrict__ extTuple,
                                  uint32_t* __restrict__ extCandSlot,
                                  uint32_t* __restrict__ nExt,
                                  int32_t* __restrict__ extNExtras) const {
      const uint32_t nC = alpaka::math::min(acc, *nCands, maxCandidates);
      for (auto j : cms::alpakatools::uniform_elements(acc, nC)) {
        if (nExtras[j] <= 0)
          continue;
        const uint32_t extIdx = alpaka::atomicAdd(acc, nExt, 1u, alpaka::hierarchy::Grids{});
        const uint32_t tupleId = candList[j];
        extTuple[extIdx] = tupleId;
        extCandSlot[extIdx] = j;
        extNExtras[extIdx] = nExtras[j];
      }
    }
  };

  // Merged (originals + kept extras) list sizes per extIdx; zero padding beyond nExt so the
  // prefix scan runs over the fixed maxCandidates extent.
  struct Kernel_extSizes {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::TrackSoAConstView tracks,
                                  const uint32_t* __restrict__ extTuple,
                                  const uint32_t* __restrict__ nExt,
                                  const int32_t* __restrict__ extNExtras,
                                  const uint32_t maxCandidates,
                                  int32_t* __restrict__ sizes) const {
      for (auto e : cms::alpakatools::uniform_elements(acc, maxCandidates)) {
        if (e >= *nExt) {
          sizes[e] = 0;
          continue;
        }
        const uint32_t i = extTuple[e];
        const uint32_t begin = (i == 0) ? 0u : tracks[i - 1].hitOffsets();
        const uint32_t end = tracks[i].hitOffsets();
        sizes[e] = int32_t(end - begin) + extNExtras[e];
        ALPAKA_ASSERT_ACC(sizes[e] <= kMaxMergedHits);
      }
    }
  };

  // Build the merged hit container: per extIdx the originals + extras insertion-sorted by arc length
  // along the host's own fitted helix, plus the CSR offsets with sentinel rows past nExt.
  struct Kernel_extFillContainer {
    // OT source (by value) so a tagged extra's arc-sort key reads the OT global position.
    // Default (nOTHits == 0) => no tagged ids can be present and only merged hits are read.
    OTHitsSource otSource_{};

    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  const int maxExtraHitsPerTrack,
                                  const float bf,
                                  ::reco::TrackSoAConstView tracks,
                                  ::reco::TrackHitSoAConstView trackHits,
                                  ::reco::TrackingRecHitConstView hits,
                                  const uint32_t* __restrict__ extTuple,
                                  const uint32_t* __restrict__ extCandSlot,
                                  const uint32_t* __restrict__ nExt,
                                  const int32_t* __restrict__ extNExtras,
                                  const uint32_t maxCandidates,
                                  const uint32_t* __restrict__ extrasIds,
                                  const int32_t* __restrict__ offsets,  // inclusive prefix of sizes
                                  SeqContainer* extContainer) const {
      const uint32_t nE = alpaka::math::min(acc, *nExt, maxCandidates);
      const int32_t total = (maxCandidates > 0) ? offsets[maxCandidates - 1] : 0;
      for (auto k : cms::alpakatools::uniform_elements(acc, maxCandidates + 1)) {
        if (k == 0u)
          extContainer->off[0] = 0;
        else if (k <= nE)
          extContainer->off[k] = uint32_t(offsets[k - 1]);
        else
          extContainer->off[k] = uint32_t(total);
      }
      for (auto e : cms::alpakatools::uniform_elements(acc, nE)) {
        const uint32_t i = extTuple[e];
        const uint32_t j = extCandSlot[e];
        const uint32_t outStart = (e == 0) ? 0u : uint32_t(offsets[e - 1]);

        const HelixState h = makeHelixState(acc, tracks, int(i), bf);
        auto arcOf = [&](uint32_t hitId) {
          float xh, yh;
          if (isOTId(hitId)) {
            const uint32_t o = otIdx(hitId);
            xh = otSource_.otHits[o].xGlobal();
            yh = otSource_.otHits[o].yGlobal();
          } else {
            xh = hits[hitId].xGlobal();
            yh = hits[hitId].yGlobal();
          }
          const float alphaH = alpaka::math::atan2(acc, yh - h.yc, xh - h.xc);
          return h.rho * foldPi(h.alphaOrigin - alphaH);
        };

        const uint32_t hitBegin = (i == 0) ? 0u : tracks[i - 1].hitOffsets();
        const uint32_t hitEnd = tracks[i].hitOffsets();
        const int nOrig = int(hitEnd - hitBegin);
        const int nAdd = extNExtras[e];
        const int nTot = nOrig + nAdd;
        ALPAKA_ASSERT_ACC(nAdd <= kMaxExtraHitsScratch);
        ALPAKA_ASSERT_ACC(nTot <= kMaxRebuildHits);

        uint32_t ids[kMaxRebuildHits];
        float arcs[kMaxRebuildHits];
        for (int a = 0; a < nOrig; ++a) {
          ids[a] = trackHits[hitBegin + a].id();
          arcs[a] = arcOf(ids[a]);
        }
        for (int a = 0; a < nAdd; ++a) {
          ids[nOrig + a] = extrasIds[j * maxExtraHitsPerTrack + a];
          arcs[nOrig + a] = arcOf(ids[nOrig + a]);
        }
        for (int a = 1; a < nTot; ++a) {
          const uint32_t id = ids[a];
          const float s = arcs[a];
          int b = a - 1;
          while (b >= 0 && arcs[b] > s) {
            ids[b + 1] = ids[b];
            arcs[b + 1] = arcs[b];
            --b;
          }
          ids[b + 1] = id;
          arcs[b + 1] = s;
        }
        for (int a = 0; a < nTot; ++a)
          extContainer->content[outStart + a] = ids[a];
      }
    }
  };

  // Admits every walk commit: there is no post-walk re-fit test: a chi2-increase test on the extended hit list would discard a
  // significant share of the correct attachments for a negligible gain in purity, since a background
  // hit that wins an argmin is close by construction and its residual carries almost no information
  // about whether it is the right hit. That population is addressed by the hole hypothesis inside the
  // walk instead. This kernel therefore only fills the scaffold slots the rewrite reads, each
  // extended candidate keeping its walk extras and its original fitted row; the merger's final GBL
  // refit is what re-fits the extended hit list.
  struct Kernel_extAcceptAll {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::TrackSoAConstView tracks,
                                  const uint32_t* __restrict__ extTuple,
                                  const uint32_t* __restrict__ nExt,
                                  const uint32_t candCapacity,
                                  const uint32_t* __restrict__ extCandSlot,
                                  const int32_t* __restrict__ nExtras,
                                  int32_t* __restrict__ extNExtras,
                                  float* __restrict__ extNewState,
                                  float* __restrict__ extNewCov,
                                  float* __restrict__ extNewChi2,
                                  int32_t* __restrict__ extNewNdof,
                                  int32_t* __restrict__ acceptedByTuple,
                                  uint32_t* __restrict__ stats) const {
      const uint32_t nE = alpaka::math::min(acc, *nExt, candCapacity);
      for (auto e : cms::alpakatools::uniform_elements(acc, nE)) {
        const uint32_t i = extTuple[e];
        const uint32_t j = extCandSlot[e];
        const int32_t nA = nExtras[j];
        if (nA <= 0)
          continue;
        extNExtras[e] = nA;
        for (int a = 0; a < 5; ++a)
          extNewState[e * 5u + uint32_t(a)] = tracks[i].state()(a);
        for (int m = 0; m < 15; ++m)
          extNewCov[e * 15u + uint32_t(m)] = tracks[i].covariance()(m);
        extNewChi2[e] = tracks[i].chi2();
        extNewNdof[e] = tracks[i].ndof();
        acceptedByTuple[i] = int32_t(e);
        alpaka::atomicAdd(acc, &stats[kStatAccepted], 1u, alpaka::hierarchy::Grids{});
      }
    }
  };

  // Snapshot the original hit lists (ids + detIds + attached flags + per-tuple ends) before the
  // in-place re-layout. snapAttached is what lets a follow-on attach pass preserve the attach marks an
  // earlier pass set.
  struct Kernel_extSnapshot {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::TrackSoAConstView tracks,
                                  ::reco::TrackHitSoAConstView trackHits,
                                  const uint32_t maxNumberOfTuples,
                                  uint32_t* __restrict__ snapIds,
                                  uint32_t* __restrict__ snapDetIds,
                                  uint8_t* __restrict__ snapAttached,
                                  uint32_t* __restrict__ snapEnds) const {
      const uint32_t nT = alpaka::math::min(acc, maxNumberOfTuples, uint32_t(std::max(0, tracks.nTracks())));
      const uint32_t totalHits = nT > 0u ? tracks[nT - 1].hitOffsets() : 0u;
      for (auto i : cms::alpakatools::uniform_elements(acc, nT))
        snapEnds[i] = tracks[i].hitOffsets();
      for (auto k : cms::alpakatools::uniform_elements(acc, totalHits)) {
        snapIds[k] = trackHits[k].id();
        snapDetIds[k] = trackHits[k].detId();
        snapAttached[k] = uint8_t(trackHits[k].attached());
      }
    }
  };

  // Final per-tuple list sizes after acceptance (zero padding beyond nTracks for the prefix scan).
  struct Kernel_extFinalSizes {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::TrackSoAConstView tracks,
                                  const int32_t* __restrict__ acceptedByTuple,
                                  const int32_t* __restrict__ extNExtras,
                                  const uint32_t maxNumberOfTuples,
                                  int32_t* __restrict__ finalSizes) const {
      const uint32_t nT = alpaka::math::min(acc, maxNumberOfTuples, uint32_t(std::max(0, tracks.nTracks())));
      for (auto i : cms::alpakatools::uniform_elements(acc, maxNumberOfTuples)) {
        if (i >= nT) {
          finalSizes[i] = 0;
          continue;
        }
        const uint32_t begin = (i == 0) ? 0u : tracks[i - 1].hitOffsets();
        const int nOrig = int(tracks[i].hitOffsets() - begin);
        const int32_t e = acceptedByTuple[i];
        finalSizes[i] = nOrig + ((e >= 0) ? extNExtras[e] : 0);
      }
    }
  };

  // Capacity guard: if the re-laid-out lists would overflow the hit-list allocation, flag the
  // event for whole-event fallback (nothing is rewritten; counted and reported).
  struct Kernel_extCapacityCheck {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::TrackSoAConstView tracks,
                                  const int32_t* __restrict__ finalOffsets,
                                  const uint32_t maxNumberOfTuples,
                                  const uint32_t hitCapacity,
                                  uint32_t* __restrict__ stats) const {
      if (alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0] != 0)
        return;
      const uint32_t nT = alpaka::math::min(acc, maxNumberOfTuples, uint32_t(std::max(0, tracks.nTracks())));
      if (nT > 0u && uint32_t(finalOffsets[nT - 1]) > hitCapacity)
        stats[kStatRewriteOverflow] = 1u;
    }
  };

  // The in-place re-layout: new hitOffsets + TrackHitSoA rows + hit-container offsets/content for
  // every tuple (accepted tuples from the merged sorted lists + the scratch fit row + an nLayers
  // recount; everything else verbatim from the snapshot). Skipped entirely on capacity fallback.
  struct Kernel_extWriteFinal {
    // OT source (by value) so a tagged extra's detId is fetched from the raw OT SoA. Default
    // (nOTHits == 0) => no tagged ids can be present and only merged hits are read.
    OTHitsSource otSource_{};

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::TrackSoAView tracks,
                                  ::reco::TrackHitSoAView trackHits,
                                  const float* __restrict__ extNewState,
                                  const float* __restrict__ extNewCov,
                                  const float* __restrict__ extNewChi2,
                                  const int32_t* __restrict__ extNewNdof,
                                  ::reco::TrackingRecHitConstView hits,
                                  ::reco::CALayersSoAConstView caLayers,
                                  SeqContainer* hitContainer,
                                  SeqContainer const* __restrict__ extContainer,
                                  const int maxExtraHitsPerTrack,
                                  const uint32_t* __restrict__ extCandSlot,
                                  const int32_t* __restrict__ extNExtras,
                                  const uint32_t* __restrict__ extrasIds,
                                  const int32_t* __restrict__ acceptedByTuple,
                                  const uint32_t* __restrict__ snapIds,
                                  const uint32_t* __restrict__ snapDetIds,
                                  const uint8_t* __restrict__ snapAttached,
                                  const uint32_t* __restrict__ snapEnds,
                                  const int32_t* __restrict__ finalOffsets,
                                  const uint32_t maxNumberOfTuples,
                                  uint32_t* __restrict__ stats) const {
      if (stats[kStatRewriteOverflow] != 0u)
        return;
      const uint32_t nT = alpaka::math::min(acc, maxNumberOfTuples, uint32_t(std::max(0, tracks.nTracks())));
      const uint32_t total = nT > 0u ? uint32_t(finalOffsets[nT - 1]) : 0u;
      // CSR offsets of the (rewritten) hit container, sentinels past the live tuples.
      for (auto k : cms::alpakatools::uniform_elements(acc, maxNumberOfTuples + 1)) {
        if (k == 0u)
          hitContainer->off[0] = 0;
        else if (k <= nT)
          hitContainer->off[k] = uint32_t(finalOffsets[k - 1]);
        else
          hitContainer->off[k] = total;
      }
      for (auto i : cms::alpakatools::uniform_elements(acc, nT)) {
        const uint32_t outStart = (i == 0) ? 0u : uint32_t(finalOffsets[i - 1]);
        const uint32_t outEnd = uint32_t(finalOffsets[i]);
        tracks[i].hitOffsets() = outEnd;
        const int32_t e = acceptedByTuple[i];
        if (e >= 0) {
          const uint32_t srcBegin = extContainer->off[e];
          const uint32_t extrasBase = uint32_t(extCandSlot[e]) * uint32_t(maxExtraHitsPerTrack);
          const int nAdd = extNExtras[e];
          // The snapshot range of this tuple, to preserve attach marks written by an earlier pass.
          // On a first pass every snapshot flag is 0, so the lookup below is a no-op.
          const uint32_t snapBegin = (i == 0) ? 0u : snapEnds[i - 1];
          const uint32_t snapEnd = snapEnds[i];
          uint16_t nCALayers = 0;
          int prevLayer = -1;
          // detectorIndex for either source; both use the same CA module numbering (OT = nPixel + geom),
          // so caLayerOf works for both. The tagged id itself is kept on trackHits[d].id() (downstream
          // OT-aware consumers key on it).
          auto detOf = [&](uint32_t theId) -> uint32_t {
            return isOTId(theId) ? uint32_t(otSource_.otHits[otIdx(theId)].detectorIndex())
                                 : hits[theId].detectorIndex();
          };
          for (uint32_t d = outStart, s = srcBegin; d < outEnd; ++d, ++s) {
            const uint32_t id = extContainer->content[s];
            const uint32_t det = detOf(id);
            trackHits[d].id() = id;
            trackHits[d].detId() = det;
            // isNewExtra: added by this pass, which is what the stats count. wasAttached: a mark
            // carried in from an earlier pass -- preserved, but never re-counted. On a first pass every
            // snapshot flag is 0, so wasAttached is always false.
            bool isNewExtra = false;
            for (int a = 0; a < nAdd && !isNewExtra; ++a)
              isNewExtra = (extrasIds[extrasBase + a] == id);
            bool wasAttached = false;
            if (!isNewExtra) {
              // id-lookup over the snapshot range: the merged list is arc-sorted, so the container
              // position does not map to the snapshot position. nOrig <= kMaxMergedHits.
              for (uint32_t q = snapBegin; q < snapEnd && !wasAttached; ++q)
                wasAttached = (snapIds[q] == id && snapAttached[q] != 0);
            }
            trackHits[d].attached() = (isNewExtra || wasAttached) ? 1 : 0;
            if (isNewExtra && isOTId(id))  // diagnostic: tagged OT hit written into an accepted tuple
              alpaka::atomicAdd(acc, &stats[kStatOTWritten], 1u, alpaka::hierarchy::Grids{});
            if (isNewExtra && !isOTId(id))  // diagnostic: merged extra written into an accepted tuple
              alpaka::atomicAdd(acc, &stats[kDiagMergedWritten], 1u, alpaka::hierarchy::Grids{});
            hitContainer->content[d] = id;
            // merged lists are arc-sorted, not layer-sorted, so count distinct layers exactly
            const int layer = caLayerOf(det, caLayers);
            if (layer != prevLayer) {
              bool seen = false;
              for (uint32_t q = outStart; q < d && !seen; ++q)
                seen = (caLayerOf(detOf(hitContainer->content[q]), caLayers) == layer);
              if (!seen)
                ++nCALayers;
              prevLayer = layer;
            }
          }
          // state/cov/chi2 from the scaffold row; kinematics recomputed from that state
          // (state(2) holds signed 1/pt, same convention as Kernel_BLFit: pt = 1/|state(2)|).
          for (int a = 0; a < 5; ++a)
            tracks[i].state()(a) = extNewState[e * 5u + uint32_t(a)];
          for (int c = 0; c < 15; ++c)
            tracks[i].covariance()(c) = extNewCov[e * 15u + uint32_t(c)];
          tracks[i].pt() = 1.f / alpaka::math::max(acc, alpaka::math::abs(acc, extNewState[e * 5u + 2u]), 1e-9f);
          tracks[i].eta() = alpaka::math::asinh(acc, extNewState[e * 5u + 3u]);
          tracks[i].chi2() = extNewChi2[e];
          tracks[i].ndof() = extNewNdof[e];
          tracks[i].nLayers() = nCALayers;
        } else {
          const uint32_t srcBegin = (i == 0) ? 0u : snapEnds[i - 1];
          const uint32_t srcEnd = snapEnds[i];
          for (uint32_t d = outStart, s = srcBegin; s < srcEnd; ++d, ++s) {
            trackHits[d].id() = snapIds[s];
            trackHits[d].detId() = snapDetIds[s];
            // Restore the pre-rewrite flag: all-zero on a first pass, and on a follow-on pass this is
            // what preserves the earlier pass's attach marks on unmodified tuples.
            trackHits[d].attached() = snapAttached[s];
            hitContainer->content[d] = snapIds[s];
          }
        }
      }
    }
  };

  // Per-event accept-stage summary appended to the attach line.
  struct Kernel_extPrintAcceptSummary {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, const uint32_t* __restrict__ stats) const {
      if (alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0] != 0)
        return;
      printf(
          "[CAExtension] accept: accepted=%u capacityFallback=%u\n", stats[kStatAccepted], stats[kStatRewriteOverflow]);
      // The OT-vs-merged attachment ledger: where extras are committed, lost and written. The OT
      // columns read 0 when no OT source is active.
      printf(
          "[CAExtension][DIAG] walk(merged=%u OT=%u) otPartner=%u slotExhaust(tot=%u withOT=%u) "
          "arbLost(merged=%u OT=%u) "
          "mergedWritten=%u OTwritten=%u\n",
          stats[kDiagWalkMerged],
          stats[kDiagWalkOT],
          stats[kStatOTPartner],
          stats[kDiagSlotExhaust],
          stats[kDiagSlotExhaustOT],
          stats[kDiagArbLostMerged],
          stats[kDiagArbLostOT],
          stats[kDiagMergedWritten],
          stats[kStatOTWritten]);
      // Mean permil decomposition of the endcap gate variance sigSec2 over the considered disk hits,
      // per source round. With secFracDiag_ off the counts are zero and the line prints n=0 all-zeros.
      const uint32_t nsm = stats[kDiagSecNM] ? stats[kDiagSecNM] : 1u;
      const uint32_t nso = stats[kDiagSecNOT] ? stats[kDiagSecNOT] : 1u;
      printf(
          "[CAExtension][SIGSEC] endcap sigSec2 permil (merged n=%u: hit=%u ms=%u pred=%u align=%u | "
          "OT n=%u: hit=%u ms=%u pred=%u align=%u)\n",
          stats[kDiagSecNM],
          stats[kDiagSecFracHitM] / nsm,
          stats[kDiagSecFracMsM] / nsm,
          stats[kDiagSecFracPredM] / nsm,
          stats[kDiagSecFracAlignM] / nsm,
          stats[kDiagSecNOT],
          stats[kDiagSecFracHitOT] / nso,
          stats[kDiagSecFracMsOT] / nso,
          stats[kDiagSecFracPredOT] / nso,
          stats[kDiagSecFracAlignOT] / nso);
      // Merged-endcap predSecVar circle/line split plus the exact-5x5 shadow projection
      // (predSecVar_full), all as permil of the same sigSec2, bucketed by prior disk accepts.
      const uint32_t nd0 = stats[kDiagNDisk0M] ? stats[kDiagNDisk0M] : 1u;
      const uint32_t nd1 = stats[kDiagNDisk1M] ? stats[kDiagNDisk1M] : 1u;
      printf(
          "[CAExtension][SIGSEC-PRED] merged-endcap predSec permil-of-sigSec2 (n=%u: circle=%u line=%u "
          "bd=%u full=%u | disk0 n=%u bd=%u full=%u | disk1+ n=%u bd=%u full=%u)\n",
          stats[kDiagSecNM],
          stats[kDiagPredCircleM] / nsm,
          stats[kDiagPredLineM] / nsm,
          stats[kDiagSecFracPredM] / nsm,
          stats[kDiagPredFullM] / nsm,
          stats[kDiagNDisk0M],
          stats[kDiagPredBd0M] / nd0,
          stats[kDiagPredFull0M] / nd0,
          stats[kDiagNDisk1M],
          stats[kDiagPredBd1M] / nd1,
          stats[kDiagPredFull1M] / nd1);
    }
  };

  // One-shot per-event OT summary. Launched only when the OT source is active, so on a merged-only
  // run this line never appears.
  struct Kernel_extPrintOTSummary {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, const uint32_t* __restrict__ stats) const {
      if (alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0] != 0)
        return;
      printf("[CAExtension][OT] otHitsWritten=%u (tagged OT rechits added to accepted tuples)\n",
             stats[kStatOTWritten]);
    }
  };

  RefitScaffold buildRefitScaffold(Queue& queue,
                                   const AttachParams& params,
                                   float bf,
                                   ::reco::TrackSoAConstView tracks,
                                   ::reco::TrackHitSoAConstView trackHits,
                                   ::reco::TrackingRecHitConstView hits,
                                   uint32_t maxNumberOfTuples,
                                   uint32_t hitCapacity,
                                   AttachBuffers& bufs,
                                   const OTHitsSource* otSource) {
    const OTHitsSource otSrcVal = (otSource != nullptr && otSource->nOTHits > 0u) ? *otSource : OTHitsSource{};
    // Runtime candidate capacity (the count pass in allocateAttachBuffers sized bufs to it): every
    // stage-B scratch array and candidate-loop grid is sized/bounded by this, not params.maxCandidates.
    const uint32_t candCapacity = bufs.candCapacity;
    const auto mc = std::size_t(candCapacity);
    const auto contentCap = mc * std::size_t(kMaxMergedHits);
    RefitScaffold s{
        cms::alpakatools::make_device_buffer<uint32_t[]>(queue, std::max<std::size_t>(1, mc)),
        cms::alpakatools::make_device_buffer<uint32_t[]>(queue, std::max<std::size_t>(1, mc)),
        // nExt + pfxCounter + pfxCounter2 in one 3-word allocation (see RefitScaffold).
        cms::alpakatools::make_device_buffer<uint32_t[]>(queue, 3),
        cms::alpakatools::make_device_buffer<int32_t[]>(queue, std::max<std::size_t>(1, mc)),
        cms::alpakatools::make_device_buffer<SeqContainer>(queue),
        cms::alpakatools::make_device_buffer<SeqContainer::Counter[]>(queue, mc + 1),
        cms::alpakatools::make_device_buffer<SeqContainer::value_type[]>(queue, contentCap + 1),
        cms::alpakatools::make_device_buffer<float[]>(queue, 5 * std::max<std::size_t>(1, mc)),
        cms::alpakatools::make_device_buffer<float[]>(queue, 15 * std::max<std::size_t>(1, mc)),
        cms::alpakatools::make_device_buffer<float[]>(queue, std::max<std::size_t>(1, mc)),
        cms::alpakatools::make_device_buffer<int32_t[]>(queue, std::max<std::size_t>(1, mc)),
        cms::alpakatools::make_device_buffer<int32_t[]>(queue, std::max<std::size_t>(1, mc)),
        cms::alpakatools::make_device_buffer<int32_t[]>(queue, std::max<std::size_t>(1, mc)),
        cms::alpakatools::make_device_buffer<int32_t[]>(queue, std::max<std::size_t>(1, maxNumberOfTuples)),
        cms::alpakatools::make_device_buffer<uint32_t[]>(queue, std::max<std::size_t>(1, hitCapacity)),
        cms::alpakatools::make_device_buffer<uint32_t[]>(queue, std::max<std::size_t>(1, hitCapacity)),
        // snapAttached: the pre-rewrite attached() flags (see the struct comment)
        cms::alpakatools::make_device_buffer<uint8_t[]>(queue, std::max<std::size_t>(1, hitCapacity)),
        cms::alpakatools::make_device_buffer<uint32_t[]>(queue, std::max<std::size_t>(1, maxNumberOfTuples)),
        cms::alpakatools::make_device_buffer<int32_t[]>(queue, std::max<std::size_t>(1, maxNumberOfTuples)),
        cms::alpakatools::make_device_buffer<int32_t[]>(queue, std::max<std::size_t>(1, maxNumberOfTuples))};
    alpaka::memset(queue, s.counters, 0);            // one memset for all three words
    alpaka::memset(queue, s.acceptedByTuple, 0xff);  // -1 = no accepted extension

    typename SeqContainer::View extContainerView{s.extContainer.data(),
                                                 s.extContainerOffsets.data(),
                                                 s.extContainerStorage.data(),
                                                 uint32_t(mc + 1),
                                                 uint32_t(contentCap + 1)};
    SeqContainer::template launchZero<Acc1D>(extContainerView, queue);
    constexpr uint32_t blockSize = 256;
    const uint32_t blocksC = cms::alpakatools::divide_up_by(candCapacity, blockSize);
    const auto wdC = cms::alpakatools::make_workdiv<Acc1D>(blocksC, blockSize);
    const uint32_t blocksC1 = cms::alpakatools::divide_up_by(candCapacity + 1, blockSize);
    const auto wdC1 = cms::alpakatools::make_workdiv<Acc1D>(blocksC1, blockSize);

    alpaka::exec<Acc1D>(queue,
                        wdC,
                        Kernel_extCompact{},
                        bufs.candList.data(),
                        bufs.nCands.data(),
                        candCapacity,
                        bufs.nExtras.data(),
                        s.extTuple.data(),
                        s.extCandSlot.data(),
                        s.nExt(),
                        s.extNExtras.data());

    alpaka::exec<Acc1D>(queue,
                        wdC,
                        Kernel_extSizes{},
                        tracks,
                        s.extTuple.data(),
                        s.nExt(),
                        s.extNExtras.data(),
                        candCapacity,
                        s.sizes.data());

    alpaka::exec<Acc1D>(queue,
                        wdC,
                        cms::alpakatools::multiBlockPrefixScan<int32_t>(),
                        s.sizes.data(),
                        s.offsets.data(),
                        int32_t(candCapacity),
                        int(blocksC),
                        s.pfxCounter(),
                        alpaka::getPreferredWarpSize(alpaka::getDev(queue)));

    alpaka::exec<Acc1D>(queue,
                        wdC1,
                        Kernel_extFillContainer{otSrcVal},
                        params.maxExtraHitsPerTrack,
                        bf,
                        tracks,
                        trackHits,
                        hits,
                        s.extTuple.data(),
                        s.extCandSlot.data(),
                        s.nExt(),
                        s.extNExtras.data(),
                        candCapacity,
                        bufs.extrasIds.data(),
                        s.offsets.data(),
                        s.extContainer.data());

    return s;
  }

  void launchVerifyRewrite(Queue& queue,
                           const AttachParams& params,
                           ::reco::TrackSoAView tracks,
                           ::reco::TrackHitSoAView trackHits,
                           ::reco::TrackingRecHitConstView hits,
                           ::reco::CALayersSoAConstView caLayers,
                           ::reco_extender::ExtenderLayersConstView extLayers,
                           const ::reco::CAModulesConstView modules,
                           const float* rhoMap,
                           float bfield,
                           caStructures::SequentialContainer* hitContainer,
                           uint32_t maxNumberOfTuples,
                           uint32_t hitCapacity,
                           RefitScaffold& scaffold,
                           AttachBuffers& bufs,
                           const OTHitsSource* otSource) {
    const OTHitsSource otSrcVal = (otSource != nullptr && otSource->nOTHits > 0u) ? *otSource : OTHitsSource{};
    // Runtime candidate capacity (see allocateAttachBuffers/buildRefitScaffold): bounds the
    // per-candidate grid + extent. Per-tuple grids (wdT/wdT1) stay on maxNumberOfTuples.
    const uint32_t candCapacity = bufs.candCapacity;
    constexpr uint32_t blockSize = 256;
    const uint32_t blocksC = cms::alpakatools::divide_up_by(candCapacity, blockSize);
    const auto wdC = cms::alpakatools::make_workdiv<Acc1D>(blocksC, blockSize);
    const uint32_t blocksT = cms::alpakatools::divide_up_by(maxNumberOfTuples, blockSize);
    const auto wdT = cms::alpakatools::make_workdiv<Acc1D>(blocksT, blockSize);
    const uint32_t blocksT1 = cms::alpakatools::divide_up_by(maxNumberOfTuples + 1, blockSize);
    const auto wdT1 = cms::alpakatools::make_workdiv<Acc1D>(blocksT1, blockSize);

    // Accept every arbitration-surviving extra: the per-hit gate has already priced each one against
    // its own prediction covariance and against the hole hypothesis, so no separate post-walk test is
    // applied.
    alpaka::exec<Acc1D>(queue,
                        wdC,
                        Kernel_extAcceptAll{},
                        tracks,
                        scaffold.extTuple.data(),
                        scaffold.nExt(),
                        candCapacity,
                        scaffold.extCandSlot.data(),
                        bufs.nExtras.data(),
                        scaffold.extNExtras.data(),
                        scaffold.extNewState.data(),
                        scaffold.extNewCov.data(),
                        scaffold.extNewChi2.data(),
                        scaffold.extNewNdof.data(),
                        scaffold.acceptedByTuple.data(),
                        bufs.stats.data());

    alpaka::exec<Acc1D>(queue,
                        wdT,
                        Kernel_extSnapshot{},
                        tracks,
                        trackHits,
                        maxNumberOfTuples,
                        scaffold.snapIds.data(),
                        scaffold.snapDetIds.data(),
                        scaffold.snapAttached.data(),
                        scaffold.snapEnds.data());

    alpaka::exec<Acc1D>(queue,
                        wdT,
                        Kernel_extFinalSizes{},
                        tracks,
                        scaffold.acceptedByTuple.data(),
                        scaffold.extNExtras.data(),
                        maxNumberOfTuples,
                        scaffold.finalSizes.data());

    alpaka::exec<Acc1D>(queue,
                        wdT,
                        cms::alpakatools::multiBlockPrefixScan<int32_t>(),
                        scaffold.finalSizes.data(),
                        scaffold.finalOffsets.data(),
                        int32_t(maxNumberOfTuples),
                        int(blocksT),
                        scaffold.pfxCounter2(),
                        alpaka::getPreferredWarpSize(alpaka::getDev(queue)));

    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(1u, 1u),
                        Kernel_extCapacityCheck{},
                        tracks,
                        scaffold.finalOffsets.data(),
                        maxNumberOfTuples,
                        hitCapacity,
                        bufs.stats.data());

    alpaka::exec<Acc1D>(queue,
                        wdT1,
                        Kernel_extWriteFinal{otSrcVal},
                        tracks,
                        trackHits,
                        scaffold.extNewState.data(),
                        scaffold.extNewCov.data(),
                        scaffold.extNewChi2.data(),
                        scaffold.extNewNdof.data(),
                        hits,
                        caLayers,
                        hitContainer,
                        scaffold.extContainer.data(),
                        params.maxExtraHitsPerTrack,
                        scaffold.extCandSlot.data(),
                        scaffold.extNExtras.data(),
                        bufs.extrasIds.data(),
                        scaffold.acceptedByTuple.data(),
                        scaffold.snapIds.data(),
                        scaffold.snapDetIds.data(),
                        scaffold.snapAttached.data(),
                        scaffold.snapEnds.data(),
                        scaffold.finalOffsets.data(),
                        maxNumberOfTuples,
                        bufs.stats.data());

    if (params.verbose) {
      const auto oneThread = cms::alpakatools::make_workdiv<Acc1D>(1u, 1u);
      alpaka::exec<Acc1D>(queue, oneThread, Kernel_extPrintAcceptSummary{}, bufs.stats.data());
      if (otSrcVal.nOTHits > 0u)
        alpaka::exec<Acc1D>(queue, oneThread, Kernel_extPrintOTSummary{}, bufs.stats.data());
    }
  }

  // Merger-side attach: the one entry point of the stage. Builds the search structures the merger does
  // not already hold, then drives the attach pipeline over the merged collection.

  // Per-CA-layer pixel-hit-row offsets over the shared rechit SoA (mirror of the CA's SetHitsLayerStart
  // in CAHitNtupletGeneratorKernelsImpl.h): hitsLayerStart[i] = moduleStart[caLayers.layerStarts()[i]].
  // It seeds the merged-hit ExtPhiBinner with the same 256-bin per-CA-layer partition prepareHits
  // builds -- so a merger-attach candidate's per-layer phi window scan is identical to the CA's.
  struct Kernel_mergerHitSetLayerStart {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::HitModuleSoAConstView mm,
                                  ::reco::CALayersSoAConstView ll,
                                  uint32_t* __restrict__ hitsLayerStart) const {
      for (auto i : cms::alpakatools::uniform_elements(acc, uint32_t(ll.metadata().size())))
        hitsLayerStart[i] = mm.moduleStart()[ll.layerStarts()[i]];
    }
  };

  // Diagnostic coverage histogram: per CA layer, total hits on the (post-attach) merged tracks and how
  // many of them carry the attached() flag, i.e. were placed by this stage. Iterates per track over the
  // compacted CSR hit span, so the unused hit-capacity tail is never read. Only launched when
  // params.verbose; it is what makes the odd OT disks' (CA layers ~34-53) coverage checkable.
  struct Kernel_mergerCoverage {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::reco::TrackSoAConstView tracks,
                                  ::reco::TrackHitSoAConstView trackHits,
                                  ::reco::CALayersSoAConstView caLayers,
                                  uint32_t nTracksCap,
                                  uint32_t* __restrict__ hitsByLayer,
                                  uint32_t* __restrict__ attachedByLayer) const {
      const uint32_t nT = alpaka::math::min(acc, nTracksCap, uint32_t(std::max(0, tracks.nTracks())));
      for (auto i : cms::alpakatools::uniform_elements(acc, nT)) {
        const uint32_t beg = (i == 0u) ? 0u : tracks[i - 1].hitOffsets();
        const uint32_t end = tracks[i].hitOffsets();
        for (uint32_t d = beg; d < end; ++d) {
          const int L = caLayerOf(trackHits[d].detId(), caLayers);
          alpaka::atomicAdd(acc, &hitsByLayer[L], 1u, alpaka::hierarchy::Grids{});
          if (trackHits[d].attached() != 0u)
            alpaka::atomicAdd(acc, &attachedByLayer[L], 1u, alpaka::hierarchy::Grids{});
        }
      }
    }
  };

  void launchMergerAttach(Queue& queue,
                          const AttachParams& params,
                          float bfield,
                          const float* rhoMap,
                          ::reco::TrackSoAView tracks,
                          ::reco::TrackHitSoAView trackHits,
                          const uint8_t* armId,  // pocket gate: per-track arm (0=prompt,1=disp); null=off/arm-blind
                          ::reco::TrackingRecHitConstView hits,
                          ::reco::HitModuleSoAConstView hitModules,
                          ::reco::TrackingRecHitsMaskingConstView hitMask,
                          ::reco::CALayersSoAConstView caLayers,
                          ::reco::CAModulesConstView modules,
                          ::reco_extender::ExtenderLayersConstView extLayers,
                          uint32_t nHits,
                          uint32_t nTracksCap,
                          uint32_t hitCapacity,
                          const OTHitsSource* otSource,
                          int32_t* extendedMaskOut) {
    using namespace cms::alpakatools;
    if (nTracksCap == 0u || nHits == 0u)
      return;
    const uint32_t nLayersP1 = uint32_t(caLayers.metadata().size());  // nLayers + 1
    const uint32_t nOTForClaims = (otSource != nullptr && otSource->nOTHits > 0u) ? otSource->nOTHits : 0u;
    const float* rho = rhoMap;

    // (a) merged-hit ExtPhiBinner over the shared pixelRecHits SoA. Pure fn of the rechit iphi column +
    // the CA-layer partition (layerStart); identical convention to the CA prepareHits() phi binner.
    auto layerStart = make_device_buffer<uint32_t[]>(queue, nLayersP1);
    auto phiHist = make_device_buffer<ExtPhiBinner>(queue);
    auto phiStorage = make_device_buffer<ExtPhiBinner::value_type[]>(queue, nHits);
    {
      constexpr uint32_t kThreads = 256u;
      alpaka::exec<Acc1D>(queue,
                          make_workdiv<Acc1D>(divide_up_by(nLayersP1, kThreads), kThreads),
                          Kernel_mergerHitSetLayerStart{},
                          hitModules,
                          caLayers,
                          layerStart.data());
    }
    ExtPhiBinner::View phiView{phiHist.data(), nullptr, phiStorage.data(), cms::alpakatools::kDynamicSize, nHits};
    fillManyFromVector<Acc1D>(phiHist.data(),
                              phiView,
                              ::pixelTopology::Phase2OTStubs::numberOfLayers,
                              hits.iphi().data(),
                              layerStart.data(),
                              nHits,
                              (uint32_t)256,
                              queue);

    // (b) mutable per-tuple SequentialContainer WriteFinal re-lays the extended hit lists into (both
    // the offsets and the content are fully overwritten by the rewrite, so it only needs to be sized +
    // initialised here -- no pre-fill from the merged trackHits is required). Sized to the same hit
    // capacity as the merged SoA so a rewrite that fits the SoA fits the container too.
    const uint32_t contentCap = std::max(hitCapacity, 1u);
    auto contHeader = make_device_buffer<SeqContainer>(queue);
    auto offBuf = make_device_buffer<SeqContainer::Counter[]>(queue, std::size_t(nTracksCap) + 1u);
    auto contentBuf = make_device_buffer<SeqContainer::value_type[]>(queue, contentCap);
    typename SeqContainer::View contView{
        contHeader.data(), offBuf.data(), contentBuf.data(), nTracksCap + 1u, contentCap};
    SeqContainer::template launchZero<Acc1D>(contView, queue);

    // (c) all-fitted pre-gate: every merged row is a fitted HP-selected track, so the pre-gate's
    // never-fitted sentinel (passBuf slot 2, the fitted-circle radius) can never fire here.
    // Kernel_extPreGate reads a null passBuf as exactly that statement: no row is never-fitted.
    const double* const passBuf = nullptr;

    // The attach pipeline: pre-gate/compact -> walk + arbitration -> scaffold -> accept + in-place
    // rewrite. The walk seeds from the merged tracks' fitted running-helix state and scans the full
    // attachable layer set including the odd OT disks.
    // knownCandCapacity = nTracksCap: the candidate set is a subset of the tuples [0, nTracksCap),
    // the range the count/fill pre-gate iterates, so the track capacity is a host-known upper bound
    // and no candidate-count readback is needed. The cost of the looser bound is memory and grid
    // width, not correctness: candCapacity also sizes the stage-B RefitScaffold arrays and the
    // per-candidate launch grids.
    auto extBufs =
        allocateAttachBuffers(queue, params, nHits, nOTForClaims, nTracksCap, tracks, passBuf, nullptr, nTracksCap);
    launchAttach(queue,
                 params,
                 bfield,
                 rho,
                 phiHist.data(),
                 tracks,
                 trackHits,
                 armId,  // pocket gate: forwarded per-track arm (null when off/arm-blind)
                 hits,
                 hitModules,
                 hitMask,
                 extLayers,
                 modules,
                 passBuf,
                 nTracksCap,
                 nHits,
                 extBufs,
                 otSource);
#ifdef CA_SIZING_DUMP
    // Per-event demand dump of the attach stage, same line form as the CA producer's. nCands now holds
    // the fill pass's candidate count, i.e. the number of merged tracks that cleared the pre-gate, and
    // candCapacity is what the candidate scratch, the refit scaffold and the per-candidate grids were
    // sized to. Fitting the ceiling (extRefitMaxCandidates) needs exactly this pair. One 4-byte D2H and
    // one host wait, compiled in only under the toggle.
    {
      auto nCandsHost = cms::alpakatools::make_host_buffer<uint32_t>(queue);
      alpaka::memcpy(queue, nCandsHost, extBufs.nCands);
      alpaka::wait(queue);
      printf("[CA Sizing] iter=mergerAttach nCands=%u capCands=%u\n", *nCandsHost.data(), extBufs.candCapacity);
    }
#endif
    auto scaffold =
        buildRefitScaffold(queue, params, bfield, tracks, trackHits, hits, nTracksCap, hitCapacity, extBufs, otSource);
    launchVerifyRewrite(queue,
                        params,
                        tracks,
                        trackHits,
                        hits,
                        caLayers,
                        extLayers,
                        modules,
                        rho,
                        bfield,
                        contHeader.data(),
                        nTracksCap,
                        hitCapacity,
                        scaffold,
                        extBufs,
                        otSource);

    // Expose this pass's per-tuple extended mask (acceptedByTuple: candidate index >= 0 iff the tuple
    // gained an accepted extra, -1 otherwise). It lets the merger scope a sharpening refit to the
    // extended subset only; non-extended tracks keep their state and are still covered by the
    // unconditional final refit. Sized to nTracksCap (== acceptedByTuple's maxNumberOfTuples).
    // Null => not written.
    if (extendedMaskOut != nullptr)
      alpaka::memcpy(queue,
                     make_device_view(queue, extendedMaskOut, nTracksCap),
                     make_device_view(queue, scaffold.acceptedByTuple.data(), nTracksCap));

    // Per-CA-layer attach coverage (verbose only): OT hits available in the scan domain (from the OT
    // source's per-CA-layer offsets) + hits/attached per layer on the post-attach merged tracks. Proves
    // the odd OT disks are covered. One host sync (the merger already syncs freely on this path).
    if (params.verbose) {
      auto hitsByLayer = make_device_buffer<uint32_t[]>(queue, nLayersP1);
      auto attachedByLayer = make_device_buffer<uint32_t[]>(queue, nLayersP1);
      alpaka::memset(queue, hitsByLayer, 0);
      alpaka::memset(queue, attachedByLayer, 0);
      constexpr uint32_t kThreads = 256u;
      alpaka::exec<Acc1D>(queue,
                          make_workdiv<Acc1D>(divide_up_by(nTracksCap, kThreads), kThreads),
                          Kernel_mergerCoverage{},
                          tracks,
                          trackHits,
                          caLayers,
                          nTracksCap,
                          hitsByLayer.data(),
                          attachedByLayer.data());
      std::vector<uint32_t> hHits(nLayersP1), hAtt(nLayersP1), hOT(nLayersP1, 0u);
      alpaka::memcpy(
          queue, make_host_view(hHits.data(), nLayersP1), make_device_view(queue, hitsByLayer.data(), nLayersP1));
      alpaka::memcpy(
          queue, make_host_view(hAtt.data(), nLayersP1), make_device_view(queue, attachedByLayer.data(), nLayersP1));
      if (nOTForClaims > 0u)
        alpaka::memcpy(queue,
                       make_host_view(hOT.data(), nLayersP1),
                       make_device_view(queue, const_cast<uint32_t*>(otSource->layerStart), nLayersP1));
      // Host wait: hHits/hAtt/hOT are host vectors printed on the next lines (verbose only).
      alpaka::wait(queue);
      const int nLayers = int(nLayersP1) - 1;
      printf("[MergerExtend] per-CA-layer coverage (nLayers=%d, nMergedHits=%u, nOTHits=%u):\n",
             nLayers,
             nHits,
             nOTForClaims);
      printf("[MergerExtend]   layer : otAvail  hitsOnTrk  attached\n");
      for (int L = 0; L < nLayers; ++L) {
        const uint32_t otA = (nOTForClaims > 0u) ? (hOT[L + 1] - hOT[L]) : 0u;
        if (otA != 0u || hHits[L] != 0u || hAtt[L] != 0u)
          printf("[MergerExtend]   L%-4d : %7u  %9u  %8u%s\n",
                 L,
                 otA,
                 hHits[L],
                 hAtt[L],
                 (L >= 34) ? "   <-- OT disk" : "");
      }
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::caExtension
