#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAExtensionKernels_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAExtensionKernels_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"
#include "RecoTracker/PixelSeeding/interface/OTHitTag.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometrySoA.h"

#include "CASizingDumpMacro.h"  // CA_SIZING_DUMP toggle for the attach stage's demand dump
#include "CAStructures.h"
#include "ExtenderGeometry.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::caExtension {

  // The attach stage is compiled only for Phase2OTStubs; alias its concrete per-layer phi
  // histogram (the CA's hit PhiBinner) so the attach search can window its hit scan by phi.
  using ExtPhiBinner = caStructures::PhiBinnerT<::pixelTopology::Phase2OTStubs>;

  // Configuration of the OT hit-attach stage. It runs once, in the merger, over the concatenated,
  // HP-selected and twin-merged collection, before the merger's final GBL refit; the CA itself only
  // builds, base-fits and HP-selects. The search and gate parameters size the per-layer hit window and
  // its accept test, the pre-gate parameters decide which merged tracks enter the search.
  struct AttachParams {
    bool enable = false;
    bool verbose = false;  // one-line per-event attach summary (counters)
    // Full-hits attach path: when true the producer builds the raw-OT-rechit source (OTHitsSource
    // below) and threads it to launchAttach, so the walk searches the full OT hit population in a
    // second per-layer round instead of only the stub-derived merged hits. When false no OT source is
    // built and the walk sees merged hits only.
    bool useOTRecHits = false;
    int maxExtraHitsPerTrack = 4;
    int maxWalkLayers = 6;  // nearest-reachable-layers visit budget K; 1 ~= fake-minimal, ~6 ~= full physics
    float chi2Cut = 2.0f;
    float endcapChi2Cut = 2.0f;
    float typePriorityBiasCm = 15.f;
    int pixHitsTarget = 3;
    float maxRPhiResidCm = 0.5f;
    float maxSecResidCm = 1.0f;
    float endcapMaxSecResidCm = 0.5f;
    // Alignment-noise floors [cm], added in quadrature to the chi2 denominators. alignSigmaPhiCm is the
    // lateral (r*dphi) alignment uncertainty, so the chi2 gets (alignSigmaPhiCm / r)^2 added to sigPhi^2;
    // alignSigmaSecCm is added directly to sigSec^2 [cm^2]. This stops over-shrunk propagation sigmas
    // from rejecting alignment-limited hits at large radius.
    float alignSigmaPhiCm = 0.005f;
    float alignSigmaSecCm = 0.01f;
    // Pre-gate: a candidate enters the search only if its fit is finite, its chi2/ndof is
    // below preGateMaxChi2 and its pt above preGateMinPt (edup and never-fitted tuples are always
    // skipped). maxCandidates caps the search volume (and sizes the extras buffers); overflow is
    // counted and reported, never silent.
    float preGateMaxChi2 = 15.f;  // host reduced-chi2 ceiling; lower restricts the walk to better-measured hosts
    float preGateMinPt = 0.9f;    // = final-selection pt cut; extending pt<0.9 is pointless (discarded downstream)
    // Pre-gate |eta| ceiling: a host with |cotTheta| above sinh(maxAbsEta) is not extended at all. OT
    // attach purity collapses beyond |eta| ~2.5, where the trajectory crosses the last disc
    // near-tangentially and the TEDD residual spread grows to several cm.
    float maxAbsEta = 2.4f;
    // Host-quality pre-gate: three independently sentinel-gated predicates; a host failing any active one
    // is skipped entirely. Impure raw-OT extras concentrate on poorly-measured hosts.
    //   extHostMaxChi2Ndof (<=0 = off): require tracks[i].chi2() < extHostMaxChi2Ndof. tracks[i].chi2()
    //     is already the reduced (per-ndof) GBL chi2, so it is compared to a chi2/ndof threshold directly.
    //   extHostMinHits (0 = off): require reco::nHits(tracks,i) >= extHostMinHits.
    //   extHostMinPt  (<=0 = off): require tracks[i].pt() >= extHostMinPt (GeV).
    float extHostMaxChi2Ndof = 0.f;
    int extHostMinHits = 0;
    float extHostMinPt = 0.f;
    uint32_t maxCandidates = 128 * 1024;
    // Ceiling on the candidate capacity the attach scratch is sized to, applied as
    // candCapacity = min(knownCandCapacity, extRefitMaxCandidates) before the [16, maxCandidates] clamp.
    // It must stay at or above the largest candidate count the pre-gate can produce: below that,
    // candidates past the ceiling are dropped by the fill pass. The default keeps knownCandCapacity.
    uint32_t extRefitMaxCandidates = 0xffffffffu;
    // Cross-track arbitration N-way sharing: an attached extra may be owned by up to this many tracks,
    // the N claimants with the smallest gate chi2, ties by tuple id; 1 = exclusive ownership. The claim
    // buffer keeps N sorted slots per hit, filled by an atomicMin insertion cascade whose final N-slot
    // set is the N smallest packed claims regardless of thread interleaving, and resolve keeps an extra
    // iff its tuple id occupies any slot.
    int extMaxSharedOwners = 1;
    // Ambiguity gate in the cross-track resolve stage: a contested extra (claimed by >= 2 tracks) is kept
    // only for its best claimant when the top-2 claimants' gate chi2 are within extAmbigDeltaChi2 of each
    // other, a hit two tracks fit nearly equally well being an ambiguous stray for the worse one. Needs
    // the second claim slot, so it is active only when extMaxSharedOwners >= 2; <= 0 disables it.
    float extAmbigDeltaChi2 = 0.f;
    // Per-class scales of the fixed-cut per-hit chi2 cut on the TOB4-6 (CA 31-33) and TID (CA 34-53)
    // layers, multiplied into baseChi2Cut at the three gate sites (merged round, raw-OT round, partner
    // scan) so the lane scan, the warp-reduce threshold and the partner threshold stay consistent.
    // >1 widens, <1 tightens, 1.0 leaves the base cut. The two layer ranges are disjoint.
    float extChi2CutScaleTOB456 = 1.0f;
    float extChi2CutScaleTID = 1.0f;
    // Source-scoped raw-OT veto: the walk's round 1, the raw-OT rechit scan on layers where no stub was
    // attached, is skipped on TOB4-6 (CA 31-33) and on TID (CA 34-53) when the respective flag is set;
    // the merged-stub round 0 is untouched. Raw OT is a less pure source than stubs, and its impure
    // extras concentrate in these two classes. false => round 1 runs on every layer.
    bool extRawOTVetoTOB456 = false;
    bool extRawOTVetoTID = false;
    // Displacement-aware inner-OT gate. When on, the walk tightens the TOB1-3 (CA 28-30) per-hit chi2
    // accept window for hosts whose fitted transverse impact parameter is large in significance,
    // |d0|/sigma_d0 >= sqrt(extDispGateSig2): TOB1-3 attaches are dominated by prompt strays for strongly
    // displaced hosts. The arm is not available in-kernel, so displacement significance is the proxy.
    bool extDisplacementAwareGate = false;
    // Displacement-significance^2 threshold of the gate above: tip^2 >= extDispGateSig2 * V(tip). Only
    // read when extDisplacementAwareGate is on; raising it restricts the tightening to more strongly
    // displaced hosts.
    float extDispGateSig2 = 400.f;
    // Forward-eta TOB1-3 pocket gate, the |eta|-scoped companion of extDisplacementAwareGate. When on,
    // the walk tightens the same TOB1-3 window for hosts in the forward band |eta| in [1.5, maxAbsEta),
    // tested as |cotTheta| in [kPocketCotThetaLo, kPocketCotThetaHi), that the displacement gate does not
    // already cover; the two are mutually exclusive, so the scales never stack.
    bool extForwardPocketGate = false;
    // When the pocket gate is on, apply it only to displaced-arm tracks (needs launchMergerAttach's
    // armId argument, filled through the filterTracks compaction); false = arm-blind, no armId needed.
    // Ignored when extForwardPocketGate is off.
    bool extPocketGateArmScoped = true;
    // MTV-aligned per-track extra cap. When on, the walk caps the appended-extra clusters per track at
    // floor((n_core_clusters - 1)/kMtvSharedFracDen), so that even if every appended extra is wrong the
    // worst-case shared fraction n_core/(n_core+n_extra) stays above 0.75. Cluster units, not raw hits:
    // a merged 2-hit stub extra resolves to 2 clusters, a raw-OT or pixel extra to 1.
    bool extMtvAlignedExtraCap = false;
    // Extra reachability slack [cm] on pixel layers (CA L < 28) for prefer-pixel hosts, i.e. hosts with
    // nPix < pixHitsTarget, whose inward OT-only-fit extrapolation can place the crossing just outside
    // the layer's z/r envelope plus kReachSlackCm. It cannot recover the predict-invalid case, where the
    // circle yields no crossing at all. 0 => no extra slack.
    float extRecallReachRelax = 0.f;
    // Pixel-first visit budget. The K = maxWalkLayers visit budget reserves up to ceil(K/2) seats for OT
    // disks, which on a prefer-pixel host can crowd out reachable pixel layers. When > 0, on prefer-pixel
    // hosts only, OT-disk forcing is suppressed while the nearest unvisited-reachable layer is a pixel
    // layer (gL < 28) and fewer than this many pixel layers have been visited. 0 => never suppresses.
    int extRecallPixelFirstBudget = 0;
    // Per-layer-class scale on the propagated fit covariance (predPhiVar, predSecVar) entering the 2-dof
    // gate chi2 denominators; a class scale s scales the correct-hit gate chi2 of that class by ~1/s in
    // the propagated-dominated tail. The classes are pixel (merged round, CA L < 28), stub (merged round,
    // CA L >= 28) and rawOT (round 1 plus the partner scan). 1.0 leaves the covariance as fitted.
    float extCovScalePixel = 1.0f;
    float extCovScaleStub = 1.0f;
    float extCovScaleRawOT = 1.0f;
    // Honest-calibration chi2 cut on pixel layers (CA L<28). chi2Cut=2.0 on a 2-dof gate is the 63.2 %
    // (1-e^-1) efficiency quantile by construction; a 95 %-efficient 2-dof cut is -2 ln(0.05) = 5.99.
    // When > 0 this replaces the pixel-layer baseChi2Cut (OT layers keep chi2Cut/endcapChi2Cut).
    // <= 0 (sentinel) => pixel keeps chi2Cut/endcapChi2Cut. It composes with extCovScalePixel, so
    // raising both at once double-loosens the pixel gate.
    float extPixelGateChi2Cut = -1.0f;
    // Standalone stub-bend veto in the merged round-0 gate: a stub's measured local bend dPhiDr is
    // compared to the track's expectation through the radius-independent half-curvature significance the
    // CA build uses, kappa = dPhiDr / sqrt(1 + r^2 dPhiDr^2), and |kappa_track - kappa_stub| /
    // sigma(kappa_stub) must not exceed extStubBendGate. PS stubs carry a large dPhiDrError, so the veto
    // is effectively 2S-only. <= 0 => not applied, as is the bend chi2 row of extBendPackage.
    float extStubBendGate = -1.0f;
    // Anchored cap exemption. The extMtvAlignedExtraCap cluster budget is 0-2 for short OT-only cores and
    // is exhausted by the first accepted TOB4-6 stub, which blocks most TOB4-6 -> TOB5/6 continuations.
    // When on, a candidate bypasses the cluster-cap check iff the track is anchored, already carrying at
    // least one accepted OT extra on a previous walk layer, and the candidate is itself on an OT layer.
    // It covers the three cap sites: walk-stop, per-accept drop and stack-partner scan.
    bool extCapExemptAnchored = false;
    // Budget floor, independent of the anchored exemption. When the MTV cap is on, the per-track cluster
    // budget becomes max(floor((n_core_clusters-1)/3), extCapBudgetFloor); the cap-off sentinel is never
    // lowered by it. 0 => unchanged.
    int extCapBudgetFloor = 0;
    // State process noise. Without a P += Q step the running-state covariance P only receives the
    // subtractive measurement downdate and grows overconfident with each accept. When > 0, on each accept
    // the just-traversed gap's Highland msVar is injected into the direction terms of P: circle-block
    // vPhi += scale*msVar, line-block vCot += scale*msVar*(1+cot^2)^2, since d(cot)/d(theta) =
    // -(1+cot^2). Scattering is elastic, so curvature is untouched. The injection is post-downdate, so
    // the current measurement is not double-counted. scale multiplies the physical Q; 0 => none.
    float extStateProcessNoise = 0.f;
    // Force-pixel-visit. For prefer-pixel hosts and pixel layers (CA L < 28) only, the reachability
    // envelope test is bypassed: the layer is confirmed reachable whenever the crossing prediction is
    // valid, and the per-layer road and gate scan then decides. Bounded by the pixel geometry and by the
    // K and pixel-first budgets.
    bool extRecallForcePixelVisit = false;
    // TOB4-6 scope: when both this and extCapExemptAnchored are on, the exemption fires only for
    // candidates on CA layers 31-33, at every exemption site. false => the exemption keeps its L >= 28
    // scope.
    bool extCapExemptTOB46Only = false;
    // Optional candidate-chi2 cap on the exemption: when > 0, an exempted candidate also needs its own
    // gate chi2 < extCapExemptMaxChi2 (applied at the per-accept site, using the winner's gate chi2).
    // <= 0 => no chi2 cap.
    float extCapExemptMaxChi2 = 0.f;
    // Runtime walk budget: the nearest-reachable-layers visit budget K as a runtime loop bound. Only the
    // walk loop bounds read it; every buffer sizing/stride keeps the compile-time maxWalkLayers. Clamped
    // in-kernel to [1, kChainMaxVisits = 8] (the shared chain/hole array size).
    int extMaxWalkLayers = 6;
    // Far-first disc ordering for OT-less forward pixel hosts. The walk normally visits the nearest
    // candidate layer by envelope proxy distance, which on a forward pixel-only host collapses to
    // min_k |Z_L - z_k| and spends the K and slot budgets inward, on hits that do not lengthen the
    // transverse lever arm. When armed, the ordering key of the endcap pixel layers is replaced by
    // -|layerZ[L]|, so the walk visits the farthest reachable disc crossing first; barrel and OT layers
    // keep the nearest-first key. Arming requires all three of
    //   A1  |cotTheta| >= sinh(extAttachFarMinAbsEta) (the host's own fitted polar angle);
    //   A2  the host core carries no OT or stub hit (coveredMask >> 28 == 0);
    //   A3  no OT layer is confirmed reachable: where raw-OT/TEDD content is still in reach it carries a
    //       far longer lever arm than any pixel disc and the nearest-first order already collects it.
    // false => the key is never rewritten and every host keeps the nearest-first order.
    bool extAttachFarFirst = false;
    // The |eta| floor of the far-first ordering (A1 above); below it the nearest-first order runs.
    float extAttachFarMinAbsEta = 2.8f;
    // Window-ambiguity condition on the far-first commit; read only when extAttachFarFirst is on. On a
    // far crossing of an armed host, an endcap pixel layer whose |Z_L| lies beyond the largest |z| of the
    // host's core hits, the argmin winner is committed only if at most this many candidates cleared the
    // crossing's gate in the merged round; otherwise the walk declines the crossing, keeps the slot and
    // carries on into the nearer discs. Larger values buy more far crossings at lower purity; <= 0
    // switches the condition off.
    int extAttachFarMaxWin = 1;
    // When true, on OT layers (CA L >= 28) the walk replaces its window/gate/rank stack by a derived one:
    //   * the r-phi prediction variance comes from the smoothed-prediction functional published per
    //     host by the merger-side pre-attach pass (ExtPredCoeff below), not from makeRunningHelix's
    //     perigee-propagated covariance (about a factor two too narrow in pull);
    //   * multiple scattering enters projected, with the exact transverse-plane G brackets and the 3-D
    //     lever (the fixed-cut road multiplies a 3-D scattering angle by the transverse arc, i.e. it is
    //     too narrow by cosh(eta)), scaled by the measured material-dispersion factor f_ms;
    //   * the endcap r-phi variance gains its polar-angle/z0 rows, absent from the fixed-cut gate;
    //   * per-accept process noise enters the state from the traversed gap's exact material moments
    //     (W, S1, S2) instead of per-visit incremental scattering added to the innovation;
    //   * selection is one chi2 on (r-phi, secondary), cut and ranked at the measured quantile map
    //     Q-hat(eps): window == gate == rank, one number (extDerivedEps);
    //   * the hole hypothesis ("attach nothing here") competes in the argmin, priced from the measured
    //     per-layer stub availability eta_L, the stub areal density rho and the window volume; inert
    //     while the statistic has only d = 2 rows (see kExtDerHoleMinDim), the bend row arms it.
    // The three cm residual caps become per-layer module-envelope runaway ceilings on this path, twice
    // the layer half-extent plus the reachability slack. Pixel layers (CA L < 28) keep the fixed-cut
    // gate: every table below is measured on OT roads only.
    bool extDerivedSelection = false;
    // The one free number of the package: the efficiency epsilon spent once, on the measured quantile
    // map, for the window, the gate, the rank and the hole prior. Larger eps = wider window, gate and
    // rank quantile, i.e. more attach yield at more fake exposure. Only read when extDerivedSelection.
    double extDerivedEps = 0.31;
    // The hole hypothesis, part of the same package. Inert while the measurement dimension is below
    // kExtDerHoleMinDim: at d = 2 the window volume prices it about 6 chi2 units too cheaply.
    bool extDerivedHole = true;
    // Hole detection prior. The plain form sets the hit hypothesis's prior to P(hit found) = eta_L * eps
    // and multiplies it by the full Gaussian density N(m; pred, R), but the Gaussian's own integral over
    // the window is eps, so the window mass is counted twice. The correct pairing is prior eta_L with the
    // untruncated density, the standard PDA no-detection weight
    // b = lambda (2 pi)^{d/2} |S|^{1/2} (1 - P_D P_G)/P_D, in which the gate mass P_G appears only inside
    // (1 - P_D P_G). References: Bar-Shalom & Fortmann, "Tracking and Data Association", Academic Press
    // 1988, ch. 6; Bar-Shalom, Daum & Huang, IEEE Control Systems 29(6) 82-100, 2009,
    // doi:10.1109/MCS.2009.934469; Petersen & Beard, IEEE Trans. Aerosp. Electron. Syst. 2022,
    // doi:10.1109/TAES.2022.3214803 (arXiv:2108.07265), Lemma 5 eq. (46)-(47) and Appendix B.
    // ON  => chi2_hole = 2 ln[ eta_L / ((1 - eta_L eps) nu) ]   (+ 2 ln(1/eps))
    // OFF => the plain numerator eta_L*eps.
    // No new number: the same eta_L table and the same single eps.
    bool extHoleDetectionPrior = false;
    // Hole raw-round prior. The raw-OT round runs only where the stub round attached nothing, so the
    // hypothesis it arbitrates is whether the layer produced a usable raw cluster given that no stub
    // formed: the conditional availability eta_cond = (eta_rawOT - eta_stub) / (1 - eta_stub), against a
    // background of raw clusters rather than stubs. On consumes extEtaLRaw and extRhoRaw for round-1
    // winners only; off prices both rounds with the single stub row.
    bool extHoleRawRoundPrior = false;
    // [kExtOTLayers] raw round's conditional availability / raw-cluster areal density. nullptr => OFF.
    const float* extEtaLRaw = nullptr;
    const float* extRhoRaw = nullptr;

    // The measured tables (device pointers, owned by the merger; see ExtDerivedTables.h).
    // Q-hat: the empirical quantile of the correct-hit 2-dof pull statistic X2 = p_R^2 + p_S^2 under the
    // frozen sigma model, keyed (layer class x |eta| x visit index) and tabulated on a fixed eps grid.
    // The producer interpolates the map at extDerivedEps once on the host and publishes the per-cell
    // threshold; the kernel only indexes it. Layout:
    //   [(cls * kExtQEtaBins + etaBin) * kExtQVisitBins + visitBin].
    const float* extQhat = nullptr;  // [kExtQCells] Q-hat(extDerivedEps) per cell
    // eta_L: P(the host's own particle left a stub on this layer), one scalar per CA layer 28..53.
    // Both eta_L and rho are layer-keyed with no |eta| axis, so a layer's row is dominated by whichever
    // |eta| population supplies most of its samples.
    const float* extEtaL = nullptr;  // [kExtOTLayers]
    // Rho: the stub areal density per layer [cm^-2], occupancy-weighted; the first-order input to the
    // hole (chi2_hole moves as 2 ln rho).
    const float* extRho = nullptr;  // [kExtOTLayers]
    // Arms the 5th Q-hat |eta| bin. The producer sets it exactly when a road can legitimately be
    // forward, i.e. when its |eta| pre-gate ceiling exceeds 2.4. See extQhatCell for why the pre-gate
    // alone does not imply it.
    bool extFwdEtaBin = false;
    // The target-side additive variance dV [cm^2] per module class (see kExtMatCls* below) and the
    // measured material-dispersion scale f_ms (barrel, endcap).
    const float* extDV = nullptr;  // [kExtMatClasses]
    float extFmsBarrel = 1.f;
    float extFmsEndcap = 1.f;
    // Per-host smoothed-prediction payload written by the merger-side pre-attach pass, indexed by
    // attach slot.
    const struct ExtPredCoeff* extPred = nullptr;
    // The third row of the walk's selection statistic, and with it the information basis the hole's own
    // threshold is priced against; dropping it roughly doubles the wrong-attach count. When true, and
    // only together with extDerivedSelection:
    //   * the stub bend enters as the third chi2 row, p_b = (dPhiDr_hit - dPhiDr_track)/sqrt(R_bb),
    //     with R_bb = sigma_b^2 + H_b C H_b^T + Q_MS,bb, i.e. the track-side prediction uncertainty is
    //     in the denominator (the standalone extStubBendGate veto divides by the measurement error
    //     alone, which with a corrected sigma_b costs a large fraction of the correct attachments);
    //   * the track side is the fitted local direction of the same closed-form crossing the walk already
    //     linearises, barrel dphi/dR and endcap (dphi/dz)/(dr/dz), in analytic form;
    //   * sigma_b is the leak-free formation value (TrackingRecHits::dPhiDrErrorPrec) times the
    //     measured per-class excess below;
    //   * the standalone bend veto is not applied wherever this row is active;
    //   * the statistic becomes 3-dof for stubs (Q-hat_3) and stays 2-dof for raw-OT / non-stub
    //     candidates (Q-hat_2);
    //   * the hole arms (see kExtDerMeasDim below): nu = rho (2 pi)^{3/2} sqrt(|R|).
    bool extBendPackage = false;
    // There is no separate bend trim: with the bend as a chi2 row the 3-dof ball already excludes what
    // such a trim would cut, and it would add a second efficiency choice next to the single eps.
    // Q-hat_3: the same measured-quantile map as extQhat, for the 3-dof statistic. Same keying and
    // same cell layout; consumed by stub candidates only.
    const float* extQhat3 = nullptr;  // [kExtQCells]
    // The measured per-module-class excess of the honest bend error over the two-cluster precision floor
    // dPhiDrErrorPrec delivers, indexed by kExtSigBCls* below.
    const float* extSigBExcess = nullptr;  // [kExtSigBClasses]
    // rho_3: the stub density in the 3-dof measurement space, rho_A / (2 b99) [cm^-1 rad^-1], with b99
    // the measured per-layer 99th-percentile bend half-range. The hole prices "attach nothing" against
    // the window volume, so with a bend row in the statistic the density that makes
    // nu = rho (2 pi)^{d/2} sqrt(|R|) dimensionless is a per-area-per-bend one, not the areal rho_A.
    const float* extRho3 = nullptr;  // [kExtOTLayers]
  };

  // The smoothed-prediction payload, one record per walk host.
  // The smoothed prediction of the fast-BL band at a virtual target node, reduced to the 3x3 local
  // covariance of (u, u', kappa) at the last fitted node (C_loc = T A^-1 T^T with
  // T = [[0,1,0],[-1/h,1/h,-h/2],[0,0,1]]), plus the anchor the walk measures its extrapolation from
  // and the last fitted gap's exit-direction kink variance. The walk forms
  //     V_u(ds) = g^T C_loc g ,  g = [1, ds, -ds^2/2] ,  ds = s_target - anchorS
  // which is identically the V_out = g^T A^-1 g of the bordered band, and then converts it to the target
  // surface with conv^2.
  struct ExtPredCoeff {
    float c00, c01, c02, c11, c12, c22;  // symmetric C_loc on (u [cm], u' [rad], kappa [1/cm])
    float anchorS;                       // transverse arc of the last fitted node, in the WALK's convention
    float anchorR, anchorZ;              // its global (r, z) -- the material anchor of every road integral
    float qgapCoef;                      // theta_T^2 * (1 - m1^2/m2) of the last fitted gap [rad^2]
    // The per-host effective bending field the published q/pT is expressed in, i.e. the
    // blEffectiveBField the fast BL fit divided its geometric curvature by (copyFromCircle's
    // 1/bFieldEff). The walk inverts that conversion, rho_geom = 1/(q/pT * B); with the origin scalar
    // instead the reconstructed circle would be off by B_eff/B(0,0), the forward field correction the
    // fit applied. <= 0 (or an invalid payload) => the walk falls back to the scalar.
    float bFieldEff;  // [GeV/cm], the fit's own curvature->pT conversion field
    // The ionization energy-loss road centre. The fast BL fit models one curvature, while the real
    // track's unsigned curvature grows along the path as it loses momentum,
    // kappa(s) = kappa_0 (1 + dE(X(s))/p); the fit subtracts the resulting deterministic radial offset
    // from its own residuals, so the curvature it publishes is kappa_0 at node 0. In the fit's
    // convention u is the radial offset from the reference circle's centre, outward positive, and
    // u'' = -dkappa(s) with dkappa >= 0, so with the walk's anchor at the last fitted node the offset at
    // a crossing an arc ds further out is
    //     u(ds) = elossU + elossUp*ds - 0.5*( elossDkAnchor*ds^2 + elossK*S2 )
    // with S2 the second moment of the traversed material about the arrival end, which the per-visit
    // segmentXX0Moments march publishes as w1*W*d1^2. fitCorrections off, or an invalid payload, leaves
    // all four at 0.
    float elossK;         // dkappa per unit material column = kappa_0*(dE/dX)_eff/p   [1/(cm X0)]
    float elossDkAnchor;  // the curvature excess ALREADY accumulated at the anchor     [1/cm]
    float elossU;         // u at the anchor (the fit's own u_eloss at node n-1)        [cm]
    float elossUp;        // du/ds at the anchor                                        [rad]
    float valid;          // 1 = usable; 0 = the walk falls back to the fixed-cut gate
  };

  // Q-hat keying: 3 OT layer classes x 5 |eta| bins x 4 visit bins.
  //   eta bin 4 (|eta| >= 2.4) is reachable only with the forward bin armed, which the producer does
  //   from its own |eta| ceiling; see extQhatCell.
  constexpr int kExtQClasses = 3;    // 0 = TOB1-3 (CA 28-30), 1 = TOB4-6 (31-33), 2 = TID (34-53)
  constexpr int kExtQEtaBins = 5;    // |eta| < 0.8 / 0.8-1.5 / 1.5-2.0 / 2.0-2.4 / >= 2.4
  constexpr int kExtQVisitBins = 4;  // walk visit index 0 / 1 / 2 / >= 3
  constexpr int kExtQCells = kExtQClasses * kExtQEtaBins * kExtQVisitBins;  // 60
  constexpr int kExtOTLayers = 26;                                          // CA layers 28..53

  // The coefficient pass's N-bin ladder. The band solve is templated on the hit count, so the walk hosts
  // are binned by multiplicity: one bin per N in [3, kExtPredMaxN], the top bin absorbing every host
  // above the cap. The bins are launched together in one grid, block b -> bin b / blocksPerBin.
  // kExtPredMaxN must track HelixFit<Phase2OTStubs>::kRefitMaxN; the launcher static_asserts it.
  constexpr int kExtPredMinN = 3;
  constexpr int kExtPredMaxN = 12;
  constexpr uint32_t kExtPredNBins = uint32_t(kExtPredMaxN - kExtPredMinN + 1);  // 10

  // The measurement dimension of the derived selection statistic: two rows, (r-phi, secondary), and
  // three once the bend row is live. The hole hypothesis prices "attach nothing" against the window
  // volume, nu = rho (2 pi)^{d/2} |R|^{1/2}, so its threshold moves with d: dropping the bend coordinate
  // lowers chi2_hole by about 6 units and puts it below the median gate threshold, where the hole fires
  // on a large fraction of argmin winners. The hole therefore arms only at d = 3.
  constexpr int kExtDerMeasDim2 = 2;    // (r-phi, secondary)          -- raw-OT / non-stub candidates
  constexpr int kExtDerMeasDim3 = 3;    // (r-phi, secondary, bend)    -- stub candidates
  constexpr int kExtDerHoleMinDim = 3;  // the hole hypothesis arms only at d >= 3 (see above)

  // Module classes of the measured bend-error excess.
  // dPhiDrErrorPrec delivers the two-cluster precision floor; the honest error exceeds it by a
  // class-dependent factor, the split being PS against 2S rather than barrel against endcap. The excess
  // and the precision floor belong together: in endcap PS the floor without the excess under-covers.
  constexpr int kExtSigBClsBarrelFlatPS = 0;
  constexpr int kExtSigBClsBarrelFlat2S = 1;
  constexpr int kExtSigBClsBarrelTilted = 2;
  constexpr int kExtSigBClsEndcapPS = 3;
  constexpr int kExtSigBClsEndcap2S = 4;
  constexpr int kExtSigBClasses = 5;

  // Class of a stub from its own packed StubFlags (isBarrel / isFlat / isPS).
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int extSigBClass(uint8_t stubFlags) {
    const bool bar = ::reco::StubFlags::isBarrel(stubFlags);
    const bool ps = ::reco::StubFlags::isPS(stubFlags);
    if (!bar)
      return ps ? kExtSigBClsEndcapPS : kExtSigBClsEndcap2S;
    if (!::reco::StubFlags::isFlat(stubFlags))
      return kExtSigBClsBarrelTilted;
    return ps ? kExtSigBClsBarrelFlatPS : kExtSigBClsBarrelFlat2S;
  }
  // Module classes of the target-side dV measurement.
  constexpr int kExtMatClsPXB = 0;
  constexpr int kExtMatClsPXD = 1;
  constexpr int kExtMatClsTBPS = 2;
  constexpr int kExtMatClsTB2S = 3;
  constexpr int kExtMatClsTEDDPS = 4;
  constexpr int kExtMatClsTEDD2S = 5;
  constexpr int kExtMatClasses = 6;

  // |eta| edges as |cot(theta)| = sinh(eta), so the hot path compares against constants instead of
  // taking a transcendental -- the same convention the pre-gate uses.
  constexpr float kExtCot08 = 0.888106f;  // sinh(0.8)
  constexpr float kExtCot15 = 2.129279f;  // sinh(1.5)
  constexpr float kExtCot20 = 3.626860f;  // sinh(2.0)
  constexpr float kExtCot24 = 5.466229f;  // sinh(2.4)  -- the top of the last OT-measured |eta| bin

  // Q-hat cell id from (CA layer, |cot(theta)|, visit index). A pixel layer has no measured class and
  // returns -1, which is what switches the derived selection off there.
  //
  // fwdEtaBin arms the 5th |eta| bin. It is not implied by the pre-gate: the |cot| keying this cell uses
  // is the walk state's, taken at the crossing after the Kalman updates, and that drifts past sinh(2.4)
  // on a fraction of ordinary sub-2.4 hosts, so leaving the bin always on would re-key those roads too.
  // With it off the ladder saturates at bin 3.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int extQhatCell(int L, float absCot, int visit, bool fwdEtaBin = false) {
    if (L < 28)
      return -1;
    const int cls = (L <= 30) ? 0 : ((L <= 33) ? 1 : 2);
    const int eb =
        (absCot < kExtCot08)
            ? 0
            : ((absCot < kExtCot15) ? 1 : ((absCot < kExtCot20) ? 2 : ((fwdEtaBin && absCot >= kExtCot24) ? 4 : 3)));
    const int vb = (visit < 0) ? 0 : ((visit > 3) ? 3 : visit);
    return (cls * kExtQEtaBins + eb) * kExtQVisitBins + vb;
  }

  // Candidate-level attach dump (diagnostic; compiled in only under -DEXT_CAND_DUMP).
  // Per-(candidate, visited layer) trace of the attach walk, read back to host and emitted as
  // fixed-format edm::LogInfo lines. The kernel writes these buffers only when its candDump_ member is
  // true, which the launcher sets only in an EXT_CAND_DUMP build; in a production build the buffers are
  // never allocated and there is no readback and no sync.
  //
  // Per-visited-layer outcome (the layer-level result recorded once per scanned layer).
  enum ExtCandDumpOutcome : int32_t {
    kExtDumpAccept = 0,    // a hit committed on this layer (winnerHitId >= 0)
    kExtDumpRejEmpty = 1,  // no candidate hit fell in the phi window (nWinMerged + nWinOT == 0)
    kExtDumpRejGate = 2,   // candidates present but none cleared the per-hit chi2 + abs-residual gate
    kExtDumpRejCap = 3,    // a gate-passing winner existed but the MTV cluster cap marked it over-budget
                           // and the walk dropped it (the arrival-order trim).
    kExtDumpRejOther = 4,  // gate-passing winner uncommitted for another reason (defensive; ~unreachable)
  };
  // ExtCandLayerRec::flags bit meanings.
  constexpr int32_t kExtDumpFlagVetoSkip = 1 << 0;  // raw-OT (round 1) veto-skipped on this layer
  constexpr int32_t kExtDumpFlagPartner = 1 << 1;   // a stack-partner second raw-OT extra was attached here
  // The layer's overall-best (min-gate-chi2) road candidate, over all considered hits of both rounds,
  // itself cleared the base per-hit gate (chi2 < baseChi2Cut/hitCut and abs-residual). Rides alongside
  // bestHitId, so that a geometric continuation that exists but is not committed is still visible.
  constexpr int32_t kExtDumpFlagBestPass = 1 << 2;

  // Per-road-candidate member cap. Each (candidate slot j, visit index vi) owns a block of up to
  // kExtDumpMaxMembers ExtCandMemberRec, one per scored road candidate of either round, i.e. the hits
  // that reached the gate chi2. The cap bounds the buffer and log volume on dense layers, whose window
  // occupancies reach several hundred; a layer exceeding it is clipped and candMemberOvf counts the clip.
  constexpr uint32_t kExtDumpMaxMembers = 64u;

  // Per-road-candidate member record. The hit id is the offline join key to the all-hit truth table, so
  // this dump carries only the ids, the per-candidate gate chi2 and the round/pass context. Buffer index
  // = (j * maxWalkLayers + vi) * kExtDumpMaxMembers + m; hitId < 0 marks an unwritten slot.
  struct ExtCandMemberRec {
    int32_t hitId;  // scored road-candidate id (bit30-tagged if raw-OT); join key to HitTruth tpKey/features
    float chi2;     // per-candidate gate chi2 (dPhi^2/sigPhi2 + dSec^2/sigSec2), the SAME value the argmin ranks
    int16_t round;  // 0 = merged (stub/pixel) round, 1 = raw-OT round
    int16_t pass;   // 1 iff this candidate cleared the base per-hit gate (chi2 < cut && abs-residual ok)
  };

  // Per-(candidate slot j, visit index vi in [0, maxWalkLayers)) record; buffer index = j *
  // maxWalkLayers + vi. trackId < 0 marks an unwritten slot (skip offline). Counts are post-detector-
  // masking genuine candidates (recHitMask / usedInStub / ownership rejects are NOT counted in nWin*).
  struct ExtCandLayerRec {
    int32_t trackId;      // CA tuple id (candList[j]); -1 = unwritten slot
    int32_t layerId;      // CA layer visited (0..nLayers-1; OT barrel >= 28)
    int32_t nWinMerged;   // genuine candidate hits in the phi window, merged round (round 0)
    int32_t nWinOT;       // genuine candidate hits in the phi window, raw-OT round (round 1); 0 if not run
    int32_t nPassMerged;  // merged-round hits clearing the per-hit gate (chi2 < cut && abs-residual ok)
    int32_t nPassOT;      // raw-OT-round hits clearing the per-hit gate
    int32_t winnerHitId;  // committed winner id (bit30-tagged if raw-OT); -1 = none committed
    int32_t round;        // committed round (0 merged, 1 raw-OT); -1 = none
    int32_t outcome;      // ExtCandDumpOutcome
    int32_t flags;        // OR of kExtDumpFlag*
    float winnerChi2;     // committed winner gate chi2; -1 = none committed
    float bestFailChi2;   // min gate chi2 over all considered hits (both rounds); -1 = none considered
    // Id of the overall-best (min-gate-chi2) road candidate on this layer, over all considered hits
    // regardless of gate outcome (bit30-tagged if raw-OT); -1 = none considered. bestFailChi2 is its
    // chi2 and kExtDumpFlagBestPass its base-gate pass.
    int32_t bestHitId;
    // Cumulative occupancy-gated hole count (rejGate
    // layers -- occupancy present, no gate-passer; NOT rejEmpty dead-module layers) from walk start through
    // this visited layer. For an accept record, == holes before the accept (an accept adds no hole).
    int32_t holeRunToHere;
  };
  // Per-candidate header (one per slot j). The covered/reachable/visited layer bitmasks give the
  // reachable-but-unvisited set exactly (K-budget / ordering losses = reachable & ~visited & ~covered),
  // so the offline budget/captured/lost decomposition needs no reachability proxy.
  struct ExtCandHdrRec {
    int32_t trackId;         // CA tuple id (candList[j]); -1 = unwritten slot
    int32_t nVisited;        // reachable layers scanned this walk (== walk steps)
    int32_t hostFlags;       // bit0 = displacement-gate host, bit1 = forward-pocket host
    int32_t pad;             // alignment padding (unused)
    uint64_t coveredMask;    // original (pre-walk) hit layers (bit L set)
    uint64_t reachableMask;  // confirmed-reachable layers (walk state 2)
    uint64_t visitedMask;    // scanned layers
  };

  // OT-rechit id tag (bit30): a walk-extra id with this bit set indexes the raw OT-rechit source
  // (OTHitsSource) instead of the merged TrackingRecHit SoA. The single definition lives in
  // interface/OTHitTag.h, so the plugin and CATrackFeatures.h share one literal.
  using caOTHitTag::isOTId;
  using caOTHitTag::kOTHitTag;
  using caOTHitTag::otIdx;

  // Raw device handles of the full-hits OT source consumed by the attach stage. Holds the two
  // source SoA views plus device pointers into the per-event buffers owned by OTHitsBuffers below.
  // All pointers null / nOTHits == 0 when the full-hits path is off. The walk indexes phiBinner /
  // layerStart uniformly, using the SAME 256-bin, per-CA-layer partition convention as the merged-hit
  // binner (so a candidate's per-layer phi window scan is identical whether it reads merged or raw
  // OT hits).
  struct OTHitsSource {
    ::reco::OTRecHitsConstView otHits;          // raw OT rechits (local/global position, error, detectorIndex)
    ::reco::StubsConstView stubs;               // OT stubs (lower/upper rechit indices, dPhiDr, flags)
    ::reco::OTHitModuleConstView otHitModules;  // per-OT-module moduleStart / upperSensorStart
    ::reco::StackedModuleGeometryConstView stackedGeometry;  // per-OT-module lower/upper sensor frames
    const ExtPhiBinner* phiBinner = nullptr;                 // per-CA-layer phi histogram over OT iphi, sized nOTHits
    const uint32_t* layerStart = nullptr;                    // [nLayers+1] first OT row of each CA layer (fill offsets)
    const uint8_t* usedInStub = nullptr;                     // [nOTHits] 1 iff the rechit is a member of some stub
    uint32_t* ownership = nullptr;                           // [nOTHits] per-hit ownership tag; the walk's two
    // `ownership[o] != 0` tests are guarded on the pointer, so a null
    // array rejects nothing
    uint32_t nOTHits = 0;
  };

  // Owns the per-event OT-source device buffers (caching-allocator backed, transient) plus the two
  // source views. Returned by buildOTHitsSource and kept alive by the producer for the duration of the
  // attach launch; makeSource() packages the raw handles the attach stage consumes.
  struct OTHitsBuffers {
    cms::alpakatools::device_buffer<Device, int16_t[]> iphi;         // [nOTHits] discretized phi
    cms::alpakatools::device_buffer<Device, uint8_t[]> usedInStub;   // [nOTHits] stub-membership mask
    cms::alpakatools::device_buffer<Device, uint32_t[]> layerStart;  // [nLayers+1] per-layer OT-row offsets
    cms::alpakatools::device_buffer<Device, ExtPhiBinner> phiHist;   // phi histogram header
    cms::alpakatools::device_buffer<Device, ExtPhiBinner::value_type[]> phiStorage;  // [nOTHits] index payload
    ::reco::OTRecHitsConstView otHits;
    ::reco::StubsConstView stubs;
    ::reco::OTHitModuleConstView otHitModules;               // per-OT-module upperSensorStart (walk module map)
    ::reco::StackedModuleGeometryConstView stackedGeometry;  // per-OT-module lower/upper sensor frames (gate)
    uint32_t nOTHits = 0;
    // The ownership slot is left null: the veto is unarmed (see OTHitsSource::ownership above).
    OTHitsSource makeSource() {
      return OTHitsSource{otHits,
                          stubs,
                          otHitModules,
                          stackedGeometry,
                          phiHist.data(),
                          layerStart.data(),
                          usedInStub.data(),
                          /*ownership=*/nullptr,
                          nOTHits};
    }
  };

  // Build the per-event full-hits OT source: allocate the buffers, fill iphi (same convention as the
  // merged SoA), the used-in-stub mask, the per-CA-layer OT-row offsets, and the OT phi binner (sized
  // nOTHits). Called by the producer only when the attach stage is enabled and the event has OT hits.
  // otHitModules / caLayers feed the per-layer offset kernel; nStubs sizes the stub-mask pass.
  OTHitsBuffers buildOTHitsSource(Queue& queue,
                                  const AttachParams& params,
                                  ::reco::OTRecHitsConstView otHits,
                                  ::reco::OTHitModuleConstView otHitModules,
                                  ::reco::StackedModuleGeometryConstView stackedGeometry,
                                  ::reco::StubsConstView stubs,
                                  ::reco::CALayersSoAConstView caLayers,
                                  uint32_t nOTHits,
                                  uint32_t nStubs);

  // Device scratch of the attach stage, allocated per event by allocateAttachBuffers (transient,
  // caching-allocator backed). candList/extras arrays are indexed by the compacted candidate slot
  // (0..nCands), not by tuple id; candList maps the slot back to the tuple id.
  //
  // Runtime-sized: candCapacity below is the per-event candidate cap the scratch is allocated to,
  // min(max(min(knownCandCapacity, extRefitMaxCandidates), 16), params.maxCandidates). candList, extras
  // and nExtras are sized by it; nCands, hitClaims and stats are not. Every downstream kernel and launch
  // loop reads candCapacity from here.
  struct AttachBuffers {
    cms::alpakatools::device_buffer<Device, uint32_t[]> candList;   // [candCapacity] tuple ids
    cms::alpakatools::device_buffer<Device, uint32_t> nCands;       // compacted candidate count
    cms::alpakatools::device_buffer<Device, uint32_t[]> extrasIds;  // [candCapacity * maxExtra]
    cms::alpakatools::device_buffer<Device, float[]> extrasChi2;    // [candCapacity * maxExtra]
    cms::alpakatools::device_buffer<Device, int32_t[]> nExtras;     // [candCapacity]
    cms::alpakatools::device_buffer<Device, uint64_t[]> hitClaims;  // [nHits] packed (chi2, tuple)
    cms::alpakatools::device_buffer<Device, uint32_t[]> stats;      // counters, see kStats* below
    // Runtime candidate capacity the scratch above was allocated to; the extent bound for every candidate
    // loop, grid size and RefitScaffold allocation downstream.
    uint32_t candCapacity = 0;
  };

  // Row stride of the extension's per-track fit buffer (nextRef[0..3], chi2). The pre-gate reads slot 2
  // (fitted-circle radius) as the never-fitted sentinel.
  constexpr uint32_t kPassBufStride = 5;

  // stats slots
  constexpr int kStatCandidates = 0;       // tuples passing the pre-gate
  constexpr int kStatCandOverflow = 1;     // tuples lost to the maxCandidates cap
  constexpr int kStatExtended = 2;         // candidates with >= 1 extra after arbitration
  constexpr int kStatTotalExtras = 3;      // extras after arbitration
  constexpr int kStatArbLost = 4;          // extras dropped by cross-track arbitration
  constexpr int kStatAccepted = 5;         // extended candidates rewritten into the track SoA
  constexpr int kStatUnfittable = 6;       // unwritten slot; kept so the indices below do not renumber
  constexpr int kStatRewriteOverflow = 7;  // events falling back whole (hit-list capacity)
  constexpr int kStatHistFirst = 8;        // extras-per-track histogram (last bucket = overflow)
  constexpr int kStatHistBuckets = 8;
  // Diagnostic: tagged OT-rechit hits written into accepted tuples. Stays 0 on the merged-only path;
  // reported by a one-shot print only when the OT source is active.
  constexpr int kStatOTWritten = kStatHistFirst + kStatHistBuckets;  // 16
  constexpr int kStatOTFirstFind = kStatOTWritten + 1;               // 17: unwritten slot (index placeholder)
  // Attachment ledger: where the walk's extras come from and where they are lost. All grid-wide
  // atomics, printed by the one-shot summaries below.
  constexpr int kDiagWalkMerged = kStatOTFirstFind + 1;     // 18: walk-committed merged extras (round 0)
  constexpr int kDiagWalkOT = kStatOTFirstFind + 2;         // 19: walk-committed OT extras (round 1)
  constexpr int kDiagSlotExhaust = kStatOTFirstFind + 3;    // 20: candidates whose walk ended by slot budget
  constexpr int kDiagSlotExhaustOT = kStatOTFirstFind + 4;  // 21: ...with >=1 OT extra committed
  constexpr int kDiagArbLostMerged = kStatOTFirstFind + 5;  // 22: arbitration-dropped merged extras
  constexpr int kDiagArbLostOT = kStatOTFirstFind + 6;      // 23: arbitration-dropped OT extras
  // Unwritten slots, kept so the indices below do not renumber.
  constexpr int kDiagVerRejOT = kStatOTFirstFind + 7;        // 24
  constexpr int kDiagVerRejMerged = kStatOTFirstFind + 8;    // 25
  constexpr int kDiagVerAccOT = kStatOTFirstFind + 9;        // 26
  constexpr int kDiagMergedWritten = kStatOTFirstFind + 14;  // 31: merged attached hits written to tuples
  // Raw-OT stack-partner extras accepted on stub-less layers (second OT hit on the same
  // module as the round-1 winner, gated against the updated helix). Grid-wide atomic, printed in the
  // [DIAG] summary. Stays 0 on the merged-only path (round 1 never runs).
  constexpr int kStatOTPartner = kDiagMergedWritten + 1;  // 32
  // Endcap secondary-variance decomposition: sigSec2 = hit + MS + predSec + align, accumulated over disk
  // candidate-hit evaluations, merged and OT rounds separately. Each considered endcap hit adds the
  // permil fraction of each term plus a count. Gated on the kernel functor's secFracDiag_, so production
  // runs skip the global atomics entirely.
  constexpr int kDiagSecFracHitM = kStatOTPartner + 1;    // 33: merged-round sum permil(hit)
  constexpr int kDiagSecFracMsM = kStatOTPartner + 2;     // 34: merged-round sum permil(MS)
  constexpr int kDiagSecFracPredM = kStatOTPartner + 3;   // 35: merged-round sum permil(predSec)
  constexpr int kDiagSecFracAlignM = kStatOTPartner + 4;  // 36: merged-round sum permil(align)
  constexpr int kDiagSecNM = kStatOTPartner + 5;          // 37: merged-round considered endcap hits
  constexpr int kDiagSecFracHitOT = kStatOTPartner + 6;   // 38-42: same, raw-OT round
  constexpr int kDiagSecFracMsOT = kStatOTPartner + 7;
  constexpr int kDiagSecFracPredOT = kStatOTPartner + 8;
  constexpr int kDiagSecFracAlignOT = kStatOTPartner + 9;
  constexpr int kDiagSecNOT = kStatOTPartner + 10;
  // Merged-endcap decomposition of predSecVar (verbose-gated too): circle-block vs line-block split of
  // the block-diagonal predSecVar, and the exact full-5x5 shadow-cov projection predSecVar_full, split
  // by whether a prior DISK hit was already folded in.
  constexpr int kDiagPredCircleM = kDiagSecNOT + 1;  // 43: sum permil(circle-part of predSecVar_bd)
  constexpr int kDiagPredLineM = kDiagSecNOT + 2;    // 44: sum permil(line-part of predSecVar_bd)
  constexpr int kDiagPredFullM = kDiagSecNOT + 3;    // 45: sum permil(predSecVar_full, exact 5x5 shadow)
  constexpr int kDiagNDisk0M = kDiagSecNOT + 4;      // 46: considered w/ 0 prior disk accepts
  constexpr int kDiagPredBd0M = kDiagSecNOT + 5;     // 47: sum permil(predSecVar_bd)   | 0 prior disk
  constexpr int kDiagPredFull0M = kDiagSecNOT + 6;   // 48: sum permil(predSecVar_full) | 0 prior disk
  constexpr int kDiagNDisk1M = kDiagSecNOT + 7;      // 49: considered w/ >=1 prior disk accept
  constexpr int kDiagPredBd1M = kDiagSecNOT + 8;     // 50: sum permil(predSecVar_bd)   | >=1 prior disk
  constexpr int kDiagPredFull1M = kDiagSecNOT + 9;   // 51: sum permil(predSecVar_full) | >=1 prior disk
  // Host-quality pre-gate and per-layer-class attach diagnostics, read back only when verbose.
  // kStatPreGateSkipped counts hosts skipped by the host-quality predicate alone, not by the always-on
  // base pre-gate. The per-class counters tally walk-committed extras of both rounds, pre-arbitration.
  constexpr int kStatPreGateSkipped = kDiagPredFull1M + 1;  // 52: hosts skipped by the host-quality pre-gate
  constexpr int kStatExtraTOB13 = kDiagPredFull1M + 2;      // 53: walk-committed extras on TOB1-3 (CA 28-30)
  constexpr int kStatExtraTOB456 = kDiagPredFull1M + 3;     // 54: ... TOB4-6 (CA 31-33)
  constexpr int kStatExtraTID = kDiagPredFull1M + 4;        // 55: ... TID (CA 34-53)
  // Ambiguity gate: extras vetoed from the worse claimant of a contested near-tie hit (verbose readout).
  constexpr int kStatAmbigVetoed = kDiagPredFull1M + 5;  // 56: extras dropped by the ambiguity gate
  // Derived-gate monitors. On the derived path the three cm residual caps are runaway ceilings, not
  // selectors, so their binding rate is the alarm: a ceiling that binds often means the delivered
  // efficiency is not the stated eps. kStatDerLayers is the denominator, the layer visits scored by the
  // derived gate. All are diagnostic-only; stats[] never reaches track content.
  constexpr int kStatDerLayers = kStatAmbigVetoed + 1;  // derived-gate layer visits
  constexpr int kStatDerCapR = kStatDerLayers + 1;      // 65: r-phi ceiling bound a ball-passer
  constexpr int kStatDerCapS = kStatDerLayers + 2;      // 66: secondary ceiling likewise
  constexpr int kStatDerHoleFire = kStatDerLayers + 3;  // 67: the hole hypothesis vetoed the argmin winner
  constexpr int kStatDerHoleCand = kStatDerLayers + 4;  // 68: winners the hole was tested on
  constexpr int kStatDerHostOff = kStatDerLayers + 5;   // 69: hosts without a usable payload
  constexpr int kStatDerHostOn = kStatDerLayers + 6;    // 70: hosts running the derived gate
  // Raw-channel monitors: the round-1 projections of the counters above, which otherwise split commits
  // by layer class or by source round but never by both. Incremented only on the round-1 path, so the
  // merged round pays nothing.
  constexpr int kStatRawTOB13 = kStatDerLayers + 7;         // 71: round-1 commits on TOB1-3 (CA 28-30)
  constexpr int kStatRawTOB456 = kStatDerLayers + 8;        // 72: ... TOB4-6 (CA 31-33)
  constexpr int kStatRawTID = kStatDerLayers + 9;           // 73: ... TID/TEDD (CA 34-53)
  constexpr int kStatDerHoleCandRaw = kStatDerLayers + 10;  // 74: hole tested on a round-1 winner
  constexpr int kStatDerHoleFireRaw = kStatDerLayers + 11;  // 75: ... and it declined the attach
  // The |eta| pre-gate's own rejection count (kStatPreGateSkipped covers only the three host-quality
  // predicates). Counted in the count pass only, matching the kStatCandidates single-count convention.
  constexpr int kStatPreGateEtaSkipped = kStatDerLayers + 12;  // 76: host skipped on |cotTheta| > sinh(maxAbsEta)
  // The far-first disc ordering arms per host on three conditions, only one of which is visible in any
  // other counter.
  constexpr int kStatAttachFarArmed = kStatDerLayers + 13;  // 77: host whose disc ordering was re-keyed far-first
  // A far commit declined by the window-ambiguity condition leaves no other trace: the layer's outcome
  // reads as "no gate-passer" in every other counter.
  constexpr int kStatAttachFarDecline = kStatDerLayers + 14;  // 78: far commit declined on window multiplicity
  // Length of the stats[] counter buffer: one slot past the last counter above.
  constexpr int kStatSize = kStatDerLayers + 15;  // 79

  // The smoothed-prediction pass. Defined in its own translation unit (ExtPredCoeff.dev.cc) for the
  // reason documented at the top of that file: compiling its band instantiations alongside Kernel_BLFit
  // changes the FMA contraction the compiler applies to the fit itself. Runs the fast-BL circle band
  // over every merged track selected by hostMask and publishes the walk's prediction covariance.
  // Read-only on the tracks.
  void launchExtPredCoeff(Queue& queue,
                          ::reco::TrackingRecHitConstView hv,
                          ::reco::CAModulesConstView cm,
                          ::reco::TrackSoAView mergedTracks,
                          ::reco::TrackHitSoAView mergedHits,
                          const int32_t* hostMask,
                          const OTHitsSource* otSource,
                          float bfield,
                          const float* rhoMap,
                          // The normalized (Bz,Br) r-z field map (BLBFieldMap), the same EventSetup
                          // condition the GBL refits read. Consumed only under fitCorrections; null =>
                          // the origin scalar everywhere.
                          const float* bMap,
                          bool fitCorrections,
                          ExtPredCoeff* out,
                          uint32_t nTracksCap);

  // Replicate the walk's own pre-gate predicate as a per-tuple mask, so the smoothed-prediction pass
  // runs on exactly the hosts the walk will use and on nothing else. Writes 0 for a host and -1
  // otherwise (the refit-ladder compaction's own convention); when pred is given, also clears its
  // `valid` field for every slot.
  void launchExtHostMask(Queue& queue,
                         const AttachParams& params,
                         ::reco::TrackSoAConstView tracks,
                         uint32_t nTracksCap,
                         int32_t* hostMask,
                         struct ExtPredCoeff* pred = nullptr);

  // Allocate the attach-stage scratch. Runs the count-only pre-gate pass (Kernel_extPreGate with a null
  // candList) over the input tuples, which fills stats[kStatCandidates] / stats[kStatCandOverflow] but
  // is not read back; the fill pass in launchAttach re-runs the same predicate to populate candList.
  // tracks + passBuf feed the pre-gate.
  // nOTHits: when the OT full-hits path is active (params.useOTRecHits && a source is built), the
  // per-hit arbitration claim buffer must cover the OT domain too (indices nHits + otIdx). Pass 0 to
  // keep hitClaims sized to nHits exactly.
  // candidateMask: when non-null, a tuple enters the (count + fill) pre-gate only if
  // candidateMask[tuple] >= 0. nullptr => no restriction.
  // knownCandCapacity: required, > 0. A host-known upper bound on the caller's candidate set; the
  // scratch is sized to it (first capped at params.extRefitMaxCandidates, then clamped to
  // [16, params.maxCandidates]) with no device->host readback, and the result is recorded in the
  // returned struct's candCapacity. Oversizing drops no candidate but also sizes every stage-B scratch
  // array and the per-candidate launch grids, which is what params.extRefitMaxCandidates bounds.
  AttachBuffers allocateAttachBuffers(Queue& queue,
                                      const AttachParams& params,
                                      uint32_t nHits,
                                      uint32_t nOTHits,
                                      uint32_t maxNumberOfTuples,
                                      ::reco::TrackSoAConstView tracks,
                                      const double* passBuf,
                                      const int32_t* candidateMask,
                                      uint32_t knownCandCapacity);

  // Post-walk scratch: the compacted extended candidates and the containers of their merged
  // (arc-sorted) hit lists, consumed by the accept stage + the final in-place rewrite.
  // extIdx (the dense candidate index) maps back to the CA tuple id via extTuple.
  struct RefitScaffold {
    cms::alpakatools::device_buffer<Device, uint32_t[]> extTuple;     // [candCapacity] extIdx -> tuple
    cms::alpakatools::device_buffer<Device, uint32_t[]> extCandSlot;  // [candCapacity] extIdx -> attach slot
    // nExt, pfxCounter and pfxCounter2 share one uint32_t[3] allocation (one allocation and one memset)
    // with three accessors. Each word is reached only through its own accessor, so no word is ever
    // touched as two different types; the prefix-scan counters are handed out as int32_t* because
    // multiBlockPrefixScan takes int32_t*, and an atomicAdd of 1 on a 4-byte aligned word is
    // bit-identical either way.
    cms::alpakatools::device_buffer<Device, uint32_t[]> counters;   // [3] = {nExt, pfxCounter, pfxCounter2}
    cms::alpakatools::device_buffer<Device, int32_t[]> extNExtras;  // [candCapacity] kept extras
    cms::alpakatools::device_buffer<Device, caStructures::SequentialContainer> extContainer;
    cms::alpakatools::device_buffer<Device, caStructures::SequentialContainer::Counter[]> extContainerOffsets;
    cms::alpakatools::device_buffer<Device, caStructures::SequentialContainer::value_type[]> extContainerStorage;
    // The fit row the rewrite writes back for each accepted candidate (indexed by dense extIdx):
    // 5-param state, 15-element covariance, chi2/ndof-normalized chi2, and ndof.
    cms::alpakatools::device_buffer<Device, float[]> extNewState;        // [5*candCapacity]
    cms::alpakatools::device_buffer<Device, float[]> extNewCov;          // [15*candCapacity]
    cms::alpakatools::device_buffer<Device, float[]> extNewChi2;         // [candCapacity]
    cms::alpakatools::device_buffer<Device, int32_t[]> extNewNdof;       // [candCapacity]
    cms::alpakatools::device_buffer<Device, int32_t[]> sizes;            // [candCapacity] merged sizes
    cms::alpakatools::device_buffer<Device, int32_t[]> offsets;          // [candCapacity] inclusive prefix
    cms::alpakatools::device_buffer<Device, int32_t[]> acceptedByTuple;  // [maxNumberOfTuples] extIdx or -1
    // Snapshot of the original per-tuple hit lists + list ends, taken before the in-place
    // re-layout (the new offsets shift every list after the first accepted extension).
    cms::alpakatools::device_buffer<Device, uint32_t[]> snapIds;     // [hitCapacity]
    cms::alpakatools::device_buffer<Device, uint32_t[]> snapDetIds;  // [hitCapacity]
    // Pre-rewrite attached() flags. WriteFinal recomputes attached() from the CURRENT pass's extras
    // only, so a follow-on attach pass would otherwise clear the marks an earlier pass set; restoring
    // from this snapshot preserves them. All-zero on a first pass (the CA initialises attached = 0 and
    // no extras are in the lists yet), where restoring it is a no-op.
    cms::alpakatools::device_buffer<Device, uint8_t[]> snapAttached;  // [hitCapacity]
    cms::alpakatools::device_buffer<Device, uint32_t[]> snapEnds;     // [maxNumberOfTuples]
    cms::alpakatools::device_buffer<Device, int32_t[]> finalSizes;    // [maxNumberOfTuples]
    cms::alpakatools::device_buffer<Device, int32_t[]> finalOffsets;  // [maxNumberOfTuples] inclusive

    // Accessors for the three counter words.
    uint32_t* nExt() { return counters.data(); }
    int32_t* pfxCounter() { return reinterpret_cast<int32_t*>(counters.data()) + 1; }
    int32_t* pfxCounter2() { return reinterpret_cast<int32_t*>(counters.data()) + 2; }
  };

  // Post-walk phase 1 (after launchAttach): compact the extended candidates and build the merged
  // (arc-sorted) hit container the accept stage and the final rewrite consume.
  RefitScaffold buildRefitScaffold(Queue& queue,
                                   const AttachParams& params,
                                   float bf,
                                   ::reco::TrackSoAConstView tracks,
                                   ::reco::TrackHitSoAConstView trackHits,
                                   ::reco::TrackingRecHitConstView hits,
                                   uint32_t maxNumberOfTuples,
                                   uint32_t hitCapacity,
                                   AttachBuffers& bufs,
                                   // The full-hits OT source, so the merged (arc-sorted) container build can fetch
                                   // an OT extra's global position for its arc key. nullptr => merged-hits-only.
                                   const OTHitsSource* otSource = nullptr);

  // Post-walk phase 2: accept every extended candidate (the per-hit derived gate and the hole
  // hypothesis have already priced each extra inside the walk, so no separate re-fit test runs here),
  // then rewrite the track SoA row from the scratch fit, the per-tuple hit lists (TrackHitSoA + hit
  // container + hitOffsets, via a snapshot copy + full re-layout) and the nLayers count (CA layer
  // partition). Unextended tuples keep their fitted row and original hit list untouched; if the
  // re-layout would overflow the hit-list capacity the whole event falls back (counted, reported).
  // extLayers (the extender's own layer partition), modules, rhoMap and bfield are the geometry and
  // material handles of this stage.
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
                           // The full-hits OT source, so the final rewrite can fetch a tagged extra's OT
                           // position / frame / detId. nullptr => merged-hits-only.
                           const OTHitsSource* otSource = nullptr);

  // Run the attach search: pre-gate + compaction, per-candidate layer walk with the chi2 gates, and
  // cross-track arbitration of shared attached hits (atomic best-chi2 claim per hit). On return the
  // buffers hold, per candidate slot, the arbitration-surviving extra hit ids. passBuf is the
  // two-pass fit buffer: its zeroed rows identify never-fitted tuples (the caller must memset it
  // before use).
  void launchAttach(Queue& queue,
                    const AttachParams& params,
                    float bf,
                    const float* rhoMap,
                    const ExtPhiBinner* phiBinner,
                    ::reco::TrackSoAConstView tracks,
                    ::reco::TrackHitSoAConstView trackHits,
                    // pocket gate: per-track arm (0=prompt, 1=displaced), in the merged-SoA order the walk
                    // indexes; nullptr when the gate is off or arm-blind (then the kernel never reads it).
                    const uint8_t* armId,
                    ::reco::TrackingRecHitConstView hits,
                    ::reco::HitModuleSoAConstView hitModules,
                    ::reco::TrackingRecHitsMaskingConstView hitMask,
                    ::reco_extender::ExtenderLayersConstView extLayers,
                    ::reco::CAModulesConstView caModules,
                    const double* passBuf,
                    uint32_t maxNumberOfTuples,
                    uint32_t nHits,
                    AttachBuffers& bufs,
                    // The full-hits OT source, threaded to this entry point so the walk's round 1, its
                    // stack-partner scan and the rewrite can read raw OT rechits. nullptr => the walk
                    // scans merged hits only and no tagged extra can be produced.
                    const OTHitsSource* otSource = nullptr,
                    // Candidate mask: nullptr => no restriction; when set, only tuples with
                    // candidateMask[tuple] >= 0 enter the walk (a follow-on pass passes the previous
                    // pass's extended set here).
                    const int32_t* candidateMask = nullptr);

  // Merger-side attach: the one entry point of the attach stage (allocateAttachBuffers -> launchAttach
  // -> buildRefitScaffold -> launchVerifyRewrite), run once over the merged HP-selected collection with
  // the EventSetup full geometry (all CA layers, including the odd OT disks the prompt doublet graph
  // excludes). The search structures the merger does not already hold are built internally: (a) the
  // merged-hit ExtPhiBinner over the shared pixelRecHits SoA (same 256-bin per-CA-layer partition as
  // the CA's prepareHits), (b) the mutable per-tuple SequentialContainer WriteFinal re-lays the
  // extended hit lists into, (c) a null passBuf (every merged row is fitted, and Kernel_extPreGate reads
  // a null passBuf as exactly that). The caller supplies an empty hitMask ConstView ("all open"), the
  // full OTHitsSource (buildOTHitsSource with otStubs + the ES caLayers) and the extender layer surfaces.
  // Modifies the merged track SoA in place (state/cov/chi2/ndof + hit lists of accepted extensions).
  // When params.verbose, logs the per-CA-layer attach coverage. hitCapacity must cover the post-attach
  // total, so the caller allocates hit-list headroom before calling.
  // extendedMaskOut (optional, size >= nTracksCap): if non-null, receives this pass's per-tuple
  // extended mask (candidate index >= 0 iff the tuple gained an accepted extra, -1 otherwise), directly
  // usable as a refit mask (refitMergedTwins gates on mask[t] >= 0). Null => not written.
  void launchMergerAttach(Queue& queue,
                          const AttachParams& params,
                          float bfield,
                          const float* rhoMap,
                          ::reco::TrackSoAView tracks,
                          ::reco::TrackHitSoAView trackHits,
                          // pocket gate: per-track arm (0=prompt, 1=displaced), in the merged-SoA order; the
                          // merger fills it through the filterTracks compaction. nullptr => gate off/arm-blind.
                          const uint8_t* armId,
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
                          int32_t* extendedMaskOut = nullptr);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::caExtension

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAExtensionKernels_h
