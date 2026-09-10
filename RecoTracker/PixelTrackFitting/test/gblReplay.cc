// gblReplay.cc
//
// Host-side replay of device GBL fits from a BLDUMP_* text dump (the BL_LAYER_DUMP output of the
// BrokenLineFit kernels and of the merger refit). Every dumped track is re-fitted once with the CPU twin
// (interface/GeneralBrokenLine.h) from the device's own reference (fast_fit), material (matXX0/innerXX0)
// and field, so the replayed helix parameters and chi2 can be checked against the dumped ones, and the
// per-hit smoothed residual pulls are emitted in the curvilinear (U, V) frame -- biased (measurement
// included) and unbiased (leave-one-out, the DESY GblTrajectory::getMeasResults semantics) -- tagged by
// the hit's global (r, z) so analyze_layer_pulls.py can classify them by detector layer. The dumped
// material is cross-checked against the host material map (REPLAY_MATDIFF).
//
// Usage: gblReplay <dumpfile> [eLossPerX0=0.035] [tagFilter=-1] [arm=0]
//   Lines other than BLDUMP_* are ignored, so a raw cmsRun log works. tagFilter keeps one refit call site
//   (1 = merger sharpen, 2 = merger final refit, 3 = dedup-confirm; -1 = all). arm != 0 selects the A/B
//   mode described at gArm below. Not a unit test: it exits 1 without argv[1], hence NO_TESTRUN in the
//   BuildFile. Pipe the output to analyze_layer_pulls.py.
//
// Two structural differences from the device fit:
//   * node layout: the host twin only has prepareGblData (the N+2 arrival-node layout), not the 2N+1
//     two-thin-scatterer layout the kernel builds, so the replay checks the fit inputs and the solver,
//     not the production node chain bit for bit.
//   * material source: matXX0 comes from the dump; the host march in test/gblTestMaterial.h serves the
//     audit line, the upstream two-thin split and the expanded-geometry arm.
//
// Output (stdout):
//   REPLAY_TRK tk <id> N <n> mode <m> pt <> eta <> q <> chi2 <host> chi2dev <dev> dHp <max|hp diff|>
//                    tag <t> beff <B_eff> drop <fit-hit index or -1> ndof <dev>
//   REPLAY_HIT tk <id> i <idx> r <> z <> gem <> rU <> rV <> sU <> sV <> pbU <> pbV <> puU <> puV <> cond <>
//                    tag <t> det <> stub <> sflags <> ot <> sens <> pair <> lo <x y z> up <x y z>
//                    dphidr <> dphidrerr <> hid <> dropped <>
// gem is the largest eigenvalue of the global hit covariance (module-type discriminator: pixel/PS/2S),
// rU/rV the biased smoothed residual, sU/sV its sigma, pb*/pu* the biased/unbiased pulls, cond the
// condition number of the (U,V) measurement covariance (2S strips -> huge).
//
// Dump format. BLDUMP_TRK/HIT/FIT lines have a fixed part followed by optional "<key> <value>" pairs:
// TRK `beff` (the |z|-dependent effective field the device fit used) and `tag` (refit call site); HIT the
// per-hit provenance block (raw id, OT/stub flags, detectorIndex, packed stub flags, stacked sensor, both
// sensor positions of the stub pair); FIT `ndof` and `drop` (fit-hit index removed by the in-fit outlier
// stage, -1 if none). A present `beff` replaces the scalar bField everywhere the replay uses a field
// (GBL data preparation, PCA perigee, curvature->pT), because that is what the device fit used; a present
// `drop` disables that measurement before the solve, reproducing the device's post-drop re-solve. Dumps
// without these keys replay with the scalar field and no drop.

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "RecoTracker/PixelTrackFitting/interface/BrokenLine.h"
#include "RecoTracker/PixelTrackFitting/interface/GeneralBrokenLine.h"
// Host copy of the device material march (the upstream host BrokenLine.h carries none); see its header.
#include "RecoTracker/PixelTrackFitting/test/gblTestMaterial.h"

namespace {

  double gELossPerX0 = 0.035;  // keep identical to Kernel_BLFit; override via argv[2] for A/B studies
  int gTagFilter = -1;         // -1 = replay every call site; else only records carrying this tag

  // A/B arms (argv[4]). 0 = off, the per-hit-residual replay above. Any other value fits each record
  // twice, nominal and with one lever changed, and emits a single ARM_TRK line carrying both fitted pT
  // values instead of the REPLAY_TRK/REPLAY_HIT block.
  //    1  innerXX0 := 0 (no upstream material)
  //    3  yVar x100 on 2S-class stub nodes with |z| > 118 cm (endcap along-strip de-weight)
  //    7  outlier drop off: keep all N hits (ignore the dumped drop index)
  //    8  drop the 2S-disk stub measurements from the fit entirely
  //    9  2S-disk rank-1, keeping only the small-eigenvalue (across-strip) direction
  //   10  2S-disk rank-1, keeping only the large-eigenvalue (strip) direction [control]
  //   16  split every 2S-disk stub into its two per-sensor rechit nodes (each keeps the stub covariance)
  //   17  the same split with each sensor node made rank-1 across-strip
  // Arms 16/17 disable the device outlier drop on both legs: the dumped drop index addresses the original
  // hit list and cannot be mapped onto the expanded one.
  int gArm = 0;
  constexpr int kArmE1 = 1, kArmE3 = 3, kArmE6 = 7;
  constexpr int kArmE10c = 8, kArmE10a = 9, kArmE10b = 10;
  constexpr int kArmU3 = 16, kArmU3rank1 = 17;

  constexpr int kU3MaxN = 20;          // expanded multiplicity ceiling (N + #split 2S-disk stubs)
  constexpr double kGemPixMax = 3e-4;  // gem < this  => pixel
  constexpr double kGemPSMax = 0.1;    // gem < this  => PS (else 2S)
  // Geometric hit classes from gem + |z| (the thresholds analyze_layer_pulls.py::modtype uses), for the
  // rank-1 selection of arm 17, which runs on the expanded hit list where no provenance block exists.
  enum GeomClass { kGCNone = 0, kGC2Sdisk, kGC2Sbarrel, kGCPSdisk, kGCPSbarrel, kGCallOT };

  constexpr double kE12EndcapZ = 118.;  // barrel/disk boundary in |z| [cm]

  inline int geomClassOf(double gem, double z) {
    if (gem < kGemPixMax)
      return kGCNone;  // pixel
    const bool disk = std::abs(z) >= kE12EndcapZ;
    if (gem < kGemPSMax)
      return disk ? kGCPSdisk : kGCPSbarrel;
    return disk ? kGC2Sdisk : kGC2Sbarrel;
  }

  inline bool geomClassMatches(int want, int cls) {
    if (cls == kGCNone)
      return false;
    return (want == kGCallOT) ? true : (want == cls);
  }
  // 2S-disk node treatment inside solveOnce (arms 8-10). 0 = untouched (production).
  constexpr int kTwoSNone = 0, kTwoSDrop = 1, kTwoSKeepAcross = 2, kTwoSKeepStrip = 3;
  constexpr double kE10EndcapZ = 118.;  // same |z| boundary as the 2S-disk class
  constexpr uint8_t kPSBit = 0x40;      // reco::StubFlags isPS
  // Eigen-direction degeneracy guard: a node whose (U,V) covariance condition number is below this has
  // no well-defined strip / across-strip axis. Counted and reported, never silently assigned.
  constexpr double kDegenCond = 10.0;

  constexpr double kE3YVarScale = 100.;  // arm 3: yVar x100
  constexpr double kE3EndcapZ = 118.;    // arm 3: |z| > 118 cm is endcap
  constexpr double kE3TwoSGemMin = 0.1;  // "2S-class variance": ge3 max eigenvalue >= 0.1 cm^2
                                         // (the same threshold analyze_layer_pulls.py::modtype uses)

  // Per-hit provenance appended to BLDUMP_HIT by the phase-split merger refit (all zero when the dump
  // carries no provenance block, in which case the ge3-eigenvalue module inference `gem` is the only
  // classifier).
  struct HitInfo {
    bool valid = false;
    unsigned hid = 0;
    int isOT = 0;    // 1 = tagged raw-OT rechit extra
    int det = 0;     // detectorIndex
    int isStub = 0;  // 1 = merged OT stub or raw-OT extra
    int sflags = 0;  // packed reco::StubFlags byte (merged path only)
    int sens = 0;    // 0 = pixel/unknown, 1 = lower/inner sensor, 2 = upper/outer sensor
    int pair = 0;    // 1 = lo[]/up[] hold the stub pair's two sensor positions
    double lo[3] = {0, 0, 0};
    double up[3] = {0, 0, 0};
    double dphidr = 0., dphidrerr = 0., ptest = 0.;
  };

  struct DumpRec {
    unsigned tk = 0;
    int n = 0;
    double bField = 0.;
    double ff[4] = {0, 0, 0, 0};
    int q = 0;
    double innerXX0 = 0.;
    std::vector<std::array<double, 3>> xyz;
    std::vector<std::array<double, 6>> ge;
    std::vector<double> matXX0;  // per segment i->i+1 (last entry 0)
    std::vector<HitInfo> info;
    bool hasFit = false;
    double hp[5] = {0, 0, 0, 0, 0};
    double chi2 = 0.;
    // ---- appended (optional) fields of the phase-split merger dump ----
    double bEff = 0.;  // 0 = absent -> the replay falls back to the scalar bField
    int tag = -1;      // refit call site (-1 = absent)
    int ndof = 0;      // device ndof of the emitted state
    int drop = -1;     // fit-hit index removed by the in-fit outlier stage (-1 = none)
    // Field the DEVICE fit used: B_eff when the dump carries it, else the origin scalar.
    double fitB() const { return bEff > 0. ? bEff : bField; }
  };

  // Reads the trailing "<key> <value>..." pairs an extended BLDUMP line appends after its fixed part.
  // Unknown keys are skipped (single token), so a newer dump replays on an older binary too.
  void parseTrkTail(std::istringstream& ss, DumpRec& rec) {
    std::string k;
    while (ss >> k) {
      if (k == "beff")
        ss >> rec.bEff;
      else if (k == "tag")
        ss >> rec.tag;
      else
        ss >> k;
    }
  }

  void parseHitTail(std::istringstream& ss, HitInfo& hi) {
    std::string k;
    while (ss >> k) {
      if (k == "hid")
        ss >> hi.hid, hi.valid = true;
      else if (k == "ot")
        ss >> hi.isOT;
      else if (k == "det")
        ss >> hi.det;
      else if (k == "stub")
        ss >> hi.isStub;
      else if (k == "sflags")
        ss >> hi.sflags;
      else if (k == "sens")
        ss >> hi.sens;
      else if (k == "pair")
        ss >> hi.pair;
      else if (k == "lo")
        ss >> hi.lo[0] >> hi.lo[1] >> hi.lo[2];
      else if (k == "up")
        ss >> hi.up[0] >> hi.up[1] >> hi.up[2];
      else if (k == "dphidr")
        ss >> hi.dphidr;
      else if (k == "dphidrerr")
        ss >> hi.dphidrerr;
      else if (k == "ptest")
        ss >> hi.ptest;
      else
        ss >> k;
    }
  }

  void parseFitTail(std::istringstream& ss, DumpRec& rec) {
    std::string k;
    while (ss >> k) {
      if (k == "ndof")
        ss >> rec.ndof;
      else if (k == "tag")
        ss >> rec.tag;
      else if (k == "drop")
        ss >> rec.drop;
      else
        ss >> k;
    }
  }

  // Dense normal-equation assembly over a hits-only node set (mirrors the gblFitPca dense path).
  void assembleAb(const std::vector<generalBrokenLine::GblNodeData>& sub,
                  Eigen::MatrixXd& A,
                  Eigen::VectorXd& b,
                  double& chi2ref) {
    using namespace generalBrokenLine;
    const int nN = static_cast<int>(sub.size());
    const int nP = 1 + 2 * nN;
    auto uIdx = [](int k) { return 1 + 2 * k; };
    A = Eigen::MatrixXd::Zero(nP, nP);
    b = Eigen::VectorXd::Zero(nP);
    chi2ref = 0.;
    for (int i = 0; i < nN; ++i)
      if (sub[i].hasMeas) {
        A.block<2, 2>(uIdx(i), uIdx(i)) += sub[i].measPrec;
        b.segment<2>(uIdx(i)) += sub[i].measPrec * sub[i].measResidual;
        chi2ref += sub[i].measResidual.dot(sub[i].measPrec * sub[i].measResidual);
      }
    for (int i = 1; i < nN - 1; ++i)
      if (sub[i].hasScat) {
        Matrix2d prevW, prevWJ, nextW, nextWJ;
        Vector2d prevWd, nextWd;
        prevDerivatives(sub[i].jacToPrev, prevW, prevWJ, prevWd);
        nextDerivatives(sub[i + 1].jacToPrev, nextW, nextWJ, nextWd);
        const Matrix2d sumWJ = prevWJ + nextWJ;
        const Vector2d sumWd = prevWd + nextWd;
        Eigen::Matrix<double, 2, 7> D;
        D.block<2, 1>(0, 0) = -sumWd;
        D.block<2, 2>(0, 1) = prevW;
        D.block<2, 2>(0, 3) = -sumWJ;
        D.block<2, 2>(0, 5) = nextW;
        const Eigen::Matrix<double, 7, 7> K = D.transpose() * sub[i].scatPrec * D;
        const int idx[7] = {0, uIdx(i - 1), uIdx(i - 1) + 1, uIdx(i), uIdx(i) + 1, uIdx(i + 1), uIdx(i + 1) + 1};
        for (int a = 0; a < 7; ++a)
          for (int c = 0; c < 7; ++c)
            A(idx[a], idx[c]) += K(a, c);
      }
  }

  // prepareGblData and gblHelixAtPca take four runtime model switches. The dump carries no record of
  // them, so the replay pins them to the merger refit's values, all on.
  constexpr bool kTrajectoryCorrections = true;  // useTrajectoryCorrections
  constexpr bool kScatteringLogAtTotal = true;   // useScatteringLogAtTotal
  constexpr bool kElossCumulative = true;        // useCumulativeEloss
  constexpr bool kChargeSymmetric = true;        // useChargeSymmetricCorrections (gblHelixAtPca)
  // The normalized (Bz,Br) r-z field map is an EventSetup product with no host-side table, so the
  // field-profile offset half of the charge-symmetric package cannot be reproduced here; nullptr is what
  // production passes when the map is absent. The arc-sign half is reproduced in full.
  constexpr const float* kBFieldMapUnavailable = nullptr;

  template <int N>
  void processTrack(const DumpRec& rec) {
    using namespace generalBrokenLine;
    Eigen::Matrix<double, 3, N> hits;
    Eigen::Matrix<float, 6, N> hits_ge;
    for (int i = 0; i < N; ++i) {
      for (int k = 0; k < 3; ++k)
        hits(k, i) = rec.xyz[i][k];
      for (int k = 0; k < 6; ++k)
        hits_ge(k, i) = float(rec.ge[i][k]);
    }
    Eigen::Vector4d ff(rec.ff[0], rec.ff[1], rec.ff[2], rec.ff[3]);
    // The field the device fit used: the merger refit uses a per-track |z|-effective field, so a merger
    // dump replayed with the origin scalar would differ from the device. Dumps without it use the scalar.
    const double bFit = rec.fitB();

    // arc lengths / charge from the host twin (pure geometry, identical math to the device)
    brokenline::PreparedBrokenLineData<N> data;
    brokenline::prepareBrokenLineData(hits, ff, bFit, data);
    // the material rows, from the test-local march
    gblTestMaterial::MatData<N> md;
    gblTestMaterial::fillMatData<N>(hits, md);
    if (data.qCharge != rec.q)
      printf("REPLAY_WARN tk %u qCharge host %d != device %d\n", rec.tk, data.qCharge, rec.q);
    // material-map audit: host baked-in map vs the device ES product (dumped values)
    for (int i = 0; i + 1 < N; ++i)
      if (std::abs(md.matXX0[i] - rec.matXX0[i]) > 1e-9 + 1e-6 * rec.matXX0[i])
        printf("REPLAY_MATDIFF tk %u seg %d host %.9g device %.9g\n", rec.tk, i, md.matXX0[i], rec.matXX0[i]);
    if (std::abs(md.innerXX0 - rec.innerXX0) > 1e-9 + 1e-6 * rec.innerXX0)
      printf("REPLAY_MATDIFF tk %u inner host %.9g device %.9g\n", rec.tk, md.innerXX0, rec.innerXX0);
    // replay with the DEVICE material (bit-faithful to the dumped fit)
    riemannFit::VectorNd<N> matD;
    for (int i = 0; i < N; ++i)
      matD(i) = (i + 1 < N) ? rec.matXX0[i] : 0.;

    double bUse = bFit, bConv = bFit;

    // Two-thin-scatterer split of the upstream (beamline->hit0) material from the host map, with
    // segmentXX0GapSplit (trapezoid rule) as the kernel takes it, not the rectangle-weighted
    // segmentXX0Moments, which would shift innerD1/innerW1.
    double innerD1 = 0., innerW1 = 0.;
    if (rec.innerXX0 > 0.) {
      const double rHit0 = std::hypot(hits(0, 0), hits(1, 0));
      gblTestMaterial::segmentXX0GapSplit(0., 0., rHit0, hits(2, 0), innerD1, innerW1);
    }
    std::vector<GblNodeData> nodes(N + 2);
    Matrix5d jacBack;
    bool usedInner = false;
    prepareGblData<N>(hits,
                      hits_ge,
                      ff,
                      bUse,
                      rec.q,
                      data.sTransverse,
                      data.sTotal,
                      matD,
                      rec.innerXX0,
                      nodes.data(),
                      /*msScale=*/1.0,
                      gELossPerX0,
                      &jacBack,
                      innerD1,
                      innerW1,
                      &usedInner,
                      // model switches pinned to the production values (see above)
                      kBFieldMapUnavailable,
                      /*bFieldOrigin=*/rec.bField,
                      kTrajectoryCorrections,
                      kScatteringLogAtTotal,
                      kElossCumulative);

    // node set actually fitted: full system [PCA, scatterer, hits...] (inner-node layout, extraction at PCA)
    // or the fallback hits-only set with the upstream scattering re-added as angle process noise afterwards.
    double th2Inner = 0.;
    std::vector<GblNodeData> sub;
    int hit0 = 0;  // index of hit 0 within `sub`
    if (usedInner) {
      sub.assign(nodes.begin(), nodes.begin() + N + 2);
      hit0 = 2;
    } else {
      th2Inner = nodes[1].hasScat ? 1.0 / nodes[1].scatPrec(0, 0) : 0.0;
      nodes[1].hasScat = false;
      sub.assign(nodes.begin() + 1, nodes.begin() + N + 1);
    }

    // In-fit outlier drop: the device's outlier phase disables the named measurement node and re-solves
    // once on the same linearization, so disabling it here reproduces the state the dump reports.
    // `drop` is the fit-hit index (sub[hit0 + drop]).
    if (rec.drop >= 0 && rec.drop < N)
      sub[hit0 + rec.drop].hasMeas = false;

    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    double chi2ref = 0.;
    assembleAb(sub, A, b, chi2ref);
    const int nP = 1 + 2 * int(sub.size());
    auto uIdx = [](int k) { return 1 + 2 * k; };
    Eigen::LLT<Eigen::MatrixXd> llt(A);
    if (llt.info() != Eigen::Success) {
      printf("REPLAY_WARN tk %u LLT failed\n", rec.tk);
      return;
    }
    const Eigen::MatrixXd cov = llt.solve(Eigen::MatrixXd::Identity(nP, nP));
    const Eigen::VectorXd delta = cov * b;
    const double chi2 = chi2ref - delta.dot(b);

    // track-level fidelity: full pipeline vs the dump (inner-node layout extracts at the PCA directly; the
    // fallback layout adds the upstream angle noise at hit0 and propagates with jacBack)
    Vector5d gcorr;
    double gchi2 = 0.;
    Matrix5d gcov = gblFitPca(sub, &gcorr, nullptr, &gchi2);
    Vector5d corrPca;
    Matrix5d covPca;
    if (usedInner) {
      corrPca = gcorr;
      covPca = gcov;
    } else {
      const double slopeQ = -double(rec.q) / ff(3);
      gcov(1, 1) += th2Inner;
      gcov(2, 2) += th2Inner * (1.0 + slopeQ * slopeQ);
      corrPca = jacBack * gcorr;
      covPca = jacBack * gcov * jacBack.transpose();
    }
    Vector5d hp;
    Matrix5d hc;
    // trailing `chargeSymmetric`: the production value (useChargeSymmetricCorrections is ON).
    gblHelixAtPca(
        ff, rec.q, bConv, double(data.sTransverse(0)), hits(2, 0), corrPca, covPca, hp, hc, nullptr, kChargeSymmetric);
    const double pt = bConv / std::abs(hp(2));
    const double eta = std::asinh(hp(3));
    hp(2) /= bConv;  // SoA convention of the dumped hp
    double dHp = 0.;
    if (rec.hasFit)
      for (int a = 0; a < 5; ++a)
        dHp = std::max(dHp, std::abs(hp(a) - rec.hp[a]));
    printf(
        "REPLAY_TRK tk %u N %d mode %d pt %.5g eta %.4g q %d chi2 %.6g chi2dev %.6g dHp %.3g tag %d beff %.9g drop "
        "%d ndof %d\n",
        rec.tk,
        N,
        0,
        pt,
        eta,
        rec.q,
        chi2,
        rec.hasFit ? rec.chi2 : std::nan(""),
        rec.hasFit ? dHp : std::nan(""),
        rec.tag,
        bFit,
        rec.drop,
        rec.ndof);

    // per-hit smoothed residual pulls (sub[hit0 + i] = hit i)
    for (int i = 0; i < N; ++i) {
      const int si = hit0 + i;
      const bool dropped = (rec.drop == i);
      if (!sub[si].hasMeas && !dropped)
        continue;
      const double r = std::hypot(hits(0, i), hits(1, i));
      const double z = hits(2, i);
      // module-type discriminator: the largest eigenvalue of the global hit covariance (pixel ~1e-5 cm^2,
      // PS macro-pixel ~2e-3, 2S strip ~2 -- two orders of magnitude apart each)
      Eigen::Matrix3d ge3;
      ge3 << hits_ge(0, i), hits_ge(1, i), hits_ge(3, i), hits_ge(1, i), hits_ge(2, i), hits_ge(4, i), hits_ge(3, i),
          hits_ge(4, i), hits_ge(5, i);
      const double gem = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(ge3).eigenvalues()(2);
      const Eigen::Vector2d u = delta.segment<2>(uIdx(si));
      const Eigen::Matrix2d Cii = cov.block<2, 2>(uIdx(si), uIdx(si));
      const Eigen::Matrix2d V = sub[si].measPrec.inverse();  // effective (U,V) measurement covariance
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(V);
      const double cond = es.eigenvalues()(1) / std::max(es.eigenvalues()(0), 1e-300);
      const Eigen::Vector2d rb = sub[si].measResidual - u;
      // A dropped node never entered the normal equations, so its offset covariance is already the
      // leave-one-out one: the residual sigma adds (V + C) instead of subtracting (V - C).
      const Eigen::Matrix2d Sb = dropped ? Eigen::Matrix2d(V + Cii) : Eigen::Matrix2d(V - Cii);
      double pbU = std::nan(""), pbV = std::nan("");
      if (Sb(0, 0) > 0.)
        pbU = rb(0) / std::sqrt(Sb(0, 0));
      if (Sb(1, 1) > 0.)
        pbV = rb(1) / std::sqrt(Sb(1, 1));
      // unbiased (leave-one-out): remove this node's measurement, re-solve
      double puU = std::nan(""), puV = std::nan("");
      if (dropped) {  // already leave-one-out: the device fit excluded this measurement
        puU = pbU;
        puV = pbV;
      } else {
        Eigen::MatrixXd Ap = A;
        Ap.block<2, 2>(uIdx(si), uIdx(si)) -= sub[si].measPrec;
        Eigen::VectorXd bp = b;
        bp.segment<2>(uIdx(si)) -= sub[si].measPrec * sub[si].measResidual;
        Eigen::LDLT<Eigen::MatrixXd> ldlt(Ap);
        if (ldlt.info() == Eigen::Success) {
          const Eigen::MatrixXd covp = ldlt.solve(Eigen::MatrixXd::Identity(nP, nP));
          if ((Ap * covp - Eigen::MatrixXd::Identity(nP, nP)).norm() < 1e-6 * nP) {  // reject singular leave-one-out
            const Eigen::VectorXd dp = covp * bp;
            const Eigen::Vector2d up = dp.segment<2>(uIdx(si));
            const Eigen::Matrix2d Cp = covp.block<2, 2>(uIdx(si), uIdx(si));
            const Eigen::Vector2d ru = sub[si].measResidual - up;
            const Eigen::Matrix2d Su = V + Cp;
            if (Su(0, 0) > 0.)
              puU = ru(0) / std::sqrt(Su(0, 0));
            if (Su(1, 1) > 0.)
              puV = ru(1) / std::sqrt(Su(1, 1));
          }
        }
      }
      // Per-hit provenance, dumped where available, supersedes the ge3-eigenvalue `gem` module inference
      // above; `gem` is still emitted so dumps without the provenance block keep their only classifier.
      const HitInfo hi = (i < int(rec.info.size())) ? rec.info[i] : HitInfo{};
      printf(
          "REPLAY_HIT tk %u i %d r %.4g z %.5g gem %.3g rU %.6g rV %.6g sU %.6g sV %.6g pbU %.5g pbV %.5g puU "
          "%.5g puV %.5g cond %.3g tag %d det %d stub %d sflags %d ot %d sens %d pair %d lo %.6g %.6g %.6g up "
          "%.6g %.6g %.6g dphidr %.6g dphidrerr %.6g hid %u dropped %d\n",
          rec.tk,
          i,
          r,
          z,
          gem,
          rb(0),
          rb(1),
          Sb(0, 0) > 0. ? std::sqrt(Sb(0, 0)) : std::nan(""),
          Sb(1, 1) > 0. ? std::sqrt(Sb(1, 1)) : std::nan(""),
          pbU,
          pbV,
          puU,
          puV,
          cond,
          rec.tag,
          hi.valid ? hi.det : -1,
          hi.valid ? hi.isStub : -1,
          hi.valid ? hi.sflags : -1,
          hi.valid ? hi.isOT : -1,
          hi.valid ? hi.sens : -1,
          hi.valid ? hi.pair : -1,
          hi.lo[0],
          hi.lo[1],
          hi.lo[2],
          hi.up[0],
          hi.up[1],
          hi.up[2],
          hi.dphidr,
          hi.dphidrerr,
          hi.hid,
          dropped ? 1 : 0);
    }
  }

  // Arm machinery.
  struct ArmFit {
    double pt = std::nan("");
    double eta = std::nan("");
    double chi2 = std::nan("");
    double sigrel = std::nan("");  // sigma(pT)/pT from the fitted covariance = sqrt(hc(2,2))/|kappa|
    bool ok = false;
  };

  // One full host GBL solve of a dumped record, with the arm levers exposed.
  //   innerXX0Use    : upstream X/X0 fed to the fit (arm 1 passes 0)
  //   applyE3        : de-weight the along-strip direction of 2S-class endcap stub nodes (arm 3)
  //   ignoreDrop     : keep the measurement the device's outlier stage dropped (arm 7, arms 16/17)
  //   twoSMode       : 2S-disk node treatment (arms 8-10), with counters of the nodes acted on / skipped
  //   rank1GeomClass : rank-1 across-strip on every node of one geometric class (arm 17)
  // Everything else (reference, material, charge, field, the dumped outlier drop) is production.
  template <int N>
  ArmFit solveOnce(const DumpRec& rec,
                   const Eigen::Matrix<double, 3, N>& hits,
                   const Eigen::Matrix<float, 6, N>& hits_ge,
                   const Eigen::Vector4d& ff,
                   const brokenline::PreparedBrokenLineData<N>& data,
                   const riemannFit::VectorNd<N>& matD,
                   double bFit,
                   double innerXX0Use,
                   bool applyE3,
                   bool ignoreDrop = false,
                   int twoSMode = kTwoSNone,
                   int* nTwoSSel = nullptr,
                   int* nTwoSDegen = nullptr,
                   int rank1GeomClass = kGCNone) {
    using namespace generalBrokenLine;
    ArmFit out;
    double innerD1 = 0., innerW1 = 0.;
    if (innerXX0Use > 0.) {
      const double rHit0 = std::hypot(hits(0, 0), hits(1, 0));
      gblTestMaterial::segmentXX0GapSplit(0., 0., rHit0, hits(2, 0), innerD1, innerW1);
    }
    std::vector<GblNodeData> nodes(N + 2);
    Matrix5d jacBack;
    bool usedInner = false;
    prepareGblData<N>(hits,
                      hits_ge,
                      ff,
                      bFit,
                      rec.q,
                      data.sTransverse,
                      data.sTotal,
                      matD,
                      innerXX0Use,
                      nodes.data(),
                      /*msScale=*/1.0,
                      gELossPerX0,
                      &jacBack,
                      innerD1,
                      innerW1,
                      &usedInner,
                      // Same four production model switches as the plain replay path above.
                      kBFieldMapUnavailable,
                      /*bFieldOrigin=*/rec.bField,
                      kTrajectoryCorrections,
                      kScatteringLogAtTotal,
                      kElossCumulative);

    double th2Inner = 0.;
    std::vector<GblNodeData> sub;
    int hit0 = 0;
    if (usedInner) {
      sub.assign(nodes.begin(), nodes.begin() + N + 2);
      hit0 = 2;
    } else {
      th2Inner = nodes[1].hasScat ? 1.0 / nodes[1].scatPrec(0, 0) : 0.0;
      nodes[1].hasScat = false;
      sub.assign(nodes.begin() + 1, nodes.begin() + N + 1);
    }

    // Arm 3: de-weight the along-strip direction of 2S-class endcap stub measurements by kE3YVarScale.
    // The dump carries only the global 3x3 hit covariance, from which the local (xerr, yerr) pair cannot
    // be separated, so the scaling is applied in the (U,V) measurement frame to the larger eigenvalue of
    // the 2x2 measurement covariance, the strip-length direction for a 2S sensor. Scaling the global
    // covariance wholesale would also de-weight the precise across-strip coordinate.
    if (applyE3) {
      for (int i = 0; i < N; ++i) {
        const int si = hit0 + i;
        if (!sub[si].hasMeas)
          continue;
        static const HitInfo kNoInfo{};
        const HitInfo& hi = (i < int(rec.info.size())) ? rec.info[i] : kNoInfo;
        if (hi.valid && hi.isStub != 1)
          continue;  // pixel node
        if (std::abs(hits(2, i)) <= kE3EndcapZ)
          continue;  // barrel
        Eigen::Matrix3d ge3;
        ge3 << hits_ge(0, i), hits_ge(1, i), hits_ge(3, i), hits_ge(1, i), hits_ge(2, i), hits_ge(4, i), hits_ge(3, i),
            hits_ge(4, i), hits_ge(5, i);
        const double gem = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(ge3).eigenvalues()(2);
        if (gem < kE3TwoSGemMin)
          continue;  // not 2S-class variance
        const Eigen::Matrix2d V = sub[si].measPrec.inverse();
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(V);
        Eigen::Vector2d ev = es.eigenvalues();
        const Eigen::Matrix2d Q = es.eigenvectors();
        ev(1) *= kE3YVarScale;  // eigenvalues ascending: (1) is the along-strip direction
        const Eigen::Matrix2d Vnew = Q * ev.asDiagonal() * Q.transpose();
        sub[si].measPrec = Vnew.inverse();
      }
    }

    // Arms 8-10: 2S-disk node ablation and channel isolation. A 2S-disk node is a merged-SoA stub
    // (dumped stub==1, ot==0) that is not PS (StubFlags PS bit clear when sflags is set, else ge3 max
    // eigenvalue >= 0.1 cm^2) and sits at |z| >= 118 cm. Raw-OT extras (ot==1) are excluded.
    if (twoSMode != kTwoSNone) {
      for (int i = 0; i < N; ++i) {
        const int si = hit0 + i;
        if (!sub[si].hasMeas)
          continue;
        static const HitInfo kNoInfo2{};
        const HitInfo& hi = (i < int(rec.info.size())) ? rec.info[i] : kNoInfo2;
        if (!hi.valid || hi.isStub != 1 || hi.isOT == 1)
          continue;
        if (std::abs(hits(2, i)) < kE10EndcapZ)
          continue;
        Eigen::Matrix3d ge3;
        ge3 << hits_ge(0, i), hits_ge(1, i), hits_ge(3, i), hits_ge(1, i), hits_ge(2, i), hits_ge(4, i), hits_ge(3, i),
            hits_ge(4, i), hits_ge(5, i);
        const double gem = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(ge3).eigenvalues()(2);
        const bool isps = (hi.sflags > 0) ? ((hi.sflags & kPSBit) != 0) : (gem < kE3TwoSGemMin);
        if (isps)
          continue;  // PS-disk, not 2S-disk
        if (nTwoSSel)
          ++(*nTwoSSel);
        if (twoSMode == kTwoSDrop) {
          sub[si].hasMeas = false;
          continue;
        }
        // rank-1 channel isolation: keep one eigen-direction of the (U,V) measurement covariance and
        // give the other zero weight (infinite variance).
        const Eigen::Matrix2d V = sub[si].measPrec.inverse();
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(V);
        const Eigen::Vector2d ev = es.eigenvalues();  // ascending: (0) across-strip, (1) strip
        const double cond = ev(1) / std::max(ev(0), 1e-300);
        if (cond < kDegenCond) {
          if (nTwoSDegen)
            ++(*nTwoSDegen);  // no well-defined axis: leave the node untouched, and count it
          continue;
        }
        const Eigen::Vector2d e = es.eigenvectors().col(twoSMode == kTwoSKeepAcross ? 0 : 1);
        const double lam = ev(twoSMode == kTwoSKeepAcross ? 0 : 1);
        sub[si].measPrec = (e * e.transpose()) / lam;  // rank-1 precision
      }
    }

    // Arm 17: rank-1 across-strip on nodes of one geometric class. Selection is by gem and |z| rather
    // than by provenance, so it works on the expanded hit list, whose synthetic record carries none.
    if (rank1GeomClass != kGCNone) {
      for (int i = 0; i < N; ++i) {
        const int si = hit0 + i;
        if (!sub[si].hasMeas)
          continue;
        Eigen::Matrix3d ge3;
        ge3 << hits_ge(0, i), hits_ge(1, i), hits_ge(3, i), hits_ge(1, i), hits_ge(2, i), hits_ge(4, i), hits_ge(3, i),
            hits_ge(4, i), hits_ge(5, i);
        const double gem = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(ge3).eigenvalues()(2);
        if (!geomClassMatches(rank1GeomClass, geomClassOf(gem, hits(2, i))))
          continue;
        const Eigen::Matrix2d V = sub[si].measPrec.inverse();
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(V);
        const Eigen::Vector2d ev = es.eigenvalues();
        if (ev(1) / std::max(ev(0), 1e-300) < kDegenCond)
          continue;                                          // no defined axis: leave untouched
        const Eigen::Vector2d e = es.eigenvectors().col(0);  // smallest = across-strip
        sub[si].measPrec = (e * e.transpose()) / ev(0);
      }
    }

    if (!ignoreDrop && rec.drop >= 0 && rec.drop < N)
      sub[hit0 + rec.drop].hasMeas = false;

    Vector5d gcorr;
    double gchi2 = 0.;
    Matrix5d gcov = gblFitPca(sub, &gcorr, nullptr, &gchi2);
    Vector5d corrPca;
    Matrix5d covPca;
    if (usedInner) {
      corrPca = gcorr;
      covPca = gcov;
    } else {
      const double slopeQ = -double(rec.q) / ff(3);
      gcov(1, 1) += th2Inner;
      gcov(2, 2) += th2Inner * (1.0 + slopeQ * slopeQ);
      corrPca = jacBack * gcorr;
      covPca = jacBack * gcov * jacBack.transpose();
    }
    Vector5d hp;
    Matrix5d hc;
    gblHelixAtPca(ff,
                  rec.q,
                  bFit,
                  double(data.sTransverse(0)),
                  hits(2, 0),
                  corrPca,
                  covPca,
                  hp,
                  hc,
                  nullptr,
                  kChargeSymmetric);  // production value (useChargeSymmetricCorrections ON)
    out.pt = bFit / std::abs(hp(2));
    out.eta = std::asinh(hp(3));
    out.chi2 = gchi2;
    // sigma(pT)/pT: pT = B/|kappa| => d(pT)/pT = d(kappa)/|kappa|. The ratio is invariant under the
    // SoA 1/pT rescaling, both sides carrying one power of B.
    out.sigrel = (hc(2, 2) > 0. && hp(2) != 0.) ? std::sqrt(hc(2, 2)) / std::abs(hp(2)) : std::nan("");
    out.ok = std::isfinite(out.pt) && std::isfinite(out.eta) && out.pt > 0.;
    return out;
  }

  // Arms 16/17: expanded-hit solve. The expanded hit list is fitted from scratch, with arc lengths,
  // charge and material recomputed by the host twin over the new geometry, because the dump's
  // per-segment matXX0 addresses the original segmentation.
  template <int NE>
  ArmFit solveExpanded(const DumpRec& rec,
                       const std::vector<std::array<double, 3>>& xyz,
                       const std::vector<std::array<double, 6>>& ge,
                       const Eigen::Vector4d& ff,
                       double bFit,
                       bool rank1Across) {
    Eigen::Matrix<double, 3, NE> hits;
    Eigen::Matrix<float, 6, NE> hits_ge;
    for (int i = 0; i < NE; ++i) {
      for (int k = 0; k < 3; ++k)
        hits(k, i) = xyz[i][k];
      for (int k = 0; k < 6; ++k)
        hits_ge(k, i) = float(ge[i][k]);
    }
    brokenline::PreparedBrokenLineData<NE> data;
    brokenline::prepareBrokenLineData(hits, ff, bFit, data);
    gblTestMaterial::MatData<NE> md;
    gblTestMaterial::fillMatData<NE>(hits, md);
    riemannFit::VectorNd<NE> matD;
    for (int i = 0; i < NE; ++i)
      matD(i) = (i + 1 < NE) ? md.matXX0[i] : 0.;
    DumpRec r2 = rec;
    r2.drop = -1;  // the dumped drop index addresses the original hit list
    r2.info.clear();
    return solveOnce<NE>(r2,
                         hits,
                         hits_ge,
                         ff,
                         data,
                         matD,
                         bFit,
                         rec.innerXX0,
                         /*applyE3=*/false,
                         /*ignoreDrop=*/true,
                         kTwoSNone,
                         nullptr,
                         nullptr,
                         rank1Across ? kGC2Sdisk : kGCNone);
  }

  ArmFit dispatchExpanded(int ne,
                          const DumpRec& rec,
                          const std::vector<std::array<double, 3>>& xyz,
                          const std::vector<std::array<double, 6>>& ge,
                          const Eigen::Vector4d& ff,
                          double bFit,
                          bool rank1) {
    switch (ne) {
#define U3CASE(N) \
  case N:         \
    return solveExpanded<N>(rec, xyz, ge, ff, bFit, rank1)
      U3CASE(4);
      U3CASE(5);
      U3CASE(6);
      U3CASE(7);
      U3CASE(8);
      U3CASE(9);
      U3CASE(10);
      U3CASE(11);
      U3CASE(12);
      U3CASE(13);
      U3CASE(14);
      U3CASE(15);
      U3CASE(16);
      U3CASE(17);
      U3CASE(18);
      U3CASE(19);
      U3CASE(20);
#undef U3CASE
      default:
        return ArmFit{};
    }
  }

  const HitInfo kNoInfoArm{};

  // Arm mode: fit the record twice and emit one paired line. Nominal is bit-identical to the plain
  // replay path (same helper, all levers at production values), so pt0 doubles as a self-check.
  template <int N>
  void processTrackArm(const DumpRec& rec) {
    Eigen::Matrix<double, 3, N> hits;
    Eigen::Matrix<float, 6, N> hits_ge;
    for (int i = 0; i < N; ++i) {
      for (int k = 0; k < 3; ++k)
        hits(k, i) = rec.xyz[i][k];
      for (int k = 0; k < 6; ++k)
        hits_ge(k, i) = float(rec.ge[i][k]);
    }
    Eigen::Vector4d ff(rec.ff[0], rec.ff[1], rec.ff[2], rec.ff[3]);
    const double bFit = rec.fitB();
    brokenline::PreparedBrokenLineData<N> data;
    brokenline::prepareBrokenLineData(hits, ff, bFit, data);
    riemannFit::VectorNd<N> matD;
    for (int i = 0; i < N; ++i)
      matD(i) = (i + 1 < N) ? rec.matXX0[i] : 0.;

    ArmFit nomBase = solveOnce<N>(rec, hits, hits_ge, ff, data, matD, bFit, rec.innerXX0, false);
    ArmFit nomOverride;
    bool useNomOverride = false;
    ArmFit arm;
    int n2Sd = 0;    // arms 8-10: qualifying 2S-disk nodes acted on; arms 16/17: stubs split
    int n2Sdeg = 0;  // arms 9/10: qualifying nodes skipped because their (U,V) axes are degenerate
    if (gArm == kArmE1) {
      arm = solveOnce<N>(rec, hits, hits_ge, ff, data, matD, bFit, /*innerXX0=*/0., false);
    } else if (gArm == kArmE3) {
      arm = solveOnce<N>(rec, hits, hits_ge, ff, data, matD, bFit, rec.innerXX0, true);
    } else if (gArm == kArmE6) {
      arm = solveOnce<N>(rec, hits, hits_ge, ff, data, matD, bFit, rec.innerXX0, false, /*ignoreDrop=*/true);
    } else if (gArm == kArmU3 || gArm == kArmU3rank1) {
      // Build the expanded hit list: every 2S-disk stub with a valid sensor pair is replaced by its two
      // per-sensor rechits (dumped `lo`/`up`). Both sensors of a 2S stack are parallel and share the
      // pitch and the strip length, so each sensor node inherits the stub's own global covariance.
      std::vector<std::array<double, 3>> xyz;
      std::vector<std::array<double, 6>> ge;
      int nsplit = 0;
      for (int i = 0; i < N; ++i) {
        const HitInfo& hi = (i < int(rec.info.size())) ? rec.info[i] : kNoInfoArm;
        const bool twoSdisk = hi.valid && hi.isStub == 1 && hi.isOT == 0 && hi.pair == 1 &&
                              std::abs(hits(2, i)) >= kE12EndcapZ && !(hi.sflags > 0 && (hi.sflags & kPSBit));
        std::array<double, 6> g6{rec.ge[i][0], rec.ge[i][1], rec.ge[i][2], rec.ge[i][3], rec.ge[i][4], rec.ge[i][5]};
        if (twoSdisk) {
          xyz.push_back({hi.lo[0], hi.lo[1], hi.lo[2]});
          ge.push_back(g6);
          xyz.push_back({hi.up[0], hi.up[1], hi.up[2]});
          ge.push_back(g6);
          ++nsplit;
        } else {
          xyz.push_back({hits(0, i), hits(1, i), hits(2, i)});
          ge.push_back(g6);
        }
      }
      n2Sd = nsplit;
      const int ne = int(xyz.size());
      if (nsplit == 0 || ne > kU3MaxN) {
        arm = nomBase;  // nothing to split (or beyond the instantiated ceiling): arm == nominal
      } else {
        // The expanded fit needs its own reference-consistent nominal, so both legs disable the device
        // outlier drop (the drop index addresses the original hit list).
        nomOverride = solveOnce<N>(rec,
                                   hits,
                                   hits_ge,
                                   ff,
                                   data,
                                   matD,
                                   bFit,
                                   rec.innerXX0,
                                   false,
                                   /*ignoreDrop=*/true);
        useNomOverride = true;
        arm = dispatchExpanded(ne, rec, xyz, ge, ff, bFit, gArm == kArmU3rank1);
      }
    } else if (gArm == kArmE10c || gArm == kArmE10a || gArm == kArmE10b) {
      const int mode = (gArm == kArmE10c) ? kTwoSDrop : ((gArm == kArmE10a) ? kTwoSKeepAcross : kTwoSKeepStrip);
      arm = solveOnce<N>(rec, hits, hits_ge, ff, data, matD, bFit, rec.innerXX0, false, false, mode, &n2Sd, &n2Sdeg);
    }

    const ArmFit& nom = useNomOverride ? nomOverride : nomBase;
    // Per-hit context the band/class splits need: hit0 radius, and the 2S-endcap node count (arm 3 scope).
    const double r0 = std::hypot(hits(0, 0), hits(1, 0));
    int n2Sec = 0, nStub = 0;
    for (int i = 0; i < N; ++i) {
      const bool st = (i < int(rec.info.size()) && rec.info[i].valid) ? (rec.info[i].isStub == 1) : false;
      if (st)
        ++nStub;
      if (st && std::abs(hits(2, i)) > kE3EndcapZ) {
        Eigen::Matrix3d ge3;
        ge3 << hits_ge(0, i), hits_ge(1, i), hits_ge(3, i), hits_ge(1, i), hits_ge(2, i), hits_ge(4, i), hits_ge(3, i),
            hits_ge(4, i), hits_ge(5, i);
        if (Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(ge3).eigenvalues()(2) >= kE3TwoSGemMin)
          ++n2Sec;
      }
    }
    const double dpt = (nom.ok && arm.ok) ? (arm.pt - nom.pt) / nom.pt : std::nan("");
    // Type of the hit the device's in-fit outlier stage dropped (arm 7 split): -1 none, 0 pixel, 1 stub.
    int droptype = -1;
    if (rec.drop >= 0 && rec.drop < N && rec.drop < int(rec.info.size()) && rec.info[rec.drop].valid)
      droptype = (rec.info[rec.drop].isStub == 1) ? 1 : 0;
    printf(
        "ARM_TRK arm %d tk %u N %d q %d eta %.5g pt0 %.6g pt1 %.6g dpt %.8g chi20 %.6g chi21 %.6g r0 %.4g "
        "nstub %d n2Sec %d ok %d innerXX0 %.6g sigrel %.6g drop %d droptype %d sigrel1 %.6g n2Sd %d n2Sdeg %d "
        "nmeas1 %d\n",
        gArm,
        rec.tk,
        N,
        rec.q,
        nom.eta,
        nom.pt,
        arm.pt,
        dpt,
        nom.chi2,
        arm.chi2,
        r0,
        nStub,
        n2Sec,
        (nom.ok && arm.ok) ? 1 : 0,
        rec.innerXX0,
        nom.sigrel,
        rec.drop,
        droptype,
        arm.sigrel,
        n2Sd,
        n2Sdeg,
        // measurements left in the arm fit: N minus the device drop minus the arm-8 removals
        N - ((rec.drop >= 0 && rec.drop < N) ? 1 : 0) - ((gArm == kArmE10c) ? n2Sd : 0));
  }

  void dispatchArm(const DumpRec& rec) {
    switch (rec.n) {
      case 3:
        return processTrackArm<3>(rec);
      case 4:
        return processTrackArm<4>(rec);
      case 5:
        return processTrackArm<5>(rec);
      case 6:
        return processTrackArm<6>(rec);
      case 7:
        return processTrackArm<7>(rec);
      case 8:
        return processTrackArm<8>(rec);
      case 9:
        return processTrackArm<9>(rec);
      case 10:
        return processTrackArm<10>(rec);
      case 11:
        return processTrackArm<11>(rec);
      case 12:
        return processTrackArm<12>(rec);
      default:
        printf("ARM_WARN tk %u unsupported N %d\n", rec.tk, rec.n);
    }
  }

  void dispatch(const DumpRec& rec) {
    if (gArm != 0)
      return dispatchArm(rec);
    switch (rec.n) {
      case 3:
        return processTrack<3>(rec);
      case 4:
        return processTrack<4>(rec);
      case 5:
        return processTrack<5>(rec);
      case 6:
        return processTrack<6>(rec);
      case 7:
        return processTrack<7>(rec);
      case 8:
        return processTrack<8>(rec);
      case 9:
        return processTrack<9>(rec);
      case 10:
        return processTrack<10>(rec);
      // The merger refit ladder runs to kRefitMaxN = 12 (the CA main fit stops at
      // maxHitsOnTrackForFullFit = 10), and its N=11/12 bins hold the longest-lever, most stub-rich
      // tracks, so both must be replayable.
      case 11:
        return processTrack<11>(rec);
      case 12:
        return processTrack<12>(rec);
      default:
        printf("REPLAY_WARN tk %u unsupported N %d\n", rec.tk, rec.n);
    }
  }

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: gblReplay <dumpfile (cmsRun log with BLDUMP_* lines)> [eLossPerX0=0.035] [tagFilter=-1]"
              << " [arm=0]\n  arm: 0 off | 1 innerXX0:=0 | 3 2S-endcap yVar x100 | 7 outlier drop off"
              << " | 8 drop 2S-disk | 9 2S-disk rank-1 across-strip | 10 2S-disk rank-1 strip"
              << " | 16 split 2S-disk stubs into two sensor nodes | 17 split + rank-1 across-strip" << std::endl;
    return 1;
  }
  if (argc > 2)
    gELossPerX0 = std::atof(argv[2]);
  if (argc > 3)
    gTagFilter = std::atoi(argv[3]);
  if (argc > 4)
    gArm = std::atoi(argv[4]);
  if (gArm != 0 && gArm != kArmE1 && gArm != kArmE3 && gArm != kArmE6 && gArm != kArmE10c && gArm != kArmE10a &&
      gArm != kArmE10b && gArm != kArmU3 && gArm != kArmU3rank1) {
    std::cerr << "unknown arm " << gArm << std::endl;
    return 1;
  }
  std::ifstream in(argv[1]);
  if (!in) {
    std::cerr << "cannot open " << argv[1] << std::endl;
    return 1;
  }
  std::string line, tag, skip;
  long nTrk = 0, nBad = 0, nSkipTag = 0, nOrphan = 0;
  // Generation-buffered parser. The phase-split refit emits TRK/HIT/FIT per kernel: one launch prints
  // every track's TRK+HIT block, a later launch prints every track's FIT, and the outlier launch
  // re-prints FIT for the tracks whose state it overwrote, so a record cannot be closed by the next TRK
  // line. Records are buffered by track id and dispatched as a generation, which ends the moment a track
  // id repeats: ids are unique within one refit launch of one event. FIT lines update the buffered record
  // in place, the last one winning, so the post-outlier state is the one kept.
  std::vector<DumpRec> pend;
  std::unordered_map<unsigned, std::size_t> idx;
  auto flushGeneration = [&]() {
    for (const DumpRec& r : pend) {
      if (int(r.xyz.size()) != r.n)
        ++nBad;
      else if (gTagFilter >= 0 && r.tag != gTagFilter)
        ++nSkipTag;
      else
        dispatch(r), ++nTrk;
    }
    pend.clear();
    idx.clear();
  };
  while (std::getline(in, line)) {
    if (line.compare(0, 7, "BLDUMP_") != 0)
      continue;
    std::istringstream ss(line);
    ss >> tag;
    if (tag == "BLDUMP_TRK") {
      DumpRec rec;
      ss >> rec.tk >> skip >> rec.n >> skip >> rec.bField >> skip >> rec.ff[0] >> rec.ff[1] >> rec.ff[2] >> rec.ff[3] >>
          skip >> rec.q >> skip >> rec.innerXX0;
      if (!ss) {
        ++nBad;
        continue;
      }
      parseTrkTail(ss, rec);
      if (idx.count(rec.tk))  // id repeat: the previous launch/event is complete
        flushGeneration();
      idx[rec.tk] = pend.size();
      pend.push_back(std::move(rec));
    } else if (tag == "BLDUMP_HIT") {
      unsigned tk;
      int i;
      std::array<double, 3> p;
      std::array<double, 6> g;
      double m;
      ss >> tk >> i >> p[0] >> p[1] >> p[2] >> g[0] >> g[1] >> g[2] >> g[3] >> g[4] >> g[5] >> m;
      auto it = idx.find(tk);
      if (!ss || it == idx.end()) {
        ++nOrphan;
        continue;
      }
      DumpRec& rec = pend[it->second];
      if (i != int(rec.xyz.size())) {
        ++nOrphan;
        continue;
      }
      rec.xyz.push_back(p);
      rec.ge.push_back(g);
      rec.matXX0.push_back(m);
      HitInfo hi;
      parseHitTail(ss, hi);
      rec.info.push_back(hi);
    } else if (tag == "BLDUMP_FIT") {
      unsigned tk;
      auto it = idx.end();
      ss >> tk;
      it = idx.find(tk);
      if (it == idx.end()) {
        ++nOrphan;
        continue;
      }
      DumpRec& rec = pend[it->second];
      ss >> skip >> rec.hp[0] >> rec.hp[1] >> rec.hp[2] >> rec.hp[3] >> rec.hp[4] >> skip >> rec.chi2;
      rec.hasFit = bool(ss);
      parseFitTail(ss, rec);
    }
  }
  flushGeneration();
  std::cerr << "gblReplay: " << nTrk << " tracks replayed, " << nBad << " malformed records skipped, " << nOrphan
            << " orphan HIT/FIT lines";
  if (gTagFilter >= 0)
    std::cerr << ", " << nSkipTag << " skipped by tag filter " << gTagFilter;
  std::cerr << std::endl;
  return 0;
}
