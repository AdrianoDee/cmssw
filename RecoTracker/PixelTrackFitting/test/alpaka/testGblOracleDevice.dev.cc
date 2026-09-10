// testGblOracleDevice.dev.cc
//
// The DESY oracle, on the device. For each of 15 fixtures (test/gblTestFixtures.h), the GBL node chain is
// built on the device through the production path, the calls BrokenLineFitKernels.h makes for the
// merger's refit:
//
//     brokenline::prepareGblFitData        (arc lengths, charge, the Geant4 material march, per-gap splits)
//     brokenline::segmentXX0GapSplit       (the beamline -> hit0 two-thin split)
//     generalBrokenLine::prepareGblDataSplit   (the 2N+1 layout production uses; prepareGblData is the fallback)
//     generalBrokenLine::gblFitPca             (the bordered-band solve at the PCA)
//     generalBrokenLine::gblHelixAtPca         (curvilinear -> (phi, d0, 1/R, cotTheta, z0))
//
// The node chain is then copied back and the device's own nodes are given to two independent host oracles:
//
//   (1) gbl::GblTrajectory, the DESY General-Broken-Lines library, run host-side in GblHostOracles.cc.
//       It is handed the chain the device solved, so it is asked only whether the solver agrees.
//   (2) the host twin, interface/GeneralBrokenLine.h, on the same nodes: serial, CUDA and ROCm all reduce
//       to the same host numbers.
//
// The measurement-only Cramer-Rao floor and the CMSSW PerigeeConversions cross-check are printed as
// physics diagnostics, not asserted: they are ratios, and a hard-coded ratio breaks when the physics
// improves. The tolerance block below lists every number that is asserted.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <alpaka/alpaka.hpp>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"
#include "RecoTracker/PixelTrackFitting/interface/BrokenLine.h"                // upstream host fastFit
#include "RecoTracker/PixelTrackFitting/interface/GeneralBrokenLine.h"         // host twin (oracle 2)
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BrokenLine.h"         // device prep under test
#include "RecoTracker/PixelTrackFitting/interface/alpaka/GeneralBrokenLine.h"  // device twin under test
#include "RecoTracker/PixelTrackFitting/test/GblHostOracles.h"                 // DESY + CMSSW, host-only .cc
#include "RecoTracker/PixelTrackFitting/test/gblTestFixtures.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
// `using namespace ALPAKA_ACCELERATOR_NAMESPACE` pulls the device twins into scope, so the bare names are
// ambiguous -- qualify the host ones with the global scope `::`.
namespace gblh = ::generalBrokenLine;                              // host twin
namespace blh = ::brokenline;                                      // host (upstream) fastFit
namespace gbld = ALPAKA_ACCELERATOR_NAMESPACE::generalBrokenLine;  // device twin
namespace bld = ALPAKA_ACCELERATOR_NAMESPACE::brokenline;          // device material / prep

namespace {

  // Every number the test asserts against lives here.
  //
  // kTolTwin: device vs host twin. Same algebra, bit-for-bit equal on the serial backend; on a device
  // backend nvcc contracts multiply-add into FMA where the host compiler does not, so the device result
  // solves a problem perturbed at the 1-ulp level and the forward size of that difference is set by the
  // condition number of the fixture's normal matrix: ~1e-7 on the well-conditioned fixtures, ~1e-5 on the
  // two near-degenerate ones. These are conditioning bounds, not accuracy bounds.
  // kTolDesy: device vs DESY on the same chain, two different solvers on an ill-conditioned system.
  // Entries far below the diagonal scale are skipped (kCovGate): a relative test on a numerically-zero
  // correlation measures nothing.
  //!< well-conditioned fixtures.
  constexpr double kTolTwin = 1e-6;
  //!< the two near-degenerate fixtures, the same two that get kTolDesyIllCond.
  constexpr double kTolTwinIllCond = 1e-3;
  constexpr double kTolDesy = 1e-6;
  //!< the two fixtures whose normal matrix is worst conditioned.
  constexpr double kTolDesyIllCond = 1e-4;
  //!< chi2 = chi2ref - delta.b is a difference of two large, nearly equal numbers whenever the fit is good,
  //!< so it loses digits the covariance does not. Applied to every fixture.
  constexpr double kTolDesyChi2 = 1e-4;
  //!< a covariance entry is compared relatively only when it is above this fraction of sqrt(Caa*Cbb).
  constexpr double kCovGate = 1e-3;

  // The field map and the three trajectory-model corrections are off, so the chain under test is the plain
  // constant-field GBL model. They are named rather than left to the defaults so that a change shows here.
  constexpr double kELossPerX0 = 0.0;             // ionization-loss correction off
  constexpr double kBFieldOrigin = 0.0;           // no (Bz,Br) map in these fixtures -> bMap == nullptr
  constexpr bool kTrajectoryCorrections = false;  // reference-trajectory corrections
  constexpr bool kScatteringLogAtTotal = false;   // Highland's log per gap, not at the track total
  constexpr bool kElossCumulative = false;        // per-lump MPV, not the cumulative column

  // Flat device output layout, one fixture per launch.
  template <int N>
  struct Out {
    static constexpr int kNodesMax = 2 * N + 1;  // the split layout; the fallback uses N+2 <= 2N+1 for N>=1
    static constexpr int kUsedSplit = 0;
    static constexpr int kNNodes = 1;
    static constexpr int kQCharge = 2;
    static constexpr int kInnerXX0 = 3;
    static constexpr int kInnerD1 = 4;
    static constexpr int kInnerW1 = 5;
    static constexpr int kSTransverse = 6;
    static constexpr int kSTotal = kSTransverse + N;
    static constexpr int kMatXX0 = kSTotal + N;
    static constexpr int kGapD1 = kMatXX0 + N;
    static constexpr int kGapW1 = kGapD1 + N;
    static constexpr int kCov = kGapW1 + N;
    static constexpr int kCorr = kCov + 25;
    static constexpr int kChi2 = kCorr + 5;
    static constexpr int kHelixPar = kChi2 + 1;
    static constexpr int kHelixCov = kHelixPar + 5;
    static constexpr int kFullDelta = kHelixCov + 25;
    static constexpr int kSize = kFullDelta + 2 * kNodesMax + 1;
  };

  // The device kernel: the whole production chain at fixed N, one lane.
  template <int N>
  struct OracleKernel {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  double const* hitsIn,
                                  float const* geIn,
                                  double const* ffIn,
                                  float const* rho,
                                  double bField,
                                  double msScale,
                                  gbld::GblNodeData* nodes,
                                  double* scratch,
                                  double* out) const {
      using O = Out<N>;
      for ([[maybe_unused]] auto lane : cms::alpakatools::uniform_elements(acc, 1)) {
        Eigen::Matrix<double, 3, N> hits;
        Eigen::Matrix<float, 6, N> hits_ge;
        for (int c = 0; c < N; ++c) {
          for (int r = 0; r < 3; ++r)
            hits(r, c) = hitsIn[3 * c + r];
          for (int r = 0; r < 6; ++r)
            hits_ge(r, c) = geIn[6 * c + r];
        }
        const Eigen::Vector4d ff(ffIn[0], ffIn[1], ffIn[2], ffIn[3]);

        // (i) prep: arc lengths, charge, the Geant4 material march and every gap's two-thin split.
        bld::PreparedGblData<N> data;
        double gapD1[N], gapW1[N];
        bld::prepareGblFitData(acc, hits, ff, bField, rho, data, /*matCached=*/nullptr, gapD1, gapW1);

        // the upstream (beamline -> hit0) segment's own split, as BrokenLineFitKernels.h:627 takes it.
        const double rHit0 = alpaka::math::sqrt(acc, hits(0, 0) * hits(0, 0) + hits(1, 0) * hits(1, 0));
        double innerD1 = 0., innerW1 = 0.;
        if (data.innerXX0 > 0.)
          bld::segmentXX0GapSplit(acc, rho, 0., 0., rHit0, hits(2, 0), innerD1, innerW1);

        // (ii) node builder: the 2N+1 split layout, with the N+2 arrival-node layout as the fallback.
        const bool usedSplit = gbld::prepareGblDataSplit<Acc1D, N>(acc,
                                                                   hits,
                                                                   hits_ge,
                                                                   ff,
                                                                   bField,
                                                                   data.qCharge,
                                                                   data.sTransverse,
                                                                   data.sTotal,
                                                                   data.matXX0,
                                                                   gapD1,
                                                                   gapW1,
                                                                   data.innerXX0,
                                                                   innerD1,
                                                                   innerW1,
                                                                   nodes,
                                                                   msScale,
                                                                   kELossPerX0,
                                                                   /*bMap=*/nullptr,
                                                                   kBFieldOrigin,
                                                                   kTrajectoryCorrections,
                                                                   kScatteringLogAtTotal,
                                                                   kElossCumulative);
        if (!usedSplit)
          gbld::prepareGblData<Acc1D, N>(acc,
                                         hits,
                                         hits_ge,
                                         ff,
                                         bField,
                                         data.qCharge,
                                         data.sTransverse,
                                         data.sTotal,
                                         data.matXX0,
                                         data.innerXX0,
                                         nodes,
                                         msScale,
                                         kELossPerX0,
                                         /*jacHit0ToPca=*/nullptr,
                                         innerD1,
                                         innerW1,
                                         /*usedInnerNode=*/nullptr,
                                         /*bMap=*/nullptr,
                                         kBFieldOrigin,
                                         kTrajectoryCorrections,
                                         kScatteringLogAtTotal,
                                         kElossCumulative);

        // (iii) solve at the PCA. gblFitPca<M> spans M+1 nodes, so M = nNodes - 1 in both layouts.
        gbld::Vector5d corr = gbld::Vector5d::Zero();
        gbld::Matrix5d cov = gbld::Matrix5d::Zero();
        double chi2 = 0.;
        int nNodes = 0;
        if (usedSplit) {
          nNodes = 2 * N + 1;
          cov = gbld::gblFitPca<Acc1D, 2 * N>(acc, nodes, scratch, &corr, out + O::kFullDelta, &chi2);
        } else {
          nNodes = N + 2;
          cov = gbld::gblFitPca<Acc1D, N + 1>(acc, nodes, scratch, &corr, out + O::kFullDelta, &chi2);
        }

        // (iv) extract the helix at the PCA.
        gbld::Vector5d helixPar = gbld::Vector5d::Zero();
        gbld::Matrix5d helixCov = gbld::Matrix5d::Zero();
        gbld::gblHelixAtPca(
            acc, ff, data.qCharge, bField, double(data.sTransverse(0)), hits(2, 0), corr, cov, helixPar, helixCov);

        out[O::kUsedSplit] = usedSplit ? 1. : 0.;
        out[O::kNNodes] = double(nNodes);
        out[O::kQCharge] = double(data.qCharge);
        out[O::kInnerXX0] = data.innerXX0;
        out[O::kInnerD1] = innerD1;
        out[O::kInnerW1] = innerW1;
        for (int i = 0; i < N; ++i) {
          out[O::kSTransverse + i] = data.sTransverse(i);
          out[O::kSTotal + i] = data.sTotal(i);
          out[O::kMatXX0 + i] = data.matXX0(i);
          out[O::kGapD1 + i] = gapD1[i];
          out[O::kGapW1 + i] = gapW1[i];
        }
        for (int a = 0; a < 5; ++a) {
          out[O::kCorr + a] = corr(a);
          out[O::kHelixPar + a] = helixPar(a);
          for (int b = 0; b < 5; ++b) {
            out[O::kCov + a * 5 + b] = cov(a, b);
            out[O::kHelixCov + a * 5 + b] = helixCov(a, b);
          }
        }
        out[O::kChi2] = chi2;
      }
    }
  };

  // Host helpers.

  //!< max relative difference, skipping entries whose reference is numerically zero.
  struct MaxRel {
    double v = 0.;
    bool any = false;  // positive control: did anything at all clear the zero guard?
    void add(double got, double ref) { addScaled(got, ref, std::abs(ref)); }
    //!< Relative comparison of one entry, gated on that entry being a real part of its object: an entry far
    //!< below the object's own scale carries an absolute error of order eps*scale like every other entry, so
    //!< dividing it by itself reports a huge relative error that measures only the entry's smallness. This
    //!< is the kCovGate the DESY comparison below applies, for the same reason.
    void addScaled(double got, double ref, double scale) {
      if (std::abs(ref) > 1e-30 && std::abs(ref) > kCovGate * scale) {
        any = true;
        v = std::max(v, std::abs(got - ref) / std::abs(ref));
      }
    }
  };

  //!< measurement-only Cramer-Rao floors: d0 (straight line), 1/R (3-param circle Fisher), z0.
  template <int N>
  void cramerRao(const Eigen::Matrix<double, 3, N>& hits,
                 const Eigen::Matrix<float, 6, N>& hits_ge,
                 double& crD0,
                 double& crKappa,
                 double& crZ0) {
    crD0 = crKappa = crZ0 = 0.;
    const double phi = std::atan2(hits(1, N - 1) - hits(1, 0), hits(0, N - 1) - hits(0, 0));
    const double cph = std::cos(phi), sph = std::sin(phi);
    double sW = 0., sWu = 0., sWuu = 0., sWz = 0., sWzs = 0., sWzss = 0.;
    Eigen::Matrix3d fisher = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; ++i) {
      const double xx = hits_ge(0, i), xy = hits_ge(1, i), yy = hits_ge(2, i);
      const double u = hits(0, i) * cph + hits(1, i) * sph;
      const double sv2 = sph * sph * xx - 2. * sph * cph * xy + cph * cph * yy;
      const double w = (sv2 > 0.) ? 1. / sv2 : 0.;
      sW += w;
      sWu += w * u;
      sWuu += w * u * u;
      const Eigen::Vector3d g(u, 1., 0.5 * u * u);
      fisher += w * g * g.transpose();
      const double zz = hits_ge(5, i);
      const double wz = (zz > 0.) ? 1. / zz : 0.;
      sWz += wz;
      sWzs += wz * u;
      sWzss += wz * u * u;
    }
    const double dd = sW * sWuu - sWu * sWu;
    if (dd > 0.)
      crD0 = std::sqrt(sWuu / dd);
    if (std::abs(fisher.determinant()) > 0.)
      crKappa = std::sqrt(fisher.inverse()(2, 2));
    const double dz = sWz * sWzss - sWzs * sWzs;
    if (dz > 0.)
      crZ0 = std::sqrt(sWzss / dz);
  }

  // One fixture on one device.
  template <int N>
  void runFixture(Queue& queue,
                  const char* devName,
                  const char* label,
                  const double D[N][9],
                  double msScale,
                  double tolTwin,
                  double tolDesy,
                  const float* rhoDev) {
    using O = Out<N>;
    constexpr double kB = gblTestFixtures::kB;

    // hits + the 3x3 global covariance. Where a fixture has no dumped longitudinal error (gzz == 0) a
    // nominal sigma_z = 1 cm is substituted so the longitudinal subsystem is non-singular; z0/cotTheta
    // are then meaningless for that fixture and only the transverse numbers carry physics.
    Eigen::Matrix<double, 3, N> hits;
    Eigen::Matrix<float, 6, N> hits_ge = Eigen::Matrix<float, 6, N>::Zero();
    for (int k = 0; k < N; ++k) {
      hits(0, k) = D[k][0];
      hits(1, k) = D[k][1];
      hits(2, k) = D[k][2];
      const double zz = (D[k][8] > 0.) ? D[k][8] : 1.0;
      hits_ge.col(k) << D[k][3], D[k][4], D[k][5], D[k][6], D[k][7], zz;
    }
    // The reference helix is computed ON THE HOST and uploaded, so every backend linearizes around bitwise
    // the same reference and the comparison below isolates the fit, not the seed.
    Eigen::Vector4d ff;
    blh::fastFit(hits, ff);

    auto hits_h = cms::alpakatools::make_host_buffer<double[], Platform>(3 * N);
    auto ge_h = cms::alpakatools::make_host_buffer<float[], Platform>(6 * N);
    auto ff_h = cms::alpakatools::make_host_buffer<double[], Platform>(4);
    for (int c = 0; c < N; ++c) {
      for (int r = 0; r < 3; ++r)
        hits_h[3 * c + r] = hits(r, c);
      for (int r = 0; r < 6; ++r)
        ge_h[6 * c + r] = hits_ge(r, c);
    }
    for (int j = 0; j < 4; ++j)
      ff_h[j] = ff(j);

    auto hits_d = cms::alpakatools::make_device_buffer<double[]>(queue, 3 * N);
    auto ge_d = cms::alpakatools::make_device_buffer<float[]>(queue, 6 * N);
    auto ff_d = cms::alpakatools::make_device_buffer<double[]>(queue, 4);
    auto nodes_d = cms::alpakatools::make_device_buffer<gbld::GblNodeData[]>(queue, O::kNodesMax);
    auto scratch_d = cms::alpakatools::make_device_buffer<double[]>(queue, gbld::kGblScratchDoubles<2 * N>);
    auto out_d = cms::alpakatools::make_device_buffer<double[]>(queue, O::kSize);
    auto nodes_h = cms::alpakatools::make_host_buffer<gbld::GblNodeData[], Platform>(O::kNodesMax);
    auto out_h = cms::alpakatools::make_host_buffer<double[], Platform>(O::kSize);

    alpaka::memcpy(queue, hits_d, hits_h);
    alpaka::memcpy(queue, ge_d, ge_h);
    alpaka::memcpy(queue, ff_d, ff_h);
    alpaka::memset(queue, out_d, 0);
    auto div = cms::alpakatools::make_workdiv<Acc1D>(1, 1);
    alpaka::exec<Acc1D>(queue,
                        div,
                        OracleKernel<N>{},
                        hits_d.data(),
                        ge_d.data(),
                        ff_d.data(),
                        rhoDev,
                        kB,
                        msScale,
                        nodes_d.data(),
                        scratch_d.data(),
                        out_d.data());
    alpaka::memcpy(queue, out_h, out_d);
    alpaka::memcpy(queue, nodes_h, nodes_d);
    alpaka::wait(queue);

    const int nNodes = int(out_h[O::kNNodes] + 0.5);
    const int qCharge = int(out_h[O::kQCharge]);
    const bool usedSplit = out_h[O::kUsedSplit] > 0.5;
    REQUIRE(nNodes >= 3);
    REQUIRE(nNodes <= O::kNodesMax);

    // Oracle 2: the host twin, on the device's own node chain.
    std::vector<gblh::GblNodeData> hostNodes(nNodes);
    std::vector<gblHostOracles::Node> desyNodes(nNodes);
    int nMeas = 0, nScat = 0;
    for (int k = 0; k < nNodes; ++k) {
      const gbld::GblNodeData& d = nodes_h[k];
      hostNodes[k].jacToPrev = d.jacToPrev;
      hostNodes[k].measPrec = d.measPrec;
      hostNodes[k].scatPrec = d.scatPrec;
      hostNodes[k].measResidual = d.measResidual;
      hostNodes[k].hasMeas = d.hasMeas;
      hostNodes[k].hasScat = d.hasScat;
      for (int a = 0; a < 5; ++a)
        for (int b = 0; b < 5; ++b)
          desyNodes[k].jacToPrev[a * 5 + b] = d.jacToPrev(a, b);
      for (int a = 0; a < 2; ++a) {
        desyNodes[k].measResidual[a] = d.measResidual(a);
        for (int b = 0; b < 2; ++b) {
          desyNodes[k].measPrec[a * 2 + b] = d.measPrec(a, b);
          desyNodes[k].scatPrec[a * 2 + b] = d.scatPrec(a, b);
        }
      }
      desyNodes[k].hasMeas = d.hasMeas;
      desyNodes[k].hasScat = d.hasScat;
      nMeas += d.hasMeas ? 1 : 0;
      nScat += d.hasScat ? 1 : 0;
    }
    // The chain must actually carry the track: N measurements and at least one scatterer. Without this the
    // comparisons below would be a fit of nothing to nothing agreeing perfectly.
    REQUIRE(nMeas == N);
    REQUIRE(nScat >= 1);

    gblh::Vector5d hostCorr;
    Eigen::VectorXd hostFd;
    double hostChi2 = 0.;
    const gblh::Matrix5d hostCov = gblh::gblFitPca(hostNodes, &hostCorr, &hostFd, &hostChi2);
    gblh::Vector5d hostHp;
    gblh::Matrix5d hostHc;
    gblh::gblHelixAtPca(ff, qCharge, kB, out_h[O::kSTransverse], hits(2, 0), hostCorr, hostCov, hostHp, hostHc);

    // The scales each entry is gated against: a vector against its own largest entry, a covariance entry
    // against sqrt(Caa*Cbb), which makes the gate a statement about the correlation, not about the units.
    double corrScaleT = 0., fdScaleT = 0., hpScaleT = 0.;
    for (int a = 0; a < 5; ++a) {
      corrScaleT = std::max(corrScaleT, std::abs(hostCorr(a)));
      hpScaleT = std::max(hpScaleT, std::abs(hostHp(a)));
    }
    for (int a = 0; a < 2 * nNodes + 1; ++a)
      fdScaleT = std::max(fdScaleT, std::abs(hostFd(a)));

    MaxRel twinCov, twinCorr, twinFd, twinHp, twinHc, twinChi2;
    for (int a = 0; a < 5; ++a) {
      twinCorr.addScaled(out_h[O::kCorr + a], hostCorr(a), corrScaleT);
      twinHp.addScaled(out_h[O::kHelixPar + a], hostHp(a), hpScaleT);
      for (int b = 0; b < 5; ++b) {
        twinCov.addScaled(
            out_h[O::kCov + a * 5 + b], hostCov(a, b), std::sqrt(std::abs(hostCov(a, a) * hostCov(b, b))));
        twinHc.addScaled(
            out_h[O::kHelixCov + a * 5 + b], hostHc(a, b), std::sqrt(std::abs(hostHc(a, a) * hostHc(b, b))));
      }
    }
    for (int a = 0; a < 2 * nNodes + 1; ++a)
      twinFd.addScaled(out_h[O::kFullDelta + a], hostFd(a), fdScaleT);
    twinChi2.add(out_h[O::kChi2], hostChi2);

    // Oracle 1: DESY, on the same chain.
    const gblHostOracles::DesyResult desy = gblHostOracles::desyFit(desyNodes);
    REQUIRE(desy.ok);

    MaxRel desyCorr, desyChi2;
    double desyCovMaxRel = 0., desyCovMaxAbs = 0.;
    bool desyCovAny = false;
    double corrScale = 0.;
    for (int a = 0; a < 5; ++a)
      corrScale = std::max(corrScale, std::abs(desy.corr[a]));
    for (int a = 0; a < 5; ++a) {
      if (std::abs(desy.corr[a]) > kCovGate * corrScale)
        desyCorr.add(out_h[O::kCorr + a], desy.corr[a]);
      for (int b = 0; b < 5; ++b) {
        const double ref = desy.cov[a * 5 + b];
        const double scale = std::sqrt(std::abs(desy.cov[a * 5 + a] * desy.cov[b * 5 + b]));
        const double diff = std::abs(out_h[O::kCov + a * 5 + b] - ref);
        desyCovMaxAbs = std::max(desyCovMaxAbs, diff);
        if (std::abs(ref) > kCovGate * scale) {
          desyCovAny = true;
          desyCovMaxRel = std::max(desyCovMaxRel, diff / std::abs(ref));
        }
      }
    }
    desyChi2.add(out_h[O::kChi2], desy.chi2);

    // Physics diagnostics, printed and never asserted.
    double crD0 = 0., crKappa = 0., crZ0 = 0.;
    cramerRao<N>(hits, hits_ge, crD0, crKappa, crZ0);
    const double slope = -double(qCharge) / ff(3);
    const double sec2 = 1. + slope * slope;
    const double gblD0 = std::sqrt(std::abs(out_h[O::kCov + 3 * 5 + 3]));
    const double gblZ0 = std::sqrt(std::abs(out_h[O::kCov + 4 * 5 + 4]));
    const double gblKappa = kB * std::sqrt(std::abs(out_h[O::kCov])) * std::sqrt(sec2);
    const double eta = std::asinh((hits(2, N - 1) - hits(2, 0)) /
                                  (std::hypot(hits(0, N - 1), hits(1, N - 1)) - std::hypot(hits(0, 0), hits(1, 0))));

    std::printf(
        "  %-28s N=%2d pt=%6.1f eta=%+5.2f q=%+d %s nodes=%2d | twin cov=%.1e corr=%.1e fd=%.1e hp=%.1e hc=%.1e "
        "chi2=%.1e | DESY cov=%.1e corr=%.1e chi2=%.1e ndf=%d | GBLd0/CR=%.2f GBLk/CR=%.2f z0/CR=%.2f\n",
        label,
        N,
        kB * ff(2),
        eta,
        qCharge,
        usedSplit ? "split" : "arriv",
        nNodes,
        twinCov.v,
        twinCorr.v,
        twinFd.v,
        twinHp.v,
        twinHc.v,
        twinChi2.v,
        desyCovMaxRel,
        desyCorr.v,
        desyChi2.v,
        desy.ndf,
        (crD0 > 0.) ? gblD0 / crD0 : 0.,
        (crKappa > 0.) ? gblKappa / crKappa : 0.,
        (crZ0 > 0.) ? gblZ0 / crZ0 : 0.);

    const gblHostOracles::PerigeeRatios pr = gblHostOracles::perigeeRatios(ff.data(),
                                                                           qCharge,
                                                                           kB,
                                                                           out_h[O::kSTransverse],
                                                                           hits(2, 0),
                                                                           &out_h[O::kCorr],
                                                                           &out_h[O::kCov],
                                                                           &out_h[O::kHelixPar],
                                                                           &out_h[O::kHelixCov]);
    if (pr.ok)
      std::printf("      [PERIGEE ours/CMSSW] d0=%.3f z0=%.3f phi=%.3f theta=%.3f pt/pt=%.3f   (%s)\n",
                  pr.d0,
                  pr.z0,
                  pr.phi,
                  pr.theta,
                  pr.ptRel,
                  devName);

    // Positive controls first: a comparison whose reference is identically zero passes vacuously.
    REQUIRE(twinCov.any);
    REQUIRE(twinCorr.any);
    REQUIRE(twinFd.any);
    REQUIRE(twinHp.any);
    REQUIRE(twinHc.any);
    REQUIRE(twinChi2.any);
    REQUIRE(desyCovAny);
    REQUIRE(desyCorr.any);
    REQUIRE(desyChi2.any);

    REQUIRE(twinCov.v < tolTwin);
    REQUIRE(twinCorr.v < tolTwin);
    REQUIRE(twinFd.v < tolTwin);
    REQUIRE(twinHp.v < tolTwin);
    REQUIRE(twinHc.v < tolTwin);
    REQUIRE(twinChi2.v < tolTwin);

    REQUIRE(desyCovMaxRel < tolDesy);
    REQUIRE(desyCorr.v < tolDesy);
    REQUIRE(desyChi2.v < kTolDesyChi2);

    // The material march must have produced something: an all-zero matXX0 would make every scatterer
    // vanish and the fit degenerate into a measurement-only one that still agrees with both oracles.
    double matSum = 0.;
    for (int i = 0; i < N; ++i)
      matSum += out_h[O::kMatXX0 + i];
    REQUIRE(matSum > 0.);
    REQUIRE(out_h[O::kInnerXX0] > 0.);
  }

}  // namespace

TEST_CASE("GBL device chain vs the DESY oracle for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend",
          "[" EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) "]") {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty())
    FAIL("No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, test skipped.");

  // The Geant4 material map: no EventSetup needed, the table is compiled into the
  // RecoTracker/PixelTrackFitting library and uploaded here exactly as the ES product would.
  auto rho_h = cms::alpakatools::make_host_buffer<float[], Platform>(blMaterialMap::kSize);
  std::copy_n(blMaterialMap::blMaterialMapData(), blMaterialMap::kSize, rho_h.data());

  for (auto const& device : devices) {
    auto queue = Queue(device);
    auto rho_d = cms::alpakatools::make_device_buffer<float[]>(queue, blMaterialMap::kSize);
    alpaka::memcpy(queue, rho_d, rho_h);
    alpaka::wait(queue);
    const std::string dn = alpaka::getName(device);
    std::printf("\n=== %s ===\n", dn.c_str());
    std::printf("  --- eta sweep (real Pt100 tracks) ---\n");

    using namespace gblTestFixtures;
    // The last two columns are the per-fixture tolerances.
    runFixture<4>(queue, dn.c_str(), "barrel eta~0", BARREL0, 1.0, kTolTwin, kTolDesy, rho_d.data());
    runFixture<4>(queue, dn.c_str(), "barrel eta~0.4", BARREL04, 1.0, kTolTwin, kTolDesy, rho_d.data());
    // IT+OT fixtures: a 30x outer lever arm makes the normal matrix ill-conditioned, so the
    // device/host FMA difference is amplified; DESY is a different solver on the same system.
    runFixture<10>(queue, dn.c_str(), "central eta~0.36 IT+OT", CENTRAL, 1.0, kTolTwin, kTolDesy, rho_d.data());
    runFixture<10>(queue, dn.c_str(), "RECO central w/ tilted OT", RECOTILT, 1.0, kTolTwin, kTolDesy, rho_d.data());
    runFixture<10>(
        queue, dn.c_str(), "forward eta~3.2 IT only", FWD, 1.0, kTolTwinIllCond, kTolDesyIllCond, rho_d.data());
    runFixture<10>(queue, dn.c_str(), "tilted eta~1.0 OT barrel", TILT, 1.0, kTolTwin, kTolDesy, rho_d.data());
    runFixture<9>(queue, dn.c_str(), "Pt1 barrel eta~0.28", PT1, 1.0, kTolTwin, kTolDesy, rho_d.data());
    runFixture<10>(queue, dn.c_str(), "REAL barrel eta~0.27", BARRELR027, 1.0, kTolTwin, kTolDesy, rho_d.data());
    runFixture<10>(
        queue, dn.c_str(), "REAL barrel eta~0.62", BARRELR062, 1.0, kTolTwinIllCond, kTolDesyIllCond, rho_d.data());

    std::printf("  --- multiple-scattering sweep (low-pt proxy on the central track) ---\n");
    runFixture<10>(queue, dn.c_str(), "central MSx10 (~Pt32)", CENTRAL, 10., kTolTwin, kTolDesy, rho_d.data());
    runFixture<10>(queue, dn.c_str(), "central MSx100 (~Pt10)", CENTRAL, 100., kTolTwin, kTolDesy, rho_d.data());
    runFixture<10>(queue, dn.c_str(), "central MSx10000 (~Pt1)", CENTRAL, 10000., kTolTwin, kTolDesy, rho_d.data());

    std::printf("  --- displaced low-pt OT-only track (the material study) ---\n");
    runFixture<4>(queue, dn.c_str(), "displaced OT m=1.0", DISPLACED1, 1.0, kTolTwin, kTolDesy, rho_d.data());
    runFixture<4>(queue, dn.c_str(), "displaced OT m=0.3", DISPLACED1, 0.3, kTolTwin, kTolDesy, rho_d.data());
    runFixture<4>(queue, dn.c_str(), "displaced OT m=0.1", DISPLACED1, 0.1, kTolTwin, kTolDesy, rho_d.data());
  }
}
