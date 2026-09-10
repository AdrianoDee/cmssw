// Device check of the GBL measurement model: the shipping construction -- project the global 3x3 hit
// covariance onto the curvilinear (U,V) plane, then invert -- is exactly the KF-equivalent module-frame
// measurement -- the local residual and diagonal local error through the obliquity Jacobian
// H = A - (1/n.T) c b^T -- and exactly its eigendecomposition-free form Ruv M^T ge3^+ M Ruv^T.
//
// With (lx, ly, n) the sensor frame, (U, V, T) the curvilinear frame, both orthonormal, and P the 3x3
// matrix of their overlaps written in blocks
//        P = [ A    c  ]   A = [[lx.U, lx.V],[ly.U, ly.V]],  c = (lx.T, ly.T)^T,
//            [ b^T  nT ]   b = (n.U, n.V)^T,                  nT = n.T,
// P is orthogonal, so block inversion together with P^-1 = P^T gives H := A - (1/nT) c b^T = A^-T for any
// incidence angle, hence
//        H^T W^-1 H = inv2(Ruv * ge3 * Ruv^T)                      [precision]
//        H^-1 r_local = Ruv * residual3                            [residual]
//        r_local^T W^-1 r_local = residual^T (measPrec) residual   [pre-fit chi2]
// and the three node chains fit to the same answer. The projected form needs no eigendecomposition and no
// per-module geometry payload, which is why the device header always projects.
//
// The per-hit comparisons are held at 1e-4 to 1e-3, not at the 1e-12 of the pure algebra: hits_ge is a
// float array, so ge3 is rank two only to ~1e-6 relative; ||T|| - 1 reaches 7e-7 because T is a unit
// vector only where the hit lies on the fast-fit circle; and the transverse residual component vanishes
// identically at the three fast-fit anchor hits, which is why the residual test is relative to the norm.
//
// Eigen::SelfAdjointEigenSolver, used to recover the sensor frame, is host-only test algebra: it is not
// part of the device-safe Eigen subset.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <alpaka/alpaka.hpp>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"
#include "RecoTracker/PixelTrackFitting/interface/BrokenLine.h"                // upstream host fastFit
#include "RecoTracker/PixelTrackFitting/interface/GeneralBrokenLine.h"         // host twin + the ge3 helpers
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BrokenLine.h"         // device prep under test
#include "RecoTracker/PixelTrackFitting/interface/alpaka/GeneralBrokenLine.h"  // device twin under test
#include "RecoTracker/PixelTrackFitting/test/gblTestFixtures.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
namespace gblh = ::generalBrokenLine;
namespace blh = ::brokenline;
namespace gbld = ALPAKA_ACCELERATOR_NAMESPACE::generalBrokenLine;
namespace bld = ALPAKA_ACCELERATOR_NAMESPACE::brokenline;

namespace {

  constexpr int kN = 10;
  constexpr int kNodes = 2 * kN + 1;
  constexpr int kM = 2 * kN;
  constexpr int kTreatments = 3;  // 0 = as built (CURRENT), 1 = MODFRAME (H), 2 = GTILDE

  //!< H * A^T == I with the reference direction normalized; pure double algebra.
  constexpr double kTolSchur = 1e-12;
  //!< device measPrec vs the module-frame reconstruction, per 2x2 entry; limited by the float ge3 (~1e-6).
  constexpr double kTolPrec = 1e-4;
  //!< device measResidual vs the module-frame reconstruction, relative to the residual's own norm: the
  //!< fast fit passes exactly through hits 0, mId and N-1, so a per-component test is meaningless there.
  constexpr double kTolResid = 1e-3;
  //!< per-hit pre-fit chi2. Looser than kTolPrec because chi2 is quadratic in a residual whose components
  //!< nearly cancel at the three fast-fit anchor hits; the total is bounded separately.
  constexpr double kTolChi2 = 1e-3;
  //!< total pre-fit chi2 over the track.
  constexpr double kTolChi2Total = 1e-6;
  //!< device vs host twin on the same chain.
  constexpr double kTolTwin = 1e-7;
  //!< relative gate: only compare entries above this fraction of the matrix's largest entry.
  constexpr double kEntryGate = 1e-8;

  constexpr double kELossPerX0 = 0.0;
  constexpr double kBFieldOrigin = 0.0;
  constexpr bool kTrajectoryCorrections = false;
  constexpr bool kScatteringLogAtTotal = false;
  constexpr bool kElossCumulative = false;

  enum BuildOut { kOutUsedSplit = 0, kOutQCharge, kOutSTransverse, kBuildOutSize = kOutSTransverse + kN };
  enum FitOut { kFitCov = 0, kFitCorr = 25, kFitChi2 = 30, kFitOutSize = 31 };

  struct BuildKernel {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  double const* hitsIn,
                                  float const* geIn,
                                  double const* ffIn,
                                  float const* rho,
                                  double bField,
                                  gbld::GblNodeData* nodes,
                                  double* out) const {
      for ([[maybe_unused]] auto lane : cms::alpakatools::uniform_elements(acc, 1)) {
        Eigen::Matrix<double, 3, kN> hits;
        Eigen::Matrix<float, 6, kN> hits_ge;
        for (int c = 0; c < kN; ++c) {
          for (int r = 0; r < 3; ++r)
            hits(r, c) = hitsIn[3 * c + r];
          for (int r = 0; r < 6; ++r)
            hits_ge(r, c) = geIn[6 * c + r];
        }
        const Eigen::Vector4d ff(ffIn[0], ffIn[1], ffIn[2], ffIn[3]);

        bld::PreparedGblData<kN> data;
        double gapD1[kN], gapW1[kN];
        bld::prepareGblFitData(acc, hits, ff, bField, rho, data, /*matCached=*/nullptr, gapD1, gapW1);
        const double rHit0 = alpaka::math::sqrt(acc, hits(0, 0) * hits(0, 0) + hits(1, 0) * hits(1, 0));
        double innerD1 = 0., innerW1 = 0.;
        if (data.innerXX0 > 0.)
          bld::segmentXX0GapSplit(acc, rho, 0., 0., rHit0, hits(2, 0), innerD1, innerW1);

        const bool usedSplit = gbld::prepareGblDataSplit<Acc1D, kN>(acc,
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
                                                                    /*msScale=*/1.0,
                                                                    kELossPerX0,
                                                                    /*bMap=*/nullptr,
                                                                    kBFieldOrigin,
                                                                    kTrajectoryCorrections,
                                                                    kScatteringLogAtTotal,
                                                                    kElossCumulative);
        out[kOutUsedSplit] = usedSplit ? 1. : 0.;
        out[kOutQCharge] = double(data.qCharge);
        for (int i = 0; i < kN; ++i)
          out[kOutSTransverse + i] = double(data.sTransverse(i));
      }
    }
  };

  struct SolveKernel {
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc, gbld::GblNodeData const* nodesAll, double* scratch, double* out, int nChains) const {
      for (auto v : cms::alpakatools::uniform_elements(acc, nChains)) {
        gbld::Vector5d corr = gbld::Vector5d::Zero();
        double chi2 = 0.;
        const auto cov = gbld::gblFitPca<Acc1D, kM>(acc,
                                                    nodesAll + int(v) * kNodes,
                                                    scratch + int(v) * gbld::kGblScratchDoubles<kM>,
                                                    &corr,
                                                    /*fullDelta=*/nullptr,
                                                    &chi2);
        double* o = out + int(v) * kFitOutSize;
        for (int a = 0; a < 5; ++a) {
          o[kFitCorr + a] = corr(a);
          for (int b = 0; b < 5; ++b)
            o[kFitCov + a * 5 + b] = cov(a, b);
        }
        o[kFitChi2] = chi2;
      }
    }
  };

  //!< the sensor frame recovered from the rank-2 global hit covariance ge3 = R^T diag(xel, 0, yel) R.
  struct SensorFrame {
    gblh::Vector3d lx, ly, n;
    double xel = 0., yel = 0.;
  };

  SensorFrame sensorFrameFromGe3(const gblh::Matrix3d& ge3) {
    SensorFrame f;
    f.n = gblh::planeNormalFromGe3(ge3);
    const gblh::Vector3d a = (std::abs(f.n.x()) < 0.9) ? gblh::Vector3d(1, 0, 0) : gblh::Vector3d(0, 1, 0);
    const gblh::Vector3d e1 = (a - a.dot(f.n) * f.n).normalized();
    const gblh::Vector3d e2 = f.n.cross(e1);
    Eigen::Matrix2d g;
    g(0, 0) = e1.dot(ge3 * e1);
    g(0, 1) = g(1, 0) = e1.dot(ge3 * e2);
    g(1, 1) = e2.dot(ge3 * e2);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(g);  // ascending eigenvalues
    f.xel = es.eigenvalues()(0);
    f.yel = es.eigenvalues()(1);
    const Eigen::Vector2d v0 = es.eigenvectors().col(0), v1 = es.eigenvectors().col(1);
    f.lx = (v0(0) * e1 + v0(1) * e2).normalized();
    f.ly = (v1(0) * e1 + v1(1) * e2).normalized();
    return f;
  }

  struct MaxRel {
    double v = 0.;
    bool any = false;
    void add(double got, double ref) {
      if (std::abs(ref) > 1e-30) {
        any = true;
        v = std::max(v, std::abs(got - ref) / std::abs(ref));
      }
    }
  };

  //!< max relative difference between two 2x2 blocks, gated on the reference's own largest entry.
  double maxRelBlock(const Eigen::Matrix2d& got, const Eigen::Matrix2d& ref) {
    const double scale = ref.cwiseAbs().maxCoeff();
    double m = 0.;
    for (int a = 0; a < 2; ++a)
      for (int b = 0; b < 2; ++b)
        if (std::abs(ref(a, b)) > kEntryGate * scale)
          m = std::max(m, std::abs(got(a, b) - ref(a, b)) / std::abs(ref(a, b)));
    return m;
  }

}  // namespace

namespace {

  void runFixture(Queue& queue, const char* devName, const char* label, const double D[kN][9], const float* rhoDev) {
    constexpr double kB = gblTestFixtures::kB;

    Eigen::Matrix<double, 3, kN> hits;
    Eigen::Matrix<float, 6, kN> hits_ge = Eigen::Matrix<float, 6, kN>::Zero();
    for (int k = 0; k < kN; ++k) {
      hits(0, k) = D[k][0];
      hits(1, k) = D[k][1];
      hits(2, k) = D[k][2];
      const double zz = (D[k][8] > 0.) ? D[k][8] : 1.0;
      hits_ge.col(k) << D[k][3], D[k][4], D[k][5], D[k][6], D[k][7], zz;
    }
    Eigen::Vector4d ff;
    blh::fastFit(hits, ff);

    auto hits_h = cms::alpakatools::make_host_buffer<double[], Platform>(3 * kN);
    auto ge_h = cms::alpakatools::make_host_buffer<float[], Platform>(6 * kN);
    auto ff_h = cms::alpakatools::make_host_buffer<double[], Platform>(4);
    for (int c = 0; c < kN; ++c) {
      for (int r = 0; r < 3; ++r)
        hits_h[3 * c + r] = hits(r, c);
      for (int r = 0; r < 6; ++r)
        ge_h[6 * c + r] = hits_ge(r, c);
    }
    for (int j = 0; j < 4; ++j)
      ff_h[j] = ff(j);

    auto hits_d = cms::alpakatools::make_device_buffer<double[]>(queue, 3 * kN);
    auto ge_d = cms::alpakatools::make_device_buffer<float[]>(queue, 6 * kN);
    auto ff_d = cms::alpakatools::make_device_buffer<double[]>(queue, 4);
    auto nodes_d = cms::alpakatools::make_device_buffer<gbld::GblNodeData[]>(queue, kNodes);
    auto bout_d = cms::alpakatools::make_device_buffer<double[]>(queue, kBuildOutSize);
    auto nodes_h = cms::alpakatools::make_host_buffer<gbld::GblNodeData[], Platform>(kNodes);
    auto bout_h = cms::alpakatools::make_host_buffer<double[], Platform>(kBuildOutSize);
    alpaka::memcpy(queue, hits_d, hits_h);
    alpaka::memcpy(queue, ge_d, ge_h);
    alpaka::memcpy(queue, ff_d, ff_h);
    alpaka::memset(queue, bout_d, 0);
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(1, 1),
                        BuildKernel{},
                        hits_d.data(),
                        ge_d.data(),
                        ff_d.data(),
                        rhoDev,
                        kB,
                        nodes_d.data(),
                        bout_d.data());
    alpaka::memcpy(queue, nodes_h, nodes_d);
    alpaka::memcpy(queue, bout_h, bout_d);
    alpaka::wait(queue);

    if (!(bout_h[kOutUsedSplit] > 0.5))
      FAIL("prepareGblDataSplit refused a prompt N=10 fixture: the 2N+1 layout is the production layout for it.");
    const int qCharge = int(bout_h[kOutQCharge]);

    std::vector<int> hitNode;
    hitNode.reserve(kN);
    for (int k = 0; k < kNodes; ++k)
      if (nodes_h[k].hasMeas)
        hitNode.push_back(k);
    REQUIRE(int(hitNode.size()) == kN);

    // the reference helix, exactly as the device builder formed it
    const double cx = ff(0), cy = ff(1), R = ff(2);
    const double slope = -double(qCharge) / ff(3);
    const double invn = 1. / std::sqrt(1. + slope * slope);
    const double z0 = hits(2, 0) - slope * bout_h[kOutSTransverse];

    std::vector<std::vector<gblh::GblNodeData>> hv(kTreatments, std::vector<gblh::GblNodeData>(kNodes));
    for (int k = 0; k < kNodes; ++k)
      for (int t = 0; t < kTreatments; ++t) {
        hv[t][k].jacToPrev = nodes_h[k].jacToPrev;
        hv[t][k].measPrec = nodes_h[k].measPrec;
        hv[t][k].scatPrec = nodes_h[k].scatPrec;
        hv[t][k].measResidual = nodes_h[k].measResidual;
        hv[t][k].hasMeas = nodes_h[k].hasMeas;
        hv[t][k].hasScat = nodes_h[k].hasScat;
      }

    double worstSchur = 0., worstPrecMf = 0., worstResMf = 0., worstPrecGt = 0., worstResGt = 0., worstChi2 = 0.;
    double worstTNorm = 0., totKfChi2 = 0., totNodeChi2 = 0.;
    std::printf("\n  ---- %s (%s) ----\n", label, devName);
    std::printf(
        "   i  type    incid[deg]  xel[um]   yel[um]  |detA|   ||T||-1 | H*A^T-I  prec(H)   resid(H)  prec(Gt)  "
        "resid(Gt) chi2\n");
    for (int i = 0; i < kN; ++i) {
      gblh::Matrix3d ge3;
      ge3 << hits_ge(0, i), hits_ge(1, i), hits_ge(3, i), hits_ge(1, i), hits_ge(2, i), hits_ge(4, i), hits_ge(3, i),
          hits_ge(4, i), hits_ge(5, i);

      // curvilinear frame at the hit, the same expressions as alpaka/GeneralBrokenLine.h emitHit
      const double rx = (hits(0, i) - cx) / R, ry = (hits(1, i) - cy) / R;
      const gblh::Vector3d T(double(qCharge) * ry * invn, -double(qCharge) * rx * invn, slope * invn);
      const double un = 1. / std::sqrt(T.x() * T.x() + T.y() * T.y());
      const gblh::Vector3d U(-T.y() * un, T.x() * un, 0.);
      const gblh::Vector3d V = T.cross(U);
      gblh::Matrix23d Ruv;
      Ruv.row(0) = U.transpose();
      Ruv.row(1) = V.transpose();

      // the reference-helix point and the raw 3-D residual, identical to the device's
      const double dvx = hits(0, i) - cx, dvy = hits(1, i) - cy, dmag = std::hypot(dvx, dvy);
      const gblh::Vector3d rref(cx + R * dvx / dmag, cy + R * dvy / dmag, z0 + slope * bout_h[kOutSTransverse + i]);
      const gblh::Vector3d residual3 = gblh::Vector3d(hits.col(i)) - rref;

      // MODFRAME: the sensor frame and the obliquity-correct Jacobian
      const SensorFrame f = sensorFrameFromGe3(ge3);
      const double nT = f.n.dot(T);
      Eigen::Matrix2d A;
      A << f.lx.dot(U), f.lx.dot(V), f.ly.dot(U), f.ly.dot(V);
      const Eigen::Vector2d cvec(f.lx.dot(T), f.ly.dot(T));
      const Eigen::Vector2d bvec(f.n.dot(U), f.n.dot(V));
      const Eigen::Matrix2d H = A - (1. / nT) * cvec * bvec.transpose();
      // The Schur identity assumes both frames orthonormal, but (U, V, T) as the production builder forms
      // it is orthogonal and not normalized: |T| = 1 only where the hit lies on the fast-fit circle, and
      // departs from it by the hit's radial residual over R. The identity is checked on a normalized copy.
      const gblh::Vector3d Tn = T.normalized();
      const double unn = 1. / std::sqrt(Tn.x() * Tn.x() + Tn.y() * Tn.y());
      const gblh::Vector3d Un(-Tn.y() * unn, Tn.x() * unn, 0.);
      const gblh::Vector3d Vn = Tn.cross(Un);
      Eigen::Matrix2d An;
      An << f.lx.dot(Un), f.lx.dot(Vn), f.ly.dot(Un), f.ly.dot(Vn);
      const Eigen::Vector2d cn(f.lx.dot(Tn), f.ly.dot(Tn));
      const Eigen::Vector2d bn(f.n.dot(Un), f.n.dot(Vn));
      const Eigen::Matrix2d Hn = An - (1. / f.n.dot(Tn)) * cn * bn.transpose();
      const Eigen::Matrix2d wInv = Eigen::Vector2d(1. / f.xel, 1. / f.yel).asDiagonal();
      // the module-plane crossing: move along T from the reference point until n.(x - hit) == 0.
      const double tpar = -f.n.dot(rref - gblh::Vector3d(hits.col(i))) / nT;
      const gblh::Vector3d dloc = gblh::Vector3d(hits.col(i)) - (rref + tpar * T);
      const Eigen::Vector2d rLocal(f.lx.dot(dloc), f.ly.dot(dloc));
      const Eigen::Matrix2d precMf = H.transpose() * wInv * H;
      const Eigen::Vector2d resMf = H.inverse() * rLocal;
      const double kfChi2 = rLocal.dot(wInv * rLocal);

      // the Schur identity, H == A^-T, in pure double on the normalized frame
      const double schur = (Hn * An.transpose() - Eigen::Matrix2d::Identity()).cwiseAbs().maxCoeff();
      worstSchur = std::max(worstSchur, schur);
      const double tNorm = std::abs(T.norm() - 1.);
      worstTNorm = std::max(worstTNorm, tNorm);

      // GTILDE: the same measurement with no eigendecomposition
      const gblh::Vector3d nrm = gblh::planeNormalFromGe3(ge3);
      const gblh::Matrix3d gp = gblh::pseudoInverseGe3(ge3);
      const gblh::Matrix3d M = gblh::Matrix3d::Identity() - (T * nrm.transpose()) * (1.0 / nrm.dot(T));
      const gblh::Matrix3d gTilde = M.transpose() * gp * M;
      const Eigen::Matrix2d precGt = Ruv * gTilde * Ruv.transpose();
      const Eigen::Vector2d resGt = precGt.inverse() * (Ruv * (gTilde * residual3));

      // what the device actually produced
      const Eigen::Matrix2d precDev = nodes_h[hitNode[i]].measPrec;
      const Eigen::Vector2d resDev = nodes_h[hitNode[i]].measResidual;
      const double nodeChi2 = resDev.dot(precDev * resDev);

      const double relPrecMf = maxRelBlock(precDev, precMf);
      const double relPrecGt = maxRelBlock(precDev, precGt);
      const double relResMf = (resDev - resMf).norm() / std::max(1e-300, resMf.norm());
      const double relResGt = (resDev - resGt).norm() / std::max(1e-300, resGt.norm());
      MaxRel rChi2;
      rChi2.add(nodeChi2, kfChi2);
      worstPrecMf = std::max(worstPrecMf, relPrecMf);
      worstPrecGt = std::max(worstPrecGt, relPrecGt);
      worstResMf = std::max(worstResMf, relResMf);
      worstResGt = std::max(worstResGt, relResGt);
      worstChi2 = std::max(worstChi2, rChi2.v);
      totKfChi2 += kfChi2;
      totNodeChi2 += nodeChi2;
      REQUIRE(resMf.norm() > 0.);
      REQUIRE(resGt.norm() > 0.);

      const char* tp = (f.yel > 1.0) ? "OT-2S " : (f.yel > 1e-4) ? "OT-PS " : "IT-pix";
      std::printf("   %d %s %8.1f %9.2f %9.2e %7.4f %8.1e | %.2e  %.2e  %.2e  %.2e  %.2e  %.2e\n",
                  i,
                  tp,
                  std::acos(std::min(1.0, std::abs(nT / T.norm()))) * 180. / M_PI,
                  1e4 * std::sqrt(std::abs(f.xel)),
                  1e4 * std::sqrt(std::abs(f.yel)),
                  std::abs(A.determinant()),
                  tNorm,
                  schur,
                  relPrecMf,
                  relResMf,
                  relPrecGt,
                  relResGt,
                  rChi2.v);

      REQUIRE(rChi2.any);

      hv[1][hitNode[i]].measPrec = precMf;
      hv[1][hitNode[i]].measResidual = resMf;
      hv[2][hitNode[i]].measPrec = precGt;
      hv[2][hitNode[i]].measResidual = resGt;
    }
    std::printf(
        "   pre-fit chi2 total: node=%.9g  KF-REF=%.9g (rel %.1e) | worst: ||T||-1=%.1e Schur=%.1e prec(H)=%.1e "
        "res(H)=%.1e prec(Gt)=%.1e res(Gt)=%.1e chi2=%.1e\n",
        totNodeChi2,
        totKfChi2,
        std::abs(totNodeChi2 - totKfChi2) / std::max(1e-300, std::abs(totKfChi2)),
        worstTNorm,
        worstSchur,
        worstPrecMf,
        worstResMf,
        worstPrecGt,
        worstResGt,
        worstChi2);

    REQUIRE(totKfChi2 > 0.);
    REQUIRE(worstSchur < kTolSchur);
    REQUIRE(worstPrecMf < kTolPrec);
    REQUIRE(worstResMf < kTolResid);
    REQUIRE(worstPrecGt < kTolPrec);
    REQUIRE(worstResGt < kTolResid);
    REQUIRE(worstChi2 < kTolChi2);
    REQUIRE(std::abs(totNodeChi2 - totKfChi2) / std::max(1e-300, std::abs(totKfChi2)) < kTolChi2Total);

    // and therefore the three chains fit to the same answer
    auto dv_h = cms::alpakatools::make_host_buffer<gbld::GblNodeData[], Platform>(kTreatments * kNodes);
    for (int t = 0; t < kTreatments; ++t)
      for (int k = 0; k < kNodes; ++k) {
        gbld::GblNodeData d;
        d.jacToPrev = hv[t][k].jacToPrev;
        d.measPrec = hv[t][k].measPrec;
        d.scatPrec = hv[t][k].scatPrec;
        d.measResidual = hv[t][k].measResidual;
        d.hasMeas = hv[t][k].hasMeas;
        d.hasScat = hv[t][k].hasScat;
        dv_h[t * kNodes + k] = d;
      }
    auto dv_d = cms::alpakatools::make_device_buffer<gbld::GblNodeData[]>(queue, kTreatments * kNodes);
    auto scr_d = cms::alpakatools::make_device_buffer<double[]>(queue, kTreatments * gbld::kGblScratchDoubles<kM>);
    auto fout_d = cms::alpakatools::make_device_buffer<double[]>(queue, kTreatments * kFitOutSize);
    auto fout_h = cms::alpakatools::make_host_buffer<double[], Platform>(kTreatments * kFitOutSize);
    alpaka::memcpy(queue, dv_d, dv_h);
    alpaka::memset(queue, fout_d, 0);
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(1, kTreatments),
                        SolveKernel{},
                        dv_d.data(),
                        scr_d.data(),
                        fout_d.data(),
                        kTreatments);
    alpaka::memcpy(queue, fout_h, fout_d);
    alpaka::wait(queue);

    const char* tname[kTreatments] = {"CURRENT (as built)", "MODFRAME (H)", "GTILDE (prod-portable)"};
    for (int t = 0; t < kTreatments; ++t) {
      gblh::Vector5d hostCorr;
      double hostChi2 = 0.;
      const gblh::Matrix5d hostCov = gblh::gblFitPca(hv[t], &hostCorr, nullptr, &hostChi2);
      MaxRel cov, corr, chi2;
      const double* o = &fout_h[t * kFitOutSize];
      for (int a = 0; a < 5; ++a) {
        corr.add(o[kFitCorr + a], hostCorr(a));
        for (int b = 0; b < 5; ++b)
          cov.add(o[kFitCov + a * 5 + b], hostCov(a, b));
      }
      chi2.add(o[kFitChi2], hostChi2);
      std::printf("   %-24s sigma(q/p)=%.6e chi2=%9.4f | twin cov=%.1e corr=%.1e chi2=%.1e\n",
                  tname[t],
                  std::sqrt(std::abs(o[kFitCov])),
                  o[kFitChi2],
                  cov.v,
                  corr.v,
                  chi2.v);
      REQUIRE(cov.any);
      REQUIRE(corr.any);
      REQUIRE(chi2.any);
      REQUIRE(cov.v < kTolTwin);
      REQUIRE(corr.v < kTolTwin);
      REQUIRE(chi2.v < kTolTwin);
    }
    // the three treatments are one measurement model: their fits must agree to the identity tolerance.
    for (int t = 1; t < kTreatments; ++t) {
      MaxRel cross;
      for (int a = 0; a < 25; ++a)
        cross.add(fout_h[t * kFitOutSize + kFitCov + a], fout_h[kFitCov + a]);
      cross.add(fout_h[t * kFitOutSize + kFitChi2], fout_h[kFitChi2]);
      REQUIRE(cross.any);
      REQUIRE(cross.v < kTolPrec);
    }
  }

}  // namespace

TEST_CASE("GBL measurement-frame identity for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend",
          "[" EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) "]") {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty())
    FAIL("No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, test skipped.");

  auto rho_h = cms::alpakatools::make_host_buffer<float[], Platform>(blMaterialMap::kSize);
  std::copy_n(blMaterialMap::blMaterialMapData(), blMaterialMap::kSize, rho_h.data());

  using namespace gblTestFixtures;
  for (auto const& device : devices) {
    auto queue = Queue(device);
    auto rho_d = cms::alpakatools::make_device_buffer<float[]>(queue, blMaterialMap::kSize);
    alpaka::memcpy(queue, rho_d, rho_h);
    alpaka::wait(queue);
    const std::string dn = alpaka::getName(device);
    runFixture(queue, dn.c_str(), "RECOTILT: near-normal tilted OT + 2S (reco track)", RECOTILT, rho_d.data());
    runFixture(queue, dn.c_str(), "TILT: OBLIQUE tilted OT, eta = 1.087", TILT, rho_d.data());
    runFixture(queue, dn.c_str(), "CENTRAL: flat OT control, eta = 0.36", CENTRAL, rho_d.data());
  }
}
