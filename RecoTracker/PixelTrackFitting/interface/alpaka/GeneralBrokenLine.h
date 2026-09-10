#ifndef RecoTracker_PixelTrackFitting_interface_alpaka_GeneralBrokenLine_h
#define RecoTracker_PixelTrackFitting_interface_alpaka_GeneralBrokenLine_h

// General Broken Lines device fit (alpaka), normative for the fit model. The host implementation in
// interface/GeneralBrokenLine.h is a readable reference for the same algebra, compiled only into the unit
// tests, and carries the long-form derivation and the references to DESY GblTrajectory.cpp and CMSSW
// AnalyticalCurvilinearJacobian.

#include <alpaka/alpaka.hpp>

#include <Eigen/Core>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/FitUtils.h"              // brings math::cholesky::invert
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BandedSolve.h"           // bandFactor / bandSolveInPlace
#include "RecoTracker/PixelTrackFitting/interface/alpaka/CurvilinearToPerigee.h"  // curvilinear -> perigee
#include "RecoTracker/PixelTrackFitting/interface/BLBFieldMap.h"                  // normalized (Bz,Br) r-z map

namespace ALPAKA_ACCELERATOR_NAMESPACE::generalBrokenLine {

  using namespace cms::alpakatools;

  using Matrix5d = Eigen::Matrix<double, 5, 5>;
  using Matrix3d = Eigen::Matrix<double, 3, 3>;
  using Matrix2d = Eigen::Matrix<double, 2, 2>;
  using Matrix23d = Eigen::Matrix<double, 2, 3>;
  using Vector5d = Eigen::Matrix<double, 5, 1>;
  using Vector3d = Eigen::Vector3d;
  using Vector2d = Eigen::Vector2d;

  // Explicit small inverses: the Jacobian blocks are general, not SPD.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE Matrix2d inv2(const Matrix2d& m) {
    const double idet = 1. / (m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0));
    Matrix2d r;
    r(0, 0) = m(1, 1) * idet;
    r(0, 1) = -m(0, 1) * idet;
    r(1, 0) = -m(1, 0) * idet;
    r(1, 1) = m(0, 0) * idet;
    return r;
  }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE Matrix3d inv3(const Matrix3d& m) {
    const double a = m(0, 0), b = m(0, 1), c = m(0, 2);
    const double d = m(1, 0), e = m(1, 1), f = m(1, 2);
    const double g = m(2, 0), h = m(2, 1), i = m(2, 2);
    const double A = e * i - f * h, B = f * g - d * i, C = d * h - e * g;
    const double idet = 1. / (a * A + b * B + c * C);
    Matrix3d r;
    r(0, 0) = A * idet;
    r(0, 1) = (c * h - b * i) * idet;
    r(0, 2) = (b * f - c * e) * idet;
    r(1, 0) = B * idet;
    r(1, 1) = (a * i - c * g) * idet;
    r(1, 2) = (c * d - a * f) * idet;
    r(2, 0) = C * idet;
    r(2, 1) = (b * g - a * h) * idet;
    r(2, 2) = (a * e - b * d) * idet;
    return r;
  }

  // Analytic curvilinear point-to-point Jacobian (uniform B).
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE Matrix5d curvilinearJacobian(const TAcc& acc,
                                                              const Vector3d& p1,
                                                              const Vector3d& p2,
                                                              const Vector3d& dx,
                                                              double qbp,
                                                              const Vector3d& hInvGeV,
                                                              double s) {
    Matrix5d J = Matrix5d::Identity();

    const double t11 = p1.x(), t12 = p1.y(), t13 = p1.z();
    const double t21 = p2.x(), t22 = p2.y(), t23 = p2.z();
    const double cosl0 = alpaka::math::sqrt(acc, t11 * t11 + t12 * t12);
    const double cosl1 = 1. / alpaka::math::sqrt(acc, t21 * t21 + t22 * t22);

    const double hmag = alpaka::math::sqrt(acc, hInvGeV.squaredNorm());
    const Vector3d hn = hInvGeV / hmag;
    const double qp = -hmag;
    const double q = qp * qbp;
    const double theta = q * s;
    const double sint = alpaka::math::sin(acc, theta), cost = alpaka::math::cos(acc, theta);
    const double hn1 = hn.x(), hn2 = hn.y(), hn3 = hn.z();
    const double dx1 = dx.x(), dx2 = dx.y(), dx3 = dx.z();
    const double gamma = hn1 * t21 + hn2 * t22 + hn3 * t23;
    const double an1 = hn2 * t23 - hn3 * t22;
    const double an2 = hn3 * t21 - hn1 * t23;
    const double an3 = hn1 * t22 - hn2 * t21;
    double au = 1. / alpaka::math::sqrt(acc, t11 * t11 + t12 * t12);
    const double u11 = -au * t12, u12 = au * t11;
    const double v11 = -t13 * u12, v12 = t13 * u11, v13 = t11 * u12 - t12 * u11;
    au = 1. / alpaka::math::sqrt(acc, t21 * t21 + t22 * t22);
    const double u21 = -au * t22, u22 = au * t21;
    const double v21 = -t23 * u22, v22 = t23 * u21, v23 = t21 * u22 - t22 * u21;
    const double anv = -(hn1 * u21 + hn2 * u22);
    const double anu = (hn1 * v21 + hn2 * v22 + hn3 * v23);
    const double omcost = 1. - cost, tmsint = theta - sint;

    const double hu1 = -hn3 * u12, hu2 = hn3 * u11, hu3 = hn1 * u12 - hn2 * u11;
    const double hv1 = hn2 * v13 - hn3 * v12, hv2 = hn3 * v11 - hn1 * v13, hv3 = hn1 * v12 - hn2 * v11;

    // row 1 (lambda)
    J(1, 0) = -qp * anv * (t21 * dx1 + t22 * dx2 + t23 * dx3);
    J(1, 1) = cost * (v11 * v21 + v12 * v22 + v13 * v23) + sint * (hv1 * v21 + hv2 * v22 + hv3 * v23) +
              omcost * (hn1 * v11 + hn2 * v12 + hn3 * v13) * (hn1 * v21 + hn2 * v22 + hn3 * v23) +
              anv * (-sint * (v11 * t21 + v12 * t22 + v13 * t23) + omcost * (v11 * an1 + v12 * an2 + v13 * an3) -
                     tmsint * gamma * (hn1 * v11 + hn2 * v12 + hn3 * v13));
    J(1, 2) = cost * (u11 * v21 + u12 * v22) + sint * (hu1 * v21 + hu2 * v22 + hu3 * v23) +
              omcost * (hn1 * u11 + hn2 * u12) * (hn1 * v21 + hn2 * v22 + hn3 * v23) +
              anv * (-sint * (u11 * t21 + u12 * t22) + omcost * (u11 * an1 + u12 * an2) -
                     tmsint * gamma * (hn1 * u11 + hn2 * u12));
    J(1, 2) *= cosl0;
    J(1, 3) = -q * anv * (u11 * t21 + u12 * t22);
    J(1, 4) = -q * anv * (v11 * t21 + v12 * t22 + v13 * t23);

    // row 2 (phi)
    J(2, 0) = -qp * anu * (t21 * dx1 + t22 * dx2 + t23 * dx3) * cosl1;
    J(2, 1) = cost * (v11 * u21 + v12 * u22) + sint * (hv1 * u21 + hv2 * u22) +
              omcost * (hn1 * v11 + hn2 * v12 + hn3 * v13) * (hn1 * u21 + hn2 * u22) +
              anu * (-sint * (v11 * t21 + v12 * t22 + v13 * t23) + omcost * (v11 * an1 + v12 * an2 + v13 * an3) -
                     tmsint * gamma * (hn1 * v11 + hn2 * v12 + hn3 * v13));
    J(2, 1) *= cosl1;
    J(2, 2) = cost * (u11 * u21 + u12 * u22) + sint * (hu1 * u21 + hu2 * u22) +
              omcost * (hn1 * u11 + hn2 * u12) * (hn1 * u21 + hn2 * u22) +
              anu * (-sint * (u11 * t21 + u12 * t22) + omcost * (u11 * an1 + u12 * an2) -
                     tmsint * gamma * (hn1 * u11 + hn2 * u12));
    J(2, 2) *= cosl1 * cosl0;
    J(2, 3) = -q * anu * (u11 * t21 + u12 * t22) * cosl1;
    J(2, 4) = -q * anu * (v11 * t21 + v12 * t22 + v13 * t23) * cosl1;

    // rows 3,4 col 0 (x_T, y_T vs 1/p)
    const double cutCriterion = alpaka::math::abs(acc, s * qbp);
    const double limit = 5.;
    if (cutCriterion > limit) {
      const double pp = 1. / qbp;
      J(3, 0) = pp * (u21 * dx1 + u22 * dx2);
      J(4, 0) = pp * (v21 * dx1 + v22 * dx2 + v23 * dx3);
    } else {
      const double hp11 = hn2 * t13 - hn3 * t12;
      const double hp12 = hn3 * t11 - hn1 * t13;
      const double hp13 = hn1 * t12 - hn2 * t11;
      const double temp1 = hp11 * u21 + hp12 * u22;
      const double s2 = s * s;
      const double secondOrder41 = 0.5 * qp * temp1 * s2;
      const double ghnmp1 = gamma * hn1 - t11;
      const double ghnmp2 = gamma * hn2 - t12;
      const double ghnmp3 = gamma * hn3 - t13;
      const double temp2 = ghnmp1 * u21 + ghnmp2 * u22;
      const double s3 = s2 * s, s4 = s3 * s;
      const double h1 = hmag, h2 = h1 * h1, h3 = h2 * h1;
      const double qbp2 = qbp * qbp;
      const double thirdOrder41 = (1. / 3) * h2 * s3 * qbp * temp2;
      const double fourthOrder41 = (1. / 8) * h3 * s4 * qbp2 * temp1;
      J(3, 0) = secondOrder41 + (thirdOrder41 + fourthOrder41);

      const double temp3 = hp11 * v21 + hp12 * v22 + hp13 * v23;
      const double secondOrder51 = 0.5 * qp * temp3 * s2;
      const double temp4 = ghnmp1 * v21 + ghnmp2 * v22 + ghnmp3 * v23;
      const double thirdOrder51 = (1. / 3) * h2 * s3 * qbp * temp4;
      const double fourthOrder51 = (1. / 8) * h3 * s4 * qbp2 * temp3;
      J(4, 0) = secondOrder51 + (thirdOrder51 + fourthOrder51);
    }

    J(3, 1) = (sint * (v11 * u21 + v12 * u22) + omcost * (hv1 * u21 + hv2 * u22) +
               tmsint * (hn1 * u21 + hn2 * u22) * (hn1 * v11 + hn2 * v12 + hn3 * v13)) /
              q;
    J(3, 2) = (sint * (u11 * u21 + u12 * u22) + omcost * (hu1 * u21 + hu2 * u22) +
               tmsint * (hn1 * u21 + hn2 * u22) * (hn1 * u11 + hn2 * u12)) *
              cosl0 / q;
    J(3, 3) = (u11 * u21 + u12 * u22);
    J(3, 4) = (v11 * u21 + v12 * u22);

    J(4, 1) = (sint * (v11 * v21 + v12 * v22 + v13 * v23) + omcost * (hv1 * v21 + hv2 * v22 + hv3 * v23) +
               tmsint * (hn1 * v21 + hn2 * v22 + hn3 * v23) * (hn1 * v11 + hn2 * v12 + hn3 * v13)) /
              q;
    J(4, 2) = (sint * (u11 * v21 + u12 * v22) + omcost * (hu1 * v21 + hu2 * v22 + hu3 * v23) +
               tmsint * (hn1 * v21 + hn2 * v22 + hn3 * v23) * (hn1 * u11 + hn2 * u12)) *
              cosl0 / q;
    J(4, 3) = (u11 * v21 + u12 * v22);
    J(4, 4) = (v11 * v21 + v12 * v22 + v13 * v23);

    return J;
  }

  // offset-frame slope-response derivatives W, W*J, W*d (DESY GblPoint::getDerivatives).
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void nextDerivatives(const Matrix5d& jac, Matrix2d& W, Matrix2d& WJ, Vector2d& Wd) {
    const Matrix2d matJ = jac.block<2, 2>(3, 3);
    const Matrix2d matW = jac.block<2, 2>(3, 1);
    const Vector2d vecd = jac.block<2, 1>(3, 0);
    W = inv2(matW);
    WJ = W * matJ;
    Wd = W * vecd;
  }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void prevDerivatives(const Matrix5d& jac, Matrix2d& W, Matrix2d& WJ, Vector2d& Wd) {
    const Matrix3d A33 = jac.block<3, 3>(0, 0);
    const Eigen::Matrix<double, 3, 2> B32 = jac.block<3, 2>(0, 3);
    const Matrix23d C23 = jac.block<2, 3>(3, 0);
    const Matrix2d D22 = jac.block<2, 2>(3, 3);
    const Matrix23d CA = C23 * inv3(A33);
    const Matrix2d DCABinv = inv2(D22 - CA * B32);
    const Matrix23d prevD012 = -DCABinv * CA;
    const Matrix2d& matJ = DCABinv;
    const Matrix2d matW = -prevD012.block<2, 2>(0, 1);
    const Vector2d vecd = prevD012.block<2, 1>(0, 0);
    W = inv2(matW);
    WJ = W * matJ;
    Wd = W * vecd;
  }

  //!< per-node GBL inputs (every node carries a 2-D offset; PCA node 0 has no measurement/scatterer).
  struct GblNodeData {
    Matrix5d jacToPrev = Matrix5d::Identity();  //!< curvilinear p2p Jacobian from the previous node (unused for node 0)
    Matrix2d measPrec = Matrix2d::Zero();       //!< 2x2 measurement precision in the (U,V) offset frame
    Matrix2d scatPrec = Matrix2d::Zero();       //!< 2x2 scatterer (kink) precision, diagonal in (lambda,phi)
    Vector2d measResidual = Vector2d::Zero();   //!< (U,V) residual: measured hit minus the reference-helix point
    bool hasMeas = false;
    bool hasScat = false;
  };

  // Constants of the two ionization energy-loss laws below (elossMostProbable, elossTypicalColumn): both
  // evaluate the same Landau xi in the same medium and differ only in which statistic of the loss distribution
  // they return. The medium is the composite effective medium of the material the fit charges, not silicon
  // (Bragg additivity rule, PDG RPP 34.2.5); only the product (Z/A)*X0g enters xi.
  namespace elossMedium {
    constexpr double kPionMass = 0.13957;           // pion mass [GeV]
    constexpr double kElectronMass = 0.5109989e-3;  // electron mass [GeV]
    constexpr double kK = 0.307075e-3;              // 0.307 MeV cm^2/mol -> GeV
    constexpr double kZ_A = 0.500;                  // composite <Z/A> (A = 2Z closure; +-4 % band)
    constexpr double kI = 122e-9;                   // composite mean excitation energy [GeV]
    constexpr double kX0g = 28.8;                   // composite X0 [g/cm^2]; (Z/A)*X0g = 14.4
    constexpr double kHwp = 36.16e-9;               // composite plasma energy [GeV] (Sternheimer)
    // standard Landau: lambda_median - lambda_mode = 1.35578 - (-0.22278)
    constexpr double kLandauMedianMinusMode = 1.35578 + 0.22278;
  }  // namespace elossMedium

  // Most-probable (Landau) ionization energy loss [GeV] for a segment of thickness xx0 radiation lengths
  // ALONG THE TRACK, at total momentum p [GeV] (charged-pion mass), with the Sternheimer density correction:
  //   Delta_mp = xi * [ ln(2 me beta^2 gamma^2 / I) + ln(xi/I) + 0.2 - beta^2 - delta(betagamma) ]
  // with xi = (K/2)(Z/A) x / beta^2, x = xx0 * X0g [g/cm^2]. The correction has to remove the loss of the
  // typical track, not the unrestricted Bethe-Bloch mean, which includes the Landau delta-ray tail. MP is not
  // additive across sub-segments (ln xi term).
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE double elossMostProbable(const TAcc& acc, double p, double xx0) {
    constexpr double m = elossMedium::kPionMass;
    constexpr double me = elossMedium::kElectronMass;
    constexpr double K = elossMedium::kK;
    constexpr double Z_A = elossMedium::kZ_A;
    constexpr double I = elossMedium::kI;
    constexpr double X0g = elossMedium::kX0g;
    constexpr double hwp = elossMedium::kHwp;
    if (xx0 <= 0.)
      return 0.;
    const double E = alpaka::math::sqrt(acc, p * p + m * m);
    const double beta2 = p * p / (E * E);
    const double g = E / m;
    const double bg2 = beta2 * g * g;
    const double xi = 0.5 * K * Z_A * (xx0 * X0g) / beta2;
    // density effect, high-betagamma Sternheimer limit: delta -> 2 ln(hwp/I * betagamma) - 1 (>=0)
    const double dhalf = alpaka::math::log(acc, (hwp / I) * alpaka::math::sqrt(acc, bg2)) - 0.5;
    const double delta = dhalf > 0. ? 2. * dhalf : 0.;
    const double bracket =
        alpaka::math::log(acc, 2. * me * bg2 / I) + alpaka::math::log(acc, xi / I) + 0.2 - beta2 - delta;
    const double dmp = xi * bracket;
    return dmp > 0. ? dmp : 0.;
  }

  // Typical (median) ionization loss [GeV] of the CUMULATIVE charged column of thickness xx0 [X/X0], same
  // medium as elossMostProbable, selected over the per-lump most-probable law by the runtime flag
  // elossCumulative. The Landau family is stable under convolution, so the typical loss of a multi-slab column
  // is the single-column law at the SUMMED thickness, not the sum of per-slab MPVs, and median - mode =
  // (1.35578 + 0.22278) xi for a Landau (PDG RPP 34.2.9). Callers charge per-node increments
  // T(X_cum + x_lump) - T(X_cum).
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE double elossTypicalColumn(const TAcc& acc, double p, double xx0) {
    constexpr double m = elossMedium::kPionMass;
    constexpr double me = elossMedium::kElectronMass;
    constexpr double K = elossMedium::kK;
    constexpr double Z_A = elossMedium::kZ_A;
    constexpr double I = elossMedium::kI;
    constexpr double X0g = elossMedium::kX0g;
    constexpr double hwp = elossMedium::kHwp;
    constexpr double kLandauMedianMinusMode = elossMedium::kLandauMedianMinusMode;
    if (xx0 <= 0.)
      return 0.;
    const double E = alpaka::math::sqrt(acc, p * p + m * m);
    const double beta2 = p * p / (E * E);
    const double g = E / m;
    const double bg2 = beta2 * g * g;
    const double xi = 0.5 * K * Z_A * (xx0 * X0g) / beta2;
    const double dhalf = alpaka::math::log(acc, (hwp / I) * alpaka::math::sqrt(acc, bg2)) - 0.5;
    const double delta = dhalf > 0. ? 2. * dhalf : 0.;
    const double bracket =
        alpaka::math::log(acc, 2. * me * bg2 / I) + alpaka::math::log(acc, xi / I) + 0.2 - beta2 - delta;
    const double dmp = xi * bracket;
    return (dmp > 0. ? dmp : 0.) + kLandauMedianMinusMode * xi;
  }

  // Fractional stand-off of a measurement-less node from each of the two points bounding its segment
  // (PCA/hit0 for the upstream node, the two hits for a gap node): a node arbitrarily close to a bounding
  // point gives nextDerivatives a near-singular offset-to-angle block, its inverse scaling as 1/ds.
  inline constexpr double kSplitStandOff = 0.02;
  inline constexpr double kSplitStandOffHi = 0.98;

  // Assemble the per-node GBL inputs. Upstream-material model: the two-equivalent-thin-scatterer split with a
  // dedicated measurement-less node whenever segmentXX0Moments produced a valid split (`useInner` below),
  // otherwise a single thin scatterer at hit0 carrying the whole upstream lump. nodes must have N+2 slots.
  // Single-scatterer layout: node 0 = PCA, 1..N = hits (N+1 used). Inner-node layout: node 0 = PCA, node 1 =
  // measurement-less upstream scatterer (fraction innerW1 of the scattering variance at path distance innerD1
  // upstream of hit0, the 1-innerW1 remainder at hit0), nodes 2..N+1 = hits, so extraction at node 0 IS the
  // PCA. *usedInnerNode reports the layout built.
  template <typename TAcc, int N, typename M3xN, typename M6xN, typename V4, typename VN>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void prepareGblData(
      const TAcc& acc,
      const M3xN& hits,
      const M6xN& hits_ge,
      const V4& fast_fit,
      double bField,
      int qCharge,
      const VN& sTransverse,
      const VN& sTotal,
      const VN& matXX0,
      double innerXX0,
      GblNodeData* nodes,
      double msScale = 1.0,  // multiple-scattering scale (1.0 in production)
      // Enables the ionization-loss correction when > 0; only that test is read, the loss charged at a node
      // coming from elossMostProbable / elossTypicalColumn at that node's thickness.
      double eLossPerX0 = 0.0,
      Matrix5d* jacHit0ToPca = nullptr,  // optional out: hit0->PCA backward jacobian (single-scatterer layout)
      double innerD1 = 0.,               // upstream equivalent-scatterer path distance from hit0 [cm]
      double innerW1 = 0.,               // fraction of the upstream scattering variance at that node
      bool* usedInnerNode = nullptr,
      // Normalized (Bz,Br) r-z field map + the origin field it normalizes to. With a map, every segment is
      // charged the deterministic azimuth (and, under trajectoryCorrections, lambda) excess of the real field
      // profile over the single constant the fit models; a null map accumulates nothing.
      const float* bMap = nullptr,
      double bFieldOrigin = 0.,
      // Reference-trajectory corrections: the node-0 path length seeded with sin(lambda)*z0 rather than 0, the
      // arc->azimuth rotation sign that places a measurement-less node, and the B_r lambda row of the
      // field-profile offset.
      bool trajectoryCorrections = false,
      // Evaluates Highland's logarithm at the track's TOTAL declared material rather than at each gap's own
      // thickness (producer parameter useScatteringLogAtTotal).
      bool scatteringLogAtTotal = false,
      // Charges each gap the cumulative-column typical loss from the vertex to that node (elossTypicalColumn)
      // rather than the most-probable loss of its own lump (producer parameter useCumulativeEloss).
      bool elossCumulative = false) {
    constexpr int nNodes = N + 2;
    const double cx = fast_fit(0), cy = fast_fit(1), R = fast_fit(2);
    const double slope = -double(qCharge) / fast_fit(3);
    const double pt = bField * R;
    const double sec2 = 1. + slope * slope;
    const double pTot = pt * alpaka::math::sqrt(acc, sec2);
    const double cos2lam = 1. / sec2;
    const double qbp = double(qCharge) / pTot;
    const Vector3d hInvGeV(0., 0., bField);
    const double invn = 1. / alpaka::math::sqrt(acc, sec2);
    // signed charge that turns a transverse arc into a rotation of the radius vector about the fitted centre
    const double qArc = trajectoryCorrections ? -double(qCharge) : double(qCharge);

    // pos/dir rolling window: the per-node loop below only reads node k-1 and node k, so carrying just the
    // previous node keeps them in registers and off the kernel stack frame.
    const double cmag = alpaka::math::sqrt(acc, cx * cx + cy * cy);
    auto computeNode = [&](double x, double y, double z, Vector3d& p, Vector3d& d) {
      p = Vector3d(x, y, z);
      const double rx = (x - cx) / R, ry = (y - cy) / R;
      // physical FORWARD momentum tangent: the opposite sign leaves the (direction-invariant) covariance
      // alone but flips phi and the d0 sign in the perigee VALUE.
      const double tx = double(qCharge) * ry, ty = -double(qCharge) * rx;
      d = Vector3d(tx * invn, ty * invn, slope * invn);
    };
    const double sT0 = double(sTransverse(0));
    const double z0 = hits(2, 0) - slope * sT0;  // line: z = z0 + slope * sTransverse
    // inner-node layout only when the equivalent scatterer lands strictly between the PCA and hit0
    const bool useInner = innerXX0 > 0. && innerW1 > 0. && innerW1 < 1. && sT0 > 0.;
    const int hOff = useInner ? 2 : 1;  // node index of hit0
    if (usedInnerNode)
      *usedInnerNode = useInner;

    // Transverse arc of the equivalent upstream scatterer from the PCA, clamped strictly between the PCA and
    // hit0. The node's POSITION and the PATH LENGTH of the PCA -> node step must both follow the clamped
    // value, or curvilinearJacobian is handed a displacement and an arc that are not the same step: that path
    // length is innerD1 while the clamp is inactive and (sT0 - sTA)/invn once it fires.
    double sTAInner = sT0 - innerD1 * invn;
    double innerPath = innerD1;
    if (sTAInner < kSplitStandOff * sT0) {
      sTAInner = kSplitStandOff * sT0;
      innerPath = (sT0 - sTAInner) / invn;
    } else if (sTAInner > kSplitStandOffHi * sT0) {
      sTAInner = kSplitStandOffHi * sT0;
      innerPath = (sT0 - sTAInner) / invn;
    }

    Vector3d posPrev, dirPrev;  // node k-1 of the rolling window
    computeNode(cx * (1. - R / cmag), cy * (1. - R / cmag), hits(2, 0) - slope * sTransverse(0), posPrev, dirPrev);
    // node 0 (PCA) and node hOff (hit0) are read once more by the backward jacHit0ToPca at the end, so keep
    // copies outside the rolling window.
    const Vector3d pos0 = posPrev, dir0 = dirPrev;
    Vector3d posH0 = Vector3d::Zero(), dirH0 = Vector3d::Zero();

    nodes[0] = GblNodeData{};
    double sTotN[nNodes];
    // node 0 is the PCA of the reference helix, and sTotal is the path length in the (s,z) frame rotated by
    // lambda: sTotal(i) = cos(lambda)*sTransverse(i) + sin(lambda)*z(i). At sTransverse = 0 that is
    // sin(lambda)*z0, not zero, and only the PCA -> node-1 step sees the difference (the constant cancels in
    // every later difference).
    sTotN[0] = trajectoryCorrections ? slope * invn * z0 : 0.;
    if (useInner)
      sTotN[1] = double(sTotal(0)) - innerPath;  // path length along the track, to the (possibly clamped) node
    for (int i = 0; i < N; ++i)
      sTotN[i + hOff] = double(sTotal(i));
    // Total declared material of THIS layout's gap set: the upstream lump plus every inter-hit gap that gets a
    // kink (the outermost hit carries none). Only read under scatteringLogAtTotal (see th2Of).
    double xx0TotDecl = 0.;
    if (scatteringLogAtTotal) {
      if (innerXX0 > 0.)
        xx0TotDecl += innerXX0;
      for (int i = 1; i <= N - 2; ++i)
        if (double(matXX0(i - 1)) > 0.)
          xx0TotDecl += double(matXX0(i - 1));
    }
    // Highland scattering variance (pion 1/beta factor, matches CMSSW MultipleScatteringUpdator). theta0^2 is
    // not additive over a chain, so under scatteringLogAtTotal only the ARGUMENT OF THE LOG becomes the track's
    // total declared thickness, the leading xx0 keeping each gap's share.
    auto th2Of = [&](double xx0) {
      if (xx0 <= 0.)
        return 0.;
      constexpr double mPi = 0.13957;  // pion mass [GeV]
      const double betaP = pTot * pTot / alpaka::math::sqrt(acc, pTot * pTot + mPi * mPi);
      const double tt = 0.0136 / betaP;
      const double xLog = (scatteringLogAtTotal && xx0TotDecl > 0.) ? xx0TotDecl : xx0;
      const double f = 1. + 0.038 * alpaka::math::log(acc, xLog);
      return tt * tt * xx0 * f * f * msScale;
    };
    // total upstream scattering variance from the FULL innerXX0, split linearly between the equivalent
    // scatterer node (innerW1) and hit0 (1-innerW1) in the inner-node layout.
    const double th2InnerTot = th2Of(innerXX0);
    // running deterministic curvilinear state shift, accumulated PCA -> node k. The dE/dx q/p increment
    // (slot 0) and the field-profile azimuth increment (slot 2) share one accumulator: both are deterministic
    // offsets of the same reference state, propagated by the same Jacobians and removed at the same place.
    const bool useField = (bMap != nullptr);
    const bool detOffset = (eLossPerX0 > 0.) || useField;
    // The lambda row of the field-profile offset (see the increment below): its only ingredient beyond the
    // azimuth row is B_r, which the bending law already reads at the same lattice cell.
    const bool useLambdaRow = useField && trajectoryCorrections;
    // B_bend/Bz(0,0) at a node, the same local law the effective field averages: the reference circle's
    // tanLambda*cos(alpha) carries q^2, so this is charge-free.
    auto bBendOf = [&](const Vector3d& p) {
      const double r = alpaka::math::sqrt(acc, p.x() * p.x() + p.y() * p.y());
      const double den = fast_fit(3) * alpaka::math::abs(acc, R) * r;
      const double tlca = (den != 0.) ? -(cx * p.y() - cy * p.x()) / den : 0.;
      return blBFieldMap::bBendAt(bMap, r, p.z(), tlca);
    };
    // The same law, plus B_r/Bz(0,0) times sin(alpha) (the lambda row's shape, charge-ODD unlike the bending
    // term) from the SAME lattice cell. alpha is the angle from the position azimuth to the momentum azimuth;
    // its sine is the partner of the cosine folded into tlca (cos^2 + sin^2 = 1 on the reference circle).
    auto bBendBrOf = [&](const Vector3d& p, double& brSinAlpha) {
      const double r = alpaka::math::sqrt(acc, p.x() * p.x() + p.y() * p.y());
      const double den = fast_fit(3) * alpaka::math::abs(acc, R) * r;
      const double tlca = (den != 0.) ? -(cx * p.y() - cy * p.x()) / den : 0.;
      double brNorm = 0.;
      const double bb = blBFieldMap::bBendAndBrAt(bMap, r, p.z(), tlca, brNorm);
      const double sinA =
          (r > 0. && R != 0.) ? -double(qCharge) * (p.x() * (p.x() - cx) + p.y() * (p.y() - cy)) / (R * r) : 0.;
      brSinAlpha = brNorm * sinA;
      return bb;
    };
    double bbPrev = 0.;
    double bsPrev = 0.;
    if (useField)
      bbPrev = useLambdaRow ? bBendBrOf(posPrev, bsPrev) : bBendOf(posPrev);
    Vector5d Deloss = Vector5d::Zero();
    // running charged column [X/X0] for the cumulative typical-loss law (walk order = path order)
    double xx0ElossCum = 0.;
    for (int k = 1; k < N + hOff; ++k) {
      const int i = k - hOff;                    // hit index (negative for the scatterer-only node)
      const bool scatOnly = useInner && k == 1;  // upstream-material node: no measurement
      Vector3d posCur, dirCur;
      if (scatOnly) {
        // equivalent upstream scatterer on the reference helix, at the clamped transverse arc sTAInner from
        // the PCA (the same value sTotN[1] above was built from).
        const double sTA = sTAInner;
        // rotation of the PCA radius vector over the transverse arc sTA. The reference circle advances its
        // azimuth at dpsi/dsTransverse = -q/R, the convention sTransverse itself carries and the one the
        // transport Jacobians are built in, so the rotation is -q*sTA/R.
        const double phiA = qArc * sTA / R;
        const double ex = cx * (1. - R / cmag) - cx, ey = cy * (1. - R / cmag) - cy;
        const double ca = alpaka::math::cos(acc, phiA), sa = alpaka::math::sin(acc, phiA);
        computeNode(cx + ex * ca - ey * sa, cy + ex * sa + ey * ca, z0 + slope * sTA, posCur, dirCur);
      } else {
        computeNode(hits(0, i), hits(1, i), hits(2, i), posCur, dirCur);
      }
      GblNodeData nd;
      nd.jacToPrev = curvilinearJacobian(acc, dirPrev, dirCur, posPrev - posCur, qbp, hInvGeV, sTotN[k] - sTotN[k - 1]);
      if (detOffset) {
        // Field-profile increment of THIS segment. The reference trajectory turns at dphi/ds = -qbp*bField
        // with bField the single constant the fit models, the real one at -qbp*B_bend(s), so the excess azimuth
        // over the segment is -qbp*(B_bend_seg - bField)*ds with ds the 3D path length the Jacobian above
        // transports over. Half is charged at each end, which carries the offset built up inside the segment.
        double dPhiHalf = 0.;
        double dLamHalf = 0.;
        if (useField) {
          double bsCur = 0.;
          const double bbCur = useLambdaRow ? bBendBrOf(posCur, bsCur) : bBendOf(posCur);
          dPhiHalf = -0.5 * qbp * (bFieldOrigin * 0.5 * (bbPrev + bbCur) - bField) * (sTotN[k] - sTotN[k - 1]);
          // Same equation of motion, the OTHER curvilinear slope row: dlambda/ds = -q/|p| * B_r * sin(alpha).
          // A B_z-only field model makes it identically zero, so there is no reference value to subtract and
          // the whole term is the offset. B_r(r,-z) = -B_r(r,z) makes it odd in z.
          if (useLambdaRow)
            dLamHalf = -0.5 * qbp * (bFieldOrigin * 0.5 * (bsPrev + bsCur)) * (sTotN[k] - sTotN[k - 1]);
          bbPrev = bbCur;
          bsPrev = bsCur;
          Deloss(2) += dPhiHalf;
          if (useLambdaRow)
            Deloss(1) += dLamHalf;
        }
        Deloss = nd.jacToPrev * Deloss;  // propagate the accumulated state shift through EVERY node
        if (useField) {
          Deloss(2) += dPhiHalf;
          if (useLambdaRow)
            Deloss(1) += dLamHalf;
        }
      }
      if (scatOnly) {
        nd.scatPrec << 1. / (innerW1 * th2InnerTot), 0., 0., cos2lam / (innerW1 * th2InnerTot);
        nd.hasScat = true;
        if (eLossPerX0 > 0. && elossCumulative) {
          const double tPrev = elossTypicalColumn(acc, pTot, xx0ElossCum);
          xx0ElossCum += innerXX0 * innerW1;
          Deloss(0) += qbp * (elossTypicalColumn(acc, pTot, xx0ElossCum) - tPrev) / pTot;
        } else if (eLossPerX0 > 0.)
          Deloss(0) += qbp * elossMostProbable(acc, pTot, innerXX0 * innerW1) / pTot;
        nodes[k] = nd;
        posPrev = posCur;  // roll the window before the measurement-less early-out
        dirPrev = dirCur;
        continue;
      }

      // measurement: curvilinear (U = Z x T, V = T x U) frame from the unit momentum.
      const Vector3d T = dirCur;
      const double un = 1. / alpaka::math::sqrt(acc, T.x() * T.x() + T.y() * T.y());
      const Vector3d U(-T.y() * un, T.x() * un, 0.);
      const Vector3d V(-T.z() * U.y(), T.z() * U.x(), T.x() * U.y() - T.y() * U.x());  // T x U
      Matrix23d Ruv;
      Ruv.row(0) = U.transpose();
      Ruv.row(1) = V.transpose();
      Matrix3d ge3;
      ge3 << hits_ge(0, i), hits_ge(1, i), hits_ge(3, i), hits_ge(1, i), hits_ge(2, i), hits_ge(4, i), hits_ge(3, i),
          hits_ge(4, i), hits_ge(5, i);
      // residual: measured hit minus the fast-fit reference helix (closest circle point in the bending plane,
      // z = z0 + slope*sTransverse longitudinally).
      const double dvx = hits(0, i) - cx, dvy = hits(1, i) - cy;
      const double dmag = alpaka::math::sqrt(acc, dvx * dvx + dvy * dvy);
      const Vector3d residual3(hits(0, i) - (cx + R * dvx / dmag),
                               hits(1, i) - (cy + R * dvy / dmag),
                               hits(2, i) - (z0 + slope * double(sTransverse(i))));
      // measurement precision and residual in the curvilinear (U, V) offset frame
      nd.measPrec = inv2(Ruv * ge3 * Ruv.transpose());
      nd.measResidual = Ruv * residual3;
      nd.hasMeas = true;

      // deterministic offsets (value-only, cov unchanged): remove the accumulated UPSTREAM state shift, the
      // dE/dx spiral and the field-profile bend excess, from the residual so that neither biases the fit.
      if (detOffset)
        nd.measResidual -= Deloss.template segment<2>(3);

      // one thin scatterer per segment (the upstream remainder -> hit 0, matXX0(i-1) -> inner hits): one kink
      // per segment, no both-adjacent double count.
      double th2 = 0.;
      double xx0Eloss = 0.;
      if (i == 0) {
        th2 = (useInner ? (1. - innerW1) : 1.) * th2InnerTot;  // linear split of the TOTAL Highland variance
        xx0Eloss = innerXX0 * (useInner ? (1. - innerW1) : 1.);
      } else if (i <= N - 2) {
        xx0Eloss = double(matXX0(i - 1));
        th2 = th2Of(xx0Eloss);
      }
      if (th2 > 0.) {
        nd.scatPrec << 1. / th2, 0., 0., cos2lam / th2;
        nd.hasScat = true;
      }
      // this node's q/p increment for the downstream offsets: its material's ionization loss dE enters as
      // d(q/p) = (q/p)*dE/p, so |q/p| grows as the track loses momentum outward.
      if (eLossPerX0 > 0. && xx0Eloss > 0. && elossCumulative) {
        const double tPrev = elossTypicalColumn(acc, pTot, xx0ElossCum);
        xx0ElossCum += xx0Eloss;
        Deloss(0) += qbp * (elossTypicalColumn(acc, pTot, xx0ElossCum) - tPrev) / pTot;
      } else if (eLossPerX0 > 0. && xx0Eloss > 0.)
        Deloss(0) += qbp * elossMostProbable(acc, pTot, xx0Eloss) / pTot;
      nodes[k] = nd;
      if (k == hOff) {  // hit0: retained for the backward jacHit0ToPca below
        posH0 = posCur;
        dirH0 = dirCur;
      }
      posPrev = posCur;  // roll the window: node k becomes node k-1 for the next iteration
      dirPrev = dirCur;
    }
    // hit0 -> PCA backward jacobian (exact inverse of the forward PCA->hit0 transport, so no device 5x5 LU).
    // Used only by the single-scatterer layout, which references the corrections at hit0; the inner-node
    // layout extracts at node 0 = PCA directly. The arc of this backward step is the DIFFERENCE of the two
    // nodes' path lengths, which -sTotN[hOff] is only while node 0's path length is seeded with 0.
    if (jacHit0ToPca && N >= 1)
      *jacHit0ToPca = curvilinearJacobian(
          acc, dirH0, dir0, posH0 - pos0, qbp, hInvGeV, trajectoryCorrections ? (sTotN[0] - sTotN[hOff]) : -sTotN[hOff]);
  }

  //!< nodes of the two-thin-scatterer split layout at hit multiplicity N.
  template <int N>
  inline constexpr int kGblSplitNodes = 2 * N + 1;

  /*!
    \brief Node builder for the two-equivalent-thin-scatterer partition of EVERY gap.

    A single thin scatterer at one end of a gap reproduces the gap's total scattering variance but not its
    lever arm (far-end offset variance and angle-offset covariance come out as zero). Two equivalent thin
    scatterers reproduce all three moments: one at path distance d1 = S2/S1 upstream of the ARRIVAL end with
    the fraction w1 = S1^2/(S2 W) of the variance, the remainder at the arrival end (segmentXX0GapSplit). The
    interior scatterer is not a hit node, so the layout has 2N+1 nodes:

      node 0        PCA (reference; no measurement, no scatterer)
      node 1        upstream equivalent scatterer   (innerW1 of the beamline->hit0 variance)
      node 2        hit 0                           (the 1-innerW1 remainder)
      node 3+2g     gap g's interior equivalent scatterer  (w1 of gap g's variance)
      node 4+2g     hit g+1                                (the 1-w1 remainder of gap g)

    The last gap's arrival share is dropped (the last hit has no kink degree of freedom: gblFitPca forms kinks
    only at nodes 1..nNodes-2); its interior scatterer is kept.
    Anchoring: hit nodes sit at their measured positions, so an added node is placed at the reference-helix
    point plus the arc-length interpolation of its two bounding hits' offsets from the helix; anchored to the
    bare helix, the steps into and out of it would carry the bounding hits' full residual and corrupt the
    declared covariance at low pT.
    Well-posedness: every measurement-less node must carry a kink with positive variance. W -> 0 or a
    non-increasing arc pair refuses the layout and the caller falls back to the arrival-node layout.

    \param matXX0 slot g = the TOTAL X/X0 of gap g (slot N-1 unused), \param gapD1 / \param gapW1 its
           two-thin split. \param nodes must have 2N+1 slots. Returns false if the layout is refused.
  */
  template <typename TAcc, int N, typename M3xN, typename M6xN, typename V4, typename VN>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool prepareGblDataSplit(const TAcc& acc,
                                                          const M3xN& hits,
                                                          const M6xN& hits_ge,
                                                          const V4& fast_fit,
                                                          double bField,
                                                          int qCharge,
                                                          const VN& sTransverse,
                                                          const VN& sTotal,
                                                          const VN& matXX0,
                                                          const double* gapD1,
                                                          const double* gapW1,
                                                          double innerXX0,
                                                          double innerD1,
                                                          double innerW1,
                                                          GblNodeData* nodes,
                                                          double msScale = 1.0,
                                                          double eLossPerX0 = 0.0,
                                                          // (Bz,Br) map + the origin field it normalizes to; a
                                                          // null map accumulates no field-profile offset.
                                                          const float* bMap = nullptr,
                                                          double bFieldOrigin = 0.,
                                                          // Reference-trajectory corrections; see
                                                          // prepareGblData.
                                                          bool trajectoryCorrections = false,
                                                          // Highland's log at the track TOTAL rather than per
                                                          // gap (useScatteringLogAtTotal); see prepareGblData.
                                                          bool scatteringLogAtTotal = false,
                                                          // Cumulative-column typical-loss law (producer
                                                          // parameter useCumulativeEloss; elossTypicalColumn).
                                                          bool elossCumulative = false) {
    const double cx = fast_fit(0), cy = fast_fit(1), R = fast_fit(2);
    const double slope = -double(qCharge) / fast_fit(3);
    const double pt = bField * R;
    const double sec2 = 1. + slope * slope;
    const double pTot = pt * alpaka::math::sqrt(acc, sec2);
    const double cos2lam = 1. / sec2;
    const double qbp = double(qCharge) / pTot;
    const Vector3d hInvGeV(0., 0., bField);
    const double invn = 1. / alpaka::math::sqrt(acc, sec2);
    const double qArc = trajectoryCorrections ? -double(qCharge) : double(qCharge);
    const double cmag = alpaka::math::sqrt(acc, cx * cx + cy * cy);
    const double sT0 = double(sTransverse(0));
    const double z0 = hits(2, 0) - slope * sT0;
    if (!(innerXX0 > 0.) || !(innerW1 > 0.) || !(innerW1 < 1.) || !(sT0 > 0.))
      return false;

    auto computeNode = [&](double x, double y, double z, Vector3d& p, Vector3d& d) {
      p = Vector3d(x, y, z);
      const double rx = (x - cx) / R, ry = (y - cy) / R;
      const double tx = double(qCharge) * ry, ty = -double(qCharge) * rx;
      d = Vector3d(tx * invn, ty * invn, slope * invn);
    };
    // reference-helix point at transverse arc sT: the PCA radius vector rotated by the signed angle sT/R
    // about the pre-fitted centre, with the sz-plane line for z. The circle advances its azimuth at
    // dpsi/dsTransverse = -q/R, the convention sTransverse itself carries, so the rotation is -q*sT/R.
    auto helixAt = [&](double sT, double& hx, double& hy, double& hz) {
      const double phiA = qArc * sT / R;
      const double ex = cx * (1. - R / cmag) - cx, ey = cy * (1. - R / cmag) - cy;
      const double ca = alpaka::math::cos(acc, phiA), sa = alpaka::math::sin(acc, phiA);
      hx = cx + ex * ca - ey * sa;
      hy = cy + ex * sa + ey * ca;
      hz = z0 + slope * sT;
    };
    // Total declared material of the 2N+1 layout's gap set: the upstream lump plus EVERY inter-hit gap (this
    // layout places an interior scatterer in each). Only read under scatteringLogAtTotal (see th2Of).
    double xx0TotDecl = 0.;
    if (scatteringLogAtTotal) {
      if (innerXX0 > 0.)
        xx0TotDecl += innerXX0;
      for (int g = 0; g < N - 1; ++g)
        if (double(matXX0(g)) > 0.)
          xx0TotDecl += double(matXX0(g));
    }
    // scatteringLogAtTotal moves ONLY the argument of the Highland log to the track's total declared
    // thickness, so each gap keeps its share and the w1 / 1-w1 endpoint partition is untouched.
    auto th2Of = [&](double xx0) {
      if (xx0 <= 0.)
        return 0.;
      constexpr double mPi = 0.13957;
      const double betaP = pTot * pTot / alpaka::math::sqrt(acc, pTot * pTot + mPi * mPi);
      const double tt = 0.0136 / betaP;
      const double xLog = (scatteringLogAtTotal && xx0TotDecl > 0.) ? xx0TotDecl : xx0;
      const double f = 1. + 0.038 * alpaka::math::log(acc, xLog);
      return tt * tt * xx0 * f * f * msScale;
    };

    Vector3d posPrev, dirPrev;  // node k-1 of the rolling window
    computeNode(cx * (1. - R / cmag), cy * (1. - R / cmag), hits(2, 0) - slope * sT0, posPrev, dirPrev);
    nodes[0] = GblNodeData{};
    // one deterministic curvilinear offset accumulator for both terms; see the arrival-node builder.
    const bool useField = (bMap != nullptr);
    const bool detOffset = (eLossPerX0 > 0.) || useField;
    const bool useLambdaRow = useField && trajectoryCorrections;
    // B_bend/Bz(0,0) at a node; see the arrival-node builder.
    auto bBendOf = [&](const Vector3d& p) {
      const double r = alpaka::math::sqrt(acc, p.x() * p.x() + p.y() * p.y());
      const double den = fast_fit(3) * alpaka::math::abs(acc, R) * r;
      const double tlca = (den != 0.) ? -(cx * p.y() - cy * p.x()) / den : 0.;
      return blBFieldMap::bBendAt(bMap, r, p.z(), tlca);
    };
    // B_bend plus B_r/Bz(0,0) times sin(alpha); see the arrival-node builder.
    auto bBendBrOf = [&](const Vector3d& p, double& brSinAlpha) {
      const double r = alpaka::math::sqrt(acc, p.x() * p.x() + p.y() * p.y());
      const double den = fast_fit(3) * alpaka::math::abs(acc, R) * r;
      const double tlca = (den != 0.) ? -(cx * p.y() - cy * p.x()) / den : 0.;
      double brNorm = 0.;
      const double bb = blBFieldMap::bBendAndBrAt(bMap, r, p.z(), tlca, brNorm);
      const double sinA =
          (r > 0. && R != 0.) ? -double(qCharge) * (p.x() * (p.x() - cx) + p.y() * (p.y() - cy)) / (R * r) : 0.;
      brSinAlpha = brNorm * sinA;
      return bb;
    };
    double bbPrev = 0.;
    double bsPrev = 0.;
    if (useField)
      bbPrev = useLambdaRow ? bBendBrOf(posPrev, bsPrev) : bBendOf(posPrev);
    Vector5d Deloss = Vector5d::Zero();
    // running charged column [X/X0] for the cumulative typical-loss law (walk order = path order:
    // upstream w1 lump, hit0's 1-w1 remainder, then per gap the interior w1 lump + arrival 1-w1)
    double xx0ElossCum = 0.;
    // path length of node k-1. Node 0 is the PCA, whose own sTotal is sin(lambda)*z0 in the frame sTotal is
    // measured in, not zero.
    double sTotPrev = trajectoryCorrections ? slope * invn * z0 : 0.;
    // propagate the accumulated offset across one segment, charging the segment's field-profile azimuth
    // excess half at each end.
    auto carryOffset = [&](const Matrix5d& jac, const Vector3d& posCur, double dsSeg) {
      double dPhiHalf = 0.;
      double dLamHalf = 0.;
      if (useField) {
        double bsCur = 0.;
        const double bbCur = useLambdaRow ? bBendBrOf(posCur, bsCur) : bBendOf(posCur);
        dPhiHalf = -0.5 * qbp * (bFieldOrigin * 0.5 * (bbPrev + bbCur) - bField) * dsSeg;
        if (useLambdaRow)
          dLamHalf = -0.5 * qbp * (bFieldOrigin * 0.5 * (bsPrev + bsCur)) * dsSeg;
        bbPrev = bbCur;
        bsPrev = bsCur;
        Deloss(2) += dPhiHalf;
        if (useLambdaRow)
          Deloss(1) += dLamHalf;
      }
      Deloss = jac * Deloss;
      if (useField) {
        Deloss(2) += dPhiHalf;
        if (useLambdaRow)
          Deloss(1) += dLamHalf;
      }
    };

    // one measurement node, emitted for hit i at slot k.
    auto emitHit = [&](int k, int i, double th2, double xx0Eloss) {
      Vector3d posCur, dirCur;
      computeNode(hits(0, i), hits(1, i), hits(2, i), posCur, dirCur);
      GblNodeData nd;
      const double sTotCur = double(sTotal(i));
      nd.jacToPrev = curvilinearJacobian(acc, dirPrev, dirCur, posPrev - posCur, qbp, hInvGeV, sTotCur - sTotPrev);
      if (detOffset)
        carryOffset(nd.jacToPrev, posCur, sTotCur - sTotPrev);
      const Vector3d T = dirCur;
      const double un = 1. / alpaka::math::sqrt(acc, T.x() * T.x() + T.y() * T.y());
      const Vector3d U(-T.y() * un, T.x() * un, 0.);
      const Vector3d V(-T.z() * U.y(), T.z() * U.x(), T.x() * U.y() - T.y() * U.x());
      Matrix23d Ruv;
      Ruv.row(0) = U.transpose();
      Ruv.row(1) = V.transpose();
      Matrix3d ge3;
      ge3 << hits_ge(0, i), hits_ge(1, i), hits_ge(3, i), hits_ge(1, i), hits_ge(2, i), hits_ge(4, i), hits_ge(3, i),
          hits_ge(4, i), hits_ge(5, i);
      // residual: measured hit minus the reference helix (closest circle point in the bending plane,
      // z = z0 + slope*sTransverse longitudinally), the same measurement model both layouts share.
      const double dvx = hits(0, i) - cx, dvy = hits(1, i) - cy;
      const double dmag = alpaka::math::sqrt(acc, dvx * dvx + dvy * dvy);
      const Vector3d residual3(hits(0, i) - (cx + R * dvx / dmag),
                               hits(1, i) - (cy + R * dvy / dmag),
                               hits(2, i) - (z0 + slope * double(sTransverse(i))));
      nd.measPrec = inv2(Ruv * ge3 * Ruv.transpose());
      nd.measResidual = Ruv * residual3;
      nd.hasMeas = true;
      if (detOffset)
        nd.measResidual -= Deloss.template segment<2>(3);
      if (th2 > 0.) {
        nd.scatPrec << 1. / th2, 0., 0., cos2lam / th2;
        nd.hasScat = true;
      }
      if (eLossPerX0 > 0. && xx0Eloss > 0. && elossCumulative) {
        const double tPrev = elossTypicalColumn(acc, pTot, xx0ElossCum);
        xx0ElossCum += xx0Eloss;
        Deloss(0) += qbp * (elossTypicalColumn(acc, pTot, xx0ElossCum) - tPrev) / pTot;
      } else if (eLossPerX0 > 0. && xx0Eloss > 0.)
        Deloss(0) += qbp * elossMostProbable(acc, pTot, xx0Eloss) / pTot;
      nodes[k] = nd;
      posPrev = posCur;
      dirPrev = dirCur;
      sTotPrev = sTotCur;
    };
    // one measurement-less equivalent scatterer at slot k, at transverse arc sTA and path sTotCur.
    auto emitScat = [&](int k, double px, double py, double pz, double sTotCur, double th2, double xx0Eloss) {
      Vector3d posCur, dirCur;
      computeNode(px, py, pz, posCur, dirCur);
      GblNodeData nd;
      nd.jacToPrev = curvilinearJacobian(acc, dirPrev, dirCur, posPrev - posCur, qbp, hInvGeV, sTotCur - sTotPrev);
      if (detOffset)
        carryOffset(nd.jacToPrev, posCur, sTotCur - sTotPrev);
      nd.scatPrec << 1. / th2, 0., 0., cos2lam / th2;
      nd.hasScat = true;
      if (eLossPerX0 > 0. && xx0Eloss > 0. && elossCumulative) {
        const double tPrev = elossTypicalColumn(acc, pTot, xx0ElossCum);
        xx0ElossCum += xx0Eloss;
        Deloss(0) += qbp * (elossTypicalColumn(acc, pTot, xx0ElossCum) - tPrev) / pTot;
      } else if (eLossPerX0 > 0. && xx0Eloss > 0.)
        Deloss(0) += qbp * elossMostProbable(acc, pTot, xx0Eloss) / pTot;
      nodes[k] = nd;
      posPrev = posCur;
      dirPrev = dirCur;
      sTotPrev = sTotCur;
    };

    // node 1: the upstream equivalent scatterer, on the reference helix (no hit upstream of it to anchor
    // against, and the PCA is the reference itself).
    {
      double sTA = sT0 - innerD1 * invn;
      double innerPath = innerD1;
      if (sTA < kSplitStandOff * sT0) {
        sTA = kSplitStandOff * sT0;
        innerPath = (sT0 - sTA) / invn;
      } else if (sTA > kSplitStandOffHi * sT0) {
        sTA = kSplitStandOffHi * sT0;
        innerPath = (sT0 - sTA) / invn;
      }
      double px, py, pz;
      helixAt(sTA, px, py, pz);
      const double th2Inner = th2Of(innerXX0);
      if (!(innerW1 * th2Inner > 0.))
        return false;  // the upstream node would have no kink: not a well-posed measurement-less node
      // the path length must follow the position the node ends up at: innerD1 while the clamp is inactive,
      // (sT0 - sTA)/invn once it fires.
      emitScat(1, px, py, pz, double(sTotal(0)) - innerPath, innerW1 * th2Inner, innerXX0 * innerW1);
    }
    // hit 0's offset from the reference helix: the arrival node's residual, and the departure anchor of the
    // first gap.
    double offPrevX, offPrevY, offPrevZ;
    {
      double hx0, hy0, hz0;
      helixAt(sT0, hx0, hy0, hz0);
      offPrevX = hits(0, 0) - hx0;
      offPrevY = hits(1, 0) - hy0;
      offPrevZ = hits(2, 0) - hz0;
      emitHit(2, 0, (1. - innerW1) * th2Of(innerXX0), innerXX0 * (1. - innerW1));
    }
    for (int g = 0; g < N - 1; ++g) {
      const bool last = (g == N - 2);
      const double W = double(matXX0(g));
      double w1 = gapW1[g];
      double d1 = gapD1[g];
      if (!(W > 0.))
        return false;  // no material: nothing to place, and a kink-free measurement-less node is not well posed
      const double lo = double(sTransverse(g)), hi = double(sTransverse(g + 1));
      if (!(hi > lo))
        return false;  // the two hits are not ordered in the fit's own arc length: no interior node fits
      if (!(w1 > 0.) || !(d1 > 0.) || !(w1 < 1.)) {
        w1 = 1.;  // single-atom measure (or all of it at the arrival end): the interior node takes the gap
        d1 = (d1 > 0.) ? d1 : 0.;
      }
      double sa = hi - d1 * invn;
      const double loC = lo + kSplitStandOff * (hi - lo), hiC = hi - kSplitStandOff * (hi - lo);
      sa = sa < loC ? loC : (sa > hiC ? hiC : sa);
      // the arrival hit's offset from the reference helix, needed both to anchor the interior node and as the
      // arrival node's own residual.
      double ax, ay, az;
      helixAt(hi, ax, ay, az);
      const double offArrX = hits(0, g + 1) - ax, offArrY = hits(1, g + 1) - ay, offArrZ = hits(2, g + 1) - az;
      double px, py, pz;
      helixAt(sa, px, py, pz);
      const double span = hi - lo;
      const double tt = (span > 0.) ? (sa - lo) / span : 0.5;
      px += (1. - tt) * offPrevX + tt * offArrX;
      py += (1. - tt) * offPrevY + tt * offArrY;
      pz += (1. - tt) * offPrevZ + tt * offArrZ;
      const double th2Gap = th2Of(W);
      if (!(th2Gap > 0.))
        return false;  // material present but no scattering variance to place (the Highland factor vanishes)
      // the path length must follow the position the node ends up at: curvilinearJacobian is handed both the
      // geometric displacement and the arc, and the two have to be the same step.
      emitScat(3 + 2 * g, px, py, pz, double(sTotal(g + 1)) - (hi - sa) / invn, w1 * th2Gap, w1 * W);
      const double wArr = last ? 0. : (1. - w1);
      emitHit(4 + 2 * g, g + 1, wArr * th2Gap, wArr * W);
      offPrevX = offArrX;
      offPrevY = offArrY;
      offPrevZ = offArrZ;
    }
    return true;
  }

  // The GBL system A is a 1x1 border (q/p, index 0) coupling to every offset, plus a SYMMETRIC BAND over the
  // 2-D offsets: kinks couple nodes up to distance 2, so the offset half-bandwidth is kGblBand = 5 (structural,
  // any N). bandFactor / bandSolveInPlace cost O(n) storage and O(n*band^2) work against the dense
  // (2N+3)^2 = O(N^2)/O(N^3); the 1x1 border is removed by a scalar Schur complement in gblFitPca.
  constexpr int kGblBand = 5;

  /*!
    \brief The unfactorized GBL fit at fixed N, returning the curvilinear covariance at the PCA (node 0).
    Same math as the host gblFitPca, with fixed-size stack matrices and the bordered-band solve.

    \tparam N number of trajectory nodes excluding the PCA, i.e. nNodes = N+1. It is a node count, not a hit
    count: production drives it with the split layout (kSplitN = 2*nHits), where nodes alternate
    measurement-less scatterers with hits.
    \param nodes per-node inputs (node 0 = PCA reference, nodes 1..N = trajectory nodes).
    \return 5x5 curvilinear covariance (q/p, lambda, phi, x_T, y_T) at the PCA.
  */
  // Scratch overlays inside the per-fit scratch block, each licensed by a lifetime boundary marked at its site:
  //   (a) `ms` reuses `cbor`'s nB doubles: cbor's last read is the Schur accumulation `schur -= cbor[i]*w[i]`,
  //       and ms's first use zeroes [0,nB) before writing. w keeps its address.
  //   (b) `fullDelta` may reuse the head of `Mb` (fullDeltaInScratch): the epilogue writes it after Mb's last
  //       read and after the nodeVar block, and every caller consumes it before any later gblFitPca call
  //       re-enters the band. nodeVar cannot move there: it is written mid-solve by solves that read Mb.

  // Per-fit doubles the bordered-band solver needs in `scratch`: Mb[nB*(kGblBand+1)] + cbor(=ms)/w[nB],
  // nB = 2*(N+1) offsets. Callers size the buffer for the largest N they launch.
  template <int N>
  inline constexpr int kGblScratchDoubles = (2 * (N + 1)) * (kGblBand + 1) + 2 * (2 * (N + 1));

  // Offset (in doubles) of the border INFLUENCE VECTOR w = M^-1 c inside the same `scratch` block, past Mb and
  // cbor. w is written once by the border elimination and never overwritten, so after gblFitPca returns a
  // caller may read it to express the fitted curvature as a linear functional of the measurements:
  //     delta(0) = ( b(0) - sum_i w[i] b(1+i) ) / schur,
  // a residual along the first offset component at a measurement node with precision P and offset index u
  // entering the curvature with coefficient -( w[u-1] P(0,0) + w[u] P(1,0) ) / schur.
  template <int N>
  inline constexpr int kGblInfluenceOffset = (2 * (N + 1)) * (kGblBand + 1) + (2 * (N + 1));

  template <typename TAcc, int N>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE Matrix5d gblFitPca(const TAcc& acc,
                                                    const GblNodeData* nodes,
                                                    // bordered-band solver scratch (Mb/cbor/w/ms), one
                                                    // kGblScratchDoubles<N> block per fit, held off the kernel
                                                    // stack frame in a device buffer. All four arrays are fully
                                                    // written before read.
                                                    double* __restrict__ scratch,
                                                    Vector5d* corr = nullptr,
                                                    double* fullDelta = nullptr,
                                                    double* chi2 = nullptr,
                                                    // optional out (3 per node): the [var(u0), cov(u0,u1), var(u1)]
                                                    // diagonal 2x2 block of A^-1 at each MEASUREMENT node (zeros
                                                    // elsewhere) -- the smoothed-offset covariance the outlier
                                                    // rejection needs for residual pulls r^T (V - C)^-1 r.
                                                    double* nodeVar = nullptr,
                                                    // Overlay (b) opt-in: the caller declares that its `fullDelta`
                                                    // IS the head of this fit's `scratch` block, so the epilogue
                                                    // writes through `scratch` and the __restrict__ contract
                                                    // holds. A disjoint fullDelta leaves it false.
                                                    bool fullDeltaInScratch = false) {
    constexpr int nNodes = N + 1;
    constexpr int nP = 2 * nNodes + 1;  // q/p + 2 offsets per node
    auto uIdx = [](int k) { return 1 + 2 * k; };

    // extraction at node 0 (forward): (q/p, lambda, phi, x_T, y_T) from (q/p, u_0, u_1)
    Matrix2d nW, nWJ;
    Vector2d nWd;
    nextDerivatives(nodes[1].jacToPrev, nW, nWJ, nWd);
    Matrix5d L = Matrix5d::Zero();
    L(0, 0) = 1.;
    L.block<2, 1>(1, 0) = -nWd;
    L.block<2, 2>(1, 1) = -nWJ;
    L.block<2, 2>(1, 3) = nW;
    L.block<2, 2>(3, 1) = Matrix2d::Identity();
    const int s[5] = {0, uIdx(0), uIdx(0) + 1, uIdx(1), uIdx(1) + 1};

    Eigen::Matrix<double, nP, 1> b = Eigen::Matrix<double, nP, 1>::Zero();
    double chi2ref = 0.;  // chi2 of the smooth reference: sum_meas residual^T * prec * residual
    const bool wantDelta = (corr || fullDelta || chi2);
    if (wantDelta)
      for (int i = 0; i < nNodes; ++i)
        if (nodes[i].hasMeas) {
          b.template segment<2>(uIdx(i)) += nodes[i].measPrec * nodes[i].measResidual;
          chi2ref += nodes[i].measResidual.dot(nodes[i].measPrec * nodes[i].measResidual);
        }

    Matrix5d covSub;
    Eigen::Matrix<double, nP, 1> delta = Eigen::Matrix<double, nP, 1>::Zero();  // = A^-1 b (only read when wantDelta)

    constexpr int nB = nP - 1;  // band dimension = the offset indices 1..nP-1
    double dBorder = 0.;        // A(0,0)
    // Off-stack band scratch, carved from this fit's `scratch` block (kGblScratchDoubles<N> doubles):
    // Mb (lower-packed band) then the border row cbor and the two solve vectors w/ms.
    double* Mb = scratch;                     // nB*(kGblBand+1)
    double* cbor = Mb + nB * (kGblBand + 1);  // border row A(0, 1..nP-1)
    for (int i = 0; i < nB; ++i)
      cbor[i] = 0.;
    for (int i = 0; i < nB * (kGblBand + 1); ++i)
      Mb[i] = 0.;
    // scatter A(r,c) += v into the border (index 0) + the lower band (r >= c, |r-c| <= kGblBand structural).
    auto add = [&](int r, int c, double v) {
      if (r == 0) {
        if (c == 0)
          dBorder += v;
        else
          cbor[c - 1] += v;
        return;
      }
      if (c == 0)
        return;  // symmetric border entry, already accumulated via (0, r)
      if (r >= c)
        Mb[(r - 1) * (kGblBand + 1) + (r - c)] += v;  // lower triangle of M
    };
    // measurements (identity projection in the U,V offset frame)
    for (int i = 0; i < nNodes; ++i)
      if (nodes[i].hasMeas) {
        const int u = uIdx(i);
        const Matrix2d& mp = nodes[i].measPrec;
        add(u, u, mp(0, 0));
        add(u, u + 1, mp(0, 1));
        add(u + 1, u, mp(1, 0));
        add(u + 1, u + 1, mp(1, 1));
      }
    // kinks at interior nodes: D = [-sumWd | prevW | -sumWJ | nextW] over (q/p, u_{i-1}, u_i, u_{i+1})
    for (int i = 1; i < nNodes - 1; ++i)
      if (nodes[i].hasScat) {
        Matrix2d prevW, prevWJ, nextW, nextWJ;
        Vector2d prevWd, nextWd;
        prevDerivatives(nodes[i].jacToPrev, prevW, prevWJ, prevWd);
        nextDerivatives(nodes[i + 1].jacToPrev, nextW, nextWJ, nextWd);
        const Matrix2d sumWJ = prevWJ + nextWJ;
        const Vector2d sumWd = prevWd + nextWd;
        Eigen::Matrix<double, 2, 7> D;
        D.block<2, 1>(0, 0) = -sumWd;
        D.block<2, 2>(0, 1) = prevW;
        D.block<2, 2>(0, 3) = -sumWJ;
        D.block<2, 2>(0, 5) = nextW;
        const Eigen::Matrix<double, 7, 7> K = D.transpose() * nodes[i].scatPrec * D;
        const int idx[7] = {0, uIdx(i - 1), uIdx(i - 1) + 1, uIdx(i), uIdx(i) + 1, uIdx(i + 1), uIdx(i + 1) + 1};
        for (int a = 0; a < 7; ++a)
          for (int bcol = 0; bcol < 7; ++bcol)
            add(idx[a], idx[bcol], K(a, bcol));
      }
    // factor M, then eliminate the 1x1 q/p border by a scalar Schur complement: w = M^-1 c, schur = d - c^T w.
    bandFactor<kGblBand>(Mb, nB);
    double* w = cbor + nB;
    for (int i = 0; i < nB; ++i)
      w[i] = cbor[i];
    bandSolveInPlace<kGblBand>(Mb, w, nB);
    double schur = dBorder;
    for (int i = 0; i < nB; ++i)
      schur -= cbor[i] * w[i];
    // cbor's last read is the line above; overlay (a) hands its storage to `ms` from here on (ms's first use,
    // the j-loop below, zeroes [0,nB) before writing). Leading 5x5 of A^-1 from unit-column solves
    // A x = e_{s[j]}: with rhs = [r0; sBand], x0 = (r0 - w^T sBand)/schur and the band part = M^-1 sBand - x0 w.
    double* ms = cbor;
    for (int j = 0; j < 5; ++j) {
      const int g = s[j];
      double dot_ws = 0.;
      if (g == 0) {
        // s[0] == 0 is the BORDER column: its band right-hand side is identically zero, so M ms = 0 can only
        // return ms == 0. The solve is skipped and the zero substituted where it is read below; the band part
        // of this column is entirely carried by the -x0*w term.
      } else {
        for (int i = 0; i < nB; ++i)
          ms[i] = 0.;
        ms[g - 1] = 1.;  // sBand = e_{g-1}
        dot_ws = w[g - 1];
        // leading zeros: ms[i] == 0 for i < g-1. Every covSub entry is read at s[a]-1 <= s[4]-1 = 3,
        // so the back sweep must still reach index 0.
        bandSolveInPlace<kGblBand>(Mb, ms, nB, 1, /*iFirst=*/g - 1, /*iStop=*/0);
      }
      const double r0 = (g == 0) ? 1. : 0.;
      const double x0 = (r0 - dot_ws) / schur;
      for (int a = 0; a < 5; ++a) {
        const int ga = s[a];
        const double msa = (g == 0) ? 0. : ms[ga - 1];
        covSub(a, j) = (ga == 0) ? x0 : (msa - x0 * w[ga - 1]);
      }
    }
    // per-node smoothed-offset covariance: two more unit-column solves per measurement node, same machinery
    // as the 5x5 extraction above.
    if (nodeVar)
      for (int i = 0; i < nNodes; ++i) {
        nodeVar[3 * i] = nodeVar[3 * i + 1] = nodeVar[3 * i + 2] = 0.;
        if (!nodes[i].hasMeas)
          continue;
        const int u = uIdx(i);
        for (int c = 0; c < 2; ++c) {
          const int g = u + c;
          // e_{g-1} has g-1 leading zeros and only ms[u-1], ms[u] are read back, so the forward+diagonal sweeps
          // start at g-1 and the back sweep stops at u-1+c. The zeroed window only has to cover what the sweep
          // reads, down to iFirst-kGblBand; the next solve re-zeros its own window.
          const int iFirst = g - 1;
          const int zLo = (iFirst - kGblBand > 0) ? (iFirst - kGblBand) : 0;
          for (int k2 = zLo; k2 < nB; ++k2)
            ms[k2] = 0.;
          ms[g - 1] = 1.;
          const double dot_ws = w[g - 1];
          bandSolveInPlace<kGblBand>(Mb, ms, nB, 1, iFirst, /*iStop=*/u - 1 + c);
          const double x0 = (0. - dot_ws) / schur;
          if (c == 0) {
            nodeVar[3 * i] = ms[u - 1] - x0 * w[u - 1];
            nodeVar[3 * i + 1] = ms[u] - x0 * w[u];
          } else {
            nodeVar[3 * i + 2] = ms[u] - x0 * w[u];
          }
        }
      }
    // delta = A^-1 b (b has no q/p-border component: b(0) = 0, measurements never touch q/p directly).
    if (wantDelta) {
      double dot_ws = 0.;
      for (int i = 0; i < nB; ++i) {
        ms[i] = b(1 + i);
        dot_ws += w[i] * b(1 + i);
      }
      bandSolveInPlace<kGblBand>(Mb, ms, nB);
      const double x0 = (b(0) - dot_ws) / schur;
      delta(0) = x0;
      for (int i = 0; i < nB; ++i)
        delta(1 + i) = ms[i] - x0 * w[i];
    }

    // parameter corrections (the fit VALUE); propagate node 0 via the same L.
    if (wantDelta) {
      // Overlay (b): with fullDeltaInScratch the caller's `fullDelta` is the head of this fit's own scratch
      // block, so the write goes through `scratch` and the __restrict__ contract holds. Legal only here: Mb's
      // last read was the final bandSolveInPlace above, the nodeVar solves are done, and
      // nP = 2*nNodes+1 <= nB*(kGblBand+1) always, so the write stays inside Mb.
      double* fullDeltaOut = (fullDelta != nullptr && fullDeltaInScratch) ? scratch : fullDelta;
      if (fullDeltaOut)
        for (int a = 0; a < nP; ++a)
          fullDeltaOut[a] = delta(a);
      if (chi2)  // chi2_min = chi2_ref - delta^T*b (kink residuals 0; penalty enters via A^-1). ndof = 2N-5.
        *chi2 = chi2ref - delta.dot(b);
      if (corr) {
        Vector5d dsub;
        for (int a = 0; a < 5; ++a)
          dsub(a) = delta(s[a]);
        *corr = L * dsub;
      }
    }
    return L * covSub * L.transpose();
  }

  // Extract the helix params (phi, d0, k=1/R, cotTheta, z0) + 5x5 covariance at the PCA from the GBL
  // curvilinear result, via the curvilinear->perigee transformation, with a propagation from node 0 (the
  // fast-fit PCA) to the fitted helix's true transverse PCA.
  template <typename TAcc, typename V4>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void gblHelixAtPca(const TAcc& acc,
                                                    const V4& fast_fit,
                                                    int qCharge,
                                                    double bField,
                                                    double sTransverse0,
                                                    double hitZ0,
                                                    const Vector5d& corr,
                                                    const Matrix5d& cov,
                                                    Vector5d& helixPar,
                                                    Matrix5d& helixCov,
                                                    // optional: the fitted circle (cx, cy, R, cotTheta-encoded
                                                    // 4th slot) as a fast_fit reference for a GBL
                                                    // re-linearization, in the layout prepareGblData consumes
                                                    Eigen::Vector4d* nextRef = nullptr,
                                                    // Signs the node-0 -> PCA arc by the direction of travel so
                                                    // that it flips with the charge; off, the arc keeps the CCW
                                                    // turn angle's own sign for both charges.
                                                    bool chargeSymmetric = false) {
    const double cx = fast_fit(0), cy = fast_fit(1), R = fast_fit(2);
    const double slope = -double(qCharge) / fast_fit(3);
    const double sec2 = 1. + slope * slope, invn = 1. / alpaka::math::sqrt(acc, sec2);
    const double pTot = bField * R * alpaka::math::sqrt(acc, sec2);
    const double cmag = alpaka::math::sqrt(acc, cx * cx + cy * cy);
    const double pcaX = cx * (1. - R / cmag), pcaY = cy * (1. - R / cmag), z0ref = hitZ0 - slope * sTransverse0;

    const double rx = (pcaX - cx) / R, ry = (pcaY - cy) / R;
    const Vector3d T(double(qCharge) * ry * invn, -double(qCharge) * rx * invn, slope * invn);
    const Vector3d Z(0., 0., 1.);
    const Vector3d U = curvilinearToPerigee::normalize3(acc, curvilinearToPerigee::cross3(Z, T));
    const Vector3d V = curvilinearToPerigee::cross3(T, U);
    const double phiRef = alpaka::math::atan2(acc, T.y(), T.x());
    const double lamRef = alpaka::math::atan2(acc, T.z(), alpaka::math::sqrt(acc, T.x() * T.x() + T.y() * T.y()));

    const double phiFit = phiRef + corr(2), lamFit = lamRef + corr(1);
    const double qbpFit = double(qCharge) / pTot + corr(0);
    const double pFit = 1. / alpaka::math::abs(acc, qbpFit);
    const double clam = alpaka::math::cos(acc, lamFit), slam = alpaka::math::sin(acc, lamFit);
    const Vector3d mom(
        pFit * clam * alpaka::math::cos(acc, phiFit), pFit * clam * alpaka::math::sin(acc, phiFit), pFit * slam);
    const Vector3d pos(pcaX + corr(3) * U.x() + corr(4) * V.x(),
                       pcaY + corr(3) * U.y() + corr(4) * V.y(),
                       z0ref + corr(3) * U.z() + corr(4) * V.z());

    // propagate node 0 -> the fitted helix's true transverse PCA, then perigee there.
    const double ptf = alpaka::math::sqrt(acc, mom.x() * mom.x() + mom.y() * mom.y());
    const double pmag = alpaka::math::sqrt(acc, ptf * ptf + mom.z() * mom.z());
    const double Rf = ptf / bField;
    const double tnx = mom.x() / ptf, tny = mom.y() / ptf;
    const double Cx = pos.x() + Rf * double(qCharge) * tny, Cy = pos.y() - Rf * double(qCharge) * tnx;
    const double Cmag = alpaka::math::sqrt(acc, Cx * Cx + Cy * Cy);
    const double Px = Cx * (1. - Rf / Cmag), Py = Cy * (1. - Rf / Cmag);
    if (nextRef) {                             // the fitted circle, ready to re-seed a GBL re-linearization pass
      const double slopeNext = mom.z() / ptf;  // cotTheta of the fitted track
      (*nextRef)(0) = Cx;
      (*nextRef)(1) = Cy;
      (*nextRef)(2) = Rf;
      (*nextRef)(3) = -double(qCharge) / slopeNext;  // prepareGblData reads slope = -qCharge/fast_fit(3)
    }
    const double v0x = pos.x() - Cx, v0y = pos.y() - Cy, vPx = Px - Cx, vPy = Py - Cy;
    const double dAlpha = alpaka::math::atan2(acc, v0x * vPy - v0y * vPx, v0x * vPx + v0y * vPy);
    // dAlpha is the CCW turn angle about the circle centre; the FORWARD arc advances the azimuth at
    // dpsi/ds = -q/Rf, so the signed arc is -q*Rf*dAlpha. The momentum rotation below is a rigid rotation by
    // the same CCW angle and carries no charge.
    const double dsT = (chargeSymmetric ? -double(qCharge) : 1.) * Rf * dAlpha, ds = dsT * pmag / ptf;
    const double ca = alpaka::math::cos(acc, dAlpha), sa = alpaka::math::sin(acc, dAlpha);
    const Vector3d momPca(mom.x() * ca - mom.y() * sa, mom.x() * sa + mom.y() * ca, mom.z());
    const Vector3d posPca(Px, Py, pos.z() + (mom.z() / ptf) * dsT);
    const double qbp = double(qCharge) / pmag;
    const Matrix5d Jprop =
        curvilinearJacobian(acc, mom / pmag, momPca / pmag, pos - posPca, qbp, Vector3d(0., 0., bField), ds);
    const Matrix5d covPca = Jprop * cov * Jprop.transpose();

    const Vector3d bvec(0., 0., bField);
    const Vector5d perigee = curvilinearToPerigee::perigeeParameters(acc, posPca, momPca, qCharge, bField);
    const Matrix5d Jcp = curvilinearToPerigee::jacobian(acc, momPca, bvec, qCharge);
    const Matrix5d perigeeCov = Jcp * covPca * Jcp.transpose();

    const double theta = perigee(1);
    const double cott = 1. / alpaka::math::tan(acc, theta), dcot_dtheta = -(1. + cott * cott);
    helixPar(0) = perigee(2);   // phi
    helixPar(1) = perigee(3);   // d0
    helixPar(2) = -perigee(0);  // k = 1/R
    helixPar(3) = cott;         // cotTheta
    helixPar(4) = perigee(4);   // z0
    Matrix5d M = Matrix5d::Zero();
    M(0, 2) = 1.;
    M(1, 3) = 1.;
    M(2, 0) = -1.;
    M(3, 1) = dcot_dtheta;
    M(4, 4) = 1.;
    helixCov = M * perigeeCov * M.transpose();
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::generalBrokenLine

#endif  // RecoTracker_PixelTrackFitting_interface_alpaka_GeneralBrokenLine_h
