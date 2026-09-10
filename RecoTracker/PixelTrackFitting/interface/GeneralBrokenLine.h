#ifndef RecoTracker_PixelTrackFitting_GeneralBrokenLine_h
#define RecoTracker_PixelTrackFitting_GeneralBrokenLine_h

// General Broken Lines (Kleinwort, NIM A 673 (2012) 107) -- host implementation.
//
// An unfactorized 5-parameter track fit, equivalent to a Kalman filter plus smoother: a single (2n+1)
// bordered-band least-squares with full point-to-point curvilinear helix Jacobians, keeping the parameter
// covariance correlations a factorized circle+line fit drops. The point-to-point Jacobian is the analytic
// curvilinear propagation of CMSSW's AnalyticalCurvilinearJacobian (TrackingTools/AnalyticalJacobians),
// transcribed into a self-contained, device-portable function (no ROOT SMatrix, no framework objects).
//
// This is the readable reference for the algebra the device fit (interface/alpaka/GeneralBrokenLine.h) runs,
// in dense Eigen and dynamic sizes; it is included only by the unit tests and the replay tool under
// RecoTracker/PixelTrackFitting/test/, by no production module. Its fit-model options (the field map,
// trajectoryCorrections, scatteringLogAtTotal, elossCumulative, gblHelixAtPca's chargeSymmetric) default to
// off, while the fit producers drive the device implementation with all of them on. Where the two differ, the
// device source is normative for the fit model.

#include <cmath>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "RecoTracker/PixelTrackFitting/interface/CurvilinearToPerigee.h"  // curvilinear -> perigee transform
#include "RecoTracker/PixelTrackFitting/interface/BLBFieldMap.h"           // normalized (Bz,Br) r-z field map

namespace generalBrokenLine {

  using Matrix5d = Eigen::Matrix<double, 5, 5>;
  using Matrix3d = Eigen::Matrix<double, 3, 3>;
  using Matrix2d = Eigen::Matrix<double, 2, 2>;
  using Matrix23d = Eigen::Matrix<double, 2, 3>;
  using Vector5d = Eigen::Matrix<double, 5, 1>;
  using Vector3d = Eigen::Vector3d;
  using Vector2d = Eigen::Vector2d;
  /*!
    \brief Analytic curvilinear point-to-point Jacobian for a helix in a uniform magnetic field.

    Curvilinear basis (q/p, lambda, phi, x_T, y_T): inverse momentum, dip angle, azimuth, and the two
    transverse offsets in the local (U = Z x T, V = T x U) frame. Propagates the 5x5 covariance from the
    start point to the end point: cov_end = J cov_start J^T (here J = d(end)/d(start)).

    \param p1 unit momentum direction at the START point.
    \param p2 unit momentum direction at the END point.
    \param dx (xStart - xEnd), the start-minus-end position vector [cm].
    \param qbp signed inverse momentum q/|p| [1/GeV] (CMSSW signedInverseMomentum).
    \param hInvGeV magnetic field in inverse-GeV units (= B[T] * 2.99792458e-3), along the field direction.
    \param s path length from start to end [cm] (positive).
    \return the 5x5 curvilinear transport Jacobian.

    Transcribed from CMSSW's AnalyticalCurvilinearJacobian::computeFullJacobian.
  */
  inline Matrix5d curvilinearJacobian(
      const Vector3d& p1, const Vector3d& p2, const Vector3d& dx, double qbp, const Vector3d& hInvGeV, double s) {
    Matrix5d J = Matrix5d::Identity();

    const double t11 = p1.x(), t12 = p1.y(), t13 = p1.z();
    const double t21 = p2.x(), t22 = p2.y(), t23 = p2.z();
    const double cosl0 = std::sqrt(t11 * t11 + t12 * t12);       // p1.perp()
    const double cosl1 = 1. / std::sqrt(t21 * t21 + t22 * t22);  // 1 / p2.perp()

    const Vector3d hn = hInvGeV.normalized();
    const double qp = -hInvGeV.norm();
    const double q = qp * qbp;
    const double theta = q * s;
    const double sint = std::sin(theta), cost = std::cos(theta);
    const double hn1 = hn.x(), hn2 = hn.y(), hn3 = hn.z();
    const double dx1 = dx.x(), dx2 = dx.y(), dx3 = dx.z();
    const double gamma = hn1 * t21 + hn2 * t22 + hn3 * t23;
    const double an1 = hn2 * t23 - hn3 * t22;
    const double an2 = hn3 * t21 - hn1 * t23;
    const double an3 = hn1 * t22 - hn2 * t21;
    double au = 1. / std::sqrt(t11 * t11 + t12 * t12);
    const double u11 = -au * t12, u12 = au * t11;
    const double v11 = -t13 * u12, v12 = t13 * u11, v13 = t11 * u12 - t12 * u11;
    au = 1. / std::sqrt(t21 * t21 + t22 * t22);
    const double u21 = -au * t22, u22 = au * t21;
    const double v21 = -t23 * u22, v22 = t23 * u21, v23 = t21 * u22 - t22 * u21;
    const double anv = -(hn1 * u21 + hn2 * u22);
    const double anu = (hn1 * v21 + hn2 * v22 + hn3 * v23);
    const double omcost = 1. - cost, tmsint = theta - sint;

    const double hu1 = -hn3 * u12, hu2 = hn3 * u11, hu3 = hn1 * u12 - hn2 * u11;
    const double hv1 = hn2 * v13 - hn3 * v12, hv2 = hn3 * v11 - hn1 * v13, hv3 = hn1 * v12 - hn2 * v11;

    // row 0 (1/p): unchanged (J(0,0)=1, rest 0) -- already identity.

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

    // rows 3,4 col 0 (x_T, y_T vs 1/p): exact for long steps, Taylor (to 4th order) for short.
    const double cutCriterion = std::abs(s * qbp);  // s / |p|
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
      const double h1 = hInvGeV.norm(), h2 = h1 * h1, h3 = h2 * h1;
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

  // Offset-frame slope-response derivatives W, W*J, W*d (DESY GblPoint::getDerivatives), from the
  // point-to-point Jacobian to the NEXT offset.
  inline void nextDerivatives(const Matrix5d& jac, Matrix2d& W, Matrix2d& WJ, Vector2d& Wd) {
    const Matrix2d matJ = jac.block<2, 2>(3, 3);
    const Matrix2d matW = jac.block<2, 2>(3, 1);
    const Vector2d vecd = jac.block<2, 1>(3, 0);
    W = matW.inverse();
    WJ = W * matJ;
    Wd = W * vecd;
  }
  // From the point-to-point Jacobian to the PREVIOUS offset: needs the bottom rows of its inverse
  // (DESY GblPoint::addPrevJacobian block-matrix algebra), then the same getDerivatives extraction.
  inline void prevDerivatives(const Matrix5d& jac, Matrix2d& W, Matrix2d& WJ, Vector2d& Wd) {
    const Matrix3d A33 = jac.block<3, 3>(0, 0);
    const Eigen::Matrix<double, 3, 2> B32 = jac.block<3, 2>(0, 3);
    const Matrix23d C23 = jac.block<2, 3>(3, 0);
    const Matrix2d D22 = jac.block<2, 2>(3, 3);
    const Matrix23d CA = C23 * A33.inverse();             // C * A^-1
    const Matrix2d DCABinv = (D22 - CA * B32).inverse();  // (D - C*A^-1*B)^-1
    const Matrix23d prevD012 = -DCABinv * CA;             // prevJacobian rows (3,4), cols (0,1,2)
    const Matrix2d matJ = DCABinv;                        // prevJacobian block(3,3)
    const Matrix2d matW = -prevD012.block<2, 2>(0, 1);    // -prevJacobian block(3,1)
    const Vector2d vecd = prevD012.block<2, 1>(0, 0);     //  prevJacobian block(3,0)
    W = matW.inverse();
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

  // Module normal from the (rank-2) global hit covariance. ge3 = R^T diag(sigma_x^2, 0, sigma_y^2) R has the
  // local normal-direction variance == 0, so its null space is the module normal and its rows lie in the sensor
  // plane: any two independent rows cross to the normal, the largest-norm cross product being the stable choice.
  inline Vector3d planeNormalFromGe3(const Matrix3d& m) {
    const Vector3d r0(m(0, 0), m(0, 1), m(0, 2));
    const Vector3d r1(m(1, 0), m(1, 1), m(1, 2));
    const Vector3d r2(m(2, 0), m(2, 1), m(2, 2));
    const Vector3d c01 = r0.cross(r1), c02 = r0.cross(r2), c12 = r1.cross(r2);
    Vector3d n = c01;
    double best = c01.squaredNorm();
    if (c02.squaredNorm() > best) {
      n = c02;
      best = c02.squaredNorm();
    }
    if (c12.squaredNorm() > best) {
      n = c12;
      best = c12.squaredNorm();
    }
    const double nn = std::sqrt(n.squaredNorm());
    return (nn > 0.) ? Vector3d(n / nn) : Vector3d(0., 0., 1.);
  }

  // Moore-Penrose pseudo-inverse of the rank-2 hit covariance ge3 = R^T diag(sx^2, 0, sy^2) R (the local
  // normal-direction variance is 0). ge3+ = R^T diag(1/sx^2, 0, 1/sy^2) R is the in-plane measurement precision: with
  // the unit null vector n (the module normal), ge3 + n n^T has eigenvalues (sx^2, sy^2, 1), so its inverse minus
  // n n^T leaves diag(1/sx^2, 1/sy^2, 0) = ge3+.
  inline Matrix3d pseudoInverseGe3(const Matrix3d& ge3) {
    const Vector3d n = planeNormalFromGe3(ge3);
    const Matrix3d nnT = n * n.transpose();
    return (ge3 + nnT).inverse() - nnT;
  }

  // Constants of the two ionization energy-loss laws below (elossMostProbable, elossTypicalColumn): both
  // evaluate the same Landau xi in the same medium and differ only in which statistic of the loss distribution
  // they return. The medium is the composite effective medium of the charged bands (Bragg additivity rule over
  // the material budget), not pure silicon.
  namespace elossMedium {
    constexpr double kPionMass = 0.13957;           // pion mass [GeV]
    constexpr double kElectronMass = 0.5109989e-3;  // electron mass [GeV]
    constexpr double kK = 0.307075e-3;              // 0.307 MeV cm^2/mol -> GeV
    constexpr double kZ_A = 0.500;                  // composite <Z/A>
    constexpr double kI = 122e-9;                   // composite mean excitation energy [GeV]
    constexpr double kX0g = 28.8;                   // composite X0 [g/cm^2]
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
  inline double elossMostProbable(double p, double xx0) {
    constexpr double m = elossMedium::kPionMass;
    constexpr double me = elossMedium::kElectronMass;
    constexpr double K = elossMedium::kK;
    constexpr double Z_A = elossMedium::kZ_A;
    constexpr double I = elossMedium::kI;
    constexpr double X0g = elossMedium::kX0g;
    constexpr double hwp = elossMedium::kHwp;
    if (xx0 <= 0.)
      return 0.;
    const double E = std::sqrt(p * p + m * m);
    const double beta2 = p * p / (E * E);
    const double g = E / m;
    const double bg2 = beta2 * g * g;  // (beta*gamma)^2
    const double xi = 0.5 * K * Z_A * (xx0 * X0g) / beta2;
    // density effect, high-betagamma Sternheimer limit: delta -> 2 ln(hwp/I * betagamma) - 1 (>=0)
    const double dhalf = std::log((hwp / I) * std::sqrt(bg2)) - 0.5;
    const double delta = dhalf > 0. ? 2. * dhalf : 0.;
    const double bracket = std::log(2. * me * bg2 / I) + std::log(xi / I) + 0.2 - beta2 - delta;
    const double dmp = xi * bracket;
    return dmp > 0. ? dmp : 0.;
  }

  // Typical (median) ionization loss [GeV] of the CUMULATIVE charged column of thickness xx0 [X/X0], same
  // medium as elossMostProbable, selected over the per-lump most-probable law by the runtime flag
  // elossCumulative. The Landau family is stable under convolution, so the typical loss of a multi-slab column
  // is the single-column law at the SUMMED thickness, not the sum of per-slab MPVs, and median - mode =
  // (1.35578 + 0.22278) xi for a Landau (PDG RPP 34.2.9). Callers charge per-node increments
  // T(X_cum + x_lump) - T(X_cum).
  inline double elossTypicalColumn(double p, double xx0) {
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
    const double E = std::sqrt(p * p + m * m);
    const double beta2 = p * p / (E * E);
    const double g = E / m;
    const double bg2 = beta2 * g * g;
    const double xi = 0.5 * K * Z_A * (xx0 * X0g) / beta2;
    const double dhalf = std::log((hwp / I) * std::sqrt(bg2)) - 0.5;
    const double delta = dhalf > 0. ? 2. * dhalf : 0.;
    const double bracket = std::log(2. * me * bg2 / I) + std::log(xi / I) + 0.2 - beta2 - delta;
    const double dmp = xi * bracket;
    return (dmp > 0. ? dmp : 0.) + kLandauMedianMinusMode * xi;
  }

  // Fractional stand-off of the equivalent upstream scatterer from the PCA and from hit0.
  inline constexpr double kSplitStandOff = 0.02;
  inline constexpr double kSplitStandOffHi = 0.98;

  /*!
    \brief Assemble the per-node GBL inputs (Jacobians, (U,V) measurement precisions, scatterers) for a track.

    PCA reference node + one node per hit, with the analytic curvilinear Jacobian between consecutive nodes,
    the global hit covariance projected onto the curvilinear (U,V) offset frame and inverted, and the prepared
    broken-line dataset (sTransverse/sTotal/matXX0/innerXX0/qCharge) for the scatterers.

    \tparam N number of hits.
    \param nodes output array of N+2 GblNodeData slots. Layout depends on the upstream-material model:
           - single scatterer (no valid innerD1/innerW1): node 0 = PCA, nodes 1..N = hits,
           - inner-node (valid innerD1/innerW1 from segmentXX0Moments): node 0 = PCA, node 1 = a
             measurement-less scatterer at the upstream-material equivalent position (Kleinwort
             two-thin-scatterer split), nodes 2..N+1 = hits. Extraction at node 0 yields the PCA
             parameters with exact levers.
  */
  template <int N, typename M3xN, typename M6xN, typename V4, typename VN>
  inline void prepareGblData(const M3xN& hits,
                             const M6xN& hits_ge,
                             const V4& fast_fit,
                             double bField,
                             int qCharge,
                             const VN& sTransverse,
                             const VN& sTotal,
                             const VN& matXX0,
                             double innerXX0,
                             GblNodeData* nodes,
                             double msScale = 1.0,  // multiple-scattering scale factor (1.0 in production)
                             // Enables the ionization-loss correction when > 0; only that test is read, the
                             // loss charged at a node coming from elossMostProbable / elossTypicalColumn at
                             // that node's thickness.
                             double eLossPerX0 = 0.0,
                             Matrix5d* jacHit0ToPca = nullptr,  // optional output: the hit0 -> PCA backward
                                                                // curvilinear Jacobian (single-scatterer layout)
                             double innerD1 = 0.,  // upstream equivalent-scatterer path distance from hit0 [cm]
                             double innerW1 = 0.,  // fraction of the upstream scattering variance at that node
                             bool* usedInnerNode = nullptr,
                             // Normalized (Bz,Br) r-z field map (blBFieldMap) + the origin field it normalizes
                             // to. With a map, every segment is charged the deterministic azimuth (and, under
                             // trajectoryCorrections, lambda) excess of the real field profile over the single
                             // constant the fit models; a null map accumulates nothing.
                             const float* bMap = nullptr,
                             double bFieldOrigin = 0.,
                             // Reference-trajectory corrections: the node-0 path length seeded with
                             // sin(lambda)*z0 rather than 0, the sign of the arc->azimuth rotation that places
                             // a measurement-less node, and the B_r lambda row of the field-profile offset.
                             bool trajectoryCorrections = false,
                             // Evaluates Highland's log at the track's TOTAL declared material rather than at
                             // each gap's own thickness (producer parameter useScatteringLogAtTotal).
                             bool scatteringLogAtTotal = false,
                             // Charges each gap the cumulative-column typical loss from the vertex to that node
                             // rather than each lump its own most-probable loss (useCumulativeEloss).
                             bool elossCumulative = false) {
    constexpr int nNodes = N + 2;
    const double cx = fast_fit(0), cy = fast_fit(1), R = fast_fit(2);
    const double slope = -double(qCharge) / fast_fit(3);
    const double pt = bField * R;
    const double sec2 = 1. + slope * slope;
    const double pTot = pt * std::sqrt(sec2);
    const double cos2lam = 1. / sec2;
    const double qbp = double(qCharge) / pTot;
    const Vector3d hInvGeV(0., 0., bField);  // field in inverse-GeV units (|.| = bField = B[T]*2.99792458e-3)
    const double invn = 1. / std::sqrt(sec2);
    // signed charge that turns a transverse arc into a rotation of the radius vector about the fitted centre
    const double qArc = trajectoryCorrections ? -double(qCharge) : double(qCharge);

    // node positions and unit momenta (node 0 = PCA, closest point on the fast-fit circle to the origin).
    Vector3d pos[nNodes], dir[nNodes];
    const double cmag = std::sqrt(cx * cx + cy * cy);
    auto fill = [&](int k, double x, double y, double z) {
      pos[k] = Vector3d(x, y, z);
      const double rx = (x - cx) / R, ry = (y - cy) / R;
      // physical FORWARD momentum tangent: the opposite sign leaves the (direction-invariant) covariance
      // alone but flips phi by pi and the d0 sign in the perigee VALUE.
      const double tx = double(qCharge) * ry, ty = -double(qCharge) * rx;
      dir[k] = Vector3d(tx * invn, ty * invn, slope * invn);
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
    // value, or curvilinearJacobian gets a displacement and an arc that are not the same step: that path
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

    fill(0, cx * (1. - R / cmag), cy * (1. - R / cmag), hits(2, 0) - slope * sTransverse(0));
    if (useInner) {
      // equivalent upstream scatterer on the reference helix, at the clamped transverse arc sTA
      const double sTA = sTAInner;
      // rotation of the PCA radius vector over the transverse arc sTA. The reference circle advances its
      // azimuth at dpsi/dsTransverse = -q/R, the convention sTransverse itself carries and the one the
      // transport Jacobians are built in, so the rotation is -q*sTA/R.
      const double phiA = qArc * sTA / R;
      const double ex = cx * (1. - R / cmag) - cx, ey = cy * (1. - R / cmag) - cy;
      const double ca = std::cos(phiA), sa = std::sin(phiA);
      fill(1, cx + ex * ca - ey * sa, cy + ex * sa + ey * ca, z0 + slope * sTA);
    }
    for (int i = 0; i < N; ++i)
      fill(i + hOff, hits(0, i), hits(1, i), hits(2, i));

    nodes[0] = GblNodeData{};  // PCA reference: no measurement, no scatterer
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
    // Total declared material: upstream lump plus every inter-hit gap that carries a kink. Read only
    // when scatteringLogAtTotal moves the Highland logarithm's argument from the gap to this total.
    double xx0TotDecl = 0.;
    if (scatteringLogAtTotal) {
      if (innerXX0 > 0.)
        xx0TotDecl += innerXX0;
      for (int i = 1; i <= N - 2; ++i)
        if (double(matXX0(i - 1)) > 0.)
          xx0TotDecl += double(matXX0(i - 1));
    }
    // Highland scattering variance: theta0 = 0.0136/(beta p) * sqrt(x/X0) * (1 + 0.038 ln(x/X0)). theta0^2 is
    // not additive over a chain (one logarithm); scatteringLogAtTotal moves that log from this gap's thickness
    // to the track's total declared thickness, each gap keeping its share and its partition weight.
    auto th2Of = [&](double xx0) {
      if (xx0 <= 0.)
        return 0.;
      constexpr double mPi = 0.13957;  // pion mass [GeV]
      const double betaP = pTot * pTot / std::sqrt(pTot * pTot + mPi * mPi);
      const double tt = 0.0136 / betaP;
      const double xLog = (scatteringLogAtTotal && xx0TotDecl > 0.) ? xx0TotDecl : xx0;
      const double f = 1. + 0.038 * std::log(xLog);
      return tt * tt * xx0 * f * f * msScale;
    };
    // total upstream scattering variance from the FULL innerXX0, split linearly between the equivalent
    // scatterer node (innerW1) and hit0 (1-innerW1) in the inner-node layout.
    const double th2InnerTot = th2Of(innerXX0);
    // Running deterministic curvilinear state shift, accumulated PCA -> node k. Two terms share one
    // accumulator, both being known offsets of the reference state, transported by the same Jacobians and
    // removed from the same residuals:
    //   slot 0 (q/p): the dE/dx spiral. |q/p| grows as the track loses momentum outward.
    //   slot 2 (phi): the bending-field PROFILE. The fit models one constant field, so its reference turns at
    //     dphi/ds = -qbp*bField while the trajectory turns at -qbp*B_bend(s); the deterministic difference
    //     leaks into d0 and phi0 if left in the residuals, and vanishes where B_bend == bField.
    // Only the offset part (x_T,y_T) of the accumulated shift is removed from the measurement residuals.
    const bool useField = (bMap != nullptr);
    const bool detOffset = (eLossPerX0 > 0.) || useField;
    // The lambda row of the field-profile offset (see the increment below): its only ingredient beyond the
    // azimuth row is B_r, which the bending law already reads at the same lattice cell.
    const bool useLambdaRow = useField && trajectoryCorrections;
    // B_bend/Bz(0,0) at a node, the same local law the effective field averages: the reference circle's
    // tanLambda*cos(alpha) carries q^2, so this is charge-free.
    auto bBendOf = [&](const Vector3d& p) {
      const double r = std::sqrt(p.x() * p.x() + p.y() * p.y());
      const double den = fast_fit(3) * std::abs(R) * r;
      const double tlca = (den != 0.) ? -(cx * p.y() - cy * p.x()) / den : 0.;
      return blBFieldMap::bBendAt(bMap, r, p.z(), tlca);
    };
    // The same law, plus B_r/Bz(0,0) times sin(alpha) (the lambda row's shape, charge-ODD unlike the bending
    // term) from the SAME lattice cell. alpha is the angle from the position azimuth to the momentum azimuth;
    // its sine is the partner of the cosine folded into tlca (cos^2 + sin^2 = 1 on the reference circle).
    auto bBendBrOf = [&](const Vector3d& p, double& brSinAlpha) {
      const double r = std::sqrt(p.x() * p.x() + p.y() * p.y());
      const double den = fast_fit(3) * std::abs(R) * r;
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
      bbPrev = useLambdaRow ? bBendBrOf(pos[0], bsPrev) : bBendOf(pos[0]);
    Vector5d Deloss = Vector5d::Zero();
    // running charged column [X/X0] for the cumulative typical-loss law (walk order = path order)
    double xx0ElossCum = 0.;
    for (int k = 1; k < N + hOff; ++k) {
      const int i = k - hOff;                    // hit index (negative for the scatterer-only node)
      const bool scatOnly = useInner && k == 1;  // upstream-material node: no measurement
      GblNodeData nd;
      nd.jacToPrev =
          curvilinearJacobian(dir[k - 1], dir[k], pos[k - 1] - pos[k], qbp, hInvGeV, sTotN[k] - sTotN[k - 1]);
      if (detOffset) {
        // This segment's field-profile azimuth excess: -qbp*(B_bend_seg - bField)*ds, with ds the 3D path
        // length the Jacobian above transports over and B_bend_seg the segment average of the two end nodes.
        // Half is charged at each end, which carries the offset built up inside the segment.
        double dPhiHalf = 0.;
        double dLamHalf = 0.;
        if (useField) {
          double bsCur = 0.;
          const double bbCur = useLambdaRow ? bBendBrOf(pos[k], bsCur) : bBendOf(pos[k]);
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
          const double tPrev = elossTypicalColumn(pTot, xx0ElossCum);
          xx0ElossCum += innerXX0 * innerW1;
          Deloss(0) += qbp * (elossTypicalColumn(pTot, xx0ElossCum) - tPrev) / pTot;
        } else if (eLossPerX0 > 0.)
          Deloss(0) += qbp * elossMostProbable(pTot, innerXX0 * innerW1) / pTot;
        nodes[k] = nd;
        continue;
      }

      // measurement: (U = Z x T, V = T x U) curvilinear frame from the unit momentum.
      const Vector3d& T = dir[k];
      const double un = 1. / std::sqrt(T.x() * T.x() + T.y() * T.y());
      const Vector3d U(-T.y() * un, T.x() * un, 0.);
      const Vector3d V = T.cross(U);
      Matrix23d Ruv;
      Ruv.row(0) = U.transpose();
      Ruv.row(1) = V.transpose();
      Matrix3d ge3;
      ge3 << hits_ge(0, i), hits_ge(1, i), hits_ge(3, i), hits_ge(1, i), hits_ge(2, i), hits_ge(4, i), hits_ge(3, i),
          hits_ge(4, i), hits_ge(5, i);
      // residual: measured hit minus the fast-fit reference helix (closest circle point in the bending plane,
      // z = z0 + slope*sTransverse longitudinally).
      const double dvx = hits(0, i) - cx, dvy = hits(1, i) - cy;
      const double dmag = std::sqrt(dvx * dvx + dvy * dvy);
      const Vector3d residual3(hits(0, i) - (cx + R * dvx / dmag),
                               hits(1, i) - (cy + R * dvy / dmag),
                               hits(2, i) - (z0 + slope * double(sTransverse(i))));
      // Project the global hit covariance ge3 onto the (U, V) curvilinear offset frame and invert.
      nd.measPrec = (Ruv * ge3 * Ruv.transpose()).inverse();
      nd.measResidual = Ruv * residual3;
      nd.hasMeas = true;

      // deterministic offsets (value-only, cov unchanged): the accumulated UPSTREAM state shift, propagated to
      // this node's offset via jacToPrev, is a known part of the residual, and removing it keeps the dE/dx
      // spiral from biasing the fitted q/p at the PCA.
      if (detOffset)
        nd.measResidual -= Deloss.template segment<2>(3);

      // scatterer (interior nodes): one thin scatterer per segment -- the upstream material reaching hit 0 (all
      // of it in the single-scatterer layout, the 1-innerW1 remainder in the inner-node layout), matXX0(i-1)
      // reaching the inner hits. One kink per segment, no both-adjacent double count.
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
      // d(q/p) = (q/p) dE/p, so |q/p| grows as p drops outward.
      if (eLossPerX0 > 0. && xx0Eloss > 0. && elossCumulative) {
        const double tPrev = elossTypicalColumn(pTot, xx0ElossCum);
        xx0ElossCum += xx0Eloss;
        Deloss(0) += qbp * (elossTypicalColumn(pTot, xx0ElossCum) - tPrev) / pTot;
      } else if (eLossPerX0 > 0. && xx0Eloss > 0.)
        Deloss(0) += qbp * elossMostProbable(pTot, xx0Eloss) / pTot;
      nodes[k] = nd;
    }
    // hit0 -> PCA backward jacobian: the exact inverse of the forward PCA->hit0 transport, recomputed via the
    // same curvilinearJacobian rather than a numerical 5x5 inverse. The single-scatterer layout references the
    // corrections/cov at hit0 and this propagates them back to the PCA reference gblHelixAtPca consumes; the
    // inner-node layout extracts at node 0 instead. The arc of this backward step is the DIFFERENCE of the two
    // nodes' path lengths, which -sTotN[hOff] is only while node 0's path length is seeded with 0.
    if (jacHit0ToPca && N >= 1)
      *jacHit0ToPca = curvilinearJacobian(dir[hOff],
                                          dir[0],
                                          pos[hOff] - pos[0],
                                          qbp,
                                          hInvGeV,
                                          trajectoryCorrections ? (sTotN[0] - sTotN[hOff]) : -sTotN[hOff]);
  }

  // The GBL normal matrix A = 1x1 (q/p) border + a symmetric band over the 2-D offsets (kinks couple nodes up
  // to distance 2, so the offset half-bandwidth is kGblBand = 5). M is stored lower-packed:
  // Mb[i*(kGblBand+1)+p] = M(i, i-p). bandFactor overwrites Mb with its root-free LDL^T factor,
  // bandSolveInPlace applies M^-1 in place, and the 1x1 border is removed by a scalar Schur complement.
  constexpr int kGblBand = 5;

  inline void bandFactor(double* Mb, int n) {
    for (int j = 0; j < n; ++j) {
      double* Mj = Mb + j * (kGblBand + 1);
      double Dj = Mj[0];
      const int pmax = (j < kGblBand) ? j : kGblBand;
      for (int p = 1; p <= pmax; ++p)
        Dj -= Mj[p] * Mj[p] * Mb[(j - p) * (kGblBand + 1)];
      Mj[0] = Dj;
      const int imax = (j + kGblBand < n - 1) ? (j + kGblBand) : (n - 1);
      for (int i = j + 1; i <= imax; ++i) {
        double* Mi = Mb + i * (kGblBand + 1);
        double sum = Mi[i - j];  // A(i, j)
        const int kmin = (i - kGblBand > 0) ? (i - kGblBand) : 0;
        for (int k = kmin; k < j; ++k)
          sum -= Mi[i - k] * Mj[j - k] * Mb[k * (kGblBand + 1)];
        Mi[i - j] = sum / Dj;  // L(i, j)
      }
    }
  }

  inline void bandSolveInPlace(const double* Mb, double* x, int n) {
    for (int i = 0; i < n; ++i) {  // forward: L y = rhs
      const double* Mi = Mb + i * (kGblBand + 1);
      const int pmax = (i < kGblBand) ? i : kGblBand;
      double si = x[i];
      for (int p = 1; p <= pmax; ++p)
        si -= Mi[p] * x[i - p];
      x[i] = si;
    }
    for (int i = 0; i < n; ++i)  // diagonal: D z = y
      x[i] /= Mb[i * (kGblBand + 1)];
    for (int i = n - 1; i >= 0; --i) {  // back: L^T x = z
      const int kmax = (i + kGblBand < n - 1) ? (i + kGblBand) : (n - 1);
      double si = x[i];
      for (int k = i + 1; k <= kmax; ++k)
        si -= Mb[k * (kGblBand + 1) + (k - i)] * x[k];
      x[i] = si;
    }
  }

  /*!
    \brief The unfactorized GBL fit, specialized to the pixel case (every node an offset, 2-D measurements,
    2-D kinks) and returning the covariance at the PCA (node 0). Equivalent to DESY GblTrajectory + getResults(1).

    Builds the (2*nNodes+1) bordered-band system A (q/p + a 2-D offset per node) from the measurement and kink
    contributions, inverts it, and propagates to the curvilinear track parameters (q/p, lambda, phi, x_T, y_T)
    at node 0.

    \param nodes per-node inputs, node 0 = PCA reference (no measurement/scatterer), nodes 1..n = hits.
    \param corr if non-null, filled with the 5-vector parameter CORRECTION (q/p, lambda, phi, x_T, y_T) at the PCA,
                solved from the measurement residuals (the fit VALUE: add to the fast-fit reference). Null = cov only.
    \return 5x5 curvilinear covariance at the PCA.
  */
  inline Matrix5d gblFitPca(const std::vector<GblNodeData>& nodes,
                            Vector5d* corr = nullptr,
                            Eigen::VectorXd* fullDelta = nullptr,
                            double* chi2 = nullptr,
                            // optional out (3 per node): [var(u0), cov(u0,u1), var(u1)] diagonal block of A^-1 at
                            // each MEASUREMENT node (zeros elsewhere), the smoothed-offset covariance for
                            // outlier-rejection residual pulls
                            double* nodeVar = nullptr) {
    const int nN = static_cast<int>(nodes.size());
    const int nP = 1 + 2 * nN;  // q/p + 2 offsets per node
    auto uIdx = [](int k) { return 1 + 2 * k; };

    // extraction at node 0 (forward): (q/p, lambda, phi, x_T, y_T) from (q/p, u_0, u_1)
    Matrix2d nW, nWJ;
    Vector2d nWd;
    nextDerivatives(nodes[1].jacToPrev, nW, nWJ, nWd);
    Matrix5d L = Matrix5d::Zero();
    L(0, 0) = 1.;                                // q/p
    L.block<2, 1>(1, 0) = -nWd;                  // (lambda,phi) from q/p
    L.block<2, 2>(1, 1) = -nWJ;                  // from u_0
    L.block<2, 2>(1, 3) = nW;                    // from u_1
    L.block<2, 2>(3, 1) = Matrix2d::Identity();  // (x_T,y_T) = u_0
    const int s[5] = {0, uIdx(0), uIdx(0) + 1, uIdx(1), uIdx(1) + 1};

    Eigen::VectorXd b = Eigen::VectorXd::Zero(nP);
    double chi2ref = 0.;  // chi2 of the (smooth) reference trajectory: sum_meas residual^T * prec * residual
    const bool wantDelta = (corr || fullDelta || chi2);
    if (wantDelta)
      for (int i = 0; i < nN; ++i)
        if (nodes[i].hasMeas) {
          b.segment<2>(uIdx(i)) += nodes[i].measPrec * nodes[i].measResidual;
          chi2ref += nodes[i].measResidual.dot(nodes[i].measPrec * nodes[i].measResidual);
        }

    Matrix5d covSub;
    Eigen::VectorXd delta = Eigen::VectorXd::Zero(nP);

    const int nB = nP - 1;                            // band dimension = the offset indices 1..nP-1
    double dBorder = 0.;                              // A(0,0)
    std::vector<double> cbor(nB, 0.);                 // border row A(0, 1..nP-1)
    std::vector<double> Mb(nB * (kGblBand + 1), 0.);  // lower-packed band of M (size <= 22*6)
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
    // measurements: in the (U,V) offset frame the projection is the identity -> directly constrain u_i.
    for (int i = 0; i < nN; ++i)
      if (nodes[i].hasMeas) {
        const int u = uIdx(i);
        const Matrix2d& mp = nodes[i].measPrec;
        add(u, u, mp(0, 0));
        add(u, u + 1, mp(0, 1));
        add(u + 1, u, mp(1, 0));
        add(u + 1, u + 1, mp(1, 1));
      }
    // kinks at interior nodes: D = [-sumWd | prevW | -sumWJ | nextW] over (q/p, u_{i-1}, u_i, u_{i+1}).
    for (int i = 1; i < nN - 1; ++i)
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
    bandFactor(Mb.data(), nB);
    std::vector<double> w(nB), ms(nB);
    for (int i = 0; i < nB; ++i)
      w[i] = cbor[i];
    bandSolveInPlace(Mb.data(), w.data(), nB);
    double schur = dBorder;
    for (int i = 0; i < nB; ++i)
      schur -= cbor[i] * w[i];
    // leading 5x5 of A^-1 from unit-column solves A x = e_{s[j]}: with rhs = [r0; sBand], x0 = (r0 - w^T sBand)/schur
    // and the band part = M^-1 sBand - x0 w.
    for (int j = 0; j < 5; ++j) {
      const int g = s[j];
      for (int i = 0; i < nB; ++i)
        ms[i] = 0.;
      double dot_ws = 0.;
      if (g != 0) {
        ms[g - 1] = 1.;  // sBand = e_{g-1}
        dot_ws = w[g - 1];
      }
      bandSolveInPlace(Mb.data(), ms.data(), nB);
      const double r0 = (g == 0) ? 1. : 0.;
      const double x0 = (r0 - dot_ws) / schur;
      for (int a = 0; a < 5; ++a) {
        const int ga = s[a];
        covSub(a, j) = (ga == 0) ? x0 : (ms[ga - 1] - x0 * w[ga - 1]);
      }
    }
    // per-node smoothed-offset covariance: two more unit-column solves per measurement node.
    if (nodeVar)
      for (int i = 0; i < nN; ++i) {
        nodeVar[3 * i] = nodeVar[3 * i + 1] = nodeVar[3 * i + 2] = 0.;
        if (!nodes[i].hasMeas)
          continue;
        const int u = uIdx(i);
        for (int c = 0; c < 2; ++c) {
          const int g = u + c;
          for (int k2 = 0; k2 < nB; ++k2)
            ms[k2] = 0.;
          ms[g - 1] = 1.;
          const double dot_ws = w[g - 1];
          bandSolveInPlace(Mb.data(), ms.data(), nB);
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
      bandSolveInPlace(Mb.data(), ms.data(), nB);
      const double x0 = (b(0) - dot_ws) / schur;
      delta(0) = x0;
      for (int i = 0; i < nB; ++i)
        delta(1 + i) = ms[i] - x0 * w[i];
    }

    // parameter corrections (the fit VALUE): propagate node 0 via the same L. chi2_min = chi2_ref - delta^T*b
    // (kink residuals are 0, so the scatterer penalty enters only through A^-1). ndof = 2N-5.
    if (wantDelta) {
      if (fullDelta)
        *fullDelta = delta;
      if (chi2)
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

  /*!
    \brief Extract the helix parameters (phi, d0, k=1/R, cotTheta, z0) + 5x5 covariance at the PCA from the GBL
    curvilinear result, via the curvilinear->perigee transformation (curvilinearToPerigee).

    Reconstructs the fitted state (position + momentum) at the PCA: the fast-fit reference (node 0) plus the
    GBL corrections (q/p, lambda, phi shift the momentum; x_T, y_T offset the position along the curvilinear
    U, V), then curvilinear->perigee, then the perigee->helix convention map (phi=phi, d0=epsilon, k=-kappa,
    cotTheta=cot(theta), z0=z; the k sign matches the circle fit's qCharge/R convention).

    \param fast_fit (cx, cy, R, ...) -- the pre-fit circle; slope = -qCharge/fast_fit(3).
    \param corr GBL curvilinear corrections at node 0 (from gblFitPca's corr output).
    \param cov  GBL curvilinear covariance at node 0 (from gblFitPca's return).
    \param helixPar [out] (phi, d0, k, cotTheta, z0).  \param helixCov [out] its 5x5 covariance.
  */
  template <typename V4>
  inline void gblHelixAtPca(
      const V4& fast_fit,
      int qCharge,
      double bField,
      double sTransverse0,
      double hitZ0,
      const Vector5d& corr,
      const Matrix5d& cov,
      Vector5d& helixPar,
      Matrix5d& helixCov,
      // optional: the fitted transverse circle (cx, cy, R) + the cotTheta-encoded 4th slot, as a fast_fit
      // reference for a second GBL iteration, in the layout prepareGblData/gblHelixAtPca consume
      Eigen::Vector4d* nextRef = nullptr,
      // Charge-symmetric arc convention for the node-0 -> PCA step: the transverse arc that carries the node-0
      // offset into the published z0 (and sets the propagation Jacobian's path length) is signed by the
      // direction of travel, so it flips with the charge. Off, it keeps the CCW turn angle's own sign.
      bool chargeSymmetric = false) {
    using curvilinearToPerigee::cross3;
    const double cx = fast_fit(0), cy = fast_fit(1), R = fast_fit(2);
    const double slope = -double(qCharge) / fast_fit(3);
    const double sec2 = 1. + slope * slope, invn = 1. / std::sqrt(sec2);
    const double pTot = bField * R * std::sqrt(sec2);
    const double cmag = std::sqrt(cx * cx + cy * cy);
    const double pcaX = cx * (1. - R / cmag), pcaY = cy * (1. - R / cmag), z0ref = hitZ0 - slope * sTransverse0;

    // reference FORWARD momentum direction T and curvilinear axes (U = Z x T, V = T x U) at the PCA (matches prepareGblData).
    const double rx = (pcaX - cx) / R, ry = (pcaY - cy) / R;
    const Vector3d T(double(qCharge) * ry * invn, -double(qCharge) * rx * invn, slope * invn);
    const Vector3d Z(0., 0., 1.);
    const Vector3d U = cross3(Z, T).normalized();
    const Vector3d V = cross3(T, U);
    const double phiRef = std::atan2(T.y(), T.x());
    const double lamRef = std::atan2(T.z(), std::sqrt(T.x() * T.x() + T.y() * T.y()));

    // apply the GBL corrections -> fitted momentum + position at the PCA.
    const double phiFit = phiRef + corr(2), lamFit = lamRef + corr(1);
    const double qbpFit = double(qCharge) / pTot + corr(0);
    const double pFit = 1. / std::abs(qbpFit);
    const double clam = std::cos(lamFit), slam = std::sin(lamFit);
    const Vector3d mom(pFit * clam * std::cos(phiFit), pFit * clam * std::sin(phiFit), pFit * slam);
    const Vector3d pos(pcaX + corr(3) * U.x() + corr(4) * V.x(),
                       pcaY + corr(3) * U.y() + corr(4) * V.y(),
                       z0ref + corr(3) * U.z() + corr(4) * V.z());

    // Propagate the fitted state from node 0 (the fast-fit PCA) to the FITTED helix's true transverse PCA and
    // take the perigee there. The GBL corrections, notably the y_T longitudinal offset whose V axis has a
    // transverse component for dipped tracks, move the evaluation point along the track off the true PCA,
    // which inflates d0/z0 if the perigee is taken at node 0.
    const double ptf = std::sqrt(mom.x() * mom.x() + mom.y() * mom.y());
    const double pmag = std::sqrt(ptf * ptf + mom.z() * mom.z());
    const double Rf = ptf / bField;                         // fitted transverse radius (>0)
    const double tnx = mom.x() / ptf, tny = mom.y() / ptf;  // unit transverse momentum
    // transverse circle center (perpendicular to the momentum, on the curving side set by the charge).
    const double Cx = pos.x() + Rf * double(qCharge) * tny, Cy = pos.y() - Rf * double(qCharge) * tnx;
    const double Cmag = std::sqrt(Cx * Cx + Cy * Cy);
    const double Px = Cx * (1. - Rf / Cmag), Py = Cy * (1. - Rf / Cmag);  // closest point on the circle to the origin
    if (nextRef) {
      const double slopeNext = mom.z() / ptf;  // cotTheta of the fitted track
      (*nextRef)(0) = Cx;
      (*nextRef)(1) = Cy;
      (*nextRef)(2) = Rf;
      (*nextRef)(3) = -double(qCharge) / slopeNext;  // prepareGblData reads slope = -qCharge/fast_fit(3)
    }
    // signed turn angle node 0 -> PCA (around the center); rotate the position-vector and the momentum by it.
    const double v0x = pos.x() - Cx, v0y = pos.y() - Cy, vPx = Px - Cx, vPy = Py - Cy;
    const double dAlpha = std::atan2(v0x * vPy - v0y * vPx, v0x * vPx + v0y * vPy);
    // dAlpha is the CCW turn angle of the radius vector about the fitted circle's centre. With the centre at
    // (pos + Rf*q*t_perp) the FORWARD transverse arc advances the radius azimuth at dpsi/ds = -q/Rf, so the
    // signed arc travelled is -q*Rf*dAlpha, the convention sTransverse itself carries. Only the ARC is
    // charge-signed: momPca below rotates the momentum rigidly through the CCW angle and is charge-free.
    const double dsT = (chargeSymmetric ? -double(qCharge) : 1.) * Rf * dAlpha,
                 ds = dsT * pmag / ptf;  // transverse and 3D path length (signed)
    const double ca = std::cos(dAlpha), sa = std::sin(dAlpha);
    const Vector3d momPca(mom.x() * ca - mom.y() * sa, mom.x() * sa + mom.y() * ca, mom.z());
    const Vector3d posPca(Px, Py, pos.z() + (mom.z() / ptf) * dsT);
    // propagate the curvilinear covariance node 0 -> PCA (reuses the point-to-point Jacobian).
    const double qbp = double(qCharge) / pmag;
    const Matrix5d Jprop =
        curvilinearJacobian(mom / pmag, momPca / pmag, pos - posPca, qbp, Vector3d(0., 0., bField), ds);
    const Matrix5d covPca = Jprop * cov * Jprop.transpose();

    // curvilinear -> perigee at the PCA (value + covariance).
    const Vector3d bvec(0., 0., bField);
    const Vector5d perigee = curvilinearToPerigee::perigeeParameters(posPca, momPca, qCharge, bField);
    const Matrix5d Jcp = curvilinearToPerigee::jacobian(momPca, bvec, qCharge);
    const Matrix5d perigeeCov = Jcp * covPca * Jcp.transpose();

    // perigee (kappa, theta, phi, epsilon, z) -> helix (phi, d0, k=1/R, cotTheta, z0): pure convention reorder.
    const double theta = perigee(1);
    const double cott = 1. / std::tan(theta), dcot_dtheta = -(1. + cott * cott);  // d(cotTheta)/d(theta)
    helixPar(0) = perigee(2);                                                     // phi
    helixPar(1) = perigee(3);                                                     // d0 = epsilon
    helixPar(2) = -perigee(0);      // k = 1/R = -kappa (matches circle fit's qCharge/R sign)
    helixPar(3) = cott;             // cotTheta
    helixPar(4) = perigee(4);       // z0
    Matrix5d M = Matrix5d::Zero();  // helix_i = sum_j M(i,j) perigee_j
    M(0, 2) = 1.;
    M(1, 3) = 1.;
    M(2, 0) = -1.;
    M(3, 1) = dcot_dtheta;
    M(4, 4) = 1.;
    helixCov = M * perigeeCov * M.transpose();
  }

}  // namespace generalBrokenLine

#endif  // RecoTracker_PixelTrackFitting_GeneralBrokenLine_h
