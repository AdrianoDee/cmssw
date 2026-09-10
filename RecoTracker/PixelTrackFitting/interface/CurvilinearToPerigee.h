#ifndef RecoTracker_PixelTrackFitting_CurvilinearToPerigee_h
#define RecoTracker_PixelTrackFitting_CurvilinearToPerigee_h

// Curvilinear -> perigee parameter+covariance transformation, following
// TrackingTools/TrajectoryState/src/PerigeeConversions.cc (ftsToPerigeeParameters,
// jacobianCurvilinear2Perigee): the GBL fit returns the 5-D curvilinear result
// (q/p, lambda, phi, x_T, y_T) at the PCA, the stored track parameters are perigee-like.
//
// CMSSW perigee vector order: [0]=signed transverse curvature, [1]=theta, [2]=phi, [3]=transverse IP (epsilon),
// [4]=longitudinal IP (z). Reference point = the IP (origin) for the beamline PCA.

#include <cmath>
#include <Eigen/Core>

namespace curvilinearToPerigee {

  using Matrix5d = Eigen::Matrix<double, 5, 5>;
  using Vector5d = Eigen::Matrix<double, 5, 1>;
  using Vector3d = Eigen::Vector3d;

  // explicit 3-vector cross product (Eigen's .cross() needs <Eigen/Geometry>; this keeps the header device-portable).
  inline Vector3d cross3(const Vector3d& a, const Vector3d& b) {
    return Vector3d(a.y() * b.z() - a.z() * b.y(), a.z() * b.x() - a.x() * b.z(), a.x() * b.y() - a.y() * b.x());
  }

  // Perigee parameters from the fitted global position and momentum at the PCA, referencePoint = origin.
  // bz = field z-component in inverse-GeV (= B[T]*2.99792458e-3).
  inline Vector5d perigeeParameters(const Vector3d& pos, const Vector3d& mom, int charge, double bz) {
    const double pt = std::sqrt(mom.x() * mom.x() + mom.y() * mom.y());
    const double theta = std::atan2(pt, mom.z());
    const double phi = std::atan2(mom.y(), mom.x());

    // signed transverse impact parameter (epsilon): sign from momentum-phi vs position-phi, magnitude = |xy| distance.
    const double positiveMomentumPhi = (phi > 0.) ? phi : (2. * M_PI + phi);
    const double positionPhi = std::atan2(pos.y(), pos.x());
    const double positivePositionPhi = (positionPhi > 0.) ? positionPhi : (2. * M_PI + positionPhi);
    double phiDiff = positiveMomentumPhi - positivePositionPhi;
    if (phiDiff < 0.)
      phiDiff += 2. * M_PI;
    const double signEpsilon = (phiDiff > M_PI) ? -1.0 : 1.0;
    const double epsilon = signEpsilon * std::sqrt(pos.x() * pos.x() + pos.y() * pos.y());

    const double signTC = -double(charge);
    const bool isCharged = (signTC != 0.) && (std::abs(bz) > 1.e-10);
    const double kappa = isCharged ? (bz / pt * signTC) : (1. / pt);

    Vector5d p;
    p << kappa, theta, phi, epsilon, pos.z();
    return p;
  }

  // Curvilinear -> perigee 5x5 Jacobian: cov_perigee = J * cov_curvilinear * J^T. mom = global momentum at
  // the PCA, bfield = field vector in inverse-GeV. Columns (q/p, lambda, phi, x_T, y_T), rows
  // (kappa, theta, phi, epsilon, z).
  inline Matrix5d jacobian(const Vector3d& mom, const Vector3d& bfield, int charge) {
    const Vector3d Z(0., 0., 1.);
    const Vector3d T = mom.normalized();
    const Vector3d U = cross3(Z, T).normalized();
    const Vector3d V = cross3(T, U);

    Vector3d I(-mom.x(), -mom.y(), 0.);  // opposite to track transverse dir
    I.normalize();
    const Vector3d J(-I.y(), I.x(), 0.);  // counterclockwise rotation
    const Vector3d& K = Z;

    const Vector3d H = bfield.normalized();
    const Vector3d HxT = cross3(H, T);
    const Vector3d N = HxT.normalized();
    const double alpha = HxT.norm();
    const double pmag = mom.norm();
    const double qbp = double(charge) / pmag;  // signed inverse momentum
    const double Q = -bfield.norm() * qbp;
    const double alphaQ = alpha * Q;

    const double lambda = 0.5 * M_PI - std::atan2(std::sqrt(mom.x() * mom.x() + mom.y() * mom.y()), mom.z());
    const double coslambda = std::cos(lambda), sinlambda = std::sin(lambda);
    const double seclambda = 1. / coslambda;

    const double ITI = 1. / T.dot(I);
    const double NU = N.dot(U), NV = N.dot(V);
    const double UI = U.dot(I), VI = V.dot(I);
    const double UJ = U.dot(J), VJ = V.dot(J);
    const double UK = U.dot(K), VK = V.dot(K);

    const double transverseCurvature = -bfield.z() * seclambda * qbp;  // signed: field/pt*(-charge)

    Matrix5d jac = Matrix5d::Zero();
    if (std::abs(transverseCurvature) < 1.e-10) {
      jac(0, 0) = seclambda;
      jac(0, 1) = sinlambda * seclambda * seclambda * std::abs(qbp);
    } else {
      const double Bz = bfield.z();
      jac(0, 0) = -Bz * seclambda;
      jac(0, 1) = -Bz * sinlambda * seclambda * seclambda * qbp;
      jac(1, 3) = alphaQ * NV * UI * ITI;
      jac(1, 4) = alphaQ * NV * VI * ITI;
      jac(0, 3) = -jac(0, 1) * jac(1, 3);
      jac(0, 4) = -jac(0, 1) * jac(1, 4);
      jac(2, 3) = -alphaQ * seclambda * NU * UI * ITI;
      jac(2, 4) = -alphaQ * seclambda * NU * VI * ITI;
    }
    jac(1, 1) = -1.;
    jac(2, 2) = 1.;
    jac(3, 3) = VK * ITI;
    jac(3, 4) = -UK * ITI;
    jac(4, 3) = -VJ * ITI;
    jac(4, 4) = UJ * ITI;
    return jac;
  }

}  // namespace curvilinearToPerigee

#endif  // RecoTracker_PixelTrackFitting_CurvilinearToPerigee_h
