#ifndef RecoTracker_PixelTrackFitting_interface_alpaka_CurvilinearToPerigee_h
#define RecoTracker_PixelTrackFitting_interface_alpaka_CurvilinearToPerigee_h

// Device twin of interface/CurvilinearToPerigee.h (follows the CMSSW PerigeeConversions math). Identical math,
// std:: -> alpaka::math::, explicit vector ops (no Eigen .norm()/.normalized()/.cross() on device). See the host
// twin for the derivation and the perigee conventions.

#include <alpaka/alpaka.hpp>
#include <Eigen/Core>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::curvilinearToPerigee {

  using Matrix5d = Eigen::Matrix<double, 5, 5>;
  using Vector5d = Eigen::Matrix<double, 5, 1>;
  using Vector3d = Eigen::Vector3d;

  constexpr double kPi = 3.14159265358979323846;  // M_PI may be undefined in device compilation

  ALPAKA_FN_ACC ALPAKA_FN_INLINE Vector3d cross3(const Vector3d& a, const Vector3d& b) {
    return Vector3d(a.y() * b.z() - a.z() * b.y(), a.z() * b.x() - a.x() * b.z(), a.x() * b.y() - a.y() * b.x());
  }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE double dot3(const Vector3d& a, const Vector3d& b) {
    return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
  }
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE double norm3(const TAcc& acc, const Vector3d& v) {
    return alpaka::math::sqrt(acc, v.x() * v.x() + v.y() * v.y() + v.z() * v.z());
  }
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE Vector3d normalize3(const TAcc& acc, const Vector3d& v) {
    return v / norm3(acc, v);
  }

  // perigee value from the fitted global position + momentum at the PCA (follows ftsToPerigeeParameters,
  // referencePoint = origin). charge = track charge; bz = field z-component in inverse-GeV (= B[T]*2.99792458e-3).
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE Vector5d
  perigeeParameters(const TAcc& acc, const Vector3d& pos, const Vector3d& mom, int charge, double bz) {
    const double pt = alpaka::math::sqrt(acc, mom.x() * mom.x() + mom.y() * mom.y());
    const double theta = alpaka::math::atan2(acc, pt, mom.z());
    const double phi = alpaka::math::atan2(acc, mom.y(), mom.x());

    // signed transverse impact parameter (epsilon): sign from momentum-phi vs position-phi, magnitude = |xy| distance.
    const double positiveMomentumPhi = (phi > 0.) ? phi : (2. * kPi + phi);
    const double positionPhi = alpaka::math::atan2(acc, pos.y(), pos.x());
    const double positivePositionPhi = (positionPhi > 0.) ? positionPhi : (2. * kPi + positionPhi);
    double phiDiff = positiveMomentumPhi - positivePositionPhi;
    if (phiDiff < 0.)
      phiDiff += 2. * kPi;
    const double signEpsilon = (phiDiff > kPi) ? -1.0 : 1.0;
    const double epsilon = signEpsilon * alpaka::math::sqrt(acc, pos.x() * pos.x() + pos.y() * pos.y());

    const double signTC = -double(charge);
    const bool isCharged = (signTC != 0.) && (alpaka::math::abs(acc, bz) > 1.e-10);
    const double kappa = isCharged ? (bz / pt * signTC) : (1. / pt);

    Vector5d p;
    p << kappa, theta, phi, epsilon, pos.z();
    return p;
  }

  // curvilinear -> perigee 5x5 Jacobian (follows jacobianCurvilinear2Perigee). mom = global momentum at the
  // PCA, bfield = field vector in inverse-GeV, charge = track charge. Curvilinear cols (q/p, lambda, phi, x_T,
  // y_T); perigee rows (kappa, theta, phi, epsilon, z). cov_perigee = J * cov_curvilinear * J^T.
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE Matrix5d
  jacobian(const TAcc& acc, const Vector3d& mom, const Vector3d& bfield, int charge) {
    const Vector3d Z(0., 0., 1.);
    const Vector3d T = normalize3(acc, mom);
    const Vector3d U = normalize3(acc, cross3(Z, T));
    const Vector3d V = cross3(T, U);

    Vector3d I(-mom.x(), -mom.y(), 0.);  // opposite to track transverse dir
    I = normalize3(acc, I);
    const Vector3d J(-I.y(), I.x(), 0.);  // counterclockwise rotation
    const Vector3d& K = Z;

    const Vector3d H = normalize3(acc, bfield);
    const Vector3d HxT = cross3(H, T);
    const Vector3d N = normalize3(acc, HxT);
    const double alpha = norm3(acc, HxT);
    const double pmag = norm3(acc, mom);
    const double qbp = double(charge) / pmag;
    const double Q = -norm3(acc, bfield) * qbp;
    const double alphaQ = alpha * Q;

    const double pt = alpaka::math::sqrt(acc, mom.x() * mom.x() + mom.y() * mom.y());
    const double lambda = 0.5 * kPi - alpaka::math::atan2(acc, pt, mom.z());
    const double coslambda = alpaka::math::cos(acc, lambda), sinlambda = alpaka::math::sin(acc, lambda);
    const double seclambda = 1. / coslambda;

    const double ITI = 1. / dot3(T, I);
    const double NU = dot3(N, U), NV = dot3(N, V);
    const double UI = dot3(U, I), VI = dot3(V, I);
    const double UJ = dot3(U, J), VJ = dot3(V, J);
    const double UK = dot3(U, K), VK = dot3(V, K);

    // transverse curvature kappa = field/pt*signTC; here we use the same signed transverse curvature CMSSW does.
    const double transverseCurvature = -bfield.z() * seclambda * qbp;  // = field/pt*(-charge) up to sign bookkeeping

    Matrix5d jac = Matrix5d::Zero();
    if (alpaka::math::abs(acc, transverseCurvature) < 1.e-10) {
      jac(0, 0) = seclambda;
      jac(0, 1) = sinlambda * seclambda * seclambda * alpaka::math::abs(acc, qbp);
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

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::curvilinearToPerigee

#endif  // RecoTracker_PixelTrackFitting_interface_alpaka_CurvilinearToPerigee_h
