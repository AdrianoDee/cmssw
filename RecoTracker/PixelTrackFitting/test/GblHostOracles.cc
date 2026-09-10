// Host-only implementation of the oracles declared in GblHostOracles.h.

#include <cmath>

#include <Eigen/Core>
#include <Eigen/Dense>

// External General-Broken-Lines library (DESY, Kleinwort). Eigen-only core, headers included flat.
#include "GblPoint.h"
#include "GblTrajectory.h"

// Reference curvilinear -> perigee conversion for gblHelixAtPca.
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"

#include "RecoTracker/PixelTrackFitting/test/GblHostOracles.h"

namespace {
  // Uniform solenoid along z, the field the fixtures were generated in.
  struct UniformBz : public MagneticField {
    explicit UniformBz(double bz) : b_(0., 0., bz) {}
    GlobalVector inTesla(const GlobalPoint&) const override { return b_; }
    GlobalVector b_;
  };

  Eigen::Matrix<double, 5, 5> mat5(const double* p) {
    Eigen::Matrix<double, 5, 5> m;
    for (int a = 0; a < 5; ++a)
      for (int b = 0; b < 5; ++b)
        m(a, b) = p[a * 5 + b];
    return m;
  }
  Eigen::Matrix2d mat2(const double* p) {
    Eigen::Matrix2d m;
    m << p[0], p[1], p[2], p[3];
    return m;
  }
}  // namespace

namespace gblHostOracles {

  DesyResult desyFit(const std::vector<Node>& nodes) {
    DesyResult r;
    const int nN = int(nodes.size());
    if (nN < 2)
      return r;

    std::vector<gbl::GblPoint> points;
    points.reserve(nN);
    for (int k = 0; k < nN; ++k) {
      // Node 0 is the reference: its jacToPrev is meaningless and DESY wants the identity there.
      // Built in place; the vector is reserved above, so no reallocation moves a point while it is filled.
      points.emplace_back(k == 0 ? Eigen::Matrix<double, 5, 5>::Identity() : mat5(nodes[k].jacToPrev));
      gbl::GblPoint& p = points.back();
      if (nodes[k].hasMeas) {
        const Eigen::Vector2d res(nodes[k].measResidual[0], nodes[k].measResidual[1]);
        const Eigen::Matrix2d prec = mat2(nodes[k].measPrec);
        p.addMeasurement(res, prec);
      }
      if (nodes[k].hasScat) {
        const Eigen::Vector2d kink = Eigen::Vector2d::Zero();
        const Eigen::Matrix2d sp = mat2(nodes[k].scatPrec);
        p.addScatterer(kink, sp);
      }
    }

    gbl::GblTrajectory traj(points, /*flagCurv=*/true, /*flagU1dir=*/true, /*flagU2dir=*/true);
    double chi2 = 0., lostWeight = 0.;
    int ndf = 0;
    if (traj.fit(chi2, ndf, lostWeight) != 0)
      return r;

    Eigen::VectorXd corr(5);
    Eigen::MatrixXd cov(5, 5);
    traj.getResults(1, corr, cov);  // point 1 (1-based) == node 0, the PCA reference
    for (int a = 0; a < 5; ++a) {
      r.corr[a] = corr(a);
      for (int b = 0; b < 5; ++b)
        r.cov[a * 5 + b] = cov(a, b);
    }
    r.chi2 = chi2;
    r.ndf = ndf;
    r.lostWeight = lostWeight;
    r.ok = true;
    return r;
  }

  PerigeeRatios perigeeRatios(const double fastFit[4],
                              int qCharge,
                              double bField,
                              double sTransverse0,
                              double hitZ0,
                              const double corr[5],
                              const double cov[25],
                              const double helixPar[5],
                              const double helixCov[25]) {
    PerigeeRatios out;
    // Rebuild the fitted state at the PCA exactly as gblHelixAtPca does, then hand it to CMSSW.
    const double cx = fastFit[0], cy = fastFit[1], R = fastFit[2];
    const double slope = -double(qCharge) / fastFit[3];
    const double sec2 = 1. + slope * slope, invn = 1. / std::sqrt(sec2);
    const double pTot = bField * R * std::sqrt(sec2);
    const double cmag = std::sqrt(cx * cx + cy * cy);
    const double pcaX = cx * (1. - R / cmag), pcaY = cy * (1. - R / cmag);
    const double z0r = hitZ0 - slope * sTransverse0;
    const double rx = (pcaX - cx) / R, ry = (pcaY - cy) / R;
    const Eigen::Vector3d Tv(double(qCharge) * ry * invn, -double(qCharge) * rx * invn, slope * invn);
    const Eigen::Vector3d U = Eigen::Vector3d(0., 0., 1.).cross(Tv).normalized();
    const Eigen::Vector3d Vv = Tv.cross(U);
    const double phiRef = std::atan2(Tv.y(), Tv.x());
    const double lamRef = std::atan2(Tv.z(), std::hypot(Tv.x(), Tv.y()));
    const double phiFit = phiRef + corr[2], lamFit = lamRef + corr[1];
    const double qbpFit = double(qCharge) / pTot + corr[0];
    if (!(std::abs(qbpFit) > 0.))
      return out;
    const double pFit = 1. / std::abs(qbpFit);
    const double clam = std::cos(lamFit), slam = std::sin(lamFit);
    const Eigen::Vector3d mom(pFit * clam * std::cos(phiFit), pFit * clam * std::sin(phiFit), pFit * slam);
    const Eigen::Vector3d pos(pcaX + corr[3] * U.x() + corr[4] * Vv.x(),
                              pcaY + corr[3] * U.y() + corr[4] * Vv.y(),
                              z0r + corr[3] * U.z() + corr[4] * Vv.z());

    const UniformBz fld(3.8);
    const GlobalTrajectoryParameters gtp(
        GlobalPoint(pos.x(), pos.y(), pos.z()), GlobalVector(mom.x(), mom.y(), mom.z()), qCharge, &fld);
    AlgebraicSymMatrix55 curvSym;
    for (int a = 0; a < 5; ++a)
      for (int b = 0; b < 5; ++b)
        curvSym(a, b) = cov[a * 5 + b];
    const FreeTrajectoryState fts(gtp, CurvilinearTrajectoryError(curvSym));
    double ptOut = 0.;
    const auto pp = PerigeeConversions::ftsToPerigeeParameters(fts, GlobalPoint(0., 0., 0.), ptOut);
    const double rho = pp->vector()(0);  // signed transverse curvature ~ 1/pt
    const AlgebraicSymMatrix55 rc = PerigeeConversions::ftsToPerigeeError(fts).covarianceMatrix();
    // CMSSW perigee order (0..4) = (curvature, theta, phi, d0, z0);
    // ours (helixPar/helixCov, 0..4)  = (phi, d0, k = 1/R, cotTheta, z0).
    const double cmD0 = std::sqrt(rc(3, 3)), cmZ0 = std::sqrt(rc(4, 4)), cmPhi = std::sqrt(rc(2, 2));
    const double cmTh = std::sqrt(rc(1, 1)), cmPtRel = std::sqrt(rc(0, 0)) / std::abs(rho);
    const double cott = helixPar[3];
    const double ourD0 = std::sqrt(helixCov[1 * 5 + 1]), ourZ0 = std::sqrt(helixCov[4 * 5 + 4]);
    const double ourPhi = std::sqrt(helixCov[0]);
    const double ourTh = std::sqrt(helixCov[3 * 5 + 3]) / (1. + cott * cott);  // sigma(cotTheta) -> sigma(theta)
    const double ourPtRel = std::sqrt(helixCov[2 * 5 + 2]) / std::abs(helixPar[2]);
    if (!(cmD0 > 0. && cmZ0 > 0. && cmPhi > 0. && cmTh > 0. && cmPtRel > 0.))
      return out;
    out.d0 = ourD0 / cmD0;
    out.z0 = ourZ0 / cmZ0;
    out.phi = ourPhi / cmPhi;
    out.theta = ourTh / cmTh;
    out.ptRel = ourPtRel / cmPtRel;
    out.ok = true;
    return out;
  }

}  // namespace gblHostOracles
