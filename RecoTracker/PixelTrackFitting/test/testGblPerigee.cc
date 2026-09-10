// Checks that the transcribed curvilinear->perigee transformation (interface/CurvilinearToPerigee.h)
// reproduces CMSSW's PerigeeConversions, values and covariance, on tracks spanning eta, charge and pt.
// Returns non-zero on any mismatch.

#include <cmath>
#include <cstdio>
#include <Eigen/Core>

#include "RecoTracker/PixelTrackFitting/interface/CurvilinearToPerigee.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"

namespace {
  const double kBinTesla = 3.8;

  struct UniformBz : public MagneticField {
    explicit UniformBz(double bz) : b_(0., 0., bz) {}
    GlobalVector inTesla(const GlobalPoint&) const override { return b_; }
    GlobalVector b_;
  };

  bool runTrack(const char* label, double eta, double phi, double pt, int charge, const UniformBz& field) {
    const double theta = 2. * std::atan(std::exp(-eta));
    const Eigen::Vector3d mom(pt * std::cos(phi), pt * std::sin(phi), pt / std::tan(theta));
    const Eigen::Vector3d pos(0.012, -0.034, 4.7);  // arbitrary PCA-ish position (value path uses it; Jacobian doesn't)

    // a non-trivial symmetric positive curvilinear covariance
    Eigen::Matrix<double, 5, 5> curv;
    // clang-format off
    curv << 4.0e-6, 3.0e-7, 1.0e-7, 2.0e-7, 1.0e-7,
            3.0e-7, 5.0e-6, 4.0e-7, 1.5e-7, 2.0e-7,
            1.0e-7, 4.0e-7, 6.0e-6, 1.0e-7, 3.0e-7,
            2.0e-7, 1.5e-7, 1.0e-7, 7.0e-6, 5.0e-7,
            1.0e-7, 2.0e-7, 3.0e-7, 5.0e-7, 8.0e-6;
    // clang-format on

    GlobalTrajectoryParameters gtp(
        GlobalPoint(pos.x(), pos.y(), pos.z()), GlobalVector(mom.x(), mom.y(), mom.z()), charge, &field);
    AlgebraicSymMatrix55 curvSym;
    for (int a = 0; a < 5; ++a)
      for (int b = 0; b < 5; ++b)
        curvSym(a, b) = curv(a, b);
    FreeTrajectoryState fts(gtp, CurvilinearTrajectoryError(curvSym));

    double ptOut = 0.;
    auto perigeeOpt = PerigeeConversions::ftsToPerigeeParameters(fts, GlobalPoint(0., 0., 0.), ptOut);
    const AlgebraicVector5& refVal = perigeeOpt->vector();
    const AlgebraicSymMatrix55 refCov = PerigeeConversions::ftsToPerigeeError(fts).covarianceMatrix();

    // use the field CMSSW sees (inInverseGeV at the position) to avoid any constant drift
    const double bz = field.inInverseGeV(GlobalPoint(pos.x(), pos.y(), pos.z())).z();
    const Eigen::Vector3d bvec(0., 0., bz);
    const curvilinearToPerigee::Vector5d myVal = curvilinearToPerigee::perigeeParameters(pos, mom, charge, bz);
    const curvilinearToPerigee::Matrix5d jac = curvilinearToPerigee::jacobian(mom, bvec, charge);
    const curvilinearToPerigee::Matrix5d myCov = jac * curv * jac.transpose();

    double valMaxRel = 0., covMaxRel = 0., covScale = 0.;
    for (int a = 0; a < 5; ++a) {
      const double d = std::abs(myVal(a) - refVal[a]);
      const double s = std::max(1e-12, std::abs(refVal[a]));
      valMaxRel = std::max(valMaxRel, d / s);
      covScale = std::max(covScale, std::abs(refCov(a, a)));
    }
    for (int a = 0; a < 5; ++a)
      for (int b = 0; b < 5; ++b)
        if (std::abs(refCov(a, b)) > 1e-6 * covScale)
          covMaxRel = std::max(covMaxRel, std::abs(myCov(a, b) - refCov(a, b)) / std::abs(refCov(a, b)));

    // 1e-6 tolerance: CMSSW's jacobianCurvilinear2Perigee uses vdt::fast_sincos where this uses std::,
    // a ~1e-7 difference, so a correct transcription agrees to that level and not to machine precision.
    const bool ok = valMaxRel < 1e-6 && covMaxRel < 1e-6;
    std::printf("  %-22s eta=%+5.2f q=%+d  value maxRel=%.2e  cov maxRel=%.2e  %s\n",
                label,
                eta,
                charge,
                valMaxRel,
                covMaxRel,
                ok ? "OK" : "*** MISMATCH ***");
    return ok;
  }
}  // namespace

int main() {
  std::printf("\ncurvilinear->perigee transcription vs CMSSW PerigeeConversions:\n");
  const UniformBz field(kBinTesla);
  bool ok = true;
  ok &= runTrack("barrel", 0.0, 0.7, 100., 1, field);
  ok &= runTrack("barrel neg q", 0.0, 0.7, 100., -1, field);
  ok &= runTrack("eta1", 1.0, -1.2, 50., 1, field);
  ok &= runTrack("eta2 forward", 2.0, 2.4, 10., -1, field);
  ok &= runTrack("eta-1.5", -1.5, 0.1, 5., 1, field);
  ok &= runTrack("eta3 forward", 3.0, -2.8, 30., -1, field);
  ok &= runTrack("low pt barrel", 0.2, 1.5, 1., 1, field);
  std::printf("%s\n", ok ? "ALL OK" : "*** SOME MISMATCH ***");
  return ok ? 0 : 1;
}
