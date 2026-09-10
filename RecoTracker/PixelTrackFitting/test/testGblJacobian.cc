// Checks that the transcribed curvilinear Jacobian (generalBrokenLine::curvilinearJacobian) reproduces
// CMSSW's AnalyticalCurvilinearJacobian on the same inputs across pt / eta / charge / step length,
// including the long-step branch at low pt. Returns non-zero on any mismatch.

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <Eigen/Core>

#include "RecoTracker/PixelTrackFitting/interface/GeneralBrokenLine.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/AnalyticalJacobians/interface/AnalyticalCurvilinearJacobian.h"

namespace {

  struct UniformBz : public MagneticField {
    explicit UniformBz(double bz) : b_(0., 0., bz) {}
    GlobalVector inTesla(const GlobalPoint&) const override { return b_; }
    GlobalVector b_;
  };

  GlobalVector momFrom(double pt, double phi, double eta) {
    return GlobalVector(pt * std::cos(phi), pt * std::sin(phi), pt * std::sinh(eta));
  }

  int checkCase(const char* label,
                const UniformBz& field,
                const GlobalPoint& x1,
                double pt,
                double phi,
                double eta,
                int charge,
                double s) {
    const GlobalVector p1 = momFrom(pt, phi, eta);
    const double p = p1.mag();
    const double Bz = field.inTesla(x1).z();

    // roughly-physical end state: rotate the transverse momentum by the bend, take a chord for the
    // position. Both functions receive the same derived inputs, so any end state would be a valid check.
    const double cosl = p1.perp() / p;
    const double kappa = -charge * 0.0029979 * Bz / p1.perp();  // signed transverse curvature [1/cm]
    const double phi2 = phi + kappa * s * cosl;
    const GlobalVector p2(p1.perp() * std::cos(phi2), p1.perp() * std::sin(phi2), p1.z());
    const GlobalVector midDir = (p1.unit() + p2.unit()).unit();
    const GlobalPoint x2(x1.x() + s * midDir.x(), x1.y() + s * midDir.y(), x1.z() + s * midDir.z());

    const GlobalTrajectoryParameters gtp(x1, p1, charge, &field);
    const GlobalVector hInvGeV = gtp.magneticFieldInInverseGeV();
    const AnalyticalCurvilinearJacobian acj(gtp, x2, p2, hInvGeV, s);

    const Eigen::Vector3d e_p1(p1.unit().x(), p1.unit().y(), p1.unit().z());
    const Eigen::Vector3d e_p2(p2.unit().x(), p2.unit().y(), p2.unit().z());
    const Eigen::Vector3d e_dx(x1.x() - x2.x(), x1.y() - x2.y(), x1.z() - x2.z());
    const Eigen::Vector3d e_h(hInvGeV.x(), hInvGeV.y(), hInvGeV.z());
    const double qbp = double(charge) / p;
    const Eigen::Matrix<double, 5, 5> Jmine = generalBrokenLine::curvilinearJacobian(e_p1, e_p2, e_dx, qbp, e_h, s);

    // CMSSW's computeFullJacobian uses vdt::fast_sincos where this transcription uses std::sin/cos, so
    // the two agree only to the trig-approximation level (~1e-7). The relative difference grows on tiny
    // elements with 1-cos cancellation at high pt, hence the mixed atol+rtol test and the separate
    // relative figure restricted to meaningful (|ref|>1e-3) elements.
    constexpr double atol = 1e-7, rtol = 1e-4;
    double maxAbs = 0., relBig = 0.;
    bool ok = true;
    for (int a = 0; a < 5; ++a)
      for (int b = 0; b < 5; ++b) {
        const double ref = acj.jacobian()(a, b);
        const double d = std::abs(Jmine(a, b) - ref);
        maxAbs = std::max(maxAbs, d);
        if (d > atol + rtol * std::abs(ref))
          ok = false;
        if (std::abs(ref) > 1e-3)
          relBig = std::max(relBig, d / std::abs(ref));
      }
    std::printf("  %-18s pt=%6.1f eta=%5.2f q=%+d s=%6.1f | maxAbs=%.2e relBig=%.2e  %s\n",
                label,
                pt,
                eta,
                charge,
                s,
                maxAbs,
                relBig,
                ok ? "OK" : "*** MISMATCH ***");
    return ok ? 0 : 1;
  }

}  // namespace

int main() {
  const UniformBz field(3.8);
  int bad = 0;
  std::printf("Device curvilinearJacobian vs CMSSW AnalyticalCurvilinearJacobian (uniform 3.8 T):\n");
  bad += checkCase("central highpt", field, GlobalPoint(3.7, 0.2, 0.3), 98.8, 1.6, -0.36, -1, 12.);
  bad += checkCase("barrel eta0", field, GlobalPoint(2.4, -1.6, 0.4), 108., -0.58, 0.01, +1, 10.);
  bad += checkCase("barrel eta0.4", field, GlobalPoint(-1.9, 2.0, 2.0), 138., 2.34, -0.41, -1, 9.);
  bad += checkCase("forward", field, GlobalPoint(3.3, 0.02, -41.), 41., 3.13, -3.20, -1, 25.);
  bad += checkCase("lowpt short", field, GlobalPoint(2.0, 0.5, 0.0), 2.0, 0.2, 0.0, +1, 8.);
  bad += checkCase("lowpt long(cut)", field, GlobalPoint(2.0, 0.5, 0.0), 1.0, 0.2, 0.0, -1, 30.);
  bad += checkCase("highpt long(cut)", field, GlobalPoint(3.7, 0.2, 0.3), 100., 1.6, 0.0, +1, 600.);
  std::printf(bad ? "\nFAILED: %d case(s) mismatched\n" : "\nALL OK\n", bad);
  return bad;
}
