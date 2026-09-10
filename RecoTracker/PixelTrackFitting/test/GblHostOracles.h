#ifndef RecoTracker_PixelTrackFitting_test_GblHostOracles_h
#define RecoTracker_PixelTrackFitting_test_GblHostOracles_h

// Host-only oracles for the GBL device tests: the DESY General-Broken-Lines library (external `gbl`,
// Kleinwort) and CMSSW's curvilinear->perigee conversion. Defined in GblHostOracles.cc so the host
// compiler builds them: the DESY and CMSSW TrackingTools headers are host-only, while scram sends
// %.dev.cc to nvcc. This header pulls std types only.

#include <vector>

namespace gblHostOracles {

  // One GBL node, flattened; mirrors generalBrokenLine::GblNodeData. 5x5 and 2x2 blocks are row-major.
  struct Node {
    double jacToPrev[25] = {};    // curvilinear point-to-point Jacobian from the previous node (unused for node 0)
    double measPrec[4] = {};      // 2x2 measurement precision in the (U,V) offset frame
    double measResidual[2] = {};  // (U,V) residual: measured hit minus the reference-helix point
    double scatPrec[4] = {};      // 2x2 kink precision, (lambda, phi)
    bool hasMeas = false;
    bool hasScat = false;
  };

  //!< result of the DESY fit, read out at point 1 (== node 0, the PCA reference).
  struct DesyResult {
    double corr[5] = {};  // parameter corrections (q/p, lambda, phi, x_T, y_T)
    double cov[25] = {};  // their 5x5 covariance, row-major
    double chi2 = 0.;
    double lostWeight = 0.;
    int ndf = 0;
    bool ok = false;  // false = gbl::GblTrajectory::fit returned non-zero
  };

  // Run the DESY library over exactly this node chain: one gbl::GblPoint per node carrying its own
  // jacToPrev, its measurement (residual + 2x2 precision, no projection -- the measurement directions
  // are the curvilinear u1,u2) and its scatterer (zero kink residual + 2x2 precision). The trajectory
  // is built with flagCurv/flagU1dir/flagU2dir = true and read out at point 1. The node chain is an
  // input built by the device, so only the solver is compared.
  DesyResult desyFit(const std::vector<Node>& nodes);

  //!< sigma ratios ours/CMSSW for the five perigee parameters (1.0 == identical).
  struct PerigeeRatios {
    double d0 = 0., z0 = 0., phi = 0., theta = 0., ptRel = 0.;
    bool ok = false;
  };

  // Check of the gblHelixAtPca extraction against CMSSW FreeTrajectoryState -> PerigeeConversions, fed
  // the same GBL curvilinear covariance. ftsToPerigeeError gives the perigee at the FTS point (the
  // fast-fit PCA, node 0) while gblHelixAtPca adds the node0 -> true-PCA propagation, so for prompt
  // tracks the two coincide and every ratio is ~1.
  // \param fastFit (cx, cy, R, cotTheta-encoded 4th slot) -- the GBL reference helix.
  // \param corr / cov the GBL result at node 0; \param helixPar / helixCov what gblHelixAtPca produced.
  PerigeeRatios perigeeRatios(const double fastFit[4],
                              int qCharge,
                              double bField,
                              double sTransverse0,
                              double hitZ0,
                              const double corr[5],
                              const double cov[25],
                              const double helixPar[5],
                              const double helixCov[25]);

}  // namespace gblHostOracles

#endif
