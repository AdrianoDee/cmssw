// Split-build TU: extended-N refit scans runRefitScanBin<N, Phase2OTStubs>, N=9..12 (stride kRefitStride).
// Each N instantiated once (ODR).
#include "BrokenLineFitKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // The fused ladder's per-bin fast-fit scans, same N-range (see BrokenLineFit_refitLo.dev.cc).
  BLFIT_REFIT_SCAN_SIG(9);
  BLFIT_REFIT_SCAN_SIG(10);
  BLFIT_REFIT_SCAN_SIG(11);
  BLFIT_REFIT_SCAN_SIG(12);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
