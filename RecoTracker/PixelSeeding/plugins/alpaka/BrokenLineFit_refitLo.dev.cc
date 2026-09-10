// Split-build TU: extended-N refit scans runRefitScanBin<N, Phase2OTStubs>, N = 3..8 (stride
// kRefitStride); each N is instantiated in exactly one TU.
#include "BrokenLineFitKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // The fused ladder's per-bin fast-fit scans, N = 3..8.
  BLFIT_REFIT_SCAN_SIG(3);
  BLFIT_REFIT_SCAN_SIG(4);
  BLFIT_REFIT_SCAN_SIG(5);
  BLFIT_REFIT_SCAN_SIG(6);
  BLFIT_REFIT_SCAN_SIG(7);
  BLFIT_REFIT_SCAN_SIG(8);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
