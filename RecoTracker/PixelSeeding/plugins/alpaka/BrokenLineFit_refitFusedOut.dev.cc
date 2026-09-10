// Split-build TU: fused extended-N refit, Out phase. One phase per TU, since a fused kernel pulls all
// ten N and separating the phases bounds each nvcc partition.
#include "BrokenLineFitKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  BLFIT_REFIT_FUSED_OUT_SIG();

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
