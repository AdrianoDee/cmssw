// Fused extended-N refit, Prep phase. One phase per translation unit: a fused kernel pulls in all
// ten N, so splitting by phase bounds each nvcc partition.
#include "BrokenLineFitKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  BLFIT_REFIT_FUSED_PREP_SIG();

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
