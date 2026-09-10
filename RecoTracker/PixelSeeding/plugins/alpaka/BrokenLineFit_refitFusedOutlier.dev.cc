// One refit phase per translation unit: a fused kernel instantiates all ten N, so the split bounds each
// nvcc partition.
#include "BrokenLineFitKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  BLFIT_REFIT_FUSED_OUTLIER_SIG();

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
