#include "SiPixelRawToClusterKernelImpl.h"

//#define GPU_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace pixelDetails {
    template class SiPixelRawToClusterKernel<pixelTopology::HIonPhase1>;
  }  // namespace pixelDetails

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
