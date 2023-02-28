#ifndef RecoPixelVertexing_PixelVertexFinding_plugins_alpaka_PixelVertexWorkSpaceUtilities_h
#define RecoPixelVertexing_PixelVertexFinding_plugins_alpaka_PixelVertexWorkSpaceUtilities_h

#include <alpaka/alpaka.hpp>
#include "../PixelVertexWorkSpaceLayout.h"

// Methods that operate on View and ConstView of the PixelVertexWorkSpaceSoALayout.
namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace gpuVertexFinder {
    namespace workSpace {
      namespace utilities {
        using namespace ::gpuVertexFinder::workSpace;

        ALPAKA_FN_ACC ALPAKA_FN_INLINE void init(PixelVertexWorkSpaceSoAView &workspace_view) {
          workspace_view.ntrks() = 0;
          workspace_view.nvIntermediate() = 0;
        }
      }  // namespace utilities
    }    // namespace workSpace
  }      // namespace gpuVertexFinder
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
#endif
