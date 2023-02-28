#ifndef DataFormats_Vertex_ZVertexUtilities_h
#define DataFormats_Vertex_ZVertexUtilities_h

#include "DataFormats/Vertex/interface/ZVertexLayout.h"
#include "DataFormats/Vertex/interface/ZVertexDefinitions.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  // Previous ZVertexSoA class methods.
  // They operate on View and ConstView of the ZVertexSoA.
  namespace zVertex {
    namespace utilities {
      using ZVertexSoALayout = ZVertexSoAHeterogeneousLayout<>;
      using ZVertexSoAView = ZVertexSoAHeterogeneousLayout<>::View;
      using ZVertexSoAConstView = ZVertexSoAHeterogeneousLayout<>::ConstView;

      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void init(ZVertexSoAView &vertices) { vertices.nvFinal() = 0; }

    }  // namespace utilities
  }    // namespace zVertex
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
