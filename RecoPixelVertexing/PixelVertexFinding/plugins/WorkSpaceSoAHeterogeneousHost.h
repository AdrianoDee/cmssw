#ifndef RecoPixelVertexing_PixelVertexFinding_WorkSpaceSoAHeterogeneousHost_h
#define RecoPixelVertexing_PixelVertexFinding_WorkSpaceSoAHeterogeneousHost_h

#include "CUDADataFormats/Common/interface/PortableHostCollection.h"
#include "CUDADataFormats/Vertex/interface/ZVertexUtilities.h"
#include "RecoPixelVertexing/PixelVertexFinding/plugins/WorkSpaceUtilities.h"

template <int32_t S>
class WorkSpaceSoAHeterogeneousHost : public cms::cuda::PortableHostCollection<WorkSpaceSoAHeterogeneousLayout<>> {
public:
  WorkSpaceSoAHeterogeneousHost() = default;

  // Constructor which specifies the SoA size and CUDA stream
  explicit WorkSpaceSoAHeterogeneousHost(cudaStream_t stream)
      : PortableHostCollection<WorkSpaceSoAHeterogeneousLayout<>>(S, stream) {}
};

namespace gpuVertexFinder {
  namespace workSpace {
    using WorkSpaceSoAHost = WorkSpaceSoAHeterogeneousHost<zVertex::utilities::MAXTRACKS>;
  }
}  // namespace gpuVertexFinder
#endif
