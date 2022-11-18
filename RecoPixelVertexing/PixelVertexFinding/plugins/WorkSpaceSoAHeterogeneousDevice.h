#ifndef RecoPixelVertexing_PixelVertexFinding_WorkSpaceSoAHeterogeneousDevice_h
#define RecoPixelVertexing_PixelVertexFinding_WorkSpaceSoAHeterogeneousDevice_h

#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"
#include "CUDADataFormats/Vertex/interface/ZVertexUtilities.h"
#include "RecoPixelVertexing/PixelVertexFinding/plugins/WorkSpaceUtilities.h"

template <int32_t S>
class WorkSpaceSoAHeterogeneousDevice : public cms::cuda::PortableDeviceCollection<WorkSpaceSoAHeterogeneousLayout<>> {
public:
  WorkSpaceSoAHeterogeneousDevice() = default;

  // Constructor which specifies the SoA size and CUDA stream
  explicit WorkSpaceSoAHeterogeneousDevice(cudaStream_t stream)
      : PortableDeviceCollection<WorkSpaceSoAHeterogeneousLayout<>>(S, stream) {}
};

namespace gpuVertexFinder {
  namespace workSpace {
    using WorkSpaceSoADevice = WorkSpaceSoAHeterogeneousDevice<zVertex::utilities::MAXTRACKS>;
  }
}  // namespace gpuVertexFinder
#endif
