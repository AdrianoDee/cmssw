#ifndef CUDADataFormats_Vertex_ZVertexHeterogeneousDevice_H
#define CUDADataFormats_Vertex_ZVertexHeterogeneousDevice_H

#include <cstdint>

#include "CUDADataFormats/Vertex/interface/ZVertexUtilities.h"
#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

template <int32_t S>
class ZVertexSoAHeterogeneousDevice : public cms::cuda::PortableDeviceCollection<ZVertexSoAHeterogeneousLayout<>> {
public:
  ZVertexSoAHeterogeneousDevice() = default;  // cms::cuda::Product needs this

  // Constructor which specifies the SoA size
  explicit ZVertexSoAHeterogeneousDevice(cudaStream_t stream)
      : PortableDeviceCollection<ZVertexSoAHeterogeneousLayout<>>(S, stream) {}
};

namespace zVertex {

  using ZVertexSoADevice = ZVertexSoAHeterogeneousDevice<zVertex::utilities::MAXTRACKS>;

}  // namespace zVertex

#endif  // CUDADataFormats_Vertex_ZVertexHeterogeneousDevice_H
