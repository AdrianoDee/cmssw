#ifndef CUDADataFormats_Vertex_ZVertexHeterogeneousHost_H
#define CUDADataFormats_Vertex_ZVertexHeterogeneousHost_H

#include <cstdint>

#include "CUDADataFormats/Vertex/interface/ZVertexUtilities.h"
#include "CUDADataFormats/Common/interface/PortableHostCollection.h"

template <int32_t S>
class ZVertexSoAHeterogeneousHost : public cms::cuda::PortableHostCollection<ZVertexSoAHeterogeneousLayout<>> {
public:
  ZVertexSoAHeterogeneousHost() = default;  // Required for cms::cuda::Product

  // Constructor which specifies the SoA size and CUDA stream
  explicit ZVertexSoAHeterogeneousHost(cudaStream_t stream)
      : PortableHostCollection<ZVertexSoAHeterogeneousLayout<>>(S, stream) {}
};

namespace zVertex {

  using ZVertexSoAHost = ZVertexSoAHeterogeneousHost<zVertex::utilities::MAXTRACKS>;

}  // namespace zVertex

#endif  // CUDADataFormats_Vertex_ZVertexHeterogeneousHost_H
