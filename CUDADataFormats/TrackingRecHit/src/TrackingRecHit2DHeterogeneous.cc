#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

template <typename TrackerTraits>
cms::cuda::host::unique_ptr<float[]> TrackingRecHit2DGPUT<TrackerTraits>::localCoordToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<float[]>(5 * this->nHits(), stream);
  cms::cuda::copyAsync(ret, this->m_store32, 5 * this->nHits(), stream);
  return ret;
}

template <typename TrackerTraits>
cms::cuda::host::unique_ptr<uint32_t[]> TrackingRecHit2DGPUT<TrackerTraits>::hitsModuleStartToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint32_t[]>(TrackerTraits::numberOfModules + 1, stream);
  cudaCheck(
      cudaMemcpyAsync(ret.get(), this->m_hitsModuleStart, sizeof(uint32_t) * (TrackerTraits::numberOfModules + 1), cudaMemcpyDefault, stream));
  return ret;
}

// the only specialization needed
template <typename TrackerTraits>
void TrackingRecHit2DHostT<TrackerTraits>::copyFromGPU(TrackingRecHit2DGPUT<TrackerTraits> const* input, cudaStream_t stream) {
  assert(input);
  this->m_store32 = input->localCoordToHostAsync(stream);
}

template class TrackingRecHit2DCPUBaseT<cms::cudacompat::CPUTraits,pixelTopology::Phase1>;
template class TrackingRecHit2DCPUBaseT<cms::cudacompat::CPUTraits,pixelTopology::Phase2>;

template class TrackingRecHit2DGPUBaseT<cms::cudacompat::GPUTraits,pixelTopology::Phase1>;
template class TrackingRecHit2DGPUBaseT<cms::cudacompat::GPUTraits,pixelTopology::Phase2>;
