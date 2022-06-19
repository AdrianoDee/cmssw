#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"


//GPU storage copy
//FIXME use std::is_same_v<Traits, cms::cudacompat::HostTraits>
template <>
cms::cuda::host::unique_ptr<float[]> TrackingRecHit2DGPU::localCoordToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<float[]>(5 * nHits(), stream);
  cms::cuda::copyAsync(ret, m_store32, 5 * nHits(), stream);
  return ret;
}

template <>
cms::cuda::host::unique_ptr<float[]> TrackingRecHit2DGPU::store32ToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<float[]>(static_cast<int>(n32) * nHits(), stream);
  cms::cuda::copyAsync(ret, m_store32, static_cast<int>(n32) * nHits(), stream);
  return ret;
}

template <>
cms::cuda::host::unique_ptr<uint16_t[]> TrackingRecHit2DGPU::store16ToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint16_t[]>(static_cast<int>(n16) * nHits(), stream);
  cms::cuda::copyAsync(ret, m_store16, static_cast<int>(n16) * nHits(), stream);
  return ret;
}

template <>
cms::cuda::host::unique_ptr<uint32_t[]> TrackingRecHit2DGPU::hitsModuleStartToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint32_t[]>(nMaxModules() + 1, stream);
  cudaCheck(
      cudaMemcpyAsync(ret.get(), m_hitsModuleStart, sizeof(uint32_t) * (nMaxModules() + 1), cudaMemcpyDefault, stream));
  return ret;
}


//CPU storage copy
template <>
cms::cuda::host::unique_ptr<float[]> TrackingRecHit2DCPU::store32ToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<float[]>(static_cast<int>(n32) * nHits(), stream);
  std::copy(m_store32.get(), m_store32.get() + static_cast<int>(nHits()) * static_cast<int>(n32), ret.get());
  return ret;
}

template <>
cms::cuda::host::unique_ptr<uint16_t[]> TrackingRecHit2DCPU::store16ToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint16_t[]>(static_cast<int>(n16) * nHits(), stream);
  std::copy(m_store16.get(), m_store16.get() + static_cast<int>(nHits()) * static_cast<int>(n16), ret.get());
  return ret;
}

template <>
cms::cuda::host::unique_ptr<uint32_t[]> TrackingRecHit2DCPU::hitsModuleStartToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint32_t[]>(nMaxModules() + 1, stream);
  std::copy(m_hitsModuleStart, m_hitsModuleStart + static_cast<int>(nMaxModules() + 1), ret.get());
  return ret;
}


// the only specialization needed
template <>
void TrackingRecHit2DHost::copyFromGPU(TrackingRecHit2DGPU const* input, cudaStream_t stream) {
  assert(input);
  m_store32 = input->localCoordToHostAsync(stream);
}
