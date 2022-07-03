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
cms::cuda::host::unique_ptr<float[]> TrackingRecHit2DGPUT<TrackerTraits>::store32ToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<float[]>(static_cast<int>(this->n32) * this->nHits(), stream);
  cms::cuda::copyAsync(ret, this->m_store32, static_cast<int>(this->n32) * this->nHits(), stream);
  return ret;
}

template <typename TrackerTraits>
cms::cuda::host::unique_ptr<uint16_t[]> TrackingRecHit2DGPUT<TrackerTraits>::store16ToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint16_t[]>(static_cast<int>(this->n16) * this->nHits(), stream);
  cms::cuda::copyAsync(ret, this->m_store16, static_cast<int>(this->n16) * this->nHits(), stream);
  return ret;
}


template <typename TrackerTraits>
cms::cuda::host::unique_ptr<uint32_t[]> TrackingRecHit2DGPUT<TrackerTraits>::hitsModuleStartToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint32_t[]>(TrackerTraits::numberOfModules + 1, stream);
  cudaCheck(cudaMemcpyAsync(ret.get(), this->m_hitsModuleStart, sizeof(uint32_t) * (TrackerTraits::numberOfModules + 1), cudaMemcpyDefault, stream));
  return ret;
}

// the only specialization needed
template <typename TrackerTraits>
void TrackingRecHit2DHostT<TrackerTraits>::copyFromGPU(TrackingRecHit2DGPUT<TrackerTraits> const* input, cudaStream_t stream) {
  assert(input);
  this->m_store32 = input->localCoordToHostAsync(stream);
}

//
// template <>
// cms::cuda::host::unique_ptr<float[]> TrackingRecHit2DGPUT<pixelTopology::Phase1>::localCoordToHostAsync(cudaStream_t stream) const {
//   auto ret = cms::cuda::make_host_unique<float[]>(5 * this->nHits(), stream);
//   cms::cuda::copyAsync(ret, this->m_store32, 5 * this->nHits(), stream);
//   return ret;
// }
//
// template <>
// cms::cuda::host::unique_ptr<float[]> TrackingRecHit2DGPUT<pixelTopology::Phase2>::localCoordToHostAsync(cudaStream_t stream) const {
//   auto ret = cms::cuda::make_host_unique<float[]>(5 * this->nHits(), stream);
//   cms::cuda::copyAsync(ret, this->m_store32, 5 * this->nHits(), stream);
//   return ret;
// }
//
// template <>
// void TrackingRecHit2DHostT<pixelTopology::Phase1>::copyFromGPU(TrackingRecHit2DGPUT<pixelTopology::Phase1> const* input, cudaStream_t stream) {
//   assert(input);
//   this->m_store32 = input->localCoordToHostAsync(stream);
// }
//
// template <>
// void TrackingRecHit2DHostT<pixelTopology::Phase2>::copyFromGPU(TrackingRecHit2DGPUT<pixelTopology::Phase2> const* input, cudaStream_t stream) {
//   assert(input);
//   this->m_store32 = input->localCoordToHostAsync(stream);
// }
//
// template <>
// cms::cuda::host::unique_ptr<float[]> TrackingRecHit2DGPUT<pixelTopology::Phase2>::store32ToHostAsync(cudaStream_t stream) const {
//   auto ret = cms::cuda::make_host_unique<float[]>(static_cast<int>(this->n32) * this->nHits(), stream);
//   cms::cuda::copyAsync(ret, this->m_store32, static_cast<int>(this->n32) * this->nHits(), stream);
//   return ret;
// }
//
// template <>
// cms::cuda::host::unique_ptr<uint16_t[]> TrackingRecHit2DGPUT<pixelTopology::Phase2>::store16ToHostAsync(cudaStream_t stream) const {
//   auto ret = cms::cuda::make_host_unique<uint16_t[]>(static_cast<int>(this->n16) * this->nHits(), stream);
//   cms::cuda::copyAsync(ret, this->m_store16, static_cast<int>(this->n16) * this->nHits(), stream);
//   return ret;
// }
//
// template <>
// cms::cuda::host::unique_ptr<float[]> TrackingRecHit2DGPUT<pixelTopology::Phase1>::store32ToHostAsync(cudaStream_t stream) const {
//   auto ret = cms::cuda::make_host_unique<float[]>(static_cast<int>(this->n32) * this->nHits(), stream);
//   cms::cuda::copyAsync(ret, this->m_store32, static_cast<int>(this->n32) * this->nHits(), stream);
//   return ret;
// }
//
// template <>
// cms::cuda::host::unique_ptr<uint16_t[]> TrackingRecHit2DGPUT<pixelTopology::Phase1>::store16ToHostAsync(cudaStream_t stream) const {
//   auto ret = cms::cuda::make_host_unique<uint16_t[]>(static_cast<int>(this->n16) * this->nHits(), stream);
//   cms::cuda::copyAsync(ret, this->m_store16, static_cast<int>(this->n16) * this->nHits(), stream);
//   return ret;
// }


// template<class T> class A<T,int>;
// template<> class A<int,int>;

// template<> class TrackingRecHit2DCPUBaseT<cms::cudacompat::CPUTraits,pixelTopology::Phase1>;
// template<> class TrackingRecHit2DCPUBaseT<cms::cudacompat::CPUTraits,pixelTopology::Phase2>;
//
// template<> class TrackingRecHit2DGPUBaseT<cms::cudacompat::GPUTraits,pixelTopology::Phase1>;
// template<> class TrackingRecHit2DGPUBaseT<cms::cudacompat::GPUTraits,pixelTopology::Phase2>;
//
// template<> class TrackingRecHit2DHostBaseT<cms::cudacompat::HostTraits,pixelTopology::Phase1>;
// template<> class TrackingRecHit2DHostBaseT<cms::cudacompat::HostTraits,pixelTopology::Phase2>;

template class TrackingRecHit2DCPUBaseT<cms::cudacompat::CPUTraits,pixelTopology::Phase1>;
template class TrackingRecHit2DCPUBaseT<cms::cudacompat::CPUTraits,pixelTopology::Phase2>;

template class TrackingRecHit2DGPUBaseT<cms::cudacompat::GPUTraits,pixelTopology::Phase1>;
template class TrackingRecHit2DGPUBaseT<cms::cudacompat::GPUTraits,pixelTopology::Phase2>;

template<typename T> class TrackingRecHit2DCPUBaseT<cms::cudacompat::CPUTraits,T>;
template<typename T> class TrackingRecHit2DGPUBaseT<cms::cudacompat::GPUTraits,T>;

template<typename T> class TrackingRecHit2DCPUBaseT<T,pixelTopology::Phase1>;
template<typename T> class TrackingRecHit2DGPUBaseT<T,pixelTopology::Phase1>;

template<typename T> class TrackingRecHit2DCPUBaseT<T,pixelTopology::Phase2>;
template<typename T> class TrackingRecHit2DGPUBaseT<T,pixelTopology::Phase2>;

template class TrackingRecHit2DHostBaseT<cms::cudacompat::HostTraits,pixelTopology::Phase1>;
template class TrackingRecHit2DHostBaseT<cms::cudacompat::HostTraits,pixelTopology::Phase2>;
template<typename T> class TrackingRecHit2DHostBaseT<cms::cudacompat::HostTraits,T>;
template<typename T> class TrackingRecHit2DHostBaseT<T,pixelTopology::Phase1>;
template<typename T> class TrackingRecHit2DHostBaseT<T,pixelTopology::Phase2>;

template class TrackingRecHit2DHeterogeneousT<cms::cudacompat::GPUTraits,pixelTopology::Phase1>;
template class TrackingRecHit2DHeterogeneousT<cms::cudacompat::GPUTraits,pixelTopology::Phase2>;

// template class TrackingRecHit2DHeterogeneousT<cms::cudacompat::CPUTraits,pixelTopology::Phase1>;
// template class TrackingRecHit2DHeterogeneousT<cms::cudacompat::CPUTraits,pixelTopology::Phase2>;
//
// template class TrackingRecHit2DHeterogeneousT<cms::cudacompat::HostTraits,pixelTopology::Phase1>;
// template class TrackingRecHit2DHeterogeneousT<cms::cudacompat::HostTraits,pixelTopology::Phase2>;
