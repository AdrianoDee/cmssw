#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h

#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"
// #include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDASOAView.h"
#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(SiPixelDigisSoALayout,
                    SOA_COLUMN(int32_t, clus),
                    SOA_COLUMN(uint32_t, pdigi),
                    SOA_COLUMN(uint32_t, rawIdArr),
                    SOA_COLUMN(uint16_t, adc),
                    SOA_COLUMN(uint16_t, xx),
                    SOA_COLUMN(uint16_t, yy),
                    SOA_COLUMN(uint16_t, moduleId))

using SiPixelDigisCUDASOA = SiPixelDigisSoALayout<>;
using SiPixelDigisCUDASOAView = SiPixelDigisCUDASOA::View;
using SiPixelDigisCUDASOAConstView = SiPixelDigisCUDASOA::ConstView;

class SiPixelDigisCUDA : public cms::cuda::PortableDeviceCollection<SiPixelDigisSoALayout<>> {
public:
  // using StoreType = uint16_t;
  SiPixelDigisCUDA() = default;
  explicit SiPixelDigisCUDA(size_t maxFedWords, cudaStream_t stream)
      : PortableDeviceCollection<SiPixelDigisSoALayout<>>(maxFedWords + 1, stream) {}
  ~SiPixelDigisCUDA() = default;

  // SiPixelDigisCUDA(const SiPixelDigisCUDA &) = delete;
  // SiPixelDigisCUDA &operator=(const SiPixelDigisCUDA &) = delete;
  SiPixelDigisCUDA(SiPixelDigisCUDA &&) = default;
  SiPixelDigisCUDA &operator=(SiPixelDigisCUDA &&) = default;

  void setNModulesDigis(uint32_t nModules, uint32_t nDigis) {
    nModules_h = nModules;
    nDigis_h = nDigis;
  }

  uint32_t nModules() const { return nModules_h; }
  uint32_t nDigis() const { return nDigis_h; }

  // cms::cuda::host::unique_ptr<StoreType[]> copyAllToHostAsync(cudaStream_t stream) const;

  cms::cuda::host::unique_ptr<std::byte[]> copyAllToHostAsync(cudaStream_t stream) const {
    // Copy to a host buffer the host-device shared part (m_hostDeviceLayout).
    auto ret = cms::cuda::make_host_unique<std::byte[]>(bufferSize(), stream);
    cudaCheck(cudaMemcpyAsync(ret.get(), buffer().get(), bufferSize(), cudaMemcpyDeviceToHost, stream));
    return ret;
  }

  // SiPixelDigisCUDASOAView view() { return m_view; }
  // SiPixelDigisCUDASOAView const view() const { return m_view; }

private:
  // These are consumed by downstream device code
  // cms::cuda::device::unique_ptr<StoreType[]> m_store;

  // SiPixelDigisCUDASOAView m_view;

  uint32_t nModules_h = 0;
  uint32_t nDigis_h = 0;
};

#endif  // CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h
