#ifndef DataFormats_SiPixelDigi_interface_SiPixelDigisDevice_h
#define DataFormats_SiPixelDigi_interface_SiPixelDigisDevice_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisLayout.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class SiPixelDigisDevice : public PortableCollection<SiPixelDigisLayout<>> {
  public:
    SiPixelDigisDevice() = default;
    template <typename TQueue>
    explicit SiPixelDigisDevice(size_t maxFedWords, TQueue queue)
        : PortableCollection<SiPixelDigisLayout<>>(maxFedWords + 1, queue) {}
    ~SiPixelDigisDevice() = default;

    SiPixelDigisDevice(SiPixelDigisDevice &&) = default;
    SiPixelDigisDevice &operator=(SiPixelDigisDevice &&) = default;

    void setNModulesDigis(uint32_t nModules, uint32_t nDigis) {
      nModules_h = nModules;
      nDigis_h = nDigis;
    }

    uint32_t nModules() const { return nModules_h; }
    uint32_t nDigis() const { return nDigis_h; }

  private:
    uint32_t nModules_h = 0;
    uint32_t nDigis_h = 0;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
#endif  // DataFormats_SiPixelDigi_interface_SiPixelDigisDevice_h
