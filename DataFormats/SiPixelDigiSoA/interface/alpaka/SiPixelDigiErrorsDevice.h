#ifndef AlpakaDataFormats_alpaka_SiPixelDigiErrorsAlpaka_h
#define AlpakaDataFormats_alpaka_SiPixelDigiErrorsAlpaka_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "DataFormats/SiPixelRawData/interface/SiPixelErrorCompact.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelFormatterErrors.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class SiPixelDigiErrorsDevice : public PortableCollection<SiPixelDigisErrorLayout<>> {
  public:
    SiPixelDigiErrorsDevice() = default;
    ~SiPixelDigiErrorsDevice() = default;
    template <typename TQueue>
    explicit SiPixelDigiErrorsDevice(size_t maxFedWords, SiPixelFormatterErrors errors, TQueue queue)
        : PortableCollection<SiPixelDigisErrorLayout<>>(maxFedWords, queue), formatterErrors_h{std::move(errors)} {};
    SiPixelDigiErrorsDevice(SiPixelDigiErrorsDevice &&) = default;
    SiPixelDigiErrorsDevice &operator=(SiPixelDigiErrorsDevice &&) = default;

  private:
    PixelFormatterErrors formatterErrors_h;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <>
  struct CopyToHost<ALPAKA_ACCELERATOR_NAMESPACE::SiPixelDigiErrorsDevice> {
    template <typename TQueue>
    static auto copyAsync(TQueue &queue, ALPAKA_ACCELERATOR_NAMESPACE::SiPixelDigiErrorsDevice const &deviceData) {
      SiPixelDigiErrorsHost hostData(queue);
      alpaka::memcpy(queue, hostData.buffer(), deviceData.buffer());
      return hostData;
    }
  };
}  // namespace cms::alpakatools

#endif  // AlpakaDataFormats_alpaka_SiPixelDigiErrorsAlpaka_h
