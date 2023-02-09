#ifndef AlpakaDataFormats_alpaka_SiPixelDigiErrorsAlpaka_h
#define AlpakaDataFormats_alpaka_SiPixelDigiErrorsAlpaka_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "DataFormats/SiPixelRawData/interface/SiPixelErrorCompact.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelFormatterErrors.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

class SiPixelDigiErrorsDevice : public PortableCollection<SiPixelDigisErrorLayout<>> {
public:
  SiPixelDigiErrorsDevice() = default;
  ~SiPixelDigiErrorsDevice() = default;
  template <typename TQueue>
  explicit SiPixelDigiErrorsDevice(size_t maxFedWords, SiPixelFormatterErrors errors, TQueue queue)
      : PortableCollection<SiPixelDigisErrorLayout<>>(maxFedWords, queue),
        formatterErrors_h{std::move(errors)} {};
  SiPixelDigiErrorsDevice(SiPixelDigiErrorsDevice &&) = default;
  SiPixelDigiErrorsDevice &operator=(SiPixelDigiErrorsDevice &&) = default;

private:
  PixelFormatterErrors formatterErrors_h;

  };
}// namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // AlpakaDataFormats_alpaka_SiPixelDigiErrorsAlpaka_h
