#ifndef AlpakaDataFormats_alpaka_SiPixelDigiErrorsAlpaka_h
#define AlpakaDataFormats_alpaka_SiPixelDigiErrorsAlpaka_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "DataFormats/SiPixelRawData/interface/SiPixelErrorCompact.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelFormatterErrors.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisErrorLayout.h"

#include "HeterogeneousCore/AlpakaUtilities/interface/SimpleVector.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

// class SiPixelDigiErrorsDevice : public PortableCollection<SiPixelDigisErrorLayout<>> {
// public:
//   SiPixelDigiErrorsDevice() = default;
//   ~SiPixelDigiErrorsDevice() = default;
//   template <typename TQueue>
//   explicit SiPixelDigiErrorsDevice(size_t maxFedWords, SiPixelFormatterErrors errors, TQueue queue)
//       : PortableCollection<SiPixelDigisErrorLayout<>>(maxFedWords, queue),
//         formatterErrors_h{std::move(errors)} {};
//   SiPixelDigiErrorsDevice(SiPixelDigiErrorsDevice &&) = default;
//   SiPixelDigiErrorsDevice &operator=(SiPixelDigiErrorsDevice &&) = default;

// private:
//   SiPixelFormatterErrors formatterErrors_h;

//   };

class SiPixelDigiErrorsDevice {
  public:
    SiPixelDigiErrorsDevice() = delete;  // alpaka buffers are not default-constructible
    explicit SiPixelDigiErrorsDevice(size_t maxFedWords, SiPixelFormatterErrors errors, Queue& queue)
        : data_d{cms::alpakatools::make_device_buffer<SiPixelErrorCompact[]>(queue, maxFedWords)},
          error_d{cms::alpakatools::make_device_buffer<cms::alpakatools::SimpleVector<SiPixelErrorCompact>>(queue)},
          error_h{cms::alpakatools::make_host_buffer<cms::alpakatools::SimpleVector<SiPixelErrorCompact>>(queue)},
          formatterErrors_h{std::move(errors)} {
      error_h->construct(maxFedWords, data_d.data());
      ALPAKA_ASSERT_OFFLOAD(error_h->empty());
      ALPAKA_ASSERT_OFFLOAD(error_h->capacity() == static_cast<int>(maxFedWords));

      alpaka::memcpy(queue, error_d, error_h);
    }
    ~SiPixelDigiErrorsDevice() = default;

    SiPixelDigiErrorsDevice(const SiPixelDigiErrorsDevice&) = delete;
    SiPixelDigiErrorsDevice& operator=(const SiPixelDigiErrorsDevice&) = delete;
    SiPixelDigiErrorsDevice(SiPixelDigiErrorsDevice&&) = default;
    SiPixelDigiErrorsDevice& operator=(SiPixelDigiErrorsDevice&&) = default;

    const SiPixelFormatterErrors& formatterErrors() const { return formatterErrors_h; }

    cms::alpakatools::SimpleVector<SiPixelErrorCompact>* error() { return error_d.data(); }
    cms::alpakatools::SimpleVector<SiPixelErrorCompact> const* error() const { return error_d.data(); }
    cms::alpakatools::SimpleVector<SiPixelErrorCompact> const* c_error() const { return error_d.data(); }


  private:
    cms::alpakatools::device_buffer<Device, SiPixelErrorCompact[]> data_d;
    cms::alpakatools::device_buffer<Device, cms::alpakatools::SimpleVector<SiPixelErrorCompact>> error_d;
    cms::alpakatools::host_buffer<cms::alpakatools::SimpleVector<SiPixelErrorCompact>> error_h;
    SiPixelFormatterErrors formatterErrors_h;
  };

}// namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DeviceDataFormats_alpaka_SiPixelDigiErrorsDevice_h
