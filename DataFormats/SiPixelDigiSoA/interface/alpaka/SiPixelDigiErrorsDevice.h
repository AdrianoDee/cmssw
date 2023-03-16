#ifndef AlpakaDataFormats_alpaka_SiPixelDigiErrorsAlpaka_h
#define AlpakaDataFormats_alpaka_SiPixelDigiErrorsAlpaka_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsHost.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelErrorCompact.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelFormatterErrors.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisErrorLayout.h"

#include "HeterogeneousCore/AlpakaUtilities/interface/SimpleVector.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

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
    // SiPixelDigiErrorsDevice() = delete;  // alpaka buffers are not default-constructible
    explicit SiPixelDigiErrorsDevice(size_t maxFedWords, SiPixelFormatterErrors errors, Queue& queue)
        : maxFedWords_(maxFedWords), formatterErrors_h{std::move(errors)} {
      data_d = cms::alpakatools::make_device_buffer<SiPixelErrorCompact[]>(queue, maxFedWords);
      error_d = cms::alpakatools::make_device_buffer<cms::alpakatools::SimpleVector<SiPixelErrorCompact>>(queue);
      // error_h = cms::alpakatools::make_host_buffer<cms::alpakatools::SimpleVector<SiPixelErrorCompact>>(queue);
      (*error_d).data()->construct(maxFedWords, data_d->data());
      ALPAKA_ASSERT_OFFLOAD((*error_d).data()->empty());
      ALPAKA_ASSERT_OFFLOAD((*error_d).data()->capacity() == static_cast<int>(maxFedWords));

      // alpaka::memcpy(queue, (*error_d), (*error_h));
    }
    SiPixelDigiErrorsDevice() = default;
    ~SiPixelDigiErrorsDevice() = default;

    SiPixelDigiErrorsDevice(const SiPixelDigiErrorsDevice&) = delete;
    SiPixelDigiErrorsDevice& operator=(const SiPixelDigiErrorsDevice&) = delete;
    SiPixelDigiErrorsDevice(SiPixelDigiErrorsDevice&&) = default;
    SiPixelDigiErrorsDevice& operator=(SiPixelDigiErrorsDevice&&) = default;

    const SiPixelFormatterErrors& formatterErrors() const { return formatterErrors_h; }

    cms::alpakatools::SimpleVector<SiPixelErrorCompact>* error() { return (*error_d).data(); }
    cms::alpakatools::SimpleVector<SiPixelErrorCompact> const* error() const { return (*error_d).data(); }
    cms::alpakatools::SimpleVector<SiPixelErrorCompact> const* c_error() const { return (*error_d).data(); }

    auto& error_vector() const { return (*error_d); }
    auto& error_data() const { return (*data_d); }
    auto maxFedWords() const { return maxFedWords_; }

  private:
    int maxFedWords_;
    SiPixelFormatterErrors formatterErrors_h;
    std::optional<cms::alpakatools::device_buffer<Device, SiPixelErrorCompact[]>> data_d;
    std::optional<cms::alpakatools::device_buffer<Device, cms::alpakatools::SimpleVector<SiPixelErrorCompact>>> error_d;
    // std::optional<cms::alpakatools::host_buffer<cms::alpakatools::SimpleVector<SiPixelErrorCompact>>> error_h;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <>
  struct CopyToHost<ALPAKA_ACCELERATOR_NAMESPACE::SiPixelDigiErrorsDevice> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, ALPAKA_ACCELERATOR_NAMESPACE::SiPixelDigiErrorsDevice const& srcData) {
      auto error_vector = srcData.error_vector();
      auto error_data_h = cms::alpakatools::make_host_buffer<SiPixelErrorCompact[]>(error_vector.data()->capacity());
      auto error_data = srcData.error_data();

      if (not error_vector->empty()) {
        alpaka::memcpy(queue, error_data_h, error_data);
      }

      SiPixelDigiErrorsHost dstData(error_vector.data()->capacity(), srcData.formatterErrors(), error_data_h);

      // alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      return dstData;
    }
  };

  // template <>
  // struct CopyToDevice<PortableHostCollection<TLayout>> {
  //   template <typename TQueue>
  //   static auto copyAsync(TQueue& queue, PortableHostCollection<TLayout> const& srcData) {
  //     using TDevice = typename alpaka::trait::DevType<TQueue>::type;
  //     PortableDeviceCollection<TLayout, TDevice> dstData(srcData->metadata().size(), queue);
  //     alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
  //     return dstData;
  //   }
  // };
}  // namespace cms::alpakatools

// }  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DeviceDataFormats_alpaka_SiPixelDigiErrorsDevice_h
