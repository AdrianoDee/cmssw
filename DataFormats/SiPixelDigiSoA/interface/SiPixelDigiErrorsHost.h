#ifndef DataFormats_SiPixelDigi_interface_SiPixelDigiErrorsAlpaka_h
#define DataFormats_SiPixelDigi_interface_SiPixelDigiErrorsAlpaka_h

#include <utility>

#include "DataFormats/SiPixelRawData/interface/SiPixelErrorCompact.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelFormatterErrors.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisErrorLayout.h"

#include "HeterogeneousCore/AlpakaUtilities/interface/SimpleVector.h"

class SiPixelDigiErrorsHost {
public:
  SiPixelDigiErrorsHost() = default;
  ~SiPixelDigiErrorsHost() = default;
  explicit SiPixelDigiErrorsHost(int nErrorWords,
                                 SiPixelFormatterErrors errors,
                                 cms::alpakatools::host_buffer<SiPixelErrorCompact[]> data)
      // TQueue queue)
      : nErrorWords_(nErrorWords), formatterErrors_h{std::move(errors)} {
    data_h = std::move(data);  //cms::alpakatools::make_host_buffer<SiPixelErrorCompact[]>(error_vec->capacity(),queue);
    // auto data_d_view = cms::alpakatools::make_device_view(*vec_data);

    //  // but transfer only the required amount
    // if (not error_vec->empty()) {
    //   alpaka::memcpy(queue, data_h, data_d_view);
    // }

    // // auto error_vec_d = cms::alpakatools::make_device_view(error_vec.get());
    // // auto error_vec_h = cms::alpakatools::make_host_view(error_vec.get());
    error_h = cms::alpakatools::make_host_buffer<cms::alpakatools::SimpleVector<SiPixelErrorCompact>>();
    (*error_h).data()->set_data((*data_h).data());
    // alpaka::memcpy(queue, error_h, error_vec_d);
    // error_h = cms::alpakatools::make_host_buffer<cms::alpakatools::SimpleVector<SiPixelErrorCompact>>(queue);
  }
  // SiPixelDigiErrorsHost(SiPixelDigiErrorsHost &&) = default;
  // SiPixelDigiErrorsHost &operator=(SiPixelDigiErrorsHost &&) = default;

  int nErrorWords() const { return nErrorWords_; }

  cms::alpakatools::SimpleVector<SiPixelErrorCompact>* error() { return (*error_h).data(); }
  cms::alpakatools::SimpleVector<SiPixelErrorCompact> const* error() const { return (*error_h).data(); }

  const SiPixelFormatterErrors& formatterErrors() const { return formatterErrors_h; }

private:
  int nErrorWords_ = 0;
  SiPixelFormatterErrors formatterErrors_h;
  std::optional<cms::alpakatools::host_buffer<SiPixelErrorCompact[]>> data_h;
  std::optional<cms::alpakatools::host_buffer<cms::alpakatools::SimpleVector<SiPixelErrorCompact>>> error_h;
};

#endif  // AlpakaDataFormats_alpaka_SiPixelDigiErrorsAlpaka_h
