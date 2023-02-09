#ifndef DataFormats_SiPixelDigi_interface_SiPixelDigiErrorsAlpaka_h
#define DataFormats_SiPixelDigi_interface_SiPixelDigiErrorsAlpaka_h

#include <utility>

#include "DataFormats/SiPixelRawData/interface/SiPixelErrorCompact.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelFormatterErrors.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

class SiPixelDigiErrorsHost : public PortableCollection<SiPixelDigisErrorLayout<>> {
public:
  SiPixelDigiErrorsHost() = default;
  ~SiPixelDigiErrorsHost() = default;
  template <typename TQueue>
  explicit SiPixelDigiErrorsHost(size_t maxFedWords, SiPixelFormatterErrors errors, TQueue queue)
      : PortableHostCollection<SiPixelDigisErrorLayout<>>(maxFedWords, queue),
        formatterErrors_h{std::move(errors)} {}
  SiPixelDigiErrorsHost(SiPixelDigiErrorsHost &&) = default;
  SiPixelDigiErrorsHost &operator=(SiPixelDigiErrorsHost &&) = default;

private:
  PixelFormatterErrors formatterErrors_h;

  };

#endif  // AlpakaDataFormats_alpaka_SiPixelDigiErrorsAlpaka_h
