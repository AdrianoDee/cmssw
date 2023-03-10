#ifndef DataFormats_SiPixelDigi_interface_SiPixelDigiErrorsAlpaka_h
#define DataFormats_SiPixelDigi_interface_SiPixelDigiErrorsAlpaka_h

#include <utility>

#include "DataFormats/SiPixelRawData/interface/SiPixelErrorCompact.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelFormatterErrors.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisErrorLayout.h"

class SiPixelDigiErrorsHost : public PortableCollection<SiPixelDigisErrorLayout<>> {
public:
  SiPixelDigiErrorsHost() = default;
  ~SiPixelDigiErrorsHost() = default;
  template <typename TQueue>
  explicit SiPixelDigiErrorsHost(size_t maxFedWords, SiPixelFormatterErrors errors, TQueue queue)
      : PortableHostCollection<SiPixelDigisErrorLayout<>>(maxFedWords, queue), formatterErrors_h{std::move(errors)} {}
  SiPixelDigiErrorsHost(SiPixelDigiErrorsHost &&) = default;
  SiPixelDigiErrorsHost &operator=(SiPixelDigiErrorsHost &&) = default;

  int nErrorWords() const { return nErrorWords_; }

private:
  SiPixelFormatterErrors formatterErrors_h;
  int nErrorWords_ = 0;
};

#endif  // AlpakaDataFormats_alpaka_SiPixelDigiErrorsAlpaka_h
