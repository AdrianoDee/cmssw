#include "FWCore/Utilities/interface/typelookup.h"
#include "DataFormats/PixelCPEFastParams/interface/PixelCPEFastParams.h"
#include "DataFormats/TrackerCommon/interface/SimplePixelTopology.h"

using PixelCPEFastParamsPhase1 = pixelCPEforDevice::PixelCPEFastParams<alpaka_common::DevHost, pixelTopology::Phase1>;
using PixelCPEFastParamsPhase2 = pixelCPEforDevice::PixelCPEFastParams<alpaka_common::DevHost, pixelTopology::Phase2>;

TYPELOOKUP_DATA_REG(PixelCPEFastParamsPhase1);
TYPELOOKUP_DATA_REG(PixelCPEFastParamsPhase2);