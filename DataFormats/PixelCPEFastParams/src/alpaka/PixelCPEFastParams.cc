#include "HeterogeneousCore/AlpakaCore/interface/alpaka/typelookup.h"
#include "DataFormats/PixelCPEFastParams/interface/PixelCPEFastParams.h"
#include "DataFormats/TrackerCommon/interface/SimplePixelTopology.h"

template <typename TDev>
using PixelCPEFastParamsPhase1 = pixelCPEforDevice::PixelCPEFastParams<TDev, pixelTopology::Phase1>;
template <typename TDev>
using PixelCPEFastParamsPhase2 = pixelCPEforDevice::PixelCPEFastParams<TDev, pixelTopology::Phase2>;

TYPELOOKUP_ALPAKA_TEMPLATED_DATA_REG(PixelCPEFastParamsPhase1);
TYPELOOKUP_ALPAKA_TEMPLATED_DATA_REG(PixelCPEFastParamsPhase2);