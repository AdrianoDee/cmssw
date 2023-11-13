#include "FWCore/Utilities/interface/typelookup.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFastParamsDevice.h"
#include "Geometry/CommonTopologies/interface/SimplePixelStripTopology.h"

using PixelCPEFastParamsPhase1 = PixelCPEFastParamsDevice<alpaka_common::DevHost, pixelTopology::Phase1>;
using PixelCPEFastParamsPhase2 = PixelCPEFastParamsDevice<alpaka_common::DevHost, pixelTopology::Phase2>;
using PixelCPEFastParamsHIonPhase1 = PixelCPEFastParamsDevice<alpaka_common::DevHost, pixelTopology::HIonPhase1>;
using PixelCPEFastParamsPhase1Strip = PixelCPEFastParamsDevice<alpaka_common::DevHost, pixelTopology::Phase1Strip>;

TYPELOOKUP_DATA_REG(PixelCPEFastParamsPhase1);
TYPELOOKUP_DATA_REG(PixelCPEFastParamsPhase2);
TYPELOOKUP_DATA_REG(PixelCPEFastParamsHIonPhase1);
TYPELOOKUP_DATA_REG(PixelCPEFastParamsPhase1Strip);