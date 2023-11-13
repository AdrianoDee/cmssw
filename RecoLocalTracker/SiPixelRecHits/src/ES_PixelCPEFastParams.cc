#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFastParamsHost.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "Geometry/CommonTopologies/interface/SimplePixelStripTopology.h"

using PixelCPEFastParamsHostPhase1 = PixelCPEFastParamsHost<pixelTopology::Phase1>;
using PixelCPEFastParamsHostPhase2 = PixelCPEFastParamsHost<pixelTopology::Phase2>;
using PixelCPEFastParamsHostHIonPhase1 = PixelCPEFastParamsHost<pixelTopology::HIonPhase1>;
using PixelCPEFastParamsHostPhase1Strip = PixelCPEFastParamsHost<pixelTopology::Phase1Strip>;


TYPELOOKUP_DATA_REG(PixelCPEFastParamsHostPhase1);
TYPELOOKUP_DATA_REG(PixelCPEFastParamsHostPhase2);
TYPELOOKUP_DATA_REG(PixelCPEFastParamsHostHIonPhase1);
TYPELOOKUP_DATA_REG(PixelCPEFastParamsHostPhase1Strip);

