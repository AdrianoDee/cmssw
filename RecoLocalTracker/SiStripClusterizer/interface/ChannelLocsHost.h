#ifndef RecoLocalTracker_SiStripClusterizer_interface_ChannelLocsHost_H
#define RecoLocalTracker_SiStripClusterizer_interface_ChannelLocsHost_H

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/ChannelLocsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace reco {
    using ChannelLocsHost = PortableHostCollection<StripConditionsDetSoA>;
}
#endif  // RecoLocalTracker_SiStripClusterizer_interface_ChannelLocsHost_H