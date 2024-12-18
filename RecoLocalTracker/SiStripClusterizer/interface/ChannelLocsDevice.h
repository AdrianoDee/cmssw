#ifndef RecoLocalTracker_SiStripClusterizer_interface_ChannelLocsHost_H
#define RecoLocalTracker_SiStripClusterizer_interface_ChannelLocsHost_H

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/ChannelLocsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace reco {
    template <typename TDev>
    using ChannelLocsHost = PortableDeviceCollection<StripConditionsDetSoA, TDev>;
}
#endif  // RecoLocalTracker_SiStripClusterizer_interface_ChannelLocsHost_H