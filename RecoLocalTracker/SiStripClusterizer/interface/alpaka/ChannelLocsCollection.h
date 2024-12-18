#ifndef RecoLocalTracker_SiStripClusterizer_interface_ChannelLocsCollection_H
#define RecoLocalTracker_SiStripClusterizer_interface_ChannelLocsCollection_H

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/ChannelLocsDevice.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/ChannelLocsHost.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/ChannelLocsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

    using ::reco::ChannelLocsHost;
    using ::reco::ChannelLocsDevice;
    using SiStripClusterizerConditionsCollection =
        std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, ChannelLocs, ChannelLocsDevice<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(reco::ChannelLocsDevice, reco::ChannelLocsHost);

#endif  // RecoLocalTracker_SiStripClusterizer_interface_ChannelLocsCollection_H