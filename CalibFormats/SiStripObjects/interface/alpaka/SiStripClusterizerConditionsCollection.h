#ifndef CalibFormats_SiStripObjects_interface_SiStripClusterizerConditionsCollection_h
#define CalibFormats_SiStripObjects_interface_SiStripClusterizerConditionsCollection_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "CalibFormats/SiStripObjects/interface/SiStripClusterizerConditionsDevice.h"
#include "CalibFormats/SiStripObjects/interface/SiStripClusterizerConditionsHost.h"
#include "CalibFormats/SiStripObjects/interface/SiStripClusterizerConditionsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

    using ::reco::SiStripClusterizerConditionsHost;
    using ::reco::SiStripClusterizerConditionsDevice;
    using SiStripClusterizerConditionsCollection =
        std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, SiStripClusterizerConditionsHost, SiStripClusterizerConditionsDevice<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(reco::SiStripClusterizerConditionsCollection, reco::SiStripClusterizerConditionsHost);

#endif  // CalibFormats_SiStripObjects_interface_SiStripClusterizerConditionsCollection_h