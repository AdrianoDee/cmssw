#ifndef CalibFormats_SiStripObjects_interface_SiStripClusterizerConditionsDevice_H
#define CalibFormats_SiStripObjects_interface_SiStripClusterizerConditionsDevice_H

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "CalibFormats/SiStripObjects/interface/SiStripClusterizerConditionsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace reco {
    template <typename TDev>
    using SiStripClusterizerConditionsDevice = PortableDeviceMultiCollection<TDev, StripConditionsDetSoA, StripConditionsStripSoA, StripConditionsAPVSoA>;
}
#endif  // CalibFormats_SiStripObjects_interface_SiStripClusterizerConditionsDevice_H