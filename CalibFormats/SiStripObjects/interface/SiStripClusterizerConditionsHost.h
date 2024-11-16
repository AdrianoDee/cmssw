#ifndef CalibFormats_SiStripObjects_interface_SiStripClusterizerConditionsHost_H
#define CalibFormats_SiStripObjects_interface_SiStripClusterizerConditionsHost_H

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "CalibFormats/SiStripObjects/interface/SiStripClusterizerConditionsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace reco {
    using SiStripClusterizerConditionsHost = PortableHostMultiCollection<StripConditionsDetSoA, StripConditionsStripSoA, StripConditionsAPVSoA>;
}
#endif  // CalibFormats_SiStripObjects_interface_SiStripClusterizerConditionsHost_H