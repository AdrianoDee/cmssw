#ifndef CalibFormats_SiStripObjects_interface_SiStripClusterizerConditionsSoA_h
#define CalibFormats_SiStripObjects_interface_SiStripClusterizerConditionsSoA_h

#include <array>
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace reco
{   

    GENERATE_SOA_LAYOUT(StripConditionsDetLayout,
                        SOA_COLUMN(uint8_t, fedCh),
                        SOA_COLUMN(uint16_t, iPair),
                        SOA_COLUMN(float, invthick),
                        SOA_COLUMN(uint32_t, detId),
                        SOA_COLUMN(uint32_t, fedId),
                        SOA_COLUMN(uint32_t, order))

    GENERATE_SOA_LAYOUT(StripConditionsStripLayout,
                        SOA_COLUMN(uint16_t, noise))
    GENERATE_SOA_LAYOUT(StripConditionsAPVLayout,                    
                        SOA_COLUMN(float, gain))

    using StripConditionsDetSoA = StripConditionsDetLayout<>;
    using StripConditionsDetSoAView = StripConditionsDetSoA::View;
    using StripConditionsDetSoAConstView = StripConditionsDetSoA::ConstView;

    using StripConditionsStripSoA = StripConditionsStripLayout<>;
    using StripConditionsStripSoAView = StripConditionsStripSoA::View;
    using StripConditionsStripSoAConstView = StripConditionsStripSoA::ConstView;

    using StripConditionsAPVSoA = StripConditionsAPVLayout<>;
    using StripConditionsAPVSoAView = StripConditionsAPVSoA::View;
    using StripConditionsAPVSoAConstView = StripConditionsAPVSoA::ConstView;

}

#endif  // CalibFormats_SiStripObjects_interface_SiStripClusterizerConditionsSoA_h
