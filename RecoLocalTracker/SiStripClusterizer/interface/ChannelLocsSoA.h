#ifndef RecoLocalTracker_SiStripClusterizer_interface_ChannelLocsSoA_h
#define RecoLocalTracker_SiStripClusterizer_interface_ChannelLocsSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace reco
{
  GENERATE_SOA_LAYOUT(ChannelLocsLayout,
                      SOA_COLUMN(uint8_t, raw),
                      SOA_COLUMN(uint8_t, fedCh),
                      SOA_COLUMN(uint16_t, length),
                      SOA_COLUMN(uint16_t, fedID),
                      SOA_COLUMN(uint32_t, inoff),
                      SOA_COLUMN(uint32_t, offset));

  using ChannelLocsSoA = ChannelLocsLayout<>;
  using ChannelLocsSoAView = ChannelLocsSoA::View;
  using ChannelLocsSoAConstView = ChannelLocsSoA::ConstView;
}

#endif
