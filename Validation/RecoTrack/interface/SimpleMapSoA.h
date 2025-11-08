#ifndef Validation_RecoTrack_SimpleMapSoA_H
#define Validation_RecoTrack_SimpleMapSoA_H

#include <alpaka/alpaka.hpp>
#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(SimpleMapLayout,
  SOA_COLUMN(uint32_t, id)
)

using SimpleMapSoA = SimpleMapLayout<>;
using SimpleMapSoAView = SimpleMapSoA::View;
using SimpleMapSoAConstView = SimpleMapSoA::ConstView;

#endif  // Validation_RecoTrack_SimpleMapSoA_H