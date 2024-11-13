#ifndef DataFormats_Track_TracksHost_H
#define DataFormats_Track_TracksHost_H

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace reco
{
    using TracksHost = PortableHostMultiCollection<TrackSoA, TrackHitSoA>;
}

#endif  // DataFormats_Track_TracksHost_H
