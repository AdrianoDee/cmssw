#ifndef RecoTracker_PixelSeeding_interface_TrackerGeometryHost_H
#define RecoTracker_PixelSeeding_interface_TrackerGeometryHost_H

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoTracker/GeometryESProducer/interface/TrackerGeometrySoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace reco {
  using TrackerGeometryHost = PortableHostMultiCollection<TrackerModulesSoA, TrackerMaterialMapSoA>;
}
#endif  // RecoTracker_PixelSeeding_interface_TrackerGeometryHost_H
