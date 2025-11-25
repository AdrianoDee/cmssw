#ifndef RecoTracker_GeometryESProducer_interface_TrackerGeometryDevice_H
#define RecoTracker_GeometryESProducer_interface_TrackerGeometryDevice_H

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "RecoTracker/GeometryESProducer/interface/TrackerGeometrySoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace reco {
  template <typename TDev>
  using TrackerGeometryDevice = PortableDeviceMultiCollection<TDev, TrackerModulesSoA, TrackerMaterialMapSoA>;
}
#endif  // RecoTracker_GeometryESProducer_interface_TrackerGeometryDevice_H
