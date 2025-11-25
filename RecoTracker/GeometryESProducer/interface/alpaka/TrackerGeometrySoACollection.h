#ifndef RecoTracker_GeometryESProducer_interface_TrackerGeometrySoACollection_h
#define RecoTracker_GeometryESProducer_interface_TrackerGeometrySoACollection_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "RecoTracker/GeometryESProducer/interface/TrackerGeometryDevice.h"
#include "RecoTracker/GeometryESProducer/interface/TrackerGeometryHost.h"
#include "RecoTracker/GeometryESProducer/interface/TrackerGeometrySoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using ::reco::TrackerGeometryDevice;
  using ::reco::TrackerGeometryHost;
  using TrackerGeometrySoACollection =
      std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, TrackerGeometryHost, TrackerGeometryDevice<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::reco

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(reco::TrackerGeometrySoACollection, reco::TrackerGeometryHost);

#endif  // RecoTracker_GeometryESProducer_interface_TrackerGeometrySoACollection_h
