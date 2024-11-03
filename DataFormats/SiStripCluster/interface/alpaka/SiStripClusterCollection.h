#ifndef DataFormats_SiStripCluster_interface_alpaka_SiStripClusterSoACollection_h
#define DataFormats_SiStripCluster_interface_alpaka_SiStripClusterSoACollection_h

#include <cstdint>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterDevice.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterHost.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterSoA.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using SiStripClusterSoACollection = std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>,
                                                          SiStripClusterHost,
                                                          SiStripClusterDevice<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(SiStripClusterSoACollection, SiStripClusterHost);
#endif  // DataFormats_SiStripCluster_interface_alpaka_SiStripClusterSoACollection_h
