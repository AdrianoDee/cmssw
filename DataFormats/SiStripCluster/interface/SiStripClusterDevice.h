#ifndef DataFormats_SiStripCluster_interface_SiStripClusterSoADevice_h
#define DataFormats_SiStripCluster_interface_SiStripClusterSoADevice_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterSoA.h"

template <typename TDev>
using SiStripClusterDevice = PortableDeviceCollection<SiStripClusterLayout, TDev>;

#endif  // DataFormats_RecHits_interface_SiStripClusterSoADevice_h
