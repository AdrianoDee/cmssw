#ifndef DataFormats_SiStripCluster_interface_SiStripClustersHost_h
#define DataFormats_SiStripCluster_interface_SiStripClustersHost_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterSoA.h"

using SiStripClusterHost = PortableHostCollection<SiStripClusterLayout>;

#endif  // DataFormats_SiStripCluster_interface_SiStripClustersHost_h
