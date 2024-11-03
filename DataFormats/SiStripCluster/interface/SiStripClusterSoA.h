#ifndef DataFormats_SiStripCluster_interface_SiStripClusterSoA_h
#define DataFormats_SiStripCluster_interface_SiStripClusterSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"

struct SiStripClusterSoA {

  GENERATE_SOA_LAYOUT(Layout,
                      SOA_COLUMN(uint32_t, clusterIndex),
                      SOA_COLUMN(uint32_t, clusterSize),
                      SOA_COLUMN(uint32_t, clusterADCs),
                      SOA_COLUMN(uint32_t, clusterDetId),
                      SOA_COLUMN(uint16_t, firstStrip),
                      SOA_COLUMN(bool, trueCluster),
                      SOA_COLUMN(float, barycenter),
                      SOA_COLUMN(float, charge),
                      SOA_SCALAR(uint32_t, nClusters),
                      SOA_SCALAR(uint32_t, maxClusterSize))
};

using SiStripClusterLayout = SiStripClusterSoA::Layout<>;
using SiStripClusterSoAView = SiStripClusterSoA::Layout<>::View;
using SiStripClusterSoAConstView = SiStripClusterSoA::Layout<>::ConstView;

#endif
