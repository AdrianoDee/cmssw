#ifndef DataFormats_SiStripCluster_test_alpaka_Strip_test_h
#define DataFormats_SiStripCluster_test_alpaka_Strip_test_h

#include "DataFormats/SiStripCluster/interface/SiStripClusterSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::testStripClustersSoA {

  void runKernels(SiStripClusterSoAView& hits, Queue& queue);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::testStripClustersSoA

#endif  // DataFormats_SiStripCluster_test_alpaka_Strip_test_h
