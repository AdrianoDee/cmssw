#include <cstdlib>
#include <unistd.h>

#include <alpaka/alpaka.hpp>

#include "DataFormats/SiStripCluster/interface/SiStripClusterDevice.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterHost.h"
#include "DataFormats/SiStripCluster/interface/alpaka/SiStripClusterSoACollection.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "Hits_test.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

int main() {
  // Get the list of devices on the current platform
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cerr << "No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
      "the test will be skipped.\n";
    exit(EXIT_FAILURE);
  }

  // Run the test on each device
  for (const auto& device : devices) {
    Queue queue(device);

    // inner scope to deallocate memory before destroying the queue
    {
      uint32_t nClusters = 2000;
  
      SiStripClusterSoACollection stripSoA(queue, nClusters);

      testStripClusterSoA::runKernels(stripSoA.view(), queue);
    }
  }

  return EXIT_SUCCESS;
}
