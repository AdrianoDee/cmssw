#include <alpaka/alpaka.hpp>
#include <unistd.h>

#include "DataFormats/SiPixelClusterPortable/interface/alpaka/SiPixelClustersDevice.h"
#include "DataFormats/SiPixelClusterPortable/interface/SiPixelClustersHost.h"
#include "DataFormats/SiPixelClusterPortable/SiPixelClustersLayout.h"

#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace testClusterSoA {

    void runKernels(SiPixelClustersLayoutSOAView tracks_view, Queue& queue);
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

int main() {
  const auto host = cms::alpakatools::host();
  const auto device = cms::alpakatools::devices<Platform>()[0];
  Queue queue(device);

  // Inner scope to deallocate memory before destroying the stream
  {
    // Instantiate tracks on device. PortableDeviceCollection allocates
    // SoA on device automatically.
    SiPixelClustersDevice clusters(queue);
    testClusterSoA::runKernels(clusters.view(), queue);

    // Instantate tracks on host. This is where the data will be
    // copied to from device.
    SiPixelClustersHost tracks_h(queue);

    std::cout << tracks_h.view().metadata().size() << std::endl;
    alpaka::memcpy(queue, tracks_h.buffer(), tracks_d.const_buffer());
    alpaka::wait(queue);

    // Print results
    std::cout << "pt"
              << "\t"
              << "eta"
              << "\t"
              << "chi2"
              << "\t"
              << "quality"
              << "\t"
              << "nLayers"
              << "\t"
              << "hitIndices off" << std::endl;

    for (int i = 0; i < 10; ++i) {
      std::cout << tracks_h.view()[i].pt() << "\t" << tracks_h.view()[i].eta() << "\t" << tracks_h.view()[i].chi2()
                << "\t" << (int)tracks_h.view()[i].quality() << "\t" << (int)tracks_h.view()[i].nLayers() << "\t"
                << tracks_h.view().hitIndices().off[i] << std::endl;
    }
  }

  return 0;
}
