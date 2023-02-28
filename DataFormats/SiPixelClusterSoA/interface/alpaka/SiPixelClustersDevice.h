#ifndef DataFormats_SiPixelClusterSoA_interface_SiPixelClustersDevice_h
#define DataFormats_SiPixelClusterSoA_interface_SiPixelClustersDevice_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersLayout.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
// TODO: The class is created via inheritance of the PortableCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306
namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class SiPixelClustersDevice : public PortableCollection<SiPixelClustersLayout<>> {
  public:
    SiPixelClustersDevice() = default;
    ~SiPixelClustersDevice() = default;

    template <typename TQueue>
    explicit SiPixelClustersDevice(size_t maxModules, TQueue queue)
        : PortableCollection<SiPixelClustersLayout<>>(maxModules + 1, queue) {}

    SiPixelClustersDevice(SiPixelClustersDevice &&) = default;
    SiPixelClustersDevice &operator=(SiPixelClustersDevice &&) = default;

    void setNClusters(uint32_t nClusters, int32_t offsetBPIX2) {
      nClusters_h = nClusters;
      offsetBPIX2_h = offsetBPIX2;
    }

    uint32_t nClusters() const { return nClusters_h; }
    int32_t offsetBPIX2() const { return offsetBPIX2_h; }

  private:
    uint32_t nClusters_h = 0;
    int32_t offsetBPIX2_h = 0;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <>
  struct CopyToHost<ALPAKA_ACCELERATOR_NAMESPACE::SiPixelClustersDevice> {
    template <typename TQueue>
    static auto copyAsync(TQueue &queue, ALPAKA_ACCELERATOR_NAMESPACE::SiPixelClustersDevice const &deviceData) {
      SiPixelClustersHost hostData(deviceData.view().metadata().size(), queue);
      alpaka::memcpy(queue, hostData.buffer(), deviceData.buffer());
      return hostData;
    }
  };
}  // namespace cms::alpakatools
#endif  // DataFormats_SiPixelClusterSoA_interface_SiPixelClustersDevice_h
