#ifndef DataFormats_ClusterGeometrySoA_interface_alpaka_ClusterGeometrySoACollection_h
#define DataFormats_ClusterGeometrySoA_interface_alpaka_ClusterGeometrySoACollection_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/ClusterGeometrySoA/interface/ClusterGeometryLayoutDevice.h"
#include "DataFormats/ClusterGeometrySoA/interface/ClusterGeometryLayoutHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using ClusterGeometrySoACollection =
      std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, ClusterGeometryHost, ClusterGeometryDevice<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <typename TDevice>
  struct CopyToHost<ClusterGeometryDevice<TDevice>> {
    template <typename TQueue>
    static auto copyAsync(TQueue &queue, ClusterGeometryDevice<TDevice> const &srcData) {
      ClusterGeometryHost dstData(srcData.view().metadata().size(), queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      return dstData;
    }
  };
}  // namespace cms::alpakatools

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(ClusterGeometrySoACollection, ClusterGeometryHost);

#endif  // DataFormats_ClusterGeometrySoA_interface_alpaka_ClusterGeometrySoACollection_h
