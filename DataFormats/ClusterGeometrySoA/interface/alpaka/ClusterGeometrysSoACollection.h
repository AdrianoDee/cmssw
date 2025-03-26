#ifndef DataFormats_ClusterGeometrySoA_interface_alpaka_ClusterGeometrysSoACollection_h
#define DataFormats_ClusterGeometrySoA_interface_alpaka_ClusterGeometrysSoACollection_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/ClusterGeometrySoA/interface/ClusterGeometrysDevice.h"
#include "DataFormats/ClusterGeometrySoA/interface/ClusterGeometrysHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using ClusterGeometrysSoACollection =
      std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, ClusterGeometrysHost, ClusterGeometrysDevice<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <typename TDevice>
  struct CopyToHost<ClusterGeometrysDevice<TDevice>> {
    template <typename TQueue>
    static auto copyAsync(TQueue &queue, ClusterGeometrysDevice<TDevice> const &srcData) {
      ClusterGeometrysHost dstData(srcData.view().metadata().size() - 1, queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      dstData.setNModules(srcData.nModules());
      return dstData;
    }
  };
}  // namespace cms::alpakatools

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(ClusterGeometrysSoACollection, ClusterGeometrysHost);

#endif  // DataFormats_ClusterGeometrySoA_interface_alpaka_ClusterGeometrysSoACollection_h
