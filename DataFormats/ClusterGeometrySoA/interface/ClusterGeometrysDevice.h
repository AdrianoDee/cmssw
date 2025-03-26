#ifndef DataFormats_ClusterGeometrySoA_interface_ClusterGeometrysDevice_h
#define DataFormats_ClusterGeometrySoA_interface_ClusterGeometrysDevice_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/ClusterGeometrySoA/interface/ClusterGeometrysSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

template <typename TDev>
class ClusterGeometrysDevice : public PortableDeviceCollection<ClusterGeometrysSoA, TDev> {
public:

  ClusterGeometrysDevice(edm::Uninitialized) : PortableDeviceCollection<ClusterGeometrysSoA, TDev>{edm::kUninitialized} {}

  template <typename TQueue>
  explicit ClusterGeometrysDevice(size_t maxFedWords, TQueue queue)
      : PortableDeviceCollection<ClusterGeometrysSoA, TDev>(maxFedWords + 1, queue) {}

  // Constructor which specifies the SoA size
  explicit ClusterGeometrysDevice(size_t maxFedWords, TDev const &device)
      : PortableDeviceCollection<ClusterGeometrysSoA, TDev>(maxFedWords + 1, device) {}

  void setNModules(uint32_t nModules) { nModules_h = nModules; }

  uint32_t nModules() const { return nModules_h; }
  uint32_t nDigis() const { return this->view().metadata().size() - 1; }

private:
  uint32_t nModules_h = 0;
};

#endif  // DataFormats_ClusterGeometrySoA_interface_ClusterGeometrysDevice_h
