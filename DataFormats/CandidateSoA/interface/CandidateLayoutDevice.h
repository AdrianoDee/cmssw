#ifndef DataFormats_CandidateSoA_interface_CandidateDevice_h
#define DataFormats_CandidateSoA_interface_CandidateDevice_h

#include <cstdint>
#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/CandidateSoA/interface/CandidateLayout.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

template <typename TDev>
class CandidateDevice : public PortableDeviceCollection<CandidateSoA, TDev> {
public:
  // Constructor that initializes the collection with the size and queue (for device memory)
  template <typename TQueue>
  explicit CandidateDevice(size_t nClusters, TQueue queue)
      : PortableDeviceCollection<CandidateSoA, TDev>(nClusters, queue) {}

  // Constructor that initializes the collection with the size and device reference
  explicit CandidateDevice(size_t nClusters, TDev const& device)
      : PortableDeviceCollection<CandidateSoA, TDev>(nClusters, device) {}

  // Set number of modules (for example, in case of geometry data like detector modules)
  void setNModules(uint32_t nModules) { nModules_h = nModules; }

  uint32_t nModules() const { return nModules_h; }
  
  // Returns the number of clusters (metadata size - 1 for one-based indexing)
  uint32_t nClusters() const { return this->view().metadata().size() - 1; }

private:
  uint32_t nModules_h = 0;  // Number of modules (e.g., detector modules)
};

#endif  // DataFormats_CandidateSoA_interface_CandidateDevice_h
