#ifndef DataFormats_CandidateSoA_interface_CandidatesDevice_h
#define DataFormats_CandidateSoA_interface_CandidatesDevice_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/CandidateSoA/interface/CandidatesSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

template <typename TDev>
class CandidatesDevice : public PortableDeviceCollection<CandidatesSoA, TDev> {
public:
  CandidatesDevice() = default;
  template <typename TQueue>
  explicit CandidatesDevice(size_t maxFedWords, TQueue queue)
      : PortableDeviceCollection<CandidatesSoA, TDev>(maxFedWords + 1, queue) {}

  // Constructor which specifies the SoA size
  explicit CandidatesDevice(size_t maxFedWords, TDev const &device)
      : PortableDeviceCollection<CandidatesSoA, TDev>(maxFedWords + 1, device) {}

  void setNModules(uint32_t nModules) { nModules_h = nModules; }

  uint32_t nModules() const { return nModules_h; }
  uint32_t nDigis() const { return this->view().metadata().size() - 1; }

private:
  uint32_t nModules_h = 0;
};

#endif  // DataFormats_CandidateSoA_interface_CandidatesDevice_h
