#ifndef DataFormats_CandidateSoA_interface_alpaka_CandidateSoACollection_h
#define DataFormats_CandidateSoA_interface_alpaka_CandidateSoACollection_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/CandidateSoA/interface/CandidateLayoutDevice.h"
#include "DataFormats/CandidateSoA/interface/CandidateLayoutHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using CandidateSoACollection =
      std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, CandidateHost, CandidateDevice<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <typename TDevice>
  struct CopyToHost<CandidateDevice<TDevice>> {
    template <typename TQueue>
    static auto copyAsync(TQueue &queue, CandidateDevice<TDevice> const &srcData) {
      CandidateHost dstData(srcData.view().metadata().size(), queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      return dstData;
    }
  };
}  // namespace cms::alpakatools

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(CandidateSoACollection, CandidateHost);

#endif  // DataFormats_CandidateSoA_interface_alpaka_CandidateSoACollection_h
