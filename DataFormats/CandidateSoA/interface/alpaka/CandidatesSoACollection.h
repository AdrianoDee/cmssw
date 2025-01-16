#ifndef DataFormats_CandidateSoA_interface_alpaka_CandidatesSoACollection_h
#define DataFormats_CandidateSoA_interface_alpaka_CandidatesSoACollection_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/CandidateSoA/interface/CandidatesDevice.h"
#include "DataFormats/CandidateSoA/interface/CandidatesHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using CandidatesSoACollection =
      std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, CandidatesHost, CandidatesDevice<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <typename TDevice>
  struct CopyToHost<CandidatesDevice<TDevice>> {
    template <typename TQueue>
    static auto copyAsync(TQueue &queue, CandidatesDevice<TDevice> const &srcData) {
      CandidatesHost dstData(srcData.view().metadata().size() - 1, queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      dstData.setNModules(srcData.nModules());
      return dstData;
    }
  };
}  // namespace cms::alpakatools

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(CandidatesSoACollection, CandidatesHost);

#endif  // DataFormats_CandidateSoA_interface_alpaka_CandidatesSoACollection_h
