#ifndef DataFormats_TrackingRecHitSoA_interface_TrackingRecHitSoADevice_h
#define DataFormats_TrackingRecHitSoA_interface_TrackingRecHitSoADevice_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"

namespace reco 
{

  template <typename TDev>
  using HitPortableCollectionDevice = PortableDeviceMultiCollection<TDev, reco::TrackingRecHitSoA, reco::HitModuleSoA>;

  template <typename TDev>
  class TrackingRecHitDevice : public HitPortableCollectionDevice<TDev> {
  public:

    TrackingRecHitDevice() = default;

    // Constructor which specifies the SoA size, number of BPIX1 hits, and the modules entry points
    template <typename TQueue>
    explicit TrackingRecHitDevice(TQueue queue, uint32_t nHits, int32_t offsetBPIX2, uint32_t const* hitsModuleStart)
        : HitPortableCollectionDevice<TDev>({{int(nHits),10}}, queue), offsetBPIX2_{offsetBPIX2} {
      auto hitsView = this->template view<TrackingRecHitSoA>();
      auto start_h = cms::alpakatools::make_device_view(queue, hitsModuleStart, phase1PixelTopology::numberOfModules + 1);
      auto start_d =
          cms::alpakatools::make_device_view(queue, hitsView.hitsModuleStart().data(), phase1PixelTopology::numberOfModules + 1);
      alpaka::memcpy(queue, start_d, start_h);

      auto off_h = cms::alpakatools::make_host_view(offsetBPIX2_);
      auto off_d = cms::alpakatools::make_device_view(queue, hitsView.offsetBPIX2());
      alpaka::memcpy(queue, off_d, off_h);
    }

    uint32_t nHits() const { return this->template view<TrackingRecHitSoA>().metadata().size(); }

    int32_t offsetBPIX2() const { return offsetBPIX2_; }

    uint32_t const* hitsModuleStart() const { return this->template view<TrackingRecHitSoA>().hitsModuleStart().data(); }

    // asynchronously update the information cached within the class itself from the information on the device
    template <typename TQueue>
    void updateFromDevice(TQueue queue) {
      auto off_h = cms::alpakatools::make_host_view(offsetBPIX2_);
      auto off_d = cms::alpakatools::make_device_view(queue, this->template view<TrackingRecHitSoA>().offsetBPIX2());
      alpaka::memcpy(queue, off_h, off_d);
    }

  private:
    // offsetBPIX2 is used on host functions so is useful to have it also stored in the class and not only in the layout
    int32_t offsetBPIX2_ = 0;
  };
}

#endif  // DataFormats_RecHits_interface_TrackingRecHitSoADevice_h
