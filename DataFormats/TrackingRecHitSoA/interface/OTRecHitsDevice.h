#ifndef DataFormats_TrackingRecHitSoA_interface_OTRecHitsDevice_h
#define DataFormats_TrackingRecHitSoA_interface_OTRecHitsDevice_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsSoA.h"

namespace reco {

  template <typename TDev>
  using OTRecHitPortableCollectionDevice = PortableDeviceCollection<TDev, reco::OTRecHitBlocksSoA>;

  template <typename TDev>
  class OTRecHitsDevice : public OTRecHitPortableCollectionDevice<TDev> {
  public:
    OTRecHitsDevice() = default;

    OTRecHitsDevice(edm::Uninitialized) : OTRecHitPortableCollectionDevice<TDev>{edm::kUninitialized} {}

    // Constructor which specifies the SoA sizes
    // nHits: number of OT hits (from all sensors)
    // nModules: number of stacked modules (the SoA will have nModules+1 elements for cumulative sum)
    template <typename TQueue>
    explicit OTRecHitsDevice(TQueue queue, uint32_t nHits, uint32_t nModules)
        : OTRecHitPortableCollectionDevice<TDev>(queue, nHits, nModules + 1) {}

    // Number of OT hits in the collection
    uint32_t nHits() const { return this->view().otRecHits().metadata().size(); }

    // Number of stacked modules
    uint32_t nModules() const { return this->view().otHitModules().metadata().size() - 1; }

    // Do nothing for update from device (data already on device)
    template <typename TQueue>
    void updateFromDevice(TQueue) {}
  };

}  // namespace reco

#endif  // DataFormats_TrackingRecHitSoA_interface_OTRecHitsDevice_h
