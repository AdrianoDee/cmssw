#ifndef DataFormats_TrackingRecHitSoA_interface_StubsDevice_h
#define DataFormats_TrackingRecHitSoA_interface_StubsDevice_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsSoA.h"

// TODO: The class is created via inheritance of the PortableCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306

namespace reco {

  template <typename TDev>
  using StubPortableCollectionDevice = PortableDeviceCollection<TDev, reco::StubBlocksSoA>;

  template <typename TDev>
  class StubsDevice : public StubPortableCollectionDevice<TDev> {
  public:
    StubsDevice() = default;

    StubsDevice(edm::Uninitialized) : StubPortableCollectionDevice<TDev>{edm::kUninitialized} {}

    // Constructor which specifies the SoA size
    // nStubs: number of stubs
    // nModules: number of stacked modules
    //
    // N.B. why nModules + 1? Because the StubModulesLayout holds the
    // moduleStart vector that is a cumulative sum of all the stubs
    // in each module. The extra element (the last one) is used to hold
    // the total number of stubs. We are "hiding" this in the constructor
    // so that one can build the Stubs class in a more natural way,
    // just using the number of needed modules.
    template <typename TQueue>
    explicit StubsDevice(TQueue queue, uint32_t nStubs, uint32_t nModules)
        : StubPortableCollectionDevice<TDev>(queue, nStubs, nModules + 1) {}

    // Number of stubs in the collection
    uint32_t nStubs() const { return this->view().stubs().metadata().size(); }

    // Number of stacked modules
    uint32_t nModules() const { return this->view().stubModules().metadata().size() - 1; }

    // Offset to first stub (after pixel hits in unified indexing)
    int32_t offsetStubs() const { return offsetStubs_; }

    // Asynchronously update the information cached within the class itself
    // from the information on the device
    template <typename TQueue>
    void updateFromDevice(TQueue queue) {
      auto off_h = cms::alpakatools::make_host_view(offsetStubs_);
      auto off_d = cms::alpakatools::make_device_view(queue, this->view().stubs().offsetStubs());
      alpaka::memcpy(queue, off_h, off_d);
    }

  private:
    // offsetStubs is used on host functions so is useful to have it also
    // stored in the class and not only in the layout
    int32_t offsetStubs_ = 0;
  };

}  // namespace reco

#endif  // DataFormats_TrackingRecHitSoA_interface_StubsDevice_h
