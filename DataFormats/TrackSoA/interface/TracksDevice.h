#ifndef DataFormats_Track_interface_TracksDevice_h
#define DataFormats_Track_interface_TracksDevice_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/TracksSoA.h"

namespace reco
{
  template <typename TDev>
  using TracksDevice = PortableDeviceMultiCollection<TDev, TrackSoA, TrackHitSoA>;
}

// template <typename TrackerTraits, typename TDev>
// class TracksDevice : public PortableDeviceMultiCollection<TDev, reco::TrackLayout<TrackerTraits>,   reco::TrackHitLayout<TrackerTraits>> {
// public:
//   static constexpr int32_t S = TrackerTraits::maxNumberOfTuples;  //TODO: this could be made configurable at runtime
//   static constexpr int32_t H = TrackerTraits::avgHitsPerTrack;

//   TracksDevice() = default;                                       // necessary for ROOT dictionaries

//   using PortableDeviceMultiCollection<TDev, reco::TrackLayout<TrackerTraits>,   reco::TrackHitLayout<TrackerTraits>>::view;
//   using PortableDeviceMultiCollection<TDev, reco::TrackLayout<TrackerTraits>,   reco::TrackHitLayout<TrackerTraits>>::const_view;
//   using PortableDeviceMultiCollection<TDev, reco::TrackLayout<TrackerTraits>,   reco::TrackHitLayout<TrackerTraits>>::buffer;

//   // Constructor which specifies the SoA size
//   template <typename TQueue>
//   explicit TracksDevice<TrackerTraits, TDev>(TQueue& queue)
//       : PortableDeviceMultiCollection<TDev, reco::TrackLayout<TrackerTraits>,   reco::TrackHitLayout<TrackerTraits>>({{S,H}}, queue) {}
// };

// namespace pixelTrack {

//   template <typename TDev>
//   using TracksDevicePhase1 = TracksDevice<pixelTopology::Phase1, TDev>;
//   template <typename TDev>
//   using TracksDeviceHIonPhase1 = TracksDevice<pixelTopology::HIonPhase1, TDev>;
//   template <typename TDev>
//   using TracksDevicePhase2 = TracksDevice<pixelTopology::Phase2, TDev>;

// }  // namespace pixelTrack

#endif  // DataFormats_Track_TracksDevice_H
