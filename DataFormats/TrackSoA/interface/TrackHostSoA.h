#ifndef DataFormats_Track_TrackHostSoA_H
#define DataFormats_Track_TrackHostSoA_H

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "Geometry/CommonTopologies/interface/SimplePixelStripTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/TrackSoA/interface/TrackLayout.h"
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

// TODO: The class is created via inheritance of the PortableHostCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306
template <typename TrackerTraits>
class TrackHostSoA : public PortableHostCollection<TrackLayout<TrackerTraits>> {
public:
  static constexpr int32_t S = TrackerTraits::maxNumberOfTuples;  //TODO: this could be made configurable at runtime
  TrackHostSoA() = default;  // Needed for the dictionary; not sure if line above is needed anymore

  using PortableHostCollection<TrackLayout<TrackerTraits>>::view;
  using PortableHostCollection<TrackLayout<TrackerTraits>>::const_view;
  using PortableHostCollection<TrackLayout<TrackerTraits>>::buffer;

  // Constructor which specifies the SoA size
  template <typename TQueue>
  explicit TrackHostSoA<TrackerTraits>(TQueue queue) : PortableHostCollection<TrackLayout<TrackerTraits>>(S, queue) {}

  // Constructor which specifies the DevHost
  explicit TrackHostSoA(alpaka_common::DevHost const& host)
      : PortableHostCollection<TrackLayout<TrackerTraits>>(S, host) {}
};

namespace pixelTrack {

  using TrackHostSoAPhase1 = TrackHostSoA<pixelTopology::Phase1>;
  using TrackHostSoAPhase2 = TrackHostSoA<pixelTopology::Phase2>;
  using TrackHostSoAHIonPhase1 = TrackHostSoA<pixelTopology::HIonPhase1>;
  using TrackHostSoAPhase1Strip = TrackHostSoA<pixelTopology::Phase1Strip>;

}  // namespace pixelTrack

#endif  // DataFormats_Track_TrackHostSoA_H
