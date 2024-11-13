#ifndef DataFormats_Track_TracksHost_H
#define DataFormats_Track_TracksHost_H

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

namespace reco
{
    using TracksHost = PortableHostMultiCollection<TrackSoA, TrackHitSoA>;
}
// template <typename TrackerTraits>
// class TracksHost : public PortableHostMultiCollection<reco::TrackLayout<TrackerTraits>, reco::TrackHitLayout<TrackerTraits> > {
// public:
//   static constexpr int32_t S = TrackerTraits::maxNumberOfTuples;  //TODO: this could be made configurable at runtime
//   static constexpr int32_t H = TrackerTraits::avgHitsPerTrack;
//   TracksHost() = default;  // Needed for the dictionary;

//   using PortableHostMultiCollection<reco::TrackLayout<TrackerTraits>, reco::TrackHitLayout<TrackerTraits> >::view;
//   using PortableHostMultiCollection<reco::TrackLayout<TrackerTraits>, reco::TrackHitLayout<TrackerTraits> >::const_view;
//   using PortableHostMultiCollection<reco::TrackLayout<TrackerTraits>, reco::TrackHitLayout<TrackerTraits> >::buffer;

//   // Constructor which specifies the SoA size
//   template <typename TQueue>
//   explicit TracksHost<TrackerTraits>(TQueue& queue)
//       : PortableHostMultiCollection<reco::TrackLayout<TrackerTraits>, reco::TrackHitLayout<TrackerTraits> >({{S,H*S}}, queue) {}

//   // Constructor which specifies the DevHost
//   explicit TracksHost(alpaka_common::DevHost const& host)
//       : PortableHostMultiCollection<reco::TrackLayout<TrackerTraits>, reco::TrackHitLayout<TrackerTraits> >({{S,H*S}}, host) {}
// };

// namespace pixelTrack {

//   using TracksHostPhase1 = TracksHost<pixelTopology::Phase1>;
//   using TracksHostPhase2 = TracksHost<pixelTopology::Phase2>;
//   using TracksHostHIonPhase1 = TracksHost<pixelTopology::HIonPhase1>;

// }  // namespace pixelTrack

// using TrackPortableCollectionHostPhase1 = PortableHostCollection2< reco::TrackLayout<pixelTopology::Phase1> , reco::TrackHitLayout<pixelTopology::Phase1> >;
// using TrackPortableCollectionHostPhase2 = PortableHostCollection2< reco::TrackLayout<pixelTopology::Phase2> , reco::TrackHitLayout<pixelTopology::Phase2> >;

#endif  // DataFormats_Track_TracksHost_H
