#ifndef DataFormats_Track_interface_TracksDevice_h
#define DataFormats_Track_interface_TracksDevice_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"

// TODO: The class is created via inheritance of the PortableCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306

namespace reco
{
  template <typename TDev>
  using TracksDevice = PortableDeviceMultiCollection<TDev, TrackSoA, TrackHitSoA>;
}


#endif  // DataFormats_Track_TracksDevice_H
