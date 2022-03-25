#ifndef CUDADataFormats_Track_PixelTrackHeterogeneous_h
#define CUDADataFormats_Track_PixelTrackHeterogeneous_h

#include "CUDADataFormats/Common/interface/HeterogeneousSoA.h"
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousT.h"
#include "CUDADataFormats/TrackerGeometry/interface/SimplePixelTopology.h"

template <typename TrackerTraits>
using PixelTrackHeterogeneousT = HeterogeneousSoA<pixelTrack::TrackSoAT<TrackerTraits>>;

using PixelTrackHeterogeneous = PixelTrackHeterogeneousT<pixelTopology::Phase1>;
using PixelTrackHeterogeneousPhase2 = PixelTrackHeterogeneousT<pixelTopology::Phase2>;

#endif  // #ifndef CUDADataFormats_Track_PixelTrackHeterogeneous_h
