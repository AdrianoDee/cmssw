#ifndef CUDADataFormats_Track_TrackHeterogeneousDevice_H
#define CUDADataFormats_Track_TrackHeterogeneousDevice_H

#include <cstdint>

#include "CUDADataFormats/Track/interface/PixelTrackUtilities.h"
#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"

template <int32_t S>
class TrackSoAHeterogeneousDevice : public cms::cuda::PortableDeviceCollection<TrackSoAHeterogeneousLayout<>> {
public:
  TrackSoAHeterogeneousDevice() = default;  // cms::cuda::Product needs this

  // Constructor which specifies the SoA size
  explicit TrackSoAHeterogeneousDevice(cudaStream_t stream)
      : PortableDeviceCollection<TrackSoAHeterogeneousLayout<>>(S, stream) {}
};

namespace pixelTrack {

  using TrackSoADevice = TrackSoAHeterogeneousDevice<pixelTrack::maxNumber()>;

}  // namespace pixelTrack

#endif  // CUDADataFormats_Track_TrackHeterogeneousT_H
