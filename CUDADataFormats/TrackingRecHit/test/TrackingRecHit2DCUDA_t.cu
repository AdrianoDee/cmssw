#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"

namespace testTrackingRecHit2D {

  template <typename TrackerTraits>
  __global__ void fill(TrackingRecHit2DSOAViewT<TrackerTraits>* phits) {
    assert(phits);
    auto& hits = *phits;
    assert(hits.nHits() == 200);

    int i = threadIdx.x;
    if (i > 200)
      return;
  }

  template <typename TrackerTraits>
  __global__ void verify(TrackingRecHit2DSOAViewT<TrackerTraits> const* phits) {
    assert(phits);
    auto const& hits = *phits;
    assert(hits.nHits() == 200);

    int i = threadIdx.x;
    if (i > 200)
      return;
  }

  template <typename TrackerTraits>
  void runKernels(TrackingRecHit2DSOAViewT<TrackerTraits>* hits) {
    assert(hits);
    fill<TrackerTraits><<<1, 1024>>>(hits);
    verify<TrackerTraits><<<1, 1024>>>(hits);
  }

}  // namespace testTrackingRecHit2D
