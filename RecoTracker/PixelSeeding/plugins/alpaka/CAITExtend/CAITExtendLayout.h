#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAITExtend_CAITExtendLayout_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAITExtend_CAITExtendLayout_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/HostDeviceConstant.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace caITExtend {

  // Geometric extent of an IT layer used by the helix-crossing box test.
  // Barrel layer: r is nominal radius; (zMin, zMax) are barrel half-length extents.
  // Disk layer:   r is unused; (zMin, zMax) is a narrow z slab around the disk plane,
  //               (rMin, rMax) bounds the active radial range of the disk.
  struct ITLayerExtent {
    float r;
    float zMin;
    float zMax;
    float rMin;
    float rMax;
    uint8_t caLayerIdx;
    bool isBarrel;
  };

  // Phase2-specific IT layout. Encapsulated as a traits-keyed struct so a future
  // Phase1 specialisation only needs a different `Layout<TrackerTraits>`.
  template <typename TrackerTraits>
  struct Layout;

  struct Phase2OTStubsTag {};

  // The extent tables live at namespace scope inside `detail::` so they can use
  // HOST_DEVICE_CONSTANT (which expands to `__device__ constexpr` on device).
  // Static `constexpr` class members do NOT work in CUDA device code -- the
  // device-side compiler cannot resolve the symbol's address.
  //
  // Numbers are approximate Phase2 geometry; the (Delta phi, Delta z) windows
  // from the helix covariance set the actual cut.  See task #13 for the
  // follow-up to derive these from CMSSW geometry records.
  namespace detail {

    HOST_DEVICE_CONSTANT ITLayerExtent kPhase2BarrelExtents[4] = {
        // r,    zMin,  zMax,   rMin, rMax, caIdx, isBarrel
        {16.0f, -275.f, 275.f, 0.f, 0.f, 3, true},
        {11.0f, -275.f, 275.f, 0.f, 0.f, 2, true},
        { 7.0f, -275.f, 275.f, 0.f, 0.f, 1, true},
        { 3.0f, -275.f, 275.f, 0.f, 0.f, 0, true},
    };

    HOST_DEVICE_CONSTANT ITLayerExtent kPhase2ForwardExtents[12] = {
        {0.f, 265.f, 275.f, 4.f, 25.f, 15, false},
        {0.f, 240.f, 250.f, 4.f, 25.f, 14, false},
        {0.f, 215.f, 225.f, 4.f, 25.f, 13, false},
        {0.f, 190.f, 200.f, 4.f, 25.f, 12, false},
        {0.f, 165.f, 175.f, 4.f, 25.f, 11, false},
        {0.f, 140.f, 150.f, 4.f, 25.f, 10, false},
        {0.f, 115.f, 125.f, 4.f, 25.f,  9, false},
        {0.f,  90.f, 100.f, 4.f, 25.f,  8, false},
        {0.f,  70.f,  80.f, 4.f, 18.f,  7, false},
        {0.f,  55.f,  65.f, 4.f, 18.f,  6, false},
        {0.f,  42.f,  52.f, 4.f, 18.f,  5, false},
        {0.f,  30.f,  40.f, 4.f, 18.f,  4, false},
    };

    HOST_DEVICE_CONSTANT ITLayerExtent kPhase2BackwardExtents[12] = {
        {0.f, -275.f, -265.f, 4.f, 25.f, 27, false},
        {0.f, -250.f, -240.f, 4.f, 25.f, 26, false},
        {0.f, -225.f, -215.f, 4.f, 25.f, 25, false},
        {0.f, -200.f, -190.f, 4.f, 25.f, 24, false},
        {0.f, -175.f, -165.f, 4.f, 25.f, 23, false},
        {0.f, -150.f, -140.f, 4.f, 25.f, 22, false},
        {0.f, -125.f, -115.f, 4.f, 25.f, 21, false},
        {0.f, -100.f,  -90.f, 4.f, 25.f, 20, false},
        {0.f,  -80.f,  -70.f, 4.f, 18.f, 19, false},
        {0.f,  -65.f,  -55.f, 4.f, 18.f, 18, false},
        {0.f,  -52.f,  -42.f, 4.f, 18.f, 17, false},
        {0.f,  -40.f,  -30.f, 4.f, 18.f, 16, false},
    };

  }  // namespace detail

  template <>
  struct Layout<Phase2OTStubsTag> {
    // Counts (compile-time scalars are fine as class statics).
    static constexpr int kNBarrel = 4;
    static constexpr int kNForward = 12;
    static constexpr int kNBackward = 12;
    static constexpr int kNDiskSidePerTrack = kNForward;
    static constexpr int kMaxITLayersPerTrack = kNBarrel + kNDiskSidePerTrack;

    static constexpr uint16_t kMaxCandidatesPerTrackLayer = 8;
    static constexpr uint16_t kMaxNewLayers = 4;

    // Device-callable accessors return pointers to the namespace-scope arrays.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static ITLayerExtent const* barrel() {
      return detail::kPhase2BarrelExtents;
    }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static ITLayerExtent const* forward() {
      return detail::kPhase2ForwardExtents;
    }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static ITLayerExtent const* backward() {
      return detail::kPhase2BackwardExtents;
    }
  };

}  // namespace caITExtend

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAITExtend_CAITExtendLayout_h
