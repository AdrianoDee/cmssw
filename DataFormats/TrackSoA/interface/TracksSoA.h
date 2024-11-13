#ifndef DataFormats_TrackSoA_interface_TracksSoA_h
#define DataFormats_TrackSoA_interface_TracksSoA_h

#include <alpaka/alpaka.hpp>

#include <Eigen/Core>

#include "HeterogeneousCore/AlpakaInterface/interface/OneToManyAssoc.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"

namespace reco {

    // Aliases in order to not confuse the GENERATE_SOA_LAYOUT
    // macro with weird colons and angled brackets.
    using Vector5f = Eigen::Matrix<float, 5, 1>;
    using Vector15f = Eigen::Matrix<float, 15, 1>;
    using Quality = pixelTrack::Quality;

    GENERATE_SOA_LAYOUT(TrackLayout,
                        SOA_COLUMN(Quality, quality),
                        SOA_COLUMN(float, chi2),
                        SOA_COLUMN(int8_t, nLayers),
                        SOA_COLUMN(float, eta),
                        SOA_COLUMN(float, pt),
                        // state at the beam spot: {phi, tip, 1/pt, cotan(theta), zip}
                        SOA_EIGEN_COLUMN(Vector5f, state),
                        SOA_EIGEN_COLUMN(Vector15f, covariance),
                        SOA_SCALAR(int, nTracks),
                        SOA_COLUMN(uint32_t, hitOffsets))
                        
    GENERATE_SOA_LAYOUT(TrackHitsLayout,
                        SOA_COLUMN(uint32_t, id),
                        SOA_COLUMN(uint32_t, detId))
  
  using TrackSoA = TrackLayout<>;
  using TrackSoAView = TrackSoA::View;
  using TrackSoAConstView = TrackSoA::ConstView;

  using TrackHitSoA = TrackHitsLayout<>;
  using TrackHitSoAView = TrackHitSoA::View;
  using TrackHitSoAConstView = TrackHitSoA::ConstView;

  // If constexpr we get this:
  // note: non-literal type 'reco::TrackLayout<128, false>::ConstViewTemplateFreeParams<128, false, true, true>::const_element' 
  // cannot be used in a constant expression
  
  // move to use the layer gaps defined in CAParams
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static int nLayers(const TrackSoAConstView &tracks,
                                                                                 const TrackHitSoAConstView &hits,
                                                                                 uint16_t maxLayers,
                                                                                 uint32_t const* __restrict__ layerStarts,
                                                                                 int32_t i) {
    auto start = (i==0) ? 0 : tracks[i-1].hitOffsets();
    auto end = tracks[i].hitOffsets();
    auto hitId = hits[start].id();
    int nl = 1;
    auto ol = 0;
    while ( hitId >= layerStarts[ol+1] and ol < maxLayers)
      ++ol;
    ++start;
    for (; start < end; ++start) {
      hitId = hits[start].id();
      auto il = 0;
      while ( hitId >= layerStarts[ol+1] and ol < maxLayers)
        ++il;
      if (il != ol)
        ++nl;
      ol = il;
    }
    return nl;
  }

  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float charge(const TrackSoAConstView &tracks, int32_t i) {
    //was: std::copysign(1.f, tracks[i].state()(2)). Will be constexpr with C++23
    float v = tracks[i].state()(2);
    return float((0.0f < v) - (v < 0.0f));
  }

  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float phi(const TrackSoAConstView &tracks, int32_t i) {
    return tracks[i].state()(0);
  }

  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float tip(const TrackSoAConstView &tracks, int32_t i) {
    return tracks[i].state()(1);
  }

  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float zip(const TrackSoAConstView &tracks, int32_t i) {
    return tracks[i].state()(4);
  }

  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool isTriplet(const TrackSoAConstView &tracks, int32_t i) {
    return tracks[i].nLayers() == 3;
  }

}  // namespace reco

#endif  // DataFormats_TrackSoA_interface_TracksSoA_h
