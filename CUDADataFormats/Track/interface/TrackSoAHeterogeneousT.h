#ifndef CUDADataFormats_Track_TrackHeterogeneousT_H
#define CUDADataFormats_Track_TrackHeterogeneousT_H

#include <string>
#include <algorithm>

#include "CUDADataFormats/Track/interface/TrajectoryStateSoAT.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "CUDADataFormats/Common/interface/HeterogeneousSoA.h"

namespace pixelTrack {
  enum class Quality : uint8_t { bad = 0, edup, dup, loose, strict, tight, highPurity, notQuality };
  constexpr uint32_t qualitySize{uint8_t(Quality::notQuality)};
  const std::string qualityName[qualitySize]{"bad", "edup", "dup", "loose", "strict", "tight", "highPurity"};
  inline Quality qualityByName(std::string const &name) {
    auto qp = std::find(qualityName, qualityName + qualitySize, name) - qualityName;
    return static_cast<Quality>(qp);
  }
}  // namespace pixelTrack

  template <typename TrackerTraits>
  class TrackSoAHeterogeneousT {
  public:
  static constexpr int32_t S = TrackerTraits::maxNumberOfTuples;
  static constexpr int32_t H = TrackerTraits::maxHitsOnTrack;
  static constexpr int32_t stride() { return S; }

  using Quality = pixelTrack::Quality;
  using hindex_type = uint32_t;
  using HitContainer = cms::cuda::OneToManyAssoc<hindex_type, S + 1, H * S>; // TODO plot for average number of hits

  // Always check quality is at least loose!
  // CUDA does not support enums  in __lgc ...
private:
  eigenSoA::ScalarSoA<uint8_t, S> quality_;

public:
  constexpr Quality quality(int32_t i) const { return (Quality)(quality_(i)); }
  constexpr Quality &quality(int32_t i) { return (Quality &)(quality_(i)); }
  constexpr Quality const *qualityData() const { return (Quality const *)(quality_.data()); }
  constexpr Quality *qualityData() { return (Quality *)(quality_.data()); }

  // this is chi2/ndof as not necessarely all hits are used in the fit
  eigenSoA::ScalarSoA<float, S> chi2;

  eigenSoA::ScalarSoA<int8_t, S> nLayers;

  constexpr int nHits(int i) const { return detIndices.size(i); }

  constexpr bool isTriplet(int i) const { return nLayers(i) == 3; }

  constexpr int computeNumberOfLayers(int32_t i) const {
    // layers are in order and we assume tracks are either forward or backward
    auto pdet = detIndices.begin(i);
    int nl = 1;
    auto ol = pixelTopology::getLayer<TrackerTraits>(*pdet);
    for (; pdet < detIndices.end(i); ++pdet) {
      auto il = pixelTopology::getLayer<TrackerTraits>(*pdet);
      if (il != ol)
        ++nl;
      ol = il;
    }
    return nl;
  }

  // State at the Beam spot
  // phi,tip,1/pt,cotan(theta),zip
  TrajectoryStateSoAT<S> stateAtBS;
  eigenSoA::ScalarSoA<float, S> eta;
  eigenSoA::ScalarSoA<float, S> pt;
  constexpr float charge(int32_t i) const { return std::copysign(1.f, stateAtBS.state(i)(2)); }
  constexpr float phi(int32_t i) const { return stateAtBS.state(i)(0); }
  constexpr float tip(int32_t i) const { return stateAtBS.state(i)(1); }
  constexpr float zip(int32_t i) const { return stateAtBS.state(i)(4); }

  // state at the detector of the outermost hit
  // representation to be decided...
  // not yet filled on GPU
  // TrajectoryStateSoA<S> stateAtOuterDet;

  HitContainer hitIndices;
  HitContainer detIndices;
};


template class TrackSoAHeterogeneousT<pixelTopology::Phase1>;
template class TrackSoAHeterogeneousT<pixelTopology::Phase2>;

namespace pixelTrack {

  template <typename TrackerTraits>
  using TrackSoAT = TrackSoAHeterogeneousT<TrackerTraits>;

  template <typename TrackerTraits>
  using HitContainerT = typename TrackSoAT<TrackerTraits>::HitContainer;

  typedef TrackSoAT<pixelTopology::Phase1> TrackSoA;
  using TrajectoryState = TrajectoryStateSoAT<TrackSoA::S>;
  using HitContainer = TrackSoA::HitContainer;

  typedef TrackSoAT<pixelTopology::Phase2> TrackSoAPhase2;
  using TrajectoryStatePhase2 = TrajectoryStateSoAT<TrackSoAPhase2::S>;
  using HitContainerPhase2 = TrackSoAPhase2::HitContainer;

}  // namespace pixelTrack


#endif  // CUDADataFormats_Track_TrackHeterogeneousT_H
