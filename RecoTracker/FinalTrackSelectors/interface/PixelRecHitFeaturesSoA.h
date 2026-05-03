#ifndef RecoTracker_FinalTrackSelectors_PixelRecHitFeaturesSoA_h
#define RecoTracker_FinalTrackSelectors_PixelRecHitFeaturesSoA_h

#include <Eigen/Core>
#include "HeterogeneousCore/AlpakaMath/interface/float16_t.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace RecHitFeatures {

  constexpr int MaxHitsPerTrack = 16;
  constexpr int HitFeatures = 6;

  struct HitFeature {
    static constexpr int x = 0;
    static constexpr int y = 1;
    static constexpr int z = 2;
    static constexpr int r = 3;
    static constexpr int eta = 4;
    static constexpr int phi = 5;
  };

  using HitMatrixRM = Eigen::Matrix<cms::float16_t, MaxHitsPerTrack, HitFeatures, Eigen::ColMajor>;

  GENERATE_SOA_LAYOUT(PixelRecHitFeaturesSoALayout, SOA_EIGEN_COLUMN(HitMatrixRM, hits));

  using PixelRecHitFeaturesSoA = PixelRecHitFeaturesSoALayout<>;

}  // namespace RecHitFeatures

#endif
