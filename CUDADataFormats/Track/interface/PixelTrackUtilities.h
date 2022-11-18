#ifndef CUDADataFormats_Track_PixelTrackUtilities_h
#define CUDADataFormats_Track_PixelTrackUtilities_h

#include <Eigen/Dense>
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace pixelTrack {
  enum class Quality : uint8_t { bad = 0, edup, dup, loose, strict, tight, highPurity, notQuality };
  constexpr uint32_t qualitySize{uint8_t(Quality::notQuality)};
  const std::string qualityName[qualitySize]{"bad", "edup", "dup", "loose", "strict", "tight", "highPurity"};
  inline Quality qualityByName(std::string const &name) {
    auto qp = std::find(qualityName, qualityName + qualitySize, name) - qualityName;
    return static_cast<Quality>(qp);
  }

  // Maximum number of Tracks to be stored,
  // it's used to initialize Portable Collections
  // to a specific size
#ifdef GPU_SMALL_EVENTS
  // kept for testing and debugging
  constexpr uint32_t maxNumber() { return 2 * 1024; }
#else
  // tested on MC events with 55-75 pileup events
  constexpr uint32_t maxNumber() { return 32 * 1024; }
#endif

  using HitContainer = cms::cuda::OneToManyAssoc<uint32_t, pixelTrack::maxNumber() + 1, 5 * pixelTrack::maxNumber()>;

}  // namespace pixelTrack

// Aliases in order to not confuse the GENERATE_SOA_LAYOUT
// macro with weird colons and angled brackets.
using Vector5f = Eigen::Matrix<float, 5, 1>;
using Vector15f = Eigen::Matrix<float, 15, 1>;
using HitContainer = pixelTrack::HitContainer;
using Quality = pixelTrack::Quality;

GENERATE_SOA_LAYOUT(TrackSoAHeterogeneousLayout,
                    SOA_COLUMN(Quality, quality),
                    SOA_COLUMN(float, chi2),
                    SOA_COLUMN(int8_t, nLayers),
                    SOA_COLUMN(float, eta),
                    SOA_COLUMN(float, pt),
                    SOA_EIGEN_COLUMN(Vector5f, state),
                    SOA_EIGEN_COLUMN(Vector15f, covariance),
                    SOA_SCALAR(int, nTracks),
                    SOA_SCALAR(HitContainer, hitIndices),
                    SOA_SCALAR(HitContainer, detIndices))

// Methods that operate on View and ConstView of the TrackSoA, and cannot be class methods.
namespace pixelTrack {
  namespace utilities {
    using TrackSoAView = TrackSoAHeterogeneousLayout<>::View;
    using TrackSoAConstView = TrackSoAHeterogeneousLayout<>::ConstView;
    using hindex_type = uint32_t;
    // State at the Beam spot
    // phi,tip,1/pt,cotan(theta),zip
    __host__ __device__ inline float charge(const TrackSoAConstView &tracks, int32_t i) {
      return std::copysign(1.f, tracks[i].state()(2));
    }

    __host__ __device__ inline float phi(const TrackSoAConstView &tracks, int32_t i) { return tracks[i].state()(0); }

    __host__ __device__ inline float tip(const TrackSoAConstView &tracks, int32_t i) { return tracks[i].state()(1); }

    __host__ __device__ inline float zip(const TrackSoAConstView &tracks, int32_t i) { return tracks[i].state()(4); }

    __host__ __device__ inline bool isTriplet(const TrackSoAConstView &tracks, int i) {
      return tracks[i].nLayers() == 3;
    }

    template <typename V3, typename M3, typename V2, typename M2>
    __host__ __device__ inline void copyFromCircle(
        TrackSoAView &tracks, V3 const &cp, M3 const &ccov, V2 const &lp, M2 const &lcov, float b, int32_t i) {
      tracks[i].state() << cp.template cast<float>(), lp.template cast<float>();

      tracks[i].state()(2) = tracks[i].state()(2) * b;
      auto cov = tracks[i].covariance();
      cov(0) = ccov(0, 0);
      cov(1) = ccov(0, 1);
      cov(2) = b * float(ccov(0, 2));
      cov(4) = cov(3) = 0;
      cov(5) = ccov(1, 1);
      cov(6) = b * float(ccov(1, 2));
      cov(8) = cov(7) = 0;
      cov(9) = b * b * float(ccov(2, 2));
      cov(11) = cov(10) = 0;
      cov(12) = lcov(0, 0);
      cov(13) = lcov(0, 1);
      cov(14) = lcov(1, 1);
    }

    template <typename V5, typename M5>
    __host__ __device__ inline void copyFromDense(TrackSoAView &tracks, V5 const &v, M5 const &cov, int32_t i) {
      tracks[i].state() = v.template cast<float>();
      for (int j = 0, ind = 0; j < 5; ++j)
        for (auto k = j; k < 5; ++k)
          tracks[i].covariance()(ind++) = cov(j, k);
    }

    template <typename V5, typename M5>
    __host__ __device__ inline void copyToDense(const TrackSoAConstView &tracks, V5 &v, M5 &cov, int32_t i) {
      v = tracks[i].state().template cast<typename V5::Scalar>();
      for (int j = 0, ind = 0; j < 5; ++j) {
        cov(j, j) = tracks[i].covariance()(ind++);
        for (auto k = j + 1; k < 5; ++k)
          cov(k, j) = cov(j, k) = tracks[i].covariance()(ind++);
      }
    }

    // TODO: Not using TrackSoAConstView due to weird bugs with HitContainer
    __host__ __device__ inline int computeNumberOfLayers(TrackSoAView &tracks, int32_t i) {
      auto pdet = tracks.detIndices().begin(i);
      int nl = 1;
      auto ol = phase1PixelTopology::getLayer(*pdet);
      for (; pdet < tracks.detIndices().end(i); ++pdet) {
        auto il = phase1PixelTopology::getLayer(*pdet);
        if (il != ol)
          ++nl;
        ol = il;
      }
      return nl;
    }
    __host__ __device__ inline int nHits(const TrackSoAConstView &tracks, int i) { return tracks.detIndices().size(i); }

  }  // namespace utilities
}  // namespace pixelTrack

namespace pixelTrack {
  // Common types for both Host and Device code
  using TrackSoALayout = TrackSoAHeterogeneousLayout<>;
  using TrackSoAView = TrackSoAHeterogeneousLayout<>::View;
  using TrackSoAConstView = TrackSoAHeterogeneousLayout<>::ConstView;

}  // namespace pixelTrack

#endif
