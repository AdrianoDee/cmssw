#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAHitMaskingAndMergerKernelsImpl_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAHitMaskingAndMergerKernelsImpl_h

// #define GPU_DEBUG
// #define NTUPLE_DEBUG
// #define CA_DEBUG
// #define CA_WARNINGS

// C++ includes
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <type_traits>

// Alpaka includes
#include <alpaka/alpaka.hpp>

// CMSSW includes
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h"
#include "HeterogeneousCore/AlpakaInterface/interface/AtomicPairCounter.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "RecoTracker/PixelSeeding/interface/CAPairSoA.h"

// local includes
#include "CACell.h"
#include "CAHitMaskingAndMergerKernels.h"
#include "CAStructures.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::caHitMaskingAndMergerKernels {

  // using namespace ::caStructures;

  // constexpr uint32_t tkNotFound = std::numeric_limits<uint32_t>::max();
  // constexpr float maxScore = std::numeric_limits<float>::max();
  // constexpr float nSigma2Phase1 = 25.f;
  // constexpr float nSigma2 = 5.f;
  // constexpr int nTrackParameters = 5;
  // // map: index of a track parameter -> index of its covariance
  // HOST_DEVICE_CONSTANT std::array<uint8_t, nTrackParameters> iParam2iCov = {0u, 5u, 9u, 12u, 14u};

  // all of these below are mostly to avoid carrying around the relative namespace

  // using Quality = ::pixelTrack::Quality;
  // using TkSoAView = ::reco::TrackSoAView;
  // using TkHitSoAView = ::reco::TrackHitSoAView;

  // template <typename TrackerTraits>
  // using QualityCuts = ::pixelTrack::QualityCutsT<TrackerTraits>;

  // using HitToTuple = caStructures::GenericContainer;
  // using HitContainer = caStructures::SequentialContainer;
  // using TupleMultiplicity = caStructures::GenericContainer;
  // using HitToCell = caStructures::GenericContainer;
  // using CellToCell = caStructures::GenericContainer;
  // using CellToTrack = caStructures::GenericContainer;

  using namespace cms::alpakatools;

  class Kernel_updateMasking {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  ::reco::TrackingRecHitsMaskingView mask_view,
                                  const ::reco::TrackSoAConstView &trackd_view,
                                  const ::reco::TrackHitSoAConstView &trackhitd_view,
                                  const pixelTrack::Quality minQuality,
                                  uint32_t const &iterationIndex) const {
      if (alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0] == 0) {
        int getHit = 0;
        for (int j = 0; j < int(trackd_view.nTracks()); ++j) {
          getHit = getHit + ::reco::nHits(trackd_view, j);

          if (trackd_view[j].quality() < minQuality)
            continue;

          for (uint32_t k = 0; k < uint32_t(::reco::nHits(trackd_view, j)); ++k) {
            mask_view[trackhitd_view[getHit - k - 1].id()].recHitMask() = iterationIndex;
          }
        }
      }
    }
  };

  class Kernel_updateHitOffsets {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  int const &tksBeg,
                                  int const &tksEnd,
                                  int const &nHits,
                                  ::reco::TrackSoAView trackd_view) const {
      if (alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0] == 0) {
        for (int i = tksBeg; i < tksEnd; ++i) {
          trackd_view[i].hitOffsets() += nHits;
        }
      }
      alpaka::syncBlockThreads(acc);
    }
  };

  class Kernel_filterTracks {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  ::reco::TrackSoAView track_view,
                                  ::reco::TrackHitSoAView trackHit_view,
                                  const ::reco::TrackSoAConstView &inpTrack_view,
                                  const ::reco::TrackHitSoAConstView &inpTrackHit_view,
                                  const pixelTrack::Quality minQuality,
                                  const double matchFraction) const {
      if (alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0] == 0) {
        uint32_t auxOutputTkIndex = 0;
        uint32_t auxOutputHitIndex = 0;

        for (uint32_t i = 0; i < uint32_t(inpTrack_view.metadata().size()); ++i) {
          if (inpTrack_view[i].quality() < minQuality)
            continue;

          bool hasDuplicate = false;
          for (uint32_t j : cms::alpakatools::uniform_elements_x(acc, inpTrack_view.metadata().size())) {
            if (j < i + 1)
              continue;
            if (inpTrack_view[j].quality() < minQuality)
              continue;

            if (::reco::nHits(inpTrack_view, i) == ::reco::nHits(inpTrack_view, j)) {
              uint32_t matchedHits = 0;
              for (uint32_t k = 0; k < uint32_t(::reco::nHits(inpTrack_view, i)); ++k) {
                uint32_t auxHitOffsetsId = 0;
                if (i > 0)
                  auxHitOffsetsId = inpTrack_view[i - 1].hitOffsets();
                if (inpTrackHit_view[auxHitOffsetsId + k].id() ==
                    inpTrackHit_view[inpTrack_view[j - 1].hitOffsets() + k].id())
                  ++matchedHits;
              }
              if (double(matchedHits) / double(::reco::nHits(inpTrack_view, i)) > matchFraction)
                hasDuplicate = true;
            }

            if (hasDuplicate)
              break;
          }
          alpaka::syncBlockThreads(acc);

          if (hasDuplicate)
            continue;

          track_view[auxOutputTkIndex].quality() = inpTrack_view[i].quality();
          track_view[auxOutputTkIndex].chi2() = inpTrack_view[i].chi2();
          track_view[auxOutputTkIndex].nLayers() = inpTrack_view[i].nLayers();
          track_view[auxOutputTkIndex].eta() = inpTrack_view[i].eta();
          track_view[auxOutputTkIndex].pt() = inpTrack_view[i].pt();
          for (uint32_t k = 0; k < 5; ++k)
            track_view[auxOutputTkIndex].state()[k] = inpTrack_view[i].state()[k];
          for (uint32_t k = 0; k < 15; ++k)
            track_view[auxOutputTkIndex].covariance()[k] = inpTrack_view[i].covariance()[k];
          if (auxOutputTkIndex != 0) {
            track_view[auxOutputTkIndex].hitOffsets() =
                track_view[auxOutputTkIndex - 1].hitOffsets() + ::reco::nHits(inpTrack_view, i);
          } else {
            track_view[auxOutputTkIndex].hitOffsets() = ::reco::nHits(inpTrack_view, i);
          }

          uint32_t auxHitOffsetsIdBegin = 0;
          if (i > 0)
            auxHitOffsetsIdBegin = inpTrack_view[i - 1].hitOffsets();

          uint32_t auxHitOffsetsIdEnd = inpTrack_view[i].hitOffsets();
          if (i > 0)
            auxHitOffsetsIdEnd = inpTrack_view[i].hitOffsets();

          for (uint32_t k = auxHitOffsetsIdBegin; k < auxHitOffsetsIdEnd; ++k) {
            trackHit_view[auxOutputHitIndex].id() = inpTrackHit_view[k].id();
            trackHit_view[auxOutputHitIndex].detId() = inpTrackHit_view[k].detId();
            ++auxOutputHitIndex;
          }

          ++auxOutputTkIndex;
        }
        alpaka::syncBlockThreads(acc);
        track_view.nTracks() = auxOutputTkIndex;
      }
    }
  };


}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::caHitMaskingAndMergerKernels

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAHitMaskingAndMergerKernelsImpl_h
