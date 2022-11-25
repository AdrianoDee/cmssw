//
// Original Author: Felice Pantaleo, CERN
//

#define NTUPLE_DEBUG
#define GPU_DEBUG

#include <cmath>
#include <cstdint>
#include <limits>

#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"
#include "CUDADataFormats/Track/interface/PixelTrackUtilities.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitsUtilities.h"

#include "CAConstants.h"
#include "CAHitNtupletGeneratorKernels.h"
#include "GPUCACell.h"
#include "gpuFishbone.h"
#include "gpuPixelDoublets.h"

// using HitsOnGPU = TrackingRecHit2DSOAView;
// using HitsOnCPU = TrackingRecHit2DGPU;

using HitSoAView = trackingRecHitSoA::HitSoAView;
using HitSoAConstView = trackingRecHitSoA::HitSoAConstView;
using HitToTuple = caConstants::HitToTuple;
using TupleMultiplicity = caConstants::TupleMultiplicity;

using Quality = pixelTrack::Quality;
using TkSoAView = pixelTrack::TrackSoAView;
using TkSoAConstView = pixelTrack::TrackSoAConstView;
using HitContainer = pixelTrack::HitContainer;

namespace {

  constexpr uint16_t tkNotFound = std::numeric_limits<uint16_t>::max();
  constexpr float maxScore = std::numeric_limits<float>::max();
  constexpr float nSigma2 = 25.f;

}  // namespace

__global__ void kernel_checkOverflows(TkSoAView tracks_view,
                                      caConstants::TupleMultiplicity const *tupleMultiplicity,
                                      CAHitNtupletGeneratorKernelsGPU::HitToTuple const *hitToTuple,
                                      cms::cuda::AtomicPairCounter *apc,
                                      GPUCACell const *__restrict__ cells,
                                      uint32_t const *__restrict__ nCells,
                                      gpuPixelDoublets::CellNeighborsVector const *cellNeighbors,
                                      gpuPixelDoublets::CellTracksVector const *cellTracks,
                                      GPUCACell::OuterHitOfCell const isOuterHitOfCell,
                                      int32_t nHits,
                                      uint32_t maxNumberOfDoublets,
                                      CAHitNtupletGeneratorKernelsGPU::Counters *counters) {
  auto first = threadIdx.x + blockIdx.x * blockDim.x;

  auto &c = *counters;
  // counters once per event
  if (0 == first) {
    atomicAdd(&c.nEvents, 1);
    atomicAdd(&c.nHits, nHits);
    atomicAdd(&c.nCells, *nCells);
    atomicAdd(&c.nTuples, apc->get().m);
    atomicAdd(&c.nFitTracks, tupleMultiplicity->size());
  }

#ifdef NTUPLE_DEBUG
  if (0 == first) {
    printf("number of found cells %d, found tuples %d with total hits %d out of %d %d\n",
           *nCells,
           apc->get().m,
           apc->get().n,
           nHits,
           hitToTuple->totOnes());
    if (apc->get().m < caConstants::maxNumberOfQuadruplets) {
      assert(tracks_view.hitIndices().size(apc->get().m) == 0);
      assert(tracks_view.hitIndices().size() == apc->get().n);
    }
  }

  for (int idx = first, nt = tracks_view.hitIndices().nOnes(); idx < nt; idx += gridDim.x * blockDim.x) {
    if (tracks_view.hitIndices().size(idx) > 7)  // current real limit
      printf("ERROR %d, %d\n", idx, tracks_view.hitIndices().size(idx));
    assert(tracks_view.hitIndices().size(idx) <= caConstants::maxHitsOnTrack);
    for (auto ih = tracks_view.hitIndices().begin(idx); ih != tracks_view.hitIndices().end(idx); ++ih)
      assert(int(*ih) < nHits);
  }
#endif

  if (0 == first) {
    if (apc->get().m >= caConstants::maxNumberOfQuadruplets)
      printf("Tuples overflow\n");
    if (*nCells >= maxNumberOfDoublets)
      printf("Cells overflow\n");
    if (cellNeighbors && cellNeighbors->full())
      printf("cellNeighbors overflow\n");
    if (cellTracks && cellTracks->full())
      printf("cellTracks overflow\n");
    if (int(hitToTuple->nOnes()) < nHits)
      printf("ERROR hitToTuple  overflow %d %d\n", hitToTuple->nOnes(), nHits);
  }

  for (int idx = first, nt = (*nCells); idx < nt; idx += gridDim.x * blockDim.x) {
    auto const &thisCell = cells[idx];
    if (thisCell.hasFishbone() && !thisCell.isKilled())
      atomicAdd(&c.nFishCells, 1);
    if (thisCell.outerNeighbors().full())  //++tooManyNeighbors[thisCell.theLayerPairId];
      printf("OuterNeighbors overflow %d in %d\n", idx, thisCell.layerPairId());
    if (thisCell.tracks().full())  //++tooManyTracks[thisCell.theLayerPairId];
      printf("Tracks overflow %d in %d\n", idx, thisCell.layerPairId());
    if (thisCell.isKilled())
      atomicAdd(&c.nKilledCells, 1);
    if (!thisCell.unused())
      atomicAdd(&c.nEmptyCells, 1);
    if ((0 == hitToTuple->size(thisCell.inner_hit_id())) && (0 == hitToTuple->size(thisCell.outer_hit_id())))
      atomicAdd(&c.nZeroTrackCells, 1);
  }

  for (int idx = first, nt = nHits - isOuterHitOfCell.offset; idx < nt; idx += gridDim.x * blockDim.x) {
    if (isOuterHitOfCell.container[idx].full())  // ++tooManyOuterHitOfCell;
      printf("OuterHitOfCell overflow %d\n", idx);
  }
}

__global__ void kernel_fishboneCleaner(GPUCACell const *cells, uint32_t const *__restrict__ nCells, Quality *quality) {
  constexpr auto reject = pixelTrack::Quality::dup;

  auto first = threadIdx.x + blockIdx.x * blockDim.x;
  for (int idx = first, nt = (*nCells); idx < nt; idx += gridDim.x * blockDim.x) {
    auto const &thisCell = cells[idx];
    if (!thisCell.isKilled())
      continue;

    for (auto it : thisCell.tracks())
      quality[it] = reject;
  }
}

// remove shorter tracks if sharing a cell
// It does not seem to affect efficiency in any way!
__global__ void kernel_earlyDuplicateRemover(GPUCACell const *cells,
                                             uint32_t const *__restrict__ nCells,
                                             TkSoAView tracks_view,
                                             bool dupPassThrough) {
  // quality to mark rejected
  constexpr auto reject = pixelTrack::Quality::edup;  /// cannot be loose

  assert(nCells);
  auto first = threadIdx.x + blockIdx.x * blockDim.x;
  for (int idx = first, nt = (*nCells); idx < nt; idx += gridDim.x * blockDim.x) {
    auto const &thisCell = cells[idx];

    if (thisCell.tracks().size() < 2)
      continue;

    int8_t maxNl = 0;

    // find maxNl
    for (auto it : thisCell.tracks()) {
      auto nl = tracks_view[it].nLayers();
      maxNl = std::max(nl, maxNl);
    }

    // if (maxNl<4) continue;
    // quad pass through (leave it her for tests)
    //  maxNl = std::min(4, maxNl);

    for (auto it : thisCell.tracks()) {
      if (tracks_view[it].nLayers() < maxNl)
        tracks_view[it].quality() = reject;  //no race:  simple assignment of the same constant
    }
  }
}

// assume the above (so, short tracks already removed)
__global__ void kernel_fastDuplicateRemover(GPUCACell const *__restrict__ cells,
                                            uint32_t const *__restrict__ nCells,
                                            TkSoAView tracks_view,
                                            bool dupPassThrough) {
  // quality to mark rejected
  auto const reject = dupPassThrough ? pixelTrack::Quality::loose : pixelTrack::Quality::dup;
  constexpr auto loose = pixelTrack::Quality::loose;

  assert(nCells);

  auto first = threadIdx.x + blockIdx.x * blockDim.x;
  for (int idx = first, nt = (*nCells); idx < nt; idx += gridDim.x * blockDim.x) {
    auto const &thisCell = cells[idx];
    if (thisCell.tracks().size() < 2)
      continue;

    float mc = maxScore;
    uint16_t im = tkNotFound;

    auto score = [&](auto it) { return std::abs(pixelTrack::utilities::tip(tracks_view, it)); };

    // full crazy combinatorics
    int ntr = thisCell.tracks().size();
    for (int i = 0; i < ntr - 1; ++i) {
      auto it = thisCell.tracks()[i];
      auto qi = tracks_view[it].quality();
      if (qi <= reject)
        continue;
      auto opi = tracks_view[it].state()(2);
      auto e2opi = tracks_view[it].covariance()(9);
      auto cti = tracks_view[it].state()(3);
      auto e2cti = tracks_view[it].covariance()(12);
      for (auto j = i + 1; j < ntr; ++j) {
        auto jt = thisCell.tracks()[j];
        auto qj = tracks_view[jt].quality();
        if (qj <= reject)
          continue;
        auto opj = tracks_view[jt].state()(2);
        auto ctj = tracks_view[jt].state()(3);
        auto dct = nSigma2 * (tracks_view[jt].covariance()(12) + e2cti);
        if ((cti - ctj) * (cti - ctj) > dct)
          continue;
        auto dop = nSigma2 * (tracks_view[jt].covariance()(9) + e2opi);
        if ((opi - opj) * (opi - opj) > dop)
          continue;
        if ((qj < qi) || (qj == qi && score(it) < score(jt)))
          tracks_view[jt].quality() = reject;
        else {
          tracks_view[it].quality() = reject;
          break;
        }
      }
    }

    // find maxQual
    auto maxQual = reject;  // no duplicate!
    for (auto it : thisCell.tracks()) {
      if (tracks_view[it].quality() > maxQual)
        maxQual = tracks_view[it].quality();
    }

    if (maxQual <= loose)
      continue;

    // find min score
    for (auto it : thisCell.tracks()) {
      if (tracks_view[it].quality() == maxQual && score(it) < mc) {
        mc = score(it);
        im = it;
      }
    }

    if (tkNotFound == im)
      continue;

    // mark all other duplicates  (not yet, keep it loose)
    for (auto it : thisCell.tracks()) {
      if (tracks_view[it].quality() > loose && it != im)
        tracks_view[it].quality() = loose;  //no race:  simple assignment of the same constant
    }
  }
}

__global__ void kernel_connect(cms::cuda::AtomicPairCounter *apc1,
                               cms::cuda::AtomicPairCounter *apc2,  // just to zero them,
                               GPUCACell::HitsConstView hh,
                               GPUCACell *cells,
                               uint32_t const *__restrict__ nCells,
                               gpuPixelDoublets::CellNeighborsVector *cellNeighbors,
                               GPUCACell::OuterHitOfCell const isOuterHitOfCell,
                               float hardCurvCut,
                               float ptmin,
                               float CAThetaCutBarrel,
                               float CAThetaCutForward,
                               float dcaCutInnerTriplet,
                               float dcaCutOuterTriplet) {
  // auto &hh = *hhp;

  auto firstCellIndex = threadIdx.y + blockIdx.y * blockDim.y;
  auto first = threadIdx.x;
  auto stride = blockDim.x;

  if (0 == (firstCellIndex + first)) {
    (*apc1) = 0;
    (*apc2) = 0;
  }  // ready for next kernel

  for (int idx = firstCellIndex, nt = (*nCells); idx < nt; idx += gridDim.y * blockDim.y) {
    auto cellIndex = idx;
    auto &thisCell = cells[idx];
    auto innerHitId = thisCell.inner_hit_id();
    if (int(innerHitId) < isOuterHitOfCell.offset)
      continue;
    int numberOfPossibleNeighbors = isOuterHitOfCell[innerHitId].size();
    auto vi = isOuterHitOfCell[innerHitId].data();

    auto ri = thisCell.inner_r(hh);
    auto zi = thisCell.inner_z(hh);

    auto ro = thisCell.outer_r(hh);
    auto zo = thisCell.outer_z(hh);
    auto isBarrel = thisCell.inner_detIndex(hh) < caConstants::last_barrel_detIndex;

    for (int j = first; j < numberOfPossibleNeighbors; j += stride) {
      auto otherCell = __ldg(vi + j);
      auto &oc = cells[otherCell];
      auto r1 = oc.inner_r(hh);
      auto z1 = oc.inner_z(hh);
      bool aligned = GPUCACell::areAlignedRZ(
          r1,
          z1,
          ri,
          zi,
          ro,
          zo,
          ptmin,
          isBarrel ? CAThetaCutBarrel : CAThetaCutForward);  // 2.f*thetaCut); // FIXME tune cuts
      if (aligned && thisCell.dcaCut(hh,
                                     oc,
                                     oc.inner_detIndex(hh) < caConstants::last_bpix1_detIndex ? dcaCutInnerTriplet
                                                                                              : dcaCutOuterTriplet,
                                     hardCurvCut)) {  // FIXME tune cuts
        oc.addOuterNeighbor(cellIndex, *cellNeighbors);
        thisCell.setStatusBits(GPUCACell::StatusBit::kUsed);
        oc.setStatusBits(GPUCACell::StatusBit::kUsed);
      }
    }  // loop on inner cells
  }    // loop on outer cells
}

__global__ void kernel_find_ntuplets(GPUCACell::HitsConstView hh,
                                     GPUCACell *__restrict__ cells,
                                     uint32_t const *nCells,
                                     gpuPixelDoublets::CellTracksVector *cellTracks,
                                     TkSoAView tracks_view,
                                     cms::cuda::AtomicPairCounter *apc,
                                     unsigned int minHitsPerNtuplet) {
  // recursive: not obvious to widen
  // auto &hh = *hhp;

  auto first = threadIdx.x + blockIdx.x * blockDim.x;
  for (int idx = first, nt = (*nCells); idx < nt; idx += gridDim.x * blockDim.x) {
    auto const &thisCell = cells[idx];
    if (thisCell.isKilled())
      continue;  // cut by earlyFishbone
    // we require at least three hits...
    if (thisCell.outerNeighbors().empty())
      continue;
    auto pid = thisCell.layerPairId();
    auto doit = minHitsPerNtuplet > 3 ? pid < 3 : pid < 8 || pid > 12;
    if (doit) {
      GPUCACell::TmpTuple stack;
      stack.reset();
      thisCell.find_ntuplets<6>(hh,
                                cells,
                                *cellTracks,
                                tracks_view.hitIndices(),
                                *apc,
                                tracks_view.quality(),
                                stack,
                                minHitsPerNtuplet,
                                pid < 3);
      assert(stack.empty());
      // printf("in %d found quadruplets: %d\n", cellIndex, apc->get());
    }
  }
}

__global__ void kernel_mark_used(GPUCACell *__restrict__ cells, uint32_t const *nCells) {
  auto first = threadIdx.x + blockIdx.x * blockDim.x;
  for (int idx = first, nt = (*nCells); idx < nt; idx += gridDim.x * blockDim.x) {
    auto &thisCell = cells[idx];
    if (!thisCell.tracks().empty())
      thisCell.setStatusBits(GPUCACell::StatusBit::kInTrack);
  }
}

// TODO: change arg type to TkSoAConstview
__global__ void kernel_countMultiplicity(TkSoAView tracks_view, caConstants::TupleMultiplicity *tupleMultiplicity) {
  auto first = blockIdx.x * blockDim.x + threadIdx.x;
  for (int it = first, nt = tracks_view.hitIndices().nOnes(); it < nt; it += gridDim.x * blockDim.x) {
    auto nhits = tracks_view.hitIndices().size(it);
    if (nhits < 3)
      continue;
    if (tracks_view[it].quality() == pixelTrack::Quality::edup)
      continue;
    assert(tracks_view[it].quality() == pixelTrack::Quality::bad);
    if (nhits > 7)  // current limit
      printf("wrong mult %d %d\n", it, nhits);
    assert(nhits <= caConstants::maxHitsOnTrack);
    tupleMultiplicity->count(nhits);
  }
}

// TODO: change arg type to TkSoAConstview
__global__ void kernel_fillMultiplicity(TkSoAView tracks_view, caConstants::TupleMultiplicity *tupleMultiplicity) {
  auto first = blockIdx.x * blockDim.x + threadIdx.x;
  for (int it = first, nt = tracks_view.hitIndices().nOnes(); it < nt; it += gridDim.x * blockDim.x) {
    auto nhits = tracks_view.hitIndices().size(it);
    if (nhits < 3)
      continue;
    if (tracks_view[it].quality() == pixelTrack::Quality::edup)
      continue;
    assert(tracks_view[it].quality() == pixelTrack::Quality::bad);
    if (nhits > 7)
      printf("wrong mult %d %d\n", it, nhits);
    assert(nhits <= caConstants::maxHitsOnTrack);
    tupleMultiplicity->fill(nhits, it);
  }
}

/*
  Supply both the original TkSoA and the TkSoAView which contains
the SoA Data
 */
__global__ void kernel_classifyTracks(TkSoAView tracks_view,
                                      Quality *__restrict__ quality,
                                      CAHitNtupletGeneratorKernelsGPU::QualityCuts cuts) {
  int first = blockDim.x * blockIdx.x + threadIdx.x;

  for (int it = first, nt = tracks_view.hitIndices().nOnes(); it < nt; it += gridDim.x * blockDim.x) {
    auto nhits = tracks_view.hitIndices().size(it);
    if (nhits == 0)
      break;  // guard

    // if duplicate: not even fit
    if (quality[it] == pixelTrack::Quality::edup)
      continue;

    assert(quality[it] == pixelTrack::Quality::bad);

    // mark doublets as bad
    if (nhits < 3)
      continue;

    // if the fit has any invalid parameters, mark it as bad
    bool isNaN = false;
    for (int i = 0; i < 5; ++i) {
      isNaN |= std::isnan(tracks_view[it].state()(i));
    }
    if (isNaN) {
#ifdef NTUPLE_DEBUG
      printf("NaN in fit %d size %d chi2 %f\n", it, tracks_view.hitIndices().size(it), tracks_view[it].chi2());
#endif
      continue;
    }

    quality[it] = pixelTrack::Quality::strict;

    // compute a pT-dependent chi2 cut

    auto roughLog = [](float x) {
      // max diff [0.5,12] at 1.25 0.16143
      // average diff  0.0662998
      union IF {
        uint32_t i;
        float f;
      };
      IF z;
      z.f = x;
      uint32_t lsb = 1 < 21;
      z.i += lsb;
      z.i >>= 21;
      auto f = z.i & 3;
      int ex = int(z.i >> 2) - 127;

      // log2(1+0.25*f)
      // averaged over bins
      const float frac[4] = {0.160497f, 0.452172f, 0.694562f, 0.901964f};
      return float(ex) + frac[f];
    };

    // (see CAHitNtupletGeneratorGPU.cc)
    float pt = std::min<float>(tracks_view[it].pt(), cuts.chi2MaxPt);
    float chi2Cut = cuts.chi2Scale * (cuts.chi2Coeff[0] + roughLog(pt) * cuts.chi2Coeff[1]);
    if (tracks_view[it].chi2() >= chi2Cut) {
#ifdef NTUPLE_FIT_DEBUG
      printf("Bad chi2 %d size %d pt %f eta %f chi2 %f\n",
             it,
             tracks_view.hitIndices().size(it),
             tracks_view[it].pt(),
             tracks_view[it].eta(),
             tracks_view[it].chi2());
#endif
      continue;
    }

    quality[it] = pixelTrack::Quality::tight;

    // impose "region cuts" based on the fit results (phi, Tip, pt, cotan(theta)), Zip)
    // default cuts:
    //   - for triplets:    |Tip| < 0.3 cm, pT > 0.5 GeV, |Zip| < 12.0 cm
    //   - for quadruplets: |Tip| < 0.5 cm, pT > 0.3 GeV, |Zip| < 12.0 cm
    // (see CAHitNtupletGeneratorGPU.cc)
    auto const &region = (nhits > 3) ? cuts.quadruplet : cuts.triplet;
    bool isOk = (std::abs(pixelTrack::utilities::tip(tracks_view, it)) < region.maxTip) and
                (tracks_view[it].pt() > region.minPt) and
                (std::abs(pixelTrack::utilities::zip(tracks_view, it)) < region.maxZip);

    if (isOk) {
      quality[it] = pixelTrack::Quality::highPurity;
    }
  }
}

__global__ void kernel_doStatsForTracks(TkSoAView tracks_view,
                                        Quality const *__restrict__ quality,
                                        CAHitNtupletGeneratorKernelsGPU::Counters *counters) {
  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = first, ntot = tracks_view.hitIndices().nOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
    if (tracks_view.hitIndices().size(idx) == 0)
      break;  //guard
    if (quality[idx] < pixelTrack::Quality::loose)
      continue;
    atomicAdd(&(counters->nLooseTracks), 1);
    if (quality[idx] < pixelTrack::Quality::strict)
      continue;
    atomicAdd(&(counters->nGoodTracks), 1);
  }
}

__global__ void kernel_countHitInTracks(TkSoAView tracks_view,
                                        CAHitNtupletGeneratorKernelsGPU::HitToTuple *hitToTuple) {
  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = first, ntot = tracks_view.hitIndices().nOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
    if (tracks_view.hitIndices().size(idx) == 0)
      break;  // guard
    for (auto h = tracks_view.hitIndices().begin(idx); h != tracks_view.hitIndices().end(idx); ++h)
      hitToTuple->count(*h);
  }
}

__global__ void kernel_fillHitInTracks(TkSoAView tracks_view,  // TODO: Make ConstView
                                       CAHitNtupletGeneratorKernelsGPU::HitToTuple *hitToTuple) {
  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = first, ntot = tracks_view.hitIndices().nOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
    if (tracks_view.hitIndices().size(idx) == 0)
      break;  // guard
    for (auto h = tracks_view.hitIndices().begin(idx); h != tracks_view.hitIndices().end(idx); ++h)
      hitToTuple->fill(*h, idx);
  }
}

__global__ void kernel_fillHitDetIndices(TkSoAView tracks_view, HitSoAConstView hh) {
  int first = blockDim.x * blockIdx.x + threadIdx.x;
  // copy offsets
  for (int idx = first, ntot = tracks_view.hitIndices().totOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
    tracks_view.detIndices().off[idx] = tracks_view.hitIndices().off[idx];
  }
  // fill hit indices
  auto nhits = hh.nHits();
  for (int idx = first, ntot = tracks_view.hitIndices().size(); idx < ntot; idx += gridDim.x * blockDim.x) {
    assert(tracks_view.hitIndices().content[idx] < nhits);
    tracks_view.detIndices().content[idx] = hh[tracks_view.hitIndices().content[idx]].detectorIndex();
  }
}

/*
  Needs both TkSoA and TkSoAView for accessing SoA, computeNumberOfLayers(), nHits(), stride()
 */
__global__ void kernel_fillNLayers(TkSoAView tracks_view, cms::cuda::AtomicPairCounter *apc) {
  auto first = blockIdx.x * blockDim.x + threadIdx.x;
  // clamp the number of tracks to the capacity of the SoA
  auto ntracks = std::min<int>(apc->get().m, tracks_view.metadata().size() - 1);
  if (0 == first)
    tracks_view.nTracks() = ntracks;
  for (int idx = first, nt = ntracks; idx < nt; idx += gridDim.x * blockDim.x) {
    auto nHits = pixelTrack::utilities::nHits(tracks_view, idx);
    assert(nHits >= 3);
    tracks_view[idx].nLayers() = pixelTrack::utilities::computeNumberOfLayers(tracks_view, idx);
  }
}

__global__ void kernel_doStatsForHitInTracks(CAHitNtupletGeneratorKernelsGPU::HitToTuple const *__restrict__ hitToTuple,
                                             CAHitNtupletGeneratorKernelsGPU::Counters *counters) {
  auto &c = *counters;
  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = first, ntot = hitToTuple->nOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
    if (hitToTuple->size(idx) == 0)
      continue;  // SHALL NOT BE break
    atomicAdd(&c.nUsedHits, 1);
    if (hitToTuple->size(idx) > 1)
      atomicAdd(&c.nDupHits, 1);
  }
}

__global__ void kernel_countSharedHit(int *__restrict__ nshared,
                                      HitContainer const *__restrict__ ptuples,
                                      Quality const *__restrict__ quality,
                                      CAHitNtupletGeneratorKernelsGPU::HitToTuple const *__restrict__ phitToTuple) {
  constexpr auto loose = pixelTrack::Quality::loose;

  auto &hitToTuple = *phitToTuple;
  auto const &foundNtuplets = *ptuples;

  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = first, ntot = hitToTuple.nOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
    if (hitToTuple.size(idx) < 2)
      continue;

    int nt = 0;

    // count "good" tracks
    for (auto it = hitToTuple.begin(idx); it != hitToTuple.end(idx); ++it) {
      if (quality[*it] < loose)
        continue;
      ++nt;
    }

    if (nt < 2)
      continue;

    // now mark  each track triplet as sharing a hit
    for (auto it = hitToTuple.begin(idx); it != hitToTuple.end(idx); ++it) {
      if (foundNtuplets.size(*it) > 3)
        continue;
      atomicAdd(&nshared[*it], 1);
    }

  }  //  hit loop
}

__global__ void kernel_markSharedHit(int const *__restrict__ nshared,
                                     HitContainer const *__restrict__ tuples,
                                     Quality *__restrict__ quality,
                                     bool dupPassThrough) {
  constexpr auto dup = pixelTrack::Quality::dup;
  constexpr auto loose = pixelTrack::Quality::loose;

  // quality to mark rejected
  auto const reject = dupPassThrough ? loose : dup;

  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = first, ntot = tuples->nOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
    if (tuples->size(idx) == 0)
      break;  //guard
    if (quality[idx] <= reject)
      continue;
    if (nshared[idx] > 2)
      quality[idx] = reject;
  }
}

// mostly for very forward triplets.....
__global__ void kernel_rejectDuplicate(TkSoAView tracks_view,
                                       uint16_t nmin,
                                       bool dupPassThrough,
                                       CAHitNtupletGeneratorKernelsGPU::HitToTuple const *__restrict__ phitToTuple) {
  // quality to mark rejected
  auto const reject = dupPassThrough ? pixelTrack::Quality::loose : pixelTrack::Quality::dup;

  auto &hitToTuple = *phitToTuple;

  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = first, ntot = hitToTuple.nOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
    if (hitToTuple.size(idx) < 2)
      continue;

    auto score = [&](auto it, auto nl) { return std::abs(pixelTrack::utilities::tip(tracks_view, it)); };

    // full combinatorics
    for (auto ip = hitToTuple.begin(idx); ip < hitToTuple.end(idx) - 1; ++ip) {
      auto const it = *ip;
      auto qi = tracks_view[it].quality();
      if (qi <= reject)
        continue;
      auto opi = tracks_view[it].state()(2);
      auto e2opi = tracks_view[it].covariance()(9);
      auto cti = tracks_view[it].state()(3);
      auto e2cti = tracks_view[it].covariance()(12);
      auto nli = tracks_view[it].nLayers();
      for (auto jp = ip + 1; jp < hitToTuple.end(idx); ++jp) {
        auto const jt = *jp;
        auto qj = tracks_view[jt].quality();
        if (qj <= reject)
          continue;
        auto opj = tracks_view[jt].state()(2);
        auto ctj = tracks_view[jt].state()(3);
        auto dct = nSigma2 * (tracks_view[jt].covariance()(12) + e2cti);
        if ((cti - ctj) * (cti - ctj) > dct)
          continue;
        auto dop = nSigma2 * (tracks_view[jt].covariance()(9) + e2opi);
        if ((opi - opj) * (opi - opj) > dop)
          continue;
        auto nlj = tracks_view[jt].nLayers();
        if (nlj < nli || (nlj == nli && (qj < qi || (qj == qi && score(it, nli) < score(jt, nlj)))))
          tracks_view[jt].quality() = reject;
        else {
          tracks_view[it].quality() = reject;
          break;
        }
      }
    }
  }
}

__global__ void kernel_sharedHitCleaner(HitSoAConstView hh,
                                        TkSoAView tracks_view,
                                        int nmin,
                                        bool dupPassThrough,
                                        CAHitNtupletGeneratorKernelsGPU::HitToTuple const *__restrict__ phitToTuple) {
  // quality to mark rejected
  auto const reject = dupPassThrough ? pixelTrack::Quality::loose : pixelTrack::Quality::dup;
  // quality of longest track
  auto const longTqual = pixelTrack::Quality::highPurity;

  auto &hitToTuple = *phitToTuple;

  // auto const &hh = *hhp;
  int l1end = hh.hitsLayerStart()[1];

  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = first, ntot = hitToTuple.nOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
    if (hitToTuple.size(idx) < 2)
      continue;

    int8_t maxNl = 0;

    // find maxNl
    for (auto it = hitToTuple.begin(idx); it != hitToTuple.end(idx); ++it) {
      if (tracks_view[*it].quality() < longTqual)
        continue;
      // if (tracks_view[*it].nHits()==3) continue;
      auto nl = tracks_view[*it].nLayers();
      maxNl = std::max(nl, maxNl);
    }

    if (maxNl < 4)
      continue;

    // quad pass through (leave for tests)
    // maxNl = std::min(4, maxNl);

    // kill all tracks shorter than maxHl (only triplets???
    for (auto it = hitToTuple.begin(idx); it != hitToTuple.end(idx); ++it) {
      auto nl = tracks_view[*it].nLayers();

      //checking if shared hit is on bpix1 and if the tuple is short enough
      if (idx < l1end and nl > nmin)
        continue;

      if (nl < maxNl && tracks_view[*it].quality() > reject)
        tracks_view[*it].quality() = reject;
    }
  }
}

__global__ void kernel_tripletCleaner(TkSoAView tracks_view,
                                      uint16_t nmin,
                                      bool dupPassThrough,
                                      CAHitNtupletGeneratorKernelsGPU::HitToTuple const *__restrict__ phitToTuple) {
  // quality to mark rejected
  auto const reject = pixelTrack::Quality::loose;
  /// min quality of good
  auto const good = pixelTrack::Quality::strict;

  auto &hitToTuple = *phitToTuple;

  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = first, ntot = hitToTuple.nOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
    if (hitToTuple.size(idx) < 2)
      continue;

    float mc = maxScore;
    uint16_t im = tkNotFound;
    bool onlyTriplets = true;

    // check if only triplets
    for (auto it = hitToTuple.begin(idx); it != hitToTuple.end(idx); ++it) {
      if (tracks_view[*it].quality() <= good)
        continue;
      onlyTriplets &= pixelTrack::utilities::isTriplet(tracks_view, *it);
      if (!onlyTriplets)
        break;
    }

    // only triplets
    if (!onlyTriplets)
      continue;

    // for triplets choose best tip!  (should we first find best quality???)
    for (auto ip = hitToTuple.begin(idx); ip != hitToTuple.end(idx); ++ip) {
      auto const it = *ip;
      if (tracks_view[it].quality() >= good && std::abs(pixelTrack::utilities::tip(tracks_view, it)) < mc) {
        mc = std::abs(pixelTrack::utilities::tip(tracks_view, it));
        im = it;
      }
    }

    if (tkNotFound == im)
      continue;

    // mark worse ambiguities
    for (auto ip = hitToTuple.begin(idx); ip != hitToTuple.end(idx); ++ip) {
      auto const it = *ip;
      if (tracks_view[it].quality() > reject && it != im)
        tracks_view[it].quality() = reject;  //no race:  simple assignment of the same constant
    }

  }  // loop over hits
}

__global__ void kernel_simpleTripletCleaner(
    TkSoAView tracks_view,
    uint16_t nmin,
    bool dupPassThrough,
    CAHitNtupletGeneratorKernelsGPU::HitToTuple const *__restrict__ phitToTuple) {
  // quality to mark rejected
  auto const reject = pixelTrack::Quality::loose;
  /// min quality of good
  auto const good = pixelTrack::Quality::loose;

  auto &hitToTuple = *phitToTuple;

  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = first, ntot = hitToTuple.nOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
    if (hitToTuple.size(idx) < 2)
      continue;

    float mc = maxScore;
    uint16_t im = tkNotFound;

    // choose best tip!  (should we first find best quality???)
    for (auto ip = hitToTuple.begin(idx); ip != hitToTuple.end(idx); ++ip) {
      auto const it = *ip;
      if (tracks_view[it].quality() >= good && std::abs(pixelTrack::utilities::tip(tracks_view, it)) < mc) {
        mc = std::abs(pixelTrack::utilities::tip(tracks_view, it));
        im = it;
      }
    }

    if (tkNotFound == im)
      continue;

    // mark worse ambiguities
    for (auto ip = hitToTuple.begin(idx); ip != hitToTuple.end(idx); ++ip) {
      auto const it = *ip;
      if (tracks_view[it].quality() > reject && pixelTrack::utilities::isTriplet(tracks_view, it) && it != im)
        tracks_view[it].quality() = reject;  //no race:  simple assignment of the same constant
    }

  }  // loop over hits
}

__global__ void kernel_print_found_ntuplets(HitSoAView hh,
                                            TkSoAView tracks_view,
                                            CAHitNtupletGeneratorKernelsGPU::HitToTuple const *__restrict__ phitToTuple,
                                            int32_t firstPrint,
                                            int32_t lastPrint,
                                            int iev) {

  constexpr auto loose = pixelTrack::Quality::loose;


  int first = firstPrint + blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = first, np = std::min(lastPrint, tracks_view.hitIndices().nOnes()); i < np; i += blockDim.x * gridDim.x) {
    auto nh = tracks_view.hitIndices().size(i);
    if (nh < 3)
      continue;
    if (tracks_view[i].quality() < loose)
      continue;
    printf("TK: %d %d %d %d %f %f %f %f %f %f %f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
           10000 * iev + i,
           int(tracks_view[i].quality()),
           nh,
           tracks_view[i].nLayers(),
           pixelTrack::utilities::charge(tracks_view, i),
           tracks_view[i].pt(),
           tracks_view[i].eta(),
           pixelTrack::utilities::phi(tracks_view, i),
           pixelTrack::utilities::tip(tracks_view, i),
           pixelTrack::utilities::zip(tracks_view, i),
           tracks_view[i].chi2(),
           hh[*tracks_view.hitIndices().begin(i)].zGlobal(),
           hh[*(tracks_view.hitIndices().begin(i) + 1)].zGlobal(),
           hh[*(tracks_view.hitIndices().begin(i) + 2)].zGlobal(),
           nh > 3 ? hh[int(*(tracks_view.hitIndices().begin(i) + 3))].zGlobal() : 0,
           nh > 4 ? hh[int(*(tracks_view.hitIndices().begin(i) + 4))].zGlobal() : 0,
           nh > 5 ? hh[int(*(tracks_view.hitIndices().begin(i) + 5))].zGlobal() : 0,
           nh > 6 ? hh[int(*(tracks_view.hitIndices().begin(i) + nh - 1))].zGlobal() : 0);
  }
}

__global__ void kernel_printCounters(cAHitNtupletGenerator::Counters const *counters) {
  auto const &c = *counters;
  printf(
      "||Counters | nEvents | nHits | nCells | nTuples | nFitTacks  |  nLooseTracks  |  nGoodTracks | nUsedHits | "
      "nDupHits | "
      "nFishCells | "
      "nKilledCells | "
      "nUsedCells | nZeroTrackCells ||\n");
  printf("Counters Raw %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld\n",
         c.nEvents,
         c.nHits,
         c.nCells,
         c.nTuples,
         c.nFitTracks,
         c.nLooseTracks,
         c.nGoodTracks,
         c.nUsedHits,
         c.nDupHits,
         c.nFishCells,
         c.nKilledCells,
         c.nEmptyCells,
         c.nZeroTrackCells);
  printf("Counters Norm %lld ||  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.3f|  %.3f|  %.3f|  %.3f||\n",
         c.nEvents,
         c.nHits / double(c.nEvents),
         c.nCells / double(c.nEvents),
         c.nTuples / double(c.nEvents),
         c.nFitTracks / double(c.nEvents),
         c.nLooseTracks / double(c.nEvents),
         c.nGoodTracks / double(c.nEvents),
         c.nUsedHits / double(c.nEvents),
         c.nDupHits / double(c.nEvents),
         c.nFishCells / double(c.nCells),
         c.nKilledCells / double(c.nCells),
         c.nEmptyCells / double(c.nCells),
         c.nZeroTrackCells / double(c.nCells));
}
