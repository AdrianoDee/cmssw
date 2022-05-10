#ifndef RecoPixelVertexing_PixelTriplets_plugins_gpuPixelDoubletsAlgos_h
#define RecoPixelVertexing_PixelTriplets_plugins_gpuPixelDoubletsAlgos_h

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
#include "DataFormats/Math/interface/approx_atan2.h"
#include "HeterogeneousCore/CUDAUtilities/interface/VecArray.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"

#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "CAStructures.h"
#include "GPUCACell.h"

//#define GPU_DEBUG
//#define NTUPLE_DEBUG
//#define DOUBLET_DEBUG
namespace gpuPixelDoublets {

  template <typename TrackerTraits>
  using CellNeighbors = caStructures::CellNeighborsT<TrackerTraits>;
  template <typename TrackerTraits>
  using CellTracks = caStructures::CellTracksT<TrackerTraits>;
  template <typename TrackerTraits>
  using CellNeighborsVector = caStructures::CellNeighborsVectorT<TrackerTraits>;
  template <typename TrackerTraits>
  using CellTracksVector = caStructures::CellTracksVectorT<TrackerTraits>;
  template <typename TrackerTraits>
  using OuterHitOfCell = caStructures::OuterHitOfCellT<TrackerTraits>;
  template <typename TrackerTraits>
  using Hits = typename GPUCACellT<TrackerTraits>::Hits;

  template<typename TrackerTraits>
  __device__ __forceinline__ bool zsizeCut(float dz, float dr, uint8_t inner, uint8_t outer, int16_t mes, int16_t so) { return false;}

  template<>
  __device__ __forceinline__ bool zsizeCut<pixelTopology::Phase2>(float dz, float dr, uint8_t inner, uint8_t outer, int16_t mes, int16_t so) { return false;}

  template<>
  __device__ __forceinline__ bool zsizeCut<pixelTopology::Phase1>(float dz, float dr, uint8_t inner, uint8_t outer, int16_t mes, int16_t so)
  {
    auto onlyBarrel = outer < 4;
    auto dy = inner == 0 ? pixelTopology::Phase1::maxDYsize12 : pixelTopology::Phase1::maxDYsize;

    return onlyBarrel ? mes > 0 && so > 0 && std::abs(so - mes) > dy
                      : (inner < 4) && mes > 0 &&
                            std::abs(mes - int(std::abs(dz / dr) * pixelTopology::Phase1::dzdrFact + 0.5f)) > pixelTopology::Phase1::maxDYPred;
  }

  template<typename TrackerTraits>
  __device__ __forceinline__ bool clusterCut(int16_t mes, uint16_t mi) { return false;}

  template<>
  __device__ __forceinline__ bool clusterCut<pixelTopology::Phase2>(int16_t mes, uint16_t mi){ return false;}

  template<>
  __device__ __forceinline__ bool clusterCut<pixelTopology::Phase1>(int16_t mes, uint16_t mi) {

    bool innerB1 = mi < pixelTopology::Phase1::last_bpix1_detIndex;
    bool outerFwd = (mi >= pixelTopology::Phase1::last_barrel_detIndex);

    if (!outerFwd)
      return false;

    if (innerB1 && outerFwd)  // B1 and F1
      if (mes > 0 && mes < pixelTopology::Phase1::minYsizeB1)
        return true; // only long cluster  (5*8)
    bool innerB2 = (mi >= pixelTopology::Phase1::last_bpix1_detIndex) && (mi <=pixelTopology::Phase1::last_bpix2_detIndex); //FIXME number
    if (innerB2 && outerFwd)  // B2 and F1
      if (mes > 0 && mes < pixelTopology::Phase1::minYsizeB2)
        return true;

    return false;
  };

  template <typename TrackerTraits>
  __device__ __forceinline__ void doubletsFromHisto(uint32_t nPairs,
                                                    GPUCACellT<TrackerTraits>* cells,
                                                    uint32_t* nCells,
                                                    CellNeighborsVector<TrackerTraits>* cellNeighbors,
                                                    CellTracksVector<TrackerTraits>* cellTracks,
                                                    TrackingRecHit2DSOAViewT<TrackerTraits> const& __restrict__ hh,
                                                    OuterHitOfCell<TrackerTraits> isOuterHitOfCell,
                                                    bool ideal_cond,
                                                    bool doClusterCut,
                                                    bool doZ0Cut,
                                                    bool doPtCut,
                                                    uint32_t maxNumOfDoublets) {
    // ysize cuts (z in the barrel)  times 8
    // these are used if doClusterCut is true

    constexpr int maxDYsize12 = 28;
    constexpr int maxDYsize = 20;
    constexpr int maxDYPred = 20;
    constexpr float dzdrFact = 8 * 0.0285 / 0.015;  // from dz/dr to "DY"

    bool isOuterLadder = ideal_cond;

    using PhiBinner = typename TrackingRecHit2DSOAViewT<TrackerTraits>::PhiBinner;

    auto const& __restrict__ phiBinner = hh.phiBinner();
    uint32_t const* __restrict__ offsets = hh.hitsLayerStart();
    assert(offsets);

    auto layerSize = [=](uint8_t li) { return offsets[li + 1] - offsets[li]; };

    // nPairsMax to be optimized later (originally was 64).
    // If it should be much bigger, consider using a block-wide parallel prefix scan,
    // e.g. see  https://nvlabs.github.io/cub/classcub_1_1_warp_scan.html

    // assert(nPairs <= nPairsMax);
    __shared__ uint32_t innerLayerCumulativeSize[TrackerTraits::nPairs];
    __shared__ uint32_t ntot;
    if (threadIdx.y == 0 && threadIdx.x == 0) {
      innerLayerCumulativeSize[0] = layerSize(TrackerTraits::layerPairs[0]);
      for (uint32_t i = 1; i < nPairs; ++i) {
        innerLayerCumulativeSize[i] = innerLayerCumulativeSize[i - 1] + layerSize(TrackerTraits::layerPairs[2 * i]);
      }
      ntot = innerLayerCumulativeSize[nPairs - 1];
    }
    __syncthreads();

    // x runs faster
    auto idy = blockIdx.y * blockDim.y + threadIdx.y;
    auto first = threadIdx.x;
    auto stride = blockDim.x;

    uint32_t pairLayerId = 0;  // cannot go backward

    #ifdef DOUBLET_DEBUG
         int nDoublets[60],nZ[60],nPhi[60],nz0[60],nPt[60],nClus[60];

         for (size_t i = 0; i < 60; i++) {
           nDoublets[i]=0;
           nZ[i] = 0;
           nPhi[i] = 0;
           nz0[i] = 0;
           nPt[i] = 0;
           nClus[i] = 0;
         }

    #endif

    for (auto j = idy; j < ntot; j += blockDim.y * gridDim.y) {
      while (j >= innerLayerCumulativeSize[pairLayerId++])
        ;
      --pairLayerId;  // move to lower_bound ??

      assert(pairLayerId < nPairs);
      assert(j < innerLayerCumulativeSize[pairLayerId]);
      assert(0 == pairLayerId || j >= innerLayerCumulativeSize[pairLayerId - 1]);

      uint8_t inner = TrackerTraits::layerPairs[2 * pairLayerId];
      uint8_t outer = TrackerTraits::layerPairs[2 * pairLayerId + 1];
      assert(outer > inner);

      auto hoff = PhiBinner::histOff(outer);

      auto i = (0 == pairLayerId) ? j : j - innerLayerCumulativeSize[pairLayerId - 1];
      i += offsets[inner];

      // printf("Hit in Layer %d %d %d %d\n", i, inner, pairLayerId, j);

      assert(i >= offsets[inner]);
      assert(i < offsets[inner + 1]);

      // found hit corresponding to our cuda thread, now do the job
      auto mi = hh.detectorIndex(i);
      if (mi > gpuClustering::maxNumModules)
        continue;  // invalid

      /* maybe clever, not effective when zoCut is on
      auto bpos = (mi%8)/4;  // if barrel is 1 for z>0
      auto fpos = (outer>3) & (outer<7);
      if ( ((inner<3) & (outer>3)) && bpos!=fpos) continue;
      */

      auto mez = hh.zGlobal(i);

      if (mez < TrackerTraits::minz[pairLayerId] || mez > TrackerTraits::maxz[pairLayerId])
        continue;

      int16_t mes = -1;  // make compiler happy
      if (doClusterCut) {
        if (inner == 0)
          isOuterLadder = ideal_cond ? true : 0 == (mi / 8) % 2;  // only for B1/B2/B3 B4 is opposite, FPIX:noclue...
        mes = inner > 0 || isOuterLadder ? hh.clusterSizeY(i) : -1;
      }

      if (doClusterCut && clusterCut<TrackerTraits>(mes,mi))
        continue;



      auto mep = hh.iphi(i);
      auto mer = hh.rGlobal(i);

      // all cuts: true if fails
      constexpr float z0cut = 12.f;      // cm
      constexpr float hardPtCut = 0.5f;  // GeV
      // cm (1 GeV track has 1 GeV/c / (e * 3.8T) ~ 87 cm radius in a 3.8T field)
      constexpr float minRadius = hardPtCut * 87.78f;
      constexpr float minRadius2T4 = 4.f * minRadius * minRadius;
      auto ptcut = [&](int j, int16_t idphi) {
        auto r2t4 = minRadius2T4;
        auto ri = mer;
        auto ro = hh.rGlobal(j);
        auto dphi = short2phi(idphi);
        return dphi * dphi * (r2t4 - ri * ro) > (ro - ri) * (ro - ri);
      };
      auto z0cutoff = [&](int j) {
        auto zo = hh.zGlobal(j);
        auto ro = hh.rGlobal(j);
        auto dr = ro - mer;
        return dr > TrackerTraits::maxr[pairLayerId] || dr < 0 || std::abs((mez * ro - mer * zo)) > z0cut * dr;
      };
/*
      auto zsizeCutOld = [&](int j) {
        auto onlyBarrel = outer < 4;
        auto so = hh.clusterSizeY(j);
        auto dy = inner == 0 ? maxDYsize12 : maxDYsize;
        // in the barrel cut on difference in size
        // in the endcap on the prediction on the first layer (actually in the barrel only: happen to be safe for endcap as well)
        // FIXME move pred cut to z0cutoff to optmize loading of and computaiton ...
        auto zo = hh.zGlobal(j);
        auto ro = hh.rGlobal(j);
        return onlyBarrel ? mes > 0 && so > 0 && std::abs(so - mes) > dy
                          : (inner < 4) && mes > 0 &&
                                std::abs(mes - int(std::abs((mez - zo) / (mer - ro)) * dzdrFact + 0.5f)) > maxDYPred;
      };
*/
      auto iphicut = TrackerTraits::phicuts[pairLayerId];

      auto kl = PhiBinner::bin(int16_t(mep - iphicut));
      auto kh = PhiBinner::bin(int16_t(mep + iphicut));
      auto incr = [](auto& k) { return k = (k + 1) % PhiBinner::nbins(); };

#ifdef GPU_DEBUG
      int tot = 0;
      int nmin = 0;
      int tooMany = 0;
#endif



      auto khh = kh;
      incr(khh);
      for (auto kk = kl; kk != khh; incr(kk)) {
#ifdef GPU_DEBUG
        if (kk != kl && kk != kh)
          nmin += phiBinner.size(kk + hoff);
#endif
        auto const* __restrict__ p = phiBinner.begin(kk + hoff);
        auto const* __restrict__ e = phiBinner.end(kk + hoff);
        p += first;
        for (; p < e; p += stride) {

          #ifdef DOUBLET_DEBUG
          nDoublets[pairLayerId]++;
          #endif

          // printf("%.2f - %.2f - %.2f - %.2f", inner, mi, mep, mez, mer);
          auto oi = __ldg(p);
          assert(oi >= offsets[outer]);
          assert(oi < offsets[outer + 1]);
          auto mo = hh.detectorIndex(oi);

          // printf("%.2f - %.2f - %.2f - %.2f", outer, mo, mep, hh.zGlobal(j), hh.rGlobal(j))
          if (mo > gpuClustering::maxNumModules)
            continue;  //    invalid
          bool no = false;
          if (doZ0Cut && z0cutoff(oi))
          #ifndef DOUBLET_DEBUG
            continue;
          #else
          {nz0[pairLayerId]++;no=true;}
          #endif
          auto mop = hh.iphi(oi);
          uint16_t idphi = std::min(std::abs(int16_t(mop - mep)), std::abs(int16_t(mep - mop)));
          if (idphi > iphicut)
          #ifndef DOUBLET_DEBUG
            continue;
          #else
           {nPhi[pairLayerId]++;no=true;}
          #endif
          bool doCut = zsizeCut<TrackerTraits>(mez-hh.zGlobal(oi), mer-hh.rGlobal(oi), inner, outer, mes, hh.clusterSizeY(oi));
          //assert(zsizeCutOld(oi) == doCut);
          if (doClusterCut && doCut)
          #ifndef DOUBLET_DEBUG
            continue;
          #else
           {nClus[pairLayerId]++;no=true;}
           #endif
          if (doPtCut && ptcut(oi, idphi))
          #ifndef DOUBLET_DEBUG
            continue;
           #else
           {nPt[pairLayerId]++;no=true;}
           #endif

           if(no)
           continue;
          auto ind = atomicAdd(nCells, 1);
          if (ind >= maxNumOfDoublets) {
            atomicSub(nCells, 1);
            break;
          }  // move to SimpleVector??
          // int layerPairId, int doubletId, int innerHitId, int outerHitId)
          cells[ind].init(*cellNeighbors, *cellTracks, hh, pairLayerId, i, oi);
          isOuterHitOfCell[oi].push_back(ind);
#ifdef GPU_DEBUG
          if (isOuterHitOfCell[oi].full())
            ++tooMany;
          ++tot;
#endif
        }
      }
     //printf("gpuDoublets %d \n",*nCells);
//      #endif
#ifdef GPU_DEBUG
      if (tooMany > 0)
        printf("OuterHitOfCell full for %d in layer %d/%d, %d,%d %d\n", i, inner, outer, nmin, tot, tooMany);
#endif
    }  // loop in block...

    #ifdef DOUBLET_DEBUG
         for (int i = 0; i < 60; i++) {
           if(i>int(nPairs))
           continue;
           printf("pair %d %d %d %d %d %d %d %d %d %.2f %.2f \n", i, TrackerTraits::layerPairs[2 * i], TrackerTraits::layerPairs[2 * i + 1], nDoublets[i] ,nZ[i],nz0[i],nPhi[i],nPt[i],nClus[i],TrackerTraits::minz[i],TrackerTraits::maxz[i]);
         }
    #endif
  }

}  // namespace gpuPixelDoublets

#endif  // RecoPixelVertexing_PixelTriplets_plugins_gpuPixelDoubletsAlgos_h
