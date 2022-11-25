#ifndef RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelRecHits_h
#define RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelRecHits_h

#include <cstdint>
#include <cstdio>
#include <limits>

#include "CUDADataFormats/BeamSpot/interface/BeamSpotCUDA.h"
#include "CUDADataFormats/SiPixelCluster/interface/gpuClusteringConstants.h"
// #include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
#include "DataFormats/Math/interface/approx_atan2.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"
#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDASOAView.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitsUtilities.h"

#define GPU_DEBUG
namespace gpuPixelRecHits {

  __global__ void getHits(pixelCPEforGPU::ParamsOnGPU const* __restrict__ cpeParams,
                          BeamSpotPOD const* __restrict__ bs,
                          // SiPixelDigisCUDASOAView const digis
                          SiPixelDigisCUDASOAConstView digis,
                          int numElements,
                          SiPixelClustersCUDASoA::ConstView clusters,
                          trackingRecHitSoA::HitSoAView hits) {
    // FIXME
    // the compiler seems NOT to optimize loads from views (even in a simple test case)
    // The whole gimnastic here of copying or not is a pure heuristic exercise that seems to produce the fastest code with the above signature
    // not using views (passing a gazzilion of array pointers) seems to produce the fastest code (but it is harder to mantain)
    // assert(phits);
    assert(cpeParams);
    // auto& hits = *phits;

    // auto const& clusters = *pclusters;
    auto isPhase2 = cpeParams->commonParams().isPhase2;
    // copy average geometry corrected by beamspot . FIXME (move it somewhere else???)
    if (0 == blockIdx.x) {
      auto& agc = hits.averageGeometry();
      auto const& ag = cpeParams->averageGeometry();
      auto nLadders =
          isPhase2 ? phase2PixelTopology::numberOfLaddersInBarrel : phase1PixelTopology::numberOfLaddersInBarrel;

      for (int il = threadIdx.x, nl = nLadders; il < nl; il += blockDim.x) {
        agc.ladderZ[il] = ag.ladderZ[il] - bs->z;
        agc.ladderX[il] = ag.ladderX[il] - bs->x;
        agc.ladderY[il] = ag.ladderY[il] - bs->y;
        agc.ladderR[il] = sqrt(agc.ladderX[il] * agc.ladderX[il] + agc.ladderY[il] * agc.ladderY[il]);
        agc.ladderMinZ[il] = ag.ladderMinZ[il] - bs->z;
        agc.ladderMaxZ[il] = ag.ladderMaxZ[il] - bs->z;
      }

      if (0 == threadIdx.x) {
        agc.endCapZ[0] = ag.endCapZ[0] - bs->z;
        agc.endCapZ[1] = ag.endCapZ[1] - bs->z;
        //         printf("endcapZ %f %f\n",agc.endCapZ[0],agc.endCapZ[1]);
      }
    }

    // to be moved in common namespace...
    using gpuClustering::invalidModuleId;
    constexpr int32_t MaxHitsInIter = pixelCPEforGPU::MaxHitsInIter;

    using ClusParams = pixelCPEforGPU::ClusParams;

    // as usual one block per module
    __shared__ ClusParams clusParams;

    auto me = clusters.moduleId(blockIdx.x);
    int nclus = clusters.clusInModule(me);

    if (0 == nclus)
      return;
#ifdef GPU_DEBUG
    if (threadIdx.x == 0) {
      auto k = clusters.moduleStart(1 + blockIdx.x);
      while (digis.moduleId(k) == invalidModuleId)
        ++k;
      assert(digis.moduleId(k) == me);
    }
#endif
#ifdef GPU_DEBUG
    // if (me % 100 == 1)
    if (threadIdx.x == 0)
      printf("hitbuilder: %d clusters in module %d. will write at %d\n", nclus, me, clusters.clusModuleStart(me));
#endif
    for (int startClus = 0, endClus = nclus; startClus < endClus; startClus += MaxHitsInIter) {
      int nClusInIter = std::min(MaxHitsInIter, endClus - startClus);
      int lastClus = startClus + nClusInIter;
      assert(nClusInIter <= nclus);
      assert(nClusInIter > 0);
      assert(lastClus <= nclus);

      assert(nclus > MaxHitsInIter || (0 == startClus && nClusInIter == nclus && lastClus == nclus));

      // init
      for (int ic = threadIdx.x; ic < nClusInIter; ic += blockDim.x) {
        clusParams.minRow[ic] = std::numeric_limits<uint32_t>::max();
        clusParams.maxRow[ic] = 0;
        clusParams.minCol[ic] = std::numeric_limits<uint32_t>::max();
        clusParams.maxCol[ic] = 0;
        clusParams.charge[ic] = 0;
        clusParams.q_f_X[ic] = 0;
        clusParams.q_l_X[ic] = 0;
        clusParams.q_f_Y[ic] = 0;
        clusParams.q_l_Y[ic] = 0;
      }

      __syncthreads();

      // one thread per "digi"
      auto first = clusters.moduleStart(1 + blockIdx.x) + threadIdx.x;
      for (int i = first; i < numElements; i += blockDim.x) {
        auto id = digis.moduleId(i);
        if (id == invalidModuleId)
          continue;  // not valid
        if (id != me)
          break;  // end of module
        auto cl = digis.clus(i);
        if (cl < startClus || cl >= lastClus)
          continue;
        cl -= startClus;
        assert(cl >= 0);
        assert(cl < MaxHitsInIter);
        auto x = digis.xx(i);
        auto y = digis.yy(i);
        atomicMin(&clusParams.minRow[cl], x);
        atomicMax(&clusParams.maxRow[cl], x);
        atomicMin(&clusParams.minCol[cl], y);
        atomicMax(&clusParams.maxCol[cl], y);
      }

      __syncthreads();

      auto pixmx = cpeParams->detParams(me).pixmx;
      for (int i = first; i < numElements; i += blockDim.x) {
        auto id = digis.moduleId(i);
        if (id == invalidModuleId)
          continue;  // not valid
        if (id != me)
          break;  // end of module
        auto cl = digis.clus(i);
        if (cl < startClus || cl >= lastClus)
          continue;
        cl -= startClus;
        assert(cl >= 0);
        assert(cl < MaxHitsInIter);
        auto x = digis.xx(i);
        auto y = digis.yy(i);
        auto ch = digis.adc(i);
        atomicAdd(&clusParams.charge[cl], ch);
        ch = std::min(ch, pixmx);
        if (clusParams.minRow[cl] == x)
          atomicAdd(&clusParams.q_f_X[cl], ch);
        if (clusParams.maxRow[cl] == x)
          atomicAdd(&clusParams.q_l_X[cl], ch);
        if (clusParams.minCol[cl] == y)
          atomicAdd(&clusParams.q_f_Y[cl], ch);
        if (clusParams.maxCol[cl] == y)
          atomicAdd(&clusParams.q_l_Y[cl], ch);
      }

      __syncthreads();

      // next one cluster per thread...

      first = clusters.clusModuleStart(me) + startClus;
      for (int ic = threadIdx.x; ic < nClusInIter; ic += blockDim.x) {
        auto h = first + ic;  // output index in global memory

        assert(h < hits.nHits());
        assert(h < clusters.clusModuleStart(me + 1));

        pixelCPEforGPU::position(cpeParams->commonParams(), cpeParams->detParams(me), clusParams, ic);
        if (!isPhase2)
          pixelCPEforGPU::errorFromDB(cpeParams->commonParams(), cpeParams->detParams(me), clusParams, ic);
        else
          pixelCPEforGPU::errorFromSize(cpeParams->commonParams(), cpeParams->detParams(me), clusParams, ic);

        // store it
        hits[h].chargeAndStatus().charge = clusParams.charge[ic];
        hits[h].chargeAndStatus().status = clusParams.status[ic];
        hits[h].detectorIndex() = me;

        float xl, yl;
        hits[h].xLocal() = xl = clusParams.xpos[ic];
        hits[h].yLocal() = yl = clusParams.ypos[ic];

        hits[h].clusterSizeX() = clusParams.xsize[ic];
        hits[h].clusterSizeY() = clusParams.ysize[ic];

        hits[h].xerrLocal() = clusParams.xerr[ic] * clusParams.xerr[ic] + cpeParams->detParams(me).apeXX;
        hits[h].yerrLocal() = clusParams.yerr[ic] * clusParams.yerr[ic] + cpeParams->detParams(me).apeYY;

        // keep it local for computations
        float xg, yg, zg;
        // to global and compute phi...
        cpeParams->detParams(me).frame.toGlobal(xl, yl, xg, yg, zg);
        // here correct for the beamspot...
        xg -= bs->x;
        yg -= bs->y;
        zg -= bs->z;

        hits[h].xGlobal() = xg;
        hits[h].yGlobal() = yg;
        hits[h].zGlobal() = zg;

        hits[h].rGlobal() = std::sqrt(xg * xg + yg * yg);
        hits[h].iphi() = unsafe_atan2s<7>(yg, xg);
      }
      __syncthreads();
    }  // end loop on batches
  }

}  // namespace gpuPixelRecHits

#endif  // RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelRecHits_h
