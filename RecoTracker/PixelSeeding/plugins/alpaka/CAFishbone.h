#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAFishbone_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAFishbone_h

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Math/interface/approx_atan2.h"
#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "CACell.h"
#include "CAStructures.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::caPixelDoublets {

  using HitToCell = caStructures::GenericContainer;
  using CellToTracks = caStructures::GenericContainer;

  template <typename TrackerTraits>
  class CAFishbone {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  HitsConstView hh,
                                  CASimpleCell<TrackerTraits>* cells,
                                  uint32_t const* __restrict__ nCells,
                                  HitToCell const* __restrict__ outerHitHisto,
                                  CellToTracks const* __restrict__ cellTracksHisto,
                                  uint32_t outerHits,
                                  bool checkTrack) const {
      // constexpr auto maxCellsPerHit = TrackerTraits::maxCellsPerHit;

      // auto const isOuterHitOfCell = isOuterHitOfCellWrap->container;

      // float x[maxCellsPerHit], y[maxCellsPerHit], z[maxCellsPerHit], n[maxCellsPerHit];
      // uint32_t cc[maxCellsPerHit];
      // uint16_t d[maxCellsPerHit];
      // uint8_t l[maxCellsPerHit];

      // outermost parallel loop, using all grid elements along the slower dimension (Y or 0 in a 2D grid)
      for (uint32_t idy : cms::alpakatools::uniform_elements_y(acc, outerHits)) {
        // auto const& vc = isOuterHitOfCell[idy];
        uint32_t size = outerHitHisto->size(idy); //TODO have this offset in the histo building directly
// #ifdef GPU_DEBUG
//         printf("hist %d histSize %d \n",idy,size);
// #endif
        // printf("fishbone ---> outerhit %d size %d - ",idy,size);

        if (size < 2)
          continue;
        
        auto const* __restrict__ bin = outerHitHisto->begin(idy);
        
        // if alligned kill one of the two.
        // in principle one could try to relax the cut (only in r-z?) for jumping-doublets
        auto const& c0 = cells[bin[0]];
        auto xo = c0.outer_x(hh);
        auto yo = c0.outer_y(hh);
        auto zo = c0.outer_z(hh);
        //printf("first cell %d x0 %.2f y0 %.2f z0 %.2f - ",bin[0],c0.outer_x(hh),c0.outer_y(hh),c0.outer_z(hh));

        // this could be moved below 
        // precomputing these here has 
        // the advantage we then loop on less 
        // entries but we can anyway skip them below and avoid having 
        // the arrays above

// #ifdef GPU_DEBUG 
//         for (auto idx = 0u; idx < size; idx++) {
//           unsigned int otherCell = bin[idx];
//           printf("vc[0] %d idx %d vc[idx] %d otherCell %d \n",vc[0],idx,vc[idx],otherCell);
//         }
// #endif
//TODO CHECK nInBin == size
        // for (auto idx = 0u; idx < nInBin; idx++) {
        for (uint32_t ic : cms::alpakatools::independent_group_elements_x(acc, size)) {
        // for (int32_t ic = 0; ic < size; ++ic) {
        // for (auto ic = 0u; ic < size; ic++) {
          unsigned int otherCell = bin[ic];
          auto& ci = cells[otherCell];//vc[ic]];
          // unsigned int otherCell = bin[ic] - nHitsBPix1;
          // auto& ci = cells[otherCell];
          if (ci.unused())
            continue;  // for triplets equivalent to next
          if (checkTrack && cellTracksHisto->size(otherCell) == 0)//ci.tracks().empty())
            continue;

          auto x1 = (ci.inner_x(hh) - xo);
          auto y1 = (ci.inner_y(hh) - yo); 
          auto z1 = (ci.inner_z(hh) - zo); 
          auto n1 = x1*x1 + y1*y1 + z1*z1;

          for (auto jc = ic + 1; jc < size; ++jc) {
            auto& cj = cells[bin[jc]];
            if (ci.inner_detIndex(hh)==cj.inner_detIndex(hh)) continue;

            auto x2 = (cj.inner_x(hh) - xo);
            auto y2 = (cj.inner_y(hh) - yo); 
            auto z2 = (cj.inner_z(hh) - zo); 
            auto n2 = x2*x2 + y2*y2 + z2*z2;

            auto cos12 = x1 * x2 + y1 * y2 + z1 * z2;

            if (cos12 * cos12 >= 0.99999f * (n1*n2)) {
              // alligned:  kill farthest (prefer consecutive layers)
              // if same layer prefer farthest (longer level arm) and make space for intermediate hit
              bool sameLayer = ci.layerPairId() == cj.layerPairId();
              if (n1 > n2) {
                if (sameLayer) {
                  cj.kill();  // closest
                  ci.setFishbone(acc, cj.inner_hit_id(), cj.inner_z(hh), hh);
#ifdef GPU_DEBUG
                  printf("hit %d same layer cell %d kill %d \n",idy,bin[ic],bin[jc]);  
#endif
                } else {
                  ci.kill();  // farthest
#ifdef GPU_DEBUG
                  printf("hit %d same layer cell %d kill %d \n",idy,bin[jc],bin[ic]);  
#endif
                  // break;  // removed to improve reproducibility, keep it for reference and tests
                }
              } else {
                if (!sameLayer) {
                  cj.kill();  // farthest
#ifdef GPU_DEBUG
                  printf("hit %d diff layer cell %d kill %d \n",idy,bin[ic],bin[jc]);
#endif 
                } else {
                  ci.kill();  // closest
                  cj.setFishbone(acc, ci.inner_hit_id(), ci.inner_z(hh), hh);
#ifdef GPU_DEBUG
                  printf("hit %d diff layer cell %d kill %d \n",idy,bin[jc],bin[ic]);
#endif
                  // break;  // removed to improve reproducibility, keep it for reference and tests
                }
              }
            } // cos
          } // cj 
        }  // ci
      }  // hits
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::caPixelDoublets

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAFishbone_h

