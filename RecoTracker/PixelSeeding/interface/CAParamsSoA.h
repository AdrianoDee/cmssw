#ifndef RecoTracker_PixelSeeding_interface_CAParams_h
#define RecoTracker_PixelSeeding_interface_CAParams_h

#include <Eigen/Core>

#include <alpaka/alpaka.hpp>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace reco {

    using GraphNode = std::array<uint32_t, 2>;

    GENERATE_SOA_LAYOUT(CALayersLayout, 
                    SOA_COLUMN(uint32_t, layerStarts)
                    )

    GENERATE_SOA_LAYOUT(CACellsLayout, 
                    SOA_COLUMN(GraphNode, graph),
                    SOA_COLUMN(int16_t, phiCuts),
                    SOA_COLUMN(float, zmin),
                    SOA_COLUMN(float, zmax),
                    SOA_COLUMN(float, rmax),
                    SOA_SCALAR(float, cellPtCut),
                    SOA_SCALAR(float, cellZ0Cut),
                    SOA_SCALAR(bool, doClusterCut),
                    SOA_SCALAR(bool, idealConditions)
                    )

    GENERATE_SOA_LAYOUT(CARegionsLayout, 
                    SOA_COLUMN(uint32_t, regionStarts),
                    SOA_COLUMN(float, caThetaCut),
                    SOA_COLUMN(float, caDCACut)
                    )
    
  // Common types for both Host and Device code
  using CALayersSoA = CALayersLayout<>;
  using CALayersSoAView = CALayersSoA::View;
  using CALayersSoAConstView = CALayersSoA::ConstView;

  // Common types for both Host and Device code
  using CACellsSoA = CACellsLayout<>;
  using CACellsSoAView = CACellsSoA::View;
  using CACellsSoAConstView = CACellsSoA::ConstView;

  // Common types for both Host and Device code
  using CARegionsSoA = CARegionsLayout<>;
  using CARegionsSoAView = CARegionsSoA::View;
  using CARegionsSoAConstView = CARegionsSoA::ConstView;

}
#endif  // RecoTracker_PixelSeeding_interface_CAParams_h