#ifndef RecoTracker_PixelSeeding_interface_TrackerGeometry_h
#define RecoTracker_PixelSeeding_interface_TrackerGeometry_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/GeometrySurface/interface/SOARotation.h"
#include "DataFormats/DetId/interface/DetId.h"

namespace reco {

  using GraphNode = std::array<uint32_t, 2>;
  using DetFrame = SOAFrame<float>;

  GENERATE_SOA_LAYOUT(TrackerModulesLayout, 
                      SOA_COLUMN(DetFrame, detFrame),
                      SOA_COLUMN(DetId, detId),
                      SOA_COLUMN(uint8_t, layerNumber)
                      )

  // GENERATE_SOA_LAYOUT(TrackerLayersLayout,
  //                     SOA_COLUMN(uint32_t, layerStarts))

  GENERATE_SOA_LAYOUT(TrackerMaterialMapLayout,
                      SOA_COLUMN(float, radlen),
                      SOA_SCALAR(int16_t, nBinsR))

  // using TrackerLayersSoA = TrackerLayersLayout<>;
  // using TrackerLayersSoAView = TrackerLayersSoA::View;
  // using TrackerLayersSoAConstView = TrackerLayersSoA::ConstView;

  using TrackerMaterialMapSoA = TrackerMaterialMapLayout<>;
  using TrackerMaterialMapSoAView = TrackerMaterialMapSoA::View;
  using TrackerMaterialMapSoAConstView = TrackerMaterialMapSoA::ConstView;

  using TrackerModulesSoA = TrackerModulesLayout<>;
  using TrackerModulesView = TrackerModulesSoA::View;
  using TrackerModulesConstView = TrackerModulesSoA::ConstView;

}  // namespace reco
#endif  // RecoTracker_PixelSeeding_interface_TrackerGeometry_h
