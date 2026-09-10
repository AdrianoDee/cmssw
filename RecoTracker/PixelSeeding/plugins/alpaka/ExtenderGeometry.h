#ifndef RecoTracker_PixelSeeding_plugins_alpaka_ExtenderGeometry_h
#define RecoTracker_PixelSeeding_plugins_alpaka_ExtenderGeometry_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"

namespace reco_extender {

  // Minimal per-layer geometry the extender needs. Iteration-independent (depends only on
  // TrackerGeometry + StackedModuleGeometry). layerStarts/isBarrel/layerR/layerZ + half-extents.
  // All columns have length nLayers+1; the extra slot keeps blocks the same length.
  GENERATE_SOA_LAYOUT(ExtenderLayersLayout,
                      SOA_COLUMN(uint32_t, layerStarts),
                      SOA_COLUMN(bool, isBarrel),
                      SOA_COLUMN(float, layerR),
                      SOA_COLUMN(float, layerZ),
                      // Half-extents of the layer's module envelope for BL-style reachability tests.
                      SOA_COLUMN(float, halfExtentZ),
                      SOA_COLUMN(float, halfExtentR))

  using ExtenderLayersSoA = ExtenderLayersLayout<>;
  using ExtenderLayersView = ExtenderLayersSoA::View;
  using ExtenderLayersConstView = ExtenderLayersSoA::ConstView;

  using ExtenderGeometryHost = PortableHostCollection<ExtenderLayersSoA>;

  template <typename TDev>
  using ExtenderGeometryDevice = PortableDeviceCollection<TDev, ExtenderLayersSoA>;

  // Per-CA-module surface frames for HelixFit's BrokenLine refit. Uses the same SoA layout
  // as the CA's CAModulesSoA, so a .view() is bit-compatible with reco::CAModulesConstView
  // and HelixFit accepts it directly (only the modules block is read by the fit kernels).
  using ExtenderCAModulesHost = PortableHostCollection<::reco::CAModulesSoA>;

  template <typename TDev>
  using ExtenderCAModulesDevice = PortableDeviceCollection<TDev, ::reco::CAModulesSoA>;

}  // namespace reco_extender

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco_extender {
  using ::reco_extender::ExtenderCAModulesDevice;
  using ::reco_extender::ExtenderCAModulesHost;
  using ::reco_extender::ExtenderGeometryDevice;
  using ::reco_extender::ExtenderGeometryHost;
  // The producer's RunCache lifts the host SoA to device via MoveToDeviceCache,
  // which uses CopyToDevice<PortableHostCollection> (built-in support).
  using ExtenderGeometrySoACollection =
      std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, ExtenderGeometryHost, ExtenderGeometryDevice<Device>>;
  using ExtenderCAModulesSoACollection =
      std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, ExtenderCAModulesHost, ExtenderCAModulesDevice<Device>>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::reco_extender

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_ExtenderGeometry_h
