#ifndef RecoTracker_PixelSeeding_plugins_alpaka_OTStubFormationKernelsWrapper_h
#define RecoTracker_PixelSeeding_plugins_alpaka_OTStubFormationKernelsWrapper_h

#include <cstdint>
#include <optional>

#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometrySoA.h"
#include "RecoTracker/PixelSeeding/plugins/alpaka/OTStubFormationKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Wrapper class for OT stub formation kernels
  // Manages device buffers and launches kernels
  class OTStubFormationKernelsWrapper {
  public:
    // Constructor - allocates workspace buffers
    explicit OTStubFormationKernelsWrapper(Queue& queue);

    ~OTStubFormationKernelsWrapper() = default;

    // Count stubs per module
    void countStubs(Queue& queue,
                    ::reco::OTRecHitsConstView const& hits,
                    ::reco::OTHitModuleConstView const& moduleView,
                    ::reco::StackedModuleGeometryConstView const& geometry,
                    int32_t const* barrelFlatMaxCSDiff,
                    int32_t const* barrelTiltedMaxCSDiff,
                    int32_t const* endcapMaxCSDiff,
                    int32_t const* barrelFlatMaxCS,
                    int32_t const* barrelTiltedMaxCS,
                    int32_t const* endcapMaxCS,
                    int32_t const* barrelFlatMaxCSSum,
                    int32_t const* barrelTiltedMaxCSSum,
                    int32_t const* endcapMaxCSSum,
                    uint32_t* stubCounts,
                    uint32_t nModules);

    // Finalize stub offsets using device-side prefix scan
    void finalizeOffsets(Queue& queue, uint32_t* stubOffsets, uint32_t nModules);

    // Form stubs
    void formStubs(Queue& queue,
                   ::reco::OTRecHitsConstView const& hits,
                   ::reco::OTHitModuleConstView const& moduleView,
                   ::reco::StackedModuleGeometryConstView const& geometry,
                   int32_t const* barrelFlatMaxCSDiff,
                   int32_t const* barrelTiltedMaxCSDiff,
                   int32_t const* endcapMaxCSDiff,
                   int32_t const* barrelFlatMaxCS,
                   int32_t const* barrelTiltedMaxCS,
                   int32_t const* endcapMaxCS,
                   int32_t const* barrelFlatMaxCSSum,
                   int32_t const* barrelTiltedMaxCSSum,
                   int32_t const* endcapMaxCSSum,
                   uint32_t const* stubOffsets,
                   ::reco::StubsView stubs,
                   uint32_t nModules,
                   otStubFormation::CAModuleOffsets caOffsets);

    // Fill moduleStart array from stubOffsets
    void fillModuleStart(Queue& queue,
                         uint32_t const* stubOffsets,
                         ::reco::StubModuleView stubModuleView,
                         uint32_t nModules);

  private:
    // Workspace for prefix scan atomic counter
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> prefixScanWorkspace_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_OTStubFormationKernelsWrapper_h
