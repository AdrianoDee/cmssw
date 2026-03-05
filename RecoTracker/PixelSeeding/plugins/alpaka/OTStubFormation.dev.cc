#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/PixelSeeding/plugins/alpaka/OTStubFormationKernels.h"
#include "RecoTracker/PixelSeeding/plugins/alpaka/OTStubFormationKernelsWrapper.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Constructor
  OTStubFormationKernelsWrapper::OTStubFormationKernelsWrapper(Queue& queue)
      : prefixScanWorkspace_(cms::alpakatools::make_device_buffer<int32_t[]>(queue, 1)) {
    // Initialize workspace
    alpaka::memset(queue, *prefixScanWorkspace_, 0);
  }

  // Count stubs per module
  void OTStubFormationKernelsWrapper::countStubs(Queue& queue,
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
                                                  uint32_t nModules) {
    // Work division: one thread per module
    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(nModules, 256);

    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        otStubFormation::CountStubsKernel{},
                        hits,
                        moduleView,
                        geometry,
                        barrelFlatMaxCSDiff,
                        barrelTiltedMaxCSDiff,
                        endcapMaxCSDiff,
                        barrelFlatMaxCS,
                        barrelTiltedMaxCS,
                        endcapMaxCS,
                        barrelFlatMaxCSSum,
                        barrelTiltedMaxCSSum,
                        endcapMaxCSSum,
                        stubCounts,
                        nModules);
  }

  // Finalize stub offsets using device-side prefix scan
  void OTStubFormationKernelsWrapper::finalizeOffsets(Queue& queue, uint32_t* stubOffsets, uint32_t nModules) {
    // Reset workspace counter
    alpaka::memset(queue, *prefixScanWorkspace_, 0);

    // Compute prefix sum on device (in-place)
    // Since counting kernel writes to stubOffsets[iModule + 1], this naturally produces exclusive scan
    auto nthreads = 1024u;
    auto nblocks = (nModules + 1 + nthreads - 1) / nthreads;
    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(nblocks, nthreads);

    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        cms::alpakatools::multiBlockPrefixScan<uint32_t>(),
                        stubOffsets,                                       // input (counts at offset positions)
                        stubOffsets,                                       // output (in-place exclusive scan)
                        nModules + 1,                                      // size
                        nblocks,                                           // number of blocks
                        prefixScanWorkspace_->data(),                      // atomic counter workspace
                        alpaka::getPreferredWarpSize(alpaka::getDev(queue)));
  }

  // Form stubs
  void OTStubFormationKernelsWrapper::formStubs(Queue& queue,
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
                                                 otStubFormation::CAModuleOffsets caOffsets) {
    // Work division: one thread per module
    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(nModules, 256);

    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        otStubFormation::FormStubsKernel{},
                        hits,
                        moduleView,
                        geometry,
                        barrelFlatMaxCSDiff,
                        barrelTiltedMaxCSDiff,
                        endcapMaxCSDiff,
                        barrelFlatMaxCS,
                        barrelTiltedMaxCS,
                        endcapMaxCS,
                        barrelFlatMaxCSSum,
                        barrelTiltedMaxCSSum,
                        endcapMaxCSSum,
                        stubOffsets,
                        stubs,
                        nModules,
                        caOffsets);
  }

  // Fill moduleStart array from stubOffsets
  void OTStubFormationKernelsWrapper::fillModuleStart(Queue& queue,
                                                       uint32_t const* stubOffsets,
                                                       ::reco::StubModuleView stubModuleView,
                                                       uint32_t nModules) {
    // Work division: one thread per module (+ 1 for the final total)
    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(nModules + 1, 256);

    alpaka::exec<Acc1D>(
        queue, workDiv, otStubFormation::FillModuleStartKernel{}, stubOffsets, stubModuleView, nModules);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
