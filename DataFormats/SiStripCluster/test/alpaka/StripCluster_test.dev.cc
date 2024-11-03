#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/SiStripCluster/interface/SiStripClusterHost.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterDevice.h"
#include "DataFormats/SiStripCluster/interface/alpaka/SiStripClusterCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "StripCluster_test.h"

using namespace alpaka;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;
  namespace testStripClusterSoA {

    
    struct TestFillKernel {
      template <typename TAcc, typename = std::enable_if_t<isAccelerator<TAcc>>>
      ALPAKA_FN_ACC void operator()(TAcc const& acc, SiStripClusterSoAView soa) const {
        const uint32_t i(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
        const uint32_t j(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);

        if (cms::alpakatools::once_per_grid(acc)) {
          soa.nClusters() = 270855;
          soa[10].barycenter() = 1.10;
        }

        soa[i].firstStrip() = i % 10;
        soa[j].barycenter() = 2.0;
      }
    };

    
    struct ShowKernel {
      template <typename TAcc, typename = std::enable_if_t<isAccelerator<TAcc>>>
      ALPAKA_FN_ACC void operator()(TAcc const& acc, SiStripClusterSoAConstView soa) const {
        if (cms::alpakatools::once_per_grid(acc)) {
          printf("nClusters   = %d\n", soa.nClusters());
          printf("Layout size = %d\n", soa.metadata().size());
        }

        for (uint32_t i : cms::alpakatools::uniform_elements(acc, 30)) {
          printf("firstStrip %d -> %d\n", i, soa[i].firstStrip());
          printf("barycenter %d -> %.2f\n", i, soa[i].barycenter());
        }
      }
    };

    
    void runKernels(SiStripClusterSoAView& view, Queue& queue) {
      uint32_t items = 64;
      uint32_t groups = divide_up_by(view.metadata().size(), items);
      auto workDiv = make_workdiv<Acc1D>(groups, items);
      alpaka::exec<Acc1D>(queue, workDiv, TestFillKernel{}, view);
      alpaka::exec<Acc1D>(queue, workDiv, ShowKernel{}, view);
    }

  }  // namespace testStripClusterSoA
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
