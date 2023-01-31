#include "DataFormats/SiPixelClusterPortable/interface/alpaka/SiPixelClustersDevice.h"
#include "DataFormats/SiPixelClusterPortable/interface/SiPixelClustersHost.h"

using namespace alpaka;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;
  namespace testClusterSoA {

    class TestFillKernel {
    public:
      template <typename TAcc, typename = std::enable_if_t<is_accelerator_v<TAcc>>>
      ALPAKA_FN_ACC void operator()(TAcc const& acc, SiPixelClustersLayoutSOAView clust_view) const {
        const int32_t i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];

        for (int32_t j : elements_with_stride(acc, clust_view.metadata().size())) {
          clust_view[j].moduleStart() = j;
          clust_view[j].clusInModule() = j * 2;
          clust_view[j].moduleId() = j * 3;
          clust_view[j].clusModuleStart() = j * 4;

        }
      }
    };

    class TestVerifyKernel {
    public:
      template <typename TAcc, typename = std::enable_if_t<is_accelerator_v<TAcc>>>
      ALPAKA_FN_ACC void operator()(TAcc const& acc, SiPixelClustersLayoutSOAConstView clust_view) const {
        const int32_t i = getIdx<Grid, Threads>(acc)[0u];

        for (int32_t j : elements_with_stride(acc, clust_view.metadata().size())) {
          assert(clust_view[j].moduleStart()==j);
          assert(clust_view[j].clusInModule()==j*2);
          assert(clust_view[j].moduleId()==j*3);
          assert(clust_view[j].clusModuleStart()==j*4);
        }
      }
    };

    void runKernels(TrackSoAView<TrackerTraits> clust_view, Queue& queue) {
      uint32_t items = 64;
      uint32_t groups = divide_up_by(clust_view.metadata().size(), items);
      auto workDiv = make_workdiv<Acc1D>(groups, items);
      alpaka::exec<Acc1D>(queue, workDiv, TestFillKernel<TrackerTraits>{}, clust_view);
    }

  }  // namespace testClusterSoA
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
