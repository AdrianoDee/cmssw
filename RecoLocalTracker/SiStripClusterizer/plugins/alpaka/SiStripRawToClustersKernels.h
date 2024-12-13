#ifndef RecoLocalTracker_SiStripClusterizer_SiStripRawToClustersKernels_h
#define RecoLocalTracker_SiStripClusterizer_SiStripRawToClustersKernels_h

#include <algorithm>
#include <optional>
#include <utility>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "CalibFormats/SiStripObjects/interface/SiStripClusterizerConditionsSoA.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/OneToManyAssoc.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

    class SiStripRawToClustersKernels {
    public:

      using SequentialContainer = cms::alpakatools::OneToManyAssocSequential<uint32_t, -1, -1>;
      using SequentialContainerStorage = typename SequentialContainer::index_type;
      using SequentialContainerView = typename SequentialContainer::View;
      using SequentialContainerOffsets = typename SequentialContainer::Counter;
      
      SiStripRawToClustersKernels(){}

      ~SiStripRawToClustersKernels() = default;

      SiStripRawToClustersKernels(SiStripRawToClustersKernels&&) = delete;
      SiStripRawToClustersKernels& operator=(const SiStripRawToClustersKernels&) = delete;
      SiStripRawToClustersKernels& operator=(SiStripRawToClustersKernels&&) = delete;

      void makeStripClustersAsync(Queue& queue,
                                   const std::vector<const FEDRawData*>& rawdata,
                                   const std::vector<std::unique_ptr<sistrip::FEDBuffer>>& buffers,
                                   const ::reco::StripConditionsDetSoAConstView& detView,
                                   const ::reco::StripConditionsStripSoAConstView& stripView,
                                   const ::reco::StripConditionsAPVSoAConstView& apvView
                                   );

    //   SiPixelDigisSoACollection getDigis() {
    //     digis_d->setNModules(nModules_Clusters_h[0]);
    //     return std::move(*digis_d);
    //   }

    //   SiPixelClustersSoACollection getClusters() {
    //     clusters_d->setNClusters(nModules_Clusters_h[1], nModules_Clusters_h[2]);
    //     return std::move(*clusters_d);
    //   }

    //   SiPixelDigiErrorsSoACollection getErrors() { return std::move(*digiErrors_d); }

    //   auto nModules() { return nModules_Clusters_h[0]; }

    private:
      uint32_t nStrips_ = 0;

      // Data to be put in the event
      // cms::alpakatools::host_buffer<uint32_t[]> nModules_Clusters_h;
      // std::optional<SiPixelDigisSoACollection> digis_d;
      // std::optional<SiPixelClustersSoACollection> clusters_d;
      // std::optional<SiPixelDigiErrorsSoACollection> digiErrors_d;
    };

 
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // plugin_SiPixelClusterizer_alpaka_SiStripRawToClustersKernels_h
