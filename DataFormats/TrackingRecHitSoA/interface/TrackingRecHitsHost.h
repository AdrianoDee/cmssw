#ifndef DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsHost_h
#define DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsHost_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// TODO: The class is created via inheritance of the PortableCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306

namespace reco
{

  using HitPortableCollectionHost = PortableHostMultiCollection<reco::TrackingRecHitSoA, reco::HitModuleSoA>;

  class TrackingRecHitHost : public HitPortableCollectionHost {
  public:

    TrackingRecHitHost() = default;

    // Constructor which specifies only the SoA size, to be used when copying the results from the device to the host
    template <typename TQueue>
    explicit TrackingRecHitHost(TQueue queue, uint32_t nHits)
        : HitPortableCollectionHost({{int(nHits),10}}, queue) {}

    // Constructor which specifies the SoA size, number of BPIX1 hits, and the modules entry points
    template <typename TQueue> 
    explicit TrackingRecHitHost(TQueue queue, uint32_t nHits, int32_t offsetBPIX2, uint32_t const* hitsModuleStart)
        : HitPortableCollectionHost({{int(nHits),10}}, queue) {
      auto hitsView = this->template view<TrackingRecHitSoA>();
      std::copy(hitsModuleStart, hitsModuleStart + phase1PixelTopology::numberOfModules + 1, hitsView.hitsModuleStart().data());
      hitsView.offsetBPIX2() = offsetBPIX2;
    }

    template <typename TQueue>
    explicit TrackingRecHitHost(TQueue queue, SiPixelClustersHost const &clusters)
        : PortableHostMultiCollection({{int(clusters.nClusters()),clusters.view().metadata().size()}}, queue)  {
          
      auto hitsView = this->template view<TrackingRecHitSoA>();
      auto nHits = clusters.view().metadata().size();
      auto clusters_m = cms::alpakatools::make_host_view(queue, clusters.view().moduleStart(), nHits);
      auto hits_m = cms::alpakatools::make_host_view<float>(queue, hitsView.hitsModuleStart(), nHits);

      alpaka::memcpy(queue, hits_m, clusters_m);

      hitsView.offsetBPIX2() = clusters.offsetBPIX2();
    }

    uint32_t nHits() const { return this->template view<TrackingRecHitSoA>().metadata().size(); }

    int32_t offsetBPIX2() const { return this->template view<TrackingRecHitSoA>().offsetBPIX2(); }

    uint32_t const* hitsModuleStart() const { return this->template view<TrackingRecHitSoA>().hitsModuleStart().data(); }

    // do nothing for a host collection
    template <typename TQueue>
    void updateFromDevice(TQueue) {}
  };

 

}

#endif  // DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsHost_h
