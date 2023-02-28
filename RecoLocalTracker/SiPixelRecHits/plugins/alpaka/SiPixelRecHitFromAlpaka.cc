#include <alpaka/alpaka.hpp>

#include <fmt/printf.h>

#include "DataFormats/Portable/interface/HostProduct.h"
#include "DataFormats/Portable/interface/Product.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitSoAHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsLayout.h"
#include "DataFormats/TrackerCommon/interface/SimplePixelTopology.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforDevice.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename TrackerTraits>
  class SiPixelRecHitFromAlpakaT : public stream::SynchronizingEDProducer<> {
    using HitModuleStartArray = typename TrackingRecHitAlpakaSoA<TrackerTraits>::HitModuleStartArray;
    using hindex_type = typename TrackerTraits::hindex_type;

  public:
    explicit SiPixelRecHitFromAlpakaT(const edm::ParameterSet& iConfig);
    ~SiPixelRecHitFromAlpakaT() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    using HMSstorage = HostProduct<uint32_t[]>;

    // Data has been implicitly copied from Device to Host by the framework
    using HitsOnHost = TrackingRecHitAlpakaHost<TrackerTraits>;

  private:
    void acquire(device::Event const& iEvent, device::EventSetup const& iSetup) override;
    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override;

    const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
    const edm::EDGetTokenT<HitsOnHost> hitsToken_;                      // Alpaka hits
    const edm::EDGetTokenT<SiPixelClusterCollectionNew> clusterToken_;  // legacy clusters
    const edm::EDPutTokenT<SiPixelRecHitCollection> rechitsPutToken_;   // legacy rechits
    const edm::EDPutTokenT<HMSstorage> hostPutToken_;

    uint32_t nHits_;
  };

  template <typename TrackerTraits>
  SiPixelRecHitFromAlpakaT<TrackerTraits>::SiPixelRecHitFromAlpakaT(const edm::ParameterSet& iConfig)
      : geomToken_(esConsumes()),
        hitsToken_(consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"))),
        clusterToken_(consumes(iConfig.getParameter<edm::InputTag>("src"))),
        rechitsPutToken_(produces()),
        hostPutToken_(produces()) {}

  template <typename TrackerTraits>
  void SiPixelRecHitFromAlpakaT<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("siPixelRecHitsPreSplittingAlpaka"));
    desc.add<edm::InputTag>("src", edm::InputTag("siPixelClustersPreSplitting"));

    descriptions.addWithDefaultLabel(desc);
  }

  template <typename TrackerTraits>
  void SiPixelRecHitFromAlpakaT<TrackerTraits>::acquire(device::Event const& iEvent, device::EventSetup const& iSetup) {
    auto& hits_h_ = iEvent.get(hitsToken_);
    nHits_ = hits_h_.view().nHits();
    LogDebug("SiPixelRecHitFromAlpaka") << "converting " << nHits_ << " Hits";

    if (0 == nHits_)
      return;
  }

  template <typename TrackerTraits>
  void SiPixelRecHitFromAlpakaT<TrackerTraits>::produce(device::Event& iEvent, device::EventSetup const& es) {
    // TODO: is it a good idea to .get() from the Event in produce()?
    // This will at least prevent making an intermediate copy.
    // No suitable way to keep an intermediate a pointer was found.
    auto& hits_h_ = iEvent.get(hitsToken_);

    // allocate a buffer for the indices of the clusters
    constexpr auto nMaxModules = TrackerTraits::numberOfModules;
    auto hmsp = cms::alpakatools::make_host_buffer<hindex_type[]>(nMaxModules + 1);

    SiPixelRecHitCollection output;
    output.reserve(nMaxModules, nHits_);

    if (0 == nHits_) {
      iEvent.emplace(rechitsPutToken_, std::move(output));
      iEvent.emplace(hostPutToken_, std::move(hmsp));
      return;
    }
    output.reserve(nMaxModules, nHits_);

    // Could not make a view on hitsModuleStart
    //alpaka::memcpy(iEvent.queue(), hmsp, hitsModuleStartView);

    // std::copy not really working, cannot d
    //std::copy(hits_h_.view().hitsModuleStart(), &hits_h_.view().hitsModuleStart() + nMaxModules + 1, hmsp.data());
    std::memcpy((void*)hmsp.data(), (void*)&hits_h_.view().hitsModuleStart(), sizeof(HitModuleStartArray));

    // wrap the buffer in a HostProduct, and move it to the Event, without reallocating the buffer or affecting hitsModuleStart
    iEvent.emplace(hostPutToken_, std::move(hmsp));

    auto xl = hits_h_.view().xLocal();
    auto yl = hits_h_.view().yLocal();
    auto xe = hits_h_.view().xerrLocal();
    auto ye = hits_h_.view().yerrLocal();

    const TrackerGeometry* geom = &es.getData(geomToken_);

    edm::Handle<SiPixelClusterCollectionNew> hclusters = iEvent.getHandle(clusterToken_);
    auto const& input = *hclusters;

    constexpr uint32_t maxHitsInModule = gpuClustering::maxHitsInModule();

    int numberOfDetUnits = 0;
    int numberOfClusters = 0;
    for (auto const& dsv : input) {
      numberOfDetUnits++;
      unsigned int detid = dsv.detId();
      DetId detIdObject(detid);
      const GeomDetUnit* genericDet = geom->idToDetUnit(detIdObject);
      auto gind = genericDet->index();
      const PixelGeomDetUnit* pixDet = dynamic_cast<const PixelGeomDetUnit*>(genericDet);
      assert(pixDet);
      SiPixelRecHitCollection::FastFiller recHitsOnDetUnit(output, detid);
      auto fc = hits_h_.view().hitsModuleStart()[gind];
      auto lc = hits_h_.view().hitsModuleStart()[gind + 1];
      auto nhits = lc - fc;

      assert(lc > fc);
      LogDebug("SiPixelRecHitFromAlpaka") << "in det " << gind << ": conv " << nhits << " hits from " << dsv.size()
                                          << " legacy clusters" << ' ' << fc << ',' << lc << "\n";
      if (nhits > maxHitsInModule)
        edm::LogWarning("SiPixelRecHitFromAlpaka") << fmt::sprintf(
            "Too many clusters %d in module %d. Only the first %d hits will be converted", nhits, gind, maxHitsInModule);
      nhits = std::min(nhits, maxHitsInModule);

      LogDebug("SiPixelRecHitFromAlpaka") << "in det " << gind << "conv " << nhits << " hits from " << dsv.size()
                                          << " legacy clusters" << ' ' << lc << ',' << fc;

      if (0 == nhits)
        continue;
      auto jnd = [&](int k) { return fc + k; };
      assert(nhits <= dsv.size());
      if (nhits != dsv.size()) {
        edm::LogWarning("GPUHits2CPU") << "nhits!= nclus " << nhits << ' ' << dsv.size();
      }
      for (auto const& clust : dsv) {
        assert(clust.originalId() >= 0);
        assert(clust.originalId() < dsv.size());
        if (clust.originalId() >= nhits)
          continue;
        auto ij = jnd(clust.originalId());
        LocalPoint lp(xl[ij], yl[ij]);
        LocalError le(xe[ij], 0, ye[ij]);
        SiPixelRecHitQuality::QualWordType rqw = 0;

        numberOfClusters++;

        /* cpu version....  (for reference)
      std::tuple<LocalPoint, LocalError, SiPixelRecHitQuality::QualWordType> tuple = cpe_->getParameters( clust, *genericDet );
      LocalPoint lp( std::get<0>(tuple) );
      LocalError le( std::get<1>(tuple) );
      SiPixelRecHitQuality::QualWordType rqw( std::get<2>(tuple) );
      */

        // Create a persistent edm::Ref to the cluster
        edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster> cluster = edmNew::makeRefTo(hclusters, &clust);
        // Make a RecHit and add it to the DetSet
        recHitsOnDetUnit.emplace_back(lp, le, rqw, *genericDet, cluster);
        // =============================

        LogDebug("SiPixelRecHitFromAlpaka") << "cluster " << numberOfClusters << " at " << lp << ' ' << le;

      }  //  <-- End loop on Clusters

      //  LogDebug("SiPixelRecHitGPU")
      LogDebug("SiPixelRecHitFromAlpaka") << "found " << recHitsOnDetUnit.size() << " RecHits on " << detid;

    }  //    <-- End loop on DetUnits

    LogDebug("SiPixelRecHitFromAlpaka") << "found " << numberOfDetUnits << " dets, " << numberOfClusters << " clusters";

    iEvent.emplace(rechitsPutToken_, std::move(output));
  }

  using SiPixelRecHitFromAlpakaPhase1 = SiPixelRecHitFromAlpakaT<pixelTopology::Phase1>;
  using SiPixelRecHitFromAlpakaPhase2 = SiPixelRecHitFromAlpakaT<pixelTopology::Phase2>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"

DEFINE_FWK_ALPAKA_MODULE(SiPixelRecHitFromAlpakaPhase1);
DEFINE_FWK_ALPAKA_MODULE(SiPixelRecHitFromAlpakaPhase2);
