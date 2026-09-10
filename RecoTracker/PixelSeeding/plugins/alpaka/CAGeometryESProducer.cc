#include <memory>
#include <type_traits>
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
// Brings the (complete) generic CopyToDevice<PortableHostCollection<...>> specialization required by
// the alpaka ESProducer::setWhatProduced to register the host->device copy of the produced payload.
#include "DataFormats/Portable/interface/PortableCollection.h"

#include "RecoTracker/Record/interface/CAGeometryRecord.h"
#include "RecoTracker/Record/interface/StackedModuleGeometryRecord.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometryHost.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometryHost.h"

#include "CAGeometryBuild.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // ESProducer for the geometry-only blocks (modules, layers) of the CA geometry SoA. The
  // module/layer walk runs once per IOV so that all consumers read one copy. The cut blocks have
  // zero rows; each CA producer fills its own from its PSet.
  template <typename TrackerTraits>
  class CAGeometryESProducer : public ESProducer {
  public:
    CAGeometryESProducer(edm::ParameterSet const& iConfig)
        : ESProducer(iConfig), nLayers_(iConfig.getParameter<unsigned int>("nLayers")) {
      // nLayers sizes the layers block (nLayers+1 rows); zero is degenerate. It is not
      // cross-checked against TrackerTraits::numberOfLayers here: the CA consumers compare and throw.
      if (nLayers_ == 0)
        throw cms::Exception("Configuration") << "CAGeometryESProducer: nLayers must be greater than zero.";
      // The OT-extension walk holds its per-layer state in block shared memory sized at
      // reco::kMaxCALayers and encodes covered layers in a uint64_t bitmask, both indexed by the
      // layer number this block defines.
      if (nLayers_ > static_cast<unsigned int>(reco::kMaxCALayers))
        throw cms::Exception("Configuration") << "CAGeometryESProducer: nLayers = " << nLayers_
                                              << " is above the hard bound reco::kMaxCALayers = " << reco::kMaxCALayers
                                              << " imposed by the OT-extension layer walk.";
      auto c = setWhatProduced(this);
      geomToken_ = c.consumes();
      topoToken_ = c.consumes();
      // Only the OT-stubs build path reads the StackedModuleGeometry (CA-ordered OT module frames);
      // for the other topologies the record is neither needed nor required to exist.
      if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
        stackedToken_ = c.consumes();
      }
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;

      // The only parameter is the CA layer count; the default is the topology's own layer count,
      // which every configuration that shares this geometry must also use.
      desc.add<unsigned int>("nLayers", TrackerTraits::numberOfLayers)
          ->setComment(
              "Number of CA layers of the layer table this geometry describes. Must equal the number of layers the "
              "CA producers sharing this geometry are configured for (they throw otherwise).");

      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<reco::CAGeometryHost> produce(CAGeometryRecord const& iRecord) {
      auto const& trackerGeometry = iRecord.get(geomToken_);
      auto const& trackerTopology = iRecord.get(topoToken_);

      reco::StackedModuleGeometryHost const* stackedGeometry = nullptr;
      if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
        stackedGeometry = &iRecord.get(stackedToken_);
      }

      // Geometry blocks only: the cut blocks are left with zero rows (the two trailing sizing
      // arguments of buildCAGeometryHost default to 0).
      return std::make_unique<reco::CAGeometryHost>(reco::buildCAGeometryHost<TrackerTraits>(
          trackerGeometry, trackerTopology, stackedGeometry, static_cast<int>(nLayers_)));
    }

  private:
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
    edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
    edm::ESGetToken<reco::StackedModuleGeometryHost, StackedModuleGeometryRecord> stackedToken_;

    const unsigned int nLayers_;
  };

  // Only the Phase-2 OT tracking chain (Phase2OTStubs and Phase2OT) is registered; pixel-only
  // topologies are supported by buildCAGeometryHost but are not part of that chain.
  using CAGeometryESProducerPhase2OT = CAGeometryESProducer<pixelTopology::Phase2OT>;
  using CAGeometryESProducerPhase2OTStubs = CAGeometryESProducer<pixelTopology::Phase2OTStubs>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(CAGeometryESProducerPhase2OT);
DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(CAGeometryESProducerPhase2OTStubs);
