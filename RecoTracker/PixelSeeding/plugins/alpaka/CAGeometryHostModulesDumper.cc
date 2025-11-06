#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <algorithm>
#include <filesystem>

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"
#include "DataFormats/GeometrySurface/interface/SOARotation.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class CAGeometryHostModulesDumper : public stream::EDProducer<> {
  public:
    explicit CAGeometryHostModulesDumper(const edm::ParameterSet&);
    ~CAGeometryHostModulesDumper() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    void produce(device::Event&, const device::EventSetup&) override;

  private:
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
    std::string outputFile_;
  };

  // ----------------------------------------------------------------------
  CAGeometryHostModulesDumper::CAGeometryHostModulesDumper(const edm::ParameterSet& iConfig)
      : EDProducer(iConfig), geomToken_(esConsumes()) {
    outputFile_ = iConfig.getParameter<std::string>("outputFile");
  }

  // ----------------------------------------------------------------------
  void CAGeometryHostModulesDumper::produce(device::Event&, const device::EventSetup& iSetup) {
    const auto& trackerGeometry = iSetup.getData(geomToken_);
    const auto& dets = trackerGeometry.dets();
    const int nModules = dets.size();

    using Rotation = SOARotation<float>;
    using Frame = SOAFrame<float>;

    std::ofstream out(outputFile_, std::ios::binary);
    if (!out.is_open()) {
      throw cms::Exception("FileOpenError") << "Cannot open output file " << outputFile_;
    }

#ifdef GPU_DEBUG
    std::cout << "[GPU_DEBUG] Writing " << nModules
              << " module frames to " << outputFile_ << std::endl;
#endif

    // Write header (number of modules)
    out.write(reinterpret_cast<const char*>(&nModules), sizeof(int));

    // Write each module's SOAFrame
    for (int i = 0; i < nModules; ++i) {
      const auto det = dets[i];
      const auto& surf = det->surface();
      const auto& pos = surf.position();
      Rotation rot(surf.rotation());
      Frame frame(pos.x(), pos.y(), pos.z(), rot);
      out.write(reinterpret_cast<const char*>(&frame), sizeof(Frame));
    }

    out.close();

#ifdef GPU_DEBUG
    std::cout << "[GPU_DEBUG] Finished writing " << outputFile_
              << " (" << nModules << " modules)" << std::endl;
#endif
  }

  // ----------------------------------------------------------------------
  void CAGeometryHostModulesDumper::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>(
            "outputFile",
            std::string("CAGeometryHostModules") +
                std::string(pixelTopology::Phase1::nameModifier) + ".bin")
        ->setComment("Output binary file containing only the module detFrames (positions and rotations).");
    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(CAGeometryHostModulesDumper);
