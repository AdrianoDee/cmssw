#include <memory>
#include <string>
#include <alpaka/alpaka.hpp>

#include <TFile.h>
#include <TProfile2D.h>

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "RecoTracker/GeometryESProducer/interface/TrackerGeometrySoA.h"
#include "RecoTracker/GeometryESProducer/interface/TrackerGeometryHost.h"
#include "RecoTracker/GeometryESProducer/interface/alpaka/TrackerGeometrySoACollection.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "FWCore/Utilities/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define GPU_DEBUG

namespace
{
    constexpr auto zBinsMap = 1024;
    constexpr auto rBinsMap = 256;

    int rebinNoMoreThan(int N, int Nreq)
    {
        if (Nreq >= N || N <= 0) return 1;

        int bestGroup = 1;
        int bestDiff  = std::numeric_limits<int>::max();

        for (int g = 1; g <= N; ++g) {
            if (N % g != 0) continue;
            int Np = N / g;           
            if (Np > Nreq) continue;

            int diff = std::abs(Np - Nreq);
            if (diff < bestDiff) {
                bestDiff  = diff;
                bestGroup = g;
            }
        }

        return bestGroup;
    }
}

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class TrackerGeometrySoAESProducer : public ESProducer {
  public:
    TrackerGeometrySoAESProducer(edm::ParameterSet const& iConfig);
    std::optional<reco::TrackerGeometryHost> produce(const TrackerRecoGeometryRecord& iRecord);

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    using Rotation = SOARotation<float>;
    using Frame = SOAFrame<float>;

  private:
    const float fixedRadLen_;
    const std::string materialMap_;
    int nBinsZ_;
    int nBinsR_;

    TFile *mapFile_;
    // TProfile2D *mapProfile_;
    std::vector<float> linearMap_;
    // Layers params
    const std::vector<double> caThetaCuts_;
    const std::vector<double> caDCACuts_;

    // Cells params
    // TODO: move to unsigned int here
    const std::vector<int> pairGraph_;
    const std::vector<int> startingPairs_;
    const std::vector<int> phiCuts_;
    const std::vector<double> minZ_;
    const std::vector<double> maxZ_;
    const std::vector<double> maxR_;

    edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopologyToken_;
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tGeometryToken_;
  };

  TrackerGeometrySoAESProducer::TrackerGeometrySoAESProducer(const edm::ParameterSet& iConfig)
      : ESProducer(iConfig),
        fixedRadLen_(iConfig.getParameter<int>("fixedRadLen")),
        materialMap_(iConfig.getParameter<edm::FileInPath>("materialMapFile").fullPath()),
        nBinsZ_(iConfig.getParameter<int>("mapZBins")),
        nBinsR_(iConfig.getParameter<int>("mapRBins"))
 {
    auto cc = setWhatProduced(this);
    tTopologyToken_ = cc.consumes();
    tGeometryToken_ = cc.consumes();
    
    // Eventually loading the material map
    if (not materialMap_.empty())
    {
      mapFile_ = TFile::Open(materialMap_.c_str());
      if (not mapFile_->IsOpen())
        throw cms::Exception("TrackerGeometrySoAESProducer")
        << "Unable to open the material map file located at: "
        << materialMap_.c_str();

      TProfile2D* map = (TProfile2D *)mapFile_->Get("InvX0RZSumMap");

      if (not map)
          throw cms::Exception("TrackerGeometrySoAESProducer")
        << "Unable to load the 'InvX0RZSumMap' TPRofile2D from the file located at: "
        << materialMap_.c_str();
      
      edm::LogWarning("CAHitNtupletGeneratorOnGPU::fillDescriptions")
      << "Note: this fillDescriptions is a dummy one. Most probably you are missing some parameters.\n"
      << "please implement your TrackerTraits descriptions in CAHitNtupletGeneratorOnGPU.\n";

      int zBins = map->GetNbinsX();
      int rBins = map->GetNbinsY();
      int zRebin = 1, rRebin = 1;
      if(nBinsZ_ > 0)
      {
        if(nBinsZ_ <= zBins)
          zRebin = rebinNoMoreThan(zBins, nBinsZ_);
        else
          edm::LogWarning("TrackerGeometrySoAESProducer")
            << "Rebinning in Z: nBinsZ asked (" << nBinsZ_ << ") > no. of bins in Z in the original map (" << zBins << ").\n"
            << "No rebinning will be done.\n";
      }

      if(nBinsR_ > 0)
      {
        if(nBinsR_ <= rBins)
          rRebin = rebinNoMoreThan(rBins, nBinsR_);
        else
          edm::LogWarning("TrackerGeometrySoAESProducer")
            << "Rebinning in R: nBinsR asked (" << nBinsR_ << ") > no. of bins in R in the original map (" << rBins << ").\n"
            << "No rebinning will be done.\n";
      }

      if(zRebin > 1 or rRebin > 1)
      {
        map->Rebin2D(zRebin, rBins);
        zBins = map->GetNbinsX();
        rBins = map->GetNbinsY();
      }

      nBinsZ_ = zBins;
      nBinsR_ = rBins;

      // linear array: size = zBins * rBins (no under/overflow)
      std::vector<double> lin(zBins * rBins);

      linearMap_.resize(zBins*rBins);

      for (int ix = 1; ix <= zBins; ++ix) {
          for (int iy = 1; iy <= rBins; ++iy) {
              int idx = (ix - 1) + (iy - 1) * zBins;
              linearMap_[idx] = map->GetBinContent(ix, iy);
          }
      }
    }
    else
    {
      nBinsZ_ = 1;
      nBinsR_ = 1;
    }
  }

  std::optional<reco::TrackerGeometryHost> TrackerGeometrySoAESProducer::produce(const TrackerRecoGeometryRecord& iRecord) {
    
    bool doMap = not materialMap_.empty();

    const auto& trackerTopology = &iRecord.get(tTopologyToken_);
    const auto& trackerGeometry = &iRecord.get(tGeometryToken_);
    auto const& dets = trackerGeometry->dets();
    
    int n_modules = dets.size();
    auto mappingBins = doMap ? linearMap_.size() : 1;
    assert(nBinsZ_*nBinsR_ == int(linearMap_.size()));
    reco::TrackerGeometryHost product{{{n_modules, int(mappingBins)}}, cms::alpakatools::host()};

    auto layerSoA = product.view();
    auto mapSoA = product.view<::reco::TrackerMaterialMapSoA>();
    auto modulesSoA = product.view<::reco::TrackerModulesSoA>();

    for (int i = 0; i < n_modules; ++i) {
      auto det = dets[i];
      DetId detid = det->geographicalId();
      auto vv = det->surface().position();
      auto rr = Rotation(det->surface().rotation());
      modulesSoA[i].detFrame() = Frame(vv.x(), vv.y(), vv.z(), rr);
      modulesSoA[i].detId() = detid;
      modulesSoA[i].layerNumber() = trackerTopology->layer(detid);
    }

    layerSoA.layerStarts()[layerCount] = layerStarts[layerCount];

    mapSoA.nBinsR() = nBinsR_;

    if(doMap)
    {
      auto radlenHost = cms::alpakatools::make_host_view<float>(cms::alpakatools::host(), radlen.radlen().data(), mappingBins);
      auto linearMapView = cms::alpakatools::make_host_view<float>(linearMap_.data(), mappingBins);
      alpaka::memcpy(cms::alpakatools::host(), radlenHost, linearMapView);
    }
    else
    {
      radlen[0].radlen() = fixedRadLen_;
      radlen.nBinsR_ = 1;
    }
    
    assert(radlen.metadata().size() >= radlen.nBinsR_);
  
    return product;
  }

  void TrackerGeometrySoAESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<int>("nLayers", -1)
        ->setComment("Number of layers of the TrackerGeometry to be loaded into a SoA. Negative to use all the layers.");
    desc.add<edm::FileInPath>("matetrialMapFile", edm::FileInPath())
      ->setComment("Path to the 3D tracker material map to be loaded in the SoA.");
    desc.add<int>("mapZBins", -1)
      ->setComment("Binning for the material map in Z.");
    desc.add<int>("mapRBins", -1)
      ->setComment("Binning for the material map in R.");
    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(TrackerGeometrySoAESProducer);