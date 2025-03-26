#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <memory>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ESConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/stringize.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "TFile.h"
#include "TString.h"
#include <vector>
#include <tuple>
#include <cstdlib>
#include <unistd.h>

#include <alpaka/alpaka.hpp>

#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"  // For generic DetId access

#include "HeterogeneousCore/AlpakaInterface/interface/Backend.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDMetadata.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDMetadataSentry.h"

#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisDevice.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisHost.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigisSoACollection.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersDevice.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersHost.h"
#include "DataFormats/SiPixelClusterSoA/interface/alpaka/SiPixelClustersSoACollection.h"

#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "DataFormats/ClusterGeometrySoA/interface/ClusterGeometrysSoA.h"
#include "DataFormats/ClusterGeometrySoA/interface/alpaka/ClusterGeometrysSoACollection.h"

#include "DataFormats/CandidateSoA/interface/CandidatesSoA.h"
#include "DataFormats/CandidateSoA/interface/alpaka/CandidatesSoACollection.h"

#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"


using namespace ALPAKA_ACCELERATOR_NAMESPACE;

class HelperSplitter : public global::EDProducer<> {
public:
  explicit HelperSplitter(const edm::ParameterSet&);
  ~HelperSplitter() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  //void beginStream(edm::StreamID) override;
  //void produce(edm::Event&, const edm::EventSetup&) override;
  void produce(edm::StreamID sid, device::Event& event, device::EventSetup const& setup) const override;
  //void endStream() override;

  const double ptMin_;
  edm::ESGetToken<PixelClusterParameterEstimator, TkPixelCPERecord> const tCPE_;  
  float tanLorentzAngle_;
  float tanLorentzAngleBarrelLayer1_;  
  edm::EDGetTokenT<SiPixelClusterCollectionNew> clusterToken_;
  //const edm::EDGetTokenT<SiPixelClustersHost> SoAclusterToken_;
  edm::EDGetTokenT<edm::View<reco::Candidate>> candidateToken_;
  edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> const tTrackingGeom_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> const tTrackerTopo_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  bool verbose_;
  const device::EDPutToken<CandidatesSoACollection> CandidatesSoACollection_;
  const device::EDPutToken<ClusterGeometrysSoACollection> ClusterGeometrysSoACollection_;
  const device::EDPutToken<SiPixelDigisSoACollection> SiPixelDigisSoACollection_;
  //const device::EDPutToken<ALPAKA_ACCELERATOR_NAMESPACE::SiPixelDigisSoACollection> SiPixelDigisSoACollection_;

};

HelperSplitter::HelperSplitter(edm::ParameterSet const& iConfig)
    : EDProducer(iConfig),
      ptMin_(iConfig.getParameter<double>("ptMin")),
      tCPE_(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("pixelCPE")))),      
      tanLorentzAngle_(iConfig.getParameter<double>("tanLorentzAngle")),
      tanLorentzAngleBarrelLayer1_(iConfig.getParameter<double>("tanLorentzAngleBarrelLayer1")),
      //clusterToken_(consumes<SiPixelClusterCollectionNew>(iConfig.getParameter<edm::InputTag>("siPixelClusters"))),
      clusterToken_(consumes(iConfig.getParameter<edm::InputTag>("siPixelClusters"))),
      //SoAclusterToken_(consumes(iConfig.getParameter<edm::InputTag>("siPixelClustersSoA"))),
      candidateToken_(consumes<edm::View<reco::Candidate>>(iConfig.getParameter<edm::InputTag>("Candidate"))),
      tTrackingGeom_(esConsumes()),
      tTrackerTopo_(esConsumes()),
      geomToken_(esConsumes()),
      verbose_(iConfig.getParameter<bool>("verbose")),
      CandidatesSoACollection_{produces()},
      ClusterGeometrysSoACollection_{produces()},    
      SiPixelDigisSoACollection_{produces()}      
{}



HelperSplitter::~HelperSplitter() {
}

void HelperSplitter::produce(edm::StreamID sid, device::Event& iEvent, device::EventSetup const& iSetup) const {
    printf("*********************************Starting the HelperSplitter producer.\n");

    // Get geometry and parameter estimator
    const auto& geometry = &iSetup.getData(tTrackingGeom_);
    const TrackerGeometry* geom_ = &iSetup.getData(geomToken_);
    const PixelClusterParameterEstimator* pp = &iSetup.getData(tCPE_);

    // Create the queue for the (now GPU) device
    auto const& device = cms::alpakatools::devices<alpaka::PlatformCudaRt>()[0];
    Queue queue(device);
    if (verbose_) std::cout << "Queue done" << std::endl;


    // Get and Process candidates
    auto const& candidates = iEvent.get(candidateToken_);
    size_t nCandidates = candidates.size();
    if (verbose_) std::cout << "Number of Candidates: " << nCandidates << std::endl;

    // Count the number of valid candidates that pass the ptMin_ filter
    size_t validCandidatesCount = 0;
    for (const auto& candidate : candidates) {
        if (candidate.pt() > ptMin_) {
            ++validCandidatesCount;
        }
    }
    if (verbose_) std::cout << "Number of valid Candidates: " << validCandidatesCount << std::endl;

    // Create the CandidateSoA on the host (tkCandidates)
    CandidatesHost tkCandidates(nCandidates, queue);
    auto candidateView = tkCandidates.view();

    // Fill the CandidateSoA on the host
    size_t candidateIndex = 0;
    for (const auto& candidate : candidates) {
        if (candidate.pt() > ptMin_) {
            candidateView.px(candidateIndex) = static_cast<float>(candidate.px());
            candidateView.py(candidateIndex) = static_cast<float>(candidate.py());
            candidateView.pz(candidateIndex) = static_cast<float>(candidate.pz());
            candidateView.pt(candidateIndex) = static_cast<float>(candidate.pt());
            candidateView.eta(candidateIndex) = static_cast<float>(candidate.eta());
            candidateView.phi(candidateIndex) = static_cast<float>(candidate.phi());
            ++candidateIndex;
            if (verbose_) std::cout << "Candidate index=" << candidateIndex 
                                                          << " px=" << static_cast<float>(candidate.px())
                                                          << " py=" << static_cast<float>(candidate.py())
                                                          << " pz=" << static_cast<float>(candidate.pz())
                                                          << " pt=" << static_cast<float>(candidate.pt())
                                                          << " eta=" << static_cast<float>(candidate.eta())
                                                          << " eta=" << static_cast<float>(candidate.phi()) << std::endl;
        }
    }
    if (verbose_) std::cout << "Done with Candidates (cpu)" << std::endl;

    // Produce a device–resident copy, allocating a device candidate collection
    CandidatesSoACollection tkCandidatesDevice(nCandidates, queue);

    if (verbose_) std::cout << "Overallocation check......"  << std::endl;
    if (verbose_) std::cout << "on Host: Candidates should be " << nCandidates << std::endl;
    if (verbose_) std::cout << "on Device: tkCandidatesDevice.size() = " << candidateView.metadata().size() << std::endl;

    // Copy from the host candidate collection to the device one.
    alpaka::memcpy(queue, tkCandidatesDevice.buffer(), tkCandidates.buffer());
    alpaka::wait(queue);
    if (verbose_) std::cout << "Copied CandidateSoA to device\n\n" << std::endl;



    // Retrieve TrackerGeometry, trackerTopology from EventSetup
    const auto& trackingGeometry = iSetup.getData(tTrackingGeom_);
    const auto& trackerTopology = iSetup.getData(tTrackerTopo_);
    if (verbose_) std::cout << "TrackerGeometry/Topology got it" << std::endl;


    //auto const& clustersSoA = iEvent.get(SoAclusterToken_);


    // Get and process siPixelClusters
    auto const& PixelClusters = iEvent.get(clusterToken_);
    if (verbose_) std::cout << "siPixelClusters got it" << std::endl;

    // Process clusterToken_
    //size_t nPixelClusters = PixelClusters.size();
    //if (verbose_) std::cout << "Number of SiPixelClusters: " << nPixelClusters << std::endl;

    // Calculate the total number of Clusters (to be used later in the cluster geo SoA)
    int calculateNumberOfClusters = 0;
    int calculateNumberOfPixels = 0;
    for (auto detIt = PixelClusters.begin(); detIt != PixelClusters.end(); ++detIt) {
        calculateNumberOfClusters = calculateNumberOfClusters + detIt->size();

        const edmNew::DetSet<SiPixelCluster>& detset = *detIt;
        for (const auto& cluster : detset) {
            const SiPixelCluster& aCluster = cluster;
            calculateNumberOfPixels += aCluster.pixels().size();
        }
    }
    if (verbose_) std::cout << "Calculated " << calculateNumberOfClusters << " clusters" << std::endl;
    if (verbose_) std::cout << "Calculated " << calculateNumberOfPixels << " pixels" << std::endl;


    // Create the ClusterGeometrySoA on CPU (and its view)
    ClusterGeometrysHost geotkCluster(calculateNumberOfClusters, queue);
    SiPixelDigisHost tkDigi(calculateNumberOfPixels, queue);

    auto geoclusterView = geotkCluster.view();
    auto digiView = tkDigi.view();

    size_t clusterIndex = 0;
    size_t pixelIdx = 0;

    for (auto detIt = PixelClusters.begin(); detIt != PixelClusters.end(); ++detIt) {
        const edmNew::DetSet<SiPixelCluster>& detset = *detIt;
        const GeomDet* det = trackingGeometry.idToDet(detset.id());

        const GeomDetUnit* genericDet = geom_->idToDetUnit(detset.id());
        auto const gind = genericDet->index();
        //std::cout << "gind " << static_cast<uint32_t>(gind) << std::endl;
        uint32_t moduleId = static_cast<uint32_t>(gind);

        // Convert detset.id() to DetId
        DetId detId(detset.id());

        if (!det) continue;

        // Retrieve detector topology and pitch information
        const PixelTopology& topo = static_cast<const PixelTopology&>(det->topology());
        float pitchX, pitchY;
        std::tie(pitchX, pitchY) = topo.pitch();
        float thickness = det->surface().bounds().thickness();

        // Extract the Lorentz angle if needed
        float tanLorentzAngle = tanLorentzAngle_;

        // Extract the transformation matrix from local to global coordinates
        auto localX = det->surface().toLocal(GlobalVector(1, 0, 0));
        auto localY = det->surface().toLocal(GlobalVector(0, 1, 0));
        auto localZ = det->surface().toLocal(GlobalVector(0, 0, 1));

        // Store transformation coefficients for later use
        float transformXX = localX.x(), transformXY = localX.y(), transformXZ = localX.z();
        float transformYX = localY.x(), transformYY = localY.y(), transformYZ = localY.z();
        float transformZX = localZ.x(), transformZY = localZ.y(), transformZZ = localZ.z();

        unsigned int localClusterIdx = 0;

        // Loop over the clusters in this detector

        for (const auto& cluster : detset) {
            const SiPixelCluster& aCluster = cluster;
            std::vector<SiPixelCluster::Pixel> originalpixels = aCluster.pixels();


            // Fill GeoCluster SoA with necessary data
            geoclusterView.moduleId(clusterIndex) = moduleId;
            geoclusterView.clusterOffset(clusterIndex) = localClusterIdx;

            // Fill digiSoA with pixel information
            for (const auto& pixel : originalpixels) {
                digiView.xx(pixelIdx) = pixel.x;
                digiView.yy(pixelIdx) = pixel.y;                
                digiView.adc(pixelIdx) = pixel.adc;
                digiView.clus(pixelIdx) = localClusterIdx;                
                digiView.rawIdArr(pixelIdx) = detset.id();
                digiView.moduleId(pixelIdx) = moduleId;
                pixelIdx++;
            }

            // Use PixelCluster Parameter Estimator (CPE) to compute local parameters
            auto localParams = pp->localParametersV(cluster, (*geometry->idToDetUnit(detIt->id())));
            GlobalPoint cPos = det->surface().toGlobal(localParams[0].first);

            // Save the global cluster position and geometry info into SoA
            geoclusterView.clusterIds(clusterIndex) = detset.id();
            geoclusterView.pitchX(clusterIndex) = pitchX;
            geoclusterView.pitchY(clusterIndex) = pitchY;
            geoclusterView.thickness(clusterIndex) = thickness;
            geoclusterView.sizeX(clusterIndex) = aCluster.sizeX();
            geoclusterView.sizeY(clusterIndex) = aCluster.sizeY();
            geoclusterView.x(clusterIndex) = cPos.x();
            geoclusterView.y(clusterIndex) = cPos.y();
            geoclusterView.z(clusterIndex) = cPos.z();
            geoclusterView.transformXX(clusterIndex) = transformXX;
            geoclusterView.transformXY(clusterIndex) = transformXY;
            geoclusterView.transformXZ(clusterIndex) = transformXZ;
            geoclusterView.transformYX(clusterIndex) = transformYX;
            geoclusterView.transformYY(clusterIndex) = transformYY;
            geoclusterView.transformYZ(clusterIndex) = transformYZ;
            geoclusterView.transformZX(clusterIndex) = transformZX;
            geoclusterView.transformZY(clusterIndex) = transformZY;
            geoclusterView.transformZZ(clusterIndex) = transformZZ;

            // Debug printout for the cluster
            std::cout << "Processing clusterIndex = " << clusterIndex 
                      << ", detset id = " << detset.id() 
                      << ", module = " << moduleId
                      << ", offset = " << localClusterIdx
                      << ", pixels = " << originalpixels.size()  
                      << ", csizeX = " << aCluster.sizeX()
                      << ", csizeY = " << aCluster.sizeY()                                 
                      << ", clusterOffset = " << geoclusterView.clusterOffset(clusterIndex)
                      << ", Global Position: (x = " << cPos.x() 
                      << ", y = " << cPos.y() 
                      << ", z = " << cPos.z() << ")" 
                      << std::endl;

            ++clusterIndex;

            // Adjust cluster offset
            localClusterIdx++;

        }
    }

    if (verbose_) std::cout << "Done with siPixelClusters (cpu)" << std::endl;

    // Produce a device–resident copy, allocating a device collection
    ClusterGeometrysSoACollection tkClusterGeometryDevice(calculateNumberOfClusters, queue);
    SiPixelDigisSoACollection tkDigiDevice(calculateNumberOfPixels, queue);

    // Copy from the host collection to the device one.
    alpaka::memcpy(queue, tkClusterGeometryDevice.buffer(), geotkCluster.buffer());
    alpaka::memcpy(queue, tkDigiDevice.buffer(), tkDigi.buffer());
    alpaka::wait(queue);

    //if (verbose_) std::cout << "on Host: SiPixelClusters size (total number of pixels) " << nPixelClusters << std::endl;
    if (verbose_) std::cout << "on Device: geoclusterView.size() = " << geoclusterView.metadata().size() << std::endl;


    // produce output
    iEvent.emplace(CandidatesSoACollection_, std::move(tkCandidatesDevice));
    iEvent.emplace(ClusterGeometrysSoACollection_, std::move(tkClusterGeometryDevice));
    iEvent.emplace(SiPixelDigisSoACollection_, std::move(tkDigiDevice));
}

void HelperSplitter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

    edm::ParameterSetDescription desc;
    desc.add<bool>("verbose", false)->setComment("Verbose output");
    desc.add<double>("ptMin", 0.5)->setComment("Minimum pt for filtering candidates");
    desc.add<std::string>("pixelCPE", "PixelCPEGeneric");
    desc.add<double>("tanLorentzAngle", 0.1)->setComment("Lorentz angle tangent");
    desc.add<double>("tanLorentzAngleBarrelLayer1", 0.2)->setComment("Lorentz angle tangent for Barrel Layer 1");
    desc.add<edm::InputTag>("siPixelClusters", edm::InputTag("siPixelClusters"))->setComment("Collection for siPixelClusters");
    //desc.add<edm::InputTag>("siPixelClustersSoA", edm::InputTag("siPixelClustersSoA"))->setComment("Collection for siPixelClustersSoA");
    desc.add<edm::InputTag>("Candidate", edm::InputTag("Candidate"))->setComment("Candidates");
    descriptions.add("HelperSplitter", desc);
}

DEFINE_FWK_MODULE(HelperSplitter);

#endif
