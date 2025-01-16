#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED  // Ensure this file is only used for CUDA

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ESConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/stringize.h"

#include "TFile.h"
#include "TString.h"
#include <vector>
#include <tuple>
#include <cstdlib>
#include <unistd.h>

#include <alpaka/alpaka.hpp>

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"

#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"

#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisDevice.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisHost.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigisSoACollection.h"

#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersDevice.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersHost.h"
#include "DataFormats/SiPixelClusterSoA/interface/alpaka/SiPixelClustersSoACollection.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/VertexSoA/interface/ZVertexSoA.h"
#include "DataFormats/VertexSoA/interface/ZVertexHost.h"
#include "DataFormats/VertexSoA/interface/ZVertexDevice.h"
#include "DataFormats/VertexSoA/interface/alpaka/ZVertexSoACollection.h"

#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "Cluster_test.h"
#include "DataFormats/ClusterGeometrySoA/interface/ClusterGeometryLayout.h"
#include "DataFormats/ClusterGeometrySoA/interface/alpaka/ClusterGeometrySoACollection.h"

#include "DataFormats/CandidateSoA/interface/CandidateLayout.h"
#include "DataFormats/CandidateSoA/interface/alpaka/CandidateSoACollection.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

class trial : public edm::stream::EDProducer<> {
public:
  explicit trial(const edm::ParameterSet&);
  ~trial() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;

  std::string configString_;
  uint32_t nHits_;
  int32_t offset_;
  TFile* rootFile_;
  std::vector<Device> devices_;

  const double ptMin_;
  double deltaR_;
  double chargeFracMin_;
  float tanLorentzAngle_;
  float tanLorentzAngleBarrelLayer1_;  
  float expSizeXAtLorentzAngleIncidence_;
  float expSizeXDeltaPerTanAlpha_;
  float expSizeYAtNormalIncidence_;
  double centralMIPCharge_;
  double chargePerUnit_;
  double forceXError_;
  double forceYError_;  
  double fractionalWidth_;
  edm::EDGetTokenT<ALPAKA_ACCELERATOR_NAMESPACE::SiPixelClustersSoACollection> clusterToken_;
  edm::EDGetTokenT<ALPAKA_ACCELERATOR_NAMESPACE::SiPixelDigisSoACollection> digisToken_;
  edm::EDGetTokenT<ALPAKA_ACCELERATOR_NAMESPACE::TrackingRecHitsSoACollection<pixelTopology::Phase1>> recHitsToken_;
  edm::EDGetTokenT<edm::View<reco::Candidate>> candidateToken_;
  edm::EDGetTokenT<ALPAKA_ACCELERATOR_NAMESPACE::ZVertexSoACollection> zVertexToken_;
  edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> const tTrackingGeom_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> const tTrackerTopo_;

  bool verbose_;
};

trial::trial(const edm::ParameterSet& iConfig)
    : configString_(iConfig.getParameter<std::string>("configString")),
      nHits_(iConfig.getParameter<uint32_t>("nHits")),
      offset_(iConfig.getParameter<int32_t>("offset")),
      rootFile_(nullptr),
      ptMin_(iConfig.getParameter<double>("ptMin")),      
      deltaR_(iConfig.getParameter<double>("deltaR")),      
      chargeFracMin_(iConfig.getParameter<double>("chargeFracMin")),      
      tanLorentzAngle_(iConfig.getParameter<double>("tanLorentzAngle")),
      tanLorentzAngleBarrelLayer1_(iConfig.getParameter<double>("tanLorentzAngleBarrelLayer1")),
      expSizeXAtLorentzAngleIncidence_(iConfig.getParameter<double>("expSizeXAtLorentzAngleIncidence")),
      expSizeXDeltaPerTanAlpha_(iConfig.getParameter<double>("expSizeXDeltaPerTanAlpha")),
      expSizeYAtNormalIncidence_(iConfig.getParameter<double>("expSizeYAtNormalIncidence")),
      centralMIPCharge_(iConfig.getParameter<double>("centralMIPCharge")),
      chargePerUnit_(iConfig.getParameter<double>("chargePerUnit_")),
      forceXError_(iConfig.getParameter < double > ("forceXError")),
      forceYError_(iConfig.getParameter < double > ("forceYError")),      
      fractionalWidth_(iConfig.getParameter < double > ("fractionalWidth")),      
      clusterToken_(consumes<ALPAKA_ACCELERATOR_NAMESPACE::SiPixelClustersSoACollection>(edm::InputTag("siPixelClusters"))),
      digisToken_(consumes<ALPAKA_ACCELERATOR_NAMESPACE::SiPixelDigisSoACollection>(edm::InputTag("SiPixelDigisSoA"))),
      recHitsToken_(consumes<ALPAKA_ACCELERATOR_NAMESPACE::TrackingRecHitsSoACollection<pixelTopology::Phase1>>(edm::InputTag("TrackingRecHitsSoA"))),
      candidateToken_(consumes<edm::View<reco::Candidate>>(edm::InputTag("candidateInput"))),
      zVertexToken_(consumes<ALPAKA_ACCELERATOR_NAMESPACE::ZVertexSoACollection>(edm::InputTag("ZVertex"))),
      tTrackingGeom_(esConsumes()),
      tTrackerTopo_(esConsumes()),
      verbose_(iConfig.getParameter<bool>("verbose"))      
        {
            rootFile_ = new TFile("config_output.root", "RECREATE");
            produces<std::vector<int>>("outputHits");
        }


trial::~trial() {
  if (rootFile_) {
    rootFile_->Close();
    delete rootFile_;
  }
}

void trial::beginStream(edm::StreamID) {
  // Initialize devices
  devices_ = cms::alpakatools::devices<Platform>();
  if (devices_.empty()) {
    edm::LogError("trial") << "No devices available for the backend. The test will be skipped.";
    return;
  }
  edm::LogInfo("trial") << "Found " << devices_.size() << " device(s).";
}


void trial::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    if (devices_.empty()) {
        edm::LogWarning("trial") << "Skipping event because no devices are available.";
        return;
    }


    // ---------------------------------------------------------------
    // RETRIEVE THE SOA COLLECTIONS TO BE USED IN THE KERNEL DEVICE
    // (THE FOLLOWING DATA ARE ALREADY ALPAKA-FRIENDLY)
    edm::Handle<ALPAKA_ACCELERATOR_NAMESPACE::SiPixelClustersSoACollection> clustersHandle;
    iEvent.getByToken(clusterToken_, clustersHandle);
    if (!clustersHandle.isValid()) {
        edm::LogError("trial") << "Could not retrieve SiPixelClusters.";
        return;
    }

    edm::Handle<ALPAKA_ACCELERATOR_NAMESPACE::SiPixelDigisSoACollection> digisHandle;
    iEvent.getByToken(digisToken_, digisHandle);
    if (!digisHandle.isValid()) {
        edm::LogError("trial") << "Could not retrieve SiPixelDigisSoA.";
        return;
    }

    edm::Handle<ALPAKA_ACCELERATOR_NAMESPACE::TrackingRecHitsSoACollection<pixelTopology::Phase1>> recHitsHandle;
    iEvent.getByToken(recHitsToken_, recHitsHandle);
    if (!recHitsHandle.isValid()) {
        edm::LogError("trial") << "Could not retrieve TrackingRecHitsSoA.";
        return;
    }

    edm::Handle<ALPAKA_ACCELERATOR_NAMESPACE::ZVertexSoACollection> zVertexHandle;
    iEvent.getByToken(zVertexToken_, zVertexHandle);
    if (!zVertexHandle.isValid()) {
        edm::LogError("trial") << "Could not retrieve zVertexHandle";
        return;
    }

    // Process TrackingRecHitsSoACollection
    const auto& recHits = *recHitsHandle;
    size_t nHits = recHits.nHits();
    if (verbose_) {
        std::cout << "Number of hits: " << nHits << std::endl;
    }

    // Process SiPixelDigisSoACollection
    const auto& digis = *digisHandle;
    size_t nDigis = digis.nDigis();
    if (verbose_) {
        std::cout << "Number of digis: " << nDigis << std::endl;
    }

    // Process SiPixelClustersSoACollection
    const auto& clusters = *clustersHandle;
    uint32_t nClusters = clusters.nClusters();
    if (verbose_) {
        std::cout << "Total clusters in this event: " << nClusters << std::endl;
    }

    // Process ZVertexSoACollection
    const auto& zVertices = *zVertexHandle;
    uint32_t nVertices = zVertices.view().nvFinal();
    if (verbose_) {
        std::cout << "Number of Vertices: " << nVertices << std::endl;
    }
    
    // -DONE WITH THE SOA STUFF-------------------------------------------------------------




    // --------------------------------------------------------------------------
    // RETRIEVE THE NON-SOA DATA, NAMELY "Candidate" and "SiPixelClusters"

    // Candidate is used for retrieving the jets --------------
    edm::Handle<edm::View<reco::Candidate>> candidatesHandle;
    iEvent.getByToken(candidateToken_, candidatesHandle);
    if (!candidatesHandle.isValid()) {
        edm::LogError("trial") << "Could not retrieve Candidate collection.";
        return;
    }
    // Process Candidates
    size_t nCandidates = candidatesHandle->size();  // Retrieve the number of candidates
    if (verbose_) {
        std::cout << "Number of Candidates: " << nCandidates << std::endl;
    }

    // Populates the custom-made CandidateSoA -------------------------------------------
    CandidateSoA candidatedataSoA;
    CandidateSoAView candidateView(candidatedataSoA);  // Accessor for the columns in the SoA

    // Iterate over the candidates and populate the CandidateSoA
    size_t candidateIndex = 0;
    for (const auto& candidate : *candidatesHandle) {
        if (candidate.pt() > ptMin_) {  // Apply the ptMin_ filter
            candidateView.px(candidateIndex) = static_cast<float>(candidate.px());
            candidateView.py(candidateIndex) = static_cast<float>(candidate.py());
            candidateView.pz(candidateIndex) = static_cast<float>(candidate.pz());
            candidateView.pt(candidateIndex) = static_cast<float>(candidate.pt());
            candidateView.eta(candidateIndex) = static_cast<float>(candidate.eta());
            candidateView.phi(candidateIndex) = static_cast<float>(candidate.phi());
            ++candidateIndex;
        }
    }
    // --------------------------------------------------------------------------
    

    // SiPixelClusters is used to get the geometry of each cluster ---------------
    edm::Handle<edmNew::DetSetVector<SiPixelCluster>> inputPixelClustersHandle;
    iEvent.getByToken(clusterToken_, inputPixelClustersHandle);
    if (!inputPixelClustersHandle.isValid()) {
        edm::LogError("trial") << "Could not retrieve SiPixelClusters.";
        return;
    }

    // Retrieve TrackerGeometry, trackerTopology from EventSetup
    const auto& trackingGeometry = iSetup.getData(tTrackingGeom_);
    const auto& trackerTopology = iSetup.getData(tTrackerTopo_);


    // Populates the custom-made ClusterGeometry SoA -------------------------------------------
    ClusterGeometrySoA dataSoA;
    ClusterGeometrySoAView myview(dataSoA);  // Accessor for the columns in the SoA

    // Loop through the SiPixelClusters and populate the ClusterGeometrySoA
    for (auto detIt = inputPixelClustersHandle->begin(); detIt != inputPixelClustersHandle->end(); ++detIt) {
        // detIt is now a reference to edmNew::DetSet<SiPixelCluster>
        const edmNew::DetSet<SiPixelCluster>& detset = *detIt;

        // Retrieve the GeomDet for this DetSet using its id
        const GeomDet* det = trackingGeometry.idToDet(detset.id());  // Correct usage of geometry

        if (!det) {
            continue;  // Skip invalid detector IDs
        }

        // Extract geometry information
        const PixelTopology& topo = static_cast<const PixelTopology&>(det->topology());
        float pitchX, pitchY;
        std::tie(pitchX, pitchY) = topo.pitch();
        float thickness = det->surface().bounds().thickness();
        float tanLorentzAngle = tanLorentzAngle_;  // Use the correct tanLorentzAngle from your context

        // Populate the ClusterGeometrySoA with the information
        size_t clusterIndex = 0;  // Track the index of the cluster in the DetSet
        for (const auto& cluster : detset) {
            // Access columns via the view and assign values
            myview.clusterIds(clusterIndex) = detset.id();  // Directly use the id() (no rawId())
            myview.pitchX(clusterIndex) = pitchX;
            myview.pitchY(clusterIndex) = pitchY;
            myview.thickness(clusterIndex) = thickness;
            myview.tanLorentzAngles(clusterIndex) = tanLorentzAngle;
            ++clusterIndex;
        }
    }
    // -DONE WITH THE NON SOA STUFF---------------------------------------------------------------------------



    // Use event ID as the offset
    int32_t eventOffset = iEvent.id().event();
    std::cout << "Event offset: " << eventOffset << std::endl;

    for (const auto& device : devices_) {
        Queue queue(device);

        // Define moduleStart data
        auto moduleStartH =
            cms::alpakatools::make_host_buffer<uint32_t[]>(queue, pixelTopology::Phase1::numberOfModules + 1);
        for (size_t i = 0; i < pixelTopology::Phase1::numberOfModules + 1; ++i) {
          moduleStartH[i] = i * 2;
        }
        auto moduleStartD =
            cms::alpakatools::make_device_buffer<uint32_t[]>(queue, pixelTopology::Phase1::numberOfModules + 1);
        alpaka::memcpy(queue, moduleStartD, moduleStartH);
        alpaka::wait(queue);            // Ensure the data copy is complete


        // ------------- CREATE DEVICE BUFFERS -------------------------------

        /* RecHits
           the TrackingRecHitsSoACollection is an alias for: TrackingRecHitDevice (gpu) 
                                                            TrackingRecHitHost (cpu)  */
        TrackingRecHitsSoACollection<pixelTopology::Phase1> tkHit(queue, nHits, eventOffset, moduleStartD.data());
        //- - - - - - - - - - - - - - - - - - -

        /* Digis 
        the SiPixelDigisSoACollection is an alias for: SiPixelDigisDevice (gpu) or 
                                                          SiPixelDigisHost (cpu)
        but it's not templated so <pixelTopology> won't work
        I could also: SiPixelDigisDevice<Device> digisDevice(nDigis, queue); */
        SiPixelDigisSoACollection tkDigi(nDigis, queue);
        tkDigi.setNModules(pixelTopology::Phase1::numberOfModules);         // Set additional metadata
        //- - - - - - - - - - - - - - - - - - -

        /* Clusters
           the SiPixelClustersSoACollection is an alias for: SiPixelClustersDevice (gpu) 
                                                             SiPixelClustersHost (cpu)  */
        SiPixelClustersSoACollection tkClusters(nClusters, queue); // It seems the above class has no topology and no Modules.. not sure why
        //- - - - - - - - - - - - - - - - - - -

        /* Candidates*/
        CandidateSoACollection tkCandidates(nCandidates, queue);
        auto CandidatesdeviceView = tkCandidates.view();
        //- - - - - - - - - - - - - - - - - - -


        /* Geometry*/
        ClusterGeometrySoACollection tkgeoclusters(nClusters, queue);
        auto deviceView = tkgeoclusters.view();
        //- - - - - - - - - - - - - - - - - - -


        /* Vertices                    */
        ZVertexSoACollection tkVertices(queue);
        //- - - - - - - - - - - - - - - - - - -

        /* SoA for the output                    */
        SiPixelDigisSoACollection tkOutputDigis(nDigis, queue);
        SiPixelClustersSoACollection tkOutputClusters(nClusters, queue);

        // ------------- COPY FROM HOST TO DEVICE BUFFERS -------------------------------
        // The output SoA are initialized with the input ones (in case no cluster will be split)
        alpaka::memcpy(queue, tkHit.buffer(), recHitsHandle->buffer());
        alpaka::memcpy(queue, tkDigi.buffer(), digisHandle->buffer());
        alpaka::memcpy(queue, tkClusters.buffer(), clustersHandle->buffer());
        alpaka::memcpy(queue, tkVertices.buffer(), zVertexHandle->buffer());
        alpaka::memcpy(queue, tkOutputDigis.buffer(), digisHandle->buffer());
        alpaka::memcpy(queue, tkOutputClusters.buffer(), clustersHandle->buffer());



        // Copy the Candidates into Device          
        CandidateHost CandidatehostGeometry(nCandidates, queue);  // Host-side wrapper
        auto CandidatehostView = CandidatehostGeometry.view();
        CandidateSoAView CandidatedataView(candidatedataSoA); 

        // Copy columns manually
        for (size_t i = 0; i < nCandidates; ++i) {
            CandidatehostView.candidateIndex(i) = CandidatedataView.candidateIndex(i);
            CandidatehostView.px(i) = CandidatedataView.px(i);
            CandidatehostView.py(i) = CandidatedataView.py(i);
            CandidatehostView.pz(i) = CandidatedataView.pz(i);
            CandidatehostView.pt(i) = CandidatedataView.pt(i);
            CandidatehostView.eta(i) = CandidatedataView.eta(i);
            CandidatehostView.phi(i) = CandidatedataView.phi(i);
        }
        alpaka::memcpy(queue, tkCandidates.buffer(), CandidatehostGeometry.buffer());


        // Copy the ClusterGeometry into Device
        ClusterGeometryHost hostGeometry(nClusters, queue);  // Host-side wrapper
        auto hostView = hostGeometry.view();
        ClusterGeometrySoAView dataView(dataSoA); 

        // Copy columns manually
        for (size_t i = 0; i < nClusters; ++i) {
            hostView.clusterIds(i) = dataView.clusterIds(i);
            hostView.pitchX(i) = dataView.pitchX(i);
            hostView.pitchY(i) = dataView.pitchY(i);
            hostView.thickness(i) = dataView.thickness(i);
            hostView.tanLorentzAngles(i) = dataView.tanLorentzAngles(i);
        }
        alpaka::memcpy(queue, tkgeoclusters.buffer(), hostGeometry.buffer());


        // Handling the per cluster calculation attributes in a struct
        std::vector<clusterProperties> gpuAlgo;
        auto clusterPropertiesHost = cms::alpakatools::make_host_buffer<clusterProperties[]>(queue, nClusters);
        auto clusterPropertiesDevice = cms::alpakatools::make_device_buffer<clusterProperties[]>(queue, nClusters);
        std::copy(gpuAlgo.begin(), gpuAlgo.end(), clusterPropertiesHost.data());
        alpaka::memcpy(queue, clusterPropertiesDevice, clusterPropertiesHost);

        // Handling a global counter of the output (new) clusters (initialized in the kernel)
        auto clusterCounterDevice = cms::alpakatools::make_device_buffer<uint32_t>(queue);

        alpaka::wait(queue);  // Ensure the transfer is complete

        // Execute the kernel
        Splitting::runKernels<pixelTopology::Phase1>(
            tkHit.view(), tkDigi.view(), tkClusters.view(), tkVertices.view(), tkCandidates.view(), 
            tkgeoclusters.view(), ptMin_, deltaR_, chargeFracMin_, 
            expSizeXAtLorentzAngleIncidence_, expSizeXDeltaPerTanAlpha_, expSizeYAtNormalIncidence_, 
            centralMIPCharge_, chargePerUnit_, fractionalWidth_, 
            tkOutputDigis.view(), tkOutputClusters.view(), 
            clusterPropertiesDevice.data(), clusterCounterDevice.data(),
            forceXError_, forceYError_, queue);


        // Update from device to host
        //alpaka::memcpy(queue, gpuSharedHost, gpuSharedDevice);  // Copy device buffer to host buffer
        //alpaka::wait(queue);  // Ensure the transfer is complete
        tkHit.updateFromDevice(queue);

        //TrackingRecHitHost<pixelTopology::Phase1> hostRecHits = cms::alpakatools::CopyToHost<TrackingRecHitDevice<pixelTopology::Phase1, Device>>::copyAsync(queue, tkHit);
        //SiPixelDigisHost digisHost = cms::alpakatools::CopyToHost<SiPixelDigisDevice<Device>>::copyAsync(queue, tkDigi);
        //SiPixelClustersHost clustersHost = cms::alpakatools::CopyToHost<SiPixelClustersDevice<Device>>::copyAsync(queue, tkClusters);
        SiPixelDigisHost outputDigisHost = cms::alpakatools::CopyToHost<SiPixelDigisDevice<Device>>::copyAsync(queue, tkOutputDigis);
        SiPixelClustersHost outputClustersHost = cms::alpakatools::CopyToHost<SiPixelClustersDevice<Device>>::copyAsync(queue, tkOutputClusters);

        alpaka::wait(queue);
    }
}



void trial::endStream() {
  edm::LogInfo("trial") << "Processing completed.";
}

void trial::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("configString", "default")->setComment("Configuration string to store in ROOT file");
  desc.add<uint32_t>("nHits", 100)->setComment("Number of hits for the test");
  desc.add<int32_t>("offset", 0)->setComment("Offset for hits");
  descriptions.add("trial", desc);
}

DEFINE_FWK_MODULE(trial);

#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED
