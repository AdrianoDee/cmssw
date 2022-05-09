#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexProducerCUDA.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "RecoVertex/VertexTools/interface/GeometricAnnealing.h"

#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_noncached_unique_ptr.h"

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

PrimaryVertexProducerCUDA::PrimaryVertexProducerCUDA(const edm::ParameterSet& conf)
    : theTTBToken(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))), theConfig(conf) {
  fVerbose = conf.getUntrackedParameter<bool>("verbose", false);

  trkToken = consumes<reco::TrackCollection>(conf.getParameter<edm::InputTag>("TrackLabel"));
  bsToken = consumes<reco::BeamSpot>(conf.getParameter<edm::InputTag>("beamSpotLabel"));
  f4D = false;

  // select and configure the track clusterizer
  std::string clusteringAlgorithm =
      conf.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<std::string>("algorithm");
  if (clusteringAlgorithm == "gap") {
    theTrackClusterizer = new GapClusterizerInZ(
        conf.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<edm::ParameterSet>("TkGapClusParameters"));
  } else if (clusteringAlgorithm == "DA") {
    theTrackClusterizer = new DAClusterizerInZ(
        conf.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<edm::ParameterSet>("TkDAClusParameters"));
  }
  // provide the vectorized version of the clusterizer, if supported by the build
  else if (clusteringAlgorithm == "DA_vect") {
    theTrackClusterizer = new DAClusterizerInZ_vect(
        conf.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<edm::ParameterSet>("TkDAClusParameters"));
  } else if (clusteringAlgorithm == "DASub_vect") {
    theTrackClusterizer = new DAClusterizerInZSubCluster_vect(
        conf.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<edm::ParameterSet>("TkDASubClusParameters"));
  } else if (clusteringAlgorithm == "DA2D_vect") {
    theTrackClusterizer = new DAClusterizerInZT_vect(
        conf.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<edm::ParameterSet>("TkDAClusParameters"));
    f4D = true;
  }
  else {
    throw VertexException("PrimaryVertexProducerCUDA: unknown clustering algorithm: " + clusteringAlgorithm);
  }

  if (f4D) {
    trkTimesToken = consumes<edm::ValueMap<float> >(conf.getParameter<edm::InputTag>("TrackTimesLabel"));
    trkTimeResosToken = consumes<edm::ValueMap<float> >(conf.getParameter<edm::InputTag>("TrackTimeResosLabel"));
  }

  // select and configure the vertex fitters
  if (conf.exists("vertexCollections")) {
    std::vector<edm::ParameterSet> vertexCollections =
        conf.getParameter<std::vector<edm::ParameterSet> >("vertexCollections");

    for (std::vector<edm::ParameterSet>::const_iterator algoconf = vertexCollections.begin();
         algoconf != vertexCollections.end();
         algoconf++) {
      algo algorithm;
      std::string fitterAlgorithm = algoconf->getParameter<std::string>("algorithm");
      if (fitterAlgorithm == "KalmanVertexFitter") {
        algorithm.fitter = new KalmanVertexFitter();
      } else if (fitterAlgorithm == "AdaptiveVertexFitter") {
        algorithm.fitter = new AdaptiveVertexFitter(GeometricAnnealing(algoconf->getParameter<double>("chi2cutoff")));
      } else {
        throw VertexException("PrimaryVertexProducerCUDA: unknown algorithm: " + fitterAlgorithm);
      }
      algorithm.label = algoconf->getParameter<std::string>("label");
      algorithm.minNdof = algoconf->getParameter<double>("minNdof");
      algorithm.useBeamConstraint = algoconf->getParameter<bool>("useBeamConstraint");
      algorithm.vertexSelector =
          new VertexCompatibleWithBeam(VertexDistanceXY(), algoconf->getParameter<double>("maxDistanceToBeam"));
      algorithms.push_back(algorithm);

      produces<reco::VertexCollection>(algorithm.label);
    }
  } else {
    edm::LogWarning("MisConfiguration")
        << "this module's configuration has changed, please update to have a vertexCollections=cms.VPSet parameter.";

    algo algorithm;
    std::string fitterAlgorithm = conf.getParameter<std::string>("algorithm");
    if (fitterAlgorithm == "KalmanVertexFitter") {
      algorithm.fitter = new KalmanVertexFitter();
    } else if (fitterAlgorithm == "AdaptiveVertexFitter") {
      algorithm.fitter = new AdaptiveVertexFitter();
    } else {
      throw VertexException("PrimaryVertexProducerCUDAAlgorithm: unknown algorithm: " + fitterAlgorithm);
    }
    algorithm.label = "";
    algorithm.minNdof = conf.getParameter<double>("minNdof");
    algorithm.useBeamConstraint = conf.getParameter<bool>("useBeamConstraint");

    algorithm.vertexSelector = new VertexCompatibleWithBeam(
        VertexDistanceXY(),
        conf.getParameter<edm::ParameterSet>("PVSelParameters").getParameter<double>("maxDistanceToBeam"));

    algorithms.push_back(algorithm);
    produces<reco::VertexCollection>(algorithm.label);
  }

  //check if this is a recovery iteration
  fRecoveryIteration = conf.getParameter<bool>("isRecoveryIteration");
  if (fRecoveryIteration) {
    if (algorithms.empty()) {
      throw VertexException("PrimaryVertexProducerCUDA: No algorithm specified. ");
    } else if (algorithms.size() > 1) {
      throw VertexException(
          "PrimaryVertexProducerCUDA: Running in Recovery mode and more than one algorithm specified.  Please "
          "only one algorithm.");
    }
    recoveryVtxToken = consumes<reco::VertexCollection>(conf.getParameter<edm::InputTag>("recoveryVtxCollection"));
  }
  onGPU_ = conf.getParameter<bool>("onGPU");
  if (onGPU_){
    fParams = {
     .maxSignificance=conf.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<double>("maxD0Significance"),
     .maxdxyError=conf.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<double>("maxD0Error"),
     .maxdzError=conf.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<double>("maxDzError"),
     .minpAtIP=conf.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<double>("minPt"),
     .maxetaAtIP=conf.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<double>("maxEta"),
     .maxchi2=conf.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<double>("maxNormalizedChi2"),
     .minpixelHits=conf.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<int>("minPixelLayersWithHits"),
     .mintrackerHits=conf.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<int>("minSiliconLayersWithHits"),
     // TODO:: Move this to the proper TkFilterParameters, as we do it in the filtering now
     .vertexSize=conf.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<edm::ParameterSet>("TkDAClusParameters").getParameter<double>("vertexSize"),
     .d0CutOff  =conf.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<edm::ParameterSet>("TkDAClusParameters").getParameter<double>("d0CutOff")
    };
    cParams = {
      .Tmin   = conf.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<edm::ParameterSet>("TkDAClusParameters").getParameter<double>("Tmin"),
      .Tpurge = conf.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<edm::ParameterSet>("TkDAClusParameters").getParameter<double>("Tpurge"),
      .Tstop  = conf.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<edm::ParameterSet>("TkDAClusParameters").getParameter<double>("Tstop"),
      .vertexSize = conf.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<edm::ParameterSet>("TkDAClusParameters").getParameter<double>("vertexSize"),
      .coolingFactor = conf.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<edm::ParameterSet>("TkDAClusParameters").getParameter<double>("coolingFactor"),
      .d0CutOff = conf.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<edm::ParameterSet>("TkDAClusParameters").getParameter<double>("d0CutOff"),
      .dzCutOff = conf.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<edm::ParameterSet>("TkDAClusParameters").getParameter<double>("dzCutOff"),
      .uniquetrkweight = conf.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<edm::ParameterSet>("TkDAClusParameters").getParameter<double>("uniquetrkweight"),
      .uniquetrkminp = conf.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<edm::ParameterSet>("TkDAClusParameters").getParameter<double>("uniquetrkminp"),
      .zmerge = conf.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<edm::ParameterSet>("TkDAClusParameters").getParameter<double>("zmerge"),
      .sel_zrange = conf.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<edm::ParameterSet>("TkDAClusParameters").getParameter<double>("zrange"),
      .convergence_mode = conf.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<edm::ParameterSet>("TkDAClusParameters").getParameter<int>("convergence_mode"),
      .delta_lowT = conf.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<edm::ParameterSet>("TkDAClusParameters").getParameter<double>("delta_lowT"),
      .delta_highT = conf.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<edm::ParameterSet>("TkDAClusParameters").getParameter<double>("delta_highT")
    };
  }
}

PrimaryVertexProducerCUDA::~PrimaryVertexProducerCUDA() {
  if (theTrackClusterizer)
    delete theTrackClusterizer;
  for (std::vector<algo>::const_iterator algorithm = algorithms.begin(); algorithm != algorithms.end(); algorithm++) {
    if (algorithm->fitter)
      delete algorithm->fitter;
    if (algorithm->vertexSelector)
      delete algorithm->vertexSelector;
  }
}

void PrimaryVertexProducerCUDA::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // get the BeamSpot, it will always be needed, even when not used as a constraint
  // if (onGPU_) std::cout << "I'm producing this on CUDA!!" << std::endl;
  reco::BeamSpot beamSpot;
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByToken(bsToken, recoBeamSpotHandle);
  if (recoBeamSpotHandle.isValid()) {
    beamSpot = *recoBeamSpotHandle;
  } else {
    edm::LogError("UnusableBeamSpot") << "No beam spot available from EventSetup";
  }

  bool validBS = true;
  VertexState beamVertexState(beamSpot);
  if ((beamVertexState.error().cxx() <= 0.) || (beamVertexState.error().cyy() <= 0.) ||
      (beamVertexState.error().czz() <= 0.)) {
    validBS = false;
    edm::LogError("UnusableBeamSpot") << "Beamspot with invalid errors " << beamVertexState.error().matrix();
  }

  //if this is a recovery iteration, check if we already have a valid PV
  if (fRecoveryIteration) {
    auto const& oldVertices = iEvent.get(recoveryVtxToken);
    //look for the first valid (not-BeamSpot) vertex
    for (auto const& old : oldVertices) {
      if (!(old.isFake())) {
        //found a valid vertex, write the first one to the collection and return
        //otherwise continue with regular vertexing procedure
        auto result = std::make_unique<reco::VertexCollection>();
        result->push_back(old);
        iEvent.put(std::move(result), algorithms.begin()->label);
        return;
      }
    }
  }

  // get RECO tracks from the event
  // `tks` can be used as a ptr to a reco::TrackCollection
  edm::Handle<reco::TrackCollection> tks;
  iEvent.getByToken(trkToken, tks);

  // interface RECO tracks to vertex reconstruction
  const auto& theB = &iSetup.getData(theTTBToken);
  std::vector<reco::TransientTrack> t_tks;

  if (f4D) {
    edm::Handle<edm::ValueMap<float> > trackTimesH;
    edm::Handle<edm::ValueMap<float> > trackTimeResosH;
    iEvent.getByToken(trkTimesToken, trackTimesH);
    iEvent.getByToken(trkTimeResosToken, trackTimeResosH);
    t_tks = (*theB).build(tks, beamSpot, *(trackTimesH.product()), *(trackTimeResosH.product()));
  } else {
    t_tks = (*theB).build(tks, beamSpot);
  }
  if (fVerbose) {
    std::cout << "RecoVertex/PrimaryVertexProducerCUDA"
              << "Found: " << t_tks.size() << " reconstructed tracks"
              << "\n";
  }
  ////////////////////////////////////////////////////////////////////
  ////////////////////// SoA DataFormat building /////////////////////
  ////////////////////////////////////////////////////////////////////
  // We need a copy living in the host, in between steps, at least if we want to do the things "as in CPU"

  unsigned int ntracks = t_tks.size();
  // printf("nTracks in CPU: %u \n", ntracks); //DEBUG
  TrackForPVHeterogeneous CPUtracks(cms::cuda::make_host_unique<TrackForPV::TrackForPVSoA>(cudaStreamDefault));  // By construction we iterate over 8096 tracks, 512 vertices
  TrackForPVHeterogeneous GPUtracks(cms::cuda::make_device_unique<TrackForPV::TrackForPVSoA>(cudaStreamDefault));// By construction we iterate over 8096 tracks, 512 vertices

  auto* CPUtracksObject = CPUtracks.get();
  auto* GPUtracksObject = GPUtracks.get();

  // Hackery, to get the input in a SoA format from the start -which would happen if the tracking is done in GPU, but would need to be adapted accordingly-
  // TODO::Cleanup of this loading
  
  unsigned int nTrueTracks = 0; 
  auto CPUosumtkwt = cms::cuda::make_host_unique<double[]>(1, cudaStreamDefault);                                         // 1/T, to be kept across iterations
  auto* CPUosumtkwtObject = CPUosumtkwt.get();

  double min_z = 10000;
  double max_z = -10000;
  //std::cout << "nTrueTracks" << "," << "z" << "," << "weight" << "," << "dz2" << std::endl;

  for (unsigned int idx=0; idx < t_tks.size() ; idx++){
    double significance		= t_tks.at(idx).stateAtBeamLine().transverseImpactParameter().significance();
    double dxy2		= t_tks.at(idx).stateAtBeamLine().transverseImpactParameter().error()*t_tks.at(idx).stateAtBeamLine().transverseImpactParameter().error();
    double dz2		= t_tks.at(idx).track().dzError()*t_tks.at(idx).track().dzError();
    double pAtIP		= t_tks.at(idx).impactPointState().globalMomentum().transverse();
    double pxAtPCA		= t_tks.at(idx).stateAtBeamLine().trackStateAtPCA().momentum().x();
    double pyAtPCA		= t_tks.at(idx).stateAtBeamLine().trackStateAtPCA().momentum().y();
    double pzAtPCA		= t_tks.at(idx).stateAtBeamLine().trackStateAtPCA().momentum().z();
    double bx		= t_tks.at(idx).stateAtBeamLine().beamSpot().BeamWidthX();
    double by		= t_tks.at(idx).stateAtBeamLine().beamSpot().BeamWidthY();
    double z		= t_tks.at(idx).stateAtBeamLine().trackStateAtPCA().position().z();
    double etaAtIP		= std::fabs(t_tks.at(idx).impactPointState().globalMomentum().eta());
    double chi2		= t_tks.at(idx).normalizedChi2();
    int8_t nPixelHits		= t_tks.at(idx).hitPattern().pixelLayersWithMeasurement();
    int8_t nTrackerHits		= t_tks.at(idx).hitPattern().trackerLayersWithMeasurement();

    bool isGood = false;
    double weight = 0;
      if (significance < fParams.maxSignificance){
        if (dxy2 < fParams.maxdxyError*fParams.maxdxyError){
          if (dz2 < fParams.maxdzError*fParams.maxdzError){
            if (pAtIP > fParams.minpAtIP){
              if (std::fabs(etaAtIP) < fParams.maxetaAtIP){
                if (chi2 < fParams.maxchi2){
                  if (nPixelHits >= fParams.minpixelHits){
                    if (nTrackerHits >= fParams.mintrackerHits){
                      isGood = true;
                    }
                  }
                }
              }
            }
          }
        }
      }
      // And now the stuff for the clusterizer
      if (isGood){
        weight = 1.;  
        if (std::fabs(z) > 1000.){ 
          isGood = false;
          weight = 0;
          continue;
        }
        else{ // Get dz2 for the track
          // dz2 is zerror^2 + (bx*px + by*py)^2*pz^2/(pt^4) + vertex_size^2
          dz2 = dz2 
                         + (bx*bx*pxAtPCA*pxAtPCA + by*by*pyAtPCA*pyAtPCA)* pzAtPCA*pzAtPCA/(( pxAtPCA*pxAtPCA + pyAtPCA*pyAtPCA )* ( pxAtPCA*pxAtPCA + pyAtPCA*pyAtPCA)) 
                         + fParams.vertexSize*fParams.vertexSize; // TODO:: For sure ways to optimize this
          dz2 = 1./dz2;
          if (not(std::isfinite(dz2)) || dz2< std::numeric_limits<double>::min()){ // Bad track dz2 is taken out
            isGood = false;
            weight = 0;
            continue;
          }
          else{
            if (fParams.d0CutOff > 0){ // Track weights are activated only if there is a non-zero cutoff
              // weight is 1/(1 + e^{sig^2 - d0cutoff^2})
              weight = 1./ (1+exp(significance*significance - fParams.d0CutOff*fParams.d0CutOff ));
              if (not(std::isfinite(weight)) || weight< std::numeric_limits<double>::epsilon()){ // Bad track weight is taken out
                isGood = false;
                weight = 0;
                continue;
              }
            }
            // If we are here, the track is to be passed to the clusterizer. So initialize the clusterizer stuff
            // really save track now!
            //if (nTrueTracks > CPUtracksObject->stride()){
            if (nTrueTracks > 1023){
                //std::cout << "Error, size of tracks SoA is too small: " << CPUtracksObject->stride() << " while tracks are " << t_tks.size() << std::endl;
                break;
            }
            (*CPUosumtkwtObject) += weight;
            CPUtracksObject->z(nTrueTracks) = z;
            CPUtracksObject->weight(nTrueTracks) = weight;
            CPUtracksObject->tt_index(nTrueTracks) = idx;
            CPUtracksObject->dz2(nTrueTracks) = dz2;
            CPUtracksObject->order(nTrueTracks) = nTrueTracks;
            CPUtracksObject->sum_Z(nTrueTracks) = 0;
            CPUtracksObject->kmin(nTrueTracks) = 0; // will loop from kmin to kmax-1. At the start only one vertex
            CPUtracksObject->kmax(nTrueTracks) = 1;
            CPUtracksObject->aux1(nTrueTracks) = 0;
            CPUtracksObject->aux2(nTrueTracks) = 0;
            //std::cout << nTrueTracks << "," << z << "," << weight << "," << dz2 << std::endl;
            nTrueTracks++;
//            if (z > max_z) max_z = z;
//            if (z < min_z) min_z = z;
          }
        }
      }
  }
  CPUtracksObject->nTrueTracks = nTrueTracks;
  //std::cout << "nTrueTracks in producer: " << nTrueTracks << std::endl;
  
  (*CPUosumtkwtObject) = (*CPUosumtkwtObject) > 0 ? 1./(*CPUosumtkwtObject) : 0.; 

  ////////////////////////////////////////////////////////////////////
  ////////////////////// Track filtering on GPU //////////////////////
  ////////////////////////////////////////////////////////////////////
//  std::cout << "Begin copying 1" << std::endl;
//  std::cout << "size of tracks: " << sizeof(TrackForPV::TrackForPVSoA) << std::endl;
//  std::cout << "size of vertices: " << sizeof(TrackForPV::VertexForPVSoA) << std::endl;
  cudaCheck(cudaMemcpy(GPUtracksObject, CPUtracksObject, sizeof(TrackForPV::TrackForPVSoA), cudaMemcpyHostToDevice));
//  std::cout << "Finished copying 1\nBegin copying osumtkwt" << std::endl;
  auto osumtkwt      = cms::cuda::make_device_unique<double[]>(1, cudaStreamDefault); //Sum of all track weights, for the clusterizer later
  cudaCheck(cudaMemcpy(osumtkwt.get(), CPUosumtkwtObject, sizeof(double), cudaMemcpyHostToDevice));
//  std::cout << "End copying 2" << std::endl;
  trackFilterCUDA::sorterWrapper(ntracks, GPUtracksObject, cudaStreamDefault); //TODO:: We can also consider a minidataformat for the beamspot in GPU

//  trackFilterCUDA::filterWrapper(ntracks, GPUtracksObject, fParams, osumtkwt.get(), cudaStreamDefault); //TODO:: We can also consider a minidataformat for the beamspot in GPU


  ////////////////////////////////////////////////////////////////////
  ////////////////////// Clustering on GPU ///////////////////////////
  ////////////////////////////////////////////////////////////////////
  
  // First, object creation
  VertexForPVHeterogeneous CPUvertices(cms::cuda::make_host_unique<TrackForPV::VertexForPVSoA>(cudaStreamDefault));  // By construction we iterate over 8096 tracks, 512 vertices
  VertexForPVHeterogeneous GPUvertices(cms::cuda::make_device_unique<TrackForPV::VertexForPVSoA>(cudaStreamDefault));// By construction we iterate over 512 vertices
  auto* CPUverticesObject = CPUvertices.get();
  auto* GPUverticesObject = GPUvertices.get();

  auto CPUbeta = cms::cuda::make_host_unique<double[]>(1, cudaStreamDefault);                                         // 1/T, to be kept across iterations
  auto GPUbeta = cms::cuda::make_device_unique<double[]>(1, cudaStreamDefault);                                         // 1/T, to be kept across iterations
   
//  // Add first vertex, init all collections
//  clusterizerCUDA::initializeWrapper(ntracks, GPUtracksObject, GPUverticesObject, GPUbeta.get(), osumtkwt.get(), cParams, cudaStreamDefault);
//  // Estimate first critical temperature
//  clusterizerCUDA::getBeta0Wrapper(ntracks, GPUtracksObject, GPUverticesObject, GPUbeta.get(), osumtkwt.get(), cParams, cudaStreamDefault);
//  // First thermalization
//  clusterizerCUDA::thermalizeWrapper(ntracks, GPUtracksObject, GPUverticesObject, GPUbeta.get(), osumtkwt.get(), cParams, cudaStreamDefault);
//  // First T loop, includes splitting and merging
//  clusterizerCUDA::coolingWhileSplittingWrapper(ntracks, GPUtracksObject, GPUverticesObject, GPUbeta.get(), osumtkwt.get(), cParams, cudaStreamDefault);
//  // Without varying T, reassign tracks to vertices and possibly merge more
//  clusterizerCUDA::remergeTracksWrapper(ntracks, GPUtracksObject, GPUverticesObject, GPUbeta.get(), osumtkwt.get(), cParams, cudaStreamDefault);
//  // Without varying T, redo splitting with increasingly relaxed criteria
//  clusterizerCUDA::resplitTracksWrapper(ntracks, GPUtracksObject, GPUverticesObject, GPUbeta.get(), osumtkwt.get(), cParams, cudaStreamDefault);
//  // Outlier rejection at fixed T, low quality vertex purging and final cooling down to the stopping criteria
//  clusterizerCUDA::outlierRejectionWrapper(ntracks, GPUtracksObject, GPUverticesObject, GPUbeta.get(), osumtkwt.get(), cParams, cudaStreamDefault);
//  
 
//  std::cout << "Begin kernel" << std::endl;
  clusterizerCUDA::bigKernelWrapper(ntracks, GPUtracksObject, GPUverticesObject, GPUbeta.get(), osumtkwt.get(), cParams, cudaStreamDefault);
//  std::cout << "End kernel" << std::endl;
  
   
  ///// TODO:: update this when we put the fitter into GPU as well ////
  
  //cudaCheck(cudaFree(GPUverticesObject));
  //cudaCheck(cudaFree(CPUtracksObject));
  //cudaCheck(cudaFree(GPUtracksObject));
  //cudaCheck(cudaFree(beta.get()));
  //cudaCheck(cudaFree(osumtkwt.get()));

//  std::cout << "Begin copying back" << std::endl;
//  std::cout << "size of vertices: " << sizeof(TrackForPV::VertexForPVSoA) << std::endl;
  cudaCheck(cudaMemcpy(CPUverticesObject, GPUverticesObject, sizeof(TrackForPV::VertexForPVSoA), cudaMemcpyDeviceToHost));
//  std::cout << "End copying back" << std::endl;
//  std::cout << "Begin copying back 2" << std::endl;
  cudaCheck(cudaMemcpy(CPUtracksObject, GPUtracksObject, sizeof(TrackForPV::TrackForPVSoA), cudaMemcpyDeviceToHost));
//  std::cout << "End copying back 2" << std::endl;
  //unsigned int gridSize  = 32;
  //clusterizerCUDA::dumpTV(CPUtracksObject, CPUverticesObject, gridSize);

  cudaCheck(cudaMemcpy(CPUbeta.get(), GPUbeta.get(), sizeof(double), cudaMemcpyDeviceToHost));
//  cudaCheck(cudaFree(GPUverticesObject));
 // cudaCheck(cudaFree(GPUtracksObject));
  //cudaCheck(cudaFree(d_obj_ptr));  
//  //std::cout << "Finished copying 2" << std::endl;
  std::vector<TransientVertex> pv = clusterizerCUDA::vertices(ntracks, CPUtracksObject, CPUverticesObject, cParams, t_tks, CPUbeta.get());
  // clusterize tracks in Z
  std::vector<std::vector<reco::TransientTrack> >&& clusters = clusterizerCUDA::clusterize(pv, cParams);
 // cudaCheck(cudaFree(CPUverticesObject));
 // cudaCheck(cudaFree(CPUtracksObject));
  cudaCheck(cudaDeviceSynchronize());
  ////////////////////////////////////////////////////////////////////
  ////////////////////// Fitting on GPU //////////////////////////////
  ////////////////////////////////////////////////////////////////////
  //std::vector<reco::TransientTrack> seltks;
  // std::vector<std::vector<reco::TransientTrack> > clusters;
  
  // std::vector<std::vector<reco::TransientTrack> > clusters;

  // vertex fits
  for (std::vector<algo>::const_iterator algorithm = algorithms.begin(); algorithm != algorithms.end(); algorithm++) {
    auto result = std::make_unique<reco::VertexCollection>();
    reco::VertexCollection& vColl = (*result);

    std::vector<TransientVertex> pvs;
    for (std::vector<std::vector<reco::TransientTrack> >::const_iterator iclus = clusters.begin();
         iclus != clusters.end();
         iclus++) {
      double sumwt = 0.;
      double sumwt2 = 0.;
      double sumw = 0.;
      double meantime = 0.;
      double vartime = 0.;
      if (f4D) {
        for (const auto& tk : *iclus) {
          const double time = tk.timeExt();
          const double err = tk.dtErrorExt();
          const double inverr = err > 0. ? 1.0 / err : 0.;
          const double w = inverr * inverr;
          sumwt += w * time;
          sumwt2 += w * time * time;
          sumw += w;
        }
        meantime = sumwt / sumw;
        double sumsq = sumwt2 - sumwt * sumwt / sumw;
        double chisq = iclus->size() > 1 ? sumsq / double(iclus->size() - 1) : sumsq / double(iclus->size());
        vartime = chisq / sumw;
      }

      TransientVertex v;
      if (algorithm->useBeamConstraint && validBS && (iclus->size() > 1)) {
        v = algorithm->fitter->vertex(*iclus, beamSpot);
      } else if (!(algorithm->useBeamConstraint) && (iclus->size() > 1)) {
        v = algorithm->fitter->vertex(*iclus);
      }  // else: no fit ==> v.isValid()=False
     /*
     float x=0,y=0,z=0, wxy=0, wz=0;
     for (const auto& itrack : *iclus){ 
     //std::vector<reco::TransientTrack>::const_iterator itrack = iclus.begin(); itrack!= iclus.end(); itrack++) {
            x += itrack.impactPointState().globalPosition().x() * pow(itrack.track().dxyError(),2);
            y += itrack.impactPointState().globalPosition().y() * pow(itrack.track().dxyError(),2);
            wxy += pow(itrack.track().dxyError(),2);
            z += itrack.impactPointState().globalPosition().z() * pow(itrack.track().dzError(),2);
     }
     x /= (*iclus).size() * wxy;     
     y /= (*iclus).size() * wxy;     
     z /= (*iclus).size() * wz;     
     TransientVertex v2(GlobalPoint(x,y,z), v.positionError(), (*iclus), v.totalChiSquared());
     v = v2;
      */
     //std::cout << v.position().x() << ", " << x << "\t" <<  v.position().y() << ", " << y << "\t" << v.position().z() << ", " << z << "\t" << std::endl; 
     
      // 4D vertices: add timing information
      if (f4D and v.isValid()) {
        auto err = v.positionError().matrix4D();
        err(3, 3) = vartime;
        auto trkWeightMap3d = v.weightMap();  // copy the 3d-fit weights
        v = TransientVertex(v.position(), meantime, err, v.originalTracks(), v.totalChiSquared(), v.degreesOfFreedom());
        v.weightMap(trkWeightMap3d);
      }

      if (fVerbose) {
        if (v.isValid()) {
          std::cout << "x,y,z";
          if (f4D)
            std::cout << ",t";
          std::cout << "=" << v.position().x() << " " << v.position().y() << " " << v.position().z();
          if (f4D)
            std::cout << " " << v.time();
          std::cout << " cluster size = " << (*iclus).size() << std::endl;
        } else {
          std::cout << "Invalid fitted vertex,  cluster size=" << (*iclus).size() << std::endl;
        }
      }

      if (v.isValid() && (v.degreesOfFreedom() >= algorithm->minNdof) &&
          (!validBS || (*(algorithm->vertexSelector))(v, beamVertexState)))
        pvs.push_back(v);
    }  // end of cluster loop

    if (fVerbose) {
      std::cout << "PrimaryVertexProducerCUDAAlgorithm::vertices  candidates =" << pvs.size() << std::endl;
    }

    if (clusters.size() > 2 && clusters.size() > 2 * pvs.size())
      edm::LogWarning("PrimaryVertexProducerCUDA")
          << "more than half of candidate vertices lost " << pvs.size() << ' ' << clusters.size();
    /*
    if (pvs.empty() && seltks.size() > 5)
      edm::LogWarning("PrimaryVertexProducerCUDA")
          << "no vertex found with " << seltks.size() << " tracks and " << clusters.size() << " vertex-candidates";
    */
    // sort vertices by pt**2  vertex (aka signal vertex tagging)
    if (pvs.size() > 1) {
      sort(pvs.begin(), pvs.end(), VertexHigherPtSquared());
    }

    // convert transient vertices returned by the theAlgo to (reco) vertices
    for (std::vector<TransientVertex>::const_iterator iv = pvs.begin(); iv != pvs.end(); iv++) {
      reco::Vertex v = *iv;
      vColl.push_back(v);
    }

    if (vColl.empty()) {
      GlobalError bse(beamSpot.rotatedCovariance3D());
      if ((bse.cxx() <= 0.) || (bse.cyy() <= 0.) || (bse.czz() <= 0.)) {
        AlgebraicSymMatrix33 we;
        we(0, 0) = 10000;
        we(1, 1) = 10000;
        we(2, 2) = 10000;
        vColl.push_back(reco::Vertex(beamSpot.position(), we, 0., 0., 0));
        if (fVerbose) {
          std::cout << "RecoVertex/PrimaryVertexProducerCUDA: "
                    << "Beamspot with invalid errors " << bse.matrix() << std::endl;
          std::cout << "Will put Vertex derived from dummy-fake BeamSpot into Event.\n";
        }
      } else {
        vColl.push_back(reco::Vertex(beamSpot.position(), beamSpot.rotatedCovariance3D(), 0., 0., 0));
        if (fVerbose) {
          std::cout << "RecoVertex/PrimaryVertexProducerCUDA: "
                    << " will put Vertex derived from BeamSpot into Event.\n";
        }
      }
    }

    if (fVerbose) {
      int ivtx = 0;
      for (reco::VertexCollection::const_iterator v = vColl.begin(); v != vColl.end(); ++v) {
        std::cout << "recvtx " << ivtx++ << "#trk " << std::setw(3) << v->tracksSize() << " chi2 " << std::setw(4)
                  << v->chi2() << " ndof " << std::setw(3) << v->ndof() << " x " << std::setw(6) << v->position().x()
                  << " dx " << std::setw(6) << v->xError() << " y " << std::setw(6) << v->position().y() << " dy "
                  << std::setw(6) << v->yError() << " z " << std::setw(6) << v->position().z() << " dz " << std::setw(6)
                  << v->zError();
        if (f4D) {
          std::cout << " t " << std::setw(6) << v->t() << " dt " << std::setw(6) << v->tError();
        }
        std::cout << std::endl;
      }
    }
    /*
    int ivtx = 0;
    std::cout << "recvtx,#trk,chi2,ndof,x,dx,y,dy,z,dz" << std::endl;
    for (reco::VertexCollection::const_iterator v = vColl.begin(); v != vColl.end(); ++v) {
      std::cout << ivtx++ << "," << v->tracksSize() << "," << v->chi2() << ","  << v->ndof() << ","  << v->position().x()
                << ","  << v->xError() << ","  << v->position().y() << ","
                 << v->yError() << ","  << v->position().z() << "," 
                << v->zError();
      std::cout << std::endl;
    }
    */
    iEvent.put(std::move(result), algorithm->label);
  }
}

void PrimaryVertexProducerCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // offlinePrimaryVertices
  edm::ParameterSetDescription desc;
  {
    edm::ParameterSetDescription vpsd1;
    vpsd1.add<double>("maxDistanceToBeam", 1.0);
    vpsd1.add<std::string>("algorithm", "AdaptiveVertexFitter");
    vpsd1.add<bool>("useBeamConstraint", false);
    vpsd1.add<std::string>("label", "");
    vpsd1.add<double>("chi2cutoff", 2.5);
    vpsd1.add<double>("minNdof", 0.0);
    std::vector<edm::ParameterSet> temp1;
    temp1.reserve(2);
    {
      edm::ParameterSet temp2;
      temp2.addParameter<double>("maxDistanceToBeam", 1.0);
      temp2.addParameter<std::string>("algorithm", "AdaptiveVertexFitter");
      temp2.addParameter<bool>("useBeamConstraint", false);
      temp2.addParameter<std::string>("label", "");
      temp2.addParameter<double>("chi2cutoff", 2.5);
      temp2.addParameter<double>("minNdof", 0.0);
      temp1.push_back(temp2);
    }
    {
      edm::ParameterSet temp2;
      temp2.addParameter<double>("maxDistanceToBeam", 1.0);
      temp2.addParameter<std::string>("algorithm", "AdaptiveVertexFitter");
      temp2.addParameter<bool>("useBeamConstraint", true);
      temp2.addParameter<std::string>("label", "WithBS");
      temp2.addParameter<double>("chi2cutoff", 2.5);
      temp2.addParameter<double>("minNdof", 2.0);
      temp1.push_back(temp2);
    }
    desc.addVPSet("vertexCollections", vpsd1, temp1);
  }
  desc.addUntracked<bool>("verbose", false);
  {
    edm::ParameterSetDescription psd0;
    TrackFilterForPVFinding::fillPSetDescription(psd0);
    psd0.add<int>("numTracksThreshold", 0);  // HI only
    desc.add<edm::ParameterSetDescription>("TkFilterParameters", psd0);
  }
  desc.add<edm::InputTag>("beamSpotLabel", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("TrackLabel", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("TrackTimeResosLabel", edm::InputTag("dummy_default"));  // 4D only
  desc.add<edm::InputTag>("TrackTimesLabel", edm::InputTag("dummy_default"));      // 4D only

  {
    edm::ParameterSetDescription psd0;
    {
      edm::ParameterSetDescription psd1;
      DAClusterizerInZT_vect::fillPSetDescription(psd1);
      psd0.add<edm::ParameterSetDescription>("TkDAClusParameters", psd1);

      edm::ParameterSetDescription psd2;
      GapClusterizerInZ::fillPSetDescription(psd2);
      psd0.add<edm::ParameterSetDescription>("TkGapClusParameters", psd2);

      edm::ParameterSetDescription psd3;
      DAClusterizerInZSubCluster_vect::fillPSetDescription(psd3);
      psd0.add<edm::ParameterSetDescription>("TkDASubClusParameters", psd3);
    }
    psd0.add<std::string>("algorithm", "DA_vect");
    desc.add<edm::ParameterSetDescription>("TkClusParameters", psd0);
  }

  desc.add<bool>("isRecoveryIteration", false);
  desc.add<edm::InputTag>("recoveryVtxCollection", {""});
  desc.add<bool>("onGPU", true);
  descriptions.add("primaryVertexProducerCUDA", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PrimaryVertexProducerCUDA);
