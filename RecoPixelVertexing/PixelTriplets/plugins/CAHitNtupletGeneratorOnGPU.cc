//
// Original Author: Felice Pantaleo, CERN
//

#include <array>
#include <cassert>
#include <functional>
#include <vector>

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"

#include "CAHitNtupletGeneratorOnGPU.h"

namespace {

  template <typename T>
  T sqr(T x) {
    return x * x;
  }

  cAHitNtupletGenerator::QualityCuts makeQualityCuts(edm::ParameterSet const& pset) {
    auto coeff = pset.getParameter<std::vector<double>>("chi2Coeff");
    if (coeff.size() != 4) {
      throw edm::Exception(edm::errors::Configuration,
                           "CAHitNtupletGeneratorOnGPU.trackQualityCuts.chi2Coeff must have 4 elements");
    }
    return cAHitNtupletGenerator::QualityCuts{// polynomial coefficients for the pT-dependent chi2 cut
                                              {(float)coeff[0], (float)coeff[1], (float)coeff[2], (float)coeff[3]},
                                              // max pT used to determine the chi2 cut
                                              (float)pset.getParameter<double>("chi2MaxPt"),
                                              // chi2 scale factor: 30 for broken line fit, 45 for Riemann fit
                                              (float)pset.getParameter<double>("chi2Scale"),
                                              // regional cuts for triplets
                                              {
                                               (float)pset.getParameter<double>("tripletChi2MaxPt"),
                                               (float)pset.getParameter<double>("tripletMaxTip"),
                                               (float)pset.getParameter<double>("tripletMinPt"),
                                               (float)pset.getParameter<double>("tripletMaxZip")},
                                              // regional cuts for quadruplets
                                              {
                                               (float)pset.getParameter<double>("chi2MaxPt"),
                                               (float)pset.getParameter<double>("quadrupletMaxTip"),
                                               (float)pset.getParameter<double>("quadrupletMinPt"),
                                               (float)pset.getParameter<double>("quadrupletMaxZip")}};
  }

  // cAHitNtupletGenerator::VertexRegion makeRegionCuts(edm::ParameterSet const& pset) {
  //
  //   // edm::EDGetTokenT<reco::VertexCollection> tokenVertex_ = iC.consumes<reco::VertexCollection>(pset.getParameter<edm::InputTag>("inputVertices"));
  //   // edm::Handle<reco::VertexCollection> vtxs;
  //   // iEvent.getByToken(vtxToken_, vtxs);
  //
  //   // zVertices_
  //   // rVertices_
  //   // zVerticesError_
  //   // rVerticesError_
  //   auto zVertices = pset.getParameter<std::vector<double>>("zVertices");
  //   auto zVerticesError = pset.getParameter<std::vector<double>>("zVerticesError");
  //   auto r = pset.getParameter<std::vector<double>>("rVertices");
  //   auto rErr = pset.getParameter<std::vector<double>>("rVerticesError");
  //
  //   // auto zV = pset.getParameter<std::vector<double>>("zVertices");
  //   // auto rV = pset.getParameter<std::vector<double>>("rVertices");
  //   // auto zVErr = pset.getParameter<std::vector<double>>("zVerticesError");
  //
  //   float vtxZ[10] = {0.0};
  //   // float vtxR = 0.0;
  //   float vtxZerr[10] = {9999.9};
  //   // float vtxRerr = 9999.9;
  //
  //   if (zVerticesError.size()!=zVertices.size())
  //   {
  //     throw edm::Exception(edm::errors::Configuration,
  //                          "CAHitNtupletGeneratorOnGPU.zVertices & zVerticesError must have the same lenght (<=10).");
  //   }
  //   if (zVertices.size() >= 10) {
  //      edm::LogWarning("CAHitNtupletGeneratorOnGPU") << "You gave in input more than 10 vertices. Only the first 10 will be taken into account." << std::endl;
  //   }
  //   int c = std::min(10,int(zVertices.size()));
  //
  //   for (int i = 0; i < c; i++) {
  //     vtxZ[i] = (float) zVertices[i];
  //     vtxZerr[i] = (float) zVerticesError[i];
  //   }
  //
  //   return cAHitNtupletGenerator::VertexRegion{
  //                                               { vtxZ[0], vtxZ[1], vtxZ[2], vtxZ[3], vtxZ[4], vtxZ[5], vtxZ[6], vtxZ[7], vtxZ[8], vtxZ[9]} ,
  //                                               (float) r[0],
  //                                               { vtxZerr[0], vtxZerr[1], vtxZerr[2], vtxZerr[3], vtxZerr[4], vtxZerr[5], vtxZerr[6], vtxZerr[7], vtxZerr[8], vtxZerr[9]} ,
  //                                               (float) rErr[0]
  //                                             };
  // }

}  // namespace

using namespace std;

CAHitNtupletGeneratorOnGPU::CAHitNtupletGeneratorOnGPU(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC)
    : m_params(cfg.getParameter<bool>("onGPU"),
               cfg.getParameter<unsigned int>("minHitsPerNtuplet"),
               cfg.getParameter<unsigned int>("maxNumberOfDoublets"),
               cfg.getParameter<bool>("useRiemannFit"),
               cfg.getParameter<bool>("fit5as4"),
               cfg.getParameter<bool>("includeJumpingForwardDoublets"),
               cfg.getParameter<bool>("earlyFishbone"),
               cfg.getParameter<bool>("lateFishbone"),
               cfg.getParameter<bool>("idealConditions"),
               cfg.getParameter<bool>("fillStatistics"),
               cfg.getParameter<bool>("doClusterCut"),
               cfg.getParameter<bool>("doZ0Cut"),
               cfg.getParameter<bool>("doPtCut"),
               cfg.getParameter<double>("ptmin"),
               cfg.getParameter<double>("CAThetaCutBarrel"),
               cfg.getParameter<double>("CAThetaCutForward"),
               cfg.getParameter<double>("hardCurvCut"),
               cfg.getParameter<double>("dcaCutInnerTriplet"),
               cfg.getParameter<double>("dcaCutOuterTriplet"),
               cfg.getParameter<bool>("isUpgrade"),
               cfg.getParameter<bool>("doRegion"),
               // cfg.getParameter<std::vector<double>>("zVertices"),
               // cfg.getParameter<std::vector<double>>("zVerticesError"),
               // cfg.getParameter<std::vector<double>>("rVertices"),
               // cfg.getParameter<std::vector<double>>("rVerticesError"),
               makeQualityCuts(cfg.getParameterSet("trackQualityCuts")))
               // makeRegionCuts(cfg,iC)
               // cfg.getParameter<VertexRegion>("vertexRegion"))
               {
#ifdef DUMP_GPU_TK_TUPLES
  printf("TK: %s %s % %s %s %s %s %s %s %s %s %s %s %s %s %s\n",
         "tid",
         "qual",
         "nh",
         "charge",
         "pt",
         "eta",
         "phi",
         "tip",
         "zip",
         "chi2",
         "h1",
         "h2",
         "h3",
         "h4",
         "h5");
#endif

  if (m_params.onGPU_) {
    cudaCheck(cudaMalloc(&m_counters, sizeof(Counters)));
    cudaCheck(cudaMemset(m_counters, 0, sizeof(Counters)));
  } else {
    m_counters = new Counters();
    memset(m_counters, 0, sizeof(Counters));
  }
}

CAHitNtupletGeneratorOnGPU::~CAHitNtupletGeneratorOnGPU() {
  if (m_params.doStats_) {
    // crash on multi-gpu processes
    if (m_params.onGPU_) {
      CAHitNtupletGeneratorKernelsGPU::printCounters(m_counters);
    } else {
      CAHitNtupletGeneratorKernelsCPU::printCounters(m_counters);
    }
  }
  if (m_params.onGPU_) {
    cudaFree(m_counters);
  } else {
    delete m_counters;
  }
}

void CAHitNtupletGeneratorOnGPU::fillDescriptions(edm::ParameterSetDescription& desc) {
  // 87 cm/GeV = 1/(3.8T * 0.3)
  // take less than radius given by the hardPtCut and reject everything below
  // auto hardCurvCut = 1.f/(0.35 * 87.f);
  desc.add<double>("ptmin", 0.9f)->setComment("Cut on minimum pt");
  desc.add<double>("CAThetaCutBarrel", 0.002f)->setComment("Cut on RZ alignement for Barrel");
  desc.add<double>("CAThetaCutForward", 0.003f)->setComment("Cut on RZ alignment for Forward");
  desc.add<double>("hardCurvCut", 1.f / (0.35 * 87.f))->setComment("Cut on minimum curvature");
  desc.add<double>("dcaCutInnerTriplet", 0.15f)->setComment("Cut on origin radius when the inner hit is on BPix1");
  desc.add<double>("dcaCutOuterTriplet", 0.25f)->setComment("Cut on origin radius when the outer hit is on BPix1");
  desc.add<bool>("earlyFishbone", true);
  desc.add<bool>("lateFishbone", false);
  desc.add<bool>("idealConditions", true);
  desc.add<bool>("fillStatistics", false);
  desc.add<unsigned int>("minHitsPerNtuplet", 4);
  desc.add<unsigned int>("maxNumberOfDoublets", CAConstants::maxNumberOfDoublets());
  desc.add<bool>("includeJumpingForwardDoublets", false);
  desc.add<bool>("fit5as4", true);
  desc.add<bool>("doClusterCut", true);
  desc.add<bool>("doZ0Cut", true);
  desc.add<bool>("doPtCut", true);
  desc.add<bool>("useRiemannFit", false)->setComment("true for Riemann, false for BrokenLine");
  desc.add<bool>("isUpgrade", false);
  desc.add<bool>("doRegion", false);
  // desc.add<VertexRegion>("vertexRegion", {{0.0},0.0,{9999.9},9999.9})->setComment("Zs for vertices for region selection.");
  // desc.add<std::vector<double>>("rVertices", {0.0})->setComment("Rs for vertices for region selection.");
  // desc.add<std::vector<double>>("zVerticesError", {0.0})->setComment("Zerrs for vertices for region selection.");
  // desc.add<edm::InputTag>("inputVertices", edm::InputTag(""))->setComment("Input vertices for region selection.");
  edm::ParameterSetDescription trackQualityCuts;
  trackQualityCuts.add<double>("chi2MaxPt", 10.)->setComment("max pT used to determine the pT-dependent chi2 cut (ntuplets)");
  trackQualityCuts.add<std::vector<double>>("chi2Coeff", {0.68177776, 0.74609577, -0.08035491, 0.00315399})
      ->setComment("Polynomial coefficients to derive the pT-dependent chi2 cut");
  trackQualityCuts.add<double>("chi2Scale", 30.)
      ->setComment(
          "Factor to multiply the pT-dependent chi2 cut (currently: 30 for the broken line fit, 45 for the Riemann "
          "fit)");
  trackQualityCuts.add<double>("tripletChi2MaxPt", 10.)->setComment("max pT used to determine the pT-dependent chi2 cut (trips)");
  trackQualityCuts.add<double>("tripletMinPt", 0.5)->setComment("Min pT for triplets, in GeV");
  trackQualityCuts.add<double>("tripletMaxTip", 0.3)->setComment("Max |Tip| for triplets, in cm");
  trackQualityCuts.add<double>("tripletMaxZip", 12.)->setComment("Max |Zip| for triplets, in cm");
  trackQualityCuts.add<double>("quadrupletMinPt", 0.3)->setComment("Min pT for quadruplets, in GeV");
  trackQualityCuts.add<double>("quadrupletMaxTip", 0.5)->setComment("Max |Tip| for quadruplets, in cm");
  trackQualityCuts.add<double>("quadrupletMaxZip", 12.)->setComment("Max |Zip| for quadruplets, in cm");
  trackQualityCuts.add<bool>("upgrade",false);
  desc.add<edm::ParameterSetDescription>("trackQualityCuts", trackQualityCuts)
      ->setComment(
          "Quality cuts based on the results of the track fit:\n  - apply a pT-dependent chi2 cut;\n  - apply \"region "
          "cuts\" based on the fit results (pT, Tip, Zip).");
  // edm::ParameterSetDescription vertexRegion;
  // desc.add<edm::InputTag>("zVertices")->setComment("Z coordinates of vertices for region definition");
  // desc.add<edm::InputTag>("zVerticesError")->setComment("Z coordinate errors of vertices for region definition");
  // desc.add<edm::InputTag>("rVertices")->setComment("R coordinate (is BS) of vertices for region definition");
  // desc.add<edm::InputTag>("rVerticesError")->setComment("R coordinate error (still BS) of vertices for region definition");
  // desc.add<edm::ParameterSetDescription>("vertexRegion", vertexRegion)
  //     ->setComment(
  //         "Vertex region definition parameters.");
}

PixelTrackHeterogeneous CAHitNtupletGeneratorOnGPU::makeTuplesAsync(TrackingRecHit2DCUDA const& hits_d,
                                                                    float bfield,
                                                                    cudaStream_t stream) const {
  PixelTrackHeterogeneous tracks(cms::cuda::make_device_unique<pixelTrack::TrackSoA>(stream));

  auto* soa = tracks.get();

  CAHitNtupletGeneratorKernelsGPU kernels(m_params);
  kernels.counters_ = m_counters;
  HelixFitOnGPU fitter(bfield, m_params.fit5as4_,m_params.isUpgrade_);
 
  kernels.allocateOnGPU(stream);
  fitter.allocateOnGPU(&(soa->hitIndices), kernels.tupleMultiplicity(), soa);
  std::cout << "kernels.allocateOnGPU" << std::endl;  
  kernels.buildDoublets(hits_d, stream);
  std::cout << "kernels.buildDoublets" << std::endl;
  kernels.launchKernels(hits_d, soa, stream);
  std::cout << "kernels.launchKernels" << std::endl;
  kernels.fillHitDetIndices(hits_d.view(), soa, stream);  // in principle needed only if Hits not "available"
  if (m_params.useRiemannFit_) {
    fitter.launchRiemannKernels(hits_d.view(), hits_d.nHits(), CAConstants::maxNumberOfQuadruplets(), stream);
  } else {
    fitter.launchBrokenLineKernels(hits_d.view(), hits_d.nHits(), CAConstants::maxNumberOfQuadruplets(), stream);
  }
  kernels.classifyTuples(hits_d, soa, stream);
  std::cout << "kernels.classifyTuples" << std::endl; 

  return tracks;
}

PixelTrackHeterogeneous CAHitNtupletGeneratorOnGPU::makeTuples(TrackingRecHit2DCPU const& hits_d, VertexRegion const& vtx, float bfield) const {
  PixelTrackHeterogeneous tracks(std::make_unique<pixelTrack::TrackSoA>());

  auto* soa = tracks.get();
  assert(soa);
 
  CAHitNtupletGeneratorKernelsCPU kernels(m_params);
  kernels.counters_ = m_counters;
  kernels.allocateOnGPU(nullptr);

  // kernels.buildDoublets(hits_d, nullptr);
  kernels.buildDoubletsRegional(hits_d, vtx,nullptr);
  kernels.launchKernels(hits_d, soa, nullptr);
  kernels.fillHitDetIndices(hits_d.view(), soa, nullptr);  // in principle needed only if Hits not "available"

  if (0 == hits_d.nHits())
    return tracks;

  // now fit
  HelixFitOnGPU fitter(bfield, m_params.fit5as4_,m_params.isUpgrade_);
  fitter.allocateOnGPU(&(soa->hitIndices), kernels.tupleMultiplicity(), soa);

  if (m_params.useRiemannFit_) {
    fitter.launchRiemannKernelsOnCPU(hits_d.view(), hits_d.nHits(), CAConstants::maxNumberOfQuadruplets());
  } else {
    fitter.launchBrokenLineKernelsOnCPU(hits_d.view(), hits_d.nHits(), CAConstants::maxNumberOfQuadruplets());
  }

  kernels.classifyTuples(hits_d, soa, nullptr);

  // hits_d.view()->release();

  return tracks;
}
