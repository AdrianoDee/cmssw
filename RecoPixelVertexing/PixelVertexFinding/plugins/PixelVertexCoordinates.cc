// -*- C++ -*-
//
// Package: RecoPixelVertexing/PixelVertexFinding
// Class: PixelVertexCoordinates
//
/**\class PixelVertexCoordinates PixelVertexCoordinates.cc RecoPixelVertexing/PixelVertexFinding/plugins/PixelVertexCoordinates.cc

Description: [one line class summary]

Implementation:
[Notes on implementation]
*/
//
// Original Author: Riccardo Manzoni
// Created: Tue, 01 Apr 2014 10:11:16 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoPixelVertexing/PixelVertexFinding/interface/PVClusterComparer.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "RecoPixelVertexing/PixelTriplets/plugins/CAHitNtupletGeneratorKernels.h"
#include "CUDADataFormats/Vertex/interface/ZVertexHeterogeneous.h"

class PixelVertexCoordinates : public edm::stream::EDProducer<> {
public:

  // using VertexRegion = cAHitNtupletGenerator::VertexRegion;

  explicit PixelVertexCoordinates(const edm::ParameterSet&);
  ~PixelVertexCoordinates() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  edm::EDGetTokenT<reco::VertexCollection> vtxToken_;
  unsigned int maxVtx_;
};

PixelVertexCoordinates::PixelVertexCoordinates(const edm::ParameterSet& iConfig) {
  edm::InputTag vtxInputTag = iConfig.getParameter<edm::InputTag>("src");
  vtxToken_ = consumes<reco::VertexCollection>(vtxInputTag);

  // maxVtx_ = iConfig.getParameter<unsigned int>("maxVtx");
  produces<VertexRegion>();
}

PixelVertexCoordinates::~PixelVertexCoordinates() {}

// ------------ method called to produce the data ------------
void PixelVertexCoordinates::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  auto VV = std::make_unique<VertexRegion>();

  edm::Handle<reco::VertexCollection> v;
  iEvent.getByToken(vtxToken_, v);

  auto vtxs = *(v.product());

  if (vtxs.size() >= 100) {
       edm::LogWarning("PixelVertexCoordinates") << "You gave in input more than 10 vertices. Only the first 10 will be taken into account." << std::endl;
    }

  int m = std::min(100,int(vtxs.size()));

  // float vZ[10],vZErr[10];
  // float vR,vRErr;

  VV->vtxR = std::sqrt(vtxs[0].x()*vtxs[0].x()+vtxs[0].y()*vtxs[0].y()); //this comes always from beamspot
  VV->vtxRerr = std::sqrt(vtxs[0].xError()*vtxs[0].xError()+vtxs[0].yError()*vtxs[0].yError());
  
  for (int i = 0; i < m; i++) {
    VV->vtxZ[i] = vtxs[i].z();
    VV->vtxZerr[i] = vtxs[i].zError();
  }
  
  for (int i = m; i < 100; i++) {
    VV->vtxZ[i] = 0.0;
    VV->vtxZerr[i] = 100000.;
  }

  iEvent.put(std::move(VV));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module ------------
void PixelVertexCoordinates::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag(""))->setComment("input (pixel) vertex collection");
  // desc.add<unsigned int>("maxVtx", 100)->setComment("max output collection size (number of accepted vertices)");

  descriptions.add("pixelVertexCoordinates", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PixelVertexCoordinates);
