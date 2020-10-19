//
// // system include files
// #include <memory>
//
// // user include files
// #include "FWCore/Framework/interface/Frameworkfwd.h"
// #include "FWCore/Framework/interface/stream/EDProducer.h"
//
// #include "FWCore/Framework/interface/Event.h"
// #include "FWCore/Framework/interface/MakerMacros.h"
//
// #include "FWCore/ParameterSet/interface/ParameterSet.h"
//
// #include "DataFormats/VertexReco/interface/Vertex.h"
// #include "DataFormats/VertexReco/interface/VertexFwd.h"
// #include "FWCore/Utilities/interface/InputTag.h"
// #include "RecoPixelVertexing/PixelVertexFinding/interface/PVClusterComparer.h"
//
// #include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
// #include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
//
// #include "RecoPixelVertexing/PixelTriplets/plugins/CAHitNtupletGeneratorKernels.h"
//
// namespace cAHitNtupletGenerator
// {
// struct VertexRegion {
// //
//   float vtxZ[10];
//   float vtxR;
//   float vtxZerr[10];
//   float vtxRerr;
//
// };
// }
//
// class PixelVertexCoordinates : public edm::stream::EDProducer<> {
// public:
//
//   explicit PixelVertexCoordinates(const edm::ParameterSet&);
//   ~PixelVertexCoordinates() override;
//
//   static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
//
// private:
//   void produce(edm::Event&, const edm::EventSetup&) override;
//
//   edm::EDGetTokenT<reco::VertexCollection> vtxToken_;
//   unsigned int maxVtx_;
// };
//
// // //define this as a plug-in
// // DEFINE_FWK_MODULE(PixelVertexCoordinates);
