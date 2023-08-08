// -*- C++ -*-
//
// Package:    RecHits/RecHitAnalyzerTracks
// Class:      RecHitAnalyzerTracks
//
//
// Original Author:  Brunella D'Anzi
//         Created:  Mon, 28 May 2023 17:16:39 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/GluedGeomDet.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "TH2F.h"
#include "TH3F.h"
#include "TGraphErrors.h"
#include "TGraph.h"
#include "TStyle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TPaveStats.h"
//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<>
// This will improve performance in multithreaded jobs.

using reco::TrackCollection;

class RecHitAnalyzerTracks : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit RecHitAnalyzerTracks(const edm::ParameterSet&);
  ~RecHitAnalyzerTracks() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void plotRecHitPosition(const float x, const float y);
  void plotRecHitGlobalPosition(const float xGlobal, const float yGlobal, const float zGlobal);
  void plotRecHitErrors(const float x, const float y, const float xErr, const float yErr);
  void plotDetIdMapping( std::map<uint32_t, int> detIdToSequentialNumber_);
  void plotRecHitPositionGlobalRZ(const float r, const float z);
  void plotProjectionsGlobalPosition();
  void endJob() override;

  // ----------member data ---------------------------
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  edm::EDGetToken pixelTracksToken_;
  std::map<uint32_t, int> detIdToSequentialNumber_;
  int sequentialNumber_;
  unsigned int  maxId;
  unsigned int minId;
  bool isSingleMuon ;
  TH2F* recHitPositionHist_;
  TGraph* detIdMappingHist_;
  TH2F* recHitErrorsHist_;
  TH3F* recHitPositionGlobalHist_;
  TH2F* recHitPositionGlobalRZHist_;
  TGraph* layerIdSequentialNumberHist_;
  TGraph* rSequentialNumberHist_;
  TGraph* zSequentialNumberHist_;
  std::vector<TGraph*> LayersRZ;
  std::vector<int> LayersStart;
  
};


//
// constructors and destructor
//
RecHitAnalyzerTracks::RecHitAnalyzerTracks(const edm::ParameterSet& iConfig)
  :geomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()),
   pixelTracksToken_(consumes<std::vector<reco::Track>>(iConfig.getParameter<edm::InputTag>("pixels"))){
  //now do what ever initialization is needed
  sequentialNumber_ = 0;
  maxId= 0;
  minId= 10000000;
  detIdMappingHist_ = nullptr;
  recHitPositionHist_ = nullptr;
  recHitErrorsHist_ = nullptr;
  recHitPositionGlobalHist_ = nullptr;
  recHitPositionGlobalRZHist_ = nullptr;
  isSingleMuon = true;
  // TIB1 - TIB2 - TIB3 - TIB4 - TID1 and TID2 (positive) for three disks - TID1 and TID2 (negative) for three disks -
  // TID3 (positive) for three disks // 
  LayersStart={0,1856,2528,3392,3932,4580,4868,5156,5396,8204,8924,9716,10604,17004};
  //LayersStart={1856, 2528, 3392, 3932, 4580, 4628, 4676, 4716, 4764, 4812, 4852, 4900, 4948, 4988, 5036, 5084, 5124, 5172, 5220, 5260, 5308, 5356, 5396, 6404, 7556, 8204, 8924, 9716}; // 11 lauers
}

RecHitAnalyzerTracks::~RecHitAnalyzerTracks() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  //
  // please remove this method altogether if it would be left empty
}

//
// member functions
//

// ------------ method called for each event  ------------
void RecHitAnalyzerTracks::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  edm::ESHandle<TrackerGeometry> theTrackerGeometry = iSetup.getHandle(geomToken_);
  edm::Handle<std::vector<reco::Track>> pixelTracks;
  iEvent.getByToken(pixelTracksToken_, pixelTracks);
  // Loop over each individual reco Track
    for (auto const &track : *pixelTracks) {
      for ( auto const &recHit : track.recHits()){
      const GeomDet* thePixelDet = dynamic_cast<const GeomDet*>(theTrackerGeometry->idToDet(recHit->geographicalId()));
      GlobalPoint GP = thePixelDet->surface().toGlobal(recHit->localPosition());
      float xGlobal_pixel = GP.x();                                                                                    
	    float yGlobal_pixel = GP.y();
	    float zGlobal_pixel = GP.z();
      int indexPixel = thePixelDet->index();
      int sequentialId = 0;
      std::cout << "Pixels Global Position: " << xGlobal_pixel << ", " << yGlobal_pixel << ", " << zGlobal_pixel << std::endl;
      float rGlobal_pixel = sqrt(xGlobal_pixel*xGlobal_pixel + yGlobal_pixel*yGlobal_pixel);
	    plotRecHitPositionGlobalRZ(rGlobal_pixel,zGlobal_pixel);
	    plotRecHitGlobalPosition(xGlobal_pixel, yGlobal_pixel, zGlobal_pixel);
      int detId = thePixelDet->geographicalId().rawId();
      if (detIdToSequentialNumber_.find(detId) == detIdToSequentialNumber_.end()) {                                                                                          
	    sequentialId = indexPixel; // Get the current size as the sequential number                                                        
	    detIdToSequentialNumber_[detId] = indexPixel;
      rSequentialNumberHist_->AddPoint(sequentialId,rGlobal_pixel);
      zSequentialNumberHist_->AddPoint(sequentialId,zGlobal_pixel);                                                      
	  }                                                                                        
	  else{
	    sequentialId = indexPixel;
	  }
     for(int i = 0; i < 14; i++){
      if((sequentialId >= LayersStart[i]) && (sequentialId < LayersStart[i+1])){
      LayersRZ[i]->AddPoint(zGlobal_pixel,rGlobal_pixel);
      }
  
   }
      }
     
    }

	  /*Std::cout << "DetId of each RecHits: " << detId << " Sequential Number: " << sequentialId  << std::endl;
	  std::cout << "Local position (x, y): " << x << ", " << y << std::endl;
	  std::cout << "Global position (x, y, z): " << xGlobal << ", " << yGlobal << ", " << zGlobal <<std::endl;
	  std::cout << "Local position errors (x, y): " << xErr << ", " << yErr << std::endl;*/
	  
    
	
      
  }


void RecHitAnalyzerTracks::plotProjectionsGlobalPosition() {
  TCanvas* canvas_PlotGlobalPositionProjections = new TCanvas("PlotGlobalPositionProjections", "Projections", 1200, 400);
  canvas_PlotGlobalPositionProjections->Divide(3, 1); // Divide the canvas into 3 sub-pads
  // Plot the projection on the XY plane
  canvas_PlotGlobalPositionProjections->cd(1);
  TH2D* projXY = dynamic_cast<TH2D*>(recHitPositionGlobalHist_->Project3D("xy"));
  projXY->Draw("COLZ");

  // Plot the projection on the XZ plane
  canvas_PlotGlobalPositionProjections->cd(2);
  TH2D* projXZ = dynamic_cast<TH2D*>(recHitPositionGlobalHist_->Project3D("xz"));
  projXZ->Draw("COLZ");

  // Plot the projection on the YZ plane
  canvas_PlotGlobalPositionProjections->cd(3);
  TH2D* projYZ = dynamic_cast<TH2D*>(recHitPositionGlobalHist_->Project3D("yz"));
  projYZ->Draw("COLZ");

  // Save the canvas as an image file
  canvas_PlotGlobalPositionProjections->SaveAs("projectionsGlobalPosition.png");

  // Clean up memory
  delete canvas_PlotGlobalPositionProjections;
  delete projXY;
  delete projXZ;
  delete projYZ;
}

void RecHitAnalyzerTracks::plotRecHitPositionGlobalRZ(const float r,const float zGlobal) {

  recHitPositionGlobalRZHist_->Fill(zGlobal, r);
}

    
    void RecHitAnalyzerTracks::plotRecHitPosition(const float x, const float y) {
      
      // Plot the recHit position (2D plot)
      recHitPositionHist_->Fill(x, y);
      
    }

void RecHitAnalyzerTracks::plotRecHitGlobalPosition(const float xGlobal, const float yGlobal, const float zGlobal) {
  
  // Plot the recHit position (3D plot)                                                                                                                                              
  recHitPositionGlobalHist_->Fill(xGlobal, yGlobal, zGlobal);

}


void RecHitAnalyzerTracks::plotRecHitErrors(const float x, const float y, const float xErr, const float yErr) {
  
  // Plot the recHit errors (2D plot)
  recHitErrorsHist_->Fill(xErr, yErr);
  
}

void RecHitAnalyzerTracks::plotDetIdMapping(std::map<uint32_t, int> detIdToSequentialNumber_) {
  // Plot the DetId mapping (2D plot) : First find limits of the histo
  for (const auto& pair : detIdToSequentialNumber_) {
    //std::cout << "DetId of each RecHits: " << pair.first << " Sequential Number: " << pair.second  << std::endl;
    detIdMappingHist_->SetPoint(detIdMappingHist_->GetN(),pair.second, pair.first);
  }
}
// ------------ method called once each job just before starting event loop  ------------
void RecHitAnalyzerTracks::beginJob() {
  layerIdSequentialNumberHist_ = new TGraph();
  layerIdSequentialNumberHist_->SetTitle("Layer Id vs Sequential Number");
  layerIdSequentialNumberHist_->GetXaxis()->SetTitle("Sequential Number");
  layerIdSequentialNumberHist_->GetYaxis()->SetTitle("Layer Id");
   for(unsigned int i=0; i < 14; i++){
     LayersRZ.push_back(new TGraph());
     LayersRZ.back()->SetTitle(Form("Layer %d r vs z",i));
     LayersRZ.back()->GetXaxis()->SetTitle("z [cm]");
     LayersRZ.back()->GetYaxis()->SetTitle("r [cm]");
   }
  
   rSequentialNumberHist_ = new TGraph();
   rSequentialNumberHist_->SetTitle("r vs Sequential Number");
   rSequentialNumberHist_->GetXaxis()->SetTitle("Sequential Number");
   rSequentialNumberHist_->GetYaxis()->SetTitle("r [cm]");

   zSequentialNumberHist_ = new TGraph();
   zSequentialNumberHist_->SetTitle("z vs Sequential Number");
   zSequentialNumberHist_->GetXaxis()->SetTitle("Sequential Number");
   zSequentialNumberHist_->GetYaxis()->SetTitle("z [cm]");

  recHitPositionGlobalRZHist_ = new TH2F("recHitPositionGlobalRZHist", "RecHit Position Global in rz", 600, -300., 300.0, 600, 0., 120);
  recHitPositionGlobalRZHist_->GetXaxis()->SetTitle("z [cm]");
  recHitPositionGlobalRZHist_->GetYaxis()->SetTitle("r [cm]");

  recHitErrorsHist_ = new TH2F("recHitErrorsHist", "RecHit Errors", 100, -0.001, 0.2, 500, -0.5, 30.0);
  recHitErrorsHist_->GetXaxis()->SetTitle("x [cm]");
  recHitErrorsHist_->GetYaxis()->SetTitle("y [cm]");
  
  recHitPositionHist_ = new TH2F("recHitPositionHist", "RecHit Position", 100, -5.5, 5.5, 100, -1.0, 1.0);
  recHitPositionHist_->GetXaxis()->SetTitle("x [cm]");
  recHitPositionHist_->GetYaxis()->SetTitle("y [cm]");
  
  recHitPositionGlobalHist_ = new TH3F("recHitPositionGlobalHist", "Global Position Rec Strip Hits", 400, -200.0, 200.0, 400, -200.0, 200.0, 600, -700.0, 700.0);
  recHitPositionGlobalHist_->GetXaxis()->SetTitleOffset(1.5);
  recHitPositionGlobalHist_->GetYaxis()->SetTitleOffset(1.5);
  recHitPositionGlobalHist_->GetZaxis()->SetTitleOffset(1.5);
  recHitPositionGlobalHist_->GetXaxis()->SetTitle("x [cm]");
  recHitPositionGlobalHist_->GetYaxis()->SetTitle("y [cm]");
  recHitPositionGlobalHist_->GetZaxis()->SetTitle("z [cm]");

  
}

// ------------ method called once each job just after ending the event loop  ------------
void RecHitAnalyzerTracks::endJob() {
  
  TCanvas* canvas_recHitPosition = new TCanvas("canvas_recHitPosition", "RecHit Position", 800, 600);
  TCanvas* canvas_detIdMapping = new TCanvas("canvas_detIdMapping", "DetId Mapping", 1000, 600);
  TCanvas* canvas_recHitErrors = new TCanvas("canvas_recHitErrors", "RecHit Errors", 800, 600);
  TCanvas* canvas_recHitPositionGlobalHist = new TCanvas("canvas_recHitPositionGlobalHist", "Global Position Strip Rec Hits", 1000, 600);
  TCanvas* canvas_recHitPositionGlobalRZHist_ = new TCanvas("canvas_recHitPositionGlobalRZHist_"," Global Position in rz Strip Rec hits",1000,600);
  TCanvas* canvas_layerIdSequentialNumberHist_ = new TCanvas("canvas_layerIdSequentialNumberHist_", "Layer vs Sequential Number",1000,600);
  TCanvas* canvas_rSequentialNumberHist_ = new TCanvas("canvas_rSequentialNumberHist_","r vs Sequential Number",1000,600);
  TCanvas* canvas_zSequentialNumberHist_ = new TCanvas("canvas_zSequentialNumberHist_","z vs Sequential Number",1000,600);
  std::vector<TCanvas*> canvas_LayersRZ;
   for(unsigned int i=0; i < 14; i++){
     canvas_LayersRZ.push_back(new TCanvas(Form("canvas_LayersRZ_%d",i),Form("Layer %d r vs z",i),1000,600));
     canvas_LayersRZ.back()->cd();
     LayersRZ[i]->SetMarkerStyle(5);
     LayersRZ[i]->SetMarkerSize(0.3);
     LayersRZ[i]->SetMarkerColor(kBlue);
   if(LayersRZ[i]->GetN()){
     LayersRZ[i]->Draw("AP");}
     canvas_LayersRZ.back()->SaveAs(Form("Layer%d_rZ.png",i));
   }
  
  // canvas_layerIdSequentialNumberHist_->cd();
  // layerIdSequentialNumberHist_->SetMarkerStyle(5);
  // layerIdSequentialNumberHist_->SetMarkerSize(0.3);
  // layerIdSequentialNumberHist_->SetMarkerColor(kRed);
  // layerIdSequentialNumberHist_->Draw("AP");
  // canvas_layerIdSequentialNumberHist_->SaveAs("LayerIdSequentialNumber.png");
  // canvas_layerIdSequentialNumberHist_->SaveAs("LayerIdSequentialNumber.C"); 

  canvas_rSequentialNumberHist_->cd();
  rSequentialNumberHist_->SetMarkerStyle(5);
  rSequentialNumberHist_->SetMarkerSize(0.3);
  rSequentialNumberHist_->SetMarkerColor(kRed);
  rSequentialNumberHist_->Draw("AP");
  canvas_rSequentialNumberHist_->SaveAs("rSequentialNumber.png");
  canvas_rSequentialNumberHist_->SaveAs("rSequentialNumber.C");


  canvas_zSequentialNumberHist_->cd();
  zSequentialNumberHist_->SetMarkerStyle(5);
  zSequentialNumberHist_->SetMarkerSize(0.3);
  zSequentialNumberHist_->SetMarkerColor(kRed);
  zSequentialNumberHist_->Draw("AP");
  canvas_zSequentialNumberHist_->SaveAs("zSequentialNumber.png");
  canvas_zSequentialNumberHist_->SaveAs("zSequentialNumber.C");

  // for (const auto& pair : detIdToSequentialNumber_) {
  //   //std::cout << "DetId of each RecHits: " << pair.first << " Sequential Number: " << pair.second  << " Last Sequential Number: " << sequentialNumber_ <<std::endl;                                      
  //   if(pair.first > maxId) maxId = pair.first;
  //   if(pair.first < minId) minId = pair.first;
  // }
  // detIdMappingHist_ = new TGraph();
  // detIdMappingHist_->SetTitle("Det Id Mapping");
  // detIdMappingHist_->GetXaxis()->SetTitle("Sequential Number");
  // detIdMappingHist_->GetYaxis()->SetTitle("DetId");
  //detIdMappingHist_->SetMarkerStyle(20); 
  //detIdMappingHist_->SetMarkerColor(kBlack);

 

  // plotDetIdMapping(detIdToSequentialNumber_);
  plotProjectionsGlobalPosition();
  
  // canvas_recHitPosition->cd();
  // // Draw the histogram on the canvas                                                                                                                                          
  // recHitPositionHist_->Draw("hist");
  // // Save the canvas as a PNG image                                                                                                                                        
  // canvas_recHitPosition->SaveAs("recHitPosition.png");                                                                                                    delete canvas_recHitPosition;
  
  canvas_recHitPositionGlobalRZHist_-> cd();
  recHitPositionGlobalRZHist_->Draw();
  canvas_recHitPositionGlobalRZHist_->SaveAs("recHitPositionGlobalRZ.png");
  delete canvas_recHitPositionGlobalRZHist_;

  // Draw the histogram on the canvas                                                    
  canvas_recHitPositionGlobalHist->cd();                                                                                           
  recHitPositionGlobalHist_->Draw();
  canvas_recHitPositionGlobalHist->SaveAs("recHitGlobalPosition.png");
  delete canvas_recHitPositionGlobalHist;
  
  // canvas_detIdMapping->cd();
  // detIdMappingHist_->SetMarkerStyle(5);
  // detIdMappingHist_->SetMarkerSize(0.3);
  // detIdMappingHist_->SetMarkerColor(kRed);
  // detIdMappingHist_->Draw("AP");
  // // Save the canvas as a PNG image                                                                                                                                              
  // canvas_detIdMapping->SaveAs("detIdMapping.png");
  // canvas_detIdMapping->SaveAs("detIdMapping.C");
  // delete canvas_detIdMapping;                                                                                                                                                                 
  // Save the canvas as a PNG image                                                                                                                                                                 
  // canvas_recHitErrors->cd();
  // recHitErrorsHist_->Draw("hist");
  // canvas_recHitErrors->SaveAs("recHitErrors.png");
  // delete canvas_recHitErrors;
  // Save the canvas as a PNG image                                                                                                                                                                 
  
  TFile output("RecHitAnalyzerTracks.root", "RECREATE");
  recHitPositionHist_->Write();
  recHitPositionGlobalHist_->Write();
  // recHitErrorsHist_->Write();
  //detIdMappingHist_->Write();
  output.Close();
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void RecHitAnalyzerTracks::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(RecHitAnalyzerTracks);
