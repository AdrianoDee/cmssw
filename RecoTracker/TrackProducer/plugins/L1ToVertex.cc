#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/GeometrySurface/interface/Plane.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"

#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "DataFormats/L1TCorrelator/interface/TkPrimaryVertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include <vector>

using namespace reco;
using namespace std;
using namespace edm;

class dso_hidden L1ToVertex : public edm::global::EDProducer<> {
public:
  L1ToVertex(const edm::ParameterSet& cfg);
  ~L1ToVertex() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:

  const edm::EDGetTokenT<std::vector<l1t::TkPrimaryVertex> > theInputCollectionTag_;
  const edm::EDGetTokenT<reco::BeamSpot> tBeamSpot_;

};

L1ToVertex::L1ToVertex(const edm::ParameterSet& cfg)
    :
      theInputCollectionTag_(consumes<std::vector<l1t::TkPrimaryVertex>>(cfg.getParameter<edm::InputTag>("InputCollection"))),
      tBeamSpot_(consumes<reco::BeamSpot>(cfg.getParameter<edm::InputTag>("beamSpot")))
       {
  produces< reco::VertexCollection >();
}

void L1ToVertex::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputCollection", edm::InputTag("L1TkPrimaryVertex"));
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  descriptions.addWithDefaultLabel(desc);
}


void L1ToVertex::produce(edm::StreamID, edm::Event& ev, const edm::EventSetup& es) const {
  std::unique_ptr<reco::VertexCollection> result(new reco::VertexCollection);

  // L1 Vertex Collection
  auto const& vertex = ev.get(theInputCollectionTag_);

  //Bs
  edm::Handle<reco::BeamSpot> bsHandle;
  ev.getByToken(tBeamSpot_, bsHandle);
  const auto &bsh = *bsHandle;

  reco::Vertex::Error errVtx;
    errVtx(0, 0) = bsh.BeamWidthX() * 2.0;
    errVtx(1, 1) = bsh.BeamWidthY() * 2.0;
    errVtx(2, 2) = bsh.sigmaZ() * 2.0;
  reco::Vertex vv(reco::Vertex::Point(bsh.x0(), bsh.y0(), bsh.z0()), errVtx,1.0,1,10);
  result->emplace_back(vv); 
  for (auto const& l1 : vertex) {

    // math::XYZTLorentzVectorD p4;
    //
    // p4.SetPtEtaPhiM(l1.sum(),0.0,0.0,0.13957018);
    double a[9] = {0.1,0.1,0.1,
                  0.1,0.1,0.1,
                  0.1,0.1,0.1};

    reco::Vertex::Error errVtx;
    errVtx(0, 0) = bsh.BeamWidthX() * 2.0;
    errVtx(1, 1) = bsh.BeamWidthY() * 2.0;
    errVtx(2, 2) = bsh.sigmaZ() * 2.0
    ;

    reco::Vertex vv(reco::Vertex::Point(bsh.x0(), bsh.y0(), l1.zvertex()), errVtx,1.0,1,10);

    // std::cout <<  l1.zvertex() << std::endl;
    result->emplace_back(vv);

  }  // end loop over L1Tracks
  // std::cout << " - " << std::endl;
  ev.put(std::move(result));
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1ToVertex);
