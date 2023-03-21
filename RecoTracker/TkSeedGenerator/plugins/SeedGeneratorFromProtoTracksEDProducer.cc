#include "SeedGeneratorFromProtoTracksEDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include <vector>

using namespace edm;
using namespace reco;

template <class T>
T sqr(T t) {
  return t * t;
}
typedef SeedingHitSet::ConstRecHitPointer Hit;

struct HitLessByRadius {
  bool operator()(const Hit& h1, const Hit& h2) { return h1->globalPosition().perp2() < h2->globalPosition().perp2(); }
};

void SeedGeneratorFromProtoTracksEDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<InputTag>("InputCollection", InputTag("pixelTracks"));
  desc.add<InputTag>("InputVertexCollection", InputTag(""));
  desc.add<InputTag>("InputVertexCollectionForComplement", InputTag(""));
  desc.add<double>("originHalfLength", 1E9);
  desc.add<double>("originRadius", 1E9);
  desc.add<bool>("useProtoTrackKinematics", false);
  desc.add<bool>("useEventsWithNoVertex", true);
  desc.add<std::string>("TTRHBuilder", "TTRHBuilderWithoutAngle4PixelTriplets");
  desc.add<bool>("usePV", false);
  desc.add<bool>("includeFourthHit", false);
  desc.add<bool>("produceComplement", false);
  desc.add<int>("maxComplementTracks",1E9);
  desc.add<int>("maxComplementTrackVertex",1000);
  desc.add<double>("dzMaxComplementTrackVertex", 1000.);

  edm::ParameterSetDescription psd0;
  psd0.add<std::string>("ComponentName", std::string("SeedFromConsecutiveHitsCreator"));
  psd0.add<std::string>("propagator", std::string("PropagatorWithMaterial"));
  psd0.add<double>("SeedMomentumForBOFF", 5.0);
  psd0.add<double>("OriginTransverseErrorMultiplier", 1.0);
  psd0.add<double>("MinOneOverPtError", 1.0);
  psd0.add<std::string>("magneticField", std::string(""));
  psd0.add<std::string>("TTRHBuilder", std::string("WithTrackAngle"));
  psd0.add<bool>("forceKinematicWithRegionDirection", false);
  desc.add<edm::ParameterSetDescription>("SeedCreatorPSet", psd0);

  descriptions.add("SeedGeneratorFromProtoTracksEDProducer", desc);
}

SeedGeneratorFromProtoTracksEDProducer::SeedGeneratorFromProtoTracksEDProducer(const ParameterSet& cfg)
    : originHalfLength(cfg.getParameter<double>("originHalfLength")),
      originRadius(cfg.getParameter<double>("originRadius")),
      useProtoTrackKinematics(cfg.getParameter<bool>("useProtoTrackKinematics")),
      useEventsWithNoVertex(cfg.getParameter<bool>("useEventsWithNoVertex")),
      usePV_(cfg.getParameter<bool>("usePV")),
      includeFourthHit_(cfg.getParameter<bool>("includeFourthHit")),
      produceComplement_(cfg.getParameter<bool>("produceComplement")),
      maxComplementTracks_(cfg.getParameter<int>("maxComplementTracks")),
      maxComplementTrackVertex_(cfg.getParameter<int>("maxComplementTrackVertex")),
      dzMaxComplementTrackVertex_(cfg.getParameter<double>("dzMaxComplementTrackVertex")),
      theInputCollectionTag(consumes<reco::TrackCollection>(cfg.getParameter<InputTag>("InputCollection"))),
      theInputVertexCollectionTag(
          consumes<reco::VertexCollection>(cfg.getParameter<InputTag>("InputVertexCollection"))),
      theInputVertexCollectionForComplementTag(
          consumes<reco::VertexCollection>(cfg.getParameter<InputTag>("InputVertexCollectionForComplement"))),    
      seedCreator_(cfg.getParameter<edm::ParameterSet>("SeedCreatorPSet"), consumesCollector()),
      config_(consumesCollector()) {
  produces<TrajectorySeedCollection>();
  if (produceComplement_)
  {
    produces<reco::TrackCollection>();
  }
}

void SeedGeneratorFromProtoTracksEDProducer::produce(edm::Event& ev, const edm::EventSetup& es) {
  auto result = std::make_unique<TrajectorySeedCollection>();
  auto leftTracks = std::make_unique<reco::TrackCollection>();
  Handle<reco::TrackCollection> trks;
  ev.getByToken(theInputCollectionTag, trks);

  const TrackCollection& protos = *(trks.product());
  
  edm::Handle<reco::VertexCollection> vertices;
  bool foundVertices = ev.getByToken(theInputVertexCollectionTag, vertices);
  //const reco::VertexCollection & vertices = *(h_vertices.product());

  edm::Handle<reco::VertexCollection> verticesForComplement;
  bool foundVerticesForComplement = ev.getByToken(theInputVertexCollectionForComplementTag, verticesForComplement);

  ///
  /// need optimization: all es stuff should go out of the loop
  ///

  // parameters used for complement tracks
  int complementTracksCount = 0;
  int vertexForComplementCount = 0;
  double dzComplementTrackVertex;
  //int trackVertexForComplementPosition = 0;
  //bool foundTrackVertexForComplement = false;

  for (TrackCollection::const_iterator it = protos.begin(); it != protos.end(); ++it) {
    const Track& proto = (*it);
    GlobalPoint vtx(proto.vertex().x(), proto.vertex().y(), proto.vertex().z());

    // check the compatibility with a primary vertex
    bool keepTrack = false;
    if ((!foundVertices) || vertices->empty()) {
      if (useEventsWithNoVertex)
        keepTrack = true;
    } else if (usePV_) {
      GlobalPoint aPV(
          vertices->begin()->position().x(), vertices->begin()->position().y(), vertices->begin()->position().z());
      double distR2 = sqr(vtx.x() - aPV.x()) + sqr(vtx.y() - aPV.y());
      double distZ = fabs(vtx.z() - aPV.z());
      if (distR2 < sqr(originRadius) && distZ < originHalfLength) {
        keepTrack = true;
      }
    } else {
      for (reco::VertexCollection::const_iterator iv = vertices->begin(); iv != vertices->end(); ++iv) {
        GlobalPoint aPV(iv->position().x(), iv->position().y(), iv->position().z());
        double distR2 = sqr(vtx.x() - aPV.x()) + sqr(vtx.y() - aPV.y());
        double distZ = fabs(vtx.z() - aPV.z());
        if (distR2 < sqr(originRadius) && distZ < originHalfLength) {
          keepTrack = true;
          break;
        }
      }
    }

    
    if(produceComplement_ && !keepTrack && (complementTracksCount < maxComplementTracks_)) {
  
      if(foundVerticesForComplement) { //find if the track is close (by a dz) to first N vertices (dz,N configurable parameters)
        vertexForComplementCount = 0;
        for(reco::VertexCollection::const_iterator iv = verticesForComplement->begin(); iv != verticesForComplement->end(); ++iv) {
          const Vertex& ver = (*iv);
          dzComplementTrackVertex = fabs(vtx.z() - ver.position().z());
          if (dzComplementTrackVertex < dzMaxComplementTrackVertex_) {
            (*leftTracks).push_back(proto);
            complementTracksCount ++;
            break;
          }else if(vertexForComplementCount == maxComplementTrackVertex_){
            break;
          }else{
            vertexForComplementCount++;
          }
        }
      }else{
        (*leftTracks).push_back(proto);
        complementTracksCount ++;
      }

      // The following part of code is an alternative, for doing the association of tracks with vertices based on trackWeight() i.e. vertex fit
      // but using it as it is does not produce any weight > 0. hence all tracks get rejected. (?)
      /*  
      if(foundVerticesForComplement) { //find which is the vertex the track was fitted to
        trackVertexForComplementPosition = 0;
        vertexForComplementCount = 0;
        foundTrackVertexForComplement = false;
        for(reco::VertexCollection::const_iterator iv = verticesForComplement->begin(); iv != verticesForComplement->end(); ++iv) {
          const Vertex& ver = (*iv);
          if (ver.trackWeight(proto.extra()) > 0) {
            trackVertexForComplementPosition = vertexForComplementCount;
            foundTrackVertexForComplement = true;
            break;
          }
          vertexForComplementCount++;
        }

        if( foundTrackVertexForComplement &&  (trackVertexForComplementPosition < maxComplementTrackVertex_)) {
          (*leftTracks).push_back(proto);
          complementTracksCount ++;
        }
      }else{
        (*leftTracks).push_back(proto);
        complementTracksCount ++;
      }
      */
    }
    
    if (!keepTrack)
      continue;

    if (useProtoTrackKinematics) {
      SeedFromProtoTrack seedFromProtoTrack(config_, proto, es);
      if (seedFromProtoTrack.isValid())
        (*result).push_back(seedFromProtoTrack.trajectorySeed());
    } else {
      std::vector<Hit> hits;
      for (unsigned int iHit = 0, nHits = proto.recHitsSize(); iHit < nHits; ++iHit) {
        TrackingRecHitRef refHit = proto.recHit(iHit);
        if (refHit->isValid())
          hits.push_back((Hit) & (*refHit));
      }
      sort(hits.begin(), hits.end(), HitLessByRadius());

      if (hits.size() > 1) {
        double mom_perp =
            sqrt(proto.momentum().x() * proto.momentum().x() + proto.momentum().y() * proto.momentum().y());
        GlobalTrackingRegion region(mom_perp, vtx, 0.2, 0.2);

        seedCreator_.init(region, es, nullptr);
        if (hits.size() > 3 and not includeFourthHit_)
          seedCreator_.makeSeed(*result, {hits[0], hits[1], hits[2]});
        else
          seedCreator_.makeSeed(*result, hits);
      }
    }
  }

  ev.put(std::move(result));
  if(produceComplement_)
    ev.put(std::move(leftTracks));
}
