#include "SeedGeneratorFromProtoTracksEDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "RecoTracker/TkSeedGenerator/interface/SeedFromProtoTrack.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "SeedFromConsecutiveHitsCreator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
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
  desc.add<double>("originHalfLength", 1E9);
  desc.add<double>("originRadius", 1E9);
  desc.add<bool>("useProtoTrackKinematics", false);
  desc.add<bool>("useEventsWithNoVertex", true);
  desc.add<std::string>("TTRHBuilder", "TTRHBuilderWithoutAngle4PixelTriplets");
  desc.add<bool>("usePV", false);
  desc.add<bool>("useVertices", true);
  desc.add<bool>("includeFourthHit", false);
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<bool>("beamSpotPOCA", false);
  desc.add<bool>("fallBackPOCA", false);

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
    : theConfig(cfg),
      originHalfLength(cfg.getParameter<double>("originHalfLength")),
      originRadius(cfg.getParameter<double>("originRadius")),
      useProtoTrackKinematics(cfg.getParameter<bool>("useProtoTrackKinematics")),
      builderName(cfg.getParameter<std::string>("TTRHBuilder")),
      usePV_(cfg.getParameter<bool>("usePV")),
      useVertices_(cfg.getParameter<bool>("useVertices")),
      includeFourthHit_(cfg.getParameter<bool>("includeFourthHit")),
      beamSpotPOCA_(cfg.getParameter<bool>("beamSpotPOCA")),
      fallBackPOCA_(cfg.getParameter<bool>("fallBackPOCA")),
      theInputCollectionTag(consumes<reco::TrackCollection>(cfg.getParameter<InputTag>("InputCollection"))),
      theInputVertexCollectionTag(consumes<reco::VertexCollection>(cfg.getParameter<InputTag>("InputVertexCollection"))),
      theBeamSpotTag(consumes<reco::BeamSpot>(cfg.getParameter<edm::InputTag>("beamSpot")))
      {
  produces<TrajectorySeedCollection>();
}

void SeedGeneratorFromProtoTracksEDProducer::produce(edm::Event& ev, const edm::EventSetup& es) {
  auto result = std::make_unique<TrajectorySeedCollection>();
  Handle<reco::TrackCollection> trks;
  ev.getByToken(theInputCollectionTag, trks);

  const TrackCollection& protos = *(trks.product());

  edm::Handle<reco::BeamSpot> bs;
  bool foundBS = ev.getByToken(theBeamSpotTag,bs);

  edm::Handle<reco::VertexCollection> vertices;
  bool foundVertices = ev.getByToken(theInputVertexCollectionTag, vertices);
  //const reco::VertexCollection & vertices = *(h_vertices.product());

  ///
  /// need optimization: all es stuff should go out of the loop
  ///
  for (TrackCollection::const_iterator it = protos.begin(); it != protos.end(); ++it) {
    const Track& proto = (*it);
    GlobalPoint vtx(proto.vertex().x(), proto.vertex().y(), proto.vertex().z());

    Point pocaPoint(0.0,0.0,0.0);
    if(beamSpotPOCA_ && foundBS)
    {
      pocaPoint = Point(bs->position().x(),bs->position().y(),bs->position().z());
      vtx = GlobalPoint(vtx.x()-bs->position().x(),vtx.y()-bs->position().y(),vtx.z()-bs->position().z());
    }

    // check the compatibility with a primary vertex
    bool keepTrack = false;

    if(useVertices_)
    {
      if (((!foundVertices) || vertices->empty()))
      {
          if (fallBackPOCA_ && (std::abs(proto.dz(pocaPoint)) < originHalfLength) && (std::abs(proto.dxy(pocaPoint)) < originRadius))
            keepTrack = true;
      }
      else
      {
        auto lastVertex = (usePV_) ? 1 : vertices->size();
        for (unsigned int i = 0; i < lastVertex; i++)
        {
            if ((std::abs(proto.dz((*vertices)[i].position())) < originHalfLength) && (std::abs(proto.dxy((*vertices)[i].position())) < originRadius))
            {
              keepTrack = true;
              break;
            }
        }

      }
    }
    else if(fallBackPOCA_)
    {
      if ((std::abs(proto.dz(pocaPoint)) < originHalfLength) && (std::abs(proto.dxy(pocaPoint)) < originRadius))
        keepTrack = true;
    }
    else
    {
      keepTrack = true;
    }

    if (!keepTrack)
      continue;

    if (useProtoTrackKinematics) {
      SeedFromProtoTrack seedFromProtoTrack(proto, es);
      if (seedFromProtoTrack.isValid())
        (*result).push_back(seedFromProtoTrack.trajectorySeed());
    } else {
      edm::ESHandle<TransientTrackingRecHitBuilder> ttrhbESH;
      es.get<TransientRecHitRecord>().get(builderName, ttrhbESH);
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

        edm::ParameterSet seedCreatorPSet = theConfig.getParameter<edm::ParameterSet>("SeedCreatorPSet");
        SeedFromConsecutiveHitsCreator seedCreator(seedCreatorPSet);
        seedCreator.init(region, es, nullptr);

        if(hits.size() <5)
        {
          seedCreator.makeSeed(
              *result,
              SeedingHitSet(hits[0],
                            hits[1],
                            hits.size() > 2 ? hits[2] : SeedingHitSet::nullPtr(),
                            (includeFourthHit_ && hits.size() > 3) ? hits[3] : SeedingHitSet::nullPtr()));
         }else
         {
           seedCreator.makeSeed(*result,hits);
         }
      }
    }
  }

  ev.put(std::move(result));
}
