import FWCore.ParameterSet.Config as cms

otStubNtupleAnalyzer = cms.EDAnalyzer(
    "OTStubNtupleAnalyzer",
    recHits=cms.InputTag("pixelSeedingOTRecHitsSoAConverter"),
    stubs=cms.InputTag("otStubProducerVectorHitStyle"),
)
