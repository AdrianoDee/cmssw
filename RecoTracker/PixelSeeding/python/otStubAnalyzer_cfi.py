import FWCore.ParameterSet.Config as cms

otStubAnalyzer = cms.EDAnalyzer('OTStubAnalyzer',
    recHits = cms.InputTag("pixelSeedingOTRecHitsSoAConverter"),
    bendStubs = cms.InputTag("otStubProducer"),
    vectorHitStubs = cms.InputTag("otStubProducerVectorHitStyle"),
    maxHitsToPrint = cms.int32(20),
    maxStubsToPrint = cms.int32(10),
    printRecHits = cms.bool(False),
    printBendStubs = cms.bool(True),
    printVectorHitStubs = cms.bool(True)
)
