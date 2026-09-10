import FWCore.ParameterSet.Config as cms

hltESPPixelCPEFastParamsPhase2 = cms.ESProducer('PixelCPEFastParamsESProducerAlpakaPhase2@alpaka',
    ComponentName = cms.string("PixelCPEFastParamsPhase2"),
    appendToDataLabel = cms.string(''),
    alpaka = cms.untracked.PSet(backend = cms.untracked.string(''))
)

# Stub chain uses per-module genError parameters instead of the cluster-size table.
hltESPPixelCPEFastParamsPhase2OTStubs = cms.ESProducer('PixelCPEFastParamsESProducerAlpakaPhase2OTStubs@alpaka',
    ComponentName = cms.string("PixelCPEFastParamsPhase2OTStubs"),
    appendToDataLabel = cms.string(''),
    alpaka = cms.untracked.PSet(backend = cms.untracked.string(''))
)
