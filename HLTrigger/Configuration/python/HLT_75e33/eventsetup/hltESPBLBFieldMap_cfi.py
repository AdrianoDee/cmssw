import FWCore.ParameterSet.Config as cms

# BL-fit (Bz, Br)/Bz(0,0) field map on an r-z lattice, consumed by the merger GBL curvature->pT conversion.
hltESPBLBFieldMap = cms.ESProducer('BLBFieldMapESProducerAlpaka@alpaka',
    appendToDataLabel = cms.string(''),
    alpaka = cms.untracked.PSet(backend = cms.untracked.string(''))
)
