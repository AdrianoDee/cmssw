import FWCore.ParameterSet.Config as cms

hltPixelSeedingOTRecHitsSoA = cms.EDProducer('PixelSeedingOTRecHitsSoAConverter@alpaka',
  otRecHitSource = cms.InputTag('hltSiPhase2RecHits'),
  beamSpot = cms.InputTag('hltOnlineBeamSpot'),
  mightGet = cms.optional.untracked.vstring,
  alpaka = cms.untracked.PSet(
    backend = cms.untracked.string('')
  )
)
