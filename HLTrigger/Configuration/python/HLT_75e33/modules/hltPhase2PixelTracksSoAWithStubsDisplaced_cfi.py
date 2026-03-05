import FWCore.ParameterSet.Config as cms

from RecoTracker.PixelSeeding.caHitNtupletAlpakaPhase2OTStubsDisplaced_cfi import caHitNtupletAlpakaPhase2OTStubsDisplaced

hltPhase2PixelTracksSoAWithStubsDisplaced = caHitNtupletAlpakaPhase2OTStubsDisplaced.clone(
  pixelRecHitSrc = cms.InputTag('hltPhase2PixelRecHitsStubsMerger'),
  otRecHitsSrc = cms.InputTag('hltPixelSeedingOTRecHitsSoA'),
  stubsSrc = cms.InputTag('hltOTStubProducer'),
  mightGet = cms.optional.untracked.vstring,
  alpaka = cms.untracked.PSet(
    backend = cms.untracked.string('')
  )
)
