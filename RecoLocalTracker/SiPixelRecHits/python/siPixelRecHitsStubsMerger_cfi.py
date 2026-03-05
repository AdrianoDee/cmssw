import FWCore.ParameterSet.Config as cms

siPixelRecHitsStubsMerger = cms.EDProducer('SiPixelRecHitsStubsMerger@alpaka',
    # Input: pixel RecHits SoA collection
    pixelRecHitsSoA = cms.InputTag('siPixelRecHitsPreSplittingAlpaka'),
    # Input: OT stubs SoA collection
    stubsSoA = cms.InputTag('otStubProducer'),
    # Alpaka backend configuration
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')  # Auto-select backend
    )
)
