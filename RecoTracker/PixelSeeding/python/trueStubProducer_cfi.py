import FWCore.ParameterSet.Config as cms

trueStubProducer = cms.EDProducer('TrueStubProducer',
    recHitSrc = cms.InputTag('siPhase2RecHits'),
    trackingParticleSrc = cms.InputTag('mix', 'MergedTrackTruth'),
    otRecHitsSrc = cms.InputTag('pixelSeedingOTRecHitsSoA'),

    # TP selection cuts
    TP_minPt = cms.double(0.75),
    TP_maxEta = cms.double(2.6),
    TP_maxVtxZ = cms.double(30.0),
    TP_maxD0 = cms.double(60.0),
    TP_maxLxy = cms.double(60.0),
    TP_minNLayers = cms.int32(4),
    tpAwarePHitOnly = cms.bool(True),

    # TrackerHitAssociator configuration for Phase-2 truth matching
    hitAssociatorConfig = cms.PSet(
        associatePixel = cms.bool(True),
        associateStrip = cms.bool(True),
        usePhase2Tracker = cms.bool(True),
        associateRecoTracks = cms.bool(False),
        associateHitbySimTrack = cms.bool(False),
        phase2TrackerSimLinkSrc = cms.InputTag('simSiPixelDigis', 'Tracker'),
        pixelSimLinkSrc = cms.InputTag('simSiPixelDigis', 'Pixel'),
        stripSimLinkSrc = cms.InputTag('simSiStripDigis'),
        ROUList = cms.vstring(
            'TrackerHitsPixelBarrelLowTof',
            'TrackerHitsPixelBarrelHighTof',
            'TrackerHitsPixelEndcapLowTof',
            'TrackerHitsPixelEndcapHighTof',
            'TrackerHitsTIBLowTof',
            'TrackerHitsTIBHighTof',
            'TrackerHitsTIDLowTof',
            'TrackerHitsTIDHighTof',
            'TrackerHitsTOBLowTof',
            'TrackerHitsTOBHighTof',
            'TrackerHitsTECLowTof',
            'TrackerHitsTECHighTof',
        ),
    ),
)
