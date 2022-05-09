import FWCore.ParameterSet.Config as cms

## Validation step
tpClusterProducer = cms.EDProducer("ClusterTPAssociationProducer",
    mightGet = cms.optional.untracked.vstring,
    phase2OTClusterSrc = cms.InputTag("siPhase2Clusters"),
    phase2OTSimLinkSrc = cms.InputTag("simSiPixelDigis","Tracker"),
    pixelClusterSrc = cms.InputTag("siPixelClusters"),
    pixelSimLinkSrc = cms.InputTag("simSiPixelDigis"),
    simTrackSrc = cms.InputTag("g4SimHits"),
    stripClusterSrc = cms.InputTag("siStripClusters"),
    stripSimLinkSrc = cms.InputTag("simSiStripDigis"),
    throwOnMissingCollections = cms.bool(True),
    trackingParticleSrc = cms.InputTag("mix","MergedTrackTruth")
)



"""
from SimTracker.TrackerHitAssociation.tpClusterProducerDefault_cfi import tpClusterProducerDefault as _tpClusterProducerDefault

tpClusterProducer = _tpClusterProducerDefault.clone()
"""


quickTrackAssociatorByHits = cms.EDProducer("QuickTrackAssociatorByHitsProducer",
    AbsoluteNumberOfHits = cms.bool(False),
    Cut_RecoToSim = cms.double(0.75),
    PixelHitWeight = cms.double(1.0),
    Purity_SimToReco = cms.double(0.75),
    Quality_SimToReco = cms.double(0.5),
    SimToRecoDenominator = cms.string('reco'),
    ThreeHitTracksAreSpecial = cms.bool(True),
    cluster2TPSrc = cms.InputTag("tpClusterProducer"),
    useClusterTPAssociation = cms.bool(True)
)

trackingParticleRecoTrackAsssociation = cms.EDProducer("TrackAssociatorEDProducer",
    associator = cms.InputTag("quickTrackAssociatorByHits"),
    ignoremissingtrackcollection = cms.untracked.bool(False),
    label_tp = cms.InputTag("mix","MergedTrackTruth"),
    label_tr = cms.InputTag("generalTracks")
)

VertexAssociatorByPositionAndTracks = cms.EDProducer("VertexAssociatorByPositionAndTracksProducer",
    absT = cms.double(-1),
    absZ = cms.double(0.1),
    maxRecoT = cms.double(-1),
    maxRecoZ = cms.double(1000),
    mightGet = cms.optional.untracked.vstring,
    sharedTrackFraction = cms.double(-1),
    sigmaT = cms.double(-1),
    sigmaZ = cms.double(3),
#    sigmaXY = cms.double(100),
    trackAssociation = cms.InputTag("trackingParticleRecoTrackAsssociation")
)



vertexAnalysis = cms.EDProducer("PrimaryVertexAnalyzer4PUSlimmed",
    do_generic_sim_plots = cms.untracked.bool(True),
    root_folder = cms.untracked.string('Vertexing/PrimaryVertexV'),
    trackAssociatorMap = cms.untracked.InputTag("trackingParticleRecoTrackAsssociation"),
    trackingParticleCollection = cms.untracked.InputTag("mix","MergedTrackTruth"),
    trackingVertexCollection = cms.untracked.InputTag("mix","MergedTrackTruth"),
    use_only_charged_tracks = cms.untracked.bool(True),
    verbose = cms.untracked.bool(False),
    vertexAssociator = cms.untracked.InputTag("VertexAssociatorByPositionAndTracks"),
    vertexRecoCollections = cms.VInputTag("")
)


## DQM step
pvMonitor = cms.EDProducer("PrimaryVertexMonitor",
    AlignmentLabel = cms.string('Alignment'),
    DxyBin = cms.int32(100),
    DxyMax = cms.double(5000.0),
    DxyMin = cms.double(-5000.0),
    DzBin = cms.int32(100),
    DzMax = cms.double(2000.0),
    DzMin = cms.double(-2000.0),
    EtaBin = cms.int32(31),
    EtaBin2D = cms.int32(8),
    EtaMax = cms.double(3.0),
    EtaMin = cms.double(-3.0),
    PhiBin = cms.int32(32),
    PhiBin2D = cms.int32(12),
    PhiMax = cms.double(3.141592654),
    PhiMin = cms.double(-3.141592654),
    TkSizeBin = cms.int32(100),
    TkSizeMax = cms.double(499.5),
    TkSizeMin = cms.double(-0.5),
    TopFolderName = cms.string('OfflinePV'),
    Xpos = cms.double(0.0),
    Ypos = cms.double(0.0),
    beamSpotLabel = cms.InputTag("offlineBeamSpot"),
    ndof = cms.int32(4),
    useHPforAlignmentPlots = cms.bool(True),
    vertexLabel = cms.InputTag("")
)
