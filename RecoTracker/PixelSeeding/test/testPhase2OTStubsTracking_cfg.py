import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
from Configuration.ProcessModifiers.phase2CAStubs_cff import phase2CAStubs

process = cms.Process('RECO', Phase2C17I13M9, phase2CAStubs)

# Load standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.Geometry.GeometryExtendedRun4D110Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Accelerators_cff')

# Global Tag - use local GT database
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = cms.ESSource(
    "PoolDBESSource",
    DBParameters=cms.PSet(
        authenticationPath=cms.untracked.string(""),
        authenticationSystem=cms.untracked.int32(0),
        connectionTimeout=cms.untracked.int32(0),
        messageLevel=cms.untracked.int32(0),
        security=cms.untracked.string(""),
    ),
    DumpStat=cms.untracked.bool(False),
    JsonDumpFileName=cms.untracked.string(""),
    ReconnectEachRun=cms.untracked.bool(False),
    RefreshAlways=cms.untracked.bool(False),
    RefreshEachRun=cms.untracked.bool(False),
    RefreshOpenIOVs=cms.untracked.bool(False),
    appendToDataLabel=cms.string(""),
    connect=cms.string(
        "sqlite_file:/data/rovere/CMSSW_developments/CMSSW_16_0_0_pre4/src/150X_mcRun4_realistic_v1.db"
    ),
    frontierKey=cms.untracked.string(""),
    globaltag=cms.string("150X_mcRun4_realistic_v1"),
    pfnPostfix=cms.untracked.string(""),
    pfnPrefix=cms.untracked.string(""),
    recordsToDebug=cms.untracked.vstring(),
    snapshotTime=cms.string(""),
    toGet=cms.VPSet(),
)

# Load the StackedModuleGeometry ESProducer
process.load("RecoTracker.PixelSeeding.stackedModuleGeometryESProducer_cfi")
process.stackedModuleGeometryESProducer.minPt = cms.double(2.0)  # GeV
process.stackedModuleGeometryESProducer.magneticField = cms.double(3.8)  # Tesla

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:/data/rovere/CMSSW_developments/CMSSW_16_0_0_pre4/src/29752.75_SingleMuPt15Eta0p_0p4+Run4D110_HLT75e33Timing/step2_29752.75_SingleMuPt15Eta0p_0p4.root'
    ),
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Message Logger
process.MessageLogger.cerr.FwkReport.reportEvery = 1

# Load EventSetup modules required by HLT
from HLTrigger.Configuration.HLT_75e33.eventsetup.hltESPPixelCPEFastParams_cfi import hltESPPixelCPEFastParamsPhase2
from RecoLocalTracker.Phase2TrackerRecHits.Phase2StripCPEESProducer_cfi import phase2StripCPEESProducer

# Load HLT modules for stub-based tracking (configured by phase2CAStubs modifier)
from HLTrigger.Configuration.HLT_75e33.sequences.HLTBeginSequence_cfi import HLTBeginSequence
from HLTrigger.Configuration.HLT_75e33.modules.hltOnlineBeamSpot_cfi import hltOnlineBeamSpot
from HLTrigger.Configuration.HLT_75e33.modules.hltPhase2OnlineBeamSpotDevice_cfi import hltPhase2OnlineBeamSpotDevice
from HLTrigger.Configuration.HLT_75e33.modules.hltPhase2SiPixelClustersSoA_cfi import hltPhase2SiPixelClustersSoA
from HLTrigger.Configuration.HLT_75e33.modules.hltPhase2SiPixelRecHitsSoA_cfi import hltPhase2SiPixelRecHitsSoA
from HLTrigger.Configuration.HLT_75e33.modules.hltSiPhase2Clusters_cfi import hltSiPhase2Clusters
from HLTrigger.Configuration.HLT_75e33.modules.hltSiPhase2RecHits_cfi import hltSiPhase2RecHits
from HLTrigger.Configuration.HLT_75e33.modules.hltPixelSeedingOTRecHitsSoA_cfi import hltPixelSeedingOTRecHitsSoA
from HLTrigger.Configuration.HLT_75e33.modules.hltOTStubProducer_cfi import hltOTStubProducer
from HLTrigger.Configuration.HLT_75e33.modules.hltPhase2PixelRecHitsStubsMerger_cfi import hltPhase2PixelRecHitsStubsMerger
from HLTrigger.Configuration.HLT_75e33.modules.hltPhase2PixelTracksSoAWithStubs_cfi import hltPhase2PixelTracksSoAWithStubs

# Load legacy pixel clusters (required for legacy RecHits)
from HLTrigger.Configuration.HLT_75e33.modules.hltSiPixelClusters_cfi import hltSiPixelClusters

# Load legacy pixel RecHits (required for legacy converter)
# Note: We override hltSiPixelRecHits to use our freshly produced clusters
# (input file has hltSiPixelClusters from HLT process with invalid originalId)
hltSiPixelRecHits = cms.EDProducer('SiPixelRecHitFromSoAAlpaka',
    pixelRecHitSrc = cms.InputTag('hltPhase2SiPixelRecHitsSoA'),
    src = cms.InputTag('hltSiPixelClusters', '', 'RECO'),  # Explicitly use RECO process
)

# Legacy track converter for stubs - converts SoA tracks to legacy reco::Track
hltPhase2PixelTracksWithStubs = cms.EDProducer("PixelTrackProducerFromSoAAlpaka",
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    trackSrc = cms.InputTag("hltPhase2PixelTracksSoAWithStubs"),
    pixelRecHitLegacySrc = cms.InputTag("hltSiPixelRecHits"),
    outerTrackerRecHitSrc = cms.InputTag("hltSiPhase2RecHits"),
    outerTrackerRecHitSoAConverterSrc = cms.InputTag("hltPixelSeedingOTRecHitsSoA"),
    otRecHitsSoASrc = cms.InputTag("hltPixelSeedingOTRecHitsSoA"),
    stubsSoASrc = cms.InputTag("hltOTStubProducer"),
    minNumberOfHits = cms.int32(0),
    minQuality = cms.string('loose'),
    useOTExtension = cms.bool(True),
    expandStubs = cms.bool(True),
    requireQuadsFromConsecutiveLayers = cms.bool(True)
)

# Use the EventSetup modules
process.hltESPPixelCPEFastParamsPhase2 = hltESPPixelCPEFastParamsPhase2
process.phase2StripCPEESProducer = phase2StripCPEESProducer

# Use the HLT beamspot modules
process.hltOnlineBeamSpot = hltOnlineBeamSpot
process.hltPhase2OnlineBeamSpotDevice = hltPhase2OnlineBeamSpotDevice

# Use the HLT modules
process.hltPhase2SiPixelClustersSoA = hltPhase2SiPixelClustersSoA
process.hltPhase2SiPixelRecHitsSoA = hltPhase2SiPixelRecHitsSoA
process.hltSiPhase2Clusters = hltSiPhase2Clusters
process.hltSiPhase2RecHits = hltSiPhase2RecHits
process.hltPixelSeedingOTRecHitsSoA = hltPixelSeedingOTRecHitsSoA
process.hltOTStubProducer = hltOTStubProducer
process.hltPhase2PixelRecHitsStubsMerger = hltPhase2PixelRecHitsStubsMerger
process.hltPhase2PixelTracksSoAWithStubs = hltPhase2PixelTracksSoAWithStubs

# Legacy pixel clusters and RecHits, plus track converter
process.hltSiPixelClusters = hltSiPixelClusters
process.hltSiPixelRecHits = hltSiPixelRecHits
process.hltPhase2PixelTracksWithStubs = hltPhase2PixelTracksWithStubs

# Path - using HLT modules configured for stub-based tracking
process.stub_tracking_path = cms.Path(
    process.hltOnlineBeamSpot +
    process.hltPhase2OnlineBeamSpotDevice +
    process.hltPhase2SiPixelClustersSoA +
    process.hltPhase2SiPixelRecHitsSoA +
    process.hltSiPixelClusters +  # Legacy pixel clusters for RecHits
    process.hltSiPixelRecHits +  # Legacy pixel RecHits for converter
    process.hltSiPhase2Clusters +
    process.hltSiPhase2RecHits +
    process.hltPixelSeedingOTRecHitsSoA +
    process.hltOTStubProducer +
    process.hltPhase2PixelRecHitsStubsMerger +
    process.hltPhase2PixelTracksSoAWithStubs +
    process.hltPhase2PixelTracksWithStubs  # Legacy track converter
)

# Output
process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('output_stub_tracking.root'),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_hltPhase2PixelTracksSoAWithStubs_*_*',
        'keep *_hltPhase2PixelTracksWithStubs_*_*',  # Legacy reco::Track
        'keep *_hltOTStubProducer_*_*',
        'keep *_hltPixelSeedingOTRecHitsSoA_*_*',
    ),
)

process.output_step = cms.EndPath(process.output)

# Schedule
process.schedule = cms.Schedule(
    process.stub_tracking_path,
    process.output_step
)

# Print configuration summary
print("\n" + "=" * 80)
print("  Phase-2 OT Stubs Tracking Test Configuration")
print("=" * 80)
print(f"  Geometry: Run4 D110")
print(f"  Global Tag: 150X_mcRun4_realistic_v1")
print(f"  Process Modifier: phase2CAStubs")
print(f"  Events: 10")
print("=" * 80)
print("\nUsing HLT modules from HLTrigger/Configuration/python/HLT_75e33/")
print("  - Stub-based tracking configured by phase2CAStubs modifier")
print("=" * 80 + "\n")
