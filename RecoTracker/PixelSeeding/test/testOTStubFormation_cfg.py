import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9

process = cms.Process("TESTSTUBS", Phase2C17I13M9)

# Message logger
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.MessageLogger.cerr.threshold = cms.untracked.string("INFO")
process.MessageLogger.cerr.OTStubAnalyzer = cms.untracked.PSet(
    limit=cms.untracked.int32(-1)
)
process.MessageLogger.cerr.PixelSeedingOTRecHitsSoAConverter = cms.untracked.PSet(
    limit=cms.untracked.int32(-1)  # Unlimited messages
)
process.MessageLogger.cerr.OTStubProducer = cms.untracked.PSet(
    limit=cms.untracked.int32(-1)  # Unlimited messages
)
process.MessageLogger.cerr.OTStubProducerVectorHitStyle = cms.untracked.PSet(
    limit=cms.untracked.int32(-1)
)
process.MessageLogger.cerr.VectorHitBuilder = cms.untracked.PSet(
    limit=cms.untracked.int32(-1)
)
process.MessageLogger.cerr.VectorHitBuilderEDProducer = cms.untracked.PSet(
    limit=cms.untracked.int32(-1)
)

# Load Alpaka accelerators (required for @alpaka module resolution)
process.load("Configuration.StandardSequences.Accelerators_cff")

# Load Phase-2 Run4 geometry
process.load("Configuration.Geometry.GeometryExtendedRun4D110Reco_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

from Configuration.AlCa.GlobalTag import GlobalTag

# Use local GT database
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
# Fallback to auto GT if local database not available
# process.GlobalTag = GlobalTag(process.GlobalTag, "auto:phase2_realistic", "")

# Load the StackedModuleGeometry ESProducer
process.load("RecoTracker.PixelSeeding.stackedModuleGeometryESProducer_cfi")
process.stackedModuleGeometryESProducer.minPt = cms.double(2.0)  # GeV
process.stackedModuleGeometryESProducer.magneticField = cms.double(3.8)  # Tesla

# Input source - Phase2 tracker hits
process.source = cms.Source(
    "PoolSource",
    fileNames=cms.untracked.vstring(
        # Replace with your input file containing Phase2TrackerRecHit1D collections
        #        "file:/data/rovere/CMSSW_developments/CMSSW_16_0_0_pre4/src/29752.75_SingleMuPt15Eta0p_0p4+Run4D110_HLT75e33Timing/step2_29752.75_SingleMuPt15Eta0p_0p4.root"
        "file:/data/rovere/CMSSW_developments/CMSSW_16_0_0_pre4/src/step2_SingleMuPt15Eta1p0_2p0.root"
    ),
)

process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(10))

# Create OT RecHits
process.load("HLTrigger.Configuration.HLT_75e33.modules.hltSiPhase2Clusters_cfi")
process.load("RecoLocalTracker.Phase2TrackerRecHits.Phase2StripCPEESProducer_cfi")
process.load("RecoLocalTracker.Phase2TrackerRecHits.Phase2TrackerRecHits_cfi")
process.siPhase2RecHits.src = "hltSiPhase2Clusters"
# Load the OT RecHits SoA converter
process.load("RecoTracker.PixelSeeding.PixelSeedingOTRecHitsSoAConverter_cfi")

# Load both stub producers
process.load("RecoTracker.PixelSeeding.otStubProducer_cfi")
process.load("RecoTracker.PixelSeeding.otStubProducerVectorHitStyle_cfi")

# Customize stub producer parameters if needed
# process.otStubProducer.stubModuleOffset = cms.int32(1856)
# process.otStubProducerVectorHitStyle.stubModuleOffset = cms.int32(1856)
# process.otStubProducerVectorHitStyle.barrelCut = cms.vdouble(0.0, 0.05, 0.06, 0.08, 0.09, 0.12, 0.2)
# process.otStubProducerVectorHitStyle.endcapCut = cms.vdouble(0.0, 0.1, 0.1, 0.1, 0.1, 0.1)

# Load the analyzer
process.load("RecoTracker.PixelSeeding.otStubAnalyzer_cfi")
process.otStubAnalyzer.maxHitsToPrint = cms.int32(50)  # Print first 50 hits
process.otStubAnalyzer.maxStubsToPrint = cms.int32(20)  # Print first 20 stubs
process.otStubAnalyzer.printRecHits = cms.bool(True)  # Enable RecHits printing
process.otStubAnalyzer.printBendStubs = cms.bool(True)
process.otStubAnalyzer.printVectorHitStubs = cms.bool(True)


# import VectorHitBuilder
process.load("RecoLocalTracker.SiPhase2VectorHitBuilder.siPhase2VectorHits_cfi")
process.siPhase2VectorHits.Clusters = "hltSiPhase2Clusters"
process.load("RecoLocalTracker.SiPhase2VectorHitBuilder.siPhase2RecHitMatcher_cfi")
process.siPhase2RecHitMatcher.Clusters = "hltSiPhase2Clusters"

# TFileService for ntuple output
process.TFileService = cms.Service(
    "TFileService",
    fileName=cms.string("stubs.root"),
)

# Load the ntuple analyzer
process.load("RecoTracker.PixelSeeding.otStubNtupleAnalyzer_cfi")

# Define processing path
process.p = cms.Path(
    process.hltSiPhase2Clusters
    + process.siPhase2RecHits
    + process.pixelSeedingOTRecHitsSoAConverter
    + process.otStubProducer
    + process.otStubProducerVectorHitStyle
    + process.otStubAnalyzer
    + process.otStubNtupleAnalyzer
    + process.siPhase2VectorHits
)

# Print configuration
print("\n" + "=" * 80)
print("  OT Stub Formation Test Configuration")
print("=" * 80)
print(f"  Geometry: Run4 D110")
print(f"  Global Tag: {process.GlobalTag.globaltag._value}")
print(f"  B field: {process.stackedModuleGeometryESProducer.magneticField._value} T")
print(f"  Min pT: {process.stackedModuleGeometryESProducer.minPt._value} GeV")
print(
    f"  Input RecHits: {process.pixelSeedingOTRecHitsSoAConverter.otRecHitSource.value()}"
)
# print(f"  Stub module offset: {process.otStubProducer.stubModuleOffset._value}")  # Removed - no longer used
print(f"  Max stubs to print: {process.otStubAnalyzer.maxStubsToPrint._value}")
print(f"  Ntuple output: {process.TFileService.fileName._value}")
print("=" * 80)
print("\nProcessing path:")
print("  1. pixelSeedingOTRecHitsSoAConverter - Convert DetSetVector to SoA")
print("  2. OTStubProducer - Bend-based stub formation (GPU)")
print("  3. OTStubProducerVectorHitStyle - VectorHits-style stub formation (GPU)")
print("  4. OTStubAnalyzer - Print stub properties")
print("  5. OTStubNtupleAnalyzer - Write flat ROOT ntuple for analysis")
print("  6. VectorHitBuilderEDProducer - Create VectorHits from RecHits (CPU)")
print("=" * 80 + "\n")
