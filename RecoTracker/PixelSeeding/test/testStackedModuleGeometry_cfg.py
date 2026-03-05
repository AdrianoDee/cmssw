import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

# Message logger
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.MessageLogger.cerr.StackedModuleGeometryAnalyzer = cms.untracked.PSet(
    limit=cms.untracked.int32(-1)
)
process.MessageLogger.cerr.StackedModuleGeometryESProducer = cms.untracked.PSet(
    limit=cms.untracked.int32(-1)
)

# Load Phase-2 Run4 geometry
process.load("Configuration.Geometry.GeometryExtendedRun4D110Reco_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

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
# process.GlobalTag = GlobalTag(process.GlobalTag, "auto:phase2_realistic", "")

# Load the StackedModuleGeometry ESProducer
process.load("RecoTracker.PixelSeeding.stackedModuleGeometryESProducer_cfi")

# Configure parameters if needed (defaults should be OK)
process.stackedModuleGeometryESProducer.minPt = cms.double(2.0)  # GeV
process.stackedModuleGeometryESProducer.magneticField = cms.double(3.8)  # Tesla

# Empty source - we just need one "event" to trigger BeginRun
process.source = cms.Source(
    "EmptySource",
    firstRun=cms.untracked.uint32(1),
    numberEventsInRun=cms.untracked.uint32(1),
)

process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(1))

# The analyzer
process.analyzer = cms.EDAnalyzer(
    "StackedModuleGeometryAnalyzer",
    dumpFirstN=cms.int32(13200),  # Print details for first 20 modules
    checkValues=cms.bool(True),  # Perform validation checks
)

process.p = cms.Path(process.analyzer)

# Print configuration
print("\n" + "=" * 60)
print("  Stacked Module Geometry Test Configuration")
print("=" * 60)
print(f"  Geometry: Run4 D110")
print(f"  Global Tag: {process.GlobalTag.globaltag._value}")
print(f"  Min pT: {process.stackedModuleGeometryESProducer.minPt._value} GeV")
print(f"  B field: {process.stackedModuleGeometryESProducer.magneticField._value} T")
print(f"  Dump first N: {process.analyzer.dumpFirstN._value}")
print("=" * 60 + "\n")
