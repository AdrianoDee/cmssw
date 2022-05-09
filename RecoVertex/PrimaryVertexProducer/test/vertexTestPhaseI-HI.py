import FWCore.ParameterSet.Config as cms
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import offlinePrimaryVertices, offlinePrimaryVerticesCUDA, offlinePrimaryVerticesDumbFitter
#from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVerticesCUDA_cfi import offlinePrimaryVertices as offlinePrimaryVerticesCUDA
#from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVerticesCUDA_cfi import offlinePrimaryVertices as offlinePrimaryVerticesDumpFitter
import FWCore.ParameterSet.VarParsing as VarParsing
#from Configuration.Eras.Era_Run3_noMkFit_cff import Run3
from Configuration.Eras.Era_Run3_pp_on_PbPb_cff import Run3_pp_on_PbPb

from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

#process = cms.Process("Vertexing", Run3_pp_on_PbPb)
process = cms.Process("Vertexing")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("TrackingTools/TransientTrack/TransientTrackBuilder_cfi")
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
#process.load("Configuration.Geometry.GeometryIdeal_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("HeterogeneousCore.CUDAServices.CUDAService_cfi")
process.load('Configuration.EventContent.EventContent_cff')
process.load('DQMServices.Core.DQMStoreNonLegacy_cff')
process.load('commons_cff')


options = VarParsing.VarParsing('analysis')

options.register ('n',
                  -1, # default value
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "n")
options.register ('threads',
                  1,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,
                  "threads")
options.register ('gpu',
                  True,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.bool,
                  "gpu")
options.register ('timing',
                  False,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.bool,
                  "timing")
options.register ('both',
                  False,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.bool,
                  "gpuVScpu")
options.parseArguments()

from Configuration.AlCa.GlobalTag import GlobalTag

#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2021_realistic', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2021_realistic_hi', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T21', '')
process.MessageLogger.cerr.FwkReport.reportEvery = 1

suff = 'gpu'

process.options = cms.untracked.PSet(
    FailPath = cms.untracked.vstring(),
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring(),
    SkipEvent = cms.untracked.vstring(),
    accelerators = cms.untracked.vstring('*'),
    allowUnscheduled = cms.obsolete.untracked.bool,
    canDeleteEarly = cms.untracked.vstring(),
    deleteNonConsumedUnscheduledModules = cms.untracked.bool(True),
    dumpOptions = cms.untracked.bool(False),
    emptyRunLumiMode = cms.obsolete.untracked.string,
    eventSetup = cms.untracked.PSet(
        forceNumberOfConcurrentIOVs = cms.untracked.PSet(
            allowAnyLabel_=cms.required.untracked.uint32
        ),
        numberOfConcurrentIOVs = cms.untracked.uint32(0)
    ),
    fileMode = cms.untracked.string('FULLMERGE'),
    forceEventSetupCacheClearOnNewRun = cms.untracked.bool(False),
    makeTriggerResults = cms.obsolete.untracked.bool,
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(0),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(0),
    numberOfThreads = cms.untracked.uint32(options.threads),
    printDependencies = cms.untracked.bool(False),
    sizeOfStackForThreadsInKB = cms.optional.untracked.uint32,
    throwIfIllegalParameter = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(True)
)


if not options.gpu:
    process.options.accelerators = cms.untracked.vstring('cpu')
    suff = 'cpu'

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(options.n))

process.source = cms.Source("PoolSource",
fileNames = cms.untracked.vstring(
#'/store/relval/CMSSW_12_1_0_pre5/RelValHSCPgluino_M-1000_TuneCP5_13TeV-pythia8/GEN-SIM-RECO/PU_121X_mcRun3_2021_realistic_v15_HighStat-v1/2580000/5073d52d-ece2-4bf3-ae32-baeec737b51d.root',
#"/store/relval/CMSSW_12_4_0_pre3/RelValTTbarToDilepton_14TeV/GEN-SIM-RECO/PU_123X_mcRun3_2021_realistic_v14-v1/2580000/482fcd10-ed9c-4163-b6b9-c5d45f9b38d7.root",
#"/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun3_2021_realistic_v14-v1/2580000/16ca3cdd-33b1-457e-aacf-73732649acca.root"
#"/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun3_2021_realistic_v14-v1/2580000/02e16a6d-c980-411e-bace-21d07a891e3b.root"
#"/store/relval/CMSSW_12_4_0_pre3/RelValZEE_14_HI_2021/GEN-SIM-RECO/123X_mcRun3_2021_realistic_HI_v14-v1/2580000/8007eb37-6dfb-49d6-a191-30722eb46230.root"
"/store/group/offcomp_upgrade-sw/gpizzati/2021ZEEHI_reco.root"

#'/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/7781d089-b51a-495a-b1ba-384c15e90749.root'
#'/store/relval/CMSSW_12_4_0_pre3/RelValZMM_14_HI_2021/GEN-SIM-RECO/123X_mcRun3_2021_realistic_HI_v14-v1/2580000/633b39de-a6cf-45ca-9182-96be30f293fe.root'
#,'/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/f6b68ca4-5b0e-42bb-b1d0-f94480067693.root',
#'/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/876a46e3-477e-4c53-8a4a-c16e7c8dee0b.root'
#'file:aca7b050-5990-4576-a9ee-f41ac82e5b86.root'
),
skipEvents=cms.untracked.uint32(0),
inputCommands = cms.untracked.vstring(
        'keep *','drop *_offlinePrimaryVertices_*_*'
    )
)

if options.timing:

    process.ThroughputService = cms.Service('ThroughputService',
        eventRange = cms.untracked.uint32(10),
        eventResolution = cms.untracked.uint32(1),
        printEventSummary = cms.untracked.bool(True),
        enableDQM = cms.untracked.bool(True),
        dqmPathByProcesses = cms.untracked.bool(False),
        dqmPath = cms.untracked.string('Throughput'),
        timeRange = cms.untracked.double(1000),
        timeResolution = cms.untracked.double(1)
    )

    process.MessageLogger.cerr.ThroughputService = cms.untracked.PSet(
        limit = cms.untracked.int32(10000000)
    )

    process.FastTimerService.writeJSONSummary = cms.untracked.bool(True)
    process.FastTimerService.jsonFileName = cms.untracked.string('resources_'+suff+'.json')

    if not options.gpu:
        process.IgProfService = cms.Service("IgProfService",
          reportFirstEvent            = cms.untracked.int32(0),
          reportEventInterval         = cms.untracked.int32(25),
          reportToFileAtPostEvent     = cms.untracked.string("| gzip -c > igdqm.%I.gz")
        )



if options.gpu:
    process.vertex = offlinePrimaryVerticesDumbFitter.clone()
else:
    process.vertex = offlinePrimaryVertices.clone()

process.vertexAnalysis.vertexRecoCollections  = cms.VInputTag("vertex")
process.pvMonitor.vertexLabel = cms.InputTag("vertex")

if options.both:
    suff = "gpuVScpu"

process.output = cms.OutputModule("PoolOutputModule",
    fileName= cms.untracked.string("file:test_gpu.root"),
    outputCommands = cms.untracked.vstring(
                                'drop *_*_*_*',
                                'keep *_demo_*_*'
    )
)

##DQM Output step
process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('DQMIO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:step3_inDQM.root'),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

process.output.fileName = 'file:test_'+suff+'.root'
process.DQMoutput.fileName = 'file:test_dqm_'+suff+'.root'

#process.tracksValidationTruth = cms.Task(process.VertexAssociatorByPositionAndTracks, process.quickTrackAssociatorByHits, process.tpClusterProducer)
#process.pvValidation = cms.Sequence(process.vertexAnalysis,process.tracksValidationTruth)
#process.prevalidation_step = cms.Path(process.pvValidation)

process.tracksValidationTruth = cms.Sequence(process.tpClusterProducer *  process.quickTrackAssociatorByHits * process.trackingParticleRecoTrackAsssociation *  process.VertexAssociatorByPositionAndTracks * process.vertexAnalysis)
process.prevalidation_step = cms.Path(process.tracksValidationTruth)

process.DQMOfflineVertex = cms.Sequence(process.pvMonitor)
process.dqmoffline_step = cms.EndPath(process.DQMOfflineVertex)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

process.vertexing_step = cms.Path(process.vertex)
process.output_step = cms.EndPath(process.output)

process.schedule = cms.Schedule(process.vertexing_step)

if options.timing:

    process.consumer = cms.EDAnalyzer("GenericConsumer", eventProducts = cms.untracked.vstring("vertex"))
    process.consume_step = cms.EndPath(process.consumer)
    process.schedule.append(process.consume_step)

else:
    process.schedule = cms.Schedule(process.vertexing_step,process.prevalidation_step,process.dqmoffline_step,process.DQMoutput_step,process.output_step)

if options.both:

    process.vertex = offlinePrimaryVertices.clone()
    process.vertexCUDA = offlinePrimaryVerticesCUDA.clone()
    process.vertexing_step = cms.Path(process.vertex,process.vertexCUDA)
    process.consumerCPU = cms.EDAnalyzer("GenericConsumer", eventProducts = cms.untracked.vstring("vertex"))
    process.consumerGPU = cms.EDAnalyzer("GenericConsumer", eventProducts = cms.untracked.vstring("vertexCUDA"))
    process.consume_step = cms.EndPath(process.consumerCPU,process.consumerGPU)

    if not options.timing:
        process.vertexAnalysis.vertexRecoCollections = cms.VInputTag("vertex","vertexCUDA")

    process.schedule.append(process.consume_step)
