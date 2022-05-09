import FWCore.ParameterSet.Config as cms
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import offlinePrimaryVertices
process = cms.Process("Demo")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("TrackingTools/TransientTrack/TransientTrackBuilder_cfi")
process.load("Configuration.Geometry.GeometryIdeal_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
#process.load('HeterogeneousCore.CUDACore.ProcessAcceleratorCUDA_cfi')
process.load("HeterogeneousCore.CUDAServices.CUDAService_cfi")
process.load('Configuration.EventContent.EventContent_cff')
process.load('DQMServices.Core.DQMStoreNonLegacy_cff')
process.load('validation_cff')

#process.load( "HLTrigger.Timer.FastTimerService_cfi" )

#process.GlobalTag.globaltag = 'GR_P_V42_AN3::All'
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T21', '')
process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))
"""
process.IgProfService = cms.Service("IgProfService",
  reportFirstEvent            = cms.untracked.int32(0),
  reportEventInterval         = cms.untracked.int32(25),
  reportToFileAtPostEvent     = cms.untracked.string("| gzip -c > igdqm.%I.gz")
)
"""
process.source = cms.Source("PoolSource",
# replace 'myfile.root' with the source file you want to use
fileNames = cms.untracked.vstring(
#'file:/afs/cern.ch/user/g/gpizzati/CMSSW_12_2_0_pre3/src/step2.root'
'file:aca7b050-5990-4576-a9ee-f41ac82e5b86.root'
# /store/relval/CMSSW_12_2_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_122X_mcRun4_realistic_v2_2026D88PU200-v1/2580000/
#'/store/relval/CMSSW_12_3_0_pre5/RelValTTbar_14TeV/GEN-SIM-RECO/123X_mcRun4_realistic_v4_2026D88noPU-v1/10000/7cbdf153-c397-410e-8af5-dc3905565ced.root'
),
#firstEvent = cms.untracked.uint32(2)
skipEvents=cms.untracked.uint32(0)
)

"""
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
"""
"""
process.FastTimerService.writeJSONSummary = cms.untracked.bool(True)
process.FastTimerService.jsonFileName = cms.untracked.string('resources.json')
"""
process.demo = offlinePrimaryVertices
"""

process.demo = cms.EDProducer("PrimaryVertexProducer",
    TkClusParameters = cms.PSet(
        algorithm   = cms.string("DA_vectCUDA"),
        TkDAClusParameters = cms.PSet(
            coolingFactor = cms.double(0.6),  # moderate annealing speed
            zrange = cms.double(4.),          # consider only clusters within 4 sigma*sqrt(T) of a track
            delta_highT = cms.double(1.e-2),  # convergence requirement at high T
            delta_lowT = cms.double(1.e-3),   # convergence requirement at low T
            convergence_mode = cms.int32(0),  # 0 = two steps, 1 = dynamic with sqrt(T)
            Tmin = cms.double(2.0),           # end of vertex splitting
            Tpurge = cms.double(2.0),         # cleaning
            Tstop = cms.double(0.5),          # end of annealing
            vertexSize = cms.double(0.006),   # added in quadrature to track-z resolutions
            d0CutOff = cms.double(3.),        # downweight high IP tracks
            dzCutOff = cms.double(3.),        # outlier rejection after freeze-out (T<Tmin)
            zmerge = cms.double(1e-2),        # merge intermediat clusters separated by less than zmerge
            uniquetrkweight = cms.double(0.8),# require at least two tracks with this weight at T=Tpurge
            uniquetrkminp = cms.double(0.0)   # minimal a priori track weight for counting unique tracks
            )
    ),
    TkFilterParameters = cms.PSet(
        algorithm=cms.string('filter'),
        maxNormalizedChi2 = cms.double(10.0),
        minPixelLayersWithHits=cms.int32(2),
        minSiliconLayersWithHits = cms.int32(5),
        maxD0Significance = cms.double(4.0),
        maxD0Error = cms.double(1.0),
        maxDzError = cms.double(1.0),
        minPt = cms.double(0.0),
        maxEta = cms.double(2.4),
        trackQuality = cms.string("any")
    ),
    TrackLabel = cms.InputTag("generalTracks"), #TrackLabel = cms.InputTag("pixelTracks","","RECO"),
    beamSpotLabel = cms.InputTag("offlineBeamSpot"),
    verbose = cms.untracked.bool(False),
    vertexCollections = cms.VPSet(
     [cms.PSet(label=cms.string(""),
               algorithm=cms.string("AdaptiveVertexFitter"),
               chi2cutoff = cms.double(2.5),
               minNdof=cms.double(0.0),
               useBeamConstraint = cms.bool(False),
               maxDistanceToBeam = cms.double(1.0)
               ),
      cms.PSet(label=cms.string("WithBS"),
               algorithm = cms.string('AdaptiveVertexFitter'),
               chi2cutoff = cms.double(2.5),
               minNdof=cms.double(2.0),
               useBeamConstraint = cms.bool(True),
               maxDistanceToBeam = cms.double(1.0),
               )
      ]
    ),
#    onGPU = cms.bool(True),
    isRecoveryIteration = cms.bool(False),
    recoveryVtxCollection = cms.InputTag("")
)
"""
"""
process.demo = cms.EDProducer(
    'gpuTest',
    TrackLabel = cms.InputTag("generalTracks"),
    beamSpotLabel = cms.InputTag("offlineBeamSpot"),

)
"""

process.out = cms.OutputModule("PoolOutputModule",
    fileName= cms.untracked.string("file:test_cpu.root"),
    outputCommands = cms.untracked.vstring(
                                'drop *_*_*_*',
                                'keep *_demo_*_*'
    )
)

process.tracksValidationTruth = cms.Task(process.VertexAssociatorByPositionAndTracks, process.quickTrackAssociatorByHits, process.tpClusterProducer)
process.pvValidation = cms.Sequence(process.vertexAnalysis,process.tracksValidationTruth)
process.prevalidation_step = cms.Path(process.pvValidation)

process.DQMOfflineVertex = cms.Sequence(process.pvMonitor)
process.dqmoffline_step = cms.EndPath(process.DQMOfflineVertex)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

process.p = cms.Path(process.demo)
process.ep = cms.EndPath(process.out)

process.schedule = cms.Schedule(process.p,process.prevalidation_step,process.dqmoffline_step,process.DQMoutput_step,process.ep)
