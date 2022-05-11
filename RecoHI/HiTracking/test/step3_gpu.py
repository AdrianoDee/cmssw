# Auto generated configuration file
# using:
# Revision: 1.19
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v
# with command line options: step3 -s RAW2DIGI,L1Reco,RECO,PAT,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM --conditions auto:phase1_2021_realistic_hi -n 2 --datatier GEN-SIM-RECO,MINIAODSIM,DQMIO --eventcontent RECOSIM,MINIAODSIM,DQM --era Run3_pp_on_PbPb --procModifiers genJetSubEvent --filein file:step2.root --fileout file:step3.root --no_exec
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_pp_on_PbPb_cff import Run3_pp_on_PbPb
from Configuration.ProcessModifiers.genJetSubEvent_cff import genJetSubEvent
from Configuration.ProcessModifiers.gpu_cff import gpu
from Configuration.ProcessModifiers.pixelNtupletFit_cff import pixelNtupletFit

process = cms.Process('RECO',Run3_pp_on_PbPb,genJetSubEvent,gpu)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('PhysicsTools.PatAlgos.slimming.metFilterPaths_cff')
process.load('Configuration.StandardSequences.PATMC_cff')
process.load('Configuration.StandardSequences.Validation_cff')
process.load('DQMServices.Core.DQMStoreNonLegacy_cff')
process.load('DQMOffline.Configuration.DQMOfflineMC_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("PoolSource",
 fileNames = cms.untracked.vstring(
'/store/user/jaebeom/MB_Hydjet_PbPb5p02TeV_DIGI2RAW_Run3_CMSSW_12_1_0_patch3_v1/MB_Hydjet_PbPb5p02TeV_GENSIM_Run3Cond_CMSSW_12_1_0_patch3_v2/MB_Hydjet_PbPb5p02TeV_DIGI2RAW_Run3_CMSSW_12_1_0_patch3_v1/210930_083037/0000/step1_DIGIRAW_1.root',
'/store/user/jaebeom/MB_Hydjet_PbPb5p02TeV_DIGI2RAW_Run3_CMSSW_12_1_0_patch3_v1/MB_Hydjet_PbPb5p02TeV_GENSIM_Run3Cond_CMSSW_12_1_0_patch3_v2/MB_Hydjet_PbPb5p02TeV_DIGI2RAW_Run3_CMSSW_12_1_0_patch3_v1/210930_083037/0000/step1_DIGIRAW_10.root',
'/store/user/jaebeom/MB_Hydjet_PbPb5p02TeV_DIGI2RAW_Run3_CMSSW_12_1_0_patch3_v1/MB_Hydjet_PbPb5p02TeV_GENSIM_Run3Cond_CMSSW_12_1_0_patch3_v2/MB_Hydjet_PbPb5p02TeV_DIGI2RAW_Run3_CMSSW_12_1_0_patch3_v1/210930_083037/0000/step1_DIGIRAW_100.root',
'/store/user/jaebeom/MB_Hydjet_PbPb5p02TeV_DIGI2RAW_Run3_CMSSW_12_1_0_patch3_v1/MB_Hydjet_PbPb5p02TeV_GENSIM_Run3Cond_CMSSW_12_1_0_patch3_v2/MB_Hydjet_PbPb5p02TeV_DIGI2RAW_Run3_CMSSW_12_1_0_patch3_v1/210930_083037/0000/step1_DIGIRAW_101.root',
'/store/user/jaebeom/MB_Hydjet_PbPb5p02TeV_DIGI2RAW_Run3_CMSSW_12_1_0_patch3_v1/MB_Hydjet_PbPb5p02TeV_GENSIM_Run3Cond_CMSSW_12_1_0_patch3_v2/MB_Hydjet_PbPb5p02TeV_DIGI2RAW_Run3_CMSSW_12_1_0_patch3_v1/210930_083037/0000/step1_DIGIRAW_102.root',
'/store/user/jaebeom/MB_Hydjet_PbPb5p02TeV_DIGI2RAW_Run3_CMSSW_12_1_0_patch3_v1/MB_Hydjet_PbPb5p02TeV_GENSIM_Run3Cond_CMSSW_12_1_0_patch3_v2/MB_Hydjet_PbPb5p02TeV_DIGI2RAW_Run3_CMSSW_12_1_0_patch3_v1/210930_083037/0000/step1_DIGIRAW_103.root',
'/store/user/jaebeom/MB_Hydjet_PbPb5p02TeV_DIGI2RAW_Run3_CMSSW_12_1_0_patch3_v1/MB_Hydjet_PbPb5p02TeV_GENSIM_Run3Cond_CMSSW_12_1_0_patch3_v2/MB_Hydjet_PbPb5p02TeV_DIGI2RAW_Run3_CMSSW_12_1_0_patch3_v1/210930_083037/0000/step1_DIGIRAW_104.root',
'/store/user/jaebeom/MB_Hydjet_PbPb5p02TeV_DIGI2RAW_Run3_CMSSW_12_1_0_patch3_v1/MB_Hydjet_PbPb5p02TeV_GENSIM_Run3Cond_CMSSW_12_1_0_patch3_v2/MB_Hydjet_PbPb5p02TeV_DIGI2RAW_Run3_CMSSW_12_1_0_patch3_v1/210930_083037/0000/step1_DIGIRAW_105.root',
'/store/user/jaebeom/MB_Hydjet_PbPb5p02TeV_DIGI2RAW_Run3_CMSSW_12_1_0_patch3_v1/MB_Hydjet_PbPb5p02TeV_GENSIM_Run3Cond_CMSSW_12_1_0_patch3_v2/MB_Hydjet_PbPb5p02TeV_DIGI2RAW_Run3_CMSSW_12_1_0_patch3_v1/210930_083037/0000/step1_DIGIRAW_106.root',
'/store/user/jaebeom/MB_Hydjet_PbPb5p02TeV_DIGI2RAW_Run3_CMSSW_12_1_0_patch3_v1/MB_Hydjet_PbPb5p02TeV_GENSIM_Run3Cond_CMSSW_12_1_0_patch3_v2/MB_Hydjet_PbPb5p02TeV_DIGI2RAW_Run3_CMSSW_12_1_0_patch3_v1/210930_083037/0000/step1_DIGIRAW_107.root'
),
 skipEvents = cms.untracked.uint32(6),
    secondaryFileNames = cms.untracked.vstring()
)

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
    numberOfThreads = cms.untracked.uint32(8),
    printDependencies = cms.untracked.bool(False),
    sizeOfStackForThreadsInKB = cms.optional.untracked.uint32,
    throwIfIllegalParameter = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(False)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step3 nevts:2'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.RECOSIMoutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-RECO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:step3.root'),
    outputCommands = process.RECOSIMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

process.MINIAODSIMoutput = cms.OutputModule("PoolOutputModule",
    compressionAlgorithm = cms.untracked.string('LZMA'),
    compressionLevel = cms.untracked.int32(4),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('MINIAODSIM'),
        filterName = cms.untracked.string('')
    ),
    dropMetaData = cms.untracked.string('ALL'),
    eventAutoFlushCompressedSize = cms.untracked.int32(-900),
    fastCloning = cms.untracked.bool(False),
    fileName = cms.untracked.string('file:step3_inMINIAODSIM.root'),
    outputCommands = process.MINIAODSIMEventContent.outputCommands,
    overrideBranchesSplitLevel = cms.untracked.VPSet(
        cms.untracked.PSet(
            branch = cms.untracked.string('patPackedCandidates_packedPFCandidates__*'),
            splitLevel = cms.untracked.int32(99)
        ),
        cms.untracked.PSet(
            branch = cms.untracked.string('recoGenParticles_prunedGenParticles__*'),
            splitLevel = cms.untracked.int32(99)
        ),
        cms.untracked.PSet(
            branch = cms.untracked.string('patTriggerObjectStandAlones_slimmedPatTrigger__*'),
            splitLevel = cms.untracked.int32(99)
        ),
        cms.untracked.PSet(
            branch = cms.untracked.string('patPackedGenParticles_packedGenParticles__*'),
            splitLevel = cms.untracked.int32(99)
        ),
        cms.untracked.PSet(
            branch = cms.untracked.string('patJets_slimmedJets__*'),
            splitLevel = cms.untracked.int32(99)
        ),
        cms.untracked.PSet(
            branch = cms.untracked.string('recoVertexs_offlineSlimmedPrimaryVertices__*'),
            splitLevel = cms.untracked.int32(99)
        ),
        cms.untracked.PSet(
            branch = cms.untracked.string('recoVertexs_offlineSlimmedPrimaryVerticesWithBS__*'),
            splitLevel = cms.untracked.int32(99)
        ),
        cms.untracked.PSet(
            branch = cms.untracked.string('recoCaloClusters_reducedEgamma_reducedESClusters_*'),
            splitLevel = cms.untracked.int32(99)
        ),
        cms.untracked.PSet(
            branch = cms.untracked.string('EcalRecHitsSorted_reducedEgamma_reducedEBRecHits_*'),
            splitLevel = cms.untracked.int32(99)
        ),
        cms.untracked.PSet(
            branch = cms.untracked.string('EcalRecHitsSorted_reducedEgamma_reducedEERecHits_*'),
            splitLevel = cms.untracked.int32(99)
        ),
        cms.untracked.PSet(
            branch = cms.untracked.string('recoGenJets_slimmedGenJets__*'),
            splitLevel = cms.untracked.int32(99)
        ),
        cms.untracked.PSet(
            branch = cms.untracked.string('patJets_slimmedJetsPuppi__*'),
            splitLevel = cms.untracked.int32(99)
        ),
        cms.untracked.PSet(
            branch = cms.untracked.string('EcalRecHitsSorted_reducedEgamma_reducedESRecHits_*'),
            splitLevel = cms.untracked.int32(99)
        )
    ),
    overrideInputFileSplitLevels = cms.untracked.bool(True),
    splitLevel = cms.untracked.int32(0)
)

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('DQMIO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:step3_inDQM_gpu.root'),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
process.mix.playback = True
process.mix.digitizers = cms.PSet()
for a in process.aliases: delattr(process, a)
process.RandomNumberGeneratorService.restoreStateLabel=cms.untracked.string("randomEngineStateProducer")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2021_realistic_hi', '')

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction)
process.Flag_BadChargedCandidateFilter = cms.Path(process.BadChargedCandidateFilter)
process.Flag_BadChargedCandidateSummer16Filter = cms.Path(process.BadChargedCandidateSummer16Filter)
process.Flag_BadPFMuonDzFilter = cms.Path(process.BadPFMuonDzFilter)
process.Flag_BadPFMuonFilter = cms.Path(process.BadPFMuonFilter)
process.Flag_BadPFMuonSummer16Filter = cms.Path(process.BadPFMuonSummer16Filter)
process.Flag_CSCTightHalo2015Filter = cms.Path(process.CSCTightHalo2015Filter)
process.Flag_CSCTightHaloFilter = cms.Path(process.CSCTightHaloFilter)
process.Flag_CSCTightHaloTrkMuUnvetoFilter = cms.Path(process.CSCTightHaloTrkMuUnvetoFilter)
process.Flag_EcalDeadCellBoundaryEnergyFilter = cms.Path(process.EcalDeadCellBoundaryEnergyFilter)
process.Flag_EcalDeadCellTriggerPrimitiveFilter = cms.Path(process.EcalDeadCellTriggerPrimitiveFilter)
process.Flag_HBHENoiseFilter = cms.Path(process.HBHENoiseFilterResultProducer+process.HBHENoiseFilter)
process.Flag_HBHENoiseIsoFilter = cms.Path(process.HBHENoiseFilterResultProducer+process.HBHENoiseIsoFilter)
process.Flag_HcalStripHaloFilter = cms.Path(process.HcalStripHaloFilter)
process.Flag_METFilters = cms.Path(process.metFilters)
process.Flag_chargedHadronTrackResolutionFilter = cms.Path(process.chargedHadronTrackResolutionFilter)
process.Flag_ecalBadCalibFilter = cms.Path(process.ecalBadCalibFilter)
process.Flag_ecalLaserCorrFilter = cms.Path(process.ecalLaserCorrFilter)
process.Flag_eeBadScFilter = cms.Path(process.eeBadScFilter)
process.Flag_globalSuperTightHalo2016Filter = cms.Path(process.globalSuperTightHalo2016Filter)
process.Flag_globalTightHalo2016Filter = cms.Path(process.globalTightHalo2016Filter)
process.Flag_goodVertices = cms.Path(process.primaryVertexFilter)
process.Flag_hcalLaserEventFilter = cms.Path(process.hcalLaserEventFilter)
process.Flag_hfNoisyHitsFilter = cms.Path(process.hfNoisyHitsFilter)
process.Flag_muonBadTrackFilter = cms.Path(process.muonBadTrackFilter)
process.Flag_trackingFailureFilter = cms.Path(process.goodVertices+process.trackingFailureFilter)
process.Flag_trkPOGFilters = cms.Path(process.trkPOGFilters)
process.Flag_trkPOG_logErrorTooManyClusters = cms.Path(~process.logErrorTooManyClusters)
process.Flag_trkPOG_manystripclus53X = cms.Path(~process.manystripclus53X)
process.Flag_trkPOG_toomanystripclus53X = cms.Path(~process.toomanystripclus53X)
process.prevalidation_step = cms.Path(process.prevalidationNoHLT)
process.prevalidation_step1 = cms.Path(process.prevalidationMiniAOD)
process.validation_step = cms.EndPath(process.validationNoHLT)
process.validation_step1 = cms.EndPath(process.validationMiniAOD)
process.dqmoffline_step = cms.EndPath(process.DQMOfflineFakeHLT)
process.dqmoffline_1_step = cms.EndPath(process.DQMOfflineMiniAOD)
process.dqmofflineOnPAT_step = cms.EndPath(process.PostDQMOffline)
process.dqmofflineOnPAT_1_step = cms.EndPath(process.PostDQMOfflineMiniAOD)
process.RECOSIMoutput_step = cms.EndPath(process.RECOSIMoutput)
process.MINIAODSIMoutput_step = cms.EndPath(process.MINIAODSIMoutput)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

import validation

process.trackValidatorHIonPixelTrackingOnly = validation.trackValidatorHIonPixelTrackingOnly.clone()
process.trackingParticlePixelTrackAsssociationHIon = validation.trackingParticlePixelTrackAsssociationHIon.clone()
process.tpClusterProducerHIon = validation.tpClusterProducerHIon.clone()
process.quickTrackAssociatorByHitsHIon = validation.quickTrackAssociatorByHitsHIon.clone()
process.VertexAssociatorByPositionAndTracksHIon = validation.VertexAssociatorByPositionAndTracksHIon.clone()

process.tracksValidation = cms.Sequence(process.trackValidator+process.trackValidatorHIonPixelTrackingOnly+process.trackValidatorTPPtLess09+process.trackValidatorFromPV+process.trackValidatorFromPVAllTP+process.trackValidatorAllTPEffic+process.trackValidatorBuilding+process.trackValidatorBuildingPreSplitting+process.trackValidatorConversion+process.trackValidatorGsfTracks, process.tracksPreValidation)
process.tracksValidationTruth = cms.Task(process.VertexAssociatorByPositionAndTracks,process.VertexAssociatorByPositionAndTracksHIon, process.quickTrackAssociatorByHits, process.quickTrackAssociatorByHitsHIon, process.quickTrackAssociatorByHitsPreSplitting, process.tpClusterProducer, process.tpClusterProducerHIon, process.tpClusterProducerPreSplitting, process.trackAssociatorByChi2, process.trackingParticleNumberOfLayersProducer, process.trackingParticleRecoTrackAsssociation,process.trackingParticlePixelTrackAsssociationHIon)

# process.tracksValidation = cms.Sequence(process.tpClusterProducerHIon+process.trackingParticleNumberOfLayersProducer+ process.quickTrackAssociatorByHitsHIon+process.trackingParticlePixelTrackAsssociationHIon+process.VertexAssociatorByPositionAndTracksHIon+process.trackValidatorHIonPixelTrackingOnly)

process.pixelTracksCUDAHIon.doClusterCut = True
process.pixelTracksCUDAHIon.hardCurvCut = 0.0756 #1.f / (minPt* 87.f)
process.pixelTracksCUDAHIon.idealConditions = False
process.pixelTracksCUDAHIon.ptmin = 0.2
process.hiConformalPixelTracks.minQuality = "strict"


process.dump=cms.EDAnalyzer('EventContentAnalyzer')
process.eventContentAnalyzer = cms.Path(process.dump)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.eventContentAnalyzer,process.L1Reco_step,process.reconstruction_step,process.Flag_HBHENoiseFilter,process.Flag_HBHENoiseIsoFilter,process.Flag_CSCTightHaloFilter,process.Flag_CSCTightHaloTrkMuUnvetoFilter,process.Flag_CSCTightHalo2015Filter,process.Flag_globalTightHalo2016Filter,process.Flag_globalSuperTightHalo2016Filter,process.Flag_HcalStripHaloFilter,process.Flag_hcalLaserEventFilter,process.Flag_EcalDeadCellTriggerPrimitiveFilter,process.Flag_EcalDeadCellBoundaryEnergyFilter,process.Flag_ecalBadCalibFilter,process.Flag_goodVertices,process.Flag_eeBadScFilter,process.Flag_ecalLaserCorrFilter,process.Flag_trkPOGFilters,process.Flag_chargedHadronTrackResolutionFilter,process.Flag_muonBadTrackFilter,process.Flag_BadChargedCandidateFilter,process.Flag_BadPFMuonFilter,process.Flag_BadPFMuonDzFilter,process.Flag_hfNoisyHitsFilter,process.Flag_BadChargedCandidateSummer16Filter,process.Flag_BadPFMuonSummer16Filter,process.Flag_trkPOG_manystripclus53X,process.Flag_trkPOG_toomanystripclus53X,process.Flag_trkPOG_logErrorTooManyClusters,process.Flag_METFilters,process.prevalidation_step,process.prevalidation_step1,process.validation_step,process.validation_step1,process.dqmoffline_step,process.dqmoffline_1_step,process.dqmofflineOnPAT_step,process.dqmofflineOnPAT_1_step,process.RECOSIMoutput_step,process.MINIAODSIMoutput_step,process.DQMoutput_step)
process.schedule.associate(process.patTask)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

# customisation of the process.

process.load( "HLTrigger.Timer.FastTimerService_cfi" )
# print a text summary at the end of the job
process.FastTimerService.printEventSummary         = False
process.FastTimerService.printRunSummary           = False
process.FastTimerService.printJobSummary           = True

# enable DQM plots
process.FastTimerService.enableDQM                 = True

# enable per-path DQM plots (starting with CMSSW 9.2.3-patch2)
process.FastTimerService.enableDQMbyPath           = True

# enable per-module DQM plots
process.FastTimerService.enableDQMbyModule         = True

# enable per-event DQM plots vs lumisection
process.FastTimerService.enableDQMbyLumiSection    = True
process.FastTimerService.dqmLumiSectionsRange      = 2500

# set the time resolution of the DQM plots
tr = 10000000000.
tp = 10000000000.
tm = 2000000000.
process.FastTimerService.dqmTimeRange              = tr
process.FastTimerService.dqmTimeResolution         = tr/100.0
process.FastTimerService.dqmPathTimeRange          = tp
process.FastTimerService.dqmPathTimeResolution     = tp/100.0
process.FastTimerService.dqmModuleTimeRange        = tm
process.FastTimerService.dqmModuleTimeResolution   = tm/100.0

# set the base DQM folder for the plots
process.FastTimerService.dqmPath                   = 'HLT/TimerService'
process.FastTimerService.enableDQMbyProcesses      = True


process.FastTimerService.dqmMemoryRange            = 1000000
process.FastTimerService.dqmMemoryResolution       =    5000
process.FastTimerService.dqmPathMemoryRange        = 1000000
process.FastTimerService.dqmPathMemoryResolution   =    5000
process.FastTimerService.dqmModuleMemoryRange      =  100000
process.FastTimerService.dqmModuleMemoryResolution =     500

process.FastTimerService.writeJSONSummary = cms.untracked.bool(True)

process.FastTimerService.jsonFileName = cms.untracked.string('hion.json')

# Automatic addition of the customisation function from SimGeneral.MixingModule.fullMixCustomize_cff
from SimGeneral.MixingModule.fullMixCustomize_cff import setCrossingFrameOn

#call to customisation function setCrossingFrameOn imported from SimGeneral.MixingModule.fullMixCustomize_cff
process = setCrossingFrameOn(process)

# End of customisation functions

# customisation of the process.

# Automatic addition of the customisation function from PhysicsTools.PatAlgos.slimming.miniAOD_tools
from PhysicsTools.PatAlgos.slimming.miniAOD_tools import miniAOD_customizeAllMC

#call to customisation function miniAOD_customizeAllMC imported from PhysicsTools.PatAlgos.slimming.miniAOD_tools
process = miniAOD_customizeAllMC(process)

# End of customisation functions

# Customisation from command line

#Have logErrorHarvester wait for the same EDProducers to finish as those providing data for the OutputModule
from FWCore.Modules.logErrorHarvester_cff import customiseLogErrorHarvesterUsingOutputCommands
process = customiseLogErrorHarvesterUsingOutputCommands(process)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
