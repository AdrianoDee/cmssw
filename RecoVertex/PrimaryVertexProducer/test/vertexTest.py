import FWCore.ParameterSet.Config as cms
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import offlinePrimaryVertices, offlinePrimaryVerticesCUDA, offlinePrimaryVerticesDumbFitter
#from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVerticesCUDA_cfi import offlinePrimaryVertices as offlinePrimaryVerticesCUDA
#from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVerticesCUDA_cfi import offlinePrimaryVertices as offlinePrimaryVerticesDumpFitter
import FWCore.ParameterSet.VarParsing as VarParsing

from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

process = cms.Process("Vertexing")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("TrackingTools/TransientTrack/TransientTrackBuilder_cfi")
process.load("Configuration.Geometry.GeometryIdeal_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("HeterogeneousCore.CUDAServices.CUDAService_cfi")
process.load('Configuration.EventContent.EventContent_cff')
process.load('DQMServices.Core.DQMStoreNonLegacy_cff')
process.load('commons_cff')


options = VarParsing.VarParsing('analysis')

options.register ('n',
                  100,  # default value
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

process.GlobalTag = GlobalTag(process.GlobalTag, '124X_mcRun4_realistic_v8', '')
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
    numberOfStreams = cms.untracked.uint32(options.threads),
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

# process.source = cms.Source("PoolSource",
# fileNames = cms.untracked.vstring(
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/7781d089-b51a-495a-b1ba-384c15e90749.root'
# # ,'/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/f6b68ca4-5b0e-42bb-b1d0-f94480067693.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/876a46e3-477e-4c53-8a4a-c16e7c8dee0b.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/d6bb76ec-887c-44bf-bf00-5dd719dd5dff.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/841a606b-e33f-49db-9194-bb9ff42c81ff.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/60dbef09-8c37-47b2-a892-f5609d6e3a40.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/38ff9d45-d60a-4efe-83f8-719de194cf64.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/bf35cf3a-b951-4f8d-a210-3ea46c862aec.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/ae8e2b9d-4a13-4bb4-a60f-14f5e76bc73b.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/c21e804b-e71c-4db1-96dd-386a09c1cf48.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/343ff501-8d71-4b42-8b05-cfbb919a2f5d.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/604894af-bd4e-409c-bc43-13c0eb2bafd8.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/3228c9f0-8a7a-4132-9eee-7038f2017f11.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/f52ed17e-9206-4677-98a9-176c64cdd717.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/562dc429-d934-4d2f-820f-307adb310345.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/56bd4551-51c5-402f-8c91-6a5944733142.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/56300195-9b25-4c6a-acda-d2f828fab8ca.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/70efe662-719f-4529-a29c-901e1a591431.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/694294ab-47c3-4e7f-9750-ee9e97ae92f5.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/5072327d-e50a-48e3-81e8-88203d3f5f3d.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/b1c0b7b4-e08b-4438-ac2a-3a2ec7c2aff0.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/1caa8457-f38e-40ac-91c0-cadb065805d9.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/e2277657-d953-4fc2-ad3d-b8c28a1523f6.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/d1792618-2855-43c6-a381-8c109caf0fb0.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/26bf4e54-e802-4c18-810b-2b9095275522.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/1ed83b72-6f77-4446-9833-4bebbec03989.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/576f5337-ff6d-4590-99b3-682c121656f6.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/9d572000-bbd1-466b-b8d5-c031e3518957.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/b6a6f3d2-3358-481e-a6a4-a7832ce1b7b1.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/a3102d17-f249-4e42-a026-9319d89b782e.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/0d357055-494e-49f9-b36b-3e112eb5c473.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/d2e99970-100e-43d0-b986-116424ca57b6.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/04dce43e-b72c-4685-a30a-db468881dff5.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/9b295986-86c8-46d3-9139-1d0e289c92e6.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/6021d9f0-d493-448b-b329-718f2a4b404a.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/e56aba2c-90c1-4cc6-9a36-225b707b3319.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/10e668f9-3c77-4632-a2f0-20eefe86897e.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/7f8d4917-e5ae-4e53-a1a9-5e1986d314f3.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/b0b5afc2-6de5-439e-8a36-3e83c123d2c6.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/18e64b73-6718-4ff8-8810-e0f00a843548.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/426737a0-8172-4c20-ab88-f1ce2f120e13.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/d0072331-c3db-4b72-8eac-55a93ee99da1.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/1af72ef7-4414-4e34-bbfa-3abeb08bf825.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/756440ad-5946-497b-967e-0ac33b8568d0.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/9eaf4e12-20f5-4cdf-9da8-fe54457830f8.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/43a753cb-ac18-462b-b88b-c9301a5f6c35.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/5f4012d5-30d3-4279-ac7d-26a2769e78cd.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/830e3a55-9858-4366-9a26-7dc4f399d821.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/ec9f4885-7c1e-491b-bf60-a820abb3745f.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/a5ffa31e-d258-4f80-8c0f-2d8b7770b869.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/f38934b2-4788-4589-a7c1-fdd8c7ef8445.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/e41730b4-68e2-4e1c-bc59-cbc3415d5d9b.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/c1447207-6e26-456e-8bc5-32617fbcad82.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/51a121e3-c901-4811-a2bd-595b6c0ddaab.root',
# # '/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/fa95d4f2-28ae-4a9a-9840-d90be549b067.root',
# # #'/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/01b3b6fd-4e69-4d27-ad97-d889c9ca1f54.root'
# #
# #
# # #"/store/relval/CMSSW_12_4_0_pre3/RelValZpTT_1500_14/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/7e8f544a-834c-44a8-9773-1707b289a7c8.root",
# # #"/store/relval/CMSSW_12_4_0_pre3/RelValZpTT_1500_14/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/fe6452da-53cb-4f10-b490-9cf853936909.root",
# # #"/store/relval/CMSSW_12_4_0_pre3/RelValZpTT_1500_14/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/0409aeb4-70e7-4ac6-8949-01107cf661ff.root",
# # #"/store/relval/CMSSW_12_4_0_pre3/RelValZpTT_1500_14/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v11_2026D88PU200-v1/2580000/c8456b9b-8afd-4c67-8612-1ab74f75a2ea.root"
# #
# # #'/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/123X_mcRun4_realistic_v11_2026D88noPU-v1/2580000/084cd150-0989-4da9-bed0-7330725343ec.root',
# # #'/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/123X_mcRun4_realistic_v11_2026D88noPU-v1/2580000/1ca0c2f3-115c-410c-89a5-71b1e5c024d5.root',
# # #'/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/123X_mcRun4_realistic_v11_2026D88noPU-v1/2580000/275980fe-9e20-4709-9ff0-74ef4c485e0f.root',
# # #'/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/123X_mcRun4_realistic_v11_2026D88noPU-v1/2580000/2cd276f9-8f07-4a15-ac7f-481ff632fa12.root',
# # #'/store/relval/CMSSW_12_4_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/123X_mcRun4_realistic_v11_2026D88noPU-v1/2580000/4cb86d46-f780-4ce7-94df-9e0039e1953b.root'
# ),
# skipEvents=cms.untracked.uint32(0),
# inputCommands = cms.untracked.vstring(
#         'keep *','drop *_offlinePrimaryVertices_*_*'
#     )
# )


process.source = cms.Source("PoolSource",
fileNames = cms.untracked.vstring(
"file:/lustre/cms/store/user/adiflori/d88_ttbar/125X/reco/0c7739f4-fb27-4e77-a352-7d7452734804.root",
"file:/lustre/cms/store/user/adiflori/d88_ttbar/125X/reco/46447638-bd7c-4b04-8af6-1769a2a21ada.root",
"file:/lustre/cms/store/user/adiflori/d88_ttbar/125X/reco/47e0febb-f50d-44c8-b2f9-b05f98a3f1a3.root",
"file:/lustre/cms/store/user/adiflori/d88_ttbar/125X/reco/4b043fdb-ec69-4c47-9966-7fe36f282d97.root",
"file:/lustre/cms/store/user/adiflori/d88_ttbar/125X/reco/533ef852-a5e6-4ece-bc68-d579514395e7.root",
"file:/lustre/cms/store/user/adiflori/d88_ttbar/125X/reco/794d72d9-878f-427b-ae0c-cef3c2bb0406.root",
"file:/lustre/cms/store/user/adiflori/d88_ttbar/125X/reco/91d6ff57-02e3-4cc8-be42-fd281321f04d.root",
"file:/lustre/cms/store/user/adiflori/d88_ttbar/125X/reco/a906141e-016e-454a-99fe-a7bcc94dac20.root",
"file:/lustre/cms/store/user/adiflori/d88_ttbar/125X/reco/ba7f7b66-e505-4c43-8795-24cf99e2d354.root",
"file:/lustre/cms/store/user/adiflori/d88_ttbar/125X/reco/cc53c2b3-c802-4b5c-a032-a28ce56823db.root"
),
skipEvents=cms.untracked.uint32(0),
inputCommands = cms.untracked.vstring(
        'keep *','drop *_offlinePrimaryVertices_*_*','drop l1tTk*_*_*_*'
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
    process.vertex = offlinePrimaryVertices.clone()
    process.vertex.TkClusParameters.TkDAClusParameters.runInBlocks = True
    process.vertex.TkClusParameters.TkDAClusParameters.block_size = cms.uint32(512)
    process.vertex.TkClusParameters.TkDAClusParameters.overlap_frac = 0.5
    #process.vertex = offlinePrimaryVerticesCUDA.clone()
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

process.tracksValidationTruth = cms.Task(process.VertexAssociatorByPositionAndTracks, process.quickTrackAssociatorByHits, process.tpClusterProducer)
process.pvValidation = cms.Sequence(process.vertexAnalysis,process.tracksValidationTruth)
process.prevalidation_step = cms.Path(process.pvValidation)

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
