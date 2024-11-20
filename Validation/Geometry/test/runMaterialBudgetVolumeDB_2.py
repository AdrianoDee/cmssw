###############################################################################
# Way to use this:
#   cmsRun runMaterialBudgetVolumeDB_cfg.py year=YEAR
#
#   Options for year 2021,2022,2023,2024
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('year',
                 "2024",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "Year for the GT (2021,2022,20223,2024,...)")
### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

import FWCore.ParameterSet.Config as cms

year = options.year

from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process('PROD',Run3)

fileName = "materiabdgt_%s.root"%(year)
print("Root file Name:     ", fileName)


process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.GeometrySimDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_' + options.year + '_realistic', '')

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("SimG4Core.Application.g4SimHits_cfi")
process.load("Validation.Geometry.materialBudgetVolume_cfi")

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(1000)
if hasattr(process,'MessageLogger'):
    process.MessageLogger.MaterialBudget=dict()

process.source = cms.Source("PoolSource",
    noEventSort = cms.untracked.bool(True),
    duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
    fileNames = cms.untracked.vstring('file:single_neutrino_random.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string(fileName)
)

from Validation.Geometry.plot_utils import _LABELS2COMPS
_components = _LABELS2COMPS['Tracker']

process.g4SimHits.StackingAction.TrackNeutrino = cms.bool(True)
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/DummyPhysics'
process.g4SimHits.Physics.DummyEMPhysics = True
process.g4SimHits.Physics.CutsPerRegion = False
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    type = cms.string('MaterialBudgetAction'),
    MaterialBudgetAction = cms.PSet(
        HistosFile = cms.string('matbdg_%s.root' % (options.year)),
        AllStepsToTree = cms.bool(True),
        HistogramList = cms.string('Tracker'),
        SelectedVolumes = cms.vstring(_components),
        TreeFile = cms.string('matbdg_tree_%s.root' % (options.year)),
        StopAfterProcess = cms.string('None'),
        TextFile = cms.string('matbdg_%s.txt' % (options.year))
    )
))


process.load("Validation.Geometry.materialBudgetVolumeAnalysis_cfi")
process.p1 = cms.Path(process.g4SimHits+process.materialBudgetVolumeAnalysis)
