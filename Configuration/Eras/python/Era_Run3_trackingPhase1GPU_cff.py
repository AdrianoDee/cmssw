import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_cff import Run3
from Configuration.ProcessModifiers.gpuOfflineCA_cff import gpuOfflineCA
from Configuration.Eras.Modifier_trackingPhase1GPU_cff import trackingPhase1GPU

Run3_GPU = cms.ModifierChain(Run3, gpuOfflineCA,trackingPhase1GPU)


