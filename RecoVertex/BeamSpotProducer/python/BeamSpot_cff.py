import FWCore.ParameterSet.Config as cms

from RecoVertex.BeamSpotProducer.BeamSpot_cfi import *
from RecoVertex.BeamSpotProducer.offlineBeamSpotToCUDA_cfi import offlineBeamSpotToCUDA

offlineBeamSpotTask = cms.Task(offlineBeamSpot)

from Configuration.ProcessModifiers.gpu_cff import gpu
from Configuration.ProcessModifiers.gpuOfflineCA_cff import gpuOfflineCA 

_offlineBeamSpotTask_gpu = offlineBeamSpotTask.copy()
_offlineBeamSpotTask_gpu.add(offlineBeamSpotToCUDA)
(gpu | gpuOfflineCA).toReplaceWith(offlineBeamSpotTask, _offlineBeamSpotTask_gpu)
