import FWCore.ParameterSet.Config as cms

from RecoVertex.BeamSpotProducer.BeamSpot_cfi import *
from RecoVertex.BeamSpotProducer.offlineBeamSpotToCUDA_cfi import offlineBeamSpotToCUDA

offlineBeamSpotTask = cms.Task(offlineBeamSpot)

from Configuration.ProcessModifiers.gpu_cff import gpu
from Configuration.ProcessModifiers.gpuTracks_cff import gpuTracks

_offlineBeamSpotTask_gpu = offlineBeamSpotTask.copy()
_offlineBeamSpotTask_gpu.add(offlineBeamSpotToCUDA)
gpu.toReplaceWith(offlineBeamSpotTask, _offlineBeamSpotTask_gpu)
gpuTracks.toReplaceWith(offlineBeamSpotTask, _offlineBeamSpotTask_gpu)
