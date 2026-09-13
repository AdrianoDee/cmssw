import FWCore.ParameterSet.Config as cms

from RecoVertex.BeamSpotProducer.BeamSpot_cfi import *
from RecoVertex.BeamSpotProducer.beamSpotDeviceProducer_cfi import beamSpotDeviceProducer as _beamSpotDeviceProducer

offlineBeamSpotTask = cms.Task(offlineBeamSpot)

from Configuration.ProcessModifiers.alpaka_cff import alpaka
from Configuration.ProcessModifiers.trackingGPUOffline_cff import trackingGPUOffline

_offlineBeamSpotTask_alpaka = offlineBeamSpotTask.copy()
offlineBeamSpotDevice = _beamSpotDeviceProducer.clone(src = cms.InputTag('offlineBeamSpot'))
_offlineBeamSpotTask_alpaka.add(offlineBeamSpotDevice)
(alpaka | trackingGPUOffline).toReplaceWith(offlineBeamSpotTask, _offlineBeamSpotTask_alpaka)
