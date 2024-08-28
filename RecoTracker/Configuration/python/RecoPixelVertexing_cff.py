import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import *
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

from RecoTracker.PixelTrackFitting.PixelTracks_cff import *
from RecoTracker.PixelVertexFinding.PixelVertexes_cff import *

# legacy pixel vertex reconsruction using the divisive vertex finder
pixelVerticesTask = cms.Task(
    pixelVertices
)

# Phase2 Modifier
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker

# HIon Modifier
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA

## Pixel Vertex Reconstruction with Alpaka

# pixel vertex SoA producer with alpaka on the device
from RecoTracker.PixelVertexFinding.pixelVertexProducerAlpakaPhase1_cfi import pixelVertexProducerAlpakaPhase1 as _pixelVerticesAlpakaPhase1
from RecoTracker.PixelVertexFinding.pixelVertexProducerAlpakaPhase2_cfi import pixelVertexProducerAlpakaPhase2 as _pixelVerticesAlpakaPhase2
from RecoTracker.PixelVertexFinding.pixelVertexProducerAlpakaHIonPhase1_cfi import pixelVertexProducerAlpakaHIonPhase1 as _pixelVerticesAlpakaHIonPhase1
pixelVerticesAlpaka = _pixelVerticesAlpakaPhase1.clone()
phase2_tracker.toReplaceWith(pixelVerticesAlpaka,_pixelVerticesAlpakaPhase2.clone())
(pp_on_AA & ~phase2_tracker).toReplaceWith(pixelVerticesAlpaka,_pixelVerticesAlpakaHIonPhase1.clone(doSplitting = False))

from RecoTracker.PixelVertexFinding.pixelVertexFromSoAAlpaka_cfi import pixelVertexFromSoAAlpaka as _pixelVertexFromSoAAlpaka
alpaka.toReplaceWith(pixelVertices, _pixelVertexFromSoAAlpaka.clone())

# pixel vertex SoA producer with alpaka on the cpu, for validation
pixelVerticesAlpakaSerial = makeSerialClone(pixelVerticesAlpaka,
    pixelTrackSrc = 'pixelTracksAlpakaSerial'
)

alpaka.toReplaceWith(pixelVerticesTask, cms.Task(
    # Build the pixel vertices in SoA format with alpaka on the device
    pixelVerticesAlpaka,
    # Build the pixel vertices in SoA format with alpaka on the cpu (if requested by the validation)
    pixelVerticesAlpakaSerial,
    # Convert the pixel vertices from SoA format (on the host) to the legacy format
    pixelVertices
))

# Tasks and Sequences
recopixelvertexingTask = cms.Task(
    pixelTracksTask,
    pixelVerticesTask
)
recopixelvertexing = cms.Sequence(recopixelvertexingTask)
