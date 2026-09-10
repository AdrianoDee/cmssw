import FWCore.ParameterSet.Config as cms
from .hltPhase2PixelTracksSoA_cfi import hltPhase2PixelTracksSoA as _hltPhase2PixelTracksSoA

# Displaced iteration under the label the two-iteration sequence schedules: the prompt CA producer
# reading the hit mask, replaced by the stub version with phase2CAStubs. The sequence runs only in
# the stub chain, where the mask is built on the stub-merged rechits this version reads.
hltPhase2PixelTracksSoADisplaced = _hltPhase2PixelTracksSoA.clone(
    iterationName = cms.string('displaced'),
    hitMask       = cms.InputTag('hltPhase2PixelTrackHighPtMasking'),
)

from Configuration.ProcessModifiers.phase2CAStubs_cff import phase2CAStubs
from .hltPhase2PixelTracksSoADisplacedWithStubs_cfi import (
    hltPhase2PixelTracksSoADisplacedWithStubs as _hltPhase2PixelTracksSoADisplacedWithStubs,
)
phase2CAStubs.toReplaceWith(hltPhase2PixelTracksSoADisplaced,
                            _hltPhase2PixelTracksSoADisplacedWithStubs)
