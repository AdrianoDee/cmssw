import FWCore.ParameterSet.Config as cms

from ..modules.hltMeasurementTrackerEvent_cfi import *
from ..modules.hltSiPhase2RecHits_cfi import *

HLTOtLocalRecoSequence = cms.Sequence(hltMeasurementTrackerEvent
                                      +hltSiPhase2RecHits
                                      )

from Configuration.ProcessModifiers.hltPhase2LegacyTracking_cff import hltPhase2LegacyTracking
hltPhase2LegacyTracking.toReplaceWith(HLTOtLocalRecoSequence,
                                      HLTOtLocalRecoSequence.copyAndExclude([hltSiPhase2RecHits])
                                      )

_HLTOtLocalRecoSequenceWithHits = cms.Sequence(hltMeasurementTrackerEvent
                                               +hltSiPhase2RecHits
                                               )

from Configuration.ProcessModifiers.phase2CAExtension_cff import phase2CAExtension
phase2CAExtension.toReplaceWith(HLTOtLocalRecoSequence, _HLTOtLocalRecoSequenceWithHits)

# Stub-based tracking also needs OT RecHits
from Configuration.ProcessModifiers.phase2CAStubs_cff import phase2CAStubs
phase2CAStubs.toReplaceWith(HLTOtLocalRecoSequence, _HLTOtLocalRecoSequenceWithHits)
