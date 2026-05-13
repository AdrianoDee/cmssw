import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import makeSerialClone

from ..modules.hltPhase2OtRecHitsSoA_cfi import hltPhase2OtRecHitsSoA
from ..modules.hltPhase2PixelFitterByHelixProjections_cfi import hltPhase2PixelFitterByHelixProjections
from ..modules.hltPhase2PixelRecHitsExtendedSoA_cfi import hltPhase2PixelRecHitsExtendedSoA
from ..modules.hltPhase2PixelTrackFilterByKinematics_cfi import hltPhase2PixelTrackFilterByKinematics
from ..modules.hltPhase2PixelTracks_cfi import hltPhase2PixelTracks
from ..modules.hltPhase2PixelTracksAndHighPtStepTrackingRegions_cfi import hltPhase2PixelTracksAndHighPtStepTrackingRegions
from ..modules.hltPhase2PixelTracksHitDoublets_cfi import hltPhase2PixelTracksHitDoublets
from ..modules.hltPhase2PixelTracksHitSeeds_cfi import hltPhase2PixelTracksHitSeeds
from ..modules.hltPhase2PixelTracksSeedLayers_cfi import hltPhase2PixelTracksSeedLayers
from ..modules.hltPhase2PixelTracksSoA_cfi import hltPhase2PixelTracksSoA
from ..modules.hltPhase2PixelTrackTorchHighPuritySelector_cfi import hltPhase2PixelTrackTorchHighPuritySelector
from ..modules.hltPhase2PixelTrackSoATableProducer_cfi import hltPhase2PixelTrackSoATableProducer
from ..modules.hltPhase2PixelVertices_cfi import *
from ..sequences.HLTPhase2PixelVertexingSequence_cfi import *
from ..sequences.HLTBeamSpotSequence_cfi import HLTBeamSpotSequence

HLTPhase2PixelTracksAndVerticesSequence = cms.Sequence(
    HLTBeamSpotSequence
    +hltPhase2PixelTracksAndHighPtStepTrackingRegions # needed by highPtTripletStep iteration
    +hltPhase2PixelFitterByHelixProjections # needed by tracker muons
    +hltPhase2PixelTrackFilterByKinematics  # needed by tracker muons
    +hltPhase2OtRecHitsSoA
    +hltPhase2PixelRecHitsExtendedSoA
    +hltPhase2PixelTracksSoA
    +hltPhase2PixelTrackTorchHighPuritySelector
    #+hltPhase2PixelTrackSoATableProducer
    +hltPhase2PixelTracks
    +HLTPhase2PixelVertexingSequence
)

# Empty sequence as a placeholder to be filled when alpakaValidationHLT is active
HLTPhase2PixelTracksAndVerticesSequenceSerialSync = cms.Sequence()

hltPhase2PixelTracksSoASerialSync = makeSerialClone(hltPhase2PixelTracksSoA)
hltPhase2PixelTrackTorchHighPuritySelectorSerialSync = makeSerialClone(
    hltPhase2PixelTrackTorchHighPuritySelector.clone(
        pixelTrackSrc = "hltPhase2PixelTracksSoASerialSync"
    )
)
hltPhase2PixelTracksSerialSync = hltPhase2PixelTracks.clone(
    trackSrc = "hltPhase2PixelTrackTorchHighPuritySelectorSerialSync"
)

# Sequence for CPU vs. GPU validation, to be kept in sync with default sequence
from Configuration.ProcessModifiers.alpakaValidationHLT_cff import alpakaValidationHLT
alpakaValidationHLT.toReplaceWith(HLTPhase2PixelTracksAndVerticesSequenceSerialSync,
    cms.Sequence(
        HLTBeamSpotSequence
        +hltPhase2PixelTracksAndHighPtStepTrackingRegions # needed by highPtTripletStep iteration
        +hltPhase2PixelFitterByHelixProjections # needed by tracker muons
        +hltPhase2PixelTrackFilterByKinematics  # needed by tracker muons
        +hltPhase2OtRecHitsSoA
        +hltPhase2PixelRecHitsExtendedSoA
        +hltPhase2PixelTracksSoASerialSync
        +hltPhase2PixelTrackTorchHighPuritySelectorSerialSync
        +hltPhase2PixelTracksSerialSync
        +HLTPhase2PixelVertexingSequenceSerialSync
    )
)


from ..modules.hltPhase2TrimmedPixelVertices_cfi import hltPhase2TrimmedPixelVertices
_HLTPhase2PixelTracksAndVerticesSequenceTrimming = cms.Sequence(
    HLTBeamSpotSequence
    +hltPhase2PixelTracksAndHighPtStepTrackingRegions
    +hltPhase2PixelFitterByHelixProjections
    +hltPhase2PixelTrackFilterByKinematics
    +hltPhase2OtRecHitsSoA
    +hltPhase2PixelRecHitsExtendedSoA
    +hltPhase2PixelTracksSoA
    +hltPhase2PixelTracks
    +HLTPhase2PixelVertexingSequence
    +hltPhase2TrimmedPixelVertices
)

from Configuration.ProcessModifiers.phase2_hlt_vertexTrimming_cff import phase2_hlt_vertexTrimming
phase2_hlt_vertexTrimming.toReplaceWith(
    HLTPhase2PixelTracksAndVerticesSequence,
    _HLTPhase2PixelTracksAndVerticesSequenceTrimming
)

from Configuration.ProcessModifiers.hltPhase2LegacyTracking_cff import hltPhase2LegacyTracking
_HLTPhase2PixelTracksAndVerticesSequenceLegacy = cms.Sequence(
    hltPhase2PixelTracksSeedLayers
    +hltPhase2PixelTracksAndHighPtStepTrackingRegions
    +hltPhase2PixelTracksHitDoublets
    +hltPhase2PixelTracksHitSeeds
    +hltPhase2PixelFitterByHelixProjections
    +hltPhase2PixelTrackFilterByKinematics
    +hltPhase2PixelTracks
    +HLTPhase2PixelVertexingSequence
)
hltPhase2LegacyTracking.toReplaceWith(HLTPhase2PixelTracksAndVerticesSequence, _HLTPhase2PixelTracksAndVerticesSequenceLegacy)

from Configuration.ProcessModifiers.hltPhase2LegacyTrackingPatatrackQuadsChain_cff import hltPhase2LegacyTrackingPatatrackQuads
_HLTPhase2PixelTracksAndVerticesSequenceLegacyPatatrack = cms.Sequence(
    HLTBeamSpotSequence
    +hltPhase2PixelTracksAndHighPtStepTrackingRegions
    +hltPhase2PixelFitterByHelixProjections
    +hltPhase2PixelTrackFilterByKinematics
    +hltPhase2PixelTracksSoA
    +hltPhase2PixelTracks
    +HLTPhase2PixelVertexingSequence
)
(hltPhase2LegacyTracking & hltPhase2LegacyTrackingPatatrackQuads).toReplaceWith(
    HLTPhase2PixelTracksAndVerticesSequence,
    _HLTPhase2PixelTracksAndVerticesSequenceLegacyPatatrack
)



# Stub-based tracking sequence with OT stubs
from ..modules.hltPixelSeedingOTRecHitsSoA_cfi import hltPixelSeedingOTRecHitsSoA
from ..modules.hltOTStubProducer_cfi import hltOTStubProducer
from ..modules.hltPhase2PixelRecHitsStubsMerger_cfi import (
    hltPhase2PixelRecHitsStubsMerger,
)
from ..modules.hltSiPixelClusters_cfi import hltSiPixelClusters
from ..modules.hltSiPixelRecHits_cfi import hltSiPixelRecHits

from ..modules.hltPhase2PixelTrackHighPtMasking_cfi import hltPhase2PixelTrackHighPtMasking
from ..modules.hltPhase2PixelTracksSoALowPt_cfi import hltPhase2PixelTracksSoALowPt
from ..modules.hltPhase2PixelTrackTorchHighPuritySelectorLowPt_cfi import hltPhase2PixelTrackTorchHighPuritySelectorLowPt
from ..modules.hltPhase2PixelTracksSoAMerger_cfi import hltPhase2PixelTracksSoAMerger

_HLTPhase2PixelTracksAndVerticesSequenceCAStubs = cms.Sequence(
    HLTBeamSpotSequence
    + hltPhase2PixelTracksAndHighPtStepTrackingRegions  # needed by highPtTripletStep iteration
    + hltPhase2PixelFitterByHelixProjections  # needed by tracker muons
    + hltPhase2PixelTrackFilterByKinematics  # needed by tracker muons
    + hltSiPixelClusters  # Legacy pixel clusters for RecHits
    + hltSiPixelRecHits  # Legacy pixel RecHits for legacy track converter
    + hltPixelSeedingOTRecHitsSoA
    + hltOTStubProducer  # VectorHitStyle via modifier
    + hltPhase2PixelRecHitsStubsMerger
    + hltPhase2PixelTracksSoA  # Stub CA via modifier (label preserved)
    + hltPhase2PixelTrackTorchHighPuritySelector
    # USED FOR MASKING
    + hltPhase2PixelTrackHighPtMasking
    + hltPhase2PixelTracksSoALowPt
    + hltPhase2PixelTrackTorchHighPuritySelectorLowPt
    + hltPhase2PixelTracksSoAMerger
    # USED FOR MASKING
    # +hltPhase2PixelTrackSoATableProducer
    + hltPhase2PixelTracks
    + HLTPhase2PixelVertexingSequence # Vertexing from CAExtension tracks
)

from Configuration.ProcessModifiers.phase2CAStubs_cff import phase2CAStubs

phase2CAStubs.toReplaceWith(
    HLTPhase2PixelTracksAndVerticesSequence,
    _HLTPhase2PixelTracksAndVerticesSequenceCAStubs,
)