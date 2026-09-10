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
from ..modules.hltPhase2PixelVertices_cfi import *
from ..sequences.HLTPhase2PixelVertexingSequence_cfi import *
from ..sequences.HLTBeamSpotSequence_cfi import HLTBeamSpotSequence

HLTPhase2PixelTracksAndVerticesSequence = cms.Sequence(
    HLTBeamSpotSequence
    +hltPhase2PixelFitterByHelixProjections # Currently needed by tracker muons
    +hltPhase2PixelTrackFilterByKinematics  # Currently needed by tracker muons
    +hltPhase2OtRecHitsSoA
    +hltPhase2PixelRecHitsExtendedSoA
    +hltPhase2PixelTracksSoA
    +hltPhase2PixelTrackTorchHighPuritySelector
    +hltPhase2PixelTracks
    +HLTPhase2PixelVertexingSequence
)

# Empty sequence as a placeholder to be filled when alpakaValidationHLT is active
HLTPhase2PixelTracksAndVerticesSequenceSerialSync = cms.Sequence()

hltPhase2PixelTracksSoASerialSync = makeSerialClone(hltPhase2PixelTracksSoA)
hltPhase2PixelTrackTorchHighPuritySelectorSerialSync = makeSerialClone(
    hltPhase2PixelTrackTorchHighPuritySelector.clone(
        pixelTrackSrc = cms.InputTag("hltPhase2PixelTracksSoASerialSync")
    )
)
hltPhase2PixelTracksSerialSync = hltPhase2PixelTracks.clone(
    trackSrc = cms.InputTag("hltPhase2PixelTrackTorchHighPuritySelectorSerialSync")
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

# Stub-based tracking sequence. The SerialSync and vertex-trimming arms are not stub-aware.
from ..modules.hltPixelSeedingOTRecHitsSoA_cfi import hltPixelSeedingOTRecHitsSoA
from ..modules.hltOTStubProducer_cfi import hltOTStubProducer
from ..modules.hltPhase2PixelRecHitsStubsMerger_cfi import hltPhase2PixelRecHitsStubsMerger
from ..modules.hltSiPixelClusters_cfi import hltSiPixelClusters
from ..modules.hltSiPixelRecHits_cfi import hltSiPixelRecHits

from ..modules.hltPhase2PixelTrackHighPtMasking_cfi import hltPhase2PixelTrackHighPtMasking
from ..modules.hltPhase2PixelTracksSoADisplaced_cfi import hltPhase2PixelTracksSoADisplaced
from ..modules.hltPhase2PixelTrackHighPuritySelectorDisplaced_cfi import hltPhase2PixelTrackHighPuritySelectorDisplaced
from ..modules.hltPhase2PixelTracksSoAMerger_cfi import hltPhase2PixelTracksSoAMerger
from ..modules.hltPhase2PixelTrackHighPuritySelectorMerged_cfi import hltPhase2PixelTrackHighPuritySelectorMerged

_HLTPhase2PixelTracksAndVerticesSequenceCAStubs = cms.Sequence(
    HLTBeamSpotSequence
    +hltPhase2PixelTracksAndHighPtStepTrackingRegions # needed by highPtTripletStep iteration
    +hltPhase2PixelFitterByHelixProjections # Currently needed by tracker muons
    +hltPhase2PixelTrackFilterByKinematics  # Currently needed by tracker muons
    +hltSiPixelClusters                     # legacy pixel clusters for the legacy rechits
    +hltSiPixelRecHits                      # legacy pixel rechits for the legacy converter
    +hltPixelSeedingOTRecHitsSoA
    +hltOTStubProducer
    +hltPhase2PixelRecHitsStubsMerger
    +hltPhase2PixelTracksSoA                     # stub CA via the modifier (label preserved)
    +hltPhase2PixelTrackTorchHighPuritySelector  # forest selector via the modifier (label preserved)
    +hltPhase2PixelTracks
    +HLTPhase2PixelVertexingSequence
)

_HLTPhase2PixelTracksAndVerticesSequenceCAStubsTwoIterations = cms.Sequence(
    HLTBeamSpotSequence
    +hltPhase2PixelTracksAndHighPtStepTrackingRegions # needed by highPtTripletStep iteration
    +hltPhase2PixelFitterByHelixProjections # Currently needed by tracker muons
    +hltPhase2PixelTrackFilterByKinematics  # Currently needed by tracker muons
    +hltSiPixelClusters                     # legacy pixel clusters for the legacy rechits
    +hltSiPixelRecHits                      # legacy pixel rechits for the legacy converter
    +hltPixelSeedingOTRecHitsSoA
    +hltOTStubProducer
    +hltPhase2PixelRecHitsStubsMerger
    +hltPhase2PixelTracksSoA                     # prompt stub CA via the modifier (label preserved)
    +hltPhase2PixelTrackTorchHighPuritySelector  # prompt forest selector via the modifier (label preserved)
    +hltPhase2PixelTrackHighPtMasking            # masks the hits of the prompt tracks
    +hltPhase2PixelTracksSoADisplaced            # displaced stub CA on the masked hits
    +hltPhase2PixelTrackHighPuritySelectorDisplaced
    +hltPhase2PixelTracksSoAMerger               # merge + OT extension + refit, once
    +hltPhase2PixelTrackHighPuritySelectorMerged # final high-purity selection, on the merged collection
    +hltPhase2PixelTracks
    +HLTPhase2PixelVertexingSequence
)

from Configuration.ProcessModifiers.phase2CAStubs_cff import phase2CAStubs
from Configuration.ProcessModifiers.pixelTrackMask_cff import pixelTrackMask

(phase2CAStubs & ~pixelTrackMask).toReplaceWith(
    HLTPhase2PixelTracksAndVerticesSequence,
    _HLTPhase2PixelTracksAndVerticesSequenceCAStubs
)

# Two-iteration arm: the OT-hit attach extension, the final refit and the duplicate removal run once
# in the merger, over the high-purity tracks of both iterations. The legacy converter is pointed at
# the merged collection, so every consumer sees it.
(phase2CAStubs & pixelTrackMask).toReplaceWith(
    HLTPhase2PixelTracksAndVerticesSequence,
    _HLTPhase2PixelTracksAndVerticesSequenceCAStubsTwoIterations,
)

# SerialSync arms of the stub chain for the CPU vs. GPU validation, to be kept in sync with the two
# sequences above. The stub formation and the pixel/stub merger stay accelerated and are shared: the
# twins read their host copies, as the default arm reads the accelerated rechits.
hltPhase2PixelTrackHighPtMaskingSerialSync = makeSerialClone(hltPhase2PixelTrackHighPtMasking,
    tracksSoASrc = "hltPhase2PixelTrackTorchHighPuritySelectorSerialSync",
)
hltPhase2PixelTracksSoADisplacedSerialSync = makeSerialClone(hltPhase2PixelTracksSoADisplaced,
    hitMask = "hltPhase2PixelTrackHighPtMaskingSerialSync",
)
hltPhase2PixelTrackHighPuritySelectorDisplacedSerialSync = makeSerialClone(hltPhase2PixelTrackHighPuritySelectorDisplaced,
    pixelTrackSrc = "hltPhase2PixelTracksSoADisplacedSerialSync",
)
hltPhase2PixelTracksSoAMergerSerialSync = makeSerialClone(hltPhase2PixelTracksSoAMerger,
    inputTkSoAs = cms.VInputTag("hltPhase2PixelTrackTorchHighPuritySelectorSerialSync",
                                "hltPhase2PixelTrackHighPuritySelectorDisplacedSerialSync"),
)
hltPhase2PixelTrackHighPuritySelectorMergedSerialSync = makeSerialClone(hltPhase2PixelTrackHighPuritySelectorMerged,
    pixelTrackSrc = "hltPhase2PixelTracksSoAMergerSerialSync",
)
# The legacy converter of the two-iteration chain reads the final selection.
(phase2CAStubs & pixelTrackMask).toModify(hltPhase2PixelTracksSerialSync,
    trackSrc = "hltPhase2PixelTrackHighPuritySelectorMergedSerialSync",
)

_HLTPhase2PixelTracksAndVerticesSequenceCAStubsSerialSync = cms.Sequence(
    HLTBeamSpotSequence
    +hltPhase2PixelTracksAndHighPtStepTrackingRegions
    +hltPhase2PixelFitterByHelixProjections
    +hltPhase2PixelTrackFilterByKinematics
    +hltSiPixelClusters
    +hltSiPixelRecHits
    +hltPixelSeedingOTRecHitsSoA
    +hltOTStubProducer
    +hltPhase2PixelRecHitsStubsMerger
    +hltPhase2PixelTracksSoASerialSync
    +hltPhase2PixelTrackTorchHighPuritySelectorSerialSync
    +hltPhase2PixelTracksSerialSync
    +HLTPhase2PixelVertexingSequenceSerialSync
)

_HLTPhase2PixelTracksAndVerticesSequenceCAStubsTwoIterationsSerialSync = cms.Sequence(
    HLTBeamSpotSequence
    +hltPhase2PixelTracksAndHighPtStepTrackingRegions
    +hltPhase2PixelFitterByHelixProjections
    +hltPhase2PixelTrackFilterByKinematics
    +hltSiPixelClusters
    +hltSiPixelRecHits
    +hltPixelSeedingOTRecHitsSoA
    +hltOTStubProducer
    +hltPhase2PixelRecHitsStubsMerger
    +hltPhase2PixelTracksSoASerialSync
    +hltPhase2PixelTrackTorchHighPuritySelectorSerialSync
    +hltPhase2PixelTrackHighPtMaskingSerialSync
    +hltPhase2PixelTracksSoADisplacedSerialSync
    +hltPhase2PixelTrackHighPuritySelectorDisplacedSerialSync
    +hltPhase2PixelTracksSoAMergerSerialSync
    +hltPhase2PixelTrackHighPuritySelectorMergedSerialSync
    +hltPhase2PixelTracksSerialSync
    +HLTPhase2PixelVertexingSequenceSerialSync
)

(phase2CAStubs & ~pixelTrackMask & alpakaValidationHLT).toReplaceWith(
    HLTPhase2PixelTracksAndVerticesSequenceSerialSync,
    _HLTPhase2PixelTracksAndVerticesSequenceCAStubsSerialSync
)
(phase2CAStubs & pixelTrackMask & alpakaValidationHLT).toReplaceWith(
    HLTPhase2PixelTracksAndVerticesSequenceSerialSync,
    _HLTPhase2PixelTracksAndVerticesSequenceCAStubsTwoIterationsSerialSync
)
