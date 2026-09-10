import FWCore.ParameterSet.Config as cms

from ..modules.hltHGCalRecHit_cfi import hltHGCalRecHit
from ..modules.hltHGCalUncalibRecHit_cfi import hltHGCalUncalibRecHit
from ..modules.hltHgcalDigis_cfi import hltHgcalDigis
from ..modules.hltHgcalSoALayerClustersProducer_cfi import hltHgcalSoALayerClustersProducer
from ..modules.hltHgcalSoARecHitsLayerClustersProducer_cfi import hltHgcalSoARecHitsLayerClustersProducer
from ..modules.hltHgcalSoARecHitsProducer_cfi import hltHgcalSoARecHitsProducer
from ..modules.hltInputLST_cfi import hltInputLST
from ..modules.hltInitialStepSeeds_cfi import hltInitialStepSeeds
from ..modules.hltInitialStepTrajectorySeedsLST_cfi import hltInitialStepTrajectorySeedsLST
from ..modules.hltL1GTAcceptFilter_cfi import hltL1GTAcceptFilter
from ..modules.hltLST_cfi import hltLST
from ..modules.hltPhase2OtRecHitsSoA_cfi import hltPhase2OtRecHitsSoA
from ..modules.hltPhase2PixelRecHitsExtendedSoA_cfi import hltPhase2PixelRecHitsExtendedSoA
from ..modules.hltPhase2PixelTracks_cfi import hltPhase2PixelTracks
from ..modules.hltPhase2PixelTracksSoA_cfi import hltPhase2PixelTracksSoA
from ..modules.hltPhase2PixelTrackTorchHighPuritySelector_cfi import hltPhase2PixelTrackTorchHighPuritySelector
from ..modules.hltPhase2PixelVertices_cfi import hltPhase2PixelVertices
#from ..modules.hltPhase2PixelVerticesSoA_cfi import hltPhase2PixelVerticesSoA
from ..modules.hltPhase2SiPixelClustersSoA_cfi import hltPhase2SiPixelClustersSoA
from ..modules.hltPhase2SiPixelRecHitsSoA_cfi import hltPhase2SiPixelRecHitsSoA
from ..modules.hltSiPixelClusters_cfi import hltSiPixelClusters
from ..modules.hltSiPixelRecHits_cfi import hltSiPixelRecHits
from ..modules.hltSiPhase2Clusters_cfi import hltSiPhase2Clusters
from ..modules.hltSiPhase2RecHits_cfi import hltSiPhase2RecHits
from ..modules.hltPhase2PixelTrackHighPtMasking_cfi import hltPhase2PixelTrackHighPtMasking
from ..modules.hltPhase2PixelTracksSoADisplaced_cfi import hltPhase2PixelTracksSoADisplaced
from ..modules.hltPhase2PixelTrackHighPuritySelectorDisplaced_cfi import hltPhase2PixelTrackHighPuritySelectorDisplaced
from ..modules.hltPhase2PixelTracksSoAMerger_cfi import hltPhase2PixelTracksSoAMerger
from ..modules.hltPhase2PixelTrackHighPuritySelectorMerged_cfi import hltPhase2PixelTrackHighPuritySelectorMerged
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

#hltExtendedPhase2PixelVerticesSoA = hltPhase2PixelVerticesSoA.clone(pixelTrackSrc = 'hltExtendedPhase2PixelTracksSoA')

# This path runs the heterogeneous modules the menu schedules under the process modifiers in
# effect; the stub-based local reconstruction sits next to the pixel+OT rechit SoA path.
from ..modules.hltPixelSeedingOTRecHitsSoA_cfi import hltPixelSeedingOTRecHitsSoA
from ..modules.hltOTStubProducer_cfi import hltOTStubProducer
from ..modules.hltPhase2PixelRecHitsStubsMerger_cfi import hltPhase2PixelRecHitsStubsMerger

HLTLocalTrackerSequence = cms.Sequence(
    hltPhase2SiPixelClustersSoA
    + hltPhase2SiPixelRecHitsSoA
    + hltSiPhase2Clusters
    + hltSiPhase2RecHits
    + hltPhase2OtRecHitsSoA
    + hltPhase2PixelRecHitsExtendedSoA
    + hltPixelSeedingOTRecHitsSoA
    + hltOTStubProducer
    + hltPhase2PixelRecHitsStubsMerger
    + hltSiPixelClusters
    + hltSiPixelRecHits
)

# Second pixel-track iteration on the hits the first one left unused, plus the merger of the two
# iterations and its final selector. Like the menu, it runs only in the two-iteration stub chain:
# the mask and the merger read the stub-merged rechits, and the legacy converter then reads the
# final selection, so it follows this sequence.
HLTPixelTrackingSecondIterationSequence = cms.Sequence()

from Configuration.ProcessModifiers.phase2CAStubs_cff import phase2CAStubs
from Configuration.ProcessModifiers.pixelTrackMask_cff import pixelTrackMask
(phase2CAStubs & pixelTrackMask).toReplaceWith(
    HLTPixelTrackingSecondIterationSequence,
    cms.Sequence(
        hltPhase2PixelTrackHighPtMasking
        + hltPhase2PixelTracksSoADisplaced
        + hltPhase2PixelTrackHighPuritySelectorDisplaced
        + hltPhase2PixelTracksSoAMerger
        + hltPhase2PixelTrackHighPuritySelectorMerged
    ),
)

HLTPixelTrackingSequence = cms.Sequence(
    hltPhase2PixelTracksSoA
    + hltPhase2PixelTrackTorchHighPuritySelector
    + HLTPixelTrackingSecondIterationSequence
    + hltPhase2PixelTracks
    #+ hltExtendedPhase2PixelVerticesSoA # not yet ready
)

HLTLSTSequence = cms.Sequence(
    hltInitialStepSeeds
    + hltInputLST
    + hltLST
)

HLTHeterogeneousHGCalRecoSequence = cms.Sequence(
    hltHgcalDigis
    + hltHGCalUncalibRecHit
    + hltHGCalRecHit
    + hltHgcalSoARecHitsProducer
    + hltHgcalSoARecHitsLayerClustersProducer
    + hltHgcalSoALayerClustersProducer
)

DST_HeterogeneousReco = cms.Path(
    HLTBeginSequence
    + hltL1GTAcceptFilter
    + HLTLocalTrackerSequence
    + HLTPixelTrackingSequence
    + HLTLSTSequence
    + HLTHeterogeneousHGCalRecoSequence
    + HLTEndSequence
)
