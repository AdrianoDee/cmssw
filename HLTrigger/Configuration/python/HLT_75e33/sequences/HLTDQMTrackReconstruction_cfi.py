import FWCore.ParameterSet.Config as cms

from ..modules.hltPixelTracksSoAMonitorCPU_cfi import *
from ..modules.hltPixelTracksSoAMonitorGPU_cfi import *
from ..modules.hltPixelTracksSoACompareGPUvsCPU_cfi import *
from ..modules.hltPixelTrackToTrackSerialSync_cfi import *
from ..modules.hltInitialStepSeedsTrackToTrackSerialSync_cfi import *

# Empty sequence as a placeholder to be filled when alpakaValidationHLT is active
HLTDQMTrackReconstruction = cms.Sequence()

from Configuration.ProcessModifiers.alpakaValidationHLT_cff import alpakaValidationHLT
alpakaValidationHLT.toReplaceWith(HLTDQMTrackReconstruction,
    cms.Sequence(
        hltPixelTracksSoAMonitorCPU +
        hltPixelTracksSoAMonitorGPU +
        hltPixelTracksSoACompareGPUvsCPU +
        hltPixelTrackToTrackSerialSync +
        hltInitialStepSeedsTrackToTrackSerialSync
    )
)

# Two-iteration stub chain: the displaced and the merged selections are compared as well.
hltPixelTracksSoAMonitorCPUDisplaced = hltPixelTracksSoAMonitorCPU.clone(
    pixelTrackSrc = "hltPhase2PixelTrackHighPuritySelectorDisplacedSerialSync",
    topFolderName = "HLT/HeterogeneousMonitoring/PixelTracksCPUDisplaced",
)
hltPixelTracksSoAMonitorGPUDisplaced = hltPixelTracksSoAMonitorGPU.clone(
    pixelTrackSrc = "hltPhase2PixelTrackHighPuritySelectorDisplaced",
    topFolderName = "HLT/HeterogeneousMonitoring/PixelTracksGPUDisplaced",
)
hltPixelTracksSoACompareGPUvsCPUDisplaced = hltPixelTracksSoACompareGPUvsCPU.clone(
    pixelTrackReferenceSoA = "hltPhase2PixelTrackHighPuritySelectorDisplacedSerialSync",
    pixelTrackTargetSoA = "hltPhase2PixelTrackHighPuritySelectorDisplaced",
    topFolderName = "HLT/HeterogeneousComparisons/pixelTracksSoADisplaced",
)
hltPixelTracksSoAMonitorCPUMerged = hltPixelTracksSoAMonitorCPU.clone(
    pixelTrackSrc = "hltPhase2PixelTrackHighPuritySelectorMergedSerialSync",
    topFolderName = "HLT/HeterogeneousMonitoring/PixelTracksCPUMerged",
)
hltPixelTracksSoAMonitorGPUMerged = hltPixelTracksSoAMonitorGPU.clone(
    pixelTrackSrc = "hltPhase2PixelTrackHighPuritySelectorMerged",
    topFolderName = "HLT/HeterogeneousMonitoring/PixelTracksGPUMerged",
)
hltPixelTracksSoACompareGPUvsCPUMerged = hltPixelTracksSoACompareGPUvsCPU.clone(
    pixelTrackReferenceSoA = "hltPhase2PixelTrackHighPuritySelectorMergedSerialSync",
    pixelTrackTargetSoA = "hltPhase2PixelTrackHighPuritySelectorMerged",
    topFolderName = "HLT/HeterogeneousComparisons/pixelTracksSoAMerged",
)

from Configuration.ProcessModifiers.phase2CAStubs_cff import phase2CAStubs
from Configuration.ProcessModifiers.pixelTrackMask_cff import pixelTrackMask
(phase2CAStubs & pixelTrackMask & alpakaValidationHLT).toReplaceWith(HLTDQMTrackReconstruction,
    cms.Sequence(
        hltPixelTracksSoAMonitorCPU +
        hltPixelTracksSoAMonitorGPU +
        hltPixelTracksSoACompareGPUvsCPU +
        hltPixelTracksSoAMonitorCPUDisplaced +
        hltPixelTracksSoAMonitorGPUDisplaced +
        hltPixelTracksSoACompareGPUvsCPUDisplaced +
        hltPixelTracksSoAMonitorCPUMerged +
        hltPixelTracksSoAMonitorGPUMerged +
        hltPixelTracksSoACompareGPUvsCPUMerged +
        hltPixelTrackToTrackSerialSync +
        hltInitialStepSeedsTrackToTrackSerialSync
    )
)
