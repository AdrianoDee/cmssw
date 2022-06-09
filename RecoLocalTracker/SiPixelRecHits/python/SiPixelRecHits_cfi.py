import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA
from Configuration.ProcessModifiers.gpu_cff import gpu

# legacy pixel rechit producer
siPixelRecHits = cms.EDProducer("SiPixelRecHitConverter",
    src = cms.InputTag("siPixelClusters"),
    CPE = cms.string('PixelCPEGeneric'),
    VerboseLevel = cms.untracked.int32(0)
)

from Configuration.Eras.Modifier_phase2_brickedPixels_cff import phase2_brickedPixels
phase2_brickedPixels.toModify(siPixelRecHits,
                              CPE = 'PixelCPEGenericForBricked'
)

# SwitchProducer wrapping the legacy pixel rechit producer
siPixelRecHitsPreSplitting = SwitchProducerCUDA(
    cpu = siPixelRecHits.clone(
        src = 'siPixelClustersPreSplitting'
    )
)

# convert the pixel rechits from legacy to SoA format
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitSoAFromLegacy_cfi import siPixelRecHitSoAFromLegacy as _siPixelRecHitsPreSplittingSoA
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitSoAFromCUDA_cfi import siPixelRecHitSoAFromCUDA as _siPixelRecHitSoAFromCUDA
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitSoAFromLegacyPhase2_cfi import siPixelRecHitSoAFromLegacyPhase2 as _siPixelRecHitsPreSplittingSoAPhase2

siPixelRecHitsPreSplittingCPU = _siPixelRecHitsPreSplittingSoA.clone(convertToLegacy=True)

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toReplaceWith(siPixelRecHitsPreSplittingCPU, _siPixelRecHitsPreSplittingSoAPhase2.clone(convertToLegacy=True, CPE = cms.string('PixelCPEFastPhase2')))

# modifier used to prompt patatrack pixel tracks reconstruction on cpu
from Configuration.ProcessModifiers.pixelNtupletFit_cff import pixelNtupletFit
pixelNtupletFit.toModify(siPixelRecHitsPreSplitting,
    cpu = _siPixelRecHitsPreSplittingSoA.clone(convertToLegacy=True)
)

siPixelRecHitsPreSplittingTask = cms.Task(
    # SwitchProducer wrapping the legacy pixel rechit producer or the cpu SoA producer
    siPixelRecHitsPreSplitting
)

# reconstruct the pixel rechits on the gpu
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitCUDA_cfi import siPixelRecHitCUDA as _siPixelRecHitCUDA
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitCUDAPhase2_cfi import siPixelRecHitCUDAPhase2 as _siPixelRecHitCUDAPhase2
siPixelRecHitsPreSplittingCUDA = _siPixelRecHitCUDA.clone(
    beamSpot = "offlineBeamSpotToCUDA"
)
phase2_tracker.toReplaceWith(siPixelRecHitsPreSplittingCUDA,_siPixelRecHitCUDAPhase2.clone(
    beamSpot = "offlineBeamSpotToCUDA"
))


# transfer the pixel rechits to the host and convert them from SoA
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitFromCUDA_cfi import siPixelRecHitFromCUDA as _siPixelRecHitFromCUDA

#this is an alias for the SoA on GPU or CPU to be used for DQM
siPixelRecHitsPreSplittingSoA = SwitchProducerCUDA(
    cpu = cms.EDAlias(
            siPixelRecHitsPreSplittingCPU = cms.VPSet(
                 cms.PSet(type = cms.string("cmscudacompatCPUTraitspixelTopologyPhase1TrackingRecHit2DCPUBaseT")),
                 cms.PSet(type = cms.string("uintAsHostProduct"))
             )),
    cuda = _siPixelRecHitSoAFromCUDA.clone()
)

(gpu & pixelNtupletFit).toModify(siPixelRecHitsPreSplitting,
    cpu = cms.EDAlias(
            siPixelRecHitsPreSplittingCPU = cms.VPSet(
                 cms.PSet(type = cms.string("SiPixelRecHitedmNewDetSetVector")),
                 cms.PSet(type = cms.string("uintAsHostProduct"))
             )
         ),
    cuda = _siPixelRecHitFromCUDA.clone())

from RecoLocalTracker.SiPixelRecHits.siPixelRecHitSoAFromCUDAPhase2_cfi import siPixelRecHitSoAFromCUDAPhase2 as _siPixelRecHitSoAFromCUDAPhase2
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitFromCUDAPhase2_cfi import siPixelRecHitFromCUDAPhase2 as _siPixelRecHitFromCUDAPhase2
(gpu & pixelNtupletFit & phase2_tracker).toReplaceWith(siPixelRecHitsPreSplittingSoA.cuda, _siPixelRecHitSoAFromCUDAPhase2.clone())
(gpu & pixelNtupletFit & phase2_tracker).toReplaceWith(siPixelRecHitsPreSplittingSoA.cpu,
cms.EDAlias(
        siPixelRecHitsPreSplittingCPU = cms.VPSet(
             cms.PSet(type = cms.string("cmscudacompatCPUTraitspixelTopologyPhase2TrackingRecHit2DCPUBaseT")),
             cms.PSet(type = cms.string("uintAsHostProduct"))
         )))
(gpu & pixelNtupletFit & phase2_tracker).toReplaceWith(siPixelRecHitsPreSplitting.cuda, _siPixelRecHitFromCUDAPhase2.clone())


pixelNtupletFit.toReplaceWith(siPixelRecHitsPreSplittingTask, cms.Task(
     cms.Task(
         # reconstruct the pixel rechits on the cpu
         siPixelRecHitsPreSplittingCPU,
         # SwitchProducer wrapping an EDAlias on cpu or the converter from SoA to legacy on gpu
         siPixelRecHitsPreSplitting,
         # producing and converting on cpu (if needed)
         siPixelRecHitsPreSplittingSoA)
         )
         )

(gpu & pixelNtupletFit).toReplaceWith(siPixelRecHitsPreSplittingTask, cms.Task(
    # reconstruct the pixel rechits on the gpu or on the cpu
    # (normally only one of the two is run because only one is consumed from later stages)
    siPixelRecHitsPreSplittingCUDA,
    siPixelRecHitsPreSplittingCPU,
    # SwitchProducer wrapping an EDAlias on cpu or the converter from SoA to legacy on gpu
    siPixelRecHitsPreSplitting,
    # producing and converting on cpu (if needed)
    siPixelRecHitsPreSplittingSoA
))
