import FWCore.ParameterSet.Config as cms

hltOTStubProducer = cms.EDProducer(
    "OTStubProducerVectorHitStyle@alpaka",
    otRecHitsSoA=cms.InputTag("hltPixelSeedingOTRecHitsSoA"),
    barrelFlatCut=cms.vdouble(0.0, 0.15, 0.1, 0.1, 0.09, 0.12, 0.2),
    barrelTiltedCut=cms.vdouble(0.0, 0.15, 0.1, 0.1, 0.09, 0.12, 0.2),
    endcapCut=cms.vdouble(0.0, 0.1, 0.1, 0.1, 0.1, 0.1),
    # Per-layer cluster size cuts (999 = disabled)
    #                                     index:  0    1    2    3    4    5    6
    #                                     layer: pad   L1   L2   L3   L4   L5   L6
    barrelFlatMaxClusterSizeDiff=cms.vint32(999, 999, 999, 999, 999, 999, 999),
    barrelTiltedMaxClusterSizeDiff=cms.vint32(999, 999, 999, 999, 999, 999, 999),
    endcapMaxClusterSizeDiff=cms.vint32(999, 999, 999, 999, 999, 999),
    barrelFlatMaxClusterSize=cms.vint32(999, 999, 999, 999, 999, 999, 999),
    barrelTiltedMaxClusterSize=cms.vint32(999, 999, 999, 999, 999, 999, 999),
    endcapMaxClusterSize=cms.vint32(999, 999, 999, 999, 999, 999),
    # Per-layer cluster size sum cuts (999 = disabled)
    barrelFlatMaxClusterSizeSum=cms.vint32(999, 999, 999, 999, 999, 999, 999),
    barrelTiltedMaxClusterSizeSum=cms.vint32(999, 999, 999, 999, 999, 999, 999),
    endcapMaxClusterSizeSum=cms.vint32(999, 999, 999, 999, 999, 999),
    mightGet=cms.optional.untracked.vstring,
    alpaka=cms.untracked.PSet(backend=cms.untracked.string("")),
)

from Configuration.ProcessModifiers.phase2CATrueStubs_cff import phase2CATrueStubs
from .hltTrueStubProducer_cfi import hltTrueStubProducer as _hltOTStubProducerTrue
phase2CATrueStubs.toReplaceWith(hltOTStubProducer, _hltOTStubProducerTrue)
