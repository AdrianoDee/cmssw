import FWCore.ParameterSet.Config as cms

otStubProducerVectorHitStyle = cms.EDProducer('OTStubProducerVectorHitStyle@alpaka',
    otRecHitsSoA = cms.InputTag("pixelSeedingOTRecHitsSoAConverter"),
    maxWidthBarrelFlat = cms.vdouble(0.0, 0.05, 0.06, 0.08, 0.09, 0.12, 0.2),
    maxWidthBarrelTilted = cms.vdouble(0.0, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15),
    maxWidthEndcap = cms.vdouble(0.0, 0.1, 0.1, 0.1, 0.1, 0.1),
    # Per-layer cluster size cuts (999 = disabled)
    #                                     index:  0    1    2    3    4    5    6
    #                                     layer: pad   L1   L2   L3   L4   L5   L6
    maxClusterSizeDiffBarrelFlat = cms.vint32(999, 3, 2, 2, 5, 5, 6),
    maxClusterSizeDiffBarrelTilted = cms.vint32(999, 2, 2, 2, 999, 999, 999),
    maxClusterSizeDiffEndcap = cms.vint32(999, 2, 2, 2, 2, 2),
    maxClusterSizeBarrelFlat = cms.vint32(999, 8, 8, 8, 11, 11, 13),
    maxClusterSizeBarrelTilted = cms.vint32(999, 8, 7, 7, 999, 999, 999),
    maxClusterSizeEndcap = cms.vint32(999, 7, 7, 7, 6, 6),
    # Per-layer cluster size sum cuts (999 = disabled)
    maxClusterSizeBarrelFlatSum = cms.vint32(999, 6, 7, 7, 10, 11, 12),
    maxClusterSizeSumBarrelTilted = cms.vint32(999, 6, 6, 6, 999, 999, 999),
    maxClusterSizeSumEndcap = cms.vint32(999, 6, 6, 6, 6, 6),
)
