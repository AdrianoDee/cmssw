import FWCore.ParameterSet.Config as cms

# CA-ordered module geometry SoA for the Phase2OTStubs chain, shared by both OT-stub CA iterations
# and the merger.
from RecoTracker.PixelSeeding.caGeometryESProducerPhase2OTStubs_cfi import (
    caGeometryESProducerPhase2OTStubs as _caGeometryESProducerPhase2OTStubs,
)

hltCAGeometryESProducerPhase2OTStubs = _caGeometryESProducerPhase2OTStubs.clone(
    # 28 pixel + 26 OT CA layers; must equal the size of the `layers` table of every CA producer
    # sharing this geometry.
    nLayers = 54,
)
