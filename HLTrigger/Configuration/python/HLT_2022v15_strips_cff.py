from HLTrigger.Configuration.HLT_2022v15_cff import *

from RecoLocalTracker.SiPixelRecHits.siStripRecHitSoAHostPhase1_cfi import siStripRecHitSoAHostPhase1 as _siStripRecHitSoAHostPhase1
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi import siStripMatchedRecHits as siStripMatchedRecHits_

fragment.siStripRecHitSoAHostPhase1 = _siStripRecHitSoAHostPhase1.clone(
    pixelRecHitSoASource = cms.InputTag("hltSiPixelRecHitsFromLegacy")
)

fragment.hltPixelTracksSoA.cpu.pixelRecHitSrc = cms.InputTag("siStripRecHitSoAHostPhase1")

fragment.hltPixelTracks.useStripHits = cms.bool(True)
fragment.hltPixelTracks.hitModuleStartSrc = cms.InputTag("siStripRecHitSoAHostPhase1")

fragment.siStripMatchedRecHits = siStripMatchedRecHits_.clone(ClusterProducer=cms.InputTag('hltSiStripRawToClustersFacility'))

fragment.HLTRecoPixelTracksTask.add(fragment.siStripRecHitSoAHostPhase1)
fragment.HLTRecoPixelTracksTask.add(fragment.siStripMatchedRecHits)

fragment.HLTRecopixelvertexingTask.add(fragment.siStripRecHitSoAHostPhase1)
fragment.HLTRecopixelvertexingTask.add(fragment.siStripMatchedRecHits)

fragment.HLTRecoPixelTracksSequence.insert(0, fragment.HLTDoLocalStripSequence)
fragment.HLTRecopixelvertexingSequence.insert(0, fragment.HLTDoLocalStripSequence)
