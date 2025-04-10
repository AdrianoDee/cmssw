import FWCore.ParameterSet.Config as cms

from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTDiphoton3023IsoCaloIdL1SeededSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

HLT_Diphoton30_23_IsoCaloId_L1Seeded = cms.Path(
    HLTBeginSequence +
    HLTDiphoton3023IsoCaloIdL1SeededSequence +
    HLTEndSequence
)
