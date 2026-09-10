import FWCore.ParameterSet.Config as cms

from Configuration.ProcessModifiers.phase2CAStubs_cff import phase2CAStubs
from Configuration.ProcessModifiers.pixelTrackMask_cff import pixelTrackMask


class _ScoutingChain(cms.ModifierChain):
    """A modifier chain that also acts as the modifier of the scouting menu's own customisations: toModify,
    toReplaceWith, makeProcessModifier and the boolean operators go to its own modifier, which is chosen
    when the chain is."""

    def __init__(self, own, *chained):
        super().__init__(own, *chained)
        self._own = own

    def toModify(self, *args, **kwargs):
        return self._own.toModify(*args, **kwargs)

    def toReplaceWith(self, *args, **kwargs):
        return self._own.toReplaceWith(*args, **kwargs)

    def makeProcessModifier(self, *args, **kwargs):
        return self._own.makeProcessModifier(*args, **kwargs)

    def __and__(self, other):
        return self._own & other

    def __or__(self, other):
        return self._own | other

    def __invert__(self):
        return ~self._own


_ngtScoutingOwn = cms.Modifier()

# NGT scouting menu: the stub-seeded pixel CA with both iterations, the pixel tracks as general tracks.
ngtScouting = _ScoutingChain(_ngtScoutingOwn, phase2CAStubs, pixelTrackMask)

# Single-iteration variant: prompt iteration only, no hit masking.
ngtScoutingSingleIter = _ScoutingChain(_ngtScoutingOwn, phase2CAStubs)
