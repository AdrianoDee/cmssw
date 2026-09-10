# Abstract all early deletion settings here

import collections

import FWCore.ParameterSet.Config as cms

from RecoTracker.Configuration.customiseEarlyDeleteForSeeding import customiseEarlyDeleteForSeeding
from RecoTracker.Configuration.customiseEarlyDeleteForMkFit import customiseEarlyDeleteForMkFit
from RecoTracker.Configuration.customiseEarlyDeleteForCKF import customiseEarlyDeleteForCKF
from RecoTracker.Configuration.customiseEarlyDeleteForPixelClusterSoA import customiseEarlyDeleteForPixelClusterSoA
from RecoTracker.Configuration.customiseEarlyDeleteForPixelTrackSelectionSoA import customiseEarlyDeleteForPixelTrackSelectionSoA
from RecoTracker.Configuration.customiseEarlyDeleteForPixelTrackSoA import customiseEarlyDeleteForPixelTrackSoA
from CommonTools.ParticleFlow.Isolation.customiseEarlyDeleteForCandIsoDeposits import customiseEarlyDeleteForCandIsoDeposits

def customiseEarlyDelete(process):
    # Mapping label -> [branches]
    # for the producers whose products are to be deleted early
    products = collections.defaultdict(list)

    (products, references) = customiseEarlyDeleteForSeeding(process, products)
    (products, newReferences) = customiseEarlyDeleteForMkFit(process, products)
    references.update(newReferences)
    (products, newReferences) = customiseEarlyDeleteForCKF(process, products)
    references.update(newReferences)
    (products, newReferences) = customiseEarlyDeleteForPixelClusterSoA(process, products)
    references.update(newReferences)
    (products, newReferences) = customiseEarlyDeleteForPixelTrackSelectionSoA(process, products)
    references.update(newReferences)
    (products, newReferences) = customiseEarlyDeleteForPixelTrackSoA(process, products)
    references.update(newReferences)

    products = customiseEarlyDeleteForCandIsoDeposits(process, products)

    branchSet = set()
    for branches in products.values():
        for branch in branches:
            branchSet.add(branch)
    branchList = sorted(branchSet)
    process.options.canDeleteEarly.extend(branchList)

    for prod, refs in references.items():
        process.options.holdsReferencesToDeleteEarly.append(cms.PSet(product=cms.string(prod), references=cms.vstring(refs)))

    # LogErrorHarvester should not wait for deleted items
    for prod in process.producers_().values():
        if prod.type_() == "LogErrorHarvester":
            if not hasattr(prod,'excludeModules'):
                prod.excludeModules = cms.untracked.vstring()
            t = prod.excludeModules.value()
            t.extend([b.split('_')[1] for b in branchList])
            prod.excludeModules = t

    return process
