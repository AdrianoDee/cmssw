import collections

import FWCore.ParameterSet.Config as cms

# Early deletion of the device SoA of the Phase-2 pixel-track high-purity selection: no alpaka module reads it,
# the legacy converter (and the SoA monitors of the CPU vs. GPU validation) read its host copy, so it is dead on the
# device once written. A few MiB per event and stream (the selection is allocated at capacity).
#
# Releasing a device product early is safe only if no reader can still be reading it when the caching allocator
# hands its block to a later allocation on the producer's queue; with no device reader at all that condition is
# empty. The host copy is declared as holding a reference to the device product because the device-to-host
# transform is not a consumes() of any module and its read would otherwise race the delete. The product is listed
# only when no alpaka module the schedule runs reads it; any such reader keeps it alive. Nothing else of the default
# chain is listed: its rechit SoAs and the CA tracks have readers that read them in produce() on their own queue.

_selectors = ("PixelTrackTorchHighPuritySelector@alpaka",)
_devices = ("alpakaDevCudaRt", "alpakaDevHipRt")
# (device product type without the device prefix, host copy type)
_tracks = ("128falserecoTrackBlocksLayoutvoidPortableDeviceCollectionedmDeviceProduct",
           "128falserecoTrackBlocksLayoutPortableHostCollection")


def _inputLabels(parameters):
    # every module label an InputTag anywhere in the parameter set points at
    labels = set()
    for name in parameters.parameterNames_():
        p = getattr(parameters, name)
        if isinstance(p, cms.InputTag):
            labels.add(p.getModuleLabel())
        elif isinstance(p, cms.VInputTag):
            labels.update(cms.InputTag(t).getModuleLabel() if isinstance(t, str) else t.getModuleLabel() for t in p)
        elif isinstance(p, cms.PSet):
            labels.update(_inputLabels(p))
        elif isinstance(p, cms.VPSet):
            for pset in p:
                labels.update(_inputLabels(pset))
    return labels


def _scheduledModules(process):
    # the modules the job runs: those on the scheduled paths and the ones they consume (unscheduled)
    modules = dict(process.producers_())
    modules.update(process.filters_())
    modules.update(process.analyzers_())
    schedule = process.schedule_()
    paths = list(schedule) if schedule is not None else list(process.paths_().values()) + list(process.endpaths_().values())
    reachable = set()
    for path in paths:
        reachable.update(label for label in path.moduleNames() if label in modules)
    frontier = list(reachable)
    while frontier:
        label = frontier.pop()
        for inputLabel in _inputLabels(modules[label]):
            if inputLabel in modules and inputLabel not in reachable:
                reachable.add(inputLabel)
                frontier.append(inputLabel)
    return {label: modules[label] for label in reachable}


def customiseEarlyDeleteForPixelTrackSelectionSoA(process, products):
    references = collections.defaultdict(list)

    def branchName(productType, moduleLabel, instanceLabel=""):
        return "%s_%s_%s_%s" % (productType, moduleLabel, instanceLabel, process.name_())

    modules = _scheduledModules(process)
    deviceReaders = set()
    for module in modules.values():
        if module.type_().endswith("@alpaka"):
            deviceReaders.update(_inputLabels(module))

    for label, module in sorted(modules.items()):
        if module.type_() not in _selectors or label in deviceReaders:
            continue
        deviceBranches = [branchName(device + _tracks[0], label) for device in _devices]
        products[label].extend(deviceBranches)
        references[branchName(_tracks[1], label)].extend(deviceBranches)

    return (products, references)
