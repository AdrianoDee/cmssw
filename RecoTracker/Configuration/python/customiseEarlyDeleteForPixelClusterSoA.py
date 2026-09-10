import collections

import FWCore.ParameterSet.Config as cms

# Early deletion of the pixel digi and cluster SoAs on the device. Their only consumer is the alpaka
# pixel rechit producer, after which the framework would otherwise keep them until the end of the
# event: about 16 MiB per event and stream on QCD PU200.
#
# Only the device branches are listed, spelled for every asynchronous backend; an entry for a backend
# the job does not use is ignored, while listing the host copies, which have no reader on the
# asynchronous backends, would make the framework warn at every job start. The host copies are
# declared as holding references to the device branches, because the device-to-host transform is not
# a consumes() of any module and its read would otherwise race the early delete.
#
# Releasing a device product early is safe only if every consumer enqueues its reads on the product's
# queue before taking a queue of its own, because the caching allocator marks a freed block for reuse
# on the queue that allocated it. Do not add products here without that check on each consumer.

_clusterizers = ("SiPixelRawToClusterPhase1@alpaka", "SiPixelPhase2DigiToCluster@alpaka")
_devices = ("alpakaDevCudaRt", "alpakaDevHipRt")
# (device product type without the device prefix, host copy type)
_products = (("SiPixelDigisDeviceedmDeviceProduct", "SiPixelDigisHost"),
             ("SiPixelClustersDeviceedmDeviceProduct", "SiPixelClustersHost"))


def customiseEarlyDeleteForPixelClusterSoA(process, products):
    references = collections.defaultdict(list)

    def branchName(productType, moduleLabel, instanceLabel=""):
        return "%s_%s_%s_%s" % (productType, moduleLabel, instanceLabel, process.name_())

    for label, module in process.producers_().items():
        if module.type_() not in _clusterizers:
            continue
        for deviceType, hostType in _products:
            deviceBranches = [branchName(device + deviceType, label) for device in _devices]
            products[label].extend(deviceBranches)
            references[branchName(hostType, label)].extend(deviceBranches)

    return (products, references)
