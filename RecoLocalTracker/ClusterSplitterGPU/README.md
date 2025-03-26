This branch is using cmssw emptysource events to run in the producer class and feed alpaka kernels

Each event a kernel should be created, depending the number of hits, threads and blocks will perform:

For example:
nHits = 100 (number of hits for one event)
items = 64 (threads per block).
This results in groups = ceil(100 / 64) = 2 blocks.
