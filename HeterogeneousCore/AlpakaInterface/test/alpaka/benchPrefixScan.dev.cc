// Benchmark (not a unit test): single-kernel multiBlockPrefixScan vs two-kernel iterativePrefixScan
// as a function of the number of elements, to find the crossover point.
//
// For each size: warm up, then enqueue `reps` scans back to back on one queue and measure the host
// wall-clock time between two waits. This includes launch latency and inter-kernel gaps, which is
// what matters in production. Run under nsys to get kernel-only times as well.
// The single kernel needs a zeroed block counter: in OneToManyAssoc this is done for free by zeroAndInit,
// so here each scan gets its own counter, zeroed before the timed loop, and single_us excludes the memset.
// memset_us is the cost of that memset, for reference.
//
// usage: benchPrefixScan [reps=200] [maxLog2=24] [inPlace=0] [stepsPerOctave=4]
// inPlace=1 scans the buffer onto itself, as OneToManyAssoc::launchFinalize does (better L2 reuse).
// stepsPerOctave (a power of 2, up to 256) sets how many sizes are tested between consecutive powers of 2.

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

using namespace cms::alpakatools;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;

struct fillOnes {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, uint32_t* v, uint32_t n) const {
    for (auto i : uniform_elements(acc, n))
      v[i] = 1;
  }
};

struct countErrors {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, uint32_t const* v, uint32_t n, uint32_t* nErr) const {
    for (auto i : uniform_elements(acc, n))
      if (v[i] != i + 1)
        alpaka::atomicAdd(acc, nErr, 1u, alpaka::hierarchy::Blocks{});
  }
};

int main(int argc, char** argv) {
  int const reps = argc > 1 ? std::atoi(argv[1]) : 200;
  int const maxLog2 = argc > 2 ? std::atoi(argv[2]) : 24;
  bool const inPlace = argc > 3 && std::atoi(argv[3]) != 0;
  int const steps = argc > 4 ? std::atoi(argv[4]) : 4;
  int stepsLog2 = 0;
  while ((1 << stepsLog2) < steps)
    ++stepsLog2;
  if (steps < 1 || steps > 256 || (1 << stepsLog2) != steps) {
    std::cerr << "stepsPerOctave must be a power of 2 between 1 and 256\n";
    return EXIT_FAILURE;
  }

  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cerr << "No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend\n";
    return EXIT_FAILURE;
  }
  auto const& device = devices[0];
  auto queue = Queue(device);
  auto const warpSize = alpaka::getPreferredWarpSize(device);
  std::cout << "# device: " << alpaka::getName(device) << ", reps = " << reps << ", inPlace = " << inPlace
            << ", stepsPerOctave = " << steps << ", single excludes memset\n";

  // sizes: 2^l * (1 + k / steps) for k = 0 ... steps - 1, up to 2^maxLog2, and the real CA cell-offset size
  std::vector<uint32_t> sizes;
  for (int l = 8; l <= maxLog2; ++l) {
    for (int k = 0; k < steps; ++k) {
      if (l == maxLog2 && k > 0)
        break;
      sizes.push_back(uint32_t(steps + k) << (l - stepsLog2));
    }
  }
  sizes.push_back(12 * 512 * 1024 + 1);

  auto nErr_d = make_device_buffer<uint32_t>(queue);
  auto nErr_h = make_host_buffer<uint32_t>(queue);
  auto blockCounter_d = make_device_buffer<int32_t>(queue);
  // one block counter per scan (warm up + timed reps + check), zeroed outside the timed loop
  auto counters_d = make_device_buffer<int32_t[]>(queue, reps + 11);
  int nextCounter = 0;
  auto resetCounters = [&]() {
    alpaka::memset(queue, counters_d, 0);
    nextCounter = 0;
  };

  std::cout << std::setw(10) << "size" << std::setw(8) << "blocks" << std::setw(14) << "single_us" << std::setw(14)
            << "iter_us" << std::setw(10) << "ratio" << std::setw(12) << "memset_us" << "  check\n";

  for (auto n : sizes) {
    auto in_d = make_device_buffer<uint32_t[]>(queue, n);
    auto out_d = make_device_buffer<uint32_t[]>(queue, n);
    uint32_t* out = inPlace ? in_d.data() : out_d.data();
    uint32_t const nThreads = 1024;
    uint32_t const nBlocks = divide_up_by(n, nThreads);
    auto const workDiv = make_workdiv<Acc1D>(nBlocks, nThreads);
    alpaka::exec<Acc1D>(queue, workDiv, fillOnes{}, in_d.data(), n);

    // the single-kernel version is limited by the shared memory needed for the partial sums
    bool singleOk = true;
    try {
      checkSharedMemoryPrefixScan<Acc1D>(n, nBlocks, device);
    } catch (std::exception const&) {
      singleOk = false;
    }

    auto runSingle = [&]() {
      alpaka::exec<Acc1D>(queue,
                          workDiv,
                          multiBlockPrefixScan<uint32_t>(),
                          in_d.data(),
                          out,
                          n,
                          static_cast<int32_t>(nBlocks),
                          counters_d.data() + nextCounter++,
                          warpSize);
    };
    auto runIter = [&]() { iterativePrefixScan<Acc1D>(in_d.data(), out, n, queue); };
    auto runMemset = [&]() { alpaka::memset(queue, blockCounter_d, 0); };

    auto check = [&]() {
      alpaka::memset(queue, nErr_d, 0);
      alpaka::exec<Acc1D>(queue, workDiv, countErrors{}, out, n, nErr_d.data());
      alpaka::memcpy(queue, nErr_h, nErr_d);
      alpaka::wait(queue);
      return *nErr_h.data();
    };

    // in place, repeated scans make the values grow (and wrap): harmless for timing, so check a fresh scan
    auto checkRun = [&](auto&& run) {
      if (inPlace)
        alpaka::exec<Acc1D>(queue, workDiv, fillOnes{}, in_d.data(), n);
      run();
      return check();
    };

    auto time = [&](auto&& run) {
      for (int i = 0; i < 10; ++i)
        run();  // warm up, also fills the caching allocator
      alpaka::wait(queue);
      auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < reps; ++i)
        run();
      alpaka::wait(queue);
      auto t1 = std::chrono::steady_clock::now();
      return std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    };

    double tSingle = -1.;
    uint32_t errSingle = 0;
    if (singleOk) {
      resetCounters();
      tSingle = time(runSingle);
      resetCounters();
      errSingle = checkRun(runSingle);
    }
    double tIter = time(runIter);
    uint32_t errIter = checkRun(runIter);
    double tMemset = time(runMemset);

    std::cout << std::setw(10) << n << std::setw(8) << nBlocks << std::fixed << std::setprecision(2) << std::setw(14)
              << tSingle << std::setw(14) << tIter << std::setw(10) << (singleOk ? tSingle / tIter : 0.) << std::setw(12) << tMemset << "  "
              << ((errSingle == 0 && errIter == 0) ? "ok" : "FAILED") << (singleOk ? "" : " (single: smem limit)")
              << '\n';
  }
  return 0;
}
