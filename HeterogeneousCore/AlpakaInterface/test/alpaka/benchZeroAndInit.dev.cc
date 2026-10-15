// Benchmark (not a unit test): OneToManyAssoc zeroing + init as a function of the number of offsets.
//
// Compares
//   old    : the original single-block zeroAndInit (local copy below)
//   new    : OneToManyAssoc::launchZero as currently in the header (multi-block)
//   memset : alpaka::memset of the offsets + a 1-thread init kernel (2 operations)
//
// For each size: warm up, then enqueue `reps` operations back to back on one queue and measure the host
// wall-clock time between two waits. This includes launch latency and inter-kernel gaps. Run under nsys
// to get kernel-only times as well.
//
// usage: benchZeroAndInit [reps=200] [maxLog2=23]

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/OneToManyAssoc.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

using namespace cms::alpakatools;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;

// same type as caStructures::GenericContainer
using Assoc = OneToManyAssocRandomAccess<uint32_t, kDynamicSize, kDynamicSize>;
using View = Assoc::View;

// the original implementation: MUST run on a single block
struct oldZeroAndInit {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, View view) const {
    auto h = view.assoc;
    if (once_per_block(acc)) {
      h->psws = 0;
      h->initStorage(view);
    }
    alpaka::syncBlockThreads(acc);
    for (auto i : independent_group_elements(acc, h->totOnes())) {
      h->off[i] = 0;
    }
  }
};

// init only, for the memset variant
struct initOnly {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, View view) const {
    if (once_per_grid(acc)) {
      view.assoc->psws = 0;
      view.assoc->initStorage(view);
    }
  }
};

struct countNonZero {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, uint32_t const* v, uint32_t n, uint32_t* nErr) const {
    for (auto i : uniform_elements(acc, n))
      if (v[i] != 0)
        alpaka::atomicAdd(acc, nErr, 1u, alpaka::hierarchy::Blocks{});
  }
};

int main(int argc, char** argv) {
  int const reps = argc > 1 ? std::atoi(argv[1]) : 200;
  int const maxLog2 = argc > 2 ? std::atoi(argv[2]) : 23;

  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cerr << "No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend\n";
    return EXIT_FAILURE;
  }
  auto const& device = devices[0];
  auto queue = Queue(device);
  std::cout << "# device: " << alpaka::getName(device) << ", reps = " << reps << '\n';

  // number of offsets (nOnes + 1): powers of 2 plus the half-way points, and the real CA cell-offset size
  std::vector<uint32_t> sizes;
  for (int l = 8; l <= maxLog2; ++l) {
    sizes.push_back(1u << l);
    if (l < maxLog2)
      sizes.push_back(3u << (l - 1));
  }
  sizes.push_back(12 * 512 * 1024 + 1);

  uint32_t const contentSize = 1024;  // not touched by the zeroing
  auto assoc_d = make_device_buffer<Assoc>(queue);
  auto assoc_h = make_host_buffer<Assoc>(queue);
  auto content_d = make_device_buffer<uint32_t[]>(queue, contentSize);
  auto nErr_d = make_device_buffer<uint32_t>(queue);
  auto nErr_h = make_host_buffer<uint32_t>(queue);

  std::cout << std::setw(10) << "offsets" << std::setw(8) << "blocks" << std::setw(12) << "old_us" << std::setw(12)
            << "new_us" << std::setw(12) << "memset_us" << std::setw(10) << "old/new" << "  check\n";

  for (auto n : sizes) {
    auto off_d = make_device_buffer<uint32_t[]>(queue, n);
    View view = {assoc_d.data(), off_d.data(), content_d.data(), n, contentSize};
    uint32_t const nThreads = 1024;
    uint32_t const nBlocks = divide_up_by(n, nThreads);

    auto runOld = [&]() {
      alpaka::exec<Acc1D>(queue, make_workdiv<Acc1D>(1, nThreads), oldZeroAndInit{}, view);
    };
    auto runNew = [&]() { Assoc::template launchZero<Acc1D>(view, queue); };
    auto runMemset = [&]() {
      alpaka::memset(queue, off_d, 0);
      alpaka::exec<Acc1D>(queue, make_workdiv<Acc1D>(1, 1), initOnly{}, view);
    };

    // poison offsets and psws, run once, then check that everything is zero and the storage is set up
    auto check = [&](auto&& run) {
      alpaka::memset(queue, off_d, 0xff);
      alpaka::memset(queue, assoc_d, 0xff);
      run();
      alpaka::memset(queue, nErr_d, 0);
      alpaka::exec<Acc1D>(queue, make_workdiv<Acc1D>(nBlocks, nThreads), countNonZero{}, off_d.data(), n, nErr_d.data());
      alpaka::memcpy(queue, nErr_h, nErr_d);
      alpaka::memcpy(queue, assoc_h, assoc_d);
      alpaka::wait(queue);
      auto const& a = *assoc_h.data();
      return *nErr_h.data() == 0 && a.psws == 0 && a.totOnes() == n && a.capacity() == contentSize &&
             a.off.data() == off_d.data() && a.content.data() == content_d.data();
    };

    auto time = [&](auto&& run) {
      for (int i = 0; i < 10; ++i)
        run();  // warm up
      alpaka::wait(queue);
      auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < reps; ++i)
        run();
      alpaka::wait(queue);
      auto t1 = std::chrono::steady_clock::now();
      return std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    };

    double tOld = time(runOld);
    double tNew = time(runNew);
    double tMemset = time(runMemset);
    bool okOld = check(runOld);
    bool okNew = check(runNew);
    bool okMemset = check(runMemset);

    std::cout << std::setw(10) << n << std::setw(8) << nBlocks << std::fixed << std::setprecision(2) << std::setw(12)
              << tOld << std::setw(12) << tNew << std::setw(12) << tMemset << std::setw(10) << tOld / tNew << "  "
              << (okOld ? "" : "old:FAILED ") << (okNew ? "" : "new:FAILED ") << (okMemset ? "" : "memset:FAILED ")
              << ((okOld && okNew && okMemset) ? "ok" : "") << '\n';
  }
  return 0;
}
