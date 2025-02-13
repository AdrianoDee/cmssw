#include <algorithm>
#include <cassert>
#include <chrono>
using namespace std::chrono_literals;
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <type_traits>
#include <map>

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/radixSort.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

using namespace cms::alpakatools;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;


void sortDigis(Queue& queue, bool useShared) {

  std::chrono::high_resolution_clock::duration delta = 0ns;
  constexpr auto nbytes = 4;

  constexpr int blocks = 1200; // > 512 * 1024 digis
  constexpr int blockSize = 5000;
  constexpr int N = blockSize * blocks;
  std::vector <uint32_t> digis(N); // just to use is_sorted

  auto v_h = cms::alpakatools::make_host_buffer<uint32_t[]>(queue, N);

  for (long long int j = 0; j < N; j++)
    v_h[j] = digis[j] = j/1000; // 1000 digis per module

  //insert spurious
  for (long long int j = 0; j < N; j = j + 10)
    v_h[j] = digis[j] = j/120;

  
  if (N < 1e3) // avoid huge printouts
  {
    std::cout << "> The unsorted digis" << std::endl;
    for (long long int j = 0; j < N; j++)
      std::cout << digis[j] << ";";
    std::cout << std::endl;
  }

  assert(not std::is_sorted(digis.begin(),digis.end()));

  auto offsets_h = cms::alpakatools::make_host_buffer<uint32_t[]>(queue, blocks + 1);
  offsets_h[0] = 0;
  for (int j = 1; j < blocks + 1; ++j) {
    offsets_h[j] = offsets_h[j - 1] + blockSize - blockSize/10;
    assert(offsets_h[j] <= N);
  }

  auto v_d = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, N);
  auto ind_d = cms::alpakatools::make_device_buffer<uint16_t[]>(queue, N);
  auto ind_h = cms::alpakatools::make_host_buffer<uint16_t[]>(queue, N);
  auto ws_d = cms::alpakatools::make_device_buffer<uint16_t[]>(queue, N);
  auto off_d = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, blocks + 1);

  alpaka::memcpy(queue, v_d, v_h);
  alpaka::memcpy(queue, off_d, offsets_h);

  auto constexpr nthreads = 256;

  auto workdiv = make_workdiv<Acc1D>(blocks, nthreads);

  if(useShared)
  { 
    int iteration = 0;
    std::vector<uint32_t> sortedDigis(N);
    do{
      std::cout << "Iteration: " << iteration++ << std::endl;

      sortedDigis.clear();
      auto start = std::chrono::high_resolution_clock::now();
      alpaka::enqueue(queue,
        alpaka::createTaskKernel<Acc1D>(workdiv,
                                        radixSortMultiWrapper<uint32_t, nbytes>{},
                                        v_d.data(),
                                        ind_d.data(),
                                        off_d.data(),
                                        nullptr,
                                        blockSize * sizeof(uint16_t)));

      alpaka::memcpy(queue, ind_h, ind_d);
      alpaka::wait(queue);
      delta += std::chrono::high_resolution_clock::now() - start;
      
      auto I = 0;
      for (int ib = 0; ib < blocks; ++ib) {
        for (auto j = offsets_h[ib] + 1; j < offsets_h[ib + 1]; j++) {

          auto a = v_h.data() + offsets_h[ib];
          auto k1 = a[ind_h[j]];
          sortedDigis[I++] = k1;
          if (N < 1e3)
            std::cout << k1 << ";";
        }
      }
      if (N < 1e3)
        std::cout << std::endl;

    } while (not std::is_sorted(sortedDigis.begin(),sortedDigis.end()));
    
    std::cout << "Done in " << iteration << " iterations and " 
        << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count() << " milliseconds " << std::endl;
}
}

int main() {
  // get the list of devices on the current platform
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cerr << "No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
      "the test will be skipped.\n";
    exit(EXIT_FAILURE);
  }

  for (auto const& device : devices) {
    Queue queue(device);

    std::cout << "Sorting digis!" << std::endl;

    sortDigis(queue, true);
  }
  return 0;
}
