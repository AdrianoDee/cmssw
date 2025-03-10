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
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"

#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigisSoACollection.h"

using namespace cms::alpakatools;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;

#define HISTO_DIGI_SORT_PRINTOUTS

void sortDigis(Queue& queue, bool shuffle, const unsigned int N = 50000) {
  

  std::chrono::high_resolution_clock::duration delta = 0ns;

  std::cout << " - N : " << N << std::endl;

  SiPixelDigisSoACollection digis(N, queue);

  auto digisView = digis.view();
  auto nDigis = digis.view().metadata().size();
  auto v_d = cms::alpakatools::make_device_view(alpaka::getDev(queue), digisView.moduleId(), nDigis);
  auto v_h = cms::alpakatools::make_host_buffer<uint16_t[]>(queue, N);

  auto i_d = cms::alpakatools::make_device_view(alpaka::getDev(queue), digisView.sortedDigiIdx(), nDigis);

  constexpr unsigned int maxNumModules = 4000u;
  const unsigned int good = N > 100000 ? N / 10000 : N / 100; // to have less than 4000 "modules"
  const unsigned int spurious = 10;
  
  for (unsigned int j = 0; j < N; j++)
    v_h[j] = j/good >= maxNumModules ? maxNumModules - 1 : j/good; 
  
  //insert spurious
  for (unsigned int j = 0; j < N; j = j + 100)
    v_h[j] = v_h[j] + spurious >= maxNumModules ? maxNumModules : v_h[j] + spurious;

  if (shuffle)
  {
    std::mt19937 eng;
    std::shuffle(v_h.data(), v_h.data() + N, eng);
  }
  
#ifdef HISTO_DIGI_SORT_PRINTOUTS
   
  std::cout << "> The unsorted digis" << std::endl;
  for (unsigned int j = 0; j < N; j++)
    std::cout << v_h[j] << ";";
  std::cout << std::endl;

#endif

  // auto v_d = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, N);
  alpaka::memcpy(queue, v_d, v_h);

  auto start = std::chrono::high_resolution_clock::now();
  // 4000 < 2^12
  using DigisHisto = HistoContainer<uint16_t, 5000 + 1, -1, 13, uint32_t>;
  auto histo = make_device_buffer<DigisHisto>(queue);
  auto histo_storage = make_device_buffer<typename DigisHisto::index_type[]>(queue, N);
  auto histo_storage_h = make_host_buffer<typename DigisHisto::index_type[]>(queue, N);
  auto offsets_h = make_host_buffer<uint32_t[]>(queue,2);
  auto offsets_d = make_device_buffer<uint32_t[]>(queue,2);

  alpaka::memset(queue, histo, 0);
  offsets_h[0] = 0;
  offsets_h[1] = N;
  alpaka::memcpy(queue, offsets_d, offsets_h);

  typename DigisHisto::View view;
  view.assoc = histo.data();
  view.offSize = -1;
  view.offStorage = nullptr;
  view.contentSize = N;
  view.contentStorage = digis.view().sortedDigiIdx();//histo_storage.data();

  fillManyFromVector<Acc1D>(histo.data(), view, 1, v_d.data(), offsets_d.data(), N, 1024, queue);
  
  auto histo_h = make_host_buffer<DigisHisto>(queue);
  
  alpaka::memcpy(queue, histo_h, histo);
  alpaka::memcpy(queue, histo_storage_h, i_d);
  alpaka::wait(queue);

  typename DigisHisto::View hrv;
  hrv.assoc = histo_h.data();
  hrv.offSize = -1;
  hrv.offStorage = nullptr;
  hrv.contentSize = N;
  hrv.contentStorage = histo_storage_h.data();
  histo_h->initStorage(hrv);

  delta += std::chrono::high_resolution_clock::now() - start;

  for (uint32_t i = 0; i < DigisHisto::nbins(); ++i) {
    auto off = DigisHisto::histOff(0);
    auto ii = i + off;
    auto b = histo_h->begin(ii);
    auto e = histo_h->end(ii);
    auto s = histo_h->size(ii);
    if (s>0)
     std::cout << "Bin" << ii << " - " << *b << ":"<< *e <<" ->" << v_h[*b] << " : " << s << std::endl;
  }

  for (long long int j = 0; j < N-1; j++)
  {
    if(v_h[histo_storage_h[j]] > v_h[histo_storage_h[j+1]])
      std::cout << "!!! Error for: " << v_h[histo_storage_h[j]] << "-" << v_h[histo_storage_h[j+1]] << std::endl;
    assert(v_h[histo_storage_h[j]]<=v_h[histo_storage_h[j+1]]);
  }
#ifdef HISTO_DIGI_SORT_PRINTOUTS

  std::cout << "Histo storage digis" << std::endl;
  for (long long int j = 0; j < N; j++)
  {
    std::cout << histo_storage_h[j] << ";";
  }
  
  std::cout << "Sorted digis" << std::endl;
  for (long long int j = 0; j < N; j++)
  {
    std::cout << v_h[histo_storage_h[j]] << ";";
  }
  std::cout << std::endl;

#endif
  
  std::cout << "Done in " << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count() << " milliseconds " << std::endl;

}

int main() {
  // get the list of devices on the current platform
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cerr << "No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
      "the test will be skipped.\n";
    exit(EXIT_FAILURE);
  }

  int devCount = 1;
  for (auto const& device : devices) {
    Queue queue(device);
    std::cout << "= Device " << devCount++ << std::endl;
    std::cout << "> Sorting digis with spurious!" << std::endl;

    sortDigis(queue, false, 100000);
    sortDigis(queue, false, 500000);
    sortDigis(queue, false, 100000);

    std::cout << "> Sorting shuffled digis!" << std::endl;
    
    sortDigis(queue, true, 100000);
    sortDigis(queue, true, 500000);
    sortDigis(queue, true, 1000000);
  }
  return 0;
}
