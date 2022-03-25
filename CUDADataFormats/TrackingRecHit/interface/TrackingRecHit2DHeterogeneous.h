#ifndef CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneousT_h
#define CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneousT_h

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DSOAView.h"
#include "CUDADataFormats/Common/interface/HeterogeneousSoA.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"
#include "CUDADataFormats/TrackerGeometry/interface/SimplePixelTopology.h"

template <typename Traits, typename TrackerTraits>
class TrackingRecHit2DHeterogeneousT {
public:
  template <typename T>
  using unique_ptr = typename Traits::template unique_ptr<T>;

  using TrackingRecHit2DSOAView = TrackingRecHit2DSOAViewT<TrackerTraits>;
  using PhiBinner = typename TrackingRecHit2DSOAView::PhiBinner;
  using AverageGeometry = typename TrackingRecHit2DSOAView::AverageGeometry;

  TrackingRecHit2DHeterogeneousT() = default;

  explicit TrackingRecHit2DHeterogeneousT(
      uint32_t nHits,
      int32_t offsetBPIX2,
      pixelCPEforGPU::ParamsOnGPUT<TrackerTraits> const* cpeParams,
      uint32_t const* hitsModuleStart,
      cudaStream_t stream,
      TrackingRecHit2DHeterogeneousT<cms::cudacompat::GPUTraits,TrackerTraits> const* input = nullptr);

  ~TrackingRecHit2DHeterogeneousT() = default;

  TrackingRecHit2DHeterogeneousT(const TrackingRecHit2DHeterogeneousT&) = delete;
  TrackingRecHit2DHeterogeneousT& operator=(const TrackingRecHit2DHeterogeneousT&) = delete;
  TrackingRecHit2DHeterogeneousT(TrackingRecHit2DHeterogeneousT&&) = default;
  TrackingRecHit2DHeterogeneousT& operator=(TrackingRecHit2DHeterogeneousT&&) = default;

  TrackingRecHit2DSOAView* view() { return m_view.get(); }
  TrackingRecHit2DSOAView const* view() const { return m_view.get(); }

  auto nHits() const { return m_nHits; }
  auto nMaxModules() const { return m_nMaxModules; }
  auto offsetBPIX2() const { return m_offsetBPIX2; }

  auto hitsModuleStart() const { return m_hitsModuleStart; }
  auto hitsLayerStart() { return m_hitsLayerStart; }
  auto phiBinner() { return m_phiBinner; }
  auto phiBinnerStorage() { return m_phiBinnerStorage; }
  auto iphi() { return m_iphi; }

  // only the local coord and detector index
  cms::cuda::host::unique_ptr<float[]> localCoordToHostAsync(cudaStream_t stream) const;
  cms::cuda::host::unique_ptr<uint32_t[]> hitsModuleStartToHostAsync(cudaStream_t stream) const;

  // needs specialization for Host
  void copyFromGPU(TrackingRecHit2DHeterogeneousT<cms::cudacompat::GPUTraits,TrackerTraits> const* input, cudaStream_t stream);

protected:
  static constexpr uint32_t n16 = 4;                 // number of elements in m_store16
  static constexpr uint32_t n32 = 10;                // number of elements in m_store32
  static_assert(sizeof(uint32_t) == sizeof(float));  // just stating the obvious

  unique_ptr<uint16_t[]> m_store16;  //!
  unique_ptr<float[]> m_store32;     //!

  unique_ptr<PhiBinner> m_PhiBinnerStore;              //!
  unique_ptr<AverageGeometry> m_AverageGeometryStore;  //!

  unique_ptr<TrackingRecHit2DSOAView> m_view;  //!

  uint32_t m_nHits;
  int32_t m_offsetBPIX2;

  uint32_t const* m_hitsModuleStart;  // needed for legacy, this is on GPU!

  uint32_t m_nMaxModules;
  // needed as kernel params...
  PhiBinner* m_phiBinner;
  typename PhiBinner::index_type* m_phiBinnerStorage;
  uint32_t* m_hitsLayerStart;
  int16_t* m_iphi;
};


//TrackingRecHit2DGPU/CPU/Host workaround to avoid partial specialization (forbidden)

template <typename Traits,typename TrackerTraits>
class TrackingRecHit2DGPUBaseT : public TrackingRecHit2DHeterogeneousT<cms::cudacompat::GPUTraits,TrackerTraits>{};

template <typename Traits,typename TrackerTraits>
class TrackingRecHit2DCPUBaseT : public TrackingRecHit2DHeterogeneousT<cms::cudacompat::CPUTraits,TrackerTraits>{};

template <typename Traits,typename TrackerTraits>
class TrackingRecHit2DHostBaseT : public TrackingRecHit2DHeterogeneousT<cms::cudacompat::HostTraits,TrackerTraits>{};


//Specilize and overload only what we need to overload, for other memebers use this->
//GPU
template <typename TrackerTraits>
class TrackingRecHit2DGPUBaseT<cms::cudacompat::GPUTraits,TrackerTraits> : public TrackingRecHit2DHeterogeneousT<cms::cudacompat::GPUTraits,TrackerTraits> {
  public:
    using TrackingRecHit2DHeterogeneousT<cms::cudacompat::GPUTraits,TrackerTraits>::TrackingRecHit2DHeterogeneousT;
    cms::cuda::host::unique_ptr<float[]> localCoordToHostAsync(cudaStream_t stream) const;
    cms::cuda::host::unique_ptr<uint32_t[]> hitsModuleStartToHostAsync(cudaStream_t stream) const;
};
// Alias to avoid bringing GPUTraits around
template<typename TrackerTraits>
using TrackingRecHit2DGPUT = TrackingRecHit2DGPUBaseT<cms::cudacompat::GPUTraits,TrackerTraits>;

//CPU
template <typename TrackerTraits>
class TrackingRecHit2DCPUBaseT<cms::cudacompat::CPUTraits,TrackerTraits> : public TrackingRecHit2DHeterogeneousT<cms::cudacompat::CPUTraits,TrackerTraits> {
public:
  using TrackingRecHit2DHeterogeneousT<cms::cudacompat::CPUTraits,TrackerTraits>::TrackingRecHit2DHeterogeneousT;
};

// Alias to avoid bringing GPUTraits around
template<typename TrackerTraits>
using TrackingRecHit2DCPUT = TrackingRecHit2DCPUBaseT<cms::cudacompat::CPUTraits,TrackerTraits>;

//HOST
template <typename TrackerTraits>
class TrackingRecHit2DHostBaseT<cms::cudacompat::HostTraits,TrackerTraits> : public TrackingRecHit2DHeterogeneousT<cms::cudacompat::HostTraits,TrackerTraits> {
  public:
    using TrackingRecHit2DHeterogeneousT<cms::cudacompat::HostTraits,TrackerTraits>::TrackingRecHit2DHeterogeneousT;
    void copyFromGPU(TrackingRecHit2DGPUT<TrackerTraits> const* input, cudaStream_t stream);
};
// Alias to avoid bringing HostTraits around
template<typename TrackerTraits>
using TrackingRecHit2DHostT = TrackingRecHit2DHostBaseT<cms::cudacompat::HostTraits,TrackerTraits>;

//Classes definition for Phase1
using TrackingRecHit2DGPU = TrackingRecHit2DGPUT<pixelTopology::Phase1>;
using TrackingRecHit2DCPU = TrackingRecHit2DCPUT<pixelTopology::Phase1>;
using TrackingRecHit2DHost = TrackingRecHit2DHostT<pixelTopology::Phase1>;
//Classes definition for Phase2
using TrackingRecHit2DGPUPhase2 = TrackingRecHit2DGPUT<pixelTopology::Phase2>;
using TrackingRecHit2DCPUPhase2 = TrackingRecHit2DCPUT<pixelTopology::Phase2>;
using TrackingRecHit2DHostPhase2 = TrackingRecHit2DHostT<pixelTopology::Phase2>;

#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

template <typename Traits, typename TrackerTraits>
TrackingRecHit2DHeterogeneousT<Traits,TrackerTraits>::TrackingRecHit2DHeterogeneousT(
    uint32_t nHits,
    int32_t offsetBPIX2,
    pixelCPEforGPU::ParamsOnGPUT<TrackerTraits> const* cpeParams,
    uint32_t const* hitsModuleStart,
    cudaStream_t stream,
    TrackingRecHit2DHeterogeneousT<cms::cudacompat::GPUTraits,TrackerTraits> const* input)
    : m_nHits(nHits), m_offsetBPIX2(offsetBPIX2), m_hitsModuleStart(hitsModuleStart) {

  using TrackingRecHit2DSOAView = TrackingRecHit2DSOAViewT<TrackerTraits>;

  auto view = Traits::template make_host_unique<TrackingRecHit2DSOAView>(stream);

  //m_nMaxModules = isPhase2 ? phase2PixelTopology::numberOfModules : phase1PixelTopology::numberOfModules;

  view->m_nHits = nHits;
  view->m_nMaxModules = TrackerTraits::numberOfModules;
  m_view = Traits::template make_unique<TrackingRecHit2DSOAView>(stream);  // leave it on host and pass it by value?
  m_AverageGeometryStore = Traits::template make_unique<typename TrackingRecHit2DSOAView::AverageGeometry>(stream);
  view->m_averageGeometry = m_AverageGeometryStore.get();
  view->m_cpeParams = cpeParams;
  view->m_hitsModuleStart = hitsModuleStart;

  // if empy do not bother
  if (0 == nHits) {
    if constexpr (std::is_same<Traits, cms::cudacompat::GPUTraits>::value) {
      cms::cuda::copyAsync(m_view, view, stream);
    } else {
      m_view.reset(view.release());  // NOLINT: std::move() breaks CUDA version
    }
    return;
  }

  // the single arrays are not 128 bit alligned...
  // the hits are actually accessed in order only in building
  // if ordering is relevant they may have to be stored phi-ordered by layer or so
  // this will break 1to1 correspondence with cluster and module locality
  // so unless proven VERY inefficient we keep it ordered as generated

  // host copy is "reduced"  (to be reviewed at some point)
  if constexpr (std::is_same<Traits, cms::cudacompat::HostTraits>::value) {
    // it has to compile for ALL cases
    copyFromGPU(input, stream);
  } else {
    assert(input == nullptr);

    //auto nL = isPhase2 ? phase2PixelTopology::numberOfLayers : phase1PixelTopology::numberOfLayers;

    m_store16 = Traits::template make_unique<uint16_t[]>(nHits * n16, stream);
    m_store32 = Traits::template make_unique<float[]>(nHits * n32 + TrackerTraits::numberOfLayers + 1, stream);
    m_PhiBinnerStore = Traits::template make_unique<typename TrackingRecHit2DSOAView::PhiBinner>(stream);
  }

  static_assert(sizeof(typename TrackingRecHit2DSOAView::hindex_type) == sizeof(float));
  static_assert(sizeof(typename TrackingRecHit2DSOAView::hindex_type) == sizeof(typename TrackingRecHit2DSOAView::PhiBinner::index_type));

  auto get32 = [&](int i) { return m_store32.get() + i * nHits; };

  // copy all the pointers
  m_phiBinner = view->m_phiBinner = m_PhiBinnerStore.get();
  m_phiBinnerStorage = view->m_phiBinnerStorage =
      reinterpret_cast<typename TrackingRecHit2DSOAView::PhiBinner::index_type*>(get32(9));

  view->m_xl = get32(0);
  view->m_yl = get32(1);
  view->m_xerr = get32(2);
  view->m_yerr = get32(3);
  view->m_chargeAndStatus = reinterpret_cast<uint32_t*>(get32(4));

  if constexpr (!std::is_same<Traits, cms::cudacompat::HostTraits>::value) {
    assert(input == nullptr);
    view->m_xg = get32(5);
    view->m_yg = get32(6);
    view->m_zg = get32(7);
    view->m_rg = get32(8);

    auto get16 = [&](int i) { return m_store16.get() + i * nHits; };
    m_iphi = view->m_iphi = reinterpret_cast<int16_t*>(get16(1));

    view->m_xsize = reinterpret_cast<int16_t*>(get16(2));
    view->m_ysize = reinterpret_cast<int16_t*>(get16(3));
    view->m_detInd = get16(0);

    m_phiBinner = view->m_phiBinner = m_PhiBinnerStore.get();
    m_hitsLayerStart = view->m_hitsLayerStart = reinterpret_cast<uint32_t*>(get32(n32));
  }

  // transfer view
  if constexpr (std::is_same<Traits, cms::cudacompat::GPUTraits>::value) {
    cms::cuda::copyAsync(m_view, view, stream);
  } else {
    m_view.reset(view.release());  // NOLINT: std::move() breaks CUDA version
  }
}

//

#endif  // CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneousT_h
