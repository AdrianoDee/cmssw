#ifndef CUDADataFormats_RecHits_TrackingRecHitsUtilities_h
#define CUDADataFormats_RecHits_TrackingRecHitsUtilities_h

#include <Eigen/Dense>
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "SiPixelHitStatus.h"

namespace trackingRecHitSoA {

  // more information on bit fields : https://en.cppreference.com/w/cpp/language/bit_field
  struct SiPixelHitStatusAndCharge {
    SiPixelHitStatus status;
    uint32_t charge : 24;
  };

  struct Test {
    int a;
  };

  using hindex_type = uint32_t;  // if above is <=2^32
  using PhiBinner = cms::cuda::HistoContainer<int16_t,
                                              256,
                                              -1,
                                              8 * sizeof(int16_t),
                                              hindex_type,
                                              phase1PixelTopology::numberOfLayers>;  //28 for phase2 geometry
  using PhiBinnerStorageType = PhiBinner::index_type;

  using AverageGeometry = pixelTopology::AverageGeometry;

  using ParamsOnGPU = pixelCPEforGPU::ParamsOnGPU;

  using HitLayerStartArray = std::array<uint32_t, phase1PixelTopology::numberOfLayers + 1>;
  using HitModuleStartArray = std::array<uint32_t, phase1PixelTopology::numberOfModules + 1>;

}  // namespace trackingRecHitSoA

GENERATE_SOA_LAYOUT(TrackingRecHitSoALayout,
                    SOA_COLUMN(float, xLocal),
                    SOA_COLUMN(float, yLocal),  // this is chi2/ndof as not necessarely all hits are used in the fit
                    SOA_COLUMN(float, xerrLocal),
                    SOA_COLUMN(float, yerrLocal),
                    SOA_COLUMN(float, xGlobal),
                    SOA_COLUMN(float, yGlobal),
                    SOA_COLUMN(float, zGlobal),
                    SOA_COLUMN(float, rGlobal),
                    SOA_COLUMN(int16_t, iphi),
                    SOA_COLUMN(trackingRecHitSoA::SiPixelHitStatusAndCharge, chargeAndStatus),
                    SOA_COLUMN(int16_t, clusterSizeX),
                    SOA_COLUMN(int16_t, clusterSizeY),
                    SOA_COLUMN(int16_t, detectorIndex),
                    SOA_COLUMN(trackingRecHitSoA::PhiBinnerStorageType, phiBinnerStorage),

                    SOA_SCALAR(trackingRecHitSoA::HitModuleStartArray, hitsModuleStart),
                    SOA_SCALAR(trackingRecHitSoA::HitLayerStartArray, hitsLayerStart),

                    SOA_SCALAR(trackingRecHitSoA::ParamsOnGPU, cpeParams),
                    SOA_SCALAR(trackingRecHitSoA::AverageGeometry, averageGeometry),
                    SOA_SCALAR(trackingRecHitSoA::PhiBinner, phiBinner),

                    SOA_SCALAR(uint32_t, nHits),
                    SOA_SCALAR(int32_t, offsetBPIX2),
                    SOA_SCALAR(uint32_t, nMaxModules))

namespace trackingRecHitSoA {

  using HitSoAView = TrackingRecHitSoALayout<>::View;
  using HitSoAConstView = TrackingRecHitSoALayout<>::ConstView;

  constexpr size_t columnsSizes = 8 * sizeof(float) + 4 * sizeof(int16_t) +
                                  sizeof(trackingRecHitSoA::SiPixelHitStatusAndCharge) +
                                  sizeof(trackingRecHitSoA::PhiBinnerStorageType);

}  // namespace trackingRecHitSoA
#endif
