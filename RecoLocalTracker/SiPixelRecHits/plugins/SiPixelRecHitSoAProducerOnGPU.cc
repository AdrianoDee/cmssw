#include <cstdint>

#include <cuda_runtime.h>

#include "CUDADataFormats/BeamSpot/interface/BeamSpotCUDA.h"
#include "CUDADataFormats/SiPixelCluster/interface/SiPixelClustersCUDA.h"
#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"


#include <array>
#include <cassert>
#include <functional>
#include <vector>

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"

#include "SiPixelRecHitSoAProducerOnGPU.h"


namespace pixelhitconverter {

  template<>
  TrackingRecHit2DCPU SiPixelRecHitSoAProducerOnCPU::convertHits(float* store32,
                                    uint16_t* store16,
                                    uint32_t* moduleStart) const
    {

  TrackingRecHit2DCPU hits(store32,store16, moduleStart, nHits_, moduleStart[startBPIX2_], isPhase2_, nullptr);

  constexpr int L[11] = {0, 96, 320, 672, 1184, 1296, 1408,  1520, 1632, 1744, 1856};

  for (size_t i = 0; i < 11; i++) {
    std::cout << "SiPixelRecHitSoAProducer" << __LINE__ << " " << L[i] << std::endl;
    auto start = L[i];
    // auto start = 96;//layerStart[i];//isPhase2_ ? phase2PixelTopology::layerStart[i] : phase1PixelTopology::layerStart[i];
    std::cout << "SiPixelRecHitSoAProducer" << "- starting at module: " << start << " -> " << moduleStart[start] << std::endl;
std::cout << hits.iphi()[1] << std::endl;
    hits.hitsLayerStart()[i] = moduleStart[start];

        std::cout << " - starts ad cluster: " << hits.hitsLayerStart()[i] << " -" << hits.iphi()[i] << "\n";

    }

    std::cout << "SiPixelRecHitSoAProducer" << __LINE__ << std::endl;

    cms::cuda::fillManyFromVector(hits.phiBinner(),
                                  10,
                                  hits.iphi(),
                                  hits.hitsLayerStart(),
                                  nHits_,
                                  256,
                                  hits.phiBinnerStorage());


    // typename TrackingRecHit2DSOAView::PhiBinner::View view = {hits.phiBinner(), nullptr, hits.phiBinnerStorage(), -1, nHits_};
    // cms::cuda::launchZero(view, theStream);
    // cms::cuda::countFromVector(hits.phiBinner(), nLayers, hits.iphi(), hits.hitsLayerStart());
    // hits.phiBinner()->finalize();
    // std::cout << "SiPixelRecHitSoAProducer" << __LINE__ << std::endl;
    // cms::cuda::fillFromVector(hits.phiBinner(), nLayers, hits.iphi(), hits.hitsLayerStart());
    // std::cout << "SiPixelRecHitSoAProducer" << __LINE__ << std::endl;

    LogDebug("SiPixelRecHitSoAProducer") << "created HitSoa for " << nHits_ << std::endl;
    std::cout << "SiPixelRecHitSoAProducer" << __LINE__ << std::endl;
    // iEvent.emplace(tokenHitCPU_, std::move(output));

    return hits;

  }


                                    }
