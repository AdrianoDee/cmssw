#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"

#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisDevice.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisSoA.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigisSoACollection.h"

#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersDevice.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersSoA.h"
#include "DataFormats/SiPixelClusterSoA/interface/alpaka/SiPixelClustersSoACollection.h"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/JetReco/interface/Jet.h"

#include "DataFormats/VertexSoA/interface/ZVertexSoA.h"
#include "DataFormats/VertexSoA/interface/ZVertexHost.h"
#include "DataFormats/VertexSoA/interface/ZVertexDevice.h"
#include "DataFormats/VertexSoA/interface/alpaka/ZVertexSoACollection.h"

#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include "DataFormats/Math/interface/SSEVec.h"
#include "DataFormats/Math/interface/ExtVec.h"

#include "DataFormats/ClusterGeometrySoA/interface/ClusterGeometryLayout.h"
#include "DataFormats/ClusterGeometrySoA/interface/alpaka/ClusterGeometrySoACollection.h"

#include "DataFormats/CandidateSoA/interface/CandidateLayout.h"
#include "DataFormats/CandidateSoA/interface/alpaka/CandidateSoACollection.h"

#include "Cluster_test.h"

using namespace alpaka;
using namespace reco;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;
  namespace Splitting {

    template <typename TrackerTraits>
    struct Printout {
      template <typename TAcc, typename = std::enable_if_t<isAccelerator<TAcc>>>
        ALPAKA_FN_ACC void operator()(TAcc const& acc, 
                                TrackingRecHitSoAConstView<TrackerTraits> hitView, 
                                SiPixelDigisSoAConstView digiView,
                                SiPixelClustersSoAConstView clusterView,
                                ZVertexSoAView vertexView,
                                CandidateSoAView candidateView,
                                ClusterGeometrySoAView geoclusterView) const {         
 

        // Print debug info for RecHits -----------------------------------
        if (cms::alpakatools::once_per_grid(acc)) {
          printf("TrackingRecHits Info:\n");
          printf("nbins = %d\n", hitView.phiBinner().nbins());
          printf("offsetBPIX = %d\n", hitView.offsetBPIX2());
          printf("nHits = %d\n", hitView.metadata().size());
        }
        for (uint32_t i : cms::alpakatools::uniform_elements(acc, 10)) {
          printf("Hit %d -> xLocal: %.2f, yLocal: %.2f, xerrLocal: %.2f, yerrLocal: %.2f, "
                 "xGlobal: %.2f, yGlobal: %.2f, zGlobal: %.2f, rGlobal: %.2f, iPhi: %d, "
                 "charge: %d, isBigX: %d, isOneX: %d, isBigY: %d, isOneY: %d, qBin: %d, "
                 "clusterSizeX: %d, clusterSizeY: %d, detectorIndex: %d\n",
                 i,
                 hitView[i].xLocal(),
                 hitView[i].yLocal(),
                 hitView[i].xerrLocal(),
                 hitView[i].yerrLocal(),
                 hitView[i].xGlobal(),
                 hitView[i].yGlobal(),
                 hitView[i].zGlobal(),
                 hitView[i].rGlobal(),
                 hitView[i].iphi(),
                 hitView[i].chargeAndStatus().charge,
                 hitView[i].chargeAndStatus().status.isBigX,
                 hitView[i].chargeAndStatus().status.isOneX,
                 hitView[i].chargeAndStatus().status.isBigY,
                 hitView[i].chargeAndStatus().status.isOneY,
                 hitView[i].chargeAndStatus().status.qBin,
                 hitView[i].clusterSizeX(),
                 hitView[i].clusterSizeY(),
                 hitView[i].detectorIndex());
        }


        // Print debug info for digis -----------------------------------
        if (cms::alpakatools::once_per_grid(acc)) {
          printf("SiPixelDigis Info:\n");
          printf("nDigis = %d\n", digiView.metadata().size());
        }
        for (uint32_t j : cms::alpakatools::uniform_elements(acc, 10)) {
          uint16_t x = digiView[j].xx();
          uint16_t y = digiView[j].yy();
          uint16_t adc = digiView[j].adc();
          printf("Digi %d -> x: %d, y: %d, ADC: %d\n", j, x, y, adc);
        }


        // Print debug info for Clusters -----------------------------------
        if (cms::alpakatools::once_per_grid(acc)) {
          printf("SiPixelClusters Info:\n");
          printf("nClusters = %d\n", clusterView.metadata().size());
        }
        for (uint32_t k : cms::alpakatools::uniform_elements(acc, 10)) {
            printf("Cluster %d -> moduleStart: %d, clusInModule: %d, moduleId: %d, clusModuleStart: %d\n",
                   k,
                   clusterView[k].moduleStart(),
                   clusterView[k].clusInModule(),
                   clusterView[k].moduleId(),
                   clusterView[k].clusModuleStart());
        }

        // Iterate over all clusters (assuming clusters are indexed from 0 to nClusters-1)
        for (uint32_t clusterIdx : cms::alpakatools::uniform_elements(acc, clusterView.metadata().size())) {
            // Temporary storage for cluster properties
            uint32_t minX = std::numeric_limits<uint32_t>::max();
            uint32_t maxX = 0;
            uint32_t minY = std::numeric_limits<uint32_t>::max();
            uint32_t maxY = 0;
            uint32_t totalADC = 0;
            uint32_t numPixels = 0;

            // Iterate over all digis to find those belonging to the current cluster
            for (uint32_t j : cms::alpakatools::uniform_elements(acc, digiView.metadata().size())) {
                if (static_cast<uint32_t>(digiView[j].clus()) == clusterIdx) { // Fixed comparison
                    uint16_t x = digiView[j].xx();
                    uint16_t y = digiView[j].yy();
                    uint16_t adc = digiView[j].adc();

                    // Update cluster properties
                    minX = std::min(minX, (uint32_t)x);
                    maxX = std::max(maxX, (uint32_t)x);
                    minY = std::min(minY, (uint32_t)y);
                    maxY = std::max(maxY, (uint32_t)y);
                    totalADC += adc;
                    numPixels++;
                }
            }

            // Print cluster properties
            if (numPixels > 0) { // Only print clusters that contain pixels
                printf("Cluster %d -> Pixels: %d, Total ADC: %d, Bounds: x[%d-%d], y[%d-%d]\n",
                       clusterIdx, numPixels, totalADC, minX, maxX, minY, maxY);
            }
        }


        // Print debug info for Candidates -----------------------------------
        if (cms::alpakatools::once_per_grid(acc)) {
            printf("Candidate Info:\n");
            printf("nCandidates = %d\n", candidateView.metadata().size());
        }
        // Iterate over the candidates (assuming candidates are indexed from 0 to nCandidates-1)
        for (uint32_t c : cms::alpakatools::uniform_elements(acc, candidateView.metadata().size())) {
            printf("Candidate %d -> px: %.2f, py: %.2f, pz: %.2f, pt: %.2f, eta: %.2f, phi: %.2f\n",
                   c,
                   candidateView[c].px(),
                   candidateView[c].py(),
                   candidateView[c].pz(),
                   candidateView[c].pt(),
                   candidateView[c].eta(),
                   candidateView[c].phi());
        }


        // Print debug info for Vertices -----------------------------------
        if (cms::alpakatools::once_per_grid(acc)) {
          printf("Vertex Info:\n");
          printf("nVertices = %d\n", vertexView.metadata().size());
        }
        for (uint32_t v : cms::alpakatools::uniform_elements(acc, 10)) {
          printf("Vertex %d -> z: %.2f, w: %.2f, chi2: %.2f, pt^2: %.2f, sortedIndex: %d\n",
                 v,
                 vertexView[v].zv(),
                 vertexView[v].wv(),
                 vertexView[v].chi2(),
                 vertexView[v].ptv2(),
                 vertexView[v].sortInd());
        }


        // Print debug info for ClusterGeometry -----------------------------------
        if (cms::alpakatools::once_per_grid(acc)) {
          printf("ClusterGeometry Info:\n");
          printf("nClusterGeometries = %d\n", geoclusterView.metadata().size());
        }
        for (uint32_t g : cms::alpakatools::uniform_elements(acc, 10)) {
          printf("geoclusters %d -> clusterId: %d, pitchX: %.2f, pitchY: %.2f, thickness: %.2f, tanLorentzAngle: %.2f\n",
                 g,
                 geoclusterView[g].clusterIds(),
                 geoclusterView[g].pitchX(),
                 geoclusterView[g].pitchY(),
                 geoclusterView[g].thickness(),
                 geoclusterView[g].tanLorentzAngles());
        }



      }
    };



    template <typename TrackerTraits>
    struct JetSplit {

        // Main operator function
        template <typename TAcc, typename = std::enable_if_t<isAccelerator<TAcc>>>
        ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                      TrackingRecHitSoAConstView<TrackerTraits> hitView,
                                      SiPixelDigisSoAConstView digiView,
                                      SiPixelClustersSoAConstView clusterView,
                                      ZVertexSoAView vertexView,
                                      CandidateSoAView candidateView,
                                      ClusterGeometrySoAView geoclusterView,
                                      double ptMin_,
                                      double deltaR_,
                                      double chargeFracMin_,
                                      float expSizeXAtLorentzAngleIncidence_,
                                      float expSizeXDeltaPerTanAlpha_,
                                      float expSizeYAtNormalIncidence_,
                                      double centralMIPCharge_,
                                      double chargePerUnit_,
                                      double fractionalWidth_,
                                      SiPixelDigisSoAView outputDigis,
                                      SiPixelClustersSoAView outputClusters,
                                      clusterProperties* clusterPropertiesDevice,
                                      uint32_t* clusterCounterDevice,
                                      double forceXError_,
                                      double forceYError_) const {

            // Initialize the clusterCounterDevice to 0 (only on thread 0)
            auto threadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);     
            if (threadIdx == 0) clusterCounterDevice[0] = 0;

            // Iterate over clusters
            for (uint32_t clusterIdx : cms::alpakatools::uniform_elements(acc, clusterView.metadata().size())) {

                // Fetch the cluster's position and geometry
                float pitchX = geoclusterView[clusterIdx].pitchX();
                float pitchY = geoclusterView[clusterIdx].pitchY();
                float thickness = geoclusterView[clusterIdx].thickness();

                // Loop through all candidates (jets)
                for (uint32_t candIdx : cms::alpakatools::uniform_elements(acc, candidateView.metadata().size())) {
                    const auto& jet = candidateView[candIdx];

                    // Skip low-pt jets
                    if (jet.pt() < ptMin_)
                        continue;

                    // Extract jet direction components
                    float jetPx = jet.px();
                    float jetPy = jet.py();
                    float jetPz = jet.pz();

                    // Calculate deltaR directly with scalar values
                    float deltaEta = hitView[clusterIdx].yGlobal() - jetPy;
                    float deltaPhi = hitView[clusterIdx].xGlobal() - jetPx;
                    float deltaR = sqrt(deltaEta * deltaEta + deltaPhi * deltaPhi);

                    // Check deltaR condition and split clusters if applicable
                    if (deltaR < deltaR_) {
                        splitCluster(acc,
                                     hitView,
                                     digiView,
                                     clusterView,
                                     clusterIdx,
                                     jetPx, jetPy, jetPz, 
                                     pitchX, pitchY, thickness,
                                     geoclusterView[clusterIdx].tanLorentzAngles(),
                                     chargeFracMin_,
                                     expSizeXAtLorentzAngleIncidence_,
                                     expSizeXDeltaPerTanAlpha_,
                                     expSizeYAtNormalIncidence_,
                                     centralMIPCharge_,
                                     chargePerUnit_,
                                     fractionalWidth_,
                                     outputDigis,
                                     outputClusters,
                                     clusterPropertiesDevice,
                                     clusterCounterDevice,
                                     forceXError_,
                                     forceYError_);
                    }
                }
            }
        }

        ALPAKA_FN_ACC void closestClusters(clusterProperties* clusterData, int pixelIdx, float& minDist, float& secondMinDist) const {
            minDist = std::numeric_limits<float>::max();
            secondMinDist = std::numeric_limits<float>::max();

            // Loop over all sub-clusters to calculate distance for a specific pixel
            for (uint32_t clusterIdx = 0; clusterIdx < maxSubClusters; clusterIdx++) {
                float dist = clusterData->distanceMap[pixelIdx][clusterIdx];  // Access the distanceMap

                if (dist < minDist) {
                    secondMinDist = minDist;
                    minDist = dist;
                } else if (dist < secondMinDist) {
                    secondMinDist = dist;
                }
            }
        }

        ALPAKA_FN_ACC void secondDistDiffScore(clusterProperties* clusterData) const {
            for (uint32_t pixelIdx = 0; pixelIdx < clusterData->pixelCounter; pixelIdx++) {
                float minDist, secondMinDist;
                // Call closestClusters to calculate minDist and secondMinDist for each pixel
                closestClusters(clusterData, pixelIdx, minDist, secondMinDist);
                clusterData->scoresIndices[pixelIdx] = pixelIdx;
                clusterData->scoresValues[pixelIdx] = secondMinDist - minDist;
            }
        }

        ALPAKA_FN_ACC void secondDistScore(clusterProperties* clusterData) const {
            for (uint32_t pixelIdx = 0; pixelIdx < clusterData->pixelCounter; pixelIdx++) {
                float minDist, secondMinDist;
                // Call closestClusters to calculate minDist and secondMinDist for each pixel
                closestClusters(clusterData, pixelIdx, minDist, secondMinDist);
                clusterData->scoresIndices[pixelIdx] = pixelIdx;
                clusterData->scoresValues[pixelIdx] = -secondMinDist;
            }
        }

        ALPAKA_FN_ACC void distScore(clusterProperties* clusterData) const {
            for (uint32_t pixelIdx = 0; pixelIdx < clusterData->pixelCounter; pixelIdx++) {
                float minDist, secondMinDist;
                // Call closestClusters to calculate minDist and secondMinDist for each pixel
                closestClusters(clusterData, pixelIdx, minDist, secondMinDist);
                clusterData->scoresIndices[pixelIdx] = pixelIdx;
                clusterData->scoresValues[pixelIdx] = -minDist;
            }
        }

        ALPAKA_FN_ACC void sortScores(clusterProperties* clusterData) const {
            for (uint32_t i = 0; i < clusterData->pixelCounter - 1; i++) {
                for (uint32_t j = 0; j < clusterData->pixelCounter - i - 1; j++) {
                    if (clusterData->scoresValues[j] < clusterData->scoresValues[j + 1]) {  // Sort in descending order
                        // Swap scoresValues
                        float tempValue = clusterData->scoresValues[j];
                        clusterData->scoresValues[j] = clusterData->scoresValues[j + 1];
                        clusterData->scoresValues[j + 1] = tempValue;

                        // Swap scoresIndices
                        int tempIndex = clusterData->scoresIndices[j];
                        clusterData->scoresIndices[j] = clusterData->scoresIndices[j + 1];
                        clusterData->scoresIndices[j + 1] = tempIndex;
                    }
                }
            }
        }


        template <typename TAcc, typename = std::enable_if_t<isAccelerator<TAcc>>>        
        ALPAKA_FN_ACC void splitCluster(TAcc const& acc,
                                        TrackingRecHitSoAConstView<TrackerTraits> hitView,
                                        SiPixelDigisSoAConstView digiView,
                                        SiPixelClustersSoAConstView clusterView,
                                        uint32_t clusterIdx,
                                        float jetPx, float jetPy, float jetPz,
                                        float pitchX, float pitchY, float thickness,
                                        float tanLorentzAngles,
                                        double chargeFracMin_,
                                        float expSizeXAtLorentzAngleIncidence_,
                                        float expSizeXDeltaPerTanAlpha_,
                                        float expSizeYAtNormalIncidence_,
                                        double centralMIPCharge_,
                                        double chargePerUnit_,
                                        double fractionalWidth_,
                                        SiPixelDigisSoAView& outputDigi,
                                        SiPixelClustersSoAView& outputClusters,
                                        clusterProperties* clusterPropertiesDevice,
                                        uint32_t* clusterCounterDevice,
                                        double forceXError_,
                                        double forceYError_) const {

            bool split = false;
            float jetTanAlpha = jetPx / jetPz;
            float jetTanBeta = jetPy / jetPz;
            float jetZOverRho = std::sqrt(jetTanAlpha * jetTanAlpha + jetTanBeta * jetTanBeta);

            float expSizeX = expSizeXAtLorentzAngleIncidence_ +
                             std::abs(expSizeXDeltaPerTanAlpha_ * (jetTanAlpha - tanLorentzAngles));
            float expSizeY = std::sqrt((expSizeYAtNormalIncidence_ * expSizeYAtNormalIncidence_) +
                                       thickness * thickness / (pitchY * pitchY) * jetTanBeta * jetTanBeta);

            if (expSizeX < 1.f) expSizeX = 1.f;
            if (expSizeY < 1.f) expSizeY = 1.f;

            float expectedADC = std::sqrt(1.08f + jetZOverRho * jetZOverRho) * centralMIPCharge_;


            if ( hitView[clusterIdx].chargeAndStatus().charge > expectedADC * chargeFracMin_ &&
                   (hitView[clusterIdx].clusterSizeX() > expSizeX + 1 || hitView[clusterIdx].clusterSizeY() > expSizeY + 1)) {
                split = true;
            }

            if (split) {

                // Aligning to the original "fittingSplit" variables
                int sizeY = expSizeY;
                int sizeX = expSizeX;

                unsigned int meanExp = std::floor( hitView[clusterIdx].chargeAndStatus().charge / expectedADC + 0.5f);

                if (meanExp <= 1) {

                    // Iterate over all digis to find those belonging to the current cluster
                    for (uint32_t pixel : cms::alpakatools::uniform_elements(acc, digiView.metadata().size())) {

                        // Use atomicAdd to ensure pixels are added correctly
                        uint32_t idx = alpaka::atomicAdd(acc, clusterCounterDevice, uint32_t(1));
                        if (static_cast<uint32_t>(digiView[pixel].clus()) == clusterIdx) {
                            outputDigi[idx].clus() = clusterIdx;
                            outputDigi[idx].xx() = digiView[pixel].xx();
                            outputDigi[idx].yy() = digiView[pixel].yy();
                            outputDigi[idx].xx() = digiView[pixel].adc();
                            outputDigi[idx].rawIdArr() = digiView[pixel].rawIdArr();
                            outputDigi[idx].moduleId() = digiView[pixel].moduleId();
                        }
                    }
                    return;                    
                }
    

                // Splitting the pixels and writing them for the current clusterIdx
                for (uint32_t j : cms::alpakatools::uniform_elements(acc, digiView.metadata().size())) {
                    // Check if the pixel belongs to the current cluster (clusterIdx)
                    if (static_cast<uint32_t>(digiView[j].clus()) == clusterIdx) {
                        int sub = static_cast<int>(digiView[j].adc()) / chargePerUnit_ * expectedADC / centralMIPCharge_;
                        if (sub < 1) sub = 1;
                        int perDiv = digiView[j].adc() / sub;

                        // Iterate over the sub-clusters (split pixels)
                        for (int k = 0; k < sub; k++) {
                            if (k == sub - 1) perDiv = digiView[j].adc() - perDiv * k;  // Adjust for the last pixel

                            // Use atomicAdd to ensure pixels are added correctly to pixelCounter
                            uint32_t idx = alpaka::atomicAdd(acc, &(clusterPropertiesDevice[clusterIdx].pixelCounter), uint32_t(1));

                            // Write the new split pixels at the obtained index
                            clusterPropertiesDevice[clusterIdx].pixel_X[idx] = digiView[j].xx(); // Copy x-coordinate from original pixel
                            clusterPropertiesDevice[clusterIdx].pixel_Y[idx] = digiView[j].yy(); // Copy y-coordinate from original pixel
                            clusterPropertiesDevice[clusterIdx].pixel_ADC[idx] = perDiv;       // Assign divided charge (ADC)
                            clusterPropertiesDevice[clusterIdx].pixels[idx] = j;
                        }
                    }
                }


                // Compute the initial values, set all distances and centers to -999
                for (unsigned int j = 0; j < meanExp; j++) {
                    clusterPropertiesDevice[clusterIdx].oldclx[j] = -999;
                    clusterPropertiesDevice[clusterIdx].oldcly[j] = -999;
                    clusterPropertiesDevice[clusterIdx].clx[j] = hitView[0].xLocal() + j;
                    clusterPropertiesDevice[clusterIdx].cly[j] = hitView[0].xLocal() + j;
                    clusterPropertiesDevice[clusterIdx].cls[j] = 0;
                }
                bool stop = false;
                int remainingSteps = 100;

                while (!stop && remainingSteps > 0) {
                    remainingSteps--;

                    // Compute distances
                    for (uint32_t j : cms::alpakatools::uniform_elements(acc, digiView.metadata().size())) {
                        if (j >= maxPixels) continue; // Safety check for bounds

                        for (unsigned int i = 0; i < meanExp; i++) {
                            if (i >= maxSubClusters) continue; // Safety check for bounds

                            // Calculate the distance in X and Y for each cluster
                            float distanceX = 1.f * digiView[j].xx() - clusterPropertiesDevice[clusterIdx].clx[i];
                            float distanceY = 1.f * digiView[j].yy() - clusterPropertiesDevice[clusterIdx].cly[i];
                            float dist = 0;

                            if (std::abs(distanceX) > sizeX / 2.f) {
                                dist += (std::abs(distanceX) - sizeX / 2.f + 1.f) * (std::abs(distanceX) - sizeX / 2.f + 1.f);
                            } else {
                                dist += (2.f * distanceX / sizeX) * (2.f * distanceX / sizeX);
                            }

                            if (std::abs(distanceY) > sizeY / 2.f) {
                                dist += (std::abs(distanceY) - sizeY / 2.f + 1.f) * (std::abs(distanceY) - sizeY / 2.f + 1.f);
                            } else {
                                dist += (2.f * distanceY / sizeY) * (2.f * distanceY / sizeY);
                            }

                            // Store the computed distance in the 2D array
                            clusterPropertiesDevice[clusterIdx].distanceMap[j][i] = sqrt(dist);
                        }
                    }

                    secondDistScore(clusterPropertiesDevice);
                    // In the original code:
                    // - the first index is the distance, in whatever metrics we use, 
                    // - the second is the pixel index w.r.t which the distance is computed.
                    //std::multimap < float, int > scores;
                    // In this code the first index is in scoresIndices, the second in scoresValues
                    // to mimic the multimap, I score manually both arrays
                    sortScores(clusterPropertiesDevice);


                    
                    // Iterating over Scores Indices and Values
                    for (unsigned int i = 0; i < clusterPropertiesDevice[clusterIdx].pixelCounter; i++) {

                        int pixel_index = clusterPropertiesDevice[clusterIdx].scoresIndices[i];
                        float score_value = clusterPropertiesDevice[clusterIdx].scoresValues[i];

                        int subpixel_counter = 0;


                        // Iterating over subpixels
                        for (unsigned int subpixel = 0; subpixel < clusterPropertiesDevice[clusterIdx].pixelCounter; subpixel++) {
                            if (clusterPropertiesDevice[clusterIdx].pixels[subpixel] > pixel_index) {
                                break;
                            } else if (clusterPropertiesDevice[clusterIdx].pixels[subpixel] != pixel_index) {
                                continue;
                            } else {
                                float maxEst = 0;
                                int cl = -1;

                                // Iterating over subclusters to calculate the best fit
                                for (unsigned int subcluster_index = 0; subcluster_index < meanExp; subcluster_index++) {
                                    float nsig = (clusterPropertiesDevice[clusterIdx].cls[subcluster_index] - expectedADC) /
                                        (expectedADC * fractionalWidth_); 
                                    float clQest = 1.f / (1.f + std::exp(nsig)) + 1e-6f; 
                                    float clDest = 1.f / (clusterPropertiesDevice[clusterIdx].distanceMap[pixel_index][subcluster_index] + 0.05f);

                                    float est = clQest * clDest;
                                    if (est > maxEst) {
                                        cl = subcluster_index;
                                        maxEst = est;
                                    }
                                }

                                // Use atomicAdd to safely update cls
                                uint32_t idx = alpaka::atomicAdd(acc, &(clusterPropertiesDevice[clusterIdx].cls[cl]), static_cast<float>(clusterPropertiesDevice[clusterIdx].pixel_ADC[subpixel]));

                                // Updating other cluster properties
                                clusterPropertiesDevice[clusterIdx].clusterForPixel[subpixel_counter] = cl;
                                clusterPropertiesDevice[clusterIdx].weightOfPixel[subpixel_counter] = maxEst;
                                subpixel_counter++;
                            }
                        }
                    }


                    // Recompute cluster centers
                    stop = true;
                    for (unsigned int subcluster_index = 0; subcluster_index < meanExp; subcluster_index++) {
                        if (std::abs(clusterPropertiesDevice[clusterIdx].clx[subcluster_index] - clusterPropertiesDevice[clusterIdx].oldclx[subcluster_index]) > 0.01f)
                            stop = false; // still moving
                        if (std::abs(clusterPropertiesDevice[clusterIdx].cly[subcluster_index] - clusterPropertiesDevice[clusterIdx].oldcly[subcluster_index]) > 0.01f)
                            stop = false;
                        clusterPropertiesDevice[clusterIdx].oldclx[subcluster_index] = clusterPropertiesDevice[clusterIdx].clx[subcluster_index];
                        clusterPropertiesDevice[clusterIdx].oldcly[subcluster_index] = clusterPropertiesDevice[clusterIdx].cly[subcluster_index];
                        clusterPropertiesDevice[clusterIdx].clx[subcluster_index] = 0;
                        clusterPropertiesDevice[clusterIdx].cly[subcluster_index] = 0;
                        clusterPropertiesDevice[clusterIdx].cls[subcluster_index] = 1e-99;
                    }


                    for (unsigned int pixel_index = 0; pixel_index < clusterPropertiesDevice[clusterIdx].pixelCounter; pixel_index++) {
                        if (clusterPropertiesDevice[clusterIdx].clusterForPixel[pixel_index] < 0)
                            continue;

                        clusterPropertiesDevice[clusterIdx].clx[clusterPropertiesDevice[clusterIdx].clusterForPixel[pixel_index]] += clusterPropertiesDevice[clusterIdx].pixel_X[pixel_index] * clusterPropertiesDevice[clusterIdx].pixel_ADC[pixel_index];
                        clusterPropertiesDevice[clusterIdx].cly[clusterPropertiesDevice[clusterIdx].clusterForPixel[pixel_index]] += clusterPropertiesDevice[clusterIdx].pixel_Y[pixel_index] * clusterPropertiesDevice[clusterIdx].pixel_ADC[pixel_index];
                        clusterPropertiesDevice[clusterIdx].cls[clusterPropertiesDevice[clusterIdx].clusterForPixel[pixel_index]] += clusterPropertiesDevice[clusterIdx].pixel_ADC[pixel_index];
                    }
                    for (unsigned int subcluster_index = 0; subcluster_index < meanExp; subcluster_index++) {
                        if (clusterPropertiesDevice[clusterIdx].cls[subcluster_index] != 0) {
                            clusterPropertiesDevice[clusterIdx].clx[subcluster_index] /= clusterPropertiesDevice[clusterIdx].cls[subcluster_index];
                            clusterPropertiesDevice[clusterIdx].cly[subcluster_index] /= clusterPropertiesDevice[clusterIdx].cls[subcluster_index];
                        }
                        clusterPropertiesDevice[clusterIdx].cls[subcluster_index] = 0;
                    }
                }

                // accumulate pixel with same cl
                for (int cl = 0; cl < (int) meanExp; cl++) {
                    for (unsigned int j = 0; j < clusterPropertiesDevice[clusterIdx].pixelCounter; j++) {
                        if (clusterPropertiesDevice[clusterIdx].clusterForPixel[j] == cl and clusterPropertiesDevice[clusterIdx].pixel_ADC[j] != 0) {

                            // cl find the other pixels
                            // with same x,y and
                            // accumulate+reset their adc
                            for (unsigned int k = j + 1; k < clusterPropertiesDevice[clusterIdx].pixelCounter; k++) {
                                if (clusterPropertiesDevice[clusterIdx].pixel_ADC[k] != 0 and clusterPropertiesDevice[clusterIdx].pixel_X[k] == clusterPropertiesDevice[clusterIdx].pixel_X[j] and clusterPropertiesDevice[clusterIdx].pixel_Y[k] == clusterPropertiesDevice[clusterIdx].pixel_Y[j] and clusterPropertiesDevice[clusterIdx].clusterForPixel[k] == cl) {
                                    clusterPropertiesDevice[clusterIdx].pixel_ADC[j] += clusterPropertiesDevice[clusterIdx].pixel_ADC[k];
                                    clusterPropertiesDevice[clusterIdx].pixel_ADC[k] = 0;
                                }
                            }

                            //increase pixelsForClCounter and copy pixels into pixelsForCl
                            uint32_t idx = alpaka::atomicAdd(acc, &(clusterPropertiesDevice[clusterIdx].pixelsForClCounter), uint32_t(1));

                            clusterPropertiesDevice[clusterIdx].pixelsForCl_X[cl] = clusterPropertiesDevice[clusterIdx].pixel_X[j];
                            clusterPropertiesDevice[clusterIdx].pixelsForCl_Y[cl] = clusterPropertiesDevice[clusterIdx].pixel_Y[j];
                            clusterPropertiesDevice[clusterIdx].pixelsForCl_ADC[cl] = clusterPropertiesDevice[clusterIdx].pixel_ADC[j];
                        }
                    }
                }

                for (int cl = 0; cl < (int) meanExp; cl++) {
                    uint32_t idx = alpaka::atomicAdd(acc, clusterCounterDevice, uint32_t(1));
                    for (unsigned int j = 0; j < clusterPropertiesDevice[clusterIdx].pixelsForClCounter; j++) {
                        outputDigi[idx].clus() = clusterIdx;
                        outputDigi[idx].xx() = clusterPropertiesDevice[cl].pixelsForCl_X[j];
                        outputDigi[idx].yy() = clusterPropertiesDevice[cl].pixelsForCl_Y[j];
                        outputDigi[idx].adc() = clusterPropertiesDevice[cl].pixelsForCl_ADC[j];
                        outputDigi[idx].rawIdArr() = 0; // Copy raw ID from original pixel
                        outputDigi[idx].moduleId() = 0; // Copy module ID from original pixel
                    }            
                }
            }
        }
    };



    template <typename TrackerTraits>
    void runKernels(TrackingRecHitSoAView<TrackerTraits>& hitView,
                    SiPixelDigisSoAView& digiView,
                    SiPixelClustersSoAView& clusterView,
                    ZVertexSoAView& vertexView,
                    CandidateSoAView& candidateView,
                    ClusterGeometrySoAView& geoclusterView,
                    double ptMin_,
                    double deltaR_,
                    double chargeFracMin_,
                    float expSizeXAtLorentzAngleIncidence_,
                    float expSizeXDeltaPerTanAlpha_,
                    float expSizeYAtNormalIncidence_,
                    double centralMIPCharge_,
                    double chargePerUnit_,
                    double fractionalWidth_,
                    SiPixelDigisSoAView& outputDigis,                    
                    SiPixelClustersSoAView& outputClusters,
                    clusterProperties* clusterPropertiesDevice,
                    uint32_t* clusterCounterDevice,
                    double forceXError_,
                    double forceYError_,
                    Queue& queue) {


                uint32_t items = 64;
                uint32_t groupsHits = divide_up_by(hitView.metadata().size(), items);
                uint32_t groupsDigis = divide_up_by(digiView.metadata().size(), items);
                uint32_t groupsClusters = divide_up_by(clusterView.metadata().size(), items);

                uint32_t groups = std::max({groupsHits, groupsDigis, groupsClusters});

                auto workDiv = make_workdiv<Acc1D>(groups, items);

                // Kernel executions
                //alpaka::exec<Acc1D>(queue, workDiv, Printout<TrackerTraits>{}, hitView, digiView, clusterView, vertexView, candidateView, geoclusterView);
                alpaka::exec<Acc1D>(queue, 
                                    workDiv, 
                                    JetSplit<TrackerTraits>{}, 
                                    hitView, 
                                    digiView, 
                                    clusterView, 
                                    vertexView, 
                                    candidateView, 
                                    geoclusterView,
                                    ptMin_,
                                    deltaR_,
                                    chargeFracMin_,
                                    expSizeXAtLorentzAngleIncidence_,
                                    expSizeXDeltaPerTanAlpha_,
                                    expSizeYAtNormalIncidence_,
                                    centralMIPCharge_,
                                    chargePerUnit_,
                                    fractionalWidth_,
                                    outputDigis,
                                    outputClusters,
                                    clusterPropertiesDevice,
                                    clusterCounterDevice,
                                    forceXError_,
                                    forceYError_);
            }





    // Explicit template instantiation for Phase 1
    template void runKernels<pixelTopology::Phase1>(TrackingRecHitSoAView<pixelTopology::Phase1>& hitView,
                                                    SiPixelDigisSoAView& digiView,
                                                    SiPixelClustersSoAView& clusterView,
                                                    ZVertexSoAView& vertexView,
                                                    CandidateSoAView& candidateView,
                                                    ClusterGeometrySoAView& geoclusterView,
                                                    double ptMin_,
                                                    double deltaR_,
                                                    double chargeFracMin_,
                                                    float expSizeXAtLorentzAngleIncidence_,
                                                    float expSizeXDeltaPerTanAlpha_,
                                                    float expSizeYAtNormalIncidence_,
                                                    double centralMIPCharge_,
                                                    double chargePerUnit_,
                                                    double fractionalWidth_,
                                                    SiPixelDigisSoAView& outputDigis,
                                                    SiPixelClustersSoAView& outputClusters,
                                                    clusterProperties* clusterPropertiesDevice,
                                                    uint32_t* clusterCounterDevice,
                                                    double forceXError_,
                                                    double forceYError_,
                                                    Queue& queue);

    // Explicit template instantiation for Phase 2
    template void runKernels<pixelTopology::Phase2>(TrackingRecHitSoAView<pixelTopology::Phase2>& hitView,
                                                    SiPixelDigisSoAView& digiView,
                                                    SiPixelClustersSoAView& clusterView,
                                                    ZVertexSoAView& vertexView,
                                                    CandidateSoAView& candidateView,
                                                    ClusterGeometrySoAView& geoclusterView,
                                                    double ptMin_,
                                                    double deltaR_,
                                                    double chargeFracMin_,
                                                    float expSizeXAtLorentzAngleIncidence_,
                                                    float expSizeXDeltaPerTanAlpha_,
                                                    float expSizeYAtNormalIncidence_,
                                                    double centralMIPCharge_,
                                                    double chargePerUnit_,
                                                    double fractionalWidth_,
                                                    SiPixelDigisSoAView& outputDigis,
                                                    SiPixelClustersSoAView& outputClusters,
                                                    clusterProperties* clusterPropertiesDevice,
                                                    uint32_t* clusterCounterDevice, 
                                                    double forceXError_,
                                                    double forceYError_,
                                                    Queue& queue);

    
  }  // namespace Splitting
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE