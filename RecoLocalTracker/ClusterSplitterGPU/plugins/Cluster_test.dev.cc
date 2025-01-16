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

#include "DataFormats/ClusterGeometrySoA/interface/ClusterGeometrysSoA.h"
#include "DataFormats/ClusterGeometrySoA/interface/alpaka/ClusterGeometrysSoACollection.h"

#include "DataFormats/CandidateSoA/interface/CandidatesSoA.h"
#include "DataFormats/CandidateSoA/interface/alpaka/CandidatesSoACollection.h"

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
                                CandidatesSoAView candidateView,
                                ClusterGeometrysSoAView geoclusterView) const {         
 

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
                                      CandidatesSoAView candidateView,
                                      ClusterGeometrysSoAView geoclusterView,
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

            //inside jetsplit
            // Get thread and grid indices
            auto threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; // Thread index within the block
            auto blockIdx  = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];   // Block index
            auto blockDim  = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]; // Threads per block

            // Compute the global thread ID
            uint32_t globalThreadId = blockIdx * blockDim + threadIdx;

            //const char* info = "TESTA";
            //if (globalThreadId == 0) printDebug(acc, digiView, clusterView, info);
            //if (globalThreadId == 0) printf("Cl_SoA entry = %u\nCl_SoA moduleStart = %u\nCl_SoA clusInModule = %u\nCl_SoA moduleId = %u\nCl_SoA clusModuleStart = %u\n----\n", n, clusterView.moduleStart(n), clusterView.clusInModule(n), clusterView.moduleId(n), clusterView.clusModuleStart(n));


/*
            // Debugging printout ---------------------------------
            int CalculatedClusters = 0;
            if (globalThreadId == 0) {

                for (int n = 0; n < static_cast<int>(clusterView.metadata().size()); n++) {
                    for (uint32_t foundCluster = 0; foundCluster < clusterView.clusInModule(n); foundCluster++) {
                        for (uint32_t pixel = 0; pixel < static_cast<uint32_t>(digiView.metadata().size()); pixel++) {
                            if ( clusterView.moduleId(n) == digiView.moduleId(pixel) ) {
                                if (static_cast<int>(foundCluster) == digiView.clus(pixel)) {
                                    printf("Module = %u ", clusterView.moduleId(n) );
                                    printf("Cluster = %u ", foundCluster);
                                    printf("Pixel = %u ", pixel);
                                }
                            }
                        }
                    }
                }
            }
            //printf("CalculatedClusters = %d ", CalculatedClusters);            
            // Debugging printout ---------------------------------
*/

            // Get total Clusters and Candidates
            uint32_t numClusters = static_cast<uint32_t>(geoclusterView.metadata().size());
            uint32_t numCandidates = static_cast<uint32_t>(candidateView.metadata().size());

            // Ensure only valid threads process clusters
            //if (globalThreadId < numClusters-2) {
            if ( globalThreadId == 841 ) {

                if ( globalThreadId == 0 ) {
                    *clusterCounterDevice = 0;
                }

                uint32_t clusterIdx = globalThreadId;      // Each thread handles exactly one cluster
                uint32_t moduleId = geoclusterView.moduleId(clusterIdx);
                uint32_t clusterOffset = geoclusterView.clusterOffset(clusterIdx);
/*
                // This following approach resulted wrong because for multiple clusters would belong to a module
                // rely on the above 2 lines!
                for (uint32_t n = 0; n < static_cast<uint32_t>(clusterView.metadata().size()); n++) {
                    uint32_t clustersInModule = clusterView.clusInModule(n);  // Number of clusters in this module

                    // Debugging print to see the module size and current global index
                    printf("n = %u, cumulativeClusters = %u, clustersInModule = %u, clusterIdx = %u, moduleId = %u\n", 
                           n, cumulativeClusters, clustersInModule, clusterIdx, clusterView.moduleId(n));

                    // If clustersInModule is non-zero, check the range for clusterIdx
                    if (clustersInModule > 0) {
                        // Check if the current global index falls within the current module's cluster range
                        if (clusterIdx >= cumulativeClusters && clusterIdx < cumulativeClusters + clustersInModule) {
                            moduleId = clusterView.moduleId(n); // Set the module index for this global cluster
                            clusterOffset = clusterIdx - cumulativeClusters; // Local index of cluster within the module
                            printf("Found cluster %u in module %u, clusterOffset = %u\n", clusterIdx, moduleId, clusterOffset);
                            break; // Once found, break out of the loop
                        }
                    }
                    
                    // Update cumulativeClusters after checking the module
                    cumulativeClusters += clustersInModule;  // Update cumulative clusters count
                }
*/

                // Now we have:
                // - `moduleId`: The module this cluster belongs to
                // - `clusterOffset`: The cluster number within that module
                printf("I am in thread %u, analyzing cluster %u from module %u\n", 
                       globalThreadId, clusterOffset, moduleId);


                //printf("About to run over %u, Candidates\n", numCandidates);

                for (uint32_t candIdx = 0; candIdx < numCandidates; ++candIdx) {
                    //printf("Processing Cluster: %u, Candidate: %u/%u Block index: %u, Threads per block: %u, Total threads: %u\n",
                    //    clusterIdx, candIdx, numCandidates-1, blockIdx, blockDim, blockDim * alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u]);

                    // Debugging Candidate to be compared to the one originated in the other producer
                    //double testme = static_cast<double>(candidateView[candIdx].px());
                    //printf("Candidate %u px= %f \n", candIdx, testme);   

                    // Skip low-pt jets
                    if (candidateView.pt(candIdx) < ptMin_) {
                        //printf("SKIP: Candidates has low pt %f \n", candidateView.pt(candIdx));
                        return;
                    }

                    // Extract jet direction components correctly
                    float jetPx = candidateView.px(candIdx);
                    float jetPy = candidateView.py(candIdx);
                    float jetPz = candidateView.pz(candIdx);

                    // Compute jet eta and phi
                    float jetEta = 0.5 * log((sqrt(jetPx * jetPx + jetPy * jetPy + jetPz * jetPz) + jetPz) /
                                              (sqrt(jetPx * jetPx + jetPy * jetPy + jetPz * jetPz) - jetPz));
                    float jetPhi = atan2(jetPy, jetPx);

                    float clusterEta;
                    float clusterPhi;

                    // Access hit global positions (Rechits are indexed just like clustergeo)
                    float x = hitView.xGlobal(clusterIdx);
                    float y = hitView.yGlobal(clusterIdx);
                    float z = hitView.zGlobal(clusterIdx);

                    // Compute eta and phi
                    float r = sqrt(x * x + y * y + z * z);
                    clusterEta = 0.5 * log((r + z) / (r - z));
                    clusterPhi = atan2(y, x);

                    // Compute deltaR properly
                    float deltaEta = clusterEta - jetEta;
                    float deltaPhi = atan2(sin(clusterPhi - jetPhi), cos(clusterPhi - jetPhi)); // Proper phi difference handling
                    float deltaR = sqrt(deltaEta * deltaEta + deltaPhi * deltaPhi);

                    //printf("deltaR = %f  deltaR_ = %f", deltaR, deltaR_);

                    // Check deltaR condition and split clusters if applicable
                    if (deltaR < deltaR_) {
                        printf("This cluster: %u has deltaR < deltaR_ and it might be split\n",clusterIdx);

                        splitCluster(acc,
                                     hitView,
                                     digiView,
                                     clusterView,
                                     clusterIdx,
                                     moduleId,
                                     clusterOffset,
                                     jetPx, jetPy, jetPz, 
                                     geoclusterView,                                     
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
                    else {
                        storeOutputDigis(acc, digiView, outputDigis, moduleId, clusterOffset, clusterCounterDevice);
                    }
                }
            }
            else {
                return;
            }

            //info = "TESTB";
            //if (globalThreadId == 0) printDebug(acc, digiView, clusterView, info);

        }


        ALPAKA_FN_ACC void closestClusters(clusterProperties* clusterData, uint32_t clusterIdx, int pixelIdx, float& minDist, float& secondMinDist, unsigned meanExp) const {
            minDist = std::numeric_limits<float>::max();
            secondMinDist = std::numeric_limits<float>::max();

            // Loop over all sub-clusters to calculate distance for a specific pixel
            for (uint32_t subClusterIdx = 0; subClusterIdx < meanExp; subClusterIdx++) {
                float dist = clusterData[clusterIdx].distanceMap[pixelIdx][subClusterIdx];  // Access the distanceMap

                // Debug print for each sub-cluster iteration
                printf("DEBUG: pixelIdx=%d, subClusterIdx=%u, dist=%f, current minDist=%f, current secondMinDist=%f\n",
                        pixelIdx, subClusterIdx, dist, minDist, secondMinDist);

                if (dist < minDist) {
                    secondMinDist = minDist;
                    minDist = dist;
                } else if (dist < secondMinDist) {
                    secondMinDist = dist;
                }
            }
        }

        ALPAKA_FN_ACC void secondDistDiffScore(clusterProperties* clusterData, uint32_t clusterIdx, unsigned meanExp) const {
            for (uint32_t pixelIdx = 0; pixelIdx < clusterData[clusterIdx].pixelCounter; pixelIdx++) {
                float minDist, secondMinDist;
                // Call closestClusters to calculate minDist and secondMinDist for each pixel
                closestClusters(clusterData, clusterIdx, pixelIdx, minDist, secondMinDist, meanExp);
                clusterData[clusterIdx].scoresIndices[pixelIdx] = pixelIdx;
                clusterData[clusterIdx].scoresValues[pixelIdx] = secondMinDist - minDist;
            }
        }

        ALPAKA_FN_ACC void secondDistScore(clusterProperties* clusterData, uint32_t clusterIdx, unsigned meanExp) const {
            printf("running with clusterData[clusterIdx].pixelCounter=: %u\n", clusterData[clusterIdx].pixelCounter);

            for (uint32_t pixelIdx = 0; pixelIdx < clusterData[clusterIdx].pixelCounter; pixelIdx++) {
                if ( pixelIdx < maxPixels ) {
                    float minDist, secondMinDist;
                    // Call closestClusters to calculate minDist and secondMinDist for each pixel
                    closestClusters(clusterData, clusterIdx, pixelIdx, minDist, secondMinDist, meanExp);
                    clusterData[clusterIdx].scoresIndices[pixelIdx] = pixelIdx;
                    clusterData[clusterIdx].scoresValues[pixelIdx] = -secondMinDist;
                }
                else {
                    printf("ERROR@ secondDistScore: pixelIdx (%u) exceeds maxPixels (%d)\n", pixelIdx, maxPixels);
                    return;
                }

            }
        }

        ALPAKA_FN_ACC void distScore(clusterProperties* clusterData, uint32_t clusterIdx, unsigned meanExp) const {
            for (uint32_t pixelIdx = 0; pixelIdx < clusterData[clusterIdx].pixelCounter; pixelIdx++) {
                float minDist, secondMinDist;
                // Call closestClusters to calculate minDist and secondMinDist for each pixel
                closestClusters(clusterData, pixelIdx, minDist, secondMinDist, meanExp);
                clusterData[clusterIdx].scoresIndices[pixelIdx] = pixelIdx;
                clusterData[clusterIdx].scoresValues[pixelIdx] = -minDist;
            }
        }

        ALPAKA_FN_ACC void sortScores(clusterProperties* clusterData, uint32_t clusterIdx) const {
            //printf("Pixel counter: %u\n", clusterData[clusterIdx].pixelCounter);

            for (uint32_t i = 0; i < clusterData[clusterIdx].pixelCounter - 1; i++) {
                for (uint32_t j = 0; j < clusterData[clusterIdx].pixelCounter - i - 2; j++) {

                    if (j >= maxPixels-1) {
                        printf("ERROR@ sortScores: j (%u) exceeds maxPixels (%d)\n", j, maxPixels);
                        return;
                    }

                    else {
                        //printf("clusterData[clusterIdx].scoresValues: %u, %f\n", j, clusterData[clusterIdx].scoresValues[j+1]);

                        if (clusterData[clusterIdx].scoresValues[j] < clusterData[clusterIdx].scoresValues[j + 1]) {  // Sort in descending order
                            // Swap scoresValues
                            float tempValue = clusterData[clusterIdx].scoresValues[j];
                            clusterData[clusterIdx].scoresValues[j] = clusterData[clusterIdx].scoresValues[j + 1];
                            clusterData[clusterIdx].scoresValues[j + 1] = tempValue;

                            // Swap scoresIndices
                            int tempIndex = clusterData[clusterIdx].scoresIndices[j];
                            clusterData[clusterIdx].scoresIndices[j] = clusterData[clusterIdx].scoresIndices[j + 1];
                            clusterData[clusterIdx].scoresIndices[j + 1] = tempIndex;
                        }
                    }
                }
            }
        // Print out the sorted scores
          for (uint32_t k = 0; k < clusterData[clusterIdx].pixelCounter; k++) {
            printf("After sort: score[%u] = %f, index = %d\n",
                   k,
                   clusterData[clusterIdx].scoresValues[k],
                   clusterData[clusterIdx].scoresIndices[k]);
          }                
            
        }




        template <typename TAcc, typename = std::enable_if_t<isAccelerator<TAcc>>>        
        ALPAKA_FN_ACC void printDebug(
            TAcc const& acc,
            const SiPixelDigisSoAConstView digiView,
            const SiPixelClustersSoAConstView clusterView,
            const char* info) const {

            // Debugging printout ---------------------------------
            //int CalculatedClusters = 0;

            for (int n = 0; n < static_cast<int>(clusterView.metadata().size()); n++) {
                for (uint32_t foundCluster = 0; foundCluster < clusterView.clusInModule(n); foundCluster++) {
                    for (uint32_t pixel = 0; pixel < static_cast<uint32_t>(digiView.metadata().size()); pixel++) {
                        if ( clusterView.moduleId(n) == digiView.moduleId(pixel) ) {
                            if (static_cast<int>(foundCluster) == digiView.clus(pixel)) {
                                printf("%s Module = %u ", info, clusterView.moduleId(n) );
                                printf("%s Cluster = %u ",info, foundCluster);
                                printf("%s Pixel = %u \n", info, pixel);
                            }
                        }
                    }
                }
            }
            //printf("CalculatedClusters = %d ", CalculatedClusters);            
            // Debugging printout ---------------------------------
        }



        template <typename TAcc, typename = std::enable_if_t<isAccelerator<TAcc>>>        
        ALPAKA_FN_ACC void storeOutputDigis(
            TAcc const& acc,
            const SiPixelDigisSoAConstView digiView,
            SiPixelDigisSoAView outputDigis,
            uint32_t moduleId,
            uint32_t clusterOffset,
            uint32_t* clusterCounterDevice) const {

            // Iterate over all digis to find those belonging to the current cluster

            uint32_t idx = alpaka::atomicAdd(acc, clusterCounterDevice, uint32_t(0));
            uint32_t storeIdx = idx;

            //printf("AtomicAdd result: %u \n", idx);
            //printf("DigiView size: %u\n", static_cast<uint32_t>(digiView.metadata().size()));
            //printf("output size: %u\n", static_cast<uint32_t>(outputDigis.metadata().size()));

            // Reminder:
            // - `moduleId`: The module this cluster belongs to
            // - `clusterOffset`: The cluster number within that module
            for (uint32_t pixel = 0; pixel < static_cast<uint32_t>(digiView.metadata().size()); pixel++) {
                if ( moduleId == static_cast<uint32_t>(digiView.moduleId(pixel) )) {
                    if ( clusterOffset == static_cast<uint32_t>(digiView.clus(pixel) )) {

                        if (storeIdx >= static_cast<uint32_t>(outputDigis.metadata().size())) {
                            printf("ERROR: storeIdx %u out of bounds (max %u)\n", storeIdx, outputDigis.metadata().size());
                            return;  // Prevent out-of-bounds write
                        }
                        outputDigis.clus(storeIdx) = digiView.clus(pixel);
                        outputDigis.xx(storeIdx) = digiView.xx(pixel);
                        outputDigis.yy(storeIdx) = digiView.yy(pixel);
                        outputDigis.adc(storeIdx) = digiView.adc(pixel);
                        outputDigis.rawIdArr(storeIdx) = digiView.rawIdArr(pixel);
                        outputDigis.moduleId(storeIdx) = digiView.moduleId(pixel);

                        // Store Digi information in output
                        storeIdx++; // Using pixel as offset
                        //printf("AtomicAdd result: %u storeIdx: %u\n", idx, storeIdx);
                    }                                   
                }
            }
            // Use atomicAdd to ensure pixels are added correctly
            idx = alpaka::atomicAdd(acc, clusterCounterDevice, uint32_t(1));

        }



        template <typename TAcc, typename = std::enable_if_t<isAccelerator<TAcc>>>        
        ALPAKA_FN_ACC void splitCluster(TAcc const& acc,
                                        TrackingRecHitSoAConstView<TrackerTraits> hitView,
                                        SiPixelDigisSoAConstView digiView,
                                        SiPixelClustersSoAConstView clusterView,
                                        uint32_t clusterIdx,
                                        uint32_t moduleId,
                                        uint32_t clusterOffset,  
                                        float jetPx, float jetPy, float jetPz,
                                        ClusterGeometrysSoAView geoclusterView,
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
                                        double forceYError_) const {

            //printf("This cluster: %u now processed in SplitCluster routine\n",clusterIdx);

            bool split = false;

            float pitchX = geoclusterView.pitchX(clusterIdx);
            float pitchY = geoclusterView.pitchY(clusterIdx);
            float thickness = geoclusterView.thickness(clusterIdx);
            float tanLorentzAngles = geoclusterView.tanLorentzAngles(clusterIdx);

            // Apply precomputed transformation matrix
            float jetDirLocalX = geoclusterView.transformXX(clusterIdx) * jetPx + geoclusterView.transformXY(clusterIdx) * jetPy + geoclusterView.transformXZ(clusterIdx) * jetPz;
            float jetDirLocalY = geoclusterView.transformYX(clusterIdx) * jetPx + geoclusterView.transformYY(clusterIdx) * jetPy + geoclusterView.transformYZ(clusterIdx) * jetPz;
            float jetDirLocalZ = geoclusterView.transformZX(clusterIdx) * jetPx + geoclusterView.transformZY(clusterIdx) * jetPy + geoclusterView.transformZZ(clusterIdx) * jetPz;

            // Now, proceed with your calculations
            float jetTanAlpha = jetDirLocalX / jetDirLocalZ;
            float jetTanBeta = jetDirLocalY / jetDirLocalZ;

            float jetZOverRho = std::sqrt(jetTanAlpha * jetTanAlpha + jetTanBeta * jetTanBeta);

            float expSizeX = expSizeXAtLorentzAngleIncidence_ +
                             std::abs(expSizeXDeltaPerTanAlpha_ * (jetTanAlpha - tanLorentzAngles));
            float expSizeY = std::sqrt((expSizeYAtNormalIncidence_ * expSizeYAtNormalIncidence_) +
                                       thickness * thickness / (pitchY * pitchY) * jetTanBeta * jetTanBeta);

            if (expSizeX < 1.f) expSizeX = 1.f;
            if (expSizeY < 1.f) expSizeY = 1.f;

            float expectedADC = std::sqrt(1.08f + jetZOverRho * jetZOverRho) * centralMIPCharge_;

            if ( hitView.chargeAndStatus(clusterIdx).charge > expectedADC * chargeFracMin_ &&
                   (hitView.clusterSizeX(clusterIdx) > expSizeX + 1 || hitView.clusterSizeY(clusterIdx) > expSizeY + 1)) {
                split = true;
            }

            if (split) {

                // Aligning to the original "fittingSplit" variables..
                int sizeY = expSizeY;
                int sizeX = expSizeX;

                unsigned int meanExp = std::floor( hitView.chargeAndStatus(clusterIdx).charge / expectedADC + 0.5f);

                if (meanExp <= 1) {
                    printf("--------------------------");
                    printf("meanExp <= 1 writing cluster %u", clusterIdx);
                    storeOutputDigis(acc, digiView, outputDigis, moduleId, clusterOffset, clusterCounterDevice);
                }
                else {
                    // Splitting the pixels and writing them for the current clusterIdx
                    printf("cluster %u has meanExp %d\n", clusterIdx, meanExp);


                    // Copy the original pixels into a fixed array for simpler handling (and sequential access)
                    // as in the original SoA pixels have whatever index j
                    clusterPropertiesDevice[clusterIdx].pixelCounter = 0;
                    for (uint32_t j = 0; j < static_cast<uint32_t>(digiView.metadata().size()); ++j) {
                        if (clusterView.moduleId(moduleId) == digiView.moduleId(j)) {
                            if (static_cast<int>(clusterOffset) == digiView.clus(j)) {
                                clusterPropertiesDevice[clusterIdx].pixelCounter++;                                    
                                    clusterPropertiesDevice[clusterIdx].originalpixels_x[ clusterPropertiesDevice[clusterIdx].pixelCounter ] = digiView.xx(j); // Copy x-coordinate from original pixel
                                    clusterPropertiesDevice[clusterIdx].originalpixels_y[ clusterPropertiesDevice[clusterIdx].pixelCounter ] = digiView.yy(j); // Copy y-coordinate from original pixel
                                    clusterPropertiesDevice[clusterIdx].originalpixels_ADC[ clusterPropertiesDevice[clusterIdx].pixelCounter ] = digiView.adc(j);
                                    clusterPropertiesDevice[clusterIdx].originalpixels_rawIdArr[j] = digiView.rawIdArr(j);
                            }
                        }
                    }


                    for (uint32_t j = 0; j < clusterPropertiesDevice[clusterIdx].pixelCounter; ++j) {

                        int sub = static_cast<int>(clusterPropertiesDevice[clusterIdx].originalpixels_ADC[j]) / chargePerUnit_ * expectedADC / centralMIPCharge_;
                        if (sub < 1) sub = 1;

                        int perDiv = clusterPropertiesDevice[clusterIdx].originalpixels_ADC[j] / sub;

                        printf("digi has adc=%f pixelcounter=%u   sub=%d   perDiv = %d\n", clusterPropertiesDevice[clusterIdx].originalpixels_ADC[j], clusterPropertiesDevice[clusterIdx].pixelCounter, sub, perDiv);

                        // Iterate over the sub-clusters (split pixels)
                        for (int k = 0; k < sub; k++) {
                            if (k == sub - 1) perDiv = clusterPropertiesDevice[clusterIdx].originalpixels_ADC[j] - perDiv * k;  // Adjust for the last pixel

                            printf("Write pixel %u to the cluster %u \n", k, clusterIdx);
                            // Write the new split pixels at the obtained index
                            clusterPropertiesDevice[clusterIdx].pixels[k] = j;                                    
                            clusterPropertiesDevice[clusterIdx].pixel_X[k] = clusterPropertiesDevice[clusterIdx].originalpixels_x[j]; // Copy x-coordinate from original pixel
                            clusterPropertiesDevice[clusterIdx].pixel_Y[k] = clusterPropertiesDevice[clusterIdx].originalpixels_y[j]; // Copy y-coordinate from original pixel
                            clusterPropertiesDevice[clusterIdx].pixel_ADC[k] = perDiv;       // Assign divided charge (ADC)
                            clusterPropertiesDevice[clusterIdx].rawIdArr[k] = clusterPropertiesDevice[clusterIdx].originalpixels_rawIdArr[j]; // Copy rawIdArr from original pixel
                        }
                    }
                    printf("Computing initial values, set all distances");
                    // Compute the initial values, set all distances and centers to -999
                    for (unsigned int j = 0; j < meanExp; j++) {
                        clusterPropertiesDevice[clusterIdx].oldclx[j] = -999;
                        clusterPropertiesDevice[clusterIdx].oldcly[j] = -999;
                        clusterPropertiesDevice[clusterIdx].clx[j] = clusterPropertiesDevice[clusterIdx].originalpixels_x[0] + j;
                        clusterPropertiesDevice[clusterIdx].cly[j] = clusterPropertiesDevice[clusterIdx].originalpixels_y[0]  + j;
                        clusterPropertiesDevice[clusterIdx].cls[j] = 0;
                    }
                    bool stop = false;
                    int remainingSteps = 100;

//here check

                    while (!stop && remainingSteps > 0) {
                        remainingSteps--;
                        printf("Remaining steps: %d\n", remainingSteps);

                        // Compute distances
                        for (uint32_t j = 0; j < clusterPropertiesDevice[clusterIdx].pixelCounter; ++j) {

                            for (unsigned int i = 0; i < meanExp; i++) {
                                //if (i >= maxSubClusters) continue; // Safety check for bounds

                                // Calculate the distance in X and Y for each pixel
                                float distanceX = 1.f * clusterPropertiesDevice[clusterIdx].originalpixels_x[j] - clusterPropertiesDevice[clusterIdx].clx[i];
                                float distanceY = 1.f * clusterPropertiesDevice[clusterIdx].originalpixels_y[j] - clusterPropertiesDevice[clusterIdx].cly[i];
                                float dist = 0;
                                printf("i=%u, distanceX = %f, distanceY = %f\n", i, distanceX, distanceY);
                                
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
                                printf("dist = %f\n", dist);

                                // Store the computed distance in the 2D array
                                clusterPropertiesDevice[clusterIdx].distanceMap[j][i] = sqrt(dist);
                                printf("clusterIdx=%u distanceMap[%u][%u] = %f\n", clusterIdx, j, i, clusterPropertiesDevice[clusterIdx].distanceMap[j][i]);
                            }
                        } // compute distances done

                        printf("About to calculate secondDistScore\n");
                        secondDistScore(clusterPropertiesDevice, clusterIdx, meanExp);
                        
                        // In the original code:
                        // - the first index is the distance, in whatever metrics we use, 
                        // - the second is the pixel index w.r.t which the distance is computed.
                        //std::multimap < float, int > scores;
                        // In this code the first index is in scoresIndices, the second in scoresValues
                        // to mimic the multimap, I score manually both arrays
                        sortScores(clusterPropertiesDevice, clusterIdx);


                        // Iterating over Scores Indices and Values
                        for (unsigned int i = 0; i < clusterPropertiesDevice[clusterIdx].pixelCounter; i++) {
                            if (i < maxPixels) {
                                int pixel_index = clusterPropertiesDevice[clusterIdx].scoresIndices[i];
                                //float score_value = clusterPropertiesDevice[clusterIdx].scoresValues[i];

                                int subpixel_counter = 0;

                                // Iterating over subpixels
                                for (unsigned int subpixel = 0; subpixel < clusterPropertiesDevice[clusterIdx].pixelCounter; subpixel++) {
                                    if (subpixel< maxPixels) {

                                        if (clusterPropertiesDevice[clusterIdx].pixels[subpixel] > pixel_index) {
                                            break;
                                        } else if (clusterPropertiesDevice[clusterIdx].pixels[subpixel] != pixel_index) {
                                            continue;
                                        } else {
                                            float maxEst = 0;
                                            int cl = -1;

                                            // Iterating over subclusters to calculate the best fit
                                            for (unsigned int subcluster_index = 0; subcluster_index < meanExp; subcluster_index++) {
                                                if (subcluster_index< maxSubClusters) {
                                                    float nsig = (clusterPropertiesDevice[clusterIdx].cls[subcluster_index] - expectedADC) /
                                                        (expectedADC * fractionalWidth_); 
                                                    float clQest = 1.f / (1.f + std::exp(nsig)) + 1e-6f; 
                                                    float clDest = 1.f / (clusterPropertiesDevice[clusterIdx].distanceMap[pixel_index][subcluster_index] + 0.05f);

                                                    float est = clQest * clDest;
                                                    if (est > maxEst) {
                                                        cl = subcluster_index;
                                                        maxEst = est;
                                                        //printf("cl = %d",cl);
                                                    }
                                                }
                                            }

                                            if (cl >= 0 && cl < maxSubClusters-1) {

                                                // Updating other cluster properties
                                                clusterPropertiesDevice[clusterIdx].cls[cl] += clusterPropertiesDevice[clusterIdx].pixel_ADC[subpixel];                                                
                                                clusterPropertiesDevice[clusterIdx].clusterForPixel[subpixel_counter] = cl;
                                                clusterPropertiesDevice[clusterIdx].weightOfPixel[subpixel_counter] = maxEst;
                                                subpixel_counter++;
                                            }
                                        }

                                    }
                                    else {
                                        //printf("ERROR iterating over scores indices and values exceeds maxPixels %u", maxPixels);
                                    }
                                }
                            }
                        }

                        // Recompute cluster centers
                        stop = true;
                        for (unsigned int subcluster_index = 0; subcluster_index < meanExp; subcluster_index++) {
                            if (subcluster_index < maxSubClusters-1) {
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
                        }

                        for (unsigned int pixel_index = 0; pixel_index < clusterPropertiesDevice[clusterIdx].pixelCounter; pixel_index++) {
                            if (pixel_index < maxPixels-1) {
                                if (clusterPropertiesDevice[clusterIdx].clusterForPixel[pixel_index] < 0)
                                    continue;

                                clusterPropertiesDevice[clusterIdx].clx[clusterPropertiesDevice[clusterIdx].clusterForPixel[pixel_index]] += clusterPropertiesDevice[clusterIdx].pixel_X[pixel_index] * clusterPropertiesDevice[clusterIdx].pixel_ADC[pixel_index];
                                clusterPropertiesDevice[clusterIdx].cly[clusterPropertiesDevice[clusterIdx].clusterForPixel[pixel_index]] += clusterPropertiesDevice[clusterIdx].pixel_Y[pixel_index] * clusterPropertiesDevice[clusterIdx].pixel_ADC[pixel_index];
                                clusterPropertiesDevice[clusterIdx].cls[clusterPropertiesDevice[clusterIdx].clusterForPixel[pixel_index]] += clusterPropertiesDevice[clusterIdx].pixel_ADC[pixel_index];
                            }
                        }
                        for (unsigned int subcluster_index = 0; subcluster_index < meanExp; subcluster_index++) {
                            if (subcluster_index < maxSubClusters-1) {                            
                                if (clusterPropertiesDevice[clusterIdx].cls[subcluster_index] != 0) {
                                    clusterPropertiesDevice[clusterIdx].clx[subcluster_index] /= clusterPropertiesDevice[clusterIdx].cls[subcluster_index];
                                    clusterPropertiesDevice[clusterIdx].cly[subcluster_index] /= clusterPropertiesDevice[clusterIdx].cls[subcluster_index];
                                }
                                clusterPropertiesDevice[clusterIdx].cls[subcluster_index] = 0;
                            }
                        }
                    }

                    // accumulate pixel with same cl
                    int p = 0;
                    for (int cl = 0; cl < (int) meanExp; cl++) {
                        for (unsigned int j = 0; j < clusterPropertiesDevice[clusterIdx].pixelCounter; j++) {
                            if (j < maxPixels) {                            
                                if (clusterPropertiesDevice[clusterIdx].clusterForPixel[j] == cl and clusterPropertiesDevice[clusterIdx].pixel_ADC[j] != 0) {

                                    // cl find the other pixels
                                    // with same x,y and
                                    // accumulate+reset their adc
                                    for (unsigned int k = j + 1; k < clusterPropertiesDevice[clusterIdx].pixelCounter; k++) {
                                        if (k < maxPixels) {                            
                                            if (clusterPropertiesDevice[clusterIdx].pixel_ADC[k] != 0 and 
                                                clusterPropertiesDevice[clusterIdx].pixel_X[k] == clusterPropertiesDevice[clusterIdx].pixel_X[j] and 
                                                clusterPropertiesDevice[clusterIdx].pixel_Y[k] == clusterPropertiesDevice[clusterIdx].pixel_Y[j] and 
                                                clusterPropertiesDevice[clusterIdx].clusterForPixel[k] == cl) {
                                                    clusterPropertiesDevice[clusterIdx].pixel_ADC[j] += clusterPropertiesDevice[clusterIdx].pixel_ADC[k];
                                                    clusterPropertiesDevice[clusterIdx].pixel_ADC[k] = 0;
                                                }
                                        }
                                    }

                                    clusterPropertiesDevice[clusterIdx].pixelsForCl_X[cl][p] = clusterPropertiesDevice[clusterIdx].pixel_X[j];
                                    clusterPropertiesDevice[clusterIdx].pixelsForCl_Y[cl][p] = clusterPropertiesDevice[clusterIdx].pixel_Y[j];
                                    clusterPropertiesDevice[clusterIdx].pixelsForCl_ADC[cl][p] = clusterPropertiesDevice[clusterIdx].pixel_ADC[j];
                                    clusterPropertiesDevice[clusterIdx].pixelsForCl_rawIdArr[cl][p] = clusterPropertiesDevice[clusterIdx].rawIdArr[j];
                                    p++;
                                    clusterPropertiesDevice[clusterIdx].pixelsForClCounter[cl] = p;

                                }
                            }
                        }
                    }

                    // Final writing all the subcluster along with  all pixels
                    uint32_t idx = alpaka::atomicAdd(acc, clusterCounterDevice, uint32_t(0));
                    for (int cl = 0; cl < (int) meanExp; cl++) {
                        for (unsigned int j = 0; j < static_cast<uint32_t>(clusterPropertiesDevice[clusterIdx].pixelsForClCounter[cl]); j++) {

                            if ( (idx + p) >= static_cast<uint32_t>(outputDigis.metadata().size())) {
                                printf("ERROR: Idx %u out of bounds (max %u)\n", idx, outputDigis.metadata().size());
                                return;  // Prevent out-of-bounds write
                            }
                            outputDigis.clus(idx + j) = j;
                            outputDigis.xx(idx + j) = static_cast<uint16_t>(clusterPropertiesDevice[clusterIdx].pixelsForCl_X[cl][j]);
                            outputDigis.yy(idx + j) = static_cast<uint16_t>(clusterPropertiesDevice[clusterIdx].pixelsForCl_Y[cl][j]);
                            outputDigis.adc(idx + j) = static_cast<uint16_t>(clusterPropertiesDevice[clusterIdx].pixelsForCl_ADC[cl][j]);
                            outputDigis.rawIdArr(idx + j) = static_cast<uint32_t>(clusterPropertiesDevice[clusterIdx].pixelsForCl_rawIdArr[cl][j]);
                            outputDigis.moduleId(idx + j) = moduleId;

                            printf("Output: Original Cluster %u, New-Subcluster %d, Pixel %u: moduleId = %u, xx = %u, yy = %u, adc = %u, rawId = %u\n",
                                   clusterIdx, cl, j, moduleId,
                                   outputDigis.xx(idx + j),
                                   outputDigis.yy(idx + j),
                                   outputDigis.adc(idx + j),
                                   outputDigis.rawIdArr(idx + j));


                        }
                        // Use atomicAdd to ensure pixels are added correctly
                        idx = alpaka::atomicAdd(acc, clusterCounterDevice, uint32_t(1));

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
                    CandidatesSoAView& candidateView,
                    ClusterGeometrysSoAView& geoclusterView,
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

    // Get the number of items per block (threads per block)
    const uint32_t threadsPerBlock = 128;

    // Calculate how many groups (blocks) you need for each view
    const uint32_t numBlocks = (geoclusterView.metadata().size() + threadsPerBlock - 1) / threadsPerBlock;
  
    const auto MyworkDiv = make_workdiv<Acc1D>(numBlocks, threadsPerBlock);
    //const auto MyworkDiv = make_workdiv<Acc1D>(1, 917);  //setting 916 and beyond crash!

    std::cout << "\nGot candidateView.metadata().size()=" << candidateView.metadata().size(); 
    std::cout << "\nGot geoclusterView.metadata().size()=" << geoclusterView.metadata().size()
          << "\nExecuting with " << alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(MyworkDiv)[0u] << " blocks and " 
          << threadsPerBlock << " threads per block " 
          << " and " << alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(MyworkDiv)[0u] 
          << " threads in total" << std::endl;

    std::cout << "In the kernel... " << std::endl;



    // std::cout << "Launching kernel with " << groups << " blocks and " << items << " threads per block." << std::endl;

                // Kernel executions
                alpaka::exec<Acc1D>(queue, 
                                    MyworkDiv, 
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
                                                    CandidatesSoAView& candidateView,
                                                    ClusterGeometrysSoAView& geoclusterView,
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
                                                    CandidatesSoAView& candidateView,
                                                    ClusterGeometrysSoAView& geoclusterView,
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
