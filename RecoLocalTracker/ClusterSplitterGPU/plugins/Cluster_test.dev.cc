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
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/JetReco/interface/Jet.h"

#include "DataFormats/VertexSoA/interface/ZVertexSoA.h"
#include "DataFormats/VertexSoA/interface/ZVertexHost.h"
#include "DataFormats/VertexSoA/interface/ZVertexDevice.h"
#include "DataFormats/VertexSoA/interface/alpaka/ZVertexSoACollection.h"

#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include "DataFormats/GeometrySurface/interface/SOARotation.h"

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
    struct JetSplit {

        // Main operator function
        template <typename TAcc, typename = std::enable_if_t<isAccelerator<TAcc>>>
        ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                      TrackingRecHitSoAConstView<TrackerTraits> hitView,
                                      SiPixelDigisSoAView digiView,
                                      SiPixelClustersSoAConstView clusterView,
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
                                      double forceXError_, double forceYError_,
                                      float vertexX, float vertexY, float vertexZ, float vertexEta, float vertexPhi,
                                      bool verbose_, bool debugMode, int targetDetId, int targetClusterOffset) const {

            // Get thread and grid indices
            auto threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; // Thread index within the block
            auto blockIdx  = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];   // Block index
            auto blockDim  = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]; // Threads per block

            // Compute the global thread ID
            uint32_t globalThreadId = blockIdx * blockDim + threadIdx;
            uint32_t moduleId;
            uint32_t clusterOffset;
/*
            /////////////////////////////////////////////////////
            if (globalThreadId == 0) {

                // Printout the entire DigiSoA              
                for (uint32_t pixel = 0; pixel < static_cast<uint32_t>(digiView.metadata().size()); pixel++) {
                    printf("Pixel %u | clus: %d | moduleID: %u | rawIdArr: %u | adc: %u | pdigi: %u | xx: %u | yy: %u\n",
                               pixel,
                               digiView.clus(pixel),
                               digiView.moduleId(pixel),
                               digiView.rawIdArr(pixel),
                               digiView.adc(pixel),                               
                               digiView.pdigi(pixel),
                               digiView.xx(pixel),
                               digiView.yy(pixel));

                }

                // Printout the entire ClusterSoA              
                for (uint32_t pixel = 0; pixel < static_cast<uint32_t>(clusterView.metadata().size()); pixel++) {
                    printf("Cluster %u | moduleStart: %u | clusInModule: %u | moduleId: %u | clusModuleStart: %u\n",
                               pixel,
                               clusterView.moduleStart(pixel),
                               clusterView.clusInModule(pixel),
                               clusterView.moduleId(pixel),
                               clusterView.clusModuleStart(pixel) );                              
                }

                // Printout the entire HitView              
                for (uint32_t jj = 0; jj < static_cast<uint32_t>(hitView.metadata().size()); jj++) {
                    printf("hit %u | charge: %u \n",
                               jj,
                                hitView.chargeAndStatus(jj).charge);                              
                }

            }
            /////////////////////////////////////////////////////
*/


            // Get total Clusters and Candidates
            uint32_t numClusters = static_cast<uint32_t>(geoclusterView.metadata().size());
            uint32_t numCandidates = static_cast<uint32_t>(candidateView.metadata().size());

            // Ensure only valid threads process clusters
            if (globalThreadId < numClusters-2) {

                if ( globalThreadId == 0 ) *clusterCounterDevice = 0;

                uint32_t clusterIdx = globalThreadId;      // Each thread handles exactly one cluster
                moduleId = geoclusterView.moduleId(clusterIdx);
                clusterOffset = geoclusterView.clusterOffset(clusterIdx);




                if (debugMode) {
                    //uint32_t clusterIdx = 327;
                    for (uint32_t j = 0; j < static_cast<uint32_t>(digiView.metadata().size()); j++) {
                        if ( static_cast<uint32_t>(digiView.rawIdArr(j)) == static_cast<uint32_t>(targetDetId)) {
                            moduleId = digiView.moduleId(j);
                        }
                    }
                    clusterOffset = targetClusterOffset;

                    for (uint32_t j = 0; j < static_cast<uint32_t>(geoclusterView.metadata().size()); j++) {
                        if ( static_cast<uint32_t>(geoclusterView.moduleId(j)) == static_cast<uint32_t>(moduleId)) {
                            if ( static_cast<uint32_t>(geoclusterView.clusterOffset(j)) == clusterOffset) {
                                clusterIdx = j;
                            }
                        }
                    }
                }




                int FoundPixels=0;
                for (uint32_t j = 0; j < static_cast<uint32_t>(digiView.metadata().size()); j++) {
                    if ( static_cast<uint32_t>(digiView.moduleId(j)) == moduleId) {
                        if ( static_cast<uint32_t>(digiView.clus(j)) == clusterOffset) {
                            FoundPixels++;                                    

                        }
                    }
                }
                if (verbose_) printf("Working on Detector Module %u clusterOffset %u with these pixels: %u\n", moduleId, clusterOffset, FoundPixels);
                for (uint32_t j = 0; j < static_cast<uint32_t>(digiView.metadata().size()); j++) {
                    if ( static_cast<uint32_t>(digiView.moduleId(j)) == moduleId) {
                        if ( static_cast<uint32_t>(digiView.clus(j)) == clusterOffset) {
                            if (verbose_) printf(" pixel adc %d x=%d y=%d\n",digiView.adc(j), digiView.xx(j), digiView.yy(j));
                        }
                    }
                }

/*
                // Search for the clusterIdx that is matching the ModuleID and the ClusterOffset
                // as the correspondence between SiPixelCluster vs SiPixelClusterSoA is not 1:1 (it's 99%..)
                uint32_t RetrievedModule = geoclusterView.moduleId(clusterIdx);
                uint32_t RetrievedClusterOffset = geoclusterView.clusterOffset(clusterIdx);

                printf("RetrievedModule %u\n",RetrievedModule);
                printf("RetrievedClusterOffset %u\n",RetrievedClusterOffset);

                bool foundMatch = false;  // Flag to track if a match is found
                for (uint32_t ScanningCluster = 0; ScanningCluster < static_cast<uint32_t>(clusterView.metadata().size()); ScanningCluster++) {
                    if (RetrievedModule == clusterView.moduleId(ScanningCluster)) {

                        if ( (RetrievedClusterOffset) <= clusterView.clusInModule(ScanningCluster) ) {
                            printf("Found clus in Module %u\n", clusterView.clusInModule(ScanningCluster));

                            moduleId = clusterView.moduleId(ScanningCluster);
                            clusterOffset = clusterView.moduleStart(ScanningCluster) + clusterView.clusModuleStart(ScanningCluster);                    
                            
                            foundMatch = true;  // Set flag since we found a valid match
                            break;  // Exit loop early since we found the correct match
                        } 
                        else {
                            printf("ERROR: The SoA doesn't have that Cluster Offset %u\n", RetrievedClusterOffset);
                            return;  // Exit function since this is an error condition
                        }
                    }
                }
                if (!foundMatch) printf("ERROR: The SoA doesn't have Module %u\n", RetrievedModule);

                // From the above, now we have:
                // - `moduleId`: The module this cluster belongs to
                // - `clusterOffset`: The cluster number within that module
*/


/*
                // Print all about this cluster under study.........
                for (uint32_t pixel = 0; pixel < static_cast<uint32_t>(digiView.metadata().size()); pixel++) {
                    if ( static_cast<uint32_t>(digiView.moduleId(pixel)) == moduleId) {
                        if ( static_cast<uint32_t>(digiView.clus(pixel)) == clusterOffset) {
                            printf("--  clus: %d | moduleID: %u | rawIdArr: %u | adc: %u | pdigi: %u | xx: %u | yy: %u CLX: %f CLY: %f CLZ: %f\n",
                                   digiView.clus(pixel),
                                   digiView.moduleId(pixel),
                                   digiView.rawIdArr(pixel),
                                   digiView.adc(pixel),                               
                                   digiView.pdigi(pixel),
                                   digiView.xx(pixel),
                                   digiView.yy(pixel),
                                   geoclusterView.x(clusterIdx),
                                   geoclusterView.y(clusterIdx), 
                                   geoclusterView.z(clusterIdx));                            
                        }
                    }
                }
*/

                //printf("I am in thread %u, analyzing cluster %u from module %u offset %u\n", 
                //       globalThreadId, clusterOffset, moduleId, clusterOffset);


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

                    // Access hit global positions from hitView 
                    // but geoclusterView got more refined position!
                    //float x = hitView.xGlobal(clusterIdx);
                    //float y = hitView.yGlobal(clusterIdx);
                    //float z = hitView.zGlobal(clusterIdx);

                    // Access fine-tuned Global position (previously saved into the GeoCluster SoA)
                    float x = geoclusterView.x(clusterIdx);
                    float y = geoclusterView.y(clusterIdx);
                    float z = geoclusterView.z(clusterIdx);

                    // Subtract the primary vertex position to obtain the relative position
                    float relX = x - vertexX;
                    float relY = y - vertexY;
                    float relZ = z - vertexZ;
                    if (verbose_) printf("Cluster direction (cPos - vertex):");
                    if (verbose_) printf("  dx = %f, dy = %f, dz = %f\n", relX, relY, relZ);

                    // Extract jet momentum components from candidateView
                    float jetPx = candidateView.px(candIdx);
                    float jetPy = candidateView.py(candIdx);
                    float jetPz = candidateView.pz(candIdx);

                    // Compute jet transverse momentum, eta, and phi
                    float jetPt = sqrt(jetPx * jetPx + jetPy * jetPy);
                    float jetP  = sqrt(jetPx * jetPx + jetPy * jetPy + jetPz * jetPz);
                    float jetEta = 0.5 * log((jetP + jetPz) / (jetP - jetPz));
                    float jetPhi = atan2(jetPy, jetPx);

                    // Print the jet information 
                    if (verbose_) printf("Jet Information:\n");
                    if (verbose_) printf("  jetPx = %f, jetPy = %f, jetPz = %f\n", jetPx, jetPy, jetPz);
                    if (verbose_) printf("  jetPt = %f, jetEta = %f, jetPhi = %f\n\n", jetPt, jetEta, jetPhi);

                    // Compute the cluster's relative eta and phi
                    float r = sqrt(relX * relX + relY * relY + relZ * relZ);
                    float clusterEta = 0.5 * log((r + relZ) / (r - relZ));  // Pseudorapidity formula
                    float clusterPhi = atan2(relY, relX);  // Azimuthal angle

                    // Compute differences and deltaR (assuming 'jetEta' and 'jetPhi' are known)
                    float deltaEta = clusterEta - jetEta;
                    float deltaPhi = atan2(sin(clusterPhi - jetPhi), cos(clusterPhi - jetPhi));  // Adjust for periodicity
                    float deltaR = sqrt(deltaEta * deltaEta + deltaPhi * deltaPhi);
                    //printf("  deltaEta = %f, deltaPhi = %f, deltaR = %f\n", deltaEta, deltaPhi, deltaR);

                    // Print the absolute cluster position (without vertex subtraction)
                    //float abs_r = sqrt(x * x + y * y + z * z);
                    //float absClusterEta = 0.5 * log((abs_r + z) / (abs_r - z));
                    //float absClusterPhi = atan2(y, x);
 

                    // Check deltaR condition and split clusters if applicable

                    if (deltaR < deltaR_) {
                        if (verbose_) printf("This clusterOffset: %u has deltaR < deltaR_ and it might be split\n",clusterOffset);

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
                                     forceYError_,
                                     verbose_);
                    }
                    else {
                        storeOutputDigis(acc, digiView, outputDigis, moduleId, clusterOffset, clusterCounterDevice);
                    }
                }
            }
            else {
                return;
            }
        }


        ALPAKA_FN_ACC void closestClusters(clusterProperties* clusterData, uint32_t clusterIdx, int pixelIdx, float& minDist, float& secondMinDist, unsigned meanExp) const {
            minDist = std::numeric_limits<float>::max();
            secondMinDist = std::numeric_limits<float>::max();

            // Loop over all sub-clusters to calculate distance for a specific pixel
            for (uint32_t subClusterIdx = 0; subClusterIdx < meanExp; subClusterIdx++) {
                float dist = clusterData[clusterIdx].distanceMap[pixelIdx][subClusterIdx];  // Access the distanceMap

                // Debug print for each sub-cluster iteration
                //printf("DEBUG: pixelIdx=%d, subClusterIdx=%u, dist=%f, current minDist=%f, current secondMinDist=%f\n",
                //        pixelIdx, subClusterIdx, dist, minDist, secondMinDist);

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
            //printf("running with clusterData[clusterIdx].pixelCounter=: %u\n", clusterData[clusterIdx].pixelCounter);

            for (uint32_t pixelIdx = 0; pixelIdx < clusterData[clusterIdx].pixelCounter; pixelIdx++) {
                if ( pixelIdx < maxPixels ) {
                    float minDist, secondMinDist;
                    // Call closestClusters to calculate minDist and secondMinDist for each pixel
                    closestClusters(clusterData, clusterIdx, pixelIdx, minDist, secondMinDist, meanExp);
                    clusterData[clusterIdx].scoresIndices[pixelIdx] = pixelIdx;
                    clusterData[clusterIdx].scoresValues[pixelIdx] = -secondMinDist;
                }
                else {
                    //printf("ERROR@ secondDistScore: pixelIdx (%u) exceeds maxPixels (%d)\n", pixelIdx, maxPixels);
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

        ALPAKA_FN_ACC void sortScores(clusterProperties* clusterData, uint32_t clusterIdx, bool verbose_) const {
            //if (verbose_) printf("Pixel counter: %u\n", clusterData[clusterIdx].pixelCounter);

            for (uint32_t i = 0; i < clusterData[clusterIdx].pixelCounter - 1; i++) {
                for (uint32_t j = 0; j < clusterData[clusterIdx].pixelCounter - i - 1; j++) {

                    if (j >= maxPixels-1) {
                        printf("ERROR@ sortScores: j (%u) exceeds maxPixels (%d)\n", j, maxPixels);
                        return;
                    }

                    else {
                        //if (verbose_) printf("clusterData[clusterIdx].scoresValues: %u, %f\n", j, clusterData[clusterIdx].scoresValues[j+1]);

                        if (clusterData[clusterIdx].scoresValues[j] > clusterData[clusterIdx].scoresValues[j + 1]) {  
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
            if (verbose_) printf("Scores\n");

          for (uint32_t k = 0; k < clusterData[clusterIdx].pixelCounter; k++) {
            if (verbose_) printf("Score: %f, Index = %d\n",
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
            //if (verbose_) printf("CalculatedClusters = %d ", CalculatedClusters);            
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

            //if (verbose_) printf("AtomicAdd result: %u \n", idx);
            //if (verbose_) printf("DigiView size: %u\n", static_cast<uint32_t>(digiView.metadata().size()));
            //if (verbose_) printf("output size: %u\n", static_cast<uint32_t>(outputDigis.metadata().size()));

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
                        //if (verbose_) printf("AtomicAdd result: %u storeIdx: %u\n", idx, storeIdx);
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
                                        double forceYError_,
                                        bool verbose_) const {

            //if (verbose_) printf("This cluster: %u now processed in SplitCluster routine\n",clusterIdx);

            bool split = false;

            float pitchX = geoclusterView.pitchX(clusterIdx);
            float pitchY = geoclusterView.pitchY(clusterIdx);
            float thickness = geoclusterView.thickness(clusterIdx);
            float tanLorentzAngles = geoclusterView.tanLorentzAngles(clusterIdx);

            // Apply precomputed transformation matrix
            float jetDirLocalX = geoclusterView.transformXX(clusterIdx) * jetPx + geoclusterView.transformYX(clusterIdx) * jetPy + geoclusterView.transformZX(clusterIdx) * jetPz;
            float jetDirLocalY = geoclusterView.transformXY(clusterIdx) * jetPx + geoclusterView.transformYY(clusterIdx) * jetPy + geoclusterView.transformZY(clusterIdx) * jetPz;
            float jetDirLocalZ = geoclusterView.transformXZ(clusterIdx) * jetPx + geoclusterView.transformYZ(clusterIdx) * jetPy + geoclusterView.transformZZ(clusterIdx) * jetPz;

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


            // Copy the original pixels into a fixed array for simpler handling (and sequential access)
            // as in the original SoA pixels have whatever index j
            clusterPropertiesDevice[clusterIdx].pixelCounter = 0;
            int ClusterCharge = 0;

            for (uint32_t j = 0; j < static_cast<uint32_t>(digiView.metadata().size()); j++) {
                if ( static_cast<uint32_t>(digiView.moduleId(j)) == moduleId) {
                    if ( static_cast<uint32_t>(digiView.clus(j)) == clusterOffset) {
                                    
                        clusterPropertiesDevice[clusterIdx].originalpixels_x[ clusterPropertiesDevice[clusterIdx].pixelCounter ] = digiView.xx(j); // Copy x-coordinate from original pixel
                        clusterPropertiesDevice[clusterIdx].originalpixels_y[ clusterPropertiesDevice[clusterIdx].pixelCounter ] = digiView.yy(j); // Copy y-coordinate from original pixel

                        clusterPropertiesDevice[clusterIdx].originalpixels_ADC[ clusterPropertiesDevice[clusterIdx].pixelCounter ] = digiView.adc(j);

                        clusterPropertiesDevice[clusterIdx].originalpixels_rawIdArr[j] = digiView.rawIdArr(j);
                        ClusterCharge = ClusterCharge + clusterPropertiesDevice[clusterIdx].originalpixels_ADC[ clusterPropertiesDevice[clusterIdx].pixelCounter ];
                        clusterPropertiesDevice[clusterIdx].pixelCounter++;
                    }
                }
            }
            if (verbose_) printf("Trying to split: charge=%d expSizeX=%f expSizeY=%f\n",
                    static_cast<int>(ClusterCharge), expSizeX, expSizeY);

            if ( ClusterCharge > expectedADC * chargeFracMin_ &&
                   ( ClusterCharge > expSizeX + 1 || ClusterCharge > expSizeY + 1)) {
                split = true;
            }

            if (split) {

                // Aligning to the original "fittingSplit" variables..
                int sizeY = expSizeY;
                int sizeX = expSizeX;

                unsigned int meanExp = std::floor( ClusterCharge / expectedADC + 0.5f);

                if (meanExp <= 1) {
                    if (verbose_) printf("meanExp <= 1 writing cluster");
                    storeOutputDigis(acc, digiView, outputDigis, moduleId, clusterOffset, clusterCounterDevice);
                }
                else {
                    // Splitting the pixels and writing them for the current clusterIdx
                    if (verbose_) printf("cluster has meanExp=%d\n", meanExp);

                    uint32_t pixelsSize=0;
                    for (uint32_t j = 0; j < clusterPropertiesDevice[clusterIdx].pixelCounter; j++) {

                        int sub = static_cast<int>(clusterPropertiesDevice[clusterIdx].originalpixels_ADC[j]) / chargePerUnit_ * expectedADC / centralMIPCharge_;
                        if (sub < 1) sub = 1;
                        //if (verbose_) printf("FOR j=%d  sub=%d\n",j, sub);

                        int perDiv = clusterPropertiesDevice[clusterIdx].originalpixels_ADC[j] / sub;

                        if (verbose_) printf("Splitting %d in [ %d , %d ], expected numb of clusters: %u original pixel (x,y) %d %d sub %d\n",
                               j,  pixelsSize, pixelsSize+sub, meanExp, clusterPropertiesDevice[clusterIdx].originalpixels_x[j], clusterPropertiesDevice[clusterIdx].originalpixels_y[j], sub);

                        // Iterate over the sub-clusters (split pixels)
                        for (int k = 0; k < sub; k++) {
                            if (k == sub - 1) perDiv = clusterPropertiesDevice[clusterIdx].originalpixels_ADC[j] - perDiv * k;  // Adjust for the last pixel

                            // Write the new split pixels at the obtained index
                            clusterPropertiesDevice[clusterIdx].pixels[pixelsSize] = j;                                    
                            clusterPropertiesDevice[clusterIdx].pixel_X[pixelsSize] = clusterPropertiesDevice[clusterIdx].originalpixels_x[j]; // Copy x-coordinate from original pixel
                            clusterPropertiesDevice[clusterIdx].pixel_Y[pixelsSize] = clusterPropertiesDevice[clusterIdx].originalpixels_y[j]; // Copy y-coordinate from original pixel
                            clusterPropertiesDevice[clusterIdx].pixel_ADC[pixelsSize] = perDiv;       // Assign divided charge (ADC)
                            clusterPropertiesDevice[clusterIdx].rawIdArr[pixelsSize] = clusterPropertiesDevice[clusterIdx].originalpixels_rawIdArr[j]; // Copy rawIdArr from original pixel
                            pixelsSize++;
                        }
                    }
                    // make sure pixelsSize does not exceed max pixels!

                    if (verbose_) printf("Computing initial values, set all distances");
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


                    while (!stop && remainingSteps > 0) {
                        if (verbose_) printf("---------------\n");
                        if (verbose_) printf("REMAINING STEPS : %d\n", remainingSteps);
                        remainingSteps--;

                        // Compute distances
                        for (uint32_t j = 0; j < clusterPropertiesDevice[clusterIdx].pixelCounter; ++j) {

                            //if (verbose_) printf("Original Pixel pos %d %f %f\n", j, clusterPropertiesDevice[clusterIdx].originalpixels_x[j], clusterPropertiesDevice[clusterIdx].originalpixels_y[j]);


                            for (unsigned int i = 0; i < meanExp; i++) {
                                //if (i >= maxSubClusters) continue; // Safety check for bounds

                                // Calculate the distance in X and Y for each pixel
                                float distanceX = 1.f * clusterPropertiesDevice[clusterIdx].originalpixels_x[j] - clusterPropertiesDevice[clusterIdx].clx[i];
                                float distanceY = 1.f * clusterPropertiesDevice[clusterIdx].originalpixels_y[j] - clusterPropertiesDevice[clusterIdx].cly[i];
                                float dist = 0;
                                //if (verbose_) printf("i=%u, distanceX = %f, distanceY = %f\n", i, distanceX, distanceY);
                                
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
                                //if (verbose_) printf("Cluster=%u Original Pixel %u distanceMap[%u][%u] = %f\n", i, j, j, i, clusterPropertiesDevice[clusterIdx].distanceMap[j][i]);
                            }
                        } // compute distances done

                        secondDistScore(clusterPropertiesDevice, clusterIdx, meanExp);
                        
                        // In the original code:
                        // - the first index is the distance, in whatever metrics we use, 
                        // - the second is the pixel index w.r.t which the distance is computed.
                        //std::multimap < float, int > scores;
                        // In this code the first index is in scoresIndices, the second in scoresValues
                        // to mimic the multimap, I score manually both arrays
                        sortScores(clusterPropertiesDevice, clusterIdx, verbose_);

                        // Iterating over Scores Indices and Values
                        for (unsigned int i = 0; i < clusterPropertiesDevice[clusterIdx].pixelCounter; i++) {
                            if (i < maxPixels) {
                                int pixel_index = clusterPropertiesDevice[clusterIdx].scoresIndices[i];
                                //float score_value = clusterPropertiesDevice[clusterIdx].scoresValues[i];

                                int subpixel_counter = 0;

                                // Iterating over subpixels
                                for (unsigned int subpixel = 0; subpixel < pixelsSize; subpixel++, subpixel_counter++) {
                                    if (subpixel< maxPixels) {

                                        if (clusterPropertiesDevice[clusterIdx].pixels[subpixel] > static_cast<uint32_t>(pixel_index)) {
                                            break;
                                        } else if (clusterPropertiesDevice[clusterIdx].pixels[subpixel] != static_cast<uint32_t>(pixel_index)) {
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
                                                        //if (verbose_) printf("cl = %d",cl);
                                                    }
                                                }
                                            }

                                            // Updating other cluster properties
                                            clusterPropertiesDevice[clusterIdx].cls[cl] += clusterPropertiesDevice[clusterIdx].pixel_ADC[subpixel];                                                
                                            clusterPropertiesDevice[clusterIdx].clusterForPixel[subpixel_counter] = cl;
                                            clusterPropertiesDevice[clusterIdx].weightOfPixel[subpixel_counter] = maxEst;

                                            if (verbose_) printf("Pixel weight weightOfPixel[%d]=%f  cl=%d\n", 
                                                    subpixel_counter, clusterPropertiesDevice[clusterIdx].weightOfPixel[subpixel_counter], cl);                                        
                                        }
                                    }
                                    else {
                                        //if (verbose_) printf("ERROR iterating over scores indices and values exceeds maxPixels %u", maxPixels);
                                    }
                                }
                            }
                        }

                        // Recompute cluster centers
                        if (verbose_) printf("Recomputing cluster centers.........\n ");

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
                                clusterPropertiesDevice[clusterIdx].cls[subcluster_index] = 1e-38f;//1e-99;
                            }
                        }

                        for (unsigned int pixel_index = 0; pixel_index < pixelsSize; pixel_index++) {
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
                                if (verbose_) printf("Center for cluster, clx[%u]=%f cly[%u]=%f\n",subcluster_index,clusterPropertiesDevice[clusterIdx].clx[subcluster_index], subcluster_index, clusterPropertiesDevice[clusterIdx].cly[subcluster_index]);

                                clusterPropertiesDevice[clusterIdx].cls[subcluster_index] = 0;
                            }
                        }
                    }

                    // accumulate pixel with same cl
                    if (verbose_) printf("Accumulate pixel with same cl...\n");
                    int p = 0;

                    for (int cl = 0; cl < (int) meanExp; cl++) {
                        p = 0;
                        for (unsigned int j = 0; j < pixelsSize; j++) {
                            if (j < maxPixels) {                            
                                if (clusterPropertiesDevice[clusterIdx].clusterForPixel[j] == cl and clusterPropertiesDevice[clusterIdx].pixel_ADC[j] != 0) {

                                    // cl find the other pixels
                                    // with same x,y and
                                    // accumulate+reset their adc
                                    for (unsigned int k = j + 1; k < pixelsSize; k++) {
                                        if (k < maxPixels) {                            
                                            if (clusterPropertiesDevice[clusterIdx].pixel_ADC[k] != 0 and 
                                                clusterPropertiesDevice[clusterIdx].pixel_X[k] == clusterPropertiesDevice[clusterIdx].pixel_X[j] and 
                                                clusterPropertiesDevice[clusterIdx].pixel_Y[k] == clusterPropertiesDevice[clusterIdx].pixel_Y[j] and 
                                                clusterPropertiesDevice[clusterIdx].clusterForPixel[k] == cl) {


                                                if (verbose_) printf("Resetting all sub-pixel for location %d, %d at index %d associated to cl %d\n", 
                                                       clusterPropertiesDevice[clusterIdx].pixel_X[k], clusterPropertiesDevice[clusterIdx].pixel_Y[k], k, clusterPropertiesDevice[clusterIdx].clusterForPixel[k]);

                                                clusterPropertiesDevice[clusterIdx].pixel_ADC[j] += clusterPropertiesDevice[clusterIdx].pixel_ADC[k];
                                                clusterPropertiesDevice[clusterIdx].pixel_ADC[k] = 0;
                                            }
                                        }
                                    }

                                    for (unsigned int p = 0; p < pixelsSize; ++p) {
                                        if (verbose_) printf("index, x, y, ADC: %u, %d, %d, %d associated to cl %d\n",
                                               p, clusterPropertiesDevice[clusterIdx].pixel_X[p], clusterPropertiesDevice[clusterIdx].pixel_Y[p], clusterPropertiesDevice[clusterIdx].pixel_ADC[p], clusterPropertiesDevice[clusterIdx].clusterForPixel[p]);

                                        if (verbose_) printf("Adding pixel %d, %d to cluster %d\n",
                                               clusterPropertiesDevice[clusterIdx].pixel_X[j], clusterPropertiesDevice[clusterIdx].pixel_Y[j], cl);
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
                    if (verbose_) printf("\nFinal writing");

                    uint32_t idx = alpaka::atomicAdd(acc, clusterCounterDevice, uint32_t(0));
                    for (int cl = 0; cl < (int) meanExp; cl++) {

                        if (verbose_) printf("\n---> Pixels of cl=%d ",cl);
                        for (unsigned int j = 0; j < static_cast<uint32_t>(clusterPropertiesDevice[clusterIdx].pixelsForClCounter[cl]); j++) {

                            if (verbose_) printf("pixelsForCl[cl][j].x=%d, pixelsForCl[cl][j].y=%d, pixelsForCl[cl][j].adc=%d\n",
                                   clusterPropertiesDevice[clusterIdx].pixelsForCl_X[cl][j],
                                   clusterPropertiesDevice[clusterIdx].pixelsForCl_Y[cl][j],
                                   clusterPropertiesDevice[clusterIdx].pixelsForCl_ADC[cl][j]);

                            if ( (idx + p) >= static_cast<uint32_t>(outputDigis.metadata().size())) {
                                if (verbose_) printf("ERROR: Idx %u out of bounds (max %u)\n", idx, outputDigis.metadata().size());
                                return;  // Prevent out-of-bounds write
                            }
                            outputDigis.clus(idx + j) = j;
                            outputDigis.xx(idx + j) = static_cast<uint16_t>(clusterPropertiesDevice[clusterIdx].pixelsForCl_X[cl][j]);
                            outputDigis.yy(idx + j) = static_cast<uint16_t>(clusterPropertiesDevice[clusterIdx].pixelsForCl_Y[cl][j]);
                            outputDigis.adc(idx + j) = static_cast<uint16_t>(clusterPropertiesDevice[clusterIdx].pixelsForCl_ADC[cl][j]);
                            outputDigis.rawIdArr(idx + j) = static_cast<uint32_t>(clusterPropertiesDevice[clusterIdx].pixelsForCl_rawIdArr[cl][j]);
                            outputDigis.moduleId(idx + j) = moduleId;
/*
                            if (verbose_) printf("Output: Original Cluster %u, New-Subcluster %d, Pixel %u: moduleId = %u, xx = %u, yy = %u, adc = %u, rawId = %u\n",
                                   clusterIdx, cl, j, moduleId,
                                   outputDigis.xx(idx + j),
                                   outputDigis.yy(idx + j),
                                   outputDigis.adc(idx + j),
                                   outputDigis.rawIdArr(idx + j));
*/
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
                    float vertexX, float vertexY, float vertexZ, float vertexEta, float vertexPhi,
                    bool verbose_,
                    bool debugMode, int targetDetId, int targetClusterOffset,
                    Queue& queue) {

    // Get the number of items per block (threads per block)
    const uint32_t threadsPerBlock = 128;


    // Calculate how many groups (blocks) you need for each view
    const uint32_t numBlocks = (geoclusterView.metadata().size() + threadsPerBlock - 1) / threadsPerBlock;
  
    //const auto MyworkDiv = make_workdiv<Acc1D>(numBlocks, threadsPerBlock);

    const auto MyworkDiv = make_workdiv<Acc1D>(1, 1);

    //const auto MyworkDiv = debugMode ? make_workdiv<Acc1D>(1, 1) : make_workdiv<Acc1D>(numBlocks, threadsPerBlock);


    if (verbose_) std::cout << "\nGot candidateView.metadata().size()=" << candidateView.metadata().size(); 
    if (verbose_) std::cout << "\nGot geoclusterView.metadata().size()=" << geoclusterView.metadata().size()
                          << "\nExecuting with " << alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(MyworkDiv)[0u] << " blocks and " 
                          << threadsPerBlock << " threads per block " 
                          << " and " << alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(MyworkDiv)[0u] 
                          << " threads in total" << std::endl;


    if (verbose_) std::cout << "In the kernel... " << std::endl;

    // std::cout << "Launching kernel with " << groups << " blocks and " << items << " threads per block." << std::endl;

                // Kernel executions AccCpuSerial should be Acc1D
                alpaka::exec<Acc1D>(queue, 
                                    MyworkDiv, 
                                    JetSplit<TrackerTraits>{}, 
                                    hitView, 
                                    digiView, 
                                    clusterView, 
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
                                    forceYError_,
                                    vertexX, vertexY, vertexZ, vertexEta, vertexPhi, 
                                    verbose_, debugMode, targetDetId, targetClusterOffset);
            }

    // Explicit template instantiation for Phase 1
    template void runKernels<pixelTopology::Phase1>(TrackingRecHitSoAView<pixelTopology::Phase1>& hitView,
                                                    SiPixelDigisSoAView& digiView,
                                                    SiPixelClustersSoAView& clusterView,
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
                                                    float vertexX, float vertexY, float vertexZ, float vertexEta, float vertexPhi,
                                                    bool verbose_, bool debugMode, int targetDetId, int targetClusterOffset,
                                                    Queue& queue);

    // Explicit template instantiation for Phase 2
    template void runKernels<pixelTopology::Phase2>(TrackingRecHitSoAView<pixelTopology::Phase2>& hitView,
                                                    SiPixelDigisSoAView& digiView,
                                                    SiPixelClustersSoAView& clusterView,
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
                                                    float vertexX, float vertexY, float vertexZ, float vertexEta, float vertexPhi,                                                    
                                                    bool verbose_, bool debugMode, int targetDetId, int targetClusterOffset,
                                                    Queue& queue);
  }  // namespace Splitting
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
