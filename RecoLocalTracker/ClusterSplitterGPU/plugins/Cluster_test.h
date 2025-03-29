#ifndef DataFormats_SiPixelClusterSoA_test_alpaka_Hits_test_h
#define DataFormats_SiPixelClusterSoA_test_alpaka_Hits_test_h

//#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisSoA.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersSoA.h"
#include "DataFormats/VertexSoA/interface/ZVertexSoA.h"

#include "DataFormats/ClusterGeometrySoA/interface/ClusterGeometrysSoA.h"
#include "DataFormats/ClusterGeometrySoA/interface/alpaka/ClusterGeometrysSoACollection.h"

#include "DataFormats/CandidateSoA/interface/CandidatesSoA.h"
#include "DataFormats/CandidateSoA/interface/alpaka/CandidatesSoACollection.h"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include <alpaka/alpaka.hpp>

constexpr int maxSubClusters = 20;
constexpr int maxPixels = 500;

// This represent a per-cluster data needed in the Splitting algorithm
struct clusterProperties {

    // These are used to split original cluster into subclusters
    float clx[maxSubClusters];
    float cly[maxSubClusters];
    float cls[maxSubClusters];
    float oldclx[maxSubClusters];
    float oldcly[maxSubClusters];

    // Copy the pixels from the SoA into original to keep them aligned to the following arrays
    uint32_t originalpixels_x[maxPixels];
    uint32_t originalpixels_y[maxPixels];
    uint32_t originalpixels_ADC[maxPixels];
    uint32_t originalpixels_rawIdArr[maxPixels];

    // These are used to store temporary pixel information
    uint32_t pixelCounter;                // how many pixels in the cluster under study
    uint32_t pixels[maxPixels];                // Storing the index of the pixel
    int pixel_X[maxPixels];             // X position of each pixel
    int pixel_Y[maxPixels];             // Y position of each pixel
    uint32_t pixel_ADC[maxPixels];           // adc value of each pixel
    uint32_t rawIdArr[maxPixels];         // RawAddress of each pixel

    // These are used for the final sub-cluster (each subcluster contains pixels)
    uint32_t pixelsForClCounter[maxSubClusters];             // how many pixels in this Cluster
    int pixelsForCl_X[maxSubClusters][maxPixels];          // position
    int pixelsForCl_Y[maxSubClusters][maxPixels];          // position
    uint32_t pixelsForCl_ADC[maxSubClusters][maxPixels];        // adc value
    uint32_t pixelsForCl_rawIdArr[maxSubClusters][maxPixels];        // adc value
    //int pixelsForCl[maxSubClusters][maxPixels];              // Storing the index of the pixel

    // thse are used for k-map like algorithm and scoring
    float distanceMap[maxPixels][maxSubClusters];
    int scoresIndices[maxPixels];           // need this because can't to map
    float scoresValues[maxPixels];          // need this because can't to map

    int clusterForPixel[maxPixels];
    float weightOfPixel[maxPixels];
};

using namespace reco;

namespace ALPAKA_ACCELERATOR_NAMESPACE::Splitting {

  template <typename TrackerTraits>
  void runKernels(TrackingRecHitSoAView<TrackerTraits>& hits,
                  SiPixelDigisSoAView& digis,
                  SiPixelClustersSoAView& clusters,
                  CandidatesSoAView& candidates,
                  ClusterGeometrysSoAView& geoclusters,
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
                  bool verbose_, bool debugMode, int targetDetId, int targetClusterOf,
                  Queue& queue);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::Splitting

#endif  // DataFormats_SiPixelClusterSoA_test_alpaka_Hits_test_h
