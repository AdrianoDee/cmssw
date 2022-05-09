// CUDA include files
#include <cuda_runtime.h>

// CMSSW include files
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "RecoVertex/PrimaryVertexProducer/interface/trackFilterCUDA.h"
#include <stdio.h>

#ifdef __CUDA_ARCH__
#include "HeterogeneousCore/CUDAUtilities/interface/radixSort.h"
#endif
namespace trackFilterCUDA {
// track selection should happen on cpu because we're already looping on tracks for the SoA filling => perform this step inside that loop
/*
__global__ void trackFilterKernel(unsigned int ntracks, TrackForPV::TrackForPVSoA* tracks, filterParameters params, double* osumtkwt){
    // TODO:: Study performance of explicitly writing squares vs std::pow. Study performance of atomics for the serial part
    size_t firstElement = threadIdx.x + blockIdx.x * blockDim.x;
    size_t gridSize = blockDim.x * gridDim.x;
    // First the preselection block
    if (ntracks > tracks->stride()) printf("\n\nCareful, we might loose some selected tracks because the number of tracks before selection is > # selected tracks max size\n\n");
    for (unsigned int i = firstElement; i < tracks->stride(); i += gridSize) {
      tracks->isGood(i) = false;
      tracks->weight(i) = 0;
      if (i > ntracks) break; // If in the empty region go on
      if (tracks->significance(i) < params.maxSignificance){
        if (tracks->dxy2(i) < params.maxdxyError*params.maxdxyError){
          if (tracks->dz2(i) < params.maxdzError*params.maxdzError){
            if (tracks->pAtIP(i) > params.minpAtIP){
              if (std::fabs(tracks->etaAtIP(i)) < params.maxetaAtIP){
                if (tracks->chi2(i) < params.maxchi2){
                  if (tracks->nPixelHits(i) >= params.minpixelHits){
                    if (tracks->nTrackerHits(i) >= params.mintrackerHits){
                      tracks->isGood(i) = true;
                    }
                  }
                }
              }
            }
          }
        }
      }
      // And now the stuff for the clusterizer
      if (tracks->isGood(i)){
        tracks->weight(i) = 1.;  
        if (std::fabs(tracks->z(i)) > 1000.){ 
          tracks->isGood(i) = false;
          tracks->weight(i) = 0;
          continue;
        }
        else{ // Get dz2 for the track
          // dz2 is zerror^2 + (bx*px + by*py)^2*pz^2/(pt^4) + vertex_size^2
          tracks->dz2(i) = tracks->dz2(i) 
                         + (tracks->bx(i)*tracks->bx(i)*tracks->pxAtPCA(i)*tracks->pxAtPCA(i) + tracks->by(i)*tracks->by(i)*tracks->pyAtPCA(i)*tracks->pyAtPCA(i))* tracks->pzAtPCA(i)*tracks->pzAtPCA(i)/(( tracks->pxAtPCA(i)*tracks->pxAtPCA(i) + tracks->pyAtPCA(i)*tracks->pyAtPCA(i) )* ( tracks->pxAtPCA(i)*tracks->pxAtPCA(i) + tracks->pyAtPCA(i)*tracks->pyAtPCA(i))) 
                         + params.vertexSize*params.vertexSize; // TODO:: For sure ways to optimize this
          tracks->dz2(i) = 1./tracks->dz2(i);
          if (not(std::isfinite(tracks->dz2(i))) || tracks->dz2(i)< std::numeric_limits<double>::min()){ // Bad track dz2 is taken out
            tracks->isGood(i) = false;
            tracks->weight(i) = 0;
            continue;
          }
          else{
            if (params.d0CutOff > 0){ // Track weights are activated only if there is a non-zero cutoff
              // weight is 1/(1 + e^{sig^2 - d0cutoff^2})
              tracks->weight(i) = 1./ (1+exp(tracks->significance(i)*tracks->significance(i) - params.d0CutOff*params.d0CutOff ));
              if (not(std::isfinite(tracks->weight(i))) || tracks->weight(i)< std::numeric_limits<double>::epsilon()){ // Bad track weight is taken out
                tracks->isGood(i) = false;
                tracks->weight(i) = 0;
                continue;
              }
            }
            // If we are here, the track is to be passed to the clusterizer. So initialize the clusterizer stuff
            tracks->sum_Z(i) = 0;
            tracks->kmin(i) = 0; // will loop from kmin to kmax-1. At the start only one vertex
            tracks->kmax(i) = 1;
            tracks->aux1(i) = 0;
            tracks->aux2(i) = 0;
          }
        }
      }
    }
    __syncthreads(); // Synchronize after loop
    int nSelectedTracks = 0; //DEBUG
    if (threadIdx.x == 0 && blockIdx.x==0){ // Single threaded. TODO:: Check atomicAdd or other sync ways of doing this
      (*osumtkwt) = 0;
      for (unsigned int i = 0 ; i < ntracks ; i++){
        if (tracks->isGood(i)){
          (*osumtkwt) += tracks->weight(i);
          tracks->order(nSelectedTracks) = i;
          nSelectedTracks++; //DEBUG
          //printf("itrack, z, dz2, weight: %i %1.10f %1.10f %1.10f\n", i, tracks->z(i), tracks->dz2(i), tracks->weight(i));
        }
      }
    
      (*osumtkwt) = (*osumtkwt) > 0 ? 1./(*osumtkwt) : 0.; // This really is the only thing you need in a single thread, as multiple operations at once will break it
      //tracks->nTrueTracks = (int)  nSelectedTracks/10;
      tracks->nTrueTracks = (int)  nSelectedTracks;
      //tracks->nTrueTracks = (int)  256;
      printf("N true tracks after GPU filter: %i\n",tracks->nTrueTracks); //DEBUG
        
      
      ////////// printf("osumtkwt after GPU: %1.10f\n", *osumtkwt);
    }
    __syncthreads(); // Synchronize after loop
}
*/



__global__ void trackSorterKernel(unsigned int ntracks, TrackForPV::TrackForPVSoA* tracks){
    // double tracksZ[nTrueTracks];
    /*
    if (threadIdx.x == 0 && blockIdx.x == 0) { 
      double tracksZ[8196]; // we will copy each z coord. inside this array so that when we find the min we will "remove" the min found from next iteration
      for (unsigned int itrack = 0; itrack < tracks->stride(); itrack ++ ) {
          tracksZ[itrack] = tracks->z(itrack);
      }
      double min = 100000.0;
      int iMinO = -1; 
      unsigned int iTrueMinO = 0;
      unsigned int trueTracksOrder[8196]; // will be our future tracks->order
 
      for (unsigned int itrackO = 0 ; itrackO < tracks->nTrueTracks ; itrackO++){
          min = 100000.0;
          iMinO = -1;
          for (unsigned int itrackOO = 0 ; itrackOO < tracks->nTrueTracks; itrackOO++){
              unsigned int itrack = tracks->order(itrackOO);
              if (tracksZ[itrack]<min){
                  min = tracksZ[itrack];      
                  iMinO = itrack;
              }
           }
           trueTracksOrder[iTrueMinO] = iMinO;
           tracksZ[iMinO] = 100000.0;
           iTrueMinO++;
      }
     if (iTrueMinO != tracks->nTrueTracks) printf("num sorted tracks != num unsorted tracks: (%d, %d) \n\n",iTrueMinO,  tracks->nTrueTracks);
     for (unsigned int itrackO = 0; itrackO < iTrueMinO -1 ; itrackO ++ ){
        unsigned int itrack = trueTracksOrder[itrackO];
        unsigned int itrackNext = trueTracksOrder[itrackO+1];
        if (tracks->z(itrack) > tracks->z(itrackNext)) printf("problem with the ordered track (z1,z2): (%f, %f)\n(order1, pos1): (%d, %d)\n\n", tracks->z(itrack), tracks->z(itrackNext), itrackO, itrack);
     }
    

//     printf("\n\ndump\n\n\n");
//     for (unsigned int itrackO = 0; itrackO < iTrueMinO -1 ; itrackO ++ ){
//        unsigned int itrack = trueTracksOrder[itrackO];
//        printf("(order,pos,z): (%d, %d, %f)\n\n", itrackO,itrack, tracks->z(itrack));
//     }
     for (unsigned int itrackO = 0; itrackO < iTrueMinO ; itrackO ++ ){
        tracks->order(itrackO) = trueTracksOrder[itrackO] ;
     }
     //tracks->nTrueTracks = 256;
    }    
    */
    // if (threadIdx.x == 0 && blockIdx.x == 0) printf("nTrueTracks in sorter: %d\n", tracks->nTrueTracks);
    size_t firstElement = threadIdx.x + blockIdx.x * blockDim.x; // This is going to be the track index
    size_t gridSize = blockDim.x * gridDim.x;
  __shared__ float tracksZ[5120]; // we will copy each z coord. inside this array so that when we find the min we will "remove" the min found from next iteration
   
  if (tracks->nTrueTracks < 5120) {

  for (unsigned int itrack=firstElement; itrack<tracks->nTrueTracks; itrack+=gridSize) {
        tracksZ[itrack] = tracks->z(itrack);
  }
  __syncthreads();

      __shared__ __restrict__ uint16_t trueTracksOrder[5120];
      __shared__ uint16_t sws[5120];

      unsigned int const& nvFinal = tracks->nTrueTracks;
      #ifdef __CUDA_ARCH__
      radixSort<float, 2>(tracksZ, trueTracksOrder, sws, nvFinal);
      #else
      if (threadIdx.x == 0 && blockIdx.x == 0) {
            for (unsigned int itrackO=0; itrackO<tracks->nTrueTracks; itrackO++){
                unsigned int itrack = trueTracksOrder[itrackO];
                float z = tracksZ[itrack];
                printf("itrack %d, itrackO %d, z %f\n", itrackO, itrack, z);
            } 
       }
      #endif
      __syncthreads();
      /*
       */
      for (unsigned int itrack=firstElement; itrack<tracks->nTrueTracks; itrack+=gridSize) {
            tracks->order(itrack) = trueTracksOrder[itrack];
      }
       __syncthreads();
  } else {

      // __shared__ float tracksZ[4096]; // we will copy each z coord. inside this array so that when we find the min we will "remove" the min found from next iteration
      return;
/*
      unsigned int steps = std::ceil(tracks->nTrueTracks / 4096);
      double range = (tracks->max_z - tracks->min_z )/steps ;
      for (unsigned int itrack=0; itrack<tracks->nTrueTracks; itrack++) {
            if (tracks->z(itrack) 
            tracksZ[nTrueTracks] = tracks->z(itrack); 
            nTrueTracks++;
        }
      for (unsigned int itrack=firstElement; itrack<tracks->nTrueTracks; itrack+=gridSize) {
            tracksZ[itrack] = tracks->z(itrack);
      }
*/
  __syncthreads();
      
      __shared__ __restrict__ uint16_t trueTracksOrder[4096];
      __shared__ uint16_t sws[4096];

      unsigned int const& nvFinal = tracks->nTrueTracks;
      #ifdef __CUDA_ARCH__
      radixSort<float, 2>(tracksZ, trueTracksOrder, sws, nvFinal);
      #else
      if (threadIdx.x == 0 && blockIdx.x == 0) {
            for (unsigned int itrackO=0; itrackO<tracks->nTrueTracks; itrackO++){
                unsigned int itrack = trueTracksOrder[itrackO];
                float z = tracksZ[itrack];
                printf("itrack %d, itrackO %d, z %f\n", itrackO, itrack, z);
            } 
       }
      #endif
      __syncthreads();
      /*
       */
      for (unsigned int itrack=firstElement; itrack<tracks->nTrueTracks; itrack+=gridSize) {
            tracks->order(itrack) = trueTracksOrder[itrack];
      }
       __syncthreads();
    }
}


#ifdef __CUDACC__
  // Only on GPUs, of course...
void sorterWrapper(unsigned int ntracks, TrackForPV::TrackForPVSoA* tracks, cudaStream_t stream){
    unsigned int blockSize = 512; // TODO::Optimize this
    unsigned int gridSize  = 1; // TrackForPV::TrackForPVSoA always has 8096 entries. TODO::Automatize this one
    trackSorterKernel<<<gridSize, blockSize, 0, stream>>>(ntracks, tracks);
    cudaCheck(cudaGetLastError());
}
/*
void filterWrapper(unsigned int ntracks, TrackForPV::TrackForPVSoA* tracks, filterParameters params, double* osumtkwt, cudaStream_t stream){
    unsigned int blockSize = 512; // TODO::Optimize this
    unsigned int gridSize  = 1; // TrackForPV::TrackForPVSoA always has 8096 entries. TODO::Automatize this one
//    trackFilterKernel<<<gridSize, blockSize, 0, stream>>>(ntracks, tracks, params, osumtkwt);
//    cudaCheck(cudaGetLastError());
    trackSorterKernel<<<gridSize, blockSize, 0, stream>>>(ntracks, tracks);
    cudaCheck(cudaGetLastError());
}
*/
#endif
}
