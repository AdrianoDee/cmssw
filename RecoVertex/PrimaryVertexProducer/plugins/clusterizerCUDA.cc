// CUDA include files
#include <cuda_runtime.h>

// CMSSW include files
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "RecoVertex/PrimaryVertexProducer/interface/clusterizerCUDA.h"
#include <stdio.h>
namespace clusterizerCUDA {


__device__ __forceinline__ void checkRange(TrackForPV::TrackForPVSoA* tracks, TrackForPV::VertexForPVSoA* vertices){
    __syncthreads();
    unsigned int firstElement = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int gridSize = blockDim.x * gridDim.x;
   for(unsigned int itrackO = firstElement; itrackO< tracks->nTrueTracks; itrackO += gridSize){
        unsigned int itrack = tracks->order(itrackO);
        for (unsigned int ivertexO = tracks->kmin(itrack); ivertexO < tracks->kmax(itrack); ivertexO ++){
            if (vertices->order(ivertexO) == -1){
                printf("\n\nvtx range error: %d, %d, %d, (%d, %d)\n\n", itrackO, itrack, ivertexO, threadIdx.x, blockIdx.x);
            }
        }
    }  
    __syncthreads();
}

__device__ __forceinline__ void initializeKernel(unsigned int ntracks, TrackForPV::TrackForPVSoA* tracks, TrackForPV::VertexForPVSoA* vertices, clusterParameters params){
    // TODO:: Study performance of explicitly writing squares vs std::pow. Study performance of atomics for the serial part
    size_t firstElement = threadIdx.x + blockIdx.x * blockDim.x;
    size_t gridSize = blockDim.x * gridDim.x;
    size_t maxVerticesPerBlock = (int) (vertices->stride()/gridDim.x);
    // First the preselection block
    vertices->nTrueVertex(blockIdx.x) = 1; // Will add one here. As it is hard assignment, not really care to separate by threads, a bit lazy it is
    for (unsigned int i = maxVerticesPerBlock * blockIdx.x + threadIdx.x; i < maxVerticesPerBlock * (blockIdx.x+1); i += gridSize) {
      vertices->sw(i)     = 0.;
      vertices->se(i)     = 0.;
      vertices->swz(i)    = 0.;
      vertices->swE(i)    = 0.;
      vertices->exp(i)    = 0.;
      vertices->exparg(i) = 0.;
      vertices->z(i)      = 0.;
      vertices->rho(i)    = 0.;
      vertices->isGood(i) = false;
      // The order vector is special, -1 means undefined
      vertices->order(i)  = -1;
      if (i == maxVerticesPerBlock * blockIdx.x){ // TODO:: Cleanup
        vertices->z(i)     = 0.;
        vertices->rho(i)   = 1.; // First vertex has all the swag
        vertices->order(i) = maxVerticesPerBlock * blockIdx.x;  // And it is the only one to loop over
        vertices->isGood(i)= true;
      }
    }
    __syncthreads(); // Synchronize after loop,just in case
    for (unsigned int itrackO = firstElement; itrackO < tracks->nTrueTracks; itrackO += gridSize) {
        unsigned int itrack = tracks->order(itrackO);
        tracks->kmin(itrack) = maxVerticesPerBlock * blockIdx.x; 
        tracks->kmax(itrack) = maxVerticesPerBlock * blockIdx.x + 1; 
        int ivertex = vertices->order(maxVerticesPerBlock*blockIdx.x);
        if (ivertex==-1) printf("\n\nPROBLEM in initializeKernel\n\n");
    }
    __syncthreads(); // Synchronize after loop,just in case
    //checkRange(tracks, vertices);
}

__device__ __forceinline__ void getBeta0Kernel(unsigned int ntracks, TrackForPV::TrackForPVSoA* tracks, TrackForPV::VertexForPVSoA* vertices, clusterParameters params, double *beta){
//    double znew = 0;
//    double wnew = 0;
    size_t firstElement = threadIdx.x + blockIdx.x*blockDim.x; // This is going to be the track index
    size_t gridSize = blockDim.x*gridDim.x;
    size_t maxVerticesPerBlock = (int) (vertices->stride()/gridDim.x);

    for (unsigned int itrackO = firstElement; itrackO < tracks->nTrueTracks ; itrackO += gridSize){
      unsigned int itrack = tracks->order(itrackO);
      //if (not(tracks->isGood(itrack))) continue;
      //printf("Track %i is Good, adding stuff\n", itrack);
      tracks->aux1(itrack)  = tracks->weight(itrack)*tracks->dz2(itrack); // Will be sumw
      tracks->aux2(itrack)  = tracks->weight(itrack)*tracks->dz2(itrack)*tracks->z(itrack); // Will be sumwz
    }
    __syncthreads();
    //if (0 == threadIdx.x){ // Serialized code. TODO:: Test atomicAdd instead, or better even a syncable sum 
    __shared__ double znew, wnew;
    if (0 == threadIdx.x){
        znew = 0;
        wnew = 0;
    }
    __syncthreads();
    for (unsigned int itrackO = firstElement ; itrackO < tracks->nTrueTracks ; itrackO += gridSize){
      unsigned int itrack = tracks->order(itrackO);
     // for (unsigned int itrack = 0; itrack < ntracks ; itrack++){
     //   if (not(tracks->isGood(itrack))) continue;
        atomicAdd(&znew, tracks->aux2(itrack)); //sumwz
        atomicAdd(&wnew, tracks->aux1(itrack)); //sumw
        //printf("znew, wnew, itrack: %1.16f, %1.16f, %i \n", znew, wnew, itrack);
      }
    __syncthreads();
      if (0 == threadIdx.x) vertices->z(maxVerticesPerBlock*blockIdx.x) = znew/wnew; //New z is the quotient 
      //if (0 == threadIdx.x) printf("\n\nFirst vertex for block (%d): %f\n\n", blockIdx.x, znew/wnew);
    
    __syncthreads();
    for (unsigned int itrackO = firstElement; itrackO < tracks->nTrueTracks ; itrackO += gridSize){
      unsigned int itrack = tracks->order(itrackO);
//    for (unsigned int itrack = firstElement; itrack < ntracks ; itrack += gridSize){
//      if (not(tracks->isGood(itrack))) continue;
      tracks->aux2(itrack) = tracks->aux1(itrack)*(vertices->z(maxVerticesPerBlock*blockIdx.x) - tracks->z(itrack) )*(vertices->z(maxVerticesPerBlock*blockIdx.x) - tracks->z(itrack))*tracks->dz2(itrack);
    }
    __syncthreads();
    __shared__  double a;
    if (threadIdx.x == 0) a = 0;
    __syncthreads();
    for (unsigned int itrackO = firstElement; itrackO < tracks->nTrueTracks ; itrackO += gridSize){
      unsigned int itrack = tracks->order(itrackO);
//      for (unsigned int itrack = 0; itrack < ntracks ; itrack++){
//        if (not(tracks->isGood(itrack))) continue;
        atomicAdd(&a, tracks->aux2(itrack));
      }
    __syncthreads();
    if (0 == threadIdx.x){ // More serialized code. TODO:: Test atomicAdd instead, or better even a syncable sum 
      (*beta) = 2 * a/wnew; // Here it is technically 1/*beta0, i.e. Tc, but ok to save on memory allocation
      ////////// printf("*beta0 before comparing: %1.3f\n", ((*beta)));
      ////////// printf("znew, wnew, a: %1.16f, %1.16f, %1.16f \n", znew, wnew, a);

      double betamax_ = 1./params.Tmin; //From the config file, 
      ////////// printf("*betamax, coolingFactor %1.10f, %1.10f\n", *betamax_, params.coolingFactor);
      if ((*beta) > 1./betamax_){
        int coolingsteps = 1 - int(std::log((*beta) * betamax_) / std::log(params.coolingFactor)); // A tricky conversion to round the number
        (*beta) = betamax_ * std::pow(params.coolingFactor, coolingsteps);
        ////////// printf("*betamax, coolingFactor, coolingsteps %1.10f, %1.10f %i \n", *betamax_, params.coolingFactor, coolingsteps);
      }
      else{
        (*beta) = betamax_ * params.coolingFactor;
      }
      //TODO::Add debugging option
      ////////// printf("1./*beta0 %1.3f\n", 1./((*beta)));
      //printf("%1.10f",*beta[0]);
    }
    __syncthreads();
    //checkRange(tracks, vertices);
}

__device__ __forceinline__ void thermalizeKernel(unsigned int ntracks, TrackForPV::TrackForPVSoA* tracks, TrackForPV::VertexForPVSoA* vertices, clusterParameters params, double * osumtkwt, double *beta, double delta_max0, double rho0){
   thermalize(ntracks, tracks, vertices, params, osumtkwt, beta, delta_max0, rho0);
   //Dif (0 == threadIdx.x && 0 == blockIdx.x){
   //D  for (unsigned int ivertexO = 0; ivertexO < vertices->nTrueVertex(blockIdx.x) ; ivertexO++){
   //D    printf("At T=%1.3f, ivertex=%i, pos=%1.10f, rho=%1.10f", 1./((**beta)), ivertexO, vertices->z(vertices->order(ivertexO)), vertices->rho(vertices->order(ivertexO)));
   //D  }
   //D}
   __syncthreads();
    //checkRange(tracks, vertices);
}
__device__ __forceinline__ void coolingWhileSplittingKernel(unsigned int ntracks, TrackForPV::TrackForPVSoA* tracks, TrackForPV::VertexForPVSoA* vertices, clusterParameters params, double* osumtkwt, double *beta){
  double betafreeze = (1./params.Tmin) * sqrt(params.coolingFactor); //Last T to be updated
  // First the T loop
////  ////////// if (0 == threadIdx.x && 0 == blockIdx.x)   printf("Start cooling! \n");
  while (((*beta)) < betafreeze) {
    __syncthreads();
    unsigned int nprev = vertices->nTrueVertex(blockIdx.x);
////    if (0 == threadIdx.x && 0 == blockIdx.x)   printf("New T: %1.5f ; nv = %i \n",1./((*beta)), nprev);
    
//    clock_t mstart = clock();
//    if (threadIdx.x == 0)   printf("begin merge\n");
    __syncthreads();
    merge(ntracks, tracks, vertices, params, osumtkwt, beta);
//    if (threadIdx.x == 0)   printf("end merge\n");
    __syncthreads();
    //checkRange(tracks, vertices);
 //   clock_t mstop = clock();
////// //   if (threadIdx.x == 0 && 0 == blockIdx.x)   printf("Clock for merge: %i\n", (int) (mstop-mstart));
    
    //checkOrder(0, vertices);
    __syncthreads();
    
    //merge(ntracks, tracks, vertices, params, osumtkwt, *beta);
    //__syncthreads();
    //checkOrder(1, vertices);
//////    ////////// if (0 == threadIdx.x && 0 == blockIdx.x)   printf("After merging nv = %i \n", vertices->nTrueVertex(blockIdx.x));

    while (nprev !=  vertices->nTrueVertex(blockIdx.x)) { // While merge is true
      nprev = vertices->nTrueVertex(blockIdx.x);
      __syncthreads();
      
 //     clock_t ustart = clock();
//    if (threadIdx.x == 0 )   printf("begin update\n");
    update(ntracks, tracks, vertices, params, osumtkwt, beta, 0.0, false); //Udpdate them 
//    if (threadIdx.x == 0 )   printf("finished update\n");
    __syncthreads();
    //checkRange(tracks, vertices);
    __syncthreads();
 //     clock_t ustop = clock();
////// //     if (threadIdx.x == 0 && 0 == blockIdx.x)   printf("Clock for update: %i\n", (int) (ustop-ustart));
 //     mstart = clock();
//    if (threadIdx.x == 0 && 0 == blockIdx.x)   printf("begin merge\n");
      merge(ntracks, tracks, vertices, params, osumtkwt, beta);
//    if (threadIdx.x == 0 && 0 == blockIdx.x)   printf("finished merge\n");
    __syncthreads();
    //checkRange(tracks, vertices);
    __syncthreads();
 //     mstop = clock();
////// //     if (threadIdx.x == 0 && 0 == blockIdx.x)   printf("Clock for merge: %i\n", (int) (mstop-mstart));
      
      //update(ntracks, tracks, vertices, params, osumtkwt, *beta, 0.0, false); //Udpdate them 

//////      ////////// if (0 == threadIdx.x && 0 == blockIdx.x)   printf("After merging nv = %i \n", vertices->nTrueVertex(blockIdx.x));
      //
      //checkOrder(2, vertices);
    }
    //checkOrder(21, vertices);
//////     // if (0 == threadIdx.x && 0 == blockIdx.x)   printf("After merge loop nv = %i \n", vertices->nTrueVertex(blockIdx.x));
    
 //   clock_t sstart = clock();


//    if (threadIdx.x == 0 && 0 == blockIdx.x)   printf("begin split\n");
    split(ntracks, tracks, vertices, params, osumtkwt, beta, 1.); // Then split if we need to
//    if (threadIdx.x == 0 && 0 == blockIdx.x)   printf("finished split\n");
    __syncthreads();
    //checkRange(tracks, vertices);
    __syncthreads();


 //   clock_t sstop = clock();
////// //   if (threadIdx.x == 0 && 0 == blockIdx.x)   printf("Clock for split: %i\n", (int) (sstop-sstart));
    
    //split(ntracks, tracks, vertices, params, osumtkwt, *beta, 1.); // Then split if we need to
    //checkOrder(3, vertices);
////    if (0 == threadIdx.x && 0 == blockIdx.x)   printf("After splitting nv = %i \n", vertices->nTrueVertex(blockIdx.x));

    if (0 == threadIdx.x) ((*beta)) = ((*beta)) / params.coolingFactor; // Reduce temperature
////    if (0 == threadIdx.x && 0 == blockIdx.x)   printf("New T = %1.5f \n", 1./((*beta)));

    __syncthreads();
 //   clock_t tstart = clock();
//    if (threadIdx.x == 0 && 0 == blockIdx.x)   printf("begin therm\n");
    thermalize(ntracks, tracks, vertices, params, osumtkwt, beta, params.delta_highT, 0.0); // And recompute everything at the new temperature
//    if (threadIdx.x == 0 && 0 == blockIdx.x)   printf("finished therm\n");
    __syncthreads();
    //checkRange(tracks, vertices);
    __syncthreads();
////    if (0 == threadIdx.x && 0 == blockIdx.x)   printf("After therm nv = %i \nT: %f \n", vertices->nTrueVertex(blockIdx.x), 1./(*beta));
 //   clock_t tstop = clock();
////// //   if (threadIdx.x == 0 && 0 == blockIdx.x)   printf("Clock for thermalize: %i\n", (int) (tstop-tstart));
    //checkOrder(4, vertices);
  }
  // After the T loop, reassign vertices, and update again
 // clock_t srstart = clock();
  set_vtx_range(ntracks, tracks, vertices, params, osumtkwt, beta);
//    if (threadIdx.x == 0 && 0 == blockIdx.x)   printf("finished set range\n");
    __syncthreads();
    //checkRange(tracks, vertices);
    __syncthreads();
 // clock_t srstop = clock();
////// // if (threadIdx.x == 0 && 0 == blockIdx.x)   printf("Clock for set vtx range: %i\n", (int) (srstop-srstart));
  update(ntracks, tracks, vertices, params, osumtkwt, beta, 0.0, false);
  //checkOrder(5, vertices);
  __syncthreads();
}

__device__ __forceinline__ void remergeTracksKernel(unsigned int ntracks, TrackForPV::TrackForPVSoA* tracks, TrackForPV::VertexForPVSoA* vertices, clusterParameters params, double* osumtkwt, double *beta){
  unsigned int nprev = vertices->nTrueVertex(blockIdx.x);
  merge(ntracks, tracks, vertices, params, osumtkwt, beta);
  __syncthreads();
  while (nprev !=  vertices->nTrueVertex(blockIdx.x)) {
//    if (threadIdx.x == 0 && 0 == blockIdx.x)   printf("begin set range\n");
    set_vtx_range(ntracks, tracks, vertices, params, osumtkwt, beta);
//    if (threadIdx.x == 0 && 0 == blockIdx.x)   printf("end set range\n");
    __syncthreads();
//    if (threadIdx.x == 0 && 0 == blockIdx.x)   printf("begin update\n");
    update(ntracks, tracks, vertices, params, osumtkwt, beta, 0.0, false);
//    if (threadIdx.x == 0 && 0 == blockIdx.x)   printf("end update\n");
    __syncthreads();
    nprev = vertices->nTrueVertex(blockIdx.x);   
    merge(ntracks, tracks, vertices, params, osumtkwt, beta);
    __syncthreads();
  }
}

__device__ __forceinline__ void resplitTracksKernel(unsigned int ntracks, TrackForPV::TrackForPVSoA* tracks, TrackForPV::VertexForPVSoA* vertices, clusterParameters params, double* osumtkwt, double *beta){
  unsigned int ntry = 0; 
  double threshold = 1.0;
  unsigned int nprev = vertices->nTrueVertex(blockIdx.x);
  split(ntracks, tracks, vertices, params, osumtkwt, beta, threshold);
  __syncthreads();
  while (nprev !=  vertices->nTrueVertex(blockIdx.x) && (ntry++<10)) {
    thermalize(ntracks, tracks, vertices, params, osumtkwt, beta, params.delta_highT, 0.0); // if split, we recompute everything 
    __syncthreads();
    // We might need to merge afterwards!
    nprev = vertices->nTrueVertex(blockIdx.x);
    __syncthreads();
    merge(ntracks, tracks, vertices, params, osumtkwt, beta);
    __syncthreads();
    while (nprev !=  vertices->nTrueVertex(blockIdx.x)) {
      nprev = vertices->nTrueVertex(blockIdx.x);
      update(ntracks, tracks, vertices, params, osumtkwt, beta, 0.0, false); //Udpdate them 
      __syncthreads();
      merge(ntracks, tracks, vertices, params, osumtkwt, beta);
      __syncthreads();
    }
    nprev = vertices->nTrueVertex(blockIdx.x);
    threshold *= 1.1;
    split(ntracks, tracks, vertices, params, osumtkwt, beta, threshold);
    __syncthreads();
  }
}


__device__ __forceinline__ void outlierRejectionKernel(unsigned int ntracks, TrackForPV::TrackForPVSoA* tracks, TrackForPV::VertexForPVSoA* vertices, clusterParameters params, double* osumtkwt, double *beta){
  double rho0 = 0.0;
  if (params.dzCutOff > 0){
    rho0 = vertices->nTrueVertex(blockIdx.x) > 1 ? 1./vertices->nTrueVertex(blockIdx.x) : 1.;
    for (unsigned int a = 0; a < 5 ; a++){ //Can't be parallelized in any reasonable way
      update(ntracks, tracks, vertices, params, osumtkwt, beta, a*rho0/5., false);
      __syncthreads();
    }
  }
  thermalize(ntracks, tracks, vertices, params, osumtkwt, beta, params.delta_lowT, rho0); // If we don't do outlier rejection, we just thermalize afther the resplitting step at either rho0=0 or rho0 = 1/nv
  __syncthreads();
  // With outlier rejection we might lose tracks in some vertices, so merge again
  unsigned int nprev = vertices->nTrueVertex(blockIdx.x);
  __syncthreads();
  merge(ntracks, tracks, vertices, params, osumtkwt, beta);
  __syncthreads();
  while (nprev !=  vertices->nTrueVertex(blockIdx.x)) {
    set_vtx_range(ntracks, tracks, vertices, params, osumtkwt, beta);
    __syncthreads();
    update(ntracks, tracks, vertices, params, osumtkwt, beta, rho0, false); // Now at the proper rho0! This changes Z_init and so the whole partition function for tracks
    __syncthreads();
    nprev = vertices->nTrueVertex(blockIdx.x);
    __syncthreads();
    merge(ntracks, tracks, vertices, params, osumtkwt, beta);
    __syncthreads();
  }
  // Now we go to the purge temperature, without splitting or merging
  double betapurge = 1./params.Tpurge;
//  if (0==threadIdx.x)  printf("T loop begin\n\n");
  while (((*beta)) < betapurge){
//    if (0==threadIdx.x)  printf("Loop for beta: %f\n\n", (float) (*beta));
    if (0==threadIdx.x)  ((*beta)) = std::min(((*beta))/params.coolingFactor, betapurge);
    __syncthreads();
    thermalize(ntracks, tracks, vertices, params, osumtkwt, beta, params.delta_lowT, rho0);
    __syncthreads();
  }
//  if (0==threadIdx.x)  printf("T loop end\n\n");
  // Now it is time for the purging
  nprev = vertices->nTrueVertex(blockIdx.x);
  purge(ntracks, tracks, vertices, params, osumtkwt, beta, rho0);
  __syncthreads();
//  if (0==threadIdx.x)  printf("Purge loop begin\n\n");
  while (nprev !=  vertices->nTrueVertex(blockIdx.x)) {
    thermalize(ntracks, tracks, vertices, params, osumtkwt, beta, params.delta_lowT, rho0);
    __syncthreads();
    nprev = vertices->nTrueVertex(blockIdx.x);
    __syncthreads();
    purge(ntracks, tracks, vertices, params, osumtkwt, beta, rho0);
  }
//  if (0==threadIdx.x)  printf("Purge loop end\n\n");

  // And cool down more to make the assignment harder
  double betastop = 1./params.Tstop;
//  if (0==threadIdx.x)  printf("T loop 2 begin\n\n");
  while (((*beta)) < betastop){
    if (0==threadIdx.x) ((*beta)) = std::min(((*beta))/params.coolingFactor, betastop);
    __syncthreads();
    thermalize(ntracks, tracks, vertices, params, osumtkwt, beta, params.delta_lowT, rho0);
    __syncthreads();
  }
//  if (0==threadIdx.x)  printf("T loop 2 end\n\n");
  // A final assignment, before sending it to the fitter
  __syncthreads();
  set_vtx_range(ntracks, tracks, vertices, params, osumtkwt, beta);
}



__device__  __forceinline__ void dumpTVkernel(TrackForPV::TrackForPVSoA* tracks, TrackForPV::VertexForPVSoA* vertices, double * beta){
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\n\nBEGIN TRACKS\n\n");
        for(unsigned itrackO = 0; itrackO < tracks->nTrueTracks; itrackO ++){
             unsigned int itrack = tracks->order(itrackO);
             printf("%d,%d,%d,%f\n", itrackO, itrack, tracks->tt_index(itrack), tracks->z(itrack));
        }
        printf("\n\nEND TRACKS\n\nBEGIN TV\n\n");
        for(unsigned itrackO = 0; itrackO < tracks->nTrueTracks; itrackO ++){
             unsigned int itrack = tracks->order(itrackO);
             for (unsigned int ivertexO = tracks->kmin(itrack); ivertexO < tracks->kmax(itrack); ivertexO ++){
                 printf("%d,%d\n",itrackO,ivertexO);
            }
        }
        printf("\n\nEND TV\n\nBEGIN VERTICES\n\n");
        //size_t maxVerticesPerBlock = (int) (vertices->stride()/gridSize);
        for(unsigned int ivertexO = 0; ivertexO < vertices->stride(); ivertexO ++){
            unsigned int ivertex = vertices->order(ivertexO);
            if (vertices->isGood(ivertex)) {
                printf("%d,%f\n", ivertexO, vertices->z(ivertex));
            }
        }
        printf("\n\nEND VERTICES\n\n");
    }
    __syncthreads();
 }

__device__  __forceinline__ void dumpVkernel(TrackForPV::TrackForPVSoA* tracks, TrackForPV::VertexForPVSoA* vertices, double * beta){
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\n\nBEGIN TV\n\n");
        for(unsigned itrackO = 0; itrackO < tracks->nTrueTracks; itrackO ++){
             unsigned int itrack = tracks->order(itrackO);
             for (unsigned int ivertexO = tracks->kmin(itrack); ivertexO < tracks->kmax(itrack); ivertexO ++){
                 printf("\n%d,%d\n",itrackO,ivertexO);
            }
        }
        printf("\n\nEND TV\n\nBEGIN VERTICES\n\n");
        //size_t maxVerticesPerBlock = (int) (vertices->stride()/gridSize);
        for(unsigned int ivertexO = 0; ivertexO < vertices->stride(); ivertexO ++){
            unsigned int ivertex = vertices->order(ivertexO);
            if (vertices->isGood(ivertex)) {
                printf("%d,%d,%f\n", ivertexO, ivertex, vertices->z(ivertex));
            }
        }
        printf("\n\nEND VERTICES\n\n");
    }
    __syncthreads();
 }


__global__ void overlapTracks(TrackForPV::TrackForPVSoA* tracks, unsigned int blockDim, unsigned int gridDim){
    // we should copy blockSize/2 tracks for each block
    //unsigned int newOrder[tracks->stride()];
    unsigned int newNtracks = tracks->nTrueTracks;
    //for (unsigned int itrackO = 0; itrackO < tracks->nTrueTracks(); itrackO ++){
    for (unsigned int blockId = 0; blockId < 2 * std::ceil(tracks->nTrueTracks/blockDim); blockId++){
        unsigned int begin = int(blockDim/2) + blockDim * blockId;
        unsigned int newPos = (blockId + 1) * blockDim;
        if (begin >= newNtracks || begin > tracks->stride() || newPos>=newNtracks) break;
        unsigned int end =  std::min(blockDim*(blockId+1), newNtracks);
        
        for (unsigned int i = begin; i< end; i++){
             if (newNtracks >= tracks->stride()) {
                if (threadIdx.x == 0 && blockIdx.x == 0) printf("nTrueTracks after overlap: %d\n", newNtracks);
                tracks->nTrueTracks = newNtracks;
                return;
             }
             for (unsigned int j = newNtracks; j>newPos; j--){
                 // if (tracks->order(j) == tracks->order(j-1)) printf("\nProblem, copying %d from %d to %d but they are the same\n\n", tracks->order(j), j, j-1);
                 tracks->order(j) = tracks->order(j-1); 
             }     
             //FIXME copy data of the track with index tracks->order[newPos] -> tracks->data(newNtracks)
             // begin copying
             
             unsigned int oldTrack = tracks->order(i);
             tracks->significance(newNtracks) = tracks->significance(oldTrack);
             tracks->dz2(newNtracks) = tracks->dz2(oldTrack);
             tracks->z(newNtracks) = tracks->z(oldTrack);
             tracks->weight(newNtracks) = tracks->weight(oldTrack);
             tracks->tt_index(newNtracks) = tracks->tt_index(oldTrack);
             tracks->isGood(newNtracks) = true;
             // end copying
             if (newNtracks==0) printf("WTF, newNtracks is 0!\n\n");
             tracks->order(newPos) = newNtracks;
             newPos ++;
             newNtracks ++;
        }
    } 
     // if (threadIdx.x == 0 && blockIdx.x == 0) printf("nTrueTracks after overlap: %d\n", newNtracks);
    tracks->nTrueTracks = newNtracks;
}

__global__ void resortVerticesAndAssign(TrackForPV::TrackForPVSoA* tracks, TrackForPV::VertexForPVSoA* vertices, double * beta,clusterParameters params, unsigned int blockdim, unsigned int griddim){

    /*
    __shared__ float z[1024];
    __shared__ float zSorting[1024];
    __shared__ float rho[1024];
    __shared__ unsigned int orderedIndices[1024];
   __shared__ unsigned int nTrueVertex;
    __syncthreads();
   if (threadIdx.x == 0 && blockIdx.x == 0){ 
        nTrueVertex = 0;
//        float z[1024];
//        float rho[1024];
        //unsigned int vertex_index[1024];
       size_t maxVerticesPerBlock = (int) (vertices->stride()/griddim);
       for (unsigned int blockid = 0; blockid < griddim; blockid++){
           for(unsigned int ivtx = blockid * maxVerticesPerBlock; ivtx < blockid * maxVerticesPerBlock + vertices->nTrueVertex(blockid); ivtx++)   {
                unsigned int ivertex = vertices->order(ivtx);
                if ((std::abs(vertices->rho(ivertex)< 10000)) && (std::abs(vertices->z(ivertex))<15)) {
                    z[nTrueVertex] = vertices->z(ivertex);
                    zSorting[nTrueVertex] = vertices->z(ivertex);
                    rho[nTrueVertex] = vertices->rho(ivertex);
                    // printf("Copying vertex with ivertexO: %d, in position %d, z: %f , rho: %f \tSaving it in %d with z: %f\n", ivtx, ivertex, vertices->z(ivertex), vertices->rho(ivertex), nTrueVertex, z[nTrueVertex]);
                    nTrueVertex++;
                }
            }
       }
        //printf("\n\n"); 
        float min = 1000000.0; 
        int iMinO = -1; 
        for (unsigned int ivtxO = 0; ivtxO<nTrueVertex; ivtxO++){
            min = 1000000.0; 
            iMinO = -1;
            for (unsigned int ivtxOO = 0; ivtxOO<nTrueVertex; ivtxOO++){
               if (zSorting[ivtxOO]<min){
                    min = zSorting[ivtxOO];
                    iMinO = ivtxOO;
                }
            }
//            orderedZ[ivtxO] = z[iMinO];
//            orderedRho[ivtxO] = rho[iMinO];
            orderedIndices[ivtxO] = iMinO;
           // printf("Found minimum i: %d , z: %f, rho: %f\n", iMinO, z[iMinO], rho[iMinO]);
            zSorting[iMinO] = 1000000.0; 
            
        }

        //FIXME find a way to merge very similar vertices
        
        // remove duplicates of tracks
        unsigned int begin = blockdim;
        unsigned int end, newPos;
        unsigned int newNtracks = tracks->nTrueTracks;
        while (begin < newNtracks) {
           newPos = begin + int(blockdim/2);
           end = std::min(begin + int(blockdim/2), newNtracks); 
           for (unsigned int i = begin; i < end; i++){
                tracks->order(i) = tracks->order(newPos);
                for (unsigned int j = newPos; j< newNtracks-1; j++){
                    tracks->order(j) = tracks->order(j+1);
                }
                newNtracks --;
          }
          begin=newPos;
        }
        tracks->nTrueTracks = newNtracks;
//        // check everything went fine
//        printf("\n\nBEGIN FINAL VERTEX DUMP\n\n");
//        for (unsigned int ivtx = 0; ivtx < nTrueVertex; ivtx++){
//            printf("ivtx %d,%f\n", ivtx, orderedZ[ivtx]);
//        }
//        printf("\n\nEND FINAL VERTEX DUMP\n\n");
//
//        printf("\n\nBEGIN FINAL TRACKS DUMP\n\n");
//        for (unsigned int itrackO = 0; itrackO < tracks->nTrueTracks; itrackO++){
//            unsigned int itrack = tracks->order(itrackO);
//            printf("itrackO  %d,%d,%f\n", itrackO, itrack, tracks->z(itrack));
//        }
//        printf("\n\nEND FINAL TRACKS DUMP\n\n");
    }
    */

    __syncthreads();
    __shared__ float z[1024];
    __shared__ float rho[1024];

   //clock_t bigstart = clock();
    __syncthreads();

       //clock_t start = clock();
   if (threadIdx.x == 0 && blockIdx.x == 0){ 
        unsigned int nTrueVertex = 0;
        //unsigned int vertex_index[1024];
       //start = clock();
       size_t maxVerticesPerBlock = (int) (vertices->stride()/griddim);
       for (unsigned int blockid = 0; blockid < griddim ; blockid++){
           for(unsigned int ivtx = blockid * maxVerticesPerBlock; ivtx < blockid * maxVerticesPerBlock + vertices->nTrueVertex(blockid); ivtx++)   {
                unsigned int ivertex = vertices->order(ivtx);
                if ((std::abs(vertices->rho(ivertex)< 10000)) && (std::abs(vertices->z(ivertex))<30)) {
                    z[nTrueVertex] = vertices->z(ivertex);
                    rho[nTrueVertex] = vertices->rho(ivertex);
                    // printf("Copying vertex with ivertexO: %d, in position %d, z: %f , rho: %f \tSaving it in %d with z: %f\n", ivtx, ivertex, vertices->z(ivertex), vertices->rho(ivertex), _nTrueVertex, z[nTrueVertex]);
                    nTrueVertex ++;
                    if (nTrueVertex==1024) break;
                }
            }
       }
       vertices->nTrueVertex(0) = nTrueVertex;
       //nTrueVertex = _nTrueVertex;
//       printf("vertices : %d\n", nTrueVertex); 
//       printf("overlapped tracks : %d\n", tracks->nTrueTracks); 
       //printf("Took %d to copy vertices\n", (int)( clock() - start));
        //start = clock();
        /*
        unsigned int newNtracks = tracks->nTrueTracks;
        unsigned int begin = blockdim*int(newNtracks/blockdim);
        unsigned int end, newPos;
        while (begin > 0) {
           newPos = begin + (blockdim - int(blockdim/2));
           end = std::min(begin + (blockdim - int(blockdim/2)), newNtracks); 
           for (unsigned int i = begin; i < end; i++){
                tracks->order(i) = tracks->order(newPos);
                for (unsigned int j = newPos; j< newNtracks-1; j++){
                    tracks->order(j) = tracks->order(j+1);
                }
                newNtracks --;
          }
          begin -= blockdim;
        }
        tracks->nTrueTracks = newNtracks;
        */
       //printf("Took %d to remove overlap of tracks\n", (int)( clock() - start));
    }
    __syncthreads();
    //assert(vertices->nTrueVertex(0) < 1024);

    /*
    if (threadIdx.x == 0 && blockIdx.x == 0) {
            for (unsigned int ivtxO=0; ivtxO < nTrueVertex; ivtxO++){
                //unsigned int ivtx = orderedIndices[ivtxO];
                float z_i = z[ivtxO];
                printf("ivtxO %d, z %f\n", ivtxO, z_i);
            } 
    }
    __syncthreads();
    */

    //if (threadIdx.x == 0 && blockIdx.x == 0) printf("finished overlap removal\n");
    //__syncthreads();

    //start = clock();
    __shared__ __restrict__ uint16_t orderedIndices[1024];
    __shared__ uint16_t sws[1024];
    unsigned int const& nvFinal = vertices->nTrueVertex(0);
    //__shared__ unsigned int const& nvFinal;
    //if ( threadIdx.x == 0 && blockIdx.x == 0) nvFinal = nTrueVertex;

  #ifdef __CUDA_ARCH__
  radixSort<float, 2>(z, orderedIndices, sws, nvFinal); 
//    if (threadIdx.x == 0 && blockIdx.x == 0) {
//            printf("ivtxO,z\n");
//            for (unsigned int ivtxO=0; ivtxO < nTrueVertex; ivtxO++){
//                unsigned int ivtx = orderedIndices[ivtxO];
//                float z_i = z[ivtx];
//                printf("%d,%f\n", ivtxO, z_i);
//            } 
//    }
  #else
      if (threadIdx.x == 0 && blockIdx.x == 0) printf("vtx 0 - rho %f - z %f", rho[0], z[0]);
//            for (unsigned int itrackO=0; itrackO<tracks->nTrueTracks; itrackO++){
//                unsigned int itrack = trueTracksOrder[itrackO];
//                float z = tracksZ[itrack];
//                printf("itrack %d, itrackO %d, z %f\n", itrackO, itrack, z);
//            } 
//       }
  #endif

  __syncthreads();
   // if (threadIdx.x == 0 && blockIdx.x == 0) printf("For vertices sort it took %d\n", (int) (clock() - start));
 
    // reassign tracks to
    size_t firstElement = threadIdx.x + blockIdx.x * blockDim.x; // This is going to be the vertex index
    size_t gridSize = blockDim.x * gridDim.x;
   // if (threadIdx.x == 0 && blockIdx.x == 0) vertices->nTrueVertex(0) = nTrueVertex;
    __syncthreads();
    // copy sorted vertices back to the SoA
    for (unsigned int ivtx=firstElement; ivtx< vertices->nTrueVertex(0); ivtx+=gridSize){
        vertices->z(ivtx) = z[ivtx];
        vertices->rho(ivtx) = rho[ivtx];
        vertices->order(ivtx) = orderedIndices[ivtx];
    } 
    //if (threadIdx.x == 0 && blockIdx.x == 0) (*beta) = 0.1;
    __syncthreads();

    
//    resplitTracksKernel(10, tracks, vertices, params, 0.1, beta);
//  __syncthreads();

/*
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\n\nDUMP vertex in final cuda step\n\n");
        for (unsigned int ivertexO=0; ivertexO< nTrueVertex; ivertexO++){
           unsigned int ivertex = vertices->order(ivertexO);
            printf("vtx %d, %d, %f, %f\n", ivertexO, ivertex, vertices->z(ivertex), vertices->rho(ivertex));
        }
    }
  __syncthreads();
*/


    double zrange_min_ = 0.1;
     
    for (unsigned int itrackO = firstElement; itrackO < tracks->nTrueTracks ; itrackO += gridSize){
      unsigned int itrack = tracks->order(itrackO);
//    for (unsigned int itrack = firstElement; itrack < ntracks ; itrack+=gridSize){
//      if (not(tracks->isGood(itrack))) continue;
      // printf("%i vtx_range 1\n", threadIdx.x); 
      double zrange     = std::max(params.sel_zrange/ sqrt((*beta) * tracks->dz2(itrack)), zrange_min_);
      // printf("%i vtx_range 1.1, %p\n", threadIdx.x, (void*)&zrange);
      double zmin       = tracks->z(itrack) - zrange;
      // printf("%i vtx_range 1.2, %p\n", threadIdx.x, (void*)&zmin);
      //unsigned int kmin = std::min(nTrueVertex - 1,  tracks->kmin(itrack)); //We might have deleted a vertex, this might be complicated
      unsigned int kmin = vertices->nTrueVertex(0)-1;
      // printf("%i vtx_range 2, %p\n", threadIdx.x, (void*)&kmin);

      if (vertices->z(vertices->order(kmin)) > zmin){ // vertex properties always accessed through vertices->order
//        while ((kmin > maxVerticesPerBlock * blockIdx.x) && (vertices->z(vertices->order(std::max(kmin - 1,(unsigned int) maxVerticesPerBlock * blockIdx.x))) > zmin)) { // i.e., while we find another vertex within range that is before the previous initial step
        while ((kmin > 0) && (vertices->z(vertices->order(kmin - 1)) > zmin)) { // i.e., while we find another vertex within range that is before the previous initial step
          kmin--;
        }
      }
      else {
        while ((kmin < vertices->nTrueVertex(0)) && (vertices->z(vertices->order(kmin)) < zmin)) { // Or it might happen that we have to take out vertices from the thing
          kmin++;
        }
      }
      // printf("%i vtx_range 3\n", threadIdx.x);

      // Now the same for the upper bound
      double zmax       = tracks->z(itrack) + zrange;
      unsigned int kmax = 0;
      if (vertices->z(vertices->order(kmax))< zmax) {
        while (( kmax < vertices->nTrueVertex(0)  - 1) && ( vertices->z(vertices->order(kmax+1))< zmax )) { // As long as we have more vertex above kmax but within z range, we can add them to the collection, keep going
          kmax++;
        }
      }
      else { //Or maybe we have to restrict it
        while (( kmax > 0) && (vertices->z(vertices->order(kmax)) > zmax)) {
          kmax--;
        }
      }
      // printf("%i vtx_range 4\n", threadIdx.x);
      // Here kmin is the minimal index, kmax is the maximal
      if (kmin <= kmax) {
        // printf("start of vertices block + ntruevertex: %d\n\n",(unsigned int) (maxVerticesPerBlock * blockIdx.x) + vertices->nTrueVertex(blockIdx.x));
        
        tracks->kmin(itrack) = kmin;
        tracks->kmax(itrack) = kmax + 1; //always looping to tracks->kmax(i) - 1
      }
      else { // If it is here, the whole vertex are under
//        printf("\n\nWTF, kmin> kmax?! \n\n");
//        printf("start of vertices block + ntruevertex: %d\n\n",(unsigned int) (maxVerticesPerBlock * blockIdx.x) + vertices->nTrueVertex(blockIdx.x));
        tracks->kmin(itrack) = std::max(0U, std::min(kmin, kmax));
        tracks->kmax(itrack) = std::min(vertices->nTrueVertex(0), std::max(kmin, kmax) + 1);
      }
      // printf("%i vtx_range is finished\n", threadIdx.x);
    }
    // printf("%i device vtx_range is finished\n", threadIdx.x);
      __syncthreads(); 
   //if (threadIdx.x == 0 && blockIdx.x == 0) printf("For re assignment it took %d\n", (int) (clock() - start));
    
//   if (threadIdx.x == 0 && blockIdx.x == 0){ 
//        printf("tracks (itrackO, itrack, kmin, kmax\n");
//        for (unsigned itrackO = 0; itrackO < tracks->nTrueTracks; itrackO++){
//            unsigned int itrack = tracks->order(itrackO);
//            printf("track %d, %d, %d %d\n", itrackO, itrack, tracks->kmin(itrack), tracks->kmax(itrack));     
//        }
//   }

   __syncthreads();

  // if (threadIdx.x == 0 && blockIdx.x == 0) (*beta) = 2;
  __syncthreads();

  double mintrkweight_ = 0.5;
  double rho0 = vertices->nTrueVertex(0) > 1 ? 1./vertices->nTrueVertex(0) : 1.;
  double z_sum_init = rho0*exp(-(*beta)*params.dzCutOff*params.dzCutOff);
  //start = clock(); 
  //std::vector<std::vector<unsigned int> > vtx_track_indices(vertices->nTrueVertex);

  for (unsigned int itrackO = firstElement; itrackO < tracks->nTrueTracks; itrackO+=gridSize) {
    unsigned int itrack = tracks->order(itrackO);
    // if (not(tracks->isGood(i))) continue;
    unsigned int kmin = tracks->kmin(itrack);
    unsigned int kmax = tracks->kmax(itrack);
    

    //FIXME the original algorithm used to calculate sum_Z for each track and then use it as a denominator for all the vertex-track association probability. Since we want the maximum of the probability, and since each track should belong to one vertex it should be useless to calculate a common denominator but rather to use only the numerator
    //
    //
    double p_max = -1; 
    unsigned int iMax = 10000; 
    double sum_Z = z_sum_init;
    for (auto k = kmin; k < kmax; k++) {
      double v_exp = exp(-(*beta) * std::pow( tracks->z(itrack) - vertices->z(vertices->order(k)), 2) * tracks->dz2(itrack)) ;
      sum_Z += vertices->rho(vertices->order(k)) * v_exp;
      //sum_Z += v_exp;
    }
    double invZ = sum_Z > 1e-100 ? 1. / sum_Z : 0.0;
    for (auto k = kmin; k < kmax; k++) {
      float v_exp = exp(-(*beta) * std::pow( tracks->z(itrack) - vertices->z(vertices->order(k)), 2) * tracks->dz2(itrack)) ;
      float p = vertices->rho(vertices->order(k)) * v_exp * invZ; 
      //float p = v_exp * invZ; 
      //if (p > p_max && p > mintrkweight_) {
      if (p > p_max && p > mintrkweight_) {
        // assign  track i -> vertex k (hard, mintrkweight_ should be >= 0.5 here
        p_max = p;
        iMax = k;
      }
    }
    //associationItrack[iO] = tracks->tt_index(iO);
    //associationIvertex[iO] = iMax;
    tracks->kmin(itrack) = iMax; 
    tracks->kmax(itrack) = iMax+1; 

//    double sum_Z = z_sum_init;
//
//    for (auto k = kmin; k < kmax; k++) {
//      vertices->exp(ivertex) = exp(-(*beta) * std::pow( tracks->z(i) - orderedZ[k], 2) * tracks->dz2(i)) ;
//    }
//
//    //local_exp_list_range(y.exp_arg, y.exp, kmin, kmax);
//
//
//    tracks->sum_Z(i) = z_sum_init;
//    for (auto k = kmin; k < kmax; k++) {
//      tracks->sum_Z(i) += vertices->rho(ivertex) * vertices->exp(ivertex);
//    }
//    const double invZ = tracks->sum_Z(i) > 1e-100 ? 1. / tracks->sum_Z(i) : 0.0;
//
//    for (auto k = kmin; k < kmax; k++) {
//      unsigned int ivertex = vertices->order(k);
//      double p = vertices->rho(ivertex) * vertices->exp(ivertex) *invZ; 
//      if (p > mintrkweight_) {
//        // assign  track i -> vertex k (hard, mintrkweight_ should be >= 0.5 here
//        vtx_track_indices[k].push_back(i);
//        break;
//      }
//    }


  }  // track loop
  //   if (threadIdx.x == 0 && blockIdx.x == 0) printf("For hard assignment it took %d\n", (int) (clock() - start));
  __syncthreads();
   //if (threadIdx.x == 0 && blockIdx.x == 0) printf("Overall it took %d for all resort\n", (int) (clock() - bigstart));
//   if (threadIdx.x == 0 && blockIdx.x == 0){ 
//        printf("tracks vertex association (tt_index, itrack, ivertexO, track - z , ivertex z\n");
//        for (unsigned itrackO = 0; itrackO < tracks->nTrueTracks; itrackO++){
//            unsigned int itrack = tracks->order(itrackO);
//            printf("track - vertex %d, %d, %d, %f, %f\n", tracks->tt_index(itrack), itrack, tracks->kmin(itrack), tracks->z(itrack), orderedZ[tracks->kmin(itrack)]);     
//        }
//   }
}

__global__ void bigKernel(unsigned int ntracks, TrackForPV::TrackForPVSoA* tracks, TrackForPV::VertexForPVSoA* vertices, clusterParameters params, double* osumtkwt, double* beta){
    
  extern __shared__ double rbeta[];
  double * _beta = &rbeta[blockIdx.x];

  initializeKernel(ntracks, tracks, vertices, params);  
  __syncthreads();
  getBeta0Kernel(ntracks, tracks, vertices, params, _beta);
  __syncthreads();
//  if (threadIdx.x == 0 && blockIdx.x == 0)  printf("\n\nFinished get_beta0, \nBeta: %f \n", (float)rbeta[0]);
  thermalizeKernel(ntracks, tracks, vertices, params, osumtkwt, _beta, params.delta_highT, 0.0); // At the first iteration, rho0=0, no purging of any kind
  __syncthreads();
//  if (threadIdx.x == 0 && blockIdx.x == 0)  printf("\n\nFinished thermalize\n_beta: %f \n", (float)rbeta[0]);

  coolingWhileSplittingKernel(ntracks, tracks, vertices, params, osumtkwt, _beta); // At the first iteration, rho0=0, no purging of any kind 
  __syncthreads();
//  if (threadIdx.x == 0 && blockIdx.x == 0)  printf("\n\nFinished cooling\n");

  remergeTracksKernel(ntracks, tracks, vertices, params, osumtkwt, _beta); 
  __syncthreads();
//  if (threadIdx.x == 0 && blockIdx.x == 0)  printf("\n\nFinished remerge\n");

  resplitTracksKernel(ntracks, tracks, vertices, params, osumtkwt, _beta);
  __syncthreads();
//  if (threadIdx.x == 0 && blockIdx.x == 0)  printf("\n\nFinished resplit\n");

  outlierRejectionKernel(ntracks, tracks, vertices, params, osumtkwt, _beta);
  __syncthreads();
  //if (threadIdx.x == 0) printf("Beta at the end for block %d is %f\n", blockIdx.x, rbeta[0]);

  if (threadIdx.x == 0 && blockIdx.x == 0) (*beta) = rbeta[0];
  ////  if (threadIdx.x == 0 && blockIdx.x == 0)  printf("\n\nFinished outlier\n");

  //  dumpTVkernel(tracks, vertices, beta);
  __syncthreads();
}

#ifdef __CUDACC__
std::vector<TransientVertex> vertices(unsigned int ntracks, TrackForPV::TrackForPVSoA* tracks, TrackForPV::VertexForPVSoA* vertices, clusterParameters params, const std::vector<reco::TransientTrack>& t_tks, double* beta) {
  /*
  const unsigned int nv = vertices->nTrueVertex;
  for (unsigned int ivertexO = 0;ivertexO < nv; ivertexO++) {
    unsigned int ivertex = vertices->order(ivertexO);
    //std::cout << "V:" << ivertexO << " " << ivertex << " z: " << vertices->z(ivertex) << std::endl;
  }
  for (unsigned int ivertexO = 0;ivertexO < nv; ivertexO++) {
    unsigned int ivertex = vertices->order(ivertexO);
    if ( vertices->rho(ivertex) > 10000 || abs(vertices->z(ivertex)) > 10000) {
      vertices->rho(ivertex) = 0;
      vertices->z(ivertex) = 0;
    }
  }
  */
  std::vector<TransientVertex> clusters;
 // std::cout << "\n\nFound n vertices: " << vertices->nTrueVertex(0) << std::endl;
  std::vector<std::vector<unsigned int> > vtx_track_indices(vertices->nTrueVertex(0));
    
  for (unsigned int itrackO = 0; itrackO < tracks->nTrueTracks; itrackO++){
        unsigned int itrack = tracks->order(itrackO);
        unsigned int ivtx = tracks->kmin(itrack);
        if (ivtx < vertices->stride()){
          vtx_track_indices[ivtx].push_back(tracks->tt_index(itrack));
        }
        else{
//            std::cout << "rejecting vertex " << ivtx << vertices->z(vertices->order(ivtx))<< std::endl;
        }
  }

  /*
  const double mintrkweight_ = 0.5;
  double rho0 = vertices->nTrueVertex > 1 ? 1./vertices->nTrueVertex : 1.;
  const auto z_sum_init = rho0*exp(-(*beta)*params.dzCutOff*params.dzCutOff);

  std::vector<std::vector<unsigned int> > vtx_track_indices(vertices->nTrueVertex);

  for (unsigned int iO = 0; iO < tracks->nTrueTracks; iO++) {
    unsigned int i = tracks->order(iO);
    // if (not(tracks->isGood(i))) continue;
    const auto kmin = tracks->kmin(i);
    const auto kmax = tracks->kmax(i);
    for (auto k = kmin; k < kmax; k++) {
      unsigned int ivertex = vertices->order(k);
      vertices->exp(ivertex) = exp(-(*beta) * std::pow( tracks->z(i) - vertices->z(ivertex), 2) * tracks->dz2(i)) ;
    }

    //local_exp_list_range(y.exp_arg, y.exp, kmin, kmax);


    tracks->sum_Z(i) = z_sum_init;
    for (auto k = kmin; k < kmax; k++) {
      unsigned int ivertex = vertices->order(k);
      tracks->sum_Z(i) += vertices->rho(ivertex) * vertices->exp(ivertex);
    }
    const double invZ = tracks->sum_Z(i) > 1e-100 ? 1. / tracks->sum_Z(i) : 0.0;

    for (auto k = kmin; k < kmax; k++) {
      unsigned int ivertex = vertices->order(k);
      double p = vertices->rho(ivertex) * vertices->exp(ivertex) *invZ; 
      if (p > mintrkweight_) {
        // assign  track i -> vertex k (hard, mintrkweight_ should be >= 0.5 here
        vtx_track_indices[k].push_back(i);
        break;
      }
    }

  }  // track loop
  */
//  std::cout << "z vrtx,ith track" << std::endl;

  GlobalError dummyError(0.01, 0, 0.01, 0., 0., 0.01);
  for (unsigned int k = 0; k < vertices->nTrueVertex(0); k++) {
    if (!vtx_track_indices[k].empty()) {
      unsigned int ivertex = vertices->order(k);
      GlobalPoint pos(0, 0, vertices->z(ivertex));
      std::vector<reco::TransientTrack> vertexTracks;
      for (auto i : vtx_track_indices[k]) {
//        std::cout << vertices->z(ivertex) << "," << tracks->z(i) << std::endl;
//        std::cout << "Adding track " << i << " to vertex " << k << std::endl;
//        std::cout << vertices->z(ivertex) << "," << t_tks.at(i).stateAtBeamLine().trackStateAtPCA().position().z() << std::endl;
        vertexTracks.push_back(t_tks[i]);
      }
      TransientVertex v(pos, dummyError, vertexTracks, 0);
      clusters.push_back(v);
    }
  }

  return clusters;
}

std::vector<std::vector<reco::TransientTrack>> clusterize(std::vector<TransientVertex>& pv, clusterParameters params) {
  std::vector<std::vector<reco::TransientTrack> > clusters;
  // vector<TransientVertex>&& pv = vertices(tracks);


  if (pv.empty()) {
    return clusters;
  }

  // fill into clusters and merge
  std::vector<reco::TransientTrack> aCluster = pv.begin()->originalTracks();

  for (auto k = pv.begin() + 1; k != pv.end(); k++) {
    if (std::abs(k->position().z() - (k - 1)->position().z()) > (2 * params.vertexSize)) {
      // close a cluster
      if (aCluster.size() > 1) {
        clusters.push_back(aCluster);
      }
#ifdef DEBUG
      else {
        std::cout << " one track cluster at " << k->position().z() << "  suppressed" << std::endl;
      }
#endif
      aCluster.clear();
    }
    for (unsigned int i = 0; i < k->originalTracks().size(); i++) {
      aCluster.push_back(k->originalTracks()[i]);
    }
  }
  clusters.emplace_back(std::move(aCluster));

  return clusters;
}
/*
*/

/*
void dumpTV(TrackForPV::TrackForPVSoA* tracks, TrackForPV::VertexForPVSoA* vertices, unsigned int gridSize){
    std::cout << "\n\nBEGIN TRACKS\n\n" << std::endl;
    for(unsigned itrackO = 0; itrackO < tracks->nTrueTracks; itrackO ++){
         unsigned int itrack = tracks->order(itrackO);
         std::cout << itrackO << "," << itrack << "," << tracks->z(itrack) << std::endl;
    }
    std::cout <<  "\n\nEND TRACKS\n\nBEGIN TV" << std::endl;
    for(unsigned itrackO = 0; itrackO < tracks->nTrueTracks; itrackO ++){
         unsigned int itrack = tracks->order(itrackO);
         for (unsigned int ivertexO = tracks->kmin(itrack); ivertexO < tracks->kmax(itrack); ivertexO ++){
             std::cout << itrackO << "," << ivertexO << std::endl;
        }
    }
    std::cout << "\n\nEND TV\n\nBEGIN VERTICES" << std::endl;
    
    size_t maxVerticesPerBlock = (int) (vertices->stride()/gridSize);
    for(unsigned int ivertexO = 0; ivertexO < vertices->stride(); ivertexO ++){
        unsigned int ivertex = vertices->order(ivertexO);
        if (vertices->isGood(ivertex)) {
            std::cout << ivertexO << "," << vertices->z(ivertex) << std::endl; 
        }
    }
    std::cout << "\n\nEND VERTICES\n\n" << std::endl;
 }
*/
void bigKernelWrapper(unsigned int ntracks, TrackForPV::TrackForPVSoA* tracks, TrackForPV::VertexForPVSoA* vertices, double* beta, double* osumtkwt, clusterParameters params, cudaStream_t stream){
  // Initialize is vectorized across tracks
  unsigned int blockSize = 512;
  unsigned int gridSize  = 1;
  /*
  unsigned int blockSize = 128;
  unsigned int gridSize  = 64;
  */
  // auto GPUbeta = cms::cuda::make_device_unique<double[1]>(1, cudaStreamDefault);                                         // 1/T, to be kept across iterations

//  overlapTracks<<<1,1,0,stream>>>(tracks, blockSize, gridSize);
//  cudaCheck(cudaGetLastError());
  
  bigKernel<<<gridSize, blockSize,blockSize,stream>>>(ntracks, tracks, vertices, params, osumtkwt, beta);
  cudaCheck(cudaGetLastError());
  //resortVerticesAndAssign<<<1,blockSize,0,stream>>>(tracks, vertices, beta, params, blockSize, gridSize);
  resortVerticesAndAssign<<<1,blockSize,0,stream>>>(tracks, vertices, beta, params, blockSize, gridSize);
  cudaCheck(cudaGetLastError());
  /*
    __global__ void fitVertices( TrackForPV::TrackForPVSoA* tracks, TrackForPV::VertexForPVSoA* vertices ){

    }
 */
  //TrackForPV::RecoVertices<256>  
  //fitVertices<<<nClusters, 256, 0, stream>>>(tracks, vertices);
/*
  initializeKernel<<<gridSize, blockSize,0,stream>>>(ntracks, tracks, vertices, params);  
  cudaCheck(cudaGetLastError());
  getBeta0Kernel<<<gridSize, blockSize,0,stream>>>(ntracks, tracks, vertices, params, beta);
  cudaCheck(cudaGetLastError());
  thermalizeKernel<<<gridSize, blockSize,0,stream>>>(ntracks, tracks, vertices, params, osumtkwt, beta, params.delta_highT, 0.0); // At the first iteration, rho0=0, no purging of any kind
  cudaCheck(cudaGetLastError());
  coolingWhileSplittingKernel<<<gridSize, blockSize,0,stream>>>(ntracks, tracks, vertices, params, osumtkwt, beta); // At the first iteration, rho0=0, no purging of any kind 
  cudaCheck(cudaGetLastError());
  remergeTracksKernel<<<gridSize, blockSize,0,stream>>>(ntracks, tracks, vertices, params, osumtkwt, beta); 
  cudaCheck(cudaGetLastError());
  resplitTracksKernel<<<gridSize, blockSize,0,stream>>>(ntracks, tracks, vertices, params, osumtkwt, beta);
  cudaCheck(cudaGetLastError());
  outlierRejectionKernel<<<gridSize, blockSize,0,stream>>>(ntracks, tracks, vertices, params, osumtkwt, beta);
  cudaCheck(cudaGetLastError());
*/
}
/*
  // Only on GPUs, of course...
void initializeWrapper(unsigned int ntracks, TrackForPV::TrackForPVSoA* tracks, TrackForPV::VertexForPVSoA* vertices, double* beta, double* osumtkwt, clusterParameters params, cudaStream_t stream){
  // Initialize is vectorized across tracks
  unsigned int blockSize = 256;
  unsigned int gridSize  = 1;
  initializeKernel<<<gridSize, blockSize,0,stream>>>(ntracks, tracks, vertices, params);  
  cudaCheck(cudaGetLastError());
}

void getBeta0Wrapper(unsigned int ntracks, TrackForPV::TrackForPVSoA* tracks, TrackForPV::VertexForPVSoA* vertices, double* beta, double* osumtkwt, clusterParameters params, cudaStream_t stream){
  // Beta0 is vectorized across tracks
  unsigned int blockSize = 256;
  unsigned int gridSize  = 1;
  getBeta0Kernel<<<gridSize, blockSize,0,stream>>>(ntracks, tracks, vertices, params, beta);
  cudaCheck(cudaGetLastError());
}

void thermalizeWrapper(unsigned int ntracks, TrackForPV::TrackForPVSoA* tracks, TrackForPV::VertexForPVSoA* vertices, double* beta, double* osumtkwt, clusterParameters params, cudaStream_t stream){
  // Thermalize is vectorized -mostly- across tracks
  unsigned int blockSize = 256;
  unsigned int gridSize  = 1;
  thermalizeKernel<<<gridSize, blockSize,0,stream>>>(ntracks, tracks, vertices, params, osumtkwt, beta, params.delta_highT, 0.0); // At the first iteration, rho0=0, no purging of any kind
  cudaCheck(cudaGetLastError());
}


void coolingWhileSplittingWrapper(unsigned int ntracks, TrackForPV::TrackForPVSoA* tracks, TrackForPV::VertexForPVSoA* vertices, double* beta, double* osumtkwt, clusterParameters params, cudaStream_t stream){
  // Vectorized across tracks, as it calls thermalize and update copiusly
  unsigned int blockSize = 256;
  unsigned int gridSize  = 1;
  coolingWhileSplittingKernel<<<gridSize, blockSize,0,stream>>>(ntracks, tracks, vertices, params, osumtkwt, beta); // At the first iteration, rho0=0, no purging of any kind 
  cudaCheck(cudaGetLastError());
}

void remergeTracksWrapper(unsigned int ntracks, TrackForPV::TrackForPVSoA* tracks, TrackForPV::VertexForPVSoA* vertices, double* beta, double* osumtkwt, clusterParameters params, cudaStream_t stream){
  // Vectorized across tracks, as it calls update
  unsigned int blockSize = 256;
  unsigned int gridSize  = 1;
  remergeTracksKernel<<<gridSize, blockSize,0,stream>>>(ntracks, tracks, vertices, params, osumtkwt, beta); 
  cudaCheck(cudaGetLastError());
}

void resplitTracksWrapper(unsigned int ntracks, TrackForPV::TrackForPVSoA* tracks, TrackForPV::VertexForPVSoA* vertices, double* beta, double* osumtkwt, clusterParameters params, cudaStream_t stream){
  // Vectorized across tracks, as it calls update
  unsigned int blockSize = 256;
  unsigned int gridSize  = 1;
  resplitTracksKernel<<<gridSize, blockSize,0,stream>>>(ntracks, tracks, vertices, params, osumtkwt, beta);
  cudaCheck(cudaGetLastError());
}

void outlierRejectionWrapper(unsigned int ntracks, TrackForPV::TrackForPVSoA* tracks, TrackForPV::VertexForPVSoA* vertices, double* beta, double* osumtkwt, clusterParameters params, cudaStream_t stream){
  // Vectorized across tracks, as it calls update
  unsigned int blockSize = 256;
  unsigned int gridSize  = 1;
  outlierRejectionKernel<<<gridSize, blockSize,0,stream>>>(ntracks, tracks, vertices, params, osumtkwt, beta);
  cudaCheck(cudaGetLastError());
}
*/
#endif
} // namespace clusterizerCUDA
