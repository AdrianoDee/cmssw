#include "BrokenLineFitOnGPU.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

void HelixFitOnGPU::launchBrokenLineKernels(HitSoAConstView hv,
                                            uint32_t hitsInFit,
                                            uint32_t maxNumberOfTuples,
                                            cudaStream_t stream) {
  assert(tuples_);

  auto blockSize = 64;
  auto numberOfBlocks = (maxNumberOfConcurrentFits_ + blockSize - 1) / blockSize;

  //  Fit internals
  auto tkidGPU = cms::cuda::make_device_unique<caConstants::tindex_type[]>(maxNumberOfConcurrentFits_, stream);
  auto hitsGPU = cms::cuda::make_device_unique<double[]>(
      maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix3xNd<6>) / sizeof(double), stream);
  auto hits_geGPU = cms::cuda::make_device_unique<float[]>(
      maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix6xNf<6>) / sizeof(float), stream);
  auto fast_fit_resultsGPU = cms::cuda::make_device_unique<double[]>(
      maxNumberOfConcurrentFits_ * sizeof(riemannFit::Vector4d) / sizeof(double), stream);

  for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
    // fit triplets
    std::cout << "BITCH 0_" << offset << std::endl;
    kernel_BLFastFit<3><<<numberOfBlocks, blockSize, 0, stream>>>(tuples_,
                                                                  tupleMultiplicity_,
                                                                  hv,
                                                                  tkidGPU.get(),
                                                                  hitsGPU.get(),
                                                                  hits_geGPU.get(),
                                                                  fast_fit_resultsGPU.get(),
                                                                  3,
                                                                  3,
                                                                  offset);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    std::cout << "BITCH 1_" << offset << std::endl;
    kernel_BLFit<3><<<numberOfBlocks, blockSize, 0, stream>>>(tupleMultiplicity_,
                                                              bField_,
                                                              outputSoa_,
                                                              tkidGPU.get(),
                                                              hitsGPU.get(),
                                                              hits_geGPU.get(),
                                                              fast_fit_resultsGPU.get());
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    std::cout << "BITCH 2_" << offset << std::endl;
    if (fitNas4_) {
      // fit all as 4
      std::cout << "BITCH 2a_" << offset << std::endl;
      kernel_BLFastFit<4><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tuples_,
                                                                        tupleMultiplicity_,
                                                                        hv,
                                                                        tkidGPU.get(),
                                                                        hitsGPU.get(),
                                                                        hits_geGPU.get(),
                                                                        fast_fit_resultsGPU.get(),
                                                                        4,
                                                                        8,
                                                                        offset);
      cudaCheck(cudaGetLastError());
      cudaCheck(cudaDeviceSynchronize());
      std::cout << "BITCH 3_" << offset << std::endl;
      kernel_BLFit<4><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tupleMultiplicity_,
                                                                    bField_,
                                                                    outputSoa_,
                                                                    tkidGPU.get(),
                                                                    hitsGPU.get(),
                                                                    hits_geGPU.get(),
                                                                    fast_fit_resultsGPU.get());
      std::cout << "BITCH 4_" << offset << std::endl;
    } else {
      // fit quads
      std::cout << "BITCH 2b_" << offset << std::endl;
      kernel_BLFastFit<4><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tuples_,
                                                                        tupleMultiplicity_,
                                                                        hv,
                                                                        tkidGPU.get(),
                                                                        hitsGPU.get(),
                                                                        hits_geGPU.get(),
                                                                        fast_fit_resultsGPU.get(),
                                                                        4,
                                                                        4,
                                                                        offset);
      cudaCheck(cudaGetLastError());
      cudaCheck(cudaDeviceSynchronize());
      std::cout << "BITCH 5_" << offset << std::endl;
      kernel_BLFit<4><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tupleMultiplicity_,
                                                                    bField_,
                                                                    outputSoa_,
                                                                    tkidGPU.get(),
                                                                    hitsGPU.get(),
                                                                    hits_geGPU.get(),
                                                                    fast_fit_resultsGPU.get());
      std::cout << "BITCH 6_" << offset << std::endl;
      // fit penta (all 5)
      kernel_BLFastFit<5><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tuples_,
                                                                        tupleMultiplicity_,
                                                                        hv,
                                                                        tkidGPU.get(),
                                                                        hitsGPU.get(),
                                                                        hits_geGPU.get(),
                                                                        fast_fit_resultsGPU.get(),
                                                                        5,
                                                                        5,
                                                                        offset);
      cudaCheck(cudaGetLastError());
      cudaCheck(cudaDeviceSynchronize());
      std::cout << "BITCH 7_" << offset << std::endl;
      kernel_BLFit<5><<<8, blockSize, 0, stream>>>(tupleMultiplicity_,
                                                   bField_,
                                                   outputSoa_,
                                                   tkidGPU.get(),
                                                   hitsGPU.get(),
                                                   hits_geGPU.get(),
                                                   fast_fit_resultsGPU.get());
      cudaCheck(cudaGetLastError());
      std::cout << "BITCH 8_" << offset << std::endl;
      // fit sexta and above (as 6)
      kernel_BLFastFit<6><<<4, blockSize, 0, stream>>>(tuples_,
                                                       tupleMultiplicity_,
                                                       hv,
                                                       tkidGPU.get(),
                                                       hitsGPU.get(),
                                                       hits_geGPU.get(),
                                                       fast_fit_resultsGPU.get(),
                                                       6,
                                                       8,
                                                       offset);
      cudaCheck(cudaGetLastError());
      cudaCheck(cudaDeviceSynchronize());
      std::cout << "BITCH 9_" << offset << std::endl;
      kernel_BLFit<6><<<4, blockSize, 0, stream>>>(tupleMultiplicity_,
                                                   bField_,
                                                   outputSoa_,
                                                   tkidGPU.get(),
                                                   hitsGPU.get(),
                                                   hits_geGPU.get(),
                                                   fast_fit_resultsGPU.get());
      cudaCheck(cudaGetLastError());
      cudaCheck(cudaDeviceSynchronize());
      std::cout << "BITCH 10_" << offset << std::endl;
    }

  }  // loop on concurrent fits
}
