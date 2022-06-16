// #include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
// #include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
// #include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
// #include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
// #include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
//
// template<>
// __device__ __forceinline__ bool TrackingRecHit2DSOAViewT<pixelTopology::Phase1>::zSizeCut(int i, int o, int inner, int outer)
// {
//   auto dz = zGlobal(i)-zGlobal(o);
//   auto dr = zGlobal(i)-rGlobal(o);
//   auto mes = clusterSizeY(i);
//   auto so = clusterSizeY(o);
//
//   auto onlyBarrel = outer < 4;
//   auto dy = inner == 0 ? pixelTopology::Phase1::maxDYsize12 : pixelTopology::Phase1::maxDYsize;
//
//   return onlyBarrel ? mes > 0 && so > 0 && std::abs(so - mes) > dy
//                     : (inner < 4) && mes > 0 &&
//                           std::abs(mes - int(std::abs(dz / dr) * pixelTopology::Phase1::dzdrFact + 0.5f)) > pixelTopology::Phase1::maxDYPred;
// }
//
//
// template<>
// __device__ __forceinline__ bool TrackingRecHit2DSOAViewT<pixelTopology::Phase1>::clusterCut(int i, int o, int inner, int outer) {
//
//   auto mi = detectorIndex(i);
//   bool innerB1 = mi < pixelTopology::Phase1::last_bpix1_detIndex;
//   bool outerFwd = (mi >= pixelTopology::Phase1::last_barrel_detIndex);
//   auto mes = clusterSizeY(i);
//
//   if (!outerFwd)
//     return false;
//
//   if (innerB1 && outerFwd)  // B1 and F1
//     if (mes > 0 && mes < pixelTopology::Phase1::minYsizeB1)
//       return true; // only long cluster  (5*8)
//   bool innerB2 = (mi >= pixelTopology::Phase1::last_bpix1_detIndex) && (mi <=pixelTopology::Phase1::last_bpix2_detIndex); //FIXME number
//   if (innerB2 && outerFwd)  // B2 and F1
//     if (mes > 0 && mes < pixelTopology::Phase1::minYsizeB2)
//       return true;
//
//   return false;
// }
//
// template class TrackingRecHit2DSOAViewT<pixelTopology::Phase1>;
// template class TrackingRecHit2DSOAViewT<pixelTopology::Phase2>;
