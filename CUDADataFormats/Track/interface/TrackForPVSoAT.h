#ifndef CUDADataFormats_Track_TrackForPVSoAT_H
#define CUDADataFormats_Track_TrackForPVSoAT_H

#include <string>
#include <algorithm>

#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"

#include "CUDADataFormats/Common/interface/HeterogeneousSoA.h"
#include "HeterogeneousCore/CUDAUtilities/interface/eigenSoA.h"
#include <Eigen/Dense>

using Vector512d = Eigen::Matrix<double, 1024, 1>;

template <int32_t S>
class TrackForPVSoAHeterogeneousT {
public:
  static constexpr int32_t stride() { return S; }

public:
  // Track properties needed for the PV selection + fitting
  unsigned int nTrueTracks;
  double max_z;
  double min_z;
  eigenSoA::ScalarSoA<double, S> significance;
  //eigenSoA::ScalarSoA<double, S> dxy2;
  eigenSoA::ScalarSoA<double, S> dz2; // used in clusterizer
  eigenSoA::ScalarSoA<double, S> z;
  eigenSoA::ScalarSoA<double, S> weight;
  eigenSoA::ScalarSoA<double, S> sum_Z;
  eigenSoA::ScalarSoA<unsigned int, S> kmin;
  eigenSoA::ScalarSoA<unsigned int, S> kmax;
  eigenSoA::ScalarSoA<bool, S> isGood;
  eigenSoA::ScalarSoA<int, S> order;
  eigenSoA::ScalarSoA<unsigned int, S> tt_index;

  // For now, we can consider saving the full 4-momentum?
  //eigenSoA::ScalarSoA<double, S> pAtIP;
  //eigenSoA::ScalarSoA<double, S> etaAtIP;
  //eigenSoA::ScalarSoA<double, S> pxAtPCA;
  //eigenSoA::ScalarSoA<double, S> pyAtPCA;
  //eigenSoA::ScalarSoA<double, S> pzAtPCA;
  //eigenSoA::ScalarSoA<double, S> bx;
  //eigenSoA::ScalarSoA<double, S> by;

  //eigenSoA::ScalarSoA<double, S> chi2;

  //eigenSoA::ScalarSoA<int8_t, S> nPixelHits;
  //eigenSoA::ScalarSoA<int8_t, S> nTrackerHits;

  // The track-vertex association matrices
  eigenSoA::MatrixSoA<Vector512d, S> vert_sw;
  eigenSoA::MatrixSoA<Vector512d, S> vert_se;
  eigenSoA::MatrixSoA<Vector512d, S> vert_swz;
  eigenSoA::MatrixSoA<Vector512d, S> vert_swE;
  eigenSoA::MatrixSoA<Vector512d, S> vert_exp;
  eigenSoA::MatrixSoA<Vector512d, S> vert_exparg;

  // Auxiliar vectors
  eigenSoA::ScalarSoA<double, S> aux1;
  eigenSoA::ScalarSoA<double, S> aux2;
};

template <int32_t S>
class VertexForPVSoAHeterogeneousT {
public:
  static constexpr int32_t stride() { return S; }

public:
  // Track properties needed for the PV selection + fitting
  //unsigned int nTrueVertex;
  eigenSoA::ScalarSoA<unsigned int, S> nTrueVertex;  
  eigenSoA::ScalarSoA<bool, S> isGood;  
  eigenSoA::ScalarSoA<double, S> sw;
  eigenSoA::ScalarSoA<double, S> se;
  eigenSoA::ScalarSoA<double, S> swz;
  eigenSoA::ScalarSoA<double, S> swE;
  eigenSoA::ScalarSoA<double, S> exp;    // We probably don't need this
  eigenSoA::ScalarSoA<double, S> exparg; // Or this
  eigenSoA::ScalarSoA<int, S> order;
  eigenSoA::ScalarSoA<double, S> z;
  eigenSoA::ScalarSoA<double, S> rho;
  // Auxiliar vectors
  eigenSoA::ScalarSoA<double, S> aux1;
  eigenSoA::ScalarSoA<double, S> aux2;
};


namespace TrackForPV {

#ifdef GPU_SMALL_EVENTS
  // kept for testing and debugging
  constexpr uint32_t maxNumberT() { return 2 * 1024; }
  constexpr uint32_t maxNumberV() { return 1024; }
#else
  // tested on MC events with 55-75 pileup events
  constexpr uint32_t maxNumberT() { return 16 * 1024; }
  constexpr uint32_t maxNumberV() { return 1024; }
#endif

  using TrackForPVSoA  = TrackForPVSoAHeterogeneousT<maxNumberT()>;
  using VertexForPVSoA = VertexForPVSoAHeterogeneousT<maxNumberV()>;
}  
#endif
