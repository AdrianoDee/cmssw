#include "RecoPixelVertexing/PixelTriplets/plugins/GPUCACell.h"

#include <typeinfo>
#include <iostream>

template <typename T>
void print() {
  std::cout << "size of " << typeid(T).name() << ' ' << sizeof(T) << std::endl;
}

int main() {
  using namespace caConstants;

  //for Phase-I
/*  print<GPUCACellT<caConstants::trackerConstantPhase1>>();
  print<CellNeighborsT<caConstants::trackerConstantPhase1>>();
  print<CellTracksT<caConstants::trackerConstantPhase1>>();
  print<OuterHitOfCellContainerT<caConstants::trackerConstantPhase1>>();
  print<TuplesContainerT<caConstants::trackerConstantPhase1>>();
  print<HitToTupleT<caConstants::trackerConstantPhase1>>();
  print<TupleMultiplicityT<caConstants::trackerConstantPhase1>>();

  print<CellNeighborsVectorT<caConstants::trackerConstantPhase1>>();

  //for Phase-II

  print<GPUCACellT<caConstants::trackerConstantPhase2>>();
  print<CellNeighborsT<caConstants::trackerConstantPhase2>>();
  print<CellTracksT<caConstants::trackerConstantPhase2>>();
  print<OuterHitOfCellContainerT<caConstants::trackerConstantPhase2>>();
  print<TuplesContainerT<caConstants::trackerConstantPhase2>>();
  print<HitToTupleT<caConstants::trackerConstantPhase2>>();
  print<TupleMultiplicityT<caConstants::trackerConstantPhase2>>();

  print<CellNeighborsVectorT<caConstants::trackerConstantPhase2>>();
*/
  return 0;
}
