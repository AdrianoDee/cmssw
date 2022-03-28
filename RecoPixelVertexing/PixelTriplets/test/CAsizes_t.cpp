#include "RecoPixelVertexing/PixelTriplets/plugins/GPUCACell.h"

#include <typeinfo>
#include <iostream>

template <typename T>
void print() {
  std::cout << "size of " << typeid(T).name() << ' ' << sizeof(T) << std::endl;
}

int main() {
  using namespace pixelTopology;

  //for Phase-I
  print<GPUCACellT<pixelTopology::Phase1>>();
  print<CellNeighborsT<pixelTopology::Phase1>>();
  print<CellTracksT<pixelTopology::Phase1>>();
  print<OuterHitOfCellContainerT<pixelTopology::Phase1>>();
  print<TuplesContainerT<pixelTopology::Phase1>>();
  print<HitToTupleT<pixelTopology::Phase1>>();
  print<TupleMultiplicityT<pixelTopology::Phase1>>();

  print<CellNeighborsVectorT<pixelTopology::Phase1>>();

  //for Phase-II

  print<GPUCACellT<pixelTopology::Phase2>>();
  print<CellNeighborsT<pixelTopology::Phase2>>();
  print<CellTracksT<pixelTopology::Phase2>>();
  print<OuterHitOfCellContainerT<pixelTopology::Phase2>>();
  print<TuplesContainerT<pixelTopology::Phase2>>();
  print<HitToTupleT<pixelTopology::Phase2>>();
  print<TupleMultiplicityT<pixelTopology::Phase2>>();

  print<CellNeighborsVectorT<pixelTopology::Phase2>>();

  return 0;
}
