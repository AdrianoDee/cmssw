#ifndef RecoTracker_PixelSeeding_src_CellularAutomaton_h
#define RecoTracker_PixelSeeding_src_CellularAutomaton_h

#include <array>

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"

#include "CACell.h"
#include "RecoTracker/PixelSeeding/interface/CAGraph.h"

class CellularAutomaton {
public:
  CellularAutomaton(CAGraph& graph) : theLayerGraph(graph) {}

  std::vector<CACell>& getAllCells() { return allCells; }

  // counters for the per-event statistics printout
  struct Stats {
    unsigned long long candidates = 0;  // (inner, outer) cell pairs sharing a hit
    unsigned long long passTheta = 0;   // ... passing the RZ alignment (theta) cut
  };
  const Stats& stats() const { return theStats; }
  unsigned int numberOfRootCells() const { return theRootCells.size(); }

  void createAndConnectCells(
      const std::vector<const HitDoublets*>&, const TrackingRegion&, const CACut&, const CACut&, const float);

  void evolve(const unsigned int);
  void findNtuplets(std::vector<CACell::CAntuplet>&, const unsigned int);
  void findTriplets(const std::vector<const HitDoublets*>& hitDoublets,
                    std::vector<CACell::CAntuplet>& foundTriplets,
                    const TrackingRegion& region,
                    const CACut& thetaCut,
                    const CACut& phiCut,
                    const float hardPtCut);

private:
  CAGraph& theLayerGraph;

  std::vector<CACell> allCells;
  std::vector<CACellStatus> allStatus;

  std::vector<unsigned int> theRootCells;
  std::vector<std::vector<CACell*> > theNtuplets;
  Stats theStats;
};

#endif  // RecoTracker_PixelSeeding_src_CellularAutomaton_h
