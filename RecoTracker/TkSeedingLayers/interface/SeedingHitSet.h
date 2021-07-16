#ifndef TkSeedingLayers_SeedingHitSet_H
#define TkSeedingLayers_SeedingHitSet_H

#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>
#include <algorithm>

namespace seedinghits {
  constexpr size_t kMaxSeedSize = 16;  // 16 is 4+12 endcap layers for Phase II Pixel detector
}

using namespace seedinghits;

class SeedingHitSet {
public:
  using RecHit = BaseTrackerRecHit;
  using RecHitPointer = BaseTrackerRecHit *;
  using ConstRecHitPointer = BaseTrackerRecHit const *;

  static ConstRecHitPointer nullPtr() { return nullptr; }

  SeedingHitSet() {
    theRecHits[0] = theRecHits[1] = theRecHits[2] = theRecHits[3] = nullptr;
    theSize = 0;
  }

  SeedingHitSet(ConstRecHitPointer one, ConstRecHitPointer two)
  // : theRecHits{{one,two,ConstRecHitPointer()}}
  {
    theRecHits[0] = one;
    theRecHits[1] = two;
    theRecHits[2] = theRecHits[3] = nullptr;

    theSize = 2;
  }
  SeedingHitSet(ConstRecHitPointer one, ConstRecHitPointer two, ConstRecHitPointer three)
  // : theRecHits{{one,two,three}},
  {
    theRecHits[0] = one;
    theRecHits[1] = two;
    theRecHits[2] = three;
    theRecHits[3] = nullptr;

    theSize = 3;
  }

  SeedingHitSet(ConstRecHitPointer one, ConstRecHitPointer two, ConstRecHitPointer three, ConstRecHitPointer four) {
    theRecHits[0] = one;
    theRecHits[1] = two;
    theRecHits[2] = three;
    theRecHits[3] = four;

    theSize = 4;
  }

  SeedingHitSet(std::vector<ConstRecHitPointer> &hits) {
    theSize = std::min(hits.size(), kMaxSeedSize);
    std::copy(hits.begin(), hits.begin() + theSize, theRecHits);
    if (hits.size() > kMaxSeedSize) {
      edm::LogWarning("SeedingHitSet") << "More than " << kMaxSeedSize << " hits give in input. Taking only the first "
                                       << kMaxSeedSize;
    } else {
      std::fill(theRecHits + theSize, theRecHits + kMaxSeedSize, nullptr);
    }
  }

  ConstRecHitPointer const *data() const { return theRecHits; }

  unsigned int size() const { return theSize; }

  ConstRecHitPointer get(unsigned int i) const { return theRecHits[i]; }
  ConstRecHitPointer operator[](unsigned int i) const { return theRecHits[i]; }

private:
  ConstRecHitPointer theRecHits[kMaxSeedSize];
  unsigned int theSize;
};

#endif
