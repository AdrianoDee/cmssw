#ifndef TkSeedingLayers_SeedingHitSet_H
#define TkSeedingLayers_SeedingHitSet_H

#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"

#define SEEDMAXSIZE 16

class SeedingHitSet {
public:
  using RecHit = BaseTrackerRecHit;
  using RecHitPointer = BaseTrackerRecHit *;
  using ConstRecHitPointer = BaseTrackerRecHit const *;

  static ConstRecHitPointer nullPtr() { return nullptr; }

  SeedingHitSet()
  {
    theSize = 0;

    for (uint8_t i = theSize; i < SEEDMAXSIZE; i++) {
      theRecHits[i] = nullptr;
    }

  }

  SeedingHitSet(ConstRecHitPointer one, ConstRecHitPointer two)
  // : theRecHits{{one,two,ConstRecHitPointer()}}
  {
    theRecHits[0] = one;
    theRecHits[1] = two;
    theRecHits[2] = theRecHits[3] = nullptr;

    theSize = 2;

    for (uint8_t i = theSize; i < SEEDMAXSIZE; i++) {
      theRecHits[i] = nullptr;
    }

  }
  SeedingHitSet(ConstRecHitPointer one, ConstRecHitPointer two, ConstRecHitPointer three)
  // : theRecHits{{one,two,three}},
  {
    theRecHits[0] = one;
    theRecHits[1] = two;
    theRecHits[2] = three;
    theRecHits[3] = nullptr;

    theSize = 3;

    for (uint8_t i = theSize; i < SEEDMAXSIZE; i++) {
      theRecHits[i] = nullptr;
    }

  }

  SeedingHitSet(ConstRecHitPointer one, ConstRecHitPointer two, ConstRecHitPointer three, ConstRecHitPointer four) {
    theRecHits[0] = one;
    theRecHits[1] = two;
    theRecHits[2] = three;
    theRecHits[3] = four;

    theSize = 4;

    for (uint8_t i = theSize; i < SEEDMAXSIZE; i++) {
      theRecHits[i] = nullptr;
    }
  }

  SeedingHitSet(std::vector<ConstRecHitPointer> vHits) {

    theSize =  uint8_t(vHits.size());
    for (uint8_t i = 0; i < theSize; i++) {
      theRecHits[i] = vHits[i];
    }
    for (uint8_t i = theSize; i < SEEDMAXSIZE; i++) {
      theRecHits[i] = nullptr;
    }
  }

  ConstRecHitPointer const *data() const { return theRecHits; }

  unsigned int size() const { return theSize; }

  ConstRecHitPointer get(unsigned int i) const { return theRecHits[i]; }
  ConstRecHitPointer operator[](unsigned int i) const { return theRecHits[i]; }

private:
  ConstRecHitPointer theRecHits[SEEDMAXSIZE];
  uint8_t theSize;

};

#endif
