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

    for (unsigned int i=0; i < SEEDMAXSIZE; i++) {
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

    for (unsigned int i = 2; i < SEEDMAXSIZE; i++) {
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

    for (unsigned int i = 3; i < SEEDMAXSIZE; i++) {
      theRecHits[i] = nullptr;
    }

  }

  SeedingHitSet(ConstRecHitPointer one, ConstRecHitPointer two, ConstRecHitPointer three, ConstRecHitPointer four) {
    theRecHits[0] = one;
    theRecHits[1] = two;
    theRecHits[2] = three;
    theRecHits[3] = four;

    theSize = 4;

    for (unsigned int i = 4; i < SEEDMAXSIZE; i++) {
      theRecHits[i] = nullptr;
    }
  }

  SeedingHitSet(std::vector<ConstRecHitPointer> vHits) {

    theSize = (vHits.size());
    for (unsigned int i = 0; i < theSize; i++) {
      theRecHits[i] = vHits[i];
    }
    for (unsigned int i = theSize; i < SEEDMAXSIZE; i++) {
      theRecHits[i] = nullptr;
    }
  }

  ConstRecHitPointer const *data() const { return theRecHits; }

 // unsigned int size() const { return theSize; }
  //unsigned int size() const { return theRecHits[4] ? 5 :theRecHits[3] ? 4 : (theRecHits[2] ? 3 : (theRecHits[1] ? 2 : 0)); }
  unsigned int size() const { 
  for (unsigned int i = SEEDMAXSIZE-1; i >0; i--) {
    if(theRecHits[i])
      return i+1;
	}
  return 0;
}
  ConstRecHitPointer get(unsigned int i) const { return theRecHits[i]; }
  ConstRecHitPointer operator[](unsigned int i) const { return theRecHits[i]; }

private:
  ConstRecHitPointer theRecHits[SEEDMAXSIZE];
  unsigned int theSize;

};

#endif
