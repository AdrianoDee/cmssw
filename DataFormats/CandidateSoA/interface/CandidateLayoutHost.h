#ifndef DataFormats_CandidateSoA_interface_CandidateHost_h
#define DataFormats_CandidateSoA_interface_CandidateHost_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/CandidateSoA/interface/CandidateLayout.h"

class CandidateHost : public PortableHostCollection<CandidateSoA> {
public:
  // Constructor that initializes the collection with the size and queue (for host memory)
  template <typename TQueue>
  explicit CandidateHost(size_t nClusters, TQueue queue)
      : PortableHostCollection<CandidateSoA>(nClusters, queue) {}

  // Set number of modules (for example, in case of geometry data like detector modules)
  void setNModules(uint32_t nModules) { nModules_h = nModules; }

  uint32_t nModules() const { return nModules_h; }
  
  // Returns the number of clusters (metadata size)
  uint32_t nClusters() const { return view().metadata().size(); }

private:
  uint32_t nModules_h = 0;  // Number of modules (e.g., detector modules)
};

#endif  // DataFormats_CandidateSoA_interface_CandidateHost_h

