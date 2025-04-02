#ifndef DataFormats_CandidateSoA_interface_CandidatesHost_h
#define DataFormats_CandidateSoA_interface_CandidatesHost_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/CandidateSoA/interface/CandidatesSoA.h"

class CandidatesHost : public PortableHostCollection<CandidatesSoA> {
public:
  CandidatesHost(edm::Uninitialized) : PortableHostCollection<CandidatesSoA>{edm::kUninitialized} {}

  template <typename TQueue>
  explicit CandidatesHost(size_t maxFedWords, TQueue queue)
      : PortableHostCollection<CandidatesSoA>(maxFedWords + 1, queue) {}

  void setNModules(uint32_t nModules) { nModules_h = nModules; }

  uint32_t nModules() const { return nModules_h; }
  uint32_t nDigis() const { return view().metadata().size() - 1; }

private:
  uint32_t nModules_h = 0;
};

#endif  // DataFormats_CandidateSoA_interface_CandidatesHost_h