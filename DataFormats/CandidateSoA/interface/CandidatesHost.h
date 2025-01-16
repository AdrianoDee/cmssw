#ifndef DataFormats_CandidateSoA_interface_CandidatesHost_h
#define DataFormats_CandidateSoA_interface_CandidatesHost_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/CandidateSoA/interface/CandidatesSoA.h"

// TODO: The class is created via inheritance of the PortableDeviceCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306
class CandidatesHost : public PortableHostCollection<CandidatesSoA> {
public:
  CandidatesHost() = default;
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
