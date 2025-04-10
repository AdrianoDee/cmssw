#ifndef PhysicsTools_NanoAOD_TriggerOutputBranches_h
#define PhysicsTools_NanoAOD_TriggerOutputBranches_h

#include <string>
#include <vector>
#include <TTree.h>
#include "FWCore/Framework/interface/EventForOutput.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

class TriggerOutputBranches {
public:
  TriggerOutputBranches(const edm::ProductDescription *desc, const edm::EDGetToken &token)
      : m_token(token), m_lastRun(-1), m_fills(0), m_processName(desc->processName()) {
    if (desc->className() != "edm::TriggerResults")
      throw cms::Exception("Configuration",
                           "NanoAODOutputModule/TriggerOutputBranches can only write out edm::TriggerResults objects");
  }

  void updateTriggerNames(TTree &tree, const edm::TriggerNames &names, const edm::TriggerResults &ta);
  void fill(const edm::EventForOutput &iEvent, TTree &tree);
  std::string processName() { return m_processName; }

private:
  edm::TriggerNames triggerNames(
      const edm::TriggerResults
          triggerResults);  //FIXME: if we have to keep it local we may use PsetID check per event instead of run boundary

  edm::EDGetToken m_token;
  std::string m_baseName;
  UInt_t m_counter;
  struct NamedBranchPtr {
    std::string name, title;
    int idx;
    TBranch *branch;
    uint8_t buffer;
    NamedBranchPtr(const std::string &aname, const std::string &atitle, TBranch *branchptr = nullptr)
        : name(aname), title(atitle), branch(branchptr), buffer(0) {}
  };
  std::vector<NamedBranchPtr> m_triggerBranches;
  long m_lastRun;
  unsigned long m_fills;
  std::string m_processName;
  bool verifyBranchUniqueName(TTree &, std::string) const;

  template <typename T>
  void fillColumn(NamedBranchPtr &nb, const edm::TriggerResults &triggers) {
    nb.buffer = (nb.idx >= 0) ? triggers.accept(nb.idx) : 0;
    nb.branch->SetAddress(&(nb.buffer));  // Can be improved: this is not reallt needed at each event
        //but we should be sure that resize of vectors of TriggerOutputBranches do not mess up things
  }
};

#endif
