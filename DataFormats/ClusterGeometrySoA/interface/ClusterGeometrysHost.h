#ifndef DataFormats_ClusterGeometrySoA_interface_ClusterGeometrysHost_h
#define DataFormats_ClusterGeometrySoA_interface_ClusterGeometrysHost_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/ClusterGeometrySoA/interface/ClusterGeometrysSoA.h"

// TODO: The class is created via inheritance of the PortableDeviceCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306
class ClusterGeometrysHost : public PortableHostCollection<ClusterGeometrysSoA> {
public:
  ClusterGeometrysHost(edm::Uninitialized) : PortableHostCollection<ClusterGeometrysSoA>{edm::kUninitialized} {}

  template <typename TQueue>
  explicit ClusterGeometrysHost(size_t maxFedWords, TQueue queue)
      : PortableHostCollection<ClusterGeometrysSoA>(maxFedWords + 1, queue) {}

  void setNModules(uint32_t nModules) { nModules_h = nModules; }

  uint32_t nModules() const { return nModules_h; }
  uint32_t nDigis() const { return view().metadata().size() - 1; }

private:
  uint32_t nModules_h = 0;
};

#endif  // DataFormats_ClusterGeometrySoA_interface_ClusterGeometrysHost_h
