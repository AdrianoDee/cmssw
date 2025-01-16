#ifndef DataFormats_ClusterGeometrySoA_interface_ClusterGeometryHost_h
#define DataFormats_ClusterGeometrySoA_interface_ClusterGeometryHost_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/ClusterGeometrySoA/interface/ClusterGeometryLayout.h"

class ClusterGeometryHost : public PortableHostCollection<ClusterGeometrySoA> {
public:
  // Constructor that initializes the collection with the size and queue (for host memory)
  template <typename TQueue>
  explicit ClusterGeometryHost(size_t nClusters, TQueue queue)
      : PortableHostCollection<ClusterGeometrySoA>(nClusters, queue) {}

  // Set number of modules (for example, in case of geometry data like detector modules)
  void setNModules(uint32_t nModules) { nModules_h = nModules; }

  uint32_t nModules() const { return nModules_h; }
  
  // Returns the number of clusters (metadata size)
  uint32_t nClusters() const { return view().metadata().size(); }

private:
  uint32_t nModules_h = 0;  // Number of modules (e.g., detector modules)
};

#endif  // DataFormats_ClusterGeometrySoA_interface_ClusterGeometryHost_h

