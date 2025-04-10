#include "Geometry/MTDGeometryBuilder/interface/MTDTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"

#include <utility>

MTDTopology::MTDTopology(const int& topologyMode, const BTLValues& btl, const ETLValues& etl)
    : mtdTopologyMode_(topologyMode), btlVals_(btl), etlVals_(etl) {}

std::pair<uint32_t, uint32_t> MTDTopology::btlIndex(const uint32_t detId) const {
  size_t index(0);
  bool found(false);
  for (const auto& theid : btlVals_.btlDetId_) {
    if (theid == detId) {
      found = true;
      break;
    }
    index++;
  }
  if (found) {
    return std::make_pair(btlVals_.btlPhi_[index], btlVals_.btlEta_[index]);
  } else {
    edm::LogWarning("MTDTopology") << "Searching BTL topology for BTLDetId " << detId
                                   << " not in BTL geometry structure";
    return std::make_pair(std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::max());
  }
}

uint32_t MTDTopology::btlidFromIndex(const uint32_t iphi, const uint32_t ieta) const {
  uint32_t res(0);
  for (uint32_t index = 0; index < btlVals_.nBTLmodules_; index++) {
    if (iphi == btlVals_.btlPhi_[index] && ieta == btlVals_.btlEta_[index]) {
      res = btlVals_.btlDetId_[index];
      break;
    }
  }
  return res;
}

uint32_t MTDTopology::phishiftBTL(const uint32_t detid, const int phiShift) const {
  if (phiShift == 0) {
    edm::LogWarning("MTDTopology") << "asking of a null phiShift in BTL";
    return failIndex_;
  }
  // search for the next detector, check only sign from input
  int sh = phiShift > 0 ? 1 : -1;
  size_t index(0);
  bool found(false);
  for (const auto& theid : btlVals_.btlDetId_) {
    if (theid == detid) {
      found = true;
      break;
    }
    index++;
  }
  if (found) {
    int newIndex = index + sh * btlVals_.nBTLeta_;
    if (newIndex >= static_cast<int>(btlVals_.nBTLmodules_)) {
      newIndex = newIndex - btlVals_.nBTLmodules_;
    } else if (newIndex < 0) {
      newIndex = newIndex + btlVals_.nBTLmodules_;
    }
    return newIndex;
  } else {
    edm::LogWarning("MTDTopology") << "Searching for non existent BTLDetId " << detid;
    return failIndex_;
  }
}

uint32_t MTDTopology::etashiftBTL(const uint32_t detid, const int etaShift) const {
  if (etaShift == 0) {
    edm::LogWarning("MTDTopology") << "asking of a null etaShift in BTL";
    return failIndex_;
  }
  // search for the next detector, check only sign from input
  int sh = etaShift > 0 ? 1 : -1;
  size_t index(0);
  bool found(false);
  for (const auto& theid : btlVals_.btlDetId_) {
    if (theid == detid) {
      found = true;
      break;
    }
    index++;
  }
  if (found) {
    int newIndex = index + sh;
    if (newIndex < 0 || newIndex >= static_cast<int>(btlVals_.nBTLmodules_)) {
      return failIndex_;
    } else if (btlVals_.btlPhi_[newIndex] != btlVals_.btlPhi_[index]) {
      return failIndex_;
    }
    return newIndex;
  } else {
    edm::LogWarning("MTDTopology") << "Searching for non existent BTLDetId " << detid;
    return failIndex_;
  }
}

bool MTDTopology::orderETLSector(const GeomDet*& gd1, const GeomDet*& gd2) {
  ETLDetId det1(gd1->geographicalId().rawId());
  ETLDetId det2(gd2->geographicalId().rawId());

  if (det1.mtdRR() != det2.mtdRR()) {
    return det1.mtdRR() < det2.mtdRR();
  } else if (det1.modType() != det2.modType()) {
    return det1.modType() < det2.modType();
  } else if (det1.module() != det2.module()) {
    return det1.module() < det2.module();
  } else {
    return det1.sensor() < det2.sensor();
  }
}

size_t MTDTopology::hshiftETL(const uint32_t detid, const int horizontalShift) const {
  ETLDetId start_mod(detid);

  if (horizontalShift == 0) {
    edm::LogWarning("MTDTopology") << "asking of a null horizotalShift in ETL";
    return failIndex_;
  }
  int hsh = horizontalShift > 0 ? 1 : -1;

  int sensor = start_mod.sensor();
  int module = start_mod.module();
  uint32_t modtyp = start_mod.modType();
  uint32_t discface = start_mod.discSide() + 2 * (start_mod.nDisc() - 1);
  int geomDetIndex;

  // distinguish numbering in prev8 / v8 geometries
  auto topoMode = getMTDTopologyMode();
  if (static_cast<int>(MTDTopologyMode::etlLayoutFromTopoMode(topoMode)) >=
      static_cast<int>(MTDTopologyMode::EtlLayout::v8)) {
    geomDetIndex = 2 * (module - 1) + sensor;
  } else {
    geomDetIndex = module;
  }

  // ilayout number coincides at present with disc face, use this

  size_t iHome = (modtyp == etlVals_[discface].idDetType1_) ? 0 : 1;
  size_t iLeft = (etlVals_[discface].idDetType1_ == 1) ? 0 : 1;

  // for left type modules the position according to the default order is module - 1, for the rigth type modules the total number of left modules must be added

  size_t nmodOffset = (modtyp == 1) ? 0 : etlVals_[discface].start_copy_[iLeft].back() - 1;

  for (size_t iloop = 0; iloop < etlVals_[discface].start_copy_[iHome].size() - 1; iloop++) {
    if (geomDetIndex >= etlVals_[discface].start_copy_[iHome][iloop] &&
        geomDetIndex < etlVals_[discface].start_copy_[iHome][iloop + 1]) {
      if (geomDetIndex + hsh >= etlVals_[discface].start_copy_[iHome][iloop] &&
          geomDetIndex + hsh < etlVals_[discface].start_copy_[iHome][iloop + 1]) {
        return geomDetIndex + hsh - 1 + nmodOffset;
      }
      break;
    }
  }

  return failIndex_;
}

size_t MTDTopology::vshiftETL(const uint32_t detid, const int verticalShift, size_t& closest) const {
  closest = failIndex_;

  ETLDetId start_mod(detid);

  if (verticalShift == 0) {
    edm::LogWarning("MTDTopology") << "asking of a null verticalShift in ETL";
    return failIndex_;
  }
  int vsh = verticalShift > 0 ? 1 : -1;

  int sensor = start_mod.sensor();
  int module = start_mod.module();
  uint32_t modtyp = start_mod.modType();
  uint32_t discface = start_mod.discSide() + 2 * (start_mod.nDisc() - 1);
  int geomDetIndex;

  // distinguish numbering in prev8 / v8 geometries
  auto topoMode = getMTDTopologyMode();
  if (static_cast<int>(MTDTopologyMode::etlLayoutFromTopoMode(topoMode)) >=
      static_cast<int>(MTDTopologyMode::EtlLayout::v8)) {
    geomDetIndex = 2 * (module - 1) + sensor;
  } else {
    geomDetIndex = module;
  }

  // ilayout number coincides at present with disc face, use this

  size_t iHome = (modtyp == etlVals_[discface].idDetType1_) ? 0 : 1;
  size_t iOther = (iHome == 0) ? 1 : 0;
  size_t iLeft = (etlVals_[discface].idDetType1_ == 1) ? 0 : 1;

  // for right type modules the offset of the total number of left modules needs to be added,
  // what matters here is the other type, i.e. if the starting module is left the vertical shift moves towards a right type, and viceversa

  size_t nmodOffset = (modtyp == 1) ? etlVals_[discface].start_copy_[iLeft].back() - 1 : 0;

  size_t iBin(etlVals_[discface].start_copy_[iHome].size());  // never allowed
  for (size_t iloop = 0; iloop < etlVals_[discface].start_copy_[iHome].size() - 1; iloop++) {
    if (geomDetIndex >= etlVals_[discface].start_copy_[iHome][iloop] &&
        geomDetIndex < etlVals_[discface].start_copy_[iHome][iloop + 1]) {
      iBin = iloop;
      break;
    }
  }

  if (iBin == etlVals_[discface].start_copy_[iHome].size()) {
    edm::LogWarning("MTDTopology") << "Module number not compatible with layout, abort";
    return failIndex_;
  }

  // define the interval of interest for the other type according to the vertical shift sign

  int iBinOther(iBin);
  if (iHome == 0 && vsh < 0) {
    iBinOther = iBin - 1;
  }
  if (iHome == 1 && vsh > 0) {
    iBinOther = iBin + 1;
  }
  if (iBinOther < 0 || iBinOther >= static_cast<int>(etlVals_[discface].start_copy_[iOther].size()) - 1) {
    return failIndex_;
  }

  // determine the position of the other type corresponding to the same column of the home type

  int vpos = etlVals_[discface].offset_[iHome][iBin] + geomDetIndex - etlVals_[discface].start_copy_[iHome][iBin] + 1;
  if (vpos <= etlVals_[discface].offset_[iOther][iBinOther]) {
    closest = etlVals_[discface].start_copy_[iOther][iBinOther];
  } else if (vpos > etlVals_[discface].offset_[iOther][iBinOther] +
                        etlVals_[discface].start_copy_[iOther][iBinOther + 1] -
                        etlVals_[discface].start_copy_[iOther][iBinOther] ||
             (vpos == etlVals_[discface].offset_[iOther][iBinOther] +
                          etlVals_[discface].start_copy_[iOther][iBinOther + 1] -
                          etlVals_[discface].start_copy_[iOther][iBinOther] &&
              iBinOther + 1 == static_cast<int>(etlVals_[discface].start_copy_[iOther].size()))) {
    closest = etlVals_[discface].start_copy_[iOther][iBinOther + 1] - 1;
  }
  if (closest < failIndex_) {
    closest = closest + nmodOffset - 1;
    return failIndex_;
  } else {
    // number of module shifted by 1 wrt the position in the array (i.e. module 1 has index 0)
    return etlVals_[discface].start_copy_[iOther][iBinOther] + vpos - 1 -
           etlVals_[discface].offset_[iOther][iBinOther] + nmodOffset - 1;
  }
}
