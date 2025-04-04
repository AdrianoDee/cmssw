#include "DQM/EcalMonitorTasks/interface/ClusterTask.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DQM/EcalCommon/interface/MESetMulti.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtPsbWord.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace ecaldqm {
  ClusterTask::ClusterTask()
      : DQWorkerTask(),
        ebHits_(nullptr),
        eeHits_(nullptr),
        //    ievt_(0),
        //    massCalcPrescale_(_workerParams.getUntrackedParameter<int>("massCalcPrescale")),
        doExtra_(true),
        energyThreshold_(0.),
        swissCrossMaxThreshold_(3.),
        egTriggerAlgos_(),
        trigTypeToME_{0, 1, 2, 3, 4},
        L1GlobalTriggerReadoutRecordTag_(),
        L1MuGMTReadoutCollectionTag_(),
        L1GlobalTriggerReadoutRecordToken_(),
        L1MuGMTReadoutCollectionToken_() {}

  void ClusterTask::setParams(edm::ParameterSet const& _params) {
    doExtra_ = _params.getUntrackedParameter<bool>("doExtra");

    if (!doExtra_) {
      MEs_.erase(std::string("SCSizeVsEnergy"));
      MEs_.erase(std::string("SCSeedOccupancyHighE"));
      MEs_.erase(std::string("SCSeedOccupancyTrig"));
      MEs_.erase(std::string("SCSeedTimeTrigEx"));
      MEs_.erase(std::string("SCSeedTimeMapTrigEx"));
      MEs_.erase(std::string("SCOccupancyProjEta"));
      MEs_.erase(std::string("SCOccupancyProjPhi"));
      MEs_.erase(std::string("SCSwissCross"));
      MEs_.erase(std::string("Triggers"));
      MEs_.erase(std::string("ExclusiveTriggers"));

      return;
    }

    energyThreshold_ = _params.getUntrackedParameter<double>("energyThreshold");
    swissCrossMaxThreshold_ = _params.getUntrackedParameter<double>("swissCrossMaxThreshold");
    egTriggerAlgos_ = _params.getUntrackedParameter<std::vector<std::string> >("egTriggerAlgos");
    L1GlobalTriggerReadoutRecordTag_ = _params.getUntrackedParameter<edm::InputTag>("L1GlobalTriggerReadoutRecordTag");
    L1MuGMTReadoutCollectionTag_ = _params.getUntrackedParameter<edm::InputTag>("L1MuGMTReadoutCollectionTag");

    MESet::PathReplacements repl;

    std::string triggerTypeNames[nTriggerTypes] = {"ECAL", "HCAL", "CSC", "DT", "RPC"};

    MESetMulti& occupancy(static_cast<MESetMulti&>(MEs_.at("SCSeedOccupancyTrig")));
    for (unsigned iT(0); iT != nTriggerTypes; ++iT) {
      repl["trig"] = triggerTypeNames[iT];
      trigTypeToME_[iT] = occupancy.getIndex(repl);
    }
  }

  void ClusterTask::addDependencies(DependencySet& _dependencies) {
    _dependencies.push_back(Dependency(kEBSuperCluster, kEBRecHit));
    _dependencies.push_back(Dependency(kEESuperCluster, kEERecHit));
  }

  void ClusterTask::beginEvent(edm::Event const& _evt, edm::EventSetup const& _es, bool const&, bool&) {
    if (!doExtra_)
      return;

    triggered_.reset();

    // TODO IS THIS THE MOST UP-TO-DATE ACCESSOR TO L1 INFO?

    edm::Handle<L1GlobalTriggerReadoutRecord> l1GTHndl;
    _evt.getByToken(L1GlobalTriggerReadoutRecordToken_, l1GTHndl);

    if (!l1GTHndl.isValid()) {
      edm::LogError("L1GlobalTriggerReadoutRecord")
          << "Failed to retrieve L1GlobalTriggerReadoutRecord from the Event!";
      return;  // Exit the function early if the handle is invalid
    }

    DecisionWord const& dWord(l1GTHndl->decisionWord());

    //Ecal

    L1GtTriggerMenu const* menu(&_es.getData(menuRcd));

    if (!dWord.empty()) {  //protect against no L1GT in run
      for (unsigned iT(0); iT != egTriggerAlgos_.size(); ++iT) {
        if (menu->gtAlgorithmResult(egTriggerAlgos_[iT], dWord)) {
          triggered_.set(kEcalTrigger);
          break;
        }
      }
    }

    //Hcal
    bool hcal_top = false;
    bool hcal_bot = false;
    const L1GtPsbWord psb = l1GTHndl->gtPsbWord(0xbb0d, 0);
    std::vector<int> valid_phi;
    if ((psb.aData(4) & 0x3f) >= 1) {
      valid_phi.push_back((psb.aData(4) >> 10) & 0x1f);
    }
    if ((psb.bData(4) & 0x3f) >= 1) {
      valid_phi.push_back((psb.bData(4) >> 10) & 0x1f);
    }
    if ((psb.aData(5) & 0x3f) >= 1) {
      valid_phi.push_back((psb.aData(5) >> 10) & 0x1f);
    }
    if ((psb.bData(5) & 0x3f) >= 1) {
      valid_phi.push_back((psb.bData(5) >> 10) & 0x1f);
    }
    std::vector<int>::const_iterator iphi;
    for (iphi = valid_phi.begin(); iphi != valid_phi.end(); iphi++) {
      if (*iphi < 9)
        hcal_top = true;
      if (*iphi > 8)
        hcal_bot = true;
    }
    if (hcal_top && hcal_bot)
      triggered_.set(kHcalTrigger);

    //Muons
    edm::Handle<L1MuGMTReadoutCollection> l1MuHndl;
    if (!_evt.getByToken(L1MuGMTReadoutCollectionToken_, l1MuHndl))
      return;
    std::vector<L1MuGMTReadoutRecord> const& records(l1MuHndl->getRecords());

    for (unsigned iR(0); iR != records.size(); ++iR) {
      if (records[iR].getBxInEvent() != 0)
        continue;

      unsigned iC(0);

      //DT triggers
      std::vector<L1MuRegionalCand> dtBXCands(records[iR].getDTBXCands());
      for (iC = 0; iC != dtBXCands.size(); ++iC)
        if (!dtBXCands[iC].empty())
          break;
      if (iC != dtBXCands.size())
        triggered_.set(kDTTrigger);

      //RPC triggers
      std::vector<L1MuRegionalCand> brlRPCCands(records[iR].getBrlRPCCands());
      for (iC = 0; iC != brlRPCCands.size(); ++iC)
        if (!brlRPCCands[iC].empty())
          break;
      if (iC != brlRPCCands.size())
        triggered_.set(kRPCTrigger);

      //CSC Triggers
      std::vector<L1MuRegionalCand> cscCands(records[iR].getCSCCands());
      for (iC = 0; iC != cscCands.size(); ++iC)
        if (!cscCands[iC].empty())
          break;
      if (iC != cscCands.size())
        triggered_.set(kCSCTrigger);
    }

    if (triggered_.none())
      return;

    MESet& meTriggers(MEs_.at("Triggers"));
    MESet& meExclusiveTriggers(MEs_.at("ExclusiveTriggers"));

    for (unsigned iT(0); iT != nTriggerTypes; ++iT) {
      if (!triggered_[iT])
        continue;
      meTriggers.fill(getEcalDQMSetupObjects(), iT + 0.5);
      if (triggered_.count() == 1)
        meExclusiveTriggers.fill(getEcalDQMSetupObjects(), iT + 0.5);
    }
  }

  void ClusterTask::endEvent(edm::Event const&, edm::EventSetup const&) {
    //    ++ievt_;

    ebHits_ = nullptr;
    eeHits_ = nullptr;
  }

  bool ClusterTask::filterRunType(short const* _runType) {
    for (unsigned iFED(0); iFED != nDCC; iFED++) {
      if (_runType[iFED] == EcalDCCHeaderBlock::COSMIC || _runType[iFED] == EcalDCCHeaderBlock::MTCC ||
          _runType[iFED] == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
          _runType[iFED] == EcalDCCHeaderBlock::PHYSICS_GLOBAL || _runType[iFED] == EcalDCCHeaderBlock::COSMICS_LOCAL ||
          _runType[iFED] == EcalDCCHeaderBlock::PHYSICS_LOCAL)
        return true;
    }

    return false;
  }

  void ClusterTask::runOnRecHits(EcalRecHitCollection const& _hits, Collections _collection) {
    switch (_collection) {
      case kEBRecHit:
        ebHits_ = &_hits;
        break;
      case kEERecHit:
        eeHits_ = &_hits;
        break;
      default:
        break;
    }
  }

  void ClusterTask::runOnBasicClusters(edm::View<reco::CaloCluster> const& _bcs, Collections _collection) {
    MESet& meBCE(MEs_.at("BCE"));
    MESet& meBCEt(MEs_.at("BCEt"));
    MESet& meBCEMap(MEs_.at("BCEMap"));
    MESet& meBCEtMap(MEs_.at("BCEtMap"));
    MESet& meBCEMapProjEta(MEs_.at("BCEMapProjEta"));
    MESet& meBCEMapProjPhi(MEs_.at("BCEMapProjPhi"));
    MESet& meBCEtMapProjEta(MEs_.at("BCEtMapProjEta"));
    MESet& meBCEtMapProjPhi(MEs_.at("BCEtMapProjPhi"));
    MESet& meBCOccupancy(MEs_.at("BCOccupancy"));
    MESet& meBCOccupancyProjEta(MEs_.at("BCOccupancyProjEta"));
    MESet& meBCOccupancyProjPhi(MEs_.at("BCOccupancyProjPhi"));
    MESet& meBCSize(MEs_.at("BCSize"));
    MESet& meTrendBCSize(MEs_.at("TrendBCSize"));
    MESet& meBCSizeMap(MEs_.at("BCSizeMap"));
    MESet& meBCSizeMapProjEta(MEs_.at("BCSizeMapProjEta"));
    MESet& meBCSizeMapProjPhi(MEs_.at("BCSizeMapProjPhi"));
    MESet& meBCNum(MEs_.at("BCNum"));
    MESet& meTrendNBC(MEs_.at("TrendNBC"));

    int nBC[] = {0, 0};
    bool isBarrel(_collection == kEBBasicCluster);

    //    vector<reco::BasicCluster const*> lowMassCands;

    for (edm::View<reco::CaloCluster>::const_iterator bcItr(_bcs.begin()); bcItr != _bcs.end(); ++bcItr) {
      if (bcItr->caloID().detectors() != 0) {
        if (isBarrel && !bcItr->caloID().detector(reco::CaloID::DET_ECAL_BARREL))
          continue;
        if (!isBarrel && !bcItr->caloID().detector(reco::CaloID::DET_ECAL_ENDCAP))
          continue;
      }

      math::XYZPoint const& position(bcItr->position());

      DetId id(bcItr->seed());
      if (id.null()) {
        GlobalPoint gp(position.x(), position.y(), position.z());
        CaloSubdetectorGeometry const* subgeom(
            GetGeometry()->getSubdetectorGeometry(DetId::Ecal, isBarrel ? EcalBarrel : EcalEndcap));

        id = subgeom->getClosestCell(gp);
      }

      if (id.null() || (id.subdetId() == EcalBarrel && !isBarrel) || (id.subdetId() == EcalEndcap && isBarrel))
        continue;

      float energy(bcItr->energy());
      float posEta(position.eta());
      float posPhi(phi(position.phi()));
      float et(energy / std::cosh(posEta));
      int subdet(isBarrel ? EcalBarrel : EcalEndcap);
      if (subdet == EcalEndcap && position.z() < 0.)
        subdet = -EcalEndcap;

      meBCE.fill(getEcalDQMSetupObjects(), id, energy);
      meBCEt.fill(getEcalDQMSetupObjects(), id, et);

      meBCEMap.fill(getEcalDQMSetupObjects(), id, energy);
      meBCEtMap.fill(getEcalDQMSetupObjects(), id, et);
      meBCEMapProjEta.fill(getEcalDQMSetupObjects(), posEta, energy);
      meBCEMapProjPhi.fill(getEcalDQMSetupObjects(), subdet, posPhi, energy);
      meBCEtMapProjEta.fill(getEcalDQMSetupObjects(), posEta, et);
      meBCEtMapProjPhi.fill(getEcalDQMSetupObjects(), subdet, posPhi, et);

      meBCOccupancy.fill(getEcalDQMSetupObjects(), id);
      meBCOccupancyProjEta.fill(getEcalDQMSetupObjects(), posEta);
      meBCOccupancyProjPhi.fill(getEcalDQMSetupObjects(), subdet, posPhi);

      float size(bcItr->size());

      meBCSize.fill(getEcalDQMSetupObjects(), id, size);
      meTrendBCSize.fill(getEcalDQMSetupObjects(), id, double(timestamp_.iLumi), size);

      meBCSizeMap.fill(getEcalDQMSetupObjects(), id, size);
      meBCSizeMapProjEta.fill(getEcalDQMSetupObjects(), posEta, size);
      meBCSizeMapProjPhi.fill(getEcalDQMSetupObjects(), subdet, posPhi, size);

      int zside(position.z() > 0 ? 1 : 0);
      nBC[zside]++;

      //       if(ievt_ % massCalcPrescale_ != 0) continue;

      //       if(energy > 10.) continue;

      //       EcalRecHitCollection::const_iterator hitItr(isBarrel ? ebHits_->find(id) : eeHits_->find(id));
      //       if(hitItr == (isBarrel ? ebHits_->end() : eeHits_->end())) continue;

      //       // cuts here must be parametrized
      //       if(hitItr->energy() < 0.5) continue;

      //       if(hitItr->energy() / energy > 0.95) continue;

      //       lowMassCands.push_back(&(*bcItr));
    }

    if (isBarrel) {
      meBCNum.fill(getEcalDQMSetupObjects(), EcalBarrel, nBC[0] + nBC[1]);
      meTrendNBC.fill(getEcalDQMSetupObjects(), EcalBarrel, double(timestamp_.iLumi), nBC[0] + nBC[1]);
    } else {
      meBCNum.fill(getEcalDQMSetupObjects(), -EcalEndcap, nBC[0]);
      meBCNum.fill(getEcalDQMSetupObjects(), EcalEndcap, nBC[1]);
      meTrendNBC.fill(getEcalDQMSetupObjects(), EcalEndcap, double(timestamp_.iLumi), nBC[0] + nBC[1]);
    }

    //     if(ievt_ % massCalcPrescale_ != 0) return;

    //     double const pi(3.14159265);

    //     for(vector<reco::BasicCluster const*>::iterator bcItr1(lowMassCands.begin()); bcItr1 != lowMassCands.end(); ++bcItr1){
    //       reco::BasicCluster const& bc1(**bcItr1);
    //       float energy1(bc1.energy());
    //       float px1(energy1 * sin(bc1.position().theta()) * cos(bc1.phi()));
    //       float py1(energy1 * sin(bc1.position().theta()) * sin(bc1.phi()));
    //       float pz1(energy1 * cos(bc1.position().theta()));

    //       for(vector<reco::BasicCluster const*>::iterator bcItr2(lowMassCands.begin()); bcItr2 != lowMassCands.end(); ++bcItr2){
    //  if(*bcItr1 == *bcItr2) continue;
    //  reco::BasicCluster const& bc2(**bcItr2);
    //  float energy2(bc2.energy());
    //  float px2(energy2 * sin(bc2.position().theta()) * cos(bc2.phi()));
    //  float py2(energy2 * sin(bc2.position().theta()) * sin(bc2.phi()));
    //  float pz2(energy2 * cos(bc2.position().theta()));

    //  float ptpair(sqrt((px1 + px2) * (px1 + px2) + (py1 + py2) * (py1 + py2)));
    //  if(ptpair < 2.5) continue;

    //  float epair(energy1 + energy2);
    //  float pzpair(abs(pz1 + pz2));

    //         float m2(epair * epair - pzpair * pzpair - ptpair * ptpair);
    //         if(m2 < 0.) continue;

    //  float eta(0.5 * log((epair + pzpair)/(epair - pzpair)));
    //  float phi(atan2(px1 + px2, py1 + py2));

    //  float iso(0.);
    //  for(reco::BasicClusterCollection::const_iterator bcItr(_bcs.begin()); bcItr != _bcs.end(); ++bcItr){
    //    float dEta(bcItr->eta() - eta);
    //    float dPhi(bcItr->phi() - phi);
    //           if(dPhi > 2. * pi) dPhi -= 2. * pi;
    //           else if(dPhi < -2. * pi) dPhi += 2. * pi;
    //    if(sqrt(dEta * dEta + dPhi * dPhi) < 0.2) iso += bcItr->energy() * sin(bcItr->position().theta());
    //  }
    //  if(iso > 0.5) continue;

    //  float mass(sqrt(m2));
    //  MEs_[kPi0]->fill(mass);
    //  MEs_[kJPsi]->fill(mass);
    //       }
    //     }
  }

  void ClusterTask::runOnSuperClusters(reco::SuperClusterCollection const& _scs, Collections _collection) {
    bool isBarrel(_collection == kEBSuperCluster);
    EcalSubdetector subdet(isBarrel ? EcalBarrel : EcalEndcap);

    MESet& meSCE(MEs_.at("SCE"));
    MESet& meSCEt(MEs_.at("SCEt"));
    MESet& meSCELow(MEs_.at("SCELow"));
    MESet& meSCEtLow(MEs_.at("SCEtLow"));
    MESet& meSCRawE(MEs_.at("SCRawE"));
    MESet& meSCRawEt(MEs_.at("SCRawEt"));
    MESet& meSCRawELow(MEs_.at("SCRawELow"));
    MESet& meSCRawEtLow(MEs_.at("SCRawEtLow"));
    MESet& meSCRawEHigh(MEs_.at("SCRawEHigh"));
    MESet& meSCNBCs(MEs_.at("SCNBCs"));
    MESet& meSCNcrystals(MEs_.at("SCNcrystals"));
    MESet& meTrendSCSize(MEs_.at("TrendSCSize"));
    MESet& meSCSeedEnergy(MEs_.at("SCSeedEnergy"));
    MESet& meSCClusterVsSeed(MEs_.at("SCClusterVsSeed"));
    MESet& meSCSeedOccupancy(MEs_.at("SCSeedOccupancy"));
    MESet& meSingleCrystalCluster(MEs_.at("SingleCrystalCluster"));
    MESet& meSCR9(MEs_.at("SCR9"));
    MESet& meSCR9Raw(MEs_.at("SCR9Raw"));
    MESet& meSCR9Full(MEs_.at("SCR9Full"));
    MESet& meSCR9FullRaw(MEs_.at("SCR9FullRaw"));

    MESet* meSCSizeVsEnergy(doExtra_ ? &MEs_.at("SCSizeVsEnergy") : nullptr);
    MESet* meSCSeedOccupancyHighE(doExtra_ ? &MEs_.at("SCSeedOccupancyHighE") : nullptr);
    MESet* meSCSeedOccupancyTrig(doExtra_ ? &MEs_.at("SCSeedOccupancyTrig") : nullptr);
    MESet* meSCSeedTimeTrigEx(doExtra_ ? &MEs_.at("SCSeedTimeTrigEx") : nullptr);
    MESet* meSCSeedTimeMapTrigEx(doExtra_ ? &MEs_.at("SCSeedTimeMapTrigEx") : nullptr);
    MESet* meSCOccupancyProjEta(doExtra_ ? &MEs_.at("SCOccupancyProjEta") : nullptr);
    MESet* meSCOccupancyProjPhi(doExtra_ ? &MEs_.at("SCOccupancyProjPhi") : nullptr);
    MESet* meSCSwissCross(doExtra_ ? &MEs_.at("SCSwissCross") : nullptr);

    EcalRecHitCollection const* hits(isBarrel ? ebHits_ : eeHits_);

    //     reco::SuperCluster const* leading(0);
    //     reco::SuperCluster const* subLeading(0);

    int nSC(0);

    for (reco::SuperClusterCollection::const_iterator scItr(_scs.begin()); scItr != _scs.end(); ++scItr) {
      DetId seedId(scItr->seed()->seed());
      math::XYZPoint const& position(scItr->position());
      if (seedId.null()) {
        GlobalPoint gp(position.x(), position.y(), position.z());

        CaloSubdetectorGeometry const* subgeom(
            GetGeometry()->getSubdetectorGeometry(DetId::Ecal, isBarrel ? EcalBarrel : EcalEndcap));

        seedId = subgeom->getClosestCell(gp);
      }

      if (seedId.null() || (seedId.subdetId() != subdet))
        continue;

      EcalRecHitCollection::const_iterator seedItr(hits->find(seedId));
      if (seedItr == hits->end())
        continue;

      ++nSC;

      float energy(scItr->energy());
      float rawEnergy(scItr->rawEnergy());
      float posEta(position.eta());
      float et(energy / std::cosh(posEta));
      float rawEt(rawEnergy / std::cosh(posEta));
      float size(scItr->size());

      meSCE.fill(getEcalDQMSetupObjects(), seedId, energy);
      meSCEt.fill(getEcalDQMSetupObjects(), seedId, et);
      meSCELow.fill(getEcalDQMSetupObjects(), seedId, energy);
      meSCEtLow.fill(getEcalDQMSetupObjects(), seedId, et);

      meSCRawE.fill(getEcalDQMSetupObjects(), seedId, rawEnergy);
      meSCRawEt.fill(getEcalDQMSetupObjects(), seedId, rawEt);
      meSCRawELow.fill(getEcalDQMSetupObjects(), seedId, rawEnergy);
      meSCRawEtLow.fill(getEcalDQMSetupObjects(), seedId, rawEt);
      meSCRawEHigh.fill(getEcalDQMSetupObjects(), seedId, rawEnergy);

      meSCNBCs.fill(getEcalDQMSetupObjects(), seedId, scItr->clustersSize());
      meSCNcrystals.fill(getEcalDQMSetupObjects(), seedId, size);

      if (doExtra_) [[clang::suppress]]
        meSCSizeVsEnergy->fill(getEcalDQMSetupObjects(), subdet, energy, size);

      meTrendSCSize.fill(getEcalDQMSetupObjects(), seedId, double(timestamp_.iLumi), size);

      meSCSeedEnergy.fill(getEcalDQMSetupObjects(), seedId, seedItr->energy());
      meSCClusterVsSeed.fill(getEcalDQMSetupObjects(), seedId, seedItr->energy(), energy);

      meSCSeedOccupancy.fill(getEcalDQMSetupObjects(), seedId);
      if (doExtra_ && energy > energyThreshold_) [[clang::suppress]]
        meSCSeedOccupancyHighE->fill(getEcalDQMSetupObjects(), seedId);

      if (scItr->size() == 1)
        meSingleCrystalCluster.fill(getEcalDQMSetupObjects(), seedId);

      float e3x3(EcalClusterTools::e3x3(*scItr->seed(), hits, GetTopology()));
      float e3x3Full(noZS::EcalClusterTools::e3x3(*scItr->seed(), hits, GetTopology()));

      meSCR9.fill(getEcalDQMSetupObjects(), seedId, e3x3 / energy);
      meSCR9Raw.fill(getEcalDQMSetupObjects(), seedId, e3x3 / rawEnergy);
      meSCR9Full.fill(getEcalDQMSetupObjects(), seedId, e3x3Full / energy);
      meSCR9FullRaw.fill(getEcalDQMSetupObjects(), seedId, e3x3Full / rawEnergy);

      if (doExtra_) {
        for (unsigned iT(0); iT != nTriggerTypes; ++iT) {
          if (!triggered_[iT])
            continue;

          [[clang::suppress]] static_cast<MESetMulti*>(meSCSeedOccupancyTrig)->use(trigTypeToME_[iT]);
          meSCSeedOccupancyTrig->fill(getEcalDQMSetupObjects(), seedId);

          // exclusive
          if (triggered_.count() == 1) {
            static_cast<MESetMulti*>(meSCSeedTimeTrigEx)->use(trigTypeToME_[iT]);
            static_cast<MESetMulti*>(meSCSeedTimeMapTrigEx)->use(trigTypeToME_[iT]);
            meSCSeedTimeTrigEx->fill(getEcalDQMSetupObjects(), subdet, seedItr->time());
            meSCSeedTimeMapTrigEx->fill(getEcalDQMSetupObjects(), seedId, seedItr->time());
          }
        }

        meSCOccupancyProjEta->fill(getEcalDQMSetupObjects(), subdet, scItr->eta());
        meSCOccupancyProjPhi->fill(getEcalDQMSetupObjects(), subdet, phi(scItr->phi()));

        if (isBarrel) {
          float e1(EcalClusterTools::eMax(*scItr, ebHits_));
          if (e1 > swissCrossMaxThreshold_) {
            float e4(EcalClusterTools::eTop(*scItr, ebHits_, GetTopology()) +
                     EcalClusterTools::eRight(*scItr, ebHits_, GetTopology()) +
                     EcalClusterTools::eBottom(*scItr, ebHits_, GetTopology()) +
                     EcalClusterTools::eLeft(*scItr, ebHits_, GetTopology()));

            meSCSwissCross->fill(getEcalDQMSetupObjects(), 1. - e4 / e1);
          }
        }
      }

      //       if(ievt_ % massCalcPrescale_ != 0) continue;

      //       float et(energy * sin(scItr->position().theta()));
      //       if(!leading || et > leading->energy() * sin(leading->position().theta())){
      //  subLeading = leading;
      //  leading = &(*scItr);
      //       }
      //       else if(!subLeading || et > subLeading->energy() * sin(subLeading->position().theta())){
      //  subLeading = &(*scItr);
      //       }
    }

    MEs_.at("SCNum").fill(getEcalDQMSetupObjects(), subdet, nSC);
    MEs_.at("TrendNSC").fill(getEcalDQMSetupObjects(), subdet, double(timestamp_.iLumi), nSC);

    //     if(ievt_ % massCalcPrescale_ != 0) return;

    //     // implement isolation & cuts
    //     if(!leading || !subLeading) return;
    //     float e(leading->energy() + subLeading->energy());
    //     float px(leading->energy() * sin(leading->position().theta()) * cos(leading->phi()) + subLeading->energy() * sin(subLeading->position().theta()) * cos(subLeading->phi()));
    //     float py(leading->energy() * sin(leading->position().theta()) * sin(leading->phi()) + subLeading->energy() * sin(subLeading->position().theta()) * sin(subLeading->phi()));
    //     float pz(leading->energy() * cos(leading->position().theta()) + subLeading->energy() * cos(subLeading->position().theta()));
    //     float m2(e * e - px * px - py * py - pz * pz);
    //     if(m2 < 0.) return;
    //     float mass(sqrt(m2));
    //     MEs_[kZ]->fill(mass);
    //     MEs_[kHighMass]->fill(mass);
  }

  void ClusterTask::setTokens(edm::ConsumesCollector& _collector) {
    L1GlobalTriggerReadoutRecordToken_ =
        _collector.consumes<L1GlobalTriggerReadoutRecord>(L1GlobalTriggerReadoutRecordTag_);
    L1MuGMTReadoutCollectionToken_ = _collector.consumes<L1MuGMTReadoutCollection>(L1MuGMTReadoutCollectionTag_);
    menuRcd = _collector.esConsumes();
  }

  DEFINE_ECALDQM_WORKER(ClusterTask);
}  // namespace ecaldqm
