import FWCore.ParameterSet.Config as cms
from HLTrigger.Configuration.common import producers_by_type

def customizeHLTForMixedPF(process):

    # Not affecting the other seed producers
    for producer in producers_by_type(process, "SeedGeneratorFromProtoTracksEDProducer"):
        producer.produceComplement = cms.bool(False)

    # Customization for the iter0 seeds
    process.hltIter0PFLowPixelSeedsFromPixelTracks.produceComplement = cms.bool(True)
    # allow maximum number of tracks to be sampled
    #process.hltIter0PFLowPixelSeedsFromPixelTracks.maxComplementTracks = cms.int32(500)
    # get tracks only close to first N vertices, by dz<0.3
   #process.hltIter0PFLowPixelSeedsFromPixelTracks.InputVertexCollectionForComplement = cms.InputTag("hltPixelVertices")
    #process.hltIter0PFLowPixelSeedsFromPixelTracks.maxComplementTrackVertex = cms.int32(10)
    #process.hltIter0PFLowPixelSeedsFromPixelTracks.dzMaxComplementTrackVertex = cms.double(0.3)
    
    # Apply cuts for pixel tracks complement
 
    ## |eta|<2.0 requirements --> quadraplets only to reduce timing (not very big impact in performance)
    process.hltPixelTracksLowPT = cms.EDProducer( "TrackWithVertexSelector",
        normalizedChi2 = cms.double( 999999.0 ),
        numberOfValidHits = cms.uint32( 0 ),
        zetaVtx = cms.double( 0.3 ),
        rhoVtx = cms.double( 0.2 ),
        ptErrorCut = cms.double( 5.0 ),
        dzMax = cms.double( 999.0 ),
        etaMin = cms.double( 0.0 ),
        etaMax = cms.double( 5.0 ),
        quality = cms.string( "highPurity" ), ##loose, tight, highPurity
        copyTrajectories = cms.untracked.bool( False ),
        nSigmaDtVertex = cms.double( 0.0 ),
        timesTag = cms.InputTag( "" ),
        ptMin = cms.double( 0.0 ), # minimum pT cut --> try 1.,2. GeV to reduce tracks
        ptMax = cms.double( 6.0 ), # maximum pT cut
        d0Max = cms.double( 999.0 ),
        copyExtras = cms.untracked.bool( False ),
        nVertices = cms.uint32( 2 ),
        vertexTag = cms.InputTag( "hltPixelVertices" ),
        src = cms.InputTag( "hltIter0PFLowPixelSeedsFromPixelTracks" ), # the complement reco::TrackCollection
        vtxFallback = cms.bool( True ),
        numberOfLostHits = cms.uint32( 999 ),
        numberOfValidPixelHits = cms.uint32( 4 ),
        timeResosTag = cms.InputTag( "" ),
        useVtx = cms.bool( False ) ## Turning off vertex selection
    )

    



    process.HLTIterativeTrackingIteration0 = cms.Sequence(process.hltIter0PFLowPixelSeedsFromPixelTracks+process.hltPixelTracksLowPT+process.hltIter0PFlowCkfTrackCandidates+process.hltIter0PFlowCtfWithMaterialTracks+process.hltIter0PFlowTrackCutClassifier+process.hltMergedTracks)


    # Merging Iter0 tracks with the complement of pixel tracks
    process.hltPFTracks = cms.EDProducer( "TrackListMerger",
        ShareFrac = cms.double( 0.19 ),
        FoundHitBonus = cms.double( 5.0 ),
        LostHitPenalty = cms.double( 20.0 ),
        MinPT = cms.double( 0.05 ),
        Epsilon = cms.double( -0.001 ),
        MaxNormalizedChisq = cms.double( 1000.0 ),
        MinFound = cms.int32( 3 ),
        TrackProducers = cms.VInputTag( 'hltPixelTracksLowPT','hltPFMuonMerging' ),
        hasSelector = cms.vint32( 0, 0),
        indivShareFrac = cms.vdouble( 1.0, 1.0),
        selectedTrackQuals = cms.VInputTag( 'hltPixelTracksLowPT','hltPFMuonMerging' ),
        setsToMerge = cms.VPSet(
          cms.PSet(  pQual = cms.bool( False ),
            tLists = cms.vint32( 0, 1)
          )
        ),
        trackAlgoPriorityOrder = cms.string( "hltESPTrackAlgoPriorityOrder" ),
        allowFirstHitShare = cms.bool( True ),
        newQuality = cms.string( "confirmed" ),
        copyExtras = cms.untracked.bool( True ),
        writeOnlyTrkQuals = cms.bool( False ),
        copyMVA = cms.bool( False )
    )
    
    
    process.HLTTrackReconstructionForPF = cms.Sequence(process.HLTDoLocalPixelSequence+process.HLTRecopixelvertexingSequence+process.HLTDoLocalStripSequence+process.HLTIterativeTrackingIter02+process.hltPFMuonMerging+process.hltPFTracks+process.hltMuonLinks+process.hltMuons)
    
    # change to use separate hltLightPFTracks for Taus
    process.hltLightPFTracksForTaus = process.hltLightPFTracks.clone()

    for iImporter in process.hltParticleFlowBlockForTaus.elementImporters:
        if iImporter.importerName == cms.string('GeneralTracksImporter'):
            iImporter.source = 'hltLightPFTracksForTaus'
    
    process.HLTParticleFlowSequenceForTaus = cms.Sequence(process.HLTPreshowerSequence+process.hltParticleFlowRecHitECALUnseeded+process.hltParticleFlowRecHitHBHE+process.hltParticleFlowRecHitHF+process.hltParticleFlowRecHitPSUnseeded+process.hltParticleFlowClusterECALUncorrectedUnseeded+process.hltParticleFlowClusterPSUnseeded+process.hltParticleFlowClusterECALUnseeded+process.hltParticleFlowClusterHBHE+process.hltParticleFlowClusterHCAL+process.hltParticleFlowClusterHF+process.hltLightPFTracksForTaus+process.hltParticleFlowBlockForTaus+process.hltParticleFlowForTaus)
    
    # use the mixed tracks collection for particle flow
    process.hltLightPFTracks.TkColList = cms.VInputTag("hltPFTracks")

    # Customize the full hlt vertices such that they do not use the complement tracks by requiring at least 1 valid strip hit
    process.hltVerticesPF.TrackLabel = cms.InputTag("hltPFTracks")
    process.hltVerticesPF.TkFilterParameters.minValidStripHits = cms.int32(1)
    
    # Add the PFTracks in HLTTrackingForBeamspot
    process.HLTTrackingForBeamSpot = cms.Sequence(process.HLTPreAK4PFJetsRecoSequence+process.HLTL2muonrecoSequence+process.HLTL3muonrecoSequence+process.HLTDoLocalPixelSequence+process.HLTRecopixelvertexingSequence+process.HLTDoLocalStripSequence+process.HLTIterativeTrackingIter02+process.hltPFMuonMerging+process.hltPFTracks)
    
    return process

