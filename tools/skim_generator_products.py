"""CMSSW configuration: preserve all retained generator products without reconstruction.

Run with cmsRun in the pinned original-compatible CMSSW runtime. The resulting
CMS ROOT file preserves native products/provenance; a separate native exporter
adds ordinary uproot-readable TTrees without reducing the archival content.
"""
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing('analysis')
options.parseArguments()
process = cms.Process('GENBACKFILL')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 10000
process.source = cms.Source('PoolSource', fileNames=cms.untracked.vstring(options.inputFiles))
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(options.maxEvents))
process.options = cms.untracked.PSet(numberOfThreads=cms.untracked.uint32(1),
                                    numberOfStreams=cms.untracked.uint32(1),
                                    wantSummary=cms.untracked.bool(True))
process.out = cms.OutputModule('PoolOutputModule',
    fileName=cms.untracked.string(options.outputFile),
    outputCommands=cms.untracked.vstring(
        'drop *',
        'keep *_*_*_GEN',
        'keep *_generator_*_*',
        'keep *_generatorSmeared_*_*',
        'keep *_externalLHEProducer_*_*',
        'keep *_genParticles_*_*',
        'keep *_packedGenParticles_*_*',
        'keep *_prunedGenParticles_*_*',
        'keep *_genPUProtons_*_*',
        'keep *_genFilterEfficiencyProducer_*_*',
        'keep *_slimmedGenJets*_*_*',
        'keep recoGenJets_*_*_*',
        'keep recoGenMETs_*_*_*',
        'keep *_slimmedAddPileupInfo_*_*',
    ),
    compressionAlgorithm=cms.untracked.string('ZLIB'),
    compressionLevel=cms.untracked.int32(4),
    dropMetaData=cms.untracked.string('NONE'),
    fastCloning=cms.untracked.bool(False))
process.path = cms.EndPath(process.out)
