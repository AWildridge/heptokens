"""Read CMS FullSim reconstructed MiniAOD objects for exact replay events.

Run in the existing CMSSW_10_6_30 FWLite runtime (Python 2.7 compatible).
Does not modify original inputs, detector cards, or production outputs.
"""
from __future__ import print_function
import argparse,json,math,os,sys
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions=True
from DataFormats.FWLite import Events,Handle
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True);p.add_argument('--source-uri');a=p.parse_args()
reference=json.load(open(a.reference));expected=reference['events'];events=Events([a.input])
handles={name:Handle(kind) for name,kind in [('slimmedJets','std::vector<pat::Jet>'),('slimmedMETs','std::vector<pat::MET>'),('packedPFCandidates','std::vector<pat::PackedCandidate>'),('slimmedElectrons','std::vector<pat::Electron>'),('slimmedMuons','std::vector<pat::Muon>'),('slimmedPhotons','std::vector<pat::Photon>'),('generator','GenEventInfoProduct')]}
def finite(v):return not(math.isnan(v) or math.isinf(v))
def p4(obj):
 values=[float(obj.pt()),float(obj.eta()),float(obj.phi()),float(obj.mass())]
 if not all(finite(v) for v in values):raise ValueError('Nonfinite FullSim P4')
 return values
records=[]
for i,event in enumerate(events):
 if i>=len(expected):break
 aux=event.eventAuxiliary();key=[int(aux.run()),int(aux.luminosityBlock()),int(aux.event())];ref=expected[i]
 if key!=[ref['run'],ref['lumi'],ref['event']]:raise ValueError('FullSim/replay event identity mismatch at %d'%i)
 for name,handle in handles.items():
  event.getByLabel(name,handle)
  if not handle.isValid():raise RuntimeError('Missing FullSim collection '+name)
 weight=float(handles['generator'].product().weight())
 if abs(weight-ref['weight'])>1e-6*max(1,abs(weight)):raise ValueError('Generator weight mismatch')
 row={'key':key,'weight':weight}
 for name,out in [('slimmedJets','jets'),('slimmedElectrons','electrons'),('slimmedMuons','muons'),('slimmedPhotons','photons')]:row[out]=[p4(obj) for obj in handles[name].product()]
 met=handles['slimmedMETs'].product()
 if len(met)!=1:raise ValueError('Expected exactly one reconstructed MET')
 row['met']=float(met[0].pt());counts={'pf':0,'charged_pf':0,'pv_charged_pf':0}
 for obj in handles['packedPFCandidates'].product():
  if obj.pt()>0.5 and abs(obj.eta())<2.4:
   counts['pf']+=1
   if obj.charge()!=0:
    counts['charged_pf']+=1
    if obj.fromPV()>1:counts['pv_charged_pf']+=1
 row.update(counts);records.append(row)
 if (i+1)%100==0:
  print('Read matched FullSim events',i+1);sys.stdout.flush()
  with open(a.output+'.progress','w') as progress:json.dump({'matched_events':i+1},progress)
if len(records)!=len(expected):raise ValueError('Incomplete FullSim sample')
result={'schema':'heptokens-fullsim-validation-v1','input':a.input,'source_key':a.source_uri or reference.get('source_key',a.input),'events':records,'collections':list(handles),'jet_definition':'CMS MiniAOD slimmedJets: corrected AK4 PF CHS','pf_selection':'pt>0.5 GeV, abs(eta)<2.4; all packed PF candidates including original pileup'}
with open(a.output+'.partial','w') as out:json.dump(result,out);out.flush();os.fsync(out.fileno())
os.rename(a.output+'.partial',a.output);print('Validated FullSim event identities and weights:',len(records))
