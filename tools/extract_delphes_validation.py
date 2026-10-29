"""Read diagnostic observables with the same ROOT/Delphes runtime as production."""
import argparse,json,os
from pathlib import Path
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions=True
p=argparse.ArgumentParser();p.add_argument('--sample',type=Path,required=True);p.add_argument('--delphes',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
ROOT.gInterpreter.AddIncludePath(str(a.delphes));ROOT.gInterpreter.AddIncludePath(str(a.delphes/'external'))
if ROOT.gSystem.Load(str(a.delphes/'libDelphes.so'))<0:raise RuntimeError('Cannot load matching Delphes runtime')
reference=json.loads((a.sample/'pair/input-reference.json').read_text());result={}
for detector in ['CMS','ATLAS']:
 f=ROOT.TFile.Open(str(a.sample/'pair'/(detector+'.root')));tree=f.Get('Delphes');tree.SetBranchStatus('*',0)
 for name in ['Event','Jet','MissingET','EFlowTrack','EFlowPhoton','EFlowNeutralHadron','Electron','Muon']:tree.SetBranchStatus(name+'*',1)
 rows=[]
 for i,ref in enumerate(reference['events']):
  if tree.GetEntry(i)<=0:raise ValueError('Missing Delphes event')
  event=tree.Event.At(0)
  if int(event.Number)!=ref['export_event'] or abs(float(event.Weight)-ref['weight'])>1e-6*max(1,abs(ref['weight'])):raise ValueError('Delphes event/weight mismatch')
  row={'key':[ref['run'],ref['lumi'],ref['event']],'weight':float(event.Weight),'jets':[[float(j.PT),float(j.Eta),float(j.Phi),float(j.Mass)] for j in tree.Jet],'met':float(tree.MissingET.At(0).MET)}
  for name,key in [('Electron','electrons'),('Muon','muons')]:row[key]=[[float(v.PT),float(v.Eta),float(v.Phi),0.] for v in getattr(tree,name)]
  counts={}
  for name,key,field in [('EFlowTrack','charged_pf','PT'),('EFlowPhoton','photons','ET'),('EFlowNeutralHadron','neutral_hadrons','ET')]:counts[key]=sum(float(getattr(v,field))>.5 and abs(float(v.Eta))<2.4 for v in getattr(tree,name))
  row.update(charged_pf=counts['charged_pf'],neutral_pf=counts['photons']+counts['neutral_hadrons'],pf=sum(counts.values()));rows.append(row)
 f.Close();result[detector]=rows
with a.output.open('w') as stream:json.dump(result,stream)
print('Matched Delphes events per detector:',len(reference['events']),flush=True)
