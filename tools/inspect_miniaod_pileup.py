"""Inspect actual MiniAOD pileup summary fields without changing event content."""
from __future__ import print_function
import argparse,json
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions=True
from DataFormats.FWLite import Events,Handle
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--events',type=int,default=10);a=p.parse_args()
events=Events([a.input]);handle=Handle('std::vector<PileupSummaryInfo>');rows=[]
for i,event in enumerate(events):
 if i>=a.events:break
 event.getByLabel('slimmedAddPileupInfo',handle)
 if not handle.isValid():raise RuntimeError('No slimmedAddPileupInfo')
 aux=event.eventAuxiliary();row={'run':int(aux.run()),'lumi':int(aux.luminosityBlock()),'event':int(aux.event()),'pileup':[]}
 for pu in handle.product():
  info={'bunch_crossing':int(pu.getBunchCrossing()),'n_interactions':int(pu.getPU_NumInteractions()),'true_mean_interactions':float(pu.getTrueNumInteractions()),'bunch_spacing_ns':int(pu.getBunchSpacing())}
  for method in ['getPU_zpositions','getPU_EventID','getPU_pT_hats','getPU_sumpT_lowpT','getPU_sumpT_highpT']:
   try:info[method+'_size']=len(getattr(pu,method)())
   except Exception as error:info[method+'_error']=str(error)[:200]
  if pu.getBunchCrossing()==0:
   try:info['zpositions_cm']=[float(v) for v in pu.getPU_zpositions()]
   except Exception:pass
   try:info['pileup_event_ids']=[{'run':int(v.run()),'lumi':int(v.luminosityBlock()),'event':int(v.event())} for v in pu.getPU_EventID()]
   except Exception:pass
  row['pileup'].append(info)
 rows.append(row)
with open(a.output,'w') as output:json.dump({'input':a.input,'events':rows,'label':'slimmedAddPileupInfo'},output,indent=2)
print('Read pileup summaries from',len(rows),'actual MiniAOD events')
