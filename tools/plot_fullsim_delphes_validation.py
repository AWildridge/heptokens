"""Matched-event, per-QCD-bin FullSim versus native Delphes diagnostic plots."""
import argparse,csv,hashlib,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/heptokens-validation-mpl-'+str(os.getuid()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import hist
import mplhep as hep

plt.style.use(hep.style.CMS)
plt.rcParams.update({'font.size':10,'axes.labelsize':10,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':9,'axes.linewidth':1,'xtick.major.size':5,'ytick.major.size':5})

TASKS=[84,1373,2785,4151]
LABELS=['300–470','470–600','600–800','800–1000']
SIMS=['FullSim CMS','Delphes CMS','Delphes ATLAS']
COLORS=hep.styles.cms.cmap_petroff[:3]
OBS={
 'leading_pt':('Leading jet $p_T$ [GeV]',np.linspace(0,1600,41)),
 'jet_pt':('Jet $p_T$ [GeV]',np.geomspace(30,2000,36)),
 'jet_eta':(r'Jet $\eta$',np.linspace(-2.4,2.4,33)),
 'jet_mass':('Jet mass [GeV]',np.linspace(0,160,41)),
 'met':('Missing transverse momentum [GeV]',np.linspace(0,400,41)),
 'ht':('$H_T$ [GeV]',np.linspace(0,3000,41)),
 'njet':('Jet multiplicity',np.arange(-.5,20.6,1)),
 'pf':('PF-candidate multiplicity',np.arange(-.5,1400.6,35)),
 'charged_pf':('Charged PF multiplicity',np.arange(-.5,1050.6,25)),
 'neutral_pf':('Neutral PF multiplicity (incl. photons)',np.arange(-.5,700.6,20)),
 'electrons':('Electron collection multiplicity',np.arange(-.5,8.6,1)),
 'muons':('Muon collection multiplicity',np.arange(-.5,8.6,1)),
}
KIN=list(OBS)[:6];MULT=list(OBS)[6:]
CUTS=r'Jets: $p_T>30$ GeV, $|\eta|<2.4$; PF: $p_T>0.5$ GeV, $|\eta|<2.4$; leptons: $p_T>10$ GeV, $|\eta|<2.4$'

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()

def json_finite(value):
 if isinstance(value,dict):return {k:json_finite(v) for k,v in value.items()}
 if isinstance(value,list):return [json_finite(v) for v in value]
 if isinstance(value,float) and not np.isfinite(value):return None
 return value

def observables(jets,met,counts,leptons):
 out={k:[] for k in OBS}
 for i,alljets in enumerate(jets):
  good=sorted([j for j in alljets if j[0]>30 and abs(j[1])<2.4],key=lambda j:-j[0])
  out['leading_pt'].append([good[0][0]] if good else [])
  for key,col in [('jet_pt',0),('jet_eta',1),('jet_mass',3)]:out[key].append([j[col] for j in good])
  out['met'].append([met[i]]);out['ht'].append([sum(j[0] for j in good)]);out['njet'].append([len(good)])
  for key in ['pf','charged_pf','neutral_pf']:out[key].append([counts[key][i]])
  for key in ['electrons','muons']:out[key].append([sum(v[0]>10 and abs(v[1])<2.4 for v in leptons[key][i])])
 return out

def fullsim(path):
 d=json.loads(path.read_text());rows=d['events'];counts={k:[r[k] for r in rows] for k in ['pf','charged_pf']};counts['neutral_pf']=[r['pf']-r['charged_pf'] for r in rows]
 return observables([r['jets'] for r in rows],[r['met'] for r in rows],counts,{k:[r[k] for r in rows] for k in ['electrons','muons']}),np.array([r['weight'] for r in rows]),d

def delphes(path,detector,reference):
 rows=json.loads(path.read_text())[detector]
 assert [r['key'] for r in rows]==[[r['run'],r['lumi'],r['event']] for r in reference['events']]
 np.testing.assert_allclose([r['weight'] for r in rows],[r['weight'] for r in reference['events']],rtol=1e-6)
 return observables([r['jets'] for r in rows],[r['met'] for r in rows],{key:[r[key] for r in rows] for key in ['pf','charged_pf','neutral_pf']},{key:[r[key] for r in rows] for key in ['electrons','muons']})


def histmatrix(rows,edges,weights):
 # Keep event rows for paired bootstrap correlations, including inclusive jets.
 h=hist.Hist(hist.axis.Integer(0,len(rows),name='event',underflow=False,overflow=False),hist.axis.Variable(edges,name='value',underflow=True,overflow=True),storage=hist.storage.Weight())
 lengths=np.array([len(v) for v in rows]);values=np.concatenate([np.asarray(v,dtype=float) for v in rows])
 if not np.isfinite(values).all():raise ValueError('Nonfinite observable')
 h.fill(event=np.repeat(np.arange(len(rows)),lengths),value=values,weight=np.repeat(weights,lengths))
 matrix=h.values(flow=True);total=float(matrix.sum());inside=float(matrix[:,1:-1].sum())
 flow={'weighted_entries':total,'weighted_entries_in_range':inside,'underflow_weight':float(matrix[:,0].sum()),'overflow_weight':float(matrix[:,-1].sum()),'fraction_outside':(total-inside)/total if total else 0,'sumw2_with_flow':h.variances(flow=True).sum(axis=0).tolist()}
 return matrix,flow

def display_edges(edges):
 # Finite display slots only; physical flow intervals remain unbounded.
 return np.r_[edges[0]-(edges[1]-edges[0]),edges,edges[-1]+(edges[-1]-edges[-2])]

def normalized(matrix,replicas=None):
 h=matrix.sum(axis=0) if replicas is None else replicas@matrix
 sums=h.sum(axis=-1,keepdims=True)
 return np.divide(h,sums,out=np.full_like(h,np.nan),where=sums!=0)

def plot_page(pdf,data,weights,groups,keys,title,out,basename,csvrows,stats,rng,absolute=False):
 fig=plt.figure(figsize=(14,13));grid=fig.add_gridspec(3,2,hspace=.5,wspace=.28)
 replicas=np.zeros((160,len(weights)))
 for group in sorted(set(groups)):
  idx=np.where(groups==group)[0];replicas[:,idx]=rng.multinomial(len(idx),np.full(len(idx),1/len(idx)),size=160)
  if absolute:
   # Bootstrap each sample at fixed catalog cross section; its total generator
   # weight fluctuates, so recompute the sample normalization in each replica.
   denom=replicas[:,idx]@weights[idx]
   replicas[:,idx]*=(weights[idx].sum()/denom)[:,None]
 for index,key in enumerate(keys):
  label,edges=OBS[key];inner=grid[index//2,index%2].subgridspec(2,1,height_ratios=[3,1],hspace=.06);ax=fig.add_subplot(inner[0]);rat=fig.add_subplot(inner[1],sharex=ax);shown=display_edges(edges);centers=(shown[:-1]+shown[1:])/2
  shapes=[];boots=[];details={}
  for si,sim in enumerate(SIMS):
   matrix,flow=histmatrix(data[sim][key],edges,weights);shape=matrix.sum(axis=0) if absolute else normalized(matrix);boot=replicas@matrix if absolute else normalized(matrix,replicas);shapes.append(shape);boots.append(boot)
   plotted=hist.Hist(hist.axis.Variable(shown),storage=hist.storage.Double());plotted.values()[...]=shape
   hep.histplot(plotted,ax=ax,color=COLORS[si],histtype='step',yerr=False,lw=1.7,label=sim,flow='none')
   lo,hi=np.nanpercentile(boot,[16,84],axis=0);ax.fill_between(shown,np.r_[lo,lo[-1]],np.r_[hi,hi[-1]],step='post',color=COLORS[si],alpha=.12)
   entries=sum(len(v) for v in data[sim][key]);weighted_entries=sum(len(v)*w for v,w in zip(data[sim][key],weights))
   mean=sum(sum(v)*w for v,w in zip(data[sim][key],weights))/weighted_entries if weighted_entries else None
   details[sim]={'flow':flow,('bin_cross_section_pb' if absolute else 'shape_including_flow'):shape.tolist(),'edges':edges.tolist(),'display_edges':shown.tolist(),'entries':entries,'weighted_mean':mean}
   physical=np.r_[-np.inf,edges,np.inf]
   for i in range(len(centers)):csvrows.append([basename,key,sim,'underflow' if i==0 else 'overflow' if i==len(centers)-1 else 'regular',physical[i],physical[i+1],shape[i],float(matrix[:,i].sum()),flow['sumw2_with_flow'][i]])
  for si in [1,2]:
   ratio=np.divide(shapes[si],shapes[0],out=np.full_like(shapes[0],np.nan),where=shapes[0]>0)
   ratios=np.divide(boots[si],boots[0],out=np.full_like(boots[0],np.nan),where=boots[0]>0)
   valid=np.any(np.isfinite(ratios),axis=0);lo=np.full(ratio.shape,np.nan);hi=np.full(ratio.shape,np.nan)
   if valid.any():lo[valid],hi[valid]=np.nanpercentile(ratios[:,valid],[16,84],axis=0)
   rat.plot(centers,ratio,color=COLORS[si],lw=1.3);rat.fill_between(centers,lo,hi,color=COLORS[si],alpha=.17)
   details[SIMS[si]]['ratio_to_fullsim']=ratio.tolist()
   details[SIMS[si]]['ratio_68pct_lower']=lo.tolist();details[SIMS[si]]['ratio_68pct_upper']=hi.tolist()
  rat.axhline(1,color='#888888',lw=1,ls='--');rat.set_ylim(0,3);rat.set_ylabel('Fast / Full',fontsize=9);rat.set_xlabel(label);rat.grid(alpha=.2);ax.grid(alpha=.15);ax.tick_params(labelbottom=False);ax.set_ylabel('Cross section / bin [pb]' if absolute else 'Normalized entries / bin')
  if key in ['jet_pt','jet_mass','met','ht']:ax.set_yscale('log');ax.set_ylim(bottom=1e-6)
  if key=='jet_pt':ax.set_xscale('log')
  for panel in [ax,rat]:
   panel.axvspan(shown[0],edges[0],color='grey',alpha=.12);panel.axvspan(edges[-1],shown[-1],color='grey',alpha=.12)
   panel.axvline(edges[0],color='grey',ls=':',lw=.8);panel.axvline(edges[-1],color='grey',ls=':',lw=.8);panel.set_xlim(shown[0],shown[-1])
  for pos,text in [(centers[0],'UF'),(centers[-1],'OF')]:ax.text(pos,.98,text,ha='center',va='top',transform=ax.get_xaxis_transform(),fontsize=8)
  if index==0:ax.legend(fontsize=9,loc='upper right',bbox_to_anchor=(.92,.87))
  ax.set_title(label,fontsize=11)
  stats[basename+'/'+key]=details
 fig.suptitle(title,fontsize=15,y=.985)
 fig.text(.5,.956,'CMS FullSim AK4 PF CHS  |  Delphes CMS anti-$k_T$ R=0.5  |  Delphes ATLAS anti-$k_T$ R=0.6',ha='center',fontsize=10)
 fig.text(.5,.935,CUTS,ha='center',fontsize=8.5)
 fig.subplots_adjust(top=.90,bottom=.105)
 fig.text(.5,.050,'UF / OF: explicit underflow / overflow slots (unbounded intervals). Both flows retained; no width division.',ha='center',fontsize=9)
 fig.text(.5,.035,'Catalog LO cross sections / sample sum(gen weights); fixed-cross-section paired bootstrap 68% intervals.' if absolute else 'Shape comparison; raw generator weights, no cross-section normalization. Bands: paired event-bootstrap 68% intervals.',ha='center',fontsize=9)
 fig.text(.5,.018,'FullSim includes original pileup/reconstruction; replay uses packed hard-scatter stable truth without pileup. Native object definitions differ.',ha='center',fontsize=8)
 pdf.savefig(fig);fig.savefig(out/(basename+'.png'),dpi=140);plt.close(fig)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--samples',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--sample-index',type=Path,help='Explicit larger-sample references and diagnostic summaries');p.add_argument('--xsec-manifest',type=Path,help='Cross-section stitched overview and colored sample components');a=p.parse_args();out=a.output or a.results;out.mkdir(parents=True,exist_ok=True);all_data=[];all_weights=[];sources=[];stats={};csvrows=[];rng=np.random.default_rng(20261021)
 sample_index=json.loads(a.sample_index.read_text()) if a.sample_index else None
 for task,label in zip(TASKS,LABELS):
  selected=next(r for r in sample_index['samples'] if r['task']==task) if sample_index else None
  folder=a.samples/('bounds-validation-final-%05d'%task);reference_path=Path(selected['reference']) if selected else folder/'pair/input-reference.json';reference=json.loads(reference_path.read_text());fullsim_path=Path(selected['fullsim']) if selected else a.results/('fullsim-%05d.json'%task);delphes_path=Path(selected['delphes']) if selected else a.results/('delphes-%05d.json'%task);baseline,weights,record=fullsim(fullsim_path)
  assert len(weights)==(selected['events'] if selected else 1000) and [r['key'] for r in record['events']]==[[r['run'],r['lumi'],r['event']] for r in reference['events']]
  np.testing.assert_allclose(weights,[r['weight'] for r in reference['events']],rtol=1e-6)
  data={SIMS[0]:baseline,SIMS[1]:delphes(delphes_path,'CMS',reference),SIMS[2]:delphes(delphes_path,'ATLAS',reference)};all_data.append(data);all_weights.append(weights)
  sources.append({'task':task,'qcd_pt_hat_bin_GeV':label,'events':len(weights),'event_ids_and_weights':'matched exactly (weights rtol1e-6)','source_uri':record['source_key'],'reference_sha256':digest(reference_path),'fullsim_summary_sha256':digest(fullsim_path),'delphes_summary_sha256':digest(delphes_path),'cards_sha256':{sim:digest(Path(selected['cards'][sim]) if selected else folder/'pair'/('delphes_card_'+sim+'.tcl')) for sim in ['CMS','ATLAS']},'weight_range':[float(weights.min()),float(weights.max())]})
  if selected:sources[-1]['chunk_provenance']=selected['chunk_provenance']
  else:sources[-1]['delphes_root_sha256']={sim:digest(folder/'pair'/(sim+'.root')) for sim in ['CMS','ATLAS']}
 if a.xsec_manifest:
  from plot_qcd_components import make_stitched_plots
  make_stitched_plots(a.xsec_manifest,out,all_data,all_weights,sources);return
 pdfpath=out/'fullsim-delphes-validation.pdf'
 with PdfPages(pdfpath) as pdf:
  overview={sim:{key:sum([d[sim][key] for d in all_data],[]) for key in OBS} for sim in SIMS};weights=np.concatenate(all_weights);groups=np.concatenate([np.full(len(w),i) for i,w in enumerate(all_weights)])
  for keys,tag in [(KIN,'kinematics'),(MULT,'multiplicities')]:plot_page(pdf,overview,weights,groups,keys,f'{len(weights):,} matched events · equal-event mixture of four QCD bins',out,'overview-'+tag,csvrows,stats,rng)
  for task,label,data,weights in zip(TASKS,LABELS,all_data,all_weights):
   for keys,tag in [(KIN,'kinematics'),(MULT,'multiplicities')]:plot_page(pdf,data,weights,np.zeros(len(weights)),keys,'QCD $\\hat{p}_T$ '+label+f' GeV · {len(weights):,} matched events',out,'%05d-%s'%(task,tag),csvrows,stats,rng)
 with (out/'histograms.csv').open('w') as f:
  writer=csv.writer(f);writer.writerow(['sample','observable','simulation','bin_kind','bin_low','bin_high','normalized_bin_probability_including_flow','sumw','sumw2']);writer.writerows(csvrows)
 (out/'histograms-and-ratios.json').write_text(json.dumps(json_finite(stats),indent=2,allow_nan=False)+'\n')
 manifest={'status':'completed','events':4000,'scope':'Representative first1000 events per four original QCD sources; shape diagnostics, not detector closure certification. CMS FullSim is the only FullSim reference.','samples':sources,'observables':list(OBS),'cuts':{'jets':'pt>30GeV abs(eta)<2.4','PF':'pt>0.5GeV abs(eta)<2.4','leptons':'pt>10GeV abs(eta)<2.4; native collection selection, no extra ID/isolation'},'normalization':'Each distribution independently unit area inside plotted range, raw generator weights. Overview equal1000event-per-bin mixture, not physical QCD cross-section spectrum.','statistics':'160 paired multinomial event bootstraps; stratified by QCD bin in overview;16th–84th percentiles. Ratios mask zero FullSim bins; displayed0..3, raw values retained in JSON.','limitations':['CMS slimmed AK4 PF CHS corrected jets, DelphesCMS R0.5 and ATLAS R0.6 native jets: different radii/calibrations/object removal.','CMS FullSim has original pileup; Delphes uses packed hard-scatter truth without original pileup or full ancestry.','Lepton native collection identification/isolation and PF reconstruction definitions differ.','Uncertainties describe this small matched-event sample, not modelling systematics.'],'pdf':str(pdfpath),'plot_tool_sha256':digest(Path(__file__))}
 manifest.update(normalization='Each distribution independently unit area including explicit underflow and overflow, raw generator weights; probabilities per bin, no bin-width division. Overview equal1000events per pThat bin, not physical inclusive QCD.',histogram_library={'hist':hist.__version__,'mplhep':hep.__version__},palette={'name':'CMS Petroff (mplhep CMS style)','colors':COLORS},flow_display='Finite shaded end slots represent unbounded physical flow intervals; included in normalization, paired bootstraps, and ratios. Zero FullSim denominator masked.')
 (out/'validation-summary.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({'pdf':str(pdfpath),'events':4000,'figures':10,'matched_samples':4}))
if __name__=='__main__':main()
