"""Auxiliary charged-PF comparison with an explicit CMS primary-vertex cut."""
import argparse,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/heptokens-validation-mpl-'+str(os.getuid()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import hist
import mplhep as hep
from plot_fullsim_delphes_validation import display_edges
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--sample-index',type=Path);p.add_argument('--xsec-manifest',type=Path);a=p.parse_args();out=a.output or a.results;out.mkdir(parents=True,exist_ok=True)
plt.style.use(hep.style.CMS);plt.rcParams.update({'font.size':10,'axes.labelsize':10,'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':1})
tasks=[84,1373,2785,4151];labels=['300–470','470–600','600–800','800–1000'];samples=[]
sample_index=json.loads(a.sample_index.read_text()) if a.sample_index else None
xsec_manifest=json.loads(a.xsec_manifest.read_text()) if a.xsec_manifest else None
for task,label in zip(tasks,labels):
 selected=next(r for r in sample_index['samples'] if r['task']==task) if sample_index else None
 full=json.loads((Path(selected['fullsim']) if selected else a.results/('fullsim-%05d.json'%task)).read_text())['events'];fast=json.loads((Path(selected['delphes']) if selected else a.results/('delphes-%05d.json'%task)).read_text())
 weights=np.array([r['weight'] for r in full])
 if xsec_manifest:
  record=next(r for r in xsec_manifest['records'] if str(r['record'])==str(xsec_manifest['files'][task]['record']));weights*=float(record['cross_section']['total_value'])/weights.sum()
 samples.append((label,weights,[np.array([r['charged_pf'] for r in full]),np.array([r['pv_charged_pf'] for r in full]),np.array([r['charged_pf'] for r in fast['CMS']]),np.array([r['charged_pf'] for r in fast['ATLAS']])]))
overview=('Cross-section-weighted mixture' if xsec_manifest else 'Four-bin equal-count mixture',np.concatenate([s[1] for s in samples]),[np.concatenate([s[2][i] for s in samples]) for i in range(4)])
fig,axes=plt.subplots(2,3,figsize=(15,9));edges=np.arange(-.5,700.6,15);legend=['CMS FullSim: all charged PF','CMS FullSim: charged PF from PV > 1','Delphes CMS EFlowTrack','Delphes ATLAS EFlowTrack'];colors=hep.styles.cms.cmap_petroff[:4]
for ax,(label,w,collections) in zip(axes.flat,[overview]+samples):
 for i,values in enumerate(collections):
  h=hist.Hist(hist.axis.Variable(edges),storage=hist.storage.Weight());h.fill(values,weight=w);shape=h.values(flow=True)/h.values(flow=True).sum();shown=display_edges(edges)
  ax.stairs(shape,shown,label=legend[i],color=colors[i],lw=1.8,linestyle='--' if i==0 else '-')
 ax.set_title(label+(' GeV' if label in labels else ''));ax.set_xlabel('Charged PF multiplicity');ax.set_ylabel('Normalized entries / bin');ax.grid(alpha=.2);ax.set_xlim(shown[0],shown[-1])
 for left,right,tag in [(shown[0],edges[0],'UF'),(edges[-1],shown[-1],'OF')]:
  ax.axvspan(left,right,color='grey',alpha=.12);ax.text((left+right)/2,.98,tag,transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=8)
axes.flat[0].legend(fontsize=8);axes.flat[-1].axis('off');axes.flat[-1].text(0,.88,'Charged particles: pT > 0.5 GeV, |eta| < 2.4\n\nCMS FullSim PV selection: fromPV() > 1.\nDelphes does not replay original pileup.\n\nFlow slots shown; normalization includes flows.\n\nThis cut affects only charged particles.\nNeutral pileup and other object-definition\ndifferences remain. This is a diagnostic,\nnot an efficiency or detector-closure test.',va='top',fontsize=11)
fig.suptitle('Matched FullSim–Delphes events: charged-PF primary-vertex diagnostic',fontsize=16);fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(out/'charged-pf-primary-vertex.png',dpi=140);fig.savefig(out/'charged-pf-primary-vertex.pdf');plt.close(fig)
print('Saved charged-PF primary-vertex diagnostic.')
