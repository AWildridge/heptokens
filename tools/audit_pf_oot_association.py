"""Exploratory OOT-count regression, not a reconstruction ablation.

Unweighted 4k prefixes; shared slopes with source intercepts. HC3 statistical
errors do not include detector modelling or model-specification uncertainty.
The coefficient measures association with average recorded N in eight BXs.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/heptokens-pf-diagnostic-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--results', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
tasks = [84,1373,2785,4151]
rows, responses, source_hashes = [], [], {}
for j, task in enumerate(tasks):
    path = a.output/('pf-audit-%05d.json.gz'%task)
    audit = json.load(gzip.open(path,'rt'))['events']
    path_native = a.results/'fullsim-delphes-validation-100k-20261021'/('%05d'%task)/'delphes.json'
    native = json.loads(path_native.read_text())['CMS'][:1000]
    for path_input in [path,path_native]:
        with path_input.open('rb') as f:
            source_hashes[str(path_input)] = hashlib.file_digest(f,'sha256').hexdigest()
    if len(audit) != 1000 or [r['key'] for r in audit] != [r['key'] for r in native]:
        raise ValueError('Invalid matched prefix')
    for r,d in zip(audit,native):
        pu = {b['bx']:b for b in r['pileup']}
        intime, mu = pu[0]['n'], pu[0]['mean']
        oot = sum(pu[bx]['n'] for bx in [-5,-4,-3,-2,-1,1,2,3])/8.
        rows.append([int(j==k) for k in range(4)]+[intime,mu,d['charged_pf'],oot,d['neutral_pf'],intime**2,mu**2,intime*mu])
        responses.append([r['counts']['charged'][0],sum(r['counts'][c][0] for c in ['photon','neutral_hadron','other_neutral'])])
x_all, y_all = np.asarray(rows,dtype=float), np.asarray(responses,dtype=float)
report = dict(scope=__doc__,events=4000,oot_bx=[-5,-4,-3,-2,-1,1,2,3],
              units='PF candidates per unit increase in average interaction count across the eight recorded OOT BXs',
              limitations=['Exploratory association only; no OOT-free events or counterfactual reconstruction.',
                           'Common slopes across four source prefixes; nonlinear/source-dependent effects may remain.',
                           'Event-level BX metadata have no candidate truth labels; detector-dependent bunch sensitivity is not modelled.',
                           'Unweighted regression; coefficients are not cross-section-stitched multiplicity differences.'],
              source_sha256=source_hashes,models={})
for spec, columns in [('linear',[0,1,2,3,4,5,6,7]),('hard-neutral-control',[0,1,2,3,4,5,6,8,7]),('quadratic-controls',[0,1,2,3,4,5,6,8,9,10,11,7])]:
    x = x_all[:,columns]
    bread = np.linalg.inv(x.T@x)
    leverage = np.sum((x@bread)*x,axis=1)
    report['models'][spec] = {}
    for k,name in enumerate(['charged','neutral']):
        y = y_all[:,k]
        beta = np.linalg.lstsq(x,y,rcond=None)[0]
        residual = y-x@beta
        covariance = bread@(x.T@((residual/(1-leverage))[:,None]**2*x))@bread
        sem = np.sqrt(covariance[-1,-1])
        report['models'][spec][name] = dict(oot_average_coefficient=float(beta[-1]),HC3_sem=float(sem),interval95=[float(beta[-1]-1.96*sem),float(beta[-1]+1.96*sem)])
report['tool_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
controls = x_all[:,:7]
oot = x_all[:,7]
rx = oot-controls@np.linalg.lstsq(controls,oot,rcond=None)[0]
fig,axes = plt.subplots(1,2,figsize=(13,5.7))
binned = {}
for k,(name,ax) in enumerate(zip(['charged','neutral'],axes)):
    y = y_all[:,k]
    ry = y-controls@np.linalg.lstsq(controls,y,rcond=None)[0]
    edges = np.quantile(rx,np.linspace(0,1,13))
    points=[]
    for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        selected=(rx>=lo)&((rx<hi) if i<11 else (rx<=hi))
        points.append(dict(x=float(rx[selected].mean()),y=float(ry[selected].mean()),sem=float(ry[selected].std(ddof=1)/np.sqrt(selected.sum())),events=int(selected.sum())))
    ax.errorbar([r['x'] for r in points],[r['y'] for r in points],yerr=[r['sem'] for r in points],
                ls='none',marker='o',mfc='white',color='#2166ac',capsize=3,label='Residual means in equal-count bins')
    model=report['models']['linear'][name]
    xx=np.linspace(edges[0],edges[-1],100)
    ax.plot(xx,model['oot_average_coefficient']*xx,color='#b2182b',ls='--',lw=1.8,label='Linear association')
    low,high=[c*xx for c in model['interval95']]
    ax.fill_between(xx,np.minimum(low,high),np.maximum(low,high),color='#b2182b',alpha=.12)
    ax.axhline(0,color='grey',lw=.8)
    ax.set_title(name.capitalize()+f" PF: coefficient {model['oot_average_coefficient']:.2f} +/- {model['HC3_sem']:.2f}",fontsize=12)
    ax.set_xlabel('Residual mean interaction count across eight OOT BXs',fontsize=10)
    ax.set_ylabel('Residual CMS '+name+' PF count',fontsize=10)
    ax.grid(alpha=.15);ax.legend(fontsize=8)
    binned[name]=points
fig.suptitle('Exploratory OOT-count association in 4,000 existing FullSim events',fontsize=14)
fig.subplots_adjust(top=.84,bottom=.25,wspace=.3)
fig.text(.5,.15,'Unweighted first 1k/source; four source intercepts, in-time N, true mean and no-PU Delphes charged count controlled.',ha='center',fontsize=9)
fig.text(.5,.10,'Points: binned residual means +/- SEM. Band: approximate 95% HC3 interval on the linear coefficient; statistical only.',ha='center',fontsize=9)
fig.text(.5,.05,'Residual zero does not mean zero OOT pileup. This association does not measure the OOT contribution or replace an ablation.',ha='center',fontsize=9)
fig.savefig(a.output/'oot-association-exploratory.pdf')
fig.savefig(a.output/'oot-association-exploratory.png',dpi=140)
plt.close(fig)
report['binned_partial_residuals']=binned
(a.output/'oot-regression-exploratory.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report['models'],indent=2))
