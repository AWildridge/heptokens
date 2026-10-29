"""Conditional multiplicities versus the same recorded in-time PU count.

The recorded-N Delphes arm and CMS FullSim share hard events and N, while
particle identities, vertices, detector response and reconstruction differ.
"""
import argparse
import gzip
import json
from pathlib import Path

from plot_pileup_pilot import TASKS, histogram, hep, plt, np, series_style


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    records = {str(r['record']):r for r in manifest['records']}
    full, fast, counts, weights = [], [], [], []
    for task in TASKS:
        refs = json.loads((a.baseline/('%05d'%task)/'input-reference.json').read_text())['events']
        original = json.loads((a.baseline/('%05d'%task)/'fullsim.json').read_text())['events']
        metadata = [json.loads(x) for x in gzip.open(a.pilot/('metadata-%05d.jsonl.gz'%task),'rt')]
        rows = [r for path in (a.pilot/'variants/cms-recorded-count'/('%05d'%task)).glob('*/summary.json') for r in json.loads(path.read_text())['rows']]
        mapped = {tuple(r['key']):r for r in rows}
        if len(rows) != 25000 or len(mapped) != 25000:
            raise ValueError('Incomplete recorded-count comparison')
        sigma = float(records[str(manifest['files'][task]['record'])]['cross_section']['total_value'])
        w = np.array([r['weight'] for r in refs]);weights.extend(w*sigma/w.sum())
        for ref, r, meta in zip(refs,original,metadata):
            key = tuple(ref[x] for x in ['run','lumi','event'])
            v = mapped[key]
            if list(key) != r['key'] or list(key) != meta['key'] or v['n_pu'] != meta['n']:
                raise ValueError('Conditional-comparison identity differs')
            full.append([r['pf'],r['charged_pf'],r['pf']-r['charged_pf'],r['pv_charged_pf']])
            fast.append([v['pf'],v['charged_pf'],v['neutral_pf'],v['chs_charged_pf']])
            counts.append(meta['n'])
    weights=np.array(weights);counts=np.array(counts);full=np.array(full);fast=np.array(fast)
    edges=np.arange(-.5,65.,5.);stats={}
    hep.style.use('CMS');fig,axes=plt.subplots(2,2,figsize=(13,10))
    labels=['All PF candidates','Charged PF candidates','Neutral PF candidates','CMS PV-associated / Delphes CHS charged PF']
    for column,(ax,label) in enumerate(zip(axes.flat,labels)):
        rows=[]
        for lo,hi in zip(edges[:-1],edges[1:]):
            selected=(counts>=lo)&(counts<hi);w=weights[selected]
            effective=w.sum()**2/np.sum(w*w) if len(w) else 0
            if effective<10:
                continue
            x=float(np.average(counts[selected],weights=w));entry={'n_range':[float(lo),float(hi)],'n_mean':x,'events':int(selected.sum()),'effective_events':float(effective)}
            for name,values in [('FullSim',full),('CMS-recorded-N',fast)]:
                y=values[selected,column];mean=float(np.average(y,weights=w))
                error=float(np.sqrt(np.sum(w*w*(y-mean)**2))/w.sum())
                entry[name]={'mean':mean,'weighted_sem':error}
            rows.append(entry)
        for name,color,label_name,view in [('FullSim','black','CMS FullSim','FullSim'),('CMS-recorded-N',hep.styles.cms.cmap_petroff[4],'Delphes, CMS library + recorded N','cms-recorded-count')]:
            linestyle,marker=series_style(view)
            ax.errorbar([r['n_mean'] for r in rows],[r[name]['mean'] for r in rows],yerr=[r[name]['weighted_sem'] for r in rows],linestyle=linestyle,marker=marker,mfc='white',ms=4,lw=1.4,color=color,label=label_name)
        ax.set_xlabel('Original in-time pileup interactions',fontsize=12);ax.set_ylabel('Mean '+label,fontsize=11);ax.grid(alpha=.2)
        ax.tick_params(labelsize=11)
        if column==0:ax.legend(fontsize=9,handlelength=5)
        stats[label]=rows
    fig.suptitle('CMS simulation comparison · 100,000 matched hard events',fontsize=15)
    fig.subplots_adjust(top=.92,bottom=.18,hspace=.35,wspace=.27)
    fig.text(.5,.09,'Same hard events and exact in-time counts; randomly sampled pileup particles and approximate vertices in Delphes.',ha='center',fontsize=10)
    fig.text(.5,.06,'Cross-section-weighted conditional means; weighted statistical SEM; bins with effective N<10 omitted.',ha='center',fontsize=9)
    fig.text(.5,.03,'PF pT>0.5GeV, |eta|<2.4. CMS PV association and idealized Delphes CHS differ. Library-reuse/model uncertainties excluded.',ha='center',fontsize=9)
    out=a.output or a.pilot/'plots';out.mkdir(exist_ok=True);fig.savefig(out/'multiplicity-vs-recorded-pileup.pdf');fig.savefig(out/'multiplicity-vs-recorded-pileup.png',dpi=150);plt.close(fig)
    (out/'multiplicity-vs-recorded-pileup.json').write_text(json.dumps({'events':100000,'normalization':'LO cross-section weighted within original N bins','statistics':'Weighted SEM, finite-library correlations and model errors excluded','bins':stats},indent=2)+'\n')
    print('Validated conditional response comparison:100000 matched events')


if __name__=='__main__':main()
