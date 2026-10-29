"""Evidence for PF multiplicity differences, without claiming an OOT ablation.

100k matched-event diagnostics plus a separate 4k MiniAOD content audit.
Uses the frozen sample cross sections and generator weights, before cuts.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/heptokens-pf-diagnostic-mpl')
import awkward as ak
import uproot
from plot_pileup_pilot import np, plt, hep, PdfPages, TASKS, SAMPLES, histogram, shown_edges, flows, digest
from matplotlib.lines import Line2D

STYLE = {
    'full': ('CMS FullSim, all', 'black', 'solid', None),
    'full-pv': ('CMS FullSim, fromPV > 1', '#2166ac', (0, (5, 2)), None),
    'fast': ('Delphes PU card, CMS recorded N', '#b2182b', (0, (6, 2, 1, 2)), None),
    'fast-chs': ('Delphes PU card, idealized CHS', '#762a83', (0, (1, 2)), None),
    'default': ('Delphes default card, CMS Poisson', '#008060', (0, (8, 2, 2, 2)), None),
    'none': ('Delphes default card, no PU', '#666666', (0, (3, 2)), None),
}


def line(ax, key, x, y, errors=None):
    label, color, linestyle, marker = STYLE[key]
    ax.errorbar(x, y, yerr=errors, label=label, color=color, linestyle=linestyle,
                marker=marker, mfc='white', ms=4, lw=1.7, capsize=2)


def shape(ax, key, arrays, weights, sigmas, edges, masks=None):
    values = np.zeros(len(edges)+1)
    total = 0.
    for i, (v, w, sigma) in enumerate(zip(arrays, weights, sigmas)):
        mask = np.ones(len(v), dtype=bool) if masks is None else masks[i]
        ws = w*sigma/w.sum()
        values += histogram(v[mask], ws[mask], edges)
        total += ws[mask].sum()
    values /= total
    label, color, linestyle, marker = STYLE[key]
    shown = shown_edges(edges)
    ax.stairs(values, shown, color=color, linestyle=linestyle, lw=1.7)
    flows(ax, edges)
    ax.set_ylabel('Probability / bin (including flows)')
    return Line2D([], [], label=label, color=color, ls=linestyle, marker=marker, mfc='white', lw=1.7)


def mean(arrays, weights, sigmas, masks=None):
    """Conditional ratio mean with stratified delta-method statistical SEM."""
    masks = [np.ones(len(v), dtype=bool) for v in arrays] if masks is None else masks
    scaled = [w*s/w.sum() for w,s in zip(weights,sigmas)]
    denominator = sum(ws[m].sum() for ws,m in zip(scaled,masks))
    if denominator <= 0:
        return None
    mu = sum(np.sum(ws[m]*v[m]) for v,ws,m in zip(arrays,scaled,masks))/denominator
    variance = 0.
    for v,w,ws,m in zip(arrays,weights,scaled,masks):
        influence = np.where(m, v-mu, 0.)
        center = np.average(influence, weights=w)
        variance += len(w)/(len(w)-1)*np.sum((ws*(influence-center))**2)
    all_selected = np.concatenate([ws[m] for ws,m in zip(scaled,masks)])
    return dict(mean=float(mu), statistical_sem=float(np.sqrt(variance)/denominator),
                events=int(sum(m.sum() for m in masks)),
                effective_events=float(denominator**2/np.sum(all_selected**2)))


def decorate(ax, xlabel, title):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.tick_params(labelsize=9)
    ax.grid(alpha=.15)


def save(fig, pdf, out, name, title, scope):
    fig.suptitle(title, fontsize=15, y=.99)
    fig.text(.5, .045, scope, ha='center', fontsize=9)
    fig.text(.5, .022, 'Statistical errors only; finite-library correlations/model systematics excluded. Vertex association is not a bunch-crossing label.', ha='center', fontsize=8)
    fig.subplots_adjust(top=.87, bottom=.13, hspace=.42, wspace=.3)
    pdf.savefig(fig)
    fig.savefig(out/(name+'.png'), dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=True)
    base = a.results/'fullsim-delphes-validation-100k-20261021'
    pilot = a.results/'pileup-pilot-100k-20261022'
    default = a.results/'default-cms-pileup-100k-20261022'
    manifest = json.loads(a.manifest.read_text())
    records = {str(r['record']):r for r in manifest['records']}
    data, weights, sigmas, inputs = [], [], [], []
    for task in TASKS:
        folder = base/('%05d'%task)
        full_path = folder/'fullsim.json'
        full = json.loads(full_path.read_text())['events']
        original = json.loads((folder/'delphes.json').read_text())['CMS']
        metadata_path = pilot/('metadata-%05d.jsonl.gz'%task)
        metadata = [json.loads(x) for x in gzip.open(metadata_path, 'rt')]
        keys = [tuple(r['key']) for r in full]
        if len(keys) != 25000 or len(set(keys)) != 25000:
            raise ValueError('Invalid original sample')
        if keys != [tuple(r['key']) for r in original] or keys != [tuple(r['key']) for r in metadata]:
            raise ValueError('FullSim/original/metadata event order mismatch')
        sample = dict(full=full, none=original, n=np.array([r['n'] for r in metadata]))
        for label, directory, variant in [('fast',pilot,'cms-recorded-count'),('default',default,'cms-poisson')]:
            rows = [r for path in sorted((directory/'variants'/variant/('%05d'%task)).glob('*/summary.json'))
                    for r in json.loads(path.read_text())['rows']]
            mapped = {tuple(r['key']):r for r in rows}
            if len(rows) != 25000 or set(mapped) != set(keys):
                raise ValueError('Missing or duplicate overlay events')
            sample[label] = [mapped[k] for k in keys]
        w = np.array([r['weight'] for r in full])
        for name in ['none','fast','default']:
            np.testing.assert_allclose([r['weight'] for r in sample[name]],w,rtol=1e-6)
        np.testing.assert_array_equal([r['n_pu'] for r in sample['fast']],sample['n'])
        sample['values'] = {
            'full': np.array([r['charged_pf'] for r in full]),
            'full-pv': np.array([r['pv_charged_pf'] for r in full]),
            'fast': np.array([r['charged_pf'] for r in sample['fast']]),
            'fast-chs': np.array([r['chs_charged_pf'] for r in sample['fast']]),
            'none': np.array([r['charged_pf'] for r in original]),
            'default': np.array([r['charged_pf'] for r in sample['default']]),
        }
        sample['values']['full-nonpv'] = sample['values']['full']-sample['values']['full-pv']
        sample['values']['fast-removed'] = sample['values']['fast']-sample['values']['fast-chs']
        sample['neutral'] = {
            'full': np.array([r['pf']-r['charged_pf'] for r in full]),
            'fast': np.array([r['neutral_pf'] for r in sample['fast']]),
            'default': np.array([r['neutral_pf'] for r in sample['default']]),
        }
        data.append(sample)
        weights.append(w)
        record = records[str(manifest['files'][task]['record'])]
        sigmas.append(float(record['cross_section']['total_value']))
        inputs += [dict(path=str(path),sha256=digest(path)) for path in [full_path,metadata_path]]
    values = lambda key: [d['values'][key] for d in data]
    neutral = lambda key: [d['neutral'][key] for d in data]
    metrics = {k:mean(values(k),weights,sigmas) for k in data[0]['values']}
    neutral_metrics = {k:mean(neutral(k),weights,sigmas) for k in data[0]['neutral']}
    deficits = {}
    for name,left,right in [('all','full','fast'),('PV-associated vs CHS','full-pv','fast-chs'),('remaining charged','full-nonpv','fast-removed')]:
        deficits[name] = mean([x-y for x,y in zip(values(left),values(right))],weights,sigmas)
    np.testing.assert_allclose(deficits['all']['mean'],sum(deficits[k]['mean'] for k in list(deficits)[1:]),atol=1e-10)
    conditional = {}
    hep.style.use('CMS')
    plt.rcParams.update({'font.size':10,'axes.labelsize':11,'legend.fontsize':9})
    pdf_path = a.output/'pf-pileup-diagnostics.pdf'
    with PdfPages(pdf_path) as pdf:
        fig,axes = plt.subplots(1,3,figsize=(18,6.4))
        handles = [shape(axes[0],k,values(k),weights,sigmas,np.arange(-.5,820,20)) for k in ['full','full-pv','fast','fast-chs']]
        axes[0].legend(handles=handles,handlelength=4,fontsize=8)
        decorate(axes[0],'Charged PF multiplicity','Reconstructed vertex-association diagnostic')
        handles = [shape(axes[1],k,neutral(k),weights,sigmas,np.arange(-.5,361,10)) for k in ['full','fast','default']]
        axes[1].legend(handles=handles,handlelength=4,fontsize=8)
        decorate(axes[1],'Neutral PF multiplicity (including photons)','Neutral agreement depends on reconstruction card')
        names=list(deficits)
        for i,(color,hatch) in enumerate(zip(['#b2182b','#2166ac','#762a83'],['///','xxx','...'])):
            entry=deficits[names[i]]
            axes[2].bar(i,entry['mean'],yerr=entry['statistical_sem'],color=color,hatch=hatch,edgecolor='black',capsize=3)
            axes[2].text(i,entry['mean']+4,f"{entry['mean']:.1f}",ha='center',fontsize=10)
        axes[2].set_xticks(range(3),['Total charged','PV vs CHS','Remaining\ncharged'])
        axes[2].set_ylabel('Mean CMS minus Delphes count')
        axes[2].set_ylim(0,155)
        axes[2].set_title('Paired charged-count deficit',fontsize=12)
        save(fig,pdf,a.output,'multiplicity-and-association','PF multiplicity differences: evidence from 100,000 matched hard events',
             'pT > 0.5 GeV, |eta| < 2.4; fixed LO cross sections and generator weights. Shapes include explicit UF/OF.')

        fig,axes=plt.subplots(2,2,figsize=(13,10))
        for ax,(name,left,right,is_neutral) in zip(axes.flat,[('All charged','full','fast',False),('PV-associated / CHS charged','full-pv','fast-chs',False),('Non-PV / CHS-removed charged','full-nonpv','fast-removed',False),('Neutral (including photons)','full','fast',True)]):
            arrays = [neutral(left),neutral(right)] if is_neutral else [values(left),values(right)]
            entries=[]
            for lo in range(0,60,5):
                masks=[(d['n']>=lo)&(d['n']<lo+5) for d in data]
                x=mean([d['n'] for d in data],weights,sigmas,masks)
                if x is None or x['effective_events']<10:
                    continue
                entries.append(dict(range=[lo,lo+5],x=x,full=mean(arrays[0],weights,sigmas,masks),fast=mean(arrays[1],weights,sigmas,masks)))
            for style,key in [('full' if is_neutral or left in ['full','full-nonpv'] else 'full-pv','full'),('fast-chs' if right=='fast-chs' else 'fast','fast')]:
                line(ax,style,[r['x']['mean'] for r in entries],[r[key]['mean'] for r in entries],[r[key]['statistical_sem'] for r in entries])
            # Non-PV and CHS-removed are set differences, not the full collections.
            if name.startswith('Non-PV'):
                axes_labels=['CMS charged minus fromPV > 1','Delphes raw charged minus CHS']
                handles,labels=ax.get_legend_handles_labels();ax.legend(handles,axes_labels,fontsize=8,handlelength=4)
            else:
                ax.legend(fontsize=8,handlelength=4)
            ax.set_ylabel('Mean candidate count')
            decorate(ax,'Recorded in-time pileup interactions',name)
            conditional[name]=entries
        save(fig,pdf,a.output,'conditional-multiplicities','Multiplicity versus the same recorded in-time interaction count',
             '100k matched events; paired recorded-N overlay. Both FullSim selections retain their original out-of-time response.')

        fig,axes=plt.subplots(2,2,figsize=(13,10))
        selections={}
        for row,(tag,condition) in enumerate([('N in [0,5)',lambda n:n<5),('N >= 35',lambda n:n>=35)]):
            masks=[condition(d['n']) for d in data]
            for col,(label,arrays,edges) in enumerate([('Charged',values,np.arange(-.5,1001,20)),('Neutral',neutral,np.arange(-.5,601,15))]):
                ax=axes[row,col]
                handles=[shape(ax,k,arrays(k),weights,sigmas,edges,masks) for k in ['full','fast']]
                entry=mean(arrays('full'),weights,sigmas,masks)
                decorate(ax,label+' PF multiplicity',tag+f"; {entry['events']} selected events")
                ax.legend(handles=handles,handlelength=4,fontsize=8)
                selections[tag+'/'+label]=dict(full=entry,fast=mean(arrays('fast'),weights,sigmas,masks))
        save(fig,pdf,a.output,'low-high-intime-count','Selecting low in-time pileup does not remove out-of-time pileup',
             '100k parent events; conditional cross-section-weighted shapes. Only 3 parent events have recorded N=0; no OOT-free claim.')

        audits,small_weights,native_counts=[],[],[]
        timing=dict(selected=0,available=0,nonzero_valid=0)
        for i,task in enumerate(TASKS):
            path=a.output/('pf-audit-%05d.json.gz'%task)
            audit=json.load(gzip.open(path,'rt'))
            rows=audit['events'];n=len(rows)
            if n != 1000 or [r['key'] for r in rows] != [r['key'] for r in data[i]['full'][:n]]:
                raise ValueError('4k audit identity mismatch')
            w=np.array([r['weight'] for r in rows]);np.testing.assert_allclose(w,weights[i][:n],rtol=1e-6)
            for j,r in enumerate(rows):
                if r['counts']['charged'][0] != data[i]['full'][j]['charged_pf'] or r['counts']['pv_charged'][0] != data[i]['full'][j]['pv_charged_pf']:
                    raise ValueError('4k audit count differs from frozen extraction')
                if sum(r['counts'][c][0] for c in ['photon','neutral_hadron','other_neutral']) != data[i]['full'][j]['pf']-data[i]['full'][j]['charged_pf']:
                    raise ValueError('4k neutral category sum differs from frozen extraction')
                for key in timing:timing[key]+=r['timing'][key]
            audits.append(audit);small_weights.append(w)
            native=base/('%05d'%task)/'chunk-000000/sample/pair/CMS.root'
            # Explicit local-file reader avoids the default asynchronous fsspec
            # source hanging on this mounted filesystem; one CPU worker only.
            with uproot.open(native, handler=uproot.source.file.MultithreadedFileSource,
                             num_workers=1,
                             decompression_executor=uproot.source.futures.TrivialExecutor(),
                             interpretation_executor=uproot.source.futures.TrivialExecutor()) as file:
                arrays=file['Delphes'].arrays(['Event.Number','Event.Weight','EFlowTrack.PT','EFlowTrack.Eta'],library='ak')
            np.testing.assert_array_equal(ak.to_numpy(ak.firsts(arrays['Event.Number'])),np.arange(1000))
            np.testing.assert_allclose(ak.to_numpy(ak.firsts(arrays['Event.Weight'])),w,rtol=1e-6)
            native_counts.append(np.array([ak.to_numpy(ak.sum((arrays['EFlowTrack.PT']>cut)&(abs(arrays['EFlowTrack.Eta'])<2.4),axis=1)) for cut in audit['thresholds_gev']]).T)
            np.testing.assert_array_equal(native_counts[-1][:,0],data[i]['values']['none'][:n])
            inputs.append(dict(path=str(path),sha256=digest(path)))
        cuts=audits[0]['thresholds_gev']
        thresholds={}
        fig,axes=plt.subplots(1,2,figsize=(15,6.5))
        for style,category in [('full-pv','pv_charged'),('none',None)]:
            rows=[]
            for j,cut in enumerate(cuts):
                arrays=[x[:,j] for x in native_counts] if category is None else [np.array([r['counts'][category][j] for r in au['events']]) for au in audits]
                rows.append(mean(arrays,small_weights,sigmas))
            line(axes[0],style,cuts,[r['mean'] for r in rows],[r['statistical_sem'] for r in rows])
            thresholds[style]=rows
        axes[0].legend(handlelength=4,fontsize=9);axes[0].set_ylabel('Mean charged candidate count');axes[0].set_xscale('log')
        decorate(axes[0],'Minimum pT [GeV]','Hard-event comparison proxy: threshold sensitivity')
        axes[0].text(.02,.04,'CMS PV association includes reconstruction effects;\nDelphes has no pileup. These are different definitions.',transform=axes[0].transAxes,fontsize=9)
        composition={}
        x=np.arange(2)
        for k,style in enumerate(['full','fast','default']):
            means=[];errors=[]
            for category in ['photon','neutral_hadron']:
                if style=='full':
                    arrays=[np.array([r['counts'][category][0] for r in au['events']]) for au in audits]
                else:
                    arrays=[np.array([r['photons'] if category=='photon' else r['neutral_pf']-r['photons'] for r in d[style][:1000]]) for d in data]
                entry=mean(arrays,small_weights,sigmas)
                means.append(entry['mean']);errors.append(entry['statistical_sem'])
                composition[style+'/'+category]=entry
            label,color,linestyle,marker=STYLE[style]
            axes[1].bar(x+(k-1)*.25,means,.25,yerr=errors,label=label,color=color,hatch=['///','xxx','...'][k],edgecolor='black',capsize=3,alpha=.8)
        axes[1].set_xticks(x,['Photons /\nDelphes residual ECAL towers','Neutral hadrons /\nDelphes residual HCAL towers'])
        axes[1].set_ylabel('Mean selected candidate count');axes[1].legend(fontsize=8)
        axes[1].set_title('Neutral composition: aggregate agreement can hide differences',fontsize=11)
        save(fig,pdf,a.output,'thresholds-and-neutral-composition','Separate 4,000-event audit: first 1,000 events per QCD source',
             'Same four QCD sources, independently normalized 1k prefixes; |eta|<2.4. Neutral panel pT/ET>0.5 GeV; object definitions differ.')
    neutral_differences={}
    for category in ['photon','neutral_hadron','total']:
        arrays=[]
        for au,d in zip(audits,data):
            rows=au['events']
            full=np.array([sum(r['counts'][c][0] for c in ['photon','neutral_hadron','other_neutral']) if category=='total' else r['counts'][category][0] for r in rows])
            fast=np.array([r['neutral_pf'] if category=='total' else (r['photons'] if category=='photon' else r['neutral_pf']-r['photons']) for r in d['fast'][:1000]])
            arrays.append(fast-full)
        neutral_differences[category]=mean(arrays,small_weights,sigmas)
    threshold_differences=[]
    for j,cut in enumerate(cuts):
        arrays=[np.array([r['counts']['pv_charged'][j] for r in au['events']])-native[:,j] for au,native in zip(audits,native_counts)]
        threshold_differences.append(mean(arrays,small_weights,sigmas))
    content_means={category:mean([np.array([r['counts'][category][0] for r in au['events']]) for au in audits],small_weights,sigmas) for category in audits[0]['categories']}
    summary=dict(status='passed',events_main=100000,events_content_audit=4000,
                 cuts=dict(pt_min_gev=.5,abs_eta_max=2.4),normalization='sigma_sample * gen_weight / sum_selected-prefix_generator_weights, before cuts; 25k/source main, 1k/source content audit',
                 charged=metrics,neutral=neutral_metrics,paired_deficits=deficits,
                 conditional=conditional,selections=selections,thresholds_gev=cuts,
                 threshold_means=thresholds,neutral_composition_4k=composition,
                 neutral_differences_delphes_minus_cms_4k=neutral_differences,
                 paired_threshold_differences_cmsPV_minus_delphesNoPU_4k=threshold_differences,
                 fullsim_category_means_4k=content_means,
                 timing_4k=timing,zero_intime_events=[int((d['n']==0).sum()) for d in data],
                 inputs=inputs,tool_sha256=digest(Path(__file__)),pdf=str(pdf_path),pages=4,
                 limits=['No in-time-only FullSim exists in these inputs; no OOT contribution is measured.',
                         'fromPV()>1 and Delphes CHS are different reconstructed associations, not truth time labels.',
                         'Fixed sample cross sections; paired stratified delta-method statistical SEM; library correlations and modelling uncertainties excluded.',
                         'One source-file prefix per sample; 4k audit is a separate diagnostic and not a campaign-wide estimate.',
                         'Neutral Delphes towers and CMS PF candidates have different reconstruction, merging and threshold response.'])
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['status','paired_deficits','timing_4k','neutral_composition_4k','threshold_means']},indent=2))


if __name__=='__main__':
    main()
