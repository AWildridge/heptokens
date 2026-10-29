"""Matched-event, cross-section-weighted multiplicity validation with flows.

Statistical bands use an analytic paired delta method, stratified by QCD bin,
with each sample's catalog cross section held fixed. No model uncertainties.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import hist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import mplhep as hep
import numpy as np

TASKS = [84, 1373, 2785, 4151]
SAMPLES = ['300–470', '470–600', '600–800', '800–1000']
VIEWS = ['FullSim', 'original', 'zero', 'generic-poisson', 'cms-poisson', 'cms-recorded-count']
LABELS = ['CMS FullSim', 'Original Delphes, no PU', 'PU card, zero PU',
          'Generic library, Poisson', 'CMS library, Poisson', 'CMS library, recorded N']
OBS = {'pf': ('PF candidates', np.arange(0, 821, 20)),
       'charged_pf': ('Charged PF candidates', np.arange(0, 511, 10)),
       'neutral_pf': ('Neutral PF candidates (including photons)', np.arange(0, 361, 10)),
       'chs_charged_pf': ('Charged PF: CMS PV-associated / Delphes CHS', np.arange(0, 311, 10)),
       'jets': ('Jets, pT > 30 GeV', np.arange(-.5, 20.5, 1.)),
       'n_pu': ('In-time pileup interactions', np.arange(-.5, 80.5, 2.))}
COLORS = ['black', *hep.styles.cms.cmap_petroff[:5]]
# Shared by both card comparisons and the recorded-count response supplement.
# Distinct dash patterns are shared across main panels, ratios and legends.
# The user requested no markers on comparison lines (2026-10-27).
SERIES_STYLES = {
    'FullSim': ('solid', None),
    'original': ((0, (5, 2)), None),
    'zero': ((0, (1, 2)), None),
    'generic-poisson': ((0, (4, 2, 1, 2)), None),
    'cms-poisson': ((0, (8, 3)), None),
    'cms-recorded-count': ((0, (4, 2, 1, 2, 1, 2)), None),
    'default-zero': ((0, (2, 2)), None),
    'default-generic-poisson': ((0, (8, 2, 2, 2)), None),
    'default-cms-poisson': ((0, (8, 2, 1, 2)), None),
    'matched-generic': ((0, (7, 2, 1, 2)), None),
    'matched-cms': ((0, (3, 2, 1, 2, 1, 2)), None),
}


def series_style(view):
    return SERIES_STYLES[view.removeprefix('pileup-')]


LINESTYLES = [series_style(view)[0] for view in VIEWS]


def series_handle(view, label, color):
    linestyle, marker = series_style(view)
    return Line2D([], [], label=label, color=color, linestyle=linestyle,
                  marker=marker, markerfacecolor='white', markersize=4, lw=1.7)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def histogram(values, weights, edges):
    h = hist.Hist(hist.axis.Variable(edges, underflow=True, overflow=True), storage=hist.storage.Weight())
    h.fill(values, weight=weights)
    return h.values(flow=True)


def indices(values, edges):
    return np.searchsorted(edges, values, side='right')


def paired_histograms(values, raw, sigmas, edges):
    """All selected events contribute once, including underflow and overflow."""
    totals, variance, ratio_variance = {}, {}, {}
    bins = len(edges) + 1
    for view in VIEWS:
        totals[view] = np.zeros(bins)
        variance[view] = np.zeros(bins)
    scaled = [w * sigma / w.sum() for w, sigma in zip(raw, sigmas)]
    for view in VIEWS:
        for v, w, ws, sigma in zip(values[view], raw, scaled, sigmas):
            h = histogram(v, ws, edges)
            totals[view] += h
            probability = h / sigma
            w2_hist = histogram(v, ws * ws, edges)
            variance[view] += len(w) / (len(w) - 1) * (w2_hist * (1 - 2 * probability) + probability**2 * np.sum(ws**2))
    full = totals['FullSim']
    for view in VIEWS[1:]:
        ratio = np.divide(totals[view], full, out=np.full_like(full, np.nan), where=full > 0)
        numerator_variance = np.zeros(bins)
        for vf, vr, ws, sigma in zip(values[view], values['FullSim'], scaled, sigmas):
            b, br = indices(vf, edges), indices(vr, edges)
            hf, hr = histogram(vf, ws, edges), histogram(vr, ws, edges)
            h2f, h2r = histogram(vf, ws**2, edges), histogram(vr, ws**2, edges)
            same = np.bincount(b[b == br], weights=(ws**2)[b == br], minlength=bins)
            c = np.nan_to_num((hf - ratio * hr) / sigma)
            r = np.nan_to_num(ratio)
            value = h2f + r*r*h2r - 2*r*same - 2*c*(h2f-r*h2r) + c*c*np.sum(ws**2)
            numerator_variance += len(ws)/(len(ws)-1)*value
        ratio_variance[view] = np.divide(np.maximum(numerator_variance, 0), full**2,
                                         out=np.full_like(full, np.nan), where=full > 0)
    return totals, {k: np.maximum(v, 0) for k, v in variance.items()}, ratio_variance


def shown_edges(edges):
    return np.r_[edges[0] - (edges[1]-edges[0]), edges, edges[-1] + (edges[-1]-edges[-2])]


def flows(ax, edges):
    shown = shown_edges(edges)
    for lo, hi, name in [(shown[0], edges[0], 'UF'), (edges[-1], shown[-1], 'OF')]:
        ax.axvspan(lo, hi, color='grey', alpha=.12)
        ax.text((lo+hi)/2, .98, name, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=8)
    ax.set_xlim(shown[0], shown[-1])


def footer(fig, title, events=100000):
    fig.suptitle('CMS simulation comparison · ' + title, fontsize=15, y=.995)
    fig.text(.5, .952, f'{events:,} matched hard events shown; PF pT > 0.5 GeV, jets pT > 30 GeV; |η| < 2.4', ha='center', fontsize=10)
    fig.text(.5, .044, 'Four QCD pThat prefixes, 25k events each; LO cross-section stitching; explicit under/overflow; no fitted smoothing.', ha='center', fontsize=9)
    fig.text(.5, .023, 'Statistical errors only. CMS FullSim includes detector premixing/OOT response; Delphes overlays in-time particles. Native object definitions differ.', ha='center', fontsize=8)


def comparison(pdf, out, data, raw, sigmas, title, filename, statistics, normalize=False):
    fig = plt.figure(figsize=(18, 12))
    outer = fig.add_gridspec(2, 3, top=.86, bottom=.12, hspace=.38, wspace=.25)
    legend_handles = legend_labels = None
    for cell, (key, (label, edges)) in zip(outer, OBS.items()):
        inner = cell.subgridspec(2, 1, height_ratios=[3, 1], hspace=.04)
        ax = fig.add_subplot(inner[0]); ratio_ax = fig.add_subplot(inner[1], sharex=ax)
        values = {view: [d[view][key] for d in data] for view in VIEWS}
        totals, variances, ratio_variances = paired_histograms(values, raw, sigmas, edges)
        scale = sum(sigmas) if normalize else 1.
        shown = shown_edges(edges); centers = .5*(shown[1:]+shown[:-1])
        for view, name, color, linestyle in zip(VIEWS, LABELS, COLORS, LINESTYLES):
            h = hist.Hist(hist.axis.Variable(shown), storage=hist.storage.Double())
            h.values()[...] = totals[view] / scale
            hep.histplot(h, ax=ax, histtype='step', color=color, linestyle=linestyle, label=name, lw=1.7, flow='none', yerr=False)
            if view == 'FullSim':
                err = np.sqrt(variances[view]) / scale
                ax.fill_between(shown, np.r_[h.values()-err, (h.values()-err)[-1]],
                                np.r_[h.values()+err, (h.values()+err)[-1]], step='post', color='black', alpha=.15)
            else:
                full = totals['FullSim']
                ratio = np.divide(totals[view], full, out=np.full_like(full, np.nan), where=full > 0)
                error = np.sqrt(ratio_variances[view])
                ratio_ax.stairs(ratio, shown, color=color, linestyle=linestyle, lw=1.3)
                ratio_ax.fill_between(shown, np.r_[ratio-error, (ratio-error)[-1]],
                                      np.r_[ratio+error, (ratio+error)[-1]], step='post', color=color, alpha=.10)
        maximum = max(v.max() / scale for v in totals.values())
        ax.set_yscale('log'); ax.set_ylim(max(maximum * 1e-5, 1e-8), maximum*8)
        ax.set_ylabel('Probability / bin' if normalize else 'Cross section / bin [pb]', fontsize=10)
        if key == 'n_pu' and any(view.startswith('matched-') for view in VIEWS):
            ax.text(.03,.97,'Matched-overlay N = FullSim event by event',transform=ax.transAxes,va='top',fontsize=8)
        ax.tick_params(labelbottom=False); ax.grid(alpha=.15)
        ratio_ax.axhline(1, color='grey', lw=.8); ratio_ax.set_ylim(0, 3)
        ratio_ax.set_ylabel('/ FullSim', fontsize=9); ratio_ax.set_xlabel(label, fontsize=10)
        for axis in [ax, ratio_ax]:
            flows(axis, edges); axis.tick_params(labelsize=9)
        if key == 'pf':
            legend_handles = [series_handle(v, n, c) for v, n, c in zip(VIEWS, LABELS, COLORS)]
            legend_labels = LABELS
        statistics[filename+'/'+key] = {'regular_edges': edges.tolist(), 'flow_slots': ['underflow','regular bins','overflow'],
                                      'cross_section_pb': {v: totals[v].tolist() for v in VIEWS},
                                      'statistical_variance_pb2': {v: variances[v].tolist() for v in VIEWS},
                                      'paired_ratio_variance': {v: np.where(np.isfinite(x), x, -1).tolist() for v,x in ratio_variances.items()}}
    fig.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(.5,.93),
               ncol=2 if len(VIEWS)<=4 else 3, fontsize=10, handlelength=5, columnspacing=1.8)
    footer(fig, title, sum(len(w) for w in raw))
    pdf.savefig(fig); fig.savefig(out/(filename+'.png'), dpi=130); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['pilot','baseline','manifest']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--output', type=Path)
    a = p.parse_args(); out = a.output or a.pilot/'plots'; out.mkdir(exist_ok=True)
    manifest = json.loads(a.manifest.read_text())
    records = {str(r['record']):r for r in manifest['records']}
    data, raw, sigmas, source_info, metrics = [], [], [], [], {}
    for task in TASKS:
        base = a.baseline/('%05d'%task)
        refs = json.loads((base/'input-reference.json').read_text())['events']
        keys = [[r[x] for x in ['run','lumi','event']] for r in refs]
        weights = np.array([r['weight'] for r in refs], dtype=float)
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError('This LO pilot requires finite nonnegative weights')
        raw.append(weights)
        record = records[str(manifest['files'][task]['record'])]
        sigmas.append(float(record['cross_section']['total_value']))
        source_info.append({'task':task,'dataset':record['dataset'],'cross_section_pb':sigmas[-1],
                            'sum_generator_weights':float(weights.sum()),'events':25000})
        full = json.loads((base/'fullsim.json').read_text())['events']
        native = json.loads((base/'delphes.json').read_text())['CMS']
        metadata = [json.loads(x) for x in gzip.open(a.pilot/('metadata-%05d.jsonl.gz'%task),'rt')]
        if [r['key'] for r in full] != keys or [r['key'] for r in native] != keys or [r['key'] for r in metadata] != keys:
            raise ValueError('Reference FullSim/native/pileup event identity differs')
        views = {'FullSim':full, 'original':native}
        for variant in VIEWS[2:]:
            group = a.pilot/'variants'/variant/('%05d'%task)
            rows = [r for path in sorted(group.glob('*/summary.json')) for r in json.loads(path.read_text())['rows']]
            mapped = {tuple(r['key']):r for r in rows}
            if len(rows) != 25000 or len(mapped) != 25000 or set(mapped) != {tuple(k) for k in keys}:
                raise ValueError('Pilot variant missing, duplicate or unexpected events')
            views[variant] = [mapped[tuple(k)] for k in keys]
            np.testing.assert_allclose([r['weight'] for r in views[variant]], weights, rtol=1e-6)
        sample = {}
        for view, rows in views.items():
            values = {key:[] for key in OBS}
            for i,r in enumerate(rows):
                if view == 'FullSim':
                    selected = {'pf':r['pf'],'charged_pf':r['charged_pf'],'neutral_pf':r['pf']-r['charged_pf'],
                                'chs_charged_pf':r['pv_charged_pf'],'jets':sum(j[0]>30 and abs(j[1])<2.4 for j in r['jets']),'n_pu':metadata[i]['n']}
                elif view == 'original':
                    selected = {'pf':r['pf'],'charged_pf':r['charged_pf'],'neutral_pf':r['neutral_pf'],
                                'chs_charged_pf':r['charged_pf'],'jets':sum(j[0]>30 and abs(j[1])<2.4 for j in r['jets']),'n_pu':0}
                else:
                    selected = {key:r[key] for key in OBS}
                for key,value in selected.items():values[key].append(value)
            sample[view] = {k:np.asarray(v) for k,v in values.items()}
        if not np.array_equal(sample['cms-recorded-count']['n_pu'], sample['FullSim']['n_pu']):
            raise ValueError('Recorded-count pilot does not match CMS event-by-event N')
        data.append(sample)
    statistics = {}; pdf_path = out/'pileup-multiplicity-validation.pdf'
    hep.style.use('CMS')
    with PdfPages(pdf_path) as pdf:
        comparison(pdf,out,data,raw,sigmas,'Cross-section-weighted QCD, pThat 300–1000 GeV','stitched-multiplicities',statistics)
        comparison(pdf,out,data,raw,sigmas,'Normalized cross-section-weighted shapes','normalized-multiplicities',statistics,True)
        for sample,w,sigma,label,task in zip(data,raw,sigmas,SAMPLES,TASKS):
            comparison(pdf,out,[sample],[w],[sigma],'QCD pThat '+label+' GeV (25k events)','sample-%05d'%task,statistics,True)
        for view,name in zip(VIEWS,LABELS):
            fig,axes=plt.subplots(2,3,figsize=(18,11))
            for ax,(key,(label,edges)) in zip(axes.flat,OBS.items()):
                shown=shown_edges(edges);components=[]
                for sample,w,sigma in zip(data,raw,sigmas):
                    h=hist.Hist(hist.axis.Variable(shown),storage=hist.storage.Double())
                    h.values()[...]=histogram(sample[view][key],w*sigma/w.sum(),edges);components.append(h)
                hep.histplot(components,ax=ax,stack=True,histtype='fill',color=hep.styles.cms.cmap_petroff[:4],
                             label=['pThat '+x+' GeV' for x in SAMPLES],yerr=False,flow='none',alpha=.8)
                for patch,hatch,style in zip(ax.patches[-4:],['///','\\\\','xxx','...'],LINESTYLES):
                    patch.set_hatch(hatch);patch.set_linestyle(style);patch.set_edgecolor('black');patch.set_linewidth(.8)
                ax.set_yscale('log'); ax.set_ylim(bottom=1e-4);ax.set_xlabel(label,fontsize=10);ax.set_ylabel('Cross section / bin [pb]',fontsize=10)
                flows(ax,edges);ax.tick_params(labelsize=9)
            handles,names=axes.flat[0].get_legend_handles_labels()
            fig.legend(handles,names,loc='upper center',bbox_to_anchor=(.5,.93),ncol=4,fontsize=11,framealpha=1.)
            fig.subplots_adjust(top=.86,bottom=.12,hspace=.35,wspace=.25);footer(fig,name+' · QCD components')
            pdf.savefig(fig);fig.savefig(out/('components-'+view+'.png'),dpi=130);plt.close(fig)
        profile=json.loads((a.pilot/'target-profile.json').read_text())
        fig,axes=plt.subplots(1,2,figsize=(14,5))
        input_handles=[]
        for view,name,color,linestyle in zip(VIEWS,LABELS,COLORS,LINESTYLES):
            if view in ['original','zero']:continue
            values=np.concatenate([x[view]['n_pu'] for x in data]);w=np.concatenate([g*s/g.sum() for g,s in zip(raw,sigmas)])
            edges=OBS['n_pu'][1];h=hist.Hist(hist.axis.Variable(shown_edges(edges)),storage=hist.storage.Double());h.values()[...]=histogram(values,w/w.sum(),edges)
            hep.histplot(h,ax=axes[0],histtype='step',color=color,linestyle=linestyle,label=name,yerr=False,flow='none');flows(axes[0],edges)
            shown=shown_edges(edges)
            input_handles.append(series_handle(view,name,color))
        axes[0].set_xlabel('In-time pileup interactions');axes[0].set_ylabel('Probability / bin');axes[0].legend(handles=input_handles,fontsize=8,handlelength=5)
        axes[1].axis('off');axes[1].text(0,.95,'Method and limits\n\nMeasured CMS mean ≈22.4; variance ≈66.\nFixed-mean Poisson is deliberately narrower.\nRecorded-N arm matches each event count exactly.\n\nBoth CMS-library arms randomly reuse50,600 GEN events.\nGeneric library has1,000 events.\nVertices: measured mean and sigma, Gaussian approximation.\nTiming: assumed160ps; GEN ct=0.\nNo CMS detector premixing or out-of-time response.\n\nCMS PV association differs from idealized Delphes CHS.\nExact original interactions are not replayed.',va='top',fontsize=11)
        fig.tight_layout();pdf.savefig(fig);fig.savefig(out/'pileup-input-model.png',dpi=140);plt.close(fig)
    for view in VIEWS:
        metrics[view]={}
        for key in OBS:
            weighted=sum(s*np.average(d[view][key],weights=w) for d,w,s in zip(data,raw,sigmas))/sum(sigmas)
            metrics[view][key]={'cross_section_weighted_mean':float(weighted),'ratio_to_fullsim_mean':float(weighted/(sum(s*np.average(d['FullSim'][key],weights=w) for d,w,s in zip(data,raw,sigmas))/sum(sigmas)))}
    summary={'status':'completed','events_per_view':100000,'views':dict(zip(VIEWS,LABELS)),
             'normalization':source_info,'formula':'sigma_sample*generator_weight/sum_validation_generator_weights before cuts',
             'uncertainties':'Analytic paired delta method stratified by QCD sample, catalog cross sections held fixed; statistical only. Components have no uncertainty bands.',
             'assumptions':['One prefix per pThat bin, not random campaign sampling.','Native pileup card changes detector reconstruction; zero-PU card control retained.','CMS FullSim jets are AK4 CHS; native Delphes jet definitions differ.','CMS fromPV()>1 and idealized Delphes CHS are different association definitions.','CMS GEN origin is approximated by modal stable-particle vertex; relative decays retained; production ct unavailable and set0.','Fixed-mean Poisson does not reproduce CMS luminosity-profile variance.','Recorded-N overlay matches counts only; interaction IDs and recorded z values are not matched.','Both overlay methods randomly reuse a finite library, with approximate Gaussian vertices and160ps timing.','No out-of-time detector response or CMS premixing/reconstruction is reproduced.'],
             'metrics':metrics,'pdf':str(pdf_path),'pages':13,'plot_tool_sha256':digest(Path(__file__)),
             'styles':{v:dict(color=c,linestyle=series_style(v)[0],marker=series_style(v)[1]) for v,c in zip(VIEWS,COLORS)},
             'pilot_identity_sha256':digest(a.pilot/'pilot-identity.json')}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'histograms.json').write_text(json.dumps(statistics,indent=2)+'\n')
    with (out/'mean-multiplicities.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['view','observable','weighted_mean','ratio_to_fullsim_mean'])
        for view,row in metrics.items():
            for key,r in row.items():writer.writerow([view,key,r['cross_section_weighted_mean'],r['ratio_to_fullsim_mean']])
    (out/'README.txt').write_text('100k-event CMS FullSim / Delphes pileup pilot\n\n25,000 matched hard events per QCD pThat bin;100,000 per view.\nSee summary.json for selections, normalization, methods and limitations.\nUF/OF are explicit unbounded flow intervals shown as finite display slots; no density or width normalization.\nCharged-PV/CHS comparison is diagnostic, not identical object selection.\nNo smoothing, fitted pileup mean, luminosity scaling or K factors.\nNative ROOT archives require matching ARCHIVE-MANIFEST SHA256 and two stable EOS proofs; evidence remains on WORK.\n')
    for path in [a.pilot/'pilot-identity.json',a.pilot/'target-profile.json',a.pilot/'exact-interaction-coverage.json']:
        import shutil
        shutil.copy2(path,out/path.name)
    paths=sorted(p for p in out.iterdir() if p.is_file() and p.name!='SHA256SUMS')
    (out/'SHA256SUMS').write_text(''.join(digest(p)+'  '+p.name+'\n' for p in paths))
    print(json.dumps({'status':'completed','events_per_view':100000,'pdf':str(pdf_path),'metrics':metrics}))


if __name__=='__main__':main()
