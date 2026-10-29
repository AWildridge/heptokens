"""Matched 100k default-CMS versus stock pileup-card comparisons, with flows."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import shutil
import plot_pileup_pilot as plot
from plot_pileup_pilot import np, plt, hep, hist, PdfPages, TASKS, SAMPLES, digest

VIEWS = ['FullSim', 'original', 'default-zero', 'pileup-zero',
         'default-generic-poisson', 'pileup-generic-poisson',
         'default-cms-poisson', 'pileup-cms-poisson']
LABELS = ['CMS FullSim', 'Default CMS, original no PU',
          'Default CMS + merger, zero PU', 'PU card, zero PU',
          'Default CMS + generic PU', 'PU card + generic PU',
          'Default CMS + CMS MinBias PU', 'PU card + CMS MinBias PU']
COLORS = ['black', '#777777', '#f89c20', '#5790fc',
          '#e42536', '#964a8b', '#5790fc', '#7a21dd']
STYLES = [plot.series_style(view)[0] for view in VIEWS]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['pilot','prior','baseline','manifest']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    out = a.output or a.pilot/'plots'
    out.mkdir(exist_ok=True)
    manifest = json.loads(a.manifest.read_text())
    records = {str(r['record']):r for r in manifest['records']}
    plot.OBS = {k:v for k,v in plot.OBS.items() if k != 'chs_charged_pf'}
    plot.OBS['met'] = ('Missing transverse momentum [GeV]', np.arange(0,401,10))
    data, raw, sigmas, source_info = [], [], [], []
    for task in TASKS:
        base = a.baseline/('%05d'%task)
        refs = json.loads((base/'input-reference.json').read_text())['events']
        keys = [tuple(r[x] for x in ['run','lumi','event']) for r in refs]
        w = np.array([r['weight'] for r in refs])
        if len(keys) != 25000 or len(set(keys)) != 25000 or not np.isfinite(w).all() or (w<0).any():
            raise ValueError('Invalid reference sample')
        raw.append(w)
        record = records[str(manifest['files'][task]['record'])]
        sigmas.append(float(record['cross_section']['total_value']))
        source_info.append(dict(task=task, dataset=record['dataset'], events=25000,
                                cross_section_pb=sigmas[-1], sum_generator_weights=float(w.sum())))
        metadata = [json.loads(line) for line in gzip.open(a.prior/('metadata-%05d.jsonl.gz'%task),'rt')]
        if [tuple(r['key']) for r in metadata] != keys:
            raise ValueError('FullSim pileup metadata mismatch')
        views = {'FullSim':json.loads((base/'fullsim.json').read_text())['events'],
                 'original':json.loads((base/'delphes.json').read_text())['CMS']}
        for prefix, directory in [('default',a.pilot), ('pileup',a.prior)]:
            for variant in ['zero','generic-poisson','cms-poisson']:
                rows = [r for f in sorted((directory/'variants'/variant/('%05d'%task)).glob('*/summary.json'))
                        for r in json.loads(f.read_text())['rows']]
                mapped = {tuple(r['key']):r for r in rows}
                if len(rows) != 25000 or set(mapped) != set(keys):
                    raise ValueError('Missing/duplicate variant events')
                views[prefix+'-'+variant] = [mapped[k] for k in keys]
        sample = {}
        for view, rows in views.items():
            if [tuple(r['key']) for r in rows] != keys:
                raise ValueError('Matched event order differs')
            np.testing.assert_allclose([r['weight'] for r in rows],w,rtol=1e-6)
            values = {k:[] for k in plot.OBS}
            for i,r in enumerate(rows):
                selected = {k:r[k] for k in ['pf','charged_pf','met']}
                selected['neutral_pf'] = r['pf']-r['charged_pf'] if view == 'FullSim' else r['neutral_pf']
                selected['jets'] = sum(j[0]>30 and abs(j[1])<2.4 for j in r['jets']) if view in ['FullSim','original'] else r['jets']
                selected['n_pu'] = metadata[i]['n'] if view == 'FullSim' else (0 if view=='original' else r['n_pu'])
                for k in values:
                    values[k].append(selected[k])
            sample[view] = {k:np.asarray(v) for k,v in values.items()}
        data.append(sample)
    def select(names):
        plot.VIEWS = names
        indices = [VIEWS.index(v) for v in names]
        plot.LABELS = [LABELS[i] for i in indices]
        plot.COLORS = [COLORS[i] for i in indices]
        plot.LINESTYLES = [STYLES[i] for i in indices]
    stats = {}
    hep.style.use('CMS')
    pdf_path = out/'default-cms-pileup-comparison.pdf'
    pages = 0
    with PdfPages(pdf_path) as pdf:
        for variant in ['zero','generic-poisson','cms-poisson']:
            select(['FullSim','original','default-'+variant,'pileup-'+variant])
            title = 'Default CMS versus pileup card · '+variant
            plot.comparison(pdf,out,data,raw,sigmas,title,variant+'-card-comparison',stats)
            pages += 1
        select(['FullSim','default-zero','default-generic-poisson','default-cms-poisson'])
        plot.comparison(pdf,out,data,raw,sigmas,'Default CMS response · overlay models',
                        'default-overlay-models',stats)
        pages += 1
        select(['FullSim','original','default-cms-poisson','pileup-cms-poisson'])
        for sample,w,sigma,label,task in zip(data,raw,sigmas,SAMPLES,TASKS):
            plot.comparison(pdf,out,[sample],[w],[sigma],'CMS MinBias · QCD pThat '+label+' GeV',
                            'sample-%05d'%task,stats,True)
            pages += 1
        for view in ['default-zero','default-generic-poisson','default-cms-poisson']:
            fig,axes = plt.subplots(2,3,figsize=(18,11))
            for ax,(key,(label,edges)) in zip(axes.flat,plot.OBS.items()):
                shown = plot.shown_edges(edges)
                components = []
                for sample,w,sigma in zip(data,raw,sigmas):
                    h = hist.Hist(hist.axis.Variable(shown),storage=hist.storage.Double())
                    h.values()[...] = plot.histogram(sample[view][key],w*sigma/w.sum(),edges)
                    components.append(h)
                hep.histplot(components,ax=ax,stack=True,histtype='fill',color=hep.styles.cms.cmap_petroff[:4],
                             label=['pThat '+x+' GeV' for x in SAMPLES],yerr=False,flow='none',alpha=.75)
                for patch,hatch,style in zip(ax.patches[-4:],['///','\\\\','xxx','...'],STYLES):
                    patch.set_hatch(hatch);patch.set_linestyle(style);patch.set_edgecolor('black');patch.set_linewidth(.8)
                ax.set_yscale('log');ax.set_ylim(bottom=1e-4);ax.set_xlabel(label,fontsize=10)
                ax.set_ylabel('Cross section / bin [pb]',fontsize=10);plot.flows(ax,edges)
            handles,names=axes.flat[0].get_legend_handles_labels()
            fig.legend(handles,names,loc='upper center',bbox_to_anchor=(.5,.93),ncol=4,fontsize=11,framealpha=1.)
            fig.subplots_adjust(top=.86,bottom=.12,hspace=.35,wspace=.25)
            plot.footer(fig,LABELS[VIEWS.index(view)]+' · QCD components')
            pdf.savefig(fig);fig.savefig(out/('components-'+view+'.png'),dpi=130);plt.close(fig)
            pages += 1
    metrics = {}
    for view in VIEWS:
        metrics[view] = {}
        for key in plot.OBS:
            mean = sum(s*np.average(d[view][key],weights=w) for d,w,s in zip(data,raw,sigmas))/sum(sigmas)
            variance = sum(s*s*len(w)/(len(w)-1)*np.sum((w/w.sum())**2*(d[view][key]-np.average(d[view][key],weights=w))**2)
                           for d,w,s in zip(data,raw,sigmas))/sum(sigmas)**2
            metrics[view][key] = dict(cross_section_weighted_mean=float(mean),statistical_sem=float(np.sqrt(variance)))
    expected = sum(sigmas)
    for name,row in stats.items():
        total = sigmas[TASKS.index(int(name.split('/')[0].removeprefix('sample-')))] if name.startswith('sample-') else expected
        for v,h in row['cross_section_pb'].items():
            if not np.isclose(sum(h),total,rtol=1e-10):
                raise ValueError('Flow-inclusive normalization check failed')
    summary = dict(status='completed',events_per_view=100000,views=dict(zip(VIEWS,LABELS)),
                   normalization=source_info,metrics=metrics,pages=pages,
                   styles={v:dict(color=c,linestyle=s,marker=plot.series_style(v)[1]) for v,c,s in zip(VIEWS,COLORS,STYLES)},
                   statistics='Stratified paired delta-method histogram/ratio bands; fixed LO cross sections; statistical errors only',
                   limits=['One source prefix per pThat bin; not random campaign sampling',
                           'Default overlay retains stock detector/reconstruction, without CHS or rho subtraction',
                           'Both merger zero controls resample the hard vertex/time and consume RNG draws',
                           'Same seeds do not guarantee identical PU particles across reconstruction cards',
                           'Poisson means are CMS-derived; luminosity-profile variance not reproduced',
                           'Finite generic1000/CMS50600 libraries are reused; library correlations/model errors excluded',
                           'Approximate Gaussian vertices/160ps time; no original interaction or out-of-time detector replay'],
                   plot_tool_sha256=digest(Path(__file__)))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'histograms.json').write_text(json.dumps(stats,indent=2)+'\n')
    with (out/'mean-multiplicities.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['view','observable','weighted_mean','statistical_sem'])
        for view,row in metrics.items():
            for key,value in row.items():
                writer.writerow([view,key,value['cross_section_weighted_mean'],value['statistical_sem']])
    (out/'METHODOLOGY.txt').write_text(__doc__+'\n\n'+json.dumps(summary,indent=2)+'\n')
    shutil.copy2(a.pilot/'pilot-identity.json',out/'pilot-identity.json')
    (out/'SHA256SUMS').write_text(''.join(digest(f)+'  '+f.name+'\n' for f in sorted(out.iterdir()) if f.is_file() and f.name!='SHA256SUMS'))
    print(json.dumps(dict(status='completed',events_per_view=100000,pages=pages,pdf=str(pdf_path),metrics=metrics)))


if __name__ == '__main__':
    main()
