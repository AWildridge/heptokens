"""Validate and plot the100k pilot with the actual CMS identity-sharing graph."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import sqlite3

import plot_pileup_pilot as plot
from plot_pileup_pilot import np, plt, hep, PdfPages, digest, TASKS, SAMPLES
from pileup_library_inventory import inventory


def conditional_estimate(values,masks,raw,sigmas):
    """Ratio estimator with source-wise normalized-weight fluctuations."""
    scaled=[w*s/w.sum() for w,s in zip(raw,sigmas)]
    denominator=sum(ws[m].sum() for ws,m in zip(scaled,masks))
    if denominator<=0:
        raise ValueError('Empty conditional estimator')
    mean=sum(np.sum(ws[m]*y[m]) for ws,m,y in zip(scaled,masks,values))/denominator
    variance=0.
    for w,s,mask,y in zip(raw,sigmas,masks,values):
        residual=np.where(mask,y-mean,0.)
        residual-=np.average(residual,weights=w)
        variance+=s*s*len(w)/(len(w)-1)*np.sum((w/w.sum())**2*residual**2)
    return dict(mean=float(mean),statistical_sem=float(np.sqrt(variance)/denominator))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['pilot','baseline','prior-default','metadata','manifest','schedule']:
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    out=a.pilot/'plots';out.mkdir(exist_ok=True)
    schedule=json.loads((a.schedule/'index.json').read_text())
    expected=np.load(schedule['identity_file'])['uses']
    rows_by_arm={}
    occupancy={}
    for arm in ['generic','cms']:
        complete=json.loads((a.pilot/arm/'complete.json').read_text())
        if complete['status']!='complete-native-runs-and-durable-archives-validated' or complete['hard_events']!=100000:
            raise ValueError('Incomplete arm')
        ledger=a.pilot/arm/'interaction-ledger.sqlite'
        if digest(ledger)!=complete['ledger_sha256']:
            raise ValueError('Physical ledger changed')
        with sqlite3.connect('file:'+str(ledger.resolve())+'?mode=ro',uri=True) as db:
            actual=np.array(db.execute('select uses from interactions order by idx').fetchall(),dtype=np.uint8).ravel()
            np.testing.assert_array_equal(actual,expected)
            if db.execute('select count(distinct physical_id),count(distinct payload_sha256) from interactions').fetchone()!=(len(expected),len(expected)):
                raise ValueError('Physical bank identity/payload diversity failed')
            if db.execute('select count(*) from chunks').fetchone()[0]!=100:
                raise ValueError('Ledger lacks100 native chunks')
        occupancy[arm]=complete
        occupancy[arm]['particle_and_storage_inventory']=inventory(a.pilot/arm)
        source_events=0
        for source_path in sorted((a.pilot/arm/'sources').glob('*/source.json')):
            identity=json.loads(source_path.read_text())
            source_events+=identity['events']
            if arm=='cms' and identity['index']==0:
                origin=a.schedule.parent/'native-export-validation'
                extraction=json.loads((origin/'compiled-50600.pileup.summary.json').read_text())
                comparison=json.loads((origin/'comparison.json').read_text())
                if comparison['status']!='byte-identical-all-50600-interactions' or identity['sha256']!=comparison['canonical_sha256']:
                    raise ValueError('Original CMS source provenance differs')
                audit_sha=comparison['canonical_sha256']
            else:
                audit=json.loads((source_path.parent/'audit.json').read_text())
                if audit['status']!='verified-all-records-and-native-roundtrip' or audit['events']!=identity['events']:
                    raise ValueError('Physical source native audit differs')
                audit_sha=audit['sha256']
                extraction=json.loads((source_path.parent/('physical.pileup.summary.json' if arm=='cms' else 'physical.pileup.generation.json')).read_text())
            if identity['sha256']!=audit_sha or extraction['events']!=identity['events']:
                raise ValueError('Physical source identity/count differs')
            if arm=='cms':
                if extraction['sha256']!=audit_sha or not extraction['all_gen_weights_one'] or extraction['event_ids']!=identity['event_ids']:
                    raise ValueError('CMS source bytes, IDs or unit weights differ')
            elif not extraction['all_generator_weights_one'] or extraction['seed']!=202610280+identity['index']:
                raise ValueError('Generic source unit weights/seed differ')
        if source_events<len(expected) or (arm=='generic' and source_events!=len(expected)):
            raise ValueError('Physical source event inventory incomplete')
        samples={task:[] for task in TASKS}
        for chunk in schedule['chunks']:
            sequence=chunk['sequence']
            folder=a.pilot/arm/'groups'/('group-%03d'%(sequence//5))/('sequence-%03d'%sequence)
            summary=json.loads((folder/'summary.json').read_text())
            mapping=json.loads((folder/'bank.pileup.json').read_text())
            archive=json.loads((folder.parent/'archive-verified.json').read_text())
            sharing=json.loads((folder.parent/'scheduled-sharing-audit.json').read_text())
            if sharing['status']!='every-slot-matches-scheduled-physical-payload':
                raise ValueError('Full-slot particle sharing not independently checked')
            checked={r['sequence']:r for r in sharing['chunks']}[sequence]
            if checked['sha256']!=mapping['sha256'] or checked['mapping_sha256']!=mapping['mapping_sha256'] or checked['schedule_sha256']!=chunk['schedule_sha256']:
                raise ValueError('Independent payload-sharing proof differs')
            if archive['status']!='stable-and-read-back-twice':
                raise ValueError('Archive lacks two stable full readbacks')
            for name in ['summary.json','card.tcl','bank.pileup.json']:
                if archive['members_sha256'][folder.name+'/'+name]!=digest(folder/name):
                    raise ValueError('Local evidence differs from durable archive')
            if archive['members_sha256'][folder.name+'/delphes.root']!=summary['root_sha256'] or archive['members_sha256'][folder.name+'/bank.pileup']!=mapping['sha256']:
                raise ValueError('Native ROOT or scheduled payload archive identity differs')
            if summary['identity']['library_sha256']!=mapping['sha256'] or mapping['schedule_sha256']!=chunk['schedule_sha256']:
                raise ValueError('Schedule-to-native chain differs')
            if summary['status']!='validated' or summary['events']!=1000:
                raise ValueError('Missing native validation')
            samples[chunk['task']].extend(summary['rows'])
        rows_by_arm[arm]=samples
    manifest=json.loads(a.manifest.read_text())
    records={str(v['record']):v for v in manifest['records']}
    views=['FullSim','default-zero','matched-generic','matched-cms']
    labels=['CMS FullSim','Default CMS + merger, zero PU','Generic PU, matched reuse and N','CMS MinBias PU, matched reuse and N']
    colors=['black','#777777','#e42536','#5790fc']
    plot.OBS={k:v for k,v in plot.OBS.items() if k!='chs_charged_pf'}
    plot.OBS['met']=('Missing transverse momentum [GeV]',np.arange(0,401,10))
    data=[];raw=[];sigmas=[];source_info=[]
    extra_views=['default-generic-poisson','default-cms-poisson']
    all_views=views+extra_views
    for task in TASKS:
        base=a.baseline/('%05d'%task)
        refs=json.loads((base/'input-reference.json').read_text())['events']
        keys=[tuple(r[k] for k in ['run','lumi','event']) for r in refs]
        weights=np.array([r['weight'] for r in refs])
        if len(keys)!=25000 or len(set(keys))!=25000 or not np.isfinite(weights).all() or (weights<0).any():
            raise ValueError('Frozen hard-event sample incomplete')
        raw.append(weights)
        record=records[str(manifest['files'][task]['record'])]
        sigmas.append(float(record['cross_section']['total_value']))
        source_info.append(dict(task=task,dataset=record['dataset'],events=25000,cross_section_pb=sigmas[-1],sum_generator_weights=float(weights.sum())))
        metadata=[json.loads(v) for v in gzip.open(a.metadata/('metadata-%05d.jsonl.gz'%task),'rt')]
        if [tuple(r['key']) for r in metadata]!=keys:
            raise ValueError('FullSim PU metadata differs')
        rows={'FullSim':json.loads((base/'fullsim.json').read_text())['events']}
        for name,variant in [('default-zero','zero'),('default-generic-poisson','generic-poisson'),('default-cms-poisson','cms-poisson')]:
            rows[name]=[r for path in sorted((a.prior_default/'variants'/variant/('%05d'%task)).glob('*/summary.json')) for r in json.loads(path.read_text())['rows']]
        rows.update({'matched-'+arm:rows_by_arm[arm][task] for arm in ['generic','cms']})
        sample={}
        for view in all_views:
            mapped={tuple(r['key']):r for r in rows[view]}
            if len(rows[view])!=25000 or len(mapped)!=25000 or set(mapped)!=set(keys):
                raise ValueError('Missing/duplicate matched events')
            ordered=[mapped[k] for k in keys]
            np.testing.assert_allclose([r['weight'] for r in ordered],weights,rtol=1e-6,atol=1e-9)
            values={k:[] for k in plot.OBS}
            for i,r in enumerate(ordered):
                selected={k:r[k] for k in ['pf','charged_pf','met']}
                selected['neutral_pf']=r['pf']-r['charged_pf'] if view=='FullSim' else r['neutral_pf']
                selected['jets']=sum(j[0]>30 and abs(j[1])<2.4 for j in r['jets']) if view=='FullSim' else r['jets']
                selected['n_pu']=metadata[i]['n'] if view=='FullSim' else r['n_pu']
                if view.startswith('matched-') and selected['n_pu']!=metadata[i]['n']:
                    raise ValueError('Recorded N differs for a hard event')
                for key in values:
                    values[key].append(selected[key])
            sample[view]={k:np.asarray(v) for k,v in values.items()}
            if not all(np.isfinite(v).all() for v in sample[view].values()):
                raise ValueError('Nonfinite comparison observable')
        data.append(sample)
    def select(names):
        plot.VIEWS=names
        naming=dict(zip(views,labels));naming.update({'default-generic-poisson':'Earlier generic PU, Poisson N,1k bank','default-cms-poisson':'Earlier CMS PU, Poisson N,50.6k bank'})
        coloring=dict(zip(views,colors));coloring.update({'default-generic-poisson':'#f89c20','default-cms-poisson':'#7a21dd'})
        plot.LABELS=[naming[v] for v in names]
        plot.COLORS=[coloring[v] for v in names]
        plot.LINESTYLES=[plot.series_style(v)[0] for v in names]
    stats={};pdf_path=out/'matched-pileup-comparison.pdf'
    conditional={}
    hep.style.use('CMS')
    with PdfPages(pdf_path) as pdf:
        select(views)
        plot.comparison(pdf,out,data,raw,sigmas,'Observed in-time PU reuse matched · default CMS response','stitched-multiplicities',stats)
        plot.comparison(pdf,out,data,raw,sigmas,'Normalized cross-section-weighted shapes','normalized-multiplicities',stats,True)
        for sample,w,sigma,label,task in zip(data,raw,sigmas,SAMPLES,TASKS):
            plot.comparison(pdf,out,[sample],[w],[sigma],'Matched reuse · QCD pThat '+label+' GeV','sample-%05d'%task,stats,True)
        for arm in ['generic','cms']:
            select(['FullSim','default-zero','default-'+arm+'-poisson','matched-'+arm])
            plot.comparison(pdf,out,data,raw,sigmas,'Earlier versus matched '+arm+' PU · both N profile and bank changed','earlier-versus-matched-'+arm,stats)
        # Conditional ratio estimators, including normalization fluctuations
        # within each source; paired differences share the same hard events.
        select(views)
        scaled=[w*s/w.sum() for w,s in zip(raw,sigmas)]
        fig,axes=plt.subplots(2,3,figsize=(17,9),sharex='col')
        for column,key in enumerate(['pf','charged_pf','neutral_pf']):
            entries=[]
            for lo,hi in zip(np.arange(-.5,65,5)[:-1],np.arange(-.5,65,5)[1:]):
                masks=[(d['FullSim']['n_pu']>=lo)&(d['FullSim']['n_pu']<hi) for d in data]
                denominator=sum(ws[m].sum() for ws,m in zip(scaled,masks))
                sum_w2=sum(np.sum(ws[m]**2) for ws,m in zip(scaled,masks))
                effective=denominator**2/sum_w2 if sum_w2 else 0
                if effective<10:
                    continue
                entry=dict(n_range=[float(lo),float(hi)],events=sum(int(m.sum()) for m in masks),effective_events=float(effective),
                           n_mean=float(sum(np.sum(ws[m]*d['FullSim']['n_pu'][m]) for ws,m,d in zip(scaled,masks,data))/denominator),views={})
                for view in views:
                    entry['views'][view]=conditional_estimate([d[view][key] for d in data],masks,raw,sigmas)
                    if view!='FullSim':
                        entry['views'][view]['paired_difference']=conditional_estimate([d[view][key]-d['FullSim'][key] for d in data],masks,raw,sigmas)
                entries.append(entry)
            for view,label,color in zip(views,labels,colors):
                style=plot.series_style(view)[0]
                axes[0,column].errorbar([e['n_mean'] for e in entries],[e['views'][view]['mean'] for e in entries],
                    yerr=[e['views'][view]['statistical_sem'] for e in entries],color=color,linestyle=style,marker=None,lw=1.6,label=label)
                if view!='FullSim':
                    axes[1,column].errorbar([e['n_mean'] for e in entries],[e['views'][view]['paired_difference']['mean'] for e in entries],
                        yerr=[e['views'][view]['paired_difference']['statistical_sem'] for e in entries],color=color,linestyle=style,marker=None,lw=1.6)
            axes[0,column].set_ylabel('Mean '+plot.OBS[key][0],fontsize=11)
            axes[1,column].set_ylabel('Mean difference to FullSim',fontsize=11)
            axes[1,column].set_xlabel('Recorded in-time pileup interactions',fontsize=11)
            axes[1,column].axhline(0,color='black',lw=.8)
            for ax in axes[:,column]:ax.grid(alpha=.2);ax.tick_params(labelsize=10)
            conditional[key]=entries
        fig.legend([plot.series_handle(v,l,c) for v,l,c in zip(views,labels,colors)],labels,loc='upper center',bbox_to_anchor=(.5,.95),ncol=2,handlelength=5,fontsize=10)
        fig.suptitle('Same100k hard events · exact in-time counts and identity reuse',fontsize=15)
        fig.subplots_adjust(top=.83,bottom=.16,hspace=.3,wspace=.3)
        fig.text(.5,.07,'LO cross-section-weighted conditional means; paired, source-stratified statistical errors; effective N<10 bins omitted.',ha='center',fontsize=10)
        fig.text(.5,.035,'Native PF definitions differ. Gaussian vertices/time; no OOT detector response. Finite-library correlations and model errors excluded.',ha='center',fontsize=9)
        pdf.savefig(fig);fig.savefig(out/'multiplicity-vs-recorded-pileup.png',dpi=140);fig.savefig(out/'multiplicity-vs-recorded-pileup.pdf');plt.close(fig)
    metrics={};differences={}
    sigma_sum=sum(sigmas)
    for view in all_views:
        metrics[view]={}
        for key in plot.OBS:
            mean=sum(s*np.average(d[view][key],weights=w) for d,w,s in zip(data,raw,sigmas))/sigma_sum
            variance=sum(s*s*len(w)/(len(w)-1)*np.sum((w/w.sum())**2*(d[view][key]-np.average(d[view][key],weights=w))**2) for d,w,s in zip(data,raw,sigmas))/sigma_sum**2
            metrics[view][key]=dict(cross_section_weighted_mean=float(mean),statistical_sem=float(np.sqrt(variance)))
            if view!='FullSim':
                delta=[d[view][key]-d['FullSim'][key] for d in data]
                difference=sum(s*np.average(v,weights=w) for v,w,s in zip(delta,raw,sigmas))/sigma_sum
                var=sum(s*s*len(w)/(len(w)-1)*np.sum((w/w.sum())**2*(v-np.average(v,weights=w))**2) for v,w,s in zip(delta,raw,sigmas))/sigma_sum**2
                differences.setdefault(view,{})[key]=dict(mean_difference_to_fullsim=float(difference),paired_statistical_sem=float(np.sqrt(var)))
    for name,row in stats.items():
        total=sigmas[TASKS.index(int(name.split('/')[0].removeprefix('sample-')))] if name.startswith('sample-') else sigma_sum
        if any(not np.isclose(sum(h),total,rtol=1e-10) for h in row['cross_section_pb'].values()):
            raise ValueError('Flow-inclusive cross-section normalization failed')
    result=dict(status='completed',new_native_simulated_events=200000,events_per_view=100000,
                pdf=str(pdf_path),pages=9,occupancy=occupancy,normalization=source_info,
                formula='sigma_sample*gen_weight/sum_25000_validation_weights before selections, explicit flows',
                metrics=metrics,paired_differences=differences,
                styles={v:dict(color=c,linestyle=plot.series_style(v)[0],marker=None) for v,c in zip(views,colors)},
                statistical_method='Paired delta method stratified by QCD source, catalog cross sections fixed',
                limits=['One25k prefix/source; campaign representativeness unestablished',
                        'Hard inputs reuse exported MiniAOD packedGenParticles with inherited quantization/coverage; the complete original hard GEN record is not recovered',
                        'Exact in-time N and identity-sharing graph, statistically replaced particles; no original particle/vertex replay',
                        'New generic events use pinned example Tune:pp5; old bundled generic generator settings incompletely known',
                        'Selected CMS GEN files meet size/event-count staging filters; bank is not a uniform draw over the entire500M pool',
                        'FullSim PF and native Delphes track/residual-tower definitions differ; neutral includes photons',
                        'FullSim AK4 CHS jets versus default Delphes R0.5; default merger adds no CHS/rho subtraction',
                        'Approximate Gaussian beam spot/time, CMS GEN modal xyz/ct0 convention; no OOT detector premixing',
                        'Statistical errors exclude finite-library correlations and detector/generator/model uncertainties',
                        'Earlier-vs-new panels change both N distribution and library recipe/size/reuse; they do not isolate a library-size effect',
                        'Matching occupancy does not validate ML convergence or remove original shared-PU correlations across splits'],
                plot_tool_sha256=digest(Path(__file__)))
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'histograms.json').write_text(json.dumps(stats,indent=2)+'\n')
    (out/'multiplicity-vs-recorded-pileup.json').write_text(json.dumps(conditional,indent=2)+'\n')
    with (out/'mean-multiplicities.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['view','observable','cross_section_weighted_mean','statistical_sem'])
        for view,values in metrics.items():
            for key,value in values.items():
                writer.writerow([view,key,value['cross_section_weighted_mean'],value['statistical_sem']])
    (out/'METHODOLOGY.txt').write_text(__doc__+'\n\n'+json.dumps(result,indent=2)+'\n')
    (out/'SHA256SUMS').write_text(''.join(digest(p)+'  '+p.name+'\n' for p in sorted(out.iterdir()) if p.is_file() and p.name!='SHA256SUMS'))
    print(json.dumps(dict(status='completed',pdf=str(pdf_path),metrics=metrics)))


if __name__=='__main__':
    main()
