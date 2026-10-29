"""Cross-section stitching and colored components using the shared plot helpers."""
import csv,json
from pathlib import Path
from plot_fullsim_delphes_validation import (plt,np,hist,hep,PdfPages,TASKS,
 LABELS,SIMS,OBS,KIN,MULT,CUTS,digest,histmatrix,display_edges,plot_page,json_finite)

def sample_normalization(raw_weights,cross_section_pb):
 weights=np.asarray(raw_weights,dtype=float)
 if not np.isfinite(weights).all() or (weights<0).any() or weights.sum()<=0:
  raise ValueError('This LO stitching diagnostic requires finite nonnegative weights')
 return cross_section_pb/weights.sum(),weights*(cross_section_pb/weights.sum())

def component_page(pdf,data,weights,sim,keys,out,basename,stats,rows):
 colors=hep.styles.cms.cmap_petroff[:4]
 fig,axes=plt.subplots(3,2,figsize=(14,12))
 for ax,key in zip(axes.flat,keys):
  label,edges=OBS[key];shown=display_edges(edges);components=[];details=[]
  for task,pt_label,source,w in zip(TASKS,LABELS,data,weights):
   matrix,flow=histmatrix(source[sim][key],edges,w);values=matrix.sum(axis=0)
   h=hist.Hist(hist.axis.Variable(shown),storage=hist.storage.Double());h.values()[...]=values;components.append(h)
   details.append({'task':task,'pThat_GeV':pt_label,'bin_cross_section_pb':values.tolist(),'flow':flow})
   physical=np.r_[-np.inf,edges,np.inf]
   for i,value in enumerate(values):rows.append([sim,key,task,pt_label,'underflow' if i==0 else 'overflow' if i==len(values)-1 else 'regular',physical[i],physical[i+1],value])
  hep.histplot(components,ax=ax,stack=True,histtype='fill',alpha=.8,color=colors,label=[r'$\hat{p}_T$ '+s+' GeV' for s in LABELS],yerr=False,flow='none',edgecolor='white',linewidth=.3)
  total=sum(h.values() for h in components);ax.stairs(total,shown,color='black',lw=1.3,label='Sum')
  for low,high,text in [(shown[0],edges[0],'UF'),(edges[-1],shown[-1],'OF')]:
   ax.axvspan(low,high,color='grey',alpha=.12);ax.axvline(high if text=='UF' else low,color='grey',ls=':',lw=.8)
   ax.text((low+high)/2,.98,text,transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=8)
  ax.set_xlim(shown[0],shown[-1]);ax.set_yscale('log');ax.set_ylim(bottom=1e-4,top=max(total.max()*20,1));ax.set_xlabel(label);ax.set_ylabel('Cross section / bin [pb]');ax.grid(alpha=.15)
  if key=='jet_pt':ax.set_xscale('log')
  if key==keys[0]:ax.legend(fontsize=8,loc='upper right',bbox_to_anchor=(.92,.97))
  stats[basename+'/'+key]={'edges':edges.tolist(),'components':details,'total_bin_cross_section_pb':total.tolist()}
 fig.suptitle(sim+' · QCD sample contributions (stacked)',fontsize=16,y=.99)
 fig.text(.5,.956,CUTS,ha='center',fontsize=8.5)
 fig.subplots_adjust(top=.92,bottom=.13,hspace=.40,wspace=.24)
 for y,text in [(.083,f'LO sample cross section / sum of all validation gen weights, before analysis cuts; {len(weights[0]):,} events per sample.'),(.061,'Cross section per bin, not density; no width division. UF / OF are explicit unbounded intervals shown as end slots.'),(.039,'Only generated pThat 300–1000 GeV; prefixes assumed representative. Components have no uncertainty bands.'),(.017,'FullSim includes pileup; Delphes has none. Native detector definitions differ. No fitted smoothing or bin rescaling.')]:fig.text(.5,y,text,ha='center',fontsize=9)
 pdf.savefig(fig);fig.savefig(out/(basename+'.png'),dpi=140);plt.close(fig)

def make_stitched_plots(manifest_path,out,data,raw_weights,sources):
 manifest=json.loads(manifest_path.read_text());records={str(r['record']):r for r in manifest['records']};normalization=[];weights=[]
 for task,source,raw in zip(TASKS,sources,raw_weights):
  entry=manifest['files'][task];record=records[str(entry['record'])];xsec=record['cross_section'];sigma=float(xsec['total_value'])
  factor,scaled=sample_normalization(raw,sigma);weights.append(scaled)
  normalization.append({'task':task,'dataset':record['dataset'],'record':record['record'],'metadata_url':'https://opendata.cern.ch/record/'+record['record'],'cross_section_pb':sigma,'cross_section_numerical_uncertainty_pb':float(xsec['total_value_uncertainty']),'filter_efficiency_included_in_cross_section':float(xsec['filter_efficiency']),'validation_events_before_analysis_cuts':len(raw),'validation_sum_generator_weights':float(raw.sum()),'factor_pb_per_generator_weight':factor,'sum_scaled_event_weights_pb':float(scaled.sum()),'full_dataset_events_not_used_as_prefix_denominator':record['events']})
 overview={sim:{key:sum([d[sim][key] for d in data],[]) for key in OBS} for sim in SIMS};combined=np.concatenate(weights);groups=np.concatenate([np.full(len(w),i) for i,w in enumerate(weights)]);rng=np.random.default_rng(20261021);stats={};rows=[];component_stats={};component_rows=[]
 pdfpath=out/'qcd-cross-section-validation.pdf'
 with PdfPages(pdfpath) as pdf:
  for keys,tag in [(KIN,'kinematics'),(MULT,'multiplicities')]:plot_page(pdf,overview,combined,groups,keys,f'{len(combined):,} matched events · cross-section-weighted QCD (generated pThat 300–1000 GeV)',out,'stitched-'+tag,rows,stats,rng,absolute=True)
  for sim,tag in zip(SIMS,['fullsim-cms','delphes-cms','delphes-atlas']):
   for keys,kind in [(KIN,'kinematics'),(MULT,'multiplicities')]:component_page(pdf,data,weights,sim,keys,out,tag+'-components-'+kind,component_stats,component_rows)
 (out/'stitched-histograms-and-ratios.json').write_text(json.dumps(json_finite(stats),indent=2,allow_nan=False)+'\n')
 (out/'sample-component-histograms.json').write_text(json.dumps(json_finite(component_stats),indent=2,allow_nan=False)+'\n')
 for filename,header,content in [('stitched-histograms.csv',['sample','observable','simulation','bin_kind','low','high','bin_cross_section_pb','sumw_pb','sumw2_pb2'],rows),('sample-components.csv',['simulation','observable','task','pThat_GeV','bin_kind','low','high','bin_cross_section_pb'],component_rows)]:
  with (out/filename).open('w') as stream:writer=csv.writer(stream);writer.writerow(header);writer.writerows(content)
 summary={'status':'completed','events':sum(len(w) for w in weights),'normalization':normalization,'total_generated_interval_cross_section_pb':sum(r['cross_section_pb'] for r in normalization),'formula':'event_weight_pb = sample_catalog_cross_section_pb * generator_weight / sum_generator_weights_of_all_validation_events_in_that_sample_before_analysis_cuts','units':'pb per bin, including underflow and overflow; inclusive jet spectra count each selected jet','normalization_scope':'absolute cross section estimate; not independently unit-normalized and no luminosity assumed','sample_colors':dict(zip(LABELS,hep.styles.cms.cmap_petroff[:4])),'sources':sources,'source_manifest_sha256':digest(manifest_path),'tool_sha256':{p:digest(Path(__file__).parent/p) for p in ['plot_fullsim_delphes_validation.py','plot_qcd_components.py']},'statistics':'160 paired event bootstraps stratified by QCD sample. Sample cross-section normalization recomputed per bootstrap to keep catalog cross section fixed. 16th–84th percentiles; no cross-section or detector modelling uncertainty included. Component stacks have no bands.','assumptions':['Each selected source prefix is representative of its parent pThat sample; not established from this small diagnostic.','Catalog leading-order cross sections already include filter efficiency; no extra efficiency multiplication or K factors.','The four generated pThat bins are disjoint and cover only300–1000GeV; reconstructed jet pT can migrate outside these bounds.','No generator-pThat recut or reconstructed-pT cut is used to assign sample ownership.','No original pileup replay in Delphes; detector definitions and radii differ as documented in the existing validation README.','Physical normalization does not remove finite statistics, detector migration, or missing lower/higher pThat samples; no smoothing applied.'],'pdf':str(pdfpath),'pages':8,'hist_version':hist.__version__,'mplhep_version':hep.__version__}
 (out/'normalization-and-provenance.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'pdf':str(pdfpath),'pages':8,'total_cross_section_pb':summary['total_generated_interval_cross_section_pb']}))
