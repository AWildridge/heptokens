"""Plot measured CMS pileup metadata; never label it as a Delphes overlay."""
import argparse,json
from pathlib import Path
from plot_fullsim_delphes_validation import plt,np,hist,hep,COLORS,display_edges,digest

def draw(ax,values,edges,label,color):
 h=hist.Hist(hist.axis.Variable(edges),storage=hist.storage.Weight())
 h.fill(values);counts=h.values(flow=True);shown=display_edges(edges)
 ax.stairs(counts/counts.sum(),shown,label=label,color=color,lw=2)
 for low,high,text in [(shown[0],edges[0],'UF'),(edges[-1],shown[-1],'OF')]:
  ax.axvspan(low,high,color='grey',alpha=.12)
  ax.text((low+high)/2,.98,text,transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=8)
 ax.set_xlim(shown[0],shown[-1]);ax.set_ylabel('Normalized entries / bin');ax.legend(fontsize=9)
 return {'edges':edges.tolist(),'counts_including_flow':counts.tolist(),'probabilities_including_flow':(counts/counts.sum()).tolist(),'entries':len(values)}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--rucio-status',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 source=json.loads(a.metadata.read_text());events=source['events'];rows=[r for e in events for r in e['pileup']];intime=[r for r in rows if r['bunch_crossing']==0]
 status=json.loads(a.rucio_status.read_text());out=a.output;out.mkdir(parents=True,exist_ok=True)
 fig,axs=plt.subplots(2,2,figsize=(14,10));stats={}
 stats['in_time_counts']=draw(axs[0,0],[r['n_interactions'] for r in intime],np.arange(-.5,80.6,5),'Recorded in-time interactions',COLORS[0])
 stats['true_mean']=draw(axs[0,0],[r['true_mean_interactions'] for r in intime],np.arange(-.5,80.6,5),'Stored Poisson mean parameter',COLORS[1])
 axs[0,0].set_xlabel('Interactions / event, or stored mean');axs[0,0].set_title('Target metadata, not a fitted pileup profile')
 stats['z_cm']=draw(axs[0,1],[z for r in intime for z in r['zpositions_cm']],np.linspace(-20,20,41),'Recorded in-time PU vertices',COLORS[0])
 axs[0,1].set_xlabel('CMS pileup vertex z [cm]');axs[0,1].set_title('Overlay target; convert cm to mm')
 crossings=sorted({r['bunch_crossing'] for r in rows});counts=[];ids=[];zs=[];ranges=[]
 for bx in crossings:
  selected=[r for r in rows if r['bunch_crossing']==bx];den=sum(r['n_interactions'] for r in selected)
  ids.append(sum(r['getPU_EventID_size'] for r in selected)/den if den else None)
  zs.append(sum(r['getPU_zpositions_size'] for r in selected)/den if den else None)
  c=[r['n_interactions'] for r in selected];counts.append(float(np.mean(c)));ranges.append([min(c),max(c)])
 axs[1,0].plot(crossings,ids,'o-',color=COLORS[0],label='Saved pileup event IDs')
 axs[1,0].plot(crossings,zs,'x--',color=COLORS[1],label='Saved vertex z values')
 axs[1,0].set_ylim(-.05,1.15);axs[1,0].set_ylabel('Saved entries / recorded interactions');axs[1,0].set_title('Exact replay metadata exists in-time only here');axs[1,0].legend(fontsize=9)
 axs[1,1].plot(crossings,counts,'o-',color=COLORS[0],label='Mean of 10 inspected events')
 axs[1,1].fill_between(crossings,[r[0] for r in ranges],[r[1] for r in ranges],color=COLORS[0],alpha=.18,label='Observed min–max (not uncertainty)')
 axs[1,1].set_ylabel('Recorded interactions / crossing');axs[1,1].set_title('Out-of-time pileup exists in CMS reference');axs[1,1].legend(fontsize=9)
 for ax in axs[1,:]:ax.set_xlabel('Bunch crossing relative to hard collision (25 ns)');ax.axvline(0,color='grey',ls=':',lw=1)
 fig.suptitle('MinBias overlay preparation: CMS input diagnostics',fontsize=16,y=.99)
 fig.tight_layout(rect=[0,.15,1,.95])
 notes=['10 original MiniAOD events from QCD source 84; unweighted metadata only, not a representative UL16 pileup distribution.',
        'UF / OF slots are explicit and included in histogram normalization. No fitted target, reweighting, or predicted improvement.',
        'Plan: verify MinBias GEN → match saved event IDs → overlay stable particles at saved z → run separate Delphes pilot.',
        'Pilot assumptions to test: ct=0 if timing absent; no CMS electronics/premixing or out-of-time response reproduced.',
        'No MinBias overlay has been simulated in this figure. Rucio snapshot: '+str(status.get('rucio_state',status['status']))+'.']
 for y,text in zip([.115,.091,.067,.043,.019],notes):fig.text(.5,y,text,ha='center',fontsize=9)
 fig.savefig(out/'pileup-input-diagnostics.pdf');fig.savefig(out/'pileup-input-diagnostics.png',dpi=150);plt.close(fig)
 summary={'status':'input-metadata-diagnostics-only','events':len(events),'source_metadata_sha256':digest(a.metadata),'rucio_snapshot':status,'minbias_overlay_simulated':False,'production_changed':False,'histogram_library':{'hist':hist.__version__,'mplhep':hep.__version__},'palette':COLORS,'histograms':stats,'crossings':crossings,'saved_id_fractions':ids,'saved_z_fractions':zs,'mean_counts':counts,'observed_min_max':ranges,'limitations':notes,'next_required':'Actual MinBias payload and event-ID validation, then a separately identified detector pilot before overlay comparison plots.'}
 (out/'pileup-validation-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'figures':1,'status':summary['status'],'events':len(events)}))
if __name__=='__main__':main()
