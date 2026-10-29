"""Render published pilot histograms in monochrome for visual style review.

This reuses the released bin arrays and uncertainty arrays; no events are
reprocessed and no image is edited. The original color figures remain intact.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from plot_pileup_pilot import digest,shown_edges


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plots',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    summary=json.loads((a.plots/'summary.json').read_text())
    data=json.loads((a.plots/'histograms.json').read_text())
    views=['FullSim','default-zero','matched-generic','matched-cms']
    names=['CMS FullSim','Default CMS + merger, zero PU','Generic PU, matched reuse and N','CMS MinBias PU, matched reuse and N']
    labels={'pf':'PF candidates','charged_pf':'Charged PF candidates','neutral_pf':'Neutral PF candidates (including photons)',
            'jets':'Jets, pT > 30 GeV','n_pu':'In-time pileup interactions','met':'Missing transverse momentum [GeV]'}
    styles={}
    for view in views:
        style=summary['styles'][view]
        if style['marker'] is not None:
            raise ValueError('Published curve has unwanted markers')
        styles[view]=tuple(style['linestyle']) if isinstance(style['linestyle'],list) else style['linestyle']
    fig=plt.figure(figsize=(18,12))
    outer=fig.add_gridspec(2,3,top=.86,bottom=.12,hspace=.38,wspace=.25)
    for cell,key in zip(outer,labels):
        row=data['stitched-multiplicities/'+key]
        shown=shown_edges(np.asarray(row['regular_edges']))
        inner=cell.subgridspec(2,1,height_ratios=[3,1],hspace=.04)
        ax=fig.add_subplot(inner[0]);ratio_ax=fig.add_subplot(inner[1],sharex=ax)
        reference=np.asarray(row['cross_section_pb']['FullSim'])
        for view in views:
            values=np.asarray(row['cross_section_pb'][view])
            ax.stairs(values,shown,color='black',linestyle=styles[view],lw=1.7)
            if view=='FullSim':
                error=np.sqrt(row['statistical_variance_pb2'][view])
                ax.fill_between(shown,np.r_[values-error,(values-error)[-1]],np.r_[values+error,(values+error)[-1]],step='post',color='black',alpha=.15)
            else:
                ratio=np.divide(values,reference,out=np.full_like(values,np.nan),where=reference>0)
                variance=np.asarray(row['paired_ratio_variance'][view]);error=np.sqrt(np.where(variance>=0,variance,np.nan))
                ratio_ax.stairs(ratio,shown,color='black',linestyle=styles[view],lw=1.3)
                ratio_ax.fill_between(shown,np.r_[ratio-error,(ratio-error)[-1]],np.r_[ratio+error,(ratio+error)[-1]],step='post',color='black',alpha=.10)
        maximum=max(max(v) for v in row['cross_section_pb'].values())
        ax.set_yscale('log');ax.set_ylim(max(maximum*1e-5,1e-8),maximum*8)
        ax.set_ylabel('Cross section / bin [pb]',fontsize=10);ax.tick_params(labelbottom=False)
        ratio_ax.axhline(1,color='grey',lw=.8);ratio_ax.set_ylim(0,3)
        ratio_ax.set_ylabel('/ FullSim',fontsize=9);ratio_ax.set_xlabel(labels[key],fontsize=10)
        for axis in [ax,ratio_ax]:
            axis.set_xlim(shown[0],shown[-1]);axis.grid(alpha=.15);axis.tick_params(labelsize=9)
        if key=='n_pu':
            ax.text(.03,.97,'Matched-overlay N = FullSim event by event',transform=ax.transAxes,va='top',fontsize=8)
    handles=[Line2D([],[],color='black',linestyle=styles[v],marker=None,lw=1.7,label=n) for v,n in zip(views,names)]
    fig.legend(handles,names,loc='upper center',bbox_to_anchor=(.5,.93),ncol=2,fontsize=10,handlelength=5)
    fig.suptitle('Monochrome readability review · exact published histogram arrays',fontsize=15)
    fig.text(.5,.04,'Review rendering only; same spectra and statistical errors as the released color comparison. No markers.',ha='center',fontsize=10)
    fig.savefig(a.output/'grayscale-comparison.png',dpi=130);fig.savefig(a.output/'grayscale-comparison.pdf');plt.close(fig)
    result=dict(status='rendered-awaiting-visual-inspection',source_histograms_sha256=digest(a.plots/'histograms.json'),
                source_summary_sha256=digest(a.plots/'summary.json'),styles=summary['styles'],
                image_sha256=digest(a.output/'grayscale-comparison.png'),tool_sha256=digest(Path(__file__)))
    (a.output/'review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
