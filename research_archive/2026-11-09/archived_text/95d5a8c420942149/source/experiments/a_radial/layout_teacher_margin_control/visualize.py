"""Paired original-vs-strict floor sensitivity; all four objects and fractions."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..io_utils import file_hash,new_dir,write_json


def render(run):
    run=Path(run).resolve();p=json.loads((run/'RUN_PROTOCOL.json').read_text());r=json.loads((run/'REVIEW.json').read_text())
    assert r['status']=='PASS' and r['protocol_sha256']==file_hash(run/'RUN_PROTOCOL.json')
    assert r['results_sha256']==file_hash(run/'RESULTS.json') and r['sensitivity_sha256']==file_hash(run/'SENSITIVITY.json')
    out=new_dir(run/'figures');fig,axes=plt.subplots(2,2,figsize=(12,8.2),sharex=True,sharey=True)
    fractions=p['fractions'];x=np.arange(len(fractions))
    for ax,uid in zip(axes.flat,p['uids']):
        records=[next(v for v in r['paired_sensitivity'] if v['uid']==uid and v['fraction']==f) for f in fractions]
        zero=p['parent_evidence'][uid]['zero_teacher']
        ax.plot(x,[v['control_angular_pass']/16 for v in records],marker='o',color='#316aab',label='Original Fixed floor',alpha=.5 if zero else 1)
        ax.plot(x,[v['strict_angular_pass']/16 for v in records],marker='s',color='#d66512',label='Fixed floor + 0.1 degree',alpha=.5 if zero else 1)
        ax.set_title(uid+'\nSame frozen teacher start; same 64-step budget and seeds',fontsize=9,pad=10)
        ax.set_xticks(x,['0','0.0001','0.001','0.01','0.1','0.5','1']);ax.tick_params(axis='x',labelbottom=True)
        ax.set_ylim(-.04,1.06);ax.set_yticks([0,.25,.5,.75,1]);ax.grid(alpha=.25)
        ax.set_xlabel('Fraction of original teacher displacement RMS')
        ax.set_ylabel('Both original Fixed angle gates passed / 16 trials')
        if zero:
            ax.text(.5,.45,'ZERO-ENERGY CONTROL\nAll fractions are zero perturbation.\nNo robustness conclusion.',
                transform=ax.transAxes,ha='center',va='center',fontsize=10,
                bbox={'facecolor':'#fff3cf','edgecolor':'#b18e44','boxstyle':'round,pad=.5'})
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=2,bbox_to_anchor=(.5,.915),frameon=False)
    fig.suptitle('Paired floor-policy intervention in the same local optimizer\nOnly the quality floor changes; Gaussian noise is not learned prediction error',fontsize=14,y=.985)
    fig.text(.5,.02,'All registered fractions retained. Larger margin does not dominate every noise level; perturbation checks are angular only.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.825,bottom=.105,wspace=.2,hspace=.46)
    for ext in ('png','pdf'):fig.savefig(out/f'paired_sensitivity.{ext}',dpi=180)
    plt.close(fig)
    write_json(out/'FIGURE_MANIFEST.json',{'protocol_sha256':file_hash(run/'RUN_PROTOCOL.json'),'review_sha256':file_hash(run/'REVIEW.json'),
        'source_sha256':file_hash(__file__),'files':{f.name:file_hash(f) for f in out.iterdir() if f.is_file()},
        'all_UIDs':p['uids'],'all_fractions':fractions,'zero_energy_degeneration_visible':True,
        'negative_as_well_as_positive_differences_preserved':True})


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--run',required=True);args=a.parse_args();render(args.run)
