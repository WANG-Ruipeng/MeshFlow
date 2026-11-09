from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..io_utils import new_dir,file_hash,write_json

def main(run):
    run=Path(run);source=run/'metrics/RESULTS.json';data=json.loads(source.read_text());rows=data['rows']
    assert data['expected_attempts']==384 and len(rows)==384 and all(r['metric_status']=='ok' for r in rows)
    out=new_dir(run/'metric_figures');fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    colors={'a':'#2676b7','qem':'#23844f'};labels={'a':'Original A','qem':'VTK QEM'}
    records={}
    for arm in ('a','qem'):
        group=[r for r in rows if r['arm']==arm];assert len(group)==128
        gains=np.array([r['rms_gain_percent_two_measurements'] for r in group]);margins=np.array([r['p05_angle_margin_deg'] for r in group]);valid=np.array([r['audit']['valid'] for r in group])
        axes[0].scatter(gains[valid,0],margins[valid],s=22,alpha=.55,color=colors[arm],label=labels[arm])
        if (~valid).any():axes[0].scatter(gains[~valid,0],margins[~valid],s=45,marker='x',color=colors[arm],label=labels[arm]+' invalid')
        for j in range(2):axes[1].plot(np.sort(gains[:,j]),np.arange(1,129)/128,color=colors[arm],linestyle='-' if j==0 else '--',label=labels[arm]+f' MC{j+1}')
        records[arm]={'gain_mean':gains.mean(0).tolist(),'gain_median':np.median(gains,axis=0).tolist(),
            'valid':int(valid.sum()),'qualified':sum(r['qualified'] for r in group),'minimum_gain':gains.min(0).tolist()}
    axes[0].axhline(0,color='black',linewidth=.8);axes[0].axvline(0,color='black',linewidth=.8)
    axes[0].scatter([0],[0],marker='s',s=40,color='#b88020',label='Fixed reference')
    axes[0].set(xlabel='Paired RMS improvement vs Fixed (%) — MC1',ylabel='p05 minimum-angle change vs Fixed (degrees)',title='Fidelity gains accompany angular-quality loss')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.15)
    axes[1].axvline(0,color='black',linewidth=.8);axes[1].set(xlabel='Paired RMS improvement vs Fixed (%)',ylabel='Empirical fraction of 128 saved fields',title='Both MC measurements; negative gains retained')
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.15)
    fig.suptitle('Closed-set16 AE/FM: same 128 mainfields, two training seeds pooled',fontsize=12)
    fig.text(.5,.005,'No outlet repair or sample selection. This is not held-out generalization or an external-baseline ranking.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95))
    for ext in ('png','pdf'):fig.savefig(out/f'paired_tradeoff.{ext}',dpi=160)
    plt.close(fig)
    write_json(out/'FIGURES.json',{'source_metrics_sha256':file_hash(source),'source_script_sha256':file_hash(__file__),
        'records':records,'files':{f'paired_tradeoff.{x}':file_hash(out/f'paired_tradeoff.{x}') for x in ['png','pdf']}})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();main(a.run)