"""Descriptive8k/64k comparison; each FM stage has its own mainfields."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..io_utils import file_hash,new_dir,write_json

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def run(path):
    root=Path(path).resolve();p=read(root/'ARTIFACT_PROTOCOL.json');old=Path(p['source_run']);out=new_dir(root/'comparison_figures')
    inputs=[];fig,axes=plt.subplots(2,2,figsize=(11.5,8),layout='constrained');colors=['#8695a5','#165b91'];report=[]
    paths={'8k':old/'metrics','64k':root/'artifacts/metrics'}
    for i,seed in enumerate([10,11]):
        x=np.arange(3)
        for j,(stage,folder) in enumerate(paths.items()):
            path=folder/'distribution'/f'seed{seed}'/'MATRIX.json';inputs.append(path);rows=read(path)
            values=[next(r for r in rows if r['row']=='ours_'+arm) for arm in ['fixed','a','qem']]
            if not all(r['attempts']==64 and r['metrics_readable']==64 and r['valid_metric_groups']==4 for r in values):raise ValueError('Incomplete fixed distribution blocks')
            axes[0,i].bar(x+(j-.5)*.34,[r['MMD_CD'] for r in values],.34,color=colors[j],label=stage)
            report.append({'seed':seed,'stage':stage,'rows':values})
        axes[0,i].set(xticks=x,xticklabels=['Fixed','Original A','VTK QEM'],ylabel='MMD squared Chamfer',title=f'Seed {seed}: all four fixed16-sample blocks')
        axes[0,i].ticklabel_format(axis='y',style='sci',scilimits=(-3,-3));axes[0,i].legend();axes[0,i].grid(axis='y',alpha=.2)
    for i,arm in enumerate(['a','qem']):
        ax=axes[1,i]
        for j,(stage,folder) in enumerate(paths.items()):
            path=folder/'RESULTS.json';inputs.append(path);rows=[r for r in read(path)['rows'] if r['arm']==arm]
            if len(rows)!=128 or not all(r['metric_status']=='ok' for r in rows):raise ValueError('Incomplete128 same-field metric slots')
            gains=np.asarray([r['rms_gain_percent_two_measurements'] for r in rows])
            for mc in range(2):ax.plot(np.sort(gains[:,mc]),np.arange(1,129)/128,color=colors[j],ls='-' if mc==0 else '--',label=stage+f' MC{mc+1}')
        ax.axvline(0,c='k',lw=.8);ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set(xlabel='Paired RMS improvement vs own-field Fixed (%)',ylabel='Fraction of128 samples (two seeds pooled)',title=('Original A' if arm=='a' else 'VTK QEM')+': all samples, including invalid')
    fig.suptitle('Fixed FM continuation: descriptive distribution and outlet comparisons\n8k and64k use identical primary noise and AE; their generated mainfields differ',fontsize=12)
    fig.supxlabel('Each outlet gain is paired only within its own mainfield. All16 parents are training/development data.\nFour blocks are not independent training runs. Invalid outputs remain in metrics; geometry validity is reported separately.',fontsize=8.5)
    outputs=[]
    for ext in ['png','pdf']:
        path=out/f'fixed_8k_64k_comparison.{ext}';fig.savefig(path,dpi=125);outputs.append({'file':path.name,'sha256':file_hash(path)})
    plt.close(fig)
    unique=list(dict.fromkeys(inputs));write_json(out/'MANIFEST.json',{'source_sha256':file_hash(__file__),'inputs':[{'path':str(p),'sha256':file_hash(p)} for p in unique],'outputs':outputs,'distribution_rows':report,'visual_QA_pending':True,'no_significance_or_generalization_claim':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run(a.run)