"""Read-only scientific plots for the fixed FM tail diagnostic; no sample selection."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..io_utils import file_hash,write_json,new_dir

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def run(path):
    root=Path(path).resolve();out=new_dir(root/'diagnostic_figures');inputs=[]
    pair_path=root/'artifacts/paired_8k_64k/RESULTS.json';pair=read(pair_path);inputs.append(pair_path)
    if pair['successful']!=128 or pair['planned']!=128:raise ValueError('The all128 diagnostic plot requires all fixed slots')
    fig,axes=plt.subplots(2,2,figsize=(11.5,8.4),layout='constrained')
    colors={10:'#1766a0',11:'#bd501b'}
    rows=pair['rows'];steps=[8000,16000,32000,64000]
    for seed in [10,11]:
        probes=[]
        for step in steps:
            p=root/f'training/seed_{seed}/invocation_001'/f'probe{step:08d}.json';inputs.append(p);probes.append(read(p))
        axes[0,0].plot(steps,[p['network_pair_MSE'] for p in probes],'-o',color=colors[seed],label=f'Seed {seed}: model')
        axes[0,0].axhline(probes[0]['oracle_pair_MSE'],color=colors[seed],ls='--',alpha=.55,label=f'Seed {seed}: conditional oracle')
        axes[0,1].plot(steps,[p['network_vs_conditional_mean_MSE'] for p in probes],'-o',color=colors[seed],label=f'Seed {seed}')
        local=[r for r in rows if r['seed']==seed]
        axes[1,0].scatter([r['first_nearest_training_logradius_RMS'] for r in local],[r['second_nearest_training_logradius_RMS'] for r in local],s=20,alpha=.65,color=colors[seed],label=f'Seed {seed}: all64')
    for ax in axes[0]:
        ax.set_xscale('log',base=2);ax.set_xticks(steps,[f'{s//1000}k' for s in steps]);ax.set_xlabel('Global optimizer step');ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0,0].set(title='Fixed4096 paired-velocity probe',ylabel='Velocity MSE')
    axes[0,1].set(title='Error against empirical conditional mean',ylabel='Velocity MSE')
    maxval=max(max(r['first_nearest_training_logradius_RMS'],r['second_nearest_training_logradius_RMS']) for r in rows)*1.06
    axes[1,0].plot([0,maxval],[0,maxval],'k--',lw=.8);axes[1,0].set(xlim=(0,maxval),ylim=(0,maxval),xlabel='8k: nearest training log-radius RMS',ylabel='64k: nearest training log-radius RMS',title='Same noise; different generated main fields')
    axes[1,0].set_aspect('equal');axes[1,0].legend(fontsize=8);axes[1,0].grid(alpha=.2)
    labels=[];coverage=[];x=np.arange(2)
    for side in ['first','second']:
        counts=[]
        for seed in [10,11]:
            group=next(g for g in pair['groups'] if g['seed']==seed)
            counts.append(sum(v>0 for v in group['nearest_parent_counts'][side].values()))
        coverage.append(counts)
    for idx,(name,counts,color) in enumerate(zip(['8k','64k'],coverage,['#8996a6','#273e60'])):
        bars=axes[1,1].bar(x+(idx-.5)*.32,counts,width=.32,label=name,color=color)
        axes[1,1].bar_label(bars,labels=[f'{v}/16' for v in counts],fontsize=9)
    axes[1,1].set(xticks=x,xticklabels=['Seed10','Seed11'],ylim=(0,18),ylabel='Distinct nearest training parents',title='Empirical nearest-parent coverage (64 samples)')
    axes[1,1].legend(fontsize=8);axes[1,1].grid(axis='y',alpha=.2)
    fig.suptitle('Closed16 FM continuation: fixed8000 parent to fixed64000 final\nSame AE, same primary noise, Euler32; no checkpoint or sample selection',fontsize=12)
    fig.supxlabel('Nodes and nearest-parent counts are diagnostics, not surface distance or generalization.\nAll16 objects are development data. Primary Euler32 is unchanged; Euler128 stays auxiliary.',fontsize=9)
    outputs=[]
    for ext in ['png','pdf']:
        p=out/f'fixed_tail_diagnostics.{ext}';fig.savefig(p,dpi=125);outputs.append({'path':str(p.relative_to(root)),'sha256':file_hash(p)})
    plt.close(fig)
    write_json(out/'MANIFEST.json',{'scope':'DESCRIPTIVE_ALL_FIXED_SLOTS_NO_SELECTION','source_sha256':file_hash(__file__),'inputs':[{'path':str(p.relative_to(root)),'sha256':file_hash(p)} for p in inputs],'outputs':outputs,'visual_QA_pending':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run(a.run)