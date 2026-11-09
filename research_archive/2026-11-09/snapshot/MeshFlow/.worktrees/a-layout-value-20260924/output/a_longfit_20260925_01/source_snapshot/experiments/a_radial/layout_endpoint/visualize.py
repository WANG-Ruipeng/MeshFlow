"""Plots of every predeclared endpoint-study object; no checkpoint selection."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ..io_utils import file_hash,write_json

COLORS={'v1':'#64748b','c0':'#d97706','c1':'#0284c7','c2':'#059669'}


def plot(run):
    run=Path(run);out=run/'figures';out.mkdir(exist_ok=False)
    protocol=json.loads((run/'RUN_PROTOCOL.json').read_text())
    ledger=json.loads((run/'results.json').read_text());uids=sorted(protocol['uids'])
    rows={(r['uid'],r['model'],r['policy']):r for r in ledger['rows']}
    def save(fig,name,note):
        fig.text(.025,.012,note,fontsize=8,va='bottom')
        for ext in ('png','pdf'):fig.savefig(out/(name+'.'+ext),dpi=165,bbox_inches='tight')
        plt.close(fig)
    note='16 expected previously seen development parents per arm (12 train / 4 internal check); no held-out generalization claim.'
    missing_counts={f'{a}_{s}':sum((u,f'{a}_{s}','raw') not in rows for u in uids) for a in ['v1','c0','c1','c2'] for s in [10,11]}
    missing_note='Missing raw objects: '+', '.join(f'{name}={count}' for name,count in missing_counts.items())
    fig,axes=plt.subplots(2,3,figsize=(14,8),squeeze=False)
    for i,seed in enumerate([10,11]):
        for j,arm in enumerate(['c0','c1','c2']):
            ax=axes[i,j];path=run/'training'/arm/f'seed_{seed}'/'TRAIN_TRACE.json'
            if not path.exists():ax.text(.5,.5,'MISSING',transform=ax.transAxes,ha='center');continue
            trace=json.loads(path.read_text());x=np.array([r['update'] for r in trace])
            for key,label,color in [('loss','total',COLORS[arm])]+([('endpoint_loss','endpoint','#7c3aed'),('quality_penalty','quality','#dc2626')] if arm=='c2' else []):
                y=np.array([r[key] for r in trace],float)
                ax.plot(x,np.maximum(y,1e-14),color=color,alpha=.18,lw=.5)
                window=min(50,len(y));sm=np.convolve(y,np.ones(window)/window,mode='valid')
                ax.plot(x[window-1:],np.maximum(sm,1e-14),label=label+' (50-update mean)',color=color,lw=1.3)
            ax.set_yscale('log');ax.set_xlabel('Optimizer updates');ax.set_ylabel('Training objective')
            ax.set_title(f'{arm.upper()} | seed {seed}');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Complete fixed-budget training traces',fontsize=15)
    fig.subplots_adjust(left=.07,right=.98,top=.91,bottom=.12,hspace=.42,wspace=.3)
    save(fig,'training',note+'\nC0 velocity MSE has a different scale/target from C1/C2; objective values across arms are not performance rankings.')
    fig,axes=plt.subplots(1,2,figsize=(14,9),squeeze=False)
    for i,seed in enumerate([10,11]):
        ax=axes[0,i];rgb=np.ones((16,4,3))
        for j,arm in enumerate(['v1','c0','c1','c2']):
            for k,uid in enumerate(uids):
                row=rows.get((uid,f'{arm}_{seed}','raw'));base=rows.get((uid,'fixed','raw'))
                if not row or not base:
                    text='MISSING';rgb[k,j]=[.94,.7,.7]
                else:
                    gains=[100*(1-a['rms_bbox_percent']/b['rms_bbox_percent']) for a,b in zip(row['metrics'],base['metrics'])]
                    qualified=row['qualified'];strong=qualified and min(gains)>=2
                    rgb[k,j]=[.68,.87,.76] if strong else ([.98,.89,.66] if qualified else [.88,.89,.91])
                    text=f'{gains[0]:+.2f}%\n'+('qualified' if qualified else 'unqualified')
                ax.text(j,k,text,ha='center',va='center',fontsize=7.5)
        ax.imshow(rgb,aspect='auto');ax.set_xticks(range(4),['V1 initial','C0 velocity','C1 endpoint','C2 + quality'])
        labels=[u[:8]+('  [check]' if u in protocol['internal_check_uids'] else '  [train]') for u in uids]
        ax.set_yticks(range(16),labels,fontsize=8);ax.set_title(f'Seed {seed}: raw RMS gain vs Fixed')
        ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,16,1),minor=True)
        ax.grid(which='minor',color='white',lw=2);ax.tick_params(which='minor',length=0)
    fig.subplots_adjust(left=.12,right=.985,top=.92,bottom=.12,wspace=.4)
    save(fig,'all_parent_outcomes',note+'\nText: first new MC evaluation gain. Green: qualified + >=2% gain in BOTH evaluations; amber: qualified but below this joint target.\nGray retains every unqualified raw output. Quality fallback is separate and is not counted as learned improvement.')
    fig,axes=plt.subplots(1,2,figsize=(13,6),squeeze=False)
    for i,seed in enumerate([10,11]):
        ax=axes[0,i]
        for arm in ['v1','c0','c1','c2']:
            for uid in uids:
                row=rows.get((uid,f'{arm}_{seed}','raw'));base=rows.get((uid,'fixed','raw'))
                if not row or not base:continue
                gain=100*(1-row['metrics'][0]['rms_bbox_percent']/base['metrics'][0]['rms_bbox_percent'])
                margin=min(row['quality'][q]-row['quality_floor'][q] for q in ['min_triangle_angle_deg','triangle_min_angle_p05_deg'])
                ax.scatter(gain,margin,marker='s' if uid in protocol['internal_check_uids'] else 'o',s=35,
                    edgecolors=COLORS[arm],facecolors=COLORS[arm] if row['qualified'] else 'none',alpha=.8)
        ax.axhline(-1e-6,color='black',ls='--',lw=.8);ax.axvline(0,color='black',lw=.8)
        ax.set_xlabel('Paired raw RMS improvement vs Fixed (%)');ax.set_ylabel('Smaller of min-angle / p05 margins (degrees)')
        ax.set_title(f'Seed {seed}');ax.grid(alpha=.18)
    handles=[Line2D([],[],marker='o',color=c,ls='',label=a.upper()) for a,c in COLORS.items()]
    fig.legend(handles=handles,loc='upper center',ncol=4,bbox_to_anchor=(.5,.97))
    fig.subplots_adjust(left=.08,right=.98,top=.85,bottom=.2,wspace=.25)
    save(fig,'error_quality_tradeoff',note+'\n'+missing_note+'\nCircle=train, square=internal check; hollow=unqualified. Filled points still need paired RMS gain to represent useful improvement.')
    artifacts={p.name:file_hash(p) for p in sorted(out.iterdir()) if p.is_file()}
    write_json(out/'FIGURE_MANIFEST.json',{'files':artifacts,'result_rows':len(ledger['rows']),
        'expected_rows':288,'result_sha256':file_hash(run/'results.json'),'source_sha256':file_hash(__file__),
        'all_parent_ids_shown':uids,'no_model_or_mesh_modified':True})
    print(json.dumps({'figures':len(artifacts),'directory':str(out)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();plot(a.run)