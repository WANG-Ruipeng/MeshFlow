"""Immutable scientific plots from stored training/final-analysis tables only.

No model/checkpoint loads, geometry rescoring, blind-review files, or mappings.
Every invocation writes a distinct artifact directory, including final versions.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from collections import Counter
import csv
import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from .bootstrap import bind,OUT
D,E=bind()

VERSION='window_policy_rl_scientific_plots_v1'
LABELS=('FULL_DET','FULL_STOCH','RL_STOCH')
SHORT=('FULL DET','FULL STOCH','RL STOCH')
COLORS=('#758497','#377ca8','#bf6538')
STRICT='joint_both_adoptions_body_and_CF'
TRAIN_NOTE='Fresh noise and changing C/policy: online accounting, NOT a fixed-panel improvement curve.'
FINAL_NOTE='Model-blind AI_PROXY, not human certification. UNKNOWN retained; TRAIN and DEV are separate descriptive panels.'


class Sources:
    def __init__(self):self.files={};self.cache={}
    def blob(self,path):
        path=D.resolve(path);key=str(path)
        if key not in self.cache:
            data=path.read_bytes();self.cache[key]=data
            self.files[key]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
        return self.cache[key]
    def read(self,path):return json.loads(self.blob(path).decode('utf-8-sig'))
    def csv(self,path):return list(csv.DictReader(io.StringIO(self.blob(path).decode('utf-8-sig'))))


def number(value):
    if value is None or value=='':return None
    result=float(value)
    return result if math.isfinite(result) else None


def truth(value):
    if value in (True,'True','true','1'):return True
    if value in (False,'False','false','0'):return False
    if value is None or value in ('','UNKNOWN'):return None
    raise ValueError('Unexpected table boolean: '+repr(value))


def plotting():
    if 'torch' in sys.modules:raise RuntimeError('Plotting must remain model-free')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
        'axes.labelsize':9,'svg.hashsalt':VERSION,'axes.spines.top':False,'axes.spines.right':False})
    return plt


def save(kind,status,figure,data,sources,plt=None):
    sources.blob(__file__)
    now=datetime.datetime.now(datetime.timezone.utc)
    basis=dict(version=VERSION,kind=kind,status=status,data=data,source_files=sources.files)
    payload=json.dumps(basis,sort_keys=True,allow_nan=False).encode()
    digest=hashlib.sha256(payload).hexdigest()
    dest=OUT/'figures/scientific'/f'{kind}_{status.lower()}_{now.strftime("%Y%m%dT%H%M%S%fZ")}_{digest[:12]}'
    dest.mkdir(parents=True,exist_ok=False)
    files={}
    if figure is not None:
        for suffix in ('png','svg'):
            path=dest/(kind+'.'+suffix)
            options=dict(dpi=180,bbox_inches='tight',facecolor='white')
            if suffix=='svg':options['metadata']={'Date':None,'Creator':VERSION}
            figure.savefig(path,**options)
            files[str(path)]=dict(sha256=D.sha(path),bytes=path.stat().st_size)
        plt.close(figure)
    result=dict(basis,created_utc=now.isoformat(),input_digest=digest,artifacts=files,
        model_calls=0,checkpoint_loads=0,geometry_rescores=0,torch_imported='torch' in sys.modules,
        immutable_new_version=True)
    D.write(dest/'plot_data.json',result)
    return dict(status=status,kind=kind,output_dir=str(dest),data_path=str(dest/'plot_data.json'),
        data_sha256=D.sha(dest/'plot_data.json'),artifacts=files)


def training(pointer_path=None):
    sources=Sources();path=D.resolve(pointer_path or OUT/'analysis/training_latest.json')
    pointer=sources.read(path);summary_path=D.resolve(pointer['summary_path']);summary=sources.read(summary_path)
    if sources.files[str(summary_path)]['sha256']!=pointer['summary_sha256']:
        raise ValueError('Training summary pointer hash differs')
    rows=sources.csv(summary_path.parent/'training_per_batch.csv')
    if len(rows)!=64 or sorted(int(r['batch_index']) for r in rows)!=list(range(64)):
        raise ValueError('Training plot requires all64 registered rows, including pending rows')
    rows.sort(key=lambda r:int(r['batch_index']))
    complete=[r for r in rows if truth(r['logical_batch_complete'])]
    if len(complete)!=summary['completed_logical_opportunities']:
        raise ValueError('Training CSV/summary completion count differs')
    if sum(int(r['committed_optimizer_updates']) for r in complete)!=summary['committed_optimizer_updates']:
        raise ValueError('Training CSV/summary optimizer count differs')
    status=summary['status']
    if status=='COMPLETE' and len(complete)!=64:raise ValueError('Incomplete training cannot be plotted as final')
    cids=sorted({r['condition_id'] for r in rows});plt=plotting()
    palette=plt.get_cmap('tab10');cmap={cid:palette(i) for i,cid in enumerate(cids)}
    fig,axes=plt.subplots(3,2,figsize=(13.8,9.3))
    fig.subplots_adjust(top=.88,bottom=.16,hspace=.48,wspace=.24)
    observed=[r for r in rows if r['status']!='NOT_STARTED']
    xmax=64 if status=='COMPLETE' else min(64,max([int(r['batch_index'])+1 for r in observed] or [1])+1)
    for ax in axes.flat:
        ax.set_xlim(.4,xmax+.6);ax.set_xlabel('Registered batch opportunity (1-based)')
        ax.grid(axis='y',alpha=.20,linewidth=.6)
    ax=axes[0,0]
    for r in rows:
        y=number(r['reward_1'])
        if y is None:continue
        x=int(r['batch_index'])+1;color=cmap[r['condition_id']]
        ax.bar(x,y,color=color,width=.7,alpha=1 if truth(r['logical_batch_complete']) else .35)
        if y==0:ax.plot(x,0,marker='_',markersize=9,color=color)
    ax.set_ylim(-.35,8.7);ax.set_yticks(range(0,9,2));ax.set_ylabel('Binary reward sum / 8')
    ax.set_title('Reward coverage (all available completed score batches)')
    ax=axes[0,1]
    for r in observed:
        done=truth(r['logical_batch_complete']);x=int(r['batch_index'])+1
        y=int(r['committed_optimizer_updates']) if done else .5
        ax.scatter(x,y,c=[cmap[r['condition_id']]],marker='o' if done else 'x',s=38)
    ax.set_ylim(-.35,1.35);ax.set_yticks([0,.5,1],['Inactive: skipped','In progress','Active: one step'])
    ax.set_title('Committed optimizer action (no redraw / makeup)')
    metrics=(('gradient_norm_before_clip','Preclip gradient norm','L2 norm',axes[1,0]),
             ('adapter_delta_l2','Successive adapter change','Parameter L2 displacement',axes[1,1]),
             ('postupdate_trajectory_KL_mean','Old-state conditional KL','Nats: full d + 20 steps, mean over 8',axes[2,0]),
             ('preupdate_old_logp_max_abs_error','Stored vs recomputed old log probability','Maximum absolute error (nats)',axes[2,1]))
    for key,title,ylabel,ax in metrics:
        for r in rows:
            y=number(r[key])
            if y is None:continue
            done=truth(r['logical_batch_complete'])
            ax.scatter(int(r['batch_index'])+1,y,c=[cmap[r['condition_id']]],s=28,
                marker='o' if done else 'x',alpha=1 if done else .45)
        ax.set_title(title);ax.set_ylabel(ylabel)
        ax.axhline(0,color='#555555',linewidth=.6)
        if key=='gradient_norm_before_clip':
            ax.axhline(1,color='#777777',ls=':',lw=.8);ax.set_title(title+' (dotted line: clip threshold = 1)')
        if key=='postupdate_trajectory_KL_mean':
            ax.set_title(title+'\nDiagnostic on old rollout states; not terminal-mesh KL')
        if key=='preupdate_old_logp_max_abs_error':
            errors=[number(r[key]) for r in rows if number(r[key]) is not None]
            if errors and max(errors)==0:
                ax.set_ylim(-.05,.5);ax.set_yticks([0]);ax.text(.02,.95,'All recorded errors = 0',transform=ax.transAxes,va='top',fontsize=8)
    handles=[plt.Line2D([0],[0],marker='o',ls='',color=cmap[c],label=f'C{i+1}: {c.split("__")[0][:8]}') for i,c in enumerate(cids)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.065),ncol=4,fontsize=8,frameon=False)
    fig.suptitle(f'Window-policy RL training ledger | {status}\n'
        f'{len(complete)}/64 completed opportunities; {summary["committed_optimizer_updates"]} optimizer steps; '
        f'{summary["inactive_completed_batches"]} inactive skips',fontsize=14,y=.985)
    fig.text(.5,.035,TRAIN_NOTE,ha='center',fontsize=10,color='#8a3827')
    fig.text(.5,.012,'Points are unconnected; colors identify C. Faint / x marks are uncommitted. '
        'Blank future batches are not zero observations.',ha='center',fontsize=8)
    data=dict(training_summary_status=status,completed_logical_opportunities=len(complete),
        committed_optimizer_updates=summary['committed_optimizer_updates'],rows=rows,
        condition_colors=[dict(condition_id=c,display=f'C{i+1}',color=list(cmap[c])) for i,c in enumerate(cids)],
        plotted_metric_fields=[m[0] for m in metrics],actual_physical_counts=summary['actual_physical_counts'],
        interpretation=[TRAIN_NOTE,'Adapter delta is successive-update L2, not cumulative displacement.',
            'Conditional KL sums all free coordinates and all20 transitions, then averages eight trajectories.',
            'Completed logical work is distinct from physical attempts; full ledger CSV retains both.'])
    return save('training_ledger',status,fig,data,sources,plt)


def final_tables(sources):
    dest=OUT/'evaluation/analysis'
    paths={name:dest/name for name in ('summary.json','per_model_split_stage.csv',
        'review_condition_and_split.csv','paired_request_stage.csv','paired_condition_and_split_stage.csv')}
    missing=[str(p) for p in paths.values() if not p.exists()]
    if missing:return None,dict(status='PENDING_FINAL_ANALYSIS',missing=missing)
    summary=sources.read(paths['summary.json'])
    if summary['status']!='COMPLETE' or summary['outputs']!=132 or summary['stage_scores']!=264 or summary['review_rows']!=132:
        raise ValueError('Final figure requires complete numerical and actual-review analysis')
    tables={name:sources.csv(path) for name,path in paths.items() if path.suffix=='.csv'}
    panels=[]
    for split,den in (('TRAIN',32),('DEV',12)):
        counts=[]
        for label in LABELS:
            a=[r for r in tables['per_model_split_stage.csv'] if r['label']==label and r['split']==split and r['stage']=='POST']
            b=[r for r in tables['review_condition_and_split.csv'] if r['label']==label and r['scope']=='SPLIT' and r['group']==split]
            if len(a)!=1 or len(b)!=1 or int(a[0]['outputs'])!=den or int(b[0]['outputs'])!=den:
                raise ValueError('Final count panel missing/duplicated')
            event=json.loads(b[0][STRICT])
            if event['denominator']!=den or sum(event[k] for k in ('YES','NO','UNKNOWN'))!=den:
                raise ValueError('Strict joint UNKNOWN coverage differs')
            counts.append(dict(label=label,denominator=den,legal_CF_outputs=int(a[0]['legal_CF_outputs']),strict_joint=event))
        paired=[r for r in tables['paired_request_stage.csv'] if r['comparison_role']=='PRIMARY' and r['stage']=='POST' and r['split']==split]
        if len(paired)!=den or len({r['request_id'] for r in paired})!=den:
            raise ValueError('Primary paired cost coverage differs')
        if any(r['candidate_label']!='RL_STOCH' or r['baseline_label']!='FULL_STOCH' for r in paired):
            raise ValueError('Primary paired labels differ')
        aggregates=[r for r in tables['paired_condition_and_split_stage.csv'] if r['comparison_role']=='PRIMARY' and r['stage']=='POST' and r['scope']=='SPLIT' and r['group']==split]
        if len(aggregates)!=1:raise ValueError('Primary paired aggregate missing/duplicated')
        costs=[]
        for key,name in (('CF_illegal_fraction','CF'),('FF_illegal_fraction','FF'),('I_lower','I point/lower'),('I_upper','I upper')):
            values=[dict(request_id=r['request_id'],condition_id=r['condition_id'],delta_pp=number(r[key+'_delta_pp'])) for r in sorted(paired,key=lambda r:r['request_id'])]
            finite=[v['delta_pp'] for v in values if v['delta_pp'] is not None]
            mean=number(aggregates[0][key+'_mean_delta_pp'])
            if (mean is None)!=(not finite) or (finite and not math.isclose(sum(finite)/len(finite),mean,rel_tol=1e-9,abs_tol=1e-9)):
                raise ValueError('Original paired aggregate differs from stored row deltas')
            costs.append(dict(metric=key,display=name,mean_delta_pp=mean,common_known_pairs=len(finite),
                missing_pairs=den-len(finite),values=values))
        panels.append(dict(split=split,denominator=den,conditions=8 if split=='TRAIN' else 3,counts=counts,costs=costs))
    return dict(status='COMPLETE',panels=panels,original_summary_interpretation={
        k:summary[k] for k in ('primary_contrast','normalization','semantics','primary_joint','stricter_joint','limitations')},
        source_tables=tables),None


def final():
    sources=Sources();data,pending=final_tables(sources)
    if pending:return save('final_comparison',pending['status'],None,pending,sources)
    plt=plotting();fig,axes=plt.subplots(2,3,figsize=(15,8))
    fig.subplots_adjust(top=.86,bottom=.18,hspace=.42,wspace=.34)
    for index,panel in enumerate(data['panels']):
        den=panel['denominator'];split=panel['split'];counts=panel['counts']
        ax=axes[index,0]
        bars=ax.bar(range(3),[r['legal_CF_outputs'] for r in counts],color=COLORS,width=.65)
        for b,r in zip(bars,counts):ax.text(b.get_x()+b.get_width()/2,b.get_height()+den*.025,str(r['legal_CF_outputs']),ha='center')
        ax.set_xticks(range(3),SHORT);ax.set_ylim(0,den*1.2);ax.set_ylabel(f'{split}: outputs / {den}')
        ax.set_title('POST legal C/F interface present')
        ax=axes[index,1]
        yes=[r['strict_joint']['YES'] for r in counts];unknown=[r['strict_joint']['UNKNOWN'] for r in counts]
        ax.bar(range(3),yes,color=COLORS,width=.65,label='Confirmed YES')
        ax.bar(range(3),unknown,bottom=yes,color='none',edgecolor='#555555',hatch='////',width=.65,label='UNKNOWN upper extension')
        for j,r in enumerate(counts):
            v=r['strict_joint'];ax.text(j,v['YES']+v['UNKNOWN']+den*.025,f"Y{v['YES']} / U{v['UNKNOWN']}",ha='center',fontsize=8)
        ax.set_xticks(range(3),SHORT);ax.set_ylim(0,den*1.2);ax.set_ylabel(f'{split}: outputs / {den}')
        ax.set_title('Strict joint success: confirmed / unknown')
        if index==0:ax.legend(loc='upper left',fontsize=8)
        ax=axes[index,2]
        for j,cost in enumerate(panel['costs']):
            finite=[v['delta_pp'] for v in cost['values'] if v['delta_pp'] is not None]
            # Position is deterministic and unrelated to value; no jitter RNG is used.
            xs=[j+(.32*(k/max(1,len(finite)-1)-.5) if len(finite)>1 else 0) for k in range(len(finite))]
            ax.scatter(xs,finite,s=14,c='#a0abb6',alpha=.70,zorder=2)
            if cost['mean_delta_pp'] is not None:
                ax.scatter(j,cost['mean_delta_pp'],marker='D',s=45,c='#ad4932',edgecolors='white',linewidths=.5,zorder=3)
                ax.annotate(f"{cost['mean_delta_pp']:+.2f}",(j,cost['mean_delta_pp']),xytext=(6,4),textcoords='offset points',fontsize=8,color='#8b3223')
        ax.axhline(0,color='#333333',lw=.8);ax.grid(axis='y',alpha=.18)
        ax.set_xticks(range(4),[f"{c['display']}\nn={c['common_known_pairs']}, NA={c['missing_pairs']}" for c in panel['costs']],fontsize=8)
        ax.set_ylabel('RL STOCH minus FULL STOCH (pp)');ax.set_title('POST paired geometry costs; positive = worse')
        ax.text(.02,.98,'Dots: requests; diamonds: macro means\nNo confidence interval / no zero-fill',transform=ax.transAxes,va='top',fontsize=8)
    fig.suptitle('Frozen 44-request final comparison | COMPLETE\nTRAIN: 8 C x 4 noises; DEV: 3 C x 4 noises',fontsize=14,y=.98)
    fig.text(.5,.11,'Strict joint = RAW adoption YES + POST adoption YES + POST body complete YES + POST legal C/F present.',ha='center',fontsize=9)
    fig.text(.5,.078,'Hatching is the full-panel UNKNOWN upper extension, not a confidence interval. '
        'CF/FF use actual POST free-face denominators; I point/lower and upper remain separate.',ha='center',fontsize=8)
    fig.text(.5,.045,FINAL_NOTE,ha='center',fontsize=9,color='#8a3827')
    fig.text(.5,.018,'Same frozen initial and transition noises pair RL STOCH with FULL STOCH. FULL DET is a separate decoding/use reference.',ha='center',fontsize=8)
    data.update(interpretation=[FINAL_NOTE,'No missing metrics are converted to zero. Per-request deltas and original summary rows are retained below.',
        'The cost display is descriptive, not an inferential confidence interval.',
        'Strict-joint UNKNOWN uses the existing three-valued conjunction; a known NO makes conjunction NO even if another component is UNKNOWN.'])
    return save('final_comparison','COMPLETE',fig,data,sources,plt)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind',choices=('training','final'))
    parser.add_argument('--training-pointer')
    args=parser.parse_args()
    print(json.dumps(training(args.training_pointer) if args.kind=='training' else final(),sort_keys=True))
