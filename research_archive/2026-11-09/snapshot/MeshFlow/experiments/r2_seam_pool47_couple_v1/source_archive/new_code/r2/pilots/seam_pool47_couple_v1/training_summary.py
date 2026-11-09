"""Read completed normal training logs only: no torch import or extra forwards."""
from __future__ import annotations
import csv
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from . import support as S

OUT=S.ROOT/'experiments/r2_seam_pool47_couple_v1'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def rows(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():yield json.loads(line)

def csvwrite(path,rr):
    fields=list(dict.fromkeys(k for r in rr for k in r))
    with Path(path).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rr)

def count_calls(path):
    begun=Counter();done=Counter();failed=Counter();pending={};steps=[]
    for row in rows(path):
        kind=row['kind'];ident=row['call_id']
        if row['event']=='CALL_BEGIN':
            if ident in pending:raise ValueError('Repeated call identity')
            pending[ident]=kind;begun[kind]+=1
            if kind=='optimizer_update':steps.append(row['context']['global_step'])
        else:
            if pending.pop(ident,None)!=kind:raise ValueError('Call pairing differs')
            if row['event']=='CALL_DONE':done[kind]+=1
            elif row['event']=='CALL_FAILED':failed[kind]+=1
            else:raise ValueError('Unknown event')
    expected={'train_forward':24000,'train_backward':24000,'optimizer_update':3000}
    if dict(begun)!=expected or dict(done)!=expected or failed or pending or steps!=list(range(13001,16001)):
        raise ValueError('Complete exact budget not met')
    return dict(attempted=dict(begun),successful=dict(done),failed=dict(failed),pending=len(pending),
        generator_updates=3000,generator_forwards=24000,generator_backwards=24000,head_forwards=0,head_updates=0)

def main():
    dest=OUT/'training_analysis';dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():
        value=read(dest/'complete.json')
        for p,h in value['files'].items():
            if sha(p)!=h:raise ValueError('Completed analysis changed')
        print(json.dumps(value));return
    reg=read(OUT/'training_registration.json');schedule=read(OUT/'shared_schedule.json');manifest=read(OUT/'shared_data_manifest.json')
    inputs={str(p):sha(p) for p in (OUT/'training_registration.json',OUT/'shared_schedule.json',OUT/'shared_data_manifest.json')}
    updates=[];windows=[];rolling=[];geometry=[];exposures=[];initial={};budgets={};exposure_signatures={}
    for arm in S.ARMS:
        folder=OUT/'training'/arm;status=read(folder/'status.json')
        if status['status']!='COMPLETE' or status['local_step']!=3000 or status['global_step']!=16000:
            raise ValueError('Only final3000 analysis; incomplete '+arm)
        calls=count_calls(folder/'calls.jsonl');budgets[arm]=calls
        paths=[folder/n for n in ('status.json','calls.jsonl','training_log.jsonl','geometry_diagnostics.jsonl','exposure.jsonl')]
        for p in paths:inputs[str(p)]=sha(p)
        initpaths=list((folder/'attempts').glob('*/initialization.json'))
        starts=[(p,read(p)) for p in initpaths if read(p)['global_step']==13000]
        if len(starts)!=1:raise ValueError('Exactly one original initialization required')
        p,initial[arm]=starts[0];inputs[str(p)]=sha(p)
        for field,rfield in [('generator_state_sha256','source_generator_sha256'),('optimizer_state_sha256','source_optimizer_sha256'),('rng_sha256','source_rng_sha256')]:
            if initial[arm][field]!=reg[rfield]:raise ValueError('Source restoration mismatch')
        logs=list(rows(folder/'training_log.jsonl'))
        if [r['local_step'] for r in logs]!=list(range(1,3001)):raise ValueError('Missing/repeated update log')
        ar=[]
        for r in logs:
            selected={k:v for k,v in r.items() if k not in ('time_unix','pid')}
            den=r['FM_velocity_gradient_norm_sum']
            selected.update(arm=arm,couple_to_FM_velocity_norm_sum_ratio=(r['coupling_lambda']*r['coupling_velocity_gradient_norm_sum']/den if den else None))
            ar.append(selected);updates.append(selected)
        metrics=('FM','coupling','couple_to_FM_velocity_norm_sum_ratio','generator_gradient_norm','input_seconds','update_seconds','forward_seconds','backward_seconds','cuda_peak_allocated_bytes','cuda_peak_reserved_bytes')
        for name,lo,hi in [('FIRST100',1,100),('LAST100',2901,3000),('ALL',1,3000)]:
            rr=ar[lo-1:hi]
            for metric in metrics:
                vv=[r[metric] for r in rr if r.get(metric) is not None]
                windows.append(dict(arm=arm,window=name,step_start=lo,step_end=hi,metric=metric,n=len(vv),
                    mean=float(np.mean(vv)) if vv else None,median=float(np.median(vv)) if vv else None,
                    min=float(np.min(vv)) if vv else None,max=float(np.max(vv)) if vv else None))
        for hi in range(100,3001,25):
            rr=ar[hi-100:hi]
            rolling.append(dict(arm=arm,step_start=hi-99,step_end=hi,**{m:float(np.mean([r[m] for r in rr])) for m in metrics if all(r.get(m) is not None for r in rr)}))
        signature=[];counts=Counter()
        for i,r in enumerate(rows(folder/'exposure.jsonl')):
            step=i//8+1;micro=i%8;frozen=manifest['steps'][step-1]['samples'][micro];scheduled=schedule['steps'][step-1]['rows'][micro]
            if r['local_step']!=step or r['microbatch']!=micro or r['path']!=frozen['path'] or r['uid']!=frozen['uid'] or r['alpha']!=frozen['alpha']:
                raise ValueError('Exposure differs from common frozen cache')
            signature.append({k:v for k,v in r.items() if k not in ('time_unix','pid')})
            counts[r['uid']]+=1
        if len(signature)!=24000 or dict(counts)!=schedule['parent_exposure']:raise ValueError('Exposure counts differ')
        exposure_signatures[arm]=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
        for uid,n in sorted(counts.items()):exposures.append(dict(arm=arm,parent_id=uid,exposures=n))
        groups=defaultdict(lambda:dict(samples=0,scalar_count=0,velocity_sse=0.,endpoint_sse=0.,scaled_velocity_sse=0.))
        n_diag=0
        for row in rows(folder/'geometry_diagnostics.jsonl'):
            n_diag+=1;step=row['local_step'];bi=row['time_bin'];win=['ALL',f'BLOCK{(step-1)//100+1:02d}']
            if step<=100:win.append('FIRST100')
            if step>2900:win.append('LAST100')
            for region in ('source_interface_third','source_interface_other_free','bulk'):
                for w in win:
                    for b in (bi,-1):
                        acc=groups[w,b,region];acc['samples']+=1;acc['scalar_count']+=row[region+'_scalar_count']
                        acc['velocity_sse']+=row[region+'_velocity_sse'];acc['endpoint_sse']+=row[region+'_endpoint_sse']
                        acc['scaled_velocity_sse']+=row[region+'_endpoint_scaled_velocity_sse']
        if n_diag!=24000:raise ValueError('Expected exactly one diagnostic per actual training forward')
        for (win,bi,region),v in sorted(groups.items()):
            den=v['scalar_count'];geometry.append(dict(arm=arm,window=win,time_bin=bi,
                time_lower=None if bi==-1 else reg['diagnostic_time_bin_edges'][bi],
                time_upper=None if bi==-1 else reg['diagnostic_time_bin_edges'][bi+1],region=region,**v,
                velocity_mse=v['velocity_sse']/den if den else None,endpoint_mse=v['endpoint_sse']/den if den else None,
                scaled_velocity_mse=v['scaled_velocity_sse']/den if den else None,status='PASS' if den else 'NA_NO_FREE_COORDINATES'))
    shared_fields=('generator_state_sha256','optimizer_state_sha256','rng_sha256','generator_hyperparameters','EMA')
    init_equal={f:initial[S.ARMS[0]][f]==initial[S.ARMS[1]][f] for f in shared_fields}
    if not all(init_equal.values()) or len(set(exposure_signatures.values()))!=1:raise ValueError('Paired same-start/input proof failed')
    for name,data in [('per_update',updates),('first_last_all',windows),('rolling100',rolling),('time_binned_geometry',geometry),('per_parent_exposure',exposures)]:csvwrite(dest/(name+'.csv'),data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for arm in S.ARMS:
        rr=[r for r in rolling if r['arm']==arm]
        axes[0,0].plot([r['step_end'] for r in rr],[r['FM'] for r in rr],label=arm)
        axes[0,1].plot([r['step_end'] for r in rr],[r['couple_to_FM_velocity_norm_sum_ratio'] for r in rr],label=arm)
        for ax,region in zip(axes[1],('source_interface_third','bulk')):
            gg=sorted([r for r in geometry if r['arm']==arm and r['time_bin']==-1 and r['region']==region and r['window'].startswith('BLOCK')],key=lambda r:r['window'])
            ax.plot([int(r['window'][5:])*100 for r in gg],[r['velocity_mse'] for r in gg],label=arm)
    for ax,title in zip(axes.flat,('FM: rolling100','lambda * coupling / FM velocity norm sums','Seam tip velocity MSE: 100-step bins','Bulk velocity MSE: 100-step bins')):
        ax.set_title(title);ax.set_xlabel('Additional optimizer updates');ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(dest/'training_curves.png',dpi=160);plt.close(fig)
    result=dict(status='COMPLETE',arms=list(S.ARMS),updates_per_arm=3000,exposures_per_arm=24000,
        call_budgets=budgets,initialization_equal=init_equal,exposure_identity_sha256=exposure_signatures,
        input_sources=inputs,extra_model_forward=0,extra_backward=0,
        diagnostic_limit='Normal supervised interpolation fit only, not rollout; velocity-gradient norm sums are not parameter-gradient ratios',
        files={str(p):sha(p) for p in dest.iterdir() if p.is_file()})
    with (dest/'complete.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','input_sources')}),flush=True)

if __name__=='__main__':main()
