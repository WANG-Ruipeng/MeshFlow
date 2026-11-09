"""Offline exact accounting, training/probe curves and full-checkpoint audit."""
from __future__ import annotations
import argparse
import collections
import csv
import datetime
import json
import math
from pathlib import Path
from .data import ROOT, resolve, sha, read

OUT=ROOT/'experiments/r2_bench_clean_multipart_seam_v1'
KINDS=('train_forward','train_backward','optimizer_update','head_forward','head_backward','head_optimizer_update','probe_forward')


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def csvwrite(path,records):
    if not records:return
    keys=list(dict.fromkeys(k for r in records for k in r))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(records)


def aggregate(records):
    out=dict(examples=len(records))
    for name in ('all_free','bulk','seam_tip'):
        sse=sum(r[name+'_velocity_sse'] for r in records)
        count=sum(r[name+'_scalar_count'] for r in records)
        out.update({name+'_SSE':sse,name+'_scalar_count':count,name+'_MSE':sse/count if count else None})
    return out


def plot_curves(out,log,ps,historical):
    """New training plus old unequal-budget reference probes, never new calls."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,2,figsize=(14,12))
    axs[0,0].plot([r['local_step'] for r in log],[r['FM'] for r in log],alpha=.4,lw=.5)
    axs[0,0].set_title('New batch FM (changing inputs; not a fixed probe)')
    axs[0,1].plot([r['local_step'] for r in log],[r['occupancy_BCE'] for r in log],lw=.6)
    axs[0,1].set_title('New head unweighted per-C mean BCE')
    for axis,metric,title in ((axs[1,0],'all_free_MSE','Fixed probe: ALL free'),
                              (axs[1,1],'seam_tip_MSE','Fixed probe: seam tip'),
                              (axs[2,0],'bulk_MSE','Fixed probe: bulk')):
        for split,color in (('TRAIN','tab:blue'),('DEV','tab:orange')):
            rr=[r for r in ps if r['split']==split and r['part']=='ALL']
            axis.plot([12000+r['local_step'] for r in rr],[r[metric] for r in rr],
                      marker='o',color=color,label=f'NEW {split}')
            old=[r for r in historical if r['split']==split and r['part']=='ALL']
            axis.plot([r['global_step'] for r in old],[r[metric] for r in old],
                      marker='s',ls='--',color=color,alpha=.6,label=f'OLD SEAM {split} (+3000/+9000)')
        axis.axvline(13500,color='gray',lw=.8,ls=':')
        axis.set_title(title);axis.set_xlabel('Generator cumulative global step');axis.legend(fontsize=8);axis.grid(alpha=.2)
    axs[2,1].axis('off')
    axs[2,1].text(0,.95,
        'Budget and comparison boundaries\n\n'
        'NEW: COMMON12000 + fixed 1500 updates, 58 parents / 195 eligible C\n'
        'OLD: COMMON12000 + 3000 or 9000 SEAM updates, original 277 C\n'
        'Same frozen 96 probe inputs at each observed point.\n'
        'MSE = sum squared error / actual free-coordinate scalar count.\n\n'
        'Historical points are UNEQUAL-BUDGET recipe references.\n'
        'The new pool, xyz augmentation and contextual normalized head\n'
        'change jointly; these curves do not identify one causal component.\n'
        'Probe inputs contain source target interpolation: they are not rollout.\n'
        'No new model calls were made to create this figure.',
        va='top',fontsize=10,linespacing=1.5)
    for axis in axs[0]:
        axis.set_xlabel('New optimizer updates');axis.grid(alpha=.2)
    fig.suptitle('Bench clean multipart SEAM: actual training and frozen probes\nHistorical SEAM15000 / SEAM21000 are UNEQUAL-BUDGET references',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(out/'loss_and_fixed_probes.png',dpi=150);plt.close(fig)


def audit_checkpoint(root,status):
    import torch
    from meshflow_control.checkpoints import state_hash
    from meshflow_control.training.schedule40 import tree_hash
    torch.set_num_threads(1)
    path=resolve(status['checkpoint']['path'])
    assert sha(path)==status['checkpoint']['sha256']
    p=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    assert (p['schema'],p['local_step'],p['global_step'],p['head_step'],p['EMA'],p['update_in_progress'])==('bench_clean_multipart_seam_training_v1',1500,13500,1500,False,False)
    checks={k:fn(p[k])==p[h] for k,fn,h in (
        ('generator_state',state_hash,'generator_state_sha256'),('optimizer',tree_hash,'optimizer_state_sha256'),
        ('rng',tree_hash,'rng_sha256'),('head_state',state_hash,'head_state_sha256'),('head_optimizer',tree_hash,'head_optimizer_state_sha256'))}
    assert all(checks.values())
    cursors={}
    for key,expected in (('optimizer',13500),('head_optimizer',1500)):
        cc=collections.Counter(int(s['step']) for s in p[key]['state'].values())
        assert set(cc)=={expected}
        assert all(s['exp_avg'].dtype==s['exp_avg_sq'].dtype==torch.float32 and s['exp_avg'].shape==s['exp_avg_sq'].shape for s in p[key]['state'].values())
        cursors[key]=dict(cc)
    reg=read(root/'training_registration.json')
    assert p['registration_sha256']==sha(root/'training_registration.json')
    assert p['source_checkpoint_sha256']==reg['source_checkpoint_sha256']
    export={}
    for name,statekey,hkey in (('generator','generator_state','generator_state_sha256'),('occupancy','head_state','head_state_sha256')):
        dest=root/'inference'/f'BENCH_CLEAN_MULTIPART_S1500_{name}.pt'
        q=torch.load(dest,map_location='cpu',weights_only=True,mmap=True)
        assert q[hkey]==p[hkey] and state_hash(q[statekey])==p[hkey]
        assert q['training_checkpoint_sha256']==sha(path)
        if name=='occupancy':
            assert q['head_scaler_sha256']==reg['head_scaler_sha256']
            sc=read(resolve(reg['head_scaler']))
            assert torch.equal(q['head_state']['feature_mean'],torch.tensor(sc['mean'],dtype=torch.float32))
            assert torch.equal(q['head_state']['feature_scale'],torch.tensor(sc['scale'],dtype=torch.float32))
        export[name]=dict(path=str(dest),sha256=sha(dest),state_sha256=q[hkey])
        del q
    return dict(status='PASS',internal_hashes=checks,optimizer_cursors=cursors,exports=export,
                normalization_buffers_exact=True,model_constructions=0,model_forwards=0,GPU_calls=0)


def summarize(final=False,verify=False):
    root=OUT;status=read(root/'training/status.json')
    if final:assert status['status']=='COMPLETE' and status['local_step']==1500
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out=root/'training_summary'/stamp;out.mkdir(parents=True,exist_ok=False)
    log=rows(root/'training/training_log.jsonl');exposure=rows(root/'training/exposure.jsonl');calls=rows(root/'training/calls.jsonl')
    count=len(log)
    assert [r['local_step'] for r in log]==list(range(1,count+1))
    assert all(r['global_step']==12000+r['local_step'] for r in log)
    if not final:exposure=[r for r in exposure if r['local_step']<=count]
    assert len(exposure)==count*8
    bystep=collections.defaultdict(list)
    for r in exposure:bystep[r['local_step']].append(r)
    for l in log:
        rr=bystep[l['local_step']]
        assert len(rr)==8 and len({r['parent_id'] for r in rr})==4
        assert collections.Counter(r['augmentation'] for r in rr)==dict(CLEAN=4,ANISOTROPIC=4)
        a=aggregate(rr)
        assert math.isclose(a['all_free_MSE'],l['FM'],rel_tol=4e-6,abs_tol=1e-7)
        assert math.isclose(sum(r['BCE_contribution'] for r in rr),l['occupancy_BCE'],rel_tol=1e-8,abs_tol=1e-8)
        assert a['all_free_scalar_count']==a['bulk_scalar_count']+a['seam_tip_scalar_count']==l['free_coordinate_denominator']
    attempted=collections.Counter();done=collections.Counter();failed=collections.Counter();pending={}
    for r in calls:
        kind=r['kind'];ident=r['call_id'];assert kind in KINDS
        if r['event']=='CALL_BEGIN':
            assert ident not in pending;pending[ident]=r;attempted[kind]+=1
        else:
            assert ident in pending;pending.pop(ident)
            (done if r['event']=='CALL_DONE' else failed)[kind]+=1
    expected={k:count*(8 if k.endswith('forward') or k.endswith('backward') else 1) for k in KINDS if k!='probe_forward'}
    expected['probe_forward']=96*sum(s<=count for s in (0,500,1000,1500))
    if final:
        assert count==1500 and not failed and not pending
        assert dict(attempted)==expected==dict(done)
    actual=collections.Counter(r['condition_id'] for r in exposure)
    sched=read(root/'data/schedule.json')
    want=collections.Counter(r['condition_id'] for e in sched['steps'][:count] for r in e['samples'])
    assert actual==want
    accounting=dict(status='PASS',local_step=count,global_step=12000+count,attempted=dict(attempted),
                    completed=dict(done),failed=dict(failed),pending=sorted(pending),expected=expected,
                    exposures=len(exposure),per_condition_exposure_exact=True,perC_BCE_then_batch=True,
                    training_calls_replayed=0 if not failed else None,additional_model_calls=0)
    write(out/'accounting.json',accounting)
    csvwrite(out/'training_per_step.csv',[dict(r,**{n+'_MSE':r[n+'_velocity_sse']/r[n+'_scalar_count'] if r[n+'_scalar_count'] else None for n in ('all_free','bulk','seam_tip')}) for r in log])
    windows=[]
    for first in range(1,count+1,100):
        end=min(first+99,count);rr=[r for r in exposure if first<=r['local_step']<=end]
        for part in ['ALL']+sorted({r['part_type'] for r in rr}):
            for tb in ['ALL','0.0-0.2','0.2-0.4','0.4-0.6','0.6-0.8','0.8-1.0']:
                ss=[r for r in rr if (part=='ALL' or r['part_type']==part) and (tb=='ALL' or r['time_bin']==tb)]
                windows.append(dict(start_step=first,end_step=end,part=part,time_bin=tb,**aggregate(ss),
                                    per_condition_BCE_mean=sum(r['BCE_contribution']*8 for r in ss)/len(ss) if ss else None))
    csvwrite(out/'windows100_part_time.csv',windows)
    cr=[]
    for cid in sorted(actual):
        rr=[r for r in exposure if r['condition_id']==cid]
        cr.append(dict(condition_id=cid,parent_id=rr[0]['parent_id'],part=rr[0]['part_type'],exposures=len(rr),
                       clean=sum(r['augmentation']=='CLEAN' for r in rr),augmented=sum(r['augmentation']=='ANISOTROPIC' for r in rr)))
    csvwrite(out/'exposure_by_condition.csv',cr)
    probes=[]
    for step in (0,500,1000,1500):
        p=root/'training/fixed_loss'/f'step{step}.json'
        if not p.exists():continue
        v=read(p)
        assert v['generator_forwards']==96 and v['head_forwards']==0 and v['RNG_unchanged']
        probes.extend(dict(local_step=step,global_step=12000+step,**r) for r in v['rows'])
    csvwrite(out/'fixed_probe_per_input.csv',probes)
    ps=[]
    for step in sorted({r['local_step'] for r in probes}):
        for split in ('TRAIN','DEV'):
            subset=[r for r in probes if r['local_step']==step and r['split']==split]
            for part in ['ALL']+sorted({r['part'] for r in subset}):
                ps.append(dict(local_step=step,split=split,part=part,**aggregate([r for r in subset if part=='ALL' or r['part']==part])))
    csvwrite(out/'fixed_probe_summary.csv',ps)
    historical=[];reference_sources=[]
    for gs in (15000,21000):
        p=ROOT/'experiments/r2_bench_parts_convergence_v1/training/SEAM/fixed_loss'/f'step{gs}.json'
        v=read(p);reference_sources.append(dict(path=str(p),sha256=sha(p),global_step=gs))
        for split in ('TRAIN','DEV'):
            rr=[r for r in v['rows'] if r['split']==split]
            for part in ['ALL']+sorted({r['part'] for r in rr}):
                historical.append(dict(model=f'OLD_SEAM_S{gs}',global_step=gs,stage_updates=gs-12000,
                    split=split,part=part,comparison='UNEQUAL_BUDGET_HISTORICAL_RECIPE_REFERENCE',
                    **aggregate([r for r in rr if part=='ALL' or r['part']==part])))
    csvwrite(out/'historical_fixed_probe_summary.csv',historical)
    write(out/'historical_probe_sources.json',reference_sources)
    plot_curves(out,log,ps,historical)
    if verify:
        assert final
        write(out/'checkpoint_audit.json',audit_checkpoint(root,status))
    write(out/'completion.json',dict(status='COMPLETE',final=final,local_step=count,zero_additional_model_calls=True))
    write(root/'training_summary/latest.json',dict(path=str(out),final=final,local_step=count))
    print(json.dumps(dict(status='COMPLETE',path=str(out),local_step=count)))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--final',action='store_true');p.add_argument('--verify-checkpoint',action='store_true')
    args=p.parse_args();summarize(args.final,args.verify_checkpoint)
