"""Offline Chair multipart training curves/accounting. Never calls a model.

Run --final after COMPLETE. Optional --verify-checkpoint loads the final full
checkpoint on CPU and verifies AdamW/RNG hashes without constructing a network.
All training MSE summaries are sums of SSE / sums of actual scalar counts.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse, collections, csv, datetime, hashlib, json, math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
DEFAULT_ROOT=ROOT/'experiments/r2_chair_multipart_seam_v1'
KINDS=('train_forward','train_backward','optimizer_update','head_forward','head_backward','head_optimizer_update')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]

def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def csvwrite(path,rows):
    rows=list(rows)
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)

def divide(n,d):return n/d if d else None

def aggregates(rows):
    result={'examples':len(rows)}
    for name in ('all_free','bulk','seam_tip'):
        s=sum(float(r[name+'_velocity_sse']) for r in rows)
        n=sum(int(r[name+'_scalar_count']) for r in rows)
        result[name+'_sse']=s;result[name+'_scalar_count']=n;result[name+'_MSE']=divide(s,n)
    return result

def headstat(rows):
    c={k:sum(int(r[k]) for r in rows) for k in ('TP','FN','FP','TN')}
    return dict(**c,precision=divide(c['TP'],c['TP']+c['FP']),recall=divide(c['TP'],c['TP']+c['FN']),
                specificity=divide(c['TN'],c['TN']+c['FP']),accuracy=divide(c['TP']+c['TN'],sum(c.values())))

def account(root,logs,exposure,calls,final):
    count=len(logs);steps=[int(r['local_step']) for r in logs]
    assert steps==list(range(1,count+1)), 'Missing/duplicate/nonconsecutive training log'
    assert all(int(r['global_step'])==14000+int(r['local_step']) for r in logs)
    if final:assert count==1500
    if not final:exposure=[r for r in exposure if int(r['local_step'])<=count]
    assert len(exposure)==8*count
    bystep=collections.defaultdict(list)
    for r in exposure:bystep[int(r['local_step'])].append(r)
    for r in logs:
        step=int(r['local_step']);xs=bystep[step]
        assert sorted(int(x['microbatch']) for x in xs)==list(range(8))
        assert collections.Counter(x['part'] for x in xs)=={'SEAT':4,'BACK':4}
        a=aggregates(xs)
        for name in ('all_free','bulk','seam_tip'):
            assert a[name+'_scalar_count']==r[name+'_scalar_count']
            assert math.isclose(a[name+'_sse'],r[name+'_velocity_sse'],rel_tol=1e-10,abs_tol=1e-8)
        assert math.isclose(float(r['FM']),a['all_free_MSE'],rel_tol=4e-6,abs_tol=1e-7)
        assert math.isclose(sum(x['BCE_contribution'] for x in xs),r['occupancy_BCE'],rel_tol=1e-9,abs_tol=1e-8)
        assert sum(int(x['head_edge_count']) for x in xs)==r['head_edge_denominator']==sum(int(r[k]) for k in ('TP','FN','FP','TN'))
        assert a['all_free_scalar_count']==a['bulk_scalar_count']+a['seam_tip_scalar_count']
        assert math.isclose(a['all_free_sse'],a['bulk_sse']+a['seam_tip_sse'],rel_tol=1e-7,abs_tol=1e-6)
    begun={};pending={};attempted=collections.Counter();done=collections.Counter();failed=collections.Counter()
    for r in calls:
        ident=int(r['call_id']);kind=r['kind'];event=r['event']
        assert kind in KINDS
        if event=='CALL_BEGIN':
            assert ident not in begun;begun[ident]=r;pending[ident]=r;attempted[kind]+=1
        else:
            assert ident in pending and pending[ident]['kind']==kind;pending.pop(ident)
            (done if event=='CALL_DONE' else failed)[kind]+=1
    expected={k:count*(8 if k.endswith('forward') or k.endswith('backward') else 1) for k in KINDS}
    if final:
        assert not pending and not failed
        assert dict(attempted)==expected and dict(done)==expected
        for kind in ('optimizer_update','head_optimizer_update'):
            sequence=[r['context']['global_step'] for r in calls if r['event']=='CALL_BEGIN' and r['kind']==kind]
            assert sequence==list(range(14001,15501))
    schedule=read(root/'data/schedule.json')
    expected_exposure=collections.Counter(r['condition_id'] for entry in schedule['steps'][:count] for r in entry['samples'])
    actual_exposure=collections.Counter(r['condition_id'] for r in exposure)
    assert expected_exposure==actual_exposure
    return dict(status='PASS',final=final,completed_updates=count,global_step=14000+count,
                attempted=dict(attempted),completed=dict(done),failed=dict(failed),pending_call_ids=sorted(pending),expected=expected,
                actual_exposures=len(exposure),by_part=dict(collections.Counter(r['part'] for r in exposure)),
                per_condition_schedule_exact=True,SSE_partition_exact=True,model_calls_added=0), exposure

def verify_checkpoint(root,status):
    import sys
    sys.path.insert(0,str(ROOT/'r2/src'))
    import torch
    from meshflow_control.checkpoints import state_hash
    from meshflow_control.training.schedule40 import tree_hash
    torch.set_num_threads(1)
    path=Path(status['checkpoint']['path']);actual=sha(path)
    assert actual==status['checkpoint']['sha256']
    p=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    assert (p['schema'],p['local_step'],p['global_step'],p['head_step'],p['EMA'],p['update_in_progress'])==('chair_multipart_seam_training_v1',1500,15500,1500,False,False)
    checks={k:(fn(p[k])==p[h]) for k,fn,h in (
        ('generator_state',state_hash,'generator_state_sha256'),('optimizer',tree_hash,'optimizer_state_sha256'),
        ('rng',tree_hash,'rng_sha256'),('head_state',state_hash,'head_state_sha256'),('head_optimizer',tree_hash,'head_optimizer_state_sha256'))}
    assert all(checks.values())
    cursors={}
    for key,expected in (('optimizer',15500),('head_optimizer',1500)):
        counts=collections.Counter(int(s['step']) for s in p[key]['state'].values());assert set(counts)=={expected}
        assert all(s['exp_avg'].shape==s['exp_avg_sq'].shape and s['exp_avg'].dtype==s['exp_avg_sq'].dtype==torch.float32 for s in p[key]['state'].values())
        cursors[key]=dict(counts)
    assert p['registration_sha256']==sha(root/'training_registration.json')
    reg=read(root/'training_registration.json');assert p['source_checkpoint_sha256']==reg['source_checkpoint_sha256']
    assert len(p['optimizer_parameter_names'])==1 and len(p['optimizer_parameter_names'][0])==len(p['optimizer']['param_groups'][0]['params'])
    export_checks={}
    for key,name,statekey,hashkey in (('inference_export','MULTIPART_S1500_generator.pt','generator_state','generator_state_sha256'),
                                   ('occupancy_export','MULTIPART_S1500_occupancy.pt','head_state','head_state_sha256')):
        dest=root/'inference'/name;assert sha(dest)==status[key]['sha256']
        q=torch.load(dest,map_location='cpu',weights_only=True,mmap=True)
        assert q[hashkey]==p[hashkey] and state_hash(q[statekey])==p[hashkey]
        assert q['training_checkpoint_sha256']==actual and q['local_step']==1500 and q['global_step']==15500
        export_checks[name]='PASS';del q
    return dict(status='PASS',path=str(path),sha256=actual,internal_state_hashes=checks,
                optimizer_step_counts=cursors,FP32_matching_moments=True,exports_match_full_live_state=export_checks,
                model_constructions=0,model_forwards=0,GPU_calls=0)

def draw(out,logs,windows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    x=[r['local_step'] for r in logs]
    axs[0,0].plot(x,[r['FM'] for r in logs],alpha=.2,lw=.5,color='gray',label='batch FM')
    allrows=[r for r in windows if r['part']=='ALL' and r['time_bin']=='ALL']
    for metric,label in (('all_free_MSE','all free'),('bulk_MSE','bulk'),('seam_tip_MSE','seam tip')):
        axs[0,0].plot([r['end_step'] for r in allrows],[r[metric] for r in allrows],label=label)
    for part in ('SEAT','BACK'):
        rows=[r for r in windows if r['part']==part and r['time_bin']=='ALL']
        axs[0,1].plot([r['end_step'] for r in rows],[r['all_free_MSE'] for r in rows],label=part)
        axs[1,0].plot([r['end_step'] for r in rows],[r['seam_tip_MSE'] for r in rows],label=part)
    for metric in ('precision','recall','specificity'):
        axs[1,1].plot([r['end_step'] for r in allrows],[r['head_'+metric] for r in allrows],label=metric)
    titles=['Online FM, SSE / scalar count','Online all-free MSE by part','Online seam-tip MSE by part','C-only head, pooled batch counts']
    for ax,title in zip(axs.flat,titles):ax.set_title(title);ax.set_xlabel('new optimizer updates');ax.legend();ax.grid(alpha=.2)
    axs[1,1].set_ylim(-.02,1.02)
    fig.suptitle('Chair SEAT47 + BACK31, fixed 1500 updates\n100-update sums; changing training inputs, not fixed-probe convergence or mesh quality')
    fig.tight_layout();fig.savefig(out/'loss.png',dpi=150);plt.close(fig)

def summarize(root,*,final=False,verify=False):
    root=Path(root).resolve();training=root/'training';status=read(training/'status.json')
    if final:assert status['status']=='COMPLETE' and int(status['local_step'])==1500
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out=root/'training_summary'/stamp;out.mkdir(parents=True,exist_ok=False)
    sources=[training/n for n in ('status.json','training_log.jsonl','exposure.jsonl','calls.jsonl','checkpoint_history.jsonl')]
    sources += [root/'training_registration.json',root/'data/manifest.json',root/'data/schedule.json',Path(__file__)]
    logs=jsonl(training/'training_log.jsonl');exposure=jsonl(training/'exposure.jsonl');calls=jsonl(training/'calls.jsonl')
    accounting,exposure=account(root,logs,exposure,calls,final);write(out/'accounting.json',accounting)
    logmap={int(r['local_step']):r for r in logs}
    perstep=[]
    for r in logs:
        q=dict(r)
        for p in ('all_free','bulk','seam_tip'):q[p+'_MSE']=divide(r[p+'_velocity_sse'],r[p+'_scalar_count'])
        q.update({'head_'+k:v for k,v in headstat([r]).items()});perstep.append(q)
    csvwrite(out/'training_per_step.csv',perstep)
    windows=[];overall=[]
    ranges=[(1,len(logs),'ALL')]+[(a,min(a+99,len(logs)),'WINDOW100') for a in range(1,len(logs)+1,100)]
    for first,last,scope in ranges:
        rows=[r for r in exposure if first<=int(r['local_step'])<=last]
        for part in ('ALL','SEAT','BACK'):
            for tb in ('ALL','0.0-0.2','0.2-0.4','0.4-0.6','0.6-0.8','0.8-1.0'):
                subset=[r for r in rows if (part=='ALL' or r['part']==part) and (tb=='ALL' or r['time_bin']==tb)]
                q=dict(scope=scope,start_step=first,end_step=last,part=part,time_bin=tb,**aggregates(subset))
                edges=sum(int(r['head_edge_count']) for r in subset)
                bcesum=sum(float(r['BCE_contribution'])*logmap[int(r['local_step'])]['head_edge_denominator'] for r in subset)
                q.update(head_edges=edges,head_BCE_sum=bcesum,head_BCE=divide(bcesum,edges))
                if part=='ALL' and tb=='ALL':q.update({'head_'+k:v for k,v in headstat([r for r in logs if first<=r['local_step']<=last]).items()});q['head_counts_status']='RECORDED_BATCH_COUNTS'
                else:q.update(head_precision=None,head_recall=None,head_specificity=None,head_counts_status='NA_PER_EXAMPLE_TP_FN_FP_TN_NOT_RECORDED_NO_NEW_FORWARD')
                (overall if scope=='ALL' else windows).append(q)
    csvwrite(out/'training_by_part_time.csv',overall);csvwrite(out/'training_windows100_by_part_time.csv',windows)
    condition=[];expected=read(root/'data/schedule.json')['condition_exposure']
    for cid in sorted(set(r['condition_id'] for r in exposure)):
        rr=[r for r in exposure if r['condition_id']==cid]
        condition.append(dict(condition_id=cid,parent_id=rr[0]['parent_id'],part=rr[0]['part'],actual_exposures=len(rr),registered_full_budget_exposures=expected[cid],
            clean_exposures=sum(float(r['alpha'])==1 for r in rr),width_augmented_exposures=sum(float(r['alpha'])!=1 for r in rr),
            alpha_min=min(r['alpha'] for r in rr),alpha_max=max(r['alpha'] for r in rr)))
    csvwrite(out/'exposure_by_condition.csv',condition)
    resource=dict(torch_peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in logs),torch_peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in logs),
        torch_peak_scope='maximum recorded per-formal-batch peak; not process VRAM',process_peak_mib=status.get('resources',{}).get('process_peak_mib'),
        process_memory_status=status.get('resources',{}).get('process_memory_status'),
        total_device_used_peak_mib=status.get('resources',{}).get('total_device_used_peak_mib'),
        process_scope='GPUObserver nvidia-smi process reporting; may be NA under WSL; total device occupancy includes unrelated graphics',
        total_formal_update_seconds=sum(r['update_seconds'] for r in logs),total_input_seconds=sum(r['input_seconds'] for r in logs),
        mean_formal_update_seconds=sum(r['update_seconds'] for r in logs)/len(logs),status_elapsed_seconds=status.get('elapsed_seconds'))
    write(out/'resources_summary.json',resource)
    if verify:
        if not final:raise ValueError('Full checkpoint verification is final only')
        write(out/'final_checkpoint_check.json',verify_checkpoint(root,status))
    draw(out,logs,windows)
    hashes=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sources]
    write(out/'source_hashes.json',hashes)
    write(out/'completion.json',dict(status='COMPLETE',final=final,local_step=len(logs),model_forwards=0,new_training_updates=0,
        checkpoint_CPU_verified=verify,head_per_part_precision_recall='NA_NOT_RECORDED; no invented estimates or head calls',
        observations='Online stochastic training diagnostics only; no DEV early stopping or quality conclusion'))
    print(json.dumps(dict(status='COMPLETE',output=str(out),updates=len(logs),final=final,verified=verify)))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(DEFAULT_ROOT));p.add_argument('--final',action='store_true');p.add_argument('--verify-checkpoint',action='store_true');a=p.parse_args()
    summarize(a.root,final=a.final,verify=a.verify_checkpoint)
if __name__=='__main__':main()
