"""Finalize a CPU-only independent audit of the two frozen C200 training arms.

No model construction, CUDA calls, OT replay or scoring. Final checkpoints are
loaded once each on CPU after both arms are COMPLETE.
"""
from pathlib import Path
import collections, csv, hashlib, io, json, math, os, re, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'experiments/r2_target_cleanup_c200_v1'
OUT=EXP/'training_audit'
ARMS=('ORIGINAL_C200','DEDUP_C200')

def resolve(p):
    s=str(p).replace('\\','/')
    if os.name=='nt' and re.match(r'^/mnt/[a-z]/',s):s=s[5].upper()+':/'+s[7:]
    elif os.name!='nt' and re.match(r'^[A-Za-z]:/',s):s='/mnt/'+s[0].lower()+s[2:]
    p=Path(s);return p if p.is_absolute() else ROOT/p
def read(p):return json.loads(resolve(p).read_text(encoding='utf-8-sig'))
def rows(p):return [json.loads(x) for x in resolve(p).read_text(encoding='utf-8').splitlines() if x.strip()]
def sha(p):
    h=hashlib.sha256()
    with resolve(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def digest_json(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def checked(p):
    p=resolve(p);assert p.resolve().is_relative_to(OUT.resolve());p.parent.mkdir(parents=True,exist_ok=True);return p
def write(p,v):
    p=checked(p);payload=(json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    if p.exists():assert p.read_bytes()==payload,'Existing audit differs: '+str(p)
    else:p.write_bytes(payload)
def csvwrite(p,values):
    values=list(values);keys=list(dict.fromkeys(k for v in values for k in v)) or ['status']
    s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=keys);w.writeheader()
    for v in values:w.writerow({k:json.dumps(x,sort_keys=True) if isinstance(x,(list,dict)) else x for k,x in v.items()})
    p=checked(p);payload=s.getvalue().encode('utf-8-sig')
    if p.exists():assert p.read_bytes()==payload
    else:p.write_bytes(payload)
def close(a,b):return math.isclose(float(a),float(b),rel_tol=3e-6,abs_tol=1e-8)

def ledger_audit(arm,rs):
    pending={};begun={};done=[];failed=[]
    for r in rs:
        assert r['arm']==arm
        key=r['call_id']
        if r['event']=='CALL_BEGIN':assert key not in begun;begun[key]=r;pending[key]=r
        elif r['event'] in ('CALL_DONE','CALL_FAILED'):
            assert key in pending and pending[key]['kind']==r['kind']
            assert pending[key]['context']==r['context'];pending.pop(key)
            (done if r['event']=='CALL_DONE' else failed).append(r)
        else:raise AssertionError('Unknown event')
    expect=dict(train_forward=1600,train_backward=1600,optimizer_update=200)
    assert dict(collections.Counter(r['kind'] for r in begun.values()))==expect
    assert dict(collections.Counter(r['kind'] for r in done))==expect and not pending and not failed
    for kind in ('train_forward','train_backward'):
        current=[r for r in begun.values() if r['kind']==kind]
        assert [r['context']['sample_index'] for r in current]==list(range(1600))
        assert all(r['context']['local_step']==i//8+1 and r['context']['global_step']==12001+i//8 and r['context']['microbatch']==i%8 for i,r in enumerate(current))
    assert [r['context']['global_step'] for r in begun.values() if r['kind']=='optimizer_update']==list(range(12001,12201))
    return dict(arm=arm,status='PASS',attempted=expect,successful=expect,failed=0,pending=0,
        calls_start_unix=min(r['time_unix'] for r in begun.values()),calls_end_unix=max(r['time_unix'] for r in done),
        measured_call_seconds={k:sum(float(r['seconds']) for r in done if r['kind']==k) for k in expect})

def main():
    done=OUT/'completion.json'
    if done.exists():
        d=read(done)
        for p,h in d['source_hashes'].items():assert sha(p)==h,p
        print(json.dumps(dict(status='VERIFIED_COMPLETE',path=str(done))));return
    statuses={arm:read(EXP/'training'/arm/'status.json') for arm in ARMS}
    assert all(x['status']=='COMPLETE' and x['local_step']==200 and x['global_step']==12200 for x in statuses.values()),'Wait for both formal arms COMPLETE; no partial final audit'
    reg=read(EXP/'registration.json');regsha=sha(EXP/'registration.json');source_hashes={str(EXP/'registration.json'):regsha}
    plans={};per_update=[];per_sample=[];per_parent=[];arm_summary=[];initializations={};otcounts=[]
    group_path=ROOT/'experiments/r2_target_cleanup_v1/exact_duplicate_groups.csv'
    with group_path.open(encoding='utf-8-sig',newline='') as f:duplicate_rows=list(csv.DictReader(f))
    winding=collections.Counter();canonical_equal=0
    for row in duplicate_rows:
        a,b=json.loads(row['source_winding_rows']);ca,cb=json.loads(row['canonical_vertex_id_rows'])
        if any(a==b[j:]+b[:j] for j in range(3)):winding['same']+=1
        elif any(a==(b[j:]+b[:j])[::-1] for j in range(3)):winding['opposite']+=1
        else:winding['other']+=1
        canonical_equal+=ca==cb
    duplicate_winding=dict(groups=len(duplicate_rows),source_winding_counts=dict(winding),canonical_vertex_rows_identical=canonical_equal,
        interpretation='Exact coincident opposite-winding source copies may encode two-sided thin surfaces; removing one preserves unoriented surface union, not face multiplicity or two-sided source semantics.')
    source_hashes[str(group_path)]=sha(group_path)
    for arm in ARMS:
        folder=EXP/'training'/arm;planpath=EXP/'plans'/(arm+'.json');plan=read(planpath);plans[arm]=plan
        assert plan['registration_sha256']==regsha and len(plan['records'])==1600
        source_hashes[str(planpath)]=sha(planpath)
        snapshots={name:rows(folder/(name+'.jsonl')) for name in ('calls','training_log','diagnostics','exposure')}
        for name in snapshots:source_hashes[str(folder/(name+'.jsonl'))]=sha(folder/(name+'.jsonl'))
        source_hashes[str(folder/'status.json')]=sha(folder/'status.json')
        audit=ledger_audit(arm,snapshots['calls'])
        log,diag,exposure=[snapshots[k] for k in ('training_log','diagnostics','exposure')]
        assert len(log)==200 and len(diag)==len(exposure)==1600
        assert [r['local_step'] for r in log]==list(range(1,201))
        assert [r['sample_index'] for r in diag]==[r['sample_index'] for r in exposure]==list(range(1600))
        parents=collections.Counter()
        for i,(p,e,d) in enumerate(zip(plan['records'],exposure,diag)):
            assert all(p[k]==e[k] for k in ('sample_index','uid','task_id','N','K','alpha','t','step'))
            assert d['local_step']==p['step'] and d['global_step']==p['global_step']
            assert all(d[k]==e[k] for k in ('sample_index','uid','N','K','t','input_bytehash','label_bytehash'))
            count=9*(p['N']-p['K']);assert count==e['free_coordinate_count']==d['free_coordinates']
            den=log[i//8]['free_coordinate_denominator']
            assert close(d['velocity_sse'],d['FM_contribution']*den)
            assert close(d['endpoint_scaled_sse'],(1-float(d['t']))**2*d['velocity_sse'])
            parents[p['uid']]+=1
            per_sample.append(dict(arm=arm,sample_index=i,local_step=p['step'],uid=p['uid'],task_id=p['task_id'],N=p['N'],K=p['K'],alpha=p['alpha'],t=p['t'],
                free_coordinates=count,velocity_sse=d['velocity_sse'],velocity_mse=d['velocity_sse']/count,
                endpoint_scaled_sse=d['endpoint_scaled_sse'],endpoint_scaled_mse=d['endpoint_scaled_sse']/count,
                timestep_bin=min(4,int(float(p['t'])*5)),window50=(p['step']-1)//50,
                FM_contribution=d['FM_contribution'],input_bytehash=e['input_bytehash'],label_bytehash=e['label_bytehash']))
        assert len(parents)==16 and set(parents.values())=={100}
        assert sum(e['free_coordinate_count'] for e in exposure)==plan['free_coordinate_count']
        for j,r in enumerate(log):
            es=exposure[j*8:j*8+8];ds=diag[j*8:j*8+8]
            assert len({e['uid'] for e in es})==8 and sum(e['alpha']==1 for e in es)==4
            assert r['global_step']==12001+j and r['free_coordinate_denominator']==sum(e['free_coordinate_count'] for e in es)
            assert close(r['FM'],sum(d['FM_contribution'] for d in ds))
            assert np.isfinite([r['FM'],r['gradient_norm']]).all()
            per_update.append(dict(arm=arm,**r,window50=j//50))
        for uid,count in sorted(parents.items()):
            es=[e for e in exposure if e['uid']==uid]
            per_parent.append(dict(arm=arm,uid=uid,exposures=count,N=es[0]['N'],K=es[0]['K'],clean_exposures=sum(e['alpha']==1 for e in es),free_coordinate_sum=sum(e['free_coordinate_count'] for e in es)))
        inits=[]
        for p in sorted(folder.glob('attempts/*/initialization.json')):
            x=read(p);source_hashes[str(p)]=sha(p);inits.append(x)
        assert inits and inits[0]['global_step']==12000
        first=inits[0]
        for key,rkey in [('generator_sha256','source_generator_sha256'),('optimizer_sha256','source_optimizer_sha256'),('rng_sha256','source_rng_sha256'),('source_file_sha256','source_checkpoint_sha256')]:assert first[key]==reg[rkey]
        initializations[arm]=inits
        failures=list(folder.glob('attempts/*/failure.json'))
        assert not failures, 'Failure records must be explicitly reviewed rather than silently PASS'
        receipts=sorted((EXP/'ot'/arm).glob('[0-9][0-9][0-9][0-9].json'))
        assert len(receipts)==200
        for j,p in enumerate(receipts):
            receipt=read(p);assert receipt['status']=='COMPLETE' and receipt['OT_returns']==8
            assert receipt['identity']['plan_sha256']==sha(planpath)
            assert receipt['identity']['manifest_sha256']==reg['datasets'][arm]['sha256']
            assert sha(p.with_suffix('.npz'))==receipt['sha256']
            source_hashes[str(p)]=sha(p);source_hashes[str(p.with_suffix('.npz'))]=receipt['sha256']
        otcounts.append(dict(arm=arm,complete_cache_batches=200,registered_sample_maps=1600,recomputed_in_audit=0))
        audit.update(exposures=1600,unique_parents=16,per_parent_exposures=100,free_coordinate_sum=plan['free_coordinate_count'],
            initialization_attempts=len(inits),failures=0,final_status=statuses[arm]['status'],wall_seconds=statuses[arm]['wall_seconds'],
            maximum_update_cuda_reserved_bytes=max(r['cuda_peak_reserved_bytes'] for r in log),maximum_update_cuda_allocated_bytes=max(r['cuda_peak_allocated_bytes'] for r in log),
            allocator_peak_scope='Per-update PyTorch allocated/reserved; not full process VRAM',gpu_observer=statuses[arm].get('resources'))
        arm_summary.append(audit)
    for a,b in zip(plans[ARMS[0]]['records'],plans[ARMS[1]]['records']):
        assert all(a[k]==b[k] for k in ('sample_index','uid','task_id','K','alpha','t','t_seed','gaussian_seed','epsilon_seed','permutation_seed'))
    for key in ('generator_sha256','optimizer_sha256','rng_sha256','hyperparameters'):
        assert initializations[ARMS[0]][0][key]==initializations[ARMS[1]][0][key]
    # This is the only checkpoint-reading block, after all non-weight checks.
    sys.path.insert(0,str(ROOT/'r2/src'))
    import torch
    from meshflow_control.checkpoints import state_hash
    from meshflow_control.training.schedule40 import tree_hash
    checkpoint_audits=[]
    for arm in ARMS:
        receipt=statuses[arm]['checkpoint'];path=resolve(receipt['path']);assert sha(path)==receipt['sha256'];source_hashes[str(path)]=receipt['sha256']
        p=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        assert p['schema']=='r2_target_dedup_c200_training_v1' and p['arm']==arm
        assert p['local_step']==200 and p['global_step']==12200 and p['update_in_progress'] is False and p['EMA'] is False
        assert p['registration_sha256']==regsha and p['plan_sha256']==sha(EXP/'plans'/(arm+'.json'))
        for key,hkey,fn in [('generator_state','generator_state_sha256',state_hash),('optimizer','optimizer_state_sha256',tree_hash),('rng','rng_sha256',tree_hash)]:assert fn(p[key])==p[hkey]==receipt[hkey]
        steps={float(v['step']) for v in p['optimizer']['state'].values()};assert steps=={12200.}
        assert len(p['optimizer']['state'])==len(p['optimizer_parameter_names'][0])
        assert p['generator_state_sha256']!=reg['source_generator_sha256']
        assert all(torch.isfinite(v).all().item() for v in p['generator_state'].values() if isinstance(v,torch.Tensor) and v.is_floating_point())
        checkpoint_audits.append(dict(arm=arm,path=str(path),sha256=receipt['sha256'],cpu_loads=1,local_step=200,global_step=12200,
            optimizer_parameter_states=len(p['optimizer']['state']),optimizer_steps=sorted(steps),generator_state_sha256=p['generator_state_sha256'],optimizer_state_sha256=p['optimizer_state_sha256'],rng_sha256=p['rng_sha256'],status='PASS'))
        del p
    groups=collections.defaultdict(list)
    for s in per_sample:groups[(s['arm'],s['window50'],s['timestep_bin'])].append(s)
    curves=[]
    for (arm,window,tbin),ss in sorted(groups.items()):
        den=sum(s['free_coordinates'] for s in ss);vs=sum(s['velocity_sse'] for s in ss);ys=sum(s['endpoint_scaled_sse'] for s in ss)
        curves.append(dict(arm=arm,step_min=50*window+1,step_max=50*(window+1),t_min=.2*tbin,t_max=.2*(tbin+1),samples=len(ss),free_coordinates=den,
            velocity_mse=vs/den,endpoint_scaled_mse=ys/den,mean_t=float(np.mean([s['t'] for s in ss])),source='DETACHED_NORMAL_TRAIN_FORWARD_RESCALING_NOT_FREE_ROLLOUT'))
    window_curves=[]
    for arm in ARMS:
        for window in range(4):
            us=[r for r in per_update if r['arm']==arm and r['window50']==window];ss=[r for r in per_sample if r['arm']==arm and r['window50']==window]
            window_curves.append(dict(arm=arm,step_min=window*50+1,step_max=(window+1)*50,mean_update_FM=float(np.mean([r['FM'] for r in us])),
                pooled_velocity_mse=sum(r['velocity_sse'] for r in ss)/sum(r['free_coordinates'] for r in ss),
                pooled_endpoint_scaled_mse=sum(r['endpoint_scaled_sse'] for r in ss)/sum(r['free_coordinates'] for r in ss),
                free_coordinate_sum=sum(r['free_coordinates'] for r in ss),samples=len(ss)))
    for name,vals in [('per_update',per_update),('per_sample_detached',per_sample),('per_parent_exposure',per_parent),('time_bin_learning_curve',curves),('learning_curve_50_updates',window_curves),('ot_cache_audit',otcounts),('checkpoint_audit',checkpoint_audits)]:csvwrite(OUT/(name+'.csv'),vals)
    write(OUT/'arm_execution_audit.json',arm_summary);write(OUT/'initializations.json',initializations)
    write(OUT/'input_duplicate_winding_audit.json',duplicate_winding)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for arm in ARMS:
        ws=[r for r in window_curves if r['arm']==arm]
        axes[0].plot([r['step_max'] for r in ws],[r['mean_update_FM'] for r in ws],marker='o',label=arm)
        axes[1].plot([r['step_max'] for r in ws],[r['pooled_endpoint_scaled_mse'] for r in ws],marker='o',label=arm)
    axes[0].set(title='Mean training FM per 50 updates',xlabel='Local update',ylabel='Training FM')
    axes[1].set(title='Detached endpoint-scaled error',xlabel='Local update',ylabel='Pooled training endpoint MSE')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Different target N / denominators: curves do not rank free-generation quality',fontsize=10)
    fig.savefig(checked(OUT/'learning_curve.png'),dpi=160);plt.close(fig)
    for p,h in {**reg['protected'],**reg['training_code_hashes']}.items():assert sha(p)==h,p;source_hashes[p]=h
    source_hashes[str(Path(__file__))]=sha(__file__)
    for p,h in source_hashes.items():assert sha(p)==h,p
    completion=dict(status='PASS',arms=arm_summary,checkpoint_audits=checkpoint_audits,source_hashes=source_hashes,
        same_initial_generator_AdamW_RNG=True,same_parent_and_random_seed_stream=True,updates_total=400,train_forward_total=3200,train_backward_total=3200,
        original_free_scalar_exposures=plans[ARMS[0]]['free_coordinate_count'],dedup_free_scalar_exposures=plans[ARMS[1]]['free_coordinate_count'],
        final_checkpoint_CPU_loads=2,new_OT=0,new_model_calls=0,new_scoring=0,input_duplicate_winding=duplicate_winding,
        interpretation='Different target N, OT dimensions, repeated-surface weights and free-coordinate denominators. Detached errors derive from ordinary supervised interpolation forwards. Cross-arm lower FM is not proof of better free generation; endpoint probe is not rollout.')
    write(done,completion)
    print(json.dumps({k:v for k,v in completion.items() if k not in ('source_hashes','arms','checkpoint_audits')},indent=2))

if __name__=='__main__':main()
