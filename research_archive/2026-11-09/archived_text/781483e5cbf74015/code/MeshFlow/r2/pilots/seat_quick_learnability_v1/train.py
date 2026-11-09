"""Two fixed 200-update EXP67 continuations. No change to FM arithmetic."""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='2'
from pathlib import Path
import sys,json,time,copy,hashlib,argparse,subprocess,signal,traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import torch
from meshflow_control import runtime
from meshflow_control.config import MODEL_CONFIG
from meshflow_control.models.native import build_model,native_execution
from meshflow_control.checkpoints import state_hash,rng_state,restore_rng
from meshflow_control.training import schedule40 as base
from meshflow_control.training.context_checkpoint import _cpu_copy
from meshflow_control.training.losses import masked_fm_loss
from meshflow_control.data.dataset import collate,model_inputs,training_time
from meshflow_control.data.io import read_json,atomic_json,freeze_json,append_event,digest,save_npz,json_hash
from r2.pilots.user_data_scaling_32_67_v1 import training_data as olddata
from r2.pilots.user_condition_finetune_v1 import trainer as support

ROOT=Path('/mnt/e/MeshFlow')
OUT=ROOT/'experiments/r2_seat_quick_learnability_v1'
OLD=ROOT/'experiments/r2_user_data_scaling_32_67_v1'
SOURCE=OLD/'checkpoints/USER_EXPANDED_67_step6000.pt'
DATA=OLD/'data/train_manifest.json'
ARMS=('SMALL_C200','FULL_C200')
EXPERIMENT='R2_SEAT_QUICK_LEARNABILITY_V1'

def seed(*x):
    return int.from_bytes(hashlib.sha256(json.dumps([EXPERIMENT,*x],separators=(',',':')).encode()).digest()[:4],'little')

def resource():
    import psutil
    r=dict(unix=time.time(),cpu_cores=os.cpu_count(),ram_available_bytes=psutil.virtual_memory().available,
           cpu_percent=psutil.cpu_percent(),load_average=os.getloadavg())
    for name,query in [('gpu','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu'),
                       ('processes','--query-compute-apps=pid,process_name,used_memory')]:
        p=subprocess.run(['nvidia-smi',query,'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
        r[name]=dict(returncode=p.returncode,stdout=p.stdout.strip(),stderr=p.stderr.strip())
    try:r['free_mib']=float(r['gpu']['stdout'].splitlines()[0].split(',')[3])
    except (KeyError,IndexError,ValueError):r['free_mib']=None
    return r

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    m=read_json(OUT/'inputs/manifest.json')
    selected=m.get('focused_task_ids')
    if selected is None: raise ValueError('Prepared manifest must provide focused_task_ids')
    d=olddata.Dataset(DATA)
    if len(selected)!=4 or len({d.tasks[x]['uid'] for x in selected})!=4: raise ValueError('Four distinct selected parents required')
    receipt=read_json(SOURCE.with_suffix('.json'))
    if digest(SOURCE)!=receipt['sha256']:raise ValueError('Source checkpoint hash mismatch')
    for t in selected:d.verify_condition(t)
    reg=dict(experiment=EXPERIMENT,source_checkpoint=str(SOURCE),source_checkpoint_sha256=receipt['sha256'],
        source_generator_sha256=receipt['generator_state_sha256'],source_global_step=6000,endpoint_global_step=6200,
        source_data=str(DATA),source_data_sha256=digest(DATA),inputs_manifest_sha256=digest(OUT/'inputs/manifest.json'),
        arms=list(ARMS),updates_per_arm=200,batch=8,microbatch=1,train_task_ids=selected,
        small_parent_policy='four fixed parents twice each per batch; independent noise/time per exposure',
        full_parent_policy='67 parents exposure balanced, eight distinct per batch; uniform unique C within parent',
        augmentation='four clean/four x-width U(.9,1.1), shared between arms per slot',
        training_randomness='new stateless namespace; same time/augmentation/random stream seed identities per slot; different K means different arrays and OT',
        evaluation='eight frozen existing requests, two held-out evaluation noise streams per C; training-seen C diagnostic only',
        optimizer='restore same source AdamW, lr1e-5 betas(.9,.95), weight_decay0, clip1; no EMA/schedule/head',
        generator='unchanged Native/T1+role+Geo, free-only FM, OT then HYBRID lambda .25',
        runtime='FP32 parameters/AdamW; BF16 forward; SDPA math; strict deterministic; TF32 off',
        gpu_budget_gib=6,concurrency=1,headroom_mib=2048,
        retry='no failed/ambiguous model call replay; safe boundary resume only',
        checkpoint_policy='100 recovery point and fixed 200 endpoint; no best selection',
        extra_model_probes=0,new_generator_rollouts=16,new_D0_forwards=16)
    freeze_json(OUT/'registration.json',reg)
    for arm in ARMS:
        tasks=sorted(selected);counts=Counter();rng=np.random.RandomState(seed('parents',arm));crng=np.random.RandomState(seed('condition',arm))
        parents=sorted(d.manifest['pools']['USER_EXPANDED_67']);by={u:sorted(k for k,t in d.tasks.items() if t['uid']==u) for u in parents}
        rows=[]
        for b in range(200):
            if arm=='SMALL_C200': chosen=[tasks[int(j)] for j in rng.permutation(np.tile(np.arange(4),2))]
            else:
                shuffled=rng.permutation(len(parents));pick=sorted(shuffled,key=lambda j:counts[parents[j]])[:8]
                chosen=[]
                for j in pick:
                    uid=parents[j];opts=by[uid];chosen.append(opts[int(crng.randint(len(opts)))])
            arng=np.random.RandomState(seed('augmentation',b));alpha=np.r_[np.ones(4,np.float32),arng.uniform(.9,1.1,4).astype(np.float32)];alpha=alpha[arng.permutation(8)]
            if np.count_nonzero(alpha==1)!=4:raise ValueError('Ambiguous augmentation draw; no redraw')
            for slot,tid in enumerate(chosen):
                t=d.tasks[tid];uid=t['uid'];counts[uid]+=1;i=b*8+slot;ts=seed('time',i)
                rows.append(dict(sample_index=i,effective_batch_index=b,step=b+1,global_step=6001+b,slot_index=slot,arm=arm,
                    uid=uid,task_id=tid,N=t['N'],K=t['K'],ratio=t['K']/t['N'],alpha=float(alpha[slot]),
                    t_seed=ts,t=float(training_time(ts)),gaussian_seed=seed('gaussian',i),epsilon_seed=seed('epsilon',i),
                    permutation_seed=seed('permutation',i),condition_type=t['condition_type'],region_type='seat' if t['condition_type'].startswith('SEAT') else 'back'))
        if max(counts.values())-min(counts.values())>1:raise ValueError('Unequal parent exposure')
        freeze_json(OUT/'plans'/f'{arm}.json',dict(arm=arm,records=rows,parent_counts=dict(counts),
            condition_counts=dict(Counter(r['task_id'] for r in rows)),registration_sha256=digest(OUT/'registration.json')))
    atomic_json(OUT/'resources_before_training.json',resource())
    return reg

def batch(d,records,arm,index):
    folder=OUT/'ot'/arm;folder.mkdir(parents=True,exist_ok=True);path=folder/f'{index:03d}.npz'
    key=json_hash(records);receipt=path.with_suffix('.json');attempt=path.with_suffix('.attempt.json')
    if receipt.exists():
        r=read_json(receipt)
        if r['records_sha256']!=key or r['sha256']!=digest(path):raise ValueError('OT cache changed')
        with np.load(path,allow_pickle=False) as z:maps={k:z[k].copy() for k in z.files}
    else:
        if attempt.exists() or path.exists():raise RuntimeError('Incomplete OT attempt; explicit repair required')
        freeze_json(attempt,dict(records_sha256=key,unix=time.time()));maps=None
    builder=olddata.Builder(d,maps)
    samples=[olddata.validate_sample(builder.sample(r),r) for r in records]
    if maps is None:
        save_npz(path,**builder.new_maps);atomic_json(receipt,dict(records_sha256=key,sha256=digest(path),actual_OT=8))
    return samples

def update(model,opt,samples,ledger,arm,step,hyper):
    # Same microbatch sum/free-coordinate denominator, backward and AdamW as
    # user_condition_finetune_v1.trainer.update; only parent-exposure validation differs.
    den=support.validate_samples(samples,arm,step,effective_batch=False)
    parents=Counter(s['object_id'] for s in samples)
    if sum(s['alpha']==1 for s in samples)!=4:raise ValueError('Four clean exposures required')
    if arm=='SMALL_C200' and sorted(parents.values())!=[2,2,2,2]:raise ValueError('Small pool repeats differ')
    if arm=='FULL_C200' and len(parents)!=8:raise ValueError('Full pool distinct parents differ')
    g=6000+step;support.validate_optimizer(model,opt,g-1,hyper)
    torch.cuda.reset_peak_memory_stats();opt.zero_grad(set_to_none=True);tick=time.perf_counter()
    result=dict(FM=0.,free_coordinate_denominator=den,forward_seconds=0.,backward_seconds=0.,sample_losses=[])
    for micro,s in enumerate(samples):
        b=collate([s],'cuda');ctx=dict(local_step=step,global_step=g,microbatch=micro,sample_index=s['sample_index'],input_bytehash=s['input_bytehash'],label_bytehash=s['label_bytehash'])
        with native_execution(model,coordinates=(b['xt'],b['t']),autocast=True):
            v,sec=ledger.call('train_forward',lambda:model(*model_inputs(b)).float(),ctx);result['forward_seconds']+=sec
            loss=masked_fm_loss(v,b['u'],b['valid_mask'],b['known_mask'],denominator=den)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite FM')
            value=float(loss.detach());_,sec=ledger.call('train_backward',loss.backward,ctx);result['backward_seconds']+=sec
        result['FM']+=value;result['sample_losses'].append(dict(sample_index=s['sample_index'],FM_contribution=value));del v,loss,b
    p=list(model.parameters())
    if any(x.grad is None for x in p):raise RuntimeError('Unused generator parameter')
    result['gradient_norm']=float(torch.nn.utils.clip_grad_norm_(p,1.,error_if_nonfinite=True))
    _,result['optimizer_seconds']=ledger.call('optimizer_update',opt.step,dict(local_step=step,global_step=g))
    if any(not bool(torch.isfinite(x).all()) for x in p):raise FloatingPointError('Nonfinite updated generator')
    support.validate_optimizer(model,opt,g,hyper)
    result.update(update_seconds=time.perf_counter()-tick,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return result

def save(path,p):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        r=read_json(path.with_suffix('.json'))
        if digest(path)!=r['sha256']:raise ValueError('Existing checkpoint changed')
        old=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        for key in ('generator_state_sha256','optimizer_state_sha256','rng_sha256','local_step','global_step','arm','plan_sha256','protocol'):
            if old.get(key)!=p.get(key):raise ValueError('Existing checkpoint state differs: '+key)
        return r
    tmp=path.with_suffix('.partial')
    with tmp.open('xb') as f:torch.save(p,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path);r=dict(path=str(path),sha256=digest(path),generator_state_sha256=p['generator_state_sha256'],local_step=p['local_step'],global_step=p['global_step'])
    atomic_json(path.with_suffix('.json'),r);return r

def train(arm):
    if arm not in ARMS:raise ValueError(arm)
    reg=read_json(OUT/'registration.json');folder=OUT/'training'/arm;folder.mkdir(parents=True,exist_ok=True)
    if digest(DATA)!=reg['source_data_sha256'] or digest(OUT/'inputs/manifest.json')!=reg['inputs_manifest_sha256']:raise ValueError('Frozen data/input identity changed')
    frozen_plan=read_json(OUT/'plans'/f'{arm}.json')
    if frozen_plan['registration_sha256']!=digest(OUT/'registration.json') or frozen_plan['arm']!=arm:raise ValueError('Plan/registration identity changed')
    status=folder/'status.json';previous=read_json(status) if status.exists() else None
    if previous and previous['status']=='COMPLETE':return previous
    if previous and previous['status']!='INTERRUPTED_SAFE':raise RuntimeError('Unsafe prior attempt; no replay')
    r=resource();atomic_json(folder/f'resources_dispatch_{int(time.time())}.json',r)
    # Old comparable reserved peak 3.13GiB + context/workspace margin: estimate
    # process 4GiB; 20% inflation + 2GiB device headroom required before loading.
    if r['free_mib'] is None or r['free_mib']<1.2*4096+2048:raise RuntimeError('RESOURCE_WAIT: need >=6963 MiB free before load')
    torch.set_num_threads(2);runtime.configure_stable_runtime()
    # Allocator ceiling leaves room for CUDA context inside the user's ~6GiB allocation.
    torch.cuda.set_per_process_memory_fraction(5.0*1024**3/torch.cuda.get_device_properties(0).total_memory)
    plan=read_json(OUT/'plans'/f'{arm}.json');d=olddata.Dataset(DATA)
    ledger=support.CallLedger(folder/'calls.jsonl',dict(train_forward=1600,train_backward=1600,optimizer_update=200),arm,f'attempt_{int(time.time())}')
    step=0;start=time.perf_counter();flag={'stop':False};state=dict(status='LOADING',arm=arm,local_step=0)
    def interrupt(*args):flag['stop']=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,interrupt)
    observer=support.GPUObserver('quick_train_'+arm,interval=3.)
    pending={};pool=ThreadPoolExecutor(max_workers=2)
    try:
        with observer:
            source=Path(previous['checkpoint']['path']) if previous else SOURCE
            expected=previous['checkpoint']['sha256'] if previous else reg['source_checkpoint_sha256']
            if digest(source)!=expected:raise ValueError('Checkpoint file changed')
            p=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
            if previous and (p['arm']!=arm or p['protocol']!=reg or p['plan_sha256']!=digest(OUT/'plans'/f'{arm}.json')):raise ValueError('Resume branch/protocol/stream changed')
            step=p['local_step'] if previous else 0;ledger.require_exact_cursor(step)
            if p['global_step']!=6000+step or p['model_config']!=MODEL_CONFIG:raise ValueError('Wrong initialization')
            model=build_model(copy.deepcopy(MODEL_CONFIG),readout_mode='none',trainable=True).eval()
            if p['optimizer_parameter_names']!=base.optimizer_names(model,None):raise ValueError('Parameter order differs')
            model.load_state_dict(p['generator_state'],strict=True);model.to('cuda');opt=base.make_optimizer(model,None);opt.load_state_dict(p['optimizer']);restore_rng(p['rng'])
            if state_hash(model.state_dict())!=p['generator_state_sha256'] or base.tree_hash(opt.state_dict())!=p['optimizer_state_sha256'] or base.tree_hash(rng_state())!=p['rng_sha256']:raise ValueError('Exact generator/AdamW/RNG restoration failed')
            hyper=support.optimizer_hyperparameters(opt);support.validate_optimizer(model,opt,6000+step,hyper)
            lineage=dict(source_file_sha256=reg['source_checkpoint_sha256'],source_generator_sha256=reg['source_generator_sha256'],optimizer_restored=True,rng_restored=True)
            atomic_json(folder/'initialization.json',dict(**lineage,loaded_generator_sha256=p['generator_state_sha256'],loaded_optimizer_sha256=p['optimizer_state_sha256'],loaded_rng_sha256=p['rng_sha256'],restored_global_step=6000+step,hyperparameters=hyper))
            del p
            state.update(status='TRAINING',local_step=step,global_step=6000+step);atomic_json(status,state)
            def checkpoint():
                ledger.require_exact_cursor(step)
                g,o,r=_cpu_copy(model.state_dict()),_cpu_copy(opt.state_dict()),_cpu_copy(rng_state())
                p=dict(schema='r2_seat_quick_full_v1',experiment=EXPERIMENT,arm=arm,local_step=step,global_step=6000+step,
                    generator_state=g,generator_state_sha256=state_hash(g),optimizer=o,optimizer_state_sha256=base.tree_hash(o),
                    rng=r,rng_sha256=base.tree_hash(r),optimizer_parameter_names=base.optimizer_names(model,None),model_config=copy.deepcopy(MODEL_CONFIG),
                    protocol=reg,lineage=lineage,plan_sha256=digest(OUT/'plans'/f'{arm}.json'),execution=ledger.snapshot())
                path=OUT/'checkpoints'/f'{arm}_step{step}.pt';state['checkpoint']=save(path,p)
                if step==200:
                    q={k:p[k] for k in ('experiment','arm','local_step','global_step','generator_state','generator_state_sha256','model_config','lineage')}
                    q.update(schema='r2_seat_quick_generator_v1',inference_contract=dict(readout_mode='none',forward='BF16',parameters='FP32',integration='FP32'))
                    state['inference_export']=save(OUT/'inference'/f'{arm}_generator_step200.pt',q)
            for current in range(step+1,201):
                # Check safe boundary before any new model call. Resource pressure
                # saves a full checkpoint and releases this worker, never other jobs.
                if current==1 or current%5==0 or flag['stop']:
                    rr=resource();append_event(folder/'resource_checks.jsonl',rr)
                    if flag['stop'] or rr['free_mib'] is None or rr['free_mib']<2048:
                        checkpoint();state['status']='INTERRUPTED_SAFE';state['reason']='signal_or_resource_headroom';break
                tick=time.perf_counter()
                for b in range(current-1,min(200,current+1)):
                    if b not in pending:pending[b]=pool.submit(batch,d,plan['records'][b*8:(b+1)*8],arm,b)
                samples=pending.pop(current-1).result();input_s=time.perf_counter()-tick
                result=update(model,opt,samples,ledger,arm,current,hyper);step=current
                append_event(folder/'training_log.jsonl',dict(step=step,global_step=6000+step,input_seconds=input_s,**result))
                state.update(local_step=step,global_step=6000+step,last_FM=result['FM'],peak_reserved_bytes=result['cuda_peak_reserved_bytes'],wall_seconds=time.perf_counter()-start)
                if step in (100,200):checkpoint()
                atomic_json(status,state)
                if step==1 or step%25==0:print(arm,step,'FM',result['FM'],'seconds',round(result['update_seconds'],2),flush=True)
            else:state['status']='COMPLETE'
    except BaseException as e:
        state.update(status='FAILED',error=repr(e),traceback=traceback.format_exc(),no_automatic_retry=True);raise
    finally:
        pool.shutdown(wait=True);state.update(local_step=step,global_step=6000+step,wall_seconds=time.perf_counter()-start,calls=ledger.snapshot(),resources=observer.summary())
        atomic_json(folder/'resources.json',dict(samples=observer.samples,summary=observer.summary()));atomic_json(status,state)
    return state

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','resources']);p.add_argument('--arm',choices=ARMS);a=p.parse_args()
    print(json.dumps(prepare() if a.stage=='prepare' else resource() if a.stage=='resources' else train(a.arm),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
