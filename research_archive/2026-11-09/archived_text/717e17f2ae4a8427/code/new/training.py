"""Finite manifest-driven R2 USER scaling training; controller dispatches one GPU.

This adapter delegates all update arithmetic to schedule40.update(COMMON).
It never changes old sources, starts sampling, changes defaults or selects best.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
from pathlib import Path
import sys,copy,json,time,signal,argparse,traceback,importlib.util
from contextlib import contextmanager
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from meshflow_control import runtime
import torch
import numpy as np
from meshflow_control.models.native import build_model,native_execution
from meshflow_control.config import MODEL_CONFIG
from meshflow_control.checkpoints import state_hash,rng_state,restore_rng
from meshflow_control.training import schedule40 as base,recipe_factorial as recipe
from meshflow_control.training.context_checkpoint import _cpu_copy,_check_generator
from meshflow_control.training.losses import masked_fm_loss
from meshflow_control.data.dataset import collate,model_inputs
from meshflow_control.data.io import atomic_json,read_json,digest,json_hash,append_event
import training_data as data
import training_diagnostics as diagnostics

_spec=importlib.util.spec_from_file_location('_scaling_ledger_support',Path(__file__).resolve().parents[1]/'user_condition_finetune_v1/trainer.py')
support=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=support;_spec.loader.exec_module(support)
EXPERIMENT=data.EXPERIMENT
SCHEMA='r2_user_scaling_training_v1'
INFERENCE_SCHEMA='r2_user_scaling_generator_v1'
INITIAL_SCHEMA='r2_user_scaling_common_initialization_v1'

def protocol(updates):
    if updates not in (4000,6000):raise ValueError('Only common 4000 or 6000 endpoint allowed')
    return dict(experiment=EXPERIMENT,updates_per_arm=updates,initialization='same official generator EMA + role/Geo; fresh AdamW step0',
        official_generator_sha256=base.OFFICIAL_FILE_SHA256,official_EMA_state_sha256=base.OFFICIAL_BACKBONE_SHA256,
        model=copy.deepcopy(MODEL_CONFIG),model_seed=base.MODEL_SEED,geo_seed=1010,batch=8,microbatch=1,
        distinct_parents=8,pool_parent_counts={'USER_SMALL':32,'USER_EXPANDED_67':67},conditions='USER SEAT/BACK FULL/HALF only; within-parent uniform after deduplication',
        clean_per_batch=4,width=[.9,1.1],hybrid_lambda=.25,hybrid='OT original Gaussian then mix; no re-OT',
        lr=1e-5,betas=[.9,.95],weight_decay=0.,clip=1.,scheduler=False,EMA=False,teacher=False,
        auxiliary_loss=False,readout='none',loss='sum free SSE / (9 * sum_batch(N-K))',
        forward='BF16',parameters='FP32',optimizer='FP32',integration='FP32',model_mode='eval with all parameters trainable',
        deterministic_strict=True,tf32=False,sdpa='math',checkpoint_interval=500,rotation_segment=2000,
        preserve_fixed4000=True,endpoint_selection='resource/time lock before either arm; no best checkpoint')

def registration(out):
    out=data.checked_out(out);r=read_json(out/'registration.json')
    if r.get('experiment')!=EXPERIMENT:raise ValueError('Wrong experiment registration')
    for key in ('training_deadline_unix','deadline_unix','official_generator_path'):
        if key not in r:raise ValueError('Registration missing '+key)
    if r['training_deadline_unix']>=r['deadline_unix']:raise ValueError('Training cutoff must precede whole deadline')
    data.load_manifest(out/'data/train_manifest.json')
    return r

def code_identity():
    paths=[Path(__file__),Path(data.__file__),Path(diagnostics.__file__),Path(base.__file__),Path(recipe.__file__),
        Path(support.__file__),Path(build_model.__code__.co_filename),Path(masked_fm_loss.__code__.co_filename),
        Path(sys.modules[data.HybridStream.__module__].__file__)]
    paths=paths+[Path(sys.modules[data.HybridStream.__module__].__file__),Path(sys.modules[data.TrainingDataset.__module__].__file__),
        Path(data.coupling.__file__),Path(runtime.__file__)]
    src=Path(base.__file__).parents[1]
    paths += [src/relative for relative in ('config.py','precision.py','checkpoints.py','data/io.py','models/geometry.py','training/context_checkpoint.py')]
    paths += sorted((src/'models/backbone').glob('*.py'))
    return {str(p.resolve()):digest(p) for p in paths}

@contextmanager
def gpu_owner(out):
    import fcntl
    out=data.checked_out(out)
    lock=data.writable(out,out/'gpu_owner.lock');lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a+') as h:
        fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(h,fcntl.LOCK_UN)

def save(out,path,p,replace=False):
    path=data.writable(out,path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not replace:raise FileExistsError(path)
    tmp=path.with_name(path.name+'.partial')
    if tmp.exists():raise RuntimeError('Prior incomplete checkpoint package: '+str(tmp))
    with tmp.open('xb') as f:torch.save(p,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    r=dict(path=str(path),sha256=digest(path),global_step=p['global_step'],generator_state_sha256=p['generator_state_sha256'])
    atomic_json(path.with_suffix('.json'),r);return r

class Ledger(support.CallLedger):
    def require_exact_cursor(self,step):
        expected=dict(train_forward=8*step,train_backward=8*step,optimizer_update=step)
        if self.pending or self.failed or any(self.attempted[k]!=v or self.successful[k]!=v for k,v in expected.items()):
            raise RuntimeError('Consumed work differs from checkpoint; replay forbidden')
        updates=[r['context']['step'] for r in self.rows if r['kind']=='optimizer_update' and r['event']=='CALL_BEGIN']
        if updates!=list(range(1,step+1)):raise ValueError('Update cursor skipped or replayed')

def initialize(out):
    out=data.checked_out(out);r=registration(out);path=out/'initialization/common_step0000.pt'
    if path.exists():
        receipt=read_json(path.with_suffix('.json'))
        if digest(path)!=receipt['sha256']:raise ValueError('Common initialization changed')
        return receipt
    with gpu_owner(out):
        torch.set_num_threads(2);runtime.configure_stable_runtime()
        model,opt,audit=recipe.initialize_official(r['official_generator_path'],device='cuda')
        generator=_cpu_copy(model.state_dict());_check_generator(generator)
        state=_cpu_copy(rng_state());optimizer=_cpu_copy(opt.state_dict())
        if optimizer['state']:raise AssertionError('Common initialization inherited AdamW moments')
        p=dict(schema=INITIAL_SCHEMA,experiment=EXPERIMENT,global_step=0,model_config=copy.deepcopy(MODEL_CONFIG),
            generator_state=generator,generator_state_sha256=state_hash(generator),optimizer=optimizer,
            optimizer_state_sha256=base.tree_hash(optimizer),rng=state,rng_sha256=base.tree_hash(state),
            lineage=audit,manifest_sha256=digest(out/'data/train_manifest.json'))
        receipt=save(out,path,p)
        atomic_json(out/'initialization/audit.json',dict(**audit,**receipt,code_hashes=code_identity(),
            model_mode='eval',dropout_modules=[dict(name=n,p=float(m.p),training=m.training) for n,m in model.named_modules() if isinstance(m,torch.nn.Dropout)],
            all_parameters_FP32=all(x.dtype==torch.float32 for x in model.parameters()),
            all_parameters_trainable=all(x.requires_grad for x in model.parameters()),forwards=0,backwards=0,updates=0))
    return receipt

def common_model(out):
    path=out/'initialization/common_step0000.pt';receipt=read_json(path.with_suffix('.json'))
    if digest(path)!=receipt['sha256']:raise ValueError('Initialization receipt differs')
    p=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if p['schema']!=INITIAL_SCHEMA or p['global_step']!=0 or p['manifest_sha256']!=digest(out/'data/train_manifest.json'):
        raise ValueError('Common initialization lineage differs')
    if state_hash(p['generator_state'])!=p['generator_state_sha256'] or base.tree_hash(p['rng'])!=p['rng_sha256'] or p['optimizer']['state']:
        raise ValueError('Common state or fresh optimizer differs')
    model=build_model(copy.deepcopy(MODEL_CONFIG),readout_mode='none',trainable=True).eval()
    model.load_state_dict(p['generator_state'],strict=True);model.to('cuda')
    opt=base.make_optimizer(model,None);opt.load_state_dict(p['optimizer']);restore_rng(p['rng'])
    base.validate_optimizer(model,None,opt,'COMMON',0)
    if state_hash(model.state_dict())!=p['generator_state_sha256'] or base.tree_hash(opt.state_dict())!=p['optimizer_state_sha256'] or base.tree_hash(rng_state())!=p['rng_sha256']:
        raise ValueError('Common FP32 parameters/AdamW/RNG did not restore exactly')
    lineage=dict(p['lineage'],common_initialization_sha256=receipt['sha256'],common_rng_sha256=p['rng_sha256'])
    return model,opt,lineage

def validate_batch(samples,step,arm):
    if len(samples)!=8 or len({s['object_id'] for s in samples})!=8 or sum(s['alpha']==1. for s in samples)!=4:
        raise ValueError('Batch parent/augmentation policy differs')
    for s in samples:
        if s['step']!=step or s['global_step']!=step or s['arm']!=arm or s['lambda_value']!=.25:
            raise ValueError('Sample clock/arm/coupling differs')
        if s['free_coordinate_count']!=9*(int(s['y'])-s['K']):raise ValueError('Free denominator differs')
    return sum(s['free_coordinate_count'] for s in samples)

def preflight(out):
    out=data.checked_out(out);reg=registration(out);folder=out/'preflight';folder.mkdir(parents=True,exist_ok=True)
    report=folder/'report.json'
    if report.exists():
        r=read_json(report)
        if r['status']!='PASS' or r['manifest_sha256']!=digest(out/'data/train_manifest.json'):raise ValueError('Preflight identity differs')
        return r
    if (folder/'calls.jsonl').exists():raise RuntimeError('Preflight already attempted; no automatic replay')
    manifest=data.load_manifest(out/'data/train_manifest.json');large='USER_EXPANDED_67'
    diagnostics.freeze(out,out/'data/train_manifest.json')
    ledger=Ledger(folder/'calls.jsonl',dict(train_forward=128,train_backward=128,optimizer_update=16,preflight_parity_forward=32),'PREFLIGHT','initial')
    observer=support.GPUObserver('preflight',interval=2.);stream=None;tick=time.perf_counter()
    state=dict(status='RUNNING',manifest_sha256=digest(out/'data/train_manifest.json'),code_hashes=code_identity(),
        torch_peak_scope='reset before each of 16 real effective updates; eight microbatch F/B and AdamW included; nvidia-smi separately samples process/context',
        preflight_call_limits=dict(forwards=160,backwards=128,optimizer_updates=16),formal_state_reused=False)
    try:
        with gpu_owner(out),observer:
            torch.set_num_threads(2);runtime.configure_stable_runtime();model,opt,lineage=common_model(out)
            initial={group:state_hash({k:v for k,v in model.state_dict().items() if k.startswith(prefix)}) for group,prefix in
                (('backbone','backbone.'),('role','role_embedding'),('Geo','context_encoder.'))}
            fixtures=diagnostics.load(out);parity=[]
            with torch.no_grad():
                for s in fixtures:
                    b=collate([s],'cuda')
                    with native_execution(model,coordinates=(b['xt'],b['t']),autocast=True):
                        a,_=ledger.call('preflight_parity_forward',lambda:model(*model_inputs(b)).float(),dict(fixture=s['sample_index'],path='historical_allowlist'))
                    with native_execution(model,coordinates=(b['xt'],b['t']),autocast=True):
                        c,_=ledger.call('preflight_parity_forward',lambda:model(b['xt'],b['t'],b['y'],b['valid_mask'],b['known_mask']).float(),dict(fixture=s['sample_index'],path='manifest_allowlist'))
                    if not torch.equal(a,c):raise ValueError('Manifest/core fixture forward mismatch')
                    loss=masked_fm_loss(a,b['u'],b['valid_mask'],b['known_mask'])
                    manual=(a-b['u']).square()[b['valid_mask']&~b['known_mask']].sum()/s['free_coordinate_count']
                    if not torch.equal(loss,manual):raise ValueError('Free denominator arithmetic differs')
                    parity.append(dict(fixture=s['sample_index'],N=int(s['y']),K=s['K'],condition_type=s['condition_type'],exact=True))
            stream=data.Stream(out,out/'data/train_manifest.json',large,16,phase='preflight',workers=2)
            logs=[]
            for step in range(1,17):
                if time.time()>=reg['training_deadline_unix']:raise TimeoutError('Training cutoff reached during preflight')
                began=time.perf_counter();samples=stream.next_effective_batch();input_s=time.perf_counter()-began
                validate_batch(samples,step,large)
                if time.time()>=reg['training_deadline_unix']:raise TimeoutError('Training cutoff reached before preflight model dispatch')
                result=base.update(model,None,opt,samples,None,ledger,'COMMON',step)
                row=dict(step=step,input_seconds=input_s,total_step_seconds=time.perf_counter()-began,**result)
                append_event(folder/'steps.jsonl',row);logs.append(row)
            changed={group:state_hash({k:v for k,v in model.state_dict().items() if k.startswith(prefix)})!=initial[group] for group,prefix in
                (('backbone','backbone.'),('role','role_embedding'),('Geo','context_encoder.'))}
            if not all(changed.values()):raise ValueError('Backbone or new parameters did not change')
            ledger.require_exact_cursor(16)
            times=[r['total_step_seconds'] for r in logs[2:]]
            state.update(status='PASS',lineage=lineage,parity=parity,changed_parameter_groups=changed,
                discard_preflight_state=True,updates=16,actual_forward=160,actual_backward=128,
                timing=dict(measured_steps=16,excluded_warmup_for_estimate=2,mean=float(np.mean(times)),p90=float(np.quantile(times,.9)),
                    maximum=max(times),conservative_seconds_per_update=max(times)*1.2+.05),
                cuda_peak_allocated_bytes=max(x['cuda_peak_allocated_bytes'] for x in logs),
                cuda_peak_reserved_bytes=max(x['cuda_peak_reserved_bytes'] for x in logs))
    except BaseException as exc:
        state.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        if stream:stream.close()
        state.update(calls=ledger.snapshot(),wall_seconds=time.perf_counter()-tick,resources=observer.summary())
        atomic_json(folder/'resources.json',dict(samples=observer.samples,summary=observer.summary()))
        atomic_json(report,state)
    return state

def budget(out):
    b=read_json(out/'training_budget_lock.json')
    if b.get('status')!='LOCKED' or b['updates_per_arm'] not in (4000,6000):raise ValueError('Common budget not locked')
    return b

def payload(model,opt,step,stream,identities,lineage,ledger,updates):
    ledger.require_exact_cursor(step);base.validate_optimizer(model,None,opt,'COMMON',step)
    if stream.batch_index!=step:raise ValueError('Stream ahead of completed update')
    g,o,r=_cpu_copy(model.state_dict()),_cpu_copy(opt.state_dict()),_cpu_copy(rng_state());_check_generator(g)
    return dict(schema=SCHEMA,experiment=EXPERIMENT,arm=stream.arm,global_step=step,update_in_progress=False,
        protocol=protocol(updates),model_config=copy.deepcopy(MODEL_CONFIG),generator_state=g,generator_state_sha256=state_hash(g),
        optimizer=o,optimizer_state_sha256=base.tree_hash(o),optimizer_parameter_names=base.optimizer_names(model,None),
        rng=r,rng_sha256=base.tree_hash(r),stream_state=stream.state_dict(),identities=identities,lineage=lineage,execution_receipt=ledger.snapshot())

def validate_payload(p):
    if p['schema']!=SCHEMA or p['experiment']!=EXPERIMENT or p['update_in_progress'] or p['model_config']!=MODEL_CONFIG:
        raise ValueError('Wrong scaling checkpoint')
    if p['protocol']!=protocol(p['protocol']['updates_per_arm']) or not 0<=p['global_step']<=p['protocol']['updates_per_arm']:
        raise ValueError('Checkpoint protocol/cursor differs')
    if p['stream_state']['batch_index']!=p['global_step']:raise ValueError('Checkpoint stream differs')
    for k,h,hkey in (('generator_state',state_hash,'generator_state_sha256'),('optimizer',base.tree_hash,'optimizer_state_sha256'),('rng',base.tree_hash,'rng_sha256')):
        if h(p[k])!=p[hkey]:raise ValueError('Checkpoint state hash differs: '+k)
    _check_generator(p['generator_state']);return p

def export_generator(out,full):
    p=validate_payload(torch.load(full,map_location='cpu',weights_only=True,mmap=True));step=p['global_step']
    if step not in (4000,p['protocol']['updates_per_arm']):raise ValueError('Only fixed 4000/final exports permitted')
    v={k:p[k] for k in ('experiment','arm','global_step','protocol','model_config','generator_state','generator_state_sha256','identities','lineage')}
    v.update(schema=INFERENCE_SCHEMA,training_checkpoint_sha256=digest(full),inference_contract=dict(readout_mode='none',forward='BF16',parameters='FP32',integration='FP32'))
    return save(out,out/'inference'/f'{p["arm"]}_generator_step{step}.pt',v)

def load_generator(checkpoint,device='cuda',expected_sha256=None):
    actual=digest(checkpoint)
    if expected_sha256 is not None and actual!=expected_sha256:raise ValueError('Generator file identity differs')
    p=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    if p['schema']!=INFERENCE_SCHEMA or p['experiment']!=EXPERIMENT or p['global_step'] not in (4000,p['protocol']['updates_per_arm']):
        raise ValueError('Not a fixed scaling inference endpoint')
    if p['protocol']!=protocol(p['protocol']['updates_per_arm']) or p['model_config']!=MODEL_CONFIG:
        raise ValueError('Generator architecture/protocol differs')
    if any(k in p for k in ('optimizer','rng','teacher_state','aligner_state')):raise ValueError('Training tensors in inference export')
    if state_hash(p['generator_state'])!=p['generator_state_sha256']:raise ValueError('Generator state changed')
    _check_generator(p['generator_state']);runtime.configure_stable_runtime()
    model=build_model(copy.deepcopy(MODEL_CONFIG),readout_mode='none',trainable=False).eval()
    model.load_state_dict(p['generator_state'],strict=True);model.to(device).set_trainable(False)
    return model,dict({k:v for k,v in p.items() if k!='generator_state'},file_sha256=actual)

def train(out,arm,stop_step,attempt):
    out=data.checked_out(out);reg=registration(out);lock=budget(out);updates=lock['updates_per_arm']
    if time.time()>=reg['training_deadline_unix']:raise TimeoutError('Training cutoff reached; no GPU dispatch')
    if lock['training_deadline_unix']!=reg['training_deadline_unix']:raise ValueError('Budget cutoff changed')
    pre=read_json(out/'preflight/report.json')
    if pre['status']!='PASS' or pre['manifest_sha256']!=digest(out/'data/train_manifest.json') or pre['code_hashes']!=code_identity():
        raise ValueError('Preflight must pass for the unchanged frozen data and implementation')
    manifest=out/'data/train_manifest.json';m=data.load_manifest(manifest)
    if arm not in m['pools'] or stop_step not in (2000,4000,6000) or stop_step>updates:raise ValueError('Wrong registered physical segment')
    attempt=support.safe_attempt_id(attempt);folder=out/'training'/arm;folder.mkdir(parents=True,exist_ok=True)
    status_path=folder/'status.json';previous=read_json(status_path) if status_path.exists() else None
    if previous and previous['status'] not in ('SEGMENT_COMPLETE','INTERRUPTED_SAFE','TIME_CUTOFF'):
        raise RuntimeError('Existing unsafe/complete run cannot restart')
    ident=dict(manifest_sha256=digest(manifest),budget_sha256=digest(out/'training_budget_lock.json'),
        registration_sha256=digest(out/'registration.json'),common_initialization_sha256=read_json(out/'initialization/common_step0000.json')['sha256'],
        code_hashes=code_identity())
    ledger=Ledger(folder/'calls.jsonl',dict(train_forward=8*updates,train_backward=8*updates,optimizer_update=updates),arm,attempt)
    dledger=Ledger(out/'diagnostics/calls.jsonl',dict(diagnostic_forward=128),'DIAGNOSTICS',attempt)
    adir=folder/'attempts'/attempt;adir.mkdir(parents=True,exist_ok=False)
    state=dict(status='LOADING',arm=arm,global_step=0,attempt=attempt,pid=os.getpid(),checkpoint=None)
    stream=None;step=0;tick=time.perf_counter();observer=support.GPUObserver('train_'+arm,interval=5.)
    flag={'signal':None}
    def interrupted(sig,frame):flag['signal']=sig
    handlers={s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        with gpu_owner(out),observer:
            torch.set_num_threads(2);runtime.configure_stable_runtime();stream=data.Stream(out,manifest,arm,updates,workers=2)
            if previous is None:
                ledger.require_exact_cursor(0);model,opt,lineage=common_model(out)
            else:
                receipt=previous['checkpoint'];path=Path(receipt['path'])
                if digest(path)!=receipt['sha256']:raise ValueError('Safe recovery checkpoint changed')
                p=validate_payload(torch.load(path,map_location='cpu',weights_only=True,mmap=True))
                if p['identities']!=ident or p['arm']!=arm:raise ValueError('Resume identity changed')
                step=p['global_step'];ledger.require_exact_cursor(step)
                model=build_model(copy.deepcopy(MODEL_CONFIG),readout_mode='none',trainable=True).eval()
                model.load_state_dict(p['generator_state'],strict=True);model.to('cuda')
                if p['optimizer_parameter_names']!=base.optimizer_names(model,None):raise ValueError('Saved optimizer parameter order differs')
                opt=base.make_optimizer(model,None);opt.load_state_dict(p['optimizer']);restore_rng(p['rng'])
                if state_hash(model.state_dict())!=p['generator_state_sha256'] or base.tree_hash(opt.state_dict())!=p['optimizer_state_sha256'] or base.tree_hash(rng_state())!=p['rng_sha256']:
                    raise ValueError('Exact model/AdamW/RNG restoration failed')
                stream.load_state_dict(p['stream_state']);lineage=p['lineage'];del p
                base.validate_optimizer(model,None,opt,'COMMON',step)
            if not step<stop_step:raise ValueError('Segment would replay or empty-dispatch')
            state.update(status='TRAINING',global_step=step,lineage=lineage)
            atomic_json(status_path,state)
            def checkpoint():
                start=time.perf_counter();p=payload(model,opt,step,stream,ident,lineage,ledger,updates)
                fixed=step in (4000,updates);name=f'{arm}_step{step}.pt' if fixed else f'{arm}_latest.pt'
                target=out/'checkpoints'/name
                if fixed and target.exists():
                    receipt=read_json(target.with_suffix('.json'))
                    if digest(target)!=receipt['sha256'] or p['generator_state_sha256']!=receipt['generator_state_sha256']:
                        raise ValueError('Previously saved fixed checkpoint differs')
                else:receipt=save(out,target,p,replace=not fixed)
                state['checkpoint']=receipt
                append_event(adir/'events.jsonl',dict(event='CHECKPOINT',seconds=time.perf_counter()-start,**receipt));del p
                if fixed:
                    ep=out/'inference'/f'{arm}_generator_step{step}.pt'
                    if ep.exists():
                        er=read_json(ep.with_suffix('.json'))
                        if digest(ep)!=er['sha256'] or er['generator_state_sha256']!=receipt['generator_state_sha256']:
                            raise ValueError('Previously saved generator differs')
                        state['inference_export']=er
                    else:state['inference_export']=export_generator(out,receipt['path'])
                return receipt
            if step==0:diagnostics.measure(model,out,arm,0,dledger)
            for current in range(step+1,stop_step+1):
                if flag['signal'] is not None or time.time()>=reg['training_deadline_unix']:
                    checkpoint();state['status']='INTERRUPTED_SAFE' if flag['signal'] is not None else 'TIME_CUTOFF';break
                started=time.perf_counter();samples=stream.next_effective_batch(stop_step=stop_step);input_s=time.perf_counter()-started
                validate_batch(samples,current,arm)
                if time.time()>=reg['training_deadline_unix'] or flag['signal'] is not None:
                    # Input OT may have finished while waiting; no model call was made.
                    stream.batch_index-=1
                    append_event(adir/'events.jsonl',dict(event='INPUT_READY_NO_UPDATE_AT_CUTOFF',prepared_step=current,model_calls=0))
                    checkpoint();state['status']='INTERRUPTED_SAFE' if flag['signal'] is not None else 'TIME_CUTOFF';break
                for s in samples:
                    append_event(adir/'exposure.jsonl',dict(step=current,sample_index=s['sample_index'],uid=s['object_id'],
                        task_id=s['task_id'],condition_type=s['condition_type'],N=int(s['y']),K=s['K'],alpha=s['alpha'],t=float(s['t']),
                        free_coordinate_count=s['free_coordinate_count'],input_bytehash=s['input_bytehash'],label_bytehash=s['label_bytehash'],ot_cache_key=s['ot_cache_key']))
                result=base.update(model,None,opt,samples,None,ledger,'COMMON',current);step=current
                append_event(adir/'training_log.jsonl',dict(step=step,input_seconds=input_s,total_step_seconds=time.perf_counter()-started,**result))
                state.update(global_step=step,last_FM=result['FM'],update_seconds=result['update_seconds'],input_seconds=input_s,
                    cuda_peak_allocated_bytes=result['cuda_peak_allocated_bytes'],cuda_peak_reserved_bytes=result['cuda_peak_reserved_bytes'],
                    calls=ledger.snapshot(),wall_seconds=time.perf_counter()-tick)
                if step in (1000,3000,updates):diagnostics.measure(model,out,arm,step,dledger)
                if step%500==0 or step==stop_step:checkpoint()
                atomic_json(status_path,state)
                if step%25==0 or step==1:print(arm,step,'FM',result['FM'],flush=True)
            else:state['status']='COMPLETE' if step==updates else 'SEGMENT_COMPLETE'
    except BaseException as exc:
        state.update(status='OOM' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'BLOCKED',error=repr(exc),traceback=traceback.format_exc(),no_automatic_retry=True)
        atomic_json(adir/'failure.json',state);raise
    finally:
        if stream:stream.close()
        for s,h in handlers.items():signal.signal(s,h)
        state.update(global_step=step,calls=ledger.snapshot(),diagnostic_calls=dledger.snapshot(),wall_seconds=time.perf_counter()-tick,resources=observer.summary())
        atomic_json(adir/'resources.json',dict(samples=observer.samples,summary=observer.summary()))
        atomic_json(status_path,state);atomic_json(adir/'final_status.json',state)
    return state

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['init','fixtures','preflight','train'],required=True)
    p.add_argument('--out',type=Path,default=data.OUT);p.add_argument('--arm');p.add_argument('--stop-step',type=int);p.add_argument('--attempt',default='initial')
    a=p.parse_args()
    if a.stage=='init':r=initialize(a.out)
    elif a.stage=='fixtures':r=diagnostics.freeze(a.out,a.out/'data/train_manifest.json')
    elif a.stage=='preflight':r=preflight(a.out)
    else:r=train(a.out,a.arm,a.stop_step,a.attempt)
    print(json.dumps(r,indent=2))
if __name__=='__main__':main()
