"""One official-initialized USER-majority R2 arm; bounded calls, no retries."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
import sys, json, copy, time, signal, argparse, importlib.util, traceback
from pathlib import Path
CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE.parents[1]/'src'))
from meshflow_control import runtime
import torch
from meshflow_control.training import schedule40 as base, recipe_factorial as recipe
from meshflow_control.config import MODEL_CONFIG
from meshflow_control.models.native import build_model
from meshflow_control.checkpoints import state_hash, rng_state, restore_rng
from meshflow_control.training.context_checkpoint import _cpu_copy, _check_generator
from meshflow_control.data.io import atomic_json, read_json, digest, json_hash, append_event

spec=importlib.util.spec_from_file_location('_user_mix_training_support',CODE.parent/'user_condition_finetune_v1/trainer.py')
support=importlib.util.module_from_spec(spec);sys.modules[spec.name]=support;spec.loader.exec_module(support)
ARM='R2_USER_MIX'
EXPERIMENT='R2_USER_MIX_FROM_OFFICIAL_V1'
SCHEMA='r2_user_mix_official_training_v1'
INFERENCE_SCHEMA='r2_user_mix_official_generator_v1'
TOTAL=6000
LIMITS=dict(train_forward=48000,train_backward=48000,optimizer_update=6000)
OFFICIAL=Path('/mnt/e/MeshFlow/native_t1/runs/control_mvp48_v1/checkpoints/official_chair_last.pt')

def protocol():
    return dict(experiment=EXPERIMENT,arm=ARM,total_updates=TOTAL,initialization='official EMA; fresh AdamW step0',
        official_sha256=base.OFFICIAL_FILE_SHA256,model=MODEL_CONFIG,model_seed=base.MODEL_SEED,
        batch=8,microbatch=1,USER_per_batch=6,OLD_per_batch=2,clean_per_batch=4,width=[.9,1.1],
        parents=25,user_conditions=117,old_conditions=395,hybrid_lambda=.25,hybrid_schedule='all',
        loss='free-only FM; whole-batch valid free scalar denominator',lr=1e-5,betas=[.9,.95],weight_decay=0.,clip=1.,
        forward='BF16',parameters='FP32',integration='FP32',deterministic_strict=True,tf32=False,sdpa='math',
        auxiliary_head=False,teacher=False,readout='none',selection_step=6000,checkpoint_interval=500,
        recovery='exact safe cursor only; no attempted/completed update replay',sample_mix_is_loss_weight=False)

class Ledger(support.CallLedger):
    def require_exact_cursor(self,step):
        expected=dict(train_forward=8*step,train_backward=8*step,optimizer_update=step)
        if self.pending or self.failed or any(self.attempted[k]!=v or self.successful[k]!=v for k,v in expected.items()):
            raise ValueError('Unresolved calls or checkpoint behind consumed updates; replay forbidden')
        steps=[r['context']['step'] for r in self.rows if r['kind']=='optimizer_update' and r['event']=='CALL_BEGIN']
        if steps!=list(range(1,step+1)):raise ValueError('Optimizer cursor skipped/replayed')

def validate_samples(samples,step):
    denominator=support.validate_samples(samples,effective_batch=True)
    from collections import Counter
    if Counter(s['source_kind'] for s in samples)!={'USER':6,'OLD':2}:raise ValueError('Mixed batch differs')
    for kind,count in (('USER',3),('OLD',1)):
        if sum(s['source_kind']==kind and s['alpha']==1. for s in samples)!=count:raise ValueError('Source/width balance differs')
    old=[s for s in samples if s['source_kind']=='OLD']
    if sorted(s['ratio'] for s in old)!=[.2,.4]:raise ValueError('OLD C20/C40 balance differs')
    for s in samples:
        if s['source_kind']=='OLD' and int(s['K'])!=round(int(s['y'])*s['ratio']):raise ValueError('OLD K/ratio differs')
        if s['source_kind']=='USER' and abs(float(s['ratio'])-int(s['K'])/int(s['y']))>1e-12:raise ValueError('USER K/ratio differs')
        if s['arm']!=ARM or s['step']!=step or s['global_step']!=step:raise ValueError('Input stream clock differs')
    return denominator

def check_registration(out):
    p=out/'registration.json';r=read_json(p)
    if r['experiment']!=EXPERIMENT or r['protocol']!=protocol() or r['train_limits']!=LIMITS:raise ValueError('Wrong registration')
    for path,h in read_json(out/'code_manifest.json')['source_hashes'].items():
        if digest(path)!=h:raise ValueError('Frozen source changed: '+path)
    return r

def payload(model,opt,step,stream,identities,lineage,ledger):
    base.validate_optimizer(model,None,opt,'COMMON',step);ledger.require_exact_cursor(step)
    state=stream.state_dict()
    if state['batch_index']!=step:raise ValueError('Stream ahead of update')
    g,o,r=_cpu_copy(model.state_dict()),_cpu_copy(opt.state_dict()),_cpu_copy(rng_state())
    _check_generator(g)
    return dict(schema=SCHEMA,experiment=EXPERIMENT,arm=ARM,global_step=step,update_in_progress=False,
        protocol=protocol(),protocol_sha256=json_hash(protocol()),model_config=copy.deepcopy(MODEL_CONFIG),
        generator_state=g,generator_state_sha256=state_hash(g),optimizer=o,optimizer_state_sha256=base.tree_hash(o),
        optimizer_parameter_names=base.optimizer_names(model,None),rng=r,rng_sha256=base.tree_hash(r),
        stream_state=state,stream_state_sha256=json_hash(state),identities=identities,lineage=lineage,
        execution_receipt=ledger.snapshot())

def validate_payload(p):
    if p['schema']!=SCHEMA or p['experiment']!=EXPERIMENT or p['arm']!=ARM or p['update_in_progress']:
        raise ValueError('Checkpoint schema differs')
    if p['protocol']!=protocol() or p['protocol_sha256']!=json_hash(protocol()) or p['model_config']!=MODEL_CONFIG:
        raise ValueError('Checkpoint recipe differs')
    if not 0<=p['global_step']<=TOTAL or p['stream_state']['batch_index']!=p['global_step']:raise ValueError('Checkpoint cursor differs')
    for key,hasher,hkey in [('generator_state',state_hash,'generator_state_sha256'),('optimizer',base.tree_hash,'optimizer_state_sha256'),('rng',base.tree_hash,'rng_sha256'),('stream_state',json_hash,'stream_state_sha256')]:
        if hasher(p[key])!=p[hkey]:raise ValueError('Tensor/state hash changed: '+key)
    _check_generator(p['generator_state'])
    return p

def save(path,p,replace=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not replace:raise FileExistsError(path)
    tmp=path.with_name(path.name+'.partial')
    if tmp.exists():raise FileExistsError(tmp)
    with tmp.open('xb') as f:torch.save(p,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    r=dict(path=str(path),sha256=digest(path),global_step=p['global_step'],generator_state_sha256=p['generator_state_sha256'])
    atomic_json(path.with_suffix('.json'),r);return r

def export_generator(full,out):
    p=validate_payload(torch.load(full,map_location='cpu',weights_only=True,mmap=True))
    if p['global_step']!=TOTAL:raise ValueError('Only fixed final endpoint exports')
    v={k:p[k] for k in ('experiment','arm','global_step','protocol','protocol_sha256','model_config','generator_state','generator_state_sha256','identities','lineage')}
    v.update(schema=INFERENCE_SCHEMA,training_checkpoint_sha256=digest(full),inference_contract=dict(readout_mode='none',forward='BF16',parameters='FP32',integration='FP32',teacher=False))
    return save(out,v)

def load_generator(checkpoint,device='cuda',expected_sha256=None):
    actual=digest(checkpoint)
    if expected_sha256 is not None and actual!=expected_sha256:raise ValueError('Export hash differs')
    p=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    if p['schema']!=INFERENCE_SCHEMA or p['experiment']!=EXPERIMENT or p['arm']!=ARM or p['global_step']!=TOTAL:
        raise ValueError('Wrong final generator identity')
    if p['protocol']!=protocol() or p['protocol_sha256']!=json_hash(protocol()) or p['model_config']!=MODEL_CONFIG:raise ValueError('Generator recipe differs')
    if any(k in p for k in ('optimizer','rng','teacher_state','aligner_state')):raise ValueError('Training tensors in export')
    if state_hash(p['generator_state'])!=p['generator_state_sha256']:raise ValueError('Generator state differs')
    _check_generator(p['generator_state']);runtime.configure_stable_runtime()
    model=build_model(copy.deepcopy(MODEL_CONFIG),readout_mode='none',trainable=False).eval()
    model.load_state_dict(p['generator_state'],strict=True);model.to(device).set_trainable(False)
    return model,dict({k:v for k,v in p.items() if k!='generator_state'},file_sha256=actual)

def train(out,resume=None,attempt='initial'):
    import data
    out=Path(out).resolve();registered=check_registration(out)
    torch.set_num_threads(2);runtime.configure_stable_runtime()
    folder=out/'training'/ARM
    with support.exclusive_writer(folder):
        stream=data.Stream(out)
        identities=dict(registration_sha256=digest(out/'registration.json'),code_manifest_sha256=digest(out/'code_manifest.json'),data=stream.identities)
        ledger=Ledger(folder/'calls.jsonl',LIMITS,ARM,attempt)
        status_path=folder/'status.json'
        previous=read_json(status_path) if status_path.exists() else None
        if resume is None and (previous or ledger.rows):raise ValueError('Existing run cannot restart')
        if resume is not None:
            if not previous or previous['status']!='INTERRUPTED_SAFE' or str(Path(resume).resolve())!=previous['checkpoint']['path']:
                raise ValueError('Only latest safe checkpoint can resume')
            if digest(resume)!=previous['checkpoint']['sha256']:raise ValueError('Safe checkpoint changed')
        adir=folder/'attempts'/support.safe_attempt_id(attempt);adir.mkdir(parents=True,exist_ok=False)
        state=dict(status='LOADING',arm=ARM,global_step=0,pid=os.getpid(),attempt=attempt,checkpoint=None)
        atomic_json(status_path,state)
        observer=support.GPUObserver('train_'+ARM,interval=2.)
        started=time.perf_counter();step=first=0;flag={'signal':None}
        def interrupted(sig,frame):flag['signal']=sig
        handlers={s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT)}
        try:
            with observer:
                if resume is None:
                    model,opt,lineage=recipe.initialize_official(OFFICIAL,device='cuda')
                    ledger.require_exact_cursor(0)
                    lineage.update(optimizer_hyperparameters=support.optimizer_hyperparameters(opt),condition_training_step0=True)
                    atomic_json(adir/'initialization_audit.json',lineage)
                else:
                    p=validate_payload(torch.load(resume,map_location='cpu',weights_only=True,mmap=True))
                    if p['identities']!=identities:raise ValueError('Resume input/code identity differs')
                    step=first=p['global_step'];ledger.require_exact_cursor(step)
                    model=build_model(copy.deepcopy(MODEL_CONFIG),readout_mode='none',trainable=True).eval()
                    model.load_state_dict(p['generator_state'],strict=True);model.to('cuda')
                    opt=base.make_optimizer(model,None);opt.load_state_dict(p['optimizer'])
                    if p['optimizer_parameter_names']!=base.optimizer_names(model,None):raise ValueError('Optimizer names differ')
                    base.validate_optimizer(model,None,opt,'COMMON',step)
                    stream.load_state_dict(p['stream_state']);lineage=p['lineage'];restore_rng(p['rng'])
                    if state_hash(model.state_dict())!=p['generator_state_sha256'] or base.tree_hash(opt.state_dict())!=p['optimizer_state_sha256'] or base.tree_hash(rng_state())!=p['rng_sha256']:raise ValueError('Restored tensors/RNG differ')
                    del p
                state.update(status='TRAINING',global_step=step,lineage=lineage)
                peak_a=peak_r=0
                def checkpoint(safe=False):
                    p=payload(model,opt,step,stream,identities,lineage,ledger)
                    name='R2_USER_MIX_step6000.pt' if step==TOTAL else 'R2_USER_MIX_latest.pt'
                    receipt=save(out/'checkpoints'/name,p,replace=step!=TOTAL)
                    state['checkpoint']=receipt
                    append_event(adir/'events.jsonl',dict(event='CHECKPOINT',safe=safe,**receipt))
                    del p
                for current in range(step+1,TOTAL+1):
                    if flag['signal'] is not None:
                        checkpoint(safe=True);state['status']='INTERRUPTED_SAFE';break
                    tick=time.perf_counter();samples=stream.next_effective_batch();validate_samples(samples,current)
                    input_seconds=time.perf_counter()-tick
                    for s in samples:
                        append_event(adir/'exposure.jsonl',dict(step=current,sample_index=s['sample_index'],uid=s['object_id'],task_id=s['task_id'],source_kind=s['source_kind'],N=int(s['y']),K=int(s['K']),ratio=float(s['ratio']),alpha=float(s['alpha']),t=float(s['t']),free_coordinates=s['free_coordinate_count'],input_bytehash=s['input_bytehash'],label_bytehash=s['label_bytehash'],ot_cache_key=s['ot_cache_key']))
                    result=base.update(model,None,opt,samples,None,ledger,'COMMON',current)
                    if result['alignment_active'] or result['aligned_samples']:raise AssertionError('Unexpected auxiliary training')
                    step=current
                    peak_a=max(peak_a,result['cuda_peak_allocated_bytes']);peak_r=max(peak_r,result['cuda_peak_reserved_bytes'])
                    append_event(adir/'training_log.jsonl',dict(step=step,input_seconds=input_seconds,**result))
                    state.update(global_step=step,last_FM=result['FM'],update_seconds=result['update_seconds'],input_seconds=input_seconds,wall_seconds=time.perf_counter()-started,cuda_peak_allocated_bytes=peak_a,cuda_peak_reserved_bytes=peak_r,calls=ledger.snapshot())
                    if step%500==0:checkpoint()
                    atomic_json(status_path,state)
                    if step==first+1 or step%25==0:print(ARM,step,'FM',result['FM'],flush=True)
                else:state['status']='TRAIN_UPDATES_COMPLETE'
            if state['status']=='TRAIN_UPDATES_COMPLETE':
                state['inference_export']=export_generator(state['checkpoint']['path'],out/'inference/R2_USER_MIX_generator_step6000.pt')
                state['status']='COMPLETE'
        except BaseException as exc:
            state.update(status='OOM' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'BLOCKED',error=repr(exc),traceback=traceback.format_exc(),no_automatic_retry=True)
            atomic_json(adir/'failure.json',state);raise
        finally:
            for s,h in handlers.items():signal.signal(s,h)
            state.update(global_step=step,calls=ledger.snapshot(),wall_seconds=time.perf_counter()-started,resource_observation=observer.summary())
            atomic_json(adir/'resources.json',dict(summary=observer.summary(),samples=observer.samples))
            atomic_json(status_path,state);atomic_json(adir/'final_status.json',state)
            if state['status']=='COMPLETE':atomic_json(folder/'completed.json',state)
        return state

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--resume');p.add_argument('--attempt',default='initial')
    a=p.parse_args();result=train(a.out,a.resume,a.attempt);print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['status']=='COMPLETE' else 75)
