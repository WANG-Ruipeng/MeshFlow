"""Paired 200-update Native FM continuation; no implicit failed replay."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse, copy, json, signal, time, traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import numpy as np
from . import common as C
C.package()
import torch
from meshflow_control import runtime
from meshflow_control.config import MODEL_CONFIG
from meshflow_control.models.native import build_model, native_execution
from meshflow_control.checkpoints import state_hash, rng_state, restore_rng
from meshflow_control.training import schedule40 as base
from meshflow_control.training.context_checkpoint import _cpu_copy, _check_generator
from meshflow_control.training.losses import masked_fm_loss
from meshflow_control.data.dataset import collate, model_inputs
from meshflow_control.data.io import atomic_json, append_event, digest, json_hash
from r2.pilots.user_data_scaling_32_67_v1 import training_data as D
from r2.pilots.user_condition_finetune_v1 import trainer as S

SCHEMA = 'r2_target_dedup_c200_training_v1'


class CallLedger(S.CallLedger):
    def require_exact_cursor(self, local_step):
        expected=dict(train_forward=8*local_step,train_backward=8*local_step,optimizer_update=local_step)
        if self.pending or self.failed:raise ValueError('Failed/unclosed calls; no implicit replay')
        if any(self.attempted[k]!=n or self.successful[k]!=n for k,n in expected.items()):
            raise ValueError('Checkpoint trails attempted/completed calls')
        steps=[r['context']['global_step'] for r in self.rows if r['kind']=='optimizer_update' and r['event']=='CALL_BEGIN']
        if steps!=list(range(12001,12001+local_step)):raise ValueError('Global update sequence differs')


class Dataset(D.Dataset):
    def __init__(self, path):
        self.root = C.resolve(path).parent
        self.manifest = C.read(path)
        self.manifest_sha256 = C.sha(path)
        self.parents, self.tasks = self.manifest['parents'], self.manifest['tasks']
        if len(self.parents) != 16 or len(self.tasks) != 16:
            raise ValueError('Exactly16 frozen affected parents/conditions required')


def registration():
    r = C.load_registration()
    if r['status'] != 'FROZEN' or (r['updates_per_arm'],r['batch'],r['microbatch'],r['source_global_step']) != (200,8,1,12000):
        raise ValueError('Frozen C200 contract differs')
    for p,h in r['training_code_hashes'].items():
        if C.sha(p) != h: raise ValueError('Training source changed: '+p)
    for p,h in r['protected'].items():
        if C.sha(p) != h: raise ValueError('Protected small input changed: '+p)
    return r


def batch(dataset, records, arm, index, plan_sha):
    folder = C.OUT/'ot'/arm
    path = folder/f'{index:04d}.npz'; receipt = path.with_suffix('.json'); attempt = path.with_suffix('.attempt.json')
    identity = dict(plan_sha256=plan_sha, records_sha256=json_hash(records), manifest_sha256=dataset.manifest_sha256)
    if receipt.exists():
        r = C.read(receipt)
        if r['identity'] != identity or C.sha(path) != r['sha256']: raise ValueError('OT receipt differs')
        maps = C.arrays(path)
    else:
        if attempt.exists() or path.exists(): raise RuntimeError('Unresolved OT, no implicit replay')
        C.write(attempt, identity); maps = None
    builder = D.Builder(dataset,maps)
    samples = [D.validate_sample(builder.sample(r),r) for r in records]
    if maps is None:
        C.frozen_npz(path,builder.new_maps)
        C.write(receipt,dict(status='COMPLETE',identity=identity,sha256=C.sha(path),OT_returns=8))
    return samples


def update(model,opt,samples,ledger,step,arm,hyper):
    den = S.validate_samples(samples,arm,None,effective_batch=True)
    assert all(s['global_step']==12000+step and s['step']==step and s['K']>=2 for s in samples)
    base.validate_optimizer(model,None,opt,'COMMON',12000+step-1)
    assert S.optimizer_hyperparameters(opt)==hyper
    torch.cuda.reset_peak_memory_stats(); opt.zero_grad(set_to_none=True)
    tick=time.perf_counter(); fm=0.; forward=backward=0.; diagnostics=[]
    for i,s in enumerate(samples):
        b=collate([s],'cuda')
        ctx=dict(local_step=step,global_step=12000+step,microbatch=i,sample_index=s['sample_index'],input_bytehash=s['input_bytehash'],label_bytehash=s['label_bytehash'])
        with native_execution(model,coordinates=(b['xt'],b['t']),autocast=True):
            v,sec=ledger.call('train_forward',lambda:model(*model_inputs(b)).float(),ctx);forward+=sec
            loss=masked_fm_loss(v,b['u'],b['valid_mask'],b['known_mask'],denominator=den)
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite FM')
            value=float(loss.detach())
            _,sec=ledger.call('train_backward',loss.backward,ctx);backward+=sec
        fm+=value
        diagnostics.append(dict(**ctx,uid=s['object_id'],N=int(s['y']),K=s['K'],t=float(s['t']),alpha=s['alpha'],
                                free_coordinates=s['free_coordinate_count'],FM_contribution=value,
                                velocity_sse=value*den,endpoint_scaled_sse=(1-float(s['t']))**2*value*den))
        del v,loss,b
    if any(p.grad is None for p in model.parameters()): raise RuntimeError('Unused model parameter')
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
    _,optsec=ledger.call('optimizer_update',opt.step,dict(local_step=step,global_step=12000+step))
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()): raise FloatingPointError('Nonfinite parameter')
    base.validate_optimizer(model,None,opt,'COMMON',12000+step)
    return dict(FM=fm,free_coordinate_denominator=den,gradient_norm=norm,forward_seconds=forward,
                backward_seconds=backward,optimizer_seconds=optsec,update_seconds=time.perf_counter()-tick,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                peak_scope='formal update including forward/backward/optimizer; not total process VRAM'),diagnostics


def resource():
    v=C.resources()
    v['free_mib']=float(v['gpu']['stdout'].splitlines()[0].split(',')[3])
    return v


def save(path,payload,replace=False):
    C.checked(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not replace: raise FileExistsError(path)
    tmp=path.with_suffix('.partial')
    with tmp.open('xb') as f:
        torch.save(payload,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    r=dict(path=str(path),sha256=C.sha(path),bytes=path.stat().st_size,
           **{k:payload[k] for k in ('arm','local_step','global_step','generator_state_sha256','optimizer_state_sha256','rng_sha256') if k in payload})
    atomic_json(path.with_suffix('.json'),r)
    return r


def train(arm):
    assert arm in C.ARMS
    reg=registration();rh=C.sha(C.OUT/'registration.json')
    planpath=C.OUT/'plans'/f'{arm}.json';plan=C.read(planpath);ph=C.sha(planpath)
    assert plan['registration_sha256']==rh and len(plan['records'])==1600
    folder=C.OUT/'training'/arm;folder.mkdir(parents=True,exist_ok=True)
    status=folder/'status.json';previous=C.read(status) if status.exists() else None
    if previous and previous['status']=='COMPLETE':
        for key in ('checkpoint','inference_export'):
            assert C.sha(previous[key]['path'])==previous[key]['sha256']
        return previous
    if previous and previous['status']!='INTERRUPTED_SAFE': raise RuntimeError('Prior failed/ambiguous attempt, no implicit replay')
    step=previous['local_step'] if previous else 0
    rr=resource();C.write(folder/f'dispatch_{time.time_ns()}.json',rr)
    if rr['free_mib']<2048+1.2*4096: raise RuntimeError('RESOURCE_WAIT before model load')
    attempt='attempt_'+str(time.time_ns());adir=folder/'attempts'/attempt
    ledger=CallLedger(folder/'calls.jsonl',dict(train_forward=1600,train_backward=1600,optimizer_update=200),arm,attempt)
    ledger.require_exact_cursor(step)
    dataset=Dataset(reg['datasets'][arm]['path'])
    assert dataset.manifest_sha256==reg['datasets'][arm]['sha256']
    observer=S.GPUObserver('target_cleanup_'+arm,interval=2.)
    start=time.perf_counter();flag={'stop':False};pending={};pool=ThreadPoolExecutor(max_workers=2)
    def stop(*_):flag['stop']=True
    handlers={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
    state=dict(status='LOADING',arm=arm,local_step=step,global_step=12000+step,registration_sha256=rh,attempt=attempt)
    try:
        with S.exclusive_writer(folder),observer:
            atomic_json(status,state);torch.set_num_threads(2);runtime.configure_stable_runtime()
            torch.cuda.set_per_process_memory_fraction(5*1024**3/torch.cuda.get_device_properties(0).total_memory)
            path=C.resolve(previous['checkpoint']['path'] if previous else reg['source_checkpoint'])
            expected=previous['checkpoint']['sha256'] if previous else reg['source_checkpoint_sha256']
            assert C.sha(path)==expected
            p=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            assert p['global_step']==12000+step and p['model_config']==MODEL_CONFIG and p['update_in_progress'] is False
            if previous:
                assert p['schema']==SCHEMA and p['arm']==arm and p['registration_sha256']==rh and p['plan_sha256']==ph
            else:assert p['arm']=='FULL_SEAT' and p['local_step']==6000
            for key,fn,hkey in [('generator_state',state_hash,'generator_state_sha256'),('optimizer',base.tree_hash,'optimizer_state_sha256'),('rng',base.tree_hash,'rng_sha256')]:
                assert fn(p[key])==p[hkey]
            model=build_model(copy.deepcopy(MODEL_CONFIG),readout_mode='none',trainable=True).eval()
            assert p['optimizer_parameter_names']==base.optimizer_names(model,None)
            model.load_state_dict(p['generator_state'],strict=True);model.to('cuda')
            opt=base.make_optimizer(model,None);opt.load_state_dict(p['optimizer']);restore_rng(p['rng'])
            assert state_hash(model.state_dict())==p['generator_state_sha256']
            assert base.tree_hash(opt.state_dict())==p['optimizer_state_sha256'] and base.tree_hash(rng_state())==p['rng_sha256']
            hyper=S.optimizer_hyperparameters(opt)
            C.write(adir/'initialization.json',dict(source_path=str(path),source_file_sha256=expected,
                    generator_sha256=p['generator_state_sha256'],optimizer_sha256=p['optimizer_state_sha256'],rng_sha256=p['rng_sha256'],hyperparameters=hyper,global_step=p['global_step']))
            del p
            def checkpoint(final=False):
                ledger.require_exact_cursor(step)
                g,o,rng=_cpu_copy(model.state_dict()),_cpu_copy(opt.state_dict()),_cpu_copy(rng_state());_check_generator(g)
                value=dict(schema=SCHEMA,experiment=C.VERSION,arm=arm,local_step=step,global_step=12000+step,update_in_progress=False,
                    generator_state=g,generator_state_sha256=state_hash(g),optimizer=o,optimizer_state_sha256=base.tree_hash(o),
                    rng=rng,rng_sha256=base.tree_hash(rng),optimizer_parameter_names=base.optimizer_names(model,None),model_config=copy.deepcopy(MODEL_CONFIG),
                    registration_sha256=rh,plan_sha256=ph,protocol=reg,execution=ledger.snapshot(),EMA=False)
                latest=C.OUT/'checkpoints'/f'{arm}_resume_latest.pt'
                state['checkpoint']=save(latest,value,replace=latest.exists())
                if final:
                    finalpath=C.OUT/'checkpoints'/f'{arm}_step200.pt'
                    if finalpath.exists():raise FileExistsError(finalpath)
                    os.replace(latest,finalpath)
                    rec=state['checkpoint'];rec['path']=str(finalpath)
                    os.replace(latest.with_suffix('.json'),finalpath.with_suffix('.json'));atomic_json(finalpath.with_suffix('.json'),rec)
                    q={k:value[k] for k in ('experiment','arm','local_step','global_step','generator_state','generator_state_sha256','model_config')}
                    q.update(schema='r2_target_dedup_c200_generator_v1',training_checkpoint_sha256=rec['sha256'],EMA=False)
                    state['inference_export']=save(C.OUT/'inference'/f'{arm}_generator.pt',q)
                    C.write(C.OUT/'evaluation/bindings'/f'{arm}.json',dict(state['inference_export'],label=arm,checkpoint_arm=arm,EMA=False))
            begin_step=step
            state['status']='TRAINING'
            for current in range(step+1,201):
                if current==begin_step+1 or current%5==0 or flag['stop']:
                    rr=resource();append_event(folder/'resource_checks.jsonl',rr)
                    if flag['stop'] or rr['free_mib']<2048:
                        checkpoint();state.update(status='INTERRUPTED_SAFE',reason='signal_or_headroom');break
                tick=time.perf_counter()
                for bi in range(current-1,min(200,current+1)):
                    if bi not in pending: pending[bi]=pool.submit(batch,dataset,plan['records'][bi*8:(bi+1)*8],arm,bi,ph)
                samples=pending.pop(current-1).result();input_seconds=time.perf_counter()-tick
                for s in samples:
                    append_event(folder/'exposure.jsonl',dict(step=current,sample_index=s['sample_index'],uid=s['object_id'],task_id=s['task_id'],N=int(s['y']),K=s['K'],alpha=s['alpha'],t=float(s['t']),
                        free_coordinate_count=s['free_coordinate_count'],input_bytehash=s['input_bytehash'],label_bytehash=s['label_bytehash'],ot_cache_key=s['ot_cache_key']))
                result,di=update(model,opt,samples,ledger,current,arm,hyper);step=current
                append_event(folder/'training_log.jsonl',dict(local_step=step,global_step=12000+step,input_seconds=input_seconds,**result))
                for x in di:append_event(folder/'diagnostics.jsonl',x)
                state.update(local_step=step,global_step=12000+step,last_FM=result['FM'],peak_reserved_bytes=result['cuda_peak_reserved_bytes'],wall_seconds=time.perf_counter()-start)
                if step%50==0:checkpoint(final=step==200)
                atomic_json(status,state)
                if step==1 or step%10==0:print(json.dumps(dict(arm=arm,step=step,**result)),flush=True)
            else:state['status']='COMPLETE'
    except BaseException as exc:
        state.update(status='OOM' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'FAILED',error=repr(exc),traceback=traceback.format_exc(),no_automatic_retry=True)
        C.write(adir/'failure.json',state);raise
    finally:
        pool.shutdown(wait=True)
        for sig,h in handlers.items():signal.signal(sig,h)
        state.update(local_step=step,global_step=12000+step,calls=ledger.snapshot(),wall_seconds=time.perf_counter()-start,resources=observer.summary())
        C.write(adir/'resources.json',dict(samples=observer.samples,summary=observer.summary()))
        atomic_json(status,state);C.write(adir/'final_status.json',state)
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=C.ARMS,required=True)
    args=parser.parse_args();print(json.dumps(train(args.arm),indent=2))
