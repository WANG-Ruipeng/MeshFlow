"""One locked GPU worker; fixed budgets, durable accounting and no lost-update replay."""
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'
from native_t1 import runtime as _runtime
import argparse
import contextlib
import csv
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil
import signal
import time
import traceback
import numpy as np
import torch
from native_t1.training import make_optimizer
from experiments.mf_h1_local0.control_mvp48.model_mvp import load_mvp_checkpoint
from experiments.mf_h1_local0.control_mvp48.train_core import optimizer_update
from experiments.mf_h1_local0.control_mvp48.train_stream import MVPObjective
from experiments.mf_h1_local0.control_mvp48.train_io import (
    gpu_lock, resource_snapshot, capture_rng, restore_rng, tree_hash)
from experiments.mf_h1_local0.control_mvp48.train_run import GPUMonitor, dispatch_check
from .common import read_json, atomic_json, append_event, digest, json_hash
from .checkpoint import ARMS, load_checkpoint, save_checkpoint, validate_payload, tensor_state_hash
from .stream import BudgetStream

COLUMNS = ['arm','logical_step','new_updates','batch_sha256','FM','geometry','loss',
           'free_coordinate_denominator','active_samples','grad_norm_before_clip',
           'mean_active_D_edge','mean_active_L5','OT_seconds','train_seconds',
           'cuda_peak_allocated_bytes','cuda_peak_reserved_bytes']



@contextlib.contextmanager
def process_gpu_lock(out):
    """Keep the existing shared flock until this single-use worker exits."""
    global _PROCESS_GPU_LOCK
    held=gpu_lock(out)
    wait=held.__enter__()
    _PROCESS_GPU_LOCK=held
    # Retaining the entered context covers exception cleanup and CUDA context
    # destruction. OS process exit releases flock; the lock file is never removed.
    yield wait


def ledger_rows(path):
    path=Path(path)
    if not path.exists(): return []
    rows=[]
    for line in path.read_text().splitlines():
        try: rows.append(json.loads(line))
        except json.JSONDecodeError:
            raise RuntimeError('Incomplete durable ledger; manual reconciliation required: '+str(path))
    return rows


def check_recovery(events, checkpoint_step, start):
    """An attempted update not covered by the checkpoint is never replayed."""
    begins=[int(r['logical_step']) for r in events if r['event']=='UPDATE_BEGIN']
    dones=[int(r['logical_step']) for r in events if r['event']=='UPDATE_VALIDATED']
    if len(begins)!=len(set(begins)) or len(dones)!=len(set(dones)):
        raise RuntimeError('Duplicate update in durable ledger')
    if max(begins+[start])>checkpoint_step or max(dones+[start])>checkpoint_step:
        raise RuntimeError('BLOCKED_UNCHECKPOINTED_ATTEMPT: completed/attempted updates exceed reliable checkpoint; no automatic replay')
    if dones and sorted(dones)!=list(range(start+1,max(dones)+1)):
        raise RuntimeError('Training ledger has a non-contiguous update history')



def reconcile_latest(out, arm, latest, protocol, stream, events):
    """Recover an atomically installed full payload if pointer installation was interrupted."""
    out=Path(out)
    candidate=Path(latest['checkpoint']['path']) if latest else out/'checkpoints'/f'{arm}_latest.pt'
    endpoint=out/'checkpoints'/f'{arm}_step{stream.end}.pt'
    if endpoint.exists() and (not latest or latest['logical_step']<stream.end):
        candidate=endpoint
    if not candidate.exists(): return latest
    if out not in candidate.resolve().parents:
        raise ValueError('Latest recovery candidate outside isolated run')
    payload=torch.load(candidate,map_location='cpu',weights_only=True,mmap=True)
    meta=payload.get('metadata',{})
    if latest and meta.get('logical_step')==latest['logical_step']:
        return latest
    meta=validate_payload(payload)
    if not meta['full_recovery_state'] or 'optimizer' not in payload:
        raise ValueError('Interrupted save has no complete recovery payload')
    if meta['variant']!=arm or meta['protocol_sha256']!=json_hash(protocol):
        raise ValueError('Interrupted save has an unrelated identity')
    if payload['stream_state']['plan_sha256']!=stream.plan_sha256:
        raise ValueError('Interrupted save has an unrelated plan')
    if latest and meta['logical_step']<latest['logical_step']:
        raise ValueError('Recovery candidate regressed behind durable pointer')
    check_recovery(events,meta['logical_step'],stream.offset)
    if tensor_state_hash(payload['model'])!=meta['model_state_sha256']:
        raise ValueError('Interrupted save has corrupt model tensors')
    if meta.get('counts',{}).get('optimizer_updates')!=meta['logical_step']-stream.offset:
        raise ValueError('Recovery candidate update counter differs')
    sha=digest(candidate)
    recovery=dict(path=str(candidate.resolve()),checkpoint_sha256=sha,metadata=meta,strict_reload='PASS')
    repaired=dict(logical_step=meta['logical_step'],protocol_sha256=json_hash(protocol),
        plan_sha256=stream.plan_sha256,checkpoint=recovery,counts=meta['counts'])
    atomic_json(out/'training'/f'{arm}_pointer_reconciliation_step{meta["logical_step"]}.json',
        dict(previous_pointer=latest,recovered=repaired,model_forwards=0,optimizer_updates=0))
    atomic_json(candidate.with_suffix('.json'),{k:recovery[k] for k in ('path','checkpoint_sha256','metadata')})
    atomic_json(out/'training'/f'{arm}_latest.json',repaired)
    append_event(out/'training'/f'{arm}_events.jsonl',dict(event='POINTER_RECONCILED',logical_step=meta['logical_step'],checkpoint_sha256=sha))
    return repaired


def _restore_source(source_path, device, probe):
    model, meta, hashes=load_mvp_checkpoint(source_path,device)
    if meta['logical_step']!=1000:
        raise ValueError('Continuation requires MVP48 step1000')
    payload=torch.load(source_path,map_location='cpu',weights_only=True,mmap=True)
    model.set_trainable(True).eval();optimizer=make_optimizer(model)
    if probe:
        # A separate experiment RNG; no source/global training RNG is consumed by data.
        random.seed(2026100302);np.random.seed(2026100302);torch.manual_seed(2026100302)
        torch.cuda.manual_seed_all(2026100302)
    else:
        optimizer.load_state_dict(payload['optimizer'])
        if tree_hash(optimizer.state_dict())!=meta['optimizer_state_sha256']:
            raise ValueError('Restored optimizer content differs')
        restore_rng(payload['rng'])
        if tree_hash(capture_rng())!=meta['rng_sha256']:
            raise ValueError('Restored training RNG differs')
    del payload
    return model,optimizer,hashes['checkpoint_sha256']


def train_segment(source,out,arm,target,protocol):
    source,out=Path(source).resolve(),Path(out).resolve()
    if arm not in ARMS: raise ValueError('Unregistered arm')
    probe=arm=='T1_fit2_probe';start,maximum=(0,1000) if probe else (1000,5000)
    milestones=(500,1000) if probe else (2000,3500,5000)
    if target not in milestones:
        raise ValueError('Target must be a fixed checkpoint boundary')
    if protocol.get('restore_mode','RESTORE_ALL')!='RESTORE_ALL':
        raise ValueError('This verified run freezes RESTORE_ALL for every main arm')
    for folder in ('training','checkpoints','registration'): (out/folder).mkdir(parents=True,exist_ok=True)
    source_path=source/'checkpoints'/f'{"T1" if probe else arm}_step1000.pt'
    status_path=out/'training'/f'{arm}_status.json'
    latest_path=out/'training'/f'{arm}_latest.json'
    events_path=out/'training'/f'{arm}_events.jsonl'
    log_path=out/'training'/f'{arm}_training_log.csv'
    state=dict(status='PREPARING',arm=arm,target=target,logical_step=start,new_updates=0,
               counts={},plan_sha256=None,protocol_sha256=json_hash(protocol),gpu_workers=1)
    if status_path.exists() and read_json(status_path).get('status')=='BLOCKED':
        raise RuntimeError('Earlier affected stage is BLOCKED; no automatic retry: '+str(status_path))
    if torch.cuda.is_initialized(): raise RuntimeError('Worker initialized CUDA before shared lock')
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    _runtime.configure_stable_runtime()
    stream=BudgetStream(source,out,arm);state['plan_sha256']=stream.plan_sha256
    latest=read_json(latest_path) if latest_path.exists() else None
    latest=reconcile_latest(out,arm,latest,protocol,stream,ledger_rows(events_path))
    if latest:
        if latest['protocol_sha256']!=json_hash(protocol) or latest['plan_sha256']!=stream.plan_sha256:
            raise ValueError('Continuation identity differs from reliable progress')
        checkpoint_step=int(latest['logical_step'])
    else: checkpoint_step=start
    check_recovery(ledger_rows(events_path),checkpoint_step,start)
    state.update(logical_step=checkpoint_step,new_updates=checkpoint_step-start)
    if checkpoint_step>=target:
        required=out/'checkpoints'/f'{arm}_step{target}.pt'
        if not required.exists(): raise RuntimeError('Progress exists but fixed checkpoint is missing')
        return dict(state,status='REUSED',checkpoint=str(required),actual_updates_this_worker=0)
    expected_prior=start if target==milestones[0] else milestones[milestones.index(target)-1]
    if checkpoint_step<expected_prior:
        raise ValueError('Fixed stage dependencies require the prior boundary first')
    if shutil.disk_usage(out).free<8*1024**3:
        raise RuntimeError('Insufficient free storage for isolated checkpoint staging')
    resources=resource_snapshot();atomic_json(out/'training'/f'{arm}_to{target}_resources_before.json',resources)
    dispatch=dispatch_check(resources)
    atomic_json(out/'training'/f'{arm}_to{target}_dispatch.json',dispatch)
    stop_requested={'signal':None}
    def on_signal(signum,frame): stop_requested['signal']=int(signum)
    old_handlers={s:signal.signal(s,on_signal) for s in (signal.SIGTERM,signal.SIGINT)}
    model=optimizer=monitor=None;counts={};finished_step=checkpoint_step
    restored_update_count=checkpoint_step-start
    tick=time.perf_counter();worker_ot=worker_train=worker_save=0.;source_sha=None
    try:
        with process_gpu_lock(out) as wait_seconds:
            atomic_json(out/'training'/f'{arm}_to{target}_resources_locked.json',resource_snapshot())
            dispatch_check(resource_snapshot())
            load_tick=time.perf_counter()
            if latest:
                checkpoint=Path(latest['checkpoint']['path'])
                if out not in checkpoint.resolve().parents:
                    raise ValueError('Latest checkpoint outside this isolated experiment')
                model,meta,hashes=load_checkpoint(checkpoint,'cuda',trainable=True)
                if meta['variant']!=arm or meta['logical_step']!=checkpoint_step:
                    raise ValueError('Reliable checkpoint arm/cursor differs')
                payload=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
                optimizer=make_optimizer(model);optimizer.load_state_dict(payload['optimizer'])
                if tree_hash(optimizer.state_dict())!=meta['optimizer_state_sha256']:
                    raise ValueError('Resumed AdamW differs')
                stream.load_state_dict(payload['stream_state']);restore_rng(payload['rng'])
                if tree_hash(capture_rng())!=meta['rng_sha256']:
                    raise ValueError('Resumed RNG differs')
                source_sha=meta['source_checkpoint_sha256'];counts=dict(meta.get('counts',{}));del payload
            else:
                model,optimizer,source_sha=_restore_source(source_path,'cuda',probe)
            if digest(source_path)!=source_sha:
                raise ValueError('Immutable source checkpoint changed')
            load_seconds=time.perf_counter()-load_tick
            objective=MVPObjective(stream) if arm=='Full' else None
            torch.cuda.reset_peak_memory_stats();monitor=GPUMonitor(out);monitor.start()
            state.update(status='TRAINING',lock_wait_seconds=wait_seconds,load_seconds=load_seconds)
            atomic_json(status_path,state)
            shared_path=out/'training/shared_batch_hashes.jsonl'
            shared={r['logical_step']:r['batch_sha256'] for r in ledger_rows(shared_path)} if not probe else {}
            with log_path.open('a',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=COLUMNS)
                if f.tell()==0:writer.writeheader();f.flush();os.fsync(f.fileno())
                def persist(step,fixed=False):
                    nonlocal worker_save,latest
                    if fixed:
                        path=out/'checkpoints'/f'{arm}_step{step}.pt'
                        record=save_checkpoint(path,model,optimizer,stream,protocol,arm=arm,step=step,
                            source_path=source_path,source_hash=source_sha,full_state=step==maximum,
                            changes=True,counts=counts)
                        worker_save+=record['save_and_verify_seconds']
                        append_event(events_path,dict(event='FIXED_CHECKPOINT',logical_step=step,checkpoint=record))
                    if fixed and step==maximum:
                        recovery=record
                    else:
                        recovery=save_checkpoint(out/'checkpoints'/f'{arm}_latest.pt',model,optimizer,stream,protocol,
                            arm=arm,step=step,source_path=source_path,source_hash=source_sha,full_state=True,
                            changes=False,counts=counts,replace=True)
                        worker_save+=recovery['save_and_verify_seconds']
                    latest=dict(logical_step=step,protocol_sha256=json_hash(protocol),plan_sha256=stream.plan_sha256,
                                checkpoint=recovery,counts=dict(counts))
                    atomic_json(latest_path,latest)
                    append_event(events_path,dict(event='RECOVERY_CHECKPOINT',logical_step=step,
                                                  checkpoint_sha256=recovery['checkpoint_sha256']))
                for step in range(checkpoint_step+1,target+1):
                    if stop_requested['signal'] is not None:
                        if finished_step>checkpoint_step and (not latest or latest['logical_step']!=finished_step):
                            persist(finished_step)
                        state.update(status='INTERRUPTED_SAFE',signal=stop_requested['signal'])
                        break
                    ot_tick=time.perf_counter();samples=stream.next_effective_batch();ot_seconds=time.perf_counter()-ot_tick
                    worker_ot+=ot_seconds
                    bh=hashlib.sha256(''.join(s['shared_input_sha256'] for s in samples).encode()).hexdigest()
                    if not probe:
                        if step in shared and shared[step]!=bh:
                            raise ValueError('Shared T1/Geo/Full actual input bytes differ at '+str(step))
                        if step not in shared:
                            append_event(shared_path,dict(event='SHARED_INPUT_REGISTERED',logical_step=step,batch_sha256=bh,first_arm=arm))
                            shared[step]=bh
                    append_event(events_path,dict(event='UPDATE_BEGIN',arm=arm,logical_step=step,batch_sha256=bh,
                        completed_before=finished_step,counts_before=dict(counts)))
                    result=optimizer_update(model,optimizer,samples,objective=objective,
                        additional_step=step-500 if arm=='Full' else None,counts=counts)
                    finished_step=step;worker_train+=result['train_seconds']
                    def active_mean(key):
                        values=[x['raw_losses'][key] for x in result['geometry_diagnostics'] if key in x['raw_losses']]
                        return sum(values)/len(values) if values else 0.
                    row=dict(arm=arm,logical_step=step,new_updates=step-start,batch_sha256=bh,
                        **{k:result[k] for k in ('FM','geometry','loss','free_coordinate_denominator','active_samples','grad_norm_before_clip')},
                        mean_active_D_edge=active_mean('D_edge'),mean_active_L5=active_mean('L5'),
                        OT_seconds=ot_seconds,train_seconds=result['train_seconds'],
                        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
                    append_event(events_path,dict(event='UPDATE_VALIDATED',**row,counts=dict(counts)))
                    writer.writerow(row);f.flush();os.fsync(f.fileno())
                    state.update(logical_step=step,new_updates=step-start,counts=dict(counts),last_row=row)
                    atomic_json(status_path,state)
                    if step%200==0 or step==target:
                        observation=resource_snapshot()
                        atomic_json(out/'training'/f'{arm}_step{step}_resources.json',observation)
                        print(json.dumps(dict(event='TRAIN_PROGRESS',**row)),flush=True)
                        persist(step,fixed=step in milestones)
                    elif step in milestones:
                        persist(step,fixed=True)
                else:
                    state['status']='SEGMENT_COMPLETE'
            torch.cuda.synchronize()
            state.update(actual_updates_this_worker=finished_step-checkpoint_step,
                worker_OT_seconds=worker_ot,worker_train_seconds=worker_train,worker_save_seconds=worker_save,
                worker_wall_seconds=time.perf_counter()-tick,OT_counts=dict(stream.counts),counts=dict(counts),
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                global_gpu_used_peak_mib=monitor.peak_global_used_mib,process_gpu_used_peak_mib=monitor.peak_process_used_mib,
                memory_scope='Per worker after restore, including backward/AdamW and geometry; nvidia-smi global includes other processes',
                throughput_updates_per_train_second=(finished_step-checkpoint_step)/worker_train if worker_train else None)
            atomic_json(status_path,state)
            atomic_json(out/'training'/f'{arm}_to{target}_summary.json',state)
            append_event(events_path,dict(event='SEGMENT_END',**state))
            return state
    except BaseException as exc:
        actual_count=counts.get('optimizer_updates',restored_update_count)
        state.update(status='BLOCKED',logical_step=finished_step,new_updates=actual_count,
            counts=dict(counts),OT_counts=dict(stream.counts),error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc(),
            actual_updates_this_worker=actual_count-restored_update_count,worker_wall_seconds=time.perf_counter()-tick)
        atomic_json(status_path,state);append_event(events_path,dict(event='FAILURE',**state))
        raise
    finally:
        if monitor:monitor.stop()
        for s,handler in old_handlers.items():signal.signal(s,handler)
        del model,optimizer;gc.collect()
        if torch.cuda.is_initialized():torch.cuda.empty_cache()
        atomic_json(out/'training'/f'{arm}_to{target}_resources_after.json',resource_snapshot())


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--target',type=int,required=True)
    a=p.parse_args();protocol=read_json(a.out/'protocol.json')
    print(json.dumps(train_segment(a.source,a.out,a.arm,a.target,protocol),sort_keys=True),flush=True)


if __name__=='__main__':main()
