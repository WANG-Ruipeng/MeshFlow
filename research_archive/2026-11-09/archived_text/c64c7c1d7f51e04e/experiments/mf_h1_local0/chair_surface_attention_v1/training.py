"""Fixed-budget three-arm HYBRID continuation; numerical gates precede loading.

The public train_arm accepts an already restored model under the caller's GPU
lock. The run entry owns that lock itself. Neither CPU audits nor imports invoke
a model, optimizer step, or CUDA initialization.
"""
from __future__ import annotations

import argparse
import csv
import os
import signal
import time
import traceback
from pathlib import Path

from native_t1 import runtime as _runtime
import torch
from native_t1.model import native_execution
from native_t1.data import model_inputs
from native_t1.training import masked_fm_loss
from experiments.mf_h1_local0.control_mvp48.train_stream import collate
from .common import SOURCE,CHAIR,OUT,START,START_SHA,ARMS,append_event,atomic_json,read_json,ledger_rows,json_hash,digest
from .stream import SurfaceStream
from .model_adapter import set_profile
from .ledger import Budget

LOG_COLUMNS=('arm','surface_step','conditional_total_step','shared_batch_sha256','coupled_batch_sha256',
    'FM','loss','free_coordinate_denominator','grad_norm_before_clip','new_parameter_count',
    'new_parameter_norm_before','new_parameter_norm_after','new_parameter_delta_l2',
    'new_grad_norm_before_clip','new_grad_norm_after_clip','logit_correction_RMS_mean',
    'weighted_pair_value_RMS_mean','content_delta_RMS_mean','relation_observation_count',
    'OT_and_input_seconds','forward_seconds','loss_seconds','backward_seconds','clip_seconds',
    'optimizer_seconds','validation_seconds','update_total_seconds','actual_OT_attempts','actual_OT_returns','OT_cache_hits',
    'cuda_peak_allocated_bytes','cuda_peak_reserved_bytes')
EXPOSURE_COLUMNS=('arm','surface_step','sample_index','task_id','uid','N','K','ratio','alpha','t',
    'free_coordinates','shared_input_sha256','coupled_input_sha256','ot_cache_key','ot_cache_status')


def parameter_partition(model):
    names=set(model.surface_attention_metadata['new_parameter_names'])
    parameters=dict(model.named_parameters())
    actual={name for name in parameters if name.startswith('relation_modules.')}
    if actual!=names:raise ValueError('New parameter namespace differs from registered adapter metadata')
    if model._surface_attention_arm=='H_CONT' and names:raise ValueError('H_CONT cannot own new relation parameters')
    return parameters,names


def validate_optimizer_steps(named,new_names,optimizer,surface_step):
    """Old states continue from 5000; only registered new states start at zero."""
    for name,param in named.items():
        state=optimizer.state.get(param,{})
        expected=surface_step if name in new_names else 5000+surface_step
        if not state:
            if name in new_names and surface_step==0:continue  # AdamW lazy zero-state initialization.
            raise ValueError('Missing restored optimizer state: '+name)
        if int(state.get('step',-1))!=expected:raise ValueError('AdamW cursor differs for '+name)
        for key in ('exp_avg','exp_avg_sq'):
            if key not in state or state[key].shape!=param.shape or state[key].dtype!=torch.float32:
                raise ValueError('AdamW moment shape/dtype missing for '+name)
            if name in new_names and surface_step==0 and bool(torch.count_nonzero(state[key])):
                raise ValueError('New relation moment must initialize to zero')


def _validate_optimizer(model,optimizer,surface_step):
    named,new_names=parameter_partition(model);params=list(named.values())
    if not isinstance(optimizer,torch.optim.AdamW):raise ValueError('Restored AdamW required')
    if model.training or not all(p.requires_grad and p.dtype==torch.float32 and p.is_cuda for p in params):
        raise ValueError('Require original eval/dropout mode and trainable FP32 GPU parameters')
    owned=[p for group in optimizer.param_groups for p in group['params']]
    if len(owned)!=len(params) or {id(p) for p in owned}!={id(p) for p in params}:
        raise ValueError('Optimizer must own every parameter exactly once')
    for group in optimizer.param_groups:
        if (group['lr'],tuple(group['betas']),group['weight_decay'])!=(1e-5,(.9,.95),0.):
            raise ValueError('Frozen AdamW hyperparameters changed')
    validate_optimizer_steps(named,new_names,optimizer,surface_step)
    return named,new_names


def validate_recovery(events,checkpoint_step):
    begun={};completed=set();inputs_done=set()
    for row in events:
        if row.get('event') in ('INPUT_BEGIN','UPDATE_BEGIN'):
            step=int(row['surface_step'])
            if step>checkpoint_step:raise RuntimeError('Uncheckpointed attempted surface update; automatic replay prohibited')
            key=(row['event'],step)
            if key in begun:raise RuntimeError('Duplicate durable input/update attempt')
            begun[key]=row
        if row.get('event')=='INPUT_DONE':
            step=int(row['surface_step'])
            if step in inputs_done:raise RuntimeError('Duplicate input return')
            inputs_done.add(step)
        if row.get('event')=='UPDATE_DONE':
            step=int(row['surface_step'])
            if step in completed or step>checkpoint_step:raise RuntimeError('Duplicate or uncovered completed update')
            completed.add(step)
    if completed!=set(range(1,checkpoint_step+1)):raise RuntimeError('Checkpoint cursor lacks durable completed updates')
    if {step for kind,step in begun if kind=='UPDATE_BEGIN'}!=completed:raise RuntimeError('Update attempt/return disagreement')
    if {step for kind,step in begun if kind=='INPUT_BEGIN'}!=completed or inputs_done!=completed:
        raise RuntimeError('Input attempt/return disagreement')


def _timed_cuda(budget,kind,fn,context):
    torch.cuda.synchronize();tick=time.perf_counter()
    def complete():
        value=fn();torch.cuda.synchronize();return value
    value=budget.call(kind,complete,context=context)
    return value,time.perf_counter()-tick


def _norm(values):
    values=list(values)
    if not values:return 0.
    return float(torch.stack([v.detach().float().square().sum() for v in values]).sum().sqrt())


def update(model,optimizer,samples,budget,arm,step):
    if len(samples)!=8:raise ValueError('Effective batch8 / microbatch1 is frozen')
    named,new_names=_validate_optimizer(model,optimizer,step-1);params=list(named.values())
    new={name:named[name] for name in sorted(new_names)}
    before={name:param.detach().clone() for name,param in new.items()}
    denominator=9*sum(int((s['valid_mask']&~s['known_mask']).sum()) for s in samples)
    if denominator<=0:raise ValueError('No valid free coordinates')
    optimizer.zero_grad(set_to_none=True);set_profile(model,False,collect_stats=True)
    tick=time.perf_counter();result=dict(FM=0.,loss=0.,free_coordinate_denominator=denominator,
        forward_seconds=0.,loss_seconds=0.,backward_seconds=0.,clip_seconds=0.,optimizer_seconds=0.,validation_seconds=0.,
        new_parameter_count=sum(p.numel() for p in new.values()),new_parameter_norm_before=_norm(new.values()))
    per_sample=[];observations=[]
    for micro,sample in enumerate(samples):
        if sample['lambda_value']!=.25:raise ValueError('All three arms require HYBRID lambda=.25')
        batch=collate([sample],params[0].device)
        context=dict(arm=arm,surface_step=step,conditional_total_step=5000+step,microbatch=micro,
            sample_index=sample['sample_index'],shared_input_sha256=sample['shared_input_sha256'])
        with native_execution(model,coordinates=(batch['xt'],batch['t']),autocast=True) as runtime:
            velocity,seconds=_timed_cuda(budget,'training_forward',lambda:model(*model_inputs(batch)).float(),context)
            result['forward_seconds']+=seconds
            actual=[]
            for layer in model.relation_observations:
                for item in layer['batches']:
                    values={name:float(value.detach()) if isinstance(value,torch.Tensor) else value for name,value in item.items()}
                    actual.append(dict(layer=layer['layer'],**values))
            if arm=='H_CONT' and actual:raise ValueError('H_CONT has unexpected relation observations')
            if arm!='H_CONT' and len(actual)!=len(model.surface_attention_metadata['layers']):
                raise ValueError('Every enabled relation layer must expose one microbatch observation')
            observations.extend(dict(microbatch=micro,sample_index=sample['sample_index'],**r) for r in actual)
            loss_tick=time.perf_counter()
            loss=masked_fm_loss(velocity,batch['u'],batch['valid_mask'],batch['known_mask'],denominator=denominator)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite actual FM loss')
            contribution=float(loss.detach());result['FM']+=contribution;result['loss']+=contribution
            torch.cuda.synchronize();result['loss_seconds']+=time.perf_counter()-loss_tick
            _,seconds=_timed_cuda(budget,'training_backward',loss.backward,context);result['backward_seconds']+=seconds
            per_sample.append(dict(sample_index=sample['sample_index'],task_id=sample['task_id'],
                free_coordinates=sample['free_coordinate_count'],FM_contribution=contribution,
                FM_per_sample_MSE=contribution*denominator/sample['free_coordinate_count'],relation_observations=actual))
        del velocity,loss,batch
    result['new_grad_norm_before_clip']=_norm(p.grad for p in new.values() if p.grad is not None)
    clip_tick=time.perf_counter();result['grad_norm_before_clip']=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
    result['new_grad_norm_after_clip']=_norm(p.grad for p in new.values() if p.grad is not None)
    torch.cuda.synchronize();result['clip_seconds']=time.perf_counter()-clip_tick
    _,result['optimizer_seconds']=_timed_cuda(budget,'optimizer_update',optimizer.step,
        dict(arm=arm,surface_step=step,conditional_total_step=5000+step))
    validation_tick=time.perf_counter()
    if not all(bool(torch.isfinite(p).all()) for p in params):raise FloatingPointError('Nonfinite parameter after update; no retry')
    validate_optimizer_steps(named,new_names,optimizer,step)
    result['new_parameter_norm_after']=_norm(new.values())
    result['new_parameter_delta_l2']=_norm(new[name]-before[name] for name in new)
    torch.cuda.synchronize();result['validation_seconds']=time.perf_counter()-validation_tick
    for source,target in (('logit_RMS','logit_correction_RMS_mean'),('weighted_pair_value_RMS','weighted_pair_value_RMS_mean'),('content_delta_RMS','content_delta_RMS_mean')):
        values=[r[source] for r in observations if source in r]
        result[target]=sum(values)/len(values) if values else 0.
    result.update(relation_observation_count=len(observations),update_total_seconds=time.perf_counter()-tick,
        sample_diagnostics=per_sample,relation_observations=observations,runtime=runtime,
        correction_mean_scope='Arithmetic mean of per-layer per-microbatch RMS; not a global coordinate RMS')
    return result


def train_arm(out,arm,protocol,budget,model,optimizer,meta,stream_state):
    from . import checkpoint
    out=Path(out).resolve()
    if read_json(out/'gate_results.json')['status']!='PASS':raise RuntimeError('Numerical/cost gate failed: all formal training prohibited')
    if arm not in ARMS or model._surface_attention_arm!=arm:raise ValueError('Unregistered or mismatched arm')
    folder=out/'training';folder.mkdir(parents=True,exist_ok=True)
    status_path=folder/f'{arm}_status.json';events_path=folder/f'{arm}_events.jsonl'
    if status_path.exists() and read_json(status_path).get('status')=='BLOCKED':raise RuntimeError('Affected arm blocked; no silent retry')
    stream=SurfaceStream(SOURCE,CHAIR,out,arm);step=int(meta.get('surface_step',0))
    if step:
        if meta['arm']!=arm or meta['protocol_sha256']!=json_hash(protocol):raise ValueError('Restored surface identity differs')
        stream.load_state_dict(stream_state)
    elif meta.get('conditional_total_step')!=5000:
        raise ValueError('Only audited HYBRID total5000 initializes surface0')
    if stream.batch_index!=step:raise ValueError('Restored task cursor differs')
    validate_recovery(ledger_rows(events_path),step);_validate_optimizer(model,optimizer,step)
    if step==1000:return dict(status='DONE',arm=arm,surface_step=step,actual_updates_this_worker=0)
    start=step;baseline={k:budget.attempts[k] for k in ('training_forward','training_backward','optimizer_update')}
    state=dict(status='TRAINING',arm=arm,surface_step=step,conditional_total_step=5000+step,
        protocol_sha256=json_hash(protocol),plan_sha256=stream.plan_sha256,actual_updates_this_worker=0,
        registered_new_parameter_names=model.surface_attention_metadata['new_parameter_names'],lambda_variance=.25)
    shared_path=folder/'shared_batch_hashes.jsonl';shared={r['surface_step']:r for r in ledger_rows(shared_path)}
    stop={'signal':None}
    def request_stop(signum,frame):stop['signal']=int(signum)
    handlers={s:signal.signal(s,request_stop) for s in (signal.SIGINT,signal.SIGTERM)}
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter();save_seconds=0.
    timing={k:0. for k in ('OT_and_input_seconds','forward_seconds','loss_seconds','backward_seconds','clip_seconds','optimizer_seconds','validation_seconds','update_total_seconds')}
    def persist(current):
        nonlocal save_seconds
        append_event(events_path,dict(event='CHECKPOINT_BEGIN',arm=arm,surface_step=current))
        record=checkpoint.save(out,arm,model,optimizer,stream,protocol,current)
        save_seconds+=record.get('save_verify_seconds',0.)
        append_event(events_path,dict(event='CHECKPOINT_DONE',arm=arm,surface_step=current,checkpoint=record));return record
    try:
        atomic_json(status_path,state)
        with (folder/f'{arm}_training_log.csv').open('a',newline='',encoding='utf-8') as log,(folder/f'{arm}_sample_exposure.csv').open('a',newline='',encoding='utf-8') as exposure:
            writer=csv.DictWriter(log,fieldnames=LOG_COLUMNS);ew=csv.DictWriter(exposure,fieldnames=EXPOSURE_COLUMNS)
            if log.tell()==0:writer.writeheader()
            if exposure.tell()==0:ew.writeheader()
            for current in range(start+1,1001):
                if stop['signal'] is not None:
                    state.update(status='INTERRUPTED_SAFE',signal=stop['signal'])
                    if step>start and step%200:state['checkpoint']=persist(step)
                    break
                append_event(events_path,dict(event='INPUT_BEGIN',arm=arm,surface_step=current,stream_batch_index=stream.batch_index,OT_counts_before=dict(stream.counts)))
                before=dict(stream.counts);tick=time.perf_counter();samples=stream.next_effective_batch();input_seconds=time.perf_counter()-tick
                delta={k:stream.counts[k]-before[k] for k in before}
                if delta['actual_OT_attempts']!=delta['actual_OT_returns'] or delta['actual_OT_returns']+delta['OT_cache_hits']!=8:
                    raise ValueError('Incomplete actual batch OT accounting')
                bh=stream.last_audit['shared_batch_sha256'];ch=stream.last_audit['coupled_batch_sha256']
                if current in shared and (shared[current]['shared_batch_sha256']!=bh or shared[current]['coupled_batch_sha256']!=ch):
                    raise ValueError('Three-arm actual shared task/Z/epsilon/t/OT/C/alpha/FM input mismatch')
                if current not in shared:
                    shared[current]=dict(surface_step=current,shared_batch_sha256=bh,coupled_batch_sha256=ch,first_arm=arm)
                    append_event(shared_path,shared[current])
                append_event(events_path,dict(event='INPUT_DONE',arm=arm,surface_step=current,shared_batch_sha256=bh,coupled_batch_sha256=ch,
                    OT_delta=delta,effective_free_coordinates=stream.last_audit['effective_free_coordinates']))
                for sample in samples:
                    ew.writerow(dict(arm=arm,surface_step=current,sample_index=sample['sample_index'],task_id=sample['task_id'],uid=sample['object_id'],
                        N=int(sample['y']),K=sample['K'],ratio=sample['ratio'],alpha=sample['alpha'],t=float(sample['t']),
                        free_coordinates=sample['free_coordinate_count'],shared_input_sha256=sample['shared_input_sha256'],coupled_input_sha256=sample['coupled_input_sha256'],
                        ot_cache_key=sample['ot_cache_key'],ot_cache_status=sample['ot_cache_status']))
                exposure.flush();os.fsync(exposure.fileno())
                append_event(events_path,dict(event='UPDATE_BEGIN',arm=arm,surface_step=current,completed_updates_before=step,shared_batch_sha256=bh))
                result=update(model,optimizer,samples,budget,arm,current);step=current
                row=dict(arm=arm,surface_step=step,conditional_total_step=5000+step,shared_batch_sha256=bh,coupled_batch_sha256=ch,
                    OT_and_input_seconds=input_seconds,**delta,**{k:result[k] for k in LOG_COLUMNS if k in result},
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
                append_event(events_path,dict(event='UPDATE_DONE',**row,sample_diagnostics=result['sample_diagnostics'],relation_observations=result['relation_observations']))
                writer.writerow(row);log.flush();os.fsync(log.fileno())
                for key in timing:timing[key]+=row[key]
                state.update(surface_step=step,conditional_total_step=5000+step,actual_updates_this_worker=step-start,last_row=row)
                atomic_json(status_path,state)
                if step%200==0:
                    state['checkpoint']=persist(step)
                    print(str(dict(event='TRAIN_PROGRESS',arm=arm,surface_step=step,FM=row['FM'],new_grad=row['new_grad_norm_before_clip'])),flush=True)
            else:state['status']='DONE'
        state.update(OT_counts=dict(stream.counts),timing_seconds=timing,save_verify_seconds=save_seconds,
            worker_wall_seconds=time.perf_counter()-started,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            forward_attempts_this_worker=budget.attempts['training_forward']-baseline['training_forward'],
            backward_attempts_this_worker=budget.attempts['training_backward']-baseline['training_backward'],
            optimizer_attempts_this_worker=budget.attempts['optimizer_update']-baseline['optimizer_update'],
            timing_note='Update total contains forward/loss/backward/clip/optimizer/validation; OT and save are separate; never sum nested scopes.')
        atomic_json(status_path,state);atomic_json(folder/f'{arm}_summary.json',state)
        append_event(events_path,dict(event='ARM_END',**state));budget.snapshot();return state
    except BaseException as exc:
        state.update(status='BLOCKED',surface_step=step,conditional_total_step=5000+step,error=repr(exc),traceback=traceback.format_exc(),
            actual_updates_this_worker=step-start,OT_counts=dict(stream.counts))
        atomic_json(status_path,state);append_event(events_path,dict(event='ARM_FAILED',**state));budget.snapshot();raise
    finally:
        for signum,handler in handlers.items():signal.signal(signum,handler)


def dispatch_check(snapshot):
    from experiments.mf_h1_local0.control_mvp48.train_run import dispatch_check as inherited
    admission=inherited(snapshot)
    headroom=max(2048.,.15*admission['total_mib'])
    if admission['free_mib']<=headroom:
        raise RuntimeError('Surface experiment requires at least 15% VRAM headroom before loading')
    admission.update(headroom_mib=headroom,headroom_fraction=.15,OT_cache=True,
        rationale='One GPU worker owns the shared lock; registered precheck supplies measured full F+B cost and memory. No additional GPU probe.')
    return admission


def run(out,arm):
    """Single-use worker entry: gate -> identities/resources -> lock -> CUDA/load."""
    out=Path(out).resolve()
    if arm not in ARMS:raise ValueError('Unknown arm')
    if read_json(out/'gate_results.json')['status']!='PASS':raise RuntimeError('Gate must PASS before any training/model load')
    from .registration import verify_identity
    from . import checkpoint
    from experiments.mf_h1_local0.control_mvp48.train_io import resource_snapshot
    from experiments.mf_h1_local0.control_mvp48.train_run import GPUMonitor
    from experiments.mf_h1_local0.chair_domain_control_v1.training import process_gpu_lock
    verify_identity(out);stage='train_'+arm;path=out/'stages'/(stage+'.json')
    if path.exists():
        previous=read_json(path)
        if previous['status']=='DONE':return previous
        if previous['status'] in ('BLOCKED','RUNNING'):raise RuntimeError('Prior failed/interrupted stage cannot silently replay')
    if torch.cuda.is_initialized():raise RuntimeError('CUDA initialized before existing shared GPU lock')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);_runtime.configure_stable_runtime()
    before=resource_snapshot();admission=dispatch_check(before)
    atomic_json(out/'resources'/(stage+'_before.json'),dict(resources=before,admission=admission))
    state=dict(status='RUNNING',stage=stage,arm=arm,started_unix=time.time(),gpu_workers=1)
    atomic_json(path,state);monitor=None;budget=None;started=time.perf_counter()
    try:
        with process_gpu_lock(out) as lock_wait:
            dispatch_check(resource_snapshot());budget=Budget(out,stage)
            (out/'training').mkdir(parents=True,exist_ok=True);monitor=GPUMonitor(out);monitor.start()
            tick=time.perf_counter();pointer=out/'training'/(arm+'_pointer.json')
            if pointer.exists():
                ref=read_json(pointer)
                if digest(ref['path'])!=ref['checkpoint_sha256']:raise ValueError('Recovery checkpoint file changed')
                model,opt,meta,hashes,stream_state=checkpoint.load(ref['path'],'cuda:0',trainable=True,optimizer=True)
            else:model,opt,meta,hashes,stream_state=checkpoint.load_start(arm,device='cuda:0',optimizer=True)
            load_seconds=time.perf_counter()-tick
            result=train_arm(out,arm,read_json(out/'protocol.json'),budget,model,opt,meta,stream_state)
            torch.cuda.synchronize();budget.snapshot();monitor.stop();monitor=None
            state.update(status=result['status'],result=result,lock_wait_seconds=lock_wait,load_seconds=load_seconds,
                wall_seconds=time.perf_counter()-started,finished_unix=time.time(),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
            if digest(START)!=START_SHA:raise ValueError('Protected starting checkpoint changed')
            atomic_json(path,state);atomic_json(out/'resources'/(stage+'_after.json'),resource_snapshot());return state
    except BaseException as exc:
        state.update(status='BLOCKED',error=repr(exc),traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-started)
        atomic_json(path,state)
        if budget is not None:budget.snapshot()
        raise
    finally:
        if monitor is not None:monitor.stop()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);p.add_argument('--arm',choices=ARMS,required=True)
    a=p.parse_args();print(run(a.out,a.arm)['status'])
