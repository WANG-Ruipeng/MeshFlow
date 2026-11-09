"""Fixed-budget two-arm HYBRID continuation; numerical gates precede loading.

The public train_arm accepts an already restored model under the caller's GPU
lock. The parent runner owns the lock and GPU dispatch. Neither CPU audits nor imports invoke
a model, optimizer step, or CUDA initialization.
"""
from __future__ import annotations

import argparse
import csv
import os
import signal
import time
import traceback
import numpy as np
from pathlib import Path

from native_t1 import runtime as _runtime
import torch
from native_t1.model import native_execution
from native_t1.data import model_inputs
from native_t1.training import masked_fm_loss
from experiments.mf_h1_local0.control_mvp48.train_stream import collate
from .common import SOURCE,CHAIR,OUT,START,START_SHA,ARMS,append_event,atomic_json,read_json,ledger_rows,json_hash,digest
from .stream import MomentStream
from .ledger import Budget

LOG_COLUMNS=('arm','moment_step','conditional_total_step','shared_batch_sha256','coupled_batch_sha256',
    'FM','loss','free_coordinate_denominator','grad_norm_before_clip','new_parameter_count',
    'new_parameter_norm_before','new_parameter_norm_after','new_parameter_delta_l2',
    'old_grad_norm_before_clip','old_grad_norm_after_clip',
    'new_grad_norm_before_clip','new_grad_norm_after_clip',
    'new_first_projection_grad_norm','new_final_projection_grad_norm',
    'new_first_projection_nonzero_grad_tensors','new_final_projection_nonzero_grad_tensors',
    'unused_parameter_count','old_parameter_count','logit_correction_RMS_mean',
    'b_RMS_mean','b_max','u_RMS_mean','u_max',
    'weighted_pair_value_RMS_mean','content_delta_RMS_mean','relation_observation_count',
    'OT_and_input_seconds','forward_seconds','loss_seconds','backward_seconds','clip_seconds',
    'optimizer_seconds','validation_seconds','update_total_seconds','actual_OT_attempts','actual_OT_returns','OT_cache_hits',
    'cuda_peak_allocated_bytes','cuda_peak_reserved_bytes')
EXPOSURE_COLUMNS=('arm','moment_step','sample_index','task_id','uid','N','K','ratio','alpha','t',
    'free_coordinates','shared_input_sha256','coupled_input_sha256','ot_cache_key','ot_cache_status')


def validate_training_protocol(protocol):
    expected=dict(seed=20261006,updates_per_arm=1000,base_conditional_total_step=5000,
                  effective_batch=8,microbatch=1,lambda_variance=.25,checkpoint_interval=200)
    for key,value in expected.items():
        if protocol.get(key)!=value:raise ValueError('Frozen V2 protocol differs: '+key)
    if protocol.get('experiment',protocol.get('experiment_id'))!='chair_moment_channel_v2':
        raise ValueError('Wrong experiment identity')
    if tuple(protocol.get('arms',()))!=tuple(ARMS):raise ValueError('Exactly H_CONT and MOMENT_XY are required')


def batch_contract(samples):
    if len(samples)!=8:raise ValueError('Effective batch8 / microbatch1 is frozen')
    if sum(float(sample['alpha'])==1. for sample in samples)!=4:
        raise ValueError('Every effective batch requires four exact-unit alpha samples')
    denominator=0
    for sample in samples:
        valid=np.asarray(sample['valid_mask']);known=np.asarray(sample['known_mask'])
        if valid.dtype!=np.bool_ or known.dtype!=np.bool_ or valid.shape!=known.shape:
            raise ValueError('Invalid Boolean face masks')
        if np.any(known & ~valid):raise ValueError('Known face cannot be invalid padding')
        count=9*int(np.count_nonzero(valid & ~known))
        if count<=0 or count!=sample['free_coordinate_count']:
            raise ValueError('Saved free coordinate count differs from actual valid/free mask')
        if float(sample['lambda_value'])!=.25 or not .9-1e-7<=float(sample['alpha'])<=1.1+1e-7:
            raise ValueError('Frozen HYBRID mixture or alpha bounds differ')
        denominator+=count
    return denominator


def projection_parameter_names(model,new_names):
    metadata=model.surface_attention_metadata
    first=set(metadata.get('first_projection_parameter_names',
        [name for name in new_names if '.fc1.' in name]))
    final=set(metadata.get('final_projection_parameter_names',
        [name for name in new_names if '.fc2.' in name]))
    if first & final or first | final != new_names:
        raise ValueError('First/final projection name partition differs from registration')
    return first,final


def parameter_partition(model):
    names=set(model.surface_attention_metadata['new_parameter_names'])
    parameters=dict(model.named_parameters())
    actual={name for name in parameters if name.startswith('relation_modules.')}
    if actual!=names:raise ValueError('New parameter namespace differs from registered adapter metadata')
    if model._surface_attention_arm=='H_CONT' and names:raise ValueError('H_CONT cannot own new relation parameters')
    return parameters,names


def validate_optimizer_steps(named,new_names,optimizer,moment_step):
    """Old states continue from 5000; only registered new states start at zero."""
    for name,param in named.items():
        state=optimizer.state.get(param,{})
        expected=moment_step if name in new_names else 5000+moment_step
        if not state:
            if name in new_names and moment_step==0:continue  # AdamW lazy zero-state initialization.
            raise ValueError('Missing restored optimizer state: '+name)
        if int(state.get('step',-1))!=expected:raise ValueError('AdamW cursor differs for '+name)
        for key in ('exp_avg','exp_avg_sq'):
            if key not in state or state[key].shape!=param.shape or state[key].dtype!=torch.float32:
                raise ValueError('AdamW moment shape/dtype missing for '+name)
            if name in new_names and moment_step==0 and bool(torch.count_nonzero(state[key])):
                raise ValueError('New relation moment must initialize to zero')


def _validate_optimizer(model,optimizer,moment_step):
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
    validate_optimizer_steps(named,new_names,optimizer,moment_step)
    return named,new_names


def validate_recovery(events,checkpoint_step):
    begun={};completed=set();inputs_done=set()
    for row in events:
        if row.get('event') in ('INPUT_BEGIN','UPDATE_BEGIN'):
            step=int(row['moment_step'])
            if step>checkpoint_step:raise RuntimeError('Uncheckpointed attempted moment update; automatic replay prohibited')
            key=(row['event'],step)
            if key in begun:raise RuntimeError('Duplicate durable input/update attempt')
            begun[key]=row
        if row.get('event')=='INPUT_DONE':
            step=int(row['moment_step'])
            if step in inputs_done:raise RuntimeError('Duplicate input return')
            inputs_done.add(step)
        if row.get('event')=='UPDATE_DONE':
            step=int(row['moment_step'])
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
    from .model_adapter import set_profile
    denominator=batch_contract(samples)
    named,new_names=_validate_optimizer(model,optimizer,step-1);params=list(named.values())
    new={name:named[name] for name in sorted(new_names)}
    old={name:param for name,param in named.items() if name not in new_names}
    first_names,final_names=projection_parameter_names(model,new_names)
    before={name:param.detach().clone() for name,param in new.items()}
    optimizer.zero_grad(set_to_none=True);set_profile(model,False,collect_stats=True)
    tick=time.perf_counter();result=dict(FM=0.,loss=0.,free_coordinate_denominator=denominator,
        forward_seconds=0.,loss_seconds=0.,backward_seconds=0.,clip_seconds=0.,optimizer_seconds=0.,validation_seconds=0.,
        new_parameter_count=sum(p.numel() for p in new.values()),
        old_parameter_count=sum(p.numel() for p in old.values()),
        new_parameter_norm_before=_norm(new.values()))
    per_sample=[];observations=[]
    for micro,sample in enumerate(samples):
        if sample['lambda_value']!=.25:raise ValueError('Both arms require HYBRID lambda=.25')
        batch=collate([sample],params[0].device)
        context=dict(arm=arm,moment_step=step,conditional_total_step=5000+step,microbatch=micro,
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
    unused=[name for name,param in named.items() if param.grad is None]
    result['unused_parameter_count']=len(unused)
    result['unused_parameter_names']=unused
    if unused:raise RuntimeError('Unused or disconnected registered trainable parameters: '+str(unused))
    result['new_parameter_gradient_norms']={name:_norm([param.grad]) for name,param in new.items()}
    result['old_grad_norm_before_clip']=_norm(p.grad for p in old.values())
    result['new_grad_norm_before_clip']=_norm(p.grad for p in new.values())
    result['new_first_projection_grad_norm']=_norm(named[name].grad for name in sorted(first_names))
    result['new_final_projection_grad_norm']=_norm(named[name].grad for name in sorted(final_names))
    result['new_first_projection_nonzero_grad_tensors']=sum(bool(torch.count_nonzero(named[name].grad)) for name in first_names)
    result['new_final_projection_nonzero_grad_tensors']=sum(bool(torch.count_nonzero(named[name].grad)) for name in final_names)
    clip_tick=time.perf_counter();result['grad_norm_before_clip']=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
    result['new_grad_norm_after_clip']=_norm(p.grad for p in new.values())
    result['old_grad_norm_after_clip']=_norm(p.grad for p in old.values())
    torch.cuda.synchronize();result['clip_seconds']=time.perf_counter()-clip_tick
    _,result['optimizer_seconds']=_timed_cuda(budget,'optimizer_update',optimizer.step,
        dict(arm=arm,moment_step=step,conditional_total_step=5000+step))
    validation_tick=time.perf_counter()
    if not all(bool(torch.isfinite(p).all()) for p in params):raise FloatingPointError('Nonfinite parameter after update; no retry')
    validate_optimizer_steps(named,new_names,optimizer,step)
    result['new_parameter_norm_after']=_norm(new.values())
    result['new_parameter_delta_l2']=_norm(new[name]-before[name] for name in new)
    torch.cuda.synchronize();result['validation_seconds']=time.perf_counter()-validation_tick
    for source,target in (('logit_RMS','logit_correction_RMS_mean'),('b_RMS','b_RMS_mean'),('u_RMS','u_RMS_mean'),('weighted_pair_value_RMS','weighted_pair_value_RMS_mean'),('content_delta_RMS','content_delta_RMS_mean')):
        values=[r[source] for r in observations if source in r]
        result[target]=sum(values)/len(values) if values else 0.
    result['b_max']=max((r.get('b_max',r.get('logit_max',0.)) for r in observations),default=0.)
    result['u_max']=max((r.get('u_max',0.) for r in observations),default=0.)
    result.update(relation_observation_count=len(observations),update_total_seconds=time.perf_counter()-tick,
        sample_diagnostics=per_sample,relation_observations=observations,runtime=runtime,
        correction_mean_scope='Arithmetic mean of per-layer per-microbatch RMS; not a global coordinate RMS')
    return result


def train_arm(out,arm,protocol,budget,model,optimizer,meta,stream_state):
    from . import checkpoint
    out=Path(out).resolve()
    validate_training_protocol(protocol)
    if read_json(out/'gate_results.json')['status']!='PASS':raise RuntimeError('V2 padding/numerical/cost gate must pass before formal training')
    if arm not in ARMS or model._surface_attention_arm!=arm:raise ValueError('Unregistered or mismatched arm')
    folder=out/'training';folder.mkdir(parents=True,exist_ok=True)
    status_path=folder/f'{arm}_status.json';events_path=folder/f'{arm}_events.jsonl'
    if status_path.exists() and read_json(status_path).get('status')=='BLOCKED':raise RuntimeError('Affected arm blocked; no silent retry')
    stream=MomentStream(SOURCE,CHAIR,out,arm);step=int(meta.get('moment_step',0))
    if step:
        if meta['arm']!=arm or meta['protocol_sha256']!=json_hash(protocol):raise ValueError('Restored moment identity differs')
        stream.load_state_dict(stream_state)
    elif meta.get('conditional_total_step')!=5000:
        raise ValueError('Only audited HYBRID total5000 initializes moment0')
    if stream.batch_index!=step:raise ValueError('Restored task cursor differs')
    validate_recovery(ledger_rows(events_path),step);_validate_optimizer(model,optimizer,step)
    if step==1000:return dict(status='DONE',arm=arm,moment_step=step,actual_updates_this_worker=0)
    start=step;baseline={k:budget.attempts[k] for k in ('training_forward','training_backward','optimizer_update')}
    state=dict(status='TRAINING',arm=arm,moment_step=step,conditional_total_step=5000+step,
        protocol_sha256=json_hash(protocol),plan_sha256=stream.plan_sha256,actual_updates_this_worker=0,
        registered_new_parameter_names=model.surface_attention_metadata['new_parameter_names'],lambda_variance=.25)
    shared_path=folder/'shared_batch_hashes.jsonl';shared_rows=ledger_rows(shared_path)
    shared={r['moment_step']:r for r in shared_rows}
    if len(shared)!=len(shared_rows):raise ValueError('Duplicate actual shared batch registration')
    stop={'signal':None}
    def request_stop(signum,frame):stop['signal']=int(signum)
    handlers={s:signal.signal(s,request_stop) for s in (signal.SIGINT,signal.SIGTERM)}
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter();save_seconds=0.
    timing={k:0. for k in ('OT_and_input_seconds','forward_seconds','loss_seconds','backward_seconds','clip_seconds','optimizer_seconds','validation_seconds','update_total_seconds')}
    def persist(current):
        nonlocal save_seconds
        append_event(events_path,dict(event='CHECKPOINT_BEGIN',arm=arm,moment_step=current))
        record=checkpoint.save(out,arm,model,optimizer,stream,protocol,current)
        save_seconds+=record.get('save_verify_seconds',0.)
        append_event(events_path,dict(event='CHECKPOINT_DONE',arm=arm,moment_step=current,checkpoint=record));return record
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
                append_event(events_path,dict(event='INPUT_BEGIN',arm=arm,moment_step=current,stream_batch_index=stream.batch_index,OT_counts_before=dict(stream.counts)))
                before=dict(stream.counts);tick=time.perf_counter();samples=stream.next_effective_batch();input_seconds=time.perf_counter()-tick
                delta={k:stream.counts[k]-before[k] for k in before}
                if delta['actual_OT_attempts']!=delta['actual_OT_returns'] or delta['actual_OT_returns']+delta['OT_cache_hits']!=8:
                    raise ValueError('Incomplete actual batch OT accounting')
                bh=stream.last_audit['shared_batch_sha256'];ch=stream.last_audit['coupled_batch_sha256']
                if current in shared and (shared[current]['shared_batch_sha256']!=bh or shared[current]['coupled_batch_sha256']!=ch):
                    raise ValueError('Two-arm actual shared task/Z/epsilon/t/OT/C/alpha/FM input mismatch')
                if current not in shared:
                    shared[current]=dict(moment_step=current,shared_batch_sha256=bh,coupled_batch_sha256=ch,first_arm=arm,
                        effective_free_coordinates=stream.last_audit['effective_free_coordinates'],samples=stream.last_audit['samples'])
                    append_event(shared_path,shared[current])
                append_event(events_path,dict(event='INPUT_DONE',arm=arm,moment_step=current,shared_batch_sha256=bh,coupled_batch_sha256=ch,
                    OT_delta=delta,effective_free_coordinates=stream.last_audit['effective_free_coordinates']))
                for sample in samples:
                    ew.writerow(dict(arm=arm,moment_step=current,sample_index=sample['sample_index'],task_id=sample['task_id'],uid=sample['object_id'],
                        N=int(sample['y']),K=sample['K'],ratio=sample['ratio'],alpha=sample['alpha'],t=float(sample['t']),
                        free_coordinates=sample['free_coordinate_count'],shared_input_sha256=sample['shared_input_sha256'],coupled_input_sha256=sample['coupled_input_sha256'],
                        ot_cache_key=sample['ot_cache_key'],ot_cache_status=sample['ot_cache_status']))
                exposure.flush();os.fsync(exposure.fileno())
                append_event(events_path,dict(event='UPDATE_BEGIN',arm=arm,moment_step=current,completed_updates_before=step,shared_batch_sha256=bh))
                result=update(model,optimizer,samples,budget,arm,current);step=current
                if result['free_coordinate_denominator']!=stream.last_audit['effective_free_coordinates']:
                    raise ValueError('Actual FM denominator differs from registered batch coordinates')
                row=dict(arm=arm,moment_step=step,conditional_total_step=5000+step,shared_batch_sha256=bh,coupled_batch_sha256=ch,
                    OT_and_input_seconds=input_seconds,**delta,**{k:result[k] for k in LOG_COLUMNS if k in result},
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
                append_event(events_path,dict(event='UPDATE_DONE',**row,sample_diagnostics=result['sample_diagnostics'],relation_observations=result['relation_observations'],
                    unused_parameter_names=result['unused_parameter_names'],new_parameter_gradient_norms=result['new_parameter_gradient_norms']))
                writer.writerow(row);log.flush();os.fsync(log.fileno())
                for key in timing:timing[key]+=row[key]
                state.update(moment_step=step,conditional_total_step=5000+step,actual_updates_this_worker=step-start,last_row=row)
                atomic_json(status_path,state)
                if step%200==0:
                    state['checkpoint']=persist(step)
                    print(str(dict(event='TRAIN_PROGRESS',arm=arm,moment_step=step,FM=row['FM'],new_grad=row['new_grad_norm_before_clip'])),flush=True)
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
        state.update(status='BLOCKED',failure_class='ENGINEERING_IO' if isinstance(exc,OSError) else 'UNRESOLVED_OR_NUMERICAL',moment_step=step,conditional_total_step=5000+step,error=repr(exc),traceback=traceback.format_exc(),
            actual_updates_this_worker=step-start,OT_counts=dict(stream.counts))
        atomic_json(status_path,state);append_event(events_path,dict(event='ARM_FAILED',**state));budget.snapshot();raise
    finally:
        for signum,handler in handlers.items():signal.signal(signum,handler)

