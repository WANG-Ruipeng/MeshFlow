"""Bounded paired FM training; caller owns the GPU lock and strict restoration.

This module never loads a model or dispatches a worker. The numerical update
matches control_mvp48.train_core with objective=None: eval-mode FP32 parameters,
eight BF16 forwards, globally normalized free-coordinate squared error, clip1,
then the restored AdamW. Actual calls are reserved in the pilot Budget.
"""
from __future__ import annotations
import csv
import hashlib
import signal
import time
import traceback
from pathlib import Path
import torch
from native_t1.model import native_execution
from native_t1.data import model_inputs
from native_t1.training import masked_fm_loss
from experiments.mf_h1_local0.control_mvp48.train_stream import collate
from .common import SOURCE, CHAIR, ARMS, append_event, atomic_json, read_json, ledger_rows, json_hash
from .stream import HybridStream
from . import checkpoint

LOG_COLUMNS = ('arm','pilot_step','conditional_total_step','shared_batch_sha256',
    'coupled_batch_sha256','FM','loss','free_coordinate_denominator','grad_norm_before_clip',
    'OT_and_input_seconds','forward_seconds','loss_seconds','backward_seconds',
    'clip_seconds','optimizer_seconds','validation_seconds','update_total_seconds',
    'actual_OT_attempts','actual_OT_returns','OT_cache_hits',
    'cuda_peak_allocated_bytes','cuda_peak_reserved_bytes')
EXPOSURE_COLUMNS = ('arm','pilot_step','sample_index','task_id','uid','N','K','ratio','alpha','t',
    'free_coordinates','shared_input_sha256','coupled_input_sha256','ot_cache_key','ot_cache_status')


def _validate_optimizer(model,optimizer):
    params=list(model.parameters())
    if not isinstance(optimizer,torch.optim.AdamW):raise ValueError('Restored AdamW required')
    if model.training or not all(p.requires_grad and p.dtype==torch.float32 and p.is_cuda for p in params):
        raise ValueError('Require trainable FP32 GPU parameters with the original eval-mode dropout contract')
    owned=[p for group in optimizer.param_groups for p in group['params']]
    if len(owned)!=len(params) or {id(p) for p in owned}!={id(p) for p in params}:
        raise ValueError('Optimizer must own every parameter exactly once')
    for group in optimizer.param_groups:
        if (group['lr'],tuple(group['betas']),group['weight_decay'])!=(1e-5,(.9,.95),0.):
            raise ValueError('Restored optimizer hyperparameters changed')
    if not optimizer.state:raise ValueError('Fresh or empty optimizer is forbidden')
    return params


def validate_recovery(events,checkpoint_step):
    """Every attempted training input/update must be covered by loaded state."""
    begun={};completed=set();inputs_done=set()
    for row in events:
        if row.get('event') in ('INPUT_BEGIN','UPDATE_BEGIN'):
            step=int(row['pilot_step'])
            if step>checkpoint_step:
                raise RuntimeError('Uncheckpointed attempted training step '+str(step)+'; no automatic replay')
            key=(row['event'],step)
            if key in begun:raise RuntimeError('Duplicate input/update attempt in durable arm ledger')
            begun[key]=row
        if row.get('event')=='INPUT_DONE':
            value=int(row['pilot_step'])
            if value in inputs_done:raise RuntimeError('Duplicate input return')
            inputs_done.add(value)
        if row.get('event')=='UPDATE_DONE':
            step=int(row['pilot_step'])
            if step in completed or step>checkpoint_step:raise RuntimeError('Duplicate or uncovered completed update')
            completed.add(step)
    if completed!=set(range(1,checkpoint_step+1)):
        raise RuntimeError('Loaded pilot cursor is not covered by every durable completed update')
    if set(step for (kind,step) in begun if kind=='UPDATE_BEGIN')!=completed:
        raise RuntimeError('Update attempt does not have a durable completed return')
    if set(step for (kind,step) in begun if kind=='INPUT_BEGIN')!=completed or inputs_done!=completed:
        raise RuntimeError('Input attempts/returns do not match every checkpointed completed update')


def _timed_cuda(budget,kind,fn,context):
    torch.cuda.synchronize()
    tick=time.perf_counter()
    def completed():
        value=fn()
        torch.cuda.synchronize()
        return value
    value=budget.call(kind,completed,context=context)
    return value,time.perf_counter()-tick


def update(model,optimizer,samples,budget,arm,step):
    """Exactly one authorized optimizer update, never called by CPU checks."""
    if len(samples)!=8:raise ValueError('Effective batch8/microbatch1 is frozen')
    params=_validate_optimizer(model,optimizer)
    device=params[0].device
    denominator=9*sum(int((s['valid_mask'] & ~s['known_mask']).sum()) for s in samples)
    if denominator<=0:raise ValueError('No valid free coordinates')
    optimizer.zero_grad(set_to_none=True)
    tick=time.perf_counter()
    result=dict(FM=0.,loss=0.,free_coordinate_denominator=denominator,
        forward_seconds=0.,loss_seconds=0.,backward_seconds=0.,clip_seconds=0.,
        optimizer_seconds=0.,validation_seconds=0.)
    per_sample=[]
    for micro,sample in enumerate(samples):
        batch=collate([sample],device)
        context=dict(arm=arm,pilot_step=step,conditional_total_step=4000+step,microbatch=micro,
                     sample_index=sample['sample_index'],shared_input_sha256=sample['shared_input_sha256'])
        with native_execution(model,coordinates=(batch['xt'],batch['t']),autocast=True) as runtime:
            velocity,seconds=_timed_cuda(budget,'training_forward',lambda:model(*model_inputs(batch)).float(),context)
            result['forward_seconds']+=seconds
            loss_tick=time.perf_counter()
            loss=masked_fm_loss(velocity,batch['u'],batch['valid_mask'],batch['known_mask'],denominator=denominator)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite actual FM loss')
            contribution=float(loss.detach())
            result['FM']+=contribution;result['loss']+=contribution
            torch.cuda.synchronize();result['loss_seconds']+=time.perf_counter()-loss_tick
            _,seconds=_timed_cuda(budget,'training_backward',loss.backward,context)
            result['backward_seconds']+=seconds
            per_sample.append(dict(sample_index=sample['sample_index'],task_id=sample['task_id'],
                free_coordinates=sample['free_coordinate_count'],FM_contribution=contribution,
                FM_per_sample_MSE=contribution*denominator/sample['free_coordinate_count']))
        del velocity,loss,batch
    clip_tick=time.perf_counter()
    result['grad_norm_before_clip']=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
    torch.cuda.synchronize();result['clip_seconds']=time.perf_counter()-clip_tick
    _,result['optimizer_seconds']=_timed_cuda(budget,'optimizer_update',optimizer.step,
        dict(arm=arm,pilot_step=step,conditional_total_step=4000+step))
    validation_tick=time.perf_counter()
    if not all(bool(torch.isfinite(p).all()) for p in params):
        raise FloatingPointError('Nonfinite parameter after actual optimizer update; no retry')
    expected_step=4000+step
    if any(int(s['step'])!=expected_step for s in optimizer.state.values()):
        raise ValueError('AdamW step state diverged from pilot cursor')
    torch.cuda.synchronize();result['validation_seconds']=time.perf_counter()-validation_tick
    result.update(update_total_seconds=time.perf_counter()-tick,sample_diagnostics=per_sample,runtime=runtime)
    return result


def train_arm(out,arm,protocol,budget,model,optimizer,meta,stream_state):
    """Run remaining fixed updates; already-loaded state must be verified by caller.

    A normal signal is deferred to the next complete update and saved. A real
    failure becomes BLOCKED. Recovery never replays an attempted update whose
    model/AdamW/RNG/stream state was not durably saved.
    """
    out=Path(out).resolve()
    if arm not in ARMS:raise ValueError('Unregistered arm')
    folder=out/'training';folder.mkdir(parents=True,exist_ok=True)
    status_path=folder/f'{arm}_status.json';events_path=folder/f'{arm}_events.jsonl'
    if status_path.exists() and read_json(status_path).get('status')=='BLOCKED':
        raise RuntimeError('Affected arm is BLOCKED; no silent retry')
    stream=HybridStream(SOURCE,CHAIR,out,arm)
    step=int(meta.get('pilot_step',0))
    if 'pilot_step' in meta:
        if meta['arm']!=arm or meta['protocol_sha256']!=json_hash(protocol):raise ValueError('Restored pilot identity differs')
        stream.load_state_dict(stream_state)
    elif meta.get('variant')!='Geo' or meta.get('conditional_total_step')!=4000:
        raise ValueError('Only audited Chair Geo total4000 may initialize pilot0')
    if stream.batch_index!=step:raise ValueError('Restored task cursor differs')
    validate_recovery(ledger_rows(events_path),step)
    params=_validate_optimizer(model,optimizer)
    if any(int(s['step'])!=4000+step for s in optimizer.state.values()):raise ValueError('Restored AdamW cursor differs')
    if step==1000:
        return dict(status='DONE',arm=arm,pilot_step=step,actual_updates_this_worker=0)
    start=step
    state=dict(status='TRAINING',arm=arm,pilot_step=step,conditional_total_step=4000+step,
        protocol_sha256=json_hash(protocol),plan_sha256=stream.plan_sha256,actual_updates_this_worker=0,
        forward_attempts_this_worker=0,backward_attempts_this_worker=0,optimizer_attempts_this_worker=0)
    baseline_budget={k:budget.attempts[k] for k in ('training_forward','training_backward','optimizer_update')}
    shared_path=folder/'shared_batch_hashes.jsonl'
    shared={r['pilot_step']:r['shared_batch_sha256'] for r in ledger_rows(shared_path)}
    stop={'signal':None}
    def request_stop(signum,frame):stop['signal']=int(signum)
    old_handlers={s:signal.signal(s,request_stop) for s in (signal.SIGINT,signal.SIGTERM)}
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter();save_seconds=0.
    timing={key:0. for key in ('OT_and_input_seconds','forward_seconds','loss_seconds','backward_seconds',
                            'clip_seconds','optimizer_seconds','validation_seconds','update_total_seconds')}
    def persist(current):
        nonlocal save_seconds
        append_event(events_path,dict(event='CHECKPOINT_BEGIN',arm=arm,pilot_step=current))
        record=checkpoint.save(out,arm,model,optimizer,stream,protocol,current)
        save_seconds+=record['save_verify_seconds']
        append_event(events_path,dict(event='CHECKPOINT_DONE',arm=arm,pilot_step=current,checkpoint=record))
        return record
    try:
        atomic_json(status_path,state)
        log_path=folder/f'{arm}_training_log.csv';exposure_path=folder/f'{arm}_sample_exposure.csv'
        with log_path.open('a',newline='',encoding='utf-8') as log,exposure_path.open('a',newline='',encoding='utf-8') as exposure:
            writer=csv.DictWriter(log,fieldnames=LOG_COLUMNS)
            swriter=csv.DictWriter(exposure,fieldnames=EXPOSURE_COLUMNS)
            if log.tell()==0:writer.writeheader()
            if exposure.tell()==0:swriter.writeheader()
            for current in range(start+1,1001):
                if stop['signal'] is not None:
                    state.update(status='INTERRUPTED_SAFE',signal=stop['signal'])
                    if step>start and step%200:state['checkpoint']=persist(step)
                    break
                append_event(events_path,dict(event='INPUT_BEGIN',arm=arm,pilot_step=current,
                    stream_batch_index=stream.batch_index,OT_counts_before=dict(stream.counts)))
                before=dict(stream.counts);ot_tick=time.perf_counter()
                samples=stream.next_effective_batch();input_seconds=time.perf_counter()-ot_tick
                delta={k:stream.counts[k]-before[k] for k in before}
                if delta['actual_OT_attempts']!=delta['actual_OT_returns'] or delta['actual_OT_returns']+delta['OT_cache_hits']!=8:
                    raise ValueError('Incomplete actual batch OT accounting')
                bh=stream.last_audit['shared_batch_sha256']
                if current in shared and shared[current]!=bh:
                    raise ValueError('Actual paired task/Z/t/OT/C/alpha input hashes disagree')
                if current not in shared:
                    append_event(shared_path,dict(pilot_step=current,shared_batch_sha256=bh,first_arm=arm))
                    shared[current]=bh
                append_event(events_path,dict(event='INPUT_DONE',arm=arm,pilot_step=current,shared_batch_sha256=bh,
                    coupled_batch_sha256=stream.last_audit['coupled_batch_sha256'],OT_delta=delta,
                    effective_free_coordinates=stream.last_audit['effective_free_coordinates']))
                for s in samples:
                    swriter.writerow(dict(arm=arm,pilot_step=current,sample_index=s['sample_index'],task_id=s['task_id'],
                        uid=s['object_id'],N=int(s['y']),K=s['K'],ratio=s['ratio'],alpha=s['alpha'],t=float(s['t']),
                        free_coordinates=s['free_coordinate_count'],shared_input_sha256=s['shared_input_sha256'],
                        coupled_input_sha256=s['coupled_input_sha256'],ot_cache_key=s['ot_cache_key'],ot_cache_status=s['ot_cache_status']))
                exposure.flush()
                import os
                os.fsync(exposure.fileno())
                append_event(events_path,dict(event='UPDATE_BEGIN',arm=arm,pilot_step=current,completed_updates_before=step,
                    shared_batch_sha256=bh))
                result=update(model,optimizer,samples,budget,arm,current)
                step=current
                row=dict(arm=arm,pilot_step=step,conditional_total_step=4000+step,shared_batch_sha256=bh,
                    coupled_batch_sha256=stream.last_audit['coupled_batch_sha256'],OT_and_input_seconds=input_seconds,
                    **delta,**{key:result[key] for key in LOG_COLUMNS if key in result},
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
                append_event(events_path,dict(event='UPDATE_DONE',**row,sample_diagnostics=result['sample_diagnostics']))
                writer.writerow(row);log.flush();os.fsync(log.fileno())
                for key in timing:timing[key]+=row[key]
                state.update(pilot_step=step,conditional_total_step=4000+step,actual_updates_this_worker=step-start,last_row=row)
                atomic_json(status_path,state)
                if step%200==0:
                    state['checkpoint']=persist(step)
                    print(json_hash(row)+' '+str(dict(event='TRAIN_PROGRESS',arm=arm,pilot_step=step,FM=row['FM'])),flush=True)
            else:
                state['status']='DONE'
        state.update(OT_counts=dict(stream.counts),timing_seconds=timing,save_verify_seconds=save_seconds,
            worker_wall_seconds=time.perf_counter()-started,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            timing_note='Phase-exclusive OT/input, forward, loss, backward, clip, optimizer, validation. update_total_seconds contains forward through validation; never add it to its components. Save separate.')
        state.update(forward_attempts_this_worker=budget.attempts['training_forward']-baseline_budget['training_forward'],
            backward_attempts_this_worker=budget.attempts['training_backward']-baseline_budget['training_backward'],
            optimizer_attempts_this_worker=budget.attempts['optimizer_update']-baseline_budget['optimizer_update'])
        atomic_json(status_path,state);atomic_json(folder/f'{arm}_summary.json',state)
        append_event(events_path,dict(event='ARM_END',**state));budget.snapshot()
        return state
    except BaseException as exc:
        state.update(status='BLOCKED',pilot_step=step,conditional_total_step=4000+step,error=repr(exc),
            traceback=traceback.format_exc(),actual_updates_this_worker=step-start,OT_counts=dict(stream.counts),
            forward_attempts_this_worker=budget.attempts['training_forward']-baseline_budget['training_forward'],
            backward_attempts_this_worker=budget.attempts['training_backward']-baseline_budget['training_backward'],
            optimizer_attempts_this_worker=budget.attempts['optimizer_update']-baseline_budget['optimizer_update'])
        atomic_json(status_path,state);append_event(events_path,dict(event='ARM_FAILED',**state));budget.snapshot()
        raise
    finally:
        for signum,handler in old_handlers.items():signal.signal(signum,handler)
