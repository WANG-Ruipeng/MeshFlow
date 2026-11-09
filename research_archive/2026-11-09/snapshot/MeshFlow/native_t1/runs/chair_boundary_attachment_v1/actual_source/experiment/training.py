"""Exactly 1000 paired-stream updates per arm; original FM plus registered seam."""
from native_t1 import runtime as _runtime
import csv
import os
from pathlib import Path
import signal
import time
import traceback
import torch
from native_t1.model import native_execution
from native_t1.data import model_inputs
from native_t1.training import masked_fm_loss
from experiments.mf_h1_local0.control_mvp48.train_stream import collate
from experiments.mf_h1_local0.chair_control_usability_quick_v1.quick_common import resource_snapshot
from .common import *
from . import checkpoint


def validate_optimizer(model,opt,arm,step):
    parameters=list(model.parameters())
    assert not model.training and model.experiment_trainable
    assert all(p.requires_grad and p.dtype==torch.float32 and p.is_cuda for p in parameters)
    owned=[p for g in opt.param_groups for p in g['params']]
    assert len(owned)==len(parameters) and {id(p) for p in owned}=={id(p) for p in parameters}
    assert len(opt.param_groups)==(2 if arm in BOUNDARY_ARMS else 1)
    for g in opt.param_groups:assert (g['lr'],tuple(g['betas']),g['weight_decay'])==(1e-5,(.9,.95),0.)
    assert all(int(opt.state[p]['step'])==5000+step for p in opt.param_groups[0]['params'])
    if arm in BOUNDARY_ARMS:assert int(opt.state[model.e_boundary]['step'])==step
    return parameters


def call_sync(budget,kind,fn,context):
    torch.cuda.synchronize();tick=time.perf_counter()
    def completed():
        value=fn();torch.cuda.synchronize();return value
    return budget.call(kind,completed,context=context),time.perf_counter()-tick


def sample_loss(model,sample,denominator,arm,lambda_seam,budget,kind,context,*,return_details=False):
    from .method import seam_loss
    b=collate([sample],next(model.parameters()).device)
    with native_execution(model,coordinates=(b['xt'],b['t']),autocast=True):
        v,forward_seconds=call_sync(budget,kind,lambda:model(*model_inputs(b)).float(),context)
        fm=masked_fm_loss(v,b['u'],b['valid_mask'],b['known_mask'],denominator=denominator)
        seam,diagnostic=seam_loss(v,b['xt'],b['t'],sample)
        gate=float(float(sample['t'])>=.5)
        auxiliary=seam*(lambda_seam*gate/8) if arm in SEAM_ARMS else seam*0.
        total=fm+auxiliary if arm in SEAM_ARMS else fm
    values=(total,fm,seam,auxiliary,diagnostic,forward_seconds)
    return values+(dict(velocity=v,batch=b),) if return_details else values


def update(model,opt,samples,budget,arm,step,lambda_seam):
    params=validate_optimizer(model,opt,arm,step-1)
    assert len(samples)==8 and len({s['object_id'] for s in samples})==8
    denominator=9*sum(int((s['valid_mask']&~s['known_mask']).sum()) for s in samples)
    opt.zero_grad(set_to_none=True);started=time.perf_counter()
    result=dict(FM=0.,seam_unweighted_mean8=0.,seam_gated_mean8=0.,seam_weighted=0.,loss=0.,
                forward_seconds=0.,backward_seconds=0.,t_gate_fraction=0.,valid_seam_samples=0,
                free_coordinate_denominator=denominator,sample_diagnostics=[])
    representatives={name:p for name,p in model.named_parameters() if name in ('role_embedding','e_boundary')}
    first_backbone=next((name,p) for name,p in model.named_parameters() if name.startswith('backbone.'))
    representatives[first_backbone[0]]=first_backbone[1]
    geo=next(((name,p) for name,p in model.named_parameters() if name.startswith('context_encoder.')),None)
    if geo:representatives[geo[0]]=geo[1]
    before={n:p.detach().reshape(-1)[:min(256,p.numel())].clone() for n,p in representatives.items()}
    boundary_before=model.e_boundary.detach().clone() if arm in BOUNDARY_ARMS else None
    for i,s in enumerate(samples):
        context=dict(arm=arm,attachment_step=step,conditional_total_step=5000+step,microbatch=i,
                     sample_index=s['sample_index'],shared_input_sha256=s['shared_input_sha256'])
        loss,fm,seam,aux,diag,seconds=sample_loss(model,s,denominator,arm,lambda_seam,budget,'training_forward',context)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite formal loss')
        with native_execution(model,autocast=True):
            _,bs=call_sync(budget,'training_backward',loss.backward,context)
        gate=float(float(s['t'])>=.5)
        result['valid_seam_samples']+=int(diag['counts']['eligible_interface_edges']>0)
        result['FM']+=float(fm.detach());result['loss']+=float(loss.detach())
        result['seam_unweighted_mean8']+=float(seam.detach())/8
        result['seam_gated_mean8']+=float(seam.detach())*gate/8
        result['seam_weighted']+=float(aux.detach());result['t_gate_fraction']+=gate/8
        result['forward_seconds']+=seconds;result['backward_seconds']+=bs
        result['sample_diagnostics'].append(dict(sample_index=s['sample_index'],task_id=s['task_id'],
            t=float(s['t']),gate=gate,seam=diag,FM_contribution=float(fm.detach()),
            unweighted_E=float(seam.detach()),weighted_contribution=float(aux.detach())))
        if arm in BOUNDARY_ARMS:
            result['sample_diagnostics'][-1]['boundary']=getattr(model,'boundary_observations',None)
        del loss,fm,seam,aux
    if arm in BOUNDARY_ARMS:
        if model.e_boundary.grad is None:raise RuntimeError('Boundary parameter missing gradient')
        result['boundary_grad_norm_before_clip']=float(model.e_boundary.grad.norm())
    tick=time.perf_counter();result['grad_norm_before_clip']=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
    torch.cuda.synchronize();result['clip_seconds']=time.perf_counter()-tick
    _,result['optimizer_seconds']=call_sync(budget,'optimizer_update',opt.step,
        dict(arm=arm,attachment_step=step,conditional_total_step=5000+step))
    if not all(bool(torch.isfinite(p).all()) for p in params):raise FloatingPointError('Nonfinite updated parameter')
    validate_optimizer(model,opt,arm,step)
    result['representative_parameter_updates']={n:dict(sampled_coordinates=before[n].numel(),
        update_L2=float((p.detach().reshape(-1)[:before[n].numel()]-before[n]).norm()),
        parameter_L2=float(p.detach().norm())) for n,p in representatives.items()}
    if arm in BOUNDARY_ARMS:
        result.update(boundary_parameter_norm=float(model.e_boundary.detach().norm()),
                      boundary_update_norm=float((model.e_boundary.detach()-boundary_before).norm()))
    result['update_seconds']=time.perf_counter()-started
    return result


def train_arm(out,arm,protocol,budget,model,opt,meta,stream_state):
    from .stream import AttachmentStream
    out=Path(out);folder=out/'training';folder.mkdir(parents=True,exist_ok=True)
    step=int(meta['attachment_step']);stream=AttachmentStream(SOURCE,CHAIR,out,arm)
    if stream_state is not None:stream.load_state_dict(stream_state)
    assert stream.batch_index==step
    events=folder/f'{arm}_events.jsonl';status=folder/f'{arm}_status.json'
    prior=ledger_rows(events)
    done={r['attachment_step'] for r in prior if r.get('event')=='UPDATE_DONE'}
    attempted={r['attachment_step'] for r in prior if r.get('event')=='UPDATE_BEGIN'}
    if done!=set(range(1,step+1)) or attempted!=done:raise RuntimeError('Checkpoint cursor does not cover all attempted/completed updates')
    shared_path=folder/'shared_batch_hashes.jsonl'
    shared={r['attachment_step']:r for r in ledger_rows(shared_path)}
    stop={'signal':None}
    def request_stop(signum,frame):stop['signal']=int(signum)
    handlers={s:signal.signal(s,request_stop) for s in (signal.SIGINT,signal.SIGTERM)}
    state=dict(status='TRAINING',arm=arm,attachment_step=step,conditional_total_step=5000+step,
               protocol_sha256=json_hash(protocol),actual_updates_this_worker=0)
    started=time.perf_counter();start_step=step;in_update=False;saved_step=step
    fields=['arm','attachment_step','conditional_total_step','shared_batch_sha256','coupled_batch_sha256',
            'FM','loss','seam_unweighted_mean8','seam_gated_mean8','seam_weighted','t_gate_fraction',
            'free_coordinate_denominator','grad_norm_before_clip','boundary_parameter_norm',
            'boundary_grad_norm_before_clip','boundary_update_norm','OT_and_input_seconds','forward_seconds',
            'backward_seconds','clip_seconds','optimizer_seconds','update_seconds','peak_allocated_bytes','peak_reserved_bytes']
    exposure_fields=['arm','attachment_step','sample_index','uid','task_id','alpha','N','K','t',
        'free_coordinate_count','shared_input_sha256','coupled_input_sha256','ot_cache_key','ot_cache_status']
    def persist():
        nonlocal saved_step
        record=checkpoint.save(out,arm,model,opt,stream,protocol,step);saved_step=step
        append_event(events,dict(event='CHECKPOINT_DONE',attachment_step=step,checkpoint=record))
        state['checkpoint']=record;return record
    try:
        validate_optimizer(model,opt,arm,step);torch.cuda.reset_peak_memory_stats()
        with (folder/f'{arm}_training_log.csv').open('a',newline='',encoding='utf-8') as f,(folder/f'{arm}_sample_exposure.csv').open('a',newline='',encoding='utf-8') as sf:
            writer=csv.DictWriter(f,fieldnames=fields);ew=csv.DictWriter(sf,fieldnames=exposure_fields)
            if f.tell()==0:writer.writeheader()
            if sf.tell()==0:ew.writeheader()
            for current in range(step+1,1001):
                if stop['signal'] is not None:
                    if step>saved_step:persist()
                    state.update(status='INTERRUPTED_SAFE',signal=stop['signal']);break
                committed_stream_state=stream.state_dict()
                tick=time.perf_counter();samples=stream.next_effective_batch();input_seconds=time.perf_counter()-tick
                shared_hash=stream.last_audit['shared_batch_sha256'];coupled_hash=stream.last_audit['coupled_batch_sha256']
                if current in shared:
                    assert shared[current]['shared_batch_sha256']==shared_hash and shared[current]['coupled_batch_sha256']==coupled_hash
                else:
                    sh=dict(attachment_step=current,shared_batch_sha256=shared_hash,coupled_batch_sha256=coupled_hash,first_arm=arm)
                    append_event(shared_path,sh);shared[current]=sh
                for s in samples:
                    ew.writerow(dict(arm=arm,attachment_step=current,sample_index=s['sample_index'],uid=s['object_id'],
                        task_id=s['task_id'],alpha=s['alpha'],N=int(s['y']),K=s['K'],t=float(s['t']),
                        **{k:s[k] for k in exposure_fields if k in s and k not in ('arm','attachment_step','sample_index','uid','task_id','alpha','N','K','t')}))
                sf.flush();os.fsync(sf.fileno())
                append_event(events,dict(event='UPDATE_BEGIN',attachment_step=current,shared_batch_sha256=shared_hash))
                in_update=True
                result=update(model,opt,samples,budget,arm,current,protocol['lambda_seam'])
                step=current;in_update=False
                row=dict(arm=arm,attachment_step=step,conditional_total_step=5000+step,
                         shared_batch_sha256=shared_hash,coupled_batch_sha256=coupled_hash,
                         OT_and_input_seconds=input_seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         peak_reserved_bytes=torch.cuda.max_memory_reserved(),**{k:result[k] for k in fields if k in result})
                append_event(events,dict(event='UPDATE_DONE',**row,sample_diagnostics=result['sample_diagnostics'],
                             representative_parameter_updates=result['representative_parameter_updates']))
                writer.writerow(row);f.flush();os.fsync(f.fileno())
                state.update(attachment_step=step,conditional_total_step=5000+step,actual_updates_this_worker=step-start_step,last_row=row)
                atomic_json(status,state)
                if step==1:resource_snapshot(out,arm+'_first_formal_update_peak')
                if step%200==0:persist();resource_snapshot(out,arm+f'_step{step}')
                if step%25==0:print(dict(event='TRAIN_PROGRESS',arm=arm,attachment_step=step,FM=row['FM'],aux=row['seam_weighted'],seconds=time.perf_counter()-started),flush=True)
            else:state['status']='DONE'
        state.update(wall_seconds=time.perf_counter()-started,OT_counts=dict(stream.counts))
        atomic_json(status,state);budget.snapshot();return state
    except BaseException as exc:
        # An engineering failure after a completed update can preserve its exact state.
        if not in_update and stream.batch_index!=step:
            stream.load_state_dict(committed_stream_state)
        if not in_update and step>saved_step:
            try:persist()
            except BaseException as save_exc:state['emergency_save_error']=repr(save_exc)
        state.update(status='BLOCKED',attachment_step=step,error=repr(exc),in_update=in_update,traceback=traceback.format_exc())
        atomic_json(status,state);budget.snapshot();raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
