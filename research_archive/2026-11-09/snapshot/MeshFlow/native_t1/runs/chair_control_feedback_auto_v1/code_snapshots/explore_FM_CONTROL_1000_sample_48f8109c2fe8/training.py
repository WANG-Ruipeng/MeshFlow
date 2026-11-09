"""Frozen main HYBRID-FM objective plus registered optional feedback replay."""
import time
import numpy as np
import torch
from native_t1.model import native_execution
from native_t1.data import model_inputs
from native_t1.training import masked_fm_loss
from experiments.mf_h1_local0.control_mvp48.train_stream import collate
from experiments.mf_h1_local0.chair_hybrid_coupling_v1.training import _validate_optimizer,_timed_cuda,validate_recovery
from .common import *
from .stream import HybridStream
from . import checkpoint

def global_grad_norm(params):
    return float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.detach()) for p in params if p.grad is not None])))

def update(model,opt,samples,budget,phase,arm,step,out,seed):
    params=_validate_optimizer(model,opt);device=params[0].device
    assert len(samples)==8 and len({s['object_id'] for s in samples})==8
    denominator=9*sum(int((s['valid_mask'] & ~s['known_mask']).sum()) for s in samples)
    opt.zero_grad(set_to_none=True);tick=time.perf_counter()
    r=dict(FM=0.,aux_unweighted=0.,aux_weighted=0.,loss=0.,free_coordinate_denominator=denominator,
        forward_seconds=0.,backward_seconds=0.,sample_diagnostics=[])
    for micro,sample in enumerate(samples):
        batch=collate([sample],device)
        context=dict(phase=phase,arm=arm,pilot_step=step,conditional_total_step=5000+step,microbatch=micro,
            sample_index=sample['sample_index'],shared_input_sha256=sample['shared_input_sha256'])
        with native_execution(model,coordinates=(batch['xt'],batch['t']),autocast=True) as facts:
            velocity,seconds=_timed_cuda(budget,'training_forward',lambda:model(*model_inputs(batch)).float(),context)
            r['forward_seconds']+=seconds
            loss=masked_fm_loss(velocity,batch['u'],batch['valid_mask'],batch['known_mask'],denominator=denominator)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite primary FM')
            value=float(loss.detach());r['FM']+=value
            _,seconds=_timed_cuda(budget,'training_backward',loss.backward,context);r['backward_seconds']+=seconds
            r['sample_diagnostics'].append(dict(sample_index=sample['sample_index'],task_id=sample['task_id'],
                uid=sample['object_id'],N=len(sample['xt']),K=sample['K'],alpha=sample['alpha'],t=float(sample['t']),
                free_coordinates=sample['free_coordinate_count'],FM_contribution=value,
                shared_input_sha256=sample['shared_input_sha256'],ot_cache_status=sample['ot_cache_status']))
        del velocity,loss,batch
    r['FM_gradient_norm']=global_grad_norm(params)
    if arm!='FM_CONTROL':
        from .replay import aux_sample,collate_aux
        aux_arm='TRAJECTORY_REPLAY' if arm=='TRAJECTORY_REPLAY_CAPPED' else arm
        sample=aux_sample(step,aux_arm,out=out,training_seed=seed);batch=collate_aux(sample,device)
        context=dict(phase=phase,arm=arm,pilot_step=step,microbatch='aux',request_id=sample.get('request_id',sample.get('task_id')))
        with native_execution(model,coordinates=(batch['xt'],batch['t']),autocast=True):
            velocity,seconds=_timed_cuda(budget,'training_forward',lambda:model(*model_inputs(batch)).float(),context)
            r['forward_seconds']+=seconds
            auxden=9*int((batch['valid_mask'] & ~batch['known_mask']).sum())
            raw=masked_fm_loss(velocity,batch['u'],batch['valid_mask'],batch['known_mask'],denominator=auxden)
            if arm=='TRAJECTORY_REPLAY_CAPPED':
                from .loss_balance import coefficient
                weight,balance=coefficient(r['FM'],float(raw.detach()))
                loss=.1*raw if weight==.1 else weight*raw
                r['loss_balance']=balance
            else:
                loss=.1*raw
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite feedback auxiliary loss')
            r['aux_unweighted']=float(raw.detach());r['aux_weighted']=float(loss.detach())
            handles=[];aux_squares=[];fm_dots=[]
            if step==1 or step%25==0:
                def observed(parameter):
                    def hook(gradient):
                        g=gradient.detach()
                        aux_squares.append(torch.sum(g*g))
                        if parameter.grad is not None:fm_dots.append(torch.sum(g*parameter.grad.detach()))
                    return hook
                handles=[p.register_hook(observed(p)) for p in params]
            try:
                _,seconds=_timed_cuda(budget,'training_backward',loss.backward,context);r['backward_seconds']+=seconds
            finally:
                for h in handles:h.remove()
            if aux_squares:
                r['weighted_aux_gradient_norm']=float(torch.stack(aux_squares).sum().sqrt())
                dot=float(torch.stack(fm_dots).sum()) if fm_dots else 0.
                r['FM_aux_gradient_cosine']=dot/max(1e-30,r['FM_gradient_norm']*r['weighted_aux_gradient_norm'])
                r['gradient_measurement']='Actual extra backward leaf hooks before accumulation; no extra forward/backward or gradient modification'
                if r['weighted_aux_gradient_norm']==0:raise RuntimeError('Registered feedback gradient is identically zero')
            r['aux_audit']=sample.get('aux_audit',{})
            r['aux_t']=float(sample['t']);r['aux_free_denominator']=auxden
        del batch,velocity,raw,loss
    r['loss']=r['FM']+r['aux_weighted']
    r['grad_norm_before_clip']=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
    _,r['optimizer_seconds']=_timed_cuda(budget,'optimizer_update',opt.step,dict(phase=phase,arm=arm,pilot_step=step))
    if not all(bool(torch.isfinite(p).all()) for p in params):raise FloatingPointError('Nonfinite updated parameters; no retry')
    if any(int(s['step'])!=5000+step for s in opt.state.values()):raise ValueError('AdamW cursor mismatch')
    r.update(update_total_seconds=time.perf_counter()-tick,runtime=facts,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        peak_scope='reset before each training update; resident model, optimizer, backward and primary/aux workspaces')
    return r

def train(out,phase,arm,until,model,opt,meta,stream_state,protocol,budget):
    assert arm in ARMS and phase in ('explore','repeat') and until in (500,1000)
    seed=20261008 if phase=='explore' else 20261009
    label=f'{phase}_{arm}';step=int(meta['feedback_step']);log=out/'training'/f'{label}_events.jsonl'
    validate_recovery(ledger_rows(log),step)
    recipe_sha=None
    if arm=='TRAJECTORY_REPLAY_CAPPED':
        from .loss_balance import frozen_recipe
        recipe,recipe_sha=frozen_recipe(out)
        if step:assert meta['recipe_addendum_file_sha256']==recipe_sha
    stream_arm='TRAJECTORY_REPLAY' if arm=='TRAJECTORY_REPLAY_CAPPED' else arm
    stream=HybridStream(SOURCE,CHAIR,out/'streams'/phase,stream_arm,seed=seed)
    if stream_state is not None:
        assert meta['arm']==arm and meta['phase']==phase
        assert meta['protocol_sha256']==json_hash(protocol)
        if arm!='FM_CONTROL':assert meta['feedback_pool_sha256']==digest(out/'feedback/pool_v1/manifest.json')
        stream.load_state_dict(stream_state)
    assert stream.batch_index==step
    if arm!='FM_CONTROL':
        from .replay import prepare_feedback
        prepared=prepare_feedback(out)
        assert prepared['status']=='PASS',prepared
    append_event(out/'decisions/training_starts.jsonl',dict(event='TRAIN_STAGE_START',phase=phase,arm=arm,
        start_step=step,until=until,source_checkpoint_sha256=START_SHA,protocol_sha256=json_hash(protocol),recipe_addendum_sha256=recipe_sha))
    while step<until:
        if time.time()>__import__('datetime').datetime.fromisoformat(SOFT_DEADLINE_UTC.replace('Z','+00:00')).timestamp()-180:
            checkpoint.save(out,phase,arm,model,opt,stream,protocol,step)
            raise RuntimeError('Wall window reached; complete-update checkpoint saved, no next update dispatched')
        step+=1
        append_event(log,dict(event='INPUT_BEGIN',pilot_step=step));tick=time.perf_counter()
        samples=stream.next_effective_batch();input_seconds=time.perf_counter()-tick
        append_event(log,dict(event='INPUT_DONE',pilot_step=step,**stream.last_audit))
        append_event(log,dict(event='UPDATE_BEGIN',pilot_step=step))
        torch.cuda.reset_peak_memory_stats()
        result=update(model,opt,samples,budget,phase,arm,step,out,seed)
        result.update(arm=arm,phase=phase,pilot_step=step,conditional_total_step=5000+step,
            OT_and_input_seconds=input_seconds,**stream.last_audit)
        append_event(out/'training'/f'{label}_metrics.jsonl',result)
        append_event(log,dict(event='UPDATE_DONE',pilot_step=step))
        if step==1 or step%25==0:
            print('TRAIN',label,step,'FM',round(result['FM'],6),'AUX',round(result['aux_weighted'],6),
                'grad',round(result['grad_norm_before_clip'],4),'sec',round(input_seconds+result['update_total_seconds'],3),flush=True)
            budget.snapshot()
        if step%100==0:
            saved=checkpoint.save(out,phase,arm,model,opt,stream,protocol,step)
            print('CHECKPOINT',saved['path'],round(saved['save_verify_seconds'],2),flush=True)
            if step in (500,1000):
                from .worker import sample_manifest
                rng=checkpoint.capture_rng()
                sample_manifest(out,f'{label}_{step}',out/'manifests/DEV.json',model,budget,saved['checkpoint_sha256'])
                checkpoint.restore_rng(rng);model.set_trainable(True).eval();opt.zero_grad(set_to_none=True)
                print('DEV_READY',f'{label}_{step}',flush=True)
    atomic_json(out/'training'/f'{label}_complete.json',dict(status='PASS',steps=step,seed=seed,
        stream_state=stream.state_dict(),phase=phase,arm=arm))
