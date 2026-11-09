"""Same explicit separated-gradient carrier for all three authorized arms."""
import time
import torch
from native_t1.model import native_execution
from native_t1.data import model_inputs
from native_t1.training import masked_fm_loss
from experiments.mf_h1_local0.control_mvp48.train_stream import collate
from experiments.mf_h1_local0.chair_hybrid_coupling_v1.training import _validate_optimizer,_timed_cuda,validate_recovery
from experiments.mf_h1_local0.chair_control_feedback_auto_v1.replay import aux_sample,collate_aux
from .common import *
from .stream import HybridStream
from .gradients import GradientCarrier,ParameterSnapshot,should_snapshot
from . import checkpoint


def update(model,opt,samples,budget,arm,step):
    params=_validate_optimizer(model,opt);device=params[0].device
    assert len(samples)==8 and len({s['object_id'] for s in samples})==8
    denominator=9*sum(int((s['valid_mask'] & ~s['known_mask']).sum()) for s in samples)
    opt.zero_grad(set_to_none=True);tick=time.perf_counter()
    r=dict(FM=0.,free_coordinate_denominator=denominator,forward_seconds=0.,backward_seconds=0.,sample_diagnostics=[])
    carrier=GradientCarrier(model.named_parameters())
    for micro,sample in enumerate(samples):
        batch=collate([sample],device)
        context=dict(arm=arm,pilot_step=step,conditional_total_step=5000+step,microbatch=micro,
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
                FM_contribution=value,shared_input_sha256=sample['shared_input_sha256'],ot_cache_status=sample['ot_cache_status']))
        del velocity,loss,batch
    carrier.capture_primary(clear=True)
    sample=aux_sample(step,'REPLAY_FM_CONTROL' if arm=='T' else 'TRAJECTORY_REPLAY',out=OLD,training_seed=AUX_SEED)
    batch=collate_aux(sample,device);context=dict(arm=arm,pilot_step=step,microbatch='aux',request_id=sample['aux_audit']['request_id'])
    with native_execution(model,coordinates=(batch['xt'],batch['t']),autocast=True):
        velocity,seconds=_timed_cuda(budget,'training_forward',lambda:model(*model_inputs(batch)).float(),context)
        r['forward_seconds']+=seconds
        auxden=9*int((batch['valid_mask'] & ~batch['known_mask']).sum())
        raw=masked_fm_loss(velocity,batch['u'],batch['valid_mask'],batch['known_mask'],denominator=auxden)
        if not bool(torch.isfinite(raw)):raise FloatingPointError('Nonfinite auxiliary loss')
        r['aux_unweighted']=float(raw.detach())
        _,seconds=_timed_cuda(budget,'training_backward',raw.backward,context);r['backward_seconds']+=seconds
    r.update(carrier.combine(primary_loss=r['FM'],auxiliary_loss=r['aux_unweighted'],arm=arm))
    r.update(aux_audit=sample['aux_audit'],aux_t=float(sample['t']),aux_free_denominator=auxden)
    del batch,velocity,raw,carrier
    r['grad_norm_before_clip']=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
    r['clip_coefficient']=min(1.,1./(r['grad_norm_before_clip']+1e-6))
    snap=ParameterSnapshot(model.named_parameters()) if should_snapshot(step) else None
    _,r['optimizer_seconds']=_timed_cuda(budget,'optimizer_update',opt.step,dict(arm=arm,pilot_step=step))
    if snap is not None:r['actual_parameter_update']=snap.measure_and_release()
    if not all(bool(torch.isfinite(p).all()) for p in params):raise FloatingPointError('Nonfinite updated parameters')
    if any(int(s['step'])!=5000+step for s in opt.state.values()):raise ValueError('AdamW cursor mismatch')
    r.update(update_total_seconds=time.perf_counter()-tick,runtime=facts,finite=True,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        peak_scope='reset before update; includes resident parameters, restored optimizer, primary FP32 buffer, backward and geometry workspaces')
    return r


def train(out,arm,until,model,opt,meta,stream_state,protocol,budget):
    assert arm in ARMS and until in (500,1000)
    step=int(meta['feedback_step']);log=out/'training'/f'{arm}_events.jsonl'
    validate_recovery(ledger_rows(log),step)
    seed=protocol['training']['seed']
    stream=HybridStream(SOURCE,CHAIR,out/'streams/repeat',arm,seed=seed)
    if stream_state is not None:
        assert meta['arm']==arm and meta['protocol_sha256']==json_hash(protocol)
        stream.load_state_dict(stream_state)
    assert stream.batch_index==step
    while step<until:
        if time.time()+90>=budget.gpu_deadline:
            checkpoint.save(out,'repeat',arm,model,opt,stream,protocol,step)
            raise RuntimeError('GPU occupancy reserve reached: saved complete update and stopped')
        step+=1;append_event(log,dict(event='INPUT_BEGIN',pilot_step=step));tick=time.perf_counter()
        samples=stream.next_effective_batch();input_seconds=time.perf_counter()-tick
        append_event(log,dict(event='INPUT_DONE',pilot_step=step,**stream.last_audit))
        append_event(log,dict(event='UPDATE_BEGIN',pilot_step=step));torch.cuda.reset_peak_memory_stats()
        result=update(model,opt,samples,budget,arm,step)
        result.update(arm=arm,pilot_step=step,conditional_total_step=5000+step,OT_and_input_seconds=input_seconds,**stream.last_audit)
        append_event(out/'training'/f'{arm}_metrics.jsonl',result)
        append_event(log,dict(event='UPDATE_DONE',pilot_step=step))
        if step==1 or step%25==0:
            print('TRAIN',arm,step,'FM',round(result['FM'],6),'grad',round(result['grad_norm_before_clip'],4),
                  'sec',round(input_seconds+result['update_total_seconds'],3),flush=True);budget.snapshot()
        if step%100==0:
            saved=checkpoint.save(out,'repeat',arm,model,opt,stream,protocol,step)
            print('CHECKPOINT',saved['path'],round(saved['save_verify_seconds'],2),flush=True)
    atomic_json(out/'training'/f'{arm}_{until}_complete.json',dict(status='PASS',steps=step,seed=seed,stream_state=stream.state_dict(),arm=arm))
