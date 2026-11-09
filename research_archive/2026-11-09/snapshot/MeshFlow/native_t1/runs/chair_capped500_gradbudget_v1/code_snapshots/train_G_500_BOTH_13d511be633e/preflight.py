"""Twelve real F/B calls on one frozen batch; no retained optimizer update."""
import gc
import torch
from native_t1.model import native_execution
from native_t1.data import model_inputs
from native_t1.training import masked_fm_loss,make_optimizer
from experiments.mf_h1_local0.control_mvp48.train_stream import collate
from experiments.mf_h1_local0.control_mvp48.train_io import tree_cpu,tree_hash
from experiments.mf_h1_local0.chair_domain_control_v1.checkpoint import tensor_state_hash
from experiments.mf_h1_local0.chair_hybrid_coupling_v1.training import _timed_cuda
from experiments.mf_h1_local0.chair_control_feedback_auto_v1.replay import aux_sample,collate_aux
from experiments.mf_h1_local0.chair_control_feedback_auto_v1.loss_balance import coefficient
from .common import *
from .stream import HybridStream
from .gradients import GradientCarrier,ParameterSnapshot

TOLERANCES=dict(combined_gradient_relative_L2_max=.01,combined_gradient_cosine_min=.9999,
    auxiliary_gradient_relative_L2_max=.05,normalization_absolute_max=1e-8)


def run(out,model,opt,protocol,budget):
    assert protocol['real_preflight_tolerances']==TOLERANCES
    params=list(model.parameters());device=params[0].device
    state_before=tensor_state_hash(model.state_dict());optim_before=tree_hash(opt.state_dict())
    stream=HybridStream(SOURCE,CHAIR,out/'streams/repeat','T',seed=protocol['training']['seed'])
    samples=stream.next_effective_batch();den=9*sum(int((s['valid_mask'] & ~s['known_mask']).sum()) for s in samples)
    opt.zero_grad(set_to_none=True);lf=0.;normalization=[]
    for micro,sample in enumerate(samples):
        batch=collate([sample],device);ctx=dict(stage='precheck_primary',microbatch=micro)
        assert torch.equal(batch['xt'][batch['known_mask']],torch.as_tensor(sample['context'][sample['known_mask']],device=device))
        with native_execution(model,coordinates=(batch['xt'],batch['t']),autocast=True):
            velocity,_=_timed_cuda(budget,'precheck_forward',lambda:model(*model_inputs(batch)).float(),ctx)
            loss=masked_fm_loss(velocity,batch['u'],batch['valid_mask'],batch['known_mask'],denominator=den)
            free=batch['valid_mask'] & ~batch['known_mask']
            manual=((velocity-batch['u'])[free]**2).sum()/den
            residual=abs(float(loss.detach())-float(manual.detach()));normalization.append(residual)
            assert residual<=TOLERANCES['normalization_absolute_max']
            lf+=float(loss.detach());_timed_cuda(budget,'precheck_backward',loss.backward,ctx)
        del batch,velocity,loss,manual
    gF=[None if p.grad is None else p.grad.detach().cpu().clone() for p in params];comparisons=[]
    for arm in ('T','L'):
        sample=aux_sample(1,'REPLAY_FM_CONTROL' if arm=='T' else 'TRAJECTORY_REPLAY',out=OLD,training_seed=AUX_SEED)
        batch=collate_aux(sample,device);auxden=9*int((batch['valid_mask'] & ~batch['known_mask']).sum())
        for p,g in zip(params,gF):p.grad=None if g is None else g.to(device)
        ctx=dict(stage='precheck_'+arm,carrier='old_weighted_backward')
        with native_execution(model,coordinates=(batch['xt'],batch['t']),autocast=True):
            velocity,_=_timed_cuda(budget,'precheck_forward',lambda:model(*model_inputs(batch)).float(),ctx)
            old_prediction=velocity.detach().cpu()
            raw=masked_fm_loss(velocity,batch['u'],batch['valid_mask'],batch['known_mask'],denominator=auxden)
            raw_value=float(raw.detach());weight,_=coefficient(lf,raw_value);weighted=weight*raw
            _timed_cuda(budget,'precheck_backward',weighted.backward,ctx)
        old_combined=[None if p.grad is None else p.grad.detach().cpu().clone() for p in params]
        del raw,weighted,velocity
        for p,g in zip(params,gF):p.grad=None if g is None else g.to(device)
        carrier=GradientCarrier(model.named_parameters());carrier.capture_primary(clear=True)
        ctx=dict(stage='precheck_'+arm,carrier='separated_raw_backward')
        with native_execution(model,coordinates=(batch['xt'],batch['t']),autocast=True):
            velocity,_=_timed_cuda(budget,'precheck_forward',lambda:model(*model_inputs(batch)).float(),ctx)
            assert torch.equal(old_prediction,velocity.detach().cpu())
            raw=masked_fm_loss(velocity,batch['u'],batch['valid_mask'],batch['known_mask'],denominator=auxden)
            assert raw_value==float(raw.detach())
            _timed_cuda(budget,'precheck_backward',raw.backward,ctx)
        stats=carrier.combine(primary_loss=lf,auxiliary_loss=raw_value,arm=arm)
        diff=old2=new2=dot=aux2=0.;maxabs=0.
        for p,ref,primary in zip(params,old_combined,gF):
            if ref is None and p.grad is None:continue
            a=ref.double() if ref is not None else torch.zeros_like(p.detach(),device='cpu',dtype=torch.float64)
            b=p.grad.detach().cpu().double() if p.grad is not None else torch.zeros_like(a)
            f=primary.double() if primary is not None else torch.zeros_like(a)
            d=b-a;diff+=float((d*d).sum());old2+=float((a*a).sum());new2+=float((b*b).sum());dot+=float((a*b).sum())
            aux2+=float(((a-f)**2).sum());maxabs=max(maxabs,float(d.abs().max()))
        rel=(diff/max(old2,1e-60))**.5;auxrel=(diff/max(aux2,1e-60))**.5;cos=dot/max((old2*new2)**.5,1e-30)
        comparisons.append(dict(arm=arm,primary_FM=lf,raw_aux=raw_value,lambda_loss=weight,
            forward_bitwise_equal=True,combined_relative_L2=rel,auxiliary_relative_L2=auxrel,
            combined_cosine=cos,max_absolute_gradient_difference=maxabs,carrier_stats=stats,
            rounding_source='Old coefficient enters BF16 backward; separated carrier scales raw gradient in FP32. Same math, not bitwise derivative arithmetic.'))
        atomic_json(out/'preflight/real_progress.json',dict(comparisons=comparisons,tolerances=TOLERANCES))
        assert rel<=TOLERANCES['combined_gradient_relative_L2_max']
        assert auxrel<=TOLERANCES['auxiliary_gradient_relative_L2_max']
        assert cos>=TOLERANCES['combined_gradient_cosine_min']
        del batch,raw,velocity,carrier,old_combined,old_prediction
    originals=[p.detach().cpu().clone() for p in params]
    isolated=make_optimizer(model);isolated.load_state_dict(tree_cpu(opt.state_dict()))
    before=ParameterSnapshot(model.named_parameters())
    clip=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
    budget.call('precheck_optimizer_update',isolated.step,context=dict(stage='isolated_optimizer_copy',retained=False))
    measured=before.measure_and_release();assert all(int(v['step'])==5001 for v in isolated.state.values())
    with torch.no_grad():
        for p,original in zip(params,originals):p.copy_(original.to(device))
    opt.zero_grad(set_to_none=True)
    assert tensor_state_hash(model.state_dict())==state_before
    assert tree_hash(opt.state_dict())==optim_before
    del isolated,originals,gF;gc.collect();torch.cuda.empty_cache()
    result=dict(status='PASS',primary_microbatches=8,forward_calls=12,backward_calls=12,
        retained_optimizer_updates=0,isolated_discarded_optimizer_updates=1,comparisons=comparisons,
        normalization_residuals=normalization,original_model_restored=True,original_AdamW_restored=True,
        actual_isolated_parameter_update=measured,clip_before_copy_step=clip,tolerances=TOLERANCES)
    freeze_json(out/'preflight/real_complete.json',result);return result
