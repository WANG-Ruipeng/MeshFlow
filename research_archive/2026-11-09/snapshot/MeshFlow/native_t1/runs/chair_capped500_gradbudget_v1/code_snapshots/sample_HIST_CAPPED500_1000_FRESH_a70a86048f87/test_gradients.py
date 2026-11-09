"""Finite CPU-only tests of the common gradient carrier; no real model calls."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from native_t1.training import masked_fm_loss
from .gradients import GradientCarrier, ParameterSnapshot, coefficients, should_snapshot

COUNTS = dict(toy_forward=0, toy_backward=0, toy_optimizer_updates=0)
ATOL, RTOL = 1e-7, 1e-5


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(4, 7)
        self.b = torch.nn.Linear(7, 9)
        self.primary_only = torch.nn.Parameter(torch.linspace(-.1,.1,9))
        self.auxiliary_only = torch.nn.Parameter(torch.linspace(.03,-.03,9))
        self.unused = torch.nn.Parameter(torch.ones(3))

    def forward(self, x, auxiliary=False):
        COUNTS['toy_forward'] += 1
        return self.b(self.a(x).tanh()) + (self.auxiliary_only if auxiliary else self.primary_only)


def backward(loss):
    COUNTS['toy_backward'] += 1
    loss.backward()


def step(opt):
    COUNTS['toy_optimizer_updates'] += 1
    opt.step()


def optimizer(model):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(.9,.95), weight_decay=0)
    for p in model.parameters():
        opt.state[p] = dict(step=torch.tensor(5000.), exp_avg=torch.full_like(p,.001), exp_avg_sq=torch.full_like(p,.002))
    return opt


def inputs(index=0):
    x = torch.sin(torch.arange(24,dtype=torch.float32).reshape(1,6,4)*.13+index*.07)
    y = torch.cos(torch.arange(54,dtype=torch.float32).reshape(1,6,9)*.11+index*.05)
    known = torch.tensor([[True,False,True,False,False,True]])
    valid = torch.tensor([[True,True,True,True,False,False]])
    return x,y,valid,known


def objective(model, *, separate, arm='L', auxiliary_scale=1., auxiliary_offset=0., bf16=False):
    carrier = GradientCarrier(model.named_parameters()) if separate else None
    fm = 0.
    for i in range(8):
        x,y,valid,known=inputs(i)
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
            output=model(x).float()
            loss=masked_fm_loss(output,y,valid,known,denominator=144)
        fm += float(loss.detach())
        backward(loss)
    if separate:
        carrier.capture_primary()
    x,y,valid,known=inputs(9)
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
        output=model(x,auxiliary=True).float()
        raw=auxiliary_scale*masked_fm_loss(output,y+auxiliary_offset,valid,known,denominator=18)
    raw_float=float(raw.detach())
    if separate:
        backward(raw)
        stats=carrier.combine(primary_loss=fm,auxiliary_loss=raw_float,arm=arm)
    else:
        weight=min(.1,fm/max(raw_float,1e-30))
        backward(.1*raw if weight==.1 else weight*raw)
        stats=dict(lambda_loss=weight,actual_lambda=weight,loss_cap_active=weight<.1)
    return fm,raw_float,stats


def full_gradient(model):
    return torch.cat([torch.zeros_like(p).flatten() if p.grad is None else p.grad.detach().flatten() for p in model.parameters()])


def run(out):
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU test must not have initialized CUDA.')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    results=[]
    started=time.time()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(618307)
        template=Tiny()
    def record(name, facts):
        results.append(dict(name=name,status='PASS',**facts))
    for active in (False,True):
        reference,split=copy.deepcopy(template),copy.deepcopy(template)
        offset=20. if active else 0.
        f0,a0,s0=objective(reference,separate=False,auxiliary_offset=offset)
        f1,a1,s1=objective(split,separate=True,auxiliary_offset=offset)
        assert (f0,a0)==(f1,a1) and s1['loss_cap_active']==active
        g0,g1=full_gradient(reference),full_gradient(split)
        assert torch.allclose(g0,g1,atol=ATOL,rtol=RTOL)
        for (n,p),(m,q) in zip(reference.named_parameters(),split.named_parameters()):
            assert n==m and ((p.grad is None)==(q.grad is None))
        o0,o1=optimizer(reference),optimizer(split)
        torch.nn.utils.clip_grad_norm_(reference.parameters(),1,error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_(split.parameters(),1,error_if_nonfinite=True)
        step(o0);step(o1)
        error=max(float((p-q).abs().max()) for p,q in zip(reference.parameters(),split.parameters()))
        assert error<=ATOL
        assert int(o1.state[split.unused]['step'])==5000 and int(o1.state[split.a.weight]['step'])==5001
        record('original_FP32_definition_vs_separation_'+('active' if active else 'inactive'),dict(forward_loss_exact=True,max_gradient_abs_error=float((g0-g1).abs().max()),max_update_parameter_abs_error=error,tolerance=dict(atol=ATOL,rtol=RTOL),both_none_AdamW_skip_preserved=True))
    left,right=copy.deepcopy(template),copy.deepcopy(template)
    _,_,sl=objective(left,separate=True,arm='L',auxiliary_scale=1e-4)
    _,_,sg=objective(right,separate=True,arm='G',auxiliary_scale=1e-4)
    assert not sg['gradient_cap_active'] and sl['actual_lambda']==sg['actual_lambda']
    assert torch.equal(full_gradient(left),full_gradient(right))
    ol,og=optimizer(left),optimizer(right)
    torch.nn.utils.clip_grad_norm_(left.parameters(),1,error_if_nonfinite=True)
    torch.nn.utils.clip_grad_norm_(right.parameters(),1,error_if_nonfinite=True)
    step(ol);step(og)
    assert all(torch.equal(p,q) for p,q in zip(left.parameters(),right.parameters()))
    record('G_nonbinding_bitwise_L_update',dict(exact_gradient=True,exact_parameter_update=True,lambda_grad=sg['lambda_grad'],lambda_loss=sg['lambda_loss']))
    p=torch.nn.Parameter(torch.zeros(2));carrier=GradientCarrier([('p',p)])
    p.grad=torch.tensor([1.,0.]);carrier.capture_primary();p.grad=torch.tensor([30.,40.])
    stats=carrier.combine(primary_loss=100.,auxiliary_loss=1.,arm='G')
    assert stats['gradient_cap_active'] and stats['actual_lambda']==min(.1,1./(50.+1e-12))
    assert stats['weighted_aux_norm']<=stats['gF_norm']+1e-7
    expected = torch.tensor([1.,0.]) + torch.tensor([30.,40.])*stats['actual_lambda']
    assert torch.equal(p.grad,expected)
    record('G_binding_full_gradient_budget',dict(actual_weighted_aux_norm=stats['weighted_aux_norm'],primary_norm=stats['gF_norm'],FP32_budget_excess=stats['actual_FP32_budget_excess'],rho=1.))
    ps=[torch.nn.Parameter(torch.zeros(2)) for _ in range(3)]
    carrier=GradientCarrier(zip(('main','aux','unused'),ps))
    ps[0].grad=torch.tensor([1.,2.]);before=ps[0].grad
    carrier.capture_primary();before.add_(10.)
    assert all(p.grad is None for p in ps)
    ps[1].grad=torch.tensor([3.,4.]);s=carrier.combine(primary_loss=10.,auxiliary_loss=1.,arm='L')
    assert torch.equal(ps[0].grad,torch.tensor([1.,2.])) and torch.allclose(ps[1].grad,torch.tensor([.3,.4])) and ps[2].grad is None
    assert s['both_none_names']==['unused'] and s['auxiliary_none_names']==['main','unused'] and s['primary_none_names']==['aux','unused']
    record('None_disjoint_and_buffer_isolation',dict(names=s['both_none_names'],complete_union=True))
    p=torch.nn.Parameter(torch.zeros(2));c=GradientCarrier([('p',p)])
    p.grad=torch.zeros(2);c.capture_primary();p.grad=torch.tensor([2.,1.]);s=c.combine(primary_loss=1.,auxiliary_loss=1.,arm='G')
    assert s['actual_lambda']==0 and torch.equal(p.grad,torch.zeros(2)) and s['cosine'] is None and s['weighted_aux_over_FM_norm'] is None
    record('G_zero_primary_suppresses_auxiliary',dict(cosine_status=s['cosine_status'],coefficient=s['actual_lambda']))
    p=torch.nn.Parameter(torch.zeros(2));c=GradientCarrier([('p',p)])
    p.grad=torch.tensor([1.,2.]);c.capture_primary();p.grad=torch.zeros(2);s=c.combine(primary_loss=1.,auxiliary_loss=0.,arm='G')
    assert s['weighted_aux_norm']==0 and s['cosine'] is None and torch.equal(p.grad,torch.tensor([1.,2.]))
    record('zero_auxiliary_preserves_main',dict(cosine_status=s['cosine_status']))
    p=torch.nn.Parameter(torch.zeros(2));c=GradientCarrier([('p',p)])
    c.capture_primary();s=c.combine(primary_loss=0.,auxiliary_loss=0.,arm='G')
    assert p.grad is None and s['actual_lambda']==0
    record('all_None_zero_norm',dict(cosine_status=s['cosine_status'],final_grad_is_None=True))
    for phase in ('primary','auxiliary'):
        for value in (float('nan'),float('inf')):
            p=torch.nn.Parameter(torch.zeros(2));c=GradientCarrier([('p',p)])
            try:
                if phase=='auxiliary':p.grad=torch.ones(2);c.capture_primary()
                p.grad=torch.tensor([value,0.])
                if phase=='primary':c.capture_primary()
                else:c.combine(primary_loss=1.,auxiliary_loss=1.,arm='G')
            except FloatingPointError:pass
            else:raise AssertionError('Nonfinite gradient was not rejected')
    record('nonfinite_primary_and_auxiliary_rejected',dict(cases=4,optimizer_update_on_failure=False))
    for arm in ('T','L','G'):
        for invalid in [(float('nan'),1.,1.,1.),(1.,float('inf'),1.,1.),(-1.,1.,1.,1.)]:
            try:coefficients(*invalid,arm)
            except (ValueError,FloatingPointError):pass
            else:raise AssertionError('Invalid coefficient input accepted')
    assert coefficients(1.,100.,1.,1.,'T')['loss_cap_active']
    record('scalar_validation_and_teacher_actual_cap',dict(invalid_cases=9,teacher_cap_can_activate=True))
    x,y,valid,known=inputs()
    p=torch.nn.Parameter(torch.arange(54,dtype=torch.float32).reshape(1,6,9)*.01)
    c=GradientCarrier([('faces',p)])
    backward(masked_fm_loss(p,y,valid,known));c.capture_primary()
    altered=y.clone();altered[~(valid&~known)]=10000.
    backward(masked_fm_loss(p,altered,valid,known));s=c.combine(primary_loss=1.,auxiliary_loss=1.,arm='L')
    assert torch.count_nonzero(p.grad[~(valid&~known)])==0
    expected=1.1*2*(p.detach()-y)/18
    assert torch.allclose(p.grad[valid&~known],expected[valid&~known],atol=1e-7,rtol=1e-5)
    record('known_padding_masks_and_free_normalization',dict(valid_free_coordinates=18,excluded_gradient_exact_zero=True))
    cancellation=[]
    for t in (.6,.8):
        w=(1-t)**2;f,a=.1,100.
        plain=min(.1,f/a);timed=min(.1,f/(w*a))
        assert plain<.1 and timed<.1 and math.isclose(plain,timed*w,rel_tol=1e-14)
        cancellation.append(dict(t=t,time_weight=w,plain_coefficient=plain,time_weighted_effective_coefficient=timed*w))
    assert .1*.16 != .1
    record('time_weight_loss_cap_cancellation_only_when_both_active',dict(cases=cancellation,inactive_region_not_equivalent=True))
    model=copy.deepcopy(template)
    names=list(model.named_parameters());before={n:p.detach().clone() for n,p in names}
    groups={n:('linear' if '.' in n else 'other') for n,_ in names}
    snap=ParameterSnapshot(names,chunk_elements=5,parameter_groups=groups)
    with torch.no_grad():
        for i,(_,p) in enumerate(names):p.add_((i+1)*1e-5)
    m=snap.measure_and_release()
    b=torch.cat([v.flatten().double() for v in before.values()]);a=torch.cat([p.detach().flatten().double() for _,p in names]);delta=float(torch.linalg.vector_norm(a-b))
    assert math.isclose(m['actual_update_norm'],delta,rel_tol=1e-14,abs_tol=1e-14)
    assert math.isclose(m['actual_update_relative_norm'],delta/float(torch.linalg.vector_norm(b)),rel_tol=1e-14)
    assert len(m['groups'])==2 and m['released']
    try:snap.measure_and_release()
    except RuntimeError:pass
    else:raise AssertionError('Snapshot was reusable')
    record('chunked_actual_parameter_delta_and_groups',dict(update_norm=m['actual_update_norm'],reference_norm=delta,cpu_bytes=m['cpu_snapshot_bytes'],chunks=5))
    z=torch.nn.Parameter(torch.zeros(4));snap=ParameterSnapshot([('zero',z)],chunk_elements=2)
    with torch.no_grad():z.add_(1.)
    m=snap.measure_and_release();assert m['actual_update_norm']==2 and m['actual_update_relative_norm'] is None
    record('zero_parameter_relative_update_undefined',dict(relative_status=m['relative_status']))
    observed=[s for s in range(1,1001) if should_snapshot(s)]
    assert len(observed)==42 and all(s in observed for s in (1,25,500,501,1000))
    record('fixed_parameter_snapshot_schedule',dict(steps=observed,count=42))
    old,new=copy.deepcopy(template),copy.deepcopy(template)
    old_losses=objective(old,separate=False,bf16=True)
    new_losses=objective(new,separate=True,bf16=True)
    go,gn=full_gradient(old),full_gradient(new)
    assert old_losses[:2]==new_losses[:2] and torch.isfinite(go).all() and torch.isfinite(gn).all()
    bf16=dict(name='CPU_BF16_backend_rounding_descriptive',status='DESCRIPTIVE_NOT_BITWISE_REQUIREMENT',
        forward_losses_exact=old_losses[:2]==new_losses[:2],max_gradient_abs_difference=float((go-gn).abs().max()),
        relative_gradient_L2_difference=float(torch.linalg.vector_norm(go-gn)/torch.linalg.vector_norm(go)),
        note='Scaling raw gradient after BF16 backward need not equal scaling loss before BF16 backward bitwise. Both are measured; real GPU tolerance belongs to the frozen real precheck.')
    assert not torch.cuda.is_initialized()
    result=dict(status='PASS',tests=results,test_count=len(results),bf16_descriptive=bf16,counts=dict(COUNTS),
        real_model_forward=0,real_model_backward=0,real_optimizer_updates=0,CUDA_initialized=False,
        CPU_threads=torch.get_num_threads(),elapsed_seconds=time.time()-started,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('gradients.py'))},
        numerical_policy=dict(FP32_atol=ATOL,FP32_rtol=RTOL,nonbinding_L_G='BITWISE',parameter_snapshot='FP64 norm reference 1e-14',
            budget='Frozen exact scalar formula; actual FP32 weighted-gradient norm rounding reported, binding vector test allowance 1e-7'))
    (out/'gradient_cpu_tests.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],test_count=len(results),counts=COUNTS,bf16_descriptive=bf16),indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='native_t1/runs/chair_capped500_gradbudget_v1/preflight')
    run(p.parse_args().out)
