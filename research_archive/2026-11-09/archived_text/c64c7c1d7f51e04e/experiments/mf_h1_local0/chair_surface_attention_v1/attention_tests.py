"""Isolated CPU attention tests; no Chair model loading or real-model forward.

One tiny SGD test update belongs solely to a disposable RelationMessageMLP and
is explicitly recorded. Production model/optimizer state is never accepted.
"""
from __future__ import annotations
import copy
import itertools
import json
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from models.attention import SelfAttention
from .attention import RelationMessageMLP,relation_delta,attention_with_relations


def _gradients(module):
    return {n:torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for n,p in module.named_parameters()}


def _nonzero(module,seed):
    generator=torch.Generator(device='cpu').manual_seed(seed)
    with torch.no_grad():
        module.fc2.weight.copy_(torch.randn(module.fc2.weight.shape,generator=generator,dtype=module.fc2.weight.dtype)*.035)
        module.fc2.bias.copy_(torch.randn(module.fc2.bias.shape,generator=generator,dtype=module.fc2.bias.dtype)*.015)


def _close(a,b,*,atol=1e-11,rtol=1e-10):
    if not torch.allclose(a,b,atol=atol,rtol=rtol):
        raise AssertionError('Array mismatch maxabs='+str(float((a-b).abs().max())))
    return float((a-b).abs().max()) if a.numel() else 0.


def run(output=None):
    if torch.cuda.is_initialized():raise RuntimeError('Isolated CPU test process cannot own CUDA')
    torch.set_num_threads(2)
    generator=torch.Generator(device='cpu').manual_seed(90261005)
    results=dict(status='RUNNING',real_model_forwards=0,real_model_backwards=0,real_optimizer_updates=0,
                 isolated_module_optimizer_updates=0,isolated_module_backward_calls=0,checks={})
    rand=lambda *shape:torch.randn(shape,generator=generator,dtype=torch.float64)
    H,D,N=3,12,9
    free=torch.arange(3,N);known=torch.arange(3);valid=torch.ones(N,dtype=torch.bool)
    q,k,v=rand(N,H,D//H),rand(N,H,D//H),rand(N,H,D//H)
    R=rand(N-3,3,3,11)
    module=RelationMessageMLP(H,D,seed=618205).double();_nonzero(module,111)
    clone=copy.deepcopy(module)
    qq=q.clone().requires_grad_();kk=k.clone().requires_grad_();vv=v.clone().requires_grad_();rr=R.clone().requires_grad_()
    q2=q.clone().requires_grad_();k2=k.clone().requires_grad_();v2=v.clone().requires_grad_();r2=R.clone().requires_grad_()
    optimized,_=relation_delta(qq,kk,vv,valid,free,known,rr,module,query_chunk=7)
    literal,_=relation_delta(q2,k2,v2,valid,free,known,r2,clone,query_chunk=7,literal=True)
    output_error=_close(optimized,literal)
    weight=rand(*optimized.shape)
    (optimized*weight).sum().backward();(literal*weight).sum().backward();results['isolated_module_backward_calls']+=2
    grad_errors={n:_close(_gradients(module)[n],_gradients(clone)[n]) for n,_ in module.named_parameters()}
    for name,a,b in [('q',qq,q2),('k',kk,k2),('v',vv,v2),('R',rr,r2)]:grad_errors[name]=_close(a.grad,b.grad)
    results['checks']['literal_vs_rearranged_pair_value']=dict(output_maxabs=output_error,gradient_maxabs=grad_errors,
        dtype='float64',production_pair_value_D_expansion=False)

    # Literal full-message definition versus original message plus correction.
    phi=module.features(R.reshape(-1,3,11));b=module.fc2(phi)[...,:H]
    logits=torch.einsum('qhd,khd->hqk',q[free.repeat_interleave(3)],k)/(D//H)**.5
    corrected=logits+torch.zeros_like(logits).index_copy(2,known,b.permute(2,0,1))
    w=corrected.softmax(-1);basew=logits.softmax(-1)
    uv=module.fc2(phi)[...,H:].reshape(3*len(free),3,H,D//H)
    full=torch.einsum('hqk,khd->qhd',w,v)+torch.einsum('hqk,qkhd->qhd',w.index_select(2,known),uv)
    base=torch.einsum('hqk,khd->qhd',basew,v)
    results['checks']['full_message_equivalence']=dict(maxabs=_close(full,base+optimized.detach().reshape_as(base)),
        source_softmax_denominator='All nine valid source faces, three known and six free')

    # Original SDPA/output-projection arithmetic is retained at zero initialization.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(472);attn=SelfAttention(D,H,mixed_precision='fp32',qk_norm=True).double().eval()
    attn2=copy.deepcopy(attn);zero=RelationMessageMLP(H,D,seed=618205).double()
    xin=rand(1,N,D);rel=[dict(free_face_indices=free,known_indices=known,R=R)]
    mask=valid[None];outweight=rand(1,3*N,D)
    with sdpa_kernel(SDPBackend.MATH):
        original=attn(xin,mask=mask).repeat_interleave(3,dim=1)
        face,delta,_=attention_with_relations(attn2,xin,mask,rel,zero)
        adapted=face.repeat_interleave(3,dim=1)+delta
    assert torch.equal(original,adapted) and torch.count_nonzero(delta)==0
    (original*outweight).sum().backward();(adapted*outweight).sum().backward();results['isolated_module_backward_calls']+=2
    old_grad_errors={n:_close(_gradients(attn)[n],_gradients(attn2)[n]) for n,_ in attn.named_parameters()}
    results['checks']['zero_init_original_SDPA']=dict(output_bitwise=True,old_parameter_gradient_maxabs=old_grad_errors,
        dtype='float64',backend='SDPA math',external_zero_gate=False)

    # Nonzero relation parameters, multiple K, full face/corner permutation and padding.
    from .relations import build_relations
    perms=list(itertools.permutations(range(3)));permutation_rows=[]
    for K in (1,2,4):
        faces=K+4
        coords=rand(1,faces,3,3);hidden=rand(1,faces,3,D);time=torch.tensor([.9],dtype=torch.float64)
        km=torch.arange(faces)[None]<K;vm=torch.ones_like(km)
        a=SelfAttention(D,H,mixed_precision='fp32',qk_norm=True).double().eval()
        m=RelationMessageMLP(H,D,seed=K).double();_nonzero(m,400+K)
        for mode in ('MOMENT_XY','SURFACE_XY'):
            rel0=build_relations(coords.reshape(1,faces,9),time,km,vm,mode,reference=True)
            with sdpa_kernel(SDPBackend.MATH):
                base,delta,_=attention_with_relations(a,hidden.mean(2),vm,rel0,m)
            expected=base[:,:,None,:]+delta.reshape(1,faces,3,D)
            assert torch.count_nonzero(delta)>0
            for shift in range(6):
                cp=torch.tensor([perms[(i+shift)%6] for i in range(faces)])
                fp=torch.randperm(faces,generator=generator)
                transformed_c=coords[:,torch.arange(faces)[:,None],cp,:][:,fp]
                transformed_h=hidden[:,torch.arange(faces)[:,None],cp,:][:,fp]
                relp=build_relations(transformed_c.reshape(1,faces,9),time,km[:,fp],vm[:,fp],mode,reference=True)
                with sdpa_kernel(SDPBackend.MATH):
                    bp,dp,_=attention_with_relations(a,transformed_h.mean(2),vm[:,fp],relp,m)
                actual=bp[:,:,None,:]+dp.reshape(1,faces,3,D)
                want=expected[:,torch.arange(faces)[:,None],cp,:][:,fp]
                error=_close(actual,want,atol=1e-10,rtol=1e-9)
                permutation_rows.append(dict(K=K,mode=mode,corner_cycle=shift,maxabs=error))
            padded_c=torch.cat((coords,100*rand(1,2,3,3)),1);padded_h=torch.cat((hidden,100*rand(1,2,3,D)),1)
            kpad=torch.cat((km,torch.zeros((1,2),dtype=torch.bool)),1);vpad=torch.cat((vm,torch.zeros((1,2),dtype=torch.bool)),1)
            relpad=build_relations(padded_c.reshape(1,faces+2,9),time,kpad,vpad,mode,reference=True)
            with sdpa_kernel(SDPBackend.MATH):
                pb,pd,_=attention_with_relations(a,padded_h.mean(2),vpad,relpad,m)
            _close((pb[:,:,None,:]+pd.reshape(1,faces+2,3,D))[:,:faces],expected)
            assert torch.count_nonzero(pd.reshape(1,faces+2,3,D)[:,faces:])==0
    results['checks']['nonzero_permutations_padding']=dict(rows=permutation_rows,invalid_keys_excluded=True,
        padding_relation_queries_zero=True,padding_queries_excluded_from_valid_output_comparison=True)

    # No known sources: no relation message, original all-valid self-attention remains.
    none=[dict(free_face_indices=torch.arange(N),known_indices=torch.empty(0,dtype=torch.long),R=torch.empty(N,3,0,11,dtype=torch.float64))]
    with sdpa_kernel(SDPBackend.MATH):
        face,delta,_=attention_with_relations(attn,xin,mask,none,module)
        old=attn(xin,mask=mask)
    assert torch.equal(face,old) and torch.count_nonzero(delta)==0 and torch.isfinite(face).all()
    results['checks']['no_known']=dict(base_bitwise_unchanged=True,correction_exact_zero=True,finite=True)

    # One explicitly isolated module-only test update, no real model/optimizer.
    reachable=RelationMessageMLP(H,D,seed=618205).double()
    optimizer=torch.optim.SGD(reachable.parameters(),lr=.01)
    d0,_=relation_delta(q,k,v,valid,free,known,R,reachable)
    (d0*weight).sum().backward();results['isolated_module_backward_calls']+=1
    g0=_gradients(reachable)
    assert g0['fc2.weight'].abs().max()>0 and g0['fc2.bias'].abs().max()>0
    assert torch.count_nonzero(g0['fc1.weight'])==0 and torch.count_nonzero(g0['fc1.bias'])==0
    optimizer.step();results['isolated_module_optimizer_updates']+=1;optimizer.zero_grad(set_to_none=True)
    d1,_=relation_delta(q,k,v,valid,free,known,R,reachable)
    (d1*weight).sum().backward();results['isolated_module_backward_calls']+=1
    g1=_gradients(reachable);assert g1['fc1.weight'].abs().max()>0 and g1['fc1.bias'].abs().max()>0
    results['checks']['gradient_reachability']=dict(first_final_weight_grad_norm=float(g0['fc2.weight'].norm()),
        first_internal_weight_grad_norm=float(g0['fc1.weight'].norm()),
        after_one_isolated_update_internal_weight_grad_norm=float(g1['fc1.weight'].norm()),isolated_updates=1)
    # Seed identity is independent of MOMENT/SURFACE mode and does not touch global RNG.
    rng=torch.random.get_rng_state().clone();one=RelationMessageMLP(12,768,seed=618208);two=RelationMessageMLP(12,768,seed=618208)
    assert torch.equal(torch.random.get_rng_state(),rng)
    assert all(torch.equal(one.state_dict()[k],two.state_dict()[k]) for k in one.state_dict())
    results['checks']['identical_initialization']=dict(bitwise=True,global_rng_unchanged=True,
        per_layer_parameters=sum(p.numel() for p in one.parameters()),four_layer_parameters=4*sum(p.numel() for p in one.parameters()))
    assert not torch.cuda.is_initialized();results['status']='PASS'
    if output is not None:
        p=Path(output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')
    return results


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(run(a.output)))
