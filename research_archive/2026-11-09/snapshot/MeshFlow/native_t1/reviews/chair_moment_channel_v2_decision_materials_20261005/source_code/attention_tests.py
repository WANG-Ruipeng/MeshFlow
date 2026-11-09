"""Affected V2 CPU algebra/mapping tests; at most one disposable SGD update."""
from __future__ import annotations
import copy
import hashlib
import itertools
import json
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from models.attention import SelfAttention
from experiments.mf_h1_local0.chair_surface_attention_v1.attention_tests import _close,_nonzero,_gradients
from experiments.mf_h1_local0.chair_surface_attention_v1.attention import relation_delta as old_delta
from .attention import RelationMessageMLP,relation_delta,attention_with_relations
from .relations import build_relations


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'STARTED.json').exists():raise RuntimeError('CPU test attempt already exists; preserve its one-update budget')
    files=[Path(__file__).with_name(n) for n in ('attention.py','relations.py','model_adapter.py','attention_tests.py')]
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (out/'source_hashes.json').write_text(json.dumps(hashes,indent=2,sort_keys=True)+'\n')
    (out/'STARTED.json').write_text(json.dumps(dict(maximum_isolated_optimizer_updates=1,real_model_calls=0))+'\n')
    assert not torch.cuda.is_initialized();torch.set_num_threads(2)
    rng=torch.Generator(device='cpu').manual_seed(90261006)
    rand=lambda *shape:torch.randn(shape,generator=rng,dtype=torch.float64)
    results=dict(status='RUNNING',real_model_forwards=0,real_model_backwards=0,real_optimizer_updates=0,
        isolated_module_backward_calls=0,isolated_module_optimizer_updates=0,coordinate_sorting=False,checks={})
    def backward(loss):
        loss.backward();results['isolated_module_backward_calls']+=1
    def persist():
        (out/'results.json').write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')
    try:
        H,D,N=3,12,9;valid=torch.tensor([1,1,0,1,1,0,1,0,1],dtype=torch.bool)
        known=torch.tensor([0,3]);free=torch.tensor([1,4,6,8]);K=len(known)
        baseq,basek,basev=rand(N,H,4),rand(N,H,4),rand(N,H,4);R=rand(len(free),3,K,11)
        module=RelationMessageMLP(H,D,seed=618205).double();_nonzero(module,111)
        modules=[copy.deepcopy(module) for _ in range(3)];leaves=[];outputs=[]
        for m,mode in zip(modules,('compact_rearranged','compact_literal','V1_reference')):
            q,k,v,r=[a.clone().requires_grad_() for a in (baseq,basek,basev,R)];leaves.append((q,k,v,r))
            fn=old_delta if mode=='V1_reference' else relation_delta
            value,_=fn(q,k,v,valid,free,known,r,m,query_chunk=7,literal=mode=='compact_literal')
            outputs.append(value)
        weight=rand(*outputs[0].shape)
        for output in outputs:backward((output*weight).sum())
        errors={}
        for j,name in ((1,'literal'),(2,'V1_reference')):
            errors[name]=dict(output_maxabs=_close(outputs[0],outputs[j]),parameters={n:_close(_gradients(modules[0])[n],_gradients(modules[j])[n]) for n,_ in module.named_parameters()},
                inputs={n:_close(a.grad,b.grad) for n,a,b in zip(('q','k','v','R'),leaves[0],leaves[j])})
        for triple in leaves:
            for tensor in triple[:3]:assert torch.count_nonzero(tensor.grad[~valid])==0
        results['checks']['compact_literal_and_V1_forward_gradients']=errors

        # Full pair-u telemetry agrees with a small literal expansion.
        _,stats=relation_delta(baseq,basek,basev,valid,free,known,R,module,collect_stats=True)
        pair=module.fc2(module.features(R.reshape(-1,K,11)));b,u=pair[...,:H],pair[...,H:]
        telemetry={name:_close(stats[name],value) for name,value in dict(b_RMS=b.square().mean().sqrt(),b_max=b.abs().max(),u_RMS=u.square().mean().sqrt(),u_max=u.abs().max()).items()}
        results['checks']['exact_effective_pair_telemetry']=telemetry

        # Compact primitive dimensions/results cannot change with padding length
        # or values when valid QKV and relation values are unchanged.
        refs={};delta0,unused=relation_delta(baseq,basek,basev,valid,free,known,R,module,debug=refs)
        for pad in (1,4,9):
            args=[torch.cat((a,rand(pad,H,4)*1000)) for a in (baseq,basek,basev)]
            vm=torch.cat((valid,torch.zeros(pad,dtype=torch.bool)));stages={}
            delta,unused=relation_delta(*args,vm,free,known,R,module,debug=stages)
            assert torch.equal(delta,delta0)
            for name in ('compact_q','compact_k','compact_v','phi','b','logits','base_weights','weights','content_delta','pair_extra','aggregate'):
                assert torch.equal(stages[name],refs[name]),name
        results['checks']['new_channel_padding_shape_independence']=dict(status='PASS',padding_counts=[1,4,9],every_stage_bitwise=True)

        # No-known means explicit zero correction with no empty softmax.
        empty=torch.empty(0,dtype=torch.long)
        zero,_=relation_delta(baseq,basek,basev,valid,torch.nonzero(valid).flatten(),empty,
            torch.empty(int(valid.sum()),3,0,11,dtype=torch.float64),module)
        assert torch.count_nonzero(zero)==0 and torch.isfinite(zero).all()
        results['checks']['no_known']=dict(explicit_zero=True,finite=True)

        permutations=list(itertools.permutations(range(3)));symmetry=[]
        for count in (1,2,4):
            faces=count+4;coords=rand(1,faces,3,3);hidden=rand(1,faces,3,D);t=torch.tensor([.9],dtype=torch.float64)
            km=torch.arange(faces)[None]<count;vm=torch.ones_like(km)
            attention=SelfAttention(D,H,mixed_precision='fp32',qk_norm=True).double().eval()
            m=RelationMessageMLP(H,D,seed=count).double();_nonzero(m,400+count)
            r=build_relations(coords.reshape(1,faces,9),t,km,vm,reference=True)
            with sdpa_kernel(SDPBackend.MATH):
                face,delta,_=attention_with_relations(attention,hidden.mean(2),vm,r,m)
            expected=face[:,:,None]+delta.reshape(1,faces,3,D)
            for shift in range(6):
                cp=torch.tensor([permutations[(j+shift)%6] for j in range(faces)])
                fp=torch.randperm(faces,generator=rng)
                c=coords[:,torch.arange(faces)[:,None],cp,:][:,fp]
                h=hidden[:,torch.arange(faces)[:,None],cp,:][:,fp]
                rr=build_relations(c.reshape(1,faces,9),t,km[:,fp],vm[:,fp],reference=True)
                with sdpa_kernel(SDPBackend.MATH):f,d,_=attention_with_relations(attention,h.mean(2),vm[:,fp],rr,m)
                actual=f[:,:,None]+d.reshape(1,faces,3,D)
                want=expected[:,torch.arange(faces)[:,None],cp,:][:,fp]
                symmetry.append(dict(K=count,corner_cycle=shift,maxabs=_close(actual,want,atol=1e-10,rtol=1e-9)))
        results['checks']['six_corner_and_face_permutations']=symmetry

        # Interleaved slots exercise mapping/scatter, invalid known flags, values,
        # pure geometry cache invalidation, and gradients w.r.t. both padding inputs.
        mask=valid[None];known_mask=torch.zeros_like(mask);known_mask[:,known]=True
        coords=rand(1,N,9);features=rand(1,N,D);t=torch.tensor([.43],dtype=torch.float64)
        attention=SelfAttention(D,H,mixed_precision='fp32',qk_norm=True).double().eval();cache=[]
        effective=[];records=[]
        for flag in (False,True):
            km=known_mask.clone();km[:,~valid]=flag
            for magnitude in (0.,1.,1000.):
                x=coords.clone();h=features.clone();x[:,~valid]=rand(1,int((~valid).sum()),9)*magnitude;h[:,~valid]=rand(1,int((~valid).sum()),D)*magnitude
                x.requires_grad_();h.requires_grad_()
                rr=build_relations(x,t,km,mask,static_cache=cache,reference=True)
                assert torch.equal(rr[0]['valid_indices'],torch.nonzero(valid).flatten())
                assert torch.equal(rr[0]['known_indices'],known) and torch.equal(rr[0]['free_face_indices'],free)
                assert cache[0] is None  # leaf-linked C must not be reused/detached
                with sdpa_kernel(SDPBackend.MATH):f,d,_=attention_with_relations(attention,h,mask,rr,module)
                assert torch.count_nonzero(d.reshape(1,N,3,D)[:,~valid])==0
                assert torch.count_nonzero(d.reshape(1,N,3,D)[:,known])==0
                output=(f.repeat_interleave(3,1)+d).reshape(1,N,3,D)[:,valid]
                if effective:assert torch.equal(output,effective[0])
                effective.append(output.detach())
                backward(output.square().sum())
                assert torch.count_nonzero(x.grad[:,~valid])==0 and torch.count_nonzero(h.grad[:,~valid])==0
                records.append(dict(invalid_known_flag=flag,magnitude=magnitude,effective_output_bitwise=True,padding_coordinate_gradient_zero=True,padding_feature_gradient_zero=True))
        results['checks']['same_shape_padding_values_flags_and_backward_scatter']=records
        cache=[];r0=build_relations(coords,t,known_mask,mask,reference=True,static_cache=cache)[0]['R']
        old=cache[0];r1=build_relations(coords,t,known_mask,mask,reference=True,static_cache=cache)[0]['R']
        assert torch.equal(r0,r1) and cache[0]['centroid'].data_ptr()==old['centroid'].data_ptr()
        changed=coords.clone();changed[:,known]*=1.1
        r2=build_relations(changed,t,known_mask,mask,reference=True,static_cache=cache)[0]['R']
        assert not torch.equal(r0,r2)
        results['checks']['geometry_cache']=dict(same_C_reused=True,changed_C_invalidates=True,grad_C_never_cached=True,full_masks_bound=True)

        # Existing base SDPA operation is retained, and zero correction remains
        # connected to its final parameters (no parameter-dependent bypass).
        az=copy.deepcopy(attention);base=copy.deepcopy(attention);zero=RelationMessageMLP(H,D,seed=618205).double()
        rel=build_relations(coords,t,known_mask,mask,reference=True)
        with sdpa_kernel(SDPBackend.MATH):
            f,d,_=attention_with_relations(az,features,mask,rel,zero);adapted=f.repeat_interleave(3,1)+d
            original=base(features,mask=mask).repeat_interleave(3,1)
        assert torch.equal(original,adapted) and torch.count_nonzero(d)==0
        weights=rand(*original.shape);backward((original*weights).sum());backward((adapted*weights).sum())
        errors={n:_close(_gradients(base)[n],_gradients(az)[n]) for n,_ in base.named_parameters()}
        results['checks']['zero_init_SDPA']=dict(output_bitwise=True,old_gradient_maxabs=errors,external_zero_gate=False)

        # Sole authorized isolated update, placed after all non-update tests.
        reachable=RelationMessageMLP(H,D,seed=618205).double();opt=torch.optim.SGD(reachable.parameters(),lr=.01)
        d0,_=relation_delta(baseq,basek,basev,valid,free,known,R,reachable)
        backward((d0*weight).sum());g0=_gradients(reachable)
        assert g0['fc2.weight'].norm()>0 and torch.count_nonzero(g0['fc1.weight'])==0
        results['isolated_module_optimizer_updates']=1;persist()  # reserve before actual isolated update
        opt.step();opt.zero_grad(set_to_none=True)
        d1,_=relation_delta(baseq,basek,basev,valid,free,known,R,reachable)
        backward((d1*weight).sum());g1=_gradients(reachable)
        assert g1['fc1.weight'].norm()>0 and g1['fc1.bias'].norm()>0
        results['checks']['isolated_gradient_reachability']=dict(first_fc1_norm=float(g0['fc1.weight'].norm()),
            first_fc2_norm=float(g0['fc2.weight'].norm()),after_one_update_fc1_norm=float(g1['fc1.weight'].norm()),updates=1)
        assert not torch.cuda.is_initialized();results['status']='PASS';persist();return results
    except BaseException as exc:
        import traceback
        results.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc());persist();raise


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();print(json.dumps(run(args.out)))
