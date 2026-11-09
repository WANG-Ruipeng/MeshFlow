"""Frozen held-noise path and shared-hidden gradient diagnostics; zero updates.

No mesh extraction, optimizer creation, loss balancing, or model selection.
All endpoint probes use true targets and are not free generation results.
"""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import argparse
import hashlib
import math
import time
import traceback
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from common import ROOT,dump,sha
from single_state_common import arrays
from interaction_semantics import extended_semantic,METRIC_DEFINITIONS

PROBE_TIMES=(0.,.25,.5,.75,.9,.99)
GRADIENT_PROBE_IDS=(0,8,16,24)
BLOCKS=(('x',0,3,3/22),('n',3,6,3/22),('F',6,22,16/22))
GRADIENT_DEFINITIONS=dict(
    version='SHARED_HIDDEN_GRADIENT_DIAGNOSTIC_V1',probe_ids=list(GRADIENT_PROBE_IDS),times=list(PROBE_TIMES),
    hidden='128-dimensional actual shared point tensor after inserted residual block, before mean/max pooling',
    unweighted='d(mean over all elements and that block channels of velocity squared error)/dH',
    weighted='Multiply each unweighted block gradient by 3/22,3/22,16/22 respectively',
    norm='Euclidean norm across all batch/element/hidden entries',
    pair_cosine='Cosine between unweighted gradient vectors; positive scalar weights preserve the angle',
    total='d(mean velocity squared error over all22 channels)/dH; independently compared with weighted block-gradient sum',
    update_policy='autograd.grad only; no optimizer or backward accumulation into parameter.grad',
    interpretation='Diagnostics in each model hidden representation, not parameter-update magnitudes or proof of causal field dominance; do not alter loss weights')


def tensor_digest(model):
    digest=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        value=tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode());digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode());digest.update(value.tobytes())
    return digest.hexdigest()


def block_losses(velocity,target_velocity):
    sq=(velocity-target_velocity).square()
    losses={name:sq[...,start:stop].mean() for name,start,stop,_ in BLOCKS}
    total=sq.mean()
    if not bool(torch.isfinite(total)):
        raise FloatingPointError('Nonfinite diagnostic velocity loss')
    record=dict(velocity_mse=float(total.detach()))
    for name,_,_,weight in BLOCKS:
        record['velocity_'+name+'_mse']=float(losses[name].detach())
        record['velocity_'+name+'_weighted_contribution']=float((weight*losses[name]).detach())
        record['velocity_'+name+'_weight']=weight
    record['weighted_loss_sum']=sum(record['velocity_'+name+'_weighted_contribution'] for name,_,_,_ in BLOCKS)
    record['weighted_loss_sum_absolute_error']=abs(record['weighted_loss_sum']-record['velocity_mse'])
    return total,losses,record


def gradient_diagnostics(velocity,target_velocity,shared_hidden):
    total,losses,record=block_losses(velocity,target_velocity)
    grads={name:torch.autograd.grad(losses[name],shared_hidden,retain_graph=True,create_graph=False)[0].detach()
           for name,_,_,_ in BLOCKS}
    total_grad=torch.autograd.grad(total,shared_hidden,retain_graph=False,create_graph=False)[0].detach()
    combined=sum(weight*grads[name] for name,_,_,weight in BLOCKS)
    if not bool(torch.isfinite(total_grad).all() and all(torch.isfinite(g).all() for g in grads.values())):
        raise FloatingPointError('Nonfinite shared-hidden diagnostic gradient')
    record.update(shared_hidden_shape=list(shared_hidden.shape),shared_hidden_rms=float(shared_hidden.detach().square().mean().sqrt()),
                  gradient_total_norm=float(torch.linalg.vector_norm(total_grad)),
                  gradient_weighted_sum_norm=float(torch.linalg.vector_norm(combined)),
                  gradient_weighted_sum_difference_norm=float(torch.linalg.vector_norm(combined-total_grad)),
                  all_gradients_finite=bool(torch.isfinite(total_grad).all() and all(torch.isfinite(g).all() for g in grads.values())))
    for name,_,_,weight in BLOCKS:
        norm=float(torch.linalg.vector_norm(grads[name]))
        record['gradient_'+name+'_unweighted_norm']=norm
        record['gradient_'+name+'_weighted_norm']=weight*norm
    names=[b[0] for b in BLOCKS]
    for i,left in enumerate(names):
        for right in names[i+1:]:
            a=grads[left].reshape(-1).double();b=grads[right].reshape(-1).double()
            denom=torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b)
            cosine=float(torch.clamp(torch.dot(a,b)/denom,-1,1)) if float(denom)>0 else None
            record['gradient_cosine_'+left+'_'+right]=cosine
            record['gradient_angle_deg_'+left+'_'+right]=math.degrees(math.acos(cosine)) if cosine is not None else None
    return record


def numeric_means(records):
    keys=sorted({k for row in records for k,v in row.items() if isinstance(v,(int,float)) and not isinstance(v,bool)})
    out={}
    for key in keys:
        values=[float(r[key]) for r in records if isinstance(r.get(key),(int,float))
                and not isinstance(r.get(key),bool) and np.isfinite(r[key])]
        if values:out[key]=dict(mean=float(np.mean(values)),finite_count=len(values))
    return out


def main(arm,device='cuda'):
    from interaction_model import InteractionVelocity
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    data=arrays(ROOT/'data.npz');probe=arrays(ROOT/'probe_noise.npz')
    if data['states'].shape!=(1,256,22):raise ValueError('This experiment freezes one 256-element state')
    target=data['states'][0];noise=probe['noise'];times=probe['times']
    if noise.shape!=(32,256,22) or not np.array_equal(times,np.asarray(PROBE_TIMES,dtype=times.dtype)):
        raise ValueError('Frozen 32-noise/six-time probe bank differs')
    checkpoint=ROOT/'training'/arm/'final.pt'
    ck=torch.load(checkpoint,map_location=device,weights_only=False)
    if ck['status']!='COMPLETED' or ck['completed_steps']!=9000 or ck['new_updates']!=6000:
        raise ValueError('Probe requires the final 9000-step/6000-new-update checkpoint')
    for key in ['mean','scale']:
        if not torch.equal(ck[key].cpu(),torch.tensor(data[key],dtype=ck[key].dtype)):
            raise ValueError('Checkpoint/data chart mismatch: '+key)
    model=InteractionVelocity(arm=arm,width=128).to(device)
    model.load_state_dict(ck['model_state_dict'],strict=True);model.eval();model.zero_grad(set_to_none=True)
    before=tensor_digest(model)
    mean=torch.tensor(data['mean'],dtype=torch.float32,device=device)
    scale=torch.tensor(data['scale'],dtype=torch.float32,device=device)
    out=ROOT/'probes'/arm;out.mkdir(parents=True,exist_ok=False)
    dump(out/'METRIC_DEFINITIONS.json',METRIC_DEFINITIONS)
    dump(out/'GRADIENT_DEFINITIONS.json',GRADIENT_DEFINITIONS)
    metadata=dict(arm=arm,checkpoint_sha256=sha(checkpoint),data_sha256=sha(ROOT/'data.npz'),
                  probe_noise_sha256=sha(ROOT/'probe_noise.npz'),shared_hidden='post-residual pre-pooling',
                  source_files=[dict(path=str(p),sha256=sha(p)) for p in
                    [ROOT/'code/probe_interaction.py',ROOT/'code/interaction_semantics.py',ROOT/'code/interaction_model.py']],
                  generated=False,oracle_assisted=True,optimizer_updates=0,geometry_extractions=0,
                  velocity='raw network output equals direct FM total velocity; no transport term')
    dump(out/'PROBE_PROTOCOL.json',metadata)
    bbox=float(np.linalg.norm(np.ptp(target[:,:3],axis=0)))
    rows=[];gradients=[];all_endpoints=[];all_paths=[];all_permutations=[];started=time.perf_counter()
    for i in range(32):
        xyz=noise[i,:,:3].astype(np.float64)*data['scale'][:3]+data['mean'][:3]
        assignment_rows,perm=linear_sum_assignment(((xyz[:,None,:]-target[None,:,:3])**2).sum(-1))
        if not np.array_equal(assignment_rows,np.arange(256)):raise AssertionError('Hungarian row order')
        assigned=target[perm]
        zt=torch.tensor(((assigned-data['mean'])/data['scale']).astype(np.float32),device=device)[None]
        z0=torch.tensor(noise[i:i+1],device=device);u=zt-z0;endpoints=[];paths=[]
        for tvalue in times:
            t=float(tvalue);z=(1-t)*z0+t*zt
            path=(z*scale+mean)[0].cpu().numpy().astype(np.float64)
            row=dict(arm=arm,probe_id=i,time=t,generated=False,oracle_assisted=True,
                     zero_velocity_endpoint=extended_semantic(path,assigned,bbox))
            try:
                if i in GRADIENT_PROBE_IDS:
                    velocity,hidden=model.forward_with_hidden(z,torch.tensor([t],device=device))
                    if hidden.shape!=(1,256,128):raise AssertionError('Shared hidden shape differs')
                    grad=gradient_diagnostics(velocity,u,hidden)
                    gradients.append(dict(arm=arm,probe_id=i,time=t,**grad))
                else:
                    with torch.no_grad():velocity=model(z,torch.tensor([t],device=device))
                _,_,loss_record=block_losses(velocity.detach(),u)
                physical=((z+(1-t)*velocity.detach())*scale+mean)[0].cpu().numpy().astype(np.float64)
                if not np.isfinite(physical).all():
                    raise FloatingPointError('Nonfinite conditional endpoint')
                row.update(status='COMPLETE',**loss_record,endpoint=extended_semantic(physical,assigned,bbox))
            except Exception as exc:
                physical=np.full((256,22),np.nan)
                row.update(status='PROBE_FAILED',error=repr(exc),traceback=traceback.format_exc(),endpoint=dict(finite=False))
            rows.append(row);endpoints.append(physical);paths.append(path)
        all_endpoints.append(np.stack(endpoints));all_paths.append(np.stack(paths));all_permutations.append(perm)
        print('INTERACTION_PROBE',arm,i,rows[-1]['status'],flush=True)
    after=tensor_digest(model)
    weights_unchanged=before==after;grad_buffers_none=all(p.grad is None for p in model.parameters())
    np.savez_compressed(out/'probe_states.npz',endpoint_predictions=np.stack(all_endpoints),path_states=np.stack(all_paths),
                        target_permutations=np.stack(all_permutations),times=times)
    dump(out/'PROBE_ROWS.json',rows);dump(out/'GRADIENT_ROWS.json',gradients)
    by_time=[]
    for tvalue in times:
        t=float(tvalue);at_time=[r for r in rows if r['time']==t]
        by_time.append(dict(time=t,records=len(at_time),completed=sum(r['status']=='COMPLETE' for r in at_time),
            velocity_metrics=numeric_means(at_time),
            endpoint_metrics=numeric_means([r['endpoint'] for r in at_time]),
            zero_velocity_metrics=numeric_means([r['zero_velocity_endpoint'] for r in at_time]),
            gradient_metrics=numeric_means([r for r in gradients if r['time']==t])))
    expected_gradients=len(GRADIENT_PROBE_IDS)*len(PROBE_TIMES)
    passed=(weights_unchanged and grad_buffers_none and len(rows)==192 and all(r['status']=='COMPLETE' for r in rows)
            and len(gradients)==expected_gradients and all(g['all_gradients_finite'] for g in gradients))
    summary=dict(passed=passed,arm=arm,probe_count=32,times=times.tolist(),semantic_records=len(rows),
        completed_records=sum(r['status']=='COMPLETE' for r in rows),gradient_records=len(gradients),expected_gradient_records=expected_gradients,
        optimizer_updates=0,geometry_extractions=0,model_tensor_digest_before=before,model_tensor_digest_after=after,
        model_weights_unchanged=weights_unchanged,parameter_grad_buffers_remain_none=grad_buffers_none,
        by_time=by_time,seconds=time.perf_counter()-started,**{k:v for k,v in metadata.items() if k not in ['arm','optimizer_updates','geometry_extractions']},
        note='Conditional target-assisted velocity probes, not free generation. Late endpoints contain most of the target; zero-velocity baselines retained. Gradient norms are diagnostics only. Point coverage is not generative distribution coverage.')
    dump(out/'SUMMARY.json',summary)
    print('INTERACTION_PROBE_DONE',arm,passed,flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['B0','B1','B2'],required=True)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args();result=main(args.arm,args.device)
    raise SystemExit(0 if result['passed'] else 1)
