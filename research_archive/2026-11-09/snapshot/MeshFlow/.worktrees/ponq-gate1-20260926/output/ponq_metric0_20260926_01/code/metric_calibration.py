"""One-time, positive, fixed field metric calibrated on shared parameter gradients.

No optimizer, updates, extraction, target gauge change or matching intervention.
Only 3x3 Gram matrices are persisted; full gradient vectors are transient.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from common import ROOT,dump,sha
from interaction_model import InteractionVelocity
from interaction_semantics import extended_semantic
from probe_interaction import tensor_digest
from single_state_common import arrays

FIELD_NAMES=('x','n','F')
BLOCKS=((0,3),(3,6),(6,22))
OLD_WEIGHTS=np.array([3/22,3/22,16/22],dtype=np.float64)
PROBE_IDS=(0,8,16,24)
PROBE_TIMES=(0.,.25,.5,.75,.9,.99)
SHARED_PARAMETER_NAMES=tuple(module+'.'+kind for module in ['encoder.0','encoder.2','decoder.0','decoder.2'] for kind in ['weight','bias'])
EXPECTED_SHARED_PARAMETERS=85376
EPS=1e-12
MIN_GRADIENT_RMS=1e-8
MAX_WEIGHT_RATIO=1e6
COMMON_SCALE_LIMITS=(1e-6,1e6)
IDENTITY_RTOL=1e-5


def vector_digest(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def shared_parameters(model):
    named=dict(model.named_parameters())
    if set(named)!=set(SHARED_PARAMETER_NAMES)|{'decoder.4.weight','decoder.4.bias'}:raise ValueError('Model parameter set is not the frozen original DeepSets')
    params=[named[name] for name in SHARED_PARAMETER_NAMES]
    metadata=[dict(name=name,shape=list(p.shape),numel=p.numel()) for name,p in zip(SHARED_PARAMETER_NAMES,params)]
    if sum(p.numel() for p in params)!=EXPECTED_SHARED_PARAMETERS:raise ValueError('Shared parameter count mismatch')
    return params,metadata


def _gradient(loss,params,retain_graph):
    grads=torch.autograd.grad(loss,params,retain_graph=retain_graph,create_graph=False,allow_unused=False)
    value=torch.cat([g.detach().reshape(-1).cpu().double() for g in grads]).numpy()
    if not np.isfinite(value).all():raise FloatingPointError('Nonfinite shared-parameter gradient')
    return value


def _weighted_effect(coefficients,block_gradients,gram,actual_gradient):
    combined=coefficients@block_gradients;difference=float(np.linalg.norm(combined-actual_gradient))
    actual=float(np.linalg.norm(actual_gradient));predicted_sq=float(coefficients@gram@coefficients)
    if not np.isfinite(predicted_sq) or predicted_sq<0:raise FloatingPointError('Invalid weighted Gram quadratic form')
    gram_norm=float(np.sqrt(predicted_sq));relative=difference/max(actual,1e-30)
    return dict(weights=dict(zip(FIELD_NAMES,coefficients.tolist())),gradient_norm=actual,
        weighted_block_sum_norm=float(np.linalg.norm(combined)),gradient_sum_difference_norm=difference,
        gradient_sum_relative_error=relative,gram_predicted_gradient_norm=gram_norm,
        gram_vs_autograd_absolute_error=abs(gram_norm-actual),
        identity_passed=relative<=IDENTITY_RTOL and abs(gram_norm-actual)<=IDENTITY_RTOL*max(1.,actual))


def evaluate_checkpoint(checkpoint,root=ROOT,device='cuda',weights=None):
    """Evaluate exactly 24 equal-weight conditional probes; return JSON-safe dict.

    weights is None for initial calibration or {'x':a_x,'n':a_n,'F':a_F}.
    When supplied, both old and new weighted total gradients are independently
    differentiated and checked against the unweighted field-gradient Gram matrix.
    """
    root=Path(root);checkpoint=Path(checkpoint);started=time.perf_counter()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    data=arrays(root/'data.npz');probe=arrays(root/'probe_noise.npz')
    target=data['states'][0];noise=probe['noise']
    if data['states'].shape!=(1,256,22) or noise.shape!=(32,256,22):raise ValueError('Frozen target/probe shape mismatch')
    if not np.array_equal(probe['times'],np.asarray(PROBE_TIMES,dtype=probe['times'].dtype)):raise ValueError('Frozen times differ')
    coeff=None if weights is None else np.array([weights[k] for k in FIELD_NAMES],dtype=np.float64)
    if coeff is not None and (not np.isfinite(coeff).all() or not np.all(coeff>0)):raise ValueError('Weights must be finite positive constants')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    for key in ['mean','scale']:
        if not torch.equal(saved[key].cpu(),torch.tensor(data[key],dtype=saved[key].dtype)):raise ValueError('Checkpoint chart mismatch: '+key)
    model=InteractionVelocity('B0').to(device);model.load_state_dict(saved['model_state_dict'],strict=True);model.eval();model.zero_grad(set_to_none=True)
    params,parameter_metadata=shared_parameters(model);before=tensor_digest(model)
    mean=torch.tensor(data['mean'],dtype=torch.float32,device=device);scale=torch.tensor(data['scale'],dtype=torch.float32,device=device)
    bbox=float(np.linalg.norm(np.ptp(target[:,:3],axis=0)));records=[];matching=[]
    for i in PROBE_IDS:
        # This is the unchanged PHYSICAL POSITION-ONLY Hungarian metric.
        source_x=noise[i,:,:3].astype(np.float64)*data['scale'][:3]+data['mean'][:3]
        cost=np.sum((source_x[:,None,:]-target[None,:,:3])**2,axis=-1)
        ordered,perm=linear_sum_assignment(cost)
        if not np.array_equal(ordered,np.arange(256)):raise AssertionError('Hungarian row ordering')
        chosen=cost[ordered,perm]
        binding=dict(probe_id=i,permutation=perm.tolist(),permutation_sha256=vector_digest(perm),
            cost_matrix_sha256=vector_digest(cost),selected_cost_sum=float(chosen.sum()),selected_cost_mean=float(chosen.mean()),
            field_contributions_selected_sum=dict(x=float(chosen.sum()),n=0.,F=0.),
            formula='sum((physical_source_xyz_i-physical_target_xyz_j)**2); raw normal and factor contributions exactly zero',
            entire_22_target_token_permuted=True,assignment_reused_for_all_six_times=True)
        matching.append(binding);assigned=target[perm]
        z1=torch.tensor(((assigned-data['mean'])/data['scale']).astype(np.float32),device=device)[None]
        z0=torch.tensor(noise[i:i+1],dtype=torch.float32,device=device);truth=z1-z0
        for t in PROBE_TIMES:
            row=dict(probe_id=i,time=t,calibration_sample_weight=1/24,matching_permutation_sha256=binding['permutation_sha256'],status='STARTED')
            try:
                z=(1-t)*z0+t*z1;velocity=model(z,torch.tensor([t],dtype=torch.float32,device=device))
                squared=(velocity-truth).square();losses=[squared[...,a:b].mean() for a,b in BLOCKS];old_total=squared.mean()
                if not bool(torch.isfinite(old_total)):raise FloatingPointError('Nonfinite probe loss')
                vectors=np.stack([_gradient(loss,params,True) for loss in losses]);gram=vectors@vectors.T
                old_actual=_gradient(old_total,params,coeff is not None)
                old_effect=_weighted_effect(OLD_WEIGHTS,vectors,gram,old_actual)
                loss_values=np.array([float(v.detach()) for v in losses]);old_sum=float(OLD_WEIGHTS@loss_values)
                old_effect.update(loss=float(old_total.detach()),weighted_block_loss_sum=old_sum,
                    loss_sum_absolute_error=abs(float(old_total.detach())-old_sum))
                old_effect['identity_passed']=bool(old_effect['identity_passed'] and old_effect['loss_sum_absolute_error']<=IDENTITY_RTOL*max(1.,abs(old_sum)))
                row.update(unweighted_losses=dict(zip(FIELD_NAMES,loss_values.tolist())),gram=gram.tolist(),
                    unweighted_gradient_norms=dict(zip(FIELD_NAMES,np.sqrt(np.diag(gram)).tolist())),old_weighted=old_effect)
                if coeff is not None:
                    new_total=sum(float(a)*loss for a,loss in zip(coeff,losses));new_actual=_gradient(new_total,params,False)
                    new_effect=_weighted_effect(coeff,vectors,gram,new_actual);new_sum=float(coeff@loss_values)
                    new_effect.update(loss=float(new_total.detach()),weighted_block_loss_sum=new_sum,loss_sum_absolute_error=abs(float(new_total.detach())-new_sum))
                    new_effect['identity_passed']=bool(new_effect['identity_passed'] and new_effect['loss_sum_absolute_error']<=IDENTITY_RTOL*max(1.,abs(new_sum)))
                    row['new_weighted']=new_effect
                physical=((z+(1-t)*velocity.detach())*scale+mean)[0].cpu().numpy().astype(np.float64)
                row['endpoint_semantic']=extended_semantic(physical,assigned,bbox)
                row['status']='COMPLETE' if old_effect['identity_passed'] and (coeff is None or row['new_weighted']['identity_passed']) else 'IDENTITY_FAILED'
            except Exception as exc:row.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
            records.append(row)
    after=tensor_digest(model);none=all(p.grad is None for p in model.parameters())
    complete=[r for r in records if r['status']=='COMPLETE'];all_ok=len(complete)==24 and before==after and none
    summary={}
    if len(complete)==24:
        grams=np.array([r['gram'] for r in records]);mean_gram=grams.mean(axis=0);g=np.sqrt(np.diag(mean_gram))
        summary=dict(mean_gram=mean_gram.tolist(),unweighted_gradient_rms=dict(zip(FIELD_NAMES,g.tolist())),
            old_total_gradient_rms=float(np.sqrt(np.mean([r['old_weighted']['gradient_norm']**2 for r in records]))),
            old_gram_total_gradient_rms=float(np.sqrt(OLD_WEIGHTS@mean_gram@OLD_WEIGHTS)),
            old_field_gradient_rms=dict(zip(FIELD_NAMES,(OLD_WEIGHTS*g).tolist())),
            unweighted_loss_mean={k:float(np.mean([r['unweighted_losses'][k] for r in records])) for k in FIELD_NAMES})
        if coeff is not None:
            summary.update(new_total_gradient_rms=float(np.sqrt(np.mean([r['new_weighted']['gradient_norm']**2 for r in records]))),
                new_gram_total_gradient_rms=float(np.sqrt(coeff@mean_gram@coeff)),new_field_gradient_rms=dict(zip(FIELD_NAMES,(coeff*g).tolist())))
    return dict(passed=all_ok,records=records,summary=summary,matching=matching,field_order=list(FIELD_NAMES),
        probe_ids=list(PROBE_IDS),times=list(PROBE_TIMES),requested_probes=24,completed_probes=len(complete),
        shared_parameters=parameter_metadata,shared_parameter_names=list(SHARED_PARAMETER_NAMES),shared_parameter_count=EXPECTED_SHARED_PARAMETERS,
        excluded_parameters=['decoder.4.weight','decoder.4.bias'],model_tensor_digest_before=before,model_tensor_digest_after=after,
        model_unchanged=before==after,parameter_grad_buffers_remain_none=none,optimizer_updates=0,mesh_extractions=0,
        checkpoint_sha256=sha(checkpoint),checkpoint_completed_steps=saved['completed_steps'],data_sha256=sha(root/'data.npz'),probe_noise_sha256=sha(root/'probe_noise.npz'),
        device=str(device),gradient_dtype='model float32 autograd; concatenation/dot products float64 on CPU',identity_rtol=IDENTITY_RTOL,
        seconds=time.perf_counter()-started,
        definitions=dict(loss='Unweighted field mean squared DIRECT velocity error over all256 elements and3/3/16 channels',
            Gram='3x3 dot products of unweighted shared-parameter gradients; order x,n,F; all cross terms retained',
            gradient_rms='sqrt(mean across24 equal-weight probes of squared field-gradient Euclidean norm)',
            velocity='raw B0 network output; equal to unchanged direct total_velocity, no transport',
            matching='physical xyz squared distance only; normal and F contribute zero, but whole22 target token is permuted',
            semantic='target-assisted linear-path endpoint extrapolation; not free generation',
            policy='autograd.grad only; no optimizer, no parameter.grad accumulation, no model/normal/Q modifications'))


def save_evaluation(out,evaluation,existing=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=existing)
    dump(out/'GRADIENT_ROWS.json',evaluation['records']);dump(out/'MATCHING.json',evaluation['matching'])
    good=[r for r in evaluation['records'] if 'gram' in r]
    np.savez_compressed(out/'GRAMS.npz',gram=np.array([r['gram'] for r in good],dtype=np.float64),
        probe_ids=np.array([r['probe_id'] for r in good],dtype=np.int64),times=np.array([r['time'] for r in good]),field_names=np.array(FIELD_NAMES))
    dump(out/'RESULT.json',{k:v for k,v in evaluation.items() if k not in ['records','matching']})


def calibrate(checkpoint,out,root=ROOT,device='cuda'):
    checkpoint=Path(checkpoint);out=Path(out);root=Path(root);out.mkdir(parents=True,exist_ok=False)
    definition=dict(version='SINGLE_STATE_METRIC0_CALIBRATION',probe_ids=list(PROBE_IDS),times=list(PROBE_TIMES),equal_probe_weight=1/24,
        shared_parameter_names=list(SHARED_PARAMETER_NAMES),shared_parameter_count=EXPECTED_SHARED_PARAMETERS,
        old_weights=dict(zip(FIELD_NAMES,OLD_WEIGHTS.tolist())),epsilon=EPS,min_gradient_rms_exclusive=MIN_GRADIENT_RMS,
        max_weight_ratio=MAX_WEIGHT_RATIO,common_scale_limits=list(COMMON_SCALE_LIMITS),clipping_allowed=False,
        checkpoint_sha256=sha(checkpoint),data_sha256=sha(root/'data.npz'),probe_noise_sha256=sha(root/'probe_noise.npz'),
        rule='g=sqrt(mean diagonal Gram); relative=1/(g+eps); c=sqrt((old_a^T meanGram old_a)/(relative^T meanGram relative)); weights=c*relative',
        source_files=[dict(path=str(p.resolve()),sha256=sha(p)) for p in [Path(__file__),Path(__file__).with_name('interaction_model.py'),Path(__file__).with_name('fm_model.py'),Path(__file__).with_name('interaction_semantics.py')]])
    dump(out/'CALIBRATION_PROTOCOL.json',definition);dump(out/'CALIBRATION_PROTOCOL_SHA256.json',dict(sha256=sha(out/'CALIBRATION_PROTOCOL.json')))
    weights_doc=dict(passed=False,status='CALIBRATION_FAILED',weights=None,**definition)
    started=time.perf_counter()
    try:
        first=evaluate_checkpoint(checkpoint,root,device);save_evaluation(out/'initial_unweighted',first)
        if first['checkpoint_completed_steps']!=3000:raise ValueError('Calibration must use the common3000-step checkpoint')
        if not first['passed']:raise ValueError('Initial gradient/model-integrity diagnostics failed')
        g=np.array([first['summary']['unweighted_gradient_rms'][k] for k in FIELD_NAMES]);gram=np.array(first['summary']['mean_gram'])
        weights_doc.update(unweighted_gradient_rms=dict(zip(FIELD_NAMES,g.tolist())),mean_gram=gram.tolist(),shared_parameters=first['shared_parameters'])
        if not np.isfinite(g).all() or np.any(g<=MIN_GRADIENT_RMS):raise ValueError('Nonfinite or near-zero field gradient RMS')
        relative=1/(g+EPS);ratio=float(relative.max()/relative.min());old_energy=float(OLD_WEIGHTS@gram@OLD_WEIGHTS);new_energy=float(relative@gram@relative)
        weights_doc.update(relative_weights=dict(zip(FIELD_NAMES,relative.tolist())),weight_ratio=ratio,old_gradient_squared_rms=old_energy,relative_gradient_squared_rms=new_energy)
        if not np.isfinite([ratio,old_energy,new_energy]).all() or old_energy<=0 or new_energy<=0:raise ValueError('Nonfinite/nonpositive total gradient energy')
        common=float(np.sqrt(old_energy/new_energy));weights_doc['common_scale']=common
        if ratio>MAX_WEIGHT_RATIO or not COMMON_SCALE_LIMITS[0]<=common<=COMMON_SCALE_LIMITS[1]:raise ValueError('Weight ratio/common scale outside frozen guards; no clipping')
        weights=common*relative
        if not np.isfinite(weights).all() or np.any(weights<=0):raise ValueError('Calibrated weights are not finite and positive')
        weight_dict=dict(zip(FIELD_NAMES,weights.tolist()))
        verified=evaluate_checkpoint(checkpoint,root,device,weights=weight_dict)
        save_evaluation(out/'weighted_verification',verified)
        if not verified['passed']:raise ValueError('Weighted gradient identity verification failed')
        if first['model_tensor_digest_before']!=verified['model_tensor_digest_before']:raise ValueError('Common model changed between calibration passes')
        if not np.array_equal(gram,np.array(verified['summary']['mean_gram'])):raise ValueError('Same model/probes produced different Gram in verification')
        old_rms=verified['summary']['old_total_gradient_rms'];new_rms=verified['summary']['new_total_gradient_rms'];scale_error=abs(new_rms-old_rms)/old_rms
        if scale_error>IDENTITY_RTOL:raise ValueError('Common scale did not match independently differentiated total gradient RMS')
        dump(out/'GRADIENT_ROWS.json',verified['records']);dump(out/'MATCHING.json',verified['matching'])
        np.savez_compressed(out/'GRAMS.npz',gram=np.array([r['gram'] for r in verified['records']]),probe_ids=np.array([r['probe_id'] for r in verified['records']]),
            times=np.array([r['time'] for r in verified['records']]),field_names=np.array(FIELD_NAMES))
        weights_doc.update(passed=True,status='PASS',weights=weight_dict,old_total_gradient_rms=old_rms,new_total_gradient_rms=new_rms,
            common_scale_relative_error=scale_error,model_tensor_digest_before=first['model_tensor_digest_before'],model_tensor_digest_after=verified['model_tensor_digest_after'],
            parameter_grad_buffers_remain_none=first['parameter_grad_buffers_remain_none'] and verified['parameter_grad_buffers_remain_none'],
            optimizer_updates=0,mesh_extractions=0)
    except Exception as exc:weights_doc.update(error=repr(exc),traceback=traceback.format_exc())
    weights_doc['seconds']=time.perf_counter()-started
    dump(out/'WEIGHTS.json',weights_doc)
    dump(out/'RESULT.json',dict(passed=weights_doc['passed'],status=weights_doc['status'],weights_sha256=sha(out/'WEIGHTS.json'),
        weights=weights_doc['weights'],seconds=weights_doc['seconds'],optimizer_updates=0,mesh_extractions=0,error=weights_doc.get('error')))
    print(json.dumps({k:weights_doc.get(k) for k in ['passed','status','weights','unweighted_gradient_rms','common_scale','old_total_gradient_rms','new_total_gradient_rms','seconds','error']},indent=2),flush=True)
    return weights_doc


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--device',default='cuda');a=p.parse_args()
    result=calibrate(a.checkpoint,a.out,a.root,a.device);raise SystemExit(0 if result['passed'] else 1)