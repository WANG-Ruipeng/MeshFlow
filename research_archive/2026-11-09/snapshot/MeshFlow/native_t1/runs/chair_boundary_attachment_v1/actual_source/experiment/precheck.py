"""Bounded 44-forward/16-zero-update-backward gates; caller holds GPU lock.

Import and cpu_contract_checks never initialize CUDA. No optimizer step exists
in this module. Calibration gradients are detached-output leaf gradients.
"""
from __future__ import annotations
import gc
import json
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from native_t1.model import native_execution
from native_t1.data import model_inputs
from native_t1.training import masked_fm_loss
from experiments.mf_h1_local0.control_mvp48.train_stream import collate
from experiments.mf_h1_local0.control_mvp48.train_io import tree_hash,capture_rng,restore_rng
from .common import *
from . import checkpoint
from .ledger import Budget
from .method import seam_loss
from .stream import AttachmentStream
from .training import sample_loss,call_sync,validate_optimizer


def comparison(actual, reference):
    a=np.asarray(actual);b=np.asarray(reference)
    bitwise=a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    a=a.astype(np.float64);b=b.astype(np.float64);d=a-b
    maximum=float(np.abs(d).max());rms=float(np.sqrt(np.mean(d*d)))
    einf=maximum/max(1.,float(np.abs(b).max()));erms=rms/max(1.,float(np.sqrt(np.mean(b*b))))
    finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
    return dict(bitwise_equal=bitwise,max_abs=maximum,rms=rms,e_inf=einf,e_rms=erms,
        finite=finite,normalized_pass=bool(finite and einf<=5e-5 and erms<=1e-5))


def at_time(sample,t):
    result=dict(sample);result['t']=np.asarray(t,np.float32)
    result['xt']=(np.float32(1)-result['t'])*sample['x0']+result['t']*sample['x1']
    result['xt'][sample['known_mask']]=sample['context'][sample['known_mask']]
    return result


def frozen_registration(samples):
    if len(samples)!=8:raise ValueError('Calibration needs exactly first eight fixed samples')
    Ns=[int(s['y']) for s in samples];median=float(np.median(Ns))
    ordinary=min(range(8),key=lambda i:(abs(Ns[i]-median),samples[i]['object_id'],samples[i]['task_id'],i))
    large=max(range(8),key=lambda i:(Ns[i],samples[i]['object_id'],samples[i]['task_id'],i))
    D=sum(s['free_coordinate_count'] for s in samples)
    return dict(schema='attachment_precheck_registration_v1',planned_model_forwards=44,
        planned_network_backwards=16,optimizer_updates=0,
        forward_breakdown=dict(calibration=16,cost=16,zero_equivalence=6,nonzero_permutation=4,padding=2),
        first_batch_D=D,first_batch=[dict(sample_index=s['sample_index'],uid=s['object_id'],task_id=s['task_id'],N=int(s['y']),K=s['K'],alpha=s['alpha'],shared_input_sha256=s['shared_input_sha256'],coupled_input_sha256=s['coupled_input_sha256'],target_sha256=s['_seam_targets']['target_sha256']) for s in samples],
        calibration_times=[.5,.8],cost_sample_indices=dict(ordinary=ordinary,large=large),
        cost_selection='first registered batch; N nearest numeric median, then uid/task/index; largest N, then uid/task/index',
        cost_times=[.5,.8],cost_repeats='Two fixed same-task states (.5,.8), identical across arms; no uncounted warmup',
        zero_equivalence_sample_index=ordinary,zero_equivalence_time=.5,
        nonzero_vector='linspace(-0.02,0.02,768),FP32; temporary and restored exactly',
        permutation='reverse face order; corner permutation cycles through all six permutations',
        padding='ordinary t=.5,7 extra faces; zeros versus linspace(-4,4,63),same shape; BF16',
        FP32_permutation=dict(normalization='maxabs/max(1,maxabs(reference)); rms/max(1,rms(reference))',e_inf_max=5e-5,e_rms_max=1e-5),
        gradient_check=dict(method='velocity hook total-FM compared with independent output-leaf seam gradient',rtol=2e-4,atol=1e-9),
        cost_scope='Synchronized forward+loss and backward only; detached-leaf verification excluded; includes loaded model and optimizer in memory peaks',
        no_model_parameter_update=True)


def _leaf_gradients(v,batch,sample,D):
    leaf=v.detach().clone().requires_grad_(True)
    fm=masked_fm_loss(leaf,batch['u'],batch['valid_mask'],batch['known_mask'],denominator=D)
    e,diagnostic=seam_loss(leaf,batch['xt'],batch['t'],sample)
    gfm=torch.autograd.grad(fm,leaf)[0]
    gseam=torch.autograd.grad(e/8,leaf)[0]
    mask=batch['loss_mask'][...,None].expand_as(leaf)
    fm_norm=float(gfm[mask].norm());seam_norm=float(gseam[mask].norm())
    return gfm,gseam,dict(FM=float(fm.detach()),E_seam=float(e.detach()),g_FM_L2=fm_norm,g_seam_div8_L2=seam_norm,
        finite=bool(np.isfinite([fm_norm,seam_norm]).all()),target_status=diagnostic['status'],gate=diagnostic['gate'],output_leaf_gradient_calls=2)


def _calibration_summary(rows):
    selected=[r for r in rows if r['valid']]
    parents=sorted({r['uid'] for r in selected})
    lam=.25*float(np.median([r['ratio'] for r in selected])) if selected else None
    valid=len(selected)>=8 and len(parents)>=4 and lam is not None and np.isfinite(lam) and lam>0
    return dict(status='PASS' if valid else 'CALIBRATION_BLOCKED',lambda_seam=lam if valid else None,
        valid_states=len(selected),valid_parents=len(parents),parents=parents,required_states=8,required_parents=4,
        formula='0.25*median(||grad_v FM_i/D_batch|| / ||grad_v E_seam_i/8||)',
        rows=rows,model_parameter_updates=0,output_leaf_gradient_calls=2*len(rows),
        not_network_parameter_gradient_ratio=True,no_validation_selection=True)


def _save_arrays(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:np.savez_compressed(f,**arrays)
    return dict(path=str(path),sha256=digest(path))


def _forward(model,batch,bf16,budget,context):
    with torch.no_grad(),native_execution(model,coordinates=(batch['xt'],batch['t']),autocast=bf16):
        v,seconds=call_sync(budget,'precheck_forward',lambda:model(*model_inputs(batch)),context)
    if not bool(torch.isfinite(v).all()):raise FloatingPointError('Nonfinite precheck output')
    return v.detach().float(),seconds


def _permuted(batch):
    import itertools
    n=batch['xt'].shape[1];device=batch['xt'].device
    order=torch.arange(n-1,-1,-1,device=device)
    corners=torch.tensor(list(itertools.permutations(range(3))),device=device,dtype=torch.long)[torch.arange(n,device=device)%6]
    transformed={k:v.clone() for k,v in batch.items()}
    for k in ('xt','x0','x1','u','context'):
        v=batch[k][:,order].reshape(1,n,3,3)
        transformed[k]=v.gather(2,corners[None,:,:,None].expand(1,n,3,3)).reshape(1,n,9)
    for k in ('known_mask','valid_mask','loss_mask'):transformed[k]=batch[k][:,order]
    return transformed,order,corners


def _mapped(array,order,corners):
    a=torch.as_tensor(array,device=order.device).reshape(1,len(order),3,3)[:,order]
    return a.gather(2,corners[None,:,:,None].expand(1,len(order),3,3)).reshape(1,len(order),9).cpu().numpy()


def run_precheck(out=OUT,budget=None):
    """Real precheck entrypoint, called by root worker only after GPU lock."""
    out=Path(out);folder=out/'precheck';folder.mkdir(parents=True,exist_ok=True)
    completed=folder/'results.json'
    if completed.exists():return read_json(completed)
    marker=folder/'started.json'
    if marker.exists():raise RuntimeError('Prior precheck incomplete; explicit recorded engineering recovery required, no automatic replay')
    budget=Budget(out,'precheck') if budget is None else budget
    start_counts=dict(budget.attempts);start_back=dict(budget.returns)
    stream=AttachmentStream(SOURCE,CHAIR,out,'A_FM');samples=stream.next_effective_batch()
    registration=frozen_registration(samples);freeze_json(folder/'registration.json',registration)
    atomic_json(marker,dict(status='STARTED',registration_sha256=digest(folder/'registration.json'),source_sha256=digest(__file__)))
    D=registration['first_batch_D'];ordinary=registration['cost_sample_indices']['ordinary'];large=registration['cost_sample_indices']['large']
    zero_sample=at_time(samples[ordinary],.5)
    rows=[];costs=[];zero_outputs={};arm_status={a:'PASS' for a in ARMS};reasons={a:[] for a in ARMS}
    calibration=None;calrows=[];leaf_calls=0;fatal=None;started=time.perf_counter();model=None;opt=None
    def block(arms,reason):
        for a in arms:
            arm_status[a]='IMPLEMENTATION_BLOCKED';reasons[a].append(reason)
    def persist():
        atomic_json(folder/'progress.json',dict(rows=rows,costs=costs,arm_status=arm_status,reasons=reasons,calibration=calibration,leaf_gradient_calls=leaf_calls))
    try:
        for arm in ARMS:
            print(dict(event='PRECHECK_ARM_BEGIN',arm=arm),flush=True)
            model,opt,meta,_=checkpoint.load(arm=arm,device='cuda',trainable=True,optimizer=True,out=out)
            validate_optimizer(model,opt,arm,0)
            initial=dict(model=checkpoint.tensor_state_hash(model.state_dict()),optimizer=tree_hash(opt.state_dict()))
            rng=capture_rng();device=next(model.parameters()).device
            # No optimization. A is exact START; B/D zero vectors are checked
            # against the same retained FP32/BF16 arrays under identical inputs.
            if arm in ('A_FM','B_BOUNDARY','D_BOUNDARY_SEAM'):
                b=collate([zero_sample],device)
                for bf16 in (False,True):
                    mode='BF16' if bf16 else 'FP32';v,sec=_forward(model,b,bf16,budget,dict(arm=arm,check='zero_equivalence',precision=mode))
                    arr=v.cpu().numpy();artifact=_save_arrays(folder/'arrays'/f'{arm}_zero_{mode}.npz',velocity=arr)
                    if arm=='A_FM':zero_outputs[mode]=arr;result=dict(bitwise_equal=True,finite=True)
                    else:result=comparison(arr,zero_outputs[mode])
                    rows.append(dict(check='zero_equivalence',arm=arm,precision=mode,seconds=sec,comparison=result,artifact=artifact))
                    if not result['bitwise_equal']:block(BOUNDARY_ARMS,'zero_initialization_'+mode+'_not_bitwise')
                del b,v
            if arm=='A_FM':
                for i,s in enumerate(samples):
                    for t in (.5,.8):
                        probe=at_time(s,t);b=collate([probe],device)
                        v,sec=_forward(model,b,True,budget,dict(arm=arm,check='lambda_calibration',sample_index=s['sample_index'],t=t))
                        gfm,gseam,details=_leaf_gradients(v,b,probe,D);leaf_calls+=2
                        valid=details['finite'] and details['g_FM_L2']>0 and details['g_seam_div8_L2']>0
                        ratio=details['g_FM_L2']/details['g_seam_div8_L2'] if valid else None
                        row=dict(sample_index=s['sample_index'],uid=s['object_id'],task_id=s['task_id'],alpha=s['alpha'],t=t,
                            D_batch=D,valid=valid,ratio=ratio,seconds=sec,actual_xt_sha256=array_hash(b['xt'].cpu().numpy()),
                            common_input_sha256=s['shared_input_sha256'],target_sha256=s['_seam_targets']['target_sha256'],**details)
                        calrows.append(row);atomic_json(folder/'lambda_progress.json',dict(rows=calrows,model_forwards=len(calrows),leaf_gradient_calls=2*len(calrows)))
                        del b,v,gfm,gseam
                calibration=_calibration_summary(calrows)
                calibration.update(code_sha256=digest(__file__),registration_sha256=digest(folder/'registration.json'),D_batch=D,
                    checkpoint_sha256=START_SHA,forward_count=16,real_network_backwards=0)
                freeze_json(out/'lambda_calibration.json',calibration)
                if calibration['status']!='PASS':
                    for a in SEAM_ARMS:arm_status[a]='CALIBRATION_BLOCKED';reasons[a].append('Insufficient finite positive fixed calibration states')
                print(dict(event='CALIBRATION_DONE',status=calibration['status'],lambda_seam=calibration['lambda_seam']),flush=True)
            lam=calibration['lambda_seam'] if calibration and calibration['status']=='PASS' else 0.
            if arm not in SEAM_ARMS or calibration['status']=='PASS':
                for name,index in (('ordinary',ordinary),('large',large)):
                    for repeat,t in enumerate((.5,.8)):
                        probe=at_time(samples[index],t);opt.zero_grad(set_to_none=True)
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
                        context=dict(arm=arm,check='cost_and_gradient_wiring',fixture=name,repeat=repeat,t=t,sample_index=probe['sample_index'])
                        values=sample_loss(model,probe,D,arm,lam,budget,'precheck_forward',context,return_details=True)
                        loss,fm,seam,aux,diag,fsec,extra=values;v=extra['velocity'];b=extra['batch']
                        torch.cuda.synchronize();forward_loss_seconds=time.perf_counter()-tick
                        tick=time.perf_counter();gfm,gseam,gd=_leaf_gradients(v,b,probe,D);leaf_calls+=2
                        expected_aux=gseam*(lam if arm in SEAM_ARMS else 0.)
                        torch.cuda.synchronize();diagnostic_seconds=time.perf_counter()-tick
                        captured=[];handle=v.register_hook(lambda gradient:captured.append(gradient.detach().clone()))
                        with native_execution(model,coordinates=(b['xt'],b['t']),autocast=True):
                            _,bsec=call_sync(budget,'precheck_backward',loss.backward,context)
                        handle.remove()
                        if len(captured)!=1:raise RuntimeError('Velocity hook not called exactly once')
                        observed_aux=captured[0]-gfm
                        error=(observed_aux-expected_aux).abs();agreement=bool(torch.allclose(observed_aux,expected_aux,rtol=2e-4,atol=1e-9))
                        active_expected=float(expected_aux.norm())>0 if arm in SEAM_ARMS else True
                        grads={n:p.grad for n,p in model.named_parameters()}
                        unused=[n for n,g in grads.items() if g is None]
                        finite=all(g is not None and bool(torch.isfinite(g).all()) for g in grads.values())
                        backbone_sq=sum(float(g.detach().double().square().sum()) for n,g in grads.items() if n.startswith('backbone.') and g is not None)
                        boundary_norm=float(model.e_boundary.grad.norm()) if arm in BOUNDARY_ARMS and model.e_boundary.grad is not None else None
                        attached=not unused and finite and backbone_sq>0 and (arm not in BOUNDARY_ARMS or boundary_norm is not None and boundary_norm>0)
                        item=dict(context,forward_seconds=fsec,forward_and_loss_seconds=forward_loss_seconds,backward_seconds=bsec,
                            complete_forward_backward_seconds=forward_loss_seconds+bsec,leaf_verification_seconds=diagnostic_seconds,
                            FM=float(fm.detach()),E_seam=float(seam.detach()),weighted_seam=float(aux.detach()),
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                            peak_scope='one F+loss+leaf-verification+B after reset; includes resident FP32 model and full restored AdamW; allocated is within reserved',
                            unused_parameters=unused,all_gradients_finite=finite,backbone_gradient_L2=backbone_sq**.5,boundary_gradient_L2=boundary_norm,
                            observed_aux_gradient_L2=float(observed_aux.norm()),expected_aux_gradient_L2=float(expected_aux.norm()),
                            auxiliary_gradient_max_abs_error=float(error.max()),gradient_decomposition_PASS=agreement,
                            active_seam_gradient=active_expected,parameter_gradient_wiring_PASS=attached)
                        costs.append(item)
                        _save_arrays(folder/'arrays'/f'{arm}_{name}_{repeat}_gradients.npz',actual_total=captured[0].cpu().numpy(),FM=gfm.cpu().numpy(),expected_aux=expected_aux.cpu().numpy())
                        if not agreement or not attached or not active_expected:block((arm,),'cost_backward_gradient_wiring')
                        persist();opt.zero_grad(set_to_none=True)
                        del values,loss,fm,seam,aux,extra,v,b,gfm,gseam,expected_aux,observed_aux,error,captured,grads
            if arm=='B_BOUNDARY':
                # Temporary fixed nonzero vector; zero restoration is checked by
                # exact full-state identity before releasing this model.
                with torch.no_grad():model.e_boundary.copy_(torch.linspace(-.02,.02,768,device=device,dtype=torch.float32))
                b=collate([zero_sample],device);permuted,order,corners=_permuted(b)
                for bf16 in (False,True):
                    mode='BF16' if bf16 else 'FP32'
                    baseline,_=_forward(model,b,bf16,budget,dict(arm=arm,check='nonzero_original',precision=mode))
                    changed,_=_forward(model,permuted,bf16,budget,dict(arm=arm,check='nonzero_permuted',precision=mode))
                    expected=_mapped(baseline.cpu().numpy(),order,corners);actual=changed.cpu().numpy();cmp=comparison(actual,expected)
                    rows.append(dict(check='nonzero_permutation',arm=arm,precision=mode,comparison=cmp,strict_bitwise_claim=False))
                    _save_arrays(folder/'arrays'/f'nonzero_permutation_{mode}.npz',actual=actual,expected=expected)
                    if not bf16 and not cmp['normalized_pass']:block(BOUNDARY_ARMS,'FP32_nonzero_permutation')
                n=int(zero_sample['y']);padded=collate([zero_sample],device,pad_to=n+7)
                baseline,_=_forward(model,padded,True,budget,dict(arm=arm,check='padding_zero',precision='BF16'))
                padded['xt'][:,n:]=torch.linspace(-4,4,63,device=device).reshape(1,7,9)
                changed,_=_forward(model,padded,True,budget,dict(arm=arm,check='padding_changed',precision='BF16'))
                cmp=comparison(changed[:,:n].cpu().numpy(),baseline[:,:n].cpu().numpy())
                rows.append(dict(check='same_shape_padding_independence',arm=arm,precision='BF16',comparison=cmp))
                _save_arrays(folder/'arrays'/'padding_BF16.npz',actual=changed[:,:n].cpu().numpy(),expected=baseline[:,:n].cpu().numpy())
                if not cmp['bitwise_equal']:block(BOUNDARY_ARMS,'padding_influences_valid_output')
                with torch.no_grad():model.e_boundary.zero_()
                del b,permuted,baseline,changed,padded
            opt.zero_grad(set_to_none=True);restore_rng(rng)
            after=dict(model=checkpoint.tensor_state_hash(model.state_dict()),optimizer=tree_hash(opt.state_dict()))
            unchanged=initial==after and tree_hash(capture_rng())==tree_hash(rng)
            rows.append(dict(check='no_model_optimizer_RNG_mutation',arm=arm,status='PASS' if unchanged else 'FAIL',before=initial,after=after))
            if not unchanged:block((arm,),'precheck_state_mutation')
            del model,opt;model=opt=None;gc.collect();torch.cuda.empty_cache();persist()
        reference={(r['fixture'],r['repeat']):r['complete_forward_backward_seconds'] for r in costs if r['arm']=='A_FM'}
        for row in costs:
            row['ratio_to_A_same_fixture_and_t']=row['complete_forward_backward_seconds']/reference[(row['fixture'],row['repeat'])]
            if row['arm']!='A_FM' and row['ratio_to_A_same_fixture_and_t']>2:block((row['arm'],),'complete_F_plus_B_exceeds_2x_A')
    except BaseException as exc:
        fatal=dict(error=repr(exc),traceback=traceback.format_exc())
        for arm in ARMS:
            if not any(r.get('check')=='no_model_optimizer_RNG_mutation' and r.get('arm')==arm for r in rows):block((arm,),'precheck_incomplete_after_actual_failure')
    finally:
        if opt is not None:opt.zero_grad(set_to_none=True)
        del model,opt;gc.collect()
    actualF=budget.attempts['precheck_forward']-start_counts.get('precheck_forward',0)
    actualB=budget.attempts['precheck_backward']-start_counts.get('precheck_backward',0)
    result=dict(status='PASS' if any(v=='PASS' for v in arm_status.values()) else 'IMPLEMENTATION_BLOCKED',
        arm_status=arm_status,arm_blocking_reasons=reasons,full_factorial_gate_pass=all(v=='PASS' for v in arm_status.values()),
        calibration_status=calibration['status'] if calibration else 'NOT_EVALUATED',lambda_seam=calibration['lambda_seam'] if calibration else None,
        actual_model_forward_attempts=actualF,actual_network_backward_attempts=actualB,
        actual_model_forward_returns=budget.returns['precheck_forward']-start_back.get('precheck_forward',0),
        actual_network_backward_returns=budget.returns['precheck_backward']-start_back.get('precheck_backward',0),
        output_leaf_gradient_calls=leaf_calls,optimizer_updates=0,rows=rows,costs=costs,fatal=fatal,
        scientific_configuration_changed=False,source_sha256=digest(__file__),registration_sha256=digest(folder/'registration.json'),wall_seconds=time.perf_counter()-started)
    if actualF>44 or actualB>16:raise RuntimeError('Internal fixed precheck schedule exceeded')
    atomic_json(completed,result);budget.snapshot();return result


def cpu_contract_checks(out=OUT):
    """Synthetic CPU leaf checks and static budget/registration checks only."""
    from .method_tests import fixture
    import ast,inspect
    torch.set_num_threads(2)
    s=fixture();s.update(t=np.asarray(.8,np.float32),x0=np.zeros_like(s['x1']),u=s['x1'].copy(),
        xt=s['x1']*.8,object_id='synthetic',task_id='synthetic',alpha=1.,sample_index=0,
        free_coordinate_count=9,shared_input_sha256='synthetic',coupled_input_sha256='synthetic')
    s['xt'][s['known_mask']]=s['context'][s['known_mask']]
    b=collate([s]);v=torch.zeros_like(b['xt']);gfm,gs,d=_leaf_gradients(v,b,s,72)
    e,diag=seam_loss(v.clone().requires_grad_(True),b['xt'],b['t'],s)
    checks=dict(leaf_gradients_finite=d['finite'],known_leaf_gradients_zero=not bool(gfm[:,0].any() or gs[:,0].any()),
        seam_leaf_nonzero=float(gs.norm())>0,leaf_is_not_model_backward=True,
        normalization_contract=comparison(np.ones((2,3))+1e-6,np.ones((2,3)))['normalized_pass'],
        CUDA_not_initialized=not torch.cuda.is_initialized())
    cases=[dict(valid=True,uid=str(i//2),ratio=float(i+1)) for i in range(16)]
    c=_calibration_summary(cases);checks['median_calibration']=c['status']=='PASS' and c['lambda_seam']==2.125 and c['valid_parents']==8
    checks['no_silent_calibration_replacement']=_calibration_summary(cases[:6])['status']=='CALIBRATION_BLOCKED'
    text=inspect.getsource(run_precheck);tree=ast.parse(text)
    checks['no_optimizer_step']=not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='step' for n in ast.walk(tree))
    signature=inspect.signature(sample_loss);checks['optional_return_details_backward_compatible']='return_details' in signature.parameters and signature.parameters['return_details'].default is False
    result=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,CPU_output_leaf_gradient_calls=2,
        real_model_forwards=0,real_model_backwards=0,optimizer_updates=0,source_sha256=digest(__file__))
    atomic_json(Path(out)/'tests/precheck_cpu.json',result)
    if result['status']!='PASS':raise AssertionError(result)
    return result
