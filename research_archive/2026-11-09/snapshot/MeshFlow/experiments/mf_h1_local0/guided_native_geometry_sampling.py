"""Actual frozen-T1 predictor/corrector sampling; no GT or training inputs.

The FP32 Euler arithmetic and one BF16 native_execution context match the
archived native_patch_run.clamped_sample. Geometry is strictly CPU-only and
never invokes a model. A disabled corrector leaves the GPU Euler state intact.
"""
from __future__ import annotations
from .stable_runtime import configure_stable_runtime
from collections import Counter
import hashlib
import math
from pathlib import Path
import re
import time
import numpy as np
import torch
from .native_inpainting_model import native_execution,assert_native_runtime
from .guided_native_geometry_corrector import correct_geometry,prepare_boundary
from .guided_native_geometry_geometry import verified_pair,initial_geometry_ledger

CATEGORIES=('baseline_replays','development_guided_generations','confirmation_generations')


def _hash(array):
    value=np.ascontiguousarray(array)
    return hashlib.sha256(str((str(value.dtype),tuple(value.shape))).encode()+value.tobytes()).hexdigest()


def _file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8<<20),b''):digest.update(part)
    return digest.hexdigest()


def _add_counts(destination,source):
    for key,value in source.items():
        if isinstance(value,(int,float)) and not isinstance(value,bool):
            destination[key]=destination.get(key,0)+value


def _freeze_guard(model):
    if model.training or getattr(model,'experiment_trainable',True):raise RuntimeError('Sampling requires read-only native T1')
    if any(p.requires_grad or p.grad is not None or p.dtype!=torch.float32 for p in model.parameters()):
        raise RuntimeError('Model parameters must stay FP32, frozen, without gradients')
    if torch.is_grad_enabled():raise RuntimeError('Sampling must not create a model gradient graph')


def _areas(triangles):
    triangles=np.asarray(triangles,dtype=np.float64)
    return np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=-1)*.5


def _observe_CF(C,free,L):
    """Current C/free observation only; no contact cache or acceptance role."""
    start=time.perf_counter();ledger=initial_geometry_ledger();counts=Counter();pairs={}
    C=np.asarray(C,dtype=np.float64);free=np.asarray(free,dtype=np.float64).reshape(-1,3,3)
    cmn=C.min(1);cmx=C.max(1);fmn=free.min(1);fmx=free.max(1)
    cdeg=_areas(C)<=1e-11*L*L;fdeg=_areas(free)<=1e-11*L*L;eps=1e-9*L
    for fi in range(len(free)):
        candidates=np.flatnonzero((np.all(cmx>=fmn[fi]-eps,axis=1)&np.all(fmx[fi]>=cmn-eps,axis=1))|cdeg|fdeg[fi])
        ledger['observation_AABB_pair_checks']=ledger.get('observation_AABB_pair_checks',0)+len(C)
        ledger['observation_broadphase_pruned']=ledger.get('observation_broadphase_pruned',0)+len(C)-len(candidates)
        for ci in candidates:
            result=verified_pair(C[ci],free[fi],L,ledger)
            kind=result['classification'] if result.get('verified') and result.get('resolved') and result.get('illegal') is not None else 'UNRESOLVED'
            if kind!='disjoint':pairs[int(ci),fi]=kind;counts[kind]+=1
    proper=[list(p) for p,k in sorted(pairs.items()) if k=='proper_transverse_penetration']
    overlap=[list(p) for p,k in sorted(pairs.items()) if k=='coplanar_positive_area_overlap']
    unresolved=[list(p) for p,k in sorted(pairs.items()) if k=='UNRESOLVED']
    return dict(classification_counts=dict(counts),proper_pairs=proper,coplanar_overlap_pairs=overlap,
        unresolved_pairs=unresolved,illegal_pair_count=len(proper)+len(overlap)),ledger,time.perf_counter()-start


def _defect_delta(before,after,free_free=False):
    result={}
    for name in ('proper_pairs','coplanar_overlap_pairs','unresolved_pairs'):
        previous={tuple(p) for p in before[name]};current={tuple(p) for p in after[name]}
        result[name]=dict(before=len(previous),after=len(current),delta=len(current)-len(previous),
            created=[list(p) for p in sorted(current-previous)],removed=[list(p) for p in sorted(previous-current)])
    previous={i for p in before['proper_pairs'] for i in (p if free_free else [p[1]])}
    current={i for p in after['proper_pairs'] for i in (p if free_free else [p[1]])}
    result['proper_affected_free_faces']=dict(before=len(previous),after=len(current),delta=len(current)-len(previous),
        newly_affected=sorted(current-previous),no_longer_affected=sorted(previous-current))
    return result


def _active_mode(mode,k):
    if mode in ('B','off'):return None
    if mode=='R':return 'R' if k>=15 else None
    if mode=='S':return 'S' if k>=40 else None
    if mode=='RS':return 'RS' if k>=40 else 'R' if k>=15 else None
    raise ValueError('Only B/off/R/S/RS are registered')


def _save_arrays(destination,path,predictors):
    if destination.exists():raise FileExistsError(destination)
    actual_path=np.stack(path).astype(np.float32,copy=False)
    predictions=np.stack(predictors).astype(np.float32,copy=False) if predictors else np.empty((0,112,9),np.float32)
    with destination.open('xb') as stream:
        np.savez_compressed(stream,output=actual_path[-1],path=actual_path,predictor=predictions)
    return dict(raw_path=str(destination),raw_sha256=_file_hash(destination),raw_bytes=destination.stat().st_size,
        output_sha256=_hash(actual_path[-1]),path_sha256=_hash(actual_path),predictor_sha256=_hash(predictions),
        saved_states=len(actual_path),saved_predictors=len(predictions))


@torch.no_grad()
def sample_frozen(model,noise,C,L,mode,counts,output,label,category):
    """Exactly50 frozen native forwards; ``noise`` is pristine [1,112,9] FP32.

    ``counts`` is the root-owned ledger with authoritative model forward hooks.
    ``category`` is one of CATEGORIES. This function returns a JSON-safe record
    and saves one NPZ containing51 corrected states and50 Euler predictors.
    Every result is an independent clone of the supplied noise. No GT/bank API.
    """
    noise=np.asarray(noise);C=np.asarray(C)
    if noise.dtype!=np.float32 or noise.shape!=(1,112,9) or C.dtype!=np.float32 or C.shape!=(12,3,3):
        raise ValueError('Require exact FP32 noise[1,112,9] and C[12,3,3]')
    if not np.isfinite(noise).all() or not np.isfinite(C).all() or not math.isfinite(L) or L<=0:
        raise ValueError('Require finite geometry and positive declared L')
    if mode not in ('B','off','R','S','RS') or category not in CATEGORIES:raise ValueError('Unregistered sampling mode/category')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',label):raise ValueError('Label must be a plain local filename stem')
    output=Path(output);raw=output/'raw';raw.mkdir(parents=True,exist_ok=True);destination=raw/(label+'.npz')
    if destination.exists():raise FileExistsError(destination)
    if counts['complete_generation_attempts']>=42:raise RuntimeError('Hard42 generation-attempt budget')
    _freeze_guard(model);configure_stable_runtime()
    device=next(model.parameters()).device
    if device.type!='cuda':raise RuntimeError('Production frozen native sampling requires CUDA')
    source_hash=_hash(noise);C_hash=_hash(C);original_noise=noise.copy();original_C=C.copy()
    counts['complete_generation_attempts']+=1
    counts.setdefault('generation_attempts_by_category',{})[category]=counts.get('generation_attempts_by_category',{}).get(category,0)+1
    before_fw=counts['model_forward_attempts'];before_returns=counts['model_forward_returns']
    correction_counts={};observation_counts={};step_audits=[];states=[];predictors=[]
    wall_start=time.perf_counter();model_seconds=transfer_seconds=geometry_seconds=observation_seconds=0.
    cumulative_accepted_length=0.;corrector_attempts=corrector_returns=0;R_calls=S_calls=0
    result=dict(status='RUNNING',label=label,key=label,mode=mode,category=category,N_total=112,K=12,N_free=100,
        noise_sha256=source_hash,condition_sha256=C_hash,bbox_diagonal=float(L),step_audits=step_audits,
        model_backwards=0,optimizer_updates=0,model_parameter_updates=0,
        CONTINUOUS_COLLISION_SAFETY='NOT_RUN',topology_welding='NOT_RUN')
    try:
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(device)
        stamp=time.perf_counter()
        z=torch.from_numpy(original_noise).to(device=device,dtype=torch.float32)
        known_C=torch.from_numpy(original_C.reshape(12,9)).to(device=device,dtype=torch.float32)
        state=z.clone();state[:,:12]=known_C[None]
        known=torch.zeros((1,112),device=device,dtype=torch.bool);known[:,:12]=True
        valid=torch.ones_like(known);y=torch.tensor([112],device=device)
        torch.cuda.synchronize();transfer_seconds+=time.perf_counter()-stamp
        stamp=time.perf_counter();initial=state[0].cpu().numpy().copy();transfer_seconds+=time.perf_counter()-stamp
        if not np.array_equal(initial[:12],original_C.reshape(12,9)) or not np.array_equal(initial[12:],original_noise[0,12:]):
            raise RuntimeError('Initial GPU C/Gaussian writeback differs')
        states.append(initial)
        boundary=prepare_boundary(original_C,float(L))
        previous_cf,observed,elapsed=_observe_CF(original_C,initial[12:],float(L))
        _add_counts(observation_counts,observed);observation_seconds+=elapsed
        result['initial_C_free']=previous_cf;previous_ff=None
        with native_execution(model,coordinates=state) as runtime:
            for k in range(50):
                _freeze_guard(model)
                if not torch.equal(state[0,:12],known_C):raise RuntimeError('C changed before native forward')
                t=torch.full((1,),k/50,device=device,dtype=torch.float32)
                assert_native_runtime(model,coordinates=(state,t))
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                begin.record();velocity=model(state,t,y,valid,known).float();end.record();end.synchronize()
                cuda_seconds=float(begin.elapsed_time(end)/1000);model_seconds+=cuda_seconds
                counts['model_cuda_seconds']+=cuda_seconds
                # This operation order is identical to the original sampler.
                predicted=state.clone()
                predicted[:,12:]=state[:,12:]+velocity[:,12:]*(1/50)
                predicted[:,:12]=known_C[None]
                counts['euler_updates']+=1
                stamp=time.perf_counter();prediction=predicted[0].cpu().numpy().copy()
                transferred=time.perf_counter()-stamp;transfer_seconds+=transferred
                if prediction.dtype!=np.float32 or not np.isfinite(prediction).all() or not np.array_equal(prediction[:12],original_C.reshape(12,9)):
                    raise RuntimeError('Invalid FP32 Euler predictor or C preservation')
                predictors.append(prediction)
                active=_active_mode(mode,k);correction=None;obs_elapsed=0.;geo_elapsed=0.
                if active is not None:
                    counts['trajectory_corrector_calls']+=1;corrector_attempts+=1;stamp=time.perf_counter()
                    try:
                        corrected_free,correction=correct_geometry(prediction[12:].reshape(100,3,3),original_C,float(L),active,boundary)
                    finally:
                        geo_elapsed=time.perf_counter()-stamp;geometry_seconds+=geo_elapsed
                    counts['trajectory_corrector_returns']=counts.get('trajectory_corrector_returns',0)+1;corrector_returns+=1
                    _add_counts(correction_counts,correction['counts'])
                    pred_cf=correction['before']['cf'];post_cf=correction['after']['cf']
                    R_calls+=sum(q['module']=='R' for q in correction['modules']);S_calls+=sum(q['module']=='S' for q in correction['modules'])
                    for module in correction['modules']:
                        key='trajectory_'+module['module']+'_module_calls';counts[key]=counts.get(key,0)+1
                    cumulative_accepted_length+=correction['sum_cumulative_vertex_motion_lengths']
                    stamp=time.perf_counter()
                    predicted[:,12:]=torch.from_numpy(corrected_free.reshape(1,100,9)).to(device=device,dtype=torch.float32)
                    predicted[:,:12]=known_C[None]
                    torch.cuda.synchronize();elapsed_transfer=time.perf_counter()-stamp
                    transferred+=elapsed_transfer;transfer_seconds+=elapsed_transfer
                    post=prediction.copy();post[12:]=corrected_free.reshape(100,9)
                    stamp=time.perf_counter();actual_post=predicted[0].cpu().numpy().copy()
                    elapsed_transfer=time.perf_counter()-stamp;transferred+=elapsed_transfer;transfer_seconds+=elapsed_transfer
                    if not np.array_equal(actual_post,post):raise RuntimeError('CPU accepted FP32 correction differs from actual GPU writeback')
                    post=actual_post
                else:
                    pred_cf,observed,obs_elapsed=_observe_CF(original_C,prediction[12:],float(L))
                    _add_counts(observation_counts,observed);observation_seconds+=obs_elapsed
                    post_cf=pred_cf;post=prediction.copy()
                state=predicted
                if state.dtype!=torch.float32 or not torch.equal(state[0,:12],known_C) or not bool(torch.isfinite(state).all()):
                    raise RuntimeError('Invalid corrected GPU integration state')
                if not np.array_equal(post[:12],original_C.reshape(12,9)):raise RuntimeError('Corrector changed C')
                net=post[12:].astype(np.float64)-prediction[12:].astype(np.float64)
                norms=np.linalg.norm(net.reshape(100,3,3),axis=-1)
                if float(norms.max())>1e-3*L:raise RuntimeError('Shared total correction rho exceeded')
                predicted_motion=prediction[12:].astype(np.float64)-states[-1][12:].astype(np.float64)
                model_vertex_lengths=np.linalg.norm(predicted_motion.reshape(100,3,3),axis=-1)
                row=dict(k=k,free_time=float(np.float32(k/50)),known_time=1.,dt=1/50,active_correction=active,
                    known_predictor_bitwise=True,known_corrected_bitwise=True,free_faces=100,
                    input_C_free=previous_cf,predictor_C_free=pred_cf,corrected_C_free=post_cf,
                    model_defect_delta=_defect_delta(previous_cf,pred_cf),geometry_defect_delta=_defect_delta(pred_cf,post_cf),
                    model_free_free_defect_delta=None if correction is None or previous_ff is None else _defect_delta(previous_ff,correction['before']['ff'],True),
                    geometry_free_free_defect_delta=None if correction is None else _defect_delta(correction['before']['ff'],correction['after']['ff'],True),
                    predictor_free_vertex_motion_RMS=float(np.sqrt(np.mean(model_vertex_lengths**2))),
                    predictor_free_vertex_motion_max=float(model_vertex_lengths.max()),
                    correction_net_displacement_RMS=float(np.sqrt(np.mean(norms**2))),correction_net_displacement_max=float(norms.max()),
                    correction_cumulative_accepted_vertex_path_length_this_step=0. if correction is None else correction['sum_cumulative_vertex_motion_lengths'],
                    correction_cumulative_accepted_vertex_path_length_trajectory=cumulative_accepted_length,
                    accepted=0 if correction is None else sum(q['accepted'] for q in correction['modules']),
                    rejected=0 if correction is None else sum(q['rejected'] for q in correction['modules']),
                    unresolved_collision_pairs=post_cf['illegal_pair_count'],
                    uncertain_detector_pairs=len(post_cf['unresolved_pairs']),
                    free_free_status='NOT_RUN_OBSERVATION_ONLY' if correction is None else 'FULL_CURRENT_PAIR_SET_WITH_EVERY_AFFECTED_UPDATE_VERIFIED',
                    free_free_before=None if correction is None else correction['before']['ff'],
                    free_free_after=None if correction is None else correction['after']['ff'],
                    model_cuda_seconds=cuda_seconds,transfer_and_integrator_sync_seconds=transferred,
                    correction_cpu_seconds=geo_elapsed,observation_cpu_seconds=obs_elapsed,correction=correction)
                step_audits.append(row);states.append(post);previous_cf=post_cf
                previous_ff=None if correction is None else correction['after']['ff']
        torch.cuda.synchronize();_freeze_guard(model)
        if counts['model_forward_attempts']-before_fw!=50 or counts['model_forward_returns']-before_returns!=50:
            raise RuntimeError('Exactly50 attempted and returned native forwards required')
        if len(states)!=51 or len(predictors)!=50:raise RuntimeError('Incomplete fifty-step trajectory')
        if _hash(noise)!=source_hash or _hash(C)!=C_hash or not np.array_equal(noise,original_noise) or not np.array_equal(C,original_C):
            raise RuntimeError('Pristine noise or C input mutated')
        if not np.array_equal(np.stack(states)[:,:12],np.broadcast_to(original_C.reshape(12,9),(51,12,9))):
            raise RuntimeError('C not exact throughout51 states')
        if not np.array_equal(states[0][12:],original_noise[0,12:]):raise RuntimeError('Initial free Gaussian changed')
        stamp=time.perf_counter();actual=state[0].cpu().numpy().copy();transfer_seconds+=time.perf_counter()-stamp
        if not np.array_equal(actual,states[-1]):raise RuntimeError('Saved final state differs from actual GPU state')
        result.update(_save_arrays(destination,states,predictors))
        result.update(status='PASS_EXECUTION',nfe=50,known_all51_bitwise=True,initial_free_Gaussian_exact=True,
            no_input_mutation=True,runtime=runtime,raw_states_are_actual_FP32_integrator=True,
            off_path_arithmetic='Identical to native_patch_run.clamped_sample; no CPU writeback when correction disabled',
            free_free_endpoint_audit='Performed separately by common endpoint evaluator',
            geometric_updates_only=True,model_gradient_graph_created=False)
        counts['complete_generations']+=1;counts[category]+=1
    except BaseException as error:
        result.update(status='FAILED',error=repr(error),failure_type=type(error).__name__,
            nfe_attempts=counts['model_forward_attempts']-before_fw,
            nfe_returns=counts['model_forward_returns']-before_returns,completed_Euler_states=max(0,len(states)-1),
            partial_corrector_counts_on_exception='Only returned corrector audits are available; failed in-flight candidates are not silently labeled zero')
        # Publish the actual attempt record before optional I/O. KeyboardInterrupt
        # and a secondary partial-save failure must not discard this ledger or
        # replace the original failure seen by the root BaseException handler.
        counts.setdefault('sampling_failures',[]).append(result)
        error.sampling_record=result
        try:
            if states:
                failed=raw/(label+'_FAILED_partial.npz')
                result.update(_save_arrays(failed,states,predictors))
                result['partial_save_status']='SAVED'
            else:
                result['partial_save_status']='NO_COMPLETED_STATE'
        except BaseException as save_error:
            result.update(partial_save_status='FAILED',partial_save_failure=repr(save_error),
                partial_save_failure_type=type(save_error).__name__,original_failure_preserved=True)
        raise
    finally:
        counts['geometry_cpu_seconds']+=geometry_seconds
        counts['geometry_observation_cpu_seconds']=counts.get('geometry_observation_cpu_seconds',0.)+observation_seconds
        counts['transfer_seconds']+=transfer_seconds
        _add_counts(counts.setdefault('geometry',{}).setdefault('trajectory_correction',{}),correction_counts)
        _add_counts(counts['geometry'].setdefault('trajectory_observation',{}),observation_counts)
        peak=torch.cuda.max_memory_allocated(device)
        counts['peak_cuda_allocated']=max(counts.get('peak_cuda_allocated',0),peak)
        result.update(costs=dict(wall_seconds=time.perf_counter()-wall_start,model_cuda_seconds=model_seconds,
            correction_cpu_seconds=geometry_seconds,observation_cpu_seconds=observation_seconds,
            transfer_and_integrator_sync_seconds=transfer_seconds,peak_cuda_allocated=peak,
            CPU_saved_array_bytes=sum(a.nbytes for a in states)+sum(a.nbytes for a in predictors),
            CUDA_event_scope='Native forward and output.float only; no geometry or CPU observation',
            transfer_scope='Initial H2D/D2H, per-step D2H predictor, correction H2D and checked D2H; includes waiting for queued Euler state operations'),
            corrector_attempts=corrector_attempts,corrector_returns=corrector_returns,R_module_calls=R_calls,S_module_calls=S_calls,
            geometry_counts=dict(correction=correction_counts,observation=observation_counts),
            sum_cumulative_accepted_vertex_motion_lengths=cumulative_accepted_length)
    return result
