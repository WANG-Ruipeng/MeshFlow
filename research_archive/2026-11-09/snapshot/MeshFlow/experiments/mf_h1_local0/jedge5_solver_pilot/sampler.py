"""Four preregistered fixed solvers; deterministic BF16 model, FP32 coordinates.

No adaptive method, extra prediction, GT, guidance, repair, OT or training.
CPU mocks are explicit and cannot be selected as a CUDA fallback.
"""
from native_t1.runtime import configure_stable_runtime
from native_t1.model import native_execution
from native_t1.artifacts import array_hash
from contextlib import nullcontext
import time
import numpy as np
import torch

METHODS={
    'E_quad50':dict(solver='Euler',intervals=50,nfe=50,grid='quadratic'),
    'Midpoint25':dict(solver='Midpoint',intervals=25,nfe=50,grid='uniform'),
    'Heun25':dict(solver='Heun',intervals=25,nfe=50,grid='uniform'),
    'Heun50':dict(solver='Heun',intervals=50,nfe=100,grid='uniform')}
TIME_POLICY='Ideal grid t_k=(k/M)^2 or k/M in Python FP64; each t and interval dt independently casts to FP32 as used in model/integration. Midpoint uses ideal (t_k+t_k+1)/2. No time accumulation.'


def new_counts():
    return dict(model_forward_attempts=0,model_forward_returns=0,accepted_steps=0,
        full_rollout_attempts=0,complete_rollouts=0,sampling_failures=0,
        model_backward_calls=0,optimizer_updates=0,geometry_gradient_calls=0,OT_calls=0,postprocessing_repairs=0)


def _bits_equal(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))


def _validate(model,noise,C,method,cpu_mock):
    if method not in METHODS:raise ValueError('Only four registered solver methods are supported')
    if not isinstance(cpu_mock,bool):raise ValueError('cpu_mock must be an explicit boolean')
    if noise.dtype!=torch.float32 or noise.shape!=(1,112,9):raise ValueError('Original FP32 [1,112,9] noise required')
    if C.dtype!=torch.float32 or C.shape not in ((12,3,3),(12,9)):raise ValueError('Original FP32 K12 condition required')
    parameters=list(model.parameters());device='cpu' if cpu_mock else 'cuda'
    if noise.device.type!=device or C.device!=noise.device or any(p.device!=noise.device for p in parameters):
        raise ValueError('Explicit CPU mock or real CUDA inputs required; no automatic device fallback')
    if not parameters or any(p.dtype!=torch.float32 or p.requires_grad or p.grad is not None for p in parameters):
        raise RuntimeError('Frozen FP32 parameters without stored gradients required')
    if getattr(model,'experiment_trainable',True) or any(m.training for m in model.modules()):raise RuntimeError('Eval frozen model required')
    if not bool(torch.isfinite(noise).all() and torch.isfinite(C).all()):raise FloatingPointError('Nonfinite input')


def _validate_velocity(velocity,state,cpu_mock):
    if not isinstance(velocity,torch.Tensor) or velocity.shape!=state.shape or velocity.device!=state.device or velocity.requires_grad:
        raise RuntimeError('Invalid model velocity shape/device/gradient graph')
    expected=torch.float32 if cpu_mock else torch.bfloat16
    if velocity.dtype!=expected:raise RuntimeError(f'Expected actual {expected} velocity; no precision fallback')
    if not bool(torch.isfinite(velocity).all()):raise FloatingPointError('Nonfinite model velocity')


@torch.no_grad()
def sample(model,noise,C,method,counts=None,cpu_mock=False):
    """Return raw[112,9], accepted states[M+1,112,9], JSON audit plus audit['_arrays'].

    Pop `_arrays` and save it with the result NPZ. It contains every actual
    pre-forward stage state, returned velocity (FP32 export), and actual time.
    The first stage velocity is the original t=0 model call, with no extra call.
    """
    _validate(model,noise,C,method,cpu_mock)
    if not cpu_mock:configure_stable_runtime()
    spec=METHODS[method];M=spec['intervals'];NFE=spec['nfe']
    if counts is None:counts=new_counts()
    for k in new_counts():counts.setdefault(k,0)
    counts['full_rollout_attempts']+=1
    initial_attempts=counts['model_forward_attempts'];initial_returns=counts['model_forward_returns']
    noise_before=array_hash(noise);condition_before=array_hash(C);condition=C.reshape(12,9)
    state=noise.clone();state[:,:12]=condition[None]
    known=torch.zeros((1,112),dtype=torch.bool,device=noise.device);known[:,:12]=True
    valid=torch.ones_like(known);y=torch.tensor([112],device=noise.device)
    ideal=[(k/M)**2 if spec['grid']=='quadratic' else k/M for k in range(M+1)]
    stage_states=[];stage_velocities=[];stage_times=[];stage_intervals=[];stage_kinds=[]
    accepted=[];events=[];returned_dtypes=[];checks=0
    host_seconds=sync_seconds=copy_seconds=0.;tick=time.perf_counter()
    def sync():
        nonlocal sync_seconds
        if not cpu_mock:
            s=time.perf_counter();torch.cuda.synchronize(noise.device);sync_seconds+=time.perf_counter()-s
    def exported(value):
        nonlocal copy_seconds
        s=time.perf_counter();v=value.detach().float()[0].cpu().numpy().copy();copy_seconds+=time.perf_counter()-s
        return v
    def assert_C(value):
        nonlocal checks
        checks+=1
        if value.dtype!=torch.float32 or not _bits_equal(value[0,:12],condition):raise RuntimeError('Stage/accepted condition changed bits')
        if not bool(torch.isfinite(value).all()):raise FloatingPointError('Nonfinite stage/accepted state')
    def advance(base,velocity,dt):
        value=base.clone()
        value[:,12:]=base[:,12:]+velocity[:,12:]*dt
        value[:,:12]=condition[None]
        assert_C(value)
        return value
    def forward(value,ideal_time,interval,kind):
        nonlocal host_seconds
        if counts['model_forward_attempts']-initial_attempts>=NFE:raise RuntimeError('Per-rollout NFE cap exceeded')
        assert_C(value)
        t=torch.full((1,),ideal_time,dtype=torch.float32,device=noise.device)
        stage_states.append(exported(value));stage_times.append(float(np.float32(ideal_time)))
        stage_intervals.append(interval);stage_kinds.append(kind)
        if not cpu_mock:
            if not torch.is_autocast_enabled('cuda') or torch.get_autocast_dtype('cuda')!=torch.bfloat16:
                raise RuntimeError('Real forward lost BF16 autocast; no fallback')
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
        s=time.perf_counter();counts['model_forward_attempts']+=1
        velocity=model(value,t,y,valid,known)
        counts['model_forward_returns']+=1;host_seconds+=time.perf_counter()-s
        if not cpu_mock:end.record();events.append((begin,end))
        _validate_velocity(velocity,value,cpu_mock);returned_dtypes.append(str(velocity.dtype))
        v=velocity.float();stage_velocities.append(exported(v));return v
    sync()
    if not cpu_mock:torch.cuda.reset_peak_memory_stats(noise.device)
    assert_C(state);accepted.append(exported(state))
    context=nullcontext(dict(cpu_mock=True,native_precision='FP32 analytic mock only')) if cpu_mock else native_execution(model,coordinates=(state,condition),autocast=True)
    try:
        with context as runtime:
            for k in range(M):
                t0,t1=ideal[k],ideal[k+1];dt=t1-t0
                v1=forward(state,t0,k,'start')
                if spec['solver']=='Euler':
                    updated=advance(state,v1,dt)
                elif spec['solver']=='Midpoint':
                    midpoint=advance(state,v1,dt*.5)
                    v2=forward(midpoint,(t0+t1)*.5,k,'midpoint')
                    updated=advance(state,v2,dt)
                else:
                    predictor=advance(state,v1,dt)
                    # Including the final t=1 model evaluation, without shortcuts.
                    v2=forward(predictor,t1,k,'Heun_predictor')
                    updated=advance(state,(v1+v2)*.5,dt)
                state=updated;counts['accepted_steps']+=1;accepted.append(exported(state))
        sync()
        if counts['model_forward_attempts']-initial_attempts!=NFE or counts['model_forward_returns']-initial_returns!=NFE:
            raise RuntimeError('Actual NFE differs from fixed solver contract')
        if array_hash(noise)!=noise_before or array_hash(C)!=condition_before:raise RuntimeError('Input noise/condition mutated')
        states=np.stack(accepted);coords=np.stack(stage_states);velocities=np.stack(stage_velocities)
        ref=condition.cpu().numpy()
        if not np.array_equal(states[:,:12].view(np.uint32),np.broadcast_to(ref,states[:,:12].shape).copy().view(np.uint32)):
            raise RuntimeError('Exported accepted C bits changed')
        if not np.array_equal(coords[:,:12].view(np.uint32),np.broadcast_to(ref,coords[:,:12].shape).copy().view(np.uint32)):
            raise RuntimeError('Exported stage C bits changed')
        counts['complete_rollouts']+=1
        aux=dict(stage_states=coords,stage_velocities=velocities,stage_times_FP32=np.asarray(stage_times,np.float32),
            accepted_times_FP32=np.asarray(ideal,np.float32),step_sizes_FP32=np.asarray(np.diff(ideal),np.float32),
            stage_interval=np.asarray(stage_intervals,np.int64),stage_kind=np.asarray(stage_kinds,dtype='U16'))
        audit=dict(method=method,solver=spec['solver'],intervals=M,nfe=NFE,precision='bf16' if not cpu_mock else 'CPU_mock_FP32',
            time_policy=TIME_POLICY,ideal_accepted_times=ideal,actual_accepted_times_FP32=aux['accepted_times_FP32'].tolist(),
            actual_stage_times_FP32=stage_times,actual_step_sizes_FP32=aux['step_sizes_FP32'].tolist(),
            actual_endpoint_t1_forward_calls=sum(t==1. for t in stage_times),
            model_forward_attempts=NFE,model_forward_returns=NFE,accepted_states=M+1,recorded_stage_states=NFE,
            runtime=runtime,actual_velocity_dtypes=sorted(set(returned_dtypes)),integration_dtype='torch.float32',
            known_accepted_all_bitwise=True,known_stages_all_bitwise=True,known_check_count=checks,
            known_predicted_velocity_may_be_nonzero=True,known_velocity_applied=False,
            inputs_unchanged=True,noise_sha256=noise_before,condition_sha256=condition_before,
            first_stage_velocity_sha256=array_hash(velocities[0]),
            total_faces=112,known_faces=12,free_faces=100,seconds=time.perf_counter()-tick,
            model_forward_host_seconds=host_seconds,model_forward_cuda_event_seconds=None if cpu_mock else sum(a.elapsed_time(b) for a,b in events)/1000,
            explicit_cuda_synchronize_seconds=sync_seconds,stage_and_state_CPU_copy_seconds=copy_seconds,
            peak_cuda_allocated=0 if cpu_mock else torch.cuda.max_memory_allocated(noise.device),
            peak_cuda_reserved=0 if cpu_mock else torch.cuda.max_memory_reserved(noise.device),
            timing_note='Host/CUDA/sync/copy timings can overlap; not additive. Peak includes resident model.',
            model_backwards=0,optimizer_updates=0,OT_calls=0,geometry_gradient_calls=0,postprocessing=None,cpu_mock=cpu_mock,
            _arrays=aux)
        return states[-1].copy(),states,audit
    except BaseException as exc:
        counts['sampling_failures']+=1
        exc.partial_states=np.stack(accepted)
        exc.partial_stage_states=np.stack(stage_states) if stage_states else np.empty((0,112,9),np.float32)
        exc.partial_stage_velocities=np.stack(stage_velocities) if stage_velocities else np.empty((0,112,9),np.float32)
        exc.partial_stage_times_FP32=np.asarray(stage_times,np.float32)
        raise
