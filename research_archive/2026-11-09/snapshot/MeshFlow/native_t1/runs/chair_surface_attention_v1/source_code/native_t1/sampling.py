"""Original clamped Euler dynamics extracted from the verified Native patch run.

No target mesh, guidance, extra CFG, projection, welding or geometry callback.
All 51 FP32 states are preserved for numerical regression.
"""
from .runtime import configure_stable_runtime
import time
import numpy as np
import torch
from .model import native_execution
from .artifacts import array_hash

EVAL_K = (2, 4, 8, 12)


def make_noise(seed):
    return torch.randn((1, 112, 9), generator=torch.Generator(device="cpu").manual_seed(int(seed)), dtype=torch.float32)


def new_counts():
    return dict(model_forward_attempts=0, model_forward_returns=0,
                euler_updates=0, complete_rollouts=0, rollout_attempts=0,
                model_backward_calls=0, optimizer_updates=0,
                geometry_gradient_calls=0, postprocessing_repairs=0)


@torch.no_grad()
def clamped_sample(model, z, C, counts=None):
    """Return (raw[112,9], path[51,112,9], audit); only C is conditioning."""
    configure_stable_runtime()
    if model.experiment_trainable or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise RuntimeError("Sampling requires a frozen model with no stored gradients")
    if z.device != C.device or z.device != next(model.parameters()).device:
        raise ValueError("Model, C, and Gaussian must be on the same CUDA device")
    if z.device.type != "cuda" or not bool(torch.isfinite(z).all() and torch.isfinite(C).all()):
        raise ValueError("Finite CUDA FP32 inputs required")
    if counts is None:
        counts = new_counts()
    before = counts["model_forward_attempts"]
    counts["rollout_attempts"] += 1
    def pre(module, args):
        if counts["model_forward_attempts"] - before >= 50:
            raise RuntimeError("50-forward sampling budget exceeded")
        counts["model_forward_attempts"] += 1
    def post(module, args, output):
        if output.requires_grad:
            raise RuntimeError("Unexpected sampling autograd graph")
        counts["model_forward_returns"] += 1
    hooks = [model.register_forward_pre_hook(pre), model.register_forward_hook(post)]
    try:
        return _sample(model, z, C, counts)
    finally:
        for hook in hooks:
            hook.remove()


def _sample(model,z,C,counts):
    if z.dtype!=torch.float32 or z.shape!=(1,112,9) or C.dtype!=torch.float32: raise ValueError('Exact FP32 inputs required')
    C=C.reshape(-1,9);K=len(C)
    if K not in EVAL_K: raise ValueError('Only registered nested K2/4/8/12')
    before=counts['model_forward_attempts'];source=array_hash(z);ch=array_hash(C)
    state=z.clone();state[:,:K]=C[None]
    known=torch.zeros((1,112),device=z.device,dtype=torch.bool);known[:,:K]=True
    valid=torch.ones_like(known);y=torch.tensor([112],device=z.device)
    path=[state[0].cpu().numpy().copy()];checks=[]
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
    context=native_execution(model,coordinates=state)
    with context as facts:
        for k in range(50):
            if not torch.equal(state[0,:K],C): raise RuntimeError('Known coordinates changed before forward')
            t=torch.full((1,),k/50,device=z.device,dtype=torch.float32)
            v=model(state,t,y,valid,known).float()
            next_state=state.clone()
            next_state[:,K:]=state[:,K:]+v[:,K:]*(1/50)
            next_state[:,:K]=C[None]
            state=next_state;counts['euler_updates']+=1
            ok=torch.equal(state[0,:K],C);checks.append(ok)
            if not ok or state.dtype!=torch.float32 or not bool(torch.isfinite(state).all()): raise RuntimeError('Invalid clamped Euler state')
            path.append(state[0].cpu().numpy().copy())
    torch.cuda.synchronize();seconds=time.perf_counter()-tick
    if counts['model_forward_attempts']-before!=50: raise RuntimeError('Expected exactly50 forward calls')
    if array_hash(z)!=source or array_hash(C)!=ch: raise RuntimeError('Sampling input mutated')
    path=np.stack(path)
    if not np.array_equal(path[:,:K],np.broadcast_to(C.cpu().numpy(),(51,K,9))): raise RuntimeError('51-state known check failed')
    if np.array_equal(path[0,K:],path[-1,K:]): raise RuntimeError('Free coordinates did not move')
    counts['complete_rollouts']+=1
    return state[0].cpu().numpy(),path,dict(nfe=50,geometry_gradient_calls=0,seconds=seconds,
        peak_cuda_allocated=torch.cuda.max_memory_allocated(),runtime=facts,known_all51_bitwise=True,known_state_checks=51,
        known_slots=list(range(K)),free_slots=list(range(K,112)),known_export_order='Original archived C face and vertex order',
        free_updated=True,noise_sha256=source,condition_sha256=ch,initial_distribution='C in firstK slots; original Gaussian slots K:112 retained',
        known_input_dtype='torch.float32',integration_dtype='torch.float32',total_faces=112,postprocessing=None)
