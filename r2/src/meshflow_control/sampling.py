"""Original MT19937 input and lossless free-only Euler50; no geometry edits."""
from . import runtime
from pathlib import Path
import json,time
import numpy as np
import torch
from .mesh_io import validate_C,write_raw
from .artifacts import file_sha256
def make_noise(seed,N,K):
    if type(seed) is not int or not 0<=seed<2**32: raise ValueError("Seed outside uint32")
    z=np.zeros((1,N,9),np.float32)
    z[:,K:]=np.random.RandomState(seed).randn(N-K,3,3).astype(np.float32).reshape(1,N-K,9)
    return z
def sample(model,C,N,seed=None,z=None,*,budget,out=None,context=None):
    from .models.native import native_execution
    C=validate_C(C,int(N));K=len(C);ctx=context or {}
    if z is None: z=make_noise(seed,int(N),K)
    z=np.asarray(z,dtype=np.float32)
    if z.shape==(N-K,3,3):
        full=np.zeros((1,N,9),np.float32);full[:,K:]=z.reshape(1,N-K,9);z=full
    if z.shape!=(1,N,9) or not np.isfinite(z).all(): raise ValueError("Malformed frozen Gaussian")
    if any(p.requires_grad or p.grad is not None for p in model.parameters()): raise ValueError("Freeze model for sampling")
    device=next(model.parameters()).device
    if device.type!="cuda": raise ValueError("Real Euler50 sampling requires explicitly owned CUDA")
    budget.reserve("rollout_attempt",context=ctx)
    state=torch.from_numpy(z.copy()).to(device);known=torch.zeros((1,N),dtype=torch.bool,device=device);known[:,:K]=True
    valid=torch.ones_like(known);y=torch.tensor([N],device=device,dtype=torch.long)
    cond=torch.from_numpy(C.reshape(K,9).copy()).to(device);state[:,:K]=cond[None]
    path=[state[0].cpu().numpy().copy()];torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
    with torch.no_grad(),native_execution(model,coordinates=state):
        for step in range(50):
            t=torch.full((1,),step/50,dtype=torch.float32,device=device)
            velocity=budget.call("sampling_forward",lambda:model(state,t,y,valid,known),context={**ctx,"euler_step":step})
            if velocity.dtype!=torch.bfloat16 or velocity.shape!=state.shape or not torch.isfinite(velocity).all():
                raise RuntimeError("Invalid BF16 velocity")
            next_state=state.clone()
            next_state[:,K:]=state[:,K:]+velocity.float()[:,K:]*(1/50)
            next_state[:,:K]=cond[None];state=next_state
            if not torch.isfinite(state).all() or not torch.equal(state[0,:K],cond): raise RuntimeError("Euler/C contract failed")
            path.append(state[0].cpu().numpy().copy())
    torch.cuda.synchronize()
    raw=state[0].cpu().numpy();trajectory=np.stack(path)
    if any(p[:K].tobytes()!=C.tobytes() for p in trajectory): raise RuntimeError("C bit pattern not preserved")
    audit=dict(nfe=50,known_all51_bitwise=True,known_coordinates_bitwise=True,known_state_checks=51,
        N=N,K=K,seed=seed,seconds=time.perf_counter()-tick,readout_mode=model.readout_mode,model_inputs=["C","N","Gaussian","time"],
        dtype="BF16 backbone/FP32 readout/FP32 parameters+integration",cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_scope="one sampling rollout, resident weights included",
        human_usability="NOT_RUN",request=ctx)
    budget.event("ROLLOUT_COMPLETE",context=ctx,**{k:audit[k] for k in ("nfe","seconds","known_all51_bitwise")})
    if out: write_raw(out,raw,C,z,trajectory,audit)
    return raw,trajectory,audit
