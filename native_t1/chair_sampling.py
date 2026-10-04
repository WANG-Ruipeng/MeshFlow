"""START's original variable-N, C-first Gaussian and clamped Euler50 sampler.

Only C, total N and noise enter inference. No training-only HYBRID coupling,
targets, relation channels, guidance or geometry repair are applied here.
"""
from .runtime import configure_stable_runtime
import time
import numpy as np
import torch
from .artifacts import array_hash
from .model import native_execution
from .sampling import new_counts


def make_noise(seed, num_faces, known_faces):
    """Original MT19937 free-face draw, cast to FP32; known slots start at zero."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("START seed must be an integer in [0, 2**32)")
    if type(num_faces) is not int or not 128 <= num_faces <= 256:
        raise ValueError("START requires total N128..256")
    if type(known_faces) is not int or not 0 < known_faces < num_faces:
        raise ValueError("START requires 0 < known faces < total N")
    noise = np.random.RandomState(seed).randn(num_faces-known_faces, 3, 3).astype(np.float32)
    z = np.zeros((1, num_faces, 9), dtype=np.float32)
    z[:, known_faces:] = noise.reshape(1, num_faces-known_faces, 9)
    return torch.from_numpy(z)


def euler_step(state, velocity, C, known_faces):
    """The historical expression and multiplication order, with exact C clamping."""
    next_state = state.clone()
    next_state[:, known_faces:] = state[:, known_faces:] + velocity[:, known_faces:] * (1 / 50)
    next_state[:, :known_faces] = C[None]
    return next_state


@torch.no_grad()
def clamped_sample(model, z, C, counts=None):
    """Return raw[N,9], all 51 states, and an actual-call audit for a frozen START."""
    configure_stable_runtime()
    if model.experiment_trainable or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise RuntimeError("Sampling requires a frozen model without stored gradients")
    if z.device != C.device or z.device != next(model.parameters()).device or z.device.type != "cuda":
        raise ValueError("Model, C and Gaussian must share a CUDA device")
    if z.dtype != torch.float32 or z.ndim != 3 or z.shape[0] != 1 or z.shape[-1] != 9:
        raise ValueError("START Gaussian must be FP32[1,N,9]")
    if C.dtype != torch.float32 or C.ndim not in (2, 3) or C.shape[1:] not in ((9,), (3, 3)):
        raise ValueError("START C must be FP32[K,9] or FP32[K,3,3]")
    N = z.shape[1]; C = C.reshape(-1, 9); K = len(C)
    if not 128 <= N <= 256 or not 0 < K < N or not bool(torch.isfinite(z).all() and torch.isfinite(C).all()):
        raise ValueError("Finite inputs, N128..256 and 0<K<N required")
    if getattr(model, "context_encoder", None) is None:
        raise ValueError("START requires its learned Geo encoder")
    counts = new_counts() if counts is None else counts
    before = counts["model_forward_attempts"]
    counts["rollout_attempts"] += 1
    source, condition = array_hash(z), array_hash(C)
    state = z.clone(); state[:, :K] = C[None]
    known = torch.zeros((1, N), dtype=torch.bool, device=z.device); known[:, :K] = True
    valid = torch.ones_like(known); y = torch.tensor([N], device=z.device, dtype=torch.long)
    path = [state[0].cpu().numpy().copy()]
    raw_dtypes = set(); geometry_calls = 0

    def pre(module, args):
        if counts["model_forward_attempts"] - before >= 50:
            raise RuntimeError("Euler50 forward budget exceeded")
        counts["model_forward_attempts"] += 1

    def post(module, args, value):
        if value.requires_grad:
            raise RuntimeError("Unexpected sampling autograd graph")
        counts["model_forward_returns"] += 1
        raw_dtypes.add(str(value.dtype))

    def observe_geo(module, args):
        nonlocal geometry_calls
        geometry_calls += 1
        if not torch.equal(args[0][0, :K], C):
            raise RuntimeError("Geo received changed known coordinates")

    handles = [model.register_forward_pre_hook(pre), model.register_forward_hook(post),
               model.context_encoder.register_forward_pre_hook(observe_geo)]
    try:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); tick = time.perf_counter()
        with native_execution(model, coordinates=state) as facts:
            for step in range(50):
                if not torch.equal(state[0, :K], C):
                    raise RuntimeError("C changed before forward")
                t = torch.full((1,), step / 50, dtype=torch.float32, device=z.device)
                velocity = model(state, t, y, valid, known).float()
                if velocity.shape != state.shape or not bool(torch.isfinite(velocity).all()):
                    raise RuntimeError("Nonfinite or malformed velocity")
                state = euler_step(state, velocity, C, K)
                counts["euler_updates"] += 1
                if state.dtype != torch.float32 or not bool(torch.isfinite(state).all()) or not torch.equal(state[0, :K], C):
                    raise RuntimeError("Invalid clamped Euler state")
                path.append(state[0].cpu().numpy().copy())
        torch.cuda.synchronize(); seconds = time.perf_counter()-tick
        if counts["model_forward_attempts"]-before != 50 or geometry_calls != 50 or raw_dtypes != {"torch.bfloat16"}:
            raise RuntimeError("Expected exactly 50 BF16 model/Geo calls")
        if array_hash(z) != source or array_hash(C) != condition:
            raise RuntimeError("Sampling input was modified")
        path = np.stack(path)
        if not np.array_equal(path[:, :K], np.broadcast_to(C.cpu().numpy(), (51, K, 9))):
            raise RuntimeError("Known faces differ in saved trajectory")
        counts["complete_rollouts"] += 1
        return state[0].cpu().numpy(), path, dict(
            nfe=50, total_faces=N, known_faces=K, known_all51_bitwise=True, known_state_checks=51,
            actual_Geo_recomputations=geometry_calls, observed_raw_output_dtypes=sorted(raw_dtypes),
            known_raw_velocity_forced_zero=False, GT_model_inputs=False, geometry_gradient_calls=0,
            postprocessing=None, seconds=seconds, runtime=facts, integration_dtype="float32",
            noise_sha256=source, condition_sha256=condition, known_slots=list(range(K)),
            free_slots=list(range(K, N)), free_updated=not np.array_equal(path[0, K:], path[-1, K:]),
            initial_distribution="MT19937 free-face normals, FP64 draw cast FP32; C first K slots; no alpha scaling",
            peak_cuda_allocated=torch.cuda.max_memory_allocated(), peak_cuda_reserved=torch.cuda.max_memory_reserved(),
            peak_scope="One rollout after reset; includes resident model and autocast cache")
    finally:
        for handle in handles:
            handle.remove()
