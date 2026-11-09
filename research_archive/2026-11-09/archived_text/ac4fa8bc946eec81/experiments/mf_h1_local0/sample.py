"""Explicit 50-update sampling of the supplied callable, without model reload."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import torch


@torch.no_grad()
def euler_sample(model, noise, y, mask, prompts=None, steps=50, precision="fp32"):
    if steps != 50:
        raise ValueError("This frozen protocol requires exactly 50 Euler updates")
    device = noise.device
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    state = noise.clone()
    before = dict(getattr(model, "counts", {}))
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
        encoded = model.encode(prompts) if prompts is not None else None
        for k in range(steps):
            t = torch.full((state.shape[0],), k / steps, device=device, dtype=state.dtype)
            kwargs = dict(y=y, mask=mask)
            if encoded is not None:
                kwargs["encoded"] = encoded
            velocity = model(state, t, **kwargs)
            state = state + velocity.float() * (1 / steps)
    if device.type == "cuda":
        torch.cuda.synchronize()
    after = dict(getattr(model, "counts", {}))
    return state, {"status": "PASS" if torch.isfinite(state).all().item() else "FAIL_NONFINITE",
                   "nfe": steps, "euler_updates": steps, "batch": len(noise),
                   "grid": [k / steps for k in range(steps + 1)], "face_cfg": 1.0,
                   "geometry_cfg": 1.0, "seconds": time.perf_counter() - start,
                   "precision": precision, "call_delta": {k: after[k] - before[k] for k in after},
                   "peak_cuda_allocated": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0}


def save_raw(tokens, destination, noise=None, **metadata):
    """Unquantized triangle soup, in final target coordinates, without repairs."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tokens = tokens.detach().cpu().numpy() if torch.is_tensor(tokens) else np.asarray(tokens)
    arrays = dict(tokens=tokens, **metadata)
    if noise is not None:
        arrays["initial_noise"] = noise.detach().cpu().numpy() if torch.is_tensor(noise) else noise
    np.savez_compressed(destination.with_suffix(".npz"), **arrays)
    vertices = tokens.reshape(-1, 3)
    with destination.with_suffix(".obj").open("w", encoding="utf-8") as stream:
        stream.write("# raw triangle soup; final target coordinates; no postprocessing\n")
        for vertex in vertices:
            stream.write("v " + " ".join(format(float(v), ".9g") for v in vertex) + "\n")
        for face in range(len(vertices) // 3):
            stream.write(f"f {3*face+1} {3*face+2} {3*face+3}\n")
