"""Autograd mirror of the original CPU A rollout, with detached guard choices.

Only the discrete Bernstein step-size decision is read from NumPy. Network
context, all intermediate directions, projections, and normalizations remain in
one torch graph. Float64 geometry and float32 network operations match legacy.
"""
from __future__ import annotations
import numpy as np
import torch
from ..legacy.chart import retract


def rollout(model, base_u, features, faces, steps=32, return_path=False):
    """Return differentiable final directions and original discrete guard alphas.

    base_u: float64 [B,N,3]; features: float32 [B,256,4]. NumPy faces are fixed.
    The CPU guard roundtrip is a real training cost, not a GPU-only claim. Its
    selected alpha has zero derivative; gradients are local to that guard choice.
    """
    if base_u.ndim != 3 or base_u.shape[-1] != 3 or base_u.dtype != torch.float64:
        raise ValueError('base_u must be float64 [B,N,3]')
    if features.shape != (base_u.shape[0], 256, 4) or features.dtype != torch.float32:
        raise ValueError('features must be float32 [B,256,4]')
    if base_u.device != features.device:
        raise ValueError('Direction and condition tensors must share a device')
    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError('faces must be an integer [F,3] array')
    if len(faces) == 0 or faces.min() < 0 or faces.max() >= base_u.shape[1]:
        raise ValueError('Face indices are outside the direction array')
    if not isinstance(steps, int) or steps < 1:
        raise ValueError('steps must be a positive integer')
    if not bool(torch.isfinite(base_u).all()) or not bool(torch.isfinite(features).all()):
        raise ValueError('Nonfinite rollout input')
    if not bool(torch.all(torch.abs(torch.linalg.vector_norm(base_u, dim=-1) - 1) <= 1e-9)):
        raise ValueError('Base directions must have unit norm')
    # Match the original deployment's B=1 matrix kernels for every parent.
    # Concatenation preserves each parent's complete context gradient.
    contexts = [model.context(features[parent:parent+1]) for parent in range(len(features))]
    u = base_u
    history = [u] if return_path else None
    alphas = np.empty((len(base_u), steps), dtype=np.float64)
    for iteration in range(steps):
        # DomainVelocity already performs the original first projection in FP32.
        network_u = u.to(dtype=torch.float32)
        tt = torch.full((len(u), 1), iteration / steps, dtype=torch.float32, device=u.device)
        predicted = torch.cat([model(network_u[parent:parent+1], tt[parent:parent+1],
                                     features[parent:parent+1], contexts[parent])
                               for parent in range(len(u))], dim=0)
        if predicted.shape != u.shape or predicted.dtype != torch.float32:
            raise ValueError('Original network must return float32 [B,N,3]')
        delta = predicted.to(dtype=torch.float64) / steps
        # The next two FP64 projections correspond to deform.py then chart.retract.
        delta = delta - (delta * u).sum(dim=-1, keepdim=True) * u
        numpy_u = u.detach().cpu().numpy()
        numpy_delta = delta.detach().cpu().numpy()
        for parent in range(len(u)):
            _, alphas[parent, iteration] = retract(numpy_u[parent], numpy_delta[parent], faces)
        tangent = delta - (delta * u).sum(dim=-1, keepdim=True) * u
        alpha = torch.as_tensor(alphas[:, iteration].copy(), dtype=torch.float64, device=u.device)[:, None, None]
        proposal = u + alpha * tangent
        u = proposal / torch.linalg.vector_norm(proposal, dim=-1, keepdim=True)
        if return_path:
            history.append(u)
    result = {'u': u, 'alphas': alphas}
    if return_path:
        result['path'] = torch.stack(history, dim=1)
    return result

