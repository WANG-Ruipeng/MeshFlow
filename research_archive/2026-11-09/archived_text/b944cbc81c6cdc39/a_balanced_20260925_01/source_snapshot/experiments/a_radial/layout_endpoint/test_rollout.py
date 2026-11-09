"""Differentiable rollout checks against the unchanged original control path."""
import numpy as np
import pytest
import torch
from ..exporter import MeshExporter
from ..legacy.deform import deform
from ..legacy.networks import point_features
from .rollout import rollout


class ContextVelocity(torch.nn.Module):
    def __init__(self, scale=.08):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale, dtype=torch.float32))
        self.register_buffer('direction', torch.tensor([.7, -.3, .2], dtype=torch.float32))

    def context(self, features):
        return features.mean(dim=1)

    def forward(self, u, t, features, context=None):
        raw = self.scale * (self.direction[None, None] + context[:, None, :3])
        return raw.expand_as(u) - (raw * u).sum(-1, keepdim=True) * u


class StressVelocity(torch.nn.Module):
    """Synthetic node-varying tangent field deliberately activates guard shrinkage."""
    def __init__(self, count=258):
        super().__init__()
        idx = torch.arange(count, dtype=torch.float32)
        self.register_buffer('raw', 80 * torch.stack((torch.sin(idx*1.7), torch.cos(idx*2.3), torch.sin(idx*.9)), -1))

    def context(self, features):
        return features.mean(1)

    def forward(self, u, t, features, context=None):
        raw = self.raw[None].expand_as(u) * (t[:, None, :] < 1/32).to(u.dtype)
        return raw - (raw * u).sum(-1, keepdim=True) * u


@pytest.fixture(scope='module')
def prepared():
    torch.set_num_threads(1)
    exporter = MeshExporter()
    base, faces = exporter.template(512)
    proxy = exporter.proxy(np.zeros(2050))
    return exporter, base, faces, proxy


def test_full32_cpu_mirror_matches_legacy(prepared):
    exporter, base, faces, proxy = prepared
    expected, path, alphas, _ = deform(exporter.flow, proxy, base, faces, True, 32)
    with torch.no_grad():
        actual = rollout(exporter.flow, torch.from_numpy(base[None]),
                         torch.from_numpy(point_features(proxy)[None]), faces, return_path=True)
    assert np.array_equal(actual['alphas'][0], alphas)
    np.testing.assert_allclose(actual['path'][0].numpy(), path, atol=1e-6, rtol=0)
    np.testing.assert_allclose(actual['u'][0].numpy(), expected, atol=1e-6, rtol=0)


def test_large_velocity_preserves_original_shrinking_guard(prepared):
    _, base, faces, proxy = prepared
    model = StressVelocity()
    expected, path, alphas, _ = deform(model, proxy, base, faces, True, 32)
    with torch.no_grad():
        actual = rollout(model, torch.from_numpy(base[None]),
                         torch.from_numpy(point_features(proxy)[None]), faces, return_path=True)
    assert np.any(alphas < 1)
    assert np.array_equal(actual['alphas'][0], alphas)
    np.testing.assert_allclose(actual['path'][0].numpy(), path, atol=2e-6, rtol=0)
    np.testing.assert_allclose(actual['u'][0].numpy(), expected, atol=2e-6, rtol=0)


def test_gradients_cross_all32_updates_and_context(prepared):
    _, base, faces, proxy = prepared
    model = ContextVelocity()
    u = torch.from_numpy(base[None].copy()).requires_grad_(True)
    features = torch.from_numpy(point_features(proxy)[None]).requires_grad_(True)
    actual = rollout(model, u, features, faces, return_path=True)
    weights = torch.linspace(-1, 1, base.size, dtype=torch.float64).reshape(1, *base.shape)
    loss = (actual['u'] * weights).sum()
    grad = torch.autograd.grad(loss, [model.scale, u, features, actual['path']], allow_unused=True)
    assert all(value is not None and torch.isfinite(value).all() and value.abs().max() > 0 for value in grad[:3])
    assert actual['path'].shape == (1, 33, 258, 3)
    assert np.all(actual['alphas'] == 1)
    # Finite differences on the FP32 parameter use a resolvable perturbation.
    derivative = float(grad[0]); original = float(model.scale)
    losses = []
    for sign in (-1, 1):
        with torch.no_grad():
            model.scale.fill_(original + sign * 1e-3)
            result = rollout(model, u.detach(), features.detach(), faces)
            losses.append(float((result['u'] * weights).sum()))
    finite_difference = (losses[1] - losses[0]) / 2e-3
    assert abs(derivative - finite_difference) <= 3e-4 * max(1, abs(derivative))


def test_batch_context_is_per_parent_and_matches_singles(prepared):
    exporter, base, faces, proxy = prepared
    features = torch.from_numpy(point_features(proxy)[None]).repeat(4, 1, 1)
    directions = torch.from_numpy(base[None]).repeat(4, 1, 1)
    with torch.no_grad():
        batched = rollout(exporter.flow, directions, features, faces)
        single = rollout(exporter.flow, directions[:1], features[:1], faces)
    assert np.all(batched['alphas'] == single['alphas'])
    np.testing.assert_allclose(batched['u'].numpy(), single['u'].numpy().repeat(4, axis=0), atol=2e-6, rtol=0)


def test_dtype_and_nonunit_inputs_are_rejected(prepared):
    exporter, base, faces, proxy = prepared
    features = torch.from_numpy(point_features(proxy)[None])
    with pytest.raises(ValueError, match='float64'):
        rollout(exporter.flow, torch.from_numpy(base[None]).float(), features, faces)
    with pytest.raises(ValueError, match='unit norm'):
        rollout(exporter.flow, torch.from_numpy(base[None]) * 2, features, faces)

