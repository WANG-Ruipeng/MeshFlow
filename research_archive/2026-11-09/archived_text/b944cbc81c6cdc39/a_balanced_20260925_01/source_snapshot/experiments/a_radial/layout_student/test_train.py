"""Correctness tests only: no training/check split outcome is inspected here."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ..legacy.networks import DomainVelocity, point_features
from .train import (BalancedSampler, discrete_bridge, file_sha256, model_sha256,
                    spherical_bridge, train_run)

ASSETS = Path(__file__).resolve().parent.parent / 'assets'


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _create_run(path: Path, updates=12) -> Path:
    path.mkdir()
    rng = np.random.default_rng(720)
    u0 = _unit(rng.normal(size=(4, 258, 3)))
    u1 = _unit(u0 + 0.02 * rng.normal(size=u0.shape))
    u1[0] = u0[0]
    p = _unit(rng.normal(size=(4, 256, 3))) * rng.uniform(.8, 1.2, size=(4, 256, 1))
    uids = ['train_a', 'train_b', 'train_c', 'train_d']
    np.savez(path / 'train_labels.npz', train_uids=uids, directions0=u0,
             directions1=u1, proxy_points=p, point_features=point_features(p))
    protocol = {'train_uids': uids, 'internal_check_uids': ['check_forbidden'],
        'paths': {'training_labels': 'train_labels.npz',
                  'original_a_checkpoint': str(ASSETS / 'original_A_10.pt')},
        'training': {'updates': updates, 'seeds': [10, 11], 'batch_size': 4,
                     'learning_rate': 3e-4, 'weight_decay': 1e-4},
        'hashes': {'training_labels_sha256': file_sha256(path / 'train_labels.npz'),
                   'original_a_sha256': file_sha256(ASSETS / 'original_A_10.pt')}}
    (path / 'RUN_PROTOCOL.json').write_text(json.dumps(protocol))
    return path


def _same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _same(x, y)
    else:
        assert a == b


def test_analytic_bridge_endpoints_tangent_and_finite_difference():
    g = torch.Generator().manual_seed(77)
    u0 = torch.nn.functional.normalize(torch.randn(3, 9, 3, generator=g, dtype=torch.float64), dim=-1)
    u1 = torch.nn.functional.normalize(u0 + .2 * torch.randn(3, 9, 3, generator=g, dtype=torch.float64), dim=-1)
    at0, _ = spherical_bridge(u0, u1, torch.zeros(3, dtype=torch.float64))
    at1, _ = spherical_bridge(u0, u1, torch.ones(3, dtype=torch.float64))
    assert torch.allclose(at0, u0, atol=1e-14, rtol=0)
    assert torch.allclose(at1, u1, atol=1e-14, rtol=0)
    t = torch.tensor([.1, .5, .9], dtype=torch.float64)
    u, v = spherical_bridge(u0, u1, t)
    delta = 1e-6
    up, _ = spherical_bridge(u0, u1, t + delta)
    um, _ = spherical_bridge(u0, u1, t - delta)
    assert torch.allclose((up-um)/(2*delta), v, atol=2e-10, rtol=1e-8)
    assert torch.max(torch.abs((u*v).sum(-1))).item() < 1e-14


def test_discrete_bridge_one_step_exact_and_fixed_zero():
    g = torch.Generator().manual_seed(778)
    u0 = torch.nn.functional.normalize(torch.randn(32, 7, 3, generator=g, dtype=torch.float64), dim=-1)
    u1 = torch.nn.functional.normalize(u0 + .2 * torch.randn(32, 7, 3, generator=g, dtype=torch.float64), dim=-1)
    k = torch.arange(32)
    u, v, q, t = discrete_bridge(u0, u1, k)
    endpoint = torch.nn.functional.normalize(u + v/32, dim=-1)
    assert torch.allclose(endpoint, q, atol=1e-14, rtol=0)
    assert torch.max(torch.abs((u*v).sum(-1))).item() < 2e-14
    assert torch.equal(t, k.double()/32)
    _, fixed_v, _, _ = discrete_bridge(u0, u0, k)
    assert torch.count_nonzero(fixed_v).item() == 0


def test_bridge_rejects_antipodal_path_and_bad_discrete_times():
    u0 = torch.tensor([[[1., 0., 0.]]])
    with pytest.raises(ValueError, match='antipodal'):
        spherical_bridge(u0, -u0, torch.tensor([.5]))
    with pytest.raises(ValueError, match='integer'):
        discrete_bridge(u0, u0, torch.tensor([1.]))
    with pytest.raises(ValueError, match='outside'):
        discrete_bridge(u0, u0, torch.tensor([32]))


def test_unchanged_network_and_proxy_input_contract():
    model = DomainVelocity()
    original = torch.load(ASSETS / 'original_A_10.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(original, strict=True)
    assert sum(p.numel() for p in model.parameters()) == 29907
    assert [(k, tuple(v.shape)) for k, v in model.state_dict().items()] == [(k, tuple(v.shape)) for k, v in original.items()]
    p = np.ones((2, 256, 3), np.float64)
    features = point_features(p)
    assert features.shape == (2, 256, 4) and features.dtype == np.float32
    u = torch.nn.functional.normalize(torch.randn(2, 258, 3), dim=-1)
    out = model(u, torch.tensor([.2, .8]), torch.from_numpy(features))
    assert out.shape == u.shape and out.dtype == torch.float32
    assert torch.max(torch.abs((out*u).sum(-1))).item() < 1e-6


def test_balanced_sampler_resume():
    sampler = BalancedSampler(12, 10)
    for _ in range(2):
        sampler.take(4)
    state = sampler.state_dict()
    copied = BalancedSampler(12, 999)
    copied.load_state_dict(state)
    assert sampler.take(17) == copied.take(17)
    assert max(sampler.exposures) - min(sampler.exposures) <= 1
    _same(sampler.state_dict(), copied.state_dict())


def test_training_continuous_vs_resumed_identical_and_check_not_read(tmp_path, monkeypatch):
    run = _create_run(tmp_path / 'train_test')
    real_load = np.load
    opened = []
    def allowed_load(path, *args, **kwargs):
        assert Path(path).name == 'train_labels.npz', 'Training opened non-training teacher data'
        opened.append(str(path))
        return real_load(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', allowed_load)
    whole = train_run(run, 10, 12, device='cpu', checkpoint_every=5, output_dir=run/'whole')
    first = train_run(run, 10, 12, device='cpu', checkpoint_every=5, stop_after=5, output_dir=run/'first')
    resumed = train_run(run, 10, 12, Path(first['checkpoint']), device='cpu',
                        checkpoint_every=5, output_dir=run/'resumed')
    a = torch.load(whole['checkpoint'], map_location='cpu', weights_only=False)
    b = torch.load(resumed['checkpoint'], map_location='cpu', weights_only=False)
    for key in ('model', 'optimizer', 'scheduler', 'sampler', 'rng', 'config', 'trace'):
        _same(a[key], b[key])
    assert whole['model_sha256'] == resumed['model_sha256'] == model_sha256(a['model'])
    assert whole['parent_exposures'] == dict.fromkeys(['train_a', 'train_b', 'train_c', 'train_d'], 12)
    assert len(opened) == 3
    assert whole['checkpoint_selection'] == 'predeclared_final_update_only'
    assert a['is_predeclared_final'] is True
    assert torch.load(first['checkpoint'], map_location='cpu', weights_only=False)['is_predeclared_final'] is False


def test_training_rejects_label_leakage_and_changed_budget(tmp_path):
    run = _create_run(tmp_path / 'bad_contract')
    with pytest.raises(ValueError, match='total updates'):
        train_run(run, 10, 1, device='cpu')
    protocol = json.loads((run/'RUN_PROTOCOL.json').read_text())
    protocol['internal_check_uids'] = ['train_a']
    (run/'RUN_PROTOCOL.json').write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match='leakage'):
        train_run(run, 10, 12, device='cpu')
    assert not (run/'training').exists()
