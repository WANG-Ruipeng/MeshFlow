"""Synthetic boundary tests only; no trained/new-object inference or timing claim."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from . import conditioned_deployment as c
from ..layout_student import deployment as original


class Support:
    """Small synthetic surface; target and condition call arguments stay visible."""
    def __init__(self):
        self.mesh_fields, self.query_fields = [], []

    def mesh(self, field):
        self.mesh_fields.append(field.copy())
        v = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        return v * np.exp(field[0]), np.array([[0, 1, 2]], np.int64)

    def query(self, field, u):
        self.query_fields.append((field.copy(), u.copy()))
        return np.full(len(u), np.exp(field[0]), dtype=np.float64)


def condition_file(tmp_path, y, name='condition.npz', **extra):
    path = tmp_path / name
    np.savez(path, logr=y, **extra)
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), c.array_hash(y)


def inputs(tmp_path, monkeypatch, *, condition_radius=2.):
    flow = c.DomainVelocity().cpu().float().eval()
    base = np.tile(np.eye(3), (86, 1)).astype(np.float64)
    faces = np.tile(np.array([[0, 1, 2]], np.int32), (512, 1))
    target = np.zeros(2050, np.float64)
    donor = np.full(2050, np.log(condition_radius), np.float64)
    path, file_hash, field_hash = condition_file(tmp_path, donor)
    calls = []
    def fake_deform(model, proxy, u0, f, guarded, steps):
        assert model is flow and guarded is True and steps == 32
        calls.append((proxy.copy(), u0.copy(), f.copy()))
        u = u0.copy()
        return u, np.repeat(u[None], 33, axis=0), np.ones(32), .125
    monkeypatch.setattr(c, 'deform', fake_deform)
    kw = dict(target_logr=target, target_base=base, target_faces=faces,
        target_uid='target', target_field_hash=c.array_hash(target), support=Support(),
        condition_path=path, condition_uid='donor', condition_file_sha256=file_hash,
        condition_field_hash=field_hash, expected_model_sha256=c.model_sha256(flow.state_dict()))
    return flow, kw, calls, fake_deform


def test_mismatched_proxy_does_not_replace_target_field_template_or_final_radius(tmp_path, monkeypatch):
    flow, kw, calls, _ = inputs(tmp_path, monkeypatch)
    saved = {k: kw[k].copy() for k in ('target_logr', 'target_base', 'target_faces')}
    mesh, meta = c.predict_raw_with_condition(flow, **kw)
    support = kw['support']
    assert len(support.mesh_fields) == len(support.query_fields) == 1
    assert np.array_equal(support.mesh_fields[0], np.full(2050, np.log(2.)))
    assert np.array_equal(support.query_fields[0][0], kw['target_logr'])
    assert np.array_equal(calls[0][1], kw['target_base'])
    assert np.array_equal(calls[0][2], kw['target_faces'])
    assert np.array_equal(mesh['r'], np.ones(258))
    assert np.array_equal(mesh['vertices'], kw['target_base'])
    assert not np.array_equal(mesh['vertices'], kw['target_base'] * 2.)
    assert meta['target_mainfield_hash'] != meta['condition_mainfield_hash']
    assert meta['proxy_points_hash'] == c.array_hash(calls[0][0])
    assert meta['point_features_hash'] == c.array_hash(c.point_features(calls[0][0]))
    assert meta['final_radius_source'] == 'target_mainfield'
    assert all(np.array_equal(kw[k], v) for k, v in saved.items())


def test_matched_synthetic_path_matches_original_proxy_rollout_and_query(tmp_path, monkeypatch):
    flow, kw, calls, fake_deform = inputs(tmp_path, monkeypatch, condition_radius=1.)
    kw['condition_uid'] = kw['target_uid']
    monkeypatch.setattr(original, 'deform', fake_deform)
    mesh, meta = c.predict_raw_with_condition(flow, **kw)
    old_mesh, old_meta = original.predict_raw(flow, kw['target_logr'], Support(),
                                             kw['target_base'], kw['target_faces'])
    assert all(np.array_equal(mesh[key], old_mesh[key]) for key in old_mesh)
    assert np.array_equal(calls[0][0], calls[1][0])
    assert meta['proxy_points_hash'] == old_meta['proxy_points_hash']
    assert meta['rollout_seconds'] == old_meta['rollout_seconds']


def test_matched_and_wrong_each_read_condition_once_and_build_proxy_once(tmp_path, monkeypatch):
    flow, kw, _, _ = inputs(tmp_path, monkeypatch)
    donor_path = kw['condition_path']
    matched_path, matched_sha, matched_hash = condition_file(tmp_path, kw['target_logr'], 'matched.npz')
    reads = []
    read = Path.read_bytes
    def tracked(path):
        reads.append(path)
        return read(path)
    monkeypatch.setattr(Path, 'read_bytes', tracked)
    c.predict_raw_with_condition(flow, **kw)
    correct = dict(kw, condition_path=matched_path, condition_uid='target',
                   condition_file_sha256=matched_sha, condition_field_hash=matched_hash)
    c.predict_raw_with_condition(flow, **correct)
    assert reads == [donor_path, matched_path]
    assert len(kw['support'].mesh_fields) == len(kw['support'].query_fields) == 2


@pytest.mark.parametrize('corruption', ['file_hash', 'field_hash', 'extra_keys', 'nan', 'shape', 'integer'])
def test_condition_reader_rejects_unbound_or_non_condition_data(tmp_path, corruption):
    y = np.zeros(2050, np.float64)
    extra = {}
    if corruption == 'extra_keys': extra = {'base_u': np.zeros((258, 3))}
    if corruption == 'nan': y[0] = np.nan
    if corruption == 'shape': y = y[:10]
    if corruption == 'integer': y = y.astype(np.int64)
    path, sha, arr_hash = condition_file(tmp_path, y, **extra)
    if corruption == 'file_hash': sha = '0' * 64
    if corruption == 'field_hash': arr_hash = '0' * 64
    with pytest.raises(ValueError):
        c.read_condition(path, file_sha256=sha, field_hash=arr_hash)


@pytest.mark.parametrize('corruption', ['target_hash', 'model_hash', 'training', 'dtype'])
def test_target_and_model_contracts_reject_before_any_rollout(tmp_path, monkeypatch, corruption):
    flow, kw, calls, _ = inputs(tmp_path, monkeypatch)
    if corruption == 'target_hash': kw['target_field_hash'] = '0' * 64
    if corruption == 'model_hash': kw['expected_model_sha256'] = '0' * 64
    if corruption == 'training': flow.train()
    if corruption == 'dtype': flow.double()
    with pytest.raises(ValueError):
        c.predict_raw_with_condition(flow, **kw)
    assert calls == []


def test_all_preparation_and_whole_deform_call_are_included_in_generation_time(tmp_path, monkeypatch):
    flow, kw, _, _ = inputs(tmp_path, monkeypatch)
    tick = iter(np.arange(1000, dtype=float))
    monkeypatch.setattr(c.time, 'perf_counter', lambda: next(tick))
    _, meta = c.predict_raw_with_condition(flow, **kw)
    assert meta['condition_read_seconds'] == 1.
    assert meta['proxy_seconds'] == 1.
    assert meta['deform_call_wall_seconds'] == 1.
    assert meta['generation_seconds'] > sum(meta[k] for k in (
        'identity_validation_seconds', 'condition_read_seconds', 'proxy_seconds',
        'proxy_identity_seconds', 'deform_call_wall_seconds', 'final_query_seconds'))

@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_condition_values_and_dtype_survive_file_read_without_conversion(tmp_path, dtype):
    y = np.linspace(-.2, .3, 2050, dtype=dtype)
    path, sha, field_hash = condition_file(tmp_path, y)
    loaded = c.read_condition(path, file_sha256=sha, field_hash=field_hash)
    assert loaded.dtype == y.dtype
    assert np.array_equal(loaded, y)
    assert loaded.tobytes() == y.tobytes()