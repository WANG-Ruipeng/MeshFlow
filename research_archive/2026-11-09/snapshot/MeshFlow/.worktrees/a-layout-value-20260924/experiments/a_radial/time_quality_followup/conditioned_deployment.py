"""CPU A deployment with explicitly separate target geometry and proxy condition.

Both matched and mismatched conditions use this identical file-read/proxy path.
The caller freezes the target-to-condition mapping; this module never selects it.
No quality gate, fallback, optimizer, or evaluation metric is used here.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import time

import numpy as np
import torch

from ..io_utils import array_hash
from ..layout_student.train import model_sha256
from ..legacy.common import sample_surface
from ..legacy.deform import deform
from ..legacy.networks import DomainVelocity, point_features


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _field(value, name):
    value = np.asarray(value)
    _require(value.shape == (2050,) and value.dtype in (np.dtype('float32'), np.dtype('float64')),
             name + ' must be float32/float64 [2050]')
    _require(np.isfinite(value).all(), name + ' must be finite')
    return value


def read_condition(path, *, file_sha256, field_hash):
    """Hash and decode the same bytes; only a radial field may cross this boundary."""
    raw = Path(path).read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    _require(actual_sha == file_sha256, 'Condition file SHA256 mismatch')
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        _require(archive.files == ['logr'], 'Condition archive must contain only logr')
        condition = _field(archive['logr'], 'condition_logr')
    _require(array_hash(condition) == field_hash, 'Condition field hash mismatch')
    return condition


def predict_raw_with_condition(flow, *, target_logr, target_base, target_faces,
                               target_uid, target_field_hash, support,
                               condition_path, condition_uid,
                               condition_file_sha256, condition_field_hash,
                               expected_model_sha256=None):
    """Return (mesh, metadata); all output geometry belongs to the target field.

    This is the original CPU float32 network/float64 guard path. Model loading and
    support construction remain caller costs. generation_seconds includes input
    validation/hashing, condition I/O, proxy construction, the *whole* legacy
    deform call (including point features/context), and final target requery.
    Identity hashes add explicit overhead; no timing is claimed to be cold-start.
    """
    begin = time.perf_counter()
    _require(isinstance(target_uid, str) and bool(target_uid), 'Missing target UID')
    _require(isinstance(condition_uid, str) and bool(condition_uid), 'Missing condition UID')
    y = _field(target_logr, 'target_logr')
    base, faces = np.asarray(target_base), np.asarray(target_faces)
    _require(base.shape == (258, 3) and base.dtype == np.float64 and np.isfinite(base).all(),
             'Target base must be finite float64 [258,3]')
    _require(np.allclose(np.linalg.norm(base, axis=1), 1., rtol=0., atol=1e-9),
             'Target base directions must be unit length')
    _require(faces.shape == (512, 3) and faces.dtype.kind in 'iu',
             'Target faces must be integer [512,3]')
    _require(faces.min() >= 0 and faces.max() < len(base), 'Target face index out of range')
    _require(array_hash(y) == target_field_hash, 'Target field hash mismatch')
    _require(type(flow) is DomainVelocity and not flow.training,
             'Expected original DomainVelocity in eval mode')
    parameters = list(flow.parameters())
    _require(sum(p.numel() for p in parameters) == 29907, 'Network parameter contract changed')
    _require(all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in parameters),
             'Deployment requires original CPU float32 model')
    model_hash = model_sha256(flow.state_dict())
    if expected_model_sha256 is not None:
        _require(model_hash == expected_model_sha256, 'Model hash mismatch')
    before_hashes = (array_hash(y), array_hash(base), array_hash(faces))
    identity_validation_seconds = time.perf_counter() - begin

    condition_begin = time.perf_counter()
    condition = read_condition(condition_path, file_sha256=condition_file_sha256,
                               field_hash=condition_field_hash)
    condition_read_seconds = time.perf_counter() - condition_begin
    proxy_begin = time.perf_counter()
    pv, pf = support.mesh(condition)
    proxy = sample_surface(pv, pf, 256, np.random.default_rng(4501))[0]
    _require(proxy.shape == (256, 3) and np.isfinite(proxy).all() and
             np.all(np.linalg.norm(proxy, axis=1) > 0), 'Invalid condition proxy')
    proxy_seconds = time.perf_counter() - proxy_begin
    identity_begin = time.perf_counter()
    proxy_hash, feature_hash = array_hash(proxy), array_hash(point_features(proxy))
    proxy_identity_seconds = time.perf_counter() - identity_begin

    rollout_begin = time.perf_counter()
    u, path, alphas, rollout_seconds = deform(flow, proxy, base, faces, True, 32)
    deform_call_wall_seconds = time.perf_counter() - rollout_begin
    _require(u.shape == base.shape and u.dtype == np.float64 and np.isfinite(u).all(),
             'Invalid accepted directions')
    _require(path.shape == (33, 258, 3) and np.array_equal(path[0], base) and
             np.array_equal(path[-1], u), 'Original rollout endpoint contract changed')
    _require(alphas.shape == (32,) and np.isfinite(alphas).all(), 'Invalid guard alpha record')
    query_begin = time.perf_counter()
    r = support.query(y, u)  # Target field only, after all guard decisions.
    _require(r.shape == (258,) and np.isfinite(r).all() and np.all(r > 0),
             'Invalid final target radius')
    mesh = {'vertices': u * r[:, None], 'faces': np.asarray(faces, np.int64),
            'u': u, 'r': r, 'alpha': alphas}
    final_query_seconds = time.perf_counter() - query_begin
    _require(before_hashes == (array_hash(y), array_hash(base), array_hash(faces)),
             'Target inputs were mutated')
    meta = {
        'target_uid': target_uid, 'target_mainfield_hash': before_hashes[0],
        'target_base_hash': before_hashes[1], 'target_faces_hash': before_hashes[2],
        'condition_uid': condition_uid, 'condition_path': str(condition_path),
        'condition_file_sha256': condition_file_sha256,
        'condition_mainfield_hash': condition_field_hash,
        'proxy_points_hash': proxy_hash, 'point_features_hash': feature_hash,
        'model_sha256': model_hash, 'condition_archive_keys': ['logr'],
        'condition_preparation': 'same file read and proxy recipe for matched/mismatched',
        'proxy_seed': 4501, 'proxy_point_count': 256,
        'identity_validation_seconds': identity_validation_seconds,
        'condition_read_seconds': condition_read_seconds, 'proxy_seconds': proxy_seconds,
        'proxy_identity_seconds': proxy_identity_seconds,
        'deform_call_wall_seconds': deform_call_wall_seconds,
        'rollout_seconds': float(rollout_seconds), 'final_query_seconds': final_query_seconds,
        'min_alpha': float(alphas.min()), 'shrunk_steps': int((alphas < 1).sum()),
        'network_device': 'cpu', 'network_dtype': 'float32',
        'guard_geometry_dtype': 'float64', 'steps': 32,
        'final_accepted_direction_requery': True, 'final_radius_source': 'target_mainfield',
        'model_load_excluded': True, 'support_construction_excluded': True,
        'quality_or_error_selection': False, 'fallback': False,
    }
    meta['generation_seconds'] = time.perf_counter() - begin
    return mesh, meta
