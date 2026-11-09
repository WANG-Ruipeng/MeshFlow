"""Small contract tests; no production optimization, metric sampling, or GPU."""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import pytest
from . import experiment as ex


def data_fixture():
    u = np.tile(np.array([[1., 0., 0.]], np.float64), (258, 1))
    return {'base_u': u, 'teacher_u': u.copy(), 'logr': np.zeros(2050, np.float32),
        'faces': (np.arange(512 * 3).reshape(512, 3) % 258).astype(np.int32),
        'ref_vertices': np.ones((2050, 3)), 'ref_faces': np.zeros((4096, 3), np.int64)}


def protocol_fixture():
    train = list(ex.UIDS) + [f'{i:032x}' for i in range(8)]
    all_uids = train + [f'{i:032x}' for i in range(8, 12)]
    return {'schema': 'A-TEACHER-MARGIN-v1', 'quality_tolerance_deg': 1e-6,
        'no_fixed_fallback': True, 'no_GT_or_MC_selection': True, 'no_old_label_updates': True,
        'no_network_training_or_learned_online_advantage_claim': True, 'no_generalization_claim': True,
        'uids': list(ex.UIDS), 'initializers': list(ex.INITIALIZERS), 'endpoint_names': list(ex.ENDPOINTS),
        'expected_tasks': 8, 'expected_endpoint_rows': 16, 'margin_deg': .1, 'max_steps': 64,
        'max_seconds': 3600, 'snapshot_steps': [0, 16, 64], 'device': 'cpu', 'threads': 4, 'interop_threads': 1,
        'all16_sorted_uids': sorted(all_uids), 'all_train_uids': train,
        'optimizer_seed_base': ex.OPT_SEED_BASE, 'evaluation_seed_bases': list(ex.MC_SEED_BASES),
        'points_each_direction': 8192, 'sensitivity': {'endpoint_names': list(ex.SENSITIVE_ENDPOINTS),
            'fractions': list(ex.FRACTIONS), 'trials': 16, 'planned_slots': 1344, 'seed_base': ex.NOISE_SEED_BASE}}


def test_stricter_floor_changes_both_angles_once_not_source():
    q = {ex.ANGLES[0]: 4.2, ex.ANGLES[1]: 7.9, 'face_count': 512}; before = dict(q)
    actual = ex.optimizer_floor(q)
    assert actual == {'min_angle_deg': 4.3, 'angle_q05_deg': 8.0}
    assert q == before
    with pytest.raises(ValueError, match='Fixed angle'):
        ex.stricter_floor({ex.ANGLES[0]: np.nan, ex.ANGLES[1]: 7.9})


def test_contract_rejects_posthoc_margin_budget_or_parent_changes():
    p = protocol_fixture(); ex.assert_contract(p)
    for key, value in [('margin_deg', .2), ('max_steps', 128), ('expected_endpoint_rows', 15),
                       ('quality_tolerance_deg', 1e-5), ('no_old_label_updates', False),
                       ('uids', list(reversed(ex.UIDS)))]:
        mutated = copy.deepcopy(p); mutated[key] = value
        with pytest.raises(ValueError):
            ex.assert_contract(mutated)
    mutated = copy.deepcopy(p); mutated['sensitivity']['planned_slots'] = 1008
    with pytest.raises(ValueError, match='denominator'):
        ex.assert_contract(mutated)


def test_frozen_data_rejects_teacher_normalization_and_face_dtype_changes():
    data = data_fixture(); ex.validate_data(data)
    data['teacher_u'][0] *= 2
    with pytest.raises(ValueError, match='unit vectors'):
        ex.validate_data(data)
    data = data_fixture(); data['faces'] = data['faces'].astype(np.int64)
    with pytest.raises(ValueError, match='int32'):
        ex.validate_data(data)


def test_noise_uses_same_absolute_tangent_rms_for_different_endpoints():
    rng = np.random.default_rng(44)
    u = rng.normal(size=(258, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = rng.normal(size=(258, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    ambient = rng.normal(size=u.shape); energy = .003
    for center in (u, v):
        perturbed = ex.perturb(center, ambient, energy, .1)
        assert np.allclose(np.linalg.norm(perturbed, axis=1), 1, atol=1e-14, rtol=0)
        # Recover the pre-normalization tangent step: q/(q.u)-u.
        tangent = perturbed / np.sum(perturbed * center, axis=1, keepdims=True) - center
        assert np.sqrt(np.square(tangent).mean()) == pytest.approx(.1 * np.sqrt(energy), rel=1e-12)
    assert np.array_equal(ex.perturb(u, ambient, energy, .1), ex.perturb(u, ambient, energy, .1))


def test_zero_teacher_is_exact_no_motion_for_every_fraction():
    u = data_fixture()['base_u']; noise = np.random.default_rng(9).normal(size=u.shape)
    for fraction in ex.FRACTIONS:
        assert np.array_equal(ex.perturb(u, noise, 0., fraction), u)
    with pytest.raises(ValueError):
        ex.perturb(u, noise, .1, .25)


def test_no_gt_enters_optimizer_and_failed_restoration_has_no_fallback(tmp_path, monkeypatch):
    data = data_fixture(); data['teacher_u'] = np.roll(data['teacher_u'], 1, axis=1)
    p = protocol_fixture(); captured = []
    chosen = data['teacher_u'].copy()
    stats = {'status': 'infeasible', 'best_feasible_iteration': None, 'iterations': 64}
    def fake_opt(logr, initial, faces, floor, **kwargs):
        captured.append((logr, initial, faces, floor, kwargs))
        return {'u': chosen, 'vertices': chosen * 3., 'stats': stats}
    monkeypatch.setattr(ex, 'optimize_layout', fake_opt)
    monkeypatch.setattr(ex, 'serialize_optimizer', lambda result, folder: None)
    monkeypatch.setattr(ex, 'snapshot_records', lambda result, requested: ([], []))
    def fake_endpoint(run, protocol, uid, name, directions, d, support, floor, claimed):
        assert directions is chosen
        assert not np.array_equal(directions, data['base_u'])
        return {'uid': uid, 'endpoint': name, 'status': 'success', 'qualified_strict': {'qualified': False}}
    monkeypatch.setattr(ex, 'endpoint_row', fake_endpoint)
    floor = {ex.ANGLES[0]: 2., ex.ANGLES[1]: 4.}
    task, row = ex.optimize_task(tmp_path, p, ex.UIDS[0], 'teacher', data, None, floor)
    assert task['restore_failed'] and task['stats']['status'] == 'infeasible'
    assert row['qualified_strict']['qualified'] is False
    assert len(captured) == 1
    logr, initial, faces, actual_floor, kwargs = captured[0]
    assert logr is data['logr'] and initial is data['teacher_u'] and faces is data['faces']
    assert actual_floor == {'min_angle_deg': 2.1, 'angle_q05_deg': 4.1}
    assert kwargs == {'device': 'cpu', 'max_steps': 64, 'max_seconds': 3600,
        'seed': ex.OPT_SEED_BASE + p['all16_sorted_uids'].index(ex.UIDS[0]), 'snapshot_steps': (0, 16, 64)}


def test_optimizer_exception_remains_an_explicit_unavailable_slot(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError('test restoration failure')
    monkeypatch.setattr(ex, 'optimize_layout', broken)
    task, row = ex.optimize_task(tmp_path, protocol_fixture(), ex.UIDS[0], 'fixed',
        data_fixture(), None, {ex.ANGLES[0]: 2., ex.ANGLES[1]: 4.})
    assert task['status'] == 'error' and row['status'] == 'unavailable'
    assert row['endpoint'] == 'opt_fixed' and row['planned_slot_retained']
    assert (tmp_path / 'attempts' / ex.UIDS[0] / 'fixed' / 'TASK.json').exists()
    with pytest.raises(FileExistsError):
        ex.optimize_task(tmp_path, protocol_fixture(), ex.UIDS[0], 'fixed',
            data_fixture(), None, {ex.ANGLES[0]: 2., ex.ANGLES[1]: 4.})


def test_missing_or_duplicate_endpoints_cannot_shrink_denominator():
    rows = [{'uid': uid, 'endpoint': name, 'status': 'unavailable'} for uid in ex.UIDS for name in ex.ENDPOINTS]
    ex.check_endpoint_slots(rows)
    with pytest.raises(ValueError):
        ex.check_endpoint_slots(rows[:-1])
    with pytest.raises(ValueError):
        ex.check_endpoint_slots(rows[:-1] + [rows[0]])


def test_mc_gains_require_same_reference_and_exact_seed_pair():
    row = {'status': 'success', 'mainfield_hash': 'field', 'GT_reference_hash': 'gt',
        'metrics': [{'seed': 1, 'rms_bbox_percent': 1.}, {'seed': 2, 'rms_bbox_percent': 2.}]}
    reference = copy.deepcopy(row)
    reference['metrics'][0]['rms_bbox_percent'] = 2.
    reference['metrics'][1]['rms_bbox_percent'] = 4.
    assert ex.paired_gain(row, reference) == [50., 50.]
    reference['metrics'][1]['seed'] = 3
    with pytest.raises(ValueError, match='pairing'):
        ex.paired_gain(row, reference)
    assert ex.paired_gain({'status': 'unavailable'}, reference) is None
