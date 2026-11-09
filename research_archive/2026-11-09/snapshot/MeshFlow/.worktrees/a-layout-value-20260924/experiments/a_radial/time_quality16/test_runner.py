"""Small contract/control-flow tests; mocks are not geometry/timing evidence."""
import copy
import json
import numpy as np
import pytest

from . import runner as r


def protocol():
    uids = [f'uid{i:02}' for i in range(16)]
    return {'uids': uids, 'all16_sorted_uids': uids, 'arms': list(r.ARMS),
        'threads': 4, 'interop_threads': 1, 'device': 'cpu', 'max_steps': 16, 'max_seconds': 3600,
        'snapshot_steps': list(range(17)), 'observation_steps': list(r.OBSERVATIONS),
        'optimizer_seed_base': 1493501, 'evaluation_seed_bases': [1593501, 1693501],
        'points_each_direction': 8192, 'repeats': 3,
        'data': {u: {'path': 'not-read-in-mock', 'sha256': 'data'} for u in uids},
        'model_bindings': {a: {'path': 'model-only.pt', 'file_sha256': 'file', 'model_sha256': 'model'} for a in r.ARMS[1:]},
        'bound_files_sha256': {'not-rehashed-per-case': 'preflight-done-by-driver'}}


def optimizer_result(losses=(4., 5., 3., 3.), feasible=(False, True, True, True), accepted=(False, True, True, True)):
    snapshots, trace = [], []
    best, best_loss = None, None
    faces = np.array([[0, 1, 2]], np.int64)
    for i, (loss, valid, take) in enumerate(zip(losses, feasible, accepted)):
        if valid and (i == 0 or take) and (best is None or loss < best_loss):
            best, best_loss = i, loss
        u = np.eye(3) * (i + 1.)
        row = {'iteration': i, 'internal_loss': loss, 'feasible': valid, 'accepted': take,
               'best_feasible_internal_loss': best_loss}
        trace.append(row)
        snapshots.append({**row, 'u': u, 'vertices': 2 * u, 'faces': faces})
    chosen = snapshots[-1] if best is None else snapshots[best]
    return {'trace': trace, 'snapshots': snapshots, 'attempts': [],
        'stats': {'iterations': len(trace)-1, 'best_feasible_iteration': best},
        'u': chosen['u'].copy(), 'vertices': chosen['vertices'].copy(), 'faces': faces.copy()}


def test_incumbent_uses_first_accepted_feasible_strict_loss_and_preserves_ties():
    result = optimizer_result()
    values = r.reconstruct_incumbents(result)
    assert [x['best_feasible_iteration'] for x in values] == [None, 1, 2, 2]
    assert np.array_equal(values[-1]['u'], result['snapshots'][2]['u'])
    # A lower internal loss that is not an accepted feasible update cannot win.
    bad = copy.deepcopy(result)
    bad['trace'][3]['internal_loss'] = bad['snapshots'][3]['internal_loss'] = 1.
    bad['trace'][3]['accepted'] = bad['snapshots'][3]['accepted'] = False
    assert r.reconstruct_incumbents(bad)[-1]['best_feasible_iteration'] == 2


def test_no_feasible_prefix_delivers_current_and_tampered_final_is_rejected():
    result = optimizer_result(feasible=(False,)*4)
    values = r.reconstruct_incumbents(result)
    assert [x['selected_current_iteration'] for x in values] == [0, 1, 2, 3]
    result['u'][0, 0] = np.nextafter(result['u'][0, 0], 100.)
    with pytest.raises(ValueError, match='final differs bitwise'):
        r.reconstruct_incumbents(result)


def test_latin_order_matches_prepare_and_balances_positions():
    from .prepare import order
    p = protocol(); p['case_order'] = order(p['uids'])
    schedule = r.balanced_order(p)
    assert len(schedule) == 144 and schedule == p['case_order']
    assert schedule[48]['uid'] == 'uid01' and schedule[96]['uid'] == 'uid02'
    for uid in p['uids']:
        for arm in r.ARMS:
            positions = [i % 3 for i, row in enumerate(schedule) if row['uid'] == uid and row['arm'] == arm]
            assert sorted(positions) == [0, 1, 2]
    assert str(r.case_folder('/run', 'uid00', 'fixed', 0)).endswith('cases/uid00/fixed/repeat_0')


def test_geometry_and_metrics_hashes_bind_values_dtype_and_order():
    a = {'u': np.eye(3), 'r': np.ones(3), 'vertices': np.eye(3), 'faces': np.array([[0, 1, 2]], np.int64)}
    b = {k: v.copy() for k, v in a.items()}; b['r'][0] = np.nextafter(1., 2.)
    assert r.geometry_hash(a) != r.geometry_hash(b)
    assert r.metrics_hash([{'seed': 1, 'rms': 2.}, {'seed': 2, 'rms': 3.}]) != r.metrics_hash([{'seed': 2, 'rms': 3.}, {'seed': 1, 'rms': 2.}])
    assert r.metrics_hash([{'seed': 1, 'rms': 2.}]) == r.metrics_hash([{'rms': 2., 'seed': 1}])


class Clock:
    value = 0.
    def now(self):
        return self.value
    def add(self, value):
        self.value += value


def mocked_case(tmp_path, monkeypatch, *, audit_error_at=None, optimizer_error_at=None, initializer_error=False):
    """Fake arithmetic/times test scheduling and denominator behavior only."""
    clock = Clock(); calls = []
    monkeypatch.setattr(r.time, 'perf_counter', clock.now)
    monkeypatch.setattr(r, 'warm_process', lambda: {'mock_only': True})
    u = np.tile(np.array([[1., 0., 0.]]), (258, 1))
    faces = np.tile(np.array([[0, 1, 2]], np.int32), (512, 1))
    data = {'base_u': u, 'logr': np.zeros(2050), 'faces': faces,
            'ref_vertices': np.zeros((2050, 3)), 'ref_faces': np.zeros((4096, 3), np.int64)}
    def load(*args):
        clock.add(1.); return copy.deepcopy(data)
    monkeypatch.setattr(r, 'load_data', load)
    class Support:
        def __init__(self): clock.add(1.)
        def mesh(self, y): return data['ref_vertices'].copy(), data['ref_faces'].copy()
        def query(self, y, q): return np.ones(len(q))
    monkeypatch.setattr(r, 'RadialSupport', Support)
    def model(*args): clock.add(1.); return None
    monkeypatch.setattr(r, 'load_model', model)
    def fixed(support, y, base, f):
        clock.add(1.)
        if initializer_error: raise RuntimeError('original initializer failure')
        return {'u': base.copy(), 'r': np.ones(len(base)), 'vertices': base.copy(), 'faces': f.astype(np.int64)}, 1.
    monkeypatch.setattr(r, 'make_fixed', fixed)
    def evaluate(q, f, y, support, rv, rf, *, floor, baseline_metrics, seeds, points):
        calls.append((floor is None, tuple(seeds), points)); clock.add(2.)
        if len(calls) == audit_error_at: raise RuntimeError('original audit failure')
        mesh = {'u': q.copy(), 'r': np.ones(len(q)), 'vertices': q.copy(), 'faces': f.astype(np.int64)}
        quality = {'min_triangle_angle_deg': 10., 'triangle_min_angle_p05_deg': 20.}
        metrics = [{'seed': s, 'points_per_direction': points, 'rms_bbox_percent': 1.} for s in seeds]
        return mesh, {'quality': quality, 'audit': {'valid': True}, 'metrics': metrics, 'target_pass': False}
    monkeypatch.setattr(r, 'evaluate_candidate', evaluate)
    def optimize_layout(y, initial_u, faces, floor, *, device, max_steps, max_seconds, seed, snapshot_steps, on_record):
        trace, attempts, snapshots = [], [], []
        best_iteration = None
        for iteration in range(17):
            clock.add(.1)
            if iteration == optimizer_error_at: raise RuntimeError('original optimizer failure')
            current = initial_u.copy(); current[:, 1] = iteration * .001
            current /= np.linalg.norm(current, axis=1, keepdims=True)
            best_iteration = iteration
            row = {'iteration': iteration, 'internal_loss': 17.-iteration, 'feasible': True,
                   'accepted': iteration > 0, 'best_feasible_internal_loss': 17.-iteration}
            trace.append(row); snapshots.append({**row, 'u': current.copy(), 'vertices': current.copy(), 'faces': faces.astype(np.int64)})
            on_record(iteration=iteration, current_u=current.copy(), delivered_u=current.copy(),
                      faces=faces.copy(), row=copy.deepcopy(row), best_feasible_iteration=iteration)
        return {'trace': trace, 'attempts': attempts, 'snapshots': snapshots,
                'u': current.copy(), 'vertices': current.copy(), 'faces': faces.astype(np.int64),
                'stats': {'iterations': 16, 'best_feasible_iteration': 16, 'stop_reason': 'max_steps'}}
    monkeypatch.setattr(r, 'optimize_layout_observed', optimize_layout)
    p = protocol(); source = tmp_path / 'RUN_PROTOCOL.json'; source.write_text(json.dumps(p))
    out = tmp_path / 'case'
    task = r.run_case(source, 'uid00', 'fixed', 0, out)
    with np.load(out / 'ARRAYS.npz', allow_pickle=False) as z:
        saved = {k: np.array(z[k]) for k in z.files}
    return task, saved, calls


def test_synchronous_case_includes_all_prior_audits_and_exactly_seven_calls(tmp_path, monkeypatch):
    task, saved, calls = mocked_case(tmp_path, monkeypatch)
    assert task['status'] == 'completed' and task['completed_full_budget']
    assert len(calls) == 7 and calls[0][0] and all(not c[0] for c in calls[1:])
    assert all(c[1:] == ((1593501, 1693501), 8192) for c in calls)
    times = [o['audit_completed_case_seconds'] for o in task['observations']]
    assert times[0] >= 8. and all(b-a >= 2. for a, b in zip(times, times[1:]))
    assert task['timings']['total_end_to_end_seconds'] >= times[-1]
    assert task['final_return_matches_audited_grid16_bitwise']
    assert np.array_equal(saved['observed_16_u'], saved['final_u'])
    assert np.array_equal(saved['observed_0_u'], saved['raw_u'])
    assert all(x['status'] == 'available' for x in task['snapshot_slots'])


def test_audit_failure_cannot_change_optimizer_but_marks_case_error(tmp_path, monkeypatch):
    task, saved, calls = mocked_case(tmp_path, monkeypatch, audit_error_at=3)
    assert task['status'] == 'error' and task['completed_full_budget']
    assert task['observations'][1]['status'] == 'audit_error'
    assert task['observations'][1]['audit_completed_case_seconds'] is None
    assert task['failure']['optimizer_still_ran_unchanged_after_audit_error']
    assert task['optimizer']['stats']['iterations'] == 16 and len(calls) == 7


def test_initializer_failure_keeps_all_scheduled_slots_and_original_error(tmp_path, monkeypatch):
    task, saved, calls = mocked_case(tmp_path, monkeypatch, initializer_error=True)
    assert task['status'] == 'error' and 'original initializer failure' in task['failure']['traceback']
    assert len(task['observations']) == 6 and len(task['snapshot_slots']) == 17
    assert all(x['status'] == 'unavailable_not_reached' for x in task['observations'])
    assert len(calls) == 1 and 'baseline_u' in saved and 'final_u' not in saved


def test_optimizer_exception_retains_executed_trace_and_prefix_arrays(tmp_path, monkeypatch):
    task, saved, calls = mocked_case(tmp_path, monkeypatch, optimizer_error_at=3)
    assert task['status'] == 'error' and 'original optimizer failure' in task['failure']['traceback']
    assert task['optimizer']['stats']['iterations'] == 2
    assert saved['current_iterations'].tolist() == [0, 1, 2]
    assert saved['observer_iterations'].tolist() == [0, 1, 2]
    assert all(x['status'] == 'available' for x in task['snapshot_slots'][:3])
    assert all(x['status'] == 'unavailable_not_executed' for x in task['snapshot_slots'][3:])
    assert 'final_u' not in saved
