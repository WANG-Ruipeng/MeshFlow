"""CPU4 warm-process cases with synchronous, nonselecting quality observations.

All disk output is deferred until the timed case has ended. The unchanged
optimizer's internal best-feasible rule chooses each delivered prefix. Independent
evaluation records that candidate; it never chooses a candidate or stops a case.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import time
_IMPORT_STARTED = time.perf_counter()
import argparse
import copy
import hashlib
import json
import traceback
from pathlib import Path
import numpy as np
import torch

from ..exporter import RadialSupport
from ..io_utils import array_hash, file_hash, save_npz, write_json
from ..legacy.networks import DomainVelocity
from ..layout_student.train import model_sha256
from ..layout_student.deployment import make_fixed, predict_raw
from .observer_adapter import optimize_layout_observed
from .timed_eval import evaluate_candidate

_IMPORT_SECONDS = time.perf_counter() - _IMPORT_STARTED
_WARMUP = None
ARMS = ('fixed', 'early_v1_10', 'balanced_10')
OBSERVATIONS = (0, 1, 2, 4, 8, 16)
SNAPSHOTS = tuple(range(17))


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def path_from(root, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_protocol(protocol):
    """Prefer a RUN_PROTOCOL path, which resolves portable model/data paths."""
    if isinstance(protocol, (str, Path)):
        path = Path(protocol).resolve()
        p = read(path)
        return p, path.parent, file_hash(path), str(path)
    p = copy.deepcopy(protocol)
    require('run_root' in p, 'Dictionary protocol requires explicit run_root')
    root = Path(p['run_root']).resolve()
    path = root / 'RUN_PROTOCOL.json'
    require(path.exists() and read(path) == p, 'Dictionary protocol must equal saved protocol')
    return p, root, file_hash(path), str(path)


def validate_contract(p):
    require(tuple(p['arms']) == ARMS and len(p['uids']) == len(set(p['uids'])) == 16,
            'Exactly sixteen registered objects and three fixed arms required')
    require(p['all16_sorted_uids'] == sorted(p['uids']), 'Seed index order changed')
    require(p['threads'] == 4 and p['interop_threads'] == 1 and p['device'] == 'cpu', 'CPU4/interop1 required')
    require(p['max_steps'] == 16 and p['max_seconds'] == 3600, 'Fixed iteration/safety budgets changed')
    require(tuple(p['snapshot_steps']) == SNAPSHOTS and tuple(p['observation_steps']) == OBSERVATIONS,
            'Snapshot/observation grids changed')
    require(p['optimizer_seed_base'] == 1493501 and p['evaluation_seed_bases'] == [1593501, 1693501]
            and p['points_each_direction'] == 8192 and p['repeats'] == 3, 'Seed/MC/repeat contract changed')
    require(set(p['data']) == set(p['uids']) and set(p['model_bindings']) == set(ARMS[1:]), 'Input bindings incomplete')


def balanced_order(p):
    """All three arms remain adjacent for each UID; their positions rotate."""
    validate_contract(p)
    records = []
    uids = p['all16_sorted_uids']
    for repeat in range(3):
        for uid in uids[repeat:] + uids[:repeat]:
            index = uids.index(uid)
            for j in range(3):
                records.append({'uid': uid, 'arm': ARMS[(j + repeat + index) % 3], 'repeat': repeat})
    if 'case_order' in p:
        require(records == p['case_order'], 'Protocol Latin order differs')
    return records


def case_folder(run_root, uid, arm, repeat):
    return Path(run_root) / 'cases' / uid / arm / f'repeat_{repeat}'


def metrics_hash(metrics):
    return hashlib.sha256(json.dumps(metrics, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def geometry_hash(mesh):
    return array_hash(mesh['u'], mesh['r'], mesh['vertices'], mesh['faces'])


def warm_process():
    """Once per interpreter; not a cold-start or isolated hardware benchmark."""
    global _WARMUP
    if _WARMUP is None:
        start = time.perf_counter()
        require(not torch.cuda.is_initialized(), 'CPU-only runner')
        torch.set_num_threads(4)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        # Deterministic synthetic arithmetic only, with no parent/model reads.
        x = torch.arange(192, dtype=torch.float64).reshape(64, 3) / 192
        _ = x.T @ x
        _ = np.linalg.inv(np.array([[2., .1, 0.], [.1, 3., .2], [0., .2, 4.]]))
        _WARMUP = {'scope': 'warm-process/per-object-includes-load; not cold start',
                   'common_dependency_import_seconds': _IMPORT_SECONDS,
                   'synthetic_process_warmup_seconds': time.perf_counter() - start,
                   'recorded_once_per_process': True, 'torch_threads': 4, 'interop_threads': 1,
                   'cuda_initialized': torch.cuda.is_initialized()}
    require(torch.get_num_threads() == 4 and torch.get_num_interop_threads() == 1
            and not torch.cuda.is_initialized(), 'Process threading/device changed')
    return copy.deepcopy(_WARMUP)


def verify_sources(p, root):
    bindings = p.get('bound_files_sha256', {})
    require(bindings, 'Protocol must bind frozen source files')
    for path, sha in bindings.items():
        require(file_hash(path_from(root, path)) == sha, 'Bound source changed: ' + path)
    return dict(bindings)


def load_data(p, root, uid):
    binding = p['data'][uid]; path = path_from(root, binding['path'])
    with np.load(path, allow_pickle=False) as z:
        data = {k: np.array(z[k]) for k in ('logr', 'base_u', 'faces', 'ref_vertices', 'ref_faces')}
    require(data['logr'].shape == (2050,) and data['base_u'].shape == (258, 3)
            and data['base_u'].dtype == np.float64 and data['faces'].shape == (512, 3)
            and data['faces'].dtype == np.int32, 'Data schema changed')
    require(all(np.isfinite(v).all() for v in data.values()), 'Nonfinite condition')
    return data


def load_model(p, root, arm):
    if arm == 'fixed':
        return None
    binding = p['model_bindings'][arm]; path = path_from(root, binding['path'])
    state = torch.load(path, map_location='cpu', weights_only=False)
    require('model' in state, 'Missing model-only contract')
    require(not ({'optimizer', 'scheduler', 'rng', 'sampler', 'trace'} & set(state)),
            'Full training state is forbidden in timed model-only load')
    require(model_sha256(state['model']) == binding['model_sha256'], 'Model tensor identity mismatch')
    require(all(v.dtype == torch.float32 and v.device.type == 'cpu' for v in state['model'].values()), 'Model dtype/device changed')
    model = DomainVelocity().cpu().float()
    model.load_state_dict(state['model'], strict=True)
    model.eval().requires_grad_(False)
    require(sum(x.numel() for x in model.parameters()) == 29907, 'Architecture changed')
    return model


def reconstruct_incumbents(result):
    """Original strict accepted-feasible loss rule, never independent MC."""
    trace, snapshots, stats = result['trace'], result['snapshots'], result['stats']
    n = stats['iterations']
    require([r['iteration'] for r in trace] == list(range(n + 1)), 'Trace incomplete/reordered')
    require([s['iteration'] for s in snapshots] == list(range(n + 1)), 'Need every actual current snapshot')
    best, best_loss, reconstructed = None, None, []
    for iteration, (row, snapshot) in enumerate(zip(trace, snapshots)):
        require(snapshot['internal_loss'] == row['internal_loss'] and snapshot['feasible'] == row['feasible'],
                'Snapshot/trace disagree')
        eligible = bool(row['feasible']) and (iteration == 0 or bool(row['accepted']))
        if eligible and (best is None or row['internal_loss'] < best_loss):
            best, best_loss = iteration, row['internal_loss']
        require(row['best_feasible_internal_loss'] == best_loss, 'Original incumbent loss mismatch')
        chosen = snapshot if best is None else snapshots[best]
        reconstructed.append({'iteration': iteration, 'best_feasible_iteration': best,
            'selected_current_iteration': iteration if best is None else best,
            'u': chosen['u'], 'vertices': chosen['vertices'], 'faces': chosen['faces']})
    require(best == stats['best_feasible_iteration'], 'Final incumbent identity differs')
    require(np.array_equal(reconstructed[-1]['u'], result['u'])
            and np.array_equal(reconstructed[-1]['vertices'], result['vertices'])
            and np.array_equal(reconstructed[-1]['faces'], result['faces']), 'Reconstructed final differs bitwise')
    return reconstructed


def pack_mesh(storage, prefix, mesh):
    for key in ('u', 'r', 'vertices', 'faces'):
        storage[prefix + '_' + key] = np.array(mesh[key], copy=True)


def source_identity(p, uid, arm, repeat, protocol_sha):
    return {'uid': uid, 'arm': arm, 'repeat': repeat, 'protocol_sha256': protocol_sha,
            'data_file_sha256': p['data'][uid]['sha256'],
            'model_sha256': None if arm == 'fixed' else p['model_bindings'][arm]['model_sha256'],
            'model_file_sha256': None if arm == 'fixed' else p['model_bindings'][arm]['file_sha256']}


def planned_observations(identity):
    return [{**identity, 'iteration': step, 'status': 'unavailable_not_reached',
             'case_elapsed_seconds': None, 'audit_completed_case_seconds': None, 'report': None, 'mesh_prefix': None}
            for step in OBSERVATIONS]


def partial_optimizer_state(exc):
    """Retain already-executed records from an exceptional original call."""
    tb = exc.__traceback__
    recovered = None
    while tb is not None:
        frame = tb.tb_frame
        values = frame.f_locals
        if frame.f_code.co_name == 'optimize_layout' and all(k in values for k in ('trace', 'attempts', 'snapshots')):
            recovered = {'trace': copy.deepcopy(values['trace']),
                         'attempts': copy.deepcopy(values['attempts']),
                         'snapshots': copy.deepcopy(values['snapshots']),
                         'stats': {'status': 'runtime_exception',
                             'iterations': values['trace'][-1]['iteration'] if values['trace'] else None,
                             'inflight_iteration': values.get('iteration'),
                             'best_feasible_iteration': values.get('best_iteration'),
                             'max_steps': values.get('max_steps'),
                             'no_fabricated_completion': True}}
        tb = tb.tb_next
    return recovered


def retain_prefix_evidence(task, storage, events, result):
    """Failure paths keep executed arrays instead of relabeling them missing."""
    snapshots = [] if result is None else result['snapshots']
    lookup = {s['iteration']: s for s in snapshots}
    if snapshots:
        storage['current_iterations'] = np.array([s['iteration'] for s in snapshots], np.int32)
        storage['current_u'] = np.stack([s['u'] for s in snapshots])
        storage['current_vertices'] = np.stack([s['vertices'] for s in snapshots])
        storage['faces'] = snapshots[0]['faces'].copy()
    if events:
        storage['observer_iterations'] = np.array([e['iteration'] for e in events], np.int32)
        storage['observer_current_u'] = np.stack([e['current_u'] for e in events])
        storage['observer_delivered_u'] = np.stack([e['delivered_u'] for e in events])
        if all((e['iteration'] if e['best_feasible_iteration'] is None else e['best_feasible_iteration']) in lookup for e in events):
            storage['observer_delivered_vertices'] = np.stack([
                lookup[e['iteration'] if e['best_feasible_iteration'] is None else e['best_feasible_iteration']]['vertices']
                for e in events])
    event_lookup = {e['iteration']: e for e in events}
    task['snapshot_slots'] = [{'iteration': i,
        'status': 'available' if i in lookup else 'directions_only' if i in event_lookup else 'unavailable_not_executed',
        'array_index': list(lookup).index(i) if i in lookup else None,
        'observer_array_index': list(event_lookup).index(i) if i in event_lookup else None,
        'best_feasible_iteration': event_lookup[i]['best_feasible_iteration'] if i in event_lookup else None}
        for i in SNAPSHOTS]

def run_case(protocol, uid, arm, repeat, out):
    """Write one exclusive TASK.json/ARRAYS.npz after its timed execution.

    ``protocol`` is preferably a RUN_PROTOCOL.json path. ``out`` is a new case
    directory. Exceptions retain all planned observation/snapshot slots and the
    original traceback. No retry, fallback, model choice or early quality stop.
    """
    p, root, protocol_sha, protocol_path = load_protocol(protocol)
    validate_contract(p)
    require(uid in p['uids'] and arm in ARMS and repeat in range(3), 'Unregistered case')
    warmup = warm_process()
    sources = dict(p['bound_files_sha256'])  # Caller verifies full bindings once before/after the batch.
    out = Path(out).resolve(); require(not out.exists(), 'Case output already exists; no retry')
    out.mkdir(parents=True)
    identity = source_identity(p, uid, arm, repeat, protocol_sha)
    write_json(out / 'START.json', {**identity, 'source_sha256': sources})
    task = {**identity, 'status': 'running', 'protocol_file': protocol_path,
        'source_sha256': sources, 'warm_process_record': warmup,
        'timing_scope': 'warm-process/per-object includes model-only/data load and all prior synchronous observations',
        'selection_rule': 'original internal-loss best accepted feasible; current if none',
        'independent_MC_changes_optimizer_or_selection_or_stopping': False,
        'observation_grid': list(OBSERVATIONS), 'planned_snapshot_iterations': list(SNAPSHOTS),
        'baseline': None, 'raw_initialization': None, 'optimizer': None,
        'observations': planned_observations(identity), 'timings': {},
        'failure': None, 'full_budget_is_not_equal_runtime': True,
        'quality_time_is_discrete_observation_not_continuous_first_hit': True,
        'disk_and_protocol_preflight_excluded_from_case_time': True,
        'source_and_file_hash_verification_required_once_before_and_after_batch': True}
    storage, events, result = {}, [], None
    timing = task['timings']; phase = 'start'; start = time.perf_counter()

    def stage(name, function):
        nonlocal phase
        phase = name; begin = time.perf_counter()
        try:
            return function()
        finally:
            timing[name + '_seconds'] = time.perf_counter() - begin

    try:
        data = stage('data_NPZ_read_and_condition', lambda: load_data(p, root, uid))
        support = stage('radial_support_construction', RadialSupport)
        def verify_reference():
            rv, rf = support.mesh(data['logr'])
            require(np.array_equal(rv, data['ref_vertices']) and np.array_equal(rf, data['ref_faces']),
                    'Reference is not the same complete frozen radial field')
        stage('reference_identity', verify_reference)
        index = p['all16_sorted_uids'].index(uid)
        seeds = [base + index for base in p['evaluation_seed_bases']]
        task['mainfield_hash'] = array_hash(data['logr'])
        task['reference_hash'] = array_hash(data['ref_vertices'], data['ref_faces'])
        task['optimizer_seed'] = p['optimizer_seed_base'] + index
        task['metric_seeds'] = seeds
        def evaluate(u, floor=None, baseline_metrics=None):
            return evaluate_candidate(u, data['faces'], data['logr'], support,
                data['ref_vertices'], data['ref_faces'], floor=floor,
                baseline_metrics=baseline_metrics, seeds=seeds, points=8192)
        baseline, baseline_report = stage('fixed_baseline_full_audit', lambda: evaluate(data['base_u']))
        require(baseline_report['audit']['valid'] is True, 'Common Fixed baseline is not exactly valid')
        pack_mesh(storage, 'baseline', baseline)
        task['baseline'] = {'mesh_prefix': 'baseline', 'report': baseline_report,
                            'geometry_hash': geometry_hash(baseline),
                            'metrics_hash': metrics_hash(baseline_report['metrics']),
                            'case_elapsed_seconds': time.perf_counter() - start}
        floor = baseline_report['quality']
        model = stage('portable_model_load', lambda: load_model(p, root, arm))
        if arm == 'fixed':
            raw, elapsed = stage('raw_initialization', lambda: make_fixed(support, data['logr'], data['base_u'], data['faces']))
            raw_meta = {'generation_seconds': elapsed, 'network_used': False}
        else:
            raw, raw_meta = stage('raw_initialization', lambda: predict_raw(model, data['logr'], support, data['base_u'], data['faces']))
        pack_mesh(storage, 'raw', raw)
        if 'alpha' in raw:
            storage['raw_alpha'] = np.array(raw['alpha'], copy=True)
        task['raw_initialization'] = {'mesh_prefix': 'raw', 'metadata': raw_meta,
            'geometry_hash': geometry_hash(raw),
            'initialization_finished_case_seconds': time.perf_counter() - start,
            'separately_audited_before_optimizer_setup': False,
            'audit_shared_with_grid0_after_original_optimizer_setup': True}
        optimizer_floor = {'min_angle_deg': floor['min_triangle_angle_deg'],
                           'angle_q05_deg': floor['triangle_min_angle_p05_deg']}

        def on_record(*, iteration, current_u, delivered_u, faces, row, best_feasible_iteration):
            require(iteration == len(events) and iteration in SNAPSHOTS, 'Observer reordered/duplicated iteration')
            require(np.array_equal(faces, data['faces']), 'Observer connectivity differs')
            event = {'iteration': iteration, 'current_u': current_u, 'delivered_u': delivered_u,
                     'best_feasible_iteration': best_feasible_iteration,
                     'original_record': row, 'callback_start_case_seconds': time.perf_counter() - start}
            if iteration in OBSERVATIONS:
                observation = task['observations'][OBSERVATIONS.index(iteration)]
                audit_start = time.perf_counter()
                try:
                    candidate, report = evaluate(delivered_u, floor, baseline_report['metrics'])
                    if iteration == 0:
                        require(np.array_equal(candidate['u'], raw['u'])
                                and np.array_equal(candidate['r'], raw['r'])
                                and np.array_equal(candidate['vertices'], raw['vertices']), 'Grid0 differs from saved raw')
                    prefix = f'observed_{iteration}'
                    pack_mesh(storage, prefix, candidate)
                    observation.update(status='success', report=report, mesh_prefix=prefix,
                        candidate_best_feasible_iteration=best_feasible_iteration,
                        original_optimizer_trace=row,
                        candidate_hash=array_hash(candidate['vertices'], data['faces']),
                        geometry_hash=geometry_hash(candidate), metrics_hash=metrics_hash(report['metrics']))
                except Exception as exc:
                    # Evaluation cannot feed back into optimizer choices or stop it.
                    observation.update(status='audit_error', error=repr(exc), traceback=traceback.format_exc(),
                        candidate_best_feasible_iteration=best_feasible_iteration,
                        original_optimizer_trace=row)
                observation['audit_callback_seconds'] = time.perf_counter() - audit_start
                observation['case_elapsed_seconds'] = time.perf_counter() - start
                observation['audit_completed_case_seconds'] = observation['case_elapsed_seconds'] if observation['status'] == 'success' else None
                observation['all_prior_observation_costs_included'] = True
            event['callback_end_case_seconds'] = time.perf_counter() - start
            events.append(event)

        result = stage('optimizer_wall_with_observation_audits', lambda: optimize_layout_observed(
            data['logr'], raw['u'], data['faces'], optimizer_floor, device='cpu', max_steps=16,
            max_seconds=3600, seed=task['optimizer_seed'], snapshot_steps=SNAPSHOTS, on_record=on_record))
        task['optimizer'] = {'stats': result['stats'], 'trace': result['trace'], 'attempts': result['attempts']}
        def final_return_check():
            # Necessary final materialization after the unchanged function returns.
            final_r = support.query(data['logr'], result['u'])
            final_v = result['u'] * final_r[:, None]
            require(np.array_equal(final_v, result['vertices']), 'Original optimizer final radius stale')
            incumbents = reconstruct_incumbents(result)
            require(len(events) == len(incumbents), 'Observer coverage differs from original trace')
            for event, chosen in zip(events, incumbents):
                require(event['best_feasible_iteration'] == chosen['best_feasible_iteration']
                        and np.array_equal(event['delivered_u'], chosen['u']), 'Observer changed original incumbent')
                require(np.array_equal(event['current_u'], result['snapshots'][event['iteration']]['u']), 'Current snapshot differs')
            if result['stats']['iterations'] == 16:
                final_observation = task['observations'][-1]
                if final_observation['status'] == 'success':
                    require(np.array_equal(storage['observed_16_u'], result['u'])
                            and np.array_equal(storage['observed_16_r'], final_r)
                            and np.array_equal(storage['observed_16_vertices'], final_v), 'Grid16 is not the exact original final')
            storage.update(final_u=result['u'].copy(), final_r=final_r.copy(), final_vertices=final_v.copy(),
                           faces=np.asarray(result['faces']).copy(),
                           current_iterations=np.array([s['iteration'] for s in result['snapshots']], np.int32),
                           current_u=np.stack([s['u'] for s in result['snapshots']]),
                           current_vertices=np.stack([s['vertices'] for s in result['snapshots']]),
                           delivered_u=np.stack([s['u'] for s in incumbents]),
                           delivered_vertices=np.stack([s['vertices'] for s in incumbents]))
            task['final_return'] = {'geometry_hash': geometry_hash({'u': result['u'], 'r': final_r, 'vertices': final_v, 'faces': result['faces']}),
                'mesh_prefix': 'final', 'audited_observation_iteration': 16 if result['stats']['iterations'] == 16 else None,
                'metrics_hash': task['observations'][-1].get('metrics_hash') if result['stats']['iterations'] == 16 else None}
            return incumbents
        incumbents = stage('final_return_query_and_identity', final_return_check)
        task['optimizer'] = {'stats': result['stats'], 'trace': result['trace'], 'attempts': result['attempts']}
        task['snapshot_slots'] = [{'iteration': step, 'status': 'available' if step < len(incumbents) else 'not_executed',
            'array_index': step if step < len(incumbents) else None,
            'best_feasible_iteration': incumbents[step]['best_feasible_iteration'] if step < len(incumbents) else None}
            for step in SNAPSHOTS]
        audit_errors = sum(r['status'] == 'audit_error' for r in task['observations'])
        task['status'] = 'completed' if not audit_errors else 'error'
        task['completed_full_budget'] = result['stats']['iterations'] == 16
        task['audit_error_count'] = audit_errors
        if audit_errors:
            first = next(r for r in task['observations'] if r['status'] == 'audit_error')
            task['failure'] = {'phase': 'synchronous_observation_audit', 'error': first['error'],
                'traceback': first['traceback'], 'case_elapsed_seconds': first['case_elapsed_seconds'],
                'optimizer_still_ran_unchanged_after_audit_error': True}
        task['final_return_matches_reconstructed_incumbent_bitwise'] = True
        task['final_return_matches_audited_grid16_bitwise'] = bool(result['stats']['iterations'] == 16
            and task['observations'][-1]['status'] == 'success')
    except Exception as exc:
        task['status'] = 'error'
        task['completed_full_budget'] = False
        task['failure'] = {'phase': phase, 'error': repr(exc), 'traceback': traceback.format_exc(),
                           'case_elapsed_seconds': time.perf_counter() - start}
        if result is None:
            result = partial_optimizer_state(exc)
        if result is not None:
            task['optimizer'] = {k: result[k] for k in ('stats', 'trace', 'attempts')}
    finally:
        timing['total_end_to_end_seconds'] = time.perf_counter() - start
        task['cuda_initialized'] = torch.cuda.is_initialized()
    retain_prefix_evidence(task, storage, events, result)
    task['observer_events'] = [{k: v for k, v in event.items() if k not in ('current_u', 'delivered_u')} for event in events]
    timing['synchronous_observation_callback_seconds'] = sum(r.get('audit_callback_seconds', 0.) for r in task['observations'])
    timing['completed_full_audit_calls'] = int(task['baseline'] is not None) + sum(r['status'] == 'success' for r in task['observations'])
    timing['planned_full_audit_calls'] = 7
    for observation in task['observations']:
        if observation['status'] == 'unavailable_not_reached':
            observation['failure_or_termination'] = task['failure'] or (task['optimizer'] or {}).get('stats')
    # Persistence and its hashes are explicitly outside total_end_to_end_seconds.
    disk_start = time.perf_counter()
    save_npz(out / 'ARRAYS.npz', **storage)
    task['arrays_file'] = 'ARRAYS.npz'; task['arrays_file_sha256'] = file_hash(out / 'ARRAYS.npz')
    timing['persistence_before_TASK_seconds_excluded'] = time.perf_counter() - disk_start
    write_json(out / 'TASK.json', task)
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True); parser.add_argument('--uid', required=True)
    parser.add_argument('--arm', required=True, choices=ARMS); parser.add_argument('--repeat', type=int, required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    task = run_case(args.protocol, args.uid, args.arm, args.repeat, args.out)
    print(json.dumps({k: task[k] for k in ('uid', 'arm', 'repeat', 'status')}), flush=True)


if __name__ == '__main__':
    main()
