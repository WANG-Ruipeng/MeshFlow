"""Freeze the 16-parent, three-initializer checked Time-to-Quality contract."""
from __future__ import annotations
import argparse, datetime, json, platform, subprocess, time
from pathlib import Path
import numpy as np
import torch
from ..exporter import RadialSupport
from ..io_utils import array_hash, atomic_torch, file_hash, new_dir, save_npz, write_json
from ..layout_student.train import model_sha256
from ..legacy.networks import DomainVelocity

ARMS = ('fixed', 'early_v1_10', 'balanced_10')
OBSERVATIONS = (0, 1, 2, 4, 8, 16)
REPEATS = 3
BUDGETS = (1., 2., 4., 8., 16., 32., 64., 128.)
PIN = 'bd7d8a138465c583694f9f86083ddd2979d10f87'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def order(uids, arms=ARMS, repeats=REPEATS):
    require(len(uids) == len(set(uids)), 'Duplicate parent')
    result = []
    for repeat in range(repeats):
        # Rotate the parent traversal and balance the three within-parent orders.
        rotated = list(uids[repeat:]) + list(uids[:repeat])
        for uid in rotated:
            index = list(uids).index(uid)
            shift = (index + repeat) % len(arms)
            for arm in list(arms[shift:]) + list(arms[:shift]):
                result.append({'uid': uid, 'arm': arm, 'repeat': repeat})
    return result


def sources():
    module = Path(__file__).resolve().parent
    root = module.parent
    required = ('__init__.py', 'prepare.py', 'runner.py', 'timed_eval.py', 'observer_adapter.py', 'orchestrate.py', 'report.py')
    for name in required:
        require((module / name).is_file(), 'Missing executable dependency: ' + name)
    paths = list(module.glob('*.py'))
    names = ['layout_value/optimizer.py', 'layout_value/geometry.py',
             'layout_student/deployment.py', 'layout_student/train.py',
             'legacy/chart.py', 'legacy/common.py', 'legacy/deform.py',
             'legacy/exact_accel.py', 'legacy/geometry_task.py', 'legacy/networks.py',
             'mesh_ops.py', 'exporter.py', 'io_utils.py', 'assets/support.npz']
    return {str(p.resolve()): file_hash(p) for p in paths + [root / n for n in names]}


def validate(p):
    expected = {'arms': list(ARMS), 'threads': 4, 'interop_threads': 1,
                'max_steps': 16, 'max_seconds': 3600,
                'snapshot_steps': list(range(17)), 'observation_steps': list(OBSERVATIONS),
                'repeats': REPEATS, 'optimizer_seed_base': 1493501,
                'evaluation_seed_bases': [1593501, 1693501], 'points_each_direction': 8192,
                'gain_fraction_threshold': .02, 'quality_tolerance_deg': 1e-6,
                'same_time_budgets_seconds': list(BUDGETS), 'expected_cases': 144}
    for key, value in expected.items():
        require(p[key] == value, 'Frozen contract changed: ' + key)
    require(p['uids'] == sorted(set(p['uids'])) and len(p['uids']) == 16, 'Need all16 fixed parents')
    require(p['all16_sorted_uids'] == p['uids'], 'UID index mismatch')
    require(p['case_order'] == order(p['uids']), 'Task order changed')
    require(set(p['model_bindings']) == set(ARMS[1:]), 'Weight set changed')
    require(set(p['data']) == set(p['uids']), 'Data set changed')
    return p


def verify(protocol_path):
    p = validate(read(protocol_path))
    for path, digest in p['bound_files_sha256'].items():
        require(file_hash(path) == digest, 'Bound source/input changed: ' + path)
    return p


def prepare(repo, out):
    repo = Path(repo).resolve()
    require(not torch.cuda.is_initialized(), 'Preparation is CPU-only')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    require(head == PIN, 'Wrong pinned worktree')
    base = repo / 'output'
    balanced = base / 'a_balanced_20260925_01'
    oldcontrols = read(base / 'a_warm_start_controls_20260925_01/RUN_PROTOCOL.json')
    warmp = read(base / 'a_warm_start_20260925_01/RUN_PROTOCOL.json')
    bp = read(balanced / 'RUN_PROTOCOL.json')
    uids = sorted(bp['uids'])
    require(uids == oldcontrols['all16_sorted_uids'] == warmp['all16_sorted_uids'], 'Parent sets differ')
    manifest_path = balanced / 'DATA_MANIFEST.json'
    require(file_hash(manifest_path) == bp['hashes']['data_manifest_sha256'], 'Data manifest changed')
    manifest = read(manifest_path)['files']
    support = RadialSupport()
    originals = {'early_v1_10': oldcontrols['model_bindings']['early_v1_10'],
                 'balanced_10': warmp['model_bindings']['balanced_10']}
    extraction_start = time.perf_counter()
    model_states = {}
    for arm, binding in originals.items():
        require(file_hash(binding['path']) == binding['file_sha256'], 'Original weight changed')
        value = torch.load(binding['path'], map_location='cpu', weights_only=(arm == 'early_v1_10'))
        state = value['model']
        require(model_sha256(state) == binding['model_sha256'], 'Model tensors changed')
        net = DomainVelocity().cpu()
        net.load_state_dict(state, strict=True)
        require(sum(x.numel() for x in net.parameters()) == 29907, 'Model shape changed')
        if arm == 'balanced_10':
            require(value['update'] == 240000 and value['is_predeclared_final'] is True, 'Wrong balanced final')
        model_states[arm] = {k: v.detach().cpu().clone() for k, v in state.items()}
    source_load_validate_extract_seconds = time.perf_counter() - extraction_start
    frozen_sources = sources()
    # Validate all source data before creating a new run directory.
    data_arrays, identities = {}, {}
    for uid in uids:
        path = balanced / 'data' / (uid + '.npz')
        require(file_hash(path) == manifest['data/' + uid + '.npz'], 'Original data changed')
        with np.load(path, allow_pickle=False) as z:
            d = {k: np.array(z[k]) for k in ('logr', 'base_u', 'faces', 'ref_vertices', 'ref_faces')}
        require(d['logr'].shape == (2050,) and d['base_u'].shape == (258, 3)
                and d['faces'].shape == (512, 3), 'Input shape mismatch')
        rv, rf = support.mesh(d['logr'])
        require(np.array_equal(rv, d['ref_vertices']) and np.array_equal(rf, d['ref_faces']),
                'Reference is not the exact same frozen radial surface')
        require(all(np.isfinite(v).all() for v in d.values()), 'Nonfinite data')
        data_arrays[uid] = d
        identities[uid] = {'source_path': str(path), 'source_sha256': file_hash(path),
                           'mainfield_hash': array_hash(d['logr']),
                           'reference_hash': array_hash(rv, rf),
                           'split': 'train' if uid in bp['train_uids'] else 'internal_check_seen'}
    out = new_dir(out).resolve()
    data, models = {}, {}
    conversion_start = time.perf_counter()
    for arm, state in model_states.items():
        path = out / 'models' / (arm + '.pt')
        atomic_torch(path, {'model': state})
        models[arm] = {'path': str(path), 'file_sha256': file_hash(path),
                       'model_sha256': model_sha256(state), 'source': originals[arm],
                       'parameter_count': 29907, 'seed': 10,
                       'training_updates': 3000 if arm == 'early_v1_10' else 240000}
    conversion_seconds = time.perf_counter() - conversion_start
    for uid, d in data_arrays.items():
        path = out / 'data' / (uid + '.npz')
        # Deliberately no teacher, proxy or per-parent optimized layout in this input.
        save_npz(path, **d)
        data[uid] = {'path': str(path), 'sha256': file_hash(path), **identities[uid]}
    bound = frozen_sources
    provenance = [balanced / 'RUN_PROTOCOL.json', manifest_path,
                  base / 'a_warm_start_controls_20260925_01/RUN_PROTOCOL.json',
                  base / 'a_warm_start_20260925_01/RUN_PROTOCOL.json',
                  base / 'REFERENCE_SEMANTICS_CORRECTION_20260925.md',
                  base / 'a_ttq_asset_audit_20260925_01/AUDIT.json']
    for path in provenance:
        bound[str(path)] = file_hash(path)
    for arm in models:
        for path, digest in [(models[arm]['path'], models[arm]['file_sha256']),
                             (originals[arm]['path'], originals[arm]['file_sha256'])]:
            bound[str(path)] = digest
    for value in data.values():
        bound[value['path']] = value['sha256']
        bound[value['source_path']] = value['source_sha256']
    p = {'schema': 'A-TIME-TO-QUALITY16-v1', 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'repository': str(repo), 'run_root': str(out), 'head': head,
         'scope': 'EXPLORATORY_CLOSED16_CHECKED_INITIALIZER_PLUS_UNCHANGED_OPTIMIZER',
         'uids': uids, 'all16_sorted_uids': uids, 'arms': list(ARMS),
         'model_bindings': models, 'data': data, 'threads': 4, 'interop_threads': 1,
         'device': 'cpu', 'max_steps': 16, 'max_seconds': 3600,
         'snapshot_steps': list(range(17)), 'observation_steps': list(OBSERVATIONS),
         'repeats': REPEATS, 'expected_cases': 144, 'case_order': order(uids),
         'optimizer_seed_base': 1493501, 'evaluation_seed_bases': [1593501, 1693501],
         'points_each_direction': 8192, 'gain_fraction_threshold': .02,
         'quality_tolerance_deg': 1e-6, 'same_time_budgets_seconds': list(BUDGETS),
         'target': 'exact-valid actual512, both Fixed angle floors, both independent same-field RMS improvements >=2%',
         'reference': 'same frozen2050node4096face radial field, not original asset',
         'selection': 'unchanged original best-feasible INTERNAL-loss incumbent at each fixed prefix; never MC selection',
         'observation_policy': 'audit at fixed0/1/2/4/8/16, continue all16 regardless of audit outcomes',
         'checked_TTQ': 'first observed passing fixed prefix; measured wall includes ALL earlier observation audits',
         'not_continuous_first_hit': True, 'audit_results_change_optimizer_or_termination': False,
         'same_time_rule': 'latest completed checked incumbent at budget, not best MC/ever-success candidate',
         'budget_interpretation': '16-step algorithm cap is not equal elapsed cost; compare checked wall curves and final true totals',
         'failure_denominator': 16, 'runtime_failure_is_not_censored_success': True,
         'no_Fixed_fallback': True, 'no_retry_or_training': True, 'no_geometry_or_MC_cache': True,
         'baseline_audits_per_case': 1, 'candidate_audits_per_successful_case': 6,
         'timing_scope': 'warm Python process; per-case data/support/model-only load, baseline floor+MC, proxy32, full optimizer, all6 audits, final requery/identity included',
         'disk_output_and_source_preflight_excluded_from_case_but_run_wall_recorded': True,
         'process_start_import_and_one_time_synthetic_warmup_reported_separately': True,
         'cold_process_or_cold_OS_cache_claim': False, 'model_conversion_seconds': source_load_validate_extract_seconds + conversion_seconds,
         'source_load_validate_extract_seconds': source_load_validate_extract_seconds,
         'model_serialization_hash_seconds': conversion_seconds,
         'model_conversion_is_once_deployment_setup_not_percase_inference': True,
         'repeats_are_timing_replicates_not_new_objects_or_training_seeds': True,
         'geometry_must_match_across_repeats': True, 'single_seed10_scope': True,
         'no_seed11_cherry_picking_or_cross_seed_claim': True, 'all16_development_seen': True,
         'historical_comparisons_not_primary_timing': True,
         'host_gate': {'no_cl_processes': True, 'minimum_free_physical_GiB': 12,
                       'minimum_free_commit_GiB': 12, 'poll_seconds': 10,
                       'quiet_samples_before_start': 3, 'resource_interference_recorded': True,
                       'compiler_process_names': ['cl', 'clang-cl', 'link', 'lld-link'],
                       'maximum_host_cpu_percent': 25,
                       'host_checks_only_between_cases': True, 'continuous_isolation_certified': False},
         'bound_files_sha256': bound,
         'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                     'numpy': np.__version__, 'platform': platform.platform()}}
    validate(p)
    write_json(out / 'RUN_PROTOCOL.json', p)
    write_json(out / 'PREPARED.json', {'status': 'READY_FOR_TEST_GATE_NOT_STARTED',
              'protocol_sha256': file_hash(out / 'RUN_PROTOCOL.json'), 'cuda_initialized': torch.cuda.is_initialized()})
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='.')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    p = prepare(args.repo, args.out)
    print(json.dumps({'status': 'PREPARED', 'cases': p['expected_cases'], 'run': p['run_root']}))


if __name__ == '__main__':
    main()