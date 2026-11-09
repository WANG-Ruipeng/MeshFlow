"""CPU-only, immutable asset/stream registration for the bounded repeat.

No model loading, forward, matching, scoring, or CONFIRM result access.  Old
artifacts are read only; only 16 predeclared inference Gaussians are newly drawn.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import datetime
import difflib
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from experiments.mf_h1_local0.chair_control_feedback_auto_v1.assets import (
    ROOT, SOURCE, CHAIR, arrays, array_hash, bits, digest, freeze_json,
    freeze_npz, make_bank, read_json, resolve,
)
from experiments.mf_h1_local0.chair_hybrid_coupling_v1.coupling import array_hash as pool_array_hash
from .stream import SEED, TRAIN_SEED_NAMESPACE, prepare_training_plan

OUT = ROOT / 'native_t1/runs/chair_capped500_gradbudget_v1'
OLD = ROOT / 'native_t1/runs/chair_control_feedback_auto_v1'
OLD_CODE = ROOT / 'experiments/mf_h1_local0/chair_control_feedback_auto_v1'
OFFLINE = ROOT / 'native_t1/runs/chair_capped500_offline_v1'
START_SHA = 'd640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80'
GPU_LOCK = '/tmp/wrp_meshflow_jedge5_diag_gpu0.lock'
AUX_TEACHER_SEED = 20261008
HISTORICAL_LABELS = {
    'START': 'START',
    'HIST_TEACHER500': 'explore_REPLAY_FM_CONTROL_500',
    'HIST_CAPPED500': 'explore_TRAJECTORY_REPLAY_CAPPED_500',
}
LABELS = list(HISTORICAL_LABELS) + ['T500', 'T1000', 'L500', 'L1000', 'G500', 'G1000']
FRESH_CANDIDATES = (2, 3)


def derive_seed(namespace):
    return int.from_bytes(hashlib.sha256(namespace.encode('utf-8')).digest()[:4], 'little')


def _copy_bytes(source, destination):
    source, destination = Path(source), Path(destination)
    data = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != data:
            raise ValueError('Existing frozen copy differs: ' + str(destination))
    else:
        with destination.open('xb') as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
    return destination


def _free_noise_hash(bank):
    # Canonical free triangles prevent padded/known-slot shape conventions from
    # hiding a repeated actual free Gaussian.
    z = np.asarray(bank['z'], np.float32).reshape(-1, 9)
    known = np.asarray(bank['known_mask'], bool).reshape(-1)
    valid = np.asarray(bank['valid_mask'], bool).reshape(-1)
    return array_hash(z[valid & ~known].reshape(-1, 3, 3))


def _validate_bank(bank, row):
    n, k = int(row['N']), int(row['K'])
    assert bank['z'].dtype == np.float32 and bank['z'].shape == (1, n, 9)
    assert bank['C'].dtype == np.float32 and bank['C'].shape == (k, 3, 3)
    assert np.array_equal(bank['known_mask'], np.arange(n) < k)
    assert np.array_equal(bank['valid_mask'], np.ones(n, bool))
    assert int(bank['N']) == n and int(bank['K']) == k
    assert np.isfinite(bank['z']).all() and np.isfinite(bank['C']).all()
    assert not np.any(bank['z'][0, :k])
    if row.get('noise_sha256'):
        assert array_hash(bank['z']) == row['noise_sha256']


def prepare(out=OUT):
    out = Path(out).resolve()
    if out != OUT.resolve():
        raise ValueError('Registration must use its isolated experiment directory')
    completed = out / 'registration/receipt.json'
    if completed.exists():
        receipt = read_json(completed)
        for p, expected in receipt['registered_artifacts'].items():
            if digest(p) != expected:
                raise ValueError('Frozen registration artifact changed: ' + p)
        return dict(receipt, invocation_status='CACHE_VALIDATED_NO_REDRAW')
    tick = time.perf_counter()
    sources, copies = {}, []

    def track(path, expected=None):
        path = resolve(path)
        key = str(path)
        if key not in sources:
            sources[key] = dict(sha256=digest(path), bytes=path.stat().st_size)
        if expected is not None and sources[key]['sha256'] != expected:
            raise ValueError('Protected source hash changed: ' + key)
        return path

    def copied(source, destination, expected=None):
        source = track(source, expected)
        destination = _copy_bytes(source, destination)
        assert digest(destination) == sources[str(source)]['sha256']
        copies.append(dict(source=str(source), destination=str(destination),
                           sha256=digest(destination), bytes=destination.stat().st_size))
        return destination

    # Exact historical manifest snapshots keep all original identity fields.
    manifests = {}
    for name in ('TRAINING', 'TRAIN_FEEDBACK', 'DEV', 'CONFIRM'):
        p = copied(OLD / 'manifests' / (name + '.json'), out / 'manifests' / (name + '.json'))
        manifests[name] = read_json(p)
    old_protocol = read_json(track(OLD / 'protocol.json'))
    for name, value in old_protocol['manifest_sha256'].items():
        assert digest(OLD / 'manifests' / (name + '.json')) == value
    assert len(manifests['TRAINING']['parents']) == 32
    assert len(manifests['TRAINING']['tasks']) == 453
    assert len(manifests['DEV']['rows']) == len(manifests['TRAIN_FEEDBACK']['rows']) == 16
    assert len(manifests['CONFIRM']['rows']) == 32
    train_parents = set(manifests['TRAINING']['parents'])
    dev_parents = {r['uid'] for r in manifests['DEV']['rows']}
    confirm_parents = {r['uid'] for r in manifests['CONFIRM']['rows']}
    assert len(dev_parents) == len(confirm_parents) == 8
    assert not (train_parents & dev_parents or train_parents & confirm_parents or dev_parents & confirm_parents)
    for relative in ('CONTROL_FEEDBACK_AUTONOMOUS_REPORT.md', 'REPRODUCE.md',
                     'hypothesis_register.md', 'decisions/H3_recipe.json',
                     'derived/checkpoints/checkpoint_manifest.json'):
        copied(OLD / relative, out / 'registration/history' / relative)
    track(ROOT / 'native_t1/docs/chair_start.md')
    for relative in ('distribution/DISTRIBUTION_MEMO.md', 'distribution/TAIL_AND_DRIFT_MEMO.md',
                     'distribution/CAPPED500_parent_digest.csv', 'distribution/CAPPED500_output_digest.csv',
                     'dynamics/DYNAMICS_MEMO.md'):
        track(OFFLINE / relative)
    for p in ('stream.py', 'training.py', 'replay.py', 'loss_balance.py', 'assets.py', 'evaluation.py'):
        track(OLD_CODE / p)
    for p in ('experiments/mf_h1_local0/chair_hybrid_coupling_v1/coupling.py',
              'experiments/mf_h1_local0/chair_domain_control_v1/prepare.py',
              'experiments/mf_h1_local0/control_mvp48/train_stream.py', 'utils/ot_utils.py'):
        track(ROOT / p)

    # Only three selected weights are read as bytes, never deserialized.
    historical = read_json(OLD / 'derived/checkpoints/checkpoint_manifest.json')
    checkpoints = {}
    for label, old_label in HISTORICAL_LABELS.items():
        if label == 'START':
            entry = historical['START']
            path = ROOT / 'native_t1/checkpoints/chair_hybrid_start_total5000.pt'
            expected = START_SHA
        else:
            entry = next(x for x in historical['checkpoints'] if x['filename'] == old_label + '.pt')
            assert entry['feedback_step'] == 500 and entry['conditional_total_step'] == 5500
            path, expected = resolve(entry['path']), entry['actual_checkpoint_sha256']
            track(entry['sidecar_path'], entry['sidecar_sha256'])
        track(path, expected)
        assert path.stat().st_size == entry['bytes']
        checkpoints[label] = dict(path=str(path), sha256=expected, checkpoint_sha256=expected,
            bytes=path.stat().st_size, source_label=old_label, feedback_step=0 if label == 'START' else 500,
            conditional_total_step=5000 if label == 'START' else 5500,
            role='COMMON_TRAINING_START' if label == 'START' else 'HISTORICAL_EVALUATION_ONLY',
            file_hash_actually_recomputed=True, payload_loaded=False)
    freeze_json(out / 'registration/checkpoints.json', checkpoints)

    # Original pool payload and correspondence maps are preserved byte for byte.
    pool_path = copied(OLD / 'feedback/pool_v1/manifest.json', out / 'feedback/pool_v1/manifest.json')
    pool = read_json(pool_path)
    recipe = read_json(OLD / 'decisions/H3_recipe.json')
    assert digest(pool_path) == recipe['feedback_pool_sha256']
    assert pool['replay_source_sha256'] == digest(OLD_CODE / 'replay.py')
    assert pool['status'] == 'PASS' and len(pool['records']) == 16 and len(pool['sorted_parents']) == 8
    assert set(pool['sorted_parents']).issubset(train_parents)
    for p, expected in pool['source_hashes'].items():
        track(p, expected)
    for record in pool['records']:
        p = copied(OLD / 'feedback/pool_v1' / record['payload'],
                   out / 'feedback/pool_v1' / record['payload'], record['payload_sha256'])
        payload = arrays(p)
        assert {k: pool_array_hash(v) for k, v in payload.items()} == record['array_sha256']
        assert 'teacher_x0_20261008' in payload
    selected = {v['selected_request_id'] for queue in pool['queues'].values() for v in queue.values()}
    assert len(selected) == 12
    queue_plan = []
    records_by_id = {r['request_id']: r for r in pool['records']}
    for step in range(1, 1001):
        queue = ('S', 'I', 'E')[(step - 1) % 3]
        uid = pool['sorted_parents'][((step - 1) // 3) % 8]
        rid = pool['queues'][queue][uid]['selected_request_id']
        record = records_by_id[rid]
        time_index = ((step - 1) // 24) % 2
        queue_plan.append(dict(step=step, queue=queue, uid=uid, request_id=rid,
            candidate=record['candidate'], t=(.6, .8)[time_index], state_index=(30, 40)[time_index],
            payload_sha256=record['payload_sha256'], teacher_seed=AUX_TEACHER_SEED,
            teacher_x0_sha256=record['array_sha256']['teacher_x0_20261008'],
            trajectory_state_sha256=record['array_sha256']['xt' + str((30, 40)[time_index])],
            teacher_target_sha256=record['array_sha256']['teacher_target'],
            trajectory_target_sha256=record['array_sha256']['trajectory_target']))
    freeze_json(out / 'registration/auxiliary_queue_plan.json', dict(
        source_pool_path=str(OLD / 'feedback/pool_v1/manifest.json'), source_pool_sha256=digest(pool_path),
        teacher_seed=AUX_TEACHER_SEED, selected_trajectories=12, selected_state_targets=24,
        policy='New namespace changes primary FM stream only; both auxiliary states and all maps remain the original frozen pool.',
        records=queue_plan))

    # Input-only scan of explicitly enumerated historical noise registrations.
    noise_sources = []
    seen_banks = set()

    def register_old_bank(path, expected, scope, row=None):
        path = resolve(path)
        if str(path) in seen_banks:
            return
        seen_banks.add(str(path)); track(path, expected)
        bank = arrays(path)
        if row is not None:
            _validate_bank(bank, row)
        noise_sources.append(dict(path=str(path), sha256=sources[str(path)]['sha256'], scope=scope,
            full_z_sha256=array_hash(bank['z']), free_gaussian_sha256=_free_noise_hash(bank)))

    for name in ('DEV', 'TRAIN_FEEDBACK', 'CONFIRM'):
        for row in manifests[name]['rows']:
            register_old_bank(row['bank_npz'], row['bank_sha256'], 'old_feedback_' + name, row)
            if name != 'CONFIRM':
                track(row['condition_npz'], row['condition_sha256'])
    panels = read_json(track(CHAIR / 'panels.json'))
    for panel in ('A', 'B', 'C'):
        for row in panels[panel]:
            if row.get('input_path'):
                register_old_bank(CHAIR / row['input_path'], row['input_sha256'], 'original_chair_' + panel)
    for run in ('chair_control_usability_quick_v1', 'chair_boundary_attachment_v1'):
        p = track(ROOT / 'native_t1/runs' / run / 'requests.json')
        data = read_json(p); rows = data['rows'] if isinstance(data, dict) else data
        for row in rows:
            if row.get('input_npz') and Path(row['input_npz']).is_file():
                register_old_bank(row['input_npz'], row['input_sha256'], run)
    old_full = {x['full_z_sha256'] for x in noise_sources}
    old_free = {x['free_gaussian_sha256'] for x in noise_sources}

    # Copy exact DEV conditions and original fixed query/reference arrays.
    condition_meta = read_json(track(OLD / 'registration/conditions.json'))
    conditions = []
    c_by_id = {}
    for original in condition_meta:
        if original['pool'] != 'DEV':
            continue
        c = copy.deepcopy(original)
        p = copied(c['condition_npz'], out / 'inputs/conditions' / (c['condition_id'] + '.npz'), c['condition_sha256'])
        c['historical_condition_npz'] = c['condition_npz']; c['condition_npz'] = str(p)
        c['historical_pool'] = c['pool']; c['pool'] = 'OLD_AND_FRESH'
        for key, value in list(c.items()):
            if key.startswith(('legacy_control_reference_', 'legacy_quality_query_')) and key.endswith(('_json', '_npz')):
                track(value, c[key + '_sha256'])
        for name in ('control.json', 'control.npz', 'quality.json', 'quality.npz'):
            copied(OLD / 'references' / c['condition_id'] / name, out / 'references' / c['condition_id'] / name)
        conditions.append(c); c_by_id[c['condition_id']] = c
    assert len(conditions) == 8
    freeze_json(out / 'registration/conditions.json', conditions)

    old_rows, fresh_rows = [], []
    caches = []
    for original in manifests['DEV']['rows']:
        row = copy.deepcopy(original)
        row.update(panel='OLD', pool='OLD', split='OLD', historical_split='DEV', raw_cache=None)
        row.pop('legacy_score_path', None)
        row['condition_npz'] = c_by_id[row['condition_id']]['condition_npz']
        bp = copied(original['bank_npz'], out / 'inputs/inference' / (row['id'] + '.npz'), original['bank_sha256'])
        row['bank_npz'] = row['input_npz'] = str(bp)
        row['historical_cache'] = {}
        bank = arrays(bp); _validate_bank(bank, row)
        for label, old_label in HISTORICAL_LABELS.items():
            rid = row['request_id']
            joined_path = track(OLD / 'evaluation' / old_label / 'DEV/scoring/joined' / (rid + '.json'))
            joined = read_json(joined_path); identity, metrics = joined['identity'], joined['metrics']
            result_path = track(OLD / 'runtime' / old_label / rid / 'result.json', identity['result_sha256'])
            result = read_json(result_path)
            raw_path = track(OLD / 'runtime' / old_label / rid / 'raw.npz', result['raw_sha256'])
            assert identity['bank_sha256'] == row['bank_sha256']
            assert identity['condition_sha256'] == row['condition_sha256']
            assert identity['request_id'] == rid and identity['label'] == old_label and identity['split'] == 'DEV'
            assert identity['raw_sha256'] == result['raw_sha256']
            assert result['checkpoint_sha256'] == checkpoints[label]['sha256'] and result['status'] == 'PASS'
            assert int(result['metadata']['nfe']) == 50
            raw = arrays(raw_path); n, k = row['N'], row['K']
            path = raw['path'].reshape(51, n, 9); output = raw['output'].reshape(n, 9)
            assert bits(output, path[-1]) and np.isfinite(path).all()
            assert bits(raw['C'], bank['C']) and bits(raw['z'], bank['z'])
            assert bits(path[0, k:], bank['z'][0, k:])
            assert all(bits(state[:k], bank['C'].reshape(k, 9)) for state in path)
            cache = dict(status='PASS', source_request_id=rid, source_label=old_label,
                joined_path=str(joined_path), joined_sha256=digest(joined_path),
                raw_path=str(raw_path), raw_sha256=digest(raw_path),
                result_path=str(result_path), result_sha256=digest(result_path),
                checkpoint_sha256=checkpoints[label]['sha256'], bank_sha256=row['bank_sha256'],
                condition_sha256=row['condition_sha256'], C_all51_bitwise=True,
                actual_Gaussian_bitwise=True, N=n, K=k, nfe=50)
            for field in ('vectors_path', 'control_vectors_path'):
                p = track(metrics[field], metrics[field.replace('_path', '_sha256')])
                cache[field] = str(p); cache[field.replace('_path', '_sha256')] = digest(p)
            for field in ('npz', 'ply'):
                p = track(result['export'][field], result['export'][field + '_sha256'])
                cache['export_' + field] = str(p); cache['export_' + field + '_sha256'] = digest(p)
            row['historical_cache'][label] = cache
            caches.append(dict(label=label, request_id=rid, **cache))
        old_rows.append(row)

    for base in [r for r in old_rows if r['candidate'] == 0]:
        for candidate in FRESH_CANDIDATES:
            namespace = f"chair_capped500_gradbudget_v1/eval_fresh/{base['uid']}/{candidate}"
            seed = derive_seed(namespace)
            bank = make_bank(arrays(base['condition_npz'])['C'], base['N'], seed)
            full_hash, free_hash = array_hash(bank['z']), _free_noise_hash(bank)
            assert full_hash not in old_full and free_hash not in old_free
            assert all(full_hash != r['noise_sha256'] and free_hash != r['free_gaussian_sha256'] for r in fresh_rows)
            rid = 'FRESH__' + base['condition_id'] + '__n' + str(candidate)
            row = {k: copy.deepcopy(v) for k, v in base.items() if k not in (
                'historical_cache', 'legacy_input_npz', 'raw_cache', 'historical_split')}
            bp = out / 'inputs/inference' / (rid + '.npz'); freeze_npz(bp, bank)
            row.update(id=rid, request_id=rid, panel='FRESH', pool='FRESH', split='FRESH',
                candidate=candidate, noise_index=candidate, noise_seed=seed, gaussian_seed=seed,
                seed_namespace=namespace, seed_derivation='SHA256 UTF8 first4 bytes little-endian',
                bank_npz=str(bp), input_npz=str(bp), bank_sha256=digest(bp), input_sha256=digest(bp),
                noise_sha256=full_hash, free_gaussian_sha256=free_hash, historical_cache={}, raw_cache=None)
            _validate_bank(bank, row); fresh_rows.append(row)
    assert len(old_rows) == len(fresh_rows) == 16
    for panel, rows in (('OLD', old_rows), ('FRESH', fresh_rows), ('ALL', old_rows + fresh_rows)):
        freeze_json(out / 'manifests' / (panel + '.json'), dict(
            schema='chair_capped500_gradbudget_evaluation_v1', panel=panel, pool=panel, rows=rows,
            parents=8, candidates_per_parent=4 if panel == 'ALL' else 2, labels=LABELS,
            interpretation='Same observed DEV parents; FRESH is new inference noise, not new shapes'))
    freeze_json(out / 'registration/historical_cache.json', dict(status='PASS', count=len(caches), rows=caches))
    freeze_json(out / 'registration/noise_identity_audit.json', dict(status='PASS', new_gaussians=16,
        history_banks=len(noise_sources), history_unique_full_hashes=len(old_full),
        history_unique_free_hashes=len(old_free), duplicate_full_hashes=[], duplicate_free_hashes=[],
        history=noise_sources, fresh=[{k: r[k] for k in ('id', 'uid', 'candidate', 'noise_seed',
            'seed_namespace', 'noise_sha256', 'free_gaussian_sha256', 'bank_npz', 'bank_sha256')} for r in fresh_rows],
        CONFIRM_scope='Read only original manifest and input Gaussian identity; no scores, images, model or generation',
        scope='Explicit original Chair A/B/C inputs, quick usability and boundary registered inputs, old feedback DEV/TRAIN_FEEDBACK/CONFIRM banks; no whole-workspace scan'))

    # Generate just the registered stateless plan; do not compute any OT/input tensor.
    track(CHAIR / 'task_manifest.json', manifests['TRAINING']['task_manifest_sha256'])
    track(SOURCE / 'registration/train_data_binding.json')
    plan_receipt = prepare_training_plan(SOURCE, CHAIR, out / 'streams/repeat', seed=SEED)
    plan = read_json(plan_receipt['path'])
    old_plan_path = track(OLD / 'streams/explore/train_plan.json')
    old_plan = read_json(old_plan_path)
    assert SEED == derive_seed(TRAIN_SEED_NAMESPACE) and SEED not in (20261008, 20261009)
    assert plan['records'] != old_plan['records'] and len(plan['records']) == 8000
    assert set(Counter(r['uid'] for r in plan['records']).values()) == {250}
    assert all(len({r['uid'] for r in plan['records'][i:i+8]}) == 8 for i in range(0, 8000, 8))
    assert all(sum(r['alpha'] == 1. for r in plan['records'][i:i+8]) == 4 for i in range(0, 8000, 8))
    assert all(.9 <= r['alpha'] <= 1.1 for r in plan['records'])
    old_stream = OLD_CODE / 'stream.py'; new_stream = Path(__file__).with_name('stream.py')
    diff = ''.join(difflib.unified_diff(old_stream.read_text().splitlines(True), new_stream.read_text().splitlines(True),
                                      fromfile=str(old_stream), tofile=str(new_stream)))
    diff_path = out / 'registration/stream.diff'
    if diff_path.exists():
        assert diff_path.read_text() == diff
    else:
        diff_path.write_text(diff, encoding='utf-8')
    freeze_json(out / 'registration/training_plan.json', dict(status='PASS', **plan_receipt,
        namespace=TRAIN_SEED_NAMESPACE, seed_derivation='SHA256 UTF8 first4 bytes little-endian',
        old_explore_seed=old_plan['seed'], old_plan_sha256=digest(old_plan_path),
        all8000_seed_tuples_changed=all(tuple(a[k] for k in ('gaussian_seed', 'epsilon_seed', 't_seed', 'permutation_seed')) !=
            tuple(b[k] for k in ('gaussian_seed', 'epsilon_seed', 't_seed', 'permutation_seed')) for a,b in zip(plan['records'], old_plan['records'])),
        exposure_per_parent=250, distinct_parents_per_batch=8, alpha1_per_batch=4,
        actual_OT_calls=0, old_stream_sha256=digest(old_stream), new_stream_sha256=digest(new_stream),
        stream_diff_sha256=digest(diff_path)))
    freeze_json(out / 'registration/copies.json', copies)
    freeze_json(out / 'registration/source_files.json', sources)
    artifacts = {str(p): digest(p) for folder in ('registration', 'manifests', 'inputs', 'references', 'feedback')
                 for p in (out / folder).rglob('*') if p.is_file() and p.suffix != '.log'}
    artifacts[plan_receipt['path']] = plan_receipt['sha256']
    artifacts[str(Path(__file__))] = digest(__file__)
    artifacts[str(new_stream)] = digest(new_stream)
    receipt = dict(status='PASS', schema='chair_capped500_gradbudget_registration_v1',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-tick,
        main_seed=SEED, auxiliary_teacher_seed=AUX_TEACHER_SEED, labels=LABELS,
        OLD=16, FRESH=16, historical_cache_verified=48, training_parents=32, training_tasks=453,
        training_updates=1000, training_microbatches=8000, pool_trajectories=16,
        selected_auxiliary_trajectories=12, selected_auxiliary_state_targets=24,
        confirm_manifest_and_noise_identity_only=True, CONFIRM_scores_or_images_read=False,
        new_Gaussian_draws=16, CPU_OT_calls=0, model_loads=0, model_forwards=0,
        model_backwards=0, optimizer_updates=0, GPU_lock_acquired=False, gpu_lock=GPU_LOCK,
        source_files=len(sources), source_bytes_hashed=sum(v['bytes'] for v in sources.values()),
        source_sha256=digest(__file__), stream_sha256=digest(new_stream),
        checkpoint_manifest=str(out / 'registration/checkpoints.json'),
        registered_artifacts=artifacts)
    freeze_json(completed, receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', default=str(OUT))
    args = parser.parse_args()
    result = prepare(args.out)
    print(json.dumps({k: v for k, v in result.items() if k != 'registered_artifacts'}, indent=2, sort_keys=True))
