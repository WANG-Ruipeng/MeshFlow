"""Frozen paired CPU data for R2_USER_CONDITION_FINETUNE_PILOT_V1.

This isolated module imports the original sample/OT mathematics, never a model.
Stream reads materialized arrays only. CPU preparation is bounded to two workers.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import os
import sys
import time

# Bound native numerical pools before NumPy/SciPy are imported by this module.
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
_SRC = Path(__file__).resolve().parents[2] / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
import numpy as np
import scipy
from meshflow_control.data import stream as baseline
from meshflow_control.data import dataset as original_dataset
from meshflow_control.data import io as data_io
from meshflow_control.data.dataset import TrainingDataset, training_time
from meshflow_control.data.io import (array_hash, atomic_json, bits, digest,
    freeze_json, json_hash, read_json, save_npz)
from meshflow_control.training import coupling

ROOT = Path('/mnt/e/MeshFlow-Control-artifacts/experiments/r2_user_condition_finetune_pilot_v1')
SOURCE = Path('/mnt/e/MeshFlow-Control-artifacts/experiments/chair_r2_user_conditions_v1')
DATA = Path('/mnt/e/MeshFlow-Control-artifacts/data')
ARMS = ('R2_CONT_OLD', 'R2_CONT_USER')
NAMESPACE = 'R2_USER_CONDITION_FINETUNE_PILOT_V1/train/v1'
TOTAL_UPDATES = 1000
BATCH_SIZE = 8
SAMPLES_PER_ARM = 8000
HASH_FIELDS = ('x0', 'x1', 'u', 'xt', 'context', 't', 'y', 'known_mask', 'valid_mask',
    'source_face_ids', 'corner_permutations', 'face_permutation',
    'free_gaussian_before_OT', 'free_epsilon', 'free_target_before_OT',
    'ot_face_map', 'ot_corner_map')
INDEX_FIELDS = ('sample_index', 'task_id', 'object_id', 'K', 'alpha', 'ratio', 'step',
    'global_step', 'arm', 'shared_input_sha256', 'input_bytehash', 'label_bytehash',
    'ot_cache_key', 'free_coordinate_count', 'lambda_value')


def domain_seed(domain, *identity):
    payload = json.dumps([NAMESPACE, domain, *identity], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'little')


def source_identity():
    paths = (Path(__file__), Path(baseline.__file__), Path(original_dataset.__file__),
             Path(data_io.__file__), Path(coupling.__file__))
    return {str(p.resolve()): digest(p) for p in paths}


def _arm(arm):
    if arm not in ARMS:
        raise ValueError('Expected R2_CONT_OLD or R2_CONT_USER')
    return arm


def _partition(row):
    n, k = int(row['N']), int(row['K'])
    known = np.asarray(row['source_face_ids'], np.int64)
    free = np.asarray(row['free_source_ids'], np.int64)
    if not (128 <= n <= 256 and 0 < k < n and len(known) == k and len(free) == n-k):
        raise ValueError('Invalid N/K partition')
    if not np.array_equal(np.sort(np.r_[known, free]), np.arange(n)):
        raise ValueError('Source partition is not exact and disjoint')


def input_registration():
    """Read/hash the fixed 25-parent pool; no random arrays or OT here."""
    active_path = SOURCE / 'active_conditions.json'
    handoff_path = SOURCE / 'dataset_handoff_active.json'
    active, handoff = read_json(active_path), read_json(handoff_path)
    manifest_path = DATA / 'train_manifest.json'
    manifest = read_json(manifest_path)
    if digest(active_path) != handoff['prepared_data']['sha256']:
        raise ValueError('Active manifest identity changed')
    if digest(manifest_path) != handoff['future_comparison']['old_arm']['source_manifest_sha256']:
        raise ValueError('Historical training manifest changed')
    sources = {str(p): digest(p) for p in (active_path, handoff_path, manifest_path,
               SOURCE / 'usability_review.json', SOURCE / 'parent_inventory.json')}
    parent_rows = handoff['prepared_data']['common_parent_rows']
    parents = sorted(r['uid'] for r in parent_rows)
    if len(parents) != 25 or len(set(parents)) != 25:
        raise ValueError('Exactly 25 common parents required')
    original_parents = {r['uid']: r for r in manifest['parents']}
    for row in parent_rows:
        old = original_parents[row['uid']]
        path = DATA / old['npz']
        if str(path) != row['npz'] or old['sha256'] != row['sha256'] or old['N'] != row['N']:
            raise ValueError('Two arms do not reference identical M')
        if digest(path) != row['sha256']:
            raise ValueError('Parent file identity changed')
        sources[str(path)] = row['sha256']
    old_tasks = {}
    manifest_tasks = {r['task_id']: r for r in manifest['tasks']}
    old_pools = handoff['future_comparison']['old_arm']['task_pools_by_parent']
    for uid in parents:
        for ratio in ('0.2', '0.4'):
            for registered in old_pools[uid][ratio]:
                task = manifest_tasks[registered['task_id']]
                if task['uid'] != uid or task['status'] != 'READY' or task['role'] != 'train' or float(task['ratio']) != float(ratio):
                    raise ValueError('Old eligible task differs')
                if task['condition_sha256'] != registered['condition_sha256']:
                    raise ValueError('Old condition identity changed')
                path = DATA / task['condition_npz']
                if digest(path) != task['condition_sha256']:
                    raise ValueError('Old condition file changed')
                _partition(task)
                sources[str(path)] = task['condition_sha256']
                old_tasks[task['task_id']] = task
    if Counter(float(r['ratio']) for r in old_tasks.values()) != {0.2: 200, 0.4: 195}:
        raise ValueError('Expected original eligible 200 C20 + 195 C40')
    held = set(active['held_condition_ids'])
    new_tasks = {}
    for row in active['rows']:
        if row['split'] != 'train':
            continue
        if row['uid'] not in parents or row['condition_id'] in held:
            raise ValueError('Active train membership differs or HOLD included')
        _partition(row)
        path = Path(row['npz'])
        if digest(path) != row['sha256']:
            raise ValueError('Active condition file changed')
        parent = original_parents[row['uid']]
        if row['parent_sha256'] != parent['sha256']:
            raise ValueError('New condition parent identity differs')
        with np.load(path, allow_pickle=False) as z, np.load(DATA / parent['npz'], allow_pickle=False) as m:
            full = m['full_target'].reshape(row['N'], 3, 3)
            if not bits(z['C'], full[np.asarray(row['source_face_ids'])]) or not bits(z['Y'], full[np.asarray(row['free_source_ids'])]):
                raise ValueError('Active C/Y does not reproduce exact parent partition')
        sources[str(path)] = row['sha256']
        new_tasks[row['condition_id']] = dict(row, task_id=row['condition_id'], ratio=row['K']/row['N'])
    if len(new_tasks) != 117 or {r['uid'] for r in new_tasks.values()} != set(parents):
        raise ValueError('Expected 117 active C on exactly 25 parents')
    for uid in parents:
        keys = [tuple(r['source_face_ids']) for r in new_tasks.values() if r['uid'] == uid]
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate active source-face set')
    return dict(schema='r2_user_finetune_input_registration_v1', namespace=NAMESPACE,
        source_files=sources, source_sha256=json_hash(sources),
        parents={uid: original_parents[uid] for uid in parents},
        tasks={ARMS[0]: old_tasks, ARMS[1]: new_tasks}, held_condition_ids=sorted(held),
        implementation_sources=source_identity(), numpy_version=np.__version__, scipy_version=scipy.__version__)


def make_plan(registration):
    """Balanced parent/category schedule; no model or Gaussian/OT draw."""
    parents = sorted(registration['parents'])
    rng = {name: np.random.RandomState(domain_seed(name)) for name in
           ('parents', 'slot_shuffle', 'alpha', 'old_conditions', 'user_conditions')}
    tasks = registration['tasks']
    old_pools = {(uid, ratio): sorted(k for k, r in tasks[ARMS[0]].items()
        if r['uid'] == uid and float(r['ratio']) == ratio) for uid in parents for ratio in (.2, .4)}
    user_pools = {uid: sorted(k for k, r in tasks[ARMS[1]].items() if r['uid'] == uid) for uid in parents}
    records = {arm: [] for arm in ARMS}
    for block_index in range(40):
        shuffled = [parents[int(i)] for i in rng['parents'].permutation(25)]
        # 200 slots / 25 batches: each parent occurs 8 times, twice in each
        # old-ratio x width class; every contiguous eight are distinct parents.
        block = [(shuffled[j % 25], j % 4) for j in range(200)]
        for inner_batch in range(25):
            batch = block_index * 25 + inner_batch
            entries = block[inner_batch*8:(inner_batch+1)*8]
            entries = [entries[int(j)] for j in rng['slot_shuffle'].permutation(8)]
            for slot, (uid, category) in enumerate(entries):
                index = batch * 8 + slot
                clean = category < 2
                ratio = .2 if category % 2 == 0 else .4
                alpha = np.float32(1) if clean else np.float32(rng['alpha'].uniform(.9, 1.1))
                if not clean and alpha == np.float32(1):
                    raise ValueError('Augmented draw rounded to one; no redraw')
                old_options, new_options = old_pools[uid, ratio], user_pools[uid]
                selected = {ARMS[0]: old_options[int(rng['old_conditions'].randint(len(old_options)))],
                            ARMS[1]: new_options[int(rng['user_conditions'].randint(len(new_options)))]}
                common = dict(sample_index=index, effective_batch_index=batch, slot_index=slot,
                    step=batch+1, local_step=batch+1, finetune_step=batch+1,
                    global_step=6001+batch, conditional_total_step=6001+batch,
                    uid=uid, N=int(registration['parents'][uid]['N']), alpha=float(alpha),
                    clean=clean, paired_old_category=category,
                    permutation_seed=domain_seed('shared_permutation', index),
                    t_seed=domain_seed('shared_time', index))
                common['t'] = float(training_time(common['t_seed']))
                for arm in ARMS:
                    task = tasks[arm][selected[arm]]
                    records[arm].append(dict(common, arm=arm, task_id=selected[arm], K=int(task['K']),
                        ratio=ratio if arm == ARMS[0] else task['K']/task['N'],
                        region_type=task.get('region_type'),
                        gaussian_seed=domain_seed('gaussian', arm, index),
                        epsilon_seed=domain_seed('epsilon', arm, index)))
    for arm, rows in records.items():
        if Counter(r['uid'] for r in rows) != {uid: 320 for uid in parents}:
            raise ValueError('Parent balance differs')
        for offset in range(0, 8000, 8):
            batch = rows[offset:offset+8]
            if len({r['uid'] for r in batch}) != 8 or sum(r['clean'] for r in batch) != 4:
                raise ValueError('Batch distinct/clean constraints differ')
            if Counter(r['paired_old_category'] for r in batch) != {0: 2, 1: 2, 2: 2, 3: 2}:
                raise ValueError('Old mixed ratio/category schedule differs')
    fields = ('sample_index', 'uid', 'N', 'alpha', 't', 't_seed', 'permutation_seed', 'clean')
    if any(any(a[k] != b[k] for k in fields) for a, b in zip(records[ARMS[0]], records[ARMS[1]])):
        raise ValueError('Paired slot identity differs')
    exposure = {}
    for arm, rows in records.items():
        exposure[arm] = dict(total_samples=8000, by_parent=dict(Counter(r['uid'] for r in rows)),
            clean_by_parent=dict(Counter(r['uid'] for r in rows if r['clean'])),
            augmented_by_parent=dict(Counter(r['uid'] for r in rows if not r['clean'])),
            by_condition=dict(Counter(r['task_id'] for r in rows)),
            by_K=dict(sorted(Counter(str(r['K']) for r in rows).items())),
            free_coordinate_count=sum(9*(r['N']-r['K']) for r in rows),
            conditions_not_drawn=sorted(set(tasks[arm])-{r['task_id'] for r in rows}))
    return dict(schema='r2_user_finetune_paired_plan_v1', namespace=NAMESPACE,
        effective_batch_count=1000, sample_count_per_arm=8000, batch_size=8, microbatch_size=1,
        starting_global_step=6000, final_global_step=7000, coupling_lambda=.25,
        registration_sha256=json_hash(registration), source_sha256=registration['source_sha256'],
        implementation_sources=registration['implementation_sources'],
        parent_schedule='40 independent shuffled25-parent blocks, each repeated8 times over25 batches; each parent320, clean160, augmented160',
        old_condition_policy='Per batch C20clean2/C20aug2/C40clean2/C40aug2; within parent/ratio uniformly choose one registered old condition',
        user_condition_policy='Parent balanced uniformly; independent uniform choice within parent unique active C; no region/tier quotas',
        rng_contract='MT19937; shared parent/alpha/time/permutation seeds, Gaussian and epsilon seeds in separate arm domains; free shape=N-K; original free-only OT then HYBRID sqrt(.75)*Z+.5*epsilon',
        exact_free_noise_pairing=False, records=records, actual_exposures=exposure)


class _Dataset(TrainingDataset):
    def __init__(self, registration, arm):
        self.root = DATA
        self.manifest = registration
        self.manifest_sha256 = registration['source_sha256']
        self.parents = registration['parents']
        self.tasks = registration['tasks'][arm]


class _Builder(baseline.HybridStream):
    """Use original sample and OT implementation byte-for-byte."""
    def __init__(self, root, arm, registration, plan_sha256):
        self.arm = arm
        self.out = Path(root) / 'data' / arm
        self.data = _Dataset(registration, arm)
        self.plan_sha256 = plan_sha256
        self.implementation_sha256 = json_hash(registration['implementation_sources'])
        self.cache_dir = self.out / 'ot_cache'
        self.counts = dict(actual_OT_attempts=0, actual_OT_returns=0, OT_cache_hits=0)
        for uid in sorted(self.data.parents):
            self.data.parent(uid)
        self.active_attempt = None
        self.active_attempt_record = None

    def _mapping(self, pre, noise, source_ids, corners, known_ids, record):
        identity = dict(contract=coupling.CONTRACT, implementation_sha256=self.implementation_sha256,
            pre_target=array_hash(pre), raw_noise=array_hash(noise), source_ids=array_hash(source_ids),
            corners=array_hash(corners), known_ids=array_hash(known_ids), alpha=record['alpha'],
            task_id=record['task_id'], sample_index=record['sample_index'],
            plan_sha256=self.plan_sha256, data_manifest_sha256=self.data.manifest_sha256)
        key = json_hash(identity)
        if self.active_attempt is None:
            raise RuntimeError('OT requires an exclusively reserved batch attempt')
        self.active_attempt_record['OT_attempts'].append(dict(sample_index=record['sample_index'],
            key=key, identity=identity, status='DISPATCHED'))
        atomic_json(self.active_attempt, self.active_attempt_record)
        self.counts['actual_OT_attempts'] += 1
        face, corner = coupling.nested_ot_map(pre, noise)
        self.counts['actual_OT_returns'] += 1
        self.active_attempt_record['OT_attempts'][-1]['status'] = 'RETURNED'
        self.active_attempt_record['OT_attempts'][-1]['map_sha256'] = json_hash([array_hash(face), array_hash(corner)])
        atomic_json(self.active_attempt, self.active_attempt_record)
        return face, corner, key, 'MISS'


def _validate_sample(sample, record):
    n, k = int(sample['y']), int(sample['K'])
    if n != record['N'] or k != record['K'] or not bits(sample['t'], np.asarray(record['t'], np.float32)):
        raise ValueError('Sample N/K/time differs from frozen plan')
    known = sample['known_mask']
    if known.dtype != np.bool_ or int(known.sum()) != k or sample['free_coordinate_count'] != 9*(n-k):
        raise ValueError('Sample masks/free denominator differ')
    if not bits(sample['xt'][known], sample['context'][known]):
        raise ValueError('Known C was changed')
    if not np.array_equal(np.sort(sample['source_face_ids']), np.arange(n)):
        raise ValueError('Source faces lost or duplicated')
    for key in HASH_FIELDS:
        if sample[key].dtype.kind in 'fc' and not np.isfinite(sample[key]).all():
            raise ValueError('Nonfinite sample array: '+key)
    shared = {q: array_hash(sample[q]) for q in HASH_FIELDS}
    expected = json_hash(dict(shared, task_id=record['task_id'], alpha=record['alpha']))
    if sample['shared_input_sha256'] != expected or sample['label_bytehash'] != array_hash(sample['u']):
        raise ValueError('Frozen input/label bytehash differs')
    input_hash = json_hash({q: shared[q] for q in ('xt', 't', 'y', 'known_mask', 'valid_mask')})
    if input_hash != sample['input_bytehash']:
        raise ValueError('Frozen model input bytehash differs')


def _read_batch(path):
    samples = []
    with np.load(path, allow_pickle=False) as z:
        for slot in range(8):
            prefix = f's{slot}__'
            sample = json.loads(str(z[prefix+'metadata'].item()))
            sample.update({k[len(prefix):]: z[k].copy() for k in z.files
                           if k.startswith(prefix) and k != prefix+'metadata'})
            samples.append(sample)
    return samples


def _preflight_rows(plan):
    chosen = []
    for arm in ARMS:
        rows = plan['records'][arm]
        selected = []
        for label, ordering in (
            ('minimum_K', lambda r: (r['K'], r['sample_index'])),
            ('maximum_K', lambda r: (-r['K'], r['sample_index'])),
            ('maximum_N', lambda r: (-r['N'], r['sample_index'])),
            ('back_or_second_ratio', lambda r: (r.get('region_type') != 'back' if arm == ARMS[1] else r['ratio'] != .4, r['sample_index']))):
            record = next(r for r in sorted(rows, key=ordering) if r['sample_index'] not in selected)
            selected.append(record['sample_index'])
            chosen.append(dict(arm=arm, sample_index=record['sample_index'], selection_reason=label,
                N=record['N'], K=record['K'], task_id=record['task_id'], region_type=record['region_type']))
    if len(chosen) != 8 or not any(r['region_type'] == 'seat' for r in chosen) or not any(r['region_type'] == 'back' for r in chosen):
        raise ValueError('Preflight registration must cover seat and back in eight fixed samples')
    return chosen


def prepare(root=ROOT, workers=2):
    if workers != 2:
        raise ValueError('This preparation fixes exactly two CPU workers; native pools one thread each')
    root = Path(root).resolve()
    out = root / 'data'
    begin = time.perf_counter()
    registration = input_registration()
    registration_sha = freeze_json(out / 'input_registration.json', registration)
    plan = make_plan(registration)
    plan_sha = freeze_json(out / 'paired_plan.json', plan)
    freeze_json(out / 'actual_exposures.json', plan['actual_exposures'])
    freeze_json(out / 'preflight_selection.json', dict(schema='r2_user_finetune_preflight_selection_v1',
        plan_sha256=plan_sha, model_forwards=0, rows=_preflight_rows(plan)))
    freeze_json(out / 'cpu_dispatch.json', dict(schema='r2_user_finetune_cpu_dispatch_v1',
        workers=2, native_threads_per_worker=1, available_cpu_cores=len(os.sched_getaffinity(0)),
        gpu_dispatch=0, registered_OT_maps=16000, namespace=NAMESPACE))
    builders = {arm: _Builder(root, arm, registration, plan_sha) for arm in ARMS}
    for builder in builders.values():
        for path in (builder.out / 'batches').glob('*.attempt.json'):
            if not path.with_suffix('').with_suffix('.npz').exists():
                raise RuntimeError('Unresolved batch OT attempt; no silent retry: '+str(path))
    jobs = [(arm, batch) for batch in range(1000) for arm in ARMS]
    indexes = {arm: [] for arm in ARMS}
    fresh_ot = cached_ot = 0

    def index_item(sample, path, sha, slot):
        item = {key: sample[key] for key in INDEX_FIELDS}
        item.update(N=int(sample['y']), sample_npz=str(path), sample_npz_sha256=sha, batch_slot=slot,
            plan_sha256=plan_sha, gaussian_shape=list(sample['free_gaussian_before_OT'].shape),
            gaussian_sha256=array_hash(sample['free_gaussian_before_OT']),
            epsilon_sha256=array_hash(sample['free_epsilon']), source_face_ids_sha256=array_hash(sample['source_face_ids']))
        return item

    def compute(job):
        arm, batch = job
        builder = builders[arm]
        records = plan['records'][arm][8*batch:8*(batch+1)]
        path = builder.out / 'batches' / f'{batch:04d}.npz'
        attempt = path.with_suffix('.attempt.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if not attempt.exists() or read_json(attempt)['plan_sha256'] != plan_sha:
                raise ValueError('Cached batch attempt identity differs')
            samples = _read_batch(path)
            sha = digest(path)
            for sample, record in zip(samples, records):
                _validate_sample(sample, record)
            return arm, [index_item(s, path, sha, i) for i, s in enumerate(samples)], 'EXISTING_SAMPLE'
        if attempt.exists():
            raise RuntimeError('Incomplete batch; no silent re-OT: '+str(attempt))
        reservation = dict(schema='r2_user_finetune_batch_OT_attempt_v1', arm=arm,
            batch_index=batch, plan_sha256=plan_sha, sample_indices=[r['sample_index'] for r in records],
            pid=os.getpid(), time_unix=time.time(), status='STARTED', OT_attempts=[])
        with attempt.open('x', encoding='utf-8') as f:
            json.dump(reservation, f, sort_keys=True)
            f.flush(); os.fsync(f.fileno())
        builder.active_attempt, builder.active_attempt_record = attempt, reservation
        samples = []
        arrays = {}
        for slot, record in enumerate(records):
            sample = baseline.HybridStream.sample(builder, record)
            sample.update(arm=arm, local_step=record['step'], finetune_step=record['step'],
                global_step=record['global_step'], conditional_total_step=record['global_step'],
                lambda_value=.25, origin_sample_index=record['sample_index'],
                condition_policy='OLD_C20_C40' if arm == ARMS[0] else 'USER_ACTIVE',
                region_type=record['region_type'])
            _validate_sample(sample, record)
            prefix = f's{slot}__'
            arrays.update({prefix+k: v for k, v in sample.items() if isinstance(v, np.ndarray)})
            metadata = {k: v for k, v in sample.items() if not isinstance(v, np.ndarray)}
            arrays[prefix+'metadata'] = np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False))
            samples.append(sample)
        sha = save_npz(path, **arrays)
        checks = _read_batch(path)
        for sample, check, record in zip(samples, checks, records):
            _validate_sample(check, record)
            if any(not bits(v, check[k]) for k, v in sample.items() if isinstance(v, np.ndarray)):
                raise ValueError('Materialized sample roundtrip changed bytes')
        reservation.update(status='COMPLETE', batch_npz_sha256=sha)
        atomic_json(attempt, reservation)
        builder.active_attempt = None
        return arm, [index_item(s, path, sha, i) for i, s in enumerate(samples)], 'MISS'

    try:
        # One independent batch per arm, two workers; no next wave after failure.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(jobs), workers):
                results = list(pool.map(compute, jobs[start:start+workers]))
                for arm, items, status in results:
                    indexes[arm].extend(items)
                    fresh_ot += 8 * (status == 'MISS')
                completed = sum(len(v) for v in indexes.values())
                if completed % 256 == 0 or completed == 16000:
                    progress = dict(status='RUNNING' if completed < 16000 else 'CACHES_COMPLETE',
                        completed_samples=completed, registered_samples=16000, workers=workers,
                        actual_OT_returns_this_run=fresh_ot, existing_OT_cache_reads=cached_ot,
                        elapsed_seconds=time.perf_counter()-begin, plan_sha256=plan_sha)
                    atomic_json(out / 'precompute_progress.json', progress)
                    print(json.dumps(progress), flush=True)
    except BaseException as exc:
        atomic_json(out / 'precompute_failure.json', dict(status='FAILED_STOP_NO_RETRY', error=repr(exc),
            completed_samples=sum(map(len, indexes.values())),
            actual_OT_attempts=sum(b.counts['actual_OT_attempts'] for b in builders.values()),
            actual_OT_returns=sum(b.counts['actual_OT_returns'] for b in builders.values()),
            plan_sha256=plan_sha, elapsed_seconds=time.perf_counter()-begin))
        raise
    for arm in ARMS:
        if [r['sample_index'] for r in indexes[arm]] != list(range(8000)):
            raise ValueError('Incomplete registered cache index')
        freeze_json(out / arm / 'paired_inputs.json', dict(schema='r2_user_finetune_paired_inputs_v1',
            arm=arm, plan_sha256=plan_sha, source_sha256=registration['source_sha256'], records=indexes[arm]))
    after = {path: digest(path) for path in registration['source_files']}
    if after != registration['source_files']:
        raise ValueError('A protected source changed during preparation')
    receipt = dict(schema='r2_user_finetune_preparation_receipt_v1', status='PASS',
        source_before_after_equal=True, source_files_checked=len(after),
        samples_per_arm=8000, total_samples=16000, workers=2, native_threads_per_worker=1,
        actual_OT_attempts_this_run=fresh_ot, actual_OT_returns_this_run=fresh_ot,
        existing_OT_cache_reads_this_run=cached_ot,
        already_materialized_samples_this_run=16000-fresh_ot-cached_ot,
        training_Gaussian_arrays_per_kind=16000, train_Gaussian_and_epsilon_arrays_materialized=32000,
        model_loads=0, model_forwards=0, model_backwards=0, optimizer_updates=0,
        evaluation_Gaussian_draws=0, sampling_calls=0, plan_sha256=plan_sha,
        input_registration_sha256=registration_sha, implementation_sources=registration['implementation_sources'],
        elapsed_seconds=time.perf_counter()-begin,
        arm_cache_index_sha256={arm: digest(out / arm / 'paired_inputs.json') for arm in ARMS})
    freeze_json(out / 'precompute_attempts' / (str(time.time_ns())+'.json'), receipt)
    atomic_json(out / 'precompute_receipt.json', receipt)
    return receipt


class Stream:
    """Training reads only immutable sample NPZs; this class cannot run OT/draw noise."""
    def __init__(self, root=ROOT, arm='R2_CONT_USER', require_cache=True):
        if require_cache is not True:
            raise ValueError('Training Stream always requires complete materialized caches')
        self.arm = _arm(arm)
        self.root = Path(root).resolve()
        self.out = self.root / 'data'
        self.plan_path = self.out / 'paired_plan.json'
        self.plan = read_json(self.plan_path)
        self.plan_sha256 = digest(self.plan_path)
        self.registration = read_json(self.out / 'input_registration.json')
        if self.plan['registration_sha256'] != json_hash(self.registration):
            raise ValueError('Registration content differs')
        if self.registration['implementation_sources'] != source_identity():
            raise ValueError('Frozen data implementation changed')
        if self.plan != make_plan(self.registration):
            raise ValueError('Paired plan does not reproduce')
        receipt = read_json(self.out / 'precompute_receipt.json')
        if receipt['status'] != 'PASS' or receipt['plan_sha256'] != self.plan_sha256:
            raise ValueError('Preparation not complete for this plan')
        self.paired_path = self.out / self.arm / 'paired_inputs.json'
        paired = read_json(self.paired_path)
        if digest(self.paired_path) != receipt['arm_cache_index_sha256'][self.arm]:
            raise ValueError('Cache index identity changed')
        self.index = paired['records']
        self.records = self.plan['records'][self.arm]
        self.total_updates = 1000
        self.batch_index = 0
        self._last_batch_path = None
        self._last_batch_samples = None
        self.implementation_sha256 = json_hash(self.registration['implementation_sources'])
        self.identities = dict(schema='r2_user_finetune_stream_identities_v1', arm=self.arm,
            namespace=NAMESPACE, plan_sha256=self.plan_sha256,
            input_registration_sha256=digest(self.out / 'input_registration.json'),
            source_sha256=self.registration['source_sha256'],
            paired_inputs_sha256=digest(self.paired_path), implementation_sha256=self.implementation_sha256,
            coupling_contract=coupling.CONTRACT, lambda_value=.25)
        self.data = _Dataset(self.registration, self.arm)
        self.counts = dict(actual_OT_attempts=0, actual_OT_returns=0, OT_cache_hits=0, sample_cache_reads=0)

    def sample(self, record):
        i = int(record['sample_index']) if isinstance(record, dict) else int(record)
        row, expected = self.records[i], self.index[i]
        if isinstance(record, dict) and record != row:
            raise ValueError('Sample record differs from registered plan')
        path = Path(expected['sample_npz'])
        if self._last_batch_path != path:
            if digest(path) != expected['sample_npz_sha256']:
                raise ValueError('Materialized batch hash differs')
            self._last_batch_samples = _read_batch(path)
            self._last_batch_path = path
        stored = self._last_batch_samples[expected['batch_slot']]
        sample = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in stored.items()}
        _validate_sample(sample, row)
        for key in INDEX_FIELDS:
            if sample[key] != expected[key]:
                raise ValueError('Materialized metadata differs: '+key)
        sample['ot_cache_status'] = 'MATERIALIZED_READ_ONLY'
        self.counts['sample_cache_reads'] += 1
        return sample

    def samples(self, step):
        if type(step) is not int or not 1 <= step <= 1000:
            raise ValueError('Expected local step 1..1000')
        return [self.sample(r) for r in self.records[8*(step-1):8*step]]

    def next_effective_batch(self):
        if self.batch_index >= 1000:
            raise RuntimeError('Frozen stream exhausted')
        samples = self.samples(self.batch_index+1)
        self.batch_index += 1
        self.last_audit = dict(batch_index=self.batch_index,
            shared_batch_sha256=json_hash([s['shared_input_sha256'] for s in samples]),
            input_batch_sha256=json_hash([s['input_bytehash'] for s in samples]),
            label_batch_sha256=json_hash([s['label_bytehash'] for s in samples]),
            effective_free_coordinates=sum(s['free_coordinate_count'] for s in samples))
        return samples

    def state_dict(self):
        return dict(schema='r2_user_finetune_stream_state_v1', identities=self.identities,
            batch_index=self.batch_index, sample_index=8*self.batch_index,
            finetune_step=self.batch_index, global_step=6000+self.batch_index)

    def load_state_dict(self, state):
        if set(state) != set(self.state_dict()) or state['schema'] != 'r2_user_finetune_stream_state_v1' or state['identities'] != self.identities:
            raise ValueError('Stream resume identities differ')
        index = state['batch_index']
        if type(index) is not int or not 0 <= index <= 1000 or state['sample_index'] != 8*index or state['finetune_step'] != index or state['global_step'] != 6000+index:
            raise ValueError('Invalid stream cursor')
        self.batch_index = index


def preflight_samples(root=ROOT):
    registration = read_json(Path(root) / 'data' / 'preflight_selection.json')
    streams = {arm: Stream(root, arm) for arm in ARMS}
    if any(registration['plan_sha256'] != s.plan_sha256 for s in streams.values()):
        raise ValueError('Preflight plan identity differs')
    return [streams[r['arm']].sample(r['sample_index']) for r in registration['rows']]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.root, args.workers), indent=2), flush=True)


if __name__ == '__main__':
    main()