"""Frozen CPU-only USER-majority data for R2_USER_MIX_FROM_OFFICIAL_V1.

The sample, free-only OT and HYBRID mathematics are imported unchanged from the
previous audited pilot. Only the registered condition schedule and clock change.
Completed batches are immutable; an interrupted OT attempt is never retried.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
_PRIOR_PATH = Path(__file__).resolve().parents[1] / 'user_condition_finetune_v1' / 'data.py'
_SPEC = importlib.util.spec_from_file_location('_r2_user_mix_prior_data_v1', _PRIOR_PATH)
prior = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = prior
_SPEC.loader.exec_module(prior)
np = prior.np
baseline = prior.baseline
coupling = prior.coupling
array_hash, atomic_json, bits = prior.array_hash, prior.atomic_json, prior.bits
digest, freeze_json, json_hash = prior.digest, prior.freeze_json, prior.json_hash
read_json, save_npz = prior.read_json, prior.save_npz

ROOT = Path('/mnt/e/MeshFlow/experiments/r2_user_mix_from_official_v1')
ARM = 'R2_USER_MIX'
ARMS = (ARM,)
NAMESPACE = 'R2_USER_MIX_FROM_OFFICIAL_V1/train/v1'
TOTAL_UPDATES = 6000
BATCH_SIZE = 8
SAMPLES = TOTAL_UPDATES * BATCH_SIZE
HASH_FIELDS = prior.HASH_FIELDS
INDEX_FIELDS = tuple(prior.INDEX_FIELDS) + ('source_kind', 'condition_source', 'condition_policy',
    'region_type', 'clean', 'schedule_category', 'local_step', 'conditional_total_step')


def domain_seed(domain, *identity):
    payload = json.dumps([NAMESPACE, domain, *identity], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'little')


def source_identity():
    sources = prior.source_identity()
    sources[str(Path(__file__).resolve())] = digest(__file__)
    return sources


def input_registration():
    registration = prior.input_registration()
    old_registration_hash = json_hash(registration)
    registration.update(schema='r2_user_mix_input_registration_v1', namespace=NAMESPACE,
        prior_registration_content_sha256=old_registration_hash,
        implementation_sources=source_identity(), output_arm=ARM,
        condition_source_task_keys={'OLD': prior.ARMS[0], 'USER': prior.ARMS[1]})
    return registration


def _summarize(rows, registration):
    sources = ('USER', 'OLD')
    total_free = sum(9*(r['N']-r['K']) for r in rows)
    by_source = {}
    for source in sources:
        group = [r for r in rows if r['condition_source'] == source]
        free = sum(9*(r['N']-r['K']) for r in group)
        tasks = registration['tasks'][registration['condition_source_task_keys'][source]]
        by_source[source] = dict(samples=len(group), sample_fraction=len(group)/len(rows),
            by_parent=dict(Counter(r['uid'] for r in group)),
            clean_by_parent=dict(Counter(r['uid'] for r in group if r['clean'])),
            width_by_parent=dict(Counter(r['uid'] for r in group if not r['clean'])),
            by_condition=dict(Counter(r['task_id'] for r in group)),
            by_K=dict(sorted(Counter(str(r['K']) for r in group).items())),
            by_region=dict(Counter(str(r['region_type']) for r in group)),
            free_coordinate_count=free, free_coordinate_fraction=free/total_free,
            free_coordinates_by_augmentation={name: sum(9*(r['N']-r['K']) for r in group if r['clean'] == clean)
                for name, clean in (('clean', True), ('width', False))},
            conditions_not_drawn=sorted(set(tasks)-{r['task_id'] for r in group}))
    return dict(total_samples=len(rows), by_parent=dict(Counter(r['uid'] for r in rows)),
        clean_by_parent=dict(Counter(r['uid'] for r in rows if r['clean'])),
        width_by_parent=dict(Counter(r['uid'] for r in rows if not r['clean'])),
        free_coordinate_count=total_free, by_source=by_source,
        weighting_note='FM uses the sum of free-coordinate errors divided by all valid free coordinates per effective batch; 6:2 slots is not a fixed 75:25 loss weight.')


def validate_plan(plan, registration):
    rows = plan['records'][ARM]
    parents = sorted(registration['parents'])
    if len(rows) != SAMPLES or [r['sample_index'] for r in rows] != list(range(SAMPLES)):
        raise ValueError('Registered sample count/order differs')
    if Counter(r['uid'] for r in rows) != {uid: 1920 for uid in parents}:
        raise ValueError('Parent exposure balance differs')
    for source, count, clean_count in (('USER', 1440, 720), ('OLD', 480, 240)):
        group = [r for r in rows if r['condition_source'] == source]
        if Counter(r['uid'] for r in group) != {uid: count for uid in parents}:
            raise ValueError('Source/parent exposure balance differs')
        if Counter(r['uid'] for r in group if r['clean']) != {uid: clean_count for uid in parents}:
            raise ValueError('Source/parent augmentation balance differs')
    for ratio in (.2, .4):
        for clean in (True, False):
            group = [r for r in rows if r['condition_source'] == 'OLD' and r['ratio'] == ratio and r['clean'] == clean]
            if Counter(r['uid'] for r in group) != {uid: 120 for uid in parents}:
                raise ValueError('OLD parent/ratio/augmentation balance differs')
    for offset in range(0, SAMPLES, 8):
        batch = rows[offset:offset+8]
        if len({r['uid'] for r in batch}) != 8:
            raise ValueError('Repeated parent in effective batch')
        if Counter((r['condition_source'], r['clean']) for r in batch) != {('USER', True): 3,
                ('USER', False): 3, ('OLD', True): 1, ('OLD', False): 1}:
            raise ValueError('Per-batch source/augmentation composition differs')
        if Counter(r['ratio'] for r in batch if r['condition_source'] == 'OLD') != {.2: 1, .4: 1}:
            raise ValueError('Per-batch OLD ratio balance differs')
        for row in batch:
            step = offset//8+1
            if row['step'] != step or row['global_step'] != step or row['conditional_total_step'] != step:
                raise ValueError('Official-initialization condition clock differs')


def make_plan(registration):
    """240 shuffled 25-batch blocks; eight category visits per parent/block."""
    parents = sorted(registration['parents'])
    if len(parents) != 25:
        raise ValueError('Exactly 25 common parents required')
    rng = {name: np.random.RandomState(domain_seed(name)) for name in
           ('parents', 'slot_shuffle', 'alpha', 'old_conditions', 'user_conditions')}
    old_tasks = registration['tasks'][prior.ARMS[0]]
    user_tasks = registration['tasks'][prior.ARMS[1]]
    old_pools = {(uid, ratio): sorted(k for k, row in old_tasks.items()
        if row['uid'] == uid and float(row['ratio']) == ratio) for uid in parents for ratio in (.2, .4)}
    user_pools = {uid: sorted(k for k, row in user_tasks.items() if row['uid'] == uid) for uid in parents}
    if any(not pool for pool in list(old_pools.values())+list(user_pools.values())):
        raise ValueError('Every parent requires USER, OLD C20 and OLD C40 conditions')
    records = []
    for block_index in range(240):
        shuffled = [parents[int(i)] for i in rng['parents'].permutation(25)]
        for inner_batch in range(25):
            batch_index = block_index*25+inner_batch
            entries = [(shuffled[j % 25], j % 8) for j in range(inner_batch*8, (inner_batch+1)*8)]
            entries = [entries[int(j)] for j in rng['slot_shuffle'].permutation(8)]
            for slot, (uid, category) in enumerate(entries):
                index = batch_index*8+slot
                source = 'USER' if category < 6 else 'OLD'
                clean = category < 3 or category == 6
                alpha = np.float32(1) if clean else np.float32(rng['alpha'].uniform(.9, 1.1))
                if not clean and alpha == np.float32(1):
                    raise ValueError('Width augmentation rounded to one; no redraw')
                if source == 'USER':
                    pool = user_pools[uid]
                    task_id = pool[int(rng['user_conditions'].randint(len(pool)))]
                    task = user_tasks[task_id]
                    ratio = task['K']/task['N']
                else:
                    ratio = .2 if (category-6+block_index) % 2 == 0 else .4
                    pool = old_pools[uid, ratio]
                    task_id = pool[int(rng['old_conditions'].randint(len(pool)))]
                    task = old_tasks[task_id]
                time_seed = domain_seed('time', index)
                records.append(dict(sample_index=index, effective_batch_index=batch_index,
                    slot_index=slot, block_index=block_index, schedule_category=category,
                    step=batch_index+1, local_step=batch_index+1, global_step=batch_index+1,
                    conditional_total_step=batch_index+1, uid=uid, task_id=task_id,
                    N=int(task['N']), K=int(task['K']), ratio=ratio, alpha=float(alpha), clean=clean,
                    arm=ARM, source_kind=source, condition_source=source, condition_policy='USER_ACTIVE' if source == 'USER' else 'OLD_C20_C40',
                    region_type=task.get('region_type'), t_seed=time_seed, t=float(prior.training_time(time_seed)),
                    permutation_seed=domain_seed('permutation', index),
                    gaussian_seed=domain_seed('gaussian', index), epsilon_seed=domain_seed('epsilon', index)))
    plan = dict(schema='r2_user_mix_plan_v1', namespace=NAMESPACE, arm=ARM,
        effective_batch_count=TOTAL_UPDATES, sample_count=SAMPLES, batch_size=8, microbatch_size=1,
        starting_global_step=0, final_global_step=TOTAL_UPDATES, coupling_lambda=.25,
        registration_sha256=json_hash(registration), source_sha256=registration['source_sha256'],
        implementation_sources=registration['implementation_sources'],
        parent_schedule='240 independently shuffled 25-parent blocks; 25 batches/block, category=j%8 and parent=permutation[j%25], independently shuffle batch slots',
        source_schedule='6 USER + 2 OLD per batch; USER 3 clean/3 width; OLD 1 clean/1 width, C20/C40 one each with clean ratio alternating by 25-batch block',
        condition_policy='Uniform within parent unique USER pool; uniform within parent/ratio OLD pool; no region/tier quotas',
        rng_contract='Independent stateless sample seeds in new namespace; original free-only OT then HYBRID sqrt(.75)*Z+.5*epsilon; no worker-derived seeds',
        records={ARM: records}, actual_exposures={ARM: _summarize(records, registration)})
    validate_plan(plan, registration)
    return plan


class _Dataset(prior.TrainingDataset):
    def __init__(self, registration):
        self.root = prior.DATA
        self.manifest = registration
        self.manifest_sha256 = registration['source_sha256']
        self.parents = registration['parents']
        old, user = (registration['tasks'][a] for a in prior.ARMS)
        if set(old) & set(user):
            raise ValueError('Source task identifiers collide')
        self.tasks = {**old, **user}


class _Builder(prior._Builder):
    """Each CPU worker owns one builder; inherited OT math/attempt logging."""
    def __init__(self, root, registration, plan_sha256):
        self.arm = ARM
        self.out = Path(root)/'data'/ARM
        self.data = _Dataset(registration)
        self.plan_sha256 = plan_sha256
        self.implementation_sha256 = json_hash(registration['implementation_sources'])
        self.cache_dir = self.out/'ot_cache'
        self.counts = dict(actual_OT_attempts=0, actual_OT_returns=0, OT_cache_hits=0)
        for uid in sorted(self.data.parents):
            self.data.parent(uid)
        self.active_attempt = None
        self.active_attempt_record = None


def _validate_sample(sample, record):
    prior._validate_sample(sample, record)
    for key in ('arm', 'source_kind', 'condition_source', 'condition_policy', 'clean', 'schedule_category',
                'region_type', 'global_step', 'local_step', 'conditional_total_step', 'step', 'task_id'):
        if sample[key] != record[key]:
            raise ValueError('Mixed sample metadata differs: '+key)
    if sample['lambda_value'] != .25 or sample['object_id'] != record['uid']:
        raise ValueError('Mixed sample identity differs')


def prepare(root=ROOT, workers=2):
    if type(workers) is not int or workers != 2:
        raise ValueError('Registered CPU preparation has exactly two workers')
    root = Path(root).resolve()
    out = root/'data'
    begin = time.perf_counter()
    registration = input_registration()
    registration_sha = freeze_json(out/'input_registration.json', registration)
    plan = make_plan(registration)
    plan_sha = freeze_json(out/'mixed_plan.json', plan)
    freeze_json(out/'actual_exposures.json', plan['actual_exposures'])
    cores = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    freeze_json(out/'cpu_dispatch.json', dict(schema='r2_user_mix_cpu_dispatch_v1',
        workers=2, native_threads_per_worker=1, available_cpu_cores=cores,
        gpu_dispatch=0, registered_OT_maps=SAMPLES, namespace=NAMESPACE))
    builders = [_Builder(root, registration, plan_sha) for _ in range(workers)]
    batch_dir = out/ARM/'batches'
    for attempt in batch_dir.glob('*.attempt.json'):
        path = attempt.with_suffix('').with_suffix('.npz')
        state = read_json(attempt)
        if not path.exists() or state.get('status') not in ('COMPLETE', 'RECOVERED_COMPLETE', 'STARTED'):
            raise RuntimeError('Unresolved OT attempt; no silent retry: '+str(attempt))
    indexes = []
    fresh_samples = cached_samples = 0

    def index_item(sample, path, sha, slot):
        item = {key: sample[key] for key in INDEX_FIELDS}
        item.update(N=int(sample['y']), sample_npz=str(path), sample_npz_sha256=sha,
            batch_slot=slot, plan_sha256=plan_sha,
            gaussian_shape=list(sample['free_gaussian_before_OT'].shape),
            gaussian_sha256=array_hash(sample['free_gaussian_before_OT']),
            epsilon_sha256=array_hash(sample['free_epsilon']),
            source_face_ids_sha256=array_hash(sample['source_face_ids']))
        return item

    def compute(job):
        worker_index, batch_index = job
        builder = builders[worker_index]
        records = plan['records'][ARM][8*batch_index:8*(batch_index+1)]
        path = batch_dir/f'{batch_index:04d}.npz'
        attempt = path.with_suffix('.attempt.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if not attempt.exists():
                raise ValueError('Cached batch has no OT attempt ledger')
            reservation = read_json(attempt)
            if reservation['plan_sha256'] != plan_sha or reservation['sample_indices'] != [r['sample_index'] for r in records]:
                raise ValueError('Cached batch attempt identity differs')
            attempts = reservation['OT_attempts']
            if len(attempts) != 8 or any(a['status'] != 'RETURNED' for a in attempts):
                raise ValueError('Cached batch lacks eight returned OT attempts')
            samples = prior._read_batch(path)
            sha = digest(path)
            for sample, record, ot in zip(samples, records, attempts):
                _validate_sample(sample, record)
                if ot['sample_index'] != record['sample_index'] or ot['key'] != sample['ot_cache_key']:
                    raise ValueError('Cached OT identity differs')
                if ot['map_sha256'] != json_hash([array_hash(sample['ot_face_map']), array_hash(sample['ot_corner_map'])]):
                    raise ValueError('Cached OT map bytes differ')
            if reservation['status'] == 'STARTED':
                reservation.update(status='RECOVERED_COMPLETE', batch_npz_sha256=sha,
                    recovery_note='Validated committed batch and all eight OT returns; zero new draws/OT')
                atomic_json(attempt, reservation)
            elif reservation['status'] not in ('COMPLETE', 'RECOVERED_COMPLETE') or reservation['batch_npz_sha256'] != sha:
                raise ValueError('Committed batch hash/status differs')
            return [index_item(s, path, sha, i) for i, s in enumerate(samples)], False
        if attempt.exists():
            raise RuntimeError('Incomplete batch; no silent re-OT: '+str(attempt))
        reservation = dict(schema='r2_user_mix_batch_OT_attempt_v1', arm=ARM,
            batch_index=batch_index, plan_sha256=plan_sha, sample_indices=[r['sample_index'] for r in records],
            pid=os.getpid(), time_unix=time.time(), status='STARTED', OT_attempts=[])
        with attempt.open('x', encoding='utf-8') as f:
            json.dump(reservation, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        builder.active_attempt, builder.active_attempt_record = attempt, reservation
        samples, arrays = [], {}
        try:
            for slot, record in enumerate(records):
                sample = baseline.HybridStream.sample(builder, record)
                sample.update({key: record[key] for key in ('arm', 'local_step', 'global_step',
                    'conditional_total_step', 'source_kind', 'condition_source', 'condition_policy', 'region_type',
                    'clean', 'schedule_category')})
                sample.update(lambda_value=.25, origin_sample_index=record['sample_index'])
                _validate_sample(sample, record)
                prefix = f's{slot}__'
                arrays.update({prefix+k: v for k, v in sample.items() if isinstance(v, np.ndarray)})
                metadata = {k: v for k, v in sample.items() if not isinstance(v, np.ndarray)}
                arrays[prefix+'metadata'] = np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False))
                samples.append(sample)
            sha = save_npz(path, **arrays)
            checks = prior._read_batch(path)
            for sample, check, record in zip(samples, checks, records):
                _validate_sample(check, record)
                if any(not bits(v, check[k]) for k, v in sample.items() if isinstance(v, np.ndarray)):
                    raise ValueError('Materialized sample roundtrip changed bytes')
            reservation.update(status='COMPLETE', batch_npz_sha256=sha)
            atomic_json(attempt, reservation)
        except BaseException as exc:
            reservation.update(status='FAILED_STOP_NO_RETRY', error=repr(exc))
            atomic_json(attempt, reservation)
            raise
        finally:
            builder.active_attempt = None
        return [index_item(s, path, sha, i) for i, s in enumerate(samples)], True

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, TOTAL_UPDATES, workers):
                # Finish this fixed wave, record every outcome, then dispatch next.
                futures = [pool.submit(compute, (worker, start+worker)) for worker in range(workers)
                           if start+worker < TOTAL_UPDATES]
                failure = None
                for future in futures:
                    try:
                        items, fresh = future.result()
                        indexes.extend(items)
                        fresh_samples += len(items) if fresh else 0
                        cached_samples += 0 if fresh else len(items)
                    except BaseException as exc:
                        failure = exc if failure is None else failure
                if failure is not None:
                    raise failure
                if len(indexes) % 256 == 0 or len(indexes) == SAMPLES:
                    progress = dict(status='RUNNING' if len(indexes) < SAMPLES else 'CACHES_COMPLETE',
                        completed_samples=len(indexes), registered_samples=SAMPLES, workers=workers,
                        actual_OT_attempts_this_run=sum(b.counts['actual_OT_attempts'] for b in builders),
                        actual_OT_returns_this_run=sum(b.counts['actual_OT_returns'] for b in builders),
                        already_materialized_samples_this_run=cached_samples,
                        elapsed_seconds=time.perf_counter()-begin, plan_sha256=plan_sha)
                    atomic_json(out/'precompute_progress.json', progress)
                    print(json.dumps(progress), flush=True)
    except BaseException as exc:
        atomic_json(out/'precompute_failure.json', dict(status='FAILED_STOP_NO_RETRY', error=repr(exc),
            completed_samples=len(indexes), actual_OT_attempts=sum(b.counts['actual_OT_attempts'] for b in builders),
            actual_OT_returns=sum(b.counts['actual_OT_returns'] for b in builders),
            plan_sha256=plan_sha, elapsed_seconds=time.perf_counter()-begin))
        raise
    if [row['sample_index'] for row in indexes] != list(range(SAMPLES)):
        raise ValueError('Incomplete registered cache index')
    index_path = out/ARM/'paired_inputs.json'
    freeze_json(index_path, dict(schema='r2_user_mix_materialized_inputs_v1', arm=ARM,
        plan_sha256=plan_sha, source_sha256=registration['source_sha256'], records=indexes))
    after = {path: digest(path) for path in registration['source_files']}
    if after != registration['source_files'] or source_identity() != registration['implementation_sources']:
        raise ValueError('Protected source or frozen implementation changed during preparation')
    receipt = dict(schema='r2_user_mix_preparation_receipt_v1', status='PASS',
        source_before_after_equal=True, source_files_checked=len(after),
        total_samples=SAMPLES, samples_per_arm=SAMPLES, workers=2, native_threads_per_worker=1,
        actual_OT_attempts_this_run=sum(b.counts['actual_OT_attempts'] for b in builders),
        actual_OT_returns_this_run=sum(b.counts['actual_OT_returns'] for b in builders),
        already_materialized_samples_this_run=cached_samples,
        training_Gaussian_arrays_per_kind=SAMPLES,
        train_Gaussian_and_epsilon_arrays_materialized=2*SAMPLES,
        training_Gaussian_and_epsilon_arrays_drawn_this_run=2*fresh_samples,
        model_loads=0, model_forwards=0, model_backwards=0, optimizer_updates=0,
        evaluation_Gaussian_draws=0, sampling_calls=0, plan_sha256=plan_sha,
        input_registration_sha256=registration_sha, implementation_sources=registration['implementation_sources'],
        elapsed_seconds=time.perf_counter()-begin, arm_cache_index_sha256={ARM: digest(index_path)})
    freeze_json(out/'precompute_attempts'/(str(time.time_ns())+'.json'), receipt)
    atomic_json(out/'precompute_receipt.json', receipt)
    return receipt


class Stream:
    """Read-only materialized stream; cannot draw Gaussian noise or execute OT."""
    def __init__(self, root=ROOT, arm=ARM, require_cache=True):
        if arm != ARM or require_cache is not True:
            raise ValueError('Only complete R2_USER_MIX caches are accepted')
        self.arm, self.root = arm, Path(root).resolve()
        self.out = self.root/'data'
        self.plan_path = self.out/'mixed_plan.json'
        self.plan = read_json(self.plan_path)
        self.plan_sha256 = digest(self.plan_path)
        self.registration = read_json(self.out/'input_registration.json')
        if self.plan['registration_sha256'] != json_hash(self.registration):
            raise ValueError('Registration content differs')
        if self.registration['implementation_sources'] != source_identity():
            raise ValueError('Frozen data implementation changed')
        if self.plan != make_plan(self.registration):
            raise ValueError('Frozen mixed plan does not reproduce')
        receipt = read_json(self.out/'precompute_receipt.json')
        if receipt['status'] != 'PASS' or receipt['plan_sha256'] != self.plan_sha256 or receipt['total_samples'] != SAMPLES:
            raise ValueError('Preparation is incomplete')
        self.paired_path = self.out/ARM/'paired_inputs.json'
        if digest(self.paired_path) != receipt['arm_cache_index_sha256'][ARM]:
            raise ValueError('Cache index identity differs')
        self.index = read_json(self.paired_path)['records']
        self.records = self.plan['records'][ARM]
        self.total_updates, self.batch_index = TOTAL_UPDATES, 0
        self._last_batch_path = self._last_batch_samples = None
        self.implementation_sha256 = json_hash(self.registration['implementation_sources'])
        self.identities = dict(schema='r2_user_mix_stream_identities_v1', arm=ARM,
            namespace=NAMESPACE, plan_sha256=self.plan_sha256,
            input_registration_sha256=digest(self.out/'input_registration.json'),
            source_sha256=self.registration['source_sha256'], paired_inputs_sha256=digest(self.paired_path),
            implementation_sha256=self.implementation_sha256, coupling_contract=coupling.CONTRACT, lambda_value=.25)
        self.data = _Dataset(self.registration)
        self.counts = dict(actual_OT_attempts=0, actual_OT_returns=0, OT_cache_hits=0, sample_cache_reads=0)

    def sample(self, record):
        i = int(record['sample_index']) if isinstance(record, dict) else int(record)
        if not 0 <= i < SAMPLES:
            raise ValueError('Sample index outside frozen budget')
        row, expected = self.records[i], self.index[i]
        if isinstance(record, dict) and record != row:
            raise ValueError('Record differs from frozen plan')
        path = Path(expected['sample_npz'])
        if self._last_batch_path != path:
            if digest(path) != expected['sample_npz_sha256']:
                raise ValueError('Materialized batch hash differs')
            self._last_batch_samples = prior._read_batch(path)
            self._last_batch_path = path
        stored = self._last_batch_samples[expected['batch_slot']]
        sample = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in stored.items()}
        _validate_sample(sample, row)
        if any(sample[key] != expected[key] for key in INDEX_FIELDS):
            raise ValueError('Materialized metadata differs from cache index')
        sample['ot_cache_status'] = 'MATERIALIZED_READ_ONLY'
        self.counts['sample_cache_reads'] += 1
        return sample

    def samples(self, step):
        if type(step) is not int or not 1 <= step <= TOTAL_UPDATES:
            raise ValueError('Expected condition step 1..6000')
        return [self.sample(row) for row in self.records[8*(step-1):8*step]]

    def next_batch(self):
        if self.batch_index >= TOTAL_UPDATES:
            raise RuntimeError('Frozen stream exhausted')
        samples = self.samples(self.batch_index+1)
        self.batch_index += 1
        self.last_audit = dict(batch_index=self.batch_index,
            shared_batch_sha256=json_hash([s['shared_input_sha256'] for s in samples]),
            input_batch_sha256=json_hash([s['input_bytehash'] for s in samples]),
            label_batch_sha256=json_hash([s['label_bytehash'] for s in samples]),
            effective_free_coordinates=sum(s['free_coordinate_count'] for s in samples),
            free_coordinates_by_source={source: sum(s['free_coordinate_count'] for s in samples if s['condition_source'] == source)
                                        for source in ('USER', 'OLD')})
        return samples

    next_effective_batch = next_batch

    def state_dict(self):
        return dict(schema='r2_user_mix_stream_state_v1', identities=self.identities,
            batch_index=self.batch_index, sample_index=8*self.batch_index,
            global_step=self.batch_index, conditional_total_step=self.batch_index)

    def load_state_dict(self, state):
        if set(state) != set(self.state_dict()) or state['schema'] != 'r2_user_mix_stream_state_v1' or state['identities'] != self.identities:
            raise ValueError('Stream resume identities differ')
        index = state['batch_index']
        if type(index) is not int or not 0 <= index <= TOTAL_UPDATES or state['sample_index'] != 8*index or state['global_step'] != index or state['conditional_total_step'] != index:
            raise ValueError('Invalid stream cursor')
        self.batch_index = index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.root, args.workers), indent=2), flush=True)


if __name__ == '__main__':
    main()
