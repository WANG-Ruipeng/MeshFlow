"""Frozen SMALL/CREF inference for regular patches; no training or implicit replay.

Each CLI invocation owns an explicitly selected arm and NEW-request partition.
The root dispatcher, never this module, chooses concurrent GPU processes.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_name, '2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time
import traceback
import numpy as np
from . import common as C

LABELS = ('SMALL_C200', 'CREF_C200')
EV = C.OUT / 'evaluation'


def verify(files):
    for path, expected in files.items():
        if C.sha(path) != expected:
            raise ValueError('Frozen artifact changed: ' + str(path))


def registration():
    r = C.load_registration()
    if r.get('status') != 'FROZEN' or tuple(r['labels']) != LABELS:
        raise ValueError('Expected frozen two-model registration')
    qs = r['requests']
    if len(qs) != 24 or len({q['request_id'] for q in qs}) != 24:
        raise ValueError('Expected exactly 24 unique requests')
    if sum(q['action'] == 'NEW' for q in qs) != 16 or sum(q['action'] == 'CACHE' for q in qs) != 8:
        raise ValueError('Expected 16 NEW plus 8 CACHE conditions/noises per model')
    for q in qs:
        if not 128 <= int(q['N']) <= 256 or not 2 <= int(q['K']) < int(q['N']):
            raise ValueError('Registered regular-patch N/K contract')
    return r


def check_code():
    r = registration()
    files = r['code_hashes']
    if not any(C.resolve(p).resolve() == Path(__file__).resolve() for p in files):
        raise ValueError('Inference implementation is not frozen')
    verify(files)
    if 'manifest_sha256' in r:
        verify({C.OUT / 'inputs/manifest.json': r['manifest_sha256']})
    verify(r.get('blind_files', {}))
    return dict(status='PASS', requests=24, NEW=16, CACHE=8,
                registration_sha256=C.sha(C.OUT / 'registration.json'))


def binding(arm):
    if arm not in LABELS:
        raise ValueError('Unregistered model')
    return registration()['models'][arm]


def record_path(q, arm, stage):
    return C.checked(EV / stage / arm / q['request_id'] / 'receipt.json')


def worker_path(stage, arm, worker, workers, name):
    return C.checked(EV / 'execution' / arm / f'{stage}_{worker}_{workers}_{name}.json')


def partition(r, worker, workers):
    if workers < 1 or not 0 <= worker < workers:
        raise ValueError('Invalid request partition')
    return [q for i, q in enumerate(q for q in r['requests'] if q['action'] == 'NEW')
            if i % workers == worker]


@contextmanager
def owner(stage, arm, worker, workers):
    import fcntl
    # Freeze partition count before separate owners: worker0/of1 must not
    # overlap worker1/of2. This changes no registered request or scientific call.
    plan = C.checked(EV / 'execution' / arm / (stage + '_dispatch_plan.json'))
    plan.parent.mkdir(parents=True, exist_ok=True)
    with plan.with_suffix('.lock').open('a+') as dispatcher:
        fcntl.flock(dispatcher, fcntl.LOCK_EX)
        try:
            C.write(plan, dict(stage=stage, arm=arm, workers=workers,
                              policy='Fixed registered NEW order modulo workers'))
        finally:
            fcntl.flock(dispatcher, fcntl.LOCK_UN)
    path = worker_path(stage, arm, worker, workers, 'owner').with_suffix('.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This arm/stage/partition already has an owner')
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _gpu(arm):
    # Private module namespace: never mutate the maintained/old experiment module.
    G = C.module('r2.pilots.seat_failure_diagnosis_v1._regular_patch_gpu_' + arm,
                 C.ROOT / 'r2/pilots/seat_failure_diagnosis_v1/gpu.py')
    G.C = C
    G.LABEL = arm
    G.check_code = check_code
    G.registration = lambda: dict(registration(), checkpoint=binding(arm))
    G.record_path = lambda q, stage: record_path(q, arm, stage)
    G.worker_path = lambda stage, worker, workers, name: worker_path(stage, arm, worker, workers, name)
    G.read_raw = lambda q, b=None: read_raw(q, arm)
    G.read_post = lambda q, raw=None: read_post(q, arm)
    G.all_raw = lambda r: all_raw(r, arm)
    G.partition = partition
    return G


def _raw_contract(q, rec, arm):
    C.package()
    b = binding(arm)
    if rec['checkpoint_sha256'] != b['sha256']:
        raise ValueError('Checkpoint binding differs')
    verify({rec[name + '_path']: rec[name + '_sha256'] for name in ('raw', 'ply', 'result', 'velocities')})
    inp = C.old().read_input(q)
    a = C.arrays(rec['raw_path'])
    n, k = int(q['N']), int(q['K'])
    raw = a['raw'].reshape(n, 3, 3)
    world = a['trajectory'].reshape(51, n, 3, 3)
    v = C.arrays(rec['velocities_path'])['velocity'].reshape(50, n, 3, 3)
    if v.dtype != np.float32 or not np.isfinite(v).all() or np.any(v.view(np.uint32) & 0xffff):
        raise ValueError('Expected finite lossless BF16 velocities in FP32 storage')
    facts = dict(C_all51=all(C.same(x[:k], inp['C']) for x in world),
                 C_saved=C.same(a['C'].reshape(k, 3, 3), inp['C']),
                 frozen_z=C.same(a['z'], inp['z']), terminal=C.same(raw, world[-1]),
                 FP32=raw.dtype == world.dtype == np.float32,
                 finite=bool(np.isfinite(world).all()))
    if arm == 'CREF_C200':
        from r2.pilots.c_reference_frame_v1.model import area_center
        local = a['local_trajectory'].reshape(51, n, 3, 3)
        center = a['center'].reshape(3)
        facts.update(center_C_only=C.same(center, area_center(inp['C'])),
                     local_C_all51=all(C.same(x[:k], inp['C'] - center) for x in local),
                     local_start_z=C.same(local[0, k:].reshape(1, n-k, 9), inp['z'][:, k:]),
                     local_euler=all(C.same(local[t+1, k:], local[t, k:] + v[t, k:] * np.float32(1/50)) for t in range(50)),
                     world_translation=all(C.same(w[k:], l[k:] + center) for w, l in zip(world, local)),
                     local_FP32_finite=local.dtype == np.float32 and bool(np.isfinite(local).all()))
    else:
        facts.update(world_start_z=C.same(world[0, k:].reshape(1, n-k, 9), inp['z'][:, k:]),
                     world_euler=all(C.same(world[t+1, k:], world[t, k:] + v[t, k:] * np.float32(1/50)) for t in range(50)))
    from meshflow_control.mesh_io import read_ply
    ply, _ = read_ply(C.resolve(rec['ply_path']))
    facts['PLY_exact'] = C.same(ply, raw)
    run = C.read(rec['result_path']); run = run.get('sampling', run)
    facts['runtime'] = run['nfe'] == 50 and run['readout_mode'] == 'none' and all(s in run['dtype'].upper() for s in ('BF16', 'FP32'))
    if not all(facts.values()):
        raise ValueError('Frozen sampling contract failed: ' + str(facts))
    G = _gpu(arm)
    G.source(q, inp['C'])
    # The original ledger remains evidence even for a reused completed rollout.
    ledger = C.resolve(rec['raw_path']).parent / 'calls.jsonl'
    G.verify_ledger(ledger)
    return facts


def read_raw(q, arm):
    rec = C.read(record_path(q, arm, 'raw'))
    if rec['request_id'] != q['request_id'] or rec['input_sha256'] != q['input_sha256'] or rec['label'] != arm:
        raise ValueError('RAW receipt identity mismatch')
    _raw_contract(q, rec, arm)
    if q['action'] == 'CACHE':
        cache = q['cache'][arm]
        verify({cache['raw_path']: cache['raw_sha256']})
        if rec['new_rollouts'] != 0 or rec['historical_receipt_sha256'] != cache['raw_sha256']:
            raise ValueError('Cache receipt identity mismatch')
    elif rec['new_rollouts'] != 1:
        raise ValueError('NEW request requires one completed rollout')
    return rec


def _post_contract(q, rec, raw, arm):
    if rec.get('status') != 'COMPLETE' or rec['raw_sha256'] != raw['raw_sha256']:
        raise ValueError('POST/RAW identity mismatch')
    if rec['source_sha256'] != q['condition_source_sha256']:
        raise ValueError('POST source metadata mismatch')
    verify(rec['files']); verify({rec['post_path']: rec['post_sha256']})
    proposal = C.read(rec['proposal_receipt'])
    if proposal['model_forwards'] != 1 or proposal['checkpoint_sha256'] != registration()['d0_checkpoint']['sha256']:
        raise ValueError('POST must use exactly the registered single official D0')
    if proposal['raw_sha256'] != raw['raw_sha256'] or proposal['output_sha256'] != C.sha(rec['proposal_path']):
        raise ValueError('D0 proposal lineage mismatch')
    src = _gpu(arm).source(q, C.old().read_input(q)['C'])
    C.package()
    from meshflow_control.source_winding import validate_source
    source = validate_source(src, C.old().read_input(q)['C'])
    a = C.arrays(rec['post_path']); k = int(q['K'])
    if not C.same(a['output'][:k], source['C_original']):
        raise ValueError('POST original source C order/coordinates differ')
    if not C.same(a['vertices'][a['faces']], a['output']):
        raise ValueError('POST indexed geometry mismatch')
    return rec


def read_post(q, arm):
    rec = C.read(record_path(q, arm, 'post'))
    if rec['request_id'] != q['request_id'] or rec['label'] != arm:
        raise ValueError('POST request identity mismatch')
    raw = read_raw(q, arm)
    _post_contract(q, rec, raw, arm)
    if q['action'] == 'CACHE':
        cache = q['cache'][arm]
        verify({cache['post_path']: cache['post_sha256']})
        if rec['new_D0_F'] != 0 or rec['historical_receipt_sha256'] != cache['post_sha256']:
            raise ValueError('Cached POST provenance mismatch')
    return rec


def reuse():
    check_code(); count = 0
    for arm in LABELS:
        for q in registration()['requests']:
            if q['action'] != 'CACHE':
                continue
            cache = q['cache'][arm]
            verify({cache[s + '_path']: cache[s + '_sha256'] for s in ('raw', 'post')})
            original = C.read(cache['raw_path'])
            for key in ('input_sha256', 'C_sha256', 'Gaussian_sha256', 'condition_source_sha256', 'N', 'K', 'noise_seed'):
                if original[key] != q[key]:
                    raise ValueError('Cache input contract differs: ' + key)
            if original['label'] != arm:
                raise ValueError('Wrong cached model')
            _raw_contract(q, original, arm)
            oldpost = C.read(cache['post_path']); _post_contract(q, oldpost, original, arm)
            raw = dict(original); raw.update(q)
            raw.update(label=arm, status='COMPLETE', new_rollouts=0, new_sampling_F=0,
                       generation_status='VERIFIED_HISTORICAL_RAW_REUSE',
                       historical_request_id=original['request_id'], historical_receipt=cache['raw_path'],
                       historical_receipt_sha256=cache['raw_sha256'])
            postrec = dict(oldpost); postrec.update(q)
            postrec.update(label=arm, status='COMPLETE', new_rollouts=0, new_sampling_F=0,
                           new_D0_F=0, GEO_calls=0, post_status='VERIFIED_HISTORICAL_POST_REUSE',
                           historical_request_id=oldpost['request_id'], historical_receipt=cache['post_path'],
                           historical_receipt_sha256=cache['post_sha256'])
            C.write(record_path(q, arm, 'raw'), raw)
            C.write(record_path(q, arm, 'post'), postrec)
            read_raw(q, arm); read_post(q, arm); count += 1
    result = dict(status='COMPLETE', reused_model_requests=count, new_model_calls=0, geometry_calls=0)
    C.write(EV / 'cache_reuse.json', result)
    return result


def all_raw(r, arm=None):
    rows = [read_raw(q, label) for label in LABELS for q in r['requests']]
    if len(rows) != 48:
        raise ValueError('All 48 logical RAW required before any D0')
    return rows if arm is None else [x for x in rows if x['label'] == arm]


def configure_GPU(G, phase, worker, workers, estimate):
    C.package()
    from meshflow_control import runtime
    import torch
    runtime.configure_stable_runtime(); torch.set_num_threads(2)
    # Estimate includes context allowance; this check is not a process-VRAM proof.
    G.snap(phase + '_before_load', worker, workers, 1.2 * estimate)
    cap = float(registration().get('resources', {}).get('allocator_cap_MiB', 5120))
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction((cap * 1024**2) / total)
    torch.cuda.reset_peak_memory_stats()
    return torch


def generate(arm, worker=0, workers=1):
    check_code(); r = registration(); b = binding(arm); G = _gpu(arm)
    with owner('generator', arm, worker, workers):
        pending = []
        for q in partition(r, worker, workers):
            path = record_path(q, arm, 'raw')
            if path.exists():
                read_raw(q, arm)
            elif path.parent.exists():
                raise RuntimeError('Unresolved RAW attempt; no replay: ' + q['request_id'])
            else:
                C.old().read_input(q); pending.append(q)
        if not pending:
            return dict(status='COMPLETE', new_rollouts=0)
        estimate = r.get('resources', {}).get('generator_incremental_estimate_MiB', 2600.)
        torch = configure_GPU(G, 'generator_' + arm, worker, workers, estimate)
        verify({b['path']: b['sha256']})
        from meshflow_control.checkpoints import state_hash
        if arm == 'CREF_C200':
            from r2.pilots.c_reference_frame_v1.model import build_model, sample
        else:
            from meshflow_control.models.native import build_model
            from meshflow_control.sampling import sample
        payload = torch.load(C.resolve(b['path']), map_location='cpu', weights_only=True, mmap=True)
        if (payload['arm'], payload['local_step'], payload['global_step']) != (arm, 200, 6200):
            raise ValueError('Wrong fixed step200 inference endpoint')
        if any(key in payload for key in ('optimizer', 'rng', 'teacher_state', 'aligner_state')):
            raise ValueError('Use the registered inference export')
        if state_hash(payload['generator_state']) != payload['generator_state_sha256']:
            raise ValueError('Invalid generator state hash')
        if 'generator_state_sha256' in b and b['generator_state_sha256'] != payload['generator_state_sha256']:
            raise ValueError('Binding state hash differs')
        model = build_model(payload['model_config'], readout_mode='none', trainable=False).eval()
        model.load_state_dict(payload['generator_state'], strict=True)
        model.to('cuda').set_trainable(False)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        identity = {k: v for k, v in payload.items() if k != 'generator_state'}
        C.write(worker_path('generator', arm, worker, workers, 'identity'), dict(identity, file_sha256=b['sha256']))
        G.snap('generator_resident_' + arm, worker, workers)

        class Budget(C.old().CellBudget):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs); self.velocities = []
            def call(self, kind, fn, context=None):
                value = super().call(kind, fn, context)
                if kind == 'sampling_forward':
                    self.velocities.append(value.detach().float().cpu().numpy()[0])
                return value

        count = 0; tick = time.perf_counter(); allocated = reserved = 0
        try:
            for q in pending:
                G.snap('generator_cell_' + arm, worker, workers)
                inp = C.old().read_input(q); G.source(q, inp['C'])
                dest = record_path(q, arm, 'raw').parent; dest.mkdir(parents=True, exist_ok=False)
                context = dict(label=arm, request_id=q['request_id'], input_sha256=q['input_sha256'], checkpoint_sha256=b['sha256'])
                budget = Budget(dest / 'calls.jsonl', context)
                _, _, audit = sample(model, inp['C'], int(q['N']), seed=q['noise_seed'], z=inp['z'], budget=budget, out=dest, context=context)
                allocated = max(allocated, audit['cuda_peak_allocated_bytes'])
                reserved = max(reserved, audit['cuda_peak_reserved_bytes'])
                C.frozen_npz(dest / 'velocities.npz', dict(velocity=np.stack(budget.velocities), source_dtype=np.array('torch.bfloat16')))
                rec = dict(q); rec.update(context)
                rec.update(status='COMPLETE', new_rollouts=1, new_sampling_F=50, generation_status='NEW_EULER50',
                           velocities_path=str(dest / 'velocities.npz'), velocities_sha256=C.sha(dest / 'velocities.npz'), worker=worker, workers=workers)
                for field, name in (('raw', 'raw.npz'), ('ply', 'raw.ply'), ('result', 'run.json')):
                    rec[field + '_path'] = str(dest / name); rec[field + '_sha256'] = C.sha(dest / name)
                rec['contracts'] = _raw_contract(q, rec, arm)
                C.write(dest / 'receipt.json', rec); read_raw(q, arm); count += 1
                C.write(worker_path('generator', arm, worker, workers, 'progress'), dict(completed=count, new_rollouts=count, seconds=time.perf_counter()-tick, peak_allocated_bytes=allocated, peak_reserved_bytes=reserved), replace=True)
                print(json.dumps(dict(stage='RAW', arm=arm, completed=count, request_id=q['request_id'], seconds=audit['seconds'])), flush=True)
            if state_hash(model.state_dict()) != payload['generator_state_sha256']:
                raise ValueError('Frozen generator state changed')
            G.snap('generator_finish_' + arm, worker, workers)
            result = dict(status='COMPLETE', new_rollouts=count, new_sampling_F=50*count, seconds=time.perf_counter()-tick,
                          peak_allocated_bytes=allocated, peak_reserved_bytes=reserved,
                          peak_scope='Max actual per-rollout PyTorch allocator counters including resident weights; not full WDDM process VRAM', training=0, backward=0)
            C.write(worker_path('generator', arm, worker, workers, 'complete'), result)
            return result
        except BaseException as exc:
            C.event('GENERATOR_STOPPED', arm=arm, worker=worker, workers=workers, returned_rollouts=count, error=repr(exc), traceback=traceback.format_exc())
            raise
        finally:
            del model; torch.cuda.empty_cache()


def post(arm, worker=0, workers=1):
    check_code(); r = registration(); G = _gpu(arm)
    with owner('D0', arm, worker, workers):
        raw_rows = all_raw(r, arm)
        assigned = partition(r, worker, workers)
        if all(record_path(q, arm, 'post').exists() for q in assigned):
            for q in assigned:
                read_post(q, arm)
            return dict(status='COMPLETE', new_D0_F=0)
        estimate = r.get('resources', {}).get('D0_incremental_estimate_MiB', 3000.)
        configure_GPU(G, 'D0_' + arm, worker, workers, estimate)
        original_source, original_barrier = G.source, G.all_raw
        # Both arms' RAW were verified immediately above; preserve that barrier.
        G.all_raw = lambda reg: raw_rows
        entered = set()
        def guarded_source(q, condition):
            if q['request_id'] not in entered:
                G.snap('D0_cell_boundary_' + arm, worker, workers)
                entered.add(q['request_id'])
            return original_source(q, condition)
        G.source = guarded_source
        try:
            return G.postprocess(worker, workers)
        finally:
            G.source, G.all_raw = original_source, original_barrier


postprocess = post


def collect():
    check_code(); r = registration(); rows = {'raw': [], 'post': []}
    for arm in LABELS:
        for q in r['requests']:
            rows['raw'].append(read_raw(q, arm)); rows['post'].append(read_post(q, arm))
    for stage, values in rows.items():
        C.write(EV / (stage + '_index.json'), dict(status='COMPLETE', rows=values))
    result = dict(status='COMPLETE', RAW=48, POST=48, NEW_per_stage=32, CACHE_per_stage=16, model_calls=0)
    C.write(EV / 'collection_complete.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('check', 'reuse', 'generate', 'post', 'collect'))
    parser.add_argument('--arm', choices=LABELS)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.stage in ('generate', 'post') and args.arm is None:
        parser.error('--arm is required for model stages')
    if args.stage == 'check': result = check_code()
    elif args.stage == 'reuse': result = reuse()
    elif args.stage == 'collect': result = collect()
    else: result = globals()[args.stage](args.arm, args.worker, args.workers)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
