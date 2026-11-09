"""Isolated single-D0 proposals, unchanged GEO_C/source export and CPU scores.

Importing this module performs no model call. ``run_d0`` is the only explicit
GPU entrypoint and must be dispatched by the resource-admitted controller in a
separate process. ``run_cpu`` is genuinely torch-free. Original modules/files
are reused read-only; their output globals are rebound only in this process.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import re
import shutil
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'experiments/r2_window_policy_rl_v1'
VERSION = 'window_policy_rl_original_postprocess_v1'
CALIBRATION_LABELS = ('FULL_DET', 'S005', 'S010', 'S020')


def bind():
    """Process-local path binding; no source mutation or old experiment writes."""
    from r2.pilots.terminal_reward_nft_v1 import data as D, evaluation as E
    D.OUT = OUT
    E.OUT = OUT
    E.package()
    return D, E


def torch_free():
    if 'torch' in sys.modules:
        raise RuntimeError('CPU postprocessing/scoring requires a separate torch-free process')


def _safe_name(value, name):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', value) or value in ('.', '..'):
        raise ValueError('Unsafe '+name+': '+str(value))
    return value


def _checked(path):
    D, _ = bind()
    result = D.checked(path)
    if any(p.is_symlink() for p in (result, *result.parents)):
        raise ValueError('Output paths cannot contain symlinks')
    return result


def _verify(files):
    D, _ = bind()
    for path, digest in files.items():
        if D.sha(path) != digest:
            raise ValueError('Frozen artifact changed: '+str(path))


def _query_map(manifest):
    result = {}
    for key in ('queries', 'evaluation_queries', 'training_queries'):
        rows = manifest.get(key, [])
        if isinstance(rows, dict):
            rows = rows.get('queries', list(rows.values()))
        for row in rows:
            cid = row['condition_id']
            if cid in result and result[cid] != row:
                raise ValueError('Conflicting frozen query descriptors: '+cid)
            result[cid] = row
    return result


def _requests(job, manifest):
    """Prefer explicit requests; old evaluation_requests remains supported."""
    requests = job.get('requests')
    if requests is None:
        requests = manifest.get('evaluation_requests')
    if not isinstance(requests, list) or not requests:
        raise ValueError('An explicit nonempty frozen request list is required')
    if len({q['request_id'] for q in requests}) != len(requests):
        raise ValueError('Repeated request identity within stage')
    return requests


def prepare_jobs(jobs, *, manifest_path, registration_path):
    """Read/verify all RAW before any proposal, geometry or score is created.

    jobs: [{stage_dir, label, requests:[q,...]}]. q follows the original frozen
    request schema; condition_path is accepted as an alias of condition_source.
    Queries are supplied in q.query or the manifest's query descriptor lists.
    """
    D, E = bind()
    from r2.pilots.terminal_reward_nft_v1.sample import verify_receipt
    manifest = D.read(manifest_path)
    if manifest.get('status') not in ('FROZEN', 'COMPLETE'):
        raise ValueError('Manifest must already be frozen')
    registration_sha = D.sha(registration_path)
    manifest_sha = D.sha(manifest_path)
    queries = _query_map(manifest)
    rows, destinations = [], set()
    if not isinstance(jobs, list) or not jobs:
        raise ValueError('A nonempty list of stage jobs is required')
    for job in jobs:
        stage = _checked(job['stage_dir'])
        label = _safe_name(job['label'], 'label')
        for original in _requests(job, manifest):
            q = dict(original)
            rid = _safe_name(q['request_id'], 'request_id')
            condition = q.get('condition_source', q.get('condition_path'))
            if condition is None:
                raise ValueError('Missing source condition metadata')
            q['condition_source'] = str(D.resolve(condition))
            expected = q.get('condition_source_sha256', q.get('condition_sha256'))
            if expected is None or D.sha(condition) != expected:
                raise ValueError('Source condition requires its frozen hash')
            q['condition_source_sha256'] = expected
            query = q.get('query', queries.get(q['condition_id']))
            if not isinstance(query, dict) or query.get('condition_id') != q['condition_id']:
                raise ValueError('Missing or mismatched frozen query: '+q['condition_id'])
            if D.sha(query['path']) != query['sha256']:
                raise ValueError('Frozen query file changed')
            if query['identity']['C_sha256'] != q['C_sha256']:
                raise ValueError('Frozen query is for a different C')
            receipt_path = stage/'raw'/rid/'receipt.json'
            receipt = D.read(receipt_path)
            for key, value in (('request_id', rid), ('label', label), ('N', q['N']), ('K', q['K'])):
                if receipt.get(key) != value:
                    raise ValueError('RAW receipt identity differs: '+key)
            if receipt.get('registration_sha256') not in (None, registration_sha):
                raise ValueError('RAW belongs to another registration')
            if D.resolve(receipt['raw_path']).resolve() != (stage/'raw'/rid/'raw.npz').resolve():
                raise ValueError('RAW must belong to this independent stage')
            verify_receipt(q, receipt)
            raw = D.arrays(receipt['raw_path'])['raw'].reshape(q['N'], 3, 3)
            E.source_metadata(q['condition_source'], raw[:q['K']])
            dest = str(stage/'post'/rid)
            if dest in destinations:
                raise ValueError('Repeated owned output directory')
            destinations.add(dest)
            rows.append(dict(request=q, label=label, stage_dir=str(stage), query=query,
                raw=receipt, raw_receipt_path=str(receipt_path), raw_receipt_sha256=D.sha(receipt_path),
                proposal_dir=str(stage/'proposal'/rid), out_dir=dest,
                score_dir=str(stage/'scores'/rid)))
    identity = dict(version=VERSION, manifest_path=str(D.resolve(manifest_path)),
        manifest_sha256=manifest_sha, registration_path=str(D.resolve(registration_path)),
        registration_sha256=registration_sha, implementation_sha256=D.sha(__file__),
        requests=[dict(request_id=r['request']['request_id'], condition_id=r['request']['condition_id'],
            label=r['label'], stage_dir=r['stage_dir'], input_sha256=r['request']['input_sha256'],
            raw_receipt_sha256=r['raw_receipt_sha256'], raw_sha256=r['raw']['raw_sha256'],
            condition_sha256=r['request']['condition_source_sha256'], query_sha256=r['query']['sha256']) for r in rows],
        post_contract='SINGLE_OFFICIAL_D0_GEO_C_SOURCE_WINDING', coordination='none',
        noise_draws=0, binding_operations=0)
    return rows, identity


def _proposal(row):
    D, E = bind()
    from meshflow_control.denoise import OFFICIAL_SHA
    dest = Path(row['proposal_dir'])
    receipt = D.read(dest/'receipt.json')
    if (receipt.get('status') != 'PASS' or receipt.get('model_forwards') != 1
            or receipt.get('checkpoint_sha256') != OFFICIAL_SHA
            or receipt.get('raw_sha256') != row['raw']['raw_sha256']
            or receipt.get('request_id') != row['request']['request_id']
            or receipt.get('output_sha256') != D.sha(dest/'proposal.npz')):
        raise ValueError('Exactly one authenticated official D0 proposal is required')
    return {str(dest/'proposal.npz'): D.sha(dest/'proposal.npz'),
            str(dest/'receipt.json'): D.sha(dest/'receipt.json')}


def _completed(ledger, identity):
    D, _ = bind()
    done = ledger/'completion.json'
    if done.exists():
        result = D.read(done)
        if result.get('status') != 'COMPLETE' or result['identity'] != identity:
            raise ValueError('Completed worker identity differs')
        _verify(result['files'])
        return dict(result, cache_verified=True, model_calls_this_invocation=0)
    if ledger.exists():
        raise RuntimeError('Incomplete worker ledger; explicit engineering recovery required: '+str(ledger))
    return None


def run_d0(jobs, *, manifest_path, registration_path, resource_snapshot_path, ledger_dir):
    """Explicit admitted GPU phase, one resident official model for all jobs.

    The controller supplies the actual resource-admission snapshot. Do not call
    this from a CPU scoring process: frozen scorers intentionally hide CUDA.
    A completed ledger is verified only; an incomplete attempt is never replayed.
    """
    D, E = bind()
    rows, identity = prepare_jobs(jobs, manifest_path=manifest_path, registration_path=registration_path)
    ledger = _checked(ledger_dir)
    from r2.pilots.terminal_reward_nft_v1.denoise_worker import retained_d0_binding
    binding = retained_d0_binding()
    identity.update(phase='d0', official_D0=binding, resource_snapshot_path=str(D.resolve(resource_snapshot_path)),
                    resource_snapshot_sha256=D.sha(resource_snapshot_path))
    saved = _completed(ledger, identity)
    if saved is not None:
        for row in rows: _proposal(row)
        return saved
    for row in rows:
        proposal = Path(row['proposal_dir'])
        if proposal.exists() and not (proposal/'receipt.json').exists():
            raise RuntimeError('Unfinished D0 proposal cannot be retried: '+str(proposal))
        if proposal.exists(): _proposal(row)
    if os.environ.get('CUDA_VISIBLE_DEVICES') == '':
        raise RuntimeError('D0 needs a fresh explicit GPU process, not a CPU scorer process')
    D.write(ledger/'started.json', identity)
    tick = time.perf_counter()
    try:
        flat = [dict(request_id=r['request']['request_id'], label=r['label'],
            raw_path=r['raw']['raw_path'], condition_path=r['request']['condition_source'],
            N=r['request']['N'], K=r['request']['K'], proposal_dir=r['proposal_dir'], out_dir=r['out_dir']) for r in rows]
        result = E.postprocess_batch(flat, binding['path'], ledger/'resident_batch',
            resource_snapshot_path=resource_snapshot_path, proposal_only=True)
        if result['status'] != 'COMPLETE' or result['jobs'] != len(rows):
            raise RuntimeError('Resident D0 did not complete the full registered batch')
        files = {}
        for row in rows: files.update(_proposal(row))
        done = dict(status='COMPLETE', version=VERSION, identity=identity, requests=len(rows),
            files=files, resident_D0_result=result, new_D0_forwards=result['new_D0_forwards'],
            model_calls_this_invocation=result['new_D0_forwards'],
            seconds=time.perf_counter()-tick, post_status='NOT_RUN_CPU_STAGE_FOLLOWS')
        D.write(ledger/'completion.json', done)
        return done
    except BaseException as error:
        D.write(ledger/'failure.json', dict(status='STOPPED', error=repr(error), automatic_retry=False))
        raise


def _post(row):
    D, E = bind()
    q = row['request']; dest = Path(row['out_dir']); done = dest/'receipt.json'
    identity = dict(request_id=q['request_id'], label=row['label'], input_sha256=q['input_sha256'],
        raw_sha256=row['raw']['raw_sha256'], condition_sha256=q['condition_source_sha256'])
    proposal_files = _proposal(row)
    if done.exists():
        result = D.read(done)
        if result['identity'] != identity or result['status'] != 'COMPLETE':
            raise ValueError('Cached POST identity changed')
        _verify(result['files'])
        return result
    if dest.exists():
        raise RuntimeError('Unfinished POST; no silent geometry replay: '+str(dest))
    proposal = Path(row['proposal_dir'])
    result = E.postprocess_one(row['raw']['raw_path'], q['condition_source'], q['K'], dest,
        proposal_path=proposal/'proposal.npz', proposal_receipt=proposal/'receipt.json')
    torch_free()
    if result['model_forwards_this_invocation'] != 0:
        raise RuntimeError('CPU POST unexpectedly invoked a model')
    # Exact byte alias of the maintained C-restored D0 proposal, not new geometry.
    with (dest/'denoised.npz').open('rb') as source, (dest/'pre_geo.npz').open('xb') as target:
        shutil.copyfileobj(source, target)
        target.flush(); os.fsync(target.fileno())
    if D.sha(dest/'pre_geo.npz') != D.sha(dest/'denoised.npz'):
        raise ValueError('PRE_GEO byte alias differs')
    post = D.arrays(dest/'post.npz')
    raw = D.arrays(row['raw']['raw_path'])['raw'].reshape(q['N'], 3, 3)
    source = E.source_metadata(q['condition_source'], raw[:q['K']])
    if not D.same(post['output'][:q['K']], source['C_original']):
        raise ValueError('Source-oriented C is not exact')
    if not D.same(post['vertices'][post['faces']], post['output']):
        raise ValueError('POST indexed faces differ from output')
    files = {str(p): D.sha(p) for p in dest.iterdir() if p.is_file()}
    files.update(proposal_files)
    files.update({row['raw']['raw_path']: row['raw']['raw_sha256'],
                  q['condition_source']: q['condition_source_sha256']})
    record = dict(status='COMPLETE', version=VERSION, identity=identity,
        request_id=q['request_id'], label=row['label'], input_sha256=q['input_sha256'],
        post_path=str(dest/'post.npz'), post_sha256=D.sha(dest/'post.npz'),
        pre_geo_path=str(dest/'pre_geo.npz'), pre_geo_sha256=D.sha(dest/'pre_geo.npz'),
        pre_geo_is_byte_alias_of='denoised.npz', files=files, result=result,
        model_calls=0, torch_imported=False, source_face_ids_propagated_by_maintained_export=True)
    D.write(done, record)
    return record


def _cpu_cell(job):
    row, stages, registration_sha = job
    torch_free(); D, E = bind()
    from r2.pilots.terminal_reward_nft_v1.cpu_worker import _single_thread
    tick = time.perf_counter()
    with _single_thread():
        if 'post' in stages:
            post = _post(row)
        else:
            post = D.read(Path(row['out_dir'])/'receipt.json')
            _verify(post['files'])
        q = row['request']
        if post['input_sha256'] != q['input_sha256'] or post['label'] != row['label']:
            raise ValueError('POST belongs to another stage')
        score = None
        if 'score' in stages:
            score = E.score_one(q, row['raw']['raw_path'], post['post_path'], q['condition_source'],
                row['query'], row['score_dir'], label=row['label'],
                registration_sha256=registration_sha, stages=('RAW', 'POST'))
        torch_free()
    files = dict(post['files'])
    files[str(Path(row['out_dir'])/'receipt.json')] = D.sha(Path(row['out_dir'])/'receipt.json')
    if score:
        for stage, path in score['score_paths'].items():
            files[path] = D.sha(path)
            files.update(score['stages'][stage]['files'])
    return dict(request_id=q['request_id'], condition_id=q['condition_id'], label=row['label'],
        stage_dir=row['stage_dir'], raw_path=row['raw']['raw_path'], post_path=post['post_path'],
        pre_geo_path=post['pre_geo_path'], condition_path=q['condition_source'],
        score_paths=score['score_paths'] if score else {}, files=files,
        model_calls=0, seconds=time.perf_counter()-tick)


def run_cpu(jobs, *, manifest_path, registration_path, ledger_dir,
            stages=('post', 'score'), workers=1):
    """CPU-only phase, with all RAW and single-D0 receipts required up front."""
    torch_free(); D, _ = bind()
    if workers not in (1, 2):
        raise ValueError('Measured CPU worker admission must choose one or two workers')
    if not stages or len(set(stages)) != len(stages) or any(s not in ('post', 'score') for s in stages):
        raise ValueError('CPU stages are post and/or score')
    stages = tuple(s for s in ('post', 'score') if s in stages)
    rows, identity = prepare_jobs(jobs, manifest_path=manifest_path, registration_path=registration_path)
    for row in rows: _proposal(row)
    identity.update(phase='cpu', stages=list(stages))
    ledger = _checked(ledger_dir)
    saved = _completed(ledger, identity)
    if saved is not None: return saved
    D.write(ledger/'started.json', identity)
    tick = time.perf_counter()
    try:
        tasks = [(row, stages, identity['registration_sha256']) for row in rows]
        if workers == 1:
            results = [_cpu_cell(task) for task in tasks]
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers,
                    mp_context=multiprocessing.get_context('spawn')) as pool:
                results = list(pool.map(_cpu_cell, tasks))
        files = {}
        for row in results: files.update(row['files'])
        torch_free()
        result = dict(status='COMPLETE', version=VERSION, identity=identity, rows=results,
            requests=len(rows), stages=list(stages), files=files, workers=workers,
            CPU_threads_per_worker=1, model_calls=0, model_calls_this_invocation=0,
            torch_imported=False, seconds=time.perf_counter()-tick,
            semantic_review='NOT_ASSIGNED; requires actual separate image review')
        D.write(ledger/'completion.json', result)
        return result
    except BaseException as error:
        D.write(ledger/'failure.json', dict(status='STOPPED', error=repr(error), automatic_retry=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('d0', 'cpu', 'verify-inputs'), required=True)
    parser.add_argument('--job', required=True, help='Frozen JSON with jobs and common API arguments')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=None)
    parser.add_argument('--stages', default=None, help='CPU post,score or either one')
    args = parser.parse_args()
    D, _ = bind(); spec = D.read(args.job)
    common = dict(jobs=spec['jobs'], manifest_path=spec['manifest_path'], registration_path=spec['registration_path'])
    if args.phase == 'verify-inputs':
        torch_free(); rows, identity = prepare_jobs(**common)
        result = dict(status='PASS', requests=len(rows), identity=identity, model_calls=0)
    elif args.phase == 'd0':
        result = run_d0(**common, resource_snapshot_path=spec['resource_snapshot_path'], ledger_dir=spec['ledger_dir'])
    else:
        stages = args.stages.split(',') if args.stages else spec.get('stages', ('post', 'score'))
        result = run_cpu(**common, ledger_dir=spec['ledger_dir'], stages=stages,
                         workers=args.workers if args.workers is not None else spec.get('workers', 1))
    print(json.dumps({k:result[k] for k in ('status', 'requests', 'new_D0_forwards',
        'model_calls_this_invocation', 'seconds', 'cache_verified') if k in result}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
