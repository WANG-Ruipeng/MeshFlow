"""Frozen Bench evaluation using the original 124-field quality definitions.

Only artifact transport and applicable arm identities are adapted. The original
finite triangle detectors, I queries, AVMR/PMR, topology and seam/bulk graph
functions are reused unchanged in a private namespace. No model imports.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
import collections
import concurrent.futures
import inspect
import json
import multiprocessing
from pathlib import Path
import shutil
import sys
import time
import types

import numpy as np
from r2.pilots.seam_abc_v1 import evaluate as E, eval_common as C
from r2.pilots.seam_pool47_couple_v1.endpoint_scoring import metric_row

VERSION = 'bench_original_124_fields_C_only_input_transport_v1'
ARMS = ('PLAIN', 'SEAM')
ROOT = C.ROOT/'experiments/r2_bench_control_transfer_v1'


def save_npz(path, arrays):
    path = C.resolve(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError('Refusing to overwrite array evidence: '+str(path))
    temp = path.with_name(path.name+'.partial')
    with temp.open('wb') as handle:
        np.savez_compressed(handle, **arrays); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)


def artifact(root, arm, request_id, stage):
    root = C.resolve(root)
    path = root/'evaluation'/arm/request_id/stage.lower()/(stage.lower()+'.npz')
    receipt = path.parent/'receipt.json'; r = C.read(receipt)
    if r['status'] != 'COMPLETE' or r['arm'] != arm or r['request_id'] != request_id:
        raise ValueError('Producer identity mismatch: '+str(receipt))
    key = 'raw_sha256' if stage == 'RAW' else 'post_sha256'
    if r[key] != C.sha(path):
        raise ValueError('Saved mesh changed: '+str(path))
    if stage == 'RAW' and r['generator_forwards'] != 50:
        raise ValueError('Not registered Euler50 output')
    if stage == 'POST':
        if r['D0_forwards'] != 1 or r['GEO_calls'] != 1:
            raise ValueError('Not registered single-D0/GEO output')
        E.verify(r['files'])
    return path, receipt


def private_score():
    common = types.SimpleNamespace(**vars(C))
    common.ARMS = ('SEAM',)  # PLAIN has no structured seam; graph marked NA.
    namespace = dict(vars(E)); namespace['C'] = common
    for name, obj in vars(E).items():
        if inspect.isfunction(obj) and obj.__globals__ is vars(E):
            function = types.FunctionType(obj.__code__, namespace, obj.__name__, obj.__defaults__, obj.__closure__)
            function.__kwdefaults__ = obj.__kwdefaults__; namespace[name] = function
    namespace['artifact'] = artifact
    namespace['VERSION'] = VERSION
    return namespace


API = private_score()


def source_hashes():
    return {**E.source_hashes(), str(Path(metric_row.__code__.co_filename)): C.sha(metric_row.__code__.co_filename),
        str(Path(__file__)): C.sha(__file__)}


def prepare(root):
    """Freeze 12 C-area query banks before model evaluation, ideally training."""
    root = C.resolve(root); destination = root/'evaluation/score_registration.json'
    if destination.exists():
        previous = C.read(destination); E.verify(previous['protected']); E.verify(previous['source_hashes'])
        return previous
    request_path = root/'data/evaluation_requests.json'
    data = C.read(request_path)
    rows = data if isinstance(data, list) else data['requests']
    if len(rows) != 24 or len({r['request_id'] for r in rows}) != 24:
        raise ValueError('Expected 24 fixed requests')
    old, A, _, _, _ = E.helpers(); descriptors = {}; requests = []; protected = {str(request_path): C.sha(request_path)}
    for initial in rows:
        q = dict(initial)
        q['split'] = q.get('split', q.get('group'))
        q['noise_index'] = int(q.get('noise_index', q.get('noise_id')))
        q['inference_npz'] = str(C.resolve(q.get('inference_npz', q['case_path'])))
        q['metadata_npz'] = str(C.resolve(q.get('metadata_npz', q['case_path'])))
        bundle = C.arrays(q['inference_npz']); metadata = C.arrays(q['metadata_npz'])
        canonical = bundle['C_vertices'][bundle['C_faces_model']]
        original = bundle['C_vertices'][bundle['C_faces_original']]
        if canonical.dtype != np.float32 or canonical.shape != (int(q['K']), 3, 3):
            raise ValueError('Unexpected frozen C')
        if not 128 <= int(q['N']) <= 256 or int(q['K']) < 2:
            raise ValueError('Outside regular Bench input contract')
        source_ids = metadata['C_source_vertex_ids'][bundle['C_faces_original']]
        _, _, topology = A.topology(original, source_ids)
        cid = q['condition_id']; c_hash = old.array_sha(canonical)
        if cid not in descriptors:
            arrays, seed = old.query_bank(canonical, cid)
            path = root/'evaluation/queries'/(cid+'.npz'); save_npz(path, arrays)
            descriptors[cid] = dict(condition_id=cid, parent_id=q['parent_id'], path=str(path),
                sha256=C.sha(path), array_sha256={name: old.array_sha(value) for name, value in arrays.items()},
                identity=dict(condition_id=cid, C_seed=seed, C_sha256=c_hash, L0=C.L), query_count=2048,
                weight_denominator=float(arrays['C_weights'].sum()), source_boundary_audit=topology,
                sampling_method='Original sample_C deterministic CID hash; unchanged method/name seed provenance')
        elif descriptors[cid]['identity']['C_sha256'] != c_hash:
            raise ValueError('Condition ID denotes different C')
        q['query_descriptor'] = descriptors[cid]
        for key in ('inference_npz', 'metadata_npz'):
            protected[q[key]] = C.sha(q[key])
        protected[descriptors[cid]['path']] = descriptors[cid]['sha256']
        requests.append(q)
    for cid in descriptors:
        paired = [q for q in requests if q['condition_id'] == cid]
        if len(paired) != 2 or {q['noise_index'] for q in paired} != {0, 1}:
            raise ValueError('Expected exactly two paired noises per C')
    if len(descriptors) != 12:
        raise ValueError('Expected four TRAIN and eight DEV conditions')
    hashes = source_hashes()
    for source, digest in hashes.items():
        path = C.resolve(source)
        try:
            relative = path.relative_to(C.ROOT)
        except ValueError:
            relative = Path('external')/path.name
        target = root/'scoring_source_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and C.sha(target) != digest:
            raise ValueError('Source snapshot differs')
        if not target.exists():
            shutil.copyfile(path, target)
    result = dict(status='FROZEN', version=VERSION, category='Bench', synset='02828884',
        arms=list(ARMS), requests=requests, cells=len(requests)*len(ARMS)*2, stages=['RAW', 'POST'],
        protected=protected, source_hashes=hashes, L=C.L, C_query_count=2048,
        query_banks=len(descriptors), new_C_query_points=len(descriptors)*2048,
        quality_definition=old.cpu_math()['control'].frozen_definition(),
        scorer='Unchanged original evaluate.score_one in a private namespace; artifact schema and SEAM applicability adapted.',
        source_geometry_use='C geometry/topology only for scoring; no complete-parent reconstruction rank.',
        PLAIN_seam_graph='NA_NOT_APPLICABLE_FULL_ARM; ordinary C/free graph retained for direct comparison.',
        semantics='Actual AI_PROXY image review is separate; no automatic use/adoption label.', model_calls=0)
    C.write(destination, result)
    if 'torch' in sys.modules:
        raise RuntimeError('Scoring preparation imported model framework')
    return result


def worker(job):
    return API['score_one'](job)


def run(root, workers=2):
    root = C.resolve(root); reg = prepare(root); registration = root/'evaluation/score_registration.json'
    completion = root/'evaluation/score_completion.json'
    if completion.exists():
        old = C.read(completion)
        for result in old['receipts']:
            E.verify({result['path']: result['sha256']}); E.verify(C.read(result['path'])['files'])
        return old
    for arm in ARMS:
        if C.read(root/'evaluation'/arm/'post_complete.json')['outputs'] != len(reg['requests']):
            raise ValueError('All RAW/POST must finish before formal scoring')
        for q in reg['requests']:
            for stage in ('RAW', 'POST'):
                artifact(root, arm, q['request_id'], stage)
    jobs = [(str(root), arm, q, stage, C.sha(registration)) for q in reg['requests'] for arm in ARMS for stage in ('RAW', 'POST')]
    tick = time.perf_counter(); first = worker(jobs[0]); snapshot = C.resources()
    memory = {x.split(':')[0]: int(x.split()[1])*1024 for x in (snapshot.get('memory') or '').splitlines()
        if len(x.split()) > 1 and x.split()[1].isdigit()}
    can_parallel = ((os.cpu_count() or 1) >= 4 and memory.get('MemAvailable', 0) > max(4 << 30, 3*first['peak_RSS_KiB']*1024)
        and (snapshot['loadavg'] or [999])[0] < (os.cpu_count() or 1)-2)
    count = 2 if workers == 2 and can_parallel else 1
    C.write(root/'evaluation'/('score_dispatch_'+str(time.time_ns())+'.json'), dict(workers=count,
        threads_per_worker=1, first_authorized_cell=first, resources=snapshot))
    receipts = [first]
    with concurrent.futures.ProcessPoolExecutor(max_workers=count, mp_context=multiprocessing.get_context('spawn')) as pool:
        todo = iter(jobs[1:]); active = {}
        for _ in range(count):
            job = next(todo, None)
            if job is not None:
                active[pool.submit(worker, job)] = job
        while active:
            done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
            results = [future.result() for future in done]
            for future, result in zip(done, results):
                del active[future]; receipts.append(result)
                C.write(root/'evaluation/score_progress.json', dict(status='RUNNING', completed=len(receipts), registered=len(jobs)), replace=True)
                print(json.dumps(dict(stage='SCORE', completed=len(receipts), registered=len(jobs))), flush=True)
                job = next(todo, None)
                if job is not None:
                    active[pool.submit(worker, job)] = job
    E.verify(reg['protected']); E.verify(reg['source_hashes'])
    receipts.sort(key=lambda row: row['path'])
    result = dict(status='COMPLETE', cells=len(receipts), receipts=receipts,
        workers=count, seconds=time.perf_counter()-tick, registration_sha256=C.sha(registration),
        full_quality_scans=sum(not row['reused'] for row in receipts), model_calls=0, new_queries=0)
    if len(receipts) != reg['cells']:
        raise ValueError('Incomplete registered scoring matrix')
    C.write(completion, result)
    return result


def number(value):
    return isinstance(value, (int, float, bool)) and np.isfinite(float(value))


def aggregate(root):
    root = C.resolve(root); reg = prepare(root); complete = C.read(root/'evaluation/score_completion.json')
    if complete['cells'] != reg['cells']:
        raise ValueError('Incomplete score matrix')
    rows = []; roles = []; deleted = []
    for q in reg['requests']:
        for arm in ARMS:
            for stage in ('RAW', 'POST'):
                folder = root/'evaluation/scores'/arm/q['request_id']/stage
                score = C.read(folder/'score.json'); E.verify(score['files'])
                row, role, drop = metric_row(score, C.arrays(score['identity']['mesh_path']), C.read(folder/'pairs.json'))
                rows.append(row); roles.extend(role); deleted.extend(drop)
    numeric = sorted(key for key in set().union(*(row.keys() for row in rows))
        if key not in (*E.META, 'label', 'stage') and any(number(row.get(key)) for row in rows))
    if len(numeric) != 124:
        raise ValueError('Original 124 numerical field contract differs: '+str(len(numeric)))
    summaries = []; key_results = []
    for arm in ARMS:
        for stage in ('RAW', 'POST'):
            for split in ('TRAIN', 'DEV', 'ALL'):
                rr = [row for row in rows if row['label'] == arm and row['stage'] == stage and (split == 'ALL' or row['split'] == split)]
                key = dict(label=arm, stage=stage, split=split, n=len(rr))
                for metric in numeric:
                    values = [row[metric] for row in rr if number(row.get(metric))]
                    summaries.append(dict(label=arm, stage=stage, split=split, metric=metric,
                        n_registered=len(rr), n_valid=len(values), mean=float(np.mean(values)) if values else None,
                        median=float(np.median(values)) if values else None, min=float(min(values)) if values else None,
                        max=float(max(values)) if values else None, status='PASS' if len(values)==len(rr) else 'NA_OR_PARTIAL'))
                    key[metric] = float(np.mean(values)) if values else None
                key['bulk_reach30_count'] = sum(number(row.get('free_only_reached_bulk_area_fraction')) and row['free_only_reached_bulk_area_fraction'] >= .3 for row in rr)
                key_results.append(key)
    lookup = {(row['label'], row['stage'], row['request_id']): row for row in rows}; pairs = []
    for q in reg['requests']:
        for stage in ('RAW', 'POST'):
            plain, seam = lookup['PLAIN', stage, q['request_id']], lookup['SEAM', stage, q['request_id']]
            for metric in numeric:
                x, y = plain.get(metric), seam.get(metric); good = number(x) and number(y)
                pairs.append(dict(**{key:q[key] for key in E.META}, stage=stage, metric=metric,
                    baseline='PLAIN', candidate='SEAM', baseline_value=x, candidate_value=y,
                    delta=float(y)-float(x) if good else None, status='PASS' if good else 'NA_NOT_COMMON'))
    groups = collections.defaultdict(list)
    for row in pairs:
        groups[(row['parent_id'], row['split'], row['stage'], row['metric'])].append(row)
    parents = []
    for key, rr in sorted(groups.items()):
        good = [row for row in rr if row['status'] == 'PASS']
        parents.append(dict(parent_id=key[0], split=key[1], stage=key[2], metric=key[3],
            n_registered=len(rr), n_valid=len(good), **{name:float(np.mean([row[name] for row in good])) if good else None
                for name in ('baseline_value', 'candidate_value', 'delta')}, status='PASS' if len(good)==len(rr) else 'NA_OR_PARTIAL'))
    files = {}
    for name, records in (('per_output',rows), ('illegal_pairs_by_role',roles), ('deleted_faces',deleted),
        ('summary',summaries), ('key_results',key_results), ('paired_per_noise',pairs), ('paired_per_parent',parents)):
        path = root/'evaluation'/(name+'.csv'); C.csvwrite(path, records); files[str(path)] = C.sha(path)
    result = dict(status='NUMERICAL_COMPLETE_VISUAL_SEPARATE', rows=len(rows), numeric_fields=numeric,
        numeric_field_count=124, files=files, model_calls=0, new_queries=0,
        score_completion_sha256=C.sha(root/'evaluation/score_completion.json'),
        metric_policy='Complete original objective fields retained; source shape is not a unique generative GT.')
    C.write(root/'evaluation/summary_completion.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(ROOT)); parser.add_argument('--stage', choices=('prepare','score','aggregate'), required=True)
    parser.add_argument('--workers', type=int, choices=(1,2), default=2); args = parser.parse_args()
    result = prepare(args.root) if args.stage == 'prepare' else run(args.root,args.workers) if args.stage == 'score' else aggregate(args.root)
    print(json.dumps({key:value for key,value in result.items() if key not in ('requests','protected','source_hashes','receipts','numeric_fields','files')}))


if __name__ == '__main__':
    main()
