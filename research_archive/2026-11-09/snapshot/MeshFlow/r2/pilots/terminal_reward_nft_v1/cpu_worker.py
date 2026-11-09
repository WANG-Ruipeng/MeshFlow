"""Bounded model-free POST / original score / actual-image gallery workers.

GPU sampling and official D0 proposals must already be complete. This module
cannot load weights or substitute a CPU D0 call for a missing proposal.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import concurrent.futures
from contextlib import contextmanager
import json
import multiprocessing
from pathlib import Path
import sys
import time
import traceback

from . import data as D

VERSION = 'terminal_reward_nft_cpu_worker_v1'
QUERY_PATH = D.OUT/'inputs/training_queries.json'


def torch_free():
    if 'torch' in sys.modules:
        raise RuntimeError('CPU pipeline must remain genuinely torch-free')


def _force_threads():
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'


@contextmanager
def _single_thread():
    # NumPy is first imported under the module one-thread environment.
    # Preload SciPy BLAS likewise before legacy imports temporarily set env=2.
    _force_threads()
    import scipy.linalg
    try:
        yield
    finally:
        _force_threads()


def prepare_training_queries():
    """Freeze eight C2048 banks with the unchanged original query definition."""
    torch_free()
    manifest_path = D.OUT/'inputs/manifest.json'
    source_sha = D.sha(manifest_path)
    m = D.read(manifest_path)
    if QUERY_PATH.exists():
        q = D.read(QUERY_PATH)
        if q['manifest_sha256'] != source_sha:
            raise ValueError('Training query source manifest changed')
        for row in q['queries']:
            if D.sha(row['path']) != row['sha256']:
                raise ValueError('Frozen training query changed')
        return q
    if any((D.OUT/'training').glob('**/calls.jsonl')):
        raise RuntimeError('Freeze C queries before the first pilot model call')
    # Imported code has no model imports; restore its historical env defaults.
    from r2.pilots.user_condition_finetune_v1 import evaluation as original
    _force_threads()
    queries = []
    with _single_thread():
        for condition in m['train_conditions']:
            c = D.arrays(condition['condition_source'])['C'].reshape(condition['K'], 3, 3)
            if D.sha(condition['condition_source']) != condition['condition_source_sha256']:
                raise ValueError('Training condition changed')
            values, seed = original.query_bank(c, condition['condition_id'])
            path = D.save_npz(D.OUT/'inputs/training_queries'/(condition['condition_id']+'.npz'), **values)
            queries.append(dict(condition_id=condition['condition_id'], parent_id=condition['parent_id'],
                path=str(path), sha256=D.sha(path), query_count=2048,
                array_sha256={k:D.array_sha(v) for k,v in values.items()},
                identity=dict(condition_id=condition['condition_id'], C_seed=int(seed),
                              C_sha256=D.array_sha(c), L0=D.L),
                weight_denominator=float(values['C_weights'].sum()),
                condition_source=condition['condition_source'],
                condition_source_sha256=condition['condition_source_sha256'],
                split='TRAIN', panel='NFT_TRAIN_INPUT_SELECTED'))
    torch_free()
    result = dict(status='FROZEN', version=VERSION, manifest_sha256=source_sha,
        queries=queries, query_count_per_condition=2048, conditions=8, model_calls=0,
        query_definition='Original user_condition_finetune_v1.evaluation.query_bank; original C area/barycentric/normal/open-edge-distance definition',
        query_source=str(Path(original.__file__)), query_source_sha256=D.sha(original.__file__),
        original_source_hashes=original.scorer_sources(),
        new_queries=8*2048, output_geometry_modified=False, training_budget_changed=False)
    D.write(QUERY_PATH, result)
    if D.sha(manifest_path) != source_sha:
        raise ValueError('Query preparation changed frozen inputs')
    return result


def _verify_files(files):
    for path, h in files.items():
        if D.sha(path) != h:
            raise ValueError('Completed CPU artifact changed: ' + str(path))


def _query_map(m):
    sidecar = D.read(QUERY_PATH)
    if sidecar['manifest_sha256'] != D.sha(D.OUT/'inputs/manifest.json'):
        raise ValueError('Query manifest source differs')
    result = {q['condition_id']: q for q in sidecar['queries']}
    if len(result) != 8 or set(result) != {q['condition_id'] for q in m['train_conditions']}:
        raise ValueError('Expected exactly eight selected training query banks')
    _verify_files({q['path']: q['sha256'] for q in result.values()})
    return result


def _post(q, root, raw):
    from . import evaluation as E
    dest = root/'post'/q['request_id']
    done = dest/'receipt.json'
    proposal_dir = root/'proposal'/q['request_id']
    proposal = proposal_dir/'proposal.npz'
    proposal_receipt = proposal_dir/'receipt.json'
    if done.exists():
        saved = D.read(done)
        _verify_files(saved['files'])
        if saved['request_id'] != q['request_id'] or saved['input_sha256'] != q['input_sha256']:
            raise ValueError('Cached POST identity changed')
        return saved
    if dest.exists():
        raise RuntimeError('Unfinished CPU POST cell; explicit recovery required: ' + str(dest))
    proof = D.read(proposal_receipt)
    if proof['status'] != 'PASS' or proof['model_forwards'] != 1:
        raise ValueError('Exactly one completed official D0 proposal is required')
    result = E.postprocess_one(raw['raw_path'], q['condition_source'], q['K'], dest,
                              proposal_path=proposal, proposal_receipt=proposal_receipt)
    torch_free()
    if result['model_forwards_this_invocation'] != 0:
        raise RuntimeError('CPU POST performed a forbidden model call')
    files = {str(p):D.sha(p) for p in (dest/'post.npz', dest/'post_canonical.npz', dest/'denoised.npz',
              dest/'denoise.json', proposal, proposal_receipt, D.resolve(raw['raw_path']), D.resolve(q['condition_source']))}
    record = dict(status='COMPLETE', request_id=q['request_id'], input_sha256=q['input_sha256'],
                  version=VERSION, post_path=result['post_path'], post_sha256=result['post_sha256'],
                  proposal_path=str(proposal), proposal_receipt=str(proposal_receipt), files=files,
                  model_calls=0, torch_imported=False, result=result)
    D.write(done, record)
    return record


def _cell(job):
    q, root_string, query, stages, registration_sha = job
    _force_threads()
    torch_free()
    from . import evaluation as E
    E.package()
    from .sample import verify_receipt
    root = Path(root_string)
    raw = D.read(root/'raw'/q['request_id']/'receipt.json')
    tick = time.perf_counter()
    with _single_thread():
        verify_receipt(q, raw)
        if 'post' in stages:
            post = _post(q, root, raw)
        else:
            post = D.read(root/'post'/q['request_id']/'receipt.json')
            _verify_files(post['files'])
        score = None
        if 'score' in stages:
            score = E.score_one(q, raw['raw_path'], post['post_path'], q['condition_source'], query,
                root/'scores'/q['request_id'], label='NFT', registration_sha256=registration_sha,
                stages=('RAW','POST'))
        torch_free()
    return dict(request_id=q['request_id'], post_receipt=str(root/'post'/q['request_id']/'receipt.json'),
                post_receipt_sha256=D.sha(root/'post'/q['request_id']/'receipt.json'),
                score_paths=score['score_paths'] if score else {}, seconds=time.perf_counter()-tick,
                model_calls=0, torch_imported=False)


def run_round(round_index, *, stages=('post','score','render'), workers=1):
    torch_free()
    if type(round_index) is not int or not 0 <= round_index < 16 or workers not in (1,2):
        raise ValueError('Registered rounds 0..15; measured CPU workers must be 1 or 2')
    stages = tuple(x for x in ('post','score','render') if x in stages)
    if not stages:
        raise ValueError('At least one registered stage is required')
    manifest_path = D.OUT/'inputs/manifest.json'
    m = D.read(manifest_path)
    queries = _query_map(m)
    round_info = m['rounds'][round_index]
    if round_info['round_index'] != round_index:
        raise ValueError('Round order changed')
    qmap = {q['request_id']:q for q in m['training_requests']}
    qs = [qmap[key] for key in round_info['request_ids']]
    if len(qs) != 32 or len({q['condition_id'] for q in qs}) != 4:
        raise ValueError('Expected four conditions x eight complete endpoints')
    root = D.OUT/'training'/f'round_{round_index:02d}'
    # Stage dependency: all32 RAW must be saved before any CPU postprocessing.
    for q in qs:
        r = D.read(root/'raw'/q['request_id']/'receipt.json')
        if r['status'] != 'COMPLETE' or r['input_sha256'] != q['input_sha256']:
            raise ValueError('Complete all registered RAW before Stage2')
        if 'post' in stages:
            D.read(root/'proposal'/q['request_id']/'receipt.json')
    regpath = D.OUT/'registration.json'
    registration_sha = D.sha(regpath) if regpath.exists() else D.sha(manifest_path)
    done = root/'cpu'/('_'.join(stages)+'_complete.json')
    if done.exists():
        r = D.read(done)
        if r['registration_sha256'] != registration_sha:
            raise ValueError('CPU completion registration changed')
        _verify_files(r['files'])
        return r
    tick = time.perf_counter()
    rows = []
    try:
        if any(s in stages for s in ('post','score')):
            jobs = [(q,str(root),queries[q['condition_id']],stages,registration_sha) for q in qs]
            if workers == 1:
                rows = [_cell(j) for j in jobs]
            else:
                with concurrent.futures.ProcessPoolExecutor(max_workers=workers,
                        mp_context=multiprocessing.get_context('spawn')) as pool:
                    rows = list(pool.map(_cell,jobs))
        gallery = None
        if 'render' in stages:
            from .render import render_panel
            cases = []
            for q in qs:
                r = D.read(root/'raw'/q['request_id']/'receipt.json')
                p = D.read(root/'post'/q['request_id']/'receipt.json')
                _verify_files(p['files'])
                cases.append(dict(q,label='NFT',raw_path=r['raw_path'],post_path=p['post_path'],condition_path=q['condition_source']))
            with _single_thread():
                gallery = render_panel(cases,root/'gallery',label_order=['NFT'],expected_cases=32,workers=workers)
        torch_free()
        files = {}
        for row in rows:
            files[row['post_receipt']] = row['post_receipt_sha256']
            files.update({p:D.sha(p) for p in row['score_paths'].values()})
        if gallery is not None:
            files[str(root/'gallery/manifest.json')] = D.sha(root/'gallery/manifest.json')
        result = dict(status='COMPLETE',version=VERSION,round_index=round_index,stages=stages,
                      registration_sha256=registration_sha,manifest_sha256=D.sha(manifest_path),
                      training_queries_sha256=D.sha(QUERY_PATH),requests=32,workers=workers,
                      CPU_threads_per_worker=1,model_calls=0,torch_imported=False,
                      seconds=time.perf_counter()-tick,rows=rows,files=files,
                      gallery_manifest=str(root/'gallery/manifest.json') if gallery else None)
        D.write(done,result)
        return result
    except BaseException as exc:
        failure = root/'cpu'/('_'.join(stages)+'_failure.json')
        if not failure.exists():
            D.write(failure,dict(status='FAILED_NO_AUTO_RETRY',round_index=round_index,stages=stages,
                error=repr(exc),traceback=traceback.format_exc(),model_calls=0))
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--round',type=int)
    parser.add_argument('--stage',default='all',help='post, score, render, all, or comma-separated combination')
    parser.add_argument('--workers',type=int,choices=(1,2),default=1)
    parser.add_argument('--prepare-queries',action='store_true')
    args=parser.parse_args()
    if args.prepare_queries:
        result=prepare_training_queries()
        print(json.dumps(dict(status=result['status'],queries=len(result['queries']),model_calls=0)),flush=True)
        return
    if args.round is None:
        parser.error('--round is required for CPU pipeline execution')
    stages=('post','score','render') if args.stage=='all' else tuple(args.stage.replace('+',',').split(','))
    if any(s not in ('post','score','render') for s in stages):
        parser.error('Unknown CPU stage')
    result=run_round(args.round,stages=stages,workers=args.workers)
    print(json.dumps(dict(status=result['status'],round_index=args.round,stages=result['stages'],
                         workers=result['workers'],seconds=result['seconds'],model_calls=0)),flush=True)


if __name__=='__main__':
    main()
