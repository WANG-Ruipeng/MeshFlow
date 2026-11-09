"""CPU-only HALF scoring and exact 64-cell baseline reuse.

Original formulas/functions are imported read-only. No model is loaded, no
baseline is rescored, and every reference query is copied from the frozen run.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import types

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '2')
import numpy as np
from experiments.mf_h1_local0.chair_capped500_gradbudget_v1 import evaluation as prior

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'native_t1/runs/chair_capped500_half_v1'
SOURCE = prior.OUT
VERSION = 'chair_capped500_half_evaluation_v1'
LABELS = ('START', 'HIST_CAPPED500', 'HALF_CAPPED500')
BASELINES = LABELS[:2]
PANELS = ('OLD', 'FRESH')
old, gates, quality, control = prior.old, prior.gates, prior.quality, prior.control
resolve, digest, read_json, arrays = prior.resolve, prior.digest, prior.read_json, prior.arrays
write_json, write_csv = prior.write_json, prior.write_csv
freeze_json, freeze_npz = prior.freeze_json, prior.freeze_npz


def assert_out(out):
    out = resolve(out).resolve()
    if out != OUT.resolve():
        raise ValueError('This adapter writes only the registered independent HALF run')
    return out


def manifest(out=OUT, panel='OLD'):
    return prior.manifest(assert_out(out), panel)


def sources():
    return dict(version=VERSION, adapter_sha256=digest(__file__),
                reused_scoring_adapter_sha256=digest(prior.__file__),
                scientific_sources=old.identities(), formulas='Unchanged original frozen formulas and queries')


def _clone(function, **overrides):
    namespace = dict(function.__globals__)
    namespace.update(overrides)
    result = types.FunctionType(function.__code__, namespace, function.__name__, function.__defaults__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def prepare_references(out=OUT):
    out = assert_out(out)
    seen = set()
    for request in manifest(out, 'ALL'):
        cid = request['condition_id']
        if cid in seen:
            continue
        seen.add(cid)
        for name in ('control.json', 'control.npz', 'quality.json', 'quality.npz'):
            source = SOURCE/'references'/cid/name
            target = out/'references'/cid/name
            expected = digest(source)
            if target.exists():
                if digest(target) != expected:
                    raise ValueError('Frozen reference bytes differ: '+str(target))
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                if digest(target) != expected:
                    raise ValueError('Reference copy failed identity check')
    # This helper sees all copied references, so its query-generation fallback
    # is unreachable. It verifies the stored internal array hashes as well.
    function = _clone(prior.prepare_references, OUT=OUT, assert_out=assert_out,
                      manifest=manifest, sources=sources)
    return function(out)


def _source_row(label, request):
    if label not in BASELINES:
        raise ValueError('Only the two registered baselines may reuse scores')
    panel = request['split']
    path = SOURCE/'evaluation'/label/panel/'per_output.json'
    analysis = read_json(SOURCE/'analysis/results.json')
    expected = analysis['source_files'].get(str(path))
    if not expected or digest(path) != expected:
        raise ValueError('Baseline per-output file differs from saved final analysis hash')
    rows = read_json(path)
    matches = [r for r in rows if r['request_id'] == request['request_id']]
    if len(rows) != 16 or len(matches) != 1:
        raise ValueError('Baseline cohort/request mismatch')
    return matches[0], path, expected


def result_for(out, label, request):
    out = assert_out(out)
    if label in BASELINES:
        row, _, _ = _source_row(label, request)
        path = resolve(row['result_path'])
        return path, read_json(path), resolve(row['raw_path'])
    if label != 'HALF_CAPPED500':
        raise ValueError('Unregistered label')
    return prior.result_for(out, label, request)


def verify_baseline(request, label, out=OUT):
    """Check old saved identities, arrays and export receipts without scoring."""
    out = assert_out(out)
    row, per_output, output_hash = _source_row(label, request)
    joined = SOURCE/'evaluation'/label/request['split']/'scoring/joined'/(request['request_id']+'.json')
    saved = read_json(joined)
    identity = saved['identity']
    if saved['metrics'] != row or row['status'] != 'PASS':
        raise ValueError('Joined baseline is not the exact saved per-output row')
    if identity['sources'] != prior.sources() or row['scoring_sources'] != old.identities():
        raise ValueError('Baseline scoring source identity changed')
    for key in ('label', 'request_id', 'panel'):
        wanted = label if key == 'label' else request['split'] if key == 'panel' else request['request_id']
        if identity[key] != wanted or row[key] != wanted:
            raise ValueError('Baseline label/panel/request mismatch')
    for key in ('uid', 'condition_id', 'candidate', 'noise_seed', 'N', 'K'):
        if row[key] != request[key]:
            raise ValueError('Baseline pair metadata mismatch: '+key)
    checked = {str(per_output): output_hash, str(joined): digest(joined)}
    for key, pathkey in (('condition', 'condition_npz'), ('bank', 'bank_npz')):
        expected = request[key+'_sha256']
        if identity[key+'_sha256'] != expected or digest(resolve(request[pathkey])) != expected:
            raise ValueError('Baseline input identity changed: '+key)
        checked[str(resolve(request[pathkey]))] = expected
    for key in ('raw', 'result', 'vectors', 'control_vectors'):
        path = resolve(row[key+'_path'])
        if digest(path) != row[key+'_sha256']:
            raise ValueError('Baseline source changed: '+key)
        checked[str(path)] = row[key+'_sha256']
    condition = old._condition(request, out)
    raw = old._adapted_arrays(resolve(row['raw_path']))
    result = read_json(resolve(row['result_path']))
    facts = old._contracts(request, raw, condition, result)
    if not all(facts.get(key) is True for key in gates.HARD):
        raise ValueError('Baseline hard coordinate/export contract not established')
    if quality.array_sha(raw['raw']) != row['geometry_sha256']:
        raise ValueError('Baseline raw array hash changed')
    reference = old._reference(out, request, condition)
    qb = old._queries(out, request, condition)
    if (row['Q_Y2048_sha256'] != quality.array_sha(qb['Q_Y2048']) or
            row['legacy_query_sha256'] != quality.array_sha(condition['legacy_query_points'])):
        raise ValueError('Baseline fixed query identity changed')
    for name in ('control.json', 'control.npz', 'quality.json', 'quality.npz'):
        path = out/'references'/request['condition_id']/name
        checked[str(path)] = digest(path)
    audit = dict(status='PASS', source_files=checked, source_label=label,
                 request_id=request['request_id'], panel=request['split'],
                 reference_identity=reference['identity'], metrics_recomputed=0,
                 raw_C_noise_all51_and_exports=True, baseline_checkpoint_sha256=row['checkpoint_sha256'])
    return row, audit


def evaluate_request(request, raw_path=None, out_dir=OUT, label='HALF_CAPPED500'):
    out = assert_out(out_dir)
    if label not in LABELS or request['split'] not in PANELS:
        raise ValueError('Unregistered label or panel')
    if label in BASELINES:
        started = time.perf_counter()
        row, audit = verify_baseline(request, label, out)
        identity = dict(version=VERSION, sources=sources(), label=label,
                        request_id=request['request_id'], panel=request['split'],
                        condition_sha256=request['condition_sha256'], bank_sha256=request['bank_sha256'],
                        raw_sha256=row['raw_sha256'], result_sha256=row['result_sha256'],
                        source_files=audit['source_files'])
        dest = out/'evaluation'/label/request['split']/'scoring/joined'/(request['request_id']+'.json')
        if dest.exists():
            saved = read_json(dest)
            if saved['identity'] != identity:
                raise ValueError('Baseline cache changed after independent materialization')
            return saved['metrics']
        # Preserve scientific values exactly; make this run's cache cost explicit.
        derived = dict(row, source_generation_status=row.get('generation_status'),
                       source_new_scoring_seconds=row.get('new_scoring_seconds'),
                       generation_status='CACHED', generation_time_source='historical_generation',
                       new_scoring_seconds=0., cache_verification_seconds=time.perf_counter()-started)
        freeze_json(dest, dict(identity=identity, metrics=derived, cache_audit=audit))
        return derived
    _, result, registered_raw = result_for(out, label, request)
    if result.get('status') not in ('SAMPLED', 'PASS', 'DONE', 'FAILED', 'FAIL', 'BLOCKED', 'NOT_RUN'):
        return dict(request_id=request['request_id'], uid=request['uid'], candidate=request['candidate'],
                    condition_id=request['condition_id'], label=label, panel=request['split'],
                    split=request['split'], status='NOT_RUN', reason='Completed result receipt required')
    # Isolated globals retain the exact prior scorer mathematics, without
    # mutating the prior module or permitting its historical-rescore branches.
    function = _clone(prior.evaluate_request, VERSION=VERSION, OUT=OUT, LABELS=LABELS,
                      assert_out=assert_out, sources=sources, result_for=result_for,
                      _cache_pointer=lambda *_: None)
    return function(request, raw_path=raw_path or registered_raw, out_dir=out, label=label)


def _job(arguments):
    request, out, label = arguments
    started = time.perf_counter()
    try:
        row = evaluate_request(request, out_dir=out, label=label)
    except Exception as error:
        row = {key: request.get(key) for key in ('request_id', 'uid', 'condition_id', 'candidate', 'N', 'K', 'split')}
        row.update(label=label, status='SCORING_BLOCKED', reason=repr(error), traceback=traceback.format_exc())
    return row, time.perf_counter()-started


def score_set(out=OUT, label='HALF_CAPPED500', panel='OLD', *, workers=2, available_only=False):
    out = assert_out(out)
    if label not in LABELS or workers not in (1, 2):
        raise ValueError('Unregistered label or CPU concurrency')
    if panel == 'ALL':
        stages = [score_set(out, label, p, workers=workers, available_only=available_only) for p in PANELS]
        rows = sum((read_json(out/'evaluation'/label/p/'per_output.json') for p in PANELS), [])
        elapsed = sum(s['invocation_seconds'] for s in stages)
    else:
        prepare_references(out)
        requests = manifest(out, panel)
        started = time.perf_counter()
        jobs = [(r, str(out), label) for r in requests]
        if workers == 1:
            values = list(map(_job, jobs))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                values = list(pool.map(_job, jobs))
        rows = [row for row, _ in values]
        elapsed = time.perf_counter()-started
        journal = out/'evaluation'/label/panel/'invocations.jsonl'
        journal.parent.mkdir(parents=True, exist_ok=True)
        with journal.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), label=label, panel=panel,
                workers=workers, worker_threads=2, wall_seconds=elapsed,
                worker_seconds=sum(seconds for _, seconds in values), completed=sum(r['status']=='PASS' for r in rows),
                denominator_retained=16, model_calls=0, baseline_metrics_recomputed=0))+'\n')
    summary = gates.summarize(rows, manifest(out, panel))
    summary.update(label=label, panel=panel, invocation_seconds=elapsed, cpu_workers=workers,
                   historical_scores_reused=sum(r.get('generation_status')=='CACHED' for r in rows),
                   new_scored_outputs=sum(r.get('generation_status')=='NEW_GENERATED' for r in rows),
                   model_calls=0, baseline_metrics_recomputed=0)
    folder = out/'evaluation'/label/panel
    write_json(folder/'per_output.json', rows)
    write_csv(folder/'per_output_metrics.csv', rows)
    write_json(folder/'summary.json', summary)
    write_csv(folder/'per_parent_metrics.csv', summary['parents'])
    write_csv(folder/'per_noise_metrics.csv', summary['noise'])
    return summary


def freeze_baseline_manifest(out=OUT):
    out = assert_out(out)
    prepare_references(out)
    records = []
    for label in BASELINES:
        for panel in PANELS:
            for request in manifest(out, panel):
                row = evaluate_request(request, out_dir=out, label=label)
                saved = read_json(out/'evaluation'/label/panel/'scoring/joined'/(request['request_id']+'.json'))
                records.append(dict(saved['cache_audit'], label=label,
                                    geometry_sha256=row['geometry_sha256']))
    result = dict(status='PASS', cells=64, sources=sources(), records=records,
                  source_analysis_sha256=digest(SOURCE/'analysis/results.json'),
                  model_calls=0, metrics_recomputed=0)
    freeze_json(out/'registration/baseline_cache_manifest.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(OUT))
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--baseline-manifest', action='store_true')
    parser.add_argument('--label', choices=LABELS, default='HALF_CAPPED500')
    parser.add_argument('--panel', choices=(*PANELS, 'ALL'), default='OLD')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    result = freeze_baseline_manifest(args.out) if args.baseline_manifest else prepare_references(args.out) if args.prepare else score_set(args.out, args.label, args.panel, workers=args.workers)
    print(json.dumps(quality.clean(result), ensure_ascii=False))
