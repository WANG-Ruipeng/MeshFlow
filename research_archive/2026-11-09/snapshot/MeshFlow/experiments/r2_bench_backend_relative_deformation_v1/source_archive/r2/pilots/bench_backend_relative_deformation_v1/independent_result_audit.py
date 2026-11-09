"""Audit saved numbers, receipts and sample accounting without running scorers."""
from __future__ import annotations
import collections, csv, datetime, hashlib, json, math, statistics, time
from pathlib import Path
from .common import ROOT, OUT, OLD, PREV, read, resolve, sha, write


def number(value):
    if value is None or isinstance(value, bool) or value in ('', 'True', 'False'):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def same(a, b):
    a, b = number(a), number(b)
    return (a is None and b is None) or (a is not None and b is not None and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12))


def table(name):
    with (OUT / 'scores' / name).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def run():
    tick = time.perf_counter()
    errors, checks, expected, receipts, omitted_large = [], [], {}, [], []

    def check(ok, name, detail=None):
        checks.append(dict(name=name, passed=bool(ok), detail=detail))
        if not ok:
            errors.append(name)

    def receipt(path):
        result = read(path)
        receipts.append(str(path))
        for key in ('files', 'source_hashes', 'reference_hashes', 'protected'):
            for filename, digest in result.get(key, {}).items():
                p = resolve(filename)
                # Completed runtime already checks large checkpoint hashes. Do
                # not impose another multi-GB read while CPU scoring is active.
                if p.suffix in ('.pt', '.pth', '.ckpt'):
                    omitted_large.append(dict(path=str(p), expected_sha256=digest, exists=p.is_file()))
                    continue
                if str(p) in expected and expected[str(p)] != digest:
                    errors.append('Conflicting expected hash: ' + str(p))
                expected[str(p)] = digest
        return result

    train = receipt(OUT / 'training/complete.json')
    scores = receipt(OUT / 'scores/COMPLETE.json')
    receipt(OUT / 'evaluation/CPU_registration.json')
    receipt(OUT / 'gallery/COMPLETE.json')
    manifest = read(PREV / 'data/manifest.json')
    tasks = read(OLD / 'manifest.json')['tasks']
    arm_counts = {}
    for arm in ('CANDIDATE_RAW', 'CANDIDATE_SOURCE'):
        armroot = OUT / 'evaluation' / arm
        done = receipt(armroot / 'complete.json')
        outputs = [receipt(p) for p in sorted(armroot.glob('*/receipt.json'))]
        arm_counts[arm] = dict(outputs=len(outputs), D0_forwards=sum(r['D0_forwards'] for r in outputs),
            generator_forwards=sum(r['generator_forwards'] for r in outputs), new_noise_draws=sum(r['new_noise_draws'] for r in outputs))
        check(done['status'] == 'COMPLETE' and arm_counts[arm] == dict(outputs=30, D0_forwards=30, generator_forwards=0, new_noise_draws=0), 'exact_evaluation_' + arm)
    check(not (OUT / 'evaluation/OFFICIAL_SOURCE').exists(), 'official_source_reused_without_new_dispatch')

    ledger = [json.loads(x) for x in (OUT / 'training/ledger.jsonl').read_text().splitlines()]
    losses = [json.loads(x) for x in (OUT / 'training/losses.jsonl').read_text().splitlines()]
    event_counts = collections.Counter(r['event'] for r in ledger)
    for event, count in [('FORWARD_STARTED', 2000), ('FORWARD_RETURNED', 2000), ('BACKWARD_STARTED', 2000),
                         ('BACKWARD_RETURNED', 2000), ('OPTIMIZER_STEP_STARTED', 500), ('OPTIMIZER_STEP_RETURNED', 500)]:
        rows = [r for r in ledger if r['event'] == event]
        check(len(rows) == count and len({(r['update'], r.get('micro')) for r in rows}) == count, 'exact_' + event, len(rows))
    check([r['update'] for r in losses] == list(range(1, 501)), 'loss_updates_1_to_500')
    samples = [s for r in losses for s in r['samples']]
    check([s['row_index'] for s in samples] == [i for group in manifest['stream'] for i in group], 'exact_original_stream')
    check(all(len(r['samples']) == 4 and r['cumulative_forwards'] == 4*r['update'] and r['cumulative_backwards'] == 4*r['update'] for r in losses), 'loss_counters')
    for r in losses:
        for field in ('mse_denoiser', 'mse_model', 'relative_huber', 'weighted_relative', 'total_loss'):
            check(same(r[field], statistics.mean(s[field] for s in r['samples'])), 'loss_mean_' + field + '_' + str(r['update']))
        for s in r['samples']:
            check(same(s['weighted_relative'], .001*s['relative_huber']) and
                  math.isclose(s['total_loss'], s['mse_denoiser'] + s['weighted_relative'], rel_tol=3e-7, abs_tol=1e-12),
                  'loss_composition_FP32_roundoff_' + str(s['row_index']))
    exposures = collections.Counter(s['parent'] for s in samples)
    check(len(exposures) == 16 and set(exposures.values()) == {125}, 'parent_exposure_16x125')
    check(not set(exposures) & {r['parent_id'] for r in manifest['validation_rows']}, 'backend_train_eval_parent_disjoint')
    failures = [str(p.relative_to(OUT)) for p in OUT.rglob('*failure.json')]

    rows, new, refs = table('per_output.csv'), table('new_per_output.csv'), table('original_reference_scores.csv')
    fields = scores['metric_fields']
    by = {(r['label'], r['stage'], r['request_id']): r for r in rows}
    check(len(rows) == 330 and len(by) == 330 and len(new) == 60 and len(refs) == 270 and len(fields) == 177, 'complete_score_matrix')
    check(all(r['C_preserved_exact'] == 'True' and r['seam_applicability'] == 'NA_NOT_SEAM_REPRESENTATION' for r in new), 'exact_C_and_native_applicability')
    reference_cells = 0
    for r in refs:
        if r.get('source_metric_rows_path'):
            path = resolve(r['source_metric_rows_path'])
            original = [x for x in read(path) if x['stage'] == r['stage'] and x['label'] == r['label']]
            assert len(original) == 1
            original = original[0]
        else:
            path = resolve(r['source_score_path']).parent / 'row.json' if r.get('source_score_path') else OLD / 'scores' / r['request_id'] / 'row.json'
            original = read(path)
        expected[str(path)] = sha(path)
        for field in fields:
            reference_cells += 1
            if field not in original or not same(r.get(field), original.get(field)):
                errors.append('Reference value mismatch ' + r['request_id'] + ' ' + r['label'] + ' ' + r['stage'] + ' ' + field)
    new_cells = 0
    for t in tasks:
        path = OUT / 'cpu/RELDEF_D0' / t['request_id'] / 'metric_rows.json'
        receipt(path.parent / 'complete.json')
        receipt(OUT / 'geometry' / t['request_id'] / 'complete.json')
        for r in read(path):
            saved = by[('RELDEF_D0', r['stage'], t['request_id'])]
            for field in fields:
                new_cells += 1
                if not same(r[field], saved[field]):
                    errors.append('New value mismatch ' + t['request_id'] + ' ' + field)

    def aggregate(data, keys):
        grouped = collections.defaultdict(list)
        for r in data:
            grouped[tuple(r[k] for k in keys)].append(r)
        result = []
        for key, group in grouped.items():
            r = dict(zip(keys, key), rows=len(group))
            for field in fields:
                values = [number(x.get(field)) for x in group]
                values = [x for x in values if x is not None]
                r[field] = statistics.mean(values) if values else None
                r[field + '__valid_n'] = len(values)
            result.append(r)
        return result

    pk = ['label', 'stage', 'eval_group', 'parent_id']; ck = pk + ['condition_id']; mk = pk[:-1]
    conditions = aggregate(rows, ck); parents = aggregate(conditions, pk); macros = aggregate(parents, mk)
    aggregate_cells = 0
    for filename, data, keys in [('per_condition.csv', conditions, ck), ('per_parent.csv', parents, pk), ('parent_macro_summary.csv', macros, mk)]:
        saved = {tuple(r[k] for k in keys): r for r in table(filename)}
        check(len(saved) == len(data), 'aggregate_size_' + filename)
        for r in data:
            q = saved[tuple(r[k] for k in keys)]
            for field in ['rows'] + [name for f in fields for name in (f, f+'__valid_n')]:
                aggregate_cells += 1
                if not same(r[field], q[field]):
                    errors.append('Aggregate mismatch ' + filename + ' ' + str(tuple(r[k] for k in keys)) + ' ' + field)
    pairs = table('paired_per_output.csv'); pair_cells = 0
    check(len(pairs) == 420, '420_paired_rows')
    for p in pairs:
        a = by[(p['reference_label'], p['reference_stage'], p['request_id'])]
        b = by[(p['candidate_label'], p['candidate_stage'], p['request_id'])]
        for field in fields:
            av, bv = number(a.get(field)), number(b.get(field))
            dv = bv-av if av is not None and bv is not None else None
            for suffix, value in [('reference', av), ('candidate', bv), ('delta', dv)]:
                pair_cells += 1
                if not same(value, p[field+'__'+suffix]):
                    errors.append('Pair mismatch ' + p['request_id'] + ' ' + field + '__' + suffix)
    hash_count = 0
    for p, digest in sorted(expected.items()):
        if not Path(p).is_file() or sha(p) != digest:
            errors.append('File hash mismatch: ' + p)
        hash_count += 1
    selected = ['CF_illegal_affected_free_fraction', 'FF_illegal_affected_free_fraction', 'ConflictCov_C',
        'free_minimum_angle_lt5_fraction', 'actual_free_faces', 'deleted_free_faces', 'nonmanifold_edges',
        'C_F_valid_edge_count', 'opposite_winding_reached_free_area_fraction']
    key_macros = [{k: r[k] for k in mk + ['rows'] + selected} for r in macros if (r['label'], r['stage']) in
        [('COVERAGE_FINAL', 'CLEAN_FINAL_POST'), ('COVERAGE_FINAL', 'GEO_ONLY'), ('SHAPE_D0', 'CLEAN_FINAL_POST'), ('RELDEF_D0', 'CLEAN_FINAL_POST')]]
    result = dict(status='PASS' if not errors else 'FAIL', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        method='Independent stdlib CSV/JSON recomputation and receipt hashes; no production aggregate or scorer execution.',
        training_event_counts=dict(event_counts), evaluation_counts=arm_counts, parent_exposures=dict(exposures),
        sample_kinds=dict(collections.Counter(s['kind'] for s in samples)), recorded_failure_files=failures,
        reference_numeric_cells_checked=reference_cells, new_numeric_cells_checked=new_cells,
        aggregate_numeric_and_denominator_cells_checked=aggregate_cells, pair_cells_checked=pair_cells,
        hash_files_checked=hash_count, hash_receipts_checked=len(receipts), large_weights_not_rehashed=omitted_large,
        checks=checks, errors=errors, selected_parent_equal_metrics=key_macros,
        model_calls=0, GPU_calls=0, scoring_calls=0, geometry_calls=0, seconds=time.perf_counter()-tick,
        limitations=['This checks inherited metric arithmetic and provenance, not detector validity or visual quality.',
                     'Large checkpoint hashes rely on completed runtime receipts and are not read again here.',
                     'Source calibration is synthetic denoising, not trusted real-RAW repair supervision.'])
    write(OUT / 'audits/independent_result_audit.json', result)
    print(json.dumps(dict(status=result['status'], errors=len(errors), hash_files=hash_count,
        references=reference_cells, new_cells=new_cells, aggregate_cells=aggregate_cells, pair_cells=pair_cells)))
    if errors:
        raise RuntimeError('Independent result audit found mismatches: ' + str(errors[:5]))
    return result


if __name__ == '__main__':
    run()
