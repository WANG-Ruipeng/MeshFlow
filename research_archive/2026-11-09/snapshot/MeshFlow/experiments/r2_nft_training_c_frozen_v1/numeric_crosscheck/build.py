"""Independent one-thread numeric-only aggregation and later table comparison.

Reads manifest, registration and existing scores; --compare reads only numeric
analysis tables. No review files, mapping, images, model, or geometry scorer.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
EV = OUT / 'evaluation'
LABELS = ('BASE', 'NFT', 'SELF_FM')
METRIC_MAP = dict(CF_illegal_fraction='CF_illegal_affected_free_fraction',
                  FF_illegal_fraction='FF_illegal_affected_free_fraction',
                  I_lower='ConflictCov_C', I_upper='ConflictCov_C_upper',
                  free_near_degenerate_fraction='free_near_degenerate_fraction',
                  free_small_angle_fraction='free_minimum_angle_lt5_fraction',
                  free_surface_area='free_total_surface_area')
METRICS = tuple(METRIC_MAP) + ('free_retained_fraction', 'legal_CF_edges', 'legal_CF_length', 'reached_free_area_fraction')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Refuse changed crosscheck artifact: ' + str(path))
        return
    with path.open('xb') as f:
        f.write(data)


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    data = buffer.getvalue().encode('utf-8-sig')
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Refuse changed crosscheck CSV: ' + str(path))
        return
    with path.open('xb') as f:
        f.write(data)


def number(value):
    if value is None or value == '':
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def stats(values):
    finite = [v for v in values if v is not None]
    return dict(count=len(values), finite=len(finite), unavailable=len(values) - len(finite),
                mean=sum(finite) / len(finite) if finite else None,
                median=statistics.median(finite) if finite else None,
                minimum=min(finite) if finite else None, maximum=max(finite) if finite else None)


def summarize(rows):
    cids = sorted({r['condition_id'] for r in rows})
    result = dict(label=rows[0]['label'], stage=rows[0]['stage'], outputs=len(rows), conditions=len(cids),
                  interface_outputs=sum(r['has_legal_CF'] for r in rows),
                  legal_CF_edges_sum=sum(r['legal_CF_edges'] for r in rows),
                  legal_CF_length_sum=sum(r['legal_CF_length'] for r in rows),
                  actual_free_faces_sum=sum(r['actual_free_faces'] for r in rows),
                  original_free_faces_sum=sum(r['original_free_faces'] for r in rows),
                  I_status_counts=dict(Counter(r['I_status'] for r in rows)),
                  topology_status_counts=dict(Counter(r['topology_status'] for r in rows)), metrics={})
    for metric in METRICS:
        result['metrics'][metric] = dict(output_distribution=stats([r[metric] for r in rows]),
            equal_condition_distribution=stats([stats([r[metric] for r in rows if r['condition_id'] == cid])['mean'] for cid in cids]))
    return result


def build():
    mp, rp = OUT / 'inputs/manifest.json', OUT / 'registration.json'
    manifest, registration = read(mp), read(rp)
    assert manifest['status'] == registration['status'] == 'FROZEN'
    assert registration['manifest_sha256'] == sha(mp)
    regsha = sha(rp)
    qs = manifest['evaluation_requests']
    assert len(qs) == 32 and len({q['request_id'] for q in qs}) == 32
    assert sorted(Counter(q['condition_id'] for q in qs).values()) == [4] * 8
    assert len({q['parent_id'] for q in qs}) == 8 and {q['split'] for q in qs} == {'TRAIN'}
    paths = [EV / label / 'scores' / q['request_id'] / stage / 'score.json'
             for label in LABELS for q in qs for stage in ('RAW', 'POST')]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        print(json.dumps(dict(status='INCOMPLETE', missing_count=len(missing))))
        return
    sources = {str(p): sha(p) for p in (mp, rp, Path(__file__).resolve())}
    definitions = None
    rows = []
    for label in LABELS:
        for q in qs:
            for stage in ('RAW', 'POST'):
                path = EV / label / 'scores' / q['request_id'] / stage / 'score.json'
                sources[str(path)] = sha(path)
                s = read(path)
                ident = s['identity']
                for key, value in (('label', label), ('request_id', q['request_id']), ('condition_id', q['condition_id']),
                                   ('N', q['N']), ('K', q['K']), ('stage', stage), ('registration_sha256', regsha)):
                    assert ident[key] == value, (str(path), key)
                assert s['status'] == 'PASS' and s['stage'] == stage and s['C_preserved_exact'] is True
                assert s['original_definition_unchanged'] is True and s['model_calls'] == 0
                if definitions is None:
                    definitions = ident['definition_hashes']
                    for p, h in definitions.items():
                        assert sha(p) == h, p
                        sources[p] = h
                else:
                    assert ident['definition_hashes'] == definitions
                m, t = s['metrics'], s['topology']
                actual, original = int(m['actual_free_faces']), q['N'] - q['K']
                assert actual == s['free_face_denominator']
                assert actual + m['deleted_free_faces'] == original
                assert s['CF_tested_pair_denominator'] == actual * q['K']
                assert s['FF_tested_pair_denominator'] == actual * (actual - 1) // 2
                values = {key: number(m.get(field)) for key, field in METRIC_MAP.items()}
                values.update(free_retained_fraction=actual / original, legal_CF_edges=int(t['C_F_valid_edge_count']),
                              legal_CF_length=float(t['C_F_valid_edge_total_length']),
                              reached_free_area_fraction=number(t.get('reached_free_area_fraction')))
                assert values['I_lower'] is None or values['I_upper'] is None or values['I_lower'] <= values['I_upper']
                row = dict(label=label, stage=stage, request_id=q['request_id'], condition_id=q['condition_id'],
                           parent_id=q['parent_id'], candidate=q['candidate'], split=q['split'], N=q['N'], K=q['K'],
                           actual_free_faces=actual, original_free_faces=original, deleted_free_faces=int(m['deleted_free_faces']),
                           has_legal_CF=int(values['legal_CF_edges'] > 0), I_status=m.get('ConflictCov_C_status', 'UNKNOWN'),
                           topology_status=t['status'], CF_unresolved_pairs=int(m.get('CF_unresolved_or_degenerate_pair_count', 0)),
                           FF_unresolved_pairs=int(m.get('FF_unresolved_or_degenerate_pair_count', 0)),
                           source_C_connectivity_status=s.get('source_C_connectivity_status', 'UNKNOWN'),
                           score_path=str(path), score_sha256=sources[str(path)], **values)
                rows.append(row)
    groups, conditions = [], []
    for label in LABELS:
        for stage in ('RAW', 'POST'):
            selected = [r for r in rows if r['label'] == label and r['stage'] == stage]
            groups.append(summarize(selected))
            for cid in sorted({r['condition_id'] for r in selected}):
                local = [r for r in selected if r['condition_id'] == cid]
                conditions.append(dict(condition_id=cid, parent_id=local[0]['parent_id'], **summarize(local)))
    assert len(rows) == 192 and len(conditions) == 48 and 'torch' not in sys.modules
    for p, h in sources.items():
        assert sha(p) == h, p
    result = dict(version='nft_training_C_independent_numeric_crosscheck_v1', status='COMPLETE', scope='NUMERIC_ONLY',
                  stage_rows=192, distinct_requests=32, conditions=8, noise_per_condition=4, groups=groups,
                  definitions=definitions, model_calls=0, geometry_rescored=False, review_or_mapping_read=False,
                  aggregation='Within each original C average its four registered noises, then average all eight C equally.',
                  units='Fractions are 0..1; length retains original model coordinates. CF/FF divide by actual free faces of that stage.',
                  comparison_status='PENDING_MAIN_ANALYSIS')
    csv_write(HERE / 'per_output_stage.csv', rows)
    csv_write(HERE / 'per_condition_stage.csv', conditions)
    write(HERE / 'per_output_stage.json', rows)
    write(HERE / 'per_condition_stage.json', conditions)
    write(HERE / 'input_sources.json', sources)
    write(HERE / 'summary.json', result)
    print(json.dumps(dict(status='COMPLETE', stage_rows=192, condition_stage_groups=48, definition_sources=len(definitions))))
    for g in groups:
        print(json.dumps(dict(label=g['label'], stage=g['stage'], interface_outputs=g['interface_outputs'],
            legal_edges=g['legal_CF_edges_sum'], legal_length=g['legal_CF_length_sum'], I_status=g['I_status_counts'],
            means={k: g['metrics'][k]['equal_condition_distribution']['mean'] for k in METRICS})))


def compare():
    main = EV / 'analysis'
    required = [main / 'summary.json', main / 'per_output_stage.csv', main / 'per_condition.json',
                HERE / 'summary.json', HERE / 'per_output_stage.json', HERE / 'per_condition_stage.json']
    if any(not p.is_file() for p in required):
        print(json.dumps(dict(status='PENDING_MAIN_ANALYSIS', missing=[str(p) for p in required if not p.is_file()])))
        return
    source_hashes = {str(p): sha(p) for p in required + [Path(__file__).resolve()]}
    ms, own = read(required[0]), read(required[3])
    assert ms['status'] == own['status'] == 'COMPLETE'
    assert ms['outputs'] == 96 and ms['RAW_scores'] == ms['POST_scores'] == 96
    with required[1].open(encoding='utf-8-sig', newline='') as f:
        mainrows = list(csv.DictReader(f))
    mindex = {(r['label'], r['request_id'], r['stage']): r for r in mainrows}
    independent = read(required[4])
    assert len(mainrows) == len(mindex) == 192 and set(mindex) == {(r['label'], r['request_id'], r['stage']) for r in independent}
    errors, checked = [], Counter()
    maximum_difference = 0.0

    def check(scope, identity, field, a, b, numeric=False):
        nonlocal maximum_difference
        checked[scope] += 1
        if numeric:
            a, b = number(a), number(b)
            same = a == b if a is None or b is None else math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
            if a is not None and b is not None:
                maximum_difference = max(maximum_difference, abs(a - b))
        else:
            same = a == b
        if not same:
            errors.append(dict(scope=scope, identity=identity, field=field, independent=a, main=b))

    numeric = set(METRICS) | {'candidate', 'actual_free_faces', 'original_free_faces', 'deleted_free_faces',
                             'has_legal_CF', 'CF_unresolved_pairs', 'FF_unresolved_pairs'}
    excluded = {'N', 'K'}
    for r in independent:
        key = r['label'], r['request_id'], r['stage']
        target = mindex[key]
        for field, value in r.items():
            if field in excluded:
                continue
            if field not in target:
                raise ValueError('Main stage table lacks expected numeric field: ' + field)
            check('per_output_stage', list(key), field, value, target[field], numeric=field in numeric)
    mgroups = {g['label']: g for g in ms['groups']}
    mconditions = {(g['label'], g['condition_id']): g for g in read(required[2])}
    assert len(mgroups) == 3 and len(mconditions) == 24
    for scope, groups in (('aggregate', own['groups']), ('condition', read(required[5]))):
        for g in groups:
            key = (g['label'], g.get('condition_id'), g['stage'])
            target = mgroups[g['label']] if scope == 'aggregate' else mconditions[g['label'], g['condition_id']]
            check(scope, key, 'outputs', g['outputs'], target['outputs'])
            check(scope, key, 'conditions', g['conditions'], target['conditions'])
            check(scope, key, 'I_status', g['I_status_counts'], target[g['stage'] + '_I_status_counts'])
            check(scope, key, 'topology_status', g['topology_status_counts'], target[g['stage'] + '_topology_status_counts'])
            if g['stage'] == 'POST':
                check(scope, key, 'POST_interface_outputs', g['interface_outputs'], target['POST_interface_outputs'])
            for metric in METRICS:
                for distribution in ('output_distribution', 'equal_condition_distribution'):
                    for stat, value in g['metrics'][metric][distribution].items():
                        check(scope, key, '/'.join((metric, distribution, stat)), value,
                              target['metrics'][g['stage'] + '_' + metric][distribution][stat], numeric=True)
    check('definitions', 'all', 'definitions', own['definitions'], ms['definitions'])
    for p, h in source_hashes.items():
        assert sha(p) == h, p
    result = dict(version='nft_training_C_numeric_comparison_v1', status='FAIL' if errors else 'PASS',
                  stage_rows=192, condition_stage_groups=48, checks=dict(checked), total_checks=sum(checked.values()),
                  tolerance=dict(relative=1e-12, absolute=1e-12), maximum_absolute_numeric_difference=maximum_difference,
                  exceptions=errors, sources=source_hashes, main_results_modified=False, review_or_mapping_read=False,
                  model_calls=0, geometry_rescored=False)
    write(HERE / 'comparison.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'sources'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()
    compare() if args.compare else build()
