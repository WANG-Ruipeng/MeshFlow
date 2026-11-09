"""Derived boundary components from saved closest points; no matching or scoring.

All outputs are independent sidecars. Source normals are source winding, never
inside/outside, and a closest finite face/edge is not semantic or legal evidence.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key, '2')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from . import common as C

VERSION = 'full_seat_saved_closest_point_components_v1'
MAIN = ('FULL_S0000', 'FULL_S1000', 'FULL_S3000', 'FULL_S6000')
REFERENCES = ('SMALL_C200', 'CREF_C200')
TARGETS = ('surface', 'edge', 'open_edge')
STAGES = ('RAW', 'POST')
OUT = C.OUT / 'evaluation/boundary_components_v1'
METRICS = tuple(x + '_' + stat + '_over_L' for x in ('normal_abs', 'tangent_plane') for stat in ('mean', 'p95'))
META = ('request_id', 'parent_id', 'condition_id', 'candidate', 'panel', 'split', 'N', 'K')
DEFINITION = dict(version=VERSION,
    targets=dict(surface='Saved nearest finite F triangle', edge='Saved nearest finite segment of all F edges',
                 open_edge='Saved nearest finite F edge with exactly one F incidence, including edges also incident to C'),
    displacement='delta = saved_closest_point - saved_C_boundary_point',
    signed_normal='s = dot(delta, saved_source_C_unit_normal), identical to original sample field',
    absolute_normal='abs(s)', tangent_plane='norm(delta - s * saved_source_C_unit_normal)',
    tangent_warning='Two-dimensional source C tangent-plane residual; NOT one-dimensional displacement along the boundary tangent.',
    weighting='Original positive length_weights on EDGE_MIDPOINT samples only. Boundary vertices remain excluded with zero weight.',
    mean='sum(component * valid_length_weight) / sum(valid_length_weight)',
    p95='Stable ascending sort; first value whose cumulative valid length reaches 0.95 of total valid length. Same inverse-CDF convention as the original boundary helper.',
    invalid_policy='Retain invalid point and length counts. Statistics use valid length only and status PARTIAL_NA; no finite candidate becomes NA, never zero.',
    source_normal_unit_tolerance=1e-10,
    invalid_codes=dict(source_normal_not_good=1, missing_or_nonfinite_closest_candidate=2,
                       nonfinite_query=4, source_normal_not_unit_or_nonfinite=8),
    pairing='Step0 subtraction only if original query-domain hash and target valid-query-mask hash match; no unpaired replacement.',
    parent_aggregation='Arithmetic macro mean of four per-output values; mean of output p95 is explicitly not a pooled-point p95.',
    split_aggregation='Equal-parent macro mean, separately TRAIN/DEV and ALL. No significance test or independent-seed claim.',
    limitations=['Source-normal sign is not inside/outside.', 'Closest candidates do not certify intended use, legality, thickness or actual common edges.',
                 'All source boundary is descriptive; it is not a required interface or S/SJoin definition.',
                 'These are inherited discrete midpoint approximations, not continuous contour guarantees.',
                 'RAW and POST stay separate; positive component deltas mean larger offsets but are not a universal quality verdict.'],
    new_matching_calls=0, new_quality_scans=0, new_queries=0, model_calls=0, new_geometry=0)


def digest(value):
    return hashlib.sha256(json.dumps(C.clean(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verify(path, expected):
    actual = C.sha(path)
    if actual != expected:
        raise ValueError('Frozen source changed: ' + str(path))
    return actual


def quantile(values, weights, fraction=.95):
    order = np.argsort(values, kind='stable'); cumulative = np.cumsum(weights[order])
    index = min(int(np.searchsorted(cumulative, fraction*cumulative[-1], side='left')), len(order)-1)
    return float(values[order[index]])


def _number(value):
    if value in (None, ''):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _cell(q, label, stage, ref, source_receipt_path, registration_sha):
    identity = ref['identity']
    if ref['status'] != 'COMPLETE' or (identity['label'], identity['request_id'], identity['stage']) != (label, q['request_id'], stage):
        raise ValueError('Boundary completion identity differs')
    if identity['registration_sha256'] != registration_sha or identity['source_sha256'] != q['condition_source_sha256']:
        raise ValueError('Boundary registration or source C differs')
    paths = dict(vectors=ref['vectors_path'], summary=ref['summary_path'])
    paths['samples'] = next(p for p in ref['files'] if Path(p).name == 'samples.csv')
    hashes = {str(p): verify(p, ref['files'][str(p)]) for p in paths.values()}
    code = {str(Path(__file__)): C.sha(__file__), str(Path(C.__file__)): C.sha(C.__file__)}
    proof = dict(version=VERSION, boundary_identity=identity, boundary_receipt_digest=digest(ref),
                 definition=DEFINITION, input_hashes=hashes, source_hashes=code)
    folder = OUT/'cells'/label/q['request_id']/stage
    final = folder/'completion.json'
    if final.exists():
        old = C.read(final)
        if old['identity'] != proof:
            raise ValueError('Existing component sidecar has changed identity')
        for p, h in old['files'].items():
            verify(p, h)
        return C.read(old['summary_path']), dict(path=str(final), sha256=C.sha(final)), hashes
    a = C.arrays(paths['vectors']); summary = C.read(paths['summary']); length_scale = float(summary['L'])
    if length_scale != C.L:
        raise ValueError('Reference length changed')
    with C.resolve(paths['samples']).open(encoding='utf-8-sig', newline='') as handle:
        samples = list(csv.DictReader(handle))
    points = a['points']; normals = a['source_normals']; weights = a['length_weights']; kinds = a['sample_kind']
    if len(samples) != len(points) or any(int(row['sample_id']) != i for i, row in enumerate(samples)):
        raise ValueError('Original sample order differs')
    edge = (kinds == 0) & (weights > 0)
    if np.any((weights > 0) != edge) or np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError('Unexpected boundary weights/domain')
    normal_good = np.asarray([row['source_normal_good'].lower() == 'true' for row in samples])
    unit_good = np.isfinite(normals).all(1) & (np.abs(np.linalg.norm(normals, axis=1)-1) <= 1e-10)
    query_good = np.isfinite(points).all(1)
    total_length = float(weights[edge].sum())
    if not np.isclose(total_length, summary['source_boundary_length'], rtol=1e-13, atol=1e-13):
        raise ValueError('Original length denominator differs')
    domain_sha = digest({k: C.array_sha(a[k]) for k in ('points', 'source_normals', 'sample_kind', 'length_weights')})
    audit = []; output = []; vectors = dict(sample_id=np.arange(len(points)), length_weights=weights,
        sample_kind=kinds, source_normal_good=normal_good, source_normal_unit_good=unit_good)
    test_ids = np.flatnonzero(edge)
    test_ids = test_ids[np.unique(np.linspace(0, len(test_ids)-1, min(3, len(test_ids))).astype(int))] if len(test_ids) else []
    for target in TARGETS:
        closest = a[target+'_point']; candidate = (a[target+'_id'] >= 0) & np.isfinite(closest).all(1) & np.isfinite(a[target+'_distance'])
        with np.errstate(invalid='ignore'):
            delta = closest-points
            signed = np.einsum('ij,ij->i', delta, normals)
            normal_abs = np.abs(signed)
            tangent = np.linalg.norm(delta-signed[:, None]*normals, axis=1)
        invalid = ((~normal_good).astype(np.uint8) | ((~candidate).astype(np.uint8)*2) |
                   ((~query_good).astype(np.uint8)*4) | ((~unit_good).astype(np.uint8)*8))
        valid = edge & (invalid == 0)
        valid_length = float(weights[valid].sum())
        for sample_id in test_ids:
            row = samples[int(sample_id)]
            observed_signed = _number(row[target+'_signed_source_normal_offset_over_L'])
            observed_distance = _number(row[target+'_distance_over_L'])
            computed_signed = float(signed[sample_id]/length_scale) if np.isfinite(signed[sample_id]) else None
            computed_distance = float(np.linalg.norm(delta[sample_id])/length_scale) if candidate[sample_id] else None
            for old, new in ((observed_signed, computed_signed), (observed_distance, computed_distance)):
                if (old is None) != (new is None) or (old is not None and not np.isclose(old, new, rtol=2e-12, atol=2e-12)):
                    raise ValueError('Saved point-field consistency failed')
            audit.append(dict(target=target, sample_id=int(sample_id), signed_original_over_L=observed_signed,
                signed_from_saved_points_over_L=computed_signed, distance_original_over_L=observed_distance,
                distance_from_saved_points_over_L=computed_distance, status='PASS'))
        status = 'NA_ZERO_BOUNDARY_LENGTH' if total_length <= 0 else 'NA_NO_VALID_LENGTH' if valid_length <= 0 else 'PARTIAL_NA' if not np.array_equal(valid, edge) else 'COMPLETE'
        result = dict(target=target, status=status, boundary_query_domain_sha256=domain_sha,
            valid_query_mask_sha256=C.array_sha(valid), L=length_scale, total_midpoint_count=int(edge.sum()),
            zero_weight_vertex_count=int((kinds == 1).sum()), valid_midpoint_count=int(valid.sum()),
            invalid_midpoint_count=int((edge & ~valid).sum()), source_normal_invalid_count=int((edge & ~(normal_good & unit_good)).sum()),
            missing_candidate_count=int((edge & ~candidate).sum()), query_nonfinite_count=int((edge & ~query_good).sum()),
            total_boundary_length=total_length, valid_boundary_length=valid_length,
            invalid_boundary_length=total_length-valid_length,
            valid_boundary_length_fraction=valid_length/total_length if total_length > 0 else None)
        for name, values in (('normal_abs', normal_abs), ('tangent_plane', tangent)):
            mean = float(np.dot(values[valid], weights[valid])/valid_length) if valid_length > 0 else None
            p95 = quantile(values[valid], weights[valid]) if valid_length > 0 else None
            result.update({name+'_mean': mean, name+'_p95': p95,
                name+'_mean_over_L': None if mean is None else mean/length_scale,
                name+'_p95_over_L': None if p95 is None else p95/length_scale})
            vectors[target+'_'+name+'_over_L'] = np.where(valid, values/length_scale, np.nan)
        vectors[target+'_valid'] = valid; vectors[target+'_invalid_code'] = invalid
        output.append(result)
    value = dict(status='COMPLETE_DERIVATION', label=label, request_id=q['request_id'], stage=stage, targets=output,
        original_boundary_receipt_path=source_receipt_path, samples_path=paths['samples'], vectors_path=paths['vectors'],
        csv_consistency_checks=len(audit), new_matching_calls=0, new_quality_scans=0, new_queries=0, model_calls=0)
    C.write(folder/'summary.json', value); C.csvwrite(folder/'point_field_consistency.csv', audit)
    C.frozen_npz(folder/'components.npz', vectors)
    files = {str(folder/name): C.sha(folder/name) for name in ('summary.json', 'point_field_consistency.csv', 'components.npz')}
    C.write(final, dict(status='COMPLETE', identity=proof, files=files, summary_path=str(folder/'summary.json')))
    return value, dict(path=str(final), sha256=C.sha(final)), hashes


def aggregate(rows, keys, metrics, expected, prefix='macro_mean__'):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for group, values in groups.items():
        row = dict(zip(keys, group)); row.update(expected_records=expected, present_records=len(values),
            status_counts=dict((status, sum(x.get('status', 'AGGREGATED') == status for x in values)) for status in sorted({x.get('status', 'AGGREGATED') for x in values})))
        for metric in metrics:
            valid = [_number(x.get(metric)) for x in values]; valid = [x for x in valid if x is not None]
            row[prefix+metric] = float(np.mean(valid)) if valid else None
            row['valid_count__'+metric] = len(valid)
        counts = [row['valid_count__'+metric] for metric in metrics]
        row['status'] = ('COMPLETE_GROUP' if all(n == len(values) for n in counts) else 'NA_NO_VALID_METRIC' if not any(counts) else 'PARTIAL_GROUP')
        result.append(row)
    return result


def run(include_references=False):
    if 'torch' in sys.modules:
        raise RuntimeError('Saved point derivation requires a model-free process')
    tick = time.perf_counter(); reg = C.read(C.OUT/'registration.json'); rh = C.sha(C.OUT/'registration.json')
    requests = reg['requests']; labels = MAIN+REFERENCES if include_references else MAIN
    if len(requests) != 24 or len({q['parent_id'] for q in requests}) != 6:
        raise ValueError('Expected frozen six-parent/24-request panel')
    sources = {str(C.OUT/'registration.json'): rh}; refs = []; long = []; wide = []; available = 0
    for label in labels:
        aggregate_path = C.OUT/'evaluation/boundary_completion'/f'{label}.json'; records = {}
        if aggregate_path.exists():
            block = C.read(aggregate_path); sources[str(aggregate_path)] = C.sha(aggregate_path)
            if block['status'] != 'COMPLETE' or block['label'] != label:
                raise ValueError('Invalid boundary aggregate')
            records = {(x['request_id'], x['stage']): (x, str(aggregate_path)) for x in block['receipts']}
            if len(records) != 48:
                raise ValueError('Incomplete committed boundary aggregate')
        for q in requests:
            for stage in STAGES:
                key = (q['request_id'], stage); individual = C.OUT/'evaluation/boundary'/label/q['request_id']/stage/'completion.json'
                if key not in records and individual.exists():
                    records[key] = (C.read(individual), str(individual)); sources[str(individual)] = C.sha(individual)
                meta = {k: q[k] for k in META}; meta.update(label=label, stage=stage,
                    checkpoint_role='LEARNING_CURVE' if label in MAIN else 'FROZEN_CONTEXT_REFERENCE')
                if key in records:
                    ref, path = records[key]; value, receipt, hashes = _cell(q, label, stage, ref, path, rh)
                    available += 1; refs.append(receipt); sources.update(hashes); targets = value['targets']
                else:
                    targets = [dict(target=target, status='NA_PENDING_BOUNDARY_COMPLETION') for target in TARGETS]
                row = dict(meta, status='AVAILABLE' if key in records else 'NA_PENDING_BOUNDARY_COMPLETION')
                for target in targets:
                    long.append(dict(meta, **target))
                    row.update({target['target']+'__'+k: v for k, v in target.items() if k != 'target'})
                wide.append(row)
    baseline = {(x['request_id'], x['stage'], x['target']): x for x in long if x['label'] == MAIN[0]}
    paired = []
    for row in long:
        if row['label'] not in MAIN[1:]:
            continue
        old = baseline[row['request_id'], row['stage'], row['target']]
        result = {k: row[k] for k in META+('label', 'stage', 'target')}
        if row['status'].startswith('NA_') or old['status'].startswith('NA_'):
            status = 'NA_ONE_OR_BOTH_ENDPOINTS_UNAVAILABLE'
        elif row['boundary_query_domain_sha256'] != old['boundary_query_domain_sha256'] or row['valid_query_mask_sha256'] != old['valid_query_mask_sha256']:
            status = 'NA_DIFFERENT_VALID_QUERY_DOMAIN'
        else:
            status = 'PAIRED_VALID_SUBSET' if 'PARTIAL_NA' in (row['status'], old['status']) else 'PAIRED_COMPLETE'
        result.update(status=status, current_status=row['status'], baseline_status=old['status'],
                      baseline_label=MAIN[0], delta_direction='CURRENT_MINUS_STEP0')
        for metric in METRICS:
            a, b = _number(old.get(metric)), _number(row.get(metric))
            result['step0__'+metric] = a; result['current__'+metric] = b
            result['delta__'+metric] = b-a if status.startswith('PAIRED_') and a is not None and b is not None else None
        paired.append(result)
    parents = aggregate(long, ('parent_id', 'split', 'label', 'stage', 'target'), METRICS, 4)
    delta_metrics = tuple('delta__'+x for x in METRICS)
    parent_pairs = aggregate(paired, ('parent_id', 'split', 'label', 'stage', 'target'), delta_metrics, 4)
    split_rows = [dict(x) for x in parents]+[dict(x, split='ALL') for x in parents]
    split_pairs = [dict(x) for x in parent_pairs]+[dict(x, split='ALL') for x in parent_pairs]
    summaries = aggregate(split_rows, ('split', 'label', 'stage', 'target'), tuple('macro_mean__'+x for x in METRICS),
                          'ALL=6; TRAIN/DEV=registered parents', prefix='equal_parent_mean__')
    pair_summaries = aggregate(split_pairs, ('split', 'label', 'stage', 'target'), tuple('macro_mean__'+x for x in delta_metrics),
                               'ALL=6; TRAIN/DEV=registered parents', prefix='equal_parent_mean__')
    source_hashes = {str(Path(__file__)): C.sha(__file__), str(Path(C.__file__)): C.sha(C.__file__),
                    str(C.ROOT/'r2/tools/full_boundary_attribution_v1/geometry.py'): C.sha(C.ROOT/'r2/tools/full_boundary_attribution_v1/geometry.py')}
    snapshot = digest(dict(input_hashes=sources, source_hashes=source_hashes, include_references=include_references))[:16]
    folder = OUT/'snapshots'/snapshot
    tables = dict(per_output=wide, per_output_target=long, paired_to_step0=paired,
                  per_parent_four_noises=parents, per_parent_paired_to_step0=parent_pairs,
                  by_split_equal_parent=summaries, by_split_paired_to_step0=pair_summaries)
    for name, rows in tables.items():
        C.csvwrite(folder/(name+'.csv'), rows)
    C.write(folder/'definition.json', DEFINITION)
    result = dict(status='COMPLETE' if available == len(wide) else 'PARTIAL', version=VERSION,
        labels=labels, expected_cells=len(wide), available_cells=available, pending_cells=len(wide)-available,
        input_hashes=sources, source_hashes=source_hashes, cell_receipts=refs,
        files={str(folder/(name+'.csv')): C.sha(folder/(name+'.csv')) for name in tables},
        definition_path=str(folder/'definition.json'), definition_sha256=C.sha(folder/'definition.json'),
        snapshot=str(folder), new_matching_calls=0, new_quality_scans=0, new_queries=0, model_calls=0, new_geometry=0)
    C.write(folder/'manifest.json', result)
    C.event('SAVED_BOUNDARY_COMPONENTS_SNAPSHOT', snapshot=str(folder), status=result['status'], available_cells=available,
            expected_cells=len(wide), seconds=time.perf_counter()-tick, model_calls=0, new_matching_calls=0)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--include-references', action='store_true')
    args = parser.parse_args(); result = run(args.include_references)
    print(json.dumps({k: result[k] for k in ('status', 'expected_cells', 'available_cells', 'pending_cells', 'snapshot')}), flush=True)


if __name__ == '__main__':
    main()
