"""Independent, resumable verification of saved subsegment metrics.

Reads CSV, NPZ identity arrays, saved face flags and contact classifications.
Does not import the scoring adapter, invoke detectors, or call a model.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_hinge_subsegment_v1'
BASE = ROOT / 'experiments/r2_hinge_opposite_v1/cells/HINGE_OPP015'
STAGES = ('HINGE_OPP015', 'SPLIT015', 'SPLIT025')
ILLEGAL = frozenset(('proper_transverse_penetration', 'coplanar_positive_area_overlap'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def number(value):
    if isinstance(value, bool) or str(value).lower() in ('true', 'false'):
        return float(str(value).lower() == 'true')
    result = float(value)
    assert math.isfinite(result), value
    return result


def changed(delta, direction):
    if abs(delta) <= 1e-12:
        return 'unchanged'
    if direction == 'context_only':
        return 'increased' if delta > 0 else 'decreased'
    return 'improved' if (delta < 0) == (direction == 'lower_descriptive') else 'worsened'


def cached_parent_unions(folder, parents, K, original_count):
    """Reduce saved flags/contact identities, without calculating geometry."""
    pairs = read_json(folder / 'pairs.json')
    shape = read_json(folder / 'shape_per_face.json')
    denominator = original_count - K
    stats = {'fixed_parent_free_face_denominator': denominator}
    for kind in ('CF', 'FF'):
        affected, relations = set(), set()
        for record in pairs[kind]['contacts']:
            if record['classification'] not in ILLEGAL:
                continue
            i, j = map(int, record['pair'])
            j += K
            if kind == 'FF':
                i += K
                affected.update((int(parents[i]), int(parents[j])))
                relations.add(tuple(sorted((int(parents[i]), int(parents[j])))))
            else:
                affected.add(int(parents[j]))
                relations.add((int(parents[i]), int(parents[j])))
        stats[f'fixed_parent_{kind}_illegal_affected_count'] = len(affected)
        stats[f'fixed_parent_{kind}_illegal_affected_fraction'] = len(affected) / denominator
        stats[f'fixed_parent_{kind}_illegal_unique_pair_count'] = len(relations)
    for flag, label in (('minimum_angle_lt5', 'minimum_angle_lt5'),
                        ('q_actual_gt10', 'q_gt10'),
                        ('area_near_degenerate', 'near_degenerate'),
                        ('area_exact_zero', 'exact_zero_area')):
        bad = {int(parents[K + int(r['face'])]) for r in shape['free']['per_face'] if r[flag]}
        stats[f'fixed_parent_{label}_count'] = len(bad)
        stats[f'fixed_parent_{label}_fraction'] = len(bad) / denominator
    assert len(stats) == 15
    return stats


def run():
    start = time.perf_counter()
    paths = {name: OUT / relative for name, relative in (
        ('outputs', 'scores/per_output.csv'), ('paired', 'paired_per_output.csv'),
        ('summary', 'metric_summary.csv'), ('parents', 'paired_per_parent.csv'),
        ('regressions', 'all_regressions.csv'))}
    tables = {name: csv_rows(path) for name, path in paths.items()}
    policy_path = ROOT / 'r2/tools/hinge_opposite_v1/metric_policy.json'
    policy = read_json(policy_path)['directions']
    original_fields = list(policy)
    assert len(original_fields) == 135
    outputs = tables['outputs']
    fixed_fields = [key for key in outputs[0] if key.startswith('fixed_parent_')]
    assert len(fixed_fields) == 15
    fields = original_fields + fixed_fields
    directions = dict(policy)
    for key in fixed_fields:
        directions[key] = ('lower_descriptive' if any(word in key for word in
                           ('illegal', 'minimum_angle_lt5', 'q_gt10', 'near_degenerate', 'duplicate'))
                           else 'context_only')
    by_key = {(r['stage'], r['request_id']): r for r in outputs}
    ids = sorted({r['request_id'] for r in outputs})
    assert len(outputs) == len(by_key) == 84 and len(ids) == 28
    assert {r['stage'] for r in outputs} == set(STAGES)
    values = {(stage, rid, metric): number(by_key[stage, rid][metric])
              for stage in STAGES for rid in ids for metric in fields}
    checks = 0

    def eq(actual, expected, identity):
        nonlocal checks
        a, b = number(actual), number(expected)
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (identity, a, b)
        checks += 1

    source_hashes = {str(path): sha(path) for path in paths.values()}
    source_hashes[str(policy_path)] = sha(policy_path)
    accepted = []
    noop = 0
    dev_exact = 0
    for rid in ids:
        baseline = by_key[STAGES[0], rid]
        old_folder = BASE / rid
        old = read_json(old_folder / 'complete.json')['row']
        for metric in original_fields:
            eq(baseline[metric], old[metric], ('baseline', rid, metric))
        with np.load(old_folder / 'post.npz', allow_pickle=False) as mesh:
            original_count = len(mesh['faces'])
            K = int(mesh['known_faces'])
        reduced = cached_parent_unions(old_folder, np.arange(original_count), K, original_count)
        for metric, expected in reduced.items():
            eq(baseline[metric], expected, ('cached_parent', rid, metric))
        for stage in STAGES[1:]:
            row = by_key[stage, rid]
            for key in ('parent_id', 'condition_id', 'noise_index', 'N', 'K', 'split'):
                assert row[key] == baseline[key], (stage, rid, key)
            receipt = read_json(OUT / 'trials' / stage / rid / 'complete.json')
            eq(row['accepted_splits'], receipt['accepted'], ('accepted', stage, rid))
            if receipt['accepted'] == 0:
                noop += 1
                for metric in fields:
                    assert row[metric] == baseline[metric], ('noop_text', stage, rid, metric)
                    assert values[stage, rid, metric] == values[STAGES[0], rid, metric]
                    checks += 1
            else:
                accepted.append((stage, rid))
                cell = OUT / 'trials' / stage / rid
                score_folder = OUT / 'scores' / stage / rid
                with np.load(cell / 'post.npz', allow_pickle=False) as mesh:
                    parents = mesh['parent_post_face_id'].copy()
                    seam = mesh['child_seam_mask'].copy()
                reduced = cached_parent_unions(score_folder, parents, K, original_count)
                for metric, expected in reduced.items():
                    eq(row[metric], expected, ('cached_parent', stage, rid, metric))
                edges = [e for e in csv_rows(score_folder / 'valid_edges.csv')
                         if e['valid_edge'].lower() == 'true']
                internal = cross = 0
                for edge in edges:
                    i, j = map(int, json.loads(edge['incident_face_ids']))
                    internal += int(parents[i] == parents[j])
                    cross += int(min(i, j) >= K and bool(seam[i]) != bool(seam[j]))
                for metric, expected in (
                    ('subdivision_internal_valid_edge_count', internal),
                    ('external_valid_edge_count', len(edges) - internal),
                    ('graph_valid_edges', len(edges)), ('legal_seam_bulk_edge_count', cross)):
                    eq(row[metric], expected, ('edge_identity', stage, rid, metric))
            if baseline['split'] == 'DEV':
                for metric in original_fields:
                    assert row[metric] == baseline[metric], ('DEV_text', stage, rid, metric)
                    assert values[stage, rid, metric] == values[STAGES[0], rid, metric]
                    dev_exact += 1
    assert noop == 54 and len(accepted) == 2
    assert {rid for _, rid in accepted} == {'85b16941_n0'}
    assert dev_exact == 12 * 2 * 135

    paired = {(r['candidate'], r['request_id'], r['metric']): r for r in tables['paired']}
    assert len(paired) == len(tables['paired']) == 8400
    regressions = {}
    for stage in STAGES[1:]:
        for rid in ids:
            for metric in fields:
                row = paired[stage, rid, metric]
                a, b = values[STAGES[0], rid, metric], values[stage, rid, metric]
                delta = b - a
                for key, expected in (('baseline_value', a), ('candidate_value', b), ('delta', delta)):
                    eq(row[key], expected, ('paired', stage, rid, metric, key))
                assert row['direction'] == directions[metric]
                assert row['change'] == changed(delta, directions[metric])
                assert row['reference'] == STAGES[0]
                for key in ('parent_id', 'split', 'noise_index'):
                    assert row[key] == by_key[stage, rid][key]
                if row['change'] == 'worsened':
                    regressions[stage, rid, metric] = row
    observed_regressions = {(r['candidate'], r['request_id'], r['metric']): r
                            for r in tables['regressions']}
    assert observed_regressions == regressions

    def aggregate(rows, expected_groups):
        keyed = {(r['group'], r['metric']): r for r in rows}
        assert len(keyed) == len(rows) == len(expected_groups) * len(fields)
        for group in expected_groups:
            group_ids = [rid for rid in ids if group == 'ALL' or
                         by_key[STAGES[0], rid]['split'] == group or
                         by_key[STAGES[0], rid]['parent_id'] == group]
            assert len(group_ids) == {'ALL': 28, 'TRAIN': 16, 'DEV': 12}.get(group, 4)
            for metric in fields:
                row = keyed[group, metric]
                eq(row['n_registered'], len(group_ids), ('summary_n', group, metric))
                assert row['direction'] == directions[metric]
                for stage in STAGES:
                    data = [values[stage, rid, metric] for rid in group_ids]
                    eq(row[stage + '_n'], len(data), ('summary_stage_n', group, stage, metric))
                    for stat, operation in (('mean', statistics.mean), ('median', statistics.median),
                                            ('sum', sum), ('min', min), ('max', max)):
                        eq(row[stage + '_' + stat], operation(data), ('summary', group, stage, metric, stat))

    aggregate(tables['summary'], ['ALL', 'TRAIN', 'DEV'])
    aggregate(tables['parents'], sorted({r['parent_id'] for r in outputs}))
    evidence_fields = ('actual_free_faces', 'FF_illegal_affected_free_fraction', 'FF_illegal_pair_count',
        'fixed_parent_FF_illegal_affected_count', 'fixed_parent_FF_illegal_affected_fraction',
        'fixed_parent_free_face_denominator', 'legal_seam_bulk_edge_count',
        'subdivision_internal_valid_edge_count', 'external_valid_edge_count', 'graph_valid_edges',
        'free_only_reached_bulk_faces', 'free_only_reached_bulk_area_fraction',
        'deleted_free_faces', 'deleted_original_POST_faces')
    evidence = {stage: {key: number(by_key[stage, '85b16941_n0'][key]) for key in evidence_fields}
                for stage in STAGES}
    for stage in STAGES[1:]:
        old, new = evidence[STAGES[0]], evidence[stage]
        for key in ('FF_illegal_pair_count', 'fixed_parent_FF_illegal_affected_count',
                    'fixed_parent_FF_illegal_affected_fraction', 'fixed_parent_free_face_denominator',
                    'free_only_reached_bulk_faces', 'deleted_free_faces', 'deleted_original_POST_faces'):
            eq(new[key], old[key], ('85_parent_unchanged', stage, key))
        eq(new['actual_free_faces'], old['actual_free_faces'] + 1, ('85_child_count', stage))
        eq(new['FF_illegal_affected_free_fraction'], 27 / 149, ('85_actual_denominator', stage))
        for key, delta in (('legal_seam_bulk_edge_count', 1),
                           ('subdivision_internal_valid_edge_count', 1),
                           ('external_valid_edge_count', 1), ('graph_valid_edges', 2)):
            eq(new[key], old[key] + delta, ('85_new_edge_kind', stage, key))
    assert evidence['SPLIT015'] == evidence['SPLIT025']
    assert 'torch' not in sys.modules
    result = dict(status='PASS_INDEPENDENT_METRICS_AUDIT', rows=len(outputs), fields=len(fields),
        original_fields=len(original_fields), fixed_parent_fields=len(fixed_fields),
        paired_values=len(paired), summary_rows=len(tables['summary']), parent_rows=len(tables['parents']),
        missing_numeric_values=0, numeric_checks=checks, regressions_verified=len(regressions),
        DEV_original135_text_and_numeric_identical_cells=dev_exact,
        all_54_noop_outputs_all150_exact=True, accepted_unique_request_ids=['85b16941_n0'],
        accepted_cells_same_physical_case=len(accepted), evidence85n0=evidence,
        new_model_calls=0, new_scores=0, seconds=time.perf_counter() - start,
        source_hashes=source_hashes, audit_source_path=str(Path(__file__).resolve()),
        audit_source_sha256=sha(__file__),
        verification_scope='Saved values, cached contact labels and face flags only; no geometry rescoring.')
    (OUT / 'independent_metrics_audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('source_hashes', 'evidence85n0')}, indent=2))
    return result


def main():
    return run()


if __name__ == '__main__':
    main()
