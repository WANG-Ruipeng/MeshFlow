"""Fixed-denominator parent-level aggregation; no model or generation calls."""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import itertools
import numpy as np
from .scoring import (TAUS, ATTACH_TAUS, arrays, clean, read_json, resolve, sha,
                      surface_distance_squared, tri, write_csv, write_json, tkey)

ARMS = ('OT_BASE', 'OT_HYBRID')
SECONDARY = ['GT_to_F_MSE2048_L0', 'F_to_GT_MSE2048_L0', 'fixed_query_MSE',
    'coverage_fraction', 'fixed_query_p95_L', 'remainder_symmetric_RMS_bbox_pct',
    'free_to_R12_RMS_bbox_pct', 'R12_to_free_RMS_bbox_pct',
    'free_minimum_angle_lt5_fraction', 'free_angle_p05_degrees',
    'GT_free_minimum_angle_lt5_fraction', 'GT_free_angle_p05_degrees',
    'free_near_degenerate_fraction', 'free_exact_zero_area_fraction',
    'CF_proper_pair_count', 'CF_coplanar_pair_count', 'CF_proper_affected_free_fraction',
    'FF_proper_affected_free_fraction', 'FF_illegal_affected_free_fraction', 'FF_proper_pair_count',
    'FF_coplanar_pair_count', 'FF_illegal_pair_count', 'generation_seconds']
SECONDARY += [f'{name}2048_{tkey(t)}' for t in TAUS for name in ('Precision', 'Recall', 'Fscore')]
SECONDARY += [f'Gamma32_{kind}_{metric}' for kind in ('edge', 'surface') for metric in ('RMS_L0', 'p95_L0')]
SECONDARY += [f'Gamma32_{kind}_AttachCov_{tkey(t)}' for kind in ('edge', 'surface') for t in ATTACH_TAUS]
COMMON = ['common_RMS_L0', 'common_p95_L0'] + ['common_recall_'+tkey(t) for t in TAUS]


def number(value):
    if value is None or value in ('NA', 'NOT_RUN'): return None
    if value == 'INF': return float('inf')
    if value == '-INF': return -float('inf')
    try: return float(value)
    except (ValueError, TypeError): return None


def mean_fixed(values, expected):
    nums = [number(v) for v in values]
    return float(np.mean(nums)) if len(nums) == expected and all(v is not None for v in nums) else None


def diversity(out, rows):
    values = [r for r in rows if r.get('status') == 'PASS']
    pairrows = []
    if len(values) < 2 or any(not r.get('vectors_path') for r in values): return None, pairrows
    cachepath = Path(out)/'scoring'/'diversity'/(values[0]['condition_id']+'_'+values[0]['arm']+'.json')
    identity = {str(r['candidate']): r.get('raw_sha256') for r in values}
    if cachepath.exists():
        cached = read_json(cachepath)
        if cached['identity'] != identity: raise ValueError('Diversity frozen candidate identity differs')
        return cached['normalized_free_pair_distance'], cached['pairs']
    loaded = {}
    for r in values:
        if not r.get('vectors_path'): continue
        if sha(resolve(r['vectors_path'])) != r['vectors_sha256']: raise ValueError('Frozen Q_F changed')
        raw = tri(arrays(resolve(r['raw_path']))['raw'])[int(r['K']):]
        loaded[r['candidate']] = (raw, arrays(resolve(r['vectors_path']))['Q_F2048'])
    L0 = float(values[0]['L0'])
    for a, b in itertools.combinations(sorted(loaded), 2):
        fa, qa = loaded[a]; fb, qb = loaded[b]
        da = surface_distance_squared(qa, fb); db = surface_distance_squared(qb, fa)
        pairrows.append(dict(uid=values[0]['uid'], condition_id=values[0]['condition_id'], arm=values[0]['arm'],
            candidate_a=a, candidate_b=b, a_to_b_MSE_L0=float(da.mean()/L0**2), b_to_a_MSE_L0=float(db.mean()/L0**2),
            normalized_pair_distance=float((da.mean()+db.mean())/(2*L0**2))))
    full = float(np.mean([r['normalized_pair_distance'] for r in pairrows])) if len(rows) == 4 and len(pairrows) == 6 else None
    # A partial cache is not frozen: later candidates may complete.
    if len(rows) == 4 and all(r.get('status') in ('PASS', 'GENERATION_FAILED', 'INVALID_OUTPUT') for r in rows):
        write_json(cachepath, dict(identity=identity, normalized_free_pair_distance=full, pairs=pairrows))
    return full, pairrows


def condition_summary(out, rows, expected):
    rows = sorted(rows, key=lambda r: int(r['candidate']))
    first = rows[0]; by = {int(r['candidate']): r for r in rows}
    complete = all(i in by and by[i].get('status') in ('PASS', 'GENERATION_FAILED', 'INVALID_OUTPUT') for i in range(expected))
    row = {k: first.get(k) for k in ('uid', 'condition_id', 'context_level', 'arm', 'panel')}
    row.update(expected_candidates=expected, present_rows=len(rows), completed_candidates=sum(r.get('status') in ('PASS', 'GENERATION_FAILED', 'INVALID_OUTPUT') for r in rows),
        valid_candidates=sum(r.get('status') == 'PASS' for r in rows), failed_candidates=sum(r.get('status') in ('GENERATION_FAILED', 'INVALID_OUTPUT') for r in rows),
        four_candidate_status='COMPLETE' if complete and expected == 4 else 'NOT_APPLICABLE' if expected != 4 else 'INCOMPLETE',
        D_sym2048_mean=mean_fixed([r.get('D_sym2048') for r in rows], expected) if complete else None,
        D_sym2048_valid_only_mean=mean_fixed([r.get('D_sym2048') for r in rows if r.get('status') == 'PASS'], sum(r.get('status') == 'PASS' for r in rows)) if any(r.get('status') == 'PASS' for r in rows) else None,
        candidate0_D_sym2048=by.get(0, {}).get('D_sym2048'), human_U_single=None, human_U_at4=None, user_study='NOT_RUN')
    for metric in SECONDARY:
        row['mean_'+metric] = mean_fixed([r.get(metric) for r in rows], expected) if complete else None
    row['four_candidate_generation_seconds'] = (row['mean_generation_seconds']*4 if expected == 4 and row['mean_generation_seconds'] is not None else None)
    for budget in (2, 4):
        eligible = budget <= expected and all(i in by and number(by[i].get('D_sym2048')) is not None for i in range(budget))
        name = f'GT_oracle_Best_at{budget}'
        winner = min(range(budget), key=lambda i: (number(by[i]['D_sym2048']), i)) if eligible else None
        row[name+'_candidate'] = winner
        row[name+'_D_sym2048'] = by[winner]['D_sym2048'] if winner is not None else None
        for metric in SECONDARY:
            row[name+'_'+metric] = by[winner].get(metric) if winner is not None else None
    if expected == 4:
        value, pairs = diversity(out, rows)
        row.update(normalized_free_pair_distance=value, valid_diversity_pairs=len(pairs), expected_diversity_pairs=6)
    else:
        pairs = []; row.update(normalized_free_pair_distance=None, valid_diversity_pairs=None, expected_diversity_pairs=None)
    return row, pairs


def higher(metric):
    return any(key in metric for key in ('Fscore', 'Precision', 'Recall', 'recall', 'AttachCov', 'coverage_fraction', 'angle_p05'))


def paired_summary(conditions):
    grouped = defaultdict(dict)
    for r in conditions:
        if int(r['context_level']) == 1: grouped[r['uid']][r['arm']] = r
    metrics = ['D_sym2048_mean', 'candidate0_D_sym2048', 'GT_oracle_Best_at2_D_sym2048', 'GT_oracle_Best_at4_D_sym2048',
               'normalized_free_pair_distance', 'four_candidate_generation_seconds'] + ['mean_'+m for m in SECONDARY]
    rows, summary = [], []
    for metric in metrics:
        collected = []
        for uid, arms in sorted(grouped.items()):
            a, b = number(arms.get(ARMS[0], {}).get(metric)), number(arms.get(ARMS[1], {}).get(metric))
            delta = b-a if a is not None and b is not None and not (np.isinf(a) and np.isinf(b)) else None
            row = dict(uid=uid, metric=metric, OT_BASE=a, OT_HYBRID=b, HYBRID_minus_BASE=delta,
                       improved=(delta > 0 if higher(metric) else delta < 0) if delta is not None else None,
                       direction='descriptive' if metric == 'normalized_free_pair_distance' else 'higher' if higher(metric) else 'lower')
            rows.append(row); collected.append(row)
        comparable = [r for r in collected if r['HYBRID_minus_BASE'] is not None]
        worst = (min(comparable, key=lambda r: r['HYBRID_minus_BASE']) if higher(metric) else max(comparable, key=lambda r: r['HYBRID_minus_BASE'])) if comparable else None
        deltas = [r['HYBRID_minus_BASE'] for r in comparable]
        summary.append(dict(metric=metric, expected_parents=8, available_parents=len(comparable),
            OT_BASE_parent_equal_mean=mean_fixed([r['OT_BASE'] for r in collected], 8),
            OT_HYBRID_parent_equal_mean=mean_fixed([r['OT_HYBRID'] for r in collected], 8),
            paired_mean_delta=mean_fixed(deltas, 8), paired_median_delta=float(np.median(deltas)) if len(deltas) == 8 else None,
            improved_parents=sum(r['improved'] for r in comparable) if metric != 'normalized_free_pair_distance' else None,
            worst_parent=worst['uid'] if worst else None, worst_delta=worst['HYBRID_minus_BASE'] if worst else None,
            direction=collected[0]['direction'] if collected else None))
    return rows, summary


def multi_summary(requests, outputs):
    chosen = sorted({r['uid'] for r in requests if int(r['context_level']) > 1})
    lookup = {(r['uid'], r['arm'], int(r['candidate']), int(r['context_level'])): r for r in outputs}
    rows = []
    for uid in chosen:
        for arm in ARMS:
            for level in (1, 2, 3):
                out = dict(uid=uid, arm=arm, context_level=level, expected_noises=2, statistic='two-noise mean within parent',
                           interpretation='Only common unseen GT region to predicted free is comparable across levels')
                rr = [lookup.get((uid, arm, noise, level), {}) for noise in (0, 1)]
                out['completed_noises'] = sum(r.get('status') == 'PASS' for r in rr)
                for metric in COMMON + ['D_sym2048', 'known_face_ratio', 'known_area_ratio', 'known_span_L0', 'free_faces', 'common_GT_area', 'known_components']:
                    out[metric] = mean_fixed([r.get(metric) for r in rr], 2)
                for metric in COMMON:
                    baseline = mean_fixed([lookup.get((uid, arm, noise, 1), {}).get(metric) for noise in (0, 1)], 2)
                    value = number(out[metric]); out[metric+'_minus_C1'] = value-baseline if value is not None and baseline is not None else None
                rows.append(out)
    grand = []
    for arm in ARMS:
        for level in (1, 2, 3):
            selected = [r for r in rows if r['arm'] == arm and r['context_level'] == level]
            item = dict(arm=arm, context_level=level, expected_parents=2, available_parents=sum(r['completed_noises'] == 2 for r in selected))
            for metric in COMMON + [m+'_minus_C1' for m in COMMON]: item[metric] = mean_fixed([r.get(metric) for r in selected], 2)
            grand.append(item)
    return rows, grand


def finalize(out):
    out = Path(out); path = out/'requests.json'
    if not path.exists(): path = out/'registration'/'requests.json'
    requests = read_json(path); requests = requests.get('rows', requests) if isinstance(requests, dict) else requests
    loaded = read_json(out/'scoring'/'per_output.json')
    lookup = {r['request_id']: r for r in loaded}; rows = []
    for request in requests:
        row = lookup.get(request['request_id'])
        if row is None:
            row = {k: request.get(k) for k in ('request_id', 'uid', 'condition_id', 'context_level', 'candidate', 'arm', 'panel', 'panels')}
            row.update(status='BLOCKED', reason='Output not available in this CPU stage')
        rows.append(row)
    # Condition topology is descriptive task metadata, never inferred from F.
    meta_path = out/'registration'/'conditions.json'
    if meta_path.exists():
        data = read_json(meta_path); data = data.get('conditions', data.get('rows', data)) if isinstance(data, dict) else data
        if isinstance(data, list):
            meta = {r['condition_id']: r for r in data}
            for r in rows:
                cm = meta.get(r['condition_id'], {})
                for key in ('known_components', 'known_component_count', 'connected_components'):
                    if key in cm: r['known_components'] = cm[key]; break
    groups = defaultdict(list)
    for row in rows: groups[(row['condition_id'], row['arm'])].append(row)
    conditions, pairs = [], []
    for key, group in sorted(groups.items()):
        value, detail = condition_summary(out, group, 4 if int(group[0]['context_level']) == 1 else 2)
        conditions.append(value); pairs.extend(detail)
    paired, primary = paired_summary(conditions)
    multi, multi_global = multi_summary(requests, rows)
    write_csv(out/'per_output_metrics.csv', rows)
    write_csv(out/'per_condition_summary.csv', conditions)
    write_csv(out/'paired_parent_summary.csv', paired)
    write_csv(out/'multi_context_summary.csv', multi)
    write_csv(out/'scoring'/'primary_summary.csv', primary)
    write_csv(out/'scoring'/'diversity_pairs.csv', pairs)
    write_csv(out/'scoring'/'multi_context_global.csv', multi_global)
    sensitivity = []
    for r in rows:
        if 'D_sym8192' not in r: continue
        item = {k: r.get(k) for k in ('request_id', 'uid', 'arm', 'candidate', 'D_sym2048', 'D_sym8192')}
        item['D_sym8192_minus_2048'] = number(r['D_sym8192'])-number(r['D_sym2048'])
        for kind in ('edge', 'surface'):
            for metric in ('RMS_L0', 'p95_L0'):
                for count in (32, 128): item[f'Gamma{count}_{kind}_{metric}'] = r.get(f'Gamma{count}_{kind}_{metric}')
        for tau in TAUS:
            for count in (2048, 8192): item[f'Fscore{count}_{tkey(tau)}'] = r.get(f'Fscore{count}_{tkey(tau)}')
        sensitivity.append(item)
    write_csv(out/'scoring'/'sensitivity_checks.csv', sensitivity)
    result = dict(status='COMPLETE' if all(r.get('status') in ('PASS', 'GENERATION_FAILED', 'INVALID_OUTPUT', 'NOT_APPLICABLE') for r in rows) else 'PARTIAL',
        registered_requests=len(requests), output_status_counts=dict(Counter(r['status'] for r in rows)),
        condition_groups=len(conditions), main_parents=8, primary=primary, multi_context=multi_global,
        sensitivity_outputs=len(sensitivity), expected_sensitivity_outputs=4,
        human_usable=None, U_single=None, U_at4=None, user_study='NOT_RUN',
        model_forwards=0, backwards=0, optimizer_updates=0,
        caveats=['Old development parents, new noise and development metrics are not independent confirmatory evidence',
                 'GT-oracle best is not deployment selection; no scalar usability or human labels inferred',
                 'Diversity is descriptive and not monotonically desirable; missing candidate groups remain NA'])
    write_json(out/'scoring'/'summary.json', result)
    write_json(out/'scoring'/'per_condition.json', conditions)
    return clean(result)
