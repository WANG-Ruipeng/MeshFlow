"""Paired reporting of the frozen finite CPU split test; never rescores meshes."""
from __future__ import annotations
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_hinge_subsegment_v1'


def read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    if str(value).lower() in ('true', 'false'):
        return float(str(value).lower() == 'true')
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def run():
    rows = read(OUT / 'scores/per_output.csv')
    stages = list(dict.fromkeys(r['stage'] for r in rows))
    assert len(stages) == 3, stages
    # The scoring adapter places the original HINGE baseline first.
    baseline = next(s for s in stages if 'SPLIT' not in s.upper())
    stages = [baseline] + [s for s in stages if s != baseline]
    lookup = {(r['stage'], r['request_id']): r for r in rows}
    ids = sorted({r['request_id'] for r in rows})
    assert len(ids) == 28 and len(rows) == len(lookup) == 84
    policy = json.loads((ROOT / 'r2/tools/hinge_opposite_v1/metric_policy.json').read_text(encoding='utf-8-sig'))
    directions = dict(policy['directions'])
    fields = list(directions)
    assert len(fields) == 135
    for key in rows[0]:
        if key.startswith('fixed_parent_') and any(number(r.get(key)) is not None for r in rows):
            if key not in directions:
                fields.append(key)
                directions[key] = 'lower_descriptive' if any(x in key for x in ('illegal', 'minimum_angle_lt5', 'q_gt10', 'near_degenerate', 'duplicate')) else 'context_only'
    paired = []
    for rid in ids:
        old = lookup[baseline, rid]
        for stage in stages[1:]:
            row = lookup[stage, rid]
            for k in ('parent_id', 'condition_id', 'noise_index', 'N', 'K', 'split'):
                assert row[k] == old[k], (rid, stage, k)
            for key in fields:
                a, b = number(old.get(key)), number(row.get(key))
                delta = b-a if a is not None and b is not None else None
                direction = directions[key]
                change = ('NA' if delta is None else 'unchanged' if abs(delta) <= 1e-12
                          else ('increased' if delta > 0 else 'decreased') if direction == 'context_only'
                          else 'improved' if (delta < 0) == (direction == 'lower_descriptive') else 'worsened')
                paired.append(dict(request_id=rid, parent_id=row['parent_id'], split=row['split'],
                                   noise_index=row['noise_index'], reference=baseline, candidate=stage,
                                   metric=key, baseline_value=a, candidate_value=b, delta=delta,
                                   direction=direction, change=change, score_origin=row.get('score_origin', '')))
    write(OUT / 'paired_per_output.csv', paired)
    write(OUT / 'all_regressions.csv', [r for r in paired if r['change'] == 'worsened'])
    summaries, parents = [], []
    for group in ['ALL', 'TRAIN', 'DEV'] + sorted({r['parent_id'] for r in rows}):
        selected = [rid for rid in ids if group == 'ALL' or lookup[baseline, rid]['split'] == group or lookup[baseline, rid]['parent_id'] == group]
        for key in fields:
            result = dict(group=group, metric=key, n_registered=len(selected), direction=directions[key])
            for stage in stages:
                values = [number(lookup[stage, rid].get(key)) for rid in selected]
                values = [v for v in values if v is not None]
                result[stage+'_n'] = len(values)
                for label, func in [('mean', statistics.mean), ('median', statistics.median), ('sum', sum), ('min', min), ('max', max)]:
                    result[stage+'_'+label] = func(values) if values else None
            (summaries if group in ('ALL', 'TRAIN', 'DEV') else parents).append(result)
    write(OUT / 'metric_summary.csv', summaries)
    write(OUT / 'paired_per_parent.csv', parents)
    result = dict(status='COMPLETE', stages=stages, outputs=len(rows), original_fields=135,
                  total_fields=len(fields), paired_values=len(paired),
                  missing_values=sum(number(r.get(k)) is None for r in rows for k in fields),
                  aggregation='Macro over registered requests; ALL is 16 TRAIN and 12 DEV; four noises per parent.',
                  original_parent_is_not_a_reconstruction_target=True, GPU_calls=0)
    (OUT / 'analysis_complete.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    run()
