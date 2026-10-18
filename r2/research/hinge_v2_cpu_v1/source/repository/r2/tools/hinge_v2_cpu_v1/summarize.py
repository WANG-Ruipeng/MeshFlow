"""Complete deterministic tabular analysis; no geometry scan or model import.

Every inherited numeric field remains present. BRIDGE descriptors are contextual;
actual-face quality and fixed original-parent quality are reported side by side.
"""
from __future__ import annotations
import csv
import hashlib
import io
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_hinge_v2_cpu_v1'
STAGES = ('BASELINE', 'MIDPOINT_REPLAY', 'ALPHA', 'HOLE', 'ALPHA_HOLE')
COMPARISONS = tuple(('BASELINE', a) for a in STAGES[1:]) + (('MIDPOINT_REPLAY', 'ALPHA'), ('ALPHA', 'ALPHA_HOLE'))
META = ('request_id', 'condition_id', 'parent_id', 'noise_index', 'N', 'K', 'split', 'label', 'stage', 'domain', 'scoring_definition_version')
TOL = 1e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    p = Path(path)
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    if p.exists():
        assert p.read_bytes() == data, f'REFUSE_CHANGED_RESULT: {p}'
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


def csvwrite(path, rows):
    assert rows
    s = io.StringIO(newline='')
    w = csv.DictWriter(s, fieldnames=list(rows[0]), lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
    p = Path(path)
    data = s.getvalue().encode('utf-8')
    if p.exists():
        assert p.read_bytes() == data, f'REFUSE_CHANGED_RESULT: {p}'
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


def number(value):
    if value is None or value == '' or str(value).upper() in ('NA', 'NAN', 'NONE', 'NULL'):
        return None
    if str(value).lower() in ('true', 'false'):
        return float(str(value).lower() == 'true')
    x = float(value)
    return x if math.isfinite(x) else None


def stats(values):
    xs = [x for x in values if x is not None]
    return dict(n_valid=len(xs), mean=statistics.fmean(xs) if xs else None,
                median=statistics.median(xs) if xs else None,
                minimum=min(xs) if xs else None, maximum=max(xs) if xs else None,
                total=math.fsum(xs) if xs else None)


def direction(key, original):
    if key in original:
        return original[key]
    # A bridge can be an intended completion or an unwanted addition. Its count,
    # area, connectivity and quality descriptors are never independently a win.
    if 'bridge' in key.lower():
        return 'context_only'
    if key.startswith('fixed_parent_') and 'denominator' not in key:
        return 'lower_descriptive'
    if key == 'deleted_original_POST_faces':
        return 'lower_descriptive'
    return 'context_only'


def outcome(delta, d):
    if delta is None:
        return 'NA'
    if abs(delta) <= TOL:
        return 'TIE'
    if d.startswith('lower'):
        return 'WIN' if delta < 0 else 'LOSS'
    if d.startswith('higher'):
        return 'WIN' if delta > 0 else 'LOSS'
    return 'INCREASE' if delta > 0 else 'DECREASE'


def count_outcomes(items):
    c = Counter(items)
    return {k.lower(): c[k] for k in ('WIN', 'TIE', 'LOSS', 'INCREASE', 'DECREASE', 'NA')}


def main():
    done = OUT / 'analysis_complete.json'
    if done.exists():
        r = read(done)
        for p, expected in {**r['inputs'], **r['outputs']}.items():
            assert sha(p) == expected, p
        assert sha(__file__) == r['analysis_code_sha256']
        print(json.dumps({k: r[k] for k in ('status', 'rows', 'metrics', 'paired_output_rows')}, indent=2))
        return r
    source = OUT / 'scores/per_output.csv'
    score_done = OUT / 'scores/completion.json'
    policy = ROOT / 'r2/tools/hinge_opposite_v1/metric_policy.json'
    audits = (OUT / 'independent_audit.json', OUT / 'independent_score_pair_audit.json')
    completion = read(score_done)
    assert completion['status'] == 'COMPLETE' and completion['rows'] == 140
    assert completion['baseline_equivalence_checks'] == 3780
    assert completion['files'][str(source)] == sha(source)
    original = read(policy)['directions']
    assert len(original) == 135
    with source.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames
    assert len(rows) == 140
    metrics = []
    status_fields = []
    for key in fields:
        if key in META:
            continue
        try:
            for row in rows:
                number(row[key])
        except ValueError:
            assert key not in original, key
            status_fields.append(key)
        else:
            metrics.append(key)
    assert set(original) <= set(metrics)
    directions = {m: direction(m, original) for m in metrics}
    bystage = {s: {r['request_id']: r for r in rows if r['stage'] == s} for s in STAGES}
    ids = sorted(bystage['BASELINE'])
    assert len(ids) == 28
    pair_groups = defaultdict(list)
    split_counts = Counter(r['split'] for r in bystage['BASELINE'].values())
    assert split_counts == dict(TRAIN=16, DEV=12)
    for s in STAGES:
        assert sorted(bystage[s]) == ids
        for rid in ids:
            a, b = bystage['BASELINE'][rid], bystage[s][rid]
            assert all(a[k] == b[k] for k in ('condition_id', 'parent_id', 'noise_index', 'N', 'K', 'split'))
            assert number(a['fixed_parent_free_face_denominator']) == number(b['fixed_parent_free_face_denominator'])
            assert number(b['actual_free_faces']) == number(b['original_POST_free_faces']) + number(b['new_bridge_faces']) - number(b['deleted_original_POST_faces'])
            for kind in ('CF', 'FF'):
                frac = number(b[f'fixed_parent_{kind}_illegal_affected_fraction'])
                expected = number(b[f'fixed_parent_{kind}_illegal_affected_count']) / number(b['fixed_parent_free_face_denominator'])
                assert abs(frac - expected) <= TOL
    values = {(r['stage'], r['request_id']): {m: number(r[m]) for m in metrics} for r in rows}
    outputs = []

    def savecsv(name, data):
        p = OUT / name
        csvwrite(p, data)
        outputs.append(p)

    def savejson(name, data):
        p = OUT / name
        write(p, data)
        outputs.append(p)

    savejson('analysis_metric_policy.json', dict(version='hinge_v2_full_numeric_accounting_v1', original_policy=str(policy),
        original_policy_sha256=sha(policy), original_metric_count=135, total_numeric_metrics=len(metrics), directions=directions,
        NA_policy='Blank and explicit NA remain missing; no imputation. n_valid and n_registered are reported.',
        context_policy='All newly introduced BRIDGE metrics are contextual, without a good/bad preference.',
        averaging='Equal-request macro; four fixed noises per parent. Parent paired rows average only matched valid requests.',
        comparison_tolerance_absolute=TOL, primary_bulk='Original BULK role area only; BRIDGE may carry paths but contributes no bulk numerator or denominator.',
        fixed_parent='Original HINGE_OPP015 free-face identity union; bridge-related original parent damage included, bridge damage separately retained.',
        I='Original frozen C queries and weights; no added-face denominator dilution.',
        metadata_omitted_from_metric_loop=META, categorical_fields=status_fields))
    summary = []
    for s in STAGES:
        for split in ('ALL', 'TRAIN', 'DEV'):
            group = [r for r in bystage[s].values() if split == 'ALL' or r['split'] == split]
            for m in metrics:
                summary.append(dict(stage=s, split=split, metric=m, direction=directions[m], inherited_135=m in original,
                    n_registered=len(group), **stats([values[(s, r['request_id'])][m] for r in group])))
    savecsv('metric_summary.csv', summary)
    paired = []
    for base, arm in COMPARISONS:
        comparison = f'{arm}_vs_{base}'
        for rid in ids:
            a, b = bystage[base][rid], bystage[arm][rid]
            for m in metrics:
                x, y = values[(base, rid)][m], values[(arm, rid)][m]
                delta = None if x is None or y is None else y - x
                r = dict(comparison=comparison, reference=base, candidate=arm, request_id=rid,
                    parent_id=a['parent_id'], condition_id=a['condition_id'], noise_index=a['noise_index'], split=a['split'],
                    metric=m, direction=directions[m], reference_value=x, candidate_value=y, delta=delta,
                    change=outcome(delta, directions[m]))
                paired.append(r)
                pair_groups[(comparison, m)].append(r)
    savecsv('paired_per_output.csv', paired)
    parent_rows = []
    pair_summary = []
    for base, arm in COMPARISONS:
        comparison = f'{arm}_vs_{base}'
        for m in metrics:
            g = pair_groups[(comparison, m)]
            for parent in sorted({r['parent_id'] for r in g}):
                pg = [r for r in g if r['parent_id'] == parent]
                valid = [r for r in pg if r['delta'] is not None]
                delta = statistics.fmean(r['delta'] for r in valid) if valid else None
                parent_rows.append(dict(comparison=comparison, reference=base, candidate=arm, parent_id=parent, split=pg[0]['split'],
                    metric=m, direction=directions[m], n_registered=len(pg), n_valid=len(valid),
                    reference_mean=statistics.fmean(r['reference_value'] for r in valid) if valid else None,
                    candidate_mean=statistics.fmean(r['candidate_value'] for r in valid) if valid else None,
                    delta_mean=delta, parent_change=outcome(delta, directions[m]), **count_outcomes(r['change'] for r in pg)))
            for split in ('ALL', 'TRAIN', 'DEV'):
                sg = [r for r in g if split == 'ALL' or r['split'] == split]
                valid = [r for r in sg if r['delta'] is not None]
                pair_summary.append(dict(comparison=comparison, reference=base, candidate=arm, split=split, metric=m,
                    direction=directions[m], n_registered=len(sg),
                    reference_mean=statistics.fmean(r['reference_value'] for r in valid) if valid else None,
                    candidate_mean=statistics.fmean(r['candidate_value'] for r in valid) if valid else None,
                    **{f'delta_{k}': v for k, v in stats(r['delta'] for r in sg).items()}, **count_outcomes(r['change'] for r in sg)))
    savecsv('paired_per_parent.csv', parent_rows)
    savecsv('paired_summary.csv', pair_summary)
    regressions = [r for r in paired if r['change'] == 'LOSS']
    savecsv('all_regressions.csv', regressions)
    identities = dict(definition='Original BULK free-only legal-path reachable area fraction >= 0.30; newly added BRIDGE area excluded.',
        by_stage={}, comparisons=[])
    for s in STAGES:
        identities['by_stage'][s] = {}
        for split in ('ALL', 'TRAIN', 'DEV'):
            req = [rid for rid in ids if split == 'ALL' or bystage[s][rid]['split'] == split]
            good = [rid for rid in req if values[(s, rid)]['bulk_reach_ge_0_30_descriptive'] == 1]
            identities['by_stage'][s][split] = dict(count=len(good), total=len(req), request_ids=good)
            for rid in req:
                assert (values[(s, rid)]['free_only_reached_bulk_area_fraction'] >= .30) == (rid in good)
    for base, arm in COMPARISONS:
        for split in ('ALL', 'TRAIN', 'DEV'):
            a = set(identities['by_stage'][base][split]['request_ids'])
            b = set(identities['by_stage'][arm][split]['request_ids'])
            identities['comparisons'].append(dict(reference=base, candidate=arm, split=split, old_positive_count=len(a),
                new_positive_count=len(b), retained=sorted(a & b), gained=sorted(b - a), lost=sorted(a - b)))
    savejson('bulk_reach_identities.json', identities)
    denominators = []
    for s in STAGES:
        for rid in ids:
            r = bystage[s][rid]
            v = values[(s, rid)]
            for kind in ('CF', 'FF'):
                actual, fixed = v[f'{kind}_illegal_affected_free_fraction'], v[f'fixed_parent_{kind}_illegal_affected_fraction']
                denominators.append(dict(stage=s, request_id=rid, split=r['split'], kind=kind,
                    original_parent_denominator=v['fixed_parent_free_face_denominator'], actual_child_denominator=v['actual_free_faces'],
                    bridge_faces=v['bridge_face_count'], actual_affected_fraction=actual, fixed_parent_affected_fraction=fixed,
                    actual_minus_fixed_fraction=actual-fixed, original_affected_count=v[f'fixed_parent_{kind}_illegal_affected_count'],
                    bridge_affected_count=v[f'bridge_{kind}_illegal_affected_count'],
                    interpretation='Zero bridge conflicts: lower actual rate than fixed rate is denominator dilution' if v[f'bridge_{kind}_illegal_affected_count'] == 0 else 'Bridge conflicts must be considered separately'))
    savecsv('actual_vs_fixed_parent.csv', denominators)
    categorical = []
    for s in STAGES:
        for key in status_fields:
            for status, count in sorted(Counter(r[key] or 'NA' for r in bystage[s].values()).items()):
                categorical.append(dict(stage=s, field=key, status=status, count=count, registered=28))
    savecsv('categorical_status.csv', categorical)
    geom = read(audits[0]); pair_audit = read(audits[1])
    assert geom['status'] == 'PASS_INDEPENDENT_GEOMETRY_AUDIT'
    assert 'PASS' in pair_audit['status']
    savejson('connection_audit_summary.json', dict(source=str(audits[0]), source_sha256=sha(audits[0]),
        status=geom['status'], totals=geom['totals'], invariant_counts={k: geom[k] for k in ('old_faces_deleted',
        'C_and_seam_anchor_changes', 'new_nonmanifold', 'new_same_winding_two_face_edges', 'old_legal_edge_losses',
        'new_illegal_CF_FF', 'new_unknown_pairs')}, pair_score_audit_source=str(audits[1]),
        pair_score_audit_sha256=sha(audits[1]), pair_score_audit_status=pair_audit['status'],
        scope='Existing independent per-face lineage and pair audits reused; no new geometric scan.'))
    core_fields = ('FF_illegal_affected_free_fraction', 'fixed_parent_FF_illegal_affected_fraction',
        'CF_illegal_affected_free_fraction', 'fixed_parent_CF_illegal_affected_fraction', 'ConflictCov_C',
        'free_minimum_angle_lt5_fraction', 'fixed_parent_minimum_angle_lt5_fraction', 'free_q_gt10_fraction',
        'fixed_parent_q_gt10_fraction', 'free_q_mean', 'free_angle_p05_degrees', 'EXACT_COORD_same_direction_two_face_edges',
        'hinge_diag_isolated_1face_edge_components', 'free_only_reached_bulk_area_fraction', 'bulk_reach_ge_0_30_descriptive',
        'C_F_valid_edge_count', 'C_F_valid_edge_total_length', 'graph_valid_edges', 'legal_seam_bulk_edge_count',
        'bridge_face_count', 'bridge_area', 'actual_free_faces', 'deleted_original_POST_faces')
    savecsv('core_metric_summary.csv', [r for r in summary if r['metric'] in core_fields])
    savecsv('core_paired_summary.csv', [r for r in pair_summary if r['metric'] in core_fields])
    sources = (source, score_done, policy, *audits)
    rec = dict(status='PASS_COMPLETE_NUMERIC_ANALYSIS', utc=datetime.now(timezone.utc).isoformat(), rows=len(rows),
        metrics=len(metrics), inherited_metrics=135, comparisons=len(COMPARISONS),
        paired_output_rows=len(paired), paired_parent_rows=len(parent_rows), macro_rows=len(summary),
        pair_summary_rows=len(pair_summary), regression_rows=len(regressions),
        baseline_equivalence_checks=3780, model_calls=0, geometric_pair_scans=0,
        analysis_code_sha256=sha(__file__), inputs={str(p): sha(p) for p in sources}, outputs={str(p): sha(p) for p in outputs})
    assert 'torch' not in sys.modules
    write(done, rec)
    print(json.dumps({k: rec[k] for k in ('status', 'rows', 'metrics', 'paired_output_rows', 'regression_rows')}, indent=2))
    return rec


if __name__ == '__main__':
    main()
