"""Descriptive HALF-versus-both-endpoints analysis, without promotion gates.

All registered parents/noises and original conservative bounds are retained.
OLD/FRESH are already-observed development panels; ALL is supplementary.
"""
from __future__ import annotations
import argparse
import json
import math
import statistics
from pathlib import Path
import numpy as np
from experiments.mf_h1_local0.chair_capped500_gradbudget_v1 import analysis as prior
from . import evaluation as ev

VERSION = 'chair_capped500_half_analysis_v1'
gates = ev.gates
finite, avg, med = prior.finite, prior.avg, prior.med
CONTRASTS = (('HALF_CAPPED500', 'START'), ('HALF_CAPPED500', 'HIST_CAPPED500'), ('HIST_CAPPED500', 'START'))
EXTRA = ('SeamCov_0.01', 'CF_proper_pair_count', 'FF_illegal_pair_count',
    'CF_unresolved_or_degenerate_pair_count', 'FF_unresolved_or_degenerate_pair_count',
    'free_near_degenerate_count', 'free_exact_zero_area_count',
    'predicted_free_angle_p05_degrees', 'predicted_free_angle_p50_degrees', 'predicted_free_angle_p95_degrees',
    'GT_free_minimum_angle_lt5_fraction', 'GT_free_angle_p05_degrees', 'GT_free_angle_p50_degrees', 'GT_free_angle_p95_degrees',
    'Gamma32_edge_RMS_L0', 'Gamma32_edge_p95_L0', 'Gamma32_surface_RMS_L0', 'Gamma32_surface_p95_L0',
    'ConflictCov_C_uncertain_fraction', 'SeamCov_uncertain_weight_fraction')
METRICS = tuple(dict.fromkeys(prior.METRICS + EXTRA + tuple(
    f'{name}_{tau}' for name in ('Precision2048', 'Recall2048', 'Fscore2048')
    for tau in ('0.005', '0.01', '0.02', '0.05')) + tuple(
    f'Gamma32_{kind}_AttachCov_{tau}' for kind in ('edge', 'surface')
    for tau in ('0.001', '0.003', '0.005', '0.01', '0.02', '0.05'))))
HIGHER = prior.HIGHER | {'SeamCov_0.01'} | {k for k in METRICS if 'AttachCov' in k or k.startswith(('Precision2048', 'Recall2048', 'Fscore2048'))}
LOWER = prior.LOWER | {'CF_proper_pair_count', 'FF_illegal_pair_count'} | {k for k in METRICS if k.startswith('Gamma32') and ('RMS' in k or 'p95' in k)}
DEFINITION = dict(version=VERSION, labels=list(ev.LABELS), contrasts=CONTRASTS, metrics=METRICS,
    aggregation='Two registered noises within each of eight parents, then parent-equal mean; ALL uses four noises per parent',
    panels='OLD and FRESH separately; ALL is supplementary, not independent confirmation',
    uncertainty='Original S/I bounds and E global bounds; NA retained. Exact same-condition same-source geometry cancels paired delta uncertainty only.',
    baseline='64 saved cells verified and copied without any metric recomputation',
    half='One fixed full-parameter interpolation alpha=0.5; no training or alpha selection',
    useful_tradeoff='Positive S gain over START must be retained while historical CAPPED shape costs are reduced. Moving toward START alone is insufficient. Report each quality direction without a scalar total score.',
    decision_scope='Descriptive necessary evidence only; no prior T/L/G A/B thresholds, no automatic promotion or default change',
    LOPO='Descriptive sensitivity only; no parent/noise removed from the actual result',
    human_usable=None, user_study='NOT_RUN', default_model_changed=False)


def _rows(out, label, panel):
    if panel == 'ALL':
        return _rows(out, label, 'OLD') + _rows(out, label, 'FRESH')
    path = Path(out)/'evaluation'/label/panel/'per_output.json'
    by = {r['request_id']: r for r in ev.read_json(path)} if path.exists() else {}
    return [by.get(q['request_id'], dict(request_id=q['request_id'], uid=q['uid'],
        condition_id=q['condition_id'], candidate=q['candidate'], label=label,
        panel=panel, split=panel, status='NOT_RUN')) for q in ev.manifest(out, panel)]


def interval(row, key):
    if key in gates.PRIMARY:
        return gates.interval(row, key)
    if key in ('SeamCov_0.005', 'SeamCov_0.01', 'SeamCov_0.02'):
        if row.get('status') != 'PASS':
            return None, None
        return finite(row.get(key)), finite(row.get(key.replace('SeamCov_', 'SeamCov_upper_')))
    value = finite(row.get(key))
    return value, value


def compare(candidate, reference, requests, candidate_label, reference_label, panel):
    function = ev._clone(prior.compare, METRICS=METRICS, HIGHER=HIGHER, LOWER=LOWER, interval=interval)
    result = function(candidate, reference, requests, candidate_label, reference_label, panel)
    groups = {}
    for row in result['per_parent']:
        groups.setdefault(row['metric'], []).append(row)
        noise = list(row['noise_deltas'].values())
        row['noise_delta_min'] = min(noise) if all(x is not None for x in noise) else None
        row['noise_delta_max'] = max(noise) if all(x is not None for x in noise) else None
    for row in result['leave_one_parent_out']:
        remaining = [p for p in groups[row['metric']] if p['uid'] != row['omitted_uid']]
        cm, rm = [avg(p[key] for p in remaining) for key in ('candidate_value', 'reference_value')]
        row.update(candidate_mean=cm, reference_mean=rm,
                   percent_change=100*(cm-rm)/rm if cm is not None and rm not in (None, 0) else None)
    return result


def freeze(out=ev.OUT):
    out = ev.assert_out(out)
    ev.freeze_json(out/'registration/evaluation_definition.json', DEFINITION)
    return dict(status='PASS', source_sha256=ev.digest(__file__), definition=DEFINITION)


def _tail(rows, label, panel, metric):
    values = [(finite(r.get(metric)), r['request_id'], r['uid'], r['candidate']) for r in rows]
    complete = all(v[0] is not None for v in values)
    usable = sorted((v for v in values if v[0] is not None), key=lambda r: (r[0], r[1]))
    result = dict(label=label, panel=panel, metric=metric, registered_outputs=len(rows),
                  observed_values=len(usable), status='COMPLETE' if complete else 'INCOMPLETE')
    if complete:
        numbers = [v[0] for v in usable]
        result.update(minimum=numbers[0], median=statistics.median(numbers), maximum=numbers[-1],
            mean=avg(numbers), minimum_request=usable[0][1], maximum_request=usable[-1][1],
            minimum_parent=usable[0][2], maximum_parent=usable[-1][2])
        if all(math.isfinite(v) for v in numbers):
            result.update(p90=float(np.quantile(numbers, .9)), p95=float(np.quantile(numbers, .95)))
    return result


def run(out=ev.OUT):
    out = ev.assert_out(out)
    freeze(out)
    tables = {k: [] for k in ('per_output', 'per_parent', 'summary', 'per_noise', 'leave_one_parent_out')}
    profiles, tails, absolute_parents, absolute_noise, identities = [], [], [], [], {}
    prerequisites, hard = {}, {}
    for panel in (*ev.PANELS, 'ALL'):
        requests = ev.manifest(out, panel)
        cohorts = {label: _rows(out, label, panel) for label in ev.LABELS}
        for label, rows in cohorts.items():
            summary = gates.summarize(rows, requests)
            profile = dict(label=label, panel=panel, status=summary['status'],
                completed=summary['completed'], expected_outputs=len(requests), **summary['metrics'])
            parent_rows = []
            for uid in sorted({q['uid'] for q in requests}):
                group = [r for r in rows if r['uid'] == uid]
                p = dict(label=label, panel=panel, uid=uid, expected_noises=len(group))
                for metric in METRICS:
                    p[metric] = avg(finite(r.get(metric)) for r in group)
                    bounds = [interval(r, metric) for r in group]
                    p[metric+'_lower'] = avg(v[0] for v in bounds)
                    p[metric+'_upper'] = avg(v[1] for v in bounds)
                parent_rows.append(p)
            absolute_parents.extend(parent_rows)
            for metric in METRICS:
                profile[metric] = avg(p[metric] for p in parent_rows)
                tails.append(_tail(rows, label, panel, metric))
            profiles.append(profile)
            for noise in sorted({q['candidate'] for q in requests}):
                group = [r for r in rows if r['candidate'] == noise]
                absolute_noise.append(dict(label=label, panel=panel, candidate=noise, parents=8,
                    **{key: avg(finite(r.get(key)) for r in group) for key in METRICS}))
            if panel in ev.PANELS:
                path = out/'evaluation'/label/panel/'per_output.json'
                if path.exists():
                    identities[str(path)] = ev.digest(path)
                hard[label+'/'+panel] = gates._hard(rows)
        for candidate, reference in CONTRASTS:
            paired = compare(cohorts[candidate], cohorts[reference], requests, candidate, reference, panel)
            for key in tables:
                tables[key].extend(paired[key])
        pair = gates.paired(cohorts['HALF_CAPPED500'], cohorts['START'], requests)
        s = pair['stats']['SeamCov']
        prerequisites[panel] = dict(
            positive_S_over_START=gates.threshold(s['lower'], s['upper'], 0., strict=True),
            S_parent_distribution=s,
            scope='Necessary descriptive condition only; not sufficient for useful tradeoff, human usability or promotion')
    folder = out/'analysis'
    tables.update(checkpoint_profiles=profiles, absolute_per_parent=absolute_parents,
                  absolute_per_noise=absolute_noise, output_tail_summary=tails)
    paths = {}
    for name, rows in tables.items():
        path = folder/(name+'.csv')
        ev.write_csv(path, rows)
        paths[name] = dict(path=str(path), sha256=ev.digest(path), rows=len(rows))
    complete = all(p['status'] == 'COMPLETE' for p in profiles)
    result = dict(version=VERSION, status='COMPLETE' if complete else 'INCOMPLETE',
        completed_cells=sum(p['completed'] for p in profiles if p['panel'] != 'ALL'), expected_cells=96,
        baseline_cells=64, new_output_cells=32, checkpoint_profiles=profiles,
        positive_attachment_prerequisite=prerequisites, hard_contracts=hard,
        hard_contract_status=gates._and([v['status'] for v in hard.values()]),
        tables=paths, source_files=identities, definition=DEFINITION,
        model_calls=0, metrics_recomputed=0, human_usable=None, user_study='NOT_RUN', default_model_changed=False)
    ev.write_json(folder/'results.json', ev.quality.clean(result))
    return result


def self_test():
    requests = [dict(request_id=f'p{i}n{n}', uid=f'p{i}', candidate=n) for i in range(8) for n in (0, 1)]
    def cohort(gain):
        return [dict(q, status='PASS', SeamCov=.2+gain, SeamCov_upper=.2+gain,
            ConflictCov_C=.03, ConflictCov_C_upper=.03, ExposedRetain=.8,
            ExposedRetain_global_lower=.8, ExposedRetain_global_upper=.8,
            fixed_query_MSE=1.+gain) for q in requests]
    data = compare(cohort(.02), cohort(0), requests, 'HALF_CAPPED500', 'START', 'OLD')
    s = next(r for r in data['summary'] if r['metric'] == 'SeamCov')
    assert abs(s['delta']-.02) < 1e-12 and s['improved_parents'] == 8
    assert len([r for r in data['leave_one_parent_out'] if r['metric']=='SeamCov']) == 8
    missing = cohort(.02); missing[0].pop('SeamCov')
    d = compare(missing, cohort(0), requests, 'HALF_CAPPED500', 'START', 'OLD')
    assert next(r for r in d['summary'] if r['metric']=='SeamCov')['delta'] is None
    assert gates.threshold(-.01, .02, 0., strict=True)['status'] == 'INCOMPLETE'
    assert interval(dict(status='PASS', **{'SeamCov_0.005':.1, 'SeamCov_upper_0.005':.2}), 'SeamCov_0.005') == (.1, .2)
    assert 'question_A' not in DEFINITION and 'question_B' not in DEFINITION
    return dict(status='PASS', checks=6, model_calls=0, geometry_calls=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(ev.OUT))
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    result = self_test() if args.self_test else freeze(args.out) if args.freeze else run(args.out)
    print(json.dumps(ev.quality.clean(result), ensure_ascii=False))
