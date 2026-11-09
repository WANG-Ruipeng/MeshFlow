"""CPU-only summaries of saved local-error records and archived teacher tables.

No model construction, tensor autograd, OT, or geometry evaluator is imported.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


FIXED_CASES = ((0, 0, 10401), (1, 0, 10401), (0, 4, 10401),
               (0, 4, 10402), (1, 2, 10401), (1, 2, 10402))
OLD_RUN = Path('native_t1/runs/jedge5_diagnosis_20261002_102215')
TEACHER_HASH = '613f0787508bfc6eb1a02e4ae9178a0e763a0f2de46c30381a979bf6365eede4'
PARAMETER_HASH = '0eeb259ff8a81240aeedf65b9a4c9eb7487558db5df3d05c40f0c93ab4bf65e7'


def distribution(values):
    values = [float(v) for v in values if v is not None]
    if any(not math.isfinite(v) for v in values):
        raise ValueError('Nonfinite saved scalar')
    return dict(count=len(values), mean=statistics.fmean(values) if values else None,
                median=statistics.median(values) if values else None,
                minimum=min(values) if values else None,
                maximum=max(values) if values else None,
                negative=sum(v < 0 for v in values), zero=sum(v == 0 for v in values),
                positive=sum(v > 0 for v in values))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False, allow_nan=False) if isinstance(v, (dict, list, tuple)) else v
                        for k, v in row.items()})


def flatten(row, prefix=''):
    result = {}
    for k, v in row.items():
        name = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            result.update(flatten(v, name))
        else:
            result[name] = v
    return result


def _read_csv(path, expected_hash):
    data = path.read_bytes()
    identity = hashlib.sha256(data).hexdigest()
    if identity != expected_hash:
        raise ValueError(f'Archived teacher source changed: {path}')
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in row.items():
            if value == '':
                row[key] = None
            elif value in ('True', 'False'):
                row[key] = value == 'True'
            elif key in ('parent', 'patch', 'noise_index', 'FM_denominator'):
                row[key] = int(value)
            elif key not in ('model', 'precision', 'state_id', 'context_group'):
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
    return rows, dict(path=str(path.resolve()), sha256=identity, size_bytes=len(data))


def _group_teacher(rows, keys, metrics):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    return [dict(**dict(zip(keys, group)), rows=len(items),
                 metrics={k: distribution([x[k] for x in items]) for k in metrics})
            for group, items in sorted(groups.items())]


def reuse_teacher(out, archive=OLD_RUN):
    """Read preexisting scalar CSVs only, never re-evaluate teacher states."""
    out, archive = Path(out), Path(archive)
    teacher, source = _read_csv(archive/'stage_c/teacher_context_time.csv', TEACHER_HASH)
    parameters, psource = _read_csv(archive/'stage_c/parameter_velocity.csv', PARAMETER_HASH)
    if len(teacher) != 400 or len(parameters) != 8:
        raise ValueError('Archived teacher grid incomplete')
    selected = [r for r in teacher if r['teacher_t'] in (.5, .75, .9)]
    metrics = ['loss_FM', 'loss_Edge', 'loss_Surface', 'loss_joint_actual',
               'joint_actual_over_FM', 'cosine_FM_joint_actual']
    by_identity = {(r['model'], r['precision'], r['state_id']): r for r in selected}
    paired = []
    for r in selected:
        if r['model'] != 'J':
            continue
        b = by_identity['B', r['precision'], r['state_id']]
        row = {k: r[k] for k in ('precision', 'state_id', 'parent', 'patch', 'noise_index', 'teacher_t', 'context_group')}
        row.update({k: r[k]-b[k] if r[k] is not None and b[k] is not None else None for k in metrics})
        paired.append(row)
    result = dict(status='PASS', schema='local_error_B_readonly_teacher_v1', sources=[source, psource],
        records_read=len(teacher), records_selected=len(selected), parameter_records=len(parameters),
        new_model_forwards=0, new_backwards=0, new_OT=0, new_geometry_scores=0,
        gradient_ratio_definition='Norm(lambda_E*g_E + lambda_5*g_L5) / Norm(g_FM); actual train gate; ramp=1',
        teacher_time_selection=[.5, .75, .9],
        parameter_probe_scope='Archived BF16 ninth transformer block MLP final Linear weight+bias, t=.75, P0/P1 patch0/4; not all-network gradients',
        groups=_group_teacher(selected, ('model', 'precision', 'context_group', 'teacher_t'), metrics),
        by_parent_patch_time=_group_teacher(selected, ('model', 'precision', 'parent', 'patch', 'teacher_t'), metrics),
        loss_J_minus_B=_group_teacher(paired, ('precision', 'context_group', 'teacher_t'), metrics),
        paired_context_loss_differences=paired, parameter_rows=parameters,
        limits=['Independent archived teacher Gaussian/GT states differ from this run free trajectories.',
                'No pointwise causal correlation with local defects is defined or computed.',
                'Training loss gate is absent from frozen generation forward.',
                'Training-C and other-C labels concern already-observed contexts, not new-object generalization.'])
    write_json(out/'analysis/teacher_reuse.json', result)
    write_csv(out/'analysis/teacher_reused_context_time.csv', selected)
    write_csv(out/'analysis/teacher_reused_parameter_probe.csv', parameters)
    return result

LOCAL_TIMES = (0., .1, .2, .35, .49, .5, .51, .65, .75, .85, .95, .98)
METRICS = ('remainder_symmetric_RMS_bbox_pct', 'free_to_R12_RMS_bbox_pct',
    'R12_to_free_RMS_bbox_pct', 'pairwise_free_to_R12_coverage_within_1pct_fraction',
    'pairwise_R12_to_free_coverage_within_1pct_fraction', 'fixed_R12_coverage_RMS_bbox_pct',
    'fixed_R12_coverage_within_1pct_fraction', 'boundary_RMS_bbox_pct',
    'boundary_p95_bbox_pct', 'boundary_coverage_fraction',
    'Gamma_to_free_edges_RMS_bbox_pct', 'Gamma_to_free_edges_p95_bbox_pct',
    'CF_proper_affected_free_fraction', 'CF_proper_pair_count', 'FF_illegal_pair_count',
    'free_near_degenerate_count', 'free_exact_zero_area_count', 'free_total_surface_area')


def identity(row):
    return str(row['model']), int(row['parent']), int(row['patch']), int(row['seed'])


def _local(row):
    return row['local']


def summarize_local(rows):
    """Summarize registered times only; any .8 dependency state is kept separate."""
    rows = sorted(rows, key=lambda r: (identity(r), float(r['t'])))
    official = [r for r in rows if float(r['t']) in LOCAL_TIMES]
    supplemental = [r for r in rows if float(r['t']) not in LOCAL_TIMES]
    expected = {(m, *c) for m in ('B', 'J') for c in FIXED_CASES}
    observed = {identity(r) for r in official}
    if observed != expected:
        raise ValueError('Local diagnostic trajectory identities differ from fixed cases')
    trajectories = []
    for key in sorted(expected):
        group = [r for r in official if identity(r) == key]
        if len(group) != 12 or {float(r['t']) for r in group} != set(LOCAL_TIMES):
            raise ValueError('Local time grid incomplete or duplicated')
        peak = max(group, key=lambda r: _local(r)['delta_h']['coordinate_rms'])
        peak_summary = _local(peak)
        ratios = [_local(r)['delta_h_over_delta_half_rms'] for r in group]
        halved = sum(_local(r)['delta_half']['coordinate_rms'] < _local(r)['delta_h']['coordinate_rms'] for r in group)
        term_sums = {k: sum(r['decomposition']['terms'][k]['l2_norm'] for r in group)
                     for k in ('space', 'time', 'cross')}
        total_sum = sum(term_sums.values())
        periods = []
        for name, lo, hi in (('early_[0,.35)', 0., .35), ('middle_[.35,.65]', .35, .65), ('late_(.65,1]', .65, 1.)):
            subset = [r for r in group if ((lo <= float(r['t']) < hi) if name.startswith('early') else
                      (lo <= float(r['t']) <= hi) if name.startswith('middle') else (lo < float(r['t']) <= hi))]
            periods.append(dict(period=name, sampled_time_count=len(subset),
                coordinate_rms_bbox=distribution([_local(r)['delta_h']['coordinate_rms_bbox'] for r in subset])))
        trajectories.append(dict(model=key[0], parent=key[1], patch=key[2], seed=key[3],
            peak_t=peak['t'], peak_delta_h=peak_summary['delta_h'],
            peak_decomposition=peak['decomposition'],
            peak_same_state_model_difference=peak.get('same_state_model_difference'),
            delta_h_over_half=distribution(ratios), half_step_difference_smaller_count=halved,
            main_time_states=12, sampled_periods=periods,
            term_norm_sum_shares={k: v/total_sum if total_sum else None for k, v in term_sums.items()},
            term_norm_sum_share_warning='Descriptive magnitudes over 12 unevenly spaced samples; not time integrals or causal explained variance.',
            cancellation_fraction=distribution([r['decomposition']['cancellation_fraction'] for r in group]),
            maximum_identity_residual=max(r['decomposition']['identity_residual']['max_abs'] for r in group)))
    paired = []
    indexed = {(identity(r), float(r['t'])): r for r in official}
    for case in FIXED_CASES:
        for t in LOCAL_TIMES:
            b, j = indexed[(('B', *case), t)], indexed[(('J', *case), t)]
            paired.append(dict(parent=case[0], patch=case[1], seed=case[2], t=t,
                J_minus_B_delta_coordinate_rms_bbox=_local(j)['delta_h']['coordinate_rms_bbox']-_local(b)['delta_h']['coordinate_rms_bbox'],
                same_state_warning='B and J own-path states differ. Same-state model velocity differences are reported separately.'))
    return dict(main_state_count=len(official), supplemental_state_count=len(supplemental),
                trajectories=trajectories, paired_own_path_local_differences=paired,
                summary_counting='All 144 registered states; .8 dependency states excluded from main-grid aggregates.')


def summarize_amplification(rows):
    """Rows contain propagation plus original_scalars and replaced_scalars."""
    answer, trajectory = [], []
    for row in sorted(rows, key=lambda r: (identity(r), float(r['t']))):
        prop = row['propagation']
        original, replaced = row['original_scalars'], row['replaced_scalars']
        changes = {k: None if original.get(k) is None or replaced.get(k) is None else replaced[k]-original[k] for k in METRICS}
        answer.append(dict(model=row['model'], parent=row['parent'], patch=row['patch'], seed=row['seed'], t=row['t'],
            initial_coordinate_rms=prop['initial_difference']['coordinate_rms'],
            final_coordinate_rms=prop['final_difference']['coordinate_rms'],
            initial_coordinate_rms_bbox=prop['initial_difference']['coordinate_rms_bbox'],
            final_coordinate_rms_bbox=prop['final_difference']['coordinate_rms_bbox'],
            initial_max_abs=prop['initial_difference']['max_abs'],
            final_max_abs=prop['final_difference']['max_abs'],
            amplification=prop['amplification'], denominator_resolution_floor=prop['denominator_resolution_floor'],
            ratio_status=prop['status'], **{'original_'+k: original.get(k) for k in METRICS},
            **{'replaced_'+k: replaced.get(k) for k in METRICS}, **{'delta_'+k: v for k, v in changes.items()}))
    for key in sorted({identity(r) for r in rows}):
        group = [r for r in answer if identity(r) == key]
        if len(group) != 3 or {float(r['t']) for r in group} != {.2, .5, .8}:
            raise ValueError('Amplification branches incomplete')
        resolved = [r for r in group if r['amplification'] is not None]
        strongest = max(resolved, key=lambda r: r['amplification']) if resolved else None
        ff_reductions = []
        for row in group:
            if row['delta_FF_illegal_pair_count'] is not None and row['delta_FF_illegal_pair_count'] < 0:
                ff_reductions.append({k: row[k] for k in ('t', 'delta_FF_illegal_pair_count',
                    'delta_CF_proper_pair_count', 'delta_remainder_symmetric_RMS_bbox_pct',
                    'delta_R12_to_free_RMS_bbox_pct', 'delta_boundary_p95_bbox_pct',
                    'delta_Gamma_to_free_edges_p95_bbox_pct')})
        trajectory.append(dict(model=key[0], parent=key[1], patch=key[2], seed=key[3],
            amplification=distribution([r['amplification'] for r in group]),
            amplification_gt1=sum(r['amplification'] is not None and r['amplification'] > 1 for r in group),
            strongest_replacement_t=None if strongest is None else strongest['t'],
            strongest_amplification=None if strongest is None else strongest['amplification'],
            RMS_improved_count=sum(r['delta_remainder_symmetric_RMS_bbox_pct'] < 0 for r in group),
            FF_reduced_count=len(ff_reductions), FF_reduction_tradeoffs=ff_reductions,
            metric_changes={k: distribution([r['delta_'+k] for r in group]) for k in METRICS}))
    return dict(rows=answer, trajectories=trajectory)


def summarize_sensitivity(rows):
    groups = {}
    for r in rows:
        groups.setdefault((identity(r), float(r['t'])), {})[float(r['eta'])] = r
    records = []
    for (key, t), group in sorted(groups.items()):
        if set(group) != {1e-4, 5e-5}:
            raise ValueError('Both preregistered perturbation scales required')
        coarse, fine = group[1e-4], group[5e-5]
        g1, g2 = coarse.get('gain'), fine.get('gain')
        records.append(dict(model=key[0], parent=key[1], patch=key[2], seed=key[3], t=t,
            gain_eta_1e4=g1, gain_eta_5e5=g2,
            eta_1e4_status=coarse['status'], eta_5e5_status=fine['status'],
            relative_gain_difference=None if g1 is None or g2 is None or max(g1,g2)==0 else abs(g1-g2)/max(g1,g2),
            coarse_actual_denominator=coarse.get('actual_denominator_coordinate_rms'),
            fine_actual_denominator=fine.get('actual_denominator_coordinate_rms')))
    return dict(states=len(records), records=records,
        gain_scale_relative_difference=distribution([r['relative_gain_difference'] for r in records]),
        interpretation='Specific local-defect direction finite differences only; eta dependence may reflect rounding or nonlinearity, not a spectral bound.')


def build_saved_analysis(out, local_rows, suffix_rows, sensitivity_rows=None, archive=OLD_RUN):
    out = Path(out)
    local = summarize_local(local_rows)
    amplification = summarize_amplification(suffix_rows)
    sensitivity = summarize_sensitivity(sensitivity_rows) if sensitivity_rows is not None else None
    originals = {identity(r): r['original_scalars'] for r in suffix_rows}
    baseline_comparisons = [dict(parent=p, patch=j, seed=seed, **{k: originals[('J', p, j, seed)].get(k)-originals[('B', p, j, seed)].get(k) if originals[('J', p, j, seed)].get(k) is not None and originals[('B', p, j, seed)].get(k) is not None else None for k in METRICS}) for p,j,seed in FIXED_CASES]
    paired_amp = []
    amp_index = {(identity(r), float(r['t'])):r for r in amplification['rows']}
    for (key, t), jrow in amp_index.items():
        if key[0] != 'J':
            continue
        brow = amp_index[(('B', *key[1:]), t)]
        paired_amp.append(dict(parent=key[1], patch=key[2], seed=key[3], t=t,
            J_minus_B_amplification=None if jrow['amplification'] is None or brow['amplification'] is None else jrow['amplification']-brow['amplification']))
    own_path_comparison = dict(
        local_J_minus_B=distribution([r['J_minus_B_delta_coordinate_rms_bbox'] for r in local['paired_own_path_local_differences']]),
        amplification_J_minus_B=distribution([r['J_minus_B_amplification'] for r in paired_amp]),
        paired_amplification=paired_amp,
        warning='Own-path state and local-defect direction differ between B and J; this is descriptive matched-case behavior, not one common-state Jacobian comparison.')
    if sensitivity is not None:
        gain_index = {(identity(r), float(r['t'])): r for r in sensitivity['records']}
        own_path_comparison['direction_gain_J_minus_B'] = {eta: distribution([r[eta]-gain_index[(('B', *key[1:]), t)][eta] for (key,t),r in gain_index.items() if key[0]=='J' and r[eta] is not None and gain_index[(('B', *key[1:]),t)][eta] is not None]) for eta in ('gain_eta_1e4','gain_eta_5e5')}
    teacher = reuse_teacher(out, archive)
    write_csv(out/'local_defect_per_state.csv', [flatten(r) for r in sorted(local_rows, key=lambda r: (identity(r), float(r['t'])))])
    write_csv(out/'amplification_per_case.csv', amplification['rows'])
    if sensitivity_rows is not None:
        write_csv(out/'analysis/directional_sensitivity.csv', [flatten(r) for r in sensitivity_rows])
    result = dict(status='PASS', schema='local_error_B_cpu_analysis_v1',
        normalization='Numerical defect RMS averages scalar free coordinates (900); geometry surface RMS averages squared 3D point distances. Divide each by fixed GT bbox, never mix the two.',
        local=local, amplification=amplification, sensitivity=sensitivity,
        original_J_minus_B_geometry=baseline_comparisons, own_path_comparison=own_path_comparison, teacher_reuse=teacher,
        scientific_calls=dict(model_forwards=0, backwards=0, optimizer_updates=0, OT=0, geometry_scores=0),
        limits=['Fixed six diagnostic cases, not a representative generalization set.',
                'Two half steps minus one step is a local step-size difference, not an exact truncation error.',
                'Finite-direction gains are not full Jacobian norms, Lipschitz bounds, chaos or stiffness proofs.',
                'Amplification uses the Euler100 suffix map, not exact continuous ODE dynamics.',
                'The training geometry gate is never executed by frozen sampling.',
                'A beneficial single-step replacement is diagnostic and is not a validated new sampler.'])
    write_json(out/'analysis/summary.json', result)
    return result


def run(out):
    out = Path(out)
    local_record = json.loads((out/'raw/local_rows.json').read_text(encoding='utf-8-sig'))
    suffix_record = json.loads((out/'raw/suffix_rows.json').read_text(encoding='utf-8-sig'))
    baseline, modified = {}, {}
    for model in ('B', 'J'):
        record = json.loads((out/'scores'/model/'results.json').read_text(encoding='utf-8-sig'))
        if record['status'] != 'PASS':
            raise ValueError('Complete CPU scoring required before final analysis')
        for r in record['baseline_rows']:
            key = identity(r)
            if key in baseline:
                raise ValueError('Duplicate original score')
            baseline[key] = r.get('augmented_scalars', r['scalars'])
        for r in record['suffix_rows']:
            step = r.get('replacement_step', r.get('index'))
            if step is None:
                step = round(float(r['t'])*100)
            key = (identity(r), int(step))
            if key in modified:
                raise ValueError('Duplicate replaced endpoint score')
            modified[key] = r['scalars']
    joined = []
    for r in suffix_record['rows']:
        step = int(r.get('index', round(float(r['t'])*100)))
        joined.append(dict(r, original_scalars=baseline[identity(r)],
                           replaced_scalars=modified[identity(r), step]))
    all_local = local_record['rows'] + local_record.get('supplemental_rows', [])
    sensitivity_path = out/'raw/sensitivity_rows.json'
    sensitivity = json.loads(sensitivity_path.read_text(encoding='utf-8-sig'))['rows'] if sensitivity_path.exists() else None
    return build_saved_analysis(out, all_local, joined, sensitivity)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--teacher-only', action='store_true')
    args = parser.parse_args()
    result = reuse_teacher(args.out) if args.teacher_only else run(args.out)
    print(json.dumps({'status': result['status'], 'model_forwards': 0, 'geometry_scores': 0}))
