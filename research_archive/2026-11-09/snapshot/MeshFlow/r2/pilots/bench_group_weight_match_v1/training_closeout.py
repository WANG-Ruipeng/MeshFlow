"""Post-endpoint CPU audit of actual matched losses, exposure and coefficients.

Reads existing JSON logs only; no trainer/model import, tensor load, OT or GPU.
Execution endpoint/resource/source-state validation is separately owned by
execution_closeout.py. This module never changes frozen scientific artifacts.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import statistics

from . import coefficient_audit as A

ROOT, OUT = A.ROOT, A.OUT
DEST = OUT / 'training_closeout'
GROUPS = A.GROUPS
EXPECTED = dict(train_forward=12000, train_backward=12000, optimizer_update=1500)
CAUTION = (
    'Unweighted FM and optimized objective are distinct. Group loss contribution '
    'is identifiable from masked SSE and the logged scalar gains; group gradient '
    'norms/directions and AdamW update shares are not logged and cannot be inferred. '
    'generator_gradient_norm is the total pre-clipping norm returned by '
    'clip_grad_norm_, with clip threshold1. Matched group coefficients do not '
    'restore B samples, group errors, gradients, or the optimization trajectory. '
    'Training losses do not rank generation quality. Curves use the same '
    'non-overlapping50-update bins and parent-equal group MSE convention as old W.'
)


def close(a, b, message, *, tight=False):
    A.require(isinstance(a, (float, int)) and isinstance(b, (float, int))
              and math.isfinite(a) and math.isfinite(b), message + ': nonfinite')
    A.require(math.isclose(a, b, rel_tol=1e-10 if tight else 5e-6,
                           abs_tol=1e-9 if tight else 2e-7), message)


def cursor(row, step):
    A.require((row['local_step'], row['stage_step'], row['global_step'],
               row['native_total_step'], row['curriculum_step'], row['seam_total_step'], row['stage']) ==
              (step, step, 21000 + step, 9000 + step, 3000 + step, 0, 'NATIVE'), 'Actual training cursor differs')


def audit_calls(rows, exposures):
    A.require(len(rows) == 2 * sum(EXPECTED.values()), 'Call event count differs')
    expected_order = [(s, j, kind) for s in range(1, 1501)
                      for j, kind in ([(j, k) for j in range(8) for k in ('train_forward', 'train_backward')]
                                      + [(None, 'optimizer_update')])]
    kinds, attempts = Counter(), set()
    for i, (step, micro, kind) in enumerate(expected_order):
        begin, end = rows[2*i:2*i+2]
        A.require((begin['event'], end['event']) == ('CALL_BEGIN', 'CALL_DONE'), 'Failed/pending/overlapping call event')
        for row in (begin, end):
            A.require(row['arm'] == 'W_MATCHED' and row['call_id'] == i + 1 and row['kind'] == kind,
                      'Actual call identity/order differs')
            c = row['context']; cursor(c, step)
            A.require(c['arm'] == 'W_MATCHED' and c.get('microbatch') == micro, 'Call microbatch differs')
            if micro is not None:
                A.require(c['condition_id'] == exposures[(step-1)*8+micro]['condition_id'], 'Call/exposure identity differs')
        A.require(begin['context'] == end['context'] and begin['attempt_id'] == end['attempt_id'], 'Call completion identity differs')
        A.require(math.isfinite(end['seconds']) and end['seconds'] >= 0, 'Call duration invalid')
        attempts.add(begin['attempt_id']); kinds[kind] += 1
    A.require(dict(kinds) == EXPECTED and len(attempts) == 1, 'Fixed one-attempt budget differs')
    return dict(successful=dict(kinds), attempted=dict(kinds), attempts=len(attempts),
                failed=0, pending=0, events=len(rows), step_micro_condition_order_exact=True)


def audit_update(step, samples, update, coefficient, expected_rows, old_exposures):
    """Verify actual scalars against prepared references, then independently recompute losses."""
    cursor(update, step)
    A.require(update['arm'] == 'W_MATCHED' and len(samples) == 8, 'Matched update shape/arm differs')
    A.require(update['loss_normalization'] == 'narrow_reference_group_MSE_coefficients_v1', 'Actual normalization identity differs')
    A.require(update['group_allocation'] == coefficient, 'Actual allocation differs from frozen coefficients')
    dw, db, rho = coefficient['D_W'], coefficient['D_B'], coefficient['rho']
    A.require(update['free_coordinate_denominator'] == dw, 'Actual denominator differs')
    unweighted, weighted, sse = [], [], []
    max_base_error = max_weight_error = 0.0
    for j, (row, expected, old) in enumerate(zip(samples, expected_rows, old_exposures)):
        cursor(row, step)
        A.require(row['arm'] == 'W_MATCHED' and row['microbatch'] == j, 'Actual exposure cursor differs')
        A.require(all(row[k] == v for k, v in expected.items()), 'Matched actual stream differs from original frozen W')
        for key in (*A.RANDOM_FIELDS, 't', 'N', 'K', 'all_free_scalar_count', 'bulk_scalar_count',
                    'seam_tip_scalar_count', 'mechanism', 'head_forward_called'):
            A.require(row[key] == old[key], 'Actual W/M sample identity/count differs: ' + key)
        count = 9 * (int(row['N']) - int(row['K']))
        A.require(row['all_free_scalar_count'] == coefficient['micro_free_scalar_counts'][j] == count,
                  'Actual mask count differs')
        A.require(row['actual_batch_denominator_D_W'] == dw and row['reference_denominator_D_B'] == db
                  and row['reference_HIGH_ratio_rho'] == rho
                  and row['loss_gain'] == coefficient['micro_gains'][j], 'Actual applied coefficient differs')
        value = row['all_free_velocity_sse']
        A.require(math.isfinite(value) and value >= 0, 'Invalid actual SSE')
        base, objective = value / dw, value / dw * row['loss_gain']
        close(base, row['unweighted_loss_contribution'], 'Unweighted micro loss differs from SSE/D_W')
        close(objective, row['loss_contribution'], 'Weighted micro objective differs from applied gain')
        if coefficient['branch'] == 'FOCUS':
            A.require(row['loss_gain'] == 1 and row['unweighted_loss_contribution'] == row['loss_contribution'],
                      'FOCUS objective changed within update')
        max_base_error = max(max_base_error, abs(base - row['unweighted_loss_contribution']))
        max_weight_error = max(max_weight_error, abs(objective - row['loss_contribution']))
        unweighted.append(row['unweighted_loss_contribution']); weighted.append(row['loss_contribution']); sse.append(value)
    A.require(sum(r['all_free_scalar_count'] for r in samples) == update['all_free_scalar_count'] == dw,
              'Actual summed scalar count differs')
    close(sum(unweighted), update['FM'], 'FM contributions do not sum', tight=True)
    A.require(update['FM'] == update['unweighted_FM'], 'FM is not unweighted_FM')
    close(sum(weighted), update['objective'], 'Objective contributions do not sum', tight=True)
    close(sum(sse), update['all_free_velocity_sse'], 'SSE contributions do not sum', tight=True)
    groups = {g: sum(r['all_free_velocity_sse'] for r in samples if r['replay_group'] == g) for g in GROUPS}
    for g in GROUPS:
        close(groups[g], update['group_velocity_sse'][g], 'Group SSE log differs', tight=True)
    predicted = (groups['OLD'] + rho * groups['HIGH']) / db if rho is not None else sum(sse) / dw
    close(predicted, update['objective'], 'Whole update weighted objective algebra differs')
    if rho is None:
        A.require(update['objective'] == update['unweighted_FM'], 'FOCUS objective differs from original formula')
    gradient = update['generator_gradient_norm']
    A.require(math.isfinite(gradient) and gradient >= 0 and update['head_step'] == update['head_calls'] == 0,
              'Invalid total gradient/head activity')
    return dict(stage_step=step, branch=coefficient['branch'], D_B=db, D_W=dw, rho=rho,
        actual_unweighted_FM=update['unweighted_FM'], reconstructed_unweighted_FM=sum(sse)/dw,
        actual_objective=update['objective'], reconstructed_objective=predicted,
        maximum_micro_unweighted_error=max_base_error, maximum_micro_weighted_error=max_weight_error,
        total_preclip_gradient_norm=gradient, clipping_threshold=1.,
        group_gradient_norms='UNKNOWN_NOT_LOGGED', actual_W_stream_exact=True,
        focus_objective_exact_noop=rho is None)


def group_rows(label, exposures, updates, matched=False):
    result = []
    for index, update in enumerate(updates):
        batch = exposures[8*index:8*index+8]
        dw = update['free_coordinate_denominator']
        objective = update['objective'] if matched else update['FM']
        fm = update['unweighted_FM'] if matched else update['FM']
        for group in GROUPS:
            subset = [r for r in batch if r['replay_group'] == group]
            coordinates = sum(r['all_free_scalar_count'] for r in subset)
            sse = sum(r['all_free_velocity_sse'] for r in subset)
            unweighted = sum(r['unweighted_loss_contribution'] if matched else r['loss_contribution'] for r in subset)
            contribution = sum(r['loss_contribution'] for r in subset)
            close(unweighted, sse/dw, 'Recorded base group loss differs')
            gain = subset[0]['loss_gain'] if matched and subset else 1.
            close(contribution, sse/dw*gain, 'Recorded weighted group loss differs')
            result.append(dict(label=label, local_step=index+1, group=group,
                observation_status='OBSERVED' if subset else 'NA_NO_EXPOSURE', instances=len(subset),
                free_scalars=coordinates, D_W=dw,
                group_coordinate_MSE=sse/coordinates if coordinates else None,
                unweighted_loss_contribution=unweighted, objective_contribution=contribution,
                physical_coordinate_fraction=coordinates/dw,
                applied_group_MSE_coefficient=coordinates/dw*gain,
                objective_share=contribution/objective if objective else None,
                unweighted_loss_share=unweighted/fm if fm else None,
                group_gradient_share='UNKNOWN_NOT_LOGGED'))
    return result


def summarize(label, exposures, updates, allocations, matched=False):
    total_objective = sum(u['objective'] if matched else u['FM'] for u in updates)
    total_fm = sum(u['FM'] for u in updates)
    total_coordinates = sum(r['all_free_scalar_count'] for r in exposures)
    result = []
    for group in GROUPS:
        rs = [r for r in allocations if r['group'] == group]
        e = [r for r in exposures if r['replay_group'] == group]
        coords = sum(r['free_scalars'] for r in rs)
        parents = defaultdict(list)
        for row in e:
            parents[row['parent_id']].append(row['all_free_velocity_sse']/row['all_free_scalar_count'])
        result.append(dict(label=label, group=group, instances=len(e), parents=len(parents),
            total_free_scalars=coords, actual_instance_share=len(e)/len(exposures),
            actual_pooled_coordinate_share=coords/total_coordinates,
            mean_update_group_MSE_coefficient=statistics.mean(r['applied_group_MSE_coefficient'] for r in rs),
            observed_coordinate_MSE=sum(r['all_free_velocity_sse'] for r in e)/coords,
            observed_parent_macro_MSE=statistics.mean(statistics.mean(v) for v in parents.values()),
            actual_objective_contribution_sum=sum(r['objective_contribution'] for r in rs),
            objective_share_of_summed_objective=sum(r['objective_contribution'] for r in rs)/total_objective,
            unweighted_loss_share_of_summed_FM=sum(r['unweighted_loss_contribution'] for r in rs)/total_fm,
            group_gradient_share='UNKNOWN_NOT_LOGGED'))
    return result


def curves(label, exposures, updates, allocations, matched=False):
    rows = []
    for start in range(0, 1500, 50):
        end = start + 50
        subset, us = exposures[start*8:end*8], updates[start:end]
        for group in ('ALL_MIXTURE', *GROUPS):
            e = [r for r in subset if group == 'ALL_MIXTURE' or r['replay_group'] == group]
            parent = defaultdict(list)
            for r in e:
                parent[r['parent_id']].append(r['all_free_velocity_sse']/r['all_free_scalar_count'])
            gs = [r for r in allocations if start < r['local_step'] <= end and r['group'] == group]
            rows.append(dict(label=label, group=group, first_local_step=start+1, last_local_step=end,
                actual_update_unweighted_FM_mean=statistics.mean(r['FM'] for r in us) if group == 'ALL_MIXTURE' else None,
                actual_update_objective_mean=statistics.mean(r['objective'] if matched else r['FM'] for r in us) if group == 'ALL_MIXTURE' else None,
                parent_macro_velocity_MSE=statistics.mean(statistics.mean(v) for v in parent.values()),
                coordinate_micro_velocity_MSE=sum(r['all_free_velocity_sse'] for r in e)/sum(r['all_free_scalar_count'] for r in e),
                mean_update_group_unweighted_contribution=statistics.mean(r['unweighted_loss_contribution'] for r in gs) if gs else None,
                mean_update_group_objective_contribution=statistics.mean(r['objective_contribution'] for r in gs) if gs else None))
    return rows


def parent_rows(label, exposures, updates, matched=False):
    by_parent = defaultdict(list)
    total_objective = sum(u['objective'] if matched else u['FM'] for u in updates)
    total_fm = sum(u['FM'] for u in updates)
    for row in exposures:
        by_parent[row['parent_id']].append(row)
    result = []
    for parent, rows in sorted(by_parent.items()):
        identities = {(r['condition_id'], r['replay_group'], r['N'], r['K']) for r in rows}
        A.require(len(identities) == 1, 'Parent condition/N/K changed within training')
        condition, group, n, k = next(iter(identities))
        coordinates = sum(r['all_free_scalar_count'] for r in rows)
        base = sum(r['unweighted_loss_contribution'] if matched else r['loss_contribution'] for r in rows)
        objective = sum(r['loss_contribution'] for r in rows)
        result.append(dict(label=label, parent_id=parent, condition_id=condition, group=group,
            N=n, K=k, count_embedding_bucket=n//20, instances=len(rows),
            clean_instances=sum(r['augmentation']=='CLEAN' for r in rows),
            xwidth_instances=sum(r['augmentation']=='X_WIDTH' for r in rows),
            free_scalars=coordinates, free_scalars_per_instance=9*(n-k),
            instance_mean_velocity_MSE=statistics.mean(r['all_free_velocity_sse']/r['all_free_scalar_count'] for r in rows),
            coordinate_micro_velocity_MSE=sum(r['all_free_velocity_sse'] for r in rows)/coordinates,
            unweighted_loss_contribution_sum=base, objective_contribution_sum=objective,
            share_of_summed_unweighted_FM=base/total_fm, share_of_summed_objective=objective/total_objective,
            gradient_share='UNKNOWN_NOT_LOGGED'))
    return result


def plot_curves(rows, path, objective=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), constrained_layout=True)
    for axis, group in zip(axes, ('ALL_MIXTURE', *GROUPS)):
        if objective:
            key = 'actual_update_objective_mean' if group == 'ALL_MIXTURE' else 'mean_update_group_objective_contribution'
            title = group + (' / optimized objective' if group == 'ALL_MIXTURE' else ' / objective contribution')
        else:
            key = 'actual_update_unweighted_FM_mean' if group == 'ALL_MIXTURE' else 'parent_macro_velocity_MSE'
            title = group + (' / unweighted FM' if group == 'ALL_MIXTURE' else ' / parent-equal MSE')
        for label in ('W', 'M'):
            points = [r for r in rows if r['label'] == label and r['group'] == group]
            axis.plot([r['last_local_step'] for r in points], [r[key] for r in points], label=label)
        axis.set_title(title); axis.set_xlabel('New optimizer updates'); axis.set_ylabel('Loss')
        axis.grid(alpha=.25); axis.legend()
    fig.suptitle('W original vs M group-weight matched; 50-update bins. Loss shares are not gradient shares or generation quality.', fontsize=11)
    fig.savefig(path, dpi=160, metadata={'Software': 'MeshFlow independent training_closeout'}); plt.close(fig)


def run():
    done_path = DEST / 'complete.json'
    if done_path.exists():
        done = A.read(done_path)
        for path, digest in {**done['inputs'], **done['outputs'], **done['code']}.items():
            A.require(A.sha(A.resolve(path)) == digest, 'Completed audit evidence changed: ' + path)
        return done
    reference_receipt = A.read(OUT/'coefficient_audit/complete.json')
    A.require(reference_receipt['status'] == 'PASS', 'Independent coefficient audit is not complete')
    candidate_path = A.resolve(reference_receipt['candidate_comparison']['path'])
    A.require(A.sha(candidate_path) == reference_receipt['candidate_comparison']['sha256'], 'Frozen coefficient schedule changed')
    coefficients = A.read(candidate_path)['steps']
    old = {label: A.load_source(label) for label in ('B', 'W')}
    folder = OUT / 'training/W_MATCHED'
    state = A.read(folder/'status.json')
    A.require(state['status'] == 'COMPLETE' and state['local_step'] == 1500,
              'Wait for the formal training COMPLETE endpoint before auditing logs')
    paths = [folder/name for name in ('exposure.jsonl', 'training_log.jsonl', 'calls.jsonl')]
    # Cardinality gates are checked before any result or PNG is written.
    exposure, updates, calls = (A.lines(p) for p in paths)
    A.require(len(exposure) == 12000 and len(updates) == 1500, 'Endpoint not yet complete: require12000 exposures and1500 updates')
    ledger = audit_calls(calls, exposure)
    actual, allocations, aggregates, bins, comparisons, parents = [], [], [], [], [], []
    for i, coefficient in enumerate(coefficients):
        record = audit_update(i+1, exposure[8*i:8*i+8], updates[i], coefficient,
            old['W']['schedule']['steps'][i]['samples'], old['W']['exposures'][8*i:8*i+8])
        actual.append(record)
        for g in GROUPS:
            expected = coefficient['m_old'] if g == 'OLD' else coefficient['m_focus'] if g == 'FOCUS' else coefficient['m_high_B']
            true_count = sum(r['all_free_scalar_count'] for r in exposure[8*i:8*i+8] if r['replay_group'] == g)
            gain = next((r['loss_gain'] for r in exposure[8*i:8*i+8] if r['replay_group'] == g), 1.)
            close(true_count/coefficient['D_W']*gain, expected/coefficient['D_B'], 'Actual matched group coefficient differs', tight=True)
        for j, row in enumerate(exposure[8*i:8*i+8]):
            original = old['W']['exposures'][8*i+j]
            comparisons.append(dict(local_step=i+1, microbatch=j, parent_id=row['parent_id'],
                condition_id=row['condition_id'], group=row['replay_group'], N=row['N'], K=row['K'],
                free_scalars=row['all_free_scalar_count'], t=row['t'],
                random_fields_and_time_equal=True, loss_gain=row['loss_gain'],
                D_W=row['actual_batch_denominator_D_W'], D_B=row['reference_denominator_D_B'],
                rho=row['reference_HIGH_ratio_rho'],
                actual_unweighted_loss=row['unweighted_loss_contribution'], actual_weighted_loss=row['loss_contribution'],
                original_W_loss=original['loss_contribution'], gradient_share='UNKNOWN_NOT_LOGGED'))
    for label in ('B', 'W', 'M'):
        e = exposure if label == 'M' else old[label]['exposures']
        if label == 'M':
            u = updates
        else:
            experiment, arm, _ = A.SOURCES[label]
            u = A.lines(ROOT/'experiments'/experiment/'training'/arm/'training_log.jsonl')
        gr = group_rows(label, e, u, label == 'M'); allocations.extend(gr)
        aggregates.extend(summarize(label, e, u, gr, label == 'M'))
        bins.extend(curves(label, e, u, gr, label == 'M'))
        parents.extend(parent_rows(label, e, u, label == 'M'))
    A.require(Counter(r['replay_group'] for r in exposure) == {'OLD':9000, 'FOCUS':1500, 'HIGH':1500}, 'Actual exposure totals differ')
    parent_counts = Counter(r['parent_id'] for r in exposure)
    A.require(len(parent_counts) == 30 and all(parent_counts[p] == 250 for p in parent_counts if any(r['parent_id']==p and r['replay_group']=='HIGH' for r in exposure)), 'HIGH6 exposure balance differs')
    tables = dict(per_update_actual=actual, per_instance_pairing=comparisons,
                  per_update_group_losses=allocations, group_loss_summary=aggregates,
                  per_parent_loss_summary=parents, curves_50_updates=bins)
    for name, rows in tables.items():
        A.write_once(DEST/(name+'.csv'), A.csv_text(rows))
    plot_curves(bins, DEST/'unweighted_FM_curves.png')
    plot_curves(bins, DEST/'objective_curves.png', objective=True)
    inputs = {**old['B']['input_hashes'], **old['W']['input_hashes'],
              **{A.rel(p): A.sha(p) for p in paths + [candidate_path, OUT/'coefficient_audit/complete.json']}}
    overall = dict(status='PASS', actual_updates=1500, actual_instances=12000, actual_W_stream_rows_equal=12000,
        actual_W_time_random_fields_equal=12000, focus_updates_exact_noop=750, high_updates_matched_coefficients=750,
        max_micro_unweighted_reconstruction_error=max(r['maximum_micro_unweighted_error'] for r in actual),
        max_micro_objective_reconstruction_error=max(r['maximum_micro_weighted_error'] for r in actual),
        actual_objective_mean=statistics.mean(r['objective'] for r in updates),
        actual_unweighted_FM_mean=statistics.mean(r['unweighted_FM'] for r in updates),
        total_preclip_gradient_norm_mean=statistics.mean(r['generator_gradient_norm'] for r in updates),
        total_preclip_gradient_norm_max=max(r['generator_gradient_norm'] for r in updates),
        updates_above_clip_threshold=sum(r['generator_gradient_norm'] > 1 for r in updates),
        group_gradient_norms='UNKNOWN_NOT_LOGGED', ledger=ledger, caution=CAUTION,
        endpoint_parameter_state_validation='SEPARATE_execution_closeout',
        code={A.rel(__file__): A.sha(__file__), A.rel(A.__file__): A.sha(A.__file__)},
        inputs=inputs, outputs={A.rel(p): A.sha(p) for p in DEST.iterdir() if p.is_file() and p.name != 'complete.json'},
        model_calls=0, new_OT_calls=0, tensor_loads=0)
    A.write_once(done_path, A.encoded(overall))
    return overall


def cpu_check():
    """Arithmetic-only log fixtures: FOCUS no-op and HIGH rho below/above one."""
    cases = []
    for rho in (None, .5, 1.5):
        step = 1 if rho is None else 2
        group = 'FOCUS' if rho is None else 'HIGH'
        counts = [90] * 6 + [180] * 2
        dw = sum(counts); hb = 0 if rho is None else int(360*rho)
        db = dw if rho is None else 540+hb
        gains = [1.]*8 if rho is None else [dw/db*(rho if j>=6 else 1.) for j in range(8)]
        coefficient = dict(stage_step=step, branch=group, D_W=dw, D_B=db, rho=rho,
                           micro_free_scalar_counts=counts, micro_gains=gains)
        rows, expected = [], []
        position = dict(local_step=step, stage_step=step, global_step=21000+step,
            native_total_step=9000+step, curriculum_step=3000+step, seam_total_step=0, stage='NATIVE')
        for j in range(8):
            g = 'OLD' if j<6 else group
            source = dict(replay_group=g, parent_id=g+str(j//2), condition_id='c'+str(j),
                alpha=1., scale_xyz=[1.,1.,1.], augmentation='CLEAN', noise_seed=j,
                epsilon_seed=j, time_seed=j, permutation_seed=j)
            expected.append(source)
            value = counts[j]*(1. if j<6 else 3.)
            base = value/dw
            rows.append(dict(**source, **position, arm='W_MATCHED', microbatch=j, t=.5,
                N=128, K=128-counts[j]//9, all_free_scalar_count=counts[j], bulk_scalar_count=counts[j],
                seam_tip_scalar_count=0, mechanism={'free_coordinate_count':counts[j]}, head_forward_called=False,
                actual_batch_denominator_D_W=dw, reference_denominator_D_B=db, reference_HIGH_ratio_rho=rho,
                loss_gain=gains[j], all_free_velocity_sse=value,
                unweighted_loss_contribution=base, loss_contribution=base*gains[j]))
        update = dict(**position, arm='W_MATCHED', group_allocation=coefficient,
            loss_normalization='narrow_reference_group_MSE_coefficients_v1',
            free_coordinate_denominator=dw, all_free_scalar_count=dw,
            FM=sum(r['unweighted_loss_contribution'] for r in rows),
            unweighted_FM=sum(r['unweighted_loss_contribution'] for r in rows),
            objective=sum(r['loss_contribution'] for r in rows),
            all_free_velocity_sse=sum(r['all_free_velocity_sse'] for r in rows),
            group_velocity_sse={g:sum(r['all_free_velocity_sse'] for r in rows if r['replay_group']==g) for g in GROUPS},
            generator_gradient_norm=.75, head_step=0, head_calls=0)
        audit_update(step, rows, update, coefficient, expected, rows)
        bad = [dict(r) for r in rows]; bad[0]['loss_contribution'] += .1
        try:
            audit_update(step, bad, update, coefficient, expected, rows)
        except ValueError:
            pass
        else:
            raise ValueError('Corrupt weighted contribution was accepted')
        cases.append(dict(rho=rho, status='PASS', corrupted_loss_rejected=True))
    result = dict(status='PASS', cases=cases, production_logs_read=0, model_calls=0, tensor_loads=0, new_OT_calls=0,
                  code_sha256=A.sha(__file__), scope='Synthetic arithmetic rows only, not observed experiment results')
    A.write_once(DEST/('cpu_check_'+result['code_sha256'][:12]+'.json'), A.encoded(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-check', action='store_true')
    args = parser.parse_args()
    result = cpu_check() if args.cpu_check else run()
    print(A.encoded({k:v for k,v in result.items() if k not in ('inputs','outputs','code')}))
