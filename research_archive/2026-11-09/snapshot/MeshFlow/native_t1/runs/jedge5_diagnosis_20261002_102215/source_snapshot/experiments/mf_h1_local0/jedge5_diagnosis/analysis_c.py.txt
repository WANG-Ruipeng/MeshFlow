"""Pure saved-JSON C summaries. No network, loss, gradient or geometry call."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


LOSSES = ('FM', 'Edge', 'Surface', 'joint_raw', 'joint_actual')
NORMS = ('FM', 'Edge', 'Surface', 'joint_raw', 'joint_actual')
COSINES = ('FM_Edge', 'FM_Surface', 'Edge_Surface', 'FM_joint_raw', 'FM_joint_actual')
GENERATION_METRICS = ('remainder_symmetric_RMS_bbox_pct', 'free_to_R12_RMS_bbox_pct',
    'R12_to_free_RMS_bbox_pct', 'boundary_RMS_bbox_pct', 'boundary_p95_bbox_pct',
    'boundary_coverage_fraction', 'CF_proper_affected_free_fraction', 'CF_proper_pair_count',
    'FF_illegal_pair_count', 'free_near_degenerate_count', 'free_total_surface_area')


def distribution(values):
    values = [float(value) for value in values if value is not None]
    if any(not math.isfinite(value) for value in values):
        raise ValueError('Nonfinite saved analysis scalar')
    if not values:
        return dict(available=0, mean=None, median=None, minimum=None, maximum=None)
    return dict(available=len(values), mean=statistics.fmean(values), median=statistics.median(values),
                minimum=min(values), maximum=max(values))


def flatten(row, gradient_key='velocity_gradients'):
    result = {key: row[key] for key in ('state_id', 'parent', 'patch', 'noise_index', 'teacher_t', 'context_group')}
    result.update({'loss_'+key: row['losses'][key] for key in LOSSES})
    result.update(active=row['geometry_active'], FM_denominator=row['FM_coordinate_denominator'])
    gradients = row[gradient_key]
    for name in NORMS:
        result['norm_'+name] = gradients['norms'][name]
    for name in COSINES:
        result['cosine_'+name] = gradients['cosines'][name]['value']
        result['angle_'+name] = gradients['cosines'][name]['angle_degrees']
    result.update(joint_raw_over_FM=gradients['joint_raw_over_FM'],
                  joint_actual_over_FM=gradients['joint_actual_over_FM'])
    return result


def metric_columns(rows):
    return [key for key in rows[0] if key.startswith(('loss_', 'norm_', 'cosine_', 'angle_'))
            or key in ('joint_raw_over_FM', 'joint_actual_over_FM')]


def grouped(rows, keys, metrics):
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        groups.setdefault(key, []).append(row)
    return [dict(group=dict(zip(keys, key)), state_count=len(items),
        distinct_teacher_noises=len({item['noise_index'] for item in items}),
        distinct_contexts=len({(item['parent'], item['patch']) for item in items}),
        metrics={name: distribution([item[name] for item in items]) for name in metrics})
        for key, items in sorted(groups.items())]


def paired(candidate, reference, metrics):
    reference = {row['state_id']: row for row in reference}
    if len(candidate) != len(reference) or {row['state_id'] for row in candidate} != set(reference):
        raise ValueError('Paired C state identities differ')
    answer = []
    for row in candidate:
        ref = reference[row['state_id']]
        keys = ('parent', 'patch', 'noise_index', 'teacher_t', 'context_group')
        if any(row[key] != ref[key] for key in keys):
            raise ValueError('Paired C context differs')
        delta = dict(state_id=row['state_id'], **{key: row[key] for key in keys})
        for name in metrics:
            delta[name] = None if row[name] is None or ref[name] is None else row[name]-ref[name]
        answer.append(delta)
    return answer


def paired_summary(rows, keys, metrics):
    answer = grouped(rows, keys, metrics)
    for group in answer:
        subset = [row for row in rows if all(row[k] == v for k, v in group['group'].items())]
        for name in metrics:
            values = [row[name] for row in subset if row[name] is not None]
            group['metrics'][name].update(negative_count=sum(x < 0 for x in values),
                zero_count=sum(x == 0 for x in values), positive_count=sum(x > 0 for x in values))
    return answer


def _read(path, sources):
    data = path.read_bytes()
    sources.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
    return json.loads(data.decode('utf-8-sig'))


def _csv(path, rows):
    if not rows:
        raise ValueError('No fabricated empty CSV')
    fields = list(rows[0])
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_analysis(out):
    out = Path(out)
    sources, flat, original = [], [], {}
    process = {}
    for model in ('B', 'J'):
        process[model] = _read(out/'stage_c'/f'{model}_process.json', sources)
        if process[model].get('status') != 'PASS' or process[model].get('frozen_tensor_identity') is not True:
            raise ValueError('Completed frozen C model process required')
        for precision in ('bf16', 'fp32'):
            record = _read(out/'stage_c'/'teacher'/f'{model}_{precision}.json', sources)
            if (record.get('status') != 'PASS' or record.get('completed_states') != 100
                    or record.get('precision') != precision or record.get('model') != model
                    or record.get('counts', {}).get('velocity_leaf_autograd_returns') != 300):
                raise ValueError('C fixed matrix incomplete')
            original[model, precision] = {row['state_id']: row for row in record['rows']}
            if len(original[model, precision]) != 100:
                raise ValueError('Duplicate C state identity')
            flat.extend(dict(model=model, precision=precision, **flatten(row)) for row in record['rows'])
    reference = original['B', 'bf16']
    for key, rows in original.items():
        if set(rows) != set(reference):
            raise ValueError('Model/precision teacher grid differs')
        for state_id, row in rows.items():
            if any(row[name] != reference[state_id][name] for name in ('shared_input_sha256', 'target_mapping_sha256', 'FM_coordinate_denominator')):
                raise ValueError('Paired teacher supervision or denominator changed')
    metrics = metric_columns(flat)
    summaries = dict(by_parent_patch_time=grouped(flat, ('model', 'precision', 'parent', 'patch', 'teacher_t'), metrics),
        by_parent_context_group_time=grouped(flat, ('model', 'precision', 'parent', 'context_group', 'teacher_t'), metrics))
    comparisons = {}
    for precision in ('bf16', 'fp32'):
        candidate = [row for row in flat if row['model'] == 'J' and row['precision'] == precision]
        ref = [row for row in flat if row['model'] == 'B' and row['precision'] == precision]
        delta = paired(candidate, ref, metrics)
        comparisons['J_minus_B_'+precision] = dict(per_state=delta,
            by_parent_patch_time=paired_summary(delta, ('parent', 'patch', 'teacher_t'), metrics),
            by_parent_context_group_time=paired_summary(delta, ('parent', 'context_group', 'teacher_t'), metrics))
    for model in ('B', 'J'):
        candidate = [row for row in flat if row['model'] == model and row['precision'] == 'fp32']
        ref = [row for row in flat if row['model'] == model and row['precision'] == 'bf16']
        delta = paired(candidate, ref, metrics)
        comparisons[model+'_FP32_minus_BF16'] = dict(per_state=delta,
            by_parent_patch_time=paired_summary(delta, ('parent', 'patch', 'teacher_t'), metrics))
    probes = []
    for model in ('B', 'J'):
        record = _read(out/'stage_c'/'teacher'/f'{model}_parameter_probe.json', sources)
        if (record.get('status') != 'PASS' or record.get('completed_states') != 4
                or record.get('precision') != 'bf16' or record.get('restored_frozen_and_grad_cleared') is not True
                or record.get('counts', {}).get('parameter_autograd_returns') != 12):
            raise ValueError('One-layer parameter probe incomplete')
        for row in record['rows']:
            velocity = original[model, 'bf16'][row['state_id']]
            if row['shared_input_sha256'] != velocity['shared_input_sha256']:
                raise ValueError('Parameter/velocity probe input mismatch')
            item = dict(model=model, **{name: row[name] for name in ('state_id', 'parent', 'patch', 'teacher_t')})
            for name in COSINES:
                p = row['parameter_gradients']['cosines'][name]['value']
                v = velocity['velocity_gradients']['cosines'][name]['value']
                item['parameter_cosine_'+name] = p
                item['velocity_cosine_'+name] = v
                item['sign_disagrees_'+name] = None if p is None or v is None else p*v < 0
            item['parameter_joint_over_FM'] = row['parameter_gradients']['joint_actual_over_FM']
            item['velocity_joint_over_FM'] = velocity['velocity_gradients']['joint_actual_over_FM']
            probes.append(item)
    generation = []
    for path in sorted((out/'scores').glob('*/results.json')):
        saved = _read(path, sources)
        if saved.get('status') != 'PASS':
            continue
        for row in saved.get('rows', []):
            if row.get('phase') == 'A' and row.get('model') in ('B', 'J'):
                generation.append(row)
    contrasts = []
    for precision in ('bf16', 'fp32'):
        for parent in (0, 1):
            for patch in range(5):
                teacher = [row for row in comparisons['J_minus_B_'+precision]['per_state']
                           if row['parent'] == parent and row['patch'] == patch]
                for nfe in (50, 100):
                    g = [row for row in generation if row.get('precision') == precision and row.get('nfe') == nfe
                         and row['parent'] == parent and row['patch'] == patch]
                    indexed = {(row['model'], row['seed']): row for row in g}
                    seeds = sorted({row['seed'] for row in g})
                    complete = len(g) == 4 and len(indexed) == 4 and len(seeds) == 2 and all((m, seed) in indexed for m in ('B', 'J') for seed in seeds)
                    contrasts.append(dict(parent=parent, patch=patch, precision=precision, nfe=nfe,
                        teacher_J_minus_B_by_time={str(t): {name: distribution([row['loss_'+name] for row in teacher if row['teacher_t'] == t])
                            for name in LOSSES} for t in (.1, .3, .5, .75, .9)},
                        generation_status='AVAILABLE' if complete else 'NOT_AVAILABLE_INCOMPLETE_SCORING',
                        generation_J_minus_B={name: distribution([indexed['J', seed]['scalars'].get(name)-indexed['B', seed]['scalars'].get(name)
                            for seed in seeds if indexed['J', seed]['scalars'].get(name) is not None and indexed['B', seed]['scalars'].get(name) is not None])
                            for name in GENERATION_METRICS} if complete else None,
                        note='Teacher noises and free-generation noises are independent; this is a context-level side-by-side comparison, not a noise pairing or causal test.'))
    result = dict(status='PASS', schema='jedge5_C_saved_analysis_v1', sources=sources,
        counts=dict(teacher_rows=len(flat), parameter_rows=len(probes), new_model_forwards=0, new_OT=0,
                    new_autograd=0, new_geometry_scores=0),
        limits=['Two known parents; all contexts observed before.',
                'patch0 is training C; patches1-4 were not fine-tuning contexts, not new shapes.',
                'Means across context groups give each recorded state equal weight; per-patch tables retained.',
                'Losses, gradient norms and angles have different meanings; no composite score.',
                'A single-layer probe does not establish whole-network or persistent training conflict.',
                'Negative paired differences are improvements only for the named lower-is-better loss/metric; norm or cosine sign is not an improvement criterion.',
                'B geometry loss is the same hypothetical J objective, not the objective used to train B.'],
        summary=summaries, paired=comparisons, one_layer_vs_velocity=probes,
        teacher_vs_generation=contrasts)
    folder=out/'stage_c'
    _csv(folder/'teacher_context_time.csv', flat)
    _csv(folder/'parameter_velocity.csv', probes)
    from .common import write_json
    write_json(folder/'analysis.json', result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args=parser.parse_args(argv)
    result=build_analysis(args.out)
    print(json.dumps(dict(status=result['status'], counts=result['counts'])))


if __name__ == '__main__':
    main()
