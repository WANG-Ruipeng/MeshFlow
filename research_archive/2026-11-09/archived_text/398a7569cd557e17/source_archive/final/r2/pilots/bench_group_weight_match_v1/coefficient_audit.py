"""Independent, standard-library-only B/W stream and coefficient audit.

Reads frozen manifests, schedules and existing exposure/update JSON ledgers.
Never imports torch, a trainer, an OT implementation or a model. Writes only
this experiment's coefficient_audit directory; source artifacts stay read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_bench_group_weight_match_v1'
DEST = OUT / 'coefficient_audit'
SOURCES = {
    'B': ('r2_bench_natural_n_recipe_v1', 'NATURAL_HIGH', 'bench_natural_n_recipe_v1'),
    'W': ('r2_bench_high_joint_coverage_v1', 'WIDE_HIGH', 'bench_high_joint_coverage_v1'),
}
GROUPS = ('OLD', 'FOCUS', 'HIGH')
RANDOM_FIELDS = ('alpha', 'scale_xyz', 'augmentation', 'noise_seed',
                 'epsilon_seed', 'time_seed', 'permutation_seed')
COEFFICIENT_FIELDS = ('stage_step', 'branch', 'm_old', 'm_focus', 'm_high_B',
    'm_high_W', 'D_B', 'D_W', 'rho', 'base_gain', 'micro_gains', 'micro_groups')
LIMITS = [
    'Counts are reconstructed from frozen N/K and independently checked against existing actual mask/exposure counts; case tensors and new masks are not loaded.',
    'The algebra restores B per-update coefficients on group coordinate-mean MSE, not B HIGH examples or within-HIGH parent weights.',
    'Same OLD/FOCUS stream and scalar coefficients do not restore their later errors or gradients once model trajectories diverge.',
    'Different N, geometry, OT assignments, error values, gradient directions and AdamW states remain different; equal seeds do not imply equal-sized random arrays.',
    'Step-mean coefficients and pooled coordinate fractions are distinct statistics; neither is a gradient-norm share or a generation-quality metric.',
]


def require(test, message):
    if not test:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def lines(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return [json.loads(s) for s in f if s.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT)).replace('\\', '/')


def resolve(path):
    value = str(path).replace('\\', '/')
    for prefix in ('E:/MeshFlow/', '/mnt/e/MeshFlow/'):
        if value.startswith(prefix):
            return ROOT / value[len(prefix):]
    return Path(value).resolve()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'


def write_once(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_text(encoding='utf-8') == content, 'Refusing changed audit artifact: ' + str(path))
    else:
        with path.open('x', encoding='utf-8', newline='') as f:
            f.write(content)


def csv_text(rows):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, separators=(',', ':')) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()})
    return buffer.getvalue()


def load_source(label):
    experiment, arm, module = SOURCES[label]
    base = ROOT / 'experiments' / experiment
    paths = {
        'registration': base / 'training_registration.json',
        'manifest': base / 'data/manifest.json',
        'schedule': base / ('data/schedule_' + arm + '.json'),
        'exposure': base / ('training/' + arm + '/exposure.jsonl'),
        'updates': base / ('training/' + arm + '/training_log.jsonl'),
    }
    reg, manifest, schedule = (read(paths[k]) for k in ('registration', 'manifest', 'schedule'))
    require(reg['status'] == 'FROZEN' and reg['updates'] == 1500, label + ' registration differs')
    require((reg['batch'], reg['microbatch'], reg['loss_normalization']) ==
            (8, 1, 'whole_effective_batch_free_coordinates'), label + ' normalization differs')
    require(sha(paths['manifest']) == reg['manifest_sha256'], label + ' manifest hash differs')
    require(sha(paths['schedule']) == reg['schedules'][arm]['sha256'], label + ' schedule hash differs')
    require(resolve(reg['schedules'][arm]['path']) == paths['schedule'], label + ' schedule path differs')
    require(schedule['arm'] == arm and len(schedule['steps']) == 1500 and schedule['batch'] == 8,
            label + ' schedule budget differs')
    source_code = [ROOT / 'r2/pilots' / module / name for name in ('stream.py', 'training.py', 'runtime.py')]
    source_code += [ROOT / 'r2/pilots/bench_natural_n_recipe_v1/runtime.py',
                    ROOT / 'r2/src/meshflow_control/seam/model_adapter.py']
    registered_code = {resolve(k): v for k, v in reg['training_code_hashes'].items()}
    for code in source_code:
        require(code in registered_code and sha(code) == registered_code[code], label + ' frozen mask/loss/stream source changed: ' + str(code))
    exposures, updates = lines(paths['exposure']), lines(paths['updates'])
    require(len(exposures) == 12000 and len(updates) == 1500, label + ' actual ledger length differs')
    train = manifest['train']
    require(len(train) == (26 if label == 'B' else 30), label + ' parent pool differs')
    groups = Counter(r['replay_group'] for r in train)
    require(groups == {'OLD': 22, 'FOCUS': 2, 'HIGH': 2 if label == 'B' else 6}, label + ' groups differ')
    counts, exposure_groups = [], Counter()
    for i, step in enumerate(schedule['steps']):
        number = i + 1
        require((step['step'], step['stage_step'], step['stage']) == (number, number, 'NATIVE'), label + ' stage order differs')
        batch = step['samples']
        expected_groups = ['OLD'] * 6 + ['FOCUS' if number % 2 else 'HIGH'] * 2
        require(len(batch) == 8 and [s['replay_group'] for s in batch] == expected_groups, label + ' slot groups differ')
        batch_counts = []
        for j, sample in enumerate(batch):
            source = train[sample['index']]
            exposure = exposures[i * 8 + j]
            require(sample['index'] == sample['condition_index'] and sample['sample_index'] == j,
                    label + ' sample indexing differs')
            require(all(source[k] == sample[k] for k in ('condition_id', 'parent_id', 'replay_group')),
                    label + ' source identity differs')
            require(all(exposure[k] == v for k, v in sample.items()), label + ' actual stream differs from frozen row')
            n, k = int(source['N']), int(source['K'])
            count = 9 * (n - k)
            require(128 <= n <= 799 and 2 <= k < n, label + ' N/K contract differs')
            require((exposure['N'], exposure['K'], exposure['all_free_scalar_count'],
                     exposure['bulk_scalar_count'], exposure['seam_tip_scalar_count']) == (n, k, count, count, 0),
                    label + ' observed effective mask scalar count differs')
            require((exposure['local_step'], exposure['stage_step'], exposure['microbatch'], exposure['arm']) ==
                    (number, number, j, arm), label + ' exposure order differs')
            mechanism = exposure['mechanism']
            require(mechanism['free_coordinate_count'] == count and mechanism['active_seam_slots'] == 0
                    and mechanism['policy'] == 'NATIVE_ALL_FREE_OT' and exposure['head_forward_called'] is False,
                    label + ' mask mechanism differs')
            require(math.isfinite(exposure['t']) and 0 <= exposure['t'] <= 1, label + ' invalid realized t')
            batch_counts.append(count)
            exposure_groups[sample['replay_group']] += 1
        update = updates[i]
        denominator = sum(batch_counts)
        require((update['local_step'], update['stage_step'], update['arm']) == (number, number, arm), label + ' update order differs')
        require(update['free_coordinate_denominator'] == denominator == update['all_free_scalar_count'],
                label + ' actual update denominator differs')
        counts.append(batch_counts)
    require(exposure_groups == {'OLD': 9000, 'FOCUS': 1500, 'HIGH': 1500}, label + ' exposure totals differ')
    inputs = {rel(p): sha(p) for p in list(paths.values()) + source_code}
    return dict(schedule=schedule, manifest=manifest, counts=counts, exposures=exposures,
                input_hashes=inputs, arm=arm)


def build_reference():
    b, w = load_source('B'), load_source('W')
    require(b['schedule']['seed'] == w['schedule']['seed'], 'Schedule seed differs')
    pairing = Counter()
    rows, instance_rows = [], []
    for i, (sb, sw) in enumerate(zip(b['schedule']['steps'], w['schedule']['steps'])):
        number = i + 1
        require({k: v for k, v in sb.items() if k != 'samples'} ==
                {k: v for k, v in sw.items() if k != 'samples'}, 'Step metadata differs')
        for j, (rb, rw) in enumerate(zip(sb['samples'], sw['samples'])):
            group = rb['replay_group']
            require(group == rw['replay_group'], 'Group position differs')
            require(all(rb[k] == rw[k] for k in RANDOM_FIELDS), 'Paired RNG/augmentation differs')
            require(b['exposures'][i * 8 + j]['t'] == w['exposures'][i * 8 + j]['t'], 'Realized t differs')
            if group != 'HIGH':
                require(rb == rw, 'OLD/FOCUS frozen rows differ')
                left, right = b['manifest']['train'][rb['index']], w['manifest']['train'][rw['index']]
                require(all(left[k] == right[k] for k in ('N', 'K', 'case_sha256', 'condition_id', 'parent_id')),
                        'OLD/FOCUS source case identity differs')
                pairing['exact_' + group.lower() + '_rows'] += 1
            else:
                pairing['high_random_rows_equal'] += 1
                pairing['high_same_parent_rows' if rb['parent_id'] == rw['parent_id'] else 'high_changed_parent_rows'] += 1
            pairing['realized_t_equal'] += 1
        cb, cw = b['counts'][i], w['counts'][i]
        mo, mf = sum(cb[:6]), sum(cb[6:]) if number % 2 else 0
        require(mo == sum(cw[:6]), 'OLD effective counts differ')
        mhb, mhw = (0, 0) if number % 2 else (sum(cb[6:]), sum(cw[6:]))
        db, dw = sum(cb), sum(cw)
        branch = 'FOCUS' if number % 2 else 'HIGH'
        if branch == 'FOCUS':
            require(cb == cw and db == dw, 'FOCUS update counts differ')
        rho = Fraction(mhb, mhw) if mhw else None
        base = Fraction(dw, db)
        gains = [base * (rho if j >= 6 and rho is not None else 1) for j in range(8)]
        require(sum(Fraction(count, dw) * gain for count, gain in zip(cw, gains)) == 1,
                'Matched effective weights do not sum to one')
        require(sum(Fraction(cw[j], dw) * gains[j] for j in range(6)) == Fraction(mo, db), 'OLD coefficient algebra differs')
        require(sum(Fraction(cw[j], dw) * gains[j] for j in (6, 7)) == Fraction(mf + mhb, db), 'Fourth-slot coefficient algebra differs')
        row = dict(stage_step=number, branch=branch, m_old=mo, m_focus=mf, m_high_B=mhb,
            m_high_W=mhw, D_B=db, D_W=dw, rho=float(rho) if rho is not None else None,
            base_gain=float(base), micro_gains=[float(g) for g in gains],
            micro_groups=[r['replay_group'] for r in sw['samples']],
            micro_free_scalars_B=cb, micro_free_scalars_W=cw,
            q_high_B=mhb / db, q_high_W=mhw / dw, q_high_matched=mhb / db,
            q_old_B=mo / db, q_old_W=mo / dw, q_old_matched=mo / db,
            q_focus_B=mf / db, q_focus_W=mf / dw, q_focus_matched=mf / db,
            high_parent_B=sb['samples'][6]['parent_id'] if mhb else None,
            high_parent_W=sw['samples'][6]['parent_id'] if mhw else None,
            exact_rational_group_identity=True)
        rows.append(row)
        for j, (rb, rw) in enumerate(zip(sb['samples'], sw['samples'])):
            instance_rows.append(dict(stage_step=number, sample_index=j, group=rw['replay_group'],
                parent_B=rb['parent_id'], parent_W=rw['parent_id'],
                condition_B=rb['condition_id'], condition_W=rw['condition_id'],
                free_scalars_B=cb[j], free_scalars_W=cw[j],
                realized_t=b['exposures'][i * 8 + j]['t'],
                frozen_rng_equal=True, actual_t_equal=True,
                source_case_equal=group_case_equal(b, w, rb, rw)))
    groups = []
    for group in GROUPS:
        suffix = group.lower()
        kb = 'm_' + ('high_B' if group == 'HIGH' else suffix)
        kw = 'm_' + ('high_W' if group == 'HIGH' else suffix)
        observed = [r for r in rows if r[kb]]
        groups.append(dict(group=group, instances={'OLD': 9000, 'FOCUS': 1500, 'HIGH': 1500}[group],
            updates_with_group=len(observed), total_free_scalars_B=sum(r[kb] for r in rows),
            total_free_scalars_W=sum(r[kw] for r in rows),
            mean_update_coefficient_B=statistics.mean(r['q_' + suffix + '_B'] for r in rows),
            mean_update_coefficient_W=statistics.mean(r['q_' + suffix + '_W'] for r in rows),
            mean_update_coefficient_matched=statistics.mean(r['q_' + suffix + '_matched'] for r in rows),
            mean_present_update_coefficient_B=statistics.mean(r['q_' + suffix + '_B'] for r in observed),
            mean_present_update_coefficient_W=statistics.mean(r['q_' + suffix + '_W'] for r in observed),
            pooled_actual_coordinate_fraction_B=sum(r[kb] for r in rows) / sum(r['D_B'] for r in rows),
            pooled_actual_coordinate_fraction_W=sum(r[kw] for r in rows) / sum(r['D_W'] for r in rows),
            pooled_actual_coordinate_fraction_matched=sum(r[kw] for r in rows) / sum(r['D_W'] for r in rows),
            pooled_reweighted_coordinate_fraction_matched=sum(r[kb] for r in rows) / sum(r['D_B'] for r in rows)))
    sources = dict(sorted({**b['input_hashes'], **w['input_hashes']}.items()))
    reference = dict(schema='BENCH_GROUP_WEIGHT_MATCH_INDEPENDENT_REFERENCE_V1',
        status='PASS', updates=1500, instances=12000, source_hashes=sources,
        definitions=dict(m='effective free scalar coordinates, 9*(N-K)',
            D_B='m_old+m_focus+m_high_B', D_W='m_old+m_focus+m_high_W',
            rho='m_high_B/m_high_W on HIGH steps; null on FOCUS steps',
            q_high_B='m_high_B/D_B', q_high_W='m_high_W/D_W',
            base_gain='D_W/D_B', micro_gain='base_gain*rho for HIGH; base_gain otherwise',
            matched_loss='(SSE_OLD+rho*SSE_HIGH)/D_B on HIGH steps; original loss on FOCUS steps'),
        pairing=dict(pairing), steps=rows)
    summary = dict(status='PASS', updates=1500, focus_updates=750, high_updates=750,
        pairing=dict(pairing), groups=groups,
        rho_on_high=dict(min=min(r['rho'] for r in rows if r['rho'] is not None),
                         max=max(r['rho'] for r in rows if r['rho'] is not None),
                         mean=statistics.mean(r['rho'] for r in rows if r['rho'] is not None)),
        limits=LIMITS, model_loads=0, model_forwards=0, new_OT_calls=0, tensor_loads=0)
    return reference, summary, instance_rows


def group_case_equal(b, w, rb, rw):
    return b['manifest']['train'][rb['index']]['case_sha256'] == w['manifest']['train'][rw['index']]['case_sha256']


def verify_candidate(reference, path):
    candidate = read(path)
    require(len(candidate['steps']) == 1500, 'Candidate coefficient length differs')
    maximum = 0.0
    for expected, actual in zip(reference['steps'], candidate['steps']):
        for key in COEFFICIENT_FIELDS:
            left, right = expected[key], actual[key]
            pairs = zip(left, right) if key == 'micro_gains' else [(left, right)]
            if key == 'micro_gains':
                require(len(right) == 8, 'Candidate micro gain count differs')
            for a, b in pairs:
                if isinstance(a, float):
                    require(isinstance(b, (int, float)) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-14, abs_tol=2e-15),
                            f'Candidate coefficient differs at {expected["stage_step"]}/{key}')
                    maximum = max(maximum, abs(a - b))
                else:
                    require(a == b, f'Candidate count/identity differs at {expected["stage_step"]}/{key}')
    return dict(status='PASS', path=rel(path), sha256=sha(path), checked_updates=1500,
                checked_fields=list(COEFFICIENT_FIELDS), maximum_absolute_float_difference=maximum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', help='Compare independent result to prepared trainer coefficient_schedule.json and seal complete.json')
    args = parser.parse_args()
    reference, summary, instances = build_reference()
    products = {
        'reference_coefficients.json': encoded(reference),
        'per_update_coefficients.csv': csv_text(reference['steps']),
        'paired_instances.csv': csv_text(instances),
        'group_summary.csv': csv_text(summary['groups']),
        'summary.json': encoded(summary),
    }
    for name, content in products.items():
        write_once(DEST / name, content)
    result = dict(status='REFERENCE_READY', updates=1500, instances=12000,
        source_hashes=reference['source_hashes'], audit_code={rel(__file__): sha(__file__)},
        artifact_hashes={rel(DEST / name): sha(DEST / name) for name in products},
        pairing=reference['pairing'], limits=LIMITS,
        model_loads=0, model_forwards=0, new_OT_calls=0, tensor_loads=0)
    if args.candidate:
        result['candidate_comparison'] = verify_candidate(reference, resolve(args.candidate))
        result['status'] = 'PASS'
        write_once(DEST / 'complete.json', encoded(result))
    else:
        write_once(DEST / 'reference_complete.json', encoded(result))
    print(json.dumps(dict(status=result['status'], output=str(DEST), pairing=result['pairing'],
                         groups=summary['groups']), indent=2))


if __name__ == '__main__':
    main()
