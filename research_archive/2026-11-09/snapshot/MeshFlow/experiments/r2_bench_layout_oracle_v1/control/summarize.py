"""Summarize complete saved results, training curves and paired P/NULL response.

No forward, model import, sampling, metric rescoring, rendering or visual labels.
The extra RAW displacement statistics are diagnostics, not success metrics.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from collections import defaultdict
import sys
import numpy as np
from audit_io import *

CONTRASTS = (
    ('EXPOSURE', 'PLAIN1000'), ('EXPOSURE', 'LAYOUT1000'), ('EXPOSURE', 'LAYOUT_NULL'),
    ('PLAIN1000', 'LAYOUT1000'), ('PLAIN1000', 'LAYOUT_NULL'), ('LAYOUT_NULL', 'LAYOUT1000'),
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def stat(values):
    x = np.asarray([v for v in values if numeric(v)], dtype=np.float64)
    if not len(x):
        return dict(n=0, mean=None, median=None, minimum=None, maximum=None)
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)),
                minimum=float(x.min()), maximum=float(x.max()))


def metric_tables(rows, metrics, requests):
    by = {metric_identity(r): r for r in rows}
    require(len(by) == len(rows) == 144, 'Complete 144 metric cells required')
    groups = defaultdict(list)
    for q in requests:
        groups[(q['report_group'], q['eval_group'], q['parent_id'])].append(q)
    paired = []
    for ref, candidate in CONTRASTS:
        for stage in STAGES:
            for (group, eval_group, parent), qs in sorted(groups.items()):
                row = dict(reference=ref, candidate=candidate, stage=stage, report_group=group,
                           eval_group=eval_group, parent_id=parent, requests=len(qs))
                for metric in metrics:
                    # Pair first; a missing value never borrows another output's denominator.
                    pairs = [(by[ref, q['request_id'], stage][metric], by[candidate, q['request_id'], stage][metric]) for q in qs]
                    valid = [(a, b) for a, b in pairs if numeric(a) and numeric(b)]
                    row[metric + '__paired_n'] = len(valid)
                    row[metric + '__reference'] = float(np.mean([a for a, b in valid])) if valid else None
                    row[metric + '__candidate'] = float(np.mean([b for a, b in valid])) if valid else None
                    row[metric + '__delta'] = float(np.mean([b - a for a, b in valid])) if valid else None
                paired.append(row)
    macro = []
    for ref, candidate in CONTRASTS:
        for stage in STAGES:
            for group in ('ALL', *sorted({q['report_group'] for q in requests})):
                selected = [r for r in paired if r['reference'] == ref and r['candidate'] == candidate and r['stage'] == stage and (group == 'ALL' or r['report_group'] == group)]
                row = dict(reference=ref, candidate=candidate, stage=stage, report_group=group, parents=len(selected))
                for metric in metrics:
                    values = stat([r[metric + '__delta'] for r in selected])
                    row[metric + '__parent_delta_mean'] = values['mean']
                    row[metric + '__valid_parents'] = values['n']
                macro.append(row)
    return paired, macro


def training_tables(audit):
    curves, windows, exposure = [], [], []
    for arm in ('PLAIN', 'LAYOUT'):
        folder = OUT / 'training' / arm
        state = audit.read(folder / 'status.json')
        require(state['status'] == 'COMPLETE' and state['step'] == 1000, 'Wait for complete training')
        log = audit.jsonl(folder / 'training_log.jsonl')
        require([r['step'] for r in log] == list(range(1, 1001)), 'Complete learning curve')
        for r in log:
            diagnostics = r['layout_diagnostics']
            curves.append({k: r[k] for k in ('arm', 'step', 'global_step', 'objective', 'gradient_norm',
                'denominator', 'layout_active_instances', 'input_seconds', 'update_seconds',
                'peak_allocated_bytes', 'peak_reserved_bytes')} | dict(
                layout_delta_rms_mean=stat([d['delta_rms'] for d in diagnostics])['mean'],
                layout_hidden_rms_mean=stat([d['hidden_rms'] for d in diagnostics])['mean'],
                layout_changed_scalars=sum(d['actual_changed_scalars'] for d in diagnostics) if diagnostics else None,
                layout_observed_microbatches=len(diagnostics)))
        for start in range(0, 1000, 50):
            block = log[start:start + 50]
            row = dict(arm=arm, first_step=start + 1, last_step=start + 50, updates=50)
            for key in ('objective', 'gradient_norm', 'input_seconds', 'update_seconds', 'layout_active_instances'):
                row.update({key + '__' + name: value for name, value in stat([r[key] for r in block]).items()})
            windows.append(row)
        grouped = defaultdict(list)
        for r in audit.jsonl(folder / 'exposure.jsonl'):
            grouped[r['parent_id'], r['condition_id'], r['replay_group']].append(r)
        for (parent, condition, group), current in sorted(grouped.items()):
            exposure.append(dict(arm=arm, parent_id=parent, condition_id=condition, replay_group=group,
                instances=len(current), layout_drop_instances=sum(r['layout_drop'] for r in current),
                mean_microbatch_loss=float(np.mean([r['loss'] for r in current])),
                mean_gain=float(np.mean([r['gain'] for r in current])), N_min=min(r['N'] for r in current),
                N_max=max(r['N'] for r in current), K_min=min(r['K'] for r in current), K_max=max(r['K'] for r in current)))
    return curves, windows, exposure


def load_npz(audit, path):
    audit.hash(path)
    with np.load(resolve(path), allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def displacement_stats(delta):
    distances = np.linalg.norm(np.asarray(delta, np.float64).reshape(-1, 3), axis=1)
    return dict(mean_distance_pctL=float(distances.mean() / L * 100),
                rms_distance_pctL=float(np.sqrt(np.mean(distances ** 2)) / L * 100),
                median_distance_pctL=float(np.median(distances) / L * 100),
                p90_distance_pctL=float(np.quantile(distances, .9) / L * 100),
                maximum_distance_pctL=float(distances.max() / L * 100),
                changed_corner_count=int(np.count_nonzero(distances)), corners=len(distances))


def raw_response(audit, requests):
    from scipy.spatial import cKDTree
    endpoints, steps, parent_pairs = [], [], []
    deltas = {}
    for q in requests:
        rid, k, n = q['request_id'], q['K'], q['N']
        arrays, trajectories = {}, {}
        for label in ('LAYOUT_NULL', 'LAYOUT1000'):
            folder = output_dir(label, rid, 'raw')
            receipt = audit.read(folder / 'receipt.json'); audit.verify(receipt['files'])
            require(receipt['noise_sha256'] == q['noise_sha256'], 'Fixed noise identity differs')
            arrays[label] = load_npz(audit, folder / 'raw.npz')
            trajectories[label] = load_npz(audit, folder / 'trajectory.npz')['state']
            require(trajectories[label].shape == (51, n, 3, 3), 'Saved complete Euler50 trajectory')
            require(np.isfinite(trajectories[label]).all(), 'Nonfinite saved trajectory')
            require(np.array_equal(arrays[label]['raw'], trajectories[label][-1]), 'RAW not final saved state')
            require(np.all(trajectories[label][:, :k] == trajectories[label][0, :k]), 'C changed within rollout')
        p, null = trajectories['LAYOUT1000'], trajectories['LAYOUT_NULL']
        require(np.array_equal(p[0], null[0]), 'P/NULL initial state differs')
        require(np.array_equal(p[:, :k], null[:, :k]), 'P/NULL C differs')
        for step in range(51):
            steps.append(dict(request_id=rid, parent_id=q['parent_id'], condition_id=q['condition_id'],
                noise_index=q['noise_index'], report_group=q['report_group'], euler_step=step, t=step / 50,
                **displacement_stats(p[step, k:] - null[step, k:])))
        x, y = p[-1, k:].astype(np.float64), null[-1, k:].astype(np.float64)
        delta = x - y; deltas[rid] = delta
        xx, yy = x.reshape(-1, 3), y.reshape(-1, 3)
        symmetric_corner_nn = .5 * (cKDTree(xx).query(yy, workers=1)[0].mean() + cKDTree(yy).query(xx, workers=1)[0].mean())
        row = dict(request_id=rid, parent_id=q['parent_id'], condition_id=q['condition_id'], noise_index=q['noise_index'],
            report_group=q['report_group'], N=n, K=k, initial_state_identical=True, C_identical_all51=True,
            comparison='same candidate weights; source P vs masked P; same noise',
            symmetric_corner_NN_pctL=float(symmetric_corner_nn / L * 100),
            diagnostic_scope='slotwise displacement is not semantic correspondence or completion success',
            **displacement_stats(delta))
        for axis, i in zip('xyz', range(3)):
            row['free_centroid_delta_' + axis + '_pctL'] = float((xx[:, i].mean() - yy[:, i].mean()) / L * 100)
            row['free_extent_delta_' + axis + '_pctL'] = float((np.ptp(xx[:, i]) - np.ptp(yy[:, i])) / L * 100)
        endpoints.append(row)
    grouped = defaultdict(list)
    for q in requests:
        grouped[q['parent_id'], q['condition_id']].append(q)
    for (parent, condition), current in sorted(grouped.items()):
        require(len(current) == 2, 'Expected two existing noises per condition')
        current.sort(key=lambda q: q['noise_index'])
        a, b = (deltas[q['request_id']].reshape(-1) for q in current)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        parent_pairs.append(dict(parent_id=parent, condition_id=condition, report_group=current[0]['report_group'],
            noise0_request=current[0]['request_id'], noise1_request=current[1]['request_id'],
            slot_delta_cosine=float(np.dot(a, b) / denom) if denom else None,
            response_noise0_rms_pctL=displacement_stats(a.reshape(-1, 3))['rms_distance_pctL'],
            response_noise1_rms_pctL=displacement_stats(b.reshape(-1, 3))['rms_distance_pctL'],
            interpretation='Directional response consistency only; not semantic correctness, diversity or generalization evidence'))
    return endpoints, steps, parent_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default=str(OUT / 'analysis/final_v1'))
    args = parser.parse_args()
    audit = FileAudit(); dest = resolve(args.output_dir)
    if (dest / 'complete.json').exists():
        done = read(dest / 'complete.json'); audit.verify(done['files']); audit.verify(done['protected'])
        print(json.dumps({k: v for k, v in done.items() if k not in ('files', 'protected')})); return
    require(not dest.exists(), 'Partial analysis directory: retain it and select a new output directory')
    panel = audit.read(OUT / 'data/eval_manifest.json'); requests, metrics = panel['requests'], panel['metric_fields']
    require(len(requests) == 12 and len(metrics) == 177, 'Registered panel and field count')
    complete = audit.read(OUT / 'results/completion.json')
    require(complete['status'] == 'COMPLETE', 'Wait for original score collection')
    audit.verify(complete['files']); audit.verify(complete['source_rows'])
    rows = audit.read(OUT / 'results/metric_rows.json')
    paired, macro = metric_tables(rows, metrics, requests)
    curves, windows, exposure = training_tables(audit)
    response, trajectories, noises = raw_response(audit, requests)
    tables = dict(paired_parent_metrics=paired, paired_parent_macro=macro, training_curve=curves,
                  training_windows50=windows, actual_exposure=exposure, raw_P_vs_NULL=response,
                  raw_P_vs_NULL_trajectory=trajectories, raw_response_two_noises=noises)
    for name, table in tables.items():
        csvwrite(dest / (name + '.csv'), table)
    interpretation = dict(status='COMPLETE', scope='Source-privileged P diagnostic, not C-only deployment',
        metrics='Original177 metrics retained without rescoring; deltas candidate minus reference, raw units',
        aggregation='First paired existing noises within parent, then parent-equal macro; NA never zero-filled',
        training='Actual microbatch loss is weighted and normalized contribution, not per-instance mean MSE',
        response='P-versus-NULL uses same LAYOUT1000 weights, C/N and exact saved initial arrays; slotwise displacement alone does not mean useful conditioning',
        visual='No visual labels generated; use separately recorded actual AI_PROXY observations',
        calls=dict(model=0, scoring=0, training=0, new_queries=0, new_noise=0))
    write(dest / 'definitions.json', interpretation)
    if 'torch' in sys.modules:
        raise RuntimeError('Summary imported torch')
    result = dict(schema='bench_layout_oracle_saved_result_summary_v1', status='COMPLETE',
                  row_counts={name: len(table) for name, table in tables.items()}, model_calls=0,
                  scoring_calls=0, ratings_generated=0, protected=audit.files,
                  files={str(p): sha(p) for p in dest.iterdir() if p.is_file()},
                  source_sha256=sha(__file__))
    write(dest / 'complete.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('files', 'protected')}))


if __name__ == '__main__':
    main()
