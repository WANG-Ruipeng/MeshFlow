"""Scientific figures for all task-aligned A attempts, with explicit policy scope.

This module reads completed results and training traces. It never selects a
checkpoint, filters by qualification, changes meshes, or creates model outputs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

MODELS = ('fixed', 'original_a', 'teacher_reference', 'student_10', 'student_11')
LABELS = {'fixed': 'Fixed', 'original_a': 'Frozen original A',
          'teacher_reference': 'Teacher (offline)', 'student_10': 'Student seed 10 (raw)',
          'student_11': 'Student seed 11 (raw)'}
COLORS = {'fixed': '#64748b', 'original_a': '#d97706', 'teacher_reference': '#7c3aed',
          'student_10': '#0284c7', 'student_11': '#059669', 'reference': '#c6d0de'}
MARKERS = {'fixed': 'o', 'original_a': 's', 'teacher_reference': 'D', 'student_10': '^', 'student_11': 'v'}
EXPECTED_POLICIES = {'fixed': ('raw',), 'original_a': ('raw', 'quality_fallback'),
                     'teacher_reference': ('offline_teacher',), 'student_10': ('raw', 'quality_fallback'),
                     'student_11': ('raw', 'quality_fallback')}


def finite(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def rms(row):
    metrics = row.get('metrics', [])
    return finite(metrics[0].get('rms_bbox_percent')) if metrics else None


def policy_for(model):
    return 'offline_teacher' if model == 'teacher_reference' else 'raw'


def point(ax, x, y, row, size=46):
    model = row['model']
    ax.scatter([x], [y], s=size, marker=MARKERS[model], edgecolors=COLORS[model],
               facecolors=COLORS[model] if row.get('qualified') is True else 'none',
               linewidths=1.15, alpha=.87, zorder=3)


def legend_handles(models=MODELS):
    handles = [Line2D([], [], linestyle='', marker=MARKERS[model], color=COLORS[model],
                      markersize=7, label=LABELS[model]) for model in models]
    handles.append(Line2D([], [], linestyle='', marker='o', markerfacecolor='none',
                          color='#111827', markersize=7, label='Hollow: strict gate not passed'))
    return handles


def save(fig, directory, name, footer):
    fig.text(.03, .018, footer, fontsize=8, va='bottom', linespacing=1.4)
    paths = []
    for extension in ('png', 'pdf'):
        path = directory / f'{name}.{extension}'
        fig.savefig(path, dpi=170, bbox_inches='tight')
        paths.append(str(path.resolve()))
    plt.close(fig)
    return paths


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Artifact path escapes run')
    return path


def load_mesh(path, reference=False):
    with np.load(path, allow_pickle=False) as z:
        v = np.asarray(z['ref_vertices' if reference else 'vertices'], np.float64)
        f = np.asarray(z['ref_faces' if reference else 'faces'], np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError('Invalid mesh dimensions')
    if not len(v) or not len(f) or not np.isfinite(v).all() or f.min() < 0 or f.max() >= len(v):
        raise ValueError('Empty or invalid mesh')
    return v, f


def render_report(run, *, overwrite=False, loss_only=False):
    root = Path(run).resolve()
    protocol = json.loads((root / 'RUN_PROTOCOL.json').read_text())
    document = json.loads((root / 'results.json').read_text())
    rows = document['rows']
    uids = sorted(protocol['uids'])
    if len(uids) != 16 or len(set(uids)) != 16:
        raise ValueError('Expected complete 16-parent protocol')
    train = set(protocol['train_uids'])
    check = set(protocol['internal_check_uids'])
    if train & check or train | check != set(uids):
        raise ValueError('Invalid split')
    index = {}
    for row in rows:
        model, policy, uid = row['model'], row['policy'], row['uid']
        if uid not in uids or model not in EXPECTED_POLICIES or policy not in EXPECTED_POLICIES[model]:
            raise ValueError('Unregistered result')
        key = (uid, model, policy)
        if key in index:
            raise ValueError('Duplicate result: ' + str(key))
        index[key] = row
    expected = [(uid, model, policy) for uid in uids for model in MODELS for policy in EXPECTED_POLICIES[model]]
    missing = [key for key in expected if key not in index]
    unreadable = [key for key, row in index.items() if rms(row) is None]
    counts = {}
    for model in MODELS:
        for policy in EXPECTED_POLICIES[model]:
            matching = [row for row in rows if row['model'] == model and row['policy'] == policy]
            counts[model + '/' + policy] = {
                'expected': 16, 'present': len(matching),
                'qualified': sum(row.get('qualified') is True for row in matching),
                'fallback_to_fixed': sum(row.get('fallback_to_fixed') is True for row in matching),
                'train_present': sum(row['uid'] in train for row in matching),
                'internal_check_present': sum(row['uid'] in check for row in matching),
            }
    directory = root / 'figures'
    directory.mkdir(exist_ok=overwrite or loss_only)
    artifacts = {}
    teacher_note = 'Teacher is an offline full-field reference, not a deployed predictor. No quality-qualified subset selection.'
    gate_note = 'Strict gate: exactly 512 faces + valid geometry + min angle and per-triangle angle p05 no worse than the same object\'s Fixed.'
    footer = (f'All 16 previously seen development objects: 12 train / 4 internal check (not an unseen confirmation set). Missing rows: {len(missing)}; unreadable RMS: {len(unreadable)}.\n'
              + gate_note + '\n' + teacher_note)

    # Both fixed training seeds, complete raw traces, no best-checkpoint selection.
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    trace_report = {}
    for seed in protocol['training']['seeds']:
        trace_path = root / 'training' / f'seed_{seed}' / 'TRAIN_TRACE.json'
        trace = json.loads(trace_path.read_text())
        updates = np.asarray([row['update'] for row in trace], int)
        losses = np.asarray([row['loss'] for row in trace], float)
        if len(updates) != protocol['training']['updates'] or not np.array_equal(updates, np.arange(1, len(updates) + 1)):
            raise ValueError('Training trace is incomplete or not contiguous')
        if not np.isfinite(losses).all() or np.any(losses <= 0):
            raise ValueError('Training loss invalid for logarithmic plot')
        color = COLORS[f'student_{seed}']
        ax.plot(updates, losses, color=color, alpha=.13, linewidth=.55)
        width = min(100, len(losses))
        means = np.convolve(losses, np.ones(width) / width, mode='valid')
        ax.plot(updates[width-1:], means, color=color, linewidth=1.7, label=f'Seed {seed}: {width}-update trailing mean')
        trace_report[str(seed)] = {'updates': int(updates[-1]), 'raw_points': len(updates),
                                   'smoothing': f'{width}-update trailing arithmetic mean',
                                   'first_loss': float(losses[0]), 'last_loss': float(losses[-1])}
    diagnostic = json.loads((root / 'FIT_DIAGNOSTIC.json').read_text())
    zero_rows = [row for row in diagnostic['velocity_summary']
                 if row['split'] == 'train' and row['model'] == 'zero_velocity']
    if len(zero_rows) != 1 or zero_rows[0]['objects'] != len(train):
        raise ValueError('Expected one all-training-parent zero-velocity diagnostic')
    zero_mse = finite(zero_rows[0]['same_grid_mse'])
    if zero_mse is None or zero_mse <= 0:
        raise ValueError('Invalid zero-velocity MSE reference')
    ax.axhline(zero_mse, color='#4b5563', linestyle='--', linewidth=1.5,
               label=f'Zero velocity: train offline 32-step grid (MSE={zero_mse:.3e})')
    loss_reference = {'source_file': 'FIT_DIAGNOSTIC.json', 'split': 'train',
                      'model': 'zero_velocity', 'same_grid_mse': zero_mse,
                      'objects': len(train), 'not_the_random_minibatch_curve_measurement': True,
                      'used_to_select_checkpoint': False}
    ax.set_yscale('log')
    ax.set_xlabel('Optimizer update')
    ax.set_ylabel('Discrete bridge velocity MSE (log scale)')
    ax.set_title('Task-aligned A training: both complete fixed-budget runs', pad=12)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=.2)
    fig.subplots_adjust(left=.09, right=.98, top=.90, bottom=.27)
    artifacts['loss'] = save(fig, directory, 'loss',
        'Faint curves: random mini-batch training losses. Thick curves: descriptive 100-update trailing means.\n'
        'Dashed reference: zero velocity evaluated offline on all 12 train parents and all 32 teacher-bridge states; a different measurement from the curves.\n'
        'Final update 3000 was predeclared. This offline reference is never used for checkpoint selection.\n'
        'Large loss reduction from original A does not by itself establish teacher fitting or deployed surface-quality improvement.')
    if loss_only:
        report_path = directory / 'REPORT.json'
        report = json.loads(report_path.read_text())
        report['artifacts']['loss'] = artifacts['loss']
        report['training_traces'] = trace_report
        report['loss_zero_velocity_reference'] = loss_reference
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        return report

    fig, ax = plt.subplots(figsize=(18, 8))
    ratio_exclusions = []
    paired_models = ('original_a', 'teacher_reference', 'student_10', 'student_11')
    offsets = np.linspace(-.27, .27, len(paired_models))
    for i, uid in enumerate(uids):
        if uid in check:
            ax.axvspan(i-.47, i+.47, color='#fef3c7', alpha=.65, zorder=0)
        baseline = rms(index.get((uid, 'fixed', 'raw'), {}))
        if baseline is None or baseline <= 0:
            ratio_exclusions.append({'uid': uid, 'reason': 'Fixed RMS missing/nonpositive'})
            continue
        for offset, model in zip(offsets, paired_models):
            row = index.get((uid, model, policy_for(model)), {})
            value = rms(row)
            if value is not None:
                point(ax, i + offset, value / baseline, row)
    ax.axhline(1.0, color='#475569', linestyle='--', linewidth=1, label='Fixed = 1')
    labels = [uid + ('\nINTERNAL CHECK' if uid in check else '\ntrain') for uid in uids]
    ax.set_xticks(range(len(uids)), labels, rotation=58, ha='right', fontsize=7)
    ax.set_xlim(-.65, len(uids)-.35)
    ax.set_ylabel('Surface RMS / same object\'s Fixed RMS (lower is better)')
    ax.set_title('All-parent paired surface error: raw deployed students and offline teacher', pad=14)
    ax.legend(handles=legend_handles(paired_models), loc='best', fontsize=9, frameon=False)
    ax.grid(axis='y', alpha=.2)
    fig.subplots_adjust(left=.065, right=.99, top=.91, bottom=.43)
    artifacts['paired_rms'] = save(fig, directory, 'paired_rms', footer + '\n'
        'First fresh evaluation seed only (192401 + sorted UID index), 8192 area samples per direction. Shaded columns: internal check. Fallback rows are accounted separately.')

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7), sharey=True)
    quality_exclusions = []
    for ax, field, label in zip(axes,
        ('min_triangle_angle_deg', 'triangle_min_angle_p05_deg'),
        ('Global minimum 3D triangle angle (degrees)', '5th percentile of per-triangle minimum 3D angles (degrees)')):
        for uid in uids:
            for model in MODELS:
                row = index.get((uid, model, policy_for(model)), {})
                x, y = finite(row.get('quality', {}).get(field)), rms(row)
                if x is not None and y is not None:
                    point(ax, x, y, row, size=37)
                else:
                    quality_exclusions.append({'uid': uid, 'model': model, 'field': field})
        ax.set_xlabel(label)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Same-field symmetric RMS (% reference bbox diagonal)')
    axes[0].legend(handles=legend_handles(), fontsize=8, frameon=False, loc='best')
    fig.suptitle('Raw output triangle quality and surface error: every object retained', y=.97, fontsize=13)
    fig.subplots_adjust(left=.07, right=.99, top=.89, bottom=.23, wspace=.10)
    artifacts['rms_quality'] = save(fig, directory, 'rms_quality', footer + '\n'
        'Quality floors differ across objects; no single vertical line represents qualification. RMS uses the first fresh evaluation seed; no fallback substitutions.')

    # Six columns: all actual endpoints, with a shared camera and scale per row.
    columns = ('reference',) + MODELS
    mesh_errors = []
    for page_start in range(0, len(uids), 4):
        page_uids = uids[page_start:page_start+4]
        fig = plt.figure(figsize=(19, 14.8))
        fig.suptitle(f'Actual meshes, page {page_start // 4 + 1}/4: shared view and scale within each object',
                     fontsize=15, y=.985)
        fig.text(.5, .962, 'Raw student outputs are displayed even when strict quality fails. Offline teacher has full-field optimization access.',
                 fontsize=10, ha='center')
        grid = fig.add_gridspec(4, 6, left=.04, right=.99, bottom=.115, top=.885, wspace=.03, hspace=.24)
        for col, model in enumerate(columns):
            x = .04 + (.99-.04)*(col+.5)/6
            fig.text(x, .912, 'Reference F4096' if model == 'reference' else LABELS[model],
                     ha='center', va='center', fontsize=10, weight='bold')
        for local_i, uid in enumerate(page_uids):
            meshes = {}
            try:
                meshes['reference'] = load_mesh(root / 'data' / (uid + '.npz'), reference=True)
            except Exception as exc:
                mesh_errors.append({'uid': uid, 'model': 'reference', 'error': repr(exc)})
            for model in MODELS:
                row = index.get((uid, model, policy_for(model)), {})
                if row.get('mesh_file'):
                    try:
                        meshes[model] = load_mesh(inside(root, row['mesh_file']))
                    except Exception as exc:
                        mesh_errors.append({'uid': uid, 'model': model, 'error': repr(exc)})
            if meshes:
                all_v = np.vstack([mesh[0] for mesh in meshes.values()])
                centre = (all_v.min(0) + all_v.max(0)) * .5
                radius = max(float(np.ptp(all_v, axis=0).max()) * .54, 1e-10)
            for col, model in enumerate(columns):
                ax = fig.add_subplot(grid[local_i, col], projection='3d')
                ax.set_axis_off()
                row = index.get((uid, model, policy_for(model)), {})
                if model in meshes:
                    v, f = meshes[model]
                    ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f,
                                    color=COLORS[model], edgecolor='#243040', linewidth=.14,
                                    antialiased=True, alpha=1, shade=True)
                    ax.set_xlim(centre[0]-radius, centre[0]+radius)
                    ax.set_ylim(centre[1]-radius, centre[1]+radius)
                    ax.set_zlim(centre[2]-radius, centre[2]+radius)
                    ax.set_box_aspect((1, 1, 1))
                    ax.view_init(elev=20, azim=-60, vertical_axis='y')
                    ax.set_proj_type('ortho')
                    if model == 'reference':
                        title = f'{uid[:16]}\n' + ('INTERNAL CHECK' if uid in check else 'train') + f'; F={len(f)}'
                        title_color = '#92400e' if uid in check else '#111827'
                    else:
                        err = rms(row)
                        angle = finite(row.get('quality', {}).get('triangle_min_angle_p05_deg'))
                        title = f'F={len(f)}; RMS={err:.4f}%' if err is not None else f'F={len(f)}; RMS unavailable'
                        title += f'; p05={angle:.1f} deg' if angle is not None else ''
                        title += '\n' + ('STRICT GATE PASS' if row.get('qualified') is True else 'STRICT GATE FAIL')
                        title_color = '#14532d' if row.get('qualified') is True else '#991b1b'
                    ax.set_title(title, fontsize=7.5, color=title_color, pad=-1)
                else:
                    ax.text2D(.04, .45, 'NO READABLE MESH', fontsize=9, transform=ax.transAxes)
                if col == 0:
                    ax.text2D(-.02, .02, uid, transform=ax.transAxes, fontsize=6)
        artifacts[f'meshes_{page_start // 4 + 1:02d}'] = save(fig, directory, f'meshes_{page_start // 4 + 1:02d}', footer + '\n'
            'Gate failure can be triangle quality, face budget or geometric validity; it is not synonymous with self-intersection. Quality fallback is not displayed here.')

    report = {'artifacts': artifacts, 'object_order': uids, 'counts': counts,
              'input_result_rows': len(rows), 'expected_result_rows': len(expected),
              'missing_records': missing, 'unreadable_rms': unreadable,
              'ratio_exclusions': ratio_exclusions, 'quality_exclusions': quality_exclusions,
              'mesh_read_errors': mesh_errors, 'training_traces': trace_report,
              'loss_zero_velocity_reference': loss_reference,
              'no_qualification_filter': True, 'mesh_policy': 'raw plus explicitly offline teacher',
              'fallback_rows_accounted_separately_in_counts': True,
              'evaluation_plot_seed': '192401 + sorted UID index',
              'second_fresh_evaluation_seed_not_plotted': '292401 + sorted UID index; present in raw ledger',
              'interpretation': 'Previously seen development pool; internal check is not confirmation-set or category generalization evidence'}
    (directory / 'REPORT.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--loss-only', action='store_true')
    args = parser.parse_args()
    report = render_report(args.run, overwrite=args.overwrite, loss_only=args.loss_only)
    print(json.dumps({'figures': len(report['artifacts']), 'rows': report['input_result_rows'],
                      'missing_records': len(report['missing_records']), 'mesh_read_errors': len(report['mesh_read_errors'])}))


if __name__ == '__main__':
    main()

