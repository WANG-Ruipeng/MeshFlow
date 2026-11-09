"""CPU-only fixed-case plots from saved arrays; no selection or mesh repair."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .analysis import FIXED_CASES, OLD_RUN, identity, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def checked_npz(path, sha256=None):
    path = Path(path)
    if sha256 is not None and hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise ValueError(f'Array hash mismatch: {path}')
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def _finish(fig, path, records, **metadata):
    fig.savefig(path, dpi=130, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)
    records.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **metadata))


def _plot_setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .2})
    return plt


def plot_curves(out, local_rows, suffix_rows=None):
    plt = _plot_setup()
    out = Path(out)
    figdir = out/'figures'
    figdir.mkdir(parents=True, exist_ok=True)
    records = []
    local_rows = [r for r in local_rows if 'delta_half' in r['local']]
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    for case, ax in zip(FIXED_CASES, axes.flat):
        for model, color in [('B', '#2673b8'), ('J', '#d26925')]:
            rs = sorted([r for r in local_rows if identity(r) == (model, *case)], key=lambda r: r['t'])
            ax.semilogy([r['t'] for r in rs], [r['local']['delta_h']['coordinate_rms_bbox'] for r in rs],
                        '-o', color=color, markersize=3, label=f'{model}: h=.01')
            ax.semilogy([r['t'] for r in rs], [r['local']['delta_half']['coordinate_rms_bbox'] for r in rs],
                        '--', color=color, alpha=.75, label=f'{model}: h=.005')
        ax.set(title=f'P{case[0]} patch{case[1]} seed{case[2]}', xlabel='Archived Euler100 time',
               ylabel='Local difference coordinate RMS / fixed GT bbox')
        ax.legend(fontsize=8)
    fig.suptitle('Same-start two-half-step minus one-step difference; diagnostic, not exact truncation error', fontsize=12)
    _finish(fig, figdir/'local_difference_by_t.png', records, cases=list(FIXED_CASES))

    fig, axes = plt.subplots(6, 2, figsize=(14, 22), constrained_layout=True)
    for row, case in enumerate(FIXED_CASES):
        for col, model in enumerate(('B', 'J')):
            ax = axes[row, col]
            rs = sorted([r for r in local_rows if identity(r) == (model, *case)], key=lambda r: r['t'])
            for term, style, color in [('space', '-', '#2673b8'), ('time', '--', '#d26925'),
                                       ('cross', ':', '#8643a1'), ('total', '-', '#222222')]:
                ax.semilogy([r['t'] for r in rs], [r['decomposition']['terms'][term]['coordinate_rms_bbox'] for r in rs],
                            style, color=color, label=term)
            ax.set(title=f'{model} P{case[0]} patch{case[1]} seed{case[2]}', xlabel='t',
                   ylabel='Velocity difference coordinate RMS / bbox')
            ax.legend(ncol=4, fontsize=7)
    fig.suptitle('State / explicit-time / interaction contributions; norms can cancel', fontsize=13)
    _finish(fig, figdir/'velocity_decomposition_by_t.png', records, cases=list(FIXED_CASES))
    if suffix_rows is None:
        return records

    fig, axes = plt.subplots(6, 2, figsize=(14, 22), constrained_layout=True)
    for row, case in enumerate(FIXED_CASES):
        for col, model in enumerate(('B', 'J')):
            ax = axes[row, col]
            rs = sorted([r for r in suffix_rows if identity(r) == (model, *case)], key=lambda r: r['t'])
            for r, color in zip(rs, ('#2673b8', '#d26925', '#398442')):
                growth = r['growth']
                g = [v for v in growth if v.get('coordinate_rms') is not None]
                bbox = r['propagation']['initial_difference']['bbox']
                ax.semilogy([v['t'] for v in g], [v['coordinate_rms']/bbox for v in g],
                            '-o', color=color, markersize=3, label=f"replace t={r['t']:.2f}")
            ax.set(title=f'{model} P{case[0]} patch{case[1]} seed{case[2]}', xlabel='Absolute continuation time',
                   ylabel='Difference from original coordinate RMS / bbox')
            ax.legend(fontsize=8)
    fig.suptitle('One local step replacement, then original Euler100 suffix; same model/context/noise', fontsize=13)
    _finish(fig, figdir/'suffix_growth_by_t.png', records, cases=list(FIXED_CASES))
    return records


def plot_teacher(out):
    plt = _plot_setup()
    out = Path(out)
    data = read_json(out/'analysis/teacher_reuse.json')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, precision in zip(axes, ('bf16', 'fp32')):
        for model, color in [('B', '#2673b8'), ('J', '#d26925')]:
            for group, style in [('training_C', '-'), ('other_observed_C', '--')]:
                rs = sorted([r for r in data['groups'] if r['model'] == model and r['precision'] == precision
                             and r['context_group'] == group], key=lambda r: r['teacher_t'])
                ax.plot([r['teacher_t'] for r in rs], [r['metrics']['joint_actual_over_FM']['mean'] for r in rs],
                        style+'o', color=color, label=f'{model} {group}')
        ax.set(title=precision.upper()+' archived teacher', xlabel='Teacher t',
               ylabel='Mean norm(weighted joint geometry gradient) / norm(FM gradient)')
        ax.legend(fontsize=8)
    fig.suptitle('Read-only previous teacher states; not free-trajectory states or a causal correlation', fontsize=11)
    records = []
    _finish(fig, out/'figures/teacher_archived_ratios.png', records, source_sha256=[r['sha256'] for r in data['sources']])
    return records


def plot_sensitivity(out):
    plt = _plot_setup()
    out = Path(out)
    rows = read_json(out/'raw/sensitivity_rows.json')['rows']
    fig, axes = plt.subplots(6, 2, figsize=(14, 20), constrained_layout=True)
    for row, case in enumerate(FIXED_CASES):
        for col, model in enumerate(('B', 'J')):
            ax = axes[row, col]
            for eta, color, style in ((1e-4, '#2673b8', '-o'), (5e-5, '#d26925', '--s')):
                selected = sorted([r for r in rows if identity(r) == (model, *case) and r['eta'] == eta], key=lambda r:r['t'])
                good = [r for r in selected if r.get('gain') is not None]
                ax.plot([r['t'] for r in good], [r['gain'] for r in good], style, color=color,
                        label=f'eta={eta:g}, resolved {len(good)}/3')
            ax.set(title=f'{model} P{case[0]} patch{case[1]} seed{case[2]}', xlabel='t',
                   ylabel='RMS(f+ - f-) / actual RMS(x+ - x-)')
            ax.legend(fontsize=8)
    fig.suptitle('Gain along the local-defect direction; neither full Jacobian norm nor a Lipschitz bound', fontsize=12)
    records = []
    _finish(fig, out/'figures/directional_sensitivity.png', records, cases=list(FIXED_CASES))
    return records


def original_endpoints(archive=OLD_RUN):
    archive = Path(archive)
    indexed = {}
    for model in ('B', 'J'):
        record = read_json(archive/'A'/model/'results.json')
        for r in record['rows']:
            case = (r['parent'], r['patch'], r['seed'])
            if case in FIXED_CASES and r['precision'] == 'fp32' and r['nfe'] == 100:
                key = (model, *case)
                if key in indexed:
                    raise ValueError('Duplicate original FP32 Euler100 endpoint')
                indexed[key] = dict(r, absolute_path=str(archive/r['path']))
    if len(indexed) != 12:
        raise ValueError('All twelve original endpoints are required')
    return indexed


def plot_endpoints(out, suffix_rows, original_rows=None):
    plt = _plot_setup()
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    out = Path(out)
    indexed = original_endpoints() if original_rows is None else {identity(r): r for r in original_rows}
    suffix = {(identity(r), float(r['t'])): r for r in suffix_rows}
    records = []
    for case in FIXED_CASES:
        entries = []
        sources = []
        for model in ('B', 'J'):
            original = indexed[(model, *case)]
            path = Path(original['absolute_path'])
            data = checked_npz(path, original.get('sha256'))
            raw = data['raw'].reshape(112, 3, 3)
            entries.append((model, 'Original Euler100', raw))
            sources.append(dict(model=model, condition='original', path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            for t in (.2, .5, .8):
                row = suffix[(model, *case), t]
                path = out/row['path']
                data = checked_npz(path, row.get('sha256'))
                triangles = data['raw'].reshape(112, 3, 3)
                if not np.array_equal(triangles[:12].view(np.int32), raw[:12].view(np.int32)):
                    raise ValueError('Endpoint panel found changed C bits')
                entries.append((model, f'Replace step at t={t:.2f}', triangles))
                sources.append(dict(model=model, condition=f'replace_{t}', path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        allpoints = np.concatenate([v.reshape(-1, 3) for _, _, v in entries])[:, [0, 2, 1]]
        center = (allpoints.min(0)+allpoints.max(0))*.5
        radius = float(np.ptp(allpoints, axis=0).max())*.54
        if radius <= 0:
            raise ValueError('No endpoint extent')
        fig = plt.figure(figsize=(14, 13), layout='constrained')
        for i, (model, label, raw) in enumerate(entries):
            model_row, col = divmod(i, 4)
            triangles = raw[:, :, [0, 2, 1]]
            for part in range(2):
                ax = fig.add_subplot(4, 4, (model_row*2+part)*4+col+1, projection='3d')
                ax.add_collection3d(Poly3DCollection(triangles[12:], facecolors='#91a9c2', edgecolors='#30445a',
                                                    linewidths=.35, alpha=.88))
                if part == 0:
                    ax.add_collection3d(Poly3DCollection(triangles[:12], facecolors='#f6aa3d', edgecolors='#a45309',
                                                        linewidths=.65, alpha=.96))
                ax.set_xlim(center[0]-radius, center[0]+radius)
                ax.set_ylim(center[1]-radius, center[1]+radius)
                ax.set_zlim(center[2]-radius, center[2]+radius)
                ax.set_box_aspect((1, 1, 1))
                ax.view_init(elev=24, azim=-54)
                ax.set_axis_off()
                ax.set_title(f'{model}: {label}\n'+('full, orange=context' if part == 0 else 'free only'), fontsize=9)
        fig.suptitle(f'P{case[0]} patch{case[1]} seed{case[2]} | FP32 Euler100, fixed view and bounds, no postprocessing', fontsize=12)
        _finish(fig, out/'figures'/f'endpoints_P{case[0]}_patch{case[1]}_seed{case[2]}.png', records,
                case=list(case), bounds_center=center.tolist(), radius=radius, elevation=24, azimuth=-54,
                no_case_selection=True, no_postprocessing=True, sources=sources)
    return records


def run(out):
    out = Path(out)
    local = read_json(out/'raw/local_rows.json')
    suffix = read_json(out/'raw/suffix_rows.json')['rows']
    records = plot_curves(out, local['rows'], suffix)
    records.extend(plot_teacher(out))
    if (out/'raw/sensitivity_rows.json').exists():
        records.extend(plot_sensitivity(out))
    protocol = read_json(out/'registration/protocol.json')
    original_rows = [dict(r, absolute_path=r['path']) for r in protocol['trajectories']]
    records.extend(plot_endpoints(out, suffix, original_rows=original_rows))
    write_json(out/'figures/manifest.json', dict(status='PASS', figures=records,
        all_six_fixed_cases=True, model_calls=0, geometry_evaluations=0, postprocessing=0,
        archive_teacher_is_different_state_set=True))
    return records


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    a = p.parse_args()
    run(a.out)
