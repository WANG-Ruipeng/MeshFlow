"""Research figures from existing analysis CSVs; never rescore or load models.

Formal figures require all 288 registered cells and analysis.status COMPLETE.
Every displayed noise/parent value comes from the existing contrast tables;
repeated appearances of the same absolute value must agree exactly.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path

for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

DEFAULT_OUT = Path(__file__).resolve().parents[3] / 'native_t1/runs/chair_capped500_gradbudget_v1'
PANELS = ('OLD', 'FRESH')
LABELS = ('START', 'HIST_TEACHER500', 'HIST_CAPPED500', 'T500', 'T1000', 'L500', 'L1000', 'G500', 'G1000')
PRIMARY_LABELS = ('START', 'HIST_TEACHER500', 'HIST_CAPPED500', 'T500', 'L500', 'G500')
SHORT_LABELS = {'START': 'START\n5,000 total', 'HIST_TEACHER500': 'Historical\nteacher +500',
    'HIST_CAPPED500': 'Historical\ncapped +500', 'T500': 'New T\n+500', 'L500': 'New L\n+500', 'G500': 'New G\n+500'}
METRICS = ('SeamCov', 'ConflictCov_C', 'ExposedRetain', 'GT_to_F_MSE2048_L0', 'D_sym2048')
Y_LABELS = {'SeamCov': 'Seam coverage S (%)  higher is better',
    'ConflictCov_C': 'Conflict coverage I (%)  lower is better',
    'ExposedRetain': 'Exposed retention E (%)  higher is better',
    'GT_to_F_MSE2048_L0': 'GT-to-free MSE 2048 (x 1,000)  lower is better',
    'D_sym2048': 'Symmetric MSE 2048 (x 1,000)  lower is better'}
SCALE = {metric: (100. if metric in METRICS[:3] else 1000.) for metric in METRICS}
PARENT_COLORS = ('#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB', '#222222')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def number(value):
    if value is None or str(value).strip() in ('', 'NA', 'None', 'null'):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def atomic_json(path, obj):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def load_inputs(out):
    folder = Path(out) / 'analysis'
    filenames = ('results.json', 'checkpoint_profiles.csv', 'per_output.csv', 'per_parent.csv', 'summary.csv')
    payloads = {name: (folder / name).read_bytes() for name in filenames}
    result = json.loads(payloads['results.json'].decode('utf-8-sig'))
    tables = {name[:-4]: list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))
        for name, data in payloads.items() if name.endswith('.csv')}
    for name, required in {
        'checkpoint_profiles': {'label', 'panel', 'status', 'completed', 'expected_outputs', *METRICS},
        'per_output': {'panel', 'candidate_label', 'reference_label', 'metric', 'uid', 'candidate', 'candidate_value', 'reference_value'},
        'per_parent': {'panel', 'candidate_label', 'reference_label', 'metric', 'uid', 'registered_noises', 'candidate_value', 'reference_value'},
        'summary': {'panel', 'candidate_label', 'reference_label', 'metric', 'candidate_mean', 'reference_mean', 'status'},
    }.items():
        if not tables[name] or not required.issubset(tables[name][0]):
            raise ValueError('Missing required CSV fields: ' + name)
    for name in ('per_output', 'per_parent', 'summary'):
        if result['tables'][name]['sha256'] != digest(payloads[name + '.csv']):
            raise ValueError('Analysis CSV differs from its saved hash: ' + name)
    sources = {str(folder / name): {'sha256': digest(data), 'bytes': len(data)} for name, data in payloads.items()}
    return result, tables, sources


def extract(tables, result):
    noise, parent, profiles = {}, {}, {}
    duplicate_checks = 0
    for table_name, destination in (('per_output', noise), ('per_parent', parent)):
        for row in tables[table_name]:
            if row['panel'] not in PANELS or row['metric'] not in METRICS:
                continue
            for side in ('candidate', 'reference'):
                label = row[side + '_label']
                if label not in LABELS:
                    continue
                key = (row['panel'], label, row['metric'], row['uid'])
                if table_name == 'per_output':
                    key += (int(row['candidate']),)
                elif int(row['registered_noises']) != 2:
                    raise ValueError('Expected two registered noises per parent: ' + str(key))
                value = number(row[side + '_value'])
                if value is None:
                    raise ValueError('Missing/nonfinite required plotted value: ' + str(key))
                if key in destination:
                    duplicate_checks += 1
                    if destination[key].hex() != value.hex():
                        raise ValueError('Repeated absolute values disagree: ' + str(key))
                destination[key] = value
    saved_profiles = {(r['panel'], r['label']): r for r in result['checkpoint_profiles']}
    for row in tables['checkpoint_profiles']:
        if row['panel'] not in PANELS:
            continue
        key = (row['panel'], row['label'])
        if key in profiles:
            raise ValueError('Duplicate checkpoint profile: ' + str(key))
        if row['status'] != 'COMPLETE' or int(row['completed']) != 16 or int(row['expected_outputs']) != 16:
            raise ValueError('Incomplete checkpoint profile: ' + str(key))
        for metric in METRICS:
            value = number(row[metric])
            if value is None or value.hex() != float(saved_profiles[key][metric]).hex():
                raise ValueError('CSV/profile JSON mean mismatch: ' + str((key, metric)))
        profiles[key] = row
    expected_profiles = {(panel, label) for panel in PANELS for label in LABELS}
    if set(profiles) != expected_profiles:
        raise ValueError('Final figures require exactly 18 panel/checkpoint profiles = 288 cells')
    uids = sorted({key[3] for key in parent})
    if len(uids) != 8:
        raise ValueError('Expected all eight registered parents')
    for panel in PANELS:
        for label in LABELS:
            for metric in METRICS:
                pp = []
                for uid in uids:
                    pkey = (panel, label, metric, uid)
                    points = [v for k, v in noise.items() if k[:4] == pkey]
                    if pkey not in parent or len(points) != 2:
                        raise ValueError('Required parent/noise missing: ' + str(pkey))
                    if not math.isclose(sum(points)/2, parent[pkey], rel_tol=1e-13, abs_tol=1e-15):
                        raise ValueError('Noise mean disagrees with existing parent CSV: ' + str(pkey))
                    pp.append(parent[pkey])
                if not math.isclose(sum(pp)/8, float(profiles[(panel, label)][metric]), rel_tol=1e-13, abs_tol=1e-15):
                    raise ValueError('Parent means disagree with saved profile: ' + str((panel, label, metric)))
    return noise, parent, profiles, uids, duplicate_checks


def tools():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 11,
        'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    return plt, Line2D


def offset(index):
    return (index - 3.5) * .049


def plot_group(ax, panel, label, metric, x, data):
    noise, parent, profiles, uids = data
    scale = SCALE[metric]
    for i, uid in enumerate(uids):
        px = x + offset(i)
        points = sorted((k[-1], v) for k, v in noise.items() if k[:4] == (panel, label, metric, uid))
        ax.scatter([px-.018, px+.018], [v*scale for _, v in points], s=11, facecolors='none',
            edgecolors=PARENT_COLORS[i], linewidths=.7, alpha=.7, zorder=2)
        ax.scatter([px], [parent[(panel, label, metric, uid)]*scale], s=20,
            c=[PARENT_COLORS[i]], edgecolors='white', linewidths=.25, zorder=3)
    mean = float(profiles[(panel, label)][metric]) * scale
    ax.plot([x-.23, x+.23], [mean, mean], color='black', lw=1.5, zorder=4)
    ax.scatter([x], [mean], marker='D', color='black', s=26, zorder=5)
    if metric in METRICS[:3]:
        p = profiles[(panel, label)]
        low, high = number(p.get(metric + '_lower')), number(p.get(metric + '_upper'))
        if low is not None and high is not None:
            # Original profile bounds, never a confidence interval or new gate.
            ax.vlines(x, low*scale, high*scale, color='black', lw=1.8, zorder=4)
    return mean


def decorate(ax, metric):
    ax.set_ylabel(Y_LABELS[metric], fontsize=9)
    ax.grid(axis='y', alpha=.16)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='x', length=0)


def shared_y_range(axes, metric, labels, data):
    noise, parent, profiles, uids = data
    values = [v for k, v in noise.items() if k[1] in labels and k[2] == metric]
    if metric in METRICS[:3]:
        values += [number(profiles[(panel, label)].get(metric+'_upper')) for panel in PANELS for label in labels]
    maximum = max(v for v in values if v is not None) * SCALE[metric]
    for ax in axes:
        ax.set_ylim(0, maximum*1.065 if maximum > 0 else 1.)


def legend(fig, uids, line):
    handles = [line([0], [0], marker='o', color='none', markerfacecolor=PARENT_COLORS[i],
        markeredgecolor='white', markersize=5, label=uid[:6]) for i, uid in enumerate(uids)]
    handles.extend([line([0], [0], marker='o', color='none', markerfacecolor='none', markeredgecolor='#666666', markersize=4, label='Individual noise'),
        line([0], [0], marker='D', color='black', markersize=4, label='8-parent mean')])
    fig.legend(handles=handles, ncol=5, loc='upper center', bbox_to_anchor=(.5, .998), frameon=False, fontsize=8.5)


def figure_primary(dest, data):
    plt, line = tools()
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.4), sharey='row')
    for row, metric in enumerate(('SeamCov', 'GT_to_F_MSE2048_L0')):
        for col, panel in enumerate(PANELS):
            ax = axes[row, col]
            for x, label in enumerate(PRIMARY_LABELS):
                plot_group(ax, panel, label, metric, x, data)
            ax.set_xticks(range(6), [SHORT_LABELS[label] for label in PRIMARY_LABELS], fontsize=8.5)
            ax.set_xlim(-.5, 5.5)
            ax.axvline(2.5, color='#BBBBBB', lw=.8, ls='--')
            ax.set_title(panel + ' | 8 parents x 2 fixed noises')
            decorate(ax, metric)
        shared_y_range(axes[row, :], metric, PRIMARY_LABELS, data)
    legend(fig, data[3], line)
    fig.text(.5, .013, 'Filled color points: parent means; open points: both registered noises. Black diamonds/short bars: equal-parent means.\nHistorical teacher/capped are the earlier training-seed run. New T/L/G share this run\'s primary stream. No best-noise selection or new threshold.',
        ha='center', va='bottom', fontsize=8.5)
    fig.tight_layout(rect=(0, .08, 1, .915), w_pad=2.3, h_pad=2.1)
    return save_figure(fig, dest / 'primary500_control_and_shape', plt)


def figure_drift(dest, data):
    plt, line = tools()
    fig, axes = plt.subplots(5, 2, figsize=(13.4, 14.5), sharex=True, sharey='row')
    noise, parent, profiles, uids = data
    for row, metric in enumerate(METRICS):
        for col, panel in enumerate(PANELS):
            ax = axes[row, col]
            for group, arm in enumerate(('T', 'L', 'G')):
                left, right = group*3, group*3+1
                means = [plot_group(ax, panel, arm+str(step), metric, x, data) for step, x in ((500, left), (1000, right))]
                for i, uid in enumerate(uids):
                    values = [parent[(panel, arm+str(step), metric, uid)] * SCALE[metric] for step in (500, 1000)]
                    ax.plot([left+offset(i), right+offset(i)], values, color=PARENT_COLORS[i], lw=.65, alpha=.6, zorder=1)
                ax.plot([left, right], means, color='black', lw=1.5, ls='--', zorder=4)
            ax.set_xticks((0, 1, 3, 4, 6, 7), ('T\n500', 'T\n1000', 'L\n500', 'L\n1000', 'G\n500', 'G\n1000'))
            ax.set_xlim(-.5, 7.5)
            for x in (2, 5):
                ax.axvline(x, color='#DDDDDD', lw=.8)
            decorate(ax, metric)
            if row == 0:
                ax.set_title(panel + ' | 500 to 1000 updates')
        shared_y_range(axes[row, :], metric, tuple(a+str(step) for a in ('T','L','G') for step in (500,1000)), data)
    legend(fig, uids, line)
    fig.text(.5, .01, 'Thin colored lines connect the same parent; open points retain both noise outputs. Dashed black lines connect equal-parent means.\nS/I/E vertical black bounds are the original saved numerical bounds, not confidence intervals. All parent tails remain visible; no causal or promotion claim.',
        ha='center', va='bottom', fontsize=8.4)
    fig.tight_layout(rect=(0, .05, 1, .95), w_pad=2.3, h_pad=1.45)
    return save_figure(fig, dest / 'stability500_to1000', plt)


def save_figure(fig, base, plt):
    files = []
    for ext in ('png', 'svg'):
        path = base.with_suffix('.' + ext)
        fig.savefig(path, dpi=240, facecolor='white')
        data = path.read_bytes()
        files.append(dict(path=str(path), sha256=digest(data), bytes=len(data)))
    inches = list(fig.get_size_inches())
    plt.close(fig)
    return dict(files=files, inches=inches, PNG_dpi=240)


def run(out=DEFAULT_OUT, check_only=False):
    out = Path(out).resolve()
    result, tables, sources = load_inputs(out)
    readiness = dict(status='READY' if result['status'] == 'COMPLETE' else 'NOT_RUN_ANALYSIS_INCOMPLETE',
        analysis_status=result['status'], formal_figures_written=False, check_only=check_only,
        sources=sources, model_calls=0, scoring_calls=0, metric_recomputation=False,
        script_sha256=digest(Path(__file__).read_bytes()))
    if result['status'] != 'COMPLETE':
        return readiness
    noise, parent, profiles, uids, duplicates = extract(tables, result)
    readiness.update(absolute_noise_values=len(noise), parent_values=len(parent), duplicate_identity_checks=duplicates,
        profiles=len(profiles), registered_output_cells=18*16, parents=uids)
    if check_only:
        return readiness
    dest = out / 'report_figures'
    dest.mkdir(parents=True, exist_ok=True)
    data = noise, parent, profiles, uids
    figures = dict(primary500=figure_primary(dest, data), stability=figure_drift(dest, data))
    displayed = [dict(panel=k[0], label=k[1], metric=k[2], uid=k[3], candidate=k[4],
        value=v, parent_mean=parent[k[:4]], equal_parent_mean=float(profiles[k[:2]][k[2]]), display_scale=SCALE[k[2]])
        for k, v in sorted(noise.items())]
    with (dest / 'displayed_values.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(displayed[0]))
        writer.writeheader(); writer.writerows(displayed)
    readiness.update(status='COMPLETE', formal_figures_written=True, figures=figures,
        displayed_values_path=str(dest / 'displayed_values.csv'),
        notes=['All 288 cells remain represented; OLD and FRESH are never pooled in these figures.',
            'Only existing CSV values, arithmetic identity checks, and display scaling; no new scoring or statistical threshold.',
            'Parent means and both noise values are shown together. Mean markers do not replace tail observations.',
            '500 means 500 continuation updates beyond the fixed START. Historical models use the previous experiment seed.',
            'Original S/I/E profile bounds are numerical bounds, not inferential confidence intervals. E retains the original reported score; its bounds remain separately visible.'])
    atomic_json(dest / 'manifest.json', readiness)
    return readiness


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(DEFAULT_OUT))
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    report = run(args.out, args.check_only)
    print(json.dumps({k: v for k, v in report.items() if k not in ('sources', 'parents')}, indent=2))
