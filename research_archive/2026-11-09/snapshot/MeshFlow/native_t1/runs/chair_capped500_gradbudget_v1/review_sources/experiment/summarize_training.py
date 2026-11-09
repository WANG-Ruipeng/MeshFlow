"""CPU-only descriptive summaries of recorded gradient-budget training.

Final mode requires the strict final training audit to pass. Live mode writes
only training_diagnostics/live and identifies the audited prefix separately.
No model, checkpoint-tensor, optimizer, sampling, or scorer imports.
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import time

for _env in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_env] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

ARMS = ('T', 'L', 'G')
NAMES = {'T': 'Teacher LCAP', 'L': 'Capped replay', 'G': 'Gradient budget'}
COLORS = {'T': '#3978A8', 'L': '#D18428', 'G': '#368C67'}
DEFAULT_OUT = Path(__file__).resolve().parents[3] / 'native_t1/runs/chair_capped500_gradbudget_v1'


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def average(values):
    v = [float(x) for x in values if x is not None]
    return statistics.mean(v) if v else None


def middle(values):
    v = [float(x) for x in values if x is not None]
    return statistics.median(v) if v else None


def extreme(values, fn):
    v = [float(x) for x in values if x is not None]
    return fn(v) if v else None


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class Inputs:
    def __init__(self, live):
        self.live = live
        self.sources = {}
        self.missing = []
        self.partial_tails = []

    def read(self, path, kind='json', required=True):
        path = Path(path)
        if not path.exists():
            if required:
                self.missing.append(str(path))
            return None if kind == 'json' else []
        data = path.read_bytes()
        if kind == 'jsonl' and data and not data.endswith(b'\n'):
            if not self.live:
                raise ValueError('Unterminated final training log: ' + str(path))
            end = data.rfind(b'\n') + 1
            self.partial_tails.append(dict(path=str(path),ignored_bytes=len(data)-end))
            data = data[:end]
        self.sources[str(path)] = dict(sha256=digest_bytes(data), bytes=len(data), scope='Complete read snapshot, or complete-line prefix if live')
        text = data.decode('utf-8-sig')
        if kind == 'csv':
            return list(csv.DictReader(io.StringIO(text)))
        if kind == 'jsonl':
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        return json.loads(text)


def per_step(arm, row):
    nf, na = float(row['gF_norm']), float(row['gA_norm'])
    a = row['aux_audit']
    return dict(arm=arm, step=row['pilot_step'], phase='1-500' if row['pilot_step'] <= 500 else '501-1000',
        FM=row['FM'], raw_aux=row['aux_unweighted'], weighted_aux_loss_diagnostic=row['weighted_auxiliary_loss_value'],
        lambda_loss=row['lambda_loss'], actual_lambda=row['actual_lambda'], lambda_grad=row['lambda_grad'],
        loss_cap_active=row['loss_cap_active'], gradient_cap_active=row['gradient_cap_active'],
        gF_norm=nf, raw_gA_norm=na, actual_weighted_aux_norm=row['weighted_aux_norm'],
        unweighted_aux_FM_norm_ratio=na/nf if nf else None,
        actual_weighted_aux_FM_norm_ratio=row['weighted_aux_over_FM_norm'], cosine=row['cosine'],
        preclip_norm=row['combination_preclip_norm'], torch_preclip_norm=row['grad_norm_before_clip'],
        clip_coefficient=row['clip_coefficient'], clip_applied=row['clip_coefficient'] < 1.,
        norm_above_one=row['grad_norm_before_clip'] > 1.,
        primary_buffer_bytes=row['primary_buffer_bytes'], primary_capture_seconds=row['capture_seconds'],
        scalar_reduction_seconds=row['scalar_reduction_seconds'], combination_seconds=row['combination_seconds'],
        carrier_seconds=row['carrier_seconds'], update_seconds=row['update_total_seconds'], input_seconds=row['OT_and_input_seconds'],
        actual_FP32_budget_excess=row['actual_FP32_budget_excess'], theoretical_budget_excess=row['theoretical_budget_excess'],
        queue=a['queue'], uid=a['uid'], request_id=a['request_id'], t=a['t'], state_index=a['state_index'],
        target_sha256=a['target_sha256'], xt_sha256=a['xt_sha256'], u_sha256=a['u_sha256'])


def phase_summary(arm, phase, rows, updates):
    n = len(rows)
    result = dict(arm=arm, recipe=NAMES[arm], phase=phase, completed_steps=n,
        expected_steps=500, phase_complete=n == 500,
        unique_auxiliary_requests=len({r['request_id'] for r in rows}),
        unique_auxiliary_states=len({(r['request_id'], r['state_index']) for r in rows}),
        loss_cap_count=sum(r['loss_cap_active'] for r in rows),
        gradient_cap_count=sum(r['gradient_cap_active'] for r in rows),
        total_clip_count=sum(r['clip_applied'] for r in rows),
        norm_above_one_count=sum(r['norm_above_one'] for r in rows),
        teacher_actual_loss_cap_count=sum(r['loss_cap_active'] for r in rows) if arm == 'T' else None,
        loss_cap_fraction=average(r['loss_cap_active'] for r in rows),
        gradient_cap_fraction=average(r['gradient_cap_active'] for r in rows),
        total_clip_fraction=average(r['clip_applied'] for r in rows),
        negative_cosine_fraction=average(r['cosine'] < 0 for r in rows if r['cosine'] is not None),
        cosine_defined_steps=sum(r['cosine'] is not None for r in rows))
    for key in ('FM', 'raw_aux', 'weighted_aux_loss_diagnostic', 'lambda_loss', 'actual_lambda',
                'gF_norm', 'raw_gA_norm', 'actual_weighted_aux_norm', 'unweighted_aux_FM_norm_ratio',
                'actual_weighted_aux_FM_norm_ratio', 'cosine', 'clip_coefficient', 'preclip_norm',
                'primary_buffer_bytes', 'primary_capture_seconds', 'scalar_reduction_seconds',
                'combination_seconds', 'carrier_seconds', 'update_seconds', 'input_seconds'):
        result[key + '_mean'] = average(r[key] for r in rows)
        result[key + '_median'] = middle(r[key] for r in rows)
    result.update(actual_FP32_budget_excess_max=extreme((r['actual_FP32_budget_excess'] for r in rows), max),
        theoretical_budget_excess_max=extreme((r['theoretical_budget_excess'] for r in rows), max),
        actual_weighted_aux_FM_ratio_max=extreme((r['actual_weighted_aux_FM_norm_ratio'] for r in rows), max),
        actual_update_snapshot_count=len(updates), expected_update_snapshots=21,
        actual_update_norm_mean=average(r['actual_update_norm'] for r in updates),
        actual_update_norm_median=middle(r['actual_update_norm'] for r in updates),
        actual_update_relative_norm_mean=average(r['actual_update_relative_norm'] for r in updates),
        actual_update_relative_norm_median=middle(r['actual_update_relative_norm'] for r in updates),
        actual_parameter_norm_mean=average(r['parameter_before_norm'] for r in updates),
        update_scope='Only registered actual parameter snapshots; no interpolation or all-step estimate')
    return result


def plot_training(path, steps, updates):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10})
    fig, grid = plt.subplots(4, 3, figsize=(14.4, 12), sharex=True)
    axes = list(grid.flat)
    panels = [('FM', 'Primary FM loss', False), ('raw_aux', 'Raw auxiliary MSE', True),
              ('weighted_aux_loss_diagnostic', 'Detached coefficient x raw auxiliary MSE', True),
              ('actual_lambda', 'Coefficient: actual solid / loss cap dashed', False),
              ('actual_weighted_aux_FM_norm_ratio', 'Full weighted auxiliary / primary gradient norm', False),
              ('cosine', 'Full unweighted gradient cosine', False),
              ('loss_cap_active', 'Loss cap fraction', False), ('gradient_cap_active', 'Gradient cap fraction', False),
              ('clip_applied', 'Total gradient clip fraction', False)]
    for arm in ARMS:
        rows = [r for r in steps if r['arm'] == arm]
        x = [r['step'] for r in rows]
        def rolling(key):
            return [average(r[key] for r in rows[max(0, i-49):i+1]) for i in range(len(rows))]
        for ax, (key, title, log) in zip(axes, panels):
            ax.plot(x, rolling(key), color=COLORS[arm], lw=1.4)
            if key == 'actual_lambda':
                ax.plot(x, rolling('lambda_loss'), color=COLORS[arm], lw=.9, ls='--')
            ax.set_title(title)
            if log:
                positive = any(r[key] > 0 for r in rows)
                if positive:
                    ax.set_yscale('log')
        axes[9].plot(x, rolling('primary_capture_seconds'), color=COLORS[arm], lw=1.3)
        axes[9].plot(x, rolling('scalar_reduction_seconds'), color=COLORS[arm], lw=1.1, ls='--')
        measured = [r for r in updates if r['arm'] == arm]
        for ax, key in ((axes[10], 'actual_update_norm'), (axes[11], 'actual_update_relative_norm')):
            ax.scatter([r['step'] for r in measured], [r[key] for r in measured], color=COLORS[arm], s=12, alpha=.8)
    axes[4].axhline(1, color='#555555', lw=.8, ls=':')
    axes[5].axhline(0, color='#555555', lw=.8, ls=':')
    for ax in axes[6:9]:
        ax.set_ylim(-.03, 1.03)
    axes[9].set_title('Seconds: primary capture solid / scalar reduction dashed')
    axes[10].set_title('Actual parameter update norm: registered snapshots only')
    axes[11].set_title('Actual update / parameter norm: snapshots only')
    for ax in axes:
        ax.axvline(500, color='#999999', lw=.8, ls='--')
        ax.set_xlim(1, 1000)
        ax.grid(True, alpha=.16)
        ax.spines[['top', 'right']].set_visible(False)
    for ax in axes[9:]:
        ax.set_xlabel('Training update')
    handles = [Line2D([0], [0], color=COLORS[a], lw=2, label=a + ': ' + NAMES[a]) for a in ARMS]
    fig.legend(handles=handles, loc='upper center', ncol=3, bbox_to_anchor=(.5, .995), frameon=False)
    fig.text(.5, .016, 'Curves: trailing 50 recorded updates, partial windows at the start. Dots: true parameter differences at 42 fixed steps per complete arm.\nTraining measurements alone do not establish generated-mesh quality or a causal effect on AdamW updates.', ha='center', va='bottom', fontsize=8)
    fig.tight_layout(rect=(0, .055, 1, .96), h_pad=1.7, w_pad=1.3)
    fig.savefig(path, dpi=180, facecolor='white')
    plt.close(fig)


def run(out=DEFAULT_OUT, live=False):
    started = time.time()
    out = Path(out).resolve()
    dest = out / 'training_diagnostics' / ('live' if live else 'final')
    dest.mkdir(parents=True, exist_ok=True)
    inputs = Inputs(live)
    audit_dir = out / 'training_audit' / ('live' if live else 'final')
    audit = inputs.read(audit_dir / 'summary.json')
    audit_steps = inputs.read(audit_dir / 'per_step_gradient_budget.csv', 'csv')
    audit_updates = inputs.read(audit_dir / 'actual_parameter_updates.csv', 'csv')
    logs = {arm: inputs.read(out / 'training' / (arm + '_metrics.jsonl'), 'jsonl', required=not live) for arm in ARMS}
    errors = []
    checks = Counter()
    def check(ok, name, detail=None):
        checks[name] += 1
        if not ok:
            errors.append(dict(check=name, detail=detail))
    if audit:
        check(audit['status'] in ('LIVE_PASS', 'LIVE_WAITING') if live else audit['status'] == 'PASS', 'strict_training_audit_status', audit['status'])
    if not live:
        for arm in ARMS:
            check(len(logs[arm]) == 1000, 'all_1000_actual_steps', arm)
            source = audit.get('sources', {}).get(str(out / 'training' / (arm + '_metrics.jsonl'))) if audit else None
            current = inputs.sources.get(str(out / 'training' / (arm + '_metrics.jsonl')))
            check(source is not None and current is not None and source['sha256'] == current['sha256'], 'logs_same_as_strict_final_audit', arm)
    steps, updates = [], []
    for arm, rows in logs.items():
        check([r['pilot_step'] for r in rows] == list(range(1, len(rows)+1)), 'contiguous_steps', arm)
        for row in rows:
            step = per_step(arm, row)
            steps.append(step)
            check(math.isclose(step['actual_lambda']*step['raw_aux'], step['weighted_aux_loss_diagnostic'], rel_tol=1e-14, abs_tol=0), 'detached_weighted_loss_scope', [arm, step['step']])
            snap = row.get('actual_parameter_update')
            if snap is not None:
                updates.append(dict(arm=arm, step=row['pilot_step'], phase=step['phase'],
                    **{k: snap[k] for k in ('actual_update_norm', 'actual_update_relative_norm', 'parameter_before_norm', 'parameter_after_norm', 'changed_elements', 'elements', 'max_abs_update', 'cpu_snapshot_bytes', 'snapshot_seconds', 'measure_seconds')},
                    actual_lambda=step['actual_lambda'], gradient_cap_active=step['gradient_cap_active'],
                    full_weighted_aux_FM_ratio=step['actual_weighted_aux_FM_norm_ratio'], clip_coefficient=step['clip_coefficient']))
    by_step = {(r['arm'], r['step']): r for r in steps}
    for row in audit_steps:
        key = (row['arm'], int(row['step']))
        check(key in by_step, 'audited_step_in_actual_log', key)
        if key in by_step:
            for audit_key, log_key in (('FM', 'FM'), ('raw_aux', 'raw_aux'), ('actual_lambda', 'actual_lambda'), ('gF_norm', 'gF_norm'), ('gA_norm', 'raw_gA_norm')):
                check(float(row[audit_key]).hex() == float(by_step[key][log_key]).hex(), 'audited_scalar_exact', [*key, audit_key])
    by_update = {(r['arm'], r['step']): r for r in updates}
    for row in audit_updates:
        key = (row['arm'], int(row['step']))
        check(key in by_update, 'audited_snapshot_in_actual_log', key)
        if key in by_update:
            check(float(row['actual_update_norm']).hex() == float(by_update[key]['actual_update_norm']).hex(), 'audited_update_norm_exact', key)
    if not live:
        check(len(audit_steps) == len(steps) == 3000, 'full_audited_step_table')
        check(len(audit_updates) == len(updates) == 126, 'full_audited_snapshot_table')
    phases = [phase_summary(arm, phase, [r for r in steps if r['arm'] == arm and r['phase'] == phase],
        [r for r in updates if r['arm'] == arm and r['phase'] == phase]) for arm in ARMS for phase in ('1-500', '501-1000')]
    paired = []
    for r in updates:
        if r['arm'] != 'G':
            continue
        l = by_update.get(('L', r['step']))
        if l is None:
            continue
        paired.append(dict(step=r['step'], phase=r['phase'], G_update_norm=r['actual_update_norm'], L_update_norm=l['actual_update_norm'],
            G_minus_L_update_norm=r['actual_update_norm']-l['actual_update_norm'],
            G_over_L_update_norm=r['actual_update_norm']/l['actual_update_norm'] if l['actual_update_norm'] else None,
            G_relative_update=r['actual_update_relative_norm'], L_relative_update=l['actual_update_relative_norm'],
            G_full_weighted_aux_FM_ratio=r['full_weighted_aux_FM_ratio'], L_full_weighted_aux_FM_ratio=l['full_weighted_aux_FM_ratio']))
    comparison = []
    for phase in ('1-500', '501-1000'):
        rows = [r for r in paired if r['phase'] == phase]
        comparison.append(dict(phase=phase, paired_snapshot_count=len(rows),
            G_lower_update_norm_count=sum(r['G_minus_L_update_norm'] < 0 for r in rows),
            G_minus_L_update_norm_mean=average(r['G_minus_L_update_norm'] for r in rows),
            G_over_L_update_norm_median=middle(r['G_over_L_update_norm'] for r in rows),
            interpretation='Descriptive paired registered snapshots. The gradient norm budget does not bound AdamW parameter updates; no causal inference.'))
    trows, grows = [r for r in steps if r['arm'] == 'T'], [r for r in steps if r['arm'] == 'G']
    result = dict(status='FAIL' if errors else 'INCOMPLETE' if inputs.missing else 'LIVE' if live else 'PASS', live=live,
        completed_steps={a: len(logs[a]) for a in ARMS}, strict_audit_status=audit.get('status') if audit else None,
        audited_prefix_steps=audit.get('completed_steps') if audit else None,
        checks=sum(checks.values()), checks_by_name=dict(checks), errors=errors, missing=inputs.missing,
        phase_summary=phases,
        teacher_actual_cap=dict(recorded_steps=len(trows), loss_cap_count=sum(r['loss_cap_active'] for r in trows),
            loss_cap_steps=[r['step'] for r in trows if r['loss_cap_active']],
            gradient_cap_count=sum(r['gradient_cap_active'] for r in trows)),
        G_budget=dict(recorded_steps=len(grows), gradient_cap_count=sum(r['gradient_cap_active'] for r in grows),
            actual_FP32_residual_max=extreme((r['actual_FP32_budget_excess'] for r in grows), max),
            actual_FP32_positive_residual_steps=sum(r['actual_FP32_budget_excess'] > 0 for r in grows),
            theoretical_residual_max=extreme((r['theoretical_budget_excess'] for r in grows), max),
            actual_weighted_aux_FM_ratio_max=extreme((r['actual_weighted_aux_FM_norm_ratio'] for r in grows), max)),
        actual_update_snapshot_counts={a: sum(r['arm'] == a for r in updates) for a in ARMS},
        G_vs_L_actual_updates=comparison,
        interpretation_limits=[
            'Weighted auxiliary loss is a detached scalar diagnostic. The actual auxiliary backward was unweighted and its full FP32 gradient was scaled exactly once.',
            'Gradient norms/cosines cover all trainable parameters on every recorded step, before total gradient clipping.',
            'Clip-applied counts use the logged coefficient < 1, including the original clipping epsilon. Norm > 1 counts are also provided.',
            'True parameter update norms cover only step 1, every 25 steps, and step 501: 42 snapshots per complete arm, 21 per half.',
            'No causal attribution from update comparisons: AdamW moments and parameter states diverge across arms.',
            'All loss means are step-weighted; changes in training loss alone do not establish generated-output improvements.',
            'Live data beyond the listed audited prefix are provisional and are rechecked against the final audit before formal reporting.'],
        sources=inputs.sources, ignored_live_partial_tails=inputs.partial_tails,
        script_sha256=digest_bytes(Path(__file__).read_bytes()), model_calls=0, checkpoint_tensors_loaded=False,
        CPU_threads=2, started_unix=started)
    for name, rows in (('per_step_training', steps), ('phase500_summary', phases), ('actual_parameter_updates', updates), ('G_vs_L_actual_updates', paired)):
        write_csv(dest / (name + '.csv'), rows)
    if steps:
        plot_training(dest / 'training_overview.png', steps, updates)
        result['figure'] = str(dest / 'training_overview.png')
    result['finished_unix'] = time.time()
    write_json(dest / 'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(DEFAULT_OUT))
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    result = run(args.out, args.live)
    print(json.dumps({k: result[k] for k in ('status', 'completed_steps', 'audited_prefix_steps', 'checks', 'teacher_actual_cap', 'G_budget', 'actual_update_snapshot_counts', 'errors')}, indent=2))
    raise SystemExit(1 if result['status'] == 'FAIL' else 0)
