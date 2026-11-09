"""Verify and summarize finished training only; no evaluation stage is invoked."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import json
import sys
import numpy as np
from audit_io import *
from final_verify import verify_training
from summarize import training_tables


def main():
    audit = FileAudit()
    verification = verify_training(audit)
    curves, windows, exposure = training_tables(audit)
    dest = OUT / 'analysis/training_complete_v2'
    if dest.exists():
        raise RuntimeError('Training analysis already exists; do not overwrite')
    stats = {}
    by_arm = {}
    for arm in ('PLAIN', 'LAYOUT'):
        current = [r for r in curves if r['arm'] == arm]
        by_arm[arm] = current
        state = audit.read(OUT / 'training' / arm / 'status.json')
        log = audit.jsonl(OUT / 'training' / arm / 'training_log.jsonl')
        selected = [r for r in exposure if r['arm'] == arm]
        report = dict(updates=1000, forwards=8000, backwards=8000, exposures=8000,
            parent_count=len({r['parent_id'] for r in selected}),
            conditions=len({r['condition_id'] for r in selected}),
            actual_layout_dropout_instances=sum(r['layout_drop_instances'] for r in selected),
            actual_layout_dropout_fraction=sum(r['layout_drop_instances'] for r in selected) / 8000,
            layout_active_instances=sum(r['layout_active_instances'] for r in current),
            elapsed_seconds=state['elapsed_seconds'], update_seconds_sum=sum(r['update_seconds'] for r in current),
            update_seconds_mean=float(np.mean([r['update_seconds'] for r in current])),
            input_seconds_sum=sum(r['input_seconds'] for r in current),
            peak_allocated_MiB=max(r['peak_allocated_bytes'] for r in current) / 2**20,
            peak_reserved_MiB=max(r['peak_reserved_bytes'] for r in current) / 2**20,
            nvidia_process_peak_MiB=state['resources']['process_peak_mib'],
            nvidia_process_memory_status=state['resources']['process_memory_status'],
            observed_total_device_peak_MiB=state['resources']['total_device_used_peak_mib'],
            timing_scope='Observed overlapping workers; not an isolated compute benchmark',
            inherited_base_optimizer_global_step=26500,
            fresh_layout_optimizer_local_step=1000 if arm == 'LAYOUT' else None)
        for width in (50, 100):
            for which, block in (('first', current[:width]), ('last', current[-width:])):
                prefix = which + str(width)
                report[prefix + '_objective_mean'] = float(np.mean([r['objective'] for r in block]))
                report[prefix + '_gradient_norm_mean'] = float(np.mean([r['gradient_norm'] for r in block]))
        if arm == 'LAYOUT':
            diagnostics = log[-1]['layout_diagnostics']
            report['step1000_branch_delta_rms_mean'] = float(np.mean([r['delta_rms'] for r in diagnostics]))
            report['step1000_hidden_rms_mean'] = float(np.mean([r['hidden_rms'] for r in diagnostics]))
            report['step1000_delta_to_hidden_rms_mean'] = float(np.mean([r['delta_rms'] / r['hidden_rms'] for r in diagnostics]))
            report['step1000_actual_changed_scalars'] = sum(r['actual_changed_scalars'] for r in diagnostics)
        stats[arm] = report
    paired = []
    for a, b in zip(by_arm['PLAIN'], by_arm['LAYOUT']):
        paired.append(dict(step=a['step'], global_step=a['global_step'],
            PLAIN_objective=a['objective'], LAYOUT_objective=b['objective'],
            LAYOUT_minus_PLAIN_objective=b['objective'] - a['objective'],
            PLAIN_gradient_norm=a['gradient_norm'], LAYOUT_gradient_norm=b['gradient_norm']))
    for name, rows in dict(training_curve=curves, windows50=windows, actual_exposure=exposure, paired_curve=paired).items():
        csvwrite(dest / (name + '.csv'), rows)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for arm, color in (('PLAIN', '#334155'), ('LAYOUT', '#d97706')):
        current = by_arm[arm]
        for ax, key in zip(axes, ('objective', 'gradient_norm')):
            smoothed = np.convolve([r[key] for r in current], np.ones(50) / 50, mode='valid')
            ax.plot(np.arange(50, 1001), smoothed, label=arm + ' mean over preceding 50 updates', color=color)
            ax.grid(alpha=.2)
            ax.set_ylabel('Original weighted FM' if key == 'objective' else 'Pre-clip global gradient norm')
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel('Local optimizer update (same registered stream)')
    fig.suptitle('Bench coarse-layout paired training: fixed 1,000 updates\nTraining objective alone does not establish generation or C-adoption gains', fontsize=11)
    fig.tight_layout(); fig.savefig(dest / 'training_curve.png', dpi=150); plt.close(fig)
    result = dict(status='PASS', schema='bench_layout_oracle_training_closeout_v1',
        verification=verification, arms=stats,
        paired_objective_difference=dict(first50=float(np.mean([r['LAYOUT_minus_PLAIN_objective'] for r in paired[:50]])),
            last50=float(np.mean([r['LAYOUT_minus_PLAIN_objective'] for r in paired[-50:]])),
            first100=float(np.mean([r['LAYOUT_minus_PLAIN_objective'] for r in paired[:100]])),
            last100=float(np.mean([r['LAYOUT_minus_PLAIN_objective'] for r in paired[-100:]]))),
        scope='Training-only closeout. Loss and nonzero branch responses do not establish useful conditional generation.',
        protected=audit.files, files={str(p): sha(p) for p in dest.iterdir() if p.is_file()},
        model_calls=0, scoring_calls=0, generator_state_loads=0, ratings_generated=0,
        source_sha256=sha(__file__))
    if 'torch' in sys.modules:
        raise RuntimeError('Training closeout imported torch')
    write(OUT / 'audits/training_completion.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('protected', 'files', 'verification')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
