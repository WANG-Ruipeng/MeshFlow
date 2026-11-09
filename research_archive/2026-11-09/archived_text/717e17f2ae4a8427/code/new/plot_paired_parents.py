"""Plot existing paired parent deltas only; no scoring or model imports."""
from pathlib import Path
import csv
import datetime
import hashlib
import json
import math
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

OUT = Path('/mnt/e/MeshFlow/experiments/r2_user_data_scaling_32_67_v1')
SOURCE = OUT/'report_tables/expanded_minus_small_per_parent.csv'
DEST = OUT/'training_analysis/final/paired_NEW32_POST.png'
QA = DEST.with_name('paired_NEW32_POST_qa.json')
SPECS = [
 ('CF_illegal_affected_free_fraction', 'CF | affected free faces', 'Lower is better (negative delta)', 100., 'percentage points'),
 ('FF_illegal_affected_free_fraction', 'FF | affected free faces', 'Lower is better (negative delta)', 100., 'percentage points'),
 ('ConflictCov_C', 'I | C conflict coverage', 'Lower is better (negative delta)', 100., 'percentage points'),
 ('ConflictCov_C_cross', 'I_cross | crossing coverage', 'Lower is better (negative delta)', 100., 'percentage points'),
 ('ConflictCov_C_dup', 'I_dup | duplicate-layer coverage', 'Lower is better (negative delta)', 100., 'percentage points'),
 ('free_minimum_angle_lt5_fraction', 'Free faces with angle < 5 degrees', 'Lower is better (negative delta)', 100., 'percentage points'),
 ('A_VMR__raw_tau_1e-6', r'A-VMR | raw $\tau=10^{-6}$', 'Higher matching is better; diagnostic only', 100., 'percentage points'),
 ('A_VMR__L_tau_0.01', 'A-VMR | 1% L tolerance', 'Higher matching is better; diagnostic only', 100., 'percentage points'),
 ('A_VMR__L_tau_0.02', 'A-VMR | 2% L tolerance', 'Higher matching is better; diagnostic only', 100., 'percentage points'),
 ('opposite_winding_reached_free_area_fraction', 'Direction-consistent reachable free area', 'Higher is connection evidence', 100., 'percentage points'),
 ('deleted_free_faces', 'Deleted free faces', 'Descriptive: no preferred sign', 1., 'faces'),
 ('nonmanifold_edges', 'Nonmanifold edges', 'Lower defect count is better', 1., 'edges'),
]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    for p in (SOURCE, DEST, QA, Path(__file__).resolve()):
        if any(q.is_symlink() for q in (p, *p.parents)):
            raise RuntimeError('Symlink path refused')
    if DEST.exists() or QA.exists():
        raise FileExistsError('Preserve existing figure and QA; no silent overwrite')
    source_hash = sha(SOURCE)
    with SOURCE.open(encoding='utf-8-sig', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['comparison']=='USER_EXPANDED_67-USER_SMALL'
                and r['panel']=='NEW32' and r['stage']=='POST'
                and r['stratum']=='ALL' and r['stratum_value']=='ALL']
    table = {}
    for row in rows:
        key = row['parent_id'], row['metric']
        if key in table:
            raise ValueError('Duplicate parent/metric')
        table[key] = row
    parents = sorted({r['parent_id'] for r in rows})
    if len(parents) != 32 or len({p[:6] for p in parents}) != 32:
        raise ValueError('Require 32 parents and unique six-character display IDs')
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.titlesize':11,
                         'axes.labelsize':9, 'xtick.labelsize':7, 'ytick.labelsize':8,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'savefig.facecolor':'white'})
    fig, axes = plt.subplots(4, 3, figsize=(16.5, 16.8), sharex=True)
    fig.subplots_adjust(left=.065, right=.985, top=.87, bottom=.105, hspace=.58, wspace=.26)
    fig.suptitle('Expanded 67 vs. Small 32: paired changes on unseen USER parents', y=.977,
                 fontsize=19, fontweight='semibold')
    fig.text(.5, .950, 'NEW32 / POST  |  delta = Expanded - Small  |  each dot is one parent  |  dashed line = macro mean',
             ha='center', fontsize=11, color='#374151')
    fig.text(.5, .929, 'All 12 panels use the same 32 parent IDs in lexicographic order; metric-specific ordering and tail clipping are not used.',
             ha='center', fontsize=10, color='#555b65')
    audit = []
    x = list(range(32))
    for index, (ax, spec) in enumerate(zip(axes.flat, SPECS)):
        metric, title, direction, scale, unit = spec
        selected = [table[(parent, metric)] for parent in parents]
        if any(r['status'] != 'PASS' or r['detector_uncertain'] != 'False' for r in selected):
            raise ValueError('Missing/uncertain input must be disclosed, not silently plotted as valid')
        values = [float(r['delta']) * scale for r in selected]
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Nonfinite supplied parent delta')
        if any(not math.isclose(float(r['candidate_value'])-float(r['baseline_value']), float(r['delta']),
                                rel_tol=1e-10, abs_tol=1e-12) for r in selected):
            raise ValueError('Supplied delta arithmetic differs')
        mean = statistics.mean(values)
        limit = max(max(abs(v) for v in values), abs(mean), .25) * 1.26
        ax.set_ylim(-limit, limit)
        ax.set_xlim(-.7, 31.7)
        ax.axhline(0, color='#747b86', linewidth=.8, zorder=1)
        ax.vlines(x, 0, values, color='#8eaac3', linewidth=.7, alpha=.65, zorder=2)
        ax.scatter(x, values, color='#326b99', edgecolor='white', linewidth=.35, s=29, zorder=4)
        ax.axhline(mean, color='#1f2937', linestyle=(0,(4,3)), linewidth=1.15, zorder=3)
        ax.grid(axis='y', color='#e5e7eb', linewidth=.65, zorder=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))
        ax.set_title(f'({chr(97+index)}) {title}', loc='left', pad=27, fontweight='semibold')
        ax.text(0, 1.045, direction, transform=ax.transAxes, fontsize=9, color='#525965')
        suffix = 'pp' if scale==100. else unit
        ax.text(.98, .96, f'mean {mean:+.2f} {suffix}', transform=ax.transAxes, ha='right', va='top',
                fontsize=9, color='#1f2937', bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=2.5))
        ax.set_ylabel('Delta ('+unit+')')
        ax.set_xticks(x)
        ax.set_xticklabels([p[:6] for p in parents], rotation=90, ha='center')
        ax.tick_params(axis='x', length=2, color='#9aa1aa')
        if index >= 9:
            ax.set_xlabel('Parent ID prefix (fixed order)', labelpad=9)
        audit.append(dict(metric=metric, title=title, direction=direction, original_direction=sorted({r['direction'] for r in selected}),
                          scale=scale, unit=unit, count=32, all_PASS=True, detector_uncertain=False,
                          parent_ids=parents, mean_delta=mean, min_delta=min(values), max_delta=max(values),
                          plotted_y_limits=list(ax.get_ylim()), all_points_within_limits=all(-limit<=v<=limit for v in values),
                          n_registered_values=sorted({r['n_registered'] for r in selected}),
                          values_in_display_order=values))
    fig.text(.065, .035, 'Fractions are displayed in percentage points; face/edge counts retain their original units. All meshes shown here are POST.',
             fontsize=9.5, color='#424955')
    fig.text(.065, .018, 'Raw tau denotes model-coordinate tolerance; L is the frozen request scale. Matching/reachability do not certify complete or conflict-free attachment.',
             fontsize=9.5, color='#424955')
    fig.savefig(DEST, dpi=220, metadata={'Title':'NEW32 POST paired parent deltas',
                                        'Description':'Existing table only; fixed parent order; no model calls or rescoring'})
    plt.close(fig)
    if sha(SOURCE) != source_hash:
        raise RuntimeError('Input changed while plotting')
    qa = dict(schema='paired_parent_plot_qa_v1', status='AUTOMATED_PASS_PENDING_VISUAL_QA',
              utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), source=str(SOURCE), source_sha256=source_hash,
              source_unmodified=True, code=str(Path(__file__).resolve()), code_sha256=sha(Path(__file__).resolve()),
              image=str(DEST), image_sha256=sha(DEST), panels=12, parent_count=32, observation_points=384,
              same_parent_order_all_panels=True, order='lexicographic parent ID, independent of values',
              new_model_calls=0, new_scoring_calls=0, numeric_transform='Only existing delta x100 for fractional units and existing parent macro mean',
              panel_audits=audit, actual_visual_QA=False)
    QA.write_text(json.dumps(qa, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:qa[k] for k in ('status','image','image_sha256','panels','parent_count','observation_points')}))


if __name__=='__main__':
    main()
