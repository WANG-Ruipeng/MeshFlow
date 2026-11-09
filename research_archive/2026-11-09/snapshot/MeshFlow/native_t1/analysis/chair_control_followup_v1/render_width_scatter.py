"""Display-only render from already completed CSV; zero raw/metric/model evaluation."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent/'width_analysis'
with (p/'width_start_end_per_parent.csv').open() as f:
    rows=list(csv.DictReader(f))
fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
colors={'T1':'#3875ad','Geo':'#d07817','Full':'#478b54'}
for ax,split in zip(axes,['train','validation']):
    for arm in ['T1','Geo','Full']:
        members=[r for r in rows if r['split']==split and r['arm']==arm]
        ax.scatter([float(r['baseline_abs_L0_delta']) for r in members],
                   [float(r['mean_response_abs_residual_delta']) for r in members],label=arm,color=colors[arm])
    ax.axhline(0,color='gray',linewidth=.6); ax.axvline(0,color='gray',linewidth=.6)
    ax.set_xlabel('Change in baseline absolute width error / L0 (lower better)')
    ax.set_ylabel('Change in mean absolute response residual (lower better)')
    ax.set_title(f'{split}: first-noise start to endpoint'); ax.legend()
fig.savefig(p/'baseline_vs_response_changes.png',dpi=150)
print('PASS: display-only scatter rendered from 24 existing paired rows')
