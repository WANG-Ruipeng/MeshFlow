"""Offline upright view of the preregistered parent0/noise9201 raw outputs.
Display-axis permutation only; no geometry/model/scoring changes.
"""
from pathlib import Path
import json, hashlib
import numpy as np


def render(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
    out=Path(output).resolve();r=json.loads((out/'results.json').read_text())
    if r['status']!='PASS_EXECUTION':raise RuntimeError('Formal run must be complete')
    destination=out/'fixed_parent0_seed9201_upright.png'
    if destination.exists():raise FileExistsError(destination)
    groups=('Native0','T1','T2');ks=(2,4,8,12);geometry={};contexts={}
    for K in ks:
        for group in groups:
            row=next(q for q in r['generated'] if (q['model'],q['parent'],q['K'],q['seed'])==(group,0,K,9201))
            path=Path(row['raw_path'])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['raw_sha256']:raise RuntimeError('Raw hash mismatch')
            with np.load(path,allow_pickle=False) as f:raw=f['output'].reshape(112,3,3).copy()
            geometry[K,group]=raw[:,:,[0,2,1]]
            if group=='Native0':contexts[K]=geometry[K,group][:K].copy()
            if not np.array_equal(geometry[K,group][:K],contexts[K]):raise RuntimeError('Contexts differ')
    points=np.concatenate([v.reshape(-1,3) for v in geometry.values()])
    low,high=points.min(0),points.max(0);center=(low+high)/2;radius=float((high-low).max())*.55
    fig=plt.figure(figsize=(14,23),constrained_layout=True)
    for ki,K in enumerate(ks):
        for ri,part in enumerate(('full','free')):
            for ci,group in enumerate(('C',)+groups):
                ax=fig.add_subplot(8,4,(ki*2+ri)*4+ci+1,projection='3d')
                desired=contexts[K]
                if group=='C':tri=desired if part=='full' else np.empty((0,3,3))
                else:tri=geometry[K,group] if part=='full' else geometry[K,group][K:]
                if len(tri):
                    ax.add_collection3d(Line3DCollection(np.concatenate([tri,tri[:,:1]],axis=1),colors='#486474',linewidths=.6,alpha=.85))
                if part=='full':ax.add_collection3d(Poly3DCollection(desired,facecolors='#ef463d',edgecolors='#961c1c',linewidths=.8,alpha=.3))
                ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius)
                ax.set_box_aspect((1,1,1));ax.view_init(elev=24,azim=-54);ax.set_axis_off()
                ax.set_title(f'K={K} | {group} | {part}'+ (' (untrained K)' if K==2 else ''),fontsize=10)
    fig.suptitle('Fixed parent0 / reused noise9201: all declared K and models\nRed = exact known C; free rows exclude C. Display (X,Z,Y), original Y vertical. Shared camera and bounds.',fontsize=13)
    fig.savefig(destination,dpi=150);plt.close(fig)
    return dict(path=str(destination),sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        fixed_parent=0,fixed_regression_seed=9201,K=list(ks),groups=list(groups),
        display_axes=['X','Z','Y'],camera={'elevation':24,'azimuth':-54},shared_center=center.tolist(),shared_radius=radius,
        raw_coordinates_modified=False,scoring_modified=False,context_overlay_in_free_rows=False,
        model_calls=0,optimizer_updates=0,postprocessing='NONE',source='All preregistered raw outputs for fixed P0/9201; no success-based image selection')
