"""Illustrate the two saved critical frames without rerunning extraction."""
import sys,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'code'))
from common import load_mesh
rows=json.loads((R/'COMBINED_LEDGER.json').read_text());by={(r['source_object'],r['frame'],r['state']):r for r in rows}
arms=['freeze','q_only','n_only','transport'];fig=plt.figure(figsize=(15,8))
for k,frame in enumerate([6,7]):
    rv,_=load_mesh(R/'inputs'/f'thin_u_frame{frame:02d}'/'reference.obj');center=(rv.min(0)+rv.max(0))/2;radius=np.ptp(rv,axis=0).max()*.56
    for i,arm in enumerate(arms):
        row=by['thin_u',frame,arm];base=Path(row['measurement_run']);path=base/'runs'/f"{row['object']}__{arm}";v,f=load_mesh(path/'mesh.obj')
        ax=fig.add_subplot(2,4,k*4+i+1,projection='3d');color=[.3,.56,.73,.9] if row['joint_valid'] else [.82,.34,.25,.95]
        ax.add_collection3d(Poly3DCollection(v[f],facecolors=color,edgecolors=(.06,.06,.06,.32),linewidths=.24))
        if not row['joint_valid']:
            for p in row['strict_noncycle_vertex_ids']:ax.scatter(*v[p],c='#ad0021',s=30,depthshade=False)
        for setter,a in [(ax.set_xlim,0),(ax.set_ylim,1),(ax.set_zlim,2)]:setter(center[a]-radius,center[a]+radius)
        ax.set_box_aspect((1,1,1));ax.set_proj_type('ortho');ax.view_init(elev=52,azim=-65);ax.set_axis_off()
        ax.set_title(f"{arm} | frame {frame} | {'VALID' if row['joint_valid'] else 'INVALID'}\nF={row['faces']} | RMS={row['rms_pct_bboxdiag']:.6f}%\nBetti=({row['beta0']},{row['beta1']},{row['beta2']})",fontsize=10)
fig.suptitle('Thin-U: unchanged positions, different Q / normal updates\nRed marks invalid geometry; only Q preserves the joint-transport output in both critical frames.',fontsize=13)
fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.84,wspace=.015,hspace=.12)
fig.savefig(R/'figures/critical_frames.png',dpi=140);plt.close(fig)
print('KEY_FRAME_FIGURE_SAVED')