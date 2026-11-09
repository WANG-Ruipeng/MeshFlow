"""Post-run analysis and failure illustration; never change protocol, meshes or scores."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'code'))
from common import load_mesh,dump,sha
rows=json.loads((R/'LEDGER.json').read_text()); arms=['freeze','transport','refit'];sources=['sphere','thin_u','bottle_local_1']
by={(r['source_object'],r['frame'],r['state']):r for r in rows}
def faceset(x):return set(map(tuple,np.sort(x,axis=1)))
def rd(s,f,a):return R/'runs'/f'{s}_frame{f:02d}__{a}'
records=[];same=[];end=[]
for s in sources:
    for f in range(8):
        a,b=by[s,f,'transport'],by[s,f,'refit']
        same.append(dict(source=s,frame=f,transport_refit_mesh_bytes_identical=a['mesh_sha256']==b['mesh_sha256']))
        for arm in arms:
            row=by[s,f,arm]
            if f:
                with np.load(rd(s,f,arm)/'extractor_trace.npz') as z:now=faceset(z['delaunay_simplices'])
                with np.load(rd(s,f-1,arm)/'extractor_trace.npz') as z:prev=faceset(z['delaunay_simplices'])
                old=by[s,f-1,arm]
                records.append(dict(source=s,arm=arm,from_frame=f-1,to_frame=f,
                    tetrahedra_added=len(now-prev),tetrahedra_removed=len(prev-now),
                    surface_connection_changed=row['surface_input_faces_sha256']!=old['surface_input_faces_sha256'],
                    betti_changed=[row['beta'+str(i)] for i in range(3)]!=[old['beta'+str(i)] for i in range(3)]))
    ref=by[s,7,'freeze']['rms_pct_bboxdiag']
    end.append(dict(source=s,transport_rms_improvement_relative_to_freeze=1-by[s,7,'transport']['rms_pct_bboxdiag']/ref,
                    refit_rms_improvement_relative_to_freeze=1-by[s,7,'refit']['rms_pct_bboxdiag']/ref))
record=dict(ledger_sha256=sha(R/'LEDGER.json'),read_only_analysis_of_saved_outputs=True,
    no_new_extractions=True,transport_refit_identical=sum(r['transport_refit_mesh_bytes_identical'] for r in same),
    transport_refit_comparisons=same,adjacent_frame_changes=records,endpoint_relative_improvement=end)
dump(R/'post_analysis/RESULTS.json',record)
# Explicit illustration of the discovered failure, separate from prespecified frames 0/4/7.
w=json.loads((rd('thin_u',6,'freeze')/'vertex_link_witnesses.json').read_text())['witnesses'][0]
point=np.array(w['vertex']);fullv,_=load_mesh(R/'inputs/thin_u_frame06/reference.obj')
center=(fullv.min(0)+fullv.max(0))/2;radius=np.ptp(fullv,axis=0).max()*.57
fig=plt.figure(figsize=(12,8))
for ai,arm in enumerate(arms):
    row=by['thin_u',6,arm];v,f=load_mesh(rd('thin_u',6,arm)/'mesh.obj');tri=v[f]
    with np.load(rd('thin_u',6,arm)/'mesh.npz') as z:inputids=z['input_vertex_ids']
    vid=int(np.linalg.norm(v-point,axis=1).argmin());assert np.array_equal(v[vid],point)
    incident=np.any(f==vid,axis=1)
    for zoom in range(2):
        ax=fig.add_subplot(2,3,zoom*3+ai+1,projection='3d')
        rgba=np.tile([.3,.56,.73,.85],(len(f),1));rgba[incident]=[.86,.34,.17,1]
        ax.add_collection3d(Poly3DCollection(tri,facecolors=rgba,edgecolors=(.05,.05,.05,.32),linewidths=.28))
        ax.scatter(*point,color='#c4001b',s=22,depthshade=False)
        c=point if zoom else center;r=.17 if zoom else radius
        ax.set_xlim(c[0]-r,c[0]+r);ax.set_ylim(c[1]-r,c[1]+r);ax.set_zlim(c[2]-r,c[2]+r)
        ax.set_box_aspect((1,1,1));ax.set_proj_type('ortho');ax.view_init(elev=52,azim=-65);ax.set_axis_off()
        title=f"{arm} | F={row['faces']} | Betti=({row['beta0']},{row['beta1']},{row['beta2']})" if not zoom else 'Same spatial point, incident triangles highlighted'
        ax.set_title(title,fontsize=10)
fig.suptitle('Observed intermediate failure: Thin-U, frame 6 (t=6/7)\nFreeze has two disconnected link cycles at one vertex; transport/refit have one cycle.',fontsize=12)
fig.tight_layout(rect=(0,0,1,.92));fig.savefig(R/'figures/thin_u_failure.png',dpi=150);plt.close(fig)
print(json.dumps(dict(transport_refit_identical=record['transport_refit_identical'],endpoints=end),indent=2))