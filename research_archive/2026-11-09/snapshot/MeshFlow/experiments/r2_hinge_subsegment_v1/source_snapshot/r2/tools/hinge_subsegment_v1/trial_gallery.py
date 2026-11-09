"""Scientific diagrams of the five saved split trials, not generated samples."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import json,time,textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from . import engine as E

C,S=E.C,E.S
OUT=S.OUT/'trial_gallery/v2_readable'


def shape_title(record):
    children=[r for r in record['affected_face_shapes'] if r['is_new_child']]
    return ' | '.join(f"child {i+1}: min angle {r['minimum_angle']:.2f} deg, q {r['q']:.2f}" for i,r in enumerate(children))


def poly3(ax,tri,color,alpha=.6,lw=.5):
    if not len(tri):return
    obj=Poly3DCollection(tri,facecolor=color,edgecolor='#25313b',linewidth=lw,alpha=alpha)
    ax.add_collection3d(obj)


def draw(record,q):
    c=record['candidate'];rank=record['rank'];dest=E.OUT/'SPLIT025'/q['request_id']
    base=C.arrays(S.paths(q)['mesh']);trial=C.arrays(dest/f'attempt_{rank:03d}.npz')
    v,f=base['vertices'].astype(float),base['faces'];tv,tf=trial['vertices'].astype(float),trial['faces']
    parent=int(record['split_parent_post_face_id']);bparent=int(c['bulk_post_face_id']);h=int(record['hinge_vertex']);far=int(record['seam_far_vertex']);u=int(record['moved_vertex'])
    children=np.asarray(record['new_child_rows']);affected=set(record['affected_parent_post_face_ids']);affected.discard(parent)
    conflict=set(z for pair in record['new_illegal_parent_pairs']['FF'] for z in pair)
    # Include the complete modified cluster neighborhood and any new illegal-pair
    # partners. This makes a locally good seam insufficient to hide nearby costs.
    shown=sorted(affected|conflict|{bparent});afterrows=np.flatnonzero(np.isin(trial['parent_post_face_id'],shown))
    allpoints=np.concatenate((v[f[[parent]+shown]].reshape(-1,3),tv[tf[np.r_[children,afterrows]]].reshape(-1,3)))
    lo,hi=allpoints.min(0),allpoints.max(0);center=(lo+hi)/2;extent=max(float((hi-lo).max())*.58,.02)
    fig=plt.figure(figsize=(17,6.8),constrained_layout=True)
    ax0=fig.add_subplot(131,projection='3d');ax1=fig.add_subplot(132,projection='3d');ax2=fig.add_subplot(133)
    for ax in (ax0,ax1):
        ax.set_xlim(center[0]-extent,center[0]+extent);ax.set_ylim(center[1]-extent,center[1]+extent);ax.set_zlim(center[2]-extent,center[2]+extent)
        ax.view_init(elev=23,azim=-61);ax.set_box_aspect((1,1,1));ax.set_xlabel('x');ax.set_ylabel('y');ax.set_zlabel('z')
        ax.tick_params(labelsize=7)
    poly3(ax0,v[f[shown]],'#4baec1',.26);poly3(ax0,v[f[parent:parent+1]],'#f2c14e',.83)
    poly3(ax1,tv[tf[afterrows]],'#4baec1',.26)
    if conflict:
        ci=np.flatnonzero(np.isin(trial['parent_post_face_id'],sorted(conflict)));poly3(ax1,tv[tf[ci]],'#d24b91',.4)
    for i,col in zip(children,('#79bd73','#ec9857')):poly3(ax1,tv[tf[i:i+1]],col,.85)
    for ax in (ax0,ax1):
        ax.plot(*np.vstack((v[u],tv[u])).T,color='#b8203a',linewidth=2)
        ax.scatter(*v[h],color='#232323',s=25);ax.text(*v[h],' H',fontsize=9)
        ax.scatter(*v[far],color='#232323',s=25);ax.text(*v[far],' F',fontsize=9)
        ax.scatter(*v[u],color='#b8203a',s=28);ax.scatter(*tv[u],color='#224fa3',s=28)
    ax0.set_title('Saved baseline: seam + affected neighborhood',fontsize=11)
    ax1.set_title('Saved trial (NOT final when rejected)\nmagenta = new illegal FF participants',fontsize=11)
    # Plane coordinates preserve the actual edge lengths and triangle aspect.
    origin=v[h];t=(v[far]-origin);t/=np.linalg.norm(t)
    n=np.cross(v[f[parent,1]]-v[f[parent,0]],v[f[parent,2]]-v[f[parent,0]]);n/=np.linalg.norm(n);m=np.cross(n,t)
    def project(x):return np.stack(((x-origin)@t,(x-origin)@m),axis=-1)
    parent2=project(v[f[parent]]);ax2.add_patch(Polygon(parent2,closed=True,facecolor='none',edgecolor='#333',linewidth=2))
    for i,col in zip(children,('#79bd73','#ec9857')):ax2.add_patch(Polygon(project(tv[tf[i]]),closed=True,facecolor=col,edgecolor='#303030',linewidth=1,alpha=.68))
    for key,point,col,offset in [('H',v[h],'#222',(-12,-16)),('F',v[far],'#222',(6,5)),('u',v[u],'#b8203a',(6,-15)),('P',tv[u],'#224fa3',(6,7))]:
        pp=project(point);ax2.scatter(*pp,color=col,s=30,zorder=5);ax2.annotate(key,pp,xytext=offset,textcoords='offset points',fontsize=9,color=col)
    a,b=project(v[u]),project(tv[u]);ax2.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#b8203a',lw=1.8))
    ax2.autoscale_view();ax2.set_aspect('equal',adjustable='datalim');ax2.margins(.18);ax2.grid(alpha=.2)
    ax2.set_xlabel('Along seam side (model units)');ax2.set_ylabel('Within original seam plane')
    ax2.set_title('Same-plane projection: children at true aspect\nred arrow = projected bulk endpoint movement',fontsize=11)
    badff=len(record['new_illegal_parent_pairs']['FF']);status=record['decision']
    top=f"{q['request_id']} | seam slot {c['seam_source_raw_slot']} / bulk slot {c['bulk_source_raw_slot']} | 2.5% L trial | {status}"
    bottom=(shape_title(record)+f"\nnew illegal FF = {badff}; moved cluster = {len(record['original_cluster_members'])} original corners; "
        f"max PRE displacement = {record['original_PRE_max_displacement']/C.L*100:.3f}% L\n"
        +textwrap.fill('Guard failures: '+(', '.join(record['all_guard_failures']) or 'none'),width=150)
        +'\nH = shared index; F = far endpoint; u = original bulk endpoint; P = projected split. Yellow = parent; green/orange = children; cyan = affected bulk.')
    fig.suptitle(top+'\n'+bottom,fontsize=10)
    path=OUT/(q['request_id']+f'_seam{c["seam_source_raw_slot"]}_bulk{c["bulk_source_raw_slot"]}.png')
    fig.savefig(path,dpi=160);plt.close(fig)
    return dict(request_id=q['request_id'],seam_source_raw_slot=c['seam_source_raw_slot'],bulk_source_raw_slot=c['bulk_source_raw_slot'],
        decision=status,trial_only=True,figure=str(path),baseline_path=str(S.paths(q)['mesh']),attempt_path=str(dest/f'attempt_{rank:03d}.npz'),
        child_shape=shape_title(record),new_illegal_FF=badff,all_guard_failures=record['all_guard_failures'])


def main():
    if (OUT/'completion.json').exists():return C.read(OUT/'completion.json')
    OUT.mkdir(parents=True,exist_ok=True);tick=time.perf_counter()
    records=[r for r in C.read(E.OUT/'all_trials.json') if r['delta_fraction']==.025];assert len(records)==5
    qs={q['request_id']:q for q in S.O.OLD.requests()};rows=[draw(r,qs[r['request_id']]) for r in records]
    C.csvwrite(OUT/'figures.csv',rows);result=dict(status='COMPLETE',figures=5,seconds=time.perf_counter()-tick,
        model_calls=0,new_meshes=0,source_sha256=C.sha(__file__),records=rows,files={str(p):C.sha(p) for p in OUT.iterdir() if p.is_file()})
    C.write(OUT/'completion.json',result);return result


if __name__=='__main__':print(json.dumps(main(),ensure_ascii=False),flush=True)
