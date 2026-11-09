"""Deterministic local scientific views of saved accepted CPU operations.

These diagrams explain selected edits. They do not replace the fixed evaluation
panel or constitute actual visual review. No candidates or guards are executed.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import collections,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from . import run as R,holes

C,Q,B=R.C,R.Q,R.B
OUT=R.OUT/'local_geometry'
COLORS=dict(C='#6193ce',SEAM='#eeb347',BULK='#79afba',BRIDGE='#b457b7')


def role_array(mesh):return holes.face_roles(mesh)


def reach(mesh,pairs):
    """Only the existing cached-contact valid-edge graph; zero geometry scans."""
    K=int(mesh['known_faces']);roles=role_array(mesh);_,edges,_=Q.helpers()['G'].actual_valid_graph(mesh['vertices'],mesh['faces'],K,pairs)
    adj=collections.defaultdict(set);seeds=set()
    for row in edges:
        if not row['valid_edge']:continue
        i,j=map(int,row['incident_face_ids'])
        if i>=K and j>=K:adj[i].add(j);adj[j].add(i)
        elif i<K<=j and roles[j]=='SEAM':seeds.add(j)
        elif j<K<=i and roles[i]=='SEAM':seeds.add(i)
    reached=set(seeds);todo=list(seeds)
    while todo:
        i=todo.pop()
        for j in adj[i]-reached:reached.add(j);todo.append(j)
    tri=mesh['vertices'][mesh['faces']].astype(float);area=.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    bulk=np.flatnonzero(roles=='BULK');got=[i for i in bulk if i in reached]
    return dict(bulk_rows=set(map(int,got)),fraction=float(area[got].sum()/area[bulk].sum()) if len(bulk) and area[bulk].sum() else 0.)


def apply_saved(v,roots,operations):
    vv=v.copy();rr=roots.copy()
    for op in operations:
        ids=op['original_cluster_ids'];vv[ids]=np.asarray(op['target'],np.float32);rr[ids]=int(op['new_root'])
    return vv,rr


def collect():
    if not (R.OUT/'geometry_complete.json').exists():raise RuntimeError('Wait for complete registered 112-cell geometry')
    events=[];reachrows=[];qs=R.O.OLD.requests()
    for q in qs:
        base=C.arrays(R.basepath(q)/'post.npz');bp=C.read(R.basepath(q)/'pairs.json');br=reach(base,bp)
        for arm in R.ARMS:
            folder=R.OUT/'cells'/arm/q['request_id'];final=C.arrays(folder/'post.npz');records=C.read(folder/'records.json')
            nr=reach(final,C.read(folder/'pairs.json'));gain=nr['fraction']-br['fraction'];newbulk=len(nr['bulk_rows']-br['bulk_rows'])
            reachrows.append(dict(request_id=q['request_id'],split=q['split'],arm=arm,baseline_bulk_fraction=br['fraction'],final_bulk_fraction=nr['fraction'],gain=gain,newly_reached_original_bulk_faces=newbulk))
            if arm in ('MIDPOINT_REPLAY','ALPHA'):
                v=base['vertices'].copy();roots=np.arange(len(v));f0=base['faces']
                for rec in records:
                    if rec['decision']!='ACCEPTED':continue
                    saved=next(z for z in rec['location_attempts'] if z['decision']=='ACCEPTED')
                    vv,rr=apply_saved(v,roots,saved['merged_clusters'])
                    category='ALPHA_NEW_LOCATION' if saved['proposal_kind']=='ALPHA' else 'ORIGINAL_LOCATION_REPLAY'
                    if arm=='ALPHA' and category=='ORIGINAL_LOCATION_REPLAY':v,roots=vv,rr;continue
                    affected=set(saved['affected_source_raw_slots']);sf=base['source_face_ids'];show=np.flatnonzero(np.isin(sf,list(affected)))
                    # One existing indexed ring gives context without hiding the
                    # local edit inside an entire chair.
                    verts=set(map(int,(roots[f0])[show].reshape(-1)));ring=np.flatnonzero(np.any(np.isin(roots[f0],list(verts)),axis=1))
                    selected=np.unique(np.r_[show,ring])
                    events.append(dict(request_id=q['request_id'],split=q['split'],arm=arm,category=category,rank=int(rec['rank']),
                        alpha=saved['alpha'],before_vertices=v.copy(),after_vertices=vv.copy(),before_faces=roots[f0],after_faces=rr[f0],
                        shown_rows=selected,roles=role_array(base),record=rec,operations=saved['merged_clusters'],gain=gain,newbulk=newbulk,
                        source_path=str(folder/'records.json'),source_sha256=C.sha(folder/'records.json')))
                    v,roots=vv,rr
                assert np.array_equal(v,final['vertices']) and np.array_equal(roots[f0],final['faces']),'Saved accepted-operation replay differs'
            else:
                inp=base if arm=='HOLE' else C.arrays(R.OUT/'cells/ALPHA'/q['request_id']/'post.npz')
                roles=role_array(final);previous_faces=inp['faces'].copy()
                for rec in records:
                    if rec['decision']!='ACCEPTED':continue
                    newface=int(rec['new_bridge_output_face_id']);tri=np.asarray(rec['oriented_bridge_vertices'],np.int64)
                    after=np.concatenate((previous_faces,tri[None]))
                    assert newface==len(after)-1 and np.array_equal(final['faces'][newface],tri)
                    ring=np.flatnonzero(np.any(np.isin(after,tri),axis=1));selected=np.unique(np.r_[rec['adjacent_face_rows'],ring,newface])
                    events.append(dict(request_id=q['request_id'],split=q['split'],arm=arm,category=arm+'_BRIDGE',rank=int(rec['rank']),alpha=None,
                        before_vertices=inp['vertices'],after_vertices=inp['vertices'],before_faces=previous_faces,after_faces=after,
                        shown_rows=selected,roles=roles,record=rec,operations=[],gain=gain,newbulk=newbulk,
                        source_path=str(folder/'records.json'),source_sha256=C.sha(folder/'records.json')))
                    previous_faces=after
                assert np.array_equal(previous_faces,final['faces'])
    return events,reachrows


def choose(events,reachrows):
    """Fixed explanation selection, never a replacement evaluation subset."""
    selected={}
    for category in sorted({e['category'] for e in events}):
        for split in ('TRAIN','DEV'):
            candidates=sorted((e for e in events if e['category']==category and e['split']==split),key=lambda e:(e['request_id'],e['arm'],e['rank']))
            if candidates:
                e=candidates[0];selected[(e['arm'],e['request_id'],e['rank'])]=(e,['FIRST_'+category+'_'+split])
    dev=sorted((r for r in reachrows if r['split']=='DEV' and r['newly_reached_original_bulk_faces']>0),key=lambda r:(-r['gain'],r['arm'],r['request_id']))
    if dev:
        best=dev[0];candidates=sorted((e for e in events if e['arm']==best['arm'] and e['request_id']==best['request_id']),key=lambda e:e['rank'])
        if candidates:
            e=candidates[0];key=(e['arm'],e['request_id'],e['rank'])
            if key in selected:selected[key][1].append('MAX_NEW_BULK_REACH_DEV_REQUEST_FIRST_ACCEPTED_OPERATION')
            else:selected[key]=(e,['MAX_NEW_BULK_REACH_DEV_REQUEST_FIRST_ACCEPTED_OPERATION'])
    return [selected[k] for k in sorted(selected)]


def panel(ax,vertices,faces,rows,roles,highlight=None):
    for role,color in COLORS.items():
        rr=[i for i in rows if i<len(faces) and roles[i]==role]
        if rr:ax.add_collection3d(Poly3DCollection(vertices[faces[rr]],facecolor=color,edgecolor='#233039',linewidth=.75,alpha=.65 if role=='BRIDGE' else .4))
    if highlight is not None:
        t=vertices[np.asarray(highlight)];ax.plot(*np.vstack((t,t[0])).T,color='#a01aa0',lw=2.5)


def draw(event,reasons):
    before,after=event['before_vertices'],event['after_vertices'];bf,af=event['before_faces'],event['after_faces'];rows=event['shown_rows']
    pts=np.concatenate((before[bf[rows[rows<len(bf)]]].reshape(-1,3),after[af[rows]].reshape(-1,3)))
    low,high=pts.min(0),pts.max(0);center=(low+high)/2;span=max((high-low).max()*.58,.015)
    fig=plt.figure(figsize=(13,6.6),constrained_layout=True)
    for idx,(vv,ff) in enumerate(((before,bf),(after,af))):
        ax=fig.add_subplot(1,2,idx+1,projection='3d');highlight=event['record'].get('oriented_bridge_vertices') if idx==1 else None
        panel(ax,vv,ff,rows,event['roles'],highlight)
        ax.set_xlim(center[0]-span,center[0]+span);ax.set_ylim(center[1]-span,center[1]+span);ax.set_zlim(center[2]-span,center[2]+span)
        ax.set_box_aspect((1,1,1));ax.view_init(25,-60);ax.set_xlabel('x');ax.set_ylabel('y');ax.set_zlabel('z');ax.tick_params(labelsize=8)
        for op in event['operations']:
            ids=op['original_cluster_ids'];target=np.asarray(op['target'])
            for start in np.unique(before[ids],axis=0):
                ax.plot(*np.vstack((start,target)).T,color='#b42336',lw=1.8)
            ax.scatter(*target,color='#b42336',s=30)
        ax.set_title('Immediately before saved operation' if idx==0 else 'Immediately after accepted operation')
    extra=(f"alpha={event['alpha']}" if event['category']=='ALPHA_NEW_LOCATION' else 'original midpoint / fixed anchor' if event['category']=='ORIGINAL_LOCATION_REPLAY'
           else f"bridge angle={event['record']['minimum_angle_degrees']:.2f} deg; q={event['record']['q']:.2f}")
    fig.suptitle(f"{event['request_id']} | {event['arm']} | accepted rank {event['rank']} | {extra}\n"
        +'Blue=C; gold=seam; teal=bulk; purple=new bridge; red=accepted merge movement\n'
        +'Selected explanatory diagram; entire 28-request evaluation remains separate.',fontsize=11)
    path=OUT/f"{event['arm']}_{event['request_id']}_rank{event['rank']:03d}.png";fig.savefig(path,dpi=160);plt.close(fig)
    return dict(request_id=event['request_id'],split=event['split'],arm=event['arm'],category=event['category'],rank=event['rank'],alpha=event['alpha'],
        selection_reasons=reasons,figure=str(path),source_path=event['source_path'],source_sha256=event['source_sha256'],
        bounds=dict(center=center.tolist(),half_span=float(span),elevation=25,azimuth=-60),shown_face_rows=rows.tolist(),
        actual_view_reviewed=False,scope='Explanatory local render, not evaluation subset or completed AI review',
        request_bulk_fraction_delta=event['gain'],request_newly_reached_original_bulk_faces=event['newbulk'])


def main():
    done=OUT/'completion.json'
    if done.exists():return C.read(done)
    start=time.perf_counter();events,reachrows=collect();OUT.mkdir(parents=True,exist_ok=True);selection=choose(events,reachrows)
    rows=[draw(e,reason) for e,reason in selection];C.csvwrite(OUT/'manifest.csv',rows);C.csvwrite(OUT/'reach_selection_basis.csv',reachrows)
    result=dict(status='COMPLETE',source_sha256=C.sha(__file__),accepted_operation_count=len(events),selected_local_figures=len(rows),
        selection_rule='Per operation class first TRAIN and DEV by request/arm/rank; additionally first accepted operation in largest newly reached DEV request. No phenotype selection.',
        seconds=time.perf_counter()-start,model_calls=0,new_trials=0,new_meshes=0,actual_review_count=0,records=rows,
        files={str(p):C.sha(p) for p in OUT.iterdir() if p.is_file()})
    C.write(done,result);return result


if __name__=='__main__':print(json.dumps(main(),ensure_ascii=False),flush=True)
