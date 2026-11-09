"""Read-only component/profile facts for one bounded per-part fallback.

No final target, simplifier, model, or source mutation is produced.
"""
from __future__ import annotations
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_three_family_v1'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def loops(f):
    edges=collections.defaultdict(list)
    for fi,face in enumerate(f):
        for j in range(3):edges[tuple(sorted([int(face[j]),int(face[(j+1)%3])]))].append((fi,int(face[j]),int(face[(j+1)%3])))
    boundary=[x[0][1:] for x in edges.values() if len(x)==1]
    outgoing={a:b for a,b in boundary};result=[];unused=set(outgoing)
    while unused:
        start=min(unused);cur=start;path=[]
        while cur in unused:
            unused.remove(cur);path.append(cur);cur=outgoing[cur]
        result.append(dict(vertices=path,closed=cur==start))
    return result

def profile(v,f):
    ids=np.unique(f);xyz=v[ids];side=xyz[:,0]<0
    li,ri=ids[side],ids[~side];d=cdist(v[li,1:],v[ri,1:]);j=d.argmin(1)
    boundary=loops(f)
    spans=np.ptp(v[f][:,:,0],axis=1)
    cross=f[spans>1.]
    constxmax=max(np.ptp(v[li,0]),np.ptp(v[ri,0]))/2
    return dict(vertex_records=[dict(id=int(i),xyz=list(map(float,v[i])),side='L' if v[i,0]<0 else 'R') for i in ids],
        left_vertices=list(map(int,li)),right_vertices=list(map(int,ri)),
        boundaries=boundary,boundary_xyz=[v[np.asarray(x['vertices'])].tolist() for x in boundary],
        x_spanning_faces=len(cross),same_end_faces=len(f)-len(cross),
        nearest_YZ_left_to_right=[dict(left=int(i),right=int(ri[q]),distance_model=float(d[k,q])) for k,(i,q) in enumerate(zip(li,j))],
        yz_pair_max=float(d.min(1).max()),yz_reverse_pair_max=float(d.min(0).max()),
        left_x_range=[float(v[li,0].min()),float(v[li,0].max())],right_x_range=[float(v[ri,0].min()),float(v[ri,0].max())],
        constant_x_boundary_vertex_displacement_lower_bound_model=float(constxmax),
        note='The lower bound is vertex displacement under same-ID plane flattening only, not a finite-surface Hausdorff lower bound.')

def main():
    path=OUT/'part_recipe_probe.json'
    if path.exists():
        old=read(path)
        for p,h in old['protected'].items():assert sha(ROOT/p)==h
        print(json.dumps(dict(status='VERIFIED',path=str(path))));return
    protocol=read(OUT/'PROTOCOL.json');rows=[];protected={}
    for p in protocol['parents']:
        uid=p['parent_id']
        if not uid.startswith(('27eb79','945214')):continue
        src=Path(p['source']['clean_case_path']);inspection=OUT/'part_inspection'/(uid+'.json')
        for x in (src,inspection):protected[str(x.relative_to(ROOT))]=sha(x)
        with np.load(src,allow_pickle=False) as z:v=z['model_vertices'].astype(float);f=z['source_faces'].astype(int)
        comps=read(inspection)['components'];pieces=[]
        for c in comps:
            cf=f[c['face_ids']];ids=np.unique(cf);xyz=v[ids]
            # Exact axis-normal support structure only, no clustering threshold sweep.
            normal=np.cross(v[cf[:,1]]-v[cf[:,0]],v[cf[:,2]]-v[cf[:,0]])
            normal/=np.linalg.norm(normal,axis=1)[:,None]
            aligned=np.abs(normal).max(1)>1-1e-7
            r=dict(component_id=c['component_id'],N=len(cf),V=len(ids),
                axis_aligned_faces=int(aligned.sum()),axis_aligned_fraction=float(aligned.mean()),
                boundary_loops=loops(cf),source_face_ids=c['face_ids'])
            if uid.startswith('27eb79') and c['component_id']==1:r['X_extrusion_profile']=profile(v,cf)
            pieces.append(r)
        rows.append(dict(parent_id=uid,components=pieces))
    protected[str(Path(__file__).relative_to(ROOT))]=sha(__file__)
    payload=dict(schema='BENCH_PART_RECIPE_READONLY_PROFILE_V1',rows=rows,protected=protected,
                 final_target_written=False,source_mutations=0,models_loaded=0,model_forwards=0,simplifier_calls=0)
    with path.open('x',encoding='utf-8') as stream:json.dump(payload,stream,ensure_ascii=False,indent=2);stream.write('\n')
    for r in rows:
        for c in r['components']:
            q=c.get('X_extrusion_profile',{})
            print(json.dumps(dict(parent=r['parent_id'][:6],component=c['component_id'],N=c['N'],axis_aligned_faces=c['axis_aligned_faces'],
                boundary_lengths=[len(x['vertices']) for x in c['boundary_loops']],profile={k:v for k,v in q.items() if k in ('left_x_range','right_x_range','x_spanning_faces','same_end_faces','yz_pair_max','yz_reverse_pair_max','constant_x_boundary_vertex_displacement_lower_bound_model')})))

if __name__=='__main__':main()
