"""Eight frozen meshes: endpoint vs finite-segment diagnosis only. No mutations."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
import sys,json,csv,time,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from r2.tools.partial_edge_binding_v1 import assets
OUT=Path(__file__).resolve().parent
L=float(np.sqrt(3)*2/.3762)

def inc(faces,start=0):
    d=defaultdict(list)
    for f in range(start,len(faces)):
        for a,b in ((0,1),(1,2),(2,0)):
            d[tuple(sorted(map(int,faces[f,[a,b]])))].append((f,a,b))
    return d

def seg(points,ends):
    v=ends[1]-ends[0]
    t=(points-ends[0])@v/(v@v)
    q=ends[0]+np.clip(t,0,1)[...,None]*v
    return np.linalg.norm(points-q,axis=-1),t

def save(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    started=time.perf_counter()
    manifest=json.loads((ROOT/'experiments/r2_partial_edge_binding_v1/input_manifest.json').read_text())
    rows=[r for r in manifest['rows'] if r['short_id'].startswith(('608','610'))]
    assert len(rows)==8
    point_rows=[];edge_rows=[];summary=[];protected={}
    for row in rows:
        a=assets.load_asset(row);K=row['K'];v=a['vertices'].astype(float);f=a['faces'];sf=a['source_face_ids']
        pre=a['pre_geo'].reshape(-1,3).astype(float)
        ce=[e for e,o in sorted(inc(f[:K]).items()) if len(o)==1]
        fe=[(e,o[0]) for e,o in sorted(inc(f,K).items()) if len(o)==1]
        retained=set(map(int,sf[K:]));open_corners={3*int(sf[fi])+j for _,(fi,b,c) in fe for j in (b,c)}
        cp=list(range(3*K,len(pre)))
        distances=np.stack([seg(pre[cp],v[list(e)])[0] for e in ce],axis=1)
        nearest=np.argmin(distances,axis=1)
        cverts=np.unique(f[:K]);ndv=np.linalg.norm(pre[cp,None,:]-v[cverts][None,:,:],axis=-1).min(1)
        local_points=[]
        for j,c in enumerate(cp):
            ci=int(nearest[j]);dist=float(distances[j,ci]);ends=v[list(ce[ci])];_,t=seg(pre[c:c+1],ends)
            r=dict(short_id=row['short_id'],source_face_id=c//3,corner=c%3,retained=c//3 in retained,
                   is_open_edge_corner=c in open_corners,C_edge_index=ci,C_vertex0=ce[ci][0],C_vertex1=ce[ci][1],
                   point_segment_pctL=dist/L*100,nearest_C_vertex_pctL=float(ndv[j])/L*100,
                   projection_t=float(t[0]),projection_interior=bool(0<t[0]<1),
                   **{axis:float(pre[c,k]) for k,axis in enumerate(('x','y','z'))})
            local_points.append(r)
        point_rows.extend(local_points)
        local=[]
        for e,(fi,j,k) in fe:
            pids=[3*int(sf[fi])+j,3*int(sf[fi])+k];pp=pre[pids];post=v[list(f[fi,[j,k]])]
            axis=post[1]-post[0];length=np.linalg.norm(axis)
            for ci,be in enumerate(ce):
                ends=v[list(be)];bv=ends[1]-ends[0];bl=np.linalg.norm(bv)
                cos=abs(float(axis@bv/(length*bl))) if length*bl else 0.
                d,t=seg(pp,ends);dp,tp=seg(post,ends)
                endpoint=min(float(np.linalg.norm(pp-ends,axis=1).max()),float(np.linalg.norm(pp-ends[::-1],axis=1).max()))
                endpointpost=min(float(np.linalg.norm(post-ends,axis=1).max()),float(np.linalg.norm(post-ends[::-1],axis=1).max()))
                local.append(dict(short_id=row['short_id'],source_face_id=int(sf[fi]),face_row=fi,corner0=j,corner1=k,
                    free_vertex0=int(f[fi,j]),free_vertex1=int(f[fi,k]),C_edge_index=ci,C_vertex0=be[0],C_vertex1=be[1],
                    axis_degrees=float(np.degrees(np.arccos(np.clip(cos,-1,1)))),axis_pass=bool(cos>np.cos(np.pi/6)),
                    endpoint_match_pctL=endpoint/L*100,POST_endpoint_match_pctL=endpointpost/L*100,
                    segment_max_pctL=float(d.max())/L*100,segment_min_pctL=float(d.min())/L*100,
                    POST_segment_max_pctL=float(dp.max())/L*100,project_t0=float(t[0]),project_t1=float(t[1]),
                    both_project_interior=bool(np.all((t>0)&(t<1))),projected_overlap_fraction=float(max(0,min(1,t.max())-max(0,t.min()))),
                    free_length_pctL=float(length/L*100),C_length_pctL=float(bl/L*100),free_to_C_length_ratio=float(length/bl)))
        edge_rows.extend(local)
        aligned=[e for e in local if e['axis_pass']]
        near=[e for e in local if e['segment_max_pctL']<2]
        nearaligned=[e for e in near if e['axis_pass']]
        witnesses=sorted(nearaligned,key=lambda e:(e['segment_max_pctL'],e['endpoint_match_pctL'],e['source_face_id']))
        pr=[r for r in local_points if r['retained']];op=[r for r in pr if r['is_open_edge_corner']]
        s=dict(short_id=row['short_id'],K=K,free_open_edges=len(fe),C_open_edges=len(ce),
               all_pre_corners_min_segment_pctL=min(r['point_segment_pctL'] for r in local_points),
               retained_corners_min_segment_pctL=min(r['point_segment_pctL'] for r in pr),
               retained_corners_near2=sum(r['point_segment_pctL']<2 for r in pr),
               open_corners_near2=sum(r['point_segment_pctL']<2 for r in op),
               open_corners_interior_near2_far_Cvertices35=sum(r['point_segment_pctL']<2 and r['projection_interior'] and r['nearest_C_vertex_pctL']>3.5 for r in op),
               edge_pairs_near_segment2=len(near),axis_pass_edge_pairs_near_segment2=len(nearaligned),
               same_pairs_endpoint_over35=sum(e['endpoint_match_pctL']>3.5 for e in nearaligned),
               min_endpoint_match_any_axis_pctL=min(e['endpoint_match_pctL'] for e in local),
               min_endpoint_match_axis_pass_pctL=min(e['endpoint_match_pctL'] for e in aligned),
               original_candidate_count=sum(e['endpoint_match_pctL']<=3.5 for e in aligned),
               C_edges_with_near_aligned_pairs=len({e['C_edge_index'] for e in nearaligned}))
        assert s['original_candidate_count']==0
        if witnesses:
            for key in ('source_face_id','corner0','corner1','C_edge_index','segment_max_pctL','endpoint_match_pctL','axis_degrees','free_to_C_length_ratio','project_t0','project_t1'):
                s['witness_'+key]=witnesses[0][key]
        summary.append(s)
        for key in ('pre_geo_path','canonical_path','post_path','condition_path'):
            path=assets.resolve(row[key]);protected[str(path)]=assets.sha(path)
    save('point_distances.csv',point_rows);save('edge_pairs.csv',edge_rows);save('summary.csv',summary)
    assert 'torch' not in sys.modules
    for path,h in protected.items(): assert assets.sha(Path(path))==h
    (OUT/'input_hashes.json').write_text(json.dumps(protected,indent=2))
    (OUT/'execution.json').write_text(json.dumps(dict(status='COMPLETE',seconds=time.perf_counter()-started,requests=8,
        point_rows=len(point_rows),edge_pair_rows=len(edge_rows),L=L,CPU_workers=1,threads=2,GPU_calls=0,model_calls=0,
        mesh_mutations=0,protected_files=len(protected),source_hash=assets.sha(Path(__file__))),indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
