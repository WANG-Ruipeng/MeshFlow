"""Independent pure-CPU inspection of already saved conservative proposals.

No retriangulation/simplification/model/scorer calls. Inspects original indexed
faces, saved patch partition and output lineage at the frozen tolerances.
"""
import collections
import csv
import json
import time
import numpy as np
from . import layout_contract as A

OUT=A.ROOT/'experiments/r2_bench_deployment_task_data_v1/rebuild_verification'


def edgebook(faces):
    result=collections.defaultdict(list)
    for fi,tri in enumerate(faces):
        for j in range(3):
            u,v=int(tri[j]),int(tri[(j+1)%3])
            result[tuple(sorted((u,v)))].append((fi,u,v))
    return result


def inspect_patch(v,f,group,global_vertex_faces):
    gf=f[group];eb=edgebook(gf)
    boundary=[(rows[0][1],rows[0][2]) for rows in eb.values() if len(rows)==1]
    out=collections.defaultdict(list);inc=collections.Counter()
    for u,w in boundary:out[u].append(w);inc[w]+=1
    simple=bool(boundary) and set(inc)==set(out) and all(len(x)==1 for x in out.values()) and all(x==1 for x in inc.values())
    loops=[];unseen=set(out)
    if simple:
        while unseen:
            start=min(unseen);loop=[start];cur=out[start][0]
            while cur!=start and cur not in loop:
                loop.append(cur);cur=out[cur][0]
            if cur!=start:simple=False;break
            loops.append(loop);unseen-=set(loop)
    bverts={x for e in boundary for x in e}
    inside=set(map(int,gf.ravel()))-bverts
    exterior_shared=[x for x in inside if not global_vertex_faces[x].issubset(set(group))]
    single=simple and len(loops)==1
    removable=single and len(boundary)-2<len(group) and not exterior_shared
    return dict(faces=len(group),boundary_edges=len(boundary),boundary_vertices=len(bverts),
        simple_directed_boundary=bool(simple),loops=len(loops),interior_vertices=len(inside),
        interior_vertices_shared_outside=len(exterior_shared),
        removable_by_frozen_pre_earclip_rules=bool(removable),
        potential_faces_saved=len(group)-(len(boundary)-2) if single else 0)


def main():
    tick=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    root=OUT.parent;protocol=A.read(root/'REBUILD_PROTOCOL.json');summary=A.read(root/'rebuild_summary.json')
    normal_min=protocol['normal_dot_min'];eps=protocol['plane_tolerance_model']
    inputs={str(root/'REBUILD_PROTOCOL.json'):A.sha(root/'REBUILD_PROTOCOL.json'),
        str(root/'rebuild_summary.json'):A.sha(root/'rebuild_summary.json'),
        str(A.Path(__file__)):A.sha(__file__),str(A.ROOT/'r2/data_tools/bench_deployment_tasks_v1/planar_rebuild.py'):A.sha(A.ROOT/'r2/data_tools/bench_deployment_tasks_v1/planar_rebuild.py')}
    assert inputs[str(A.ROOT/'r2/data_tools/bench_deployment_tasks_v1/planar_rebuild.py')]==protocol['code_sha256']
    rows=[];patches=[]
    for row in summary['rows']:
        uid=row['parent_id'];source=A.resolve(row['source_row']['clean_case_path'])
        lineage=root/'rebuilt_parents'/uid/'lineage.json';target=A.resolve(row['target_case_path'])
        for p in (source,lineage,target):inputs[str(p)]=A.sha(p)
        assert inputs[str(source)]==row['source_row']['clean_case_sha256']
        assert inputs[str(target)]==row['target_case_sha256']
        a=A.arrays(source);result=A.arrays(target);book=A.read(lineage)
        v=np.asarray(a['model_vertices'],np.float64);f=a['source_faces'];tri=v[f]
        normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);norms=np.linalg.norm(normals,axis=1)
        assert np.all(norms>0);normals/=norms[:,None]
        eb=edgebook(f);vfaces=collections.defaultdict(set)
        for fi,tt in enumerate(f):
            for vid in tt:vfaces[int(vid)].add(fi)
        group_of=np.full(len(f),-1,np.int64);local=[]
        for gi,g in enumerate(book['patch_ledger']):
            ids=g['source_clean_face_ids'];assert (group_of[ids]<0).all();group_of[ids]=gi
            # The first sorted face is the production ascending seed. Every
            # member must satisfy the exact frozen seed-normal/plane predicate.
            seed=ids[0]
            assert np.all(normals[ids]@normals[seed]>=normal_min-3e-16)
            assert np.max(np.abs((tri[ids]-tri[seed,0])@normals[seed]))<=eps
            rr=inspect_patch(v,f,ids,vfaces);local.append(rr)
            patches.append(dict(parent_id=uid,patch_id=gi,action=g['action'],**rr))
        assert (group_of>=0).all()
        assert all(x['action']!='COPLANAR_INTERIOR_RETRIANGULATED' for x in book['patch_ledger'])
        assert np.array_equal(a['model_vertices'],result['model_vertices'])
        # Output ordering may change but every original ordered source face remains.
        assert collections.Counter(map(tuple,f))==collections.Counter(map(tuple,result['source_faces']))
        adj=collections.Counter();near=[]
        for edge,faces in eb.items():
            if len(faces)!=2:continue
            i,j=faces[0][0],faces[1][0]
            dot=float(normals[i]@normals[j]);distance=max(float(np.max(np.abs((tri[j]-tri[i,0])@normals[i]))),float(np.max(np.abs((tri[i]-tri[j,0])@normals[j]))))
            norm_ok=dot>=normal_min;plane_ok=distance<=eps
            adj['adjacent_two_face_edges']+=1
            adj['local_both_pass']+=int(norm_ok and plane_ok)
            adj['plane_pass_normal_fail']+=int(plane_ok and not norm_ok)
            adj['normal_pass_plane_fail']+=int(norm_ok and not plane_ok)
            adj['local_pass_but_cross_saved_patch']+=int(norm_ok and plane_ok and group_of[i]!=group_of[j])
        used=np.unique(f);unique_coords=np.unique(v[used],axis=0)
        indexed_open=sum(len(fs)==1 for fs in eb.values())
        # Coordinate-identity shadow is a diagnostic only. It is not applied to
        # assets: merging coincident vertices may cross independent part seams.
        _,inverse=np.unique(v,axis=0,return_inverse=True)
        coord_eb=edgebook(inverse[f]);coordinate_open=sum(len(fs)==1 for fs in coord_eb.values())
        r=dict(parent_id=uid,N=int(len(f)),N_after=int(len(result['source_faces'])),
            total_patches=len(local),multi_face_patches=sum(x['faces']>1 for x in local),
            faces_in_multi_patches=sum(x['faces'] for x in local if x['faces']>1),
            singleton_patches=sum(x['faces']==1 for x in local),
            patches_with_interior_vertices=sum(x['interior_vertices']>0 for x in local),
            total_patch_interior_vertices=sum(x['interior_vertices'] for x in local),
            shared_interior_vertex_patches=sum(x['interior_vertices_shared_outside']>0 for x in local),
            single_simple_loop_patches=sum(x['simple_directed_boundary'] and x['loops']==1 for x in local),
            multi_loop_patches=sum(x['loops']>1 for x in local),
            removable_frozen_rule_patches=sum(x['removable_by_frozen_pre_earclip_rules'] for x in local),
            exact_coincident_used_vertex_excess=len(used)-len(unique_coords),
            indexed_open_edges=indexed_open,coordinate_shadow_open_edges=coordinate_open,
            original_ordered_face_multiset_preserved=True,original_vertices_bitwise_preserved=True,**adj)
        rows.append(r)
    A.write_csv(OUT/'per_parent.csv',rows);A.write_csv(OUT/'per_patch.csv',patches)
    totals={k:sum(int(r.get(k,0)) for r in rows) for k in rows[0] if k not in ('parent_id','original_ordered_face_multiset_preserved','original_vertices_bitwise_preserved')}
    verdict=dict(status='COMPLETE',parents=len(rows),totals=totals,
        action_counts=dict(collections.Counter(p['action'] for p in patches)),
        neural_forwards=0,GPU_calls=0,new_OT=0,new_simplification_proposals=0,
        no_tolerance_changes=True,originals_unchanged=True,
        interpretation='No candidate within the frozen boundary-preserving coplanar-interior contract is missed if removable_frozen_rule_patches=0; this does not rule out other simplification contracts.',
        elapsed_seconds=time.perf_counter()-tick)
    A.write_json(OUT/'summary.json',verdict)
    for p,h in inputs.items():assert A.sha(p)==h
    A.write_json(OUT/'COMPLETE.json',dict(status='COMPLETE',input_hashes=inputs,
        outputs={str(p):A.sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='COMPLETE.json'},
        elapsed_seconds=time.perf_counter()-tick,new_model_calls=0))
    print(json.dumps(verdict,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
