"""Child-aware CPU evaluation, with immutable original geometric definitions.

No fake RAW slots: unique child identity and many-to-one original parent lineage
are separate. Current-face statistics and fixed-original-parent unions coexist.
The two deletion counts refer to missing original parents, not N minus children.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections
import json
from pathlib import Path
import sys
import numpy as np
from r2.pilots.seam_followup_v1 import binding as B
from r2.pilots.seam_hinge_quality_v1 import topology as T

C,E=B.C,B.E
VERSION='hinge_subsegment_child_faces_and_fixed_parent_unions_v1'


def identity_mesh(base):
    """Explicit no-split view for equivalence checks; never invent RAW slots."""
    n=len(base['faces']); k=int(base['known_faces'])
    out=dict(base)
    out.update(child_face_id=np.arange(n,dtype=np.int64),
               parent_post_face_id=np.arange(n,dtype=np.int64),
               parent_source_raw_slot=np.asarray(base['raw_slot_ids'],np.int64).copy(),
               child_seam_mask=np.asarray(base['raw_seam_slot_mask'],bool)[base['raw_slot_ids']],
               base_output_face_count=np.asarray(n),base_free_face_count=np.asarray(n-k))
    return out


def validate(mesh,q):
    tri=np.asarray(mesh['output']);f=np.asarray(mesh['faces']);v=np.asarray(mesh['vertices'])
    K=int(q['K']);n=len(tri);child=np.asarray(mesh['child_face_id'],np.int64)
    parent=np.asarray(mesh['parent_post_face_id'],np.int64)
    raw=np.asarray(mesh['parent_source_raw_slot'],np.int64)
    seam=np.asarray(mesh['child_seam_mask'],bool)
    base=int(mesh['base_output_face_count']); raw_mask=np.asarray(mesh['raw_seam_slot_mask'],bool)
    assert tri.dtype==np.float32 and tri.shape==(n,3,3) and f.shape==(n,3)
    assert np.isfinite(tri).all() and np.array_equal(v[f],tri)
    assert child.shape==parent.shape==raw.shape==seam.shape==(n,) and len(set(child.tolist()))==n
    assert np.array_equal(parent[:K],np.arange(K)) and np.array_equal(raw[:K],np.arange(K))
    assert np.all(parent[K:]>=K) and np.all(parent<base) and np.all(raw>=0) and np.all(raw<len(raw_mask))
    assert not seam[:K].any() and np.array_equal(seam,raw_mask[raw])
    assert len(raw_mask)==int(q['N'])==int(mesh['input_faces'])
    assert int(mesh['known_faces'])==K and int(mesh['base_free_face_count'])==base-K
    for p in np.unique(parent):
        mask=parent==p
        assert len(set(raw[mask].tolist()))==1 and len(set(seam[mask].tolist()))==1
    return tri,f,v,K,child,parent,raw,seam,base


def seam_reach(tri,K,edges,seam,child,parent,raw,raw_seam):
    """Same legal-free-only BFS as original E.seam_to_bulk, child identity."""
    area=.5*np.linalg.norm(np.cross(tri[:,1].astype(float)-tri[:,0],tri[:,2].astype(float)-tri[:,0]),axis=1)
    seams=set(np.flatnonzero(seam).tolist());bulk=set(range(K,len(tri)))-seams
    seeds=set();adj=collections.defaultdict(set);cross=[]
    for e in edges:
        if not e['valid_edge']:continue
        i,j=map(int,e['incident_face_ids'])
        if min(i,j)<K:
            free=max(i,j)
            if free in seams:seeds.add(free)
            continue
        adj[i].add(j);adj[j].add(i)
        if (i in seams)!=(j in seams):cross.append(e)
    seen=set(seeds);todo=sorted(seeds)
    while todo:
        i=todo.pop()
        for j in sorted(adj[i]-seen):
            assert j>=K
            seen.add(j);todo.append(j)
    rb=sorted(seen&bulk);rs=sorted(seen&seams)
    den=float(area[sorted(bulk)].sum());num=float(area[rb].sum());frac=num/den if den else None
    connected=[e for e in cross if all(int(i) in seen for i in e['incident_face_ids'])]
    result=dict(status='PASS' if den>0 else 'NA_EMPTY_OR_ZERO_AREA_BULK',
        retained_seam_faces=len(seams),retained_bulk_faces=len(bulk),
        retained_seam_area=float(area[sorted(seams)].sum()),retained_bulk_area=den,
        legal_CF_seed_seam_faces=len(seeds),legal_CF_seed_seam_area=float(area[sorted(seeds)].sum()),
        free_only_reached_seam_faces=len(rs),free_only_reached_bulk_faces=len(rb),
        free_only_reached_bulk_area=num,free_only_reached_bulk_area_fraction=frac,
        legal_seam_bulk_edge_count=len(cross),legal_seam_bulk_edge_length=sum(e['length_model'] for e in cross),
        seed_connected_legal_seam_bulk_edge_count=len(connected),
        seed_connected_legal_seam_bulk_edge_length=sum(e['length_model'] for e in connected),
        bulk_reach_ge_0_30_descriptive=None if frac is None else int(frac>=.30),
        deleted_seam_faces=int(np.asarray(raw_seam,bool).sum())-len(set(raw[seam].tolist())))
    rows=[dict(output_face_id=i,child_face_id=int(child[i]),parent_post_face_id=int(parent[i]),
               parent_source_raw_slot=int(raw[i]),role='SEAM' if seam[i] else 'BULK',
               area_model2=float(area[i]),legal_CF_seed=i in seeds,reached_free_only=i in seen)
          for i in range(K,len(tri))]
    return result,rows


def child_topology(mesh,K,edges,L):
    """Original 11 graph definitions, evaluated on actual child triangles."""
    tri=np.asarray(mesh['output'],float);faces=np.asarray(mesh['faces'],np.int64)
    parent=np.asarray(mesh['parent_post_face_id']);raw=np.asarray(mesh['parent_source_raw_slot'])
    child=np.asarray(mesh['child_face_id']);seam=np.asarray(mesh['child_seam_mask'],bool);n=len(tri)
    area=.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    by_v=collections.defaultdict(set);by_e=collections.defaultdict(set)
    for i,f in enumerate(faces):
        for v in f:by_v[int(v)].add(i)
        for k in range(3):by_e[tuple(sorted((int(f[k]),int(f[(k+1)%3]))))].add(i)
    def links(groups):
        out=[]
        for group in groups:
            a=sorted(group);out.extend((a[i],a[j]) for i in range(len(a)) for j in range(i+1,len(a)))
        return out
    legal=[tuple(map(int,e['incident_face_ids'])) for e in edges if e['valid_edge']]
    opposed=[tuple(map(int,e['incident_face_ids'])) for e in edges if e['valid_edge'] and e['winding_opposite']]
    graphs={name:T.components(n,pairs,K,area) for name,pairs in (
        ('vertex',links(by_v.values())),('edge_unfiltered',links(by_e.values())),
        ('edge_legal',legal),('edge_legal_opposed',opposed))}
    masks={name:np.asarray([records[int(cid)]['has_C'] for cid in ids],bool)
           for name,(ids,records,adj) in graphs.items()}
    free=np.arange(n)>=K;bulk=free&~seam;den=float(area[free].sum())
    vertex_only=free&masks['vertex']&~masks['edge_legal']
    unfiltered_only=free&masks['vertex']&~masks['edge_unfiltered']
    single=lambda name:[r for r in graphs[name][1] if r['face_count']==1 and r['face_indices'][0]>=K]
    metrics=dict(hinge_diag_isolated_1face_edge_components=len(single('edge_legal')),
        hinge_diag_vertex_attached_only_free_area_fraction=float(area[vertex_only].sum())/den if den else None,
        hinge_diag_bulk_only_valid_shared_edge_count=sum(bool(bulk[i] and bulk[j]) for i,j in legal),
        hinge_diag_isolated_1face_unfiltered_edge_components=len(single('edge_unfiltered')),
        hinge_diag_vertex_attached_only_unfiltered_free_area_fraction=float(area[unfiltered_only].sum())/den if den else None,
        hinge_diag_open_edge_count=sum(len(x)==1 for x in by_e.values()),
        hinge_diag_free_area_denominator=den,
        hinge_diag_vertex_attached_only_free_area_numerator=float(area[vertex_only].sum()),
        hinge_diag_fully_C_disconnected_free_area_fraction=float(area[free&~masks['vertex']].sum())/den if den else None,
        hinge_diag_legal_edge_C_reached_free_area_fraction=float(area[free&masks['edge_legal']].sum())/den if den else None,
        hinge_diag_legal_opposed_edge_C_reached_free_area_fraction=float(area[free&masks['edge_legal_opposed']].sum())/den if den else None)
    rows=[]
    for i in range(n):
        row=dict(output_face_id=i,child_face_id=int(child[i]),parent_post_face_id=int(parent[i]),
                 parent_source_raw_slot=int(raw[i]),role='C' if i<K else 'SEAM' if seam[i] else 'BULK',
                 area_model2=float(area[i]),vertex_ids=faces[i].tolist())
        for name,(ids,records,adj) in graphs.items():
            r=records[int(ids[i])]
            row.update({name+'_'+key:r[key] for key in ('component_id','face_count','has_C','area_model2')})
        rows.append(row)
    return metrics,rows


def fixed_parent_unions(pairs,shape,K,parent,base_count):
    P=base_count-K;stats=dict(fixed_parent_free_face_denominator=P);detail={}
    for kind in ('CF','FF'):
        affected=set();parent_pairs=set();siblings=[]
        for r in pairs[kind]['contacts']:
            if r['classification'] not in B.ILLEGAL:continue
            i,j=map(int,r['pair']);j+=K
            if kind=='FF':i+=K
            if kind=='CF':affected.add(int(parent[j]))
            else:affected.update((int(parent[i]),int(parent[j])))
            key=(int(parent[i]),int(parent[j])) if kind=='CF' else tuple(sorted((int(parent[i]),int(parent[j]))))
            parent_pairs.add(key)
            if kind=='FF' and key[0]==key[1]:siblings.append(dict(child_output_pair=[i,j],parent=key[0],classification=r['classification']))
        stats.update({f'fixed_parent_{kind}_illegal_affected_count':len(affected),
                      f'fixed_parent_{kind}_illegal_affected_fraction':len(affected)/P if P else None,
                      f'fixed_parent_{kind}_illegal_unique_pair_count':len(parent_pairs)})
        detail[kind]=dict(affected_parent_post_ids=sorted(affected),illegal_parent_pairs=sorted(parent_pairs),
                          illegal_sibling_child_pairs=siblings)
    stats['illegal_sibling_child_pair_count']=len(detail['FF']['illegal_sibling_child_pairs'])
    for key,name in (('minimum_angle_lt5','minimum_angle_lt5'),('q_actual_gt10','q_gt10'),
                     ('area_near_degenerate','near_degenerate'),('area_exact_zero','exact_zero_area')):
        bad={int(parent[K+int(r['face'])]) for r in shape['free']['per_face'] if r[key]}
        stats['fixed_parent_'+name+'_count']=len(bad)
        stats['fixed_parent_'+name+'_fraction']=len(bad)/P if P else None
        detail[name]=sorted(bad)
    return stats,detail


def score(q,mesh,dest=None,label='SUBSEGMENT_SPLIT'):
    """Return full row/result and optionally write a fresh CPU score directory."""
    tri,f,v,K,child,parent,raw,seam,base=validate(mesh,q)
    old,A,G,H,quality=E.helpers();mod=old.cpu_math()
    inp=C.arrays(q['inference_npz']);metadata=C.arrays(q['metadata_npz'])
    cc=inp['C_vertices'][inp['C_faces_model']];original=inp['C_vertices'][inp['C_faces_original']]
    if not old.byte_equal(tri[:K],original):raise ValueError('Original C bytes/order changed')
    F=tri[K:];metrics,cf,ff=mod['geometry'].scan_geometry(cc,F,C.L)
    pairs=dict(CF=cf,FF=ff);reference=old.load_queries(q['query_descriptor'])
    trusted=mod['control'].trusted_cf(cc,F,C.L,cf_scan=cf)
    im,iv=mod['control'].conflict_metrics(F,dict(C=cc,L0=C.L),reference,trusted);metrics.update(im)
    for kind,scan in pairs.items():
        bad=[r for r in scan['contacts'] if r['classification'] in B.ILLEGAL]
        affected={int(r['pair'][1]) for r in bad} if kind=='CF' else {int(i) for r in bad for i in r['pair']}
        metrics.update({kind+'_illegal_pair_count':len(bad),kind+'_illegal_affected_free_fraction':len(affected)/len(F) if len(F) else None})
    shape={name:mod['shape'].shape_quality(part,C.L) for name,part in (('C',cc),('free',F),('whole',tri))}
    for name,s in shape.items():
        metrics.update({name+'_minimum_angle_lt5_fraction':s['min_angle_lt5_fraction'],name+'_q_gt10_fraction':s['q_actual_gt10_fraction'],
            name+'_near_degenerate_count':s['near_degenerate_area_count'],name+'_exact_zero_area_count':s['exact_zero_area_count'],name+'_total_surface_area':s['total_area']})
    # Original N remains the generator request. Deletions count original parents;
    # splitting a face is never encoded as negative deletion or a new RAW slot.
    metrics.update(mod['geometry'].topology(tri,K),actual_free_faces=len(F),
                   deleted_free_faces=int(q['N'])-len(set(raw.tolist())))
    graph,edges,_=G.actual_valid_graph(v,f,K,pairs)
    graph.update(H.opposite_reach(tri,K,edges),C_F_has_opposite_winding_edge=int(graph['C_F_valid_opposite_winding_edge_count']>0),domain=str(mesh['index_domain']))
    internal,seam_rows=seam_reach(tri,K,edges,seam,child,parent,raw,mesh['raw_seam_slot_mask'])
    source_ids=metadata['C_source_vertex_ids'][inp['C_faces_original']]
    _,points,_=A.topology(original,source_ids);dist,_=A.nearest(points,F)
    boundary=[dict(setting=n,threshold_model=t,scale=s,tau=value,domain='ALL_C_SOURCE_TOPOLOGY_OPEN_BOUNDARY',**A.metric(dist,t)) for n,t,s,value in A.SETTINGS]
    distance={key:float(fn(dist)) if len(dist) and np.isfinite(dist).all() else None for key,fn in (
        ('boundary_distance_mean_model',np.mean),('boundary_distance_median_model',np.median),
        ('boundary_distance_p95_model',lambda x:np.quantile(x,.95)),('boundary_distance_max_model',np.max))}
    result=dict(**{k:q[k] for k in E.META},label=label,stage='POST_SPLIT',status='PASS',version=VERSION,
        metrics=metrics,topology=graph,seam_to_bulk=internal,boundary=boundary,distance=distance,
        C_preserved_exact=True,C_original_order_preserved=True,C_query_count=2048,
        C_query_weight_denominator=float(reference['arrays']['C_weights'].sum()),free_face_denominator=len(F),
        CF_tested_pair_denominator=K*len(F),FF_tested_pair_denominator=len(F)*(len(F)-1)//2,
        status_by_metric=dict(I=im['ConflictCov_C_status'],S='NA_NO_REQUIRED_INTERFACE',E='NA_NO_EXPOSURE',GT='NA_NOT_UNIQUE_GENERATIVE_TARGET',semantic_use='AI_PROXY_REVIEW_SEPARATE'),
        deletion_semantics='Original RAW parents absent from current output, not requested N minus child faces')
    row=B.flatten(result,label);roles=collections.Counter();role_rows=[]
    for kind,scan in pairs.items():
        for r in scan['contacts']:
            if r['classification'] not in B.ILLEGAL:continue
            i,j=map(int,r['pair']);j+=K
            if kind=='FF':
                i+=K;role='seam-seam' if seam[i] and seam[j] else 'seam-bulk' if seam[i] or seam[j] else 'bulk-bulk'
            else:role='C-seam' if seam[j] else 'C-bulk'
            roles[role]+=1
            role_rows.append(dict(kind=kind,role=role,child_face_i=int(child[i]),child_face_j=int(child[j]),
                                  parent_post_i=int(parent[i]),parent_post_j=int(parent[j]),classification=r['classification']))
    for role in ('seam-seam','seam-bulk','bulk-bulk','C-seam','C-bulk'):row['illegal_pairs_'+role]=roles[role]
    diag,topo_rows=child_topology(mesh,K,edges,C.L);row.update(diag)
    internal_edges=[e for e in edges if e['valid_edge'] and len(e['incident_face_ids'])==2
                    and parent[int(e['incident_face_ids'][0])]==parent[int(e['incident_face_ids'][1])]]
    external_edges=[e for e in edges if e['valid_edge'] and len(e['incident_face_ids'])==2
                    and parent[int(e['incident_face_ids'][0])]!=parent[int(e['incident_face_ids'][1])]]
    row.update(subdivision_internal_valid_edge_count=len(internal_edges),
               subdivision_internal_valid_edge_length=sum(e['length_model'] for e in internal_edges),
               external_valid_edge_count=len(external_edges),
               external_valid_edge_length=sum(e['length_model'] for e in external_edges))
    fixed,detail=fixed_parent_unions(pairs,shape,K,parent,base)
    row.update(fixed,original_POST_free_faces=base-K,added_child_faces=len(tri)-len(set(parent.tolist())),
               deleted_original_POST_faces=base-len(set(parent.tolist())),scoring_definition_version=VERSION)
    policy=C.read(C.ROOT/'r2/tools/hinge_opposite_v1/metric_policy.json')
    missing=set(policy['directions'])-set(row)
    if missing:raise AssertionError('Missing inherited numeric fields: '+str(sorted(missing)))
    assert 'torch' not in sys.modules
    payload=dict(score=result,row=row,pairs=pairs,shape=shape,edges=edges,query_vectors=iv,
                 fixed_parent_detail=detail,seam_rows=seam_rows,topology_rows=topo_rows,role_rows=role_rows)
    if dest is not None:
        dest=C.resolve(dest)
        if dest.exists() and any(dest.iterdir()):raise FileExistsError('Refuse existing score directory '+str(dest))
        dest.mkdir(parents=True,exist_ok=True)
        for name,value in (('score',result),('row',row),('pairs',pairs),('shape_per_face',shape),('fixed_parent_detail',detail)):
            C.write(dest/(name+'.json'),value)
        for name,value in (('valid_edges',edges),('seam_bulk_faces',seam_rows),('per_face_connectivity',topo_rows),('illegal_pairs_by_role',role_rows)):
            C.csvwrite(dest/(name+'.csv'),value)
        E.save_npz(dest/'vectors.npz',iv)
        C.write(dest/'complete.json',dict(status='PASS',version=VERSION,row=row,model_calls=0,new_queries=0,
                    files={str(p):C.sha(p) for p in dest.iterdir() if p.is_file()}))
    return payload


def check_baseline(request_id='608b46b0_n0'):
    root=C.ROOT/'experiments/r2_hinge_subsegment_v1'
    dest=root/'scoring_equivalence'/request_id
    source=C.ROOT/'experiments/r2_hinge_opposite_v1'
    q=next(q for q in C.read(source/'protocol.json')['requests'] if q['request_id']==request_id)
    cell=source/'cells/HINGE_OPP015'/request_id
    base=C.arrays(cell/'post.npz');previous=C.read(cell/'complete.json')['row']
    C.resources(root/'scoring_equivalence/resources_before.json')
    payload=score(q,identity_mesh(base),dest/'score',label='BASELINE_HINGE_OPP015')
    fields=C.read(C.ROOT/'r2/tools/hinge_opposite_v1/metric_policy.json')['directions']
    comparisons=[]
    for key in fields:
        a=previous[key];b=payload['row'][key]
        same=a==b or (a is not None and b is not None and np.isclose(a,b,rtol=1e-12,atol=1e-12))
        comparisons.append(dict(field=key,old=a,new=b,equal=bool(same)))
    result=dict(status='PASS' if all(r['equal'] for r in comparisons) else 'FAIL',
                fields=len(comparisons),request_id=request_id,comparisons=comparisons,
                old_mesh_sha256=C.sha(cell/'post.npz'),version=VERSION,model_calls=0,new_queries=0)
    C.write(dest/'comparison.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='comparisons'},indent=2))
    assert result['status']=='PASS',[r for r in comparisons if not r['equal']]
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check-baseline'])
    p.add_argument('--request',default='608b46b0_n0');a=p.parse_args()
    check_baseline(a.request)
