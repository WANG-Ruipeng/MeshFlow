"""CPU scoring with BRIDGE as a separate role and absent RAW provenance.

Original numeric quality mathematics is unchanged. Actual-face denominators
include new triangles; fixed-parent unions exclude bridges from the denominator
and separately retain all bridge conflicts and shape statistics. Original bulk
area is never inflated by new bridge area. No model, sampling or mesh edits.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections
import json
from pathlib import Path
import sys
import numpy as np
from . import holes
from r2.pilots.seam_hinge_quality_v1 import topology as T
from r2.pilots.seam_followup_v1 import binding as B

C,E=holes.C,B.E
VERSION='hinge_v2_original_quality_bridge_role_fixed_parent_v1'


def validate(mesh,q):
    m=holes.initial(mesh);v=m['vertices'];f=m['faces'];tri=m['output'];K=int(q['K']);n=len(f)
    ids=np.asarray(m['output_face_id'],np.int64);parent=np.asarray(m['parent_post_face_id'],np.int64)
    raw=np.asarray(m['parent_source_raw_slot'],np.int64);role=np.asarray(m['face_role']).astype(str)
    bridge=role=='BRIDGE';base=int(m['base_output_face_count'])
    assert tri.dtype==np.float32 and tri.shape==(n,3,3) and f.shape==(n,3) and np.isfinite(tri).all()
    assert np.array_equal(v[f],tri) and ids.shape==parent.shape==raw.shape==role.shape==(n,)
    assert len(set(ids.tolist()))==n and set(role)<=set(('C','SEAM','BULK','BRIDGE'))
    assert (role[:K]=='C').all() and not (role[K:]=='C').any()
    assert np.array_equal(parent[:K],np.arange(K)) and np.array_equal(raw[:K],np.arange(K))
    assert (parent[bridge]==-1).all() and (raw[bridge]==-1).all()
    assert (parent[~bridge]>=0).all() and (parent[~bridge]<base).all()
    assert len(set(parent[~bridge].tolist()))==sum(~bridge)
    assert int(m['known_faces'])==K and int(m['base_free_face_count'])==base-K
    assert int(m['input_faces'])==len(m['raw_seam_slot_mask'])==int(q['N'])
    assert np.array_equal(np.asarray(m['raw_seam_slot_mask'],bool)[raw[~bridge]],role[~bridge]=='SEAM')
    assert np.array_equal(m['bridge_mask'],bridge)
    return m,tri,v,f,K,ids,parent,raw,role,bridge,base


def role_reach(tri,K,edges,role,ids,parent,raw,raw_seam):
    """Original free-only legal BFS; bridge may be traversed but is not bulk."""
    area=.5*np.linalg.norm(np.cross(tri[:,1].astype(float)-tri[:,0],tri[:,2].astype(float)-tri[:,0]),axis=1)
    seams=set(np.flatnonzero(role=='SEAM').tolist());bulk=set(np.flatnonzero(role=='BULK').tolist())
    bridges=set(np.flatnonzero(role=='BRIDGE').tolist());seeds=set();adj=collections.defaultdict(set);cross=[]
    for e in edges:
        if not e['valid_edge']:continue
        i,j=map(int,e['incident_face_ids'])
        if min(i,j)<K:
            z=max(i,j)
            if z in seams:seeds.add(z)
            continue
        adj[i].add(j);adj[j].add(i)
        if (i in seams and j in bulk) or (j in seams and i in bulk):cross.append(e)
    seen=set(seeds);todo=sorted(seeds)
    while todo:
        i=todo.pop()
        for j in sorted(adj[i]-seen):
            assert j>=K
            seen.add(j);todo.append(j)
    rb=sorted(seen&bulk);rs=sorted(seen&seams);rh=sorted(seen&bridges)
    den=float(area[sorted(bulk)].sum());num=float(area[rb].sum());frac=num/den if den else None
    connected=[e for e in cross if all(int(i) in seen for i in e['incident_face_ids'])]
    result=dict(status='PASS' if den>0 else 'NA_EMPTY_OR_ZERO_AREA_BULK',
        retained_seam_faces=len(seams),retained_bulk_faces=len(bulk),
        retained_seam_area=float(area[sorted(seams)].sum()),retained_bulk_area=den,
        legal_CF_seed_seam_faces=len(seeds),legal_CF_seed_seam_area=float(area[sorted(seeds)].sum()),
        free_only_reached_seam_faces=len(rs),free_only_reached_bulk_faces=len(rb),
        free_only_reached_bulk_area=num,free_only_reached_bulk_area_fraction=frac,
        legal_seam_bulk_edge_count=len(cross),legal_seam_bulk_edge_length=sum(e['length_model'] for e in cross),
        seed_connected_legal_seam_bulk_edge_count=len(connected),seed_connected_legal_seam_bulk_edge_length=sum(e['length_model'] for e in connected),
        bulk_reach_ge_0_30_descriptive=None if frac is None else int(frac>=.30),
        deleted_seam_faces=int(np.asarray(raw_seam,bool).sum())-len(set(raw[role=='SEAM'].tolist())))
    bridge_stats=dict(bridge_face_count=len(bridges),bridge_area=float(area[sorted(bridges)].sum()),
        free_only_reached_bridge_faces=len(rh),free_only_reached_bridge_area=float(area[rh].sum()))
    for other in ('C','SEAM','BULK','BRIDGE'):
        ee=[e for e in edges if e['valid_edge'] and len(e['incident_face_ids'])==2
            and sorted(role[list(map(int,e['incident_face_ids']))].tolist())==sorted(['BRIDGE',other])]
        bridge_stats['legal_bridge_'+other.lower()+'_edge_count']=len(ee)
        bridge_stats['legal_bridge_'+other.lower()+'_edge_length']=sum(e['length_model'] for e in ee)
    rows=[dict(output_face_index=i,output_face_id=int(ids[i]),parent_post_face_id=int(parent[i]),
               parent_source_raw_slot=int(raw[i]),role=str(role[i]),area_model2=float(area[i]),
               legal_CF_seed=i in seeds,reached_free_only=i in seen) for i in range(K,len(tri))]
    return result,bridge_stats,rows


def topology(tri,faces,K,edges,role):
    """The same 11 graph fields on actual faces, with true BULK role only."""
    tri=np.asarray(tri,float);n=len(tri);area=.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
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
        ('vertex',links(by_v.values())),('edge_unfiltered',links(by_e.values())),('edge_legal',legal),('edge_legal_opposed',opposed))}
    masks={name:np.asarray([records[int(cid)]['has_C'] for cid in ids],bool) for name,(ids,records,adj) in graphs.items()}
    free=np.arange(n)>=K;bulk=role=='BULK';den=float(area[free].sum())
    vertex_only=free&masks['vertex']&~masks['edge_legal'];unfiltered_only=free&masks['vertex']&~masks['edge_unfiltered']
    single=lambda name:[r for r in graphs[name][1] if r['face_count']==1 and r['face_indices'][0]>=K]
    metrics=dict(hinge_diag_isolated_1face_edge_components=len(single('edge_legal')),
        hinge_diag_vertex_attached_only_free_area_fraction=float(area[vertex_only].sum())/den if den else None,
        hinge_diag_bulk_only_valid_shared_edge_count=sum(bool(bulk[i] and bulk[j]) for i,j in legal),
        hinge_diag_isolated_1face_unfiltered_edge_components=len(single('edge_unfiltered')),
        hinge_diag_vertex_attached_only_unfiltered_free_area_fraction=float(area[unfiltered_only].sum())/den if den else None,
        hinge_diag_open_edge_count=sum(len(x)==1 for x in by_e.values()),hinge_diag_free_area_denominator=den,
        hinge_diag_vertex_attached_only_free_area_numerator=float(area[vertex_only].sum()),
        hinge_diag_fully_C_disconnected_free_area_fraction=float(area[free&~masks['vertex']].sum())/den if den else None,
        hinge_diag_legal_edge_C_reached_free_area_fraction=float(area[free&masks['edge_legal']].sum())/den if den else None,
        hinge_diag_legal_opposed_edge_C_reached_free_area_fraction=float(area[free&masks['edge_legal_opposed']].sum())/den if den else None)
    return metrics,graphs


def parent_unions(pairs,shape,K,parent,bridge,base):
    P=base-K;stats=dict(fixed_parent_free_face_denominator=P);detail={}
    for kind,scan in pairs.items():
        affected=set();original_pairs=set();bridge_pairs=[];bridge_affected=set()
        for r in scan['contacts']:
            if r['classification'] not in B.ILLEGAL:continue
            i,j=map(int,r['pair']);j+=K
            if kind=='FF':i+=K
            for z in (i,j):
                if z>=K:
                    if bridge[z]:bridge_affected.add(z)
                    else:affected.add(int(parent[z]))
            if bridge[i] or bridge[j]:bridge_pairs.append(dict(pair=[i,j],classification=r['classification']))
            else:original_pairs.add(tuple(sorted((int(parent[i]),int(parent[j])))))
        stats.update({f'fixed_parent_{kind}_illegal_affected_count':len(affected),f'fixed_parent_{kind}_illegal_affected_fraction':len(affected)/P if P else None,
                      f'fixed_parent_{kind}_illegal_unique_pair_count':len(original_pairs),f'bridge_{kind}_illegal_pair_count':len(bridge_pairs),
                      f'bridge_{kind}_illegal_affected_count':len(bridge_affected)})
        detail[kind]=dict(original_affected_parents=sorted(affected),original_illegal_parent_pairs=sorted(original_pairs),bridge_illegal_pairs=bridge_pairs)
    for key,name in (('minimum_angle_lt5','minimum_angle_lt5'),('q_actual_gt10','q_gt10'),('area_near_degenerate','near_degenerate'),('area_exact_zero','exact_zero_area')):
        bad={int(parent[K+int(r['face'])]) for r in shape['free']['per_face'] if r[key] and not bridge[K+int(r['face'])]}
        newbad={K+int(r['face']) for r in shape['free']['per_face'] if r[key] and bridge[K+int(r['face'])]}
        stats.update({f'fixed_parent_{name}_count':len(bad),f'fixed_parent_{name}_fraction':len(bad)/P if P else None,f'bridge_{name}_count':len(newbad)})
        detail[name]=dict(original_affected_parents=sorted(bad),bridge_affected_output_faces=sorted(newbad))
    return stats,detail


def score(q,mesh,dest=None,label='HINGE_V2',existing_pairs=None):
    m,tri,v,f,K,ids,parent,raw,role,bridge,base=validate(mesh,q)
    old,A,G,H,quality=E.helpers();mod=old.cpu_math()
    inp=C.arrays(q['inference_npz']);metadata=C.arrays(q['metadata_npz'])
    cc=inp['C_vertices'][inp['C_faces_model']];original=inp['C_vertices'][inp['C_faces_original']]
    assert old.byte_equal(tri[:K],original)
    F=tri[K:]
    if existing_pairs is None:
        metrics,cf,ff=mod['geometry'].scan_geometry(cc,F,C.L);pairs=dict(CF=cf,FF=ff)
        pair_origin='RECOMPUTED_FULL_ORIGINAL_DETECTOR'
    else:
        if bridge.any():raise ValueError('New BRIDGE requires independent full original scoring for incremental-pair audit')
        pairs=existing_pairs;cf,ff=pairs['CF'],pairs['FF'];metrics={}
        assert cf['tested_pairs']==K*len(F) and ff['tested_pairs']==len(F)*(len(F)-1)//2
        for name,scan in pairs.items():
            for key,source in (('proper_pair_count','proper_pair_count'),('coplanar_pair_count','coplanar_positive_area_pairs'),
                               ('proper_affected_free_fraction','proper_affected_face_fraction'),('unresolved_or_degenerate_pair_count','unresolved_or_degenerate_pairs')):
                metrics[name+'_'+key]=scan[source]
        qs=mod['shape'].shape_quality(F,C.L)
        metrics.update(free_minimum_angle_lt5_fraction=qs['min_angle_lt5_fraction'],free_angle_p05_degrees=qs['angle_p05_degrees'],
            free_q_gt10_fraction=qs['q_actual_gt10_fraction'],free_q_mean=qs['q_actual']['mean'],
            free_near_degenerate_count=qs['near_degenerate_area_count'],free_near_degenerate_fraction=qs['near_degenerate_area_count']/len(F),
            free_exact_zero_area_count=qs['exact_zero_area_count'],free_exact_zero_area_fraction=qs['exact_zero_area_count']/len(F),free_total_surface_area=qs['total_area'])
        pair_origin='INHERITED_COMPLETE_PAIR_CACHE_WITH_CURRENT_FACE_COUNT_VERIFIED'
    reference=old.load_queries(q['query_descriptor']);trusted=mod['control'].trusted_cf(cc,F,C.L,cf_scan=cf)
    im,iv=mod['control'].conflict_metrics(F,dict(C=cc,L0=C.L),reference,trusted);metrics.update(im)
    for kind,scan in pairs.items():
        bad=[r for r in scan['contacts'] if r['classification'] in B.ILLEGAL]
        affected={int(r['pair'][1]) for r in bad} if kind=='CF' else {int(i) for r in bad for i in r['pair']}
        metrics.update({kind+'_illegal_pair_count':len(bad),kind+'_illegal_affected_free_fraction':len(affected)/len(F) if len(F) else None})
    shape={name:mod['shape'].shape_quality(part,C.L) for name,part in (('C',cc),('free',F),('whole',tri))}
    for name,s in shape.items():
        metrics.update({name+'_minimum_angle_lt5_fraction':s['min_angle_lt5_fraction'],name+'_q_gt10_fraction':s['q_actual_gt10_fraction'],
            name+'_near_degenerate_count':s['near_degenerate_area_count'],name+'_exact_zero_area_count':s['exact_zero_area_count'],name+'_total_surface_area':s['total_area']})
    metrics.update(mod['geometry'].topology(tri,K),actual_free_faces=len(F),deleted_free_faces=int(q['N'])-len(set(raw[~bridge].tolist())))
    graph,edges,_=G.actual_valid_graph(v,f,K,pairs)
    graph.update(H.opposite_reach(tri,K,edges),C_F_has_opposite_winding_edge=int(graph['C_F_valid_opposite_winding_edge_count']>0),domain=str(m['index_domain']))
    internal,bstats,seamrows=role_reach(tri,K,edges,role,ids,parent,raw,m['raw_seam_slot_mask'])
    source_ids=metadata['C_source_vertex_ids'][inp['C_faces_original']];_,points,_=A.topology(original,source_ids);dist,_=A.nearest(points,F)
    boundary=[dict(setting=n,threshold_model=t,scale=s,tau=value,domain='ALL_C_SOURCE_TOPOLOGY_OPEN_BOUNDARY',**A.metric(dist,t)) for n,t,s,value in A.SETTINGS]
    distance={key:float(fn(dist)) if len(dist) and np.isfinite(dist).all() else None for key,fn in (
        ('boundary_distance_mean_model',np.mean),('boundary_distance_median_model',np.median),('boundary_distance_p95_model',lambda x:np.quantile(x,.95)),('boundary_distance_max_model',np.max))}
    result=dict(**{k:q[k] for k in E.META},label=label,stage=label,status='PASS',version=VERSION,metrics=metrics,
        topology=graph,seam_to_bulk=internal,boundary=boundary,distance=distance,C_preserved_exact=True,C_original_order_preserved=True,
        C_query_count=2048,C_query_weight_denominator=float(reference['arrays']['C_weights'].sum()),free_face_denominator=len(F),
        CF_tested_pair_denominator=K*len(F),FF_tested_pair_denominator=len(F)*(len(F)-1)//2,
        status_by_metric=dict(I=im['ConflictCov_C_status'],S='NA_NO_REQUIRED_INTERFACE',E='NA_NO_EXPOSURE',GT='NA_NOT_UNIQUE_GENERATIVE_TARGET',semantic_use='AI_PROXY_REVIEW_SEPARATE'),
        bridge_contract='New independent face; no RAW parent; traversable legal graph but excluded from original BULK area',
        deletion_semantics='Missing original RAW parents, never requested N minus output faces',pair_origin=pair_origin)
    row=B.flatten(result,label);counts=collections.Counter();role_rows=[]
    for kind,scan in pairs.items():
        for r in scan['contacts']:
            if r['classification'] not in B.ILLEGAL:continue
            i,j=map(int,r['pair']);j+=K
            if kind=='FF':i+=K
            rr=(str(role[i]),str(role[j]))
            if 'BRIDGE' in rr:
                other=rr[1] if rr[0]=='BRIDGE' else rr[0];key='bridge-'+other.lower()
            elif kind=='CF':key='C-'+rr[1].lower()
            else:key='seam-seam' if rr==('SEAM','SEAM') else 'seam-bulk' if 'SEAM' in rr else 'bulk-bulk'
            counts[key]+=1;role_rows.append(dict(kind=kind,role=key,output_face_i=int(ids[i]),output_face_j=int(ids[j]),
                parent_post_i=int(parent[i]),parent_post_j=int(parent[j]),classification=r['classification']))
    for key in ('seam-seam','seam-bulk','bulk-bulk','C-seam','C-bulk','bridge-c','bridge-seam','bridge-bulk','bridge-bridge'):
        row['illegal_pairs_'+key]=counts[key]
    diag,graphs=topology(tri,f,K,edges,role);fixed,detail=parent_unions(pairs,shape,K,parent,bridge,base)
    row.update(diag)
    row.update(bstats,**fixed,original_POST_free_faces=base-K,new_bridge_faces=int(bridge.sum()),
               deleted_original_POST_faces=base-len(set(parent[~bridge].tolist())),scoring_definition_version=VERSION)
    bridge_shape=mod['shape'].shape_quality(tri[bridge],C.L)
    row.update(bridge_minimum_angle_degrees=min((r['minimum_angle_degrees'] for r in bridge_shape['per_face']),default=None),
               bridge_q_mean=bridge_shape['q_actual']['mean'],bridge_shape_status='PASS' if bridge.any() else 'NA_NO_BRIDGE')
    fields=C.read(C.ROOT/'r2/tools/hinge_opposite_v1/metric_policy.json')['directions']
    assert not set(fields)-set(row),sorted(set(fields)-set(row))
    assert 'torch' not in sys.modules
    payload=dict(row=row,score=result,pairs=pairs,shape=shape,edges=edges,query_vectors=iv,
                 bridge_shape=bridge_shape,parent_detail=detail,seam_rows=seamrows,role_rows=role_rows)
    if dest is not None:
        dest=C.resolve(dest)
        if dest.exists() and any(dest.iterdir()):raise FileExistsError('Refuse existing score directory '+str(dest))
        dest.mkdir(parents=True,exist_ok=True)
        for name,value in (('score',result),('row',row),('pairs',pairs),('shape_per_face',shape),('bridge_shape',bridge_shape),('fixed_parent_detail',detail)):
            C.write(dest/(name+'.json'),value)
        for name,value in (('valid_edges',edges),('seam_bulk_faces',seamrows),('illegal_pairs_by_role',role_rows)):C.csvwrite(dest/(name+'.csv'),value)
        E.save_npz(dest/'vectors.npz',iv)
        C.write(dest/'complete.json',dict(status='PASS',version=VERSION,row=row,model_calls=0,new_queries=0,
            files={str(p):C.sha(p) for p in dest.iterdir() if p.is_file()}))
    return payload
