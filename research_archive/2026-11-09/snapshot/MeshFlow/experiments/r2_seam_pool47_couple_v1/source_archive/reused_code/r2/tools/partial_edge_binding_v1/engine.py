"""CPU-only constrained indexed binding; no neural dependencies or reclustering."""
from __future__ import annotations
from collections import defaultdict, Counter
import numpy as np
import importlib.util
from pathlib import Path
from functools import lru_cache

@lru_cache(None)
def cpu_module(name):
    path=Path(__file__).resolve().parents[3]/"r2/src/meshflow_control"/(name+".py")
    spec=importlib.util.spec_from_file_location("_binding_"+name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

VERSION = 'partial_edge_binding_indexed_v1'
L = float(np.sqrt(3.)*2/.3762)
DELTAS = (.01, .02, .035)

def incidence(faces, start=0):
    out=defaultdict(list)
    for i in range(start,len(faces)):
        for j,k in ((0,1),(1,2),(2,0)):
            u,v=map(int,faces[i,[j,k]])
            out[tuple(sorted((u,v)))].append((i,j,k))
    return out

def shape(tri):
    t=np.asarray(tri,np.float64)
    cross=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]); twice=np.linalg.norm(cross,axis=1)
    lens=np.stack([np.linalg.norm(t[:,(j+1)%3]-t[:,j],axis=1) for j in range(3)],1)
    angles=[]
    for j in range(3):
        a=t[:,(j+1)%3]-t[:,j]; b=t[:,(j+2)%3]-t[:,j]
        denom=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
        angles.append(np.degrees(np.arccos(np.clip(np.divide((a*b).sum(1),denom,out=np.ones(len(t)),where=denom>0),-1,1))))
    area=twice/2
    q=np.divide((lens*lens).sum(1),4*np.sqrt(3)*area,out=np.full(len(t),np.inf),where=area>0)
    normals=np.divide(cross,twice[:,None],out=np.zeros_like(cross),where=twice[:,None]>0)
    return dict(area=area,angle=np.min(angles,axis=0),q=q,normal=normals)

def source_orient(vertices, faces, K, P_inverse):
    orient_fixed_C=cpu_module("winding_solver").orient_fixed_C
    f=faces.copy(); f[:K]=np.take_along_axis(f[:K],P_inverse,axis=1)
    result=orient_fixed_C(vertices,f,K)
    perm=np.tile(np.arange(3),(len(f),1)); perm[:K]=P_inverse
    perm=np.take_along_axis(perm,result['face_corner_permutation'],axis=1)
    return result,perm

def decode(pre, base, K, P_inverse, mode, delta_fraction, classify_cf):
    """Baseline retained-face topology is frozen; pre is only displacement reference.

    A reassigns selected corner IDs. B reassigns every original retained occurrence
    of a GEO_C vertex ID. Deleted-input corners participate in B displacement
    checks but deleted faces are never resurrected. Final cleanup is exact only.
    classify_cf(C,F) returns {(C row,F row)} of original illegal classifications.
    """
    assert mode in ('A','B') and delta_fraction in DELTAS
    geo=cpu_module("geo_c"); _degenerate,_face_key=geo._degenerate,geo._face_key
    vertices=base['vertices'].copy(); original_faces=base['faces']; faces=original_faces.copy()
    sf=base['source_face_ids']; sc=base['source_corner_ids']; M=len(sf)
    assert np.array_equal(sc,np.tile(np.arange(3),(M,1)))
    original_tri=vertices[original_faces]; S0=shape(original_tri); C=original_tri[:K]
    flat_pre=pre.reshape(-1,3).astype(np.float64); cap=L*delta_fraction
    cv=base['corner_vertex_ids'].reshape(-1)
    clusters={int(v):np.flatnonzero((cv==v)&(np.arange(len(cv))>=3*K)) for v in np.unique(cv[3*K:]) if v>=0}
    cvertices=set(map(int,original_faces[:K].reshape(-1)))
    all_inc=incidence(original_faces); c_inc=incidence(original_faces[:K]); f_inc=incidence(original_faces,K)
    boundary=[e for e,occ in sorted(c_inc.items()) if len(occ)==1]
    free_open=[(e,occ[0]) for e,occ in sorted(f_inc.items()) if len(occ)==1 and e[0]!=e[1]]
    occupied={e for e in boundary if any(i>=K for i,j,k in all_inc[e])}
    # Capacity is conservatively full even for an already-illegal C/F edge:
    # adding a third incident face would violate the requested capacity contract.
    original_cf=classify_cf(C,original_tri[K:])
    cf_by_face=defaultdict(set)
    for ci,fi in original_cf: cf_by_face[int(fi)+K].add(int(ci))
    geometry_cache={}
    candidates=[]; prefilter=Counter()
    for edge,(fi,a,b) in free_open:
        v0,v1=map(int,original_faces[fi,[a,b]])
        x0,x1=vertices[[v0,v1]].astype(float); axis=x1-x0; ln=np.linalg.norm(axis)
        sid=int(sf[fi]); cp0,cp1=3*sid+a,3*sid+b
        for be in boundary:
            c0,c1=be; bv=vertices[c1].astype(float)-vertices[c0]; bn=np.linalg.norm(bv)
            if not ln or not bn or abs(float(np.dot(axis,bv)/(ln*bn)))<=np.cos(np.pi/6):
                prefilter['AXIS_ANGLE_GE_30']+=1; continue
            options=[]
            for t0,t1 in ((c1,c0),(c0,c1)):
                ed=max(float(np.linalg.norm(flat_pre[cp0]-vertices[t0])),float(np.linalg.norm(flat_pre[cp1]-vertices[t1])))
                options.append((ed,t0,t1))
            ed,t0,t1=min(options)
            ids0=np.asarray([cp0]) if mode=='A' else clusters[v0]
            ids1=np.asarray([cp1]) if mode=='A' else clusters[v1]
            d0=float(np.linalg.norm(flat_pre[ids0]-vertices[t0],axis=1).max())
            d1=float(np.linalg.norm(flat_pre[ids1]-vertices[t1],axis=1).max())
            d=max(d0,d1)
            if ed>cap:
                prefilter['ENDPOINT_OVER_BUDGET']+=1; continue
            candidates.append(dict(source_face_id=sid,face_row=fi,local_edge=[a,b],free_vertex_ids=[v0,v1],
                C_edge=list(be),target_C_vertex_ids=[t0,t1],endpoint_max_displacement=ed,
                max_displacement=d,assignment_input_corner_ids=[ids0.tolist(),ids1.tolist()],
                axis_abs_cos=abs(float(np.dot(axis,bv)/(ln*bn)))))
    candidates.sort(key=lambda r:(r['max_displacement'],r['source_face_id'],*r['local_edge'],*r['C_edge'],*r['target_C_vertex_ids']))
    assigned={}; accepted=[]; records=[]; affected_union=set(); used=set(); counts=Counter(); cf_cache_hits=0
    for rank,cand in enumerate(candidates):
        r=dict(cand,rank=rank,mode=mode,delta_fraction=delta_fraction); why=None
        fi=r['face_row']; a,b=r['local_edge']; t0,t1=r['target_C_vertex_ids']; be=tuple(r['C_edge'])
        v0,v1=r['free_vertex_ids']; keys=[3*int(sf[fi])+a,3*int(sf[fi])+b] if mode=='A' else [v0,v1]
        if r['max_displacement']>cap: why='CLUSTER_OVER_BUDGET'
        elif be in occupied: why='C_EDGE_CAPACITY'
        elif (fi,a,b) in used: why='FREE_EDGE_CAPACITY'
        elif any((key in assigned and assigned[key]!=t) or (v in cvertices and v!=t) for key,v,t in zip(keys,[v0,v1],[t0,t1])): why='C_ANCHOR_CONFLICT'
        trial=faces.copy()
        if why is None:
            if mode=='A': trial[fi,a]=t0; trial[fi,b]=t1; aff=np.asarray([fi])
            else:
                aff=np.flatnonzero(np.any((original_faces==v0)|(original_faces==v1),axis=1)); aff=aff[aff>=K]
                for old,t in ((v0,t0),(v1,t1)):
                    mask=(original_faces==old);mask[:K]=False;trial[mask]=t
            r['affected_source_faces']=[int(sf[i]) for i in aff]
            if any(len(set(map(int,trial[i])))<3 for i in aff): why='COLLAPSED_VERTEX_IDS'
            T=shape(vertices[trial[aff]])
            if why is None and np.any(T['area']<=1e-12*L*L): why='AREA_DEGENERATE'
            if why is None and np.any((T['angle']<5)&(T['angle']<S0['angle'][aff]-1e-10)): why='ANGLE_NEW_OR_WORSE'
            if why is None and np.any((T['q']>10)&(T['q']>S0['q'][aff]+1e-10)): why='Q_NEW_OR_WORSE'
            normal_cos=(T['normal']*S0['normal'][aff]).sum(1)
            r['maximum_physical_normal_rotation_degrees']=float(np.degrees(np.arccos(np.clip(normal_cos,-1,1))).max())
            if why is None and np.any(normal_cos<.5-1e-12): why='NORMAL_ROTATION_GT60'
            if why is None:
                tinc=incidence(trial)
                if len(tinc.get(be,[]))!=2 or any(sum(i>=K for i,j,k in tinc.get(e,[]))>1 for e in boundary if any(i in set(aff) for i,j,k in tinc.get(e,[]))):
                    why='C_EDGE_CAPACITY_AFTER_ASSIGNMENT'
            if why is None:
                for i in aff:
                    tri=vertices[trial[i]]; key=(int(i),tri.tobytes())
                    if key in geometry_cache: illegal=geometry_cache[key];cf_cache_hits+=1
                    else:
                        illegal={int(ci) for ci,unused in classify_cf(C,tri[None])}; geometry_cache[key]=illegal
                    if illegal-cf_by_face[int(i)]: why='NEW_CF_ILLEGAL'; r['new_CF_source_pairs']=[[ci,int(sf[i])] for ci in sorted(illegal-cf_by_face[int(i)])];break
            if why is None:
                orientation,_=source_orient(vertices,trial,K,P_inverse)
                face_comp=orientation['face_component_ids']
                changed_comps={int(face_comp[i]) for i in aff}
                # Only new binding components are tested; unrelated pure-free
                # baseline conflicts are not a veto for the whole mesh.
                if any(not orientation['component_rows'][co]['satisfiable'] for co in changed_comps if orientation['component_rows'][co]['C_face_count']>0):
                    why='ORIENTATION_CONFLICT'
        if why is None:
            faces=trial;occupied.add(be);used.add((fi,a,b));assigned.update(zip(keys,[t0,t1]));affected_union.update(map(int,aff))
            accepted.append(r);why='ACCEPTED'
        r['decision']=why;counts[why]+=1;records.append(r)
    # Exactly the maintained GEO_C free-only numeric-coordinate cleanup;
    # never re-run radius consolidation or move any remaining representative.
    seen={_face_key(vertices[faces[i]]):i for i in range(K)}; keep=np.ones(M,bool); deleted=[]
    for i in range(K,M):
        tri=vertices[faces[i]]; reason=_degenerate(tri); fk=_face_key(tri)
        if reason is None and fk in seen:reason='duplicate_C' if seen[fk]<K else 'duplicate_free'
        if reason is not None:
            keep[i]=False;deleted.append(dict(source_face_id=int(sf[i]),reason=reason,baseline_area=float(S0['area'][i]),affected=i in affected_union))
        else:seen[fk]=i
    kept_faces=faces[keep];orientation,perms=source_orient(vertices,kept_faces,K,P_inverse)
    final_faces=orientation['faces']; final_tri=vertices[final_faces]
    assignments={}
    for r in accepted:
        for ids,t in zip(r['assignment_input_corner_ids'],r['target_C_vertex_ids']):
            for iid in ids:
                assert iid not in assignments or assignments[iid]==t
                assignments[iid]=t
    input_ids=np.asarray(sorted(assignments),dtype=np.int64)
    target_ids=np.asarray([assignments[i] for i in input_ids],dtype=np.int64)
    maxdisp=float(np.linalg.norm(flat_pre[input_ids]-vertices[target_ids],axis=1).max()) if len(input_ids) else 0.
    assert maxdisp<=cap+1e-12
    output=dict(vertices=vertices,faces=final_faces,output=final_tri,C=final_tri[:K].copy(),known_faces=np.asarray(K),
        source_face_ids=sf[keep],source_corner_ids=perms,baseline_keep=keep,
        canonical_faces_before_cleanup=faces,assignment_input_corner_ids=input_ids,assignment_target_C_vertex_ids=target_ids,
        input_faces=np.asarray(len(pre)),orientation_flips=orientation['flips'])
    summary=dict(mode=mode,delta_fraction=delta_fraction,delta_model=cap,accepted=len(accepted),
        candidate_count=len(records),rejections=dict(counts),prefilter=dict(prefilter),
        affected_faces=len(affected_union),deleted_faces=len(deleted),deleted_area=sum(x['baseline_area'] for x in deleted),
        max_bound_corner_displacement=maxdisp,CF_cache_hits=cf_cache_hits,orientation=orientation['summary'],
        model_calls=0,no_reclustering=True,deleted_input_faces_resurrected=0)
    return output,records,deleted,summary
