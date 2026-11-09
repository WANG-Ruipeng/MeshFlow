"""Independent NumPy explanations of existing C/F geometry; no model or old scorer.

Signed distances use the source C face normal, not an inside/outside classifier.
Layer groups and sidewall candidates are observations, never validity labels.
"""
from __future__ import annotations
import csv, hashlib, io, json, os
from collections import defaultdict
from pathlib import Path
import numpy as np

VERSION='cref_layer_finite_geometry_v1'
DEFAULTS=dict(corridor_L=.1,layer_gap_L=1e-4,normal_angle_degrees=15.,dup_distance_L=.005,
              C_border_L=.01,area_floor_L2=1e-11,line_parallel_dot_epsilon=1e-12,
              coplanar_plane_epsilon_L=1e-9,barycentric_epsilon=1e-9,query_chunk=64)


def triangles(value):
    a=np.asarray(value,dtype=np.float64).reshape(-1,3,3)
    if not np.isfinite(a).all():raise ValueError('Nonfinite triangles')
    return a


def normals(t,L=1.,area_floor=1e-11):
    t=triangles(t);cross=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);length=np.linalg.norm(cross,axis=-1)
    unit=np.divide(cross,length[:,None],out=np.zeros_like(cross),where=length[:,None]>0)
    return unit,length*.5>area_floor*L**2,length*.5


def barycentric(points,t):
    """Broadcast-compatible planar barycentric values without Gram subtraction."""
    p=np.asarray(points,np.float64);t=np.asarray(t,np.float64)
    e0=t[...,1,:]-t[...,0,:];e1=t[...,2,:]-t[...,0,:];n=np.cross(e0,e1);n2=np.sum(n*n,axis=-1)
    delta=p-t[...,0,:]
    b1=np.divide(np.sum(np.cross(delta,e1)*n,axis=-1),n2,out=np.full(np.broadcast_shapes(delta.shape[:-1],n2.shape),np.nan),where=n2>0)
    b2=np.divide(np.sum(np.cross(e0,delta)*n,axis=-1),n2,out=np.full_like(b1,np.nan),where=n2>0)
    return np.stack((1-b1-b2,b1,b2),axis=-1)


def finite_distances(points,free):
    """All finite point/triangle distances and closest points for a small chunk.

    Plane projection is accepted only inside the finite triangle. Three finite
    edges handle the exterior and exact degenerate cases. Ties choose the first
    face/edge deterministically; all other qualifying faces remain recorded.
    """
    q=np.asarray(points,np.float64).reshape(-1,3);f=triangles(free);Q,M=len(q),len(f)
    if not M:return np.empty((Q,0)),np.empty((Q,0,3)),np.empty((Q,0,3))
    target=q[:,None,:];best=np.full((Q,M),np.inf);closest=np.zeros((Q,M,3));bary=np.zeros_like(closest)
    for i,j in ((0,1),(1,2),(2,0)):
        edge=f[:,j]-f[:,i];length=np.sum(edge*edge,axis=-1)
        u=np.divide(np.sum((target-f[None,:,i])*edge[None],axis=-1),length[None],out=np.zeros((Q,M)),where=length[None]>0)
        u=np.clip(u,0.,1.);point=f[None,:,i]+u[:,:,None]*edge[None]
        d2=np.sum((target-point)**2,axis=-1);take=d2<best
        best[take]=d2[take];closest[take]=point[take]
        eb=np.zeros((Q,M,3));eb[:,:,i]=1-u;eb[:,:,j]=u;bary[take]=eb[take]
    n=np.cross(f[:,1]-f[:,0],f[:,2]-f[:,0]);n2=np.sum(n*n,axis=-1)
    offset=np.divide(np.sum((target-f[None,:,0])*n[None],axis=-1),n2[None],out=np.zeros((Q,M)),where=n2[None]>0)
    projection=target-offset[:,:,None]*n[None];bc=barycentric(projection,f[None])
    inside=(bc>=0).all(-1)&(bc<=1).all(-1)&(n2[None]>0)
    pd2=np.sum((target-projection)**2,axis=-1);take=inside&(pd2<best)
    best[take]=pd2[take];closest[take]=projection[take];bary[take]=bc[take]
    return best,closest,bary


def normal_line_hits(points,directions,free,L,config=None):
    """All isolated finite hits, including both signs and hits beyond corridor.

    Coplanar normal-line/triangle intersections are retained as intervals. They
    are not arbitrarily collapsed into a single discrete layer.
    """
    cfg=dict(DEFAULTS,**(config or {}));q=np.asarray(points,np.float64).reshape(-1,3);d=np.asarray(directions,np.float64).reshape(-1,3);f=triangles(free)
    unit,good,_=normals(f,L,cfg['area_floor_L2']);Q,M=len(q),len(f)
    empty=dict(query_id=np.empty(0,np.int64),free_face_id=np.empty(0,np.int64),signed_t=np.empty(0),barycentric=np.empty((0,3)),normal_dot=np.empty(0),in_corridor=np.empty(0,bool),barycentric_strict=np.empty(0,bool))
    intervals=dict(query_id=np.empty(0,np.int64),free_face_id=np.empty(0,np.int64),t_min=np.empty(0),t_max=np.empty(0),intersects_corridor=np.empty(0,bool))
    if not Q or not M:return empty,intervals
    dot=d@unit.T;num=np.einsum('qmi,mi->qm',f[None,:,0]-q[:,None],unit)
    regular=(np.abs(dot)>cfg['line_parallel_dot_epsilon'])&good[None]& (np.linalg.norm(d,axis=-1)[:,None]>.5)
    t=np.divide(num,dot,out=np.full((Q,M),np.nan),where=regular)
    with np.errstate(invalid='ignore'):
        point=q[:,None]+t[:,:,None]*d[:,None];b=barycentric(point,f[None])
    eps=cfg['barycentric_epsilon'];inside=regular&(b>=-eps).all(-1)&(b<=1+eps).all(-1)
    qi,fi=np.nonzero(inside);bt=b[qi,fi]
    hits=dict(query_id=qi.astype(np.int64),free_face_id=fi.astype(np.int64),signed_t=t[qi,fi],barycentric=bt,normal_dot=dot[qi,fi],in_corridor=np.abs(t[qi,fi])<=cfg['corridor_L']*L,barycentric_strict=(bt>eps).all(-1)&(bt<1-eps).all(-1))
    coplanar=(~regular)&good[None]&(np.abs(dot)<=cfg['line_parallel_dot_epsilon'])&(np.abs(num)<=cfg['coplanar_plane_epsilon_L']*L)&(np.linalg.norm(d,axis=-1)[:,None]>.5)
    saved=[]
    for i,j in zip(*np.nonzero(coplanar)):
        a=barycentric(q[i],f[j]);s=barycentric(q[i]+d[i],f[j])-a;lo,hi=-np.inf,np.inf
        for aa,ss in zip(a,s):
            if abs(ss)<1e-15:
                if aa < -eps:lo,hi=1.,0.;break
            elif ss>0:lo=max(lo,(-eps-aa)/ss)
            else:hi=min(hi,(-eps-aa)/ss)
        if lo<=hi and np.isfinite(lo) and np.isfinite(hi):saved.append((i,j,lo,hi,lo<=cfg['corridor_L']*L and hi>=-cfg['corridor_L']*L))
    if saved:
        a=np.asarray(saved);intervals=dict(query_id=a[:,0].astype(np.int64),free_face_id=a[:,1].astype(np.int64),t_min=a[:,2],t_max=a[:,3],intersects_corridor=a[:,4].astype(bool))
    return hits,intervals


def layer_groups(hits,Q,L,config=None):
    """Single-link groups of isolated signed hits inside the fixed corridor."""
    cfg=dict(DEFAULTS,**(config or {}));labels=np.full(len(hits['query_id']),-1,np.int64);rows=[];counts=np.zeros(Q,np.int64)
    for qi in range(Q):
        ids=np.flatnonzero((hits['query_id']==qi)&hits['in_corridor'])
        if not len(ids):continue
        ids=ids[np.lexsort((hits['free_face_id'][ids],hits['signed_t'][ids]))];cluster=[]
        for idx in ids:
            if cluster and hits['signed_t'][idx]-hits['signed_t'][cluster[-1]]>cfg['layer_gap_L']*L:
                group=len(rows);labels[cluster]=group
                rows.append(dict(query_id=qi,query_layer_index=int(counts[qi]),t_min=float(hits['signed_t'][cluster[0]]),t_max=float(hits['signed_t'][cluster[-1]]),signed_t_mean=float(np.mean(hits['signed_t'][cluster])),hit_count=len(cluster),free_face_ids=sorted(set(int(hits['free_face_id'][j]) for j in cluster))))
                counts[qi]+=1;cluster=[]
            cluster.append(int(idx))
        if cluster:
            group=len(rows);labels[cluster]=group
            rows.append(dict(query_id=qi,query_layer_index=int(counts[qi]),t_min=float(hits['signed_t'][cluster[0]]),t_max=float(hits['signed_t'][cluster[-1]]),signed_t_mean=float(np.mean(hits['signed_t'][cluster])),hit_count=len(cluster),free_face_ids=sorted(set(int(hits['free_face_id'][j]) for j in cluster))))
            counts[qi]+=1
    return rows,labels,counts


def _components(adjacency,allowed):
    labels=np.full(len(adjacency),-1,np.int64);groups=[]
    for start in np.flatnonzero(allowed):
        if labels[start]>=0:continue
        todo=[int(start)];labels[start]=len(groups);members=[]
        while todo:
            cur=todo.pop();members.append(cur)
            for other in sorted(adjacency[cur]):
                if allowed[other] and labels[other]<0:labels[other]=len(groups);todo.append(other)
        groups.append(sorted(members))
    return labels,groups


def exact_graph(vertices,faces,known_mask):
    """Observed exact indices, including marked nonmanifold contacts; no welding."""
    vertices=np.asarray(vertices,np.float64);faces=np.asarray(faces,np.int64);known=np.asarray(known_mask,bool)
    if faces.ndim!=2 or faces.shape[1]!=3 or len(known)!=len(faces):raise ValueError('Bad indexed geometry')
    if len(faces) and (faces.min()<0 or faces.max()>=len(vertices)):raise ValueError('Out-of-range index')
    incidence=defaultdict(list)
    for fi,ids in enumerate(faces):
        for i,j in ((0,1),(1,2),(2,0)):
            if ids[i]!=ids[j]:incidence[tuple(sorted((int(ids[i]),int(ids[j]))))].append(fi)
    adj=[set() for _ in faces];edges=[]
    for edge,raw in sorted(incidence.items()):
        owners=sorted(set(raw))
        for i in owners:adj[i].update(j for j in owners if j!=i)
        edges.append(dict(vertex0=edge[0],vertex1=edge[1],incident_face_ids=owners,incidence_count=len(raw),distinct_faces=len(owners),boundary=len(owners)==1,nonmanifold=len(raw)>2,has_C=bool(known[owners].any()),has_F=bool((~known[owners]).any()),length=float(np.linalg.norm(vertices[edge[1]]-vertices[edge[0]]))))
    all_id,all_groups=_components(adj,np.ones(len(faces),bool));free_id,free_groups=_components(adj,~known)
    return dict(adjacency=adj,edges=edges,all_component_id=all_id,free_component_id=free_id,all_components=all_groups,free_components=free_groups,known_mask=known)


def local_face_geometry(C_source,F,vertices,faces,known_mask,L,candidate_by_C,config=None):
    cfg=dict(DEFAULTS,**(config or {}));c=triangles(C_source);f=triangles(F);graph=exact_graph(vertices,faces,known_mask)
    known=np.asarray(known_mask,bool);free_global=np.flatnonzero(~known);C_global=np.flatnonzero(known)
    if len(free_global)!=len(f) or len(C_global)!=len(c):raise ValueError('Indexed C/F face count mismatch')
    indexed=np.asarray(vertices)[np.asarray(faces)]
    if not np.array_equal(indexed[free_global].astype(np.float32),f.astype(np.float32)):raise ValueError('F-local IDs do not match indexed F row order')
    cn,cgood,ca=normals(c,L,cfg['area_floor_L2']);fn,fgood,fa=normals(f,L,cfg['area_floor_L2']);centers=f.mean(1);free_lookup={int(g):i for i,g in enumerate(free_global)}
    face_rows=[];regions=[];walls=[];threshold=np.cos(np.deg2rad(cfg['normal_angle_degrees']))
    for ci in range(len(c)):
        if not cgood[ci]:continue
        anchor=c[ci].mean(0);signed=(f-anchor)@cn[ci];center_signed=(centers-anchor)@cn[ci];parallel=fgood&(np.abs(fn@cn[ci])>=threshold)
        in_corridor=(signed.min(1)<=cfg['corridor_L']*L)&(signed.max(1)>=-cfg['corridor_L']*L)
        pa=np.zeros(len(faces),bool);npaa=np.zeros(len(faces),bool);pa[free_global]=parallel&in_corridor;npaa[free_global]=~parallel&fgood&in_corridor
        pids,pgroups=_components(graph['adjacency'],pa);nids,ngroups=_components(graph['adjacency'],npaa)
        for pi,members in enumerate(pgroups):
            ids=[free_lookup[m] for m in members]
            regions.append(dict(C_face_row=ci,parallel_region_id=pi,free_face_ids=ids,source_normal_centroid_offset_min=float(center_signed[ids].min()),source_normal_centroid_offset_max=float(center_signed[ids].max()),area=float(fa[ids].sum()),has_registered_near_witness=bool(candidate_by_C[ci,ids].any())))
        for ni,members in enumerate(ngroups):
            neighbor=set(j for m in members for j in graph['adjacency'][m] if j not in members)
            preg=sorted(set(int(pids[j]) for j in neighbor if pids[j]>=0));cc=sorted(int(np.flatnonzero(C_global==j)[0]) for j in neighbor if known[j]);ids=[free_lookup[m] for m in members]
            if preg or cc:
                walls.append(dict(C_face_row=ci,nonparallel_component_id=ni,free_face_ids=ids,adjacent_parallel_region_ids=preg,adjacent_C_face_rows=cc,adjacent_parallel_region_count=len(preg),candidate_C_to_parallel=bool(preg and cc),candidate_parallel_to_parallel=len(preg)>=2,
                                  meaning='Exact-edge nonparallel corridor component touches listed regions; no thickness/legality/manifold certification'))
        for fi in range(len(f)):
            g=int(free_global[fi]);face_rows.append(dict(C_face_row=ci,free_face_id=fi,global_face_id=g,free_area=float(fa[fi]),normal_good=bool(fgood[fi]),normal_abs_dot=float(abs(fn[fi]@cn[ci])),parallel_15deg=bool(parallel[fi]),signed_vertex_min=float(signed[fi].min()),signed_vertex_max=float(signed[fi].max()),signed_centroid=float(center_signed[fi]),plane_corridor_intersects=bool(in_corridor[fi]),registered_near_witness=bool(candidate_by_C[ci,fi]),parallel_region_id=int(pids[g]),nonparallel_component_id=int(nids[g]),exact_all_component_id=int(graph['all_component_id'][g]),exact_free_component_id=int(graph['free_component_id'][g])))
    return dict(face_rows=face_rows,parallel_regions=regions,sidewall_adjacency=walls,edges=graph['edges'],free_component_id=graph['free_component_id'][free_global],
                summary=dict(exact_components=len(graph['all_components']),exact_free_components=len(graph['free_components']),exact_CF_edges=sum(e['has_C'] and e['has_F'] for e in graph['edges']),nonmanifold_edges=sum(e['nonmanifold'] for e in graph['edges']),
                             graph_semantics='Observed exact indexed edges, including explicitly marked nonmanifold and potentially intersecting faces; not the previous valid-edge graph. RAW can have zero exact links despite close geometry. No repair/weld/flip performed.',sidewall_locality='Vertex signed-range corridor around each source C plane; no finite-footprint claim beyond registered candidate-query witnesses'))


def analyze_geometry(geometry,config=None):
    cfg=dict(DEFAULTS,**(config or {}));L=float(geometry.get('L',cfg.get('L',0)))
    if not np.isfinite(L) or L<=0:raise ValueError('Positive inherited L required')
    c=triangles(geometry['C']);f=triangles(geometry['F']);source=triangles(geometry['C_original']);query=geometry['query']
    if len(c)!=len(source):raise ValueError('Source C row count mismatch')
    # Winding may differ, but each original face must be the same finite triangle.
    for a,b in zip(c,source):
        if sorted(map(tuple,a))!=sorted(map(tuple,b)):raise ValueError('Source C rows do not match canonical face geometry')
    q=np.asarray(query['C_points'],np.float64);owner=np.asarray(query['C_point_faces'],np.int64);weights=np.asarray(query['C_weights'],np.float64);border=np.asarray(query['C_border_distance'],np.float64)
    if q.shape!=(len(weights),3) or owner.shape!=weights.shape or border.shape!=weights.shape:raise ValueError('Query schema mismatch')
    if len(owner) and (owner.min()<0 or owner.max()>=len(c)):raise ValueError('Query owner outside C')
    if not np.isfinite(q).all() or not np.isfinite(weights).all() or (weights<0).any() or weights.sum()<=0:raise ValueError('Invalid query points/weights')
    cn,cgood,_=normals(source,L,cfg['area_floor_L2']);fn,fgood,_=normals(f,L,cfg['area_floor_L2']);direction=cn[owner];Q,M=len(q),len(f)
    nearest_id=np.full(Q,-1,np.int64);nearest_d=np.full(Q,np.inf);nearest_point=np.full((Q,3),np.nan);nearest_bary=np.full((Q,3),np.nan)
    parallel_id=np.full(Q,-1,np.int64);parallel_d=np.full(Q,np.inf);parallel_point=np.full((Q,3),np.nan)
    hit_parts=[];interval_parts=[];witness_q=[];witness_f=[];witness_d=[];candidate_by_C=np.zeros((len(c),M),bool);candidate_counts=np.zeros(M,np.int64);candidate_weight=np.zeros(M)
    interior=border>=cfg['C_border_L']*L;threshold=np.cos(np.deg2rad(cfg['normal_angle_degrees']))
    for start in range(0,Q,int(cfg['query_chunk'])):
        stop=min(Q,start+int(cfg['query_chunk']));chunk=q[start:stop];d2,closest,bc=finite_distances(chunk,f)
        if M:
            ids=np.argmin(d2,axis=1);ii=np.arange(stop-start);nearest_id[start:stop]=ids;nearest_d[start:stop]=np.sqrt(d2[ii,ids]);nearest_point[start:stop]=closest[ii,ids];nearest_bary[start:stop]=bc[ii,ids]
            aligned=(np.abs(direction[start:stop]@fn.T)>=threshold)&fgood[None]&cgood[owner[start:stop],None]
            pd2=np.where(aligned,d2,np.inf);pids=np.argmin(pd2,axis=1);has=np.isfinite(pd2[ii,pids]);global_ids=np.arange(start,stop)[has]
            parallel_id[global_ids]=pids[has];parallel_d[global_ids]=np.sqrt(pd2[ii[has],pids[has]]);parallel_point[global_ids]=closest[ii[has],pids[has]]
            qualifies=aligned&(d2<(cfg['dup_distance_L']*L)**2)&interior[start:stop,None]
            qi,fi=np.nonzero(qualifies);qi=qi+start
            witness_q.extend(qi.tolist());witness_f.extend(fi.tolist());witness_d.extend(np.sqrt(d2[qi-start,fi]).tolist())
            for i,j in zip(qi,fi):candidate_by_C[owner[i],j]=True;candidate_counts[j]+=1;candidate_weight[j]+=weights[i]
        hits,intervals=normal_line_hits(chunk,direction[start:stop],f,L,cfg)
        hits['query_id']+=start;intervals['query_id']+=start;hit_parts.append(hits);interval_parts.append(intervals)
    def concatenate(parts):return {k:np.concatenate([p[k] for p in parts],axis=0) for k in parts[0]} if parts else {}
    hits=concatenate(hit_parts);intervals=concatenate(interval_parts);groups,group_id,group_count=layer_groups(hits,Q,L,cfg);hits['layer_group_id']=group_id
    reconstructed_dup=interior&cgood[owner]&(parallel_d<cfg['dup_distance_L']*L)
    old=geometry.get('old_conflict_dup');compare=dict(status='NA_MISSING_OLD_MASK',mismatched_queries=None)
    if old is not None:
        old=np.asarray(old,bool)
        if old.shape!=(Q,):raise ValueError('Old mask length mismatch')
        bad=np.flatnonzero(old!=reconstructed_dup);compare=dict(status='PASS' if not len(bad) else 'MISMATCH',mismatched_queries=len(bad),mismatched_query_ids=bad.tolist(),old_weighted_fraction=float(weights@old/weights.sum()),reconstructed_weighted_fraction=float(weights@reconstructed_dup/weights.sum()))
    signed_nearest=np.einsum('qi,qi->q',nearest_point-q,direction);signed_parallel=np.einsum('qi,qi->q',parallel_point-q,direction)
    line_counts=np.bincount(hits['query_id'],minlength=Q);corridor_counts=np.bincount(hits['query_id'][hits['in_corridor']],minlength=Q);interval_counts=np.bincount(intervals['query_id'],minlength=Q)
    corridor_face_counts=np.bincount(hits['free_face_id'][hits['in_corridor']],minlength=M)
    rows=[]
    for i in range(Q):
        fid=int(nearest_id[i]);plane_t=None
        if fid>=0:
            dot=float(fn[fid]@direction[i])
            if fgood[fid] and abs(dot)>cfg['line_parallel_dot_epsilon']:plane_t=float(fn[fid]@(f[fid,0]-q[i])/dot)
        rows.append(dict(query_id=i,C_face_row=int(owner[i]),weight=float(weights[i]),x=float(q[i,0]),y=float(q[i,1]),z=float(q[i,2]),source_normal_x=float(direction[i,0]),source_normal_y=float(direction[i,1]),source_normal_z=float(direction[i,2]),source_C_normal_good=bool(cgood[owner[i]]),C_border_distance=float(border[i]),old_interior=bool(interior[i]),nearest_F_face_id=fid,nearest_F_distance=float(nearest_d[i]),signed_nearest_point_offset=float(signed_nearest[i]),nearest_F_plane_normal_line_t=plane_t,
                         nearest_parallel_F_face_id=int(parallel_id[i]),nearest_parallel_F_distance=float(parallel_d[i]),signed_nearest_parallel_point_offset=float(signed_parallel[i]),old_dup=bool(old[i]) if old is not None else None,reconstructed_dup=bool(reconstructed_dup[i]),all_isolated_normal_hits=int(line_counts[i]),corridor_isolated_normal_hits=int(corridor_counts[i]),corridor_layer_groups=int(group_count[i]),coplanar_normal_line_intervals=int(interval_counts[i])))
    candidate_rows=[dict(free_face_id=j,qualifies=bool(candidate_counts[j]),qualifying_query_count=int(candidate_counts[j]),qualifying_query_weight=float(candidate_weight[j]),witness_C_face_rows=np.flatnonzero(candidate_by_C[:,j]).tolist(),corridor_hit_query_count=int(corridor_face_counts[j])) for j in range(M)]
    graph=local_face_geometry(source,f,geometry['vertices'],geometry['faces'],geometry['known_mask'],L,candidate_by_C,cfg)
    vectors=dict(query_points=q,query_weights=weights,query_C_face=owner,source_C_normals=direction,nearest_F_face_id=nearest_id,nearest_F_distance=nearest_d,nearest_F_point=nearest_point,nearest_F_barycentric=nearest_bary,nearest_parallel_F_face_id=parallel_id,nearest_parallel_F_distance=parallel_d,nearest_parallel_F_point=parallel_point,signed_nearest_point_offset=signed_nearest,signed_nearest_parallel_point_offset=signed_parallel,reconstructed_dup=reconstructed_dup,old_interior=interior,corridor_layer_count=group_count)
    if old is not None:vectors['old_conflict_dup']=old
    witness=dict(query_id=np.asarray(witness_q,np.int64),free_face_id=np.asarray(witness_f,np.int64),finite_distance=np.asarray(witness_d,np.float64))
    summary=dict(version=VERSION,status='PASS' if compare['status'] in ('PASS','NA_MISSING_OLD_MASK') else 'MISMATCH_OLD_DUP',L=L,config=cfg,C_faces=len(c),free_faces=M,query_count=Q,query_weight_denominator=float(weights.sum()),old_dup_comparison=compare,
                 near_parallel_free_indices=np.flatnonzero(candidate_counts).tolist(),isolated_normal_hits=len(hits['query_id']),coplanar_normal_line_intervals=len(intervals['query_id']),corridor_layer_groups=len(groups),
                 layer_count_query_histogram={str(k):int((group_count==k).sum()) for k in np.unique(group_count)},graph=graph['summary'],graph_identity=geometry.get('graph_identity','UNSPECIFIED_OBSERVATIONAL_INDEX'),new_model_calls=0,new_old_scorer_calls=0,new_geometry_explanations=1,
                 interpretation=['Signed values follow the preserved source C normal only, not inside/outside. Reversing C winding reverses signs.',
                                 'All finite normal-line hits are saved, including beyond the nearest face and outside the local corridor. Coplanar line intervals remain separate.',
                                 'Layer groups use consecutive signed-t gaps, not semantic labels. A triangulated quad can give two hits but one group.',
                                 'Near-parallel witnesses reproduce the old proximity proxy; neither exact duplicate nor illegal overlap follows from it.',
                                 'Sidewall candidates use observed exact-edge adjacency. Missing RAW edges do not prove separate surfaces, and observed edges do not prove legal manifold topology.'])
    return dict(summary=summary,queries=rows,query_vectors=vectors,hits=hits,intervals=intervals,layers=groups,candidate_faces=candidate_rows,near_witnesses=witness,graph=graph)


def _clean(value):
    if isinstance(value,dict):return {str(k):_clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_clean(v) for v in value]
    if isinstance(value,np.ndarray):return _clean(value.tolist())
    if isinstance(value,np.generic):return _clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def _write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=data:raise ValueError('Existing audit output differs: '+str(path))
        return
    temp=path.with_name(path.name+'.partial')
    with temp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def _csv(path,rows):
    rows=_clean(rows);fields=list(dict.fromkeys(k for row in rows for k in row)) or ['status'];s=io.StringIO(newline='');writer=csv.DictWriter(s,fieldnames=fields);writer.writeheader()
    for row in rows:writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    _write(path,s.getvalue().encode('utf-8-sig'))


def _npz(path,arrays):
    # np.savez embeds the current timestamp. Compare array values for recovery.
    if Path(path).exists():
        with np.load(path,allow_pickle=False) as z:
            if set(z.files)!=set(arrays) or any(z[k].dtype!=v.dtype or not np.array_equal(z[k],v,equal_nan=True) for k,v in arrays.items()):raise ValueError('Existing audit arrays differ')
        return
    buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);_write(path,buffer.getvalue())


def process_cell(cell,config,outdir):
    from . import common
    dest=Path(outdir).resolve();root=Path(common.OUT).resolve()
    if not dest.is_relative_to(root):raise ValueError('Audit writes outside new experiment')
    sources={}
    for key,value in cell.items():
        if key.endswith('_path') and cell.get(key[:-5]+'_sha256'):
            path=common.resolve(value);expected=cell[key[:-5]+'_sha256']
            if common.sha(path)!=expected:raise ValueError('Registered input hash differs: '+str(path))
            sources[str(path)]=expected
    geometry=common.load_geometry(cell);value=analyze_geometry(geometry,config)
    for path,expected in sources.items():
        if common.sha(path)!=expected:raise ValueError('Input changed during audit: '+path)
    for name,key in [('per_query.csv','queries'),('candidate_faces.csv','candidate_faces'),('normal_line_layers.csv','layers')]:_csv(dest/name,value[key])
    for name,key in [('query_geometry.npz','query_vectors'),('normal_line_hits.npz','hits'),('coplanar_line_intervals.npz','intervals'),('near_parallel_witnesses.npz','near_witnesses')]:_npz(dest/name,value[key])
    for name,key in [('free_face_geometry.csv','face_rows'),('parallel_regions.csv','parallel_regions'),('sidewall_adjacency.csv','sidewall_adjacency'),('exact_edges.csv','edges')]:_csv(dest/name,value['graph'][key])
    summary=dict(value['summary'],cell_id=cell.get('cell_id'),geometry_source_hashes=sources,implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(dest.iterdir()) if p.is_file() and p.name!='summary.json'};summary['files']=files
    _write(dest/'summary.json',(json.dumps(_clean(summary),indent=2,sort_keys=True,allow_nan=False)+'\n').encode('utf8'))
    return _clean(summary)
