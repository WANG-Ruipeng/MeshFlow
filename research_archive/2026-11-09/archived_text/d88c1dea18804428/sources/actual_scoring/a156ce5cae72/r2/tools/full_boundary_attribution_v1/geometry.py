"""Finite, source-boundary attribution of existing C/F meshes (pure NumPy).

No model, generation, mesh repair, source mutation or geometry-quality rescan.
CF/FF classifications are attached from the already saved exhaustive scans.
Every distance is to a finite surface/segment/vertex. Connectivity is the saved
indexed connectivity, including explicitly labelled nonmanifold incidences.
"""
from __future__ import annotations
from collections import defaultdict, Counter
import numpy as np
from r2.tools.cref_layer_audit_v1.geometry import (
    triangles, normals, finite_distances, normal_line_hits, layer_groups, exact_graph,
)

VERSION = 'full_source_boundary_finite_attribution_v1'
DEFAULTS = dict(boundary_step_over_L=.002, query_chunk=64,
                distance_thresholds_over_L=[.001,.005,.01,.02],
                normal_angle_degrees=15., tangent_angle_degrees=15.,
                area_floor_L2=1e-11, corridor_L=.1, layer_gap_L=1e-4,
                line_parallel_dot_epsilon=1e-12,
                coplanar_plane_epsilon_L=1e-9, barycentric_epsilon=1e-9)


def _float(v):
    return float(v) if np.isfinite(v) else None


def _xyz(name, point):
    return {name+'_'+axis:_float(value) for axis,value in zip('xyz',point)}


def _pair_book(pairs, K, M):
    """Validate recorded scan cardinality; no detector is called here."""
    books={}; flags={s:[[] for _ in range(M)] for s in ('CF','FF')}
    counts={s:[Counter() for _ in range(M)] for s in ('CF','FF')}
    for scope,total in (('CF',K*M),('FF',M*(M-1)//2)):
        scan=pairs.get(scope,{})
        exhaustive=(scan.get('status')=='PASS_STATIC_SCAN' and
                    int(scan.get('tested_pairs',-1))==total and
                    int(scan.get('expected_full_pairs',-1))==total)
        contacts={}
        for row in scan.get('contacts',[]):
            a,b=map(int,row['pair'])
            if scope=='CF':
                if not (0<=a<K and 0<=b<M):raise ValueError('CF local ID mismatch')
            elif not (0<=a<b<M):raise ValueError('FF local ID mismatch')
            if (a,b) in contacts:raise ValueError('Repeated saved contact pair')
            detector=row.get('detector',{})
            item=dict(classification=row.get('classification','UNKNOWN'),
                      illegal=detector.get('illegal'),
                      resolved=detector.get('resolved'),pair=[a,b])
            contacts[(a,b)]=item
            for fi in ([b] if scope=='CF' else [a,b]):
                counts[scope][fi][item['classification']]+=1
                if item['illegal'] is True:flags[scope][fi].append([a,b])
        books[scope]=dict(contacts=contacts,exhaustive=exhaustive,status=scan.get('status','MISSING'))
    return books,flags,counts


def _cf_relationship(books, cids, fid):
    if fid<0:return []
    book=books['CF'];result=[]
    for ci in cids:
        item=book['contacts'].get((int(ci),int(fid)))
        result.append(dict(C_face_row=int(ci),free_face_id=int(fid),
                           classification=item['classification'] if item else ('disjoint' if book['exhaustive'] else 'UNKNOWN_NOT_RECORDED'),
                           illegal=item['illegal'] if item else (False if book['exhaustive'] else None)))
    return result


def _source_boundary(C, ids, L, cfg):
    ids=np.asarray(ids,np.int64)
    if ids.shape!=(len(C),3) or np.any(ids<0):raise ValueError('Reliable original source vertex IDs required')
    coords={};incidence=defaultdict(list);vertex_faces=defaultdict(set)
    for fi,face in enumerate(ids):
        if len(set(map(int,face)))!=3:raise ValueError('Degenerate source C ID triangle')
        for vi,xyz in zip(face,C[fi]):
            vi=int(vi)
            if vi in coords and not np.array_equal(coords[vi],xyz):raise ValueError('Source ID has inconsistent coordinates')
            coords[vi]=xyz;vertex_faces[vi].add(fi)
        for a,b in ((0,1),(1,2),(2,0)):
            va,vb=int(face[a]),int(face[b]);incidence[tuple(sorted((va,vb)))].append((fi,va,vb))
    if any(len(owners)>2 for owners in incidence.values()):raise ValueError('Nonmanifold source C boundary is outside this registered panel')
    cn,cgood,ca=normals(C,L,cfg['area_floor_L2'])
    rows=[];samples=[];vertex_to_edges=defaultdict(list)
    step=float(cfg['boundary_step_over_L'])*L
    if not np.isfinite(step) or step<=0:raise ValueError('Positive boundary sampling step required')
    for edge,owners in sorted(incidence.items()):
        if len(owners)!=1:continue
        fi,va,vb=owners[0];a,b=coords[va],coords[vb];length=float(np.linalg.norm(b-a))
        if length<=0:raise ValueError('Zero-length source boundary edge')
        eid=len(rows);tangent=(b-a)/length;count=max(1,int(np.ceil(length/step)))
        rows.append(dict(boundary_edge_id=eid,source_vertex0=va,source_vertex1=vb,C_face_row=fi,
                         length=length,length_over_L=length/L,sample_count=count,
                         source_normal_good=bool(cgood[fi]),**_xyz('start',a),**_xyz('end',b),
                         **_xyz('tangent',tangent),**_xyz('source_normal',cn[fi])))
        vertex_to_edges[va].append(eid);vertex_to_edges[vb].append(eid)
        for j in range(count):
            t=(j+.5)/count
            samples.append(dict(kind='EDGE_MIDPOINT',boundary_edge_id=eid,source_vertex_id=-1,
                                t=t,point=(1-t)*a+t*b,normal=cn[fi],normal_good=bool(cgood[fi]),
                                tangent=tangent,C_face_row=fi,C_incident_faces=[fi],weight=length/count,
                                normal_basis='SOURCE_BOUNDARY_OWNER_FACE'))
    if not rows:raise ValueError('No true source-ID open C boundary')
    vertex_rows=[]
    for vi,eids in sorted(vertex_to_edges.items()):
        cids=sorted(vertex_faces[vi]);summed=np.sum(cn[cids]*ca[cids,None],axis=0)
        norm=np.linalg.norm(summed);n=summed/norm if norm>0 else np.zeros(3)
        vertex_rows.append(dict(boundary_vertex_id=len(vertex_rows),source_vertex_id=vi,
                                boundary_edge_ids=eids,incident_C_face_rows=cids,
                                source_normal_basis='AREA_WEIGHTED_INCIDENT_SOURCE_C_FACES',
                                **_xyz('point',coords[vi]),**_xyz('source_normal',n)))
        samples.append(dict(kind='BOUNDARY_VERTEX',boundary_edge_id=-1,source_vertex_id=vi,t=None,
                            point=coords[vi],normal=n,normal_good=bool(norm>0),tangent=np.zeros(3),
                            C_face_row=cids[0],C_incident_faces=cids,weight=0.,
                            normal_basis='AREA_WEIGHTED_INCIDENT_SOURCE_C_FACES'))
    return rows,vertex_rows,samples


def _finite_segments(points, starts, ends):
    q=np.asarray(points,np.float64);a=np.asarray(starts,np.float64).reshape(-1,3);b=np.asarray(ends,np.float64).reshape(-1,3)
    if not len(a):return np.empty((len(q),0)),np.empty((len(q),0,3)),np.empty((len(q),0))
    delta=b-a;den=np.einsum('ij,ij->i',delta,delta)
    t=np.divide(np.einsum('qei,ei->qe',q[:,None]-a[None],delta),den[None],out=np.zeros((len(q),len(a))),where=den[None]>0)
    t=np.clip(t,0,1);closest=a[None]+t[:,:,None]*delta[None]
    return np.sum((q[:,None]-closest)**2,axis=-1),closest,t


def _nearest_storage(Q):
    return dict(id=np.full(Q,-1,np.int64),distance=np.full(Q,np.inf),
                point=np.full((Q,3),np.nan),barycentric=np.full((Q,3),np.nan),
                parameter=np.full(Q,np.nan))


def _save_nearest(storage, start, d2, closest, ids=None, bary=None, parameter=None):
    if d2.shape[1]==0:return
    selected=np.argmin(d2,axis=1);ii=np.arange(len(d2));valid=np.isfinite(d2[ii,selected]);dest=np.arange(start,start+len(d2))[valid];take=selected[valid]
    storage['id'][dest]=take if ids is None else np.asarray(ids)[take]
    storage['distance'][dest]=np.sqrt(d2[ii[valid],take]);storage['point'][dest]=closest[ii[valid],take]
    if bary is not None:storage['barycentric'][dest]=bary[ii[valid],take]
    if parameter is not None:storage['parameter'][dest]=parameter[ii[valid],take]


def _concat(parts):
    return {k:np.concatenate([p[k] for p in parts],axis=0) for k in parts[0]} if parts else {}


def _weighted_quantile(values, weights, q):
    values=np.asarray(values);weights=np.asarray(weights)
    if not len(values) or weights.sum()<=0:return None
    order=np.argsort(values,kind='stable');cum=np.cumsum(weights[order])
    return _float(values[order[min(np.searchsorted(cum,q*cum[-1],side='left'),len(order)-1)]])


def _distance_summary(values, weights, L, cfg):
    den=float(np.sum(weights));value=np.asarray(values);result={}
    if den<=0:return dict(status='NA_ZERO_BOUNDARY_LENGTH')
    result.update(length_weighted_mean_over_L=_float(np.sum(value*weights)/den/L),
                  length_weighted_median_over_L=None,
                  length_weighted_p95_over_L=None,max_over_L=_float(value.max()/L))
    for key,q in (('length_weighted_median_over_L',.5),('length_weighted_p95_over_L',.95)):
        v=_weighted_quantile(value,weights,q);result[key]=None if v is None else v/L
    for tau in cfg['distance_thresholds_over_L']:
        result['length_fraction_lt_'+format(float(tau),'.9g')+'L']=float(weights@(value<float(tau)*L)/den)
    return result


def analyze(bundle, config=None):
    """Analyze one registered existing mesh, without writing any files.

    Required bundle: C_original, C_source_vertex_ids (same corner order), F,
    vertices, faces, known_mask, pairs, L. Indexed F rows must exactly reproduce
    F, while C faces may be cyclically reordered; source C identities are never
    inferred by proximity. graph_identity explains POST actual versus RAW exact
    coordinate observational indices. Returned arrays permit independent replay.
    """
    cfg=dict(DEFAULTS,**(config or {}));L=float(bundle.get('L',cfg.get('L',0)))
    if not np.isfinite(L) or L<=0:raise ValueError('Inherited L must be positive')
    C=triangles(bundle['C_original']);F=triangles(bundle['F']);K,M=len(C),len(F)
    v=np.asarray(bundle.get('vertices',bundle.get('mesh_vertices')),np.float64)
    faces=np.asarray(bundle.get('faces',bundle.get('mesh_faces')),np.int64)
    known=np.asarray(bundle.get('known_mask',np.arange(len(faces))<K),bool)
    graph=exact_graph(v,faces,known);cglobal=np.flatnonzero(known);fglobal=np.flatnonzero(~known)
    if len(cglobal)!=K or len(fglobal)!=M:raise ValueError('Indexed C/F row identity mismatch')
    indexed=v[faces]
    if not np.array_equal(indexed[fglobal],F):raise ValueError('Saved indexed F row/corner identity mismatch')
    for a,b in zip(indexed[cglobal],C):
        if sorted(map(tuple,a))!=sorted(map(tuple,b)):raise ValueError('Saved C row does not match original C surface')
    source_ids=bundle.get('C_source_vertex_ids',bundle.get('C_original_source_vertex_ids'))
    boundary,vertices,samples=_source_boundary(C,source_ids,L,cfg)
    books,flags,classification_counts=_pair_book(bundle.get('pairs',{}),K,M)
    free_map={int(g):i for i,g in enumerate(fglobal)};cmap={int(g):i for i,g in enumerate(cglobal)}
    fn,fgood,fa=normals(F,L,cfg['area_floor_L2']);free_edges=[];edge_starts=[];edge_ends=[]
    fvertex_owners=defaultdict(set)
    for fi,g in enumerate(fglobal):
        for vi in faces[g]:fvertex_owners[int(vi)].add(fi)
    for row in graph['edges']:
        owners=[free_map[g] for g in row['incident_face_ids'] if g in free_map]
        if not owners:continue
        ci=[cmap[g] for g in row['incident_face_ids'] if g in cmap]
        raw_f_incidence=sum(sum(1 for a,b in ((0,1),(1,2),(2,0)) if tuple(sorted((int(faces[g,a]),int(faces[g,b]))))==(row['vertex0'],row['vertex1'])) for g in (fglobal[i] for i in owners))
        a,b=v[row['vertex0']],v[row['vertex1']];length=float(np.linalg.norm(b-a))
        tangent=(b-a)/length if length>0 else np.zeros(3)
        free_edges.append(dict(free_edge_id=len(free_edges),vertex0=row['vertex0'],vertex1=row['vertex1'],
                               free_face_ids=owners,C_face_rows=ci,F_incidence=raw_f_incidence,
                               F_distinct_faces=len(owners),F_open=raw_f_incidence==1,
                               whole_mesh_boundary=row['boundary'],nonmanifold=row['nonmanifold'],
                               length=length,length_over_L=length/L,free_component_ids=sorted(set(int(graph['free_component_id'][fglobal[i]]) for i in owners)),
                               incident_CF_illegal_free_faces=[i for i in owners if flags['CF'][i]],
                               incident_FF_illegal_free_faces=[i for i in owners if flags['FF'][i]],
                               **_xyz('start',a),**_xyz('end',b),**_xyz('tangent',tangent)))
        edge_starts.append(a);edge_ends.append(b)
    edge_starts=np.asarray(edge_starts).reshape(-1,3);edge_ends=np.asarray(edge_ends).reshape(-1,3)
    openids=np.asarray([e['free_edge_id'] for e in free_edges if e['F_open']],np.int64)
    fvids=np.asarray(sorted(fvertex_owners),np.int64);fverts=v[fvids]
    free_vertex_rows=[dict(free_vertex_row=i,actual_vertex_id=int(vi),free_face_ids=sorted(fvertex_owners[int(vi)]),
                           free_component_ids=sorted(set(int(graph['free_component_id'][fglobal[j]]) for j in fvertex_owners[int(vi)])),
                           **_xyz('point',v[vi])) for i,vi in enumerate(fvids)]
    q=np.array([s['point'] for s in samples]);direction=np.array([s['normal'] for s in samples]);tangents=np.array([s['tangent'] for s in samples]);Q=len(q)
    stores={key:_nearest_storage(Q) for key in ('surface','parallel_surface','edge','open_edge','vertex')}
    hitparts=[];intervalparts=[];cosnorm=np.cos(np.deg2rad(cfg['normal_angle_degrees']))
    for start in range(0,Q,int(cfg['query_chunk'])):
        stop=min(Q,start+int(cfg['query_chunk']));chunk=q[start:stop]
        d2,closest,bary=finite_distances(chunk,F)
        _save_nearest(stores['surface'],start,d2,closest,bary=bary)
        aligned=(np.abs(direction[start:stop]@fn.T)>=cosnorm)&fgood[None]
        _save_nearest(stores['parallel_surface'],start,np.where(aligned,d2,np.inf),closest,bary=bary)
        ed2,ep,et=_finite_segments(chunk,edge_starts,edge_ends)
        _save_nearest(stores['edge'],start,ed2,ep,parameter=et)
        _save_nearest(stores['open_edge'],start,ed2[:,openids],ep[:,openids],ids=openids,parameter=et[:,openids])
        vd2=np.sum((chunk[:,None]-fverts[None])**2,axis=-1)
        _save_nearest(stores['vertex'],start,vd2,np.broadcast_to(fverts[None],(len(chunk),len(fverts),3)))
        hit,interval=normal_line_hits(chunk,direction[start:stop],F,L,cfg)
        hit['query_id']+=start;interval['query_id']+=start;hitparts.append(hit);intervalparts.append(interval)
    hits=_concat(hitparts);intervals=_concat(intervalparts)
    layers,groupids,groupcounts=layer_groups(hits,Q,L,cfg);hits['layer_group_id']=groupids
    hitcount=np.bincount(hits['query_id'],minlength=Q);corridorcount=np.bincount(hits['query_id'][hits['in_corridor']],minlength=Q)
    intervalcount=np.bincount(intervals['query_id'],minlength=Q)
    rows=[]
    for i,s in enumerate(samples):
        row=dict(sample_id=i,sample_kind=s['kind'],boundary_edge_id=s['boundary_edge_id'],
                 source_vertex_id=s['source_vertex_id'],edge_parameter=s['t'],
                 C_face_row=s['C_face_row'],C_incident_face_rows=s['C_incident_faces'],
                 length_weight=s['weight'],source_normal_basis=s['normal_basis'],source_normal_good=s['normal_good'],
                 **_xyz('point',q[i]),**_xyz('source_normal',direction[i]),**_xyz('boundary_tangent',tangents[i]))
        for name,store in stores.items():
            ident=int(store['id'][i]);delta=store['point'][i]-q[i]
            row[name+'_id']=ident;row[name+'_distance']=_float(store['distance'][i]);row[name+'_distance_over_L']=_float(store['distance'][i]/L)
            row[name+'_signed_source_normal_offset_over_L']=_float(delta@direction[i]/L)
            row.update(_xyz(name+'_point',store['point'][i]))
            if name in ('surface','parallel_surface'):
                row[name+'_barycentric']=[_float(x) for x in store['barycentric'][i]]
                row[name+'_normal_abs_dot']=_float(abs(fn[ident]@direction[i])) if ident>=0 else None
                row[name+'_parallel_15deg']=bool(ident>=0 and fgood[ident] and abs(fn[ident]@direction[i])>=cosnorm)
                row[name+'_free_component_id']=int(graph['free_component_id'][fglobal[ident]]) if ident>=0 else -1
                row[name+'_saved_CF_any_illegal']=bool(flags['CF'][ident]) if ident>=0 else None
                row[name+'_saved_FF_any_illegal']=bool(flags['FF'][ident]) if ident>=0 else None
                row[name+'_saved_CF_relationship']=_cf_relationship(books,s['C_incident_faces'],ident)
            elif name in ('edge','open_edge'):
                edge=free_edges[ident] if ident>=0 else None
                row[name+'_segment_parameter']=_float(store['parameter'][i]);row[name+'_tangent_abs_dot']=None
                if edge:
                    tangent=np.array([edge['tangent_'+a] for a in 'xyz'])
                    row[name+'_tangent_abs_dot']=_float(abs(tangent@tangents[i])) if s['kind']=='EDGE_MIDPOINT' else None
                    for key in ('free_face_ids','C_face_rows','F_incidence','F_open','whole_mesh_boundary','nonmanifold','free_component_ids','incident_CF_illegal_free_faces','incident_FF_illegal_free_faces'):
                        row[name+'_'+key]=edge[key]
            else:
                row[name+'_actual_vertex_id']=int(fvids[ident]) if ident>=0 else -1
                row[name+'_incident_free_faces']=sorted(fvertex_owners[int(fvids[ident])]) if ident>=0 else []
        row.update(all_isolated_normal_hits=int(hitcount[i]),corridor_isolated_normal_hits=int(corridorcount[i]),
                   corridor_layer_groups=int(groupcounts[i]),coplanar_line_intervals=int(intervalcount[i]))
        rows.append(row)
    weights=np.asarray([s['weight'] for s in samples]);edge_mask=weights>0;vertex_mask=~edge_mask
    edge_rows=[]
    for edge in boundary:
        chosen=np.array([s['boundary_edge_id']==edge['boundary_edge_id'] for s in samples]);w=weights[chosen]
        item=dict(edge)
        for name,store in stores.items():item[name]=_distance_summary(store['distance'][chosen],w,L,cfg)
        nearest=stores['surface']['id'][chosen]
        item['nearest_surface_face_ids']=sorted(set(int(x) for x in nearest if x>=0))
        item['nearest_surface_CF_illegal_length_fraction']=float(sum(ww for ww,j in zip(w,nearest) if j>=0 and flags['CF'][j])/w.sum())
        item['nearest_surface_FF_illegal_length_fraction']=float(sum(ww for ww,j in zip(w,nearest) if j>=0 and flags['FF'][j])/w.sum())
        item['multi_normal_layer_length_fraction']=float(w@(groupcounts[chosen]>=2)/w.sum())
        edge_rows.append(item)
    nearest_counts=np.bincount(stores['surface']['id'][stores['surface']['id']>=0],minlength=M)
    candidate=np.zeros(M,bool)
    for store in (stores['surface'],stores['parallel_surface']):candidate[store['id'][store['id']>=0]]=True
    candidate[hits['free_face_id']]=True;candidate[intervals['free_face_id']]=True
    free_face_rows=[]
    for fi in range(M):
        free_face_rows.append(dict(free_face_id=fi,global_face_id=int(fglobal[fi]),
                                  free_component_id=int(graph['free_component_id'][fglobal[fi]]),
                                  all_component_id=int(graph['all_component_id'][fglobal[fi]]),area=float(fa[fi]),normal_good=bool(fgood[fi]),
                                  nearest_query_count=int(nearest_counts[fi]),candidate_nearest_or_normal_hit=bool(candidate[fi]),
                                  saved_CF_illegal_pairs=flags['CF'][fi],saved_FF_illegal_pairs=flags['FF'][fi],
                                  saved_CF_contact_classifications=dict(classification_counts['CF'][fi]),
                                  saved_FF_contact_classifications=dict(classification_counts['FF'][fi]),**_xyz('normal',fn[fi])))
    vectors=dict(points=q,source_normals=direction,boundary_tangents=tangents,
                 sample_kind=np.asarray([0 if s['kind']=='EDGE_MIDPOINT' else 1 for s in samples],np.int8),
                 boundary_edge_id=np.asarray([s['boundary_edge_id'] for s in samples],np.int64),
                 source_vertex_id=np.asarray([s['source_vertex_id'] for s in samples],np.int64),
                 source_C_face_row=np.asarray([s['C_face_row'] for s in samples],np.int64),
                 length_weights=weights,corridor_layer_count=groupcounts,
                 F_component_ids=graph['free_component_id'][fglobal],F_actual_global_face_ids=fglobal,
                 F_vertex_actual_ids=fvids,F_open_edge_ids=openids)
    for name,store in stores.items():
        for key,value in store.items():vectors[name+'_'+key]=value
    summary=dict(version=VERSION,L=L,config=cfg,source_C_faces=K,free_faces=M,
                 source_boundary_edges=len(boundary),source_boundary_vertices=len(vertices),
                 source_boundary_length=float(weights.sum()),edge_midpoint_queries=int(edge_mask.sum()),
                 boundary_vertex_queries=int(vertex_mask.sum()),sampling_max_step_L=float(cfg['boundary_step_over_L']),
                 observed_F_edges=len(free_edges),observed_F_open_edges=len(openids),observed_F_vertices=len(fvids),
                 observed_F_components=len(graph['free_components']),graph_identity=bundle.get('graph_identity','UNSPECIFIED'),
                 saved_classification_status={s:dict(status=b['status'],exhaustive=b['exhaustive']) for s,b in books.items()},
                 boundary_distances={name:_distance_summary(st['distance'][edge_mask],weights[edge_mask],L,cfg) for name,st in stores.items()},
                 boundary_vertex_distances={name:dict(mean_over_L=_float(np.mean(st['distance'][vertex_mask])/L),
                                                     max_over_L=_float(np.max(st['distance'][vertex_mask])/L)) for name,st in stores.items()},
                 isolated_normal_hits=len(hits['query_id']),coplanar_line_intervals=len(intervals['query_id']),
                 normal_layer_groups=len(layers),new_model_calls=0,new_quality_scans=0,
                 interpretation=[
                     'C boundary comes from original source vertex-ID incidence, not triangle soup or proximity.',
                     'Edge queries are deterministic segment midpoints with length weights; boundary vertices are separate, zero-length-weight observations.',
                     'F-open means exactly one incident F edge occurrence, even when C is also incident. Whole-mesh boundary is separately retained.',
                     'Distances and normal signs describe finite candidates; small distance does not certify legal attachment, semantic use or thickness.',
                     'Saved CF/FF classifications are annotations only. No old quality detector is re-run.',
                     'Actual component IDs include nonmanifold and illegal incidences, not a certified legal connection graph.',
                     'Source-normal signs are not inside/outside; vertex normals use explicitly labelled area-weighted incident source faces.',
                     'Every normal-line hit and coplanar interval is retained; multiple hits at a triangulation edge need not be multiple layers.',
                 ])
    return dict(summary=summary,boundary_edges=boundary,boundary_vertices=vertices,samples=rows,
                edge_summary=edge_rows,free_edges=free_edges,free_vertices=free_vertex_rows,
                free_faces=free_face_rows,normal_line_hits=hits,coplanar_intervals=intervals,layers=layers,vectors=vectors)
