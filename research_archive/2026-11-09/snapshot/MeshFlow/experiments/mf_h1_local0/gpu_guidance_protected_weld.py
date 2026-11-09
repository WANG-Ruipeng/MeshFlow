"""Fixed CPU protected-coordinate weld W and independent indexed topology audit.

No GT, model, optimizer, collision search, face deletion, remeshing, or new
boundary vertex is used. Geometry is reassigned only to original FP32 anchors.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import hashlib
import itertools
import math
from pathlib import Path
import tempfile
import time
import numpy as np

WELD_POLICY=dict(epsilon_bbox=1e-3,area_epsilon_bbox_squared=1e-12,
    normal_acceptance='Actual FP32 proposed normal dot original Raw face normal strictly greater than zero',
    protected_identity='Exact FP32 coordinate bit pattern, including signed zero; unequal protected coordinates never merge',
    processing_order='Numeric coordinate lexicographic, then sorted geometric corner-neighborhood coordinates; bit patterns break signed-zero ties',
    indistinguishable_corner_tie='Stable ordinal only among identical coordinate and identical unordered triangle geometry; interchangeable geometric records',
    protected_choice='Minimum squared distance then protected coordinate lexicographic; only first candidate, no farther-anchor fallback after rejection',
    free_choice='Nearest fixed existing representative then coordinate lexicographic; rejected point starts its own fixed representative',
    free_representatives='Original FP32 points; never moved or replaced; absorbed points never become representatives',
    maximum_original_vertex_displacement_bbox=1e-3,
    global_collision_checks_per_candidate=0,t_junction_distance_epsilon_bbox=1e-9,
    topology_audit='Exhaustive indexed edges and vertex links; exhaustive non-endpoint vertex versus edge T-junction scan',
    topology_welding_guarantee=False,collision_safety_guarantee=False)


def _triangles(a):
    a=np.asarray(a)
    if a.ndim==2 and a.shape[1]==9:return a.reshape(-1,3,3)
    if a.ndim==3 and a.shape[1:]==(3,3):return a
    raise ValueError('Triangles must have shape [N,9] or [N,3,3]')


def _bits(p):return tuple(map(int,np.asarray(p,dtype=np.float32).view(np.uint32)))
def _lex(p):return tuple(map(float,p))+_bits(p)
def _bit_equal(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


def _hash(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str((str(a.dtype),tuple(a.shape))).encode()+a.tobytes()).hexdigest()


def _file_hash(p):
    digest=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):digest.update(chunk)
    return digest.hexdigest()


def _normals(triangles):
    a=np.asarray(triangles,dtype=np.float64)
    return np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0])


def _inputs(raw,C,known_mask=None):
    raw=_triangles(raw);C=_triangles(C)
    if raw.dtype!=np.float32 or C.dtype!=np.float32 or not np.isfinite(raw).all() or not np.isfinite(C).all():
        raise ValueError('Raw and C must be finite FP32 arrays')
    if known_mask is None:
        known_mask=np.arange(len(raw))<len(C)
    known_mask=np.asarray(known_mask)
    if known_mask.dtype!=np.bool_ or known_mask.shape!=(len(raw),) or int(known_mask.sum())!=len(C):
        raise ValueError('Separate bool[N] known mask must select exactly C')
    if not _bit_equal(raw[known_mask],C):raise ValueError('Raw known triangles do not preserve exact original C')
    return raw,C,known_mask.copy()


def _corner_order(triangles,free_ids):
    keys=[]
    for fi in free_ids:
        neighborhood=tuple(sorted(_lex(p) for p in triangles[fi]))
        for corner in range(3):keys.append((_lex(triangles[fi,corner]),neighborhood,int(fi),corner))
    # IDs are a final tie only for identical geometric corner-neighborhoods.
    keys.sort(key=lambda row:(row[0],row[1],row[2],row[3]))
    return [(row[2],row[3]) for row in keys]


def _compact(vertices,faces,protected):
    used=np.unique(faces)
    order=sorted(map(int,used),key=lambda i:(_lex(vertices[i]),0 if i in protected else 1,i))
    remap=np.full(len(vertices),-1,dtype=np.int64);remap[order]=np.arange(len(order))
    return vertices[order].copy(),remap[faces],np.asarray([remap[p] for p in sorted(protected)],dtype=np.int64)


def _finish(vertices,faces,raw,C,known,protected,audit):
    triangles=np.asarray(vertices,dtype=np.float32)[faces]
    if faces.shape!=(len(raw),3) or not _bit_equal(triangles[known],C):raise AssertionError('Face count or protected C changed')
    if not np.isfinite(triangles).all():raise AssertionError('Nonfinite W result')
    audit.update(C_PRESERVED='PASS_BITWISE_WITH_CORNER_CORRESPONDENCE',known_face_indices=np.flatnonzero(known).tolist(),
        known_faces=len(C),face_count_before=len(raw),face_count_after=len(faces),free_faces=int((~known).sum()),
        source_raw_sha256=_hash(raw),source_C_sha256=_hash(C),output_triangles_sha256=_hash(triangles),
        vertices_sha256=_hash(vertices),faces_sha256=_hash(faces),
        COMBINATORIAL_CLOSURE_STATUS='NOT_RUN_USE_INDEPENDENT_INDEXED_AUDIT',
        EMBEDDED_GEOMETRY_VALIDITY_STATUS='NOT_RUN_REQUIRES_INDEPENDENT_CF_FF',
        candidate_collision_checks=0,GT_inputs=False,model_forwards=0,optimizer_updates=0)
    return dict(vertices=vertices,faces=faces,triangles=triangles,output=triangles.reshape(-1,9),
        known_vertex_indices=faces[known].copy(),protected_vertex_indices=np.asarray(protected,dtype=np.int64),known_mask=known,audit=audit)


def raw_exact_index(raw,C,known_mask=None):
    start=time.perf_counter();raw,C,known=_inputs(raw,C,known_mask)
    points={_bits(p):p.copy() for p in raw.reshape(-1,3)}
    ordered=sorted(points,key=lambda key:_lex(points[key]));mapping={key:i for i,key in enumerate(ordered)}
    vertices=np.asarray([points[key] for key in ordered],dtype=np.float32).reshape(-1,3)
    faces=np.asarray([mapping[_bits(p)] for p in raw.reshape(-1,3)],dtype=np.int64).reshape(-1,3)
    protected=np.unique(faces[known])
    return _finish(vertices,faces,raw,C,known,protected,dict(method='Raw exact FP32 coordinate indexing only',
        tolerance_welding=False,coordinate_changes=0,vertex_count=len(vertices),accepted_merges=0,rejected_merges=0,
        indexing_seconds=time.perf_counter()-start,operator_seconds=0.))


def _merge_candidate(vertices,faces,source,destination,protected,known,reference_normals,L):
    """Local tests of actual FP32 representative reassignment; no collisions."""
    affected=np.flatnonzero(np.any(faces==source,axis=1));reasons=[]
    if source==destination:return False,faces,dict(reasons=['SAME_VERTEX_INDEX'],affected_faces=affected.tolist())
    if source in protected and _bits(vertices[source])!=_bits(vertices[destination]):reasons.append('PROTECTED_COORDINATE_CONFLICT')
    proposed=faces[affected].copy();proposed[proposed==source]=destination
    if any(len(set(row.tolist()))<3 for row in proposed):reasons.append('REPEATED_VERTEX_INDEX_IN_FACE')
    new_triangles=vertices[proposed] # These are the actual FP32 export positions.
    new_normals=_normals(new_triangles);new_area=np.linalg.norm(new_normals,axis=1)*.5
    old_area=np.linalg.norm(_normals(vertices[faces[affected]]),axis=1)*.5
    ref=reference_normals[affected];ref_area=np.linalg.norm(ref,axis=1)*.5
    if np.any(new_area<=1e-12*L*L):reasons.append('ZERO_OR_NEAR_DEGENERATE_PROPOSAL')
    if np.any(ref_area<=1e-12*L*L):reasons.append('UNRESOLVED_ORIGINAL_NEAR_DEGENERATE_NORMAL')
    if np.any(np.einsum('ij,ij->i',new_normals,ref)<=0):reasons.append('ORIGINAL_NORMAL_FLIP_OR_ORTHOGONAL')
    if np.any(known[affected]):
        for before,after,is_known in zip(faces[affected],proposed,known[affected]):
            if is_known and not _bit_equal(vertices[before],vertices[after]):reasons.append('KNOWN_FACE_CORRESPONDENCE_CHANGED')
    record=dict(reasons=sorted(set(reasons)),affected_faces=affected.tolist(),old_areas=old_area.tolist(),new_areas=new_area.tolist(),
        actual_FP32_representative=True,normal_reference='Original Raw face normal',collision_checks=0)
    if reasons:return False,faces,record
    updated=faces.copy();updated[affected]=proposed
    return True,updated,record


def protected_weld(raw,C,L,known_mask=None):
    total_start=time.perf_counter();raw,C,known=_inputs(raw,C,known_mask)
    if not math.isfinite(L) or L<=0:raise ValueError('Positive declared L required')
    L=float(L);epsilon=1e-3*L;source_raw=raw.copy();source_C=C.copy();reference_normals=_normals(raw)
    build_start=time.perf_counter()
    protected_points={_bits(p):p.copy() for p in C.reshape(-1,3)}
    protected_keys=sorted(protected_points,key=lambda key:_lex(protected_points[key]))
    mapping={key:i for i,key in enumerate(protected_keys)}
    vertices=[protected_points[key] for key in protected_keys];protected=set(range(len(vertices)))
    faces=np.empty((len(raw),3),np.int64)
    for fi in np.flatnonzero(known):faces[fi]=[mapping[_bits(p)] for p in raw[fi]]
    order=_corner_order(raw,np.flatnonzero(~known));sources=[]
    for fi,corner in order:
        source=len(vertices);vertices.append(raw[fi,corner].copy());faces[fi,corner]=source;sources.append((source,fi,corner))
    vertices=np.asarray(vertices,dtype=np.float32).reshape(-1,3)
    build_seconds=time.perf_counter()-build_start
    operator_start=time.perf_counter();accepted=rejected=attempted=0;reasons=Counter();records=[];attached=set();fixed_representatives=[]
    protected_order=sorted(protected,key=lambda i:_lex(vertices[i]))
    def propose(source,destination,stage):
        nonlocal faces,accepted,rejected,attempted
        attempted+=1
        distance=float(np.linalg.norm(vertices[source].astype(np.float64)-vertices[destination].astype(np.float64)))
        if distance>epsilon:raise AssertionError('W proposed displacement beyond epsilon')
        ok,newfaces,details=_merge_candidate(vertices,faces,source,destination,protected,known,reference_normals,L)
        record=dict(stage=stage,source_vertex=source,representative_vertex=destination,
            source_coordinate=vertices[source].tolist(),representative_coordinate=vertices[destination].tolist(),
            displacement=distance,status='ACCEPTED' if ok else 'REJECTED',**details)
        records.append(record)
        if ok:faces=newfaces;accepted+=1;attached.add(source)
        else:rejected+=1;reasons.update(details['reasons'])
        return ok
    for source,fi,corner in sources:
        if not protected_order:continue
        squared=np.sum((vertices[protected_order].astype(np.float64)-vertices[source].astype(np.float64))**2,axis=1)
        eligible=[int(i) for i in np.flatnonzero(squared<=epsilon*epsilon)]
        if eligible:
            chosen=min(eligible,key=lambda j:(float(squared[j]),_lex(vertices[protected_order[j]])))
            propose(source,protected_order[chosen],'free_to_protected')
    for source,fi,corner in sources:
        if source in attached:continue
        candidates=[]
        for representative in fixed_representatives:
            squared=float(np.sum((vertices[source].astype(np.float64)-vertices[representative].astype(np.float64))**2))
            if squared<=epsilon*epsilon:candidates.append((squared,_lex(vertices[representative]),representative))
        if candidates:
            destination=min(candidates)[2]
            if propose(source,destination,'free_to_fixed_free_representative'):continue
        fixed_representatives.append(source)
    operator_seconds=time.perf_counter()-operator_start
    compact_start=time.perf_counter();vertices,faces,protected_final=_compact(vertices,faces,protected)
    compact_seconds=time.perf_counter()-compact_start
    output=vertices[faces];lengths=np.linalg.norm(output.astype(np.float64)-raw.astype(np.float64),axis=-1)
    if float(lengths.max(initial=0))>epsilon:raise AssertionError('A final cluster member moved beyond epsilon; chain detected')
    if not _bit_equal(raw,source_raw) or not _bit_equal(C,source_C):raise AssertionError('W changed its original input')
    raw_unique=len({_bits(p) for p in raw.reshape(-1,3)})
    audit=dict(method='W_fixed_protected_vertex_weld',policy=dict(WELD_POLICY),L=L,epsilon=epsilon,
        vertex_count_raw_exact=raw_unique,vertex_count_after=len(vertices),vertex_count_change=len(vertices)-raw_unique,
        initial_corner_index_vertices=len(protected_keys)+len(sources),protected_unique_vertices=len(protected_keys),
        attempted_merges=attempted,accepted_merges=accepted,rejected_merges=rejected,rejection_reasons=dict(reasons),
        accepted_free_to_protected=sum(r['stage']=='free_to_protected' and r['status']=='ACCEPTED' for r in records),
        accepted_free_to_free=sum(r['stage']=='free_to_fixed_free_representative' and r['status']=='ACCEPTED' for r in records),
        fixed_free_representatives=len(fixed_representatives),merge_records=records,
        max_coordinate_displacement=float(lengths.max(initial=0)),vertex_slot_displacement_RMS=float(np.sqrt(np.mean(lengths**2))),
        changed_vertex_slots=int(np.count_nonzero(np.any(output.view(np.uint32)!=raw.view(np.uint32),axis=-1))),
        fixed_representatives_never_move=True,no_transitive_chains=True,original_inputs_unchanged=True,
        local_guard_checks=attempted,indexing_seconds=build_seconds+compact_seconds,operator_seconds=operator_seconds,
        total_seconds=time.perf_counter()-total_start,no_face_deletion=True,no_added_vertices_on_C_edges=True,
        protected_coordinates_not_tolerance_merged=True,post_weld_collision_status='NOT_RUN_REQUIRES_INDEPENDENT_CF_FF')
    return _finish(vertices,faces,raw,C,known,protected_final,audit)


def audit_indexed_topology(vertices,faces,L,*,C=None,known_mask=None):
    """Independent index audit; coordinate overlap/intersection is not inferred."""
    start=time.perf_counter();v=np.asarray(vertices);f=np.asarray(faces)
    if v.dtype!=np.float32 or v.ndim!=2 or v.shape[1]!=3 or f.dtype!=np.int64 or f.ndim!=2 or f.shape[1]!=3:
        raise ValueError('Use FP32 vertices[V,3] and int64 faces[N,3]')
    if not np.isfinite(v).all() or np.any(f<0) or np.any(f>=len(v)) or not math.isfinite(L) or L<=0:
        raise ValueError('Invalid indexed mesh')
    triangles=v[f];area=np.linalg.norm(_normals(triangles),axis=-1)*.5
    edges=defaultdict(list);vertex_faces=defaultdict(list);collapsed=0
    for fi,face in enumerate(f):
        for vertex in set(map(int,face)):vertex_faces[vertex].append(fi)
        for a,b in zip(face,np.roll(face,-1)):
            a=int(a);b=int(b)
            if a==b:collapsed+=1;continue
            edges[tuple(sorted((a,b)))].append((fi,a,b))
    histogram=Counter(len(rows) for rows in edges.values());same_direction=[]
    for edge,rows in edges.items():
        if len(rows)==2 and rows[0][1:]==rows[1][1:]:same_direction.append(list(edge))
    duplicates=Counter(tuple(sorted(map(int,face))) for face in f)
    geometric_duplicates=Counter(tuple(sorted(_bits(p) for p in tri)) for tri in triangles)
    repeated=sum(len(set(map(int,face)))<3 for face in f)
    adjacency=[set() for _ in f]
    for rows in edges.values():
        incident={row[0] for row in rows}
        for fi in incident:adjacency[fi].update(incident-{fi})
    components=[];visited=set()
    for first in range(len(f)):
        if first in visited:continue
        pending=[first];component=[]
        while pending:
            current=pending.pop()
            if current in visited:continue
            visited.add(current);component.append(current);pending.extend(adjacency[current]-visited)
        components.append(sorted(component))
    link_failures=[];link_labels=[];interior_count=boundary_count=0
    for vertex in range(len(v)):
        incident=vertex_faces.get(vertex,[]);link_edges=[];invalid=False
        for fi in incident:
            face=list(map(int,f[fi]));others=[i for i in face if i!=vertex]
            if len(others)!=2 or others[0]==others[1]:invalid=True;continue
            link_edges.append(tuple(sorted(others)))
        link_adj=defaultdict(list)
        for a,b in link_edges:link_adj[a].append(b);link_adj[b].append(a)
        unique_edges=len(set(link_edges))==len(link_edges)
        link_seen=set();link_components=0
        for first in link_adj:
            if first in link_seen:continue
            link_components+=1;pending=[first]
            while pending:
                current=pending.pop()
                if current in link_seen:continue
                link_seen.add(current);pending.extend(q for q in link_adj[current] if q not in link_seen)
        degree=[len(rows) for rows in link_adj.values()]
        boundary_incident=sum(len(rows)==1 and vertex in edge for edge,rows in edges.items())
        cycle=(not invalid and unique_edges and link_components==1 and len(degree)>=3 and all(q==2 for q in degree) and boundary_incident==0)
        path=(not invalid and unique_edges and link_components==1 and len(degree)>=2 and degree.count(1)==2 and all(q in (1,2) for q in degree) and boundary_incident==2)
        if cycle:label='MANIFOLD_INTERIOR_CYCLE';interior_count+=1
        elif path:label='MANIFOLD_BOUNDARY_PATH';boundary_count+=1
        elif not incident:label='UNREFERENCED_VERTEX'
        else:
            label='NONMANIFOLD_VERTEX_LINK'
            link_failures.append(dict(vertex=vertex,incident_faces=incident,link_components=link_components,
                link_degree=degree,duplicate_link_edge=not unique_edges,boundary_incident_edges=boundary_incident,invalid_incident_face=invalid))
        link_labels.append(label)
    edge_list=np.asarray(sorted(edges),dtype=np.int64).reshape(-1,2)
    tj_hits=[];pair_count=0;skipped_short=0
    if len(v) and len(edge_list):
        segments=v[edge_list].astype(np.float64);direction=segments[:,1]-segments[:,0]
        length2=np.einsum('ei,ei->e',direction,direction);length=np.sqrt(length2)
        relative=v.astype(np.float64)[:,None]-segments[None,:,0]
        alpha=np.divide(np.einsum('vei,ei->ve',relative,direction),length2[None],out=np.zeros((len(v),len(edge_list))),where=length2[None]>0)
        closest=segments[None,:,0]+alpha[:,:,None]*direction[None]
        distance=np.linalg.norm(v.astype(np.float64)[:,None]-closest,axis=-1)
        epsilon=1e-9*L
        relative_epsilon=np.divide(epsilon,length,out=np.full_like(length,np.inf),where=length>0)
        vertex_ids=np.arange(len(v))[:,None]
        nonendpoint=(vertex_ids!=edge_list[None,:,0])&(vertex_ids!=edge_list[None,:,1])
        pair_count=int(nonendpoint.sum());skipped_short=int((length<=2*epsilon).sum())
        hits=nonendpoint&(alpha>relative_epsilon[None])&(alpha<1-relative_epsilon[None])&(distance<=epsilon)
        for vi,ei in zip(*np.where(hits)):
            tj_hits.append(dict(vertex=int(vi),edge=edge_list[ei].tolist(),fraction=float(alpha[vi,ei]),
                distance=float(distance[vi,ei]),edge_incident_faces=[row[0] for row in edges[tuple(edge_list[ei])]]))
    nonmanifold_edges=sum(len(rows)>2 for rows in edges.values());duplicate_faces=sum(n-1 for n in duplicates.values())
    orientation=not same_direction and not nonmanifold_edges and not repeated
    vertex_manifold=not link_failures and all(label!='UNREFERENCED_VERTEX' for label in link_labels)
    all_two=bool(len(edges)) and all(len(rows)==2 for rows in edges.values())
    if not len(f):closure='EMPTY_NOT_CLOSED'
    elif repeated or duplicate_faces:closure='INVALID_REPEATED_INDEX_OR_DUPLICATE_FACE'
    elif nonmanifold_edges or not vertex_manifold:closure='NONMANIFOLD_OPEN' if histogram.get(1,0) else 'NONMANIFOLD_DESPITE_CLOSED_EDGE_INCIDENCE'
    elif histogram.get(1,0):closure='OPEN_BOUNDARY'
    elif all_two and not orientation:closure='CLOSED_WITH_INCONSISTENT_ORIENTATION'
    elif all_two and vertex_manifold and orientation:closure='CLOSED_ORIENTED_COMBINATORIAL_2_MANIFOLD'
    else:closure='UNRESOLVED_COMBINATORIAL_STRUCTURE'
    preserved='NOT_CHECKED_NO_C_ARGUMENT'
    if C is not None:
        C=_triangles(C)
        known=np.arange(len(f))<len(C) if known_mask is None else np.asarray(known_mask,dtype=bool)
        preserved='PASS_BITWISE_WITH_CORNER_CORRESPONDENCE' if known.shape==(len(f),) and _bit_equal(triangles[known],C) else 'FAIL'
    return dict(status='PASS_AUDIT_EXECUTION',vertex_count=len(v),face_count=len(f),
        edge_count=len(edges),edge_incidence_counts={'1':histogram.get(1,0),'2':histogram.get(2,0),'>2':nonmanifold_edges},
        edge_incidence_histogram={str(k):int(n) for k,n in sorted(histogram.items())},all_edges_incident_two=all_two,
        collapsed_index_edges=collapsed,repeated_vertex_index_faces=repeated,duplicate_faces=duplicate_faces,
        geometric_duplicate_faces=sum(n-1 for n in geometric_duplicates.values()),
        degenerate_faces=int((area<=1e-12*L*L).sum()),area_epsilon=1e-12*L*L,
        orientation_consistent=orientation,orientation_status='CONSISTENT' if orientation else 'INCONSISTENT_OR_NONMANIFOLD',
        same_direction_two_incident_edges=same_direction,edge_connected_components=len(components),component_faces=components,
        vertex_link=dict(status='PASS_MANIFOLD_WITH_POSSIBLE_BOUNDARY' if vertex_manifold else 'FAIL_NONMANIFOLD_OR_UNUSED_VERTEX',
            manifold_vertices=interior_count+boundary_count,manifold_interior_vertices=interior_count,
            manifold_boundary_vertices=boundary_count,nonmanifold_vertices=len(link_failures),per_vertex_labels=link_labels,per_vertex_failures=link_failures),
        t_junction=dict(status='EXHAUSTIVE_VERTEX_EDGE_CHECK',distance_epsilon=1e-9*L,
            tested_nonendpoint_vertex_edge_pairs=pair_count,all_vertices=len(v),all_unique_edges=len(edges),
            near_zero_or_too_short_edges_skipped=skipped_short,hits=tj_hits,count=len(tj_hits),
            scope='Every indexed vertex versus every unique edge, strict interior and distance<=1e-9L; no tolerance weld or edge split'),
        C_PRESERVED=preserved,COMBINATORIAL_CLOSURE_STATUS=closure,
        EMBEDDED_GEOMETRY_VALIDITY_STATUS='NOT_RUN_REQUIRES_INDEPENDENT_CF_FF',
        geometry_intersection_checks=0,closure_does_not_certify_embedding=True,seconds=time.perf_counter()-start)


def export_protected_obj(path,indexed,C,known_mask=None):
    """New OBJ only, with FP32 roundtrip verification of all C corner records."""
    start=time.perf_counter();path=Path(path);vertices=indexed['vertices'];faces=indexed['faces'];C=_triangles(C)
    known=indexed.get('known_mask') if known_mask is None else known_mask
    if known is None:known=np.arange(len(faces))<len(C)
    known=np.asarray(known,dtype=bool)
    if vertices.dtype!=np.float32 or faces.dtype!=np.int64 or not _bit_equal(vertices[faces][known],C):raise ValueError('OBJ source violates protected C')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='ascii',newline='\n') as stream:
        stream.write('# Fixed protected coordinate export; no manifold or collision-safety claim\n')
        for point in vertices:stream.write('v '+' '.join(format(float(q),'.9g') for q in point)+'\n')
        for face in faces:stream.write('f '+' '.join(str(int(i)+1) for i in face)+'\n')
    parsed_vertices=[];parsed_faces=[]
    for line in path.read_text(encoding='ascii').splitlines():
        fields=line.split()
        if not fields:continue
        if fields[0]=='v':parsed_vertices.append([float(q) for q in fields[1:4]])
        elif fields[0]=='f':parsed_faces.append([int(q)-1 for q in fields[1:4]])
    restored=np.asarray(parsed_vertices,dtype=np.float32);restored_faces=np.asarray(parsed_faces,dtype=np.int64)
    if not _bit_equal(restored,vertices) or not np.array_equal(restored_faces,faces):raise AssertionError('OBJ FP32 roundtrip failed')
    if not _bit_equal(restored[restored_faces][known],C):raise AssertionError('Exported OBJ altered a protected C coordinate or correspondence')
    return dict(path=str(path),bytes=path.stat().st_size,sha256=_file_hash(path),export_seconds=time.perf_counter()-start,
        vertex_count=len(restored),face_count=len(restored_faces),FP32_vertex_roundtrip_exact=True,
        C_PRESERVED='PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE',
        COMBINATORIAL_CLOSURE_STATUS='SEE_INDEPENDENT_INDEXED_AUDIT',
        EMBEDDED_GEOMETRY_VALIDITY_STATUS='NOT_RUN_REQUIRES_INDEPENDENT_CF_FF',
        formatting='Nine significant decimal digits per FP32 component; signed zero retained',face_order_and_corner_order_preserved=True)


def run_cpu_fixtures(output=None):
    cases=[];arrays={};operator_audits=[];weld_calls=topology_calls=guard_calls=0
    def weld(raw,C,L=1.):
        nonlocal weld_calls
        weld_calls+=1;result=protected_weld(np.asarray(raw,np.float32),np.asarray(C,np.float32),L)
        operator_audits.append(result['audit']);return result
    def topology(indexed,L=1.):
        nonlocal topology_calls
        topology_calls+=1;return audit_indexed_topology(indexed['vertices'],indexed['faces'],L)
    vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32)
    tetra_faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int64)
    tetra=vertices[tetra_faces];C=tetra[:1]
    full=weld(tetra,C);full_topology=topology(full)
    assert full_topology['COMBINATORIAL_CLOSURE_STATUS']=='CLOSED_ORIENTED_COMBINATORIAL_2_MANIFOLD'
    removed=weld(tetra[:3],C);removed_topology=topology(removed)
    assert removed_topology['COMBINATORIAL_CLOSURE_STATUS']=='OPEN_BOUNDARY' and removed_topology['edge_incidence_counts']['1']==3
    cases.append(dict(name='tetrahedron_removed_face_stays_open_without_hole_filling',status='PASS',full=full_topology,removed=removed_topology))
    arrays['tetrahedron']=tetra;arrays['tetrahedron_removed_face']=tetra[:3]
    C=np.array([[[0,0,0],[1,0,0],[0,1,0]]],np.float32)
    free=np.array([[[0,0,0],[.5,0,0],[.25,-.5,0]],[[.5,0,0],[1,0,0],[.75,-.5,0]]],np.float32)
    tj=weld(np.concatenate([C,free]),C);tj_topology=topology(tj)
    assert tj_topology['t_junction']['count']>=1 and tj_topology['COMBINATORIAL_CLOSURE_STATUS']!='CLOSED_ORIENTED_COMBINATORIAL_2_MANIFOLD'
    cases.append(dict(name='AB_AM_MB_T_junction_not_removed_by_vertex_weld',status='PASS',topology=tj_topology));arrays['T_junction']=tj['triangles']
    far=np.array([[[-10,-10,-10],[-9,-10,-10],[-10,-9,-10]]],np.float32)
    chain_free=np.array([[[x,0,0],[2,1,0],[2,0,1]] for x in (0.,.00075,.0015)],np.float32)
    chain_raw=np.concatenate([far,chain_free]);chain=weld(chain_raw,far)
    assert np.array_equal(chain['triangles'][1,0],[0,0,0]) and np.array_equal(chain['triangles'][2,0],[0,0,0])
    assert np.array_equal(chain['triangles'][3,0],chain_raw[3,0]) and chain['audit']['max_coordinate_displacement']<=.001
    cases.append(dict(name='fixed_representative_rejects_transitive_radius_chain',status='PASS',audit=chain['audit']));arrays['chain_before']=chain_raw;arrays['chain_after']=chain['triangles']
    close_C=np.array([[[0,0,0],[.0005,0,0],[0,1,0]]],np.float32)
    tied=np.array([[[.00025,0,.0001],[1,0,1],[0,1,1]]],np.float32)
    protected=weld(np.concatenate([close_C,tied]),close_C)
    assert _bit_equal(protected['triangles'][:1],close_C) and np.array_equal(protected['triangles'][1,0],close_C[0,0])
    assert protected['known_vertex_indices'][0,0]!=protected['known_vertex_indices'][0,1]
    rotated_C=np.roll(close_C,1,axis=1)
    rotated_protected=weld(np.concatenate([rotated_C,tied]),rotated_C)
    assert _bit_equal(rotated_protected['triangles'][1:],protected['triangles'][1:])
    vs=close_C.reshape(-1,3);fs=np.array([[0,1,2]],np.int64);guard_calls+=1
    ok,_,guard=_merge_candidate(vs,fs,0,1,{0,1,2},np.array([True]),_normals(close_C),1.)
    assert not ok and 'PROTECTED_COORDINATE_CONFLICT' in guard['reasons']
    cases.append(dict(name='distinct_protected_anchors_preserved_and_distance_tie_uses_coordinate_order',status='PASS',audit=protected['audit'],direct_guard=guard))
    anchor=np.array([[[0,0,0],[0,2,0],[0,0,2]]],np.float32)
    deg=np.array([[[0,.0005,0],[.0005,0,0],[1,0,0]]],np.float32)
    deg_result=weld(np.concatenate([anchor,deg]),anchor)
    assert deg_result['audit']['rejection_reasons'].get('ZERO_OR_NEAR_DEGENERATE_PROPOSAL',0)>0
    flip=np.array([[[0,.0004,0],[-1,.0002,0],[1,.0002,0]]],np.float32)
    flip_result=weld(np.concatenate([anchor,flip]),anchor)
    assert flip_result['audit']['rejection_reasons'].get('ORIGINAL_NORMAL_FLIP_OR_ORTHOGONAL',0)>0
    repeated=np.array([[[.0002,0,0],[0,.0002,0],[1,1,0]]],np.float32)
    repeated_result=weld(np.concatenate([anchor,repeated]),anchor)
    assert repeated_result['audit']['rejection_reasons'].get('REPEATED_VERTEX_INDEX_IN_FACE',0)>0
    cases.append(dict(name='actual_W_candidates_reject_degeneracy_flip_and_repeated_face_index',status='PASS',
        degenerate=deg_result['audit'],flip=flip_result['audit'],repeated=repeated_result['audit']))
    permutation=[0,3,1,2]
    for corner_permutation in itertools.permutations(range(3)):
        permuted=chain_raw[permutation].copy()
        permuted[1:]=permuted[1:][:,corner_permutation]
        transformed=weld(permuted,far);restored=transformed['triangles'].copy()
        restored[1:]=restored[1:][:,np.argsort(corner_permutation)]
        restored=restored[np.argsort(permutation)]
        assert _bit_equal(restored,chain['triangles'])
        assert transformed['audit']['accepted_merges']==chain['audit']['accepted_merges'] and transformed['audit']['rejection_reasons']==chain['audit']['rejection_reasons']
    cases.append(dict(name='face_permutation_and_all_six_corner_orders_preserve_geometric_W_decisions',status='PASS',corner_permutations_verified=6))
    second_vertices=-vertices;second_vertices[0]=vertices[0]
    second=second_vertices[tetra_faces];bowtie=np.concatenate([tetra,second]);bow=raw_exact_index(bowtie,bowtie[:1]);bow_topology=topology(bow)
    assert bow_topology['all_edges_incident_two'] and bow_topology['vertex_link']['nonmanifold_vertices']==1
    assert bow_topology['COMBINATORIAL_CLOSURE_STATUS']=='NONMANIFOLD_DESPITE_CLOSED_EDGE_INCIDENCE'
    cases.append(dict(name='closed_edge_incidence_does_not_replace_vertex_link_test',status='PASS',topology=bow_topology));arrays['closed_vertex_bowtie']=bowtie
    if output is None:
        with tempfile.TemporaryDirectory(prefix='mf_protected_weld_fixture_') as directory:
            exported=export_protected_obj(Path(directory)/'tetrahedron.obj',full,tetra[:1])
        exported['path']='TEMPORARY_FIXTURE_REMOVED'
    else:exported=export_protected_obj(Path(output)/'fixture_protected_tetrahedron.obj',full,tetra[:1])
    assert exported['FP32_vertex_roundtrip_exact']
    cases.append(dict(name='actual_OBJ_FP32_roundtrip_preserves_C_corner_correspondence',status='PASS',export=exported))
    return dict(status='PASS',policy=dict(WELD_POLICY),tests=cases,fixture_count=len(cases),
        ledger=dict(weld_operator_calls=weld_calls,indexed_topology_audits=topology_calls,direct_candidate_guard_calls=guard_calls,
            OBJ_exports=1,raw_exact_index_calls=1,
            merge_candidates=sum(a['attempted_merges'] for a in operator_audits),
            merges_accepted=sum(a['accepted_merges'] for a in operator_audits),
            merges_rejected=sum(a['rejected_merges'] for a in operator_audits),
            W_operator_seconds=sum(a['operator_seconds'] for a in operator_audits),
            indexing_seconds=sum(a['indexing_seconds'] for a in operator_audits),
            model_forwards=0,optimizer_updates=0,geometry_autograd_calls=0,collision_candidate_checks=0)),arrays
