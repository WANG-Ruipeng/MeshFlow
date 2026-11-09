"""Fixed-C indexed triangle orientation audit and deterministic free-only repair.

Pure NumPy/CPU. No vertex movement, merging, deletion, duplication, model, or
quality scorer. Satisfiability concerns the stated two-incidence edge system;
it never certifies collision freedom, semantic attachment, or outward normals.
"""
from __future__ import annotations
import collections
import numpy as np

VERSION='fixed_C_two_incidence_XOR_orientation_v1'

def definition():
    return dict(version=VERSION,
        variables='One binary face reversal; every C face is fixed to zero',
        constraints='All indexed C-C/C-F/F-F edges with exactly two distinct nondegenerate incident faces: same original edge direction requires XOR1, opposite requires XOR0',
        degenerate='Repeated vertex IDs or exactly zero FP64 triangle area; no near-zero threshold introduced',
        components='Connected components of the eligible two-incidence constraint graph, including isolated faces',
        repair='If parity-consistent and all fixed C anchors compatible, apply the unique anchored solution or an unanchored solution with minimum face ID unflipped; conflicted components remain entirely unchanged',
        reversal='Per-face vertex order [0,2,1]; C face rows preserved byte-for-byte',
        exclusions='Nonmanifold incidence, boundary edges, self-incidence, or a degenerate incident face produce no XOR constraint and are explicitly listed',
        unanchored='Gauge only, no outward-orientation certification',
        limitations='No vertex-link manifold, collision, overlap, watertightness, or semantic-attachment certificate. Duplicate faces are not repaired or removed.')

def orient_fixed_C(vertices,faces,K):
    """Return faces, flips, component_rows, edge_rows, face provenance, summary.

    vertices:(V,3) finite floating array; faces:(N,3) integer indexed faces;
    first K face rows are immutable C. Caller inputs are never modified.
    """
    vertices=np.asarray(vertices);faces=np.asarray(faces)
    if vertices.ndim!=2 or vertices.shape[1]!=3 or not np.issubdtype(vertices.dtype,np.floating):raise ValueError('Finite floating vertices(V,3) required')
    if not np.isfinite(vertices).all():raise ValueError('Nonfinite input vertices')
    if faces.ndim!=2 or faces.shape[1]!=3 or not np.issubdtype(faces.dtype,np.integer):raise ValueError('Integer indexed faces(N,3) required')
    if not isinstance(K,(int,np.integer)) or not 0<=K<=len(faces):raise ValueError('K outside face array')
    if faces.size and (faces.min()<0 or faces.max()>=len(vertices)):raise ValueError('Face index outside vertex array')
    K=int(K);N=len(faces);tri=vertices[faces].astype(np.float64)
    areas=.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    repeated=np.asarray([len(set(map(int,f)))<3 for f in faces],dtype=bool)
    degenerate=repeated|(areas==0)
    incidence=collections.defaultdict(list)
    duplicate_groups=collections.defaultdict(list)
    for i,face in enumerate(faces):
        duplicate_groups[tuple(sorted(map(int,face)))].append(i)
        for corner,(u,v) in enumerate(((face[0],face[1]),(face[1],face[2]),(face[2],face[0]))):
            u,v=int(u),int(v);incidence[tuple(sorted((u,v)))].append((i,corner,u,v))
    duplicate_faces={i for group in duplicate_groups.values() if len(group)>1 for i in group}
    edges=[];adjacency=[[] for _ in range(N)]
    for edge_id,((u,v),entries) in enumerate(sorted(incidence.items())):
        ff=[x[0] for x in entries];distinct=set(ff);reasons=[]
        has_C=any(i<K for i in ff);has_F=any(i>=K for i in ff)
        if u==v:reasons.append('SELF_LOOP_EDGE')
        if len(entries)==1:reasons.append('BOUNDARY_SINGLE_INCIDENCE')
        elif len(entries)>2:reasons.append('NONMANIFOLD_MORE_THAN_TWO_INCIDENCES')
        elif len(entries)!=2:reasons.append('NOT_TWO_INCIDENCES')
        if len(entries)==2 and len(distinct)!=2:reasons.append('TWO_INCIDENCES_SAME_FACE')
        if any(degenerate[i] for i in distinct):reasons.append('DEGENERATE_INCIDENT_FACE')
        eligible=not reasons
        opposed=entries[0][2:]==entries[1][2:][::-1] if len(entries)==2 and len(distinct)==2 and u!=v else None
        required=int(not opposed) if eligible else None
        row=dict(edge_id=edge_id,vertex_u=u,vertex_v=v,incidence=len(entries),distinct_incident_faces=len(distinct),
            incident_face_ids=ff,incident_local_edges=[x[1] for x in entries],
            incident_directions=[[x[2],x[3]] for x in entries],
            domain='C_F' if has_C and has_F else 'C_C' if has_C else 'F_F',
            original_opposite_direction=opposed,eligible_constraint=eligible,required_flip_XOR=required,
            exclusion_reasons=reasons,length=float(np.linalg.norm(vertices[u].astype(float)-vertices[v])))
        edges.append(row)
        if eligible:
            i,j=ff;adjacency[i].append((j,edge_id,required));adjacency[j].append((i,edge_id,required))
    relative=np.full(N,-1,np.int8);component_ids=np.full(N,-1,np.int64);components=[]
    for root in range(N):
        if relative[root]>=0:continue
        cid=len(components);relative[root]=0;component_ids[root]=cid;queue=collections.deque([root]);members=[]
        while queue:
            i=queue.popleft();members.append(i)
            for j,edge_id,wanted in sorted(adjacency[i]):
                if relative[j]<0:
                    relative[j]=int(relative[i])^wanted;component_ids[j]=cid;queue.append(j)
        members.sort();components.append(dict(component_id=cid,root_face=min(members),face_ids=members))
    constraints_by_component=collections.defaultdict(list)
    excluded_by_component=collections.defaultdict(set)
    for e in edges:
        ids=sorted({int(component_ids[i]) for i in e['incident_face_ids']})
        e['constraint_component_ids']=ids
        if e['eligible_constraint']:
            assert len(ids)==1;constraints_by_component[ids[0]].append(e)
        else:
            for cid in ids:excluded_by_component[cid].add(e['edge_id'])
    flips=np.zeros(N,dtype=bool)
    for comp in components:
        cid=comp['component_id'];members=comp['face_ids'];cc=[i for i in members if i<K];ee=constraints_by_component[cid]
        conflict=[e['edge_id'] for e in ee if int(relative[e['incident_face_ids'][0]])^int(relative[e['incident_face_ids'][1]])!=e['required_flip_XOR']]
        parity_conflict=bool(conflict)
        # Anchor compatibility is meaningful only once the graph parity is
        # internally consistent. Do not mislabel an arbitrary inconsistent
        # spanning assignment as a separate anchor-only contradiction.
        anchor_gauges=sorted({int(relative[i]) for i in cc})
        anchored_conflict=(len(anchor_gauges)>1) if not parity_conflict else None
        excluded_faces=[i for i in members if degenerate[i]]
        satisfiable=not parity_conflict and not anchored_conflict and not excluded_faces
        gauge=anchor_gauges[0] if cc and satisfiable else 0 if satisfiable else None
        if satisfiable:
            flips[members]=(relative[members]^gauge).astype(bool)
            assert not flips[cc].any()
            repair_status='SATISFIABLE_ANCHORED' if cc else 'SATISFIABLE_UNANCHORED_GAUGE_ONLY'
        else:
            repair_status='UNCHANGED_PARITY_CONFLICT' if parity_conflict else 'UNCHANGED_FIXED_C_CONFLICT' if anchored_conflict else 'UNCHANGED_DEGENERATE_FACE'
        exclusions=[edges[i] for i in sorted(excluded_by_component[cid])]
        comp.update(face_count=len(members),C_face_ids=cc,C_face_count=len(cc),free_face_count=sum(i>=K for i in members),
            eligible_constraint_edges=len(ee),parity_conflict=parity_conflict,
            parity_conflict_edge_ids_under_deterministic_propagation=conflict,
            fixed_C_anchor_conflict=anchored_conflict,
            fixed_C_anchor_status='NA_PARITY_CONFLICT' if parity_conflict else 'NO_C_ANCHOR' if not cc else 'CONTRADICTION' if anchored_conflict else 'COMPATIBLE',
            C_anchor_required_gauges=anchor_gauges,
            degenerate_face_ids=excluded_faces,excluded_incident_edge_ids=sorted(excluded_by_component[cid]),
            duplicate_indexed_face_ids=[i for i in members if i in duplicate_faces],
            touches_duplicate_indexed_face=any(i in duplicate_faces for i in members),
            touches_nonmanifold_edge=any('NONMANIFOLD_MORE_THAN_TWO_INCIDENCES' in e['exclusion_reasons'] for e in exclusions),
            touches_degenerate_incident_edge=any('DEGENERATE_INCIDENT_FACE' in e['exclusion_reasons'] for e in exclusions),
            satisfiable=satisfiable,repair_status=repair_status,gauge=gauge,
            flipped_free_face_ids=[i for i in members if flips[i]],flipped_free_face_count=int(flips[members].sum()),
            outward_orientation_certified=False)
    repaired=faces.copy();repaired[flips]=faces[flips][:,[0,2,1]]
    assert repaired[:K].tobytes()==faces[:K].tobytes() and not flips[:K].any()
    assert np.array_equal(np.sort(repaired,axis=1),np.sort(faces,axis=1))
    permutations=np.tile(np.array([0,1,2],np.int8),(N,1));permutations[flips]=[0,2,1]
    for e in edges:
        e['incident_faces_flipped']=[bool(flips[i]) for i in e['incident_face_ids']]
        if e['original_opposite_direction'] is not None:
            i,j=e['incident_face_ids'];e['repaired_opposite_direction']=bool(e['original_opposite_direction'])^(bool(flips[i])^bool(flips[j]))
        else:e['repaired_opposite_direction']=None
        if e['eligible_constraint']:
            i,j=e['incident_face_ids'];cid=int(component_ids[i])
            e['relative_parity_conflict']=(int(relative[i])^int(relative[j]))!=e['required_flip_XOR']
            e['repaired_constraint_satisfied']=(int(flips[i])^int(flips[j]))==e['required_flip_XOR']
            if components[cid]['satisfiable']:assert e['repaired_constraint_satisfied'] and e['repaired_opposite_direction']
        else:e['relative_parity_conflict']=None;e['repaired_constraint_satisfied']=None
    eligible=[e for e in edges if e['eligible_constraint']];cf=[e for e in eligible if e['domain']=='C_F']
    parity_count=sum(c['parity_conflict'] for c in components);anchor_count=sum(c['fixed_C_anchor_conflict'] is True for c in components)
    nonmanifold=sum(e['incidence']>2 for e in edges);deg_count=int(degenerate.sum())
    summary=dict(status='PASS',scope='Executed orientation audit/repair, not a manifold/geometry-quality certificate',definition=definition(),
        vertices=len(vertices),faces=N,C_faces=K,free_faces=N-K,constraint_components=len(components),
        anchored_components=sum(c['C_face_count']>0 for c in components),
        unanchored_nondegenerate_components=sum(c['C_face_count']==0 and not c['degenerate_face_ids'] for c in components),
        satisfiable_components=sum(c['satisfiable'] for c in components),parity_conflict_components=parity_count,
        fixed_C_anchor_conflict_components=anchor_count,
        fixed_C_anchor_status_not_evaluated_due_to_parity_components=parity_count,
        degenerate_faces=deg_count,nonmanifold_edges=nonmanifold,boundary_edges=sum(e['incidence']==1 for e in edges),
        duplicate_indexed_face_groups=sum(len(v)>1 for v in duplicate_groups.values()),
        duplicate_indexed_faces=len(duplicate_faces),
        eligible_two_incidence_edges=len(eligible),eligible_C_F_edges=len(cf),
        opposite_two_incidence_edges_before=sum(bool(e['original_opposite_direction']) for e in eligible),
        opposite_two_incidence_edges_after=sum(bool(e['repaired_opposite_direction']) for e in eligible),
        opposite_C_F_edges_before=sum(bool(e['original_opposite_direction']) for e in cf),
        opposite_C_F_edges_after=sum(bool(e['repaired_opposite_direction']) for e in cf),
        unsatisfied_eligible_edges_after=sum(not e['repaired_constraint_satisfied'] for e in eligible),
        flipped_free_faces=int(flips.sum()),flipped_free_surface_area=float(areas[flips].sum()),
        fixed_C_constraints_satisfiable_on_eligible_graph=not parity_count and not anchor_count,
        whole_indexed_edge_orientation_status='NA_NONMANIFOLD_OR_DEGENERATE' if nonmanifold or deg_count else 'INCOMPATIBLE_FIXED_C_OR_PARITY' if parity_count or anchor_count else 'TWO_INCIDENCE_CONSTRAINTS_SATISFIABLE',
        C_face_rows_bytewise_preserved=True,C_triangle_coordinate_order_bytewise_preserved=vertices[repaired[:K]].tobytes()==vertices[faces[:K]].tobytes(),
        coordinates_unchanged=True,face_vertex_sets_unchanged=True,face_count_unchanged=True,
        outward_orientation_certified=False,model_calls=0)
    return dict(faces=repaired,flips=flips,face_component_ids=component_ids,
        face_corner_permutation=permutations,component_rows=components,edge_rows=edges,summary=summary)
