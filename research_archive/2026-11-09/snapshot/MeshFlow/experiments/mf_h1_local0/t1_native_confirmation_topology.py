"""Read-only source-index and exact-coordinate topology attribution.

No model, geometry energy, welding, reorientation, or output mutation occurs.
Orientation parity assignments are diagnostic witnesses only.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import time
import numpy as np

PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT / 'outputs/native_patch_20260929_010000'
EXACT_POLICY = 'Exact FP32 bit patterns; distinct signed-zero patterns stay distinct, matching the existing raw index diagnostic; no tolerance or coordinate change.'


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def _array_hash(value):
    # Same archived array hash definition used by learning_data.array_hash.
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str((value.dtype, value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _key(point):
    return tuple(np.asarray(point, dtype=np.float32).view(np.uint32).tolist())


def exact_index(triangles):
    triangles = np.asarray(triangles)
    if triangles.dtype != np.float32 or triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError('Exact index requires untouched FP32 [F,3,3].')
    lookup = {}; vertices = []; index = []
    for point in triangles.reshape(-1, 3):
        key = _key(point)
        if key not in lookup:
            lookup[key] = len(vertices)
            vertices.append(point.copy())
        index.append(lookup[key])
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(index, dtype=np.int64).reshape(-1, 3)
    if vertices[faces].tobytes() != triangles.tobytes():
        raise RuntimeError('Exact diagnostic indexing changed coordinates.')
    return vertices, faces


def _components(nodes, adjacency):
    unseen = set(nodes); result = []
    while unseen:
        seen = {min(unseen)}; queue = list(seen)
        while queue:
            current = queue.pop()
            for other in adjacency.get(current, ()):
                if other not in seen:
                    seen.add(other); queue.append(other)
        unseen -= seen; result.append(sorted(seen))
    return result


def _edge_data(faces):
    edges = defaultdict(list)
    for face_index, triangle in enumerate(faces):
        for j in range(3):
            a, b = int(triangle[j]), int(triangle[(j + 1) % 3])
            edges[tuple(sorted((a, b)))].append((face_index, 1 if a < b else -1))
    return edges


def audit_mesh(vertices, faces):
    """Combinatorial diagnostics from supplied identities; no geometric repair."""
    vertices = np.asarray(vertices); faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or (faces.size and (faces.min() < 0 or faces.max() >= len(vertices))):
        raise ValueError('Invalid indexed triangles.')
    used = sorted(set(faces.reshape(-1).tolist()))
    edges = _edge_data(faces); adjacency = defaultdict(set)
    orientation_graph = defaultdict(list); conflicts = []
    for edge, members in edges.items():
        incident = [member[0] for member in members]
        for face in incident:
            adjacency[face].update(other for other in incident if other != face)
        if len(members) == 2:
            (a, sa), (b, sb) = members
            parity = int(sa == sb)
            orientation_graph[a].append((b, parity, edge))
            orientation_graph[b].append((a, parity, edge))
            if parity:
                conflicts.append(dict(edge=list(edge), face_ids=[a, b]))
    assignments = {}; parity_conflicts = set()
    for face in range(len(faces)):
        if face in assignments:
            continue
        assignments[face] = 0; queue = [face]
        while queue:
            face = queue.pop()
            for other, parity, edge in orientation_graph[face]:
                required = assignments[face] ^ parity
                if other in assignments and assignments[other] != required:
                    parity_conflicts.add((min(face, other), max(face, other), edge))
                elif other not in assignments:
                    assignments[other] = required; queue.append(other)
    links = []
    for vertex in used:
        incident = np.flatnonzero((faces == vertex).any(axis=1)).tolist()
        link_edges = []; repeated_face = False
        for face in incident:
            triangle = faces[face].tolist()
            if triangle.count(vertex) != 1:
                repeated_face = True
                continue
            other = [v for v in triangle if v != vertex]
            link_edges.append(tuple(sorted(other)))
        link_adjacency = defaultdict(set); degree = Counter()
        for a, b in link_edges:
            link_adjacency[a].add(b); link_adjacency[b].add(a)
            degree[a] += 1; degree[b] += 1
        components = _components(degree, link_adjacency)
        boundary_degree = sum(vertex in edge and len(members) == 1 for edge, members in edges.items())
        repeated_link_edges = [list(edge) for edge, count in Counter(link_edges).items() if count > 1]
        valid = len(components) == 1 and not repeated_face and not repeated_link_edges
        cycle = valid and len(degree) >= 3 and all(d == 2 for d in degree.values()) and boundary_degree == 0
        path = valid and sum(d == 1 for d in degree.values()) == 2 and all(d in (1, 2) for d in degree.values()) and boundary_degree == 2
        classification = 'CYCLE_INTERIOR_MANIFOLD' if cycle else ('PATH_BOUNDARY_MANIFOLD' if path else 'NONMANIFOLD_LINK')
        links.append(dict(vertex_id=vertex, coordinate=np.asarray(vertices[vertex], dtype=float).tolist(),
            incident_face_ids=incident, component_count=len(components), components=components,
            degrees=[dict(neighbor_vertex_id=v, degree=d) for v, d in sorted(degree.items())],
            boundary_edge_degree=boundary_degree, link_edge_count=len(link_edges),
            repeated_link_edges=repeated_link_edges, repeated_vertex_in_incident_face=repeated_face,
            classification=classification))
    bad = [row['vertex_id'] for row in links if row['classification'] == 'NONMANIFOLD_LINK']
    edge_counts = Counter(len(value) for value in edges.values())
    over = [dict(edge=list(edge), face_ids=[r[0] for r in members]) for edge, members in edges.items() if len(members) > 2]
    repeated = np.flatnonzero(np.array([len(set(face)) < 3 for face in faces])).tolist()
    duplicate_groups = defaultdict(list)
    for i, face in enumerate(faces):
        duplicate_groups[tuple(sorted(face.tolist()))].append(i)
    duplicate_groups = [value for value in duplicate_groups.values() if len(value) > 1]
    components = _components(range(len(faces)), adjacency)
    closed = edge_counts.get(1, 0) == 0 and not over and not bad and not repeated and not duplicate_groups
    reorientable = not parity_conflicts
    if over or repeated or duplicate_groups:
        orientability = 'NOT_A_TRIANGLE_SURFACE_MANIFOLD'
    elif bad:
        orientability = 'EDGE_ORIENTATION_CONSTRAINTS_SATISFIABLE_BUT_VERTEX_LINK_NONMANIFOLD' if reorientable else 'ORIENTATION_CONSTRAINTS_INCONSISTENT'
    else:
        orientability = 'CONSISTENT_REORIENTATION_EXISTS' if reorientable else 'NONORIENTABLE_MANIFOLD'
    return dict(vertex_count=len(vertices), used_vertex_count=len(used), face_count=len(faces), edge_count=len(edges),
        shared_edge_components=dict(count=len(components), sizes=sorted(map(len, components), reverse=True), face_ids=components),
        edge_incidence=dict(one=edge_counts.get(1, 0), two=edge_counts.get(2, 0), more_than_two=len(over)),
        boundary_edges=[list(edge) for edge, members in edges.items() if len(members) == 1],
        overincident_edges=over, repeated_vertex_face_ids=repeated, duplicate_face_groups=duplicate_groups,
        vertex_links=dict(nonmanifold_vertex_count=len(bad), nonmanifold_vertex_ids=bad, all=links),
        orientation=dict(current_winding_consistent=not conflicts and not over and not repeated,
            current_conflicting_edge_count=len(conflicts), current_conflicting_edges=conflicts,
            can_consistently_reorient_two_incident_edges=reorientable,
            parity_conflicts=[dict(face_ids=[a, b], edge=list(edge)) for a, b, edge in sorted(parity_conflicts)],
            diagnostic_flip_face_ids=[f for f, flip in sorted(assignments.items()) if flip],
            orientability_status=orientability, reorientation_applied=False,
            interpretation='Winding constraints only; not evidence of triangle penetration. Nonmanifold links are reported separately.'),
        COMBINATORIAL_CLOSURE_STATUS='CLOSED_TRIANGLE_2_MANIFOLD' if closed else 'NOT_CLOSED_TRIANGLE_2_MANIFOLD',
        embedded_geometry_validity='NOT_RUN_IN_TOPOLOGY_AUDIT',
        scope='Exact supplied vertex identities. No coordinate-distance inference, T-junction test, or collision predicate in this function.')


def _coincidences(vertices):
    groups = defaultdict(list)
    for i, point in enumerate(vertices):
        groups[_key(point)].append(i)
    return [dict(vertex_ids=ids, coordinate=np.asarray(vertices[ids[0]], dtype=float).tolist()) for ids in groups.values() if len(ids) > 1]


def run_input_audit(bank, out=None):
    """Read archived identity maps and actual raw NPZ; return report, never write."""
    import json
    started = time.perf_counter()
    prior = bank.get('prior_results')
    if prior is None:
        prior = json.loads((SOURCE / 'results.json').read_text(encoding='utf-8-sig'))
    manifest = prior['assets']; input_record = manifest['inputs']; source_input = Path(input_record['path'])
    if _hash(source_input) != input_record['sha256']:
        raise RuntimeError('Input archive identity changed.')
    rows = []; ledger = dict(input_parent_audits=0, indexed_mesh_audits=0, model_forwards=0, geometry_gradients=0, repairs=0)
    with np.load(source_input, allow_pickle=False) as archive:
        for parent in (0, 1):
            c = bank['cases_by_key'][parent, 12]
            source_meta = next(row for row in manifest['raw_source_topology'] if row['parent_index'] == parent)
            source_record = source_meta['pre_OT']['source']; raw_path = Path(source_record['path'])
            if not raw_path.exists():
                rows.append(dict(parent=parent, status='UNKNOWN_SOURCE_INDEX_ASSET_MISSING', source_record=source_record))
                continue
            if _hash(raw_path) != source_record['sha256']:
                raise RuntimeError('Raw parent identity changed.')
            required = ['source_vertices', 'source_face_vertex_ids', 'canonical_to_raw_face', 'target', 'K12_constraints', 'K12_source_face_ids', 'K12_free_source_ids', 'K12_free_target']
            arrays = {name:archive[f'parent{parent}_{name}'].copy() for name in required}
            for name, value in arrays.items():
                key = f'parent{parent}_{name}'
                if _array_hash(value) != input_record['arrays'][key]['sha256']:
                    raise RuntimeError('Source identity array hash differs: ' + key)
            vertices = arrays['source_vertices']; mapped_faces = arrays['source_face_vertex_ids']; canonical_to_raw = arrays['canonical_to_raw_face']
            ids = arrays['K12_source_face_ids']; C = arrays['K12_constraints']; target = arrays['target'].reshape(112, 3, 3)
            with np.load(raw_path, allow_pickle=False) as raw:
                raw_vertices = raw['vertices'].copy(); raw_faces = raw['faces'].astype(np.int64, copy=True)
                raw_uid = str(raw['uid'].item())
            if raw_uid != c['parent_uid'] or len(raw_faces) != 112 or len(vertices) != len(raw_vertices):
                raise RuntimeError('Raw source identity, index length, or task face count differs.')
            permutations = np.asarray(source_meta['raw_corner_permutations'], dtype=np.int64)
            mapped_from_raw = np.take_along_axis(raw_faces[canonical_to_raw], permutations, axis=1)
            if not np.array_equal(mapped_faces, mapped_from_raw) or len(set(canonical_to_raw.tolist())) != 112:
                raise RuntimeError('Stored raw face identity mapping is not exact and bijective.')
            if vertices[mapped_faces].tobytes() != target.tobytes() or target[ids].tobytes() != C.tobytes():
                raise RuntimeError('Raw-identity mapped FP32 coordinates differ from frozen input.')
            if C.tobytes() != np.asarray(c['constraints'], dtype=np.float32).tobytes() or target.tobytes() != np.asarray(c['full_target'], dtype=np.float32).tobytes():
                raise RuntimeError('Sampler bank and audited input coordinates differ.')
            free_ids = arrays['K12_free_source_ids']
            if free_ids.tolist() != [i for i in range(112) if i not in ids] or arrays['K12_free_target'].tobytes() != target[free_ids].reshape(100,9).tobytes():
                raise RuntimeError('Free target decomposition differs.')
            exact_vertices, exact_faces = exact_index(C)
            full_source = audit_mesh(raw_vertices, raw_faces)
            full_archive = audit_mesh(vertices, mapped_faces)
            C_source = audit_mesh(vertices, raw_faces[canonical_to_raw[ids]])
            C_archive = audit_mesh(vertices, mapped_faces[ids])
            C_exact = audit_mesh(exact_vertices, exact_faces)
            ledger['indexed_mesh_audits'] += 5; ledger['input_parent_audits'] += 1
            full_links = {r['vertex_id']:r for r in full_archive['vertex_links']['all']}
            archive_links = {r['vertex_id']:r for r in C_archive['vertex_links']['all']}
            anomalies = []
            for vertex in C_archive['vertex_links']['nonmanifold_vertex_ids']:
                anomalies.append(dict(source_vertex_id=vertex, coordinate=vertices[vertex].tolist(),
                    C_link=archive_links[vertex], full_GT_link=full_links[vertex],
                    full_GT_repairs_subpatch_link=full_links[vertex]['classification'] != 'NONMANIFOLD_LINK'))
            source_to_exact = []
            for exact_id, point in enumerate(exact_vertices):
                source_ids = [int(v) for v in np.unique(mapped_faces[ids]) if _key(vertices[v]) == _key(point)]
                source_to_exact.append(dict(exact_vertex_id=exact_id, source_vertex_ids=source_ids))
            archive_bad_keys = {_key(vertices[v]) for v in C_archive['vertex_links']['nonmanifold_vertex_ids']}
            exact_bad_keys = {_key(exact_vertices[v]) for v in C_exact['vertex_links']['nonmanifold_vertex_ids']}
            triangles64 = target.astype(np.float64)
            area = np.linalg.norm(np.cross(triangles64[:,1]-triangles64[:,0], triangles64[:,2]-triangles64[:,0]), axis=1) / 2
            area_fraction = float(area[ids].sum() / area.sum())
            original_case = next(row for row in manifest['cases'] if row['parent_index'] == parent and row['K'] == 12)
            if abs(area_fraction - original_case['area_fraction']) > 1e-12:
                raise RuntimeError('Prompt area differs from frozen case metadata.')
            rows.append(dict(parent=parent, parent_uid=c['parent_uid'], status='PASS_IDENTITY_AND_DECOMPOSITION',
                source_record=dict(**source_record, actual_sha256=_hash(raw_path)),
                source_array_hashes={name:_array_hash(value) for name,value in arrays.items()},
                source_to_archive_map=dict(canonical_to_raw_face_ids=canonical_to_raw.tolist(), raw_corner_permutations=permutations.tolist(),
                    K12_canonical_face_ids=ids.tolist(), K12_raw_face_ids=canonical_to_raw[ids].tolist(),
                    K12_source_vertex_ids_in_C_order=mapped_faces[ids].tolist(), K12_exact_to_source_vertices=source_to_exact,
                    exact_bijective_face_map=True, all112_model_coordinates_bitwise_exact=True),
                full_source_original_winding=full_source, full_archive_source_indices=full_archive,
                C_source_original_winding=C_source, C_source_archive_winding=C_archive, C_exact_archive_winding=C_exact,
                C_link_anomalies_vs_full_GT=anomalies,
                C_link_anomalies_added_by_exact_indexing_count=len(exact_bad_keys-archive_bad_keys),
                C_link_anomalies_removed_by_exact_indexing_count=len(archive_bad_keys-exact_bad_keys),
                coincident_distinct_source_vertex_ids=dict(raw_coordinates=_coincidences(raw_vertices), model_coordinates=_coincidences(vertices)),
                area_fraction=area_fraction, patch_area=float(area[ids].sum()), full_area=float(area.sum()),
                coordinate_range=dict(min=C.reshape(-1,3).min(0).tolist(), max=C.reshape(-1,3).max(0).tolist()),
                exact_index_policy=EXACT_POLICY, geometry_intersections='NOT_RUN_IN_INPUT_TOPOLOGY_AUDIT',
                C_modifications=0, full_GT_role='Read-only identity/topology/score only; excluded from model/sampling.'))
    return dict(status='PASS' if len(rows)==2 and all(row['status']=='PASS_IDENTITY_AND_DECOMPOSITION' for row in rows) else 'PARTIAL_UNKNOWN_SOURCE_INDEX',
        parents=rows, source_input_record=input_record | {'arrays': 'Hashes of used arrays are recorded per parent; other archived arrays are not duplicated.'},
        ledger=ledger, seconds=time.perf_counter()-started,
        interpretation='A subpatch link may be disconnected while the full source fills it into a valid cycle. Reorientation witnesses never change C or generated output.')


def generated_attribution(tokens, C, input_parent_audit=None):
    """Attribute exact-index output defects to immutable C or free incidence."""
    started = time.perf_counter()
    triangles = np.asarray(tokens).reshape(112, 3, 3); C = np.asarray(C)
    if triangles.dtype != np.float32 or C.dtype != np.float32 or C.shape != (12,3,3):
        raise ValueError('Untouched FP32 N112 K12 required.')
    if triangles[:12].tobytes() != C.tobytes():
        raise RuntimeError('Output changed C.')
    vertices, faces = exact_index(triangles); C_vertices, C_faces = exact_index(C)
    topo = audit_mesh(vertices, faces); C_topo = audit_mesh(C_vertices, C_faces)
    C_keys = {_key(v):i for i,v in enumerate(C_vertices)}
    output_C_vertex_ids = [i for i,v in enumerate(vertices) if _key(v) in C_keys]
    C_bad_keys = {_key(C_vertices[v]) for v in C_topo['vertex_links']['nonmanifold_vertex_ids']}
    output_links = {row['vertex_id']:row for row in topo['vertex_links']['all']}
    inherited = []; modified = []; new_bad = []; inherited_but_free_incident = []
    for vid, link in output_links.items():
        key = _key(vertices[vid]); free_incident = any(face >= 12 for face in link['incident_face_ids'])
        bad = link['classification'] == 'NONMANIFOLD_LINK'
        if key in C_keys and free_incident:
            modified.append(vid)
        if bad and key in C_bad_keys and not free_incident:
            inherited.append(vid)
        elif bad and key in C_bad_keys and free_incident:
            inherited_but_free_incident.append(vid)
        elif bad:
            new_bad.append(vid)
    edge_data = _edge_data(faces)
    shared = [dict(edge=list(edge), known_face_ids=[f for f,_ in members if f < 12], free_face_ids=[f for f,_ in members if f >= 12])
        for edge,members in edge_data.items() if any(f < 12 for f,_ in members) and any(f >= 12 for f,_ in members)]
    conflicts = topo['orientation']['current_conflicting_edges']
    known_conflicts = [row for row in conflicts if all(face < 12 for face in row['face_ids'])]
    free_conflicts = [row for row in conflicts if any(face >= 12 for face in row['face_ids'])]
    inherited_reference = None
    if input_parent_audit is not None:
        expected = input_parent_audit['C_exact_archive_winding']
        if expected['vertex_links']['nonmanifold_vertex_count'] != C_topo['vertex_links']['nonmanifold_vertex_count'] or expected['edge_incidence'] != C_topo['edge_incidence']:
            raise RuntimeError('C topology changed from input audit.')
        inherited_reference = input_parent_audit['parent']
    return dict(exact_indexing=dict(policy=EXACT_POLICY, vertex_count=len(vertices), corner_count=336,
            coordinate_changes=0, face_count=112, coordinates_bitwise_reconstruct=True),
        topology=topo, C_only_topology=C_topo,
        attribution=dict(input_parent_audit_reference=inherited_reference,
            C_only_nonmanifold_vertex_count=len(C_bad_keys),
            inherited_C_only_unmodified_link_vertex_ids=inherited,
            C_vertices_modified_by_free_incidence=modified,
            C_inherited_link_anomalies_with_new_free_incidence=inherited_but_free_incident,
            new_free_related_nonmanifold_vertex_ids=new_bad,
            orientation_conflicts_C_only=known_conflicts, orientation_conflicts_free_related=free_conflicts,
            exact_C_free_shared_edge_count=len(shared), exact_C_free_shared_edges=shared,
            exact_C_free_shared_vertex_count=sum(any(f >= 12 for f in output_links[v]['incident_face_ids']) for v in output_C_vertex_ids),
            winding_is_not_intersection_evidence=True,
            disconnected_triangle_soup_note='Isolated triangles can each have valid boundary vertex links while the complete output has no useful shared topology.'),
        topology_audit_seconds=time.perf_counter()-started,
        ledger=dict(indexed_mesh_audits=2, exact_coordinate_indexings=2, geometry_repairs=0))


def run_cpu_fixtures():
    """Fixed combinatorial fixtures only. No real outputs, thresholds, or models."""
    ledger = dict(fixture_families=0,indexed_mesh_audits=0,exact_coordinate_indexings=0,model_forwards=0,repairs=0)
    vertices = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
    tetra = np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]])
    full = audit_mesh(vertices,tetra); ledger['indexed_mesh_audits'] += 1
    assert full['COMBINATORIAL_CLOSURE_STATUS']=='CLOSED_TRIANGLE_2_MANIFOLD'
    assert full['orientation']['current_winding_consistent']
    changed = tetra.copy(); changed[1] = changed[1,::-1]
    rotated = audit_mesh(vertices,changed); ledger['indexed_mesh_audits'] += 1
    assert not rotated['orientation']['current_winding_consistent']
    assert rotated['orientation']['can_consistently_reorient_two_incident_edges']
    ledger['fixture_families'] += 1
    # Two separated sectors around one source vertex become a valid disk when the missing sector is supplied.
    v = np.array([[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]],dtype=np.float32)
    bow = np.array([[0,1,2],[0,3,4]])
    partial = audit_mesh(v,bow); filled = audit_mesh(v,np.array([[0,1,2],[0,2,3],[0,3,4]])); ledger['indexed_mesh_audits'] += 2
    assert partial['vertex_links']['nonmanifold_vertex_ids']==[0]
    assert filled['vertex_links']['nonmanifold_vertex_count']==0
    ledger['fixture_families'] += 1
    # Distinct source identities at the same coordinates can introduce a new link defect after exact reindexing.
    duplicate_vertices = np.concatenate([v,v[:1]])
    separate_faces=np.array([[0,1,2],[5,3,4]])
    separate=audit_mesh(duplicate_vertices,separate_faces)
    ev,ef=exact_index(duplicate_vertices[separate_faces]); reindexed=audit_mesh(ev,ef)
    ledger['indexed_mesh_audits']+=2; ledger['exact_coordinate_indexings']+=1
    assert separate['vertex_links']['nonmanifold_vertex_count']==0 and reindexed['vertex_links']['nonmanifold_vertex_count']==1
    ledger['fixture_families'] += 1
    over=audit_mesh(vertices,np.array([[0,1,2],[1,0,3],[0,1,3]]));ledger['indexed_mesh_audits']+=1
    assert over['edge_incidence']['more_than_two']==1
    ledger['fixture_families'] += 1
    # A standard six-vertex triangulated projective plane has inconsistent orientation parity.
    rp2=np.array([[0,1,2],[0,1,3],[0,2,4],[0,3,5],[0,4,5],[1,2,5],[1,3,4],[1,4,5],[2,3,4],[2,3,5]])
    nonorientable=audit_mesh(np.zeros((6,3),dtype=np.float32),rp2);ledger['indexed_mesh_audits']+=1
    assert nonorientable['vertex_links']['nonmanifold_vertex_count']==0
    assert not nonorientable['orientation']['can_consistently_reorient_two_incident_edges']
    ledger['fixture_families'] += 1
    return dict(status='PASS',ledger=ledger,fixture_names=['tetra_winding_vs_reorientation','subpatch_link_completed_by_full_mesh','coincident_distinct_source_ids','overincident_edge','nonorientable_projective_plane'],
        floating_geometry_predicates=False,output_coordinates_modified=False)

