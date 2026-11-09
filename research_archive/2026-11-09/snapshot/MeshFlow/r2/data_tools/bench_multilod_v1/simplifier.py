"""Deterministic, protected, CPU-only quadric edge-collapse proposals.

This module proposes a lower-face-count version; it does not certify finite
surface fidelity or absence of intersections.  A separate admission audit is
required.  Input face IDs are expanded by one edge-adjacent ring, so callers can
pass the original C face IDs directly.  All vertices of those faces are fixed.
Original vertex indices and surviving original face IDs are retained.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math

import numpy as np


VERSION = "bench_multilod_protected_qem_v1"
NORMAL_LIMIT_DEGREES = 30.0
SHARP_LIMIT_DEGREES = 45.0


def _topology(faces, alive):
    edges = defaultdict(list)
    vertex_faces = defaultdict(set)
    neighbors = defaultdict(set)
    for fi in np.flatnonzero(alive):
        tri = tuple(map(int, faces[fi]))
        for v in tri:
            vertex_faces[v].add(int(fi))
        for i in range(3):
            u, v = tri[i], tri[(i + 1) % 3]
            key = (min(u, v), max(u, v))
            edges[key].append(int(fi))
            neighbors[u].add(v)
            neighbors[v].add(u)
    return dict(edges), dict(vertex_faces), dict(neighbors)


def _face_vectors(vertices, faces):
    tris = vertices[faces].astype(np.float64)
    return np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])


def _direction(face, u, v):
    for j in range(3):
        if int(face[j]) == u and int(face[(j + 1) % 3]) == v:
            return 1
        if int(face[j]) == v and int(face[(j + 1) % 3]) == u:
            return -1
    return 0


def _bad_vertex_links(faces, vertex_faces):
    """An ordinary interior vertex has one circular link, not a bow tie."""
    bad = set()
    for vertex, incident in vertex_faces.items():
        adjacency = defaultdict(list)
        for fi in incident:
            opposite = [int(x) for x in faces[fi] if int(x) != vertex]
            if len(opposite) != 2 or opposite[0] == opposite[1]:
                bad.add(vertex)
                break
            a, b = opposite
            adjacency[a].append(b)
            adjacency[b].append(a)
        if vertex in bad or not adjacency:
            bad.add(vertex)
            continue
        # Open-chain boundary links are handled by explicit boundary freezing.
        if any(len(nbs) > 2 for nbs in adjacency.values()):
            bad.add(vertex)
            continue
        todo = [min(adjacency)]
        visited = set()
        while todo:
            current = todo.pop()
            if current in visited:
                continue
            visited.add(current)
            todo.extend(adjacency[current])
        if visited != set(adjacency):
            bad.add(vertex)
    return bad


def _quadric_proposal(u, w, vertices, quadrics):
    quadric = quadrics[u] + quadrics[w]
    midpoint = (vertices[u].astype(np.float64) + vertices[w]) * 0.5
    choices = [vertices[u].astype(np.float64), vertices[w].astype(np.float64), midpoint]
    matrix, rhs = quadric[:3, :3], -quadric[:3, 3]
    # Planar regions deliberately use endpoint/midpoint fallbacks.
    try:
        if np.linalg.cond(matrix) < 1e12:
            solution = np.linalg.solve(matrix, rhs)
            if np.isfinite(solution).all():
                choices.append(solution)
    except np.linalg.LinAlgError:
        pass
    ranked = []
    for choice_index, point in enumerate(choices):
        # Audit the actual FP32 position which will be written, not a float64
        # position that may become degenerate after serialization.
        point = np.asarray(point, np.float32)
        h = np.append(point.astype(np.float64), 1.0)
        score = float(h @ quadric @ h)
        if np.isfinite(score) and np.isfinite(point).all():
            ranked.append((max(0.0, score), choice_index, point))
    if not ranked:
        return None
    score, choice_index, point = min(ranked, key=lambda row: (row[0], row[1]))
    return score, point, choice_index


def simplify(vertices, faces, locked_face_ids, target_faces, L):
    """Return ``(vertices, faces, surviving_source_face_ids, ledger)``.

    ``locked_face_ids`` are original C/source face IDs.  Their one-ring edge
    neighbors are also locked.  No original vertex row is removed or renumbered.
    An accepted ordinary interior collapse removes exactly two source faces;
    the output never drops below ``target_faces`` (one extra face may remain
    because of parity).  Existing defects are not automatically cleaned.
    """
    source_v = np.asarray(vertices)
    source_f = np.asarray(faces)
    if source_v.dtype != np.float32 or source_v.ndim != 2 or source_v.shape[1] != 3:
        raise ValueError("vertices must be finite float32[V,3]")
    if source_f.dtype != np.int64 or source_f.ndim != 2 or source_f.shape[1] != 3:
        raise ValueError("faces must be int64[F,3]")
    if not np.isfinite(source_v).all() or not math.isfinite(float(L)) or L <= 0:
        raise ValueError("Nonfinite geometry or invalid fixed reference L")
    if source_f.size and (source_f.min() < 0 or source_f.max() >= len(source_v)):
        raise ValueError("Face vertex index out of range")
    if int(target_faces) != target_faces or target_faces < 1:
        raise ValueError("target_faces must be a positive integer")
    initial_locked = sorted(set(map(int, locked_face_ids)))
    if initial_locked and (initial_locked[0] < 0 or initial_locked[-1] >= len(source_f)):
        raise ValueError("locked_face_ids out of source-face domain")
    v, f = source_v.copy(), source_f.copy()
    alive = np.ones(len(f), dtype=bool)
    edges, vertex_faces, _ = _topology(f, alive)
    effective_locked = set(initial_locked)
    initial_locked_set = set(initial_locked)
    for incident in edges.values():
        if initial_locked_set.intersection(incident):
            effective_locked.update(incident)
    protected_vertices = set(map(int, f[sorted(effective_locked)].reshape(-1)))
    fixed_vertices = set(protected_vertices)
    initial_boundary = {e for e, fs in edges.items() if len(fs) == 1}
    boundary_vertices = {x for e in initial_boundary for x in e}
    fixed_vertices.update(boundary_vertices)
    original_normals = _face_vectors(v, f)
    normal_lengths = np.linalg.norm(original_normals, axis=1)
    min_area = 1e-12 * float(L) ** 2
    degenerate_faces = set(map(int, np.flatnonzero(normal_lengths <= 2 * min_area)))
    canonical = defaultdict(list)
    for fi, tri in enumerate(f):
        canonical[tuple(sorted(map(int, tri)))].append(fi)
    duplicate_faces = {i for ids in canonical.values() if len(ids) > 1 for i in ids}
    defective_faces = set(degenerate_faces) | duplicate_faces
    nonmanifold_edges = {e for e, fs in edges.items() if len(fs) > 2}
    for edge in nonmanifold_edges:
        defective_faces.update(edges[edge])
    for fi in defective_faces:
        fixed_vertices.update(map(int, f[fi]))
    bad_link_vertices = _bad_vertex_links(f, vertex_faces)
    fixed_vertices.update(bad_link_vertices)
    sharp_edges = set()
    inconsistent_edges = set()
    cosine_sharp = math.cos(math.radians(SHARP_LIMIT_DEGREES))
    for edge, incident in edges.items():
        if len(incident) != 2:
            continue
        i, j = incident
        if _direction(f[i], *edge) == _direction(f[j], *edge):
            inconsistent_edges.add(edge)
            fixed_vertices.update(edge)
        if normal_lengths[i] <= 2 * min_area or normal_lengths[j] <= 2 * min_area:
            continue
        cosine = float(np.dot(original_normals[i], original_normals[j]) /
                       (normal_lengths[i] * normal_lengths[j]))
        if cosine < cosine_sharp:
            sharp_edges.add(edge)
            fixed_vertices.update(edge)
    quadrics = np.zeros((len(v), 4, 4), np.float64)
    for fi, tri in enumerate(f):
        if normal_lengths[fi] <= 2 * min_area:
            continue
        n = original_normals[fi] / normal_lengths[fi]
        plane = np.append(n, -float(np.dot(n, v[int(tri[0])].astype(np.float64))))
        q = np.outer(plane, plane) * (normal_lengths[fi] * 0.5)
        for vertex in tri:
            quadrics[int(vertex)] += q
    ledger = dict(
        version=VERSION, input_face_count=int(len(f)), requested_target_face_count=int(target_faces),
        L=float(L), initial_locked_face_ids=initial_locked,
        effective_locked_face_ids=sorted(effective_locked), protected_vertex_ids=sorted(protected_vertices),
        fixed_vertex_ids=sorted(fixed_vertices), boundary_vertices=int(len(boundary_vertices)),
        sharp_edge_count=len(sharp_edges), orientation_inconsistent_edge_count=len(inconsistent_edges),
        preexisting_degenerate_face_ids=sorted(degenerate_faces),
        preexisting_duplicate_face_ids=sorted(duplicate_faces),
        preexisting_nonmanifold_edges=[list(e) for e in sorted(nonmanifold_edges)],
        preexisting_bad_vertex_link_ids=sorted(bad_link_vertices),
        maximum_normal_change_degrees=NORMAL_LIMIT_DEGREES,
        sharp_feature_angle_degrees=SHARP_LIMIT_DEGREES,
        minimum_face_area_model_squared=min_area, vertex_index_policy="original_array_and_indices_retained",
        quadric_policy="area_weighted_original_plane_quadrics_accumulated_at_collapses",
        candidate_order="quadric_cost_then_ascending_original_vertex_ids",
        position_tie_break="lower_endpoint_then_upper_endpoint_then_midpoint_then_solve",
        boundary_policy="freeze_all_original_boundary_vertices_and_preserve_exact_boundary_edges",
        source_defect_policy="preserve_existing_defects_no_automatic_cleanup",
        validation_scope="proposal_only_requires_independent_surface_and_intersection_admission",
        collapses=[], rejection_counts={}, model_calls=0,
    )
    rejected = Counter()
    cosine_limit = math.cos(math.radians(NORMAL_LIMIT_DEGREES))
    while int(alive.sum()) - 2 >= int(target_faces):
        edges, vertex_faces, neighbors = _topology(f, alive)
        candidates = []
        for (u, w), incident in sorted(edges.items()):
            if len(incident) != 2 or len(set(incident)) != 2:
                rejected["NOT_ORDINARY_TWO_FACE_EDGE"] += 1
                continue
            if u in fixed_vertices or w in fixed_vertices:
                rejected["FROZEN_VERTEX"] += 1
                continue
            if _direction(f[incident[0]], u, w) == _direction(f[incident[1]], u, w):
                rejected["EDGE_WINDING_INCONSISTENT"] += 1
                continue
            opposing = {int(x) for fi in incident for x in f[fi] if int(x) not in (u, w)}
            if len(opposing) != 2 or neighbors[u].intersection(neighbors[w]) != opposing:
                rejected["LINK_CONDITION"] += 1
                continue
            proposal = _quadric_proposal(u, w, v, quadrics)
            if proposal is None:
                rejected["NONFINITE_QUADRIC_PROPOSAL"] += 1
                continue
            cost, point, position_choice = proposal
            candidates.append((cost, u, w, point, position_choice))
        candidates.sort(key=lambda row: (row[0], row[1], row[2]))
        old_duplicate_counts = Counter(tuple(sorted(map(int, f[i]))) for i in np.flatnonzero(alive))
        accepted = False
        for cost, keep, drop, point, position_choice in candidates:
            incident = edges[(keep, drop)]
            affected = sorted(vertex_faces[keep] | vertex_faces[drop])
            surviving_affected = [i for i in affected if i not in incident]
            trialf = f.copy()
            trialf[affected] = np.where(trialf[affected] == drop, keep, trialf[affected])
            trialalive = alive.copy()
            trialalive[incident] = False
            tv = v.copy()
            tv[keep] = point
            if any(len(set(map(int, trialf[i]))) != 3 for i in surviving_affected):
                rejected["UNEXPECTED_FACE_COLLAPSE"] += 1
                continue
            if surviving_affected:
                next_n = _face_vectors(tv, trialf[surviving_affected])
                next_len = np.linalg.norm(next_n, axis=1)
                if np.any(next_len <= 2 * min_area):
                    rejected["NEW_GEOMETRIC_DEGENERACY"] += 1
                    continue
                previous_n = _face_vectors(v, f[surviving_affected])
                original_n = original_normals[surviving_affected]
                normal_ok = True
                for reference in (previous_n, original_n):
                    reference_len = np.linalg.norm(reference, axis=1)
                    if np.any(reference_len <= 2 * min_area):
                        normal_ok = False
                        break
                    dots = np.einsum("ij,ij->i", next_n, reference) / (next_len * reference_len)
                    if np.any(dots < cosine_limit - 1e-12):
                        normal_ok = False
                        break
                if not normal_ok:
                    rejected["NORMAL_CHANGE_OVER_30_DEGREES"] += 1
                    continue
            next_counts = Counter(tuple(sorted(map(int, trialf[i]))) for i in np.flatnonzero(trialalive))
            if any(count > max(1, old_duplicate_counts.get(key, 0)) for key, count in next_counts.items()):
                rejected["NEW_DUPLICATE_FACE"] += 1
                continue
            next_edges, next_vertex_faces, _ = _topology(trialf, trialalive)
            if any(len(fs) > 2 and len(fs) > len(edges.get(edge, [])) for edge, fs in next_edges.items()):
                rejected["NEW_NONMANIFOLD_EDGE"] += 1
                continue
            if {edge for edge, fs in next_edges.items() if len(fs) == 1} != initial_boundary:
                rejected["BOUNDARY_EDGE_SET_CHANGED"] += 1
                continue
            local_links = {x: next_vertex_faces[x] for x in set(trialf[surviving_affected].reshape(-1))
                           if int(x) not in bad_link_vertices}
            if _bad_vertex_links(trialf, local_links):
                rejected["NEW_NONMANIFOLD_VERTEX_LINK"] += 1
                continue
            # Protected coordinates/face rows are checked in the actual stored dtype.
            protected = sorted(effective_locked)
            if (not trialalive[protected].all() or
                    trialf[protected].tobytes() != source_f[protected].tobytes() or
                    tv[source_f[protected]].tobytes() != source_v[source_f[protected]].tobytes()):
                raise AssertionError("Internal error: protected C/interface geometry changed")
            ledger["collapses"].append(dict(
                collapse_index=len(ledger["collapses"]), keep_original_vertex_id=int(keep),
                removed_original_vertex_id=int(drop), quadric_cost=float(cost),
                position_choice=int(position_choice), old_keep_position=v[keep].tolist(),
                old_drop_position=v[drop].tolist(), new_position=point.tolist(),
                removed_original_face_ids=sorted(map(int, incident)),
                affected_surviving_original_face_ids=list(map(int, surviving_affected)),
                face_count_after=int(trialalive.sum()),
            ))
            v, f, alive = tv, trialf, trialalive
            quadrics[keep] += quadrics[drop]
            quadrics[drop] = 0
            accepted = True
            break
        if not accepted:
            break
    kept = np.flatnonzero(alive).astype(np.int64)
    final_faces = f[kept].copy()
    locked = sorted(effective_locked)
    assert alive[locked].all()
    assert f[locked].tobytes() == source_f[locked].tobytes()
    assert v[source_f[locked]].tobytes() == source_v[source_f[locked]].tobytes()
    fixed = sorted(fixed_vertices)
    assert v[fixed].tobytes() == source_v[fixed].tobytes()
    count = len(final_faces)
    if len(source_f) <= target_faces:
        status = "UNCHANGED_TARGET_ALREADY_MET"
    elif count == target_faces:
        status = "TARGET_REACHED_PROPOSAL"
    elif count == target_faces + 1:
        status = "TARGET_REACHED_WITH_TWO_FACE_COLLAPSE_PARITY_PROPOSAL"
    else:
        status = "CONSTRAINT_LIMITED_PROPOSAL"
    ledger.update(status=status, output_face_count=count,
                  accepted_collapses=len(ledger["collapses"]),
                  removed_original_face_ids=np.flatnonzero(~alive).astype(int).tolist(),
                  rejection_counts=dict(sorted(rejected.items())),
                  effective_locked_faces_bitwise_unchanged=True,
                  fixed_vertices_bitwise_unchanged=True)
    return v, final_faces, kept, ledger
