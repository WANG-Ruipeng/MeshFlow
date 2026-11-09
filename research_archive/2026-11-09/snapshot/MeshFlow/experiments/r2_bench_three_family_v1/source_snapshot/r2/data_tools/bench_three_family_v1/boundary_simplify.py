"""Conservative shared planar-boundary simplification, CPU / NumPy only.

No coordinate is moved, no source array is overwritten, and no near-planar
chain is accepted by accumulating local tolerances.  Candidate deletions are
shared by every incident patch.  A failed patch rolls back the entire connected
component of proposed patch changes, avoiding a half-applied T junction.
"""
from collections import Counter, defaultdict
import numpy as np

VERSION = "SHARED_COLLINEAR_PATCH_BOUNDARY_V1"
NORMAL_DOT_MIN = 1.0 - 1e-10
PLANE_TOL_L = 1e-7
LINE_TOL_L = 1e-7
MODEL_L = float(np.sqrt(3) * 2 / .3762)


def edgebook(faces):
    book = defaultdict(list)
    for fi, tri in enumerate(faces):
        for j in range(3):
            u, v = int(tri[j]), int(tri[(j + 1) % 3])
            book[tuple(sorted((u, v)))].append((fi, u, v))
    return book


def _cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _area(vertices, faces):
    t = vertices[np.asarray(faces, dtype=np.int64)]
    return np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1) * .5


def _single_loop(faces):
    book = edgebook(faces)
    if any(len(r) > 2 for r in book.values()):
        return None, "NONMANIFOLD_PATCH"
    boundary = [r[0][1:] for r in book.values() if len(r) == 1]
    outgoing, incoming = defaultdict(list), defaultdict(list)
    for u, v in boundary:
        outgoing[u].append(v)
        incoming[v].append(u)
    if not boundary or set(outgoing) != set(incoming) or any(len(r) != 1 for r in outgoing.values()) or any(len(r) != 1 for r in incoming.values()):
        return None, "NON_SIMPLE_BOUNDARY"
    start = min(outgoing)
    loop, current = [start], outgoing[start][0]
    while current != start and current not in loop:
        loop.append(current)
        current = outgoing[current][0]
    if current != start or len(loop) != len(boundary):
        return None, "MULTI_LOOP_OR_HOLE"
    return loop, "SIMPLE_LOOP"


def _ear_clip(vertices, loop, normal):
    xy = np.delete(vertices[np.asarray(loop)], int(np.argmax(np.abs(normal))), axis=1)
    signed_area = sum(_cross(xy[i], xy[(i + 1) % len(xy)]) for i in range(len(xy)))
    if signed_area == 0:
        return None
    sign = 1 if signed_area > 0 else -1
    tol = 1e-13 * max(float(np.ptp(xy, axis=0).max()) ** 2, 1.)
    remaining, result = list(range(len(loop))), []
    while len(remaining) > 3:
        found = False
        for pos in range(len(remaining)):
            ia, ib, ic = remaining[pos - 1], remaining[pos], remaining[(pos + 1) % len(remaining)]
            a, b, c = xy[[ia, ib, ic]]
            if sign * _cross(b - a, c - b) <= tol:
                continue
            blocked = any(min(sign * _cross(b - a, xy[j] - a), sign * _cross(c - b, xy[j] - b), sign * _cross(a - c, xy[j] - c)) >= -tol for j in remaining if j not in (ia, ib, ic))
            if blocked:
                continue
            result.append([loop[ia], loop[ib], loop[ic]])
            remaining.pop(pos)
            found = True
            break
        if not found:
            return None
    result.append([loop[i] for i in remaining])
    return np.asarray(result, dtype=np.int64)


def _collinear_chain(vertices, chain, tol):
    """Test all original chain vertices against one final segment."""
    pts = vertices[np.asarray(chain)]
    axis = pts[-1] - pts[0]
    length = float(np.linalg.norm(axis))
    if length <= tol:
        return False
    direction = axis / length
    offsets = pts - pts[0]
    projection = offsets @ direction
    distance = np.linalg.norm(offsets - projection[:, None] * direction, axis=1)
    return bool(distance.max() <= tol and np.all(np.diff(projection) > 0) and abs(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() - length) <= tol)


def _contact_with_other_surface(vertices, faces, vid, tol):
    """Conservatively protect geometric contacts even without shared indices."""
    point = vertices[vid]
    other = faces[~np.any(faces == vid, axis=1)]
    if len(other) == 0:
        return False
    tri = vertices[other]
    near = np.all(point >= tri.min(axis=1) - tol, axis=1) & np.all(point <= tri.max(axis=1) + tol, axis=1)
    tri = tri[near]
    if len(tri) == 0:
        return False
    u, w = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    n = np.cross(u, w)
    nlen = np.linalg.norm(n, axis=1)
    valid = nlen > 0
    tri, u, w, n, nlen = tri[valid], u[valid], w[valid], n[valid], nlen[valid]
    if len(tri) == 0:
        return False
    rel = point - tri[:, 0]
    height = np.abs(np.einsum("ij,ij->i", rel, n)) / nlen
    uu, ww, uw = np.einsum("ij,ij->i", u, u), np.einsum("ij,ij->i", w, w), np.einsum("ij,ij->i", u, w)
    ru, rw = np.einsum("ij,ij->i", rel, u), np.einsum("ij,ij->i", rel, w)
    det = uu * ww - uw * uw
    alpha, beta = (ru * ww - rw * uw) / det, (rw * uu - ru * uw) / det
    btol = tol / np.maximum(np.minimum(np.sqrt(uu), np.sqrt(ww)), tol)
    return bool(np.any((height <= tol) & (alpha >= -btol) & (beta >= -btol) & (alpha + beta <= 1 + btol)))


def analyze(vertices, faces, face_labels=None, *, L=MODEL_L):
    """Read-only patch/candidate inspection. No mesh proposal is constructed."""
    v, f = np.asarray(vertices, np.float64), np.asarray(faces, np.int64)
    if f.ndim != 2 or f.shape[1] != 3 or not np.all(np.isfinite(v)):
        raise ValueError("Finite vertices and triangular faces required")
    labels = np.zeros(len(f), np.int64) if face_labels is None else np.asarray(face_labels, np.int64)
    if labels.shape != (len(f),):
        raise ValueError("One integer semantic membership label is required per face")
    triangles = v[f]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 0):
        raise ValueError("Source contains zero-area triangles; no automatic repair")
    normals /= lengths[:, None]
    book, adjacent, vertex_faces = edgebook(f), defaultdict(set), defaultdict(set)
    for entries in book.values():
        if len(entries) == 2 and entries[0][1:] == entries[1][1:]:
            continue
        if len(entries) == 2:
            i, j = entries[0][0], entries[1][0]
            adjacent[i].add(j)
            adjacent[j].add(i)
    for fi, tri in enumerate(f):
        for vid in tri:
            vertex_faces[int(vid)].add(fi)
    seen, patches, face_patch = set(), [], np.full(len(f), -1, np.int64)
    for seed in range(len(f)):
        if seed in seen:
            continue
        seen.add(seed)
        todo, group = [seed], []
        while todo:
            i = todo.pop()
            group.append(i)
            for j in sorted(adjacent[i]):
                if j not in seen and labels[j] == labels[seed] and np.dot(normals[j], normals[seed]) >= NORMAL_DOT_MIN and np.max(np.abs((triangles[j] - triangles[seed, 0]) @ normals[seed])) <= PLANE_TOL_L * L:
                    seen.add(j)
                    todo.append(j)
        group.sort()
        loop, state = _single_loop(f[group])
        pid = len(patches)
        face_patch[group] = pid
        patches.append(dict(patch_id=pid, face_ids=group, semantic_label=int(labels[seed]), loop=loop, state=state, normal=normals[seed].tolist()))
    candidates, reasons, vertex_patches = {}, {}, {}
    for vid, incident_faces in sorted(vertex_faces.items()):
        pids = sorted(set(map(int, face_patch[sorted(incident_faces)])))
        vertex_patches[vid] = pids
        local_edges = [rows for edge, rows in book.items() if vid in edge]
        if any(len(rows) > 2 for rows in local_edges):
            reasons[vid] = "NONMANIFOLD_EDGE_ENDPOINT"
            continue
        local_graph = defaultdict(set)
        for rows in local_edges:
            if len(rows) == 2:
                a, b = rows[0][0], rows[1][0]
                local_graph[a].add(b)
                local_graph[b].add(a)
        reached, todo = set(), [min(incident_faces)]
        while todo:
            item = todo.pop()
            if item in reached:
                continue
            reached.add(item)
            todo.extend(local_graph[item] - reached)
        if reached != incident_faces or sum(len(r) == 1 for r in local_edges) not in (0, 2):
            reasons[vid] = "NONMANIFOLD_VERTEX_OR_CONTACT"
            continue
        if len(pids) > 2:
            reasons[vid] = "PATCH_OR_SEMANTIC_JUNCTION"
            continue
        neighborhoods, state = [], None
        for pid in pids:
            patch = patches[pid]
            loop = patch["loop"]
            if loop is None:
                state = patch["state"]
                break
            if vid not in loop:
                state = "PATCH_INTERIOR_VERTEX_NOT_BOUNDARY_CANDIDATE"
                break
            pos = loop.index(vid)
            prev, nxt = loop[pos - 1], loop[(pos + 1) % len(loop)]
            if not _collinear_chain(v, [prev, vid, nxt], LINE_TOL_L * L):
                state = "REAL_CORNER_OR_NONCOLLINEAR"
                break
            neighborhoods.append(tuple(sorted((prev, nxt))))
        if state:
            reasons[vid] = state
            continue
        if len(set(neighborhoods)) != 1:
            reasons[vid] = "SHARED_CHAIN_NEIGHBORS_DIFFER"
            continue
        if _contact_with_other_surface(v, f, vid, LINE_TOL_L * L):
            reasons[vid] = "UNINDEXED_SURFACE_CONTACT_ENDPOINT"
            continue
        candidates[vid] = pids
    return dict(patches=patches, candidates=candidates, rejection_reasons=reasons, vertex_patches=vertex_patches,
                summary=dict(vertices_used=len(vertex_faces), patches=len(patches), simple_patches=sum(p["loop"] is not None for p in patches), candidate_vertices=len(candidates), candidate_patches=len(set(x for p in candidates.values() for x in p)), rejection_counts=dict(Counter(reasons.values())), plane_tol=PLANE_TOL_L * L, line_tol=LINE_TOL_L * L, normal_dot_min=NORMAL_DOT_MIN))


def simplify_boundaries(arrays, face_labels=None, *, L=MODEL_L):
    """Return (new arrays, new-face-to-old-patch lineage, JSON-safe ledger).

    Vertex arrays keep original indexing/values. New face identity belongs to the
    new target version; the lineage is a surface-region mapping, not source IDs.
    """
    v, f = np.asarray(arrays["model_vertices"], np.float64), np.asarray(arrays["source_faces"], np.int64)
    audit = analyze(v, f, face_labels, L=L)
    patches, candidates = audit["patches"], audit["candidates"]
    patch_graph = defaultdict(set)
    for pids in candidates.values():
        for p in pids:
            patch_graph[p].update(pids)
    seen, components = set(), []
    for seed in sorted(patch_graph):
        if seed in seen:
            continue
        todo, component = [seed], set()
        while todo:
            p = todo.pop()
            if p in component:
                continue
            component.add(p)
            todo.extend(patch_graph[p] - component)
        seen.update(component)
        components.append(sorted(component))
    proposals, removed, component_ledger = {}, set(), []
    for component in components:
        deletions = {vid for vid, pids in candidates.items() if any(p in component for p in pids)}
        local, failure = {}, None
        for pid in component:
            patch, loop = patches[pid], patches[pid]["loop"]
            newloop = [x for x in loop if x not in deletions]
            if len(newloop) < 3:
                failure = "FEWER_THAN_THREE_BOUNDARY_CORNERS"
                break
            kept = [i for i, x in enumerate(loop) if x not in deletions]
            for k, start in enumerate(kept):
                end = kept[(k + 1) % len(kept)]
                ids, i = [loop[start]], (start + 1) % len(loop)
                while i != end:
                    ids.append(loop[i])
                    i = (i + 1) % len(loop)
                ids.append(loop[end])
                if len(ids) > 2 and not _collinear_chain(v, ids, LINE_TOL_L * L):
                    failure = "GLOBAL_CHAIN_NOT_COLLINEAR"
                    break
            if failure:
                break
            newfaces = _ear_clip(v, newloop, np.asarray(patch["normal"]))
            if newfaces is None:
                failure = "EAR_CLIP_FAILED"
                break
            old_area, new_area = _area(v, f[patch["face_ids"]]).sum(), _area(v, newfaces).sum()
            tri = v[newfaces]
            nn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            boundary = {edge for edge, rows in edgebook(newfaces).items() if len(rows) == 1}
            expected = {tuple(sorted((newloop[j], newloop[(j + 1) % len(newloop)]))) for j in range(len(newloop))}
            if np.any(nn @ np.asarray(patch["normal"]) <= 0) or boundary != expected or abs(new_area - old_area) > max(1e-12 * L * L, old_area * 1e-6):
                failure = "AREA_WINDING_OR_BOUNDARY_VALIDATION_FAILED"
                break
            if len(newfaces) >= len(patch["face_ids"]):
                failure = "NO_STRICT_FACE_REDUCTION"
                break
            local[pid] = newfaces
        if failure is None:
            proposals.update(local)
            removed.update(deletions)
        component_ledger.append(dict(patch_ids=component, candidate_removed_vertex_ids=sorted(deletions), action="ACCEPTED_SHARED_CHAIN_RETRIANGULATION" if failure is None else "ROLLED_BACK_COMPONENT", rejection_reason=failure))
    out, lineage, patch_ledger = [], [], []
    for patch in patches:
        pid, group = patch["patch_id"], patch["face_ids"]
        new = proposals.get(pid)
        if new is None:
            out.extend(f[group].tolist())
            lineage.extend([[int(x)] for x in group])
        else:
            out.extend(new.tolist())
            lineage.extend([group.copy() for _ in new])
        patch_ledger.append(dict(patch_id=pid, old_face_ids=group, semantic_label=patch["semantic_label"], old_faces=len(group), new_faces=len(group) if new is None else len(new), action="UNCHANGED" if new is None else "SHARED_BOUNDARY_RETRIANGULATED", boundary_state=patch["state"]))
    result = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in arrays.items()}
    result["source_faces"] = np.asarray(out, np.int64)
    if any(vid in set(map(int, result["source_faces"].ravel())) for vid in removed):
        raise AssertionError("Partial deletion would leave a T junction")
    ledger = dict(version=VERSION, summary=dict(audit["summary"], N_original=len(f), N_proposed=len(out), removed_vertices=len(removed), changed_patches=len(proposals), accepted_components=sum(x["rejection_reason"] is None for x in component_ledger)),
                  removed_vertex_ids=sorted(removed), candidate_vertex_ids=sorted(candidates), vertex_rejections={str(k): val for k, val in audit["rejection_reasons"].items()}, components=component_ledger, patches=patch_ledger,
                  vertex_coordinates_unchanged=True, semantic_membership_preserved=True, original_face_ids_not_reused=True, full_surface_finite_pair_validation_required=True)
    return result, lineage, ledger
