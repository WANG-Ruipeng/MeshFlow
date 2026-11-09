"""Independent CPU gates for fixed-C, source-index-preserving lower-LOD meshes.

``kept_original_face_ids[j]`` is the unique source face slot inherited by new
face j, including changed surviving faces. Vertex IDs retain their source index
domain: no vertex compaction/reindexing is supported. Missing/ambiguous lineage
is BLOCKED, never an assumed clean result. This validates a candidate; it does
not repair it or certify semantic completeness, watertightness or Hausdorff
distance. All supplied arrays are read-only inputs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import importlib
import itertools
from pathlib import Path
import sys
import time
import types

import numpy as np

VERSION = "bench_fixed_c_multilod_quality_v1"
ROOT = Path(__file__).resolve().parents[3]
MIN_FACE_COUNT = 128
ILLEGAL = frozenset(("proper_transverse_penetration", "coplanar_positive_area_overlap"))
FROZEN_HASHES = {
    "intersections.py": "23a2e8c8deefcceb51ace60ad8d8a58d352d7c38bdcba2e8f85d6ceabd447d86",
    "seams.py": "9ce6927710d80eedec19aa10e2100ade582cf00e6942e9de4ef987618a005e4b",
}


@lru_cache(maxsize=1)
def _detector():
    """Only import the existing hash-verified CPU scorer, never a model package."""
    folder = ROOT / "r2/research/seam_slot_c200_v1/scoring_source/legacy/src/meshflow_control/evaluation"
    hashes = {}
    for name, expected in FROZEN_HASHES.items():
        path = folder / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError("Frozen finite-face detector hash mismatch: " + str(path))
        hashes[str(path)] = digest
    package = "_bench_multilod_frozen_detector_v1"
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(folder)]
        sys.modules[package] = module
    module = importlib.import_module(package + ".intersections")
    return module, hashes


def edgebook(faces):
    book = defaultdict(list)
    for fi, face in enumerate(faces):
        for j in range(3):
            a, b = int(face[j]), int(face[(j + 1) % 3])
            book[tuple(sorted((a, b)))].append((fi, a, b))
    return book


def _components(faces):
    book = edgebook(faces)
    parent = list(range(len(faces)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for rows in book.values():
        for row in rows[1:]:
            parent[find(row[0])] = find(rows[0][0])
    groups = defaultdict(list)
    for i in range(len(faces)):
        groups[find(i)].append(i)
    return sorted(groups.values(), key=lambda group: group[0])


def _topology(faces, lineage):
    book = edgebook(faces)
    nonmanifold, same_winding = set(), set()
    for edge, rows in book.items():
        identity = (edge, tuple(sorted(int(lineage[r[0]]) for r in rows)))
        if len(rows) > 2:
            nonmanifold.add(identity)
        elif len(rows) == 2 and rows[0][1:] != rows[1][1:][::-1]:
            same_winding.add(identity)
    return dict(components=_components(faces), nonmanifold=nonmanifold,
                same_winding=same_winding,
                incidence_histogram=dict(sorted(Counter(len(x) for x in book.values()).items())),
                used_vertex_count=int(len(np.unique(faces))), edge_count=len(book))


def _areas(triangles):
    t = np.asarray(triangles, np.float64)
    return np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1) * .5


def _duplicate_pairs(triangles, lineage):
    groups = defaultdict(list)
    for i, triangle in enumerate(triangles):
        key = tuple(sorted(tuple(float(x) for x in point) for point in triangle))
        groups[key].append(int(lineage[i]))
    return {tuple(sorted(pair)) for group in groups.values()
            for pair in itertools.combinations(group, 2)}


def probes(triangles):
    """13 deterministic probes/face: 3 vertices, 9 edge samples, 1 centroid."""
    t = np.asarray(triangles, np.float64)
    weights = [np.eye(3)[i] for i in range(3)]
    for a, b in ((0, 1), (1, 2), (2, 0)):
        for fraction in (.25, .5, .75):
            w = np.zeros(3)
            w[a], w[b] = 1 - fraction, fraction
            weights.append(w)
    weights.append(np.full(3, 1 / 3))
    return np.einsum("pc,fcd->fpd", np.asarray(weights), t).reshape(-1, 3)


def distances(points, triangles):
    """Existing data-build distance algorithm, with bounded temporary memory."""
    import trimesh
    triangles = np.asarray(triangles, np.float64)
    if not len(triangles):
        raise ValueError("Finite-surface query has no target faces")
    mesh = trimesh.Trimesh(vertices=triangles.reshape(-1, 3),
        faces=np.arange(3 * len(triangles)).reshape(-1, 3), process=False)
    results = []
    for start in range(0, len(points), 64):
        _, d, _ = trimesh.proximity.closest_point_naive(mesh, points[start:start + 64])
        results.extend(d)
    d = np.asarray(results, np.float64)
    if not np.isfinite(d).all():
        raise ValueError("Nonfinite finite-surface distances")
    return d


def _fidelity(first, second, L):
    d0 = distances(probes(first), second)
    d1 = distances(probes(second), first)
    bbox0 = np.stack((first.min((0, 1)), first.max((0, 1))))
    bbox1 = np.stack((second.min((0, 1)), second.max((0, 1))))
    return dict(source_to_new_probe_max_L=float(d0.max() / L),
                new_to_source_probe_max_L=float(d1.max() / L),
                bbox_max_coordinate_shift_L=float(np.abs(bbox0 - bbox1).max() / L),
                source_probe_count=len(d0), new_probe_count=len(d1))


@lru_cache(maxsize=8)
def _scan_cached(triangle_bytes, n, L):
    """Complete pair universe, AABB rejection then original verified classifier."""
    t = np.frombuffer(triangle_bytes, np.float64).reshape(n, 3, 3)
    detector, hashes = _detector()
    lo, hi = t.min(1), t.max(1)
    rows, unresolved, calls = [], [], {}
    aabb_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if np.any(lo[i] > hi[j] + 1e-9 * L) or np.any(lo[j] > hi[i] + 1e-9 * L):
                aabb_count += 1
                continue
            q = detector.verified_pair(t[i], t[j], L, ledger=calls)
            label = q.get("classification", "UNRESOLVED")
            if not q.get("verified", False) or not q.get("resolved", False):
                unresolved.append(dict(pair=[i, j], classification=label, reason=q.get("reason")))
            elif label in ILLEGAL:
                rows.append(dict(pair=[i, j], classification=label))
    return dict(illegal=rows, unresolved=unresolved, pair_universe=n * (n - 1) // 2,
                aabb_disjoint_pairs=aabb_count, detector_calls=calls,
                detector_policy=detector.DETECTOR_POLICY, detector_source_hashes=hashes)


def _pair_groups(rows, lineage, known):
    result = {k: set() for k in ("CC", "CF", "FF")}
    for row in rows:
        a, b = sorted(int(lineage[i]) for i in row["pair"])
        kind = "CC" if a in known and b in known else "CF" if a in known or b in known else "FF"
        result[kind].add((a, b, row["classification"]))
    return result


def evaluate(vertices, source_faces, new_vertices, new_faces,
             kept_original_face_ids, known_ids, L):
    """Return a JSON-safe independent gate report. No inputs are mutated.

    Numeric PASS is pending visual review. REJECTED means a demonstrated gate
    violation. BLOCKED means evidence is unavailable/uncertain; it can coexist
    with explicit rejection reasons. No detector failure becomes zero pairs.
    """
    started = time.perf_counter()
    result = dict(version=VERSION, status="BLOCKED", geometry_pass=False,
                  rejection_reasons=[], blocked_reasons=[], checks={},
                  scope="Fixed-C source-target LOD validation, no generated-output scoring",
                  gpu_calls=0, model_calls=0)
    rejects, blocked, checks = result["rejection_reasons"], result["blocked_reasons"], result["checks"]
    def finish():
        result["geometry_pass"] = not rejects and not blocked
        result["status"] = "BLOCKED" if blocked else "REJECTED" if rejects else "PASS_PENDING_VISUAL"
        result["wall_seconds"] = time.perf_counter() - started
        return result
    def gate(name, passed, **detail):
        checks[name] = dict(pass_=bool(passed), **detail)
        if not passed:
            rejects.append(name)
    v, f, nv, nf = map(np.asarray, (vertices, source_faces, new_vertices, new_faces))
    lineage, known = map(np.asarray, (kept_original_face_ids, known_ids))
    try:
        if not np.isfinite(L) or L <= 0:
            raise ValueError("L must be finite and positive")
        for name, a in (("vertices", v), ("new_vertices", nv)):
            if a.ndim != 2 or a.shape[1] != 3 or a.dtype.kind != "f" or not np.isfinite(a).all():
                raise ValueError(name + " must be finite floating [V,3]")
        for name, a, vv in (("source_faces", f, v), ("new_faces", nf, nv)):
            if a.ndim != 2 or a.shape[1] != 3 or not len(a) or a.dtype.kind not in "iu" or a.min() < 0 or a.max() >= len(vv):
                raise ValueError(name + " must have nonempty in-range integer [N,3] indices")
        if lineage.shape != (len(nf),) or lineage.dtype.kind not in "iu" or len(np.unique(lineage)) != len(lineage) or lineage.min() < 0 or lineage.max() >= len(f):
            raise ValueError("kept_original_face_ids must be unique newface-to-oldface integer IDs")
        if known.ndim != 1 or known.dtype.kind not in "iu" or not 0 < len(known) < len(f) or len(np.unique(known)) != len(known) or known.min() < 0 or known.max() >= len(f):
            raise ValueError("known_ids must identify unique nonempty C and leave free faces")
        if len(nv) != len(v):
            raise ValueError("Original vertex index domain must be preserved, without compaction")
    except (ValueError, TypeError) as exc:
        blocked.append("INPUT_CONTRACT: " + str(exc))
        return finish()
    L = float(L)
    oldids = np.arange(len(f))
    known_set = set(map(int, known))
    inverse = {int(old): i for i, old in enumerate(lineage)}
    oldbook = edgebook(f)
    interfaces = []
    interface_faces = set()
    for edge, rows in oldbook.items():
        ci = [r[0] for r in rows if r[0] in known_set]
        fi = [r[0] for r in rows if r[0] not in known_set]
        if ci and fi:
            interface_faces.update(fi)
            interfaces.append(dict(source_vertex_ids=list(edge), C_faces=ci, free_faces=fi))
    protected = sorted(known_set | interface_faces)
    missing = [i for i in protected if i not in inverse]
    changed = [i for i in protected if i in inverse and not np.array_equal(f[i], nf[inverse[i]])]
    pvids = np.unique(f[protected])
    exact_coords = v.dtype == nv.dtype and v[pvids].tobytes() == nv[pvids].tobytes()
    gate("C_AND_SOURCE_CF_INTERFACE_UNCHANGED", not missing and not changed and exact_coords,
         known_source_face_ids=sorted(known_set), interface_source_face_ids=sorted(interface_faces),
         protected_source_vertex_ids=pvids.tolist(), missing_face_ids=missing,
         changed_face_ids=changed, protected_coordinates_bitwise_equal=exact_coords,
         interface_policy="Original indexed CF edge incidence; not approximate coordinate contact",
         source_interfaces=interfaces)
    gate("N_AT_LEAST_128", len(nf) >= MIN_FACE_COUNT, original_N=len(f), new_N=len(nf))
    ot, nt = v[f].astype(np.float64), nv[nf].astype(np.float64)
    oa, na = _areas(ot), _areas(nt)
    if oa.sum() <= 0:
        blocked.append("SOURCE_TOTAL_AREA_ZERO")
        return finish()
    ratio = float(na.sum() / oa.sum())
    gate("TOTAL_AREA_RATIO", .98 <= ratio <= 1.02, ratio=ratio, allowed=[.98, 1.02])
    od = set(np.flatnonzero((oa <= 1e-12 * L * L) | np.array([len(set(x)) < 3 for x in f])).tolist())
    nd = {int(lineage[i]) for i in np.flatnonzero((na <= 1e-12 * L * L) | np.array([len(set(x)) < 3 for x in nf]))}
    gate("NO_NEW_DEGENERATE", not nd - od, source_face_ids=sorted(od),
         retained_source_ids=sorted(nd), new_source_ids=sorted(nd - od), removed_source_ids=sorted(od - nd),
         area_threshold_model2=1e-12 * L * L)
    dup0, dup1 = _duplicate_pairs(ot, oldids), _duplicate_pairs(nt, lineage)
    gate("NO_NEW_EXACT_DUPLICATE", not dup1 - dup0,
         source_pairs=sorted(dup0), current_pairs=sorted(dup1),
         new_pairs=sorted(dup1 - dup0), eliminated_pairs=sorted(dup0 - dup1),
         definition="Unordered exact numeric triangle coordinates, both windings included")
    top0, top1 = _topology(f, oldids), _topology(nf, lineage)
    for name, key in (("NO_NEW_NONMANIFOLD_EDGE", "nonmanifold"), ("NO_NEW_SAME_WINDING_EDGE", "same_winding")):
        gate(name, not top1[key] - top0[key], original_count=len(top0[key]), new_count=len(top1[key]),
             introduced=sorted(top1[key] - top0[key]), eliminated=sorted(top0[key] - top1[key]))
    comp_id = {i: ci for ci, group in enumerate(top0["components"]) for i in group}
    new_comp_sources = [{comp_id[int(lineage[i])] for i in group} for group in top1["components"]]
    component_rows, comp_ok = [], True
    for ci, group in enumerate(top0["components"]):
        ids = [j for j, old in enumerate(lineage) if comp_id[int(old)] == ci]
        occurrences = [j for j, sources in enumerate(new_comp_sources) if ci in sources]
        identity_ok = len(occurrences) == 1 and new_comp_sources[occurrences[0]] == {ci}
        area0, area1 = float(oa[group].sum()), float(na[ids].sum())
        r = area1 / area0 if area0 > 0 else None
        row = dict(source_component=ci, source_face_ids=group, new_face_ids=ids,
                   source_area=area0, new_area=area1, area_ratio=r,
                   topology_one_to_one=identity_ok)
        good = identity_ok and r is not None and .95 <= r <= 1.05
        if ids:
            try:
                fidelity = _fidelity(ot[group], nt[ids], L)
                row["fidelity"] = fidelity
                good &= max(fidelity[k] for k in ("source_to_new_probe_max_L", "new_to_source_probe_max_L", "bbox_max_coordinate_shift_L")) <= .0025 + 1e-12
            except Exception as exc:
                blocked.append("COMPONENT_DISTANCE_UNAVAILABLE: " + repr(exc))
                good = False
        row["pass_"] = bool(good)
        comp_ok &= good
        component_rows.append(row)
    gate("COMPONENT_PRESERVATION", comp_ok, source_count=len(top0["components"]),
         new_count=len(top1["components"]), per_component=component_rows,
         area_ratio_allowed=[.95, 1.05],
         component_definition="Indexed face adjacency through any common edge; no single-component or watertight requirement")
    result["topology"] = {label: {k: top[k] for k in ("incidence_histogram", "used_vertex_count", "edge_count")}
                          for label, top in (("source", top0), ("new", top1))}
    try:
        fidelity = _fidelity(ot, nt, L)
        gate("SAMPLED_SURFACE_AND_BBOX", max(fidelity[k] for k in ("source_to_new_probe_max_L", "new_to_source_probe_max_L", "bbox_max_coordinate_shift_L")) <= .0025 + 1e-12,
             **fidelity, threshold_L=.0025, sampling="13 probes/face: vertices, each edge at 1/4,1/2,3/4, centroid",
             guarantee="Bidirectional sampled finite-triangle distance, NOT strict/continuous Hausdorff")
    except Exception as exc:
        blocked.append("SURFACE_DISTANCE_UNAVAILABLE: " + repr(exc))
    try:
        oldscan = _scan_cached(np.ascontiguousarray(ot).tobytes(), len(ot), L)
        newscan = _scan_cached(np.ascontiguousarray(nt).tobytes(), len(nt), L)
        oldpairs, newpairs = _pair_groups(oldscan["illegal"], oldids, known_set), _pair_groups(newscan["illegal"], lineage, known_set)
        u0, u1 = _pair_groups(oldscan["unresolved"], oldids, known_set), _pair_groups(newscan["unresolved"], lineage, known_set)
        pairs = {}
        for kind in ("CC", "CF", "FF"):
            before, after = oldpairs[kind], newpairs[kind]
            pairs[kind] = dict(original_count=len(before), current_count=len(after),
                new_count=len(after - before), eliminated_count=len(before - after),
                original=sorted(before), current=sorted(after), new=sorted(after - before), eliminated=sorted(before - after),
                source_unresolved=sorted(u0[kind]), new_unresolved=sorted(u1[kind]),
                introduced_unresolved=sorted(u1[kind] - u0[kind]))
            gate("NO_NEW_ILLEGAL_" + kind, not after - before, new_pair_count=len(after - before))
        if oldscan["unresolved"] or newscan["unresolved"]:
            blocked.append("FINITE_PAIR_UNRESOLVED: historical/new uncertain pairs retained explicitly; cannot certify no new conflicts")
        result["finite_pairs"] = dict(status="BLOCKED_UNRESOLVED" if oldscan["unresolved"] or newscan["unresolved"] else "COMPLETE",
            by_partition=pairs, detector_policy=oldscan["detector_policy"], detector_source_hashes=oldscan["detector_source_hashes"],
            source_scan={k: oldscan[k] for k in ("pair_universe", "aabb_disjoint_pairs", "detector_calls")},
            new_scan={k: newscan[k] for k in ("pair_universe", "aabb_disjoint_pairs", "detector_calls")},
            pair_identity="Original source face IDs plus unchanged detector classification; no nearest-face ancestry")
    except Exception as exc:
        blocked.append("FINITE_PAIR_DETECTOR_UNAVAILABLE: " + repr(exc))
        result["finite_pairs"] = dict(status="BLOCKED", new_CF_pairs=None, new_FF_pairs=None, reason=repr(exc))
    result["lineage"] = dict(new_face_to_original_face=lineage.tolist(),
                             deleted_original_face_ids=sorted(set(range(len(f))) - set(map(int, lineage))),
                             original_vertex_index_domain_preserved=True)
    return finish()
