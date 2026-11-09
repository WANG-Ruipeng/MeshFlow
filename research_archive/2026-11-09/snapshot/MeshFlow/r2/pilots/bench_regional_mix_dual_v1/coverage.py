"""Frozen centroid coverage heuristic over actual cached BASE_POST surfaces.

No model, scorer, new GEO mapping, source Y, visual labels, smoothing or region
expansion is used. Seven-point distances are diagnostics; only the centroid
test selects seeds. A missing sampled surface is not proof of a missing part.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

VERSION = 'auto_centroid_surviving_surface_v1'
RADIUS = .015 / (.95 * .3762)
POINT_NAMES = ('vertex0', 'vertex1', 'vertex2', 'midpoint01', 'midpoint12',
               'midpoint20', 'centroid')
CONFIG = dict(
    version=VERSION, radius_model=RADIUS, missing_distance_multiplier=2.,
    seed='valid RAW free face AND centroid finite-surface distance strictly > 2R',
    surface='All triangles actually present in original cached BASE_POST, including exact C; no shortest-edge survival proxy',
    raw_area_floor_L2=1e-12, component_area_fraction=.02, component_minimum_faces=2,
    component_area_denominator='Sum of areas of ALL original RAW free faces, not C and not only seeds',
    graph='Seed-induced common complete noncollapsed edge of cached RAW-GEO pre-cleanup coordinate cluster IDs; all incidences, including nonmanifold, retained',
    component_filter='One fixed pass; size >=2 AND area >=.02 total RAW free area',
    diagnostic_points=list(POINT_NAMES), diagnostic_ALL7='All seven distances strictly > 2R; ledger only, never an additional candidate',
    selection='All three corners RAW for retained seed components; all other free faces D0 PRE_GEO; C exact RAW bytes',
    expansion_rounds=0, smoothing=False, modified_GEO=False, C_normal_near_field=False,
    interpretation='Coverage heuristic only; does not certify an empty surface/volume, support function, adoption, safe junction or low fallback ratio',
    forbidden_inputs=['Y', 'source_full_mesh', 'visual_labels', 'scorer_results'])


def areas(triangles):
    t = np.asarray(triangles, np.float64)
    return .5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)


def query_points(raw):
    t = np.asarray(raw, np.float64)
    return np.stack((t[:, 0], t[:, 1], t[:, 2], (t[:, 0] + t[:, 1]) / 2,
                     (t[:, 1] + t[:, 2]) / 2, (t[:, 2] + t[:, 0]) / 2,
                     t.mean(axis=1)), axis=1)


def same_face_coordinate_multisets(left, right):
    """Source-restored C may permute corners, but cannot move/swap its faces."""
    def sorted_bits(value):
        bits = np.ascontiguousarray(value).view(np.uint32).reshape(-1, 3, 3)
        order = np.lexsort((bits[:, :, 2], bits[:, :, 1], bits[:, :, 0]), axis=1)
        return np.take_along_axis(bits, order[:, :, None], axis=1)
    return sorted_bits(left).tobytes() == sorted_bits(right).tobytes()


def nearest(points, surface):
    """Finite closed-triangle Euclidean distances, with deterministic face ties."""
    from r2.tools.cref_layer_audit_v1.geometry import finite_distances
    points = np.asarray(points, np.float64).reshape(-1, 3)
    surface = np.asarray(surface, np.float64)
    if surface.ndim != 3 or surface.shape[1:] != (3, 3) or not len(surface):
        raise ValueError('A nonempty actual BASE_POST surface is required')
    if not np.isfinite(points).all() or not np.isfinite(surface).all():
        raise ValueError('Nonfinite finite-surface query')
    distance = np.empty(len(points), np.float64)
    face_ids = np.empty(len(points), np.int64)
    for start in range(0, len(points), 64):
        d2, _, _ = finite_distances(points[start:start + 64], surface)
        ids = np.argmin(d2, axis=1)
        face_ids[start:start + len(ids)] = ids
        distance[start:start + len(ids)] = np.sqrt(d2[np.arange(len(ids)), ids])
    if not np.isfinite(distance).all():
        raise ValueError('Nonfinite nearest-surface result')
    return distance, face_ids


def classify_distances(distances, valid, *, R=RADIUS):
    if float(R) != RADIUS:
        raise ValueError('Only the fixed original GEO radius is allowed')
    d = np.asarray(distances, np.float64)
    valid = np.asarray(valid, bool)
    if d.shape != (len(valid), 7) or not np.isfinite(d[valid]).all():
        raise ValueError('Seven finite distances per valid RAW face are required')
    beyond = d > 2 * R
    centroid = valid & beyond[:, 6]
    all7 = valid & beyond.all(axis=1)
    return centroid, all7, beyond.sum(axis=1)


def edge_components(mask, cluster_ids):
    """No radius clustering, vertex-only link, C path, or recursive expansion."""
    mask = np.asarray(mask, bool)
    ids = np.asarray(cluster_ids)
    if ids.shape != (len(mask), 3) or ids.dtype.kind not in 'iu' or (ids < 0).any():
        raise ValueError('Nonnegative cached RAW-GEO corner cluster IDs required')
    incidence = {}
    for face in np.flatnonzero(mask):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((int(ids[face, a]), int(ids[face, b]))))
            if edge[0] != edge[1]:
                incidence.setdefault(edge, set()).add(int(face))
    adjacency = {int(f): set() for f in np.flatnonzero(mask)}
    for owners in incidence.values():
        for face in owners:
            adjacency[face].update(owners - {face})
    labels = np.full(len(mask), -1, np.int64)
    components = []
    for first in sorted(adjacency):
        if labels[first] >= 0:
            continue
        todo = [first]
        labels[first] = len(components)
        members = []
        while todo:
            current = todo.pop()
            members.append(current)
            for other in sorted(adjacency[current], reverse=True):
                if labels[other] < 0:
                    labels[other] = len(components)
                    todo.append(other)
        components.append(sorted(members))
    return labels, components, incidence


def filter_components(components, raw_areas, total_raw_free_area):
    threshold = CONFIG['component_area_fraction'] * total_raw_free_area
    selected = np.zeros(len(raw_areas), bool)
    records = []
    for component_id, members in enumerate(components):
        area = float(np.asarray(raw_areas)[members].sum())
        size_pass = len(members) >= CONFIG['component_minimum_faces']
        area_pass = area >= threshold
        retained = size_pass and area_pass
        if retained:
            selected[members] = True
        reasons = ([] if size_pass else ['SINGLE_FACE_COMPONENT']) + ([] if area_pass else ['AREA_BELOW_2PCT'])
        records.append(dict(component_id=component_id, raw_slots=members, faces=len(members),
                            raw_area=area, total_RAW_free_area=float(total_raw_free_area),
                            area_fraction=area / total_raw_free_area if total_raw_free_area > 0 else None,
                            threshold_area=threshold, minimum_faces_pass=size_pass,
                            area_pass=area_pass, selected_RAW=retained,
                            reason='RETAINED_SEED_COMPONENT' if retained else '|'.join(reasons)))
    return selected, records


def mixed_boundary(output, selected, known, cluster_ids):
    """Record shared RAW-GEO clusters; proximity does not certify a legal edge."""
    flat = output.reshape(-1, 3).astype(np.float64)
    ids = cluster_ids.reshape(-1)
    selected_corner = np.repeat(selected, 3)
    free_corner = np.arange(len(flat)) >= 3 * known
    records = []
    for cluster in np.unique(ids):
        members = np.flatnonzero(ids == cluster)
        a = members[selected_corner[members]]
        b = members[~selected_corner[members] & free_corner[members]]
        c = members[~free_corner[members]]
        if len(a) and (len(b) or len(c)):
            other = np.concatenate((b, c))
            distance = np.linalg.norm(flat[a, None, :] - flat[None, other, :], axis=2)
            records.append(dict(cluster_id=int(cluster), RAW_flat_corners=a.tolist(),
                                D0_flat_corners=b.tolist(), C_flat_corners=c.tolist(),
                                maximum_mixed_gap=float(distance.max()), minimum_mixed_gap=float(distance.min()),
                                exact_shared_coordinate_pairs=int((distance == 0).sum()),
                                mixed_pairs=int(distance.size),
                                all_members_same_position=bool(np.all(flat[members] == flat[members[0]]))))
    return records


def select(raw, pre, known, base_output, cluster_ids, *, L, R=RADIUS):
    """Return one fixed candidate and complete selection/diagnostic ledgers.

    Caller verifies frozen input hashes. base_output must be the original actual
    BASE_POST triangle array, and cluster_ids the already cached RAW-GEO map.
    This function deliberately cannot load assets, run GEO, or score a mesh.
    """
    if (not isinstance(raw, np.ndarray) or not isinstance(pre, np.ndarray)
            or raw.dtype != np.float32 or pre.dtype != np.float32
            or raw.shape != pre.shape or raw.ndim != 3 or raw.shape[1:] != (3, 3)):
        raise ValueError('Matched-slot FP32 RAW and D0 PRE_GEO arrays required')
    if isinstance(known, bool) or int(known) != known or not 0 < known < len(raw):
        raise ValueError('Invalid canonical known-face count')
    known = int(known)
    base = np.asarray(base_output)
    ids = np.asarray(cluster_ids)
    if (base.dtype != np.float32 or base.ndim != 3 or base.shape[1:] != (3, 3)
            or len(base) < known or not np.isfinite(base).all()):
        raise ValueError('Actual FP32 original BASE_POST triangles required')
    if (not np.isfinite(raw).all() or not np.isfinite(pre).all()
            or not np.isfinite(L) or L <= 0 or float(R) != RADIUS):
        raise ValueError('Invalid finite geometry, L, or frozen GEO radius')
    if raw[:known].tobytes() != pre[:known].tobytes():
        raise ValueError('RAW and PRE_GEO must preserve canonical C bytes')
    if not same_face_coordinate_multisets(raw[:known], base[:known]):
        raise ValueError('Source-restored BASE_POST C face coordinates differ beyond corner permutation')
    if ids.shape != raw.shape[:2] or ids.dtype.kind not in 'iu' or (ids < 0).any():
        raise ValueError('Cached RAW-GEO pre-cleanup cluster IDs required')
    raw_area = areas(raw)
    valid = raw_area > CONFIG['raw_area_floor_L2'] * L * L
    valid[:known] = False
    points = query_points(raw)
    distances = np.full((len(raw), 7), np.nan, np.float64)
    nearest_faces = np.full((len(raw), 7), -1, np.int64)
    d, fids = nearest(points[valid].reshape(-1, 3), base)
    distances[valid] = d.reshape(-1, 7)
    nearest_faces[valid] = fids.reshape(-1, 7)
    seed, all7, counts = classify_distances(distances, valid, R=R)
    region, components, incidence = edge_components(seed, ids)
    free_area = float(raw_area[known:].sum())
    selected, component_records = filter_components(components, raw_area, free_area)
    output = pre.copy()
    output[selected] = raw[selected]
    output[:known] = raw[:known]
    rows = []
    for face in range(len(raw)):
        reason = ('C_FIXED' if face < known else 'RAW_INVALID_NOT_SELECTED' if not valid[face]
                  else 'CENTROID_COVERED_D0' if not seed[face]
                  else component_records[int(region[face])]['reason'])
        row = dict(raw_slot=face, known=face < known, raw_valid=bool(valid[face]),
                   raw_area=float(raw_area[face]), centroid_seed=bool(seed[face]),
                   ALL7_seed_diagnostic=bool(all7[face]),
                   centroid_vs_ALL7_disagree=bool(seed[face] != all7[face]),
                   component_id=int(region[face]), selected_RAW=bool(selected[face]), reason=reason,
                   distance_min=float(distances[face].min()) if valid[face] else None,
                   distance_median=float(np.median(distances[face])) if valid[face] else None,
                   distance_max=float(distances[face].max()) if valid[face] else None,
                   points_strictly_beyond_2R=int(counts[face]) if valid[face] else None,
                   maximum_from_D0=float(np.linalg.norm(output[face].astype(float) - pre[face], axis=1).max()))
        for index, name in enumerate(POINT_NAMES):
            row['distance_' + name] = float(distances[face, index]) if valid[face] else None
            row['nearest_BASE_face_' + name] = int(nearest_faces[face, index]) if valid[face] else None
        rows.append(row)
    boundary = mixed_boundary(output, selected, known, ids)
    if (output[:known].tobytes() != raw[:known].tobytes()
            or output[selected].tobytes() != raw[selected].tobytes()
            or output[~selected].tobytes() != pre[~selected].tobytes()
            or np.any(selected & ~seed) or np.any(all7 & ~seed)):
        raise AssertionError('Whole-face endpoint or seed-only contract changed')
    summary = dict(version=VERSION, N=len(raw), K=known, R_model=float(R), threshold_model=2 * R,
                   original_BASE_surface_faces=len(base), original_BASE_free_surface_faces=len(base) - known,
                   original_BASE_exact_zero_area_faces=int((areas(base) == 0).sum()),
                   original_BASE_C_corner_order_permuted_faces=sum(
                       raw[f].tobytes() != base[f].tobytes() for f in range(known)),
                   valid_RAW_free_faces=int(valid.sum()), invalid_RAW_free_faces=int((~valid[known:]).sum()),
                   seed_faces=int(seed.sum()), centroid_seed_faces=int(seed.sum()),
                   diagnostic_ALL7_seed_faces=int(all7.sum()),
                   diagnostic_centroid_only_seed_faces=int((seed & ~all7).sum()),
                   diagnostic_centroid_only_seed_RAW_area=float(raw_area[seed & ~all7].sum()),
                   seed_components=len(components), retained_components=sum(r['selected_RAW'] for r in component_records),
                   rejected_components=sum(not r['selected_RAW'] for r in component_records),
                   selected_RAW_faces=int(selected.sum()), selected_RAW_face_fraction=float(selected[known:].mean()),
                   selected_raw_area=float(raw_area[selected].sum()), free_raw_area=free_area,
                   selected_RAW_area_fraction=float(raw_area[selected].sum() / free_area) if free_area > 0 else None,
                   filtered_seed_faces=int((seed & ~selected).sum()),
                   seed_graph_nonmanifold_edges=sum(len(v) > 2 for v in incidence.values()),
                   mixed_boundary_clusters=len(boundary),
                   split_mixed_boundary_clusters=sum(not r['all_members_same_position'] for r in boundary),
                   C_bytes_preserved=True, expansion_rounds=0, seven_point_hard_gate=False,
                   model_calls=0, D0_calls=0, GEO_calls=0, scorer_calls=0, new_noise=0)
    return dict(output=output, seed=seed, selected_RAW=selected, region_ids=region,
                cluster_ids=ids.copy(), distances=distances, nearest_BASE_face_ids=nearest_faces,
                diagnostic_ALL7_seed=all7, per_face=rows, components=component_records,
                boundary=boundary, summary=summary)


def write_ledger(result, destination, *, request_id, source_identities):
    """Write new selector-only evidence, never scientific scores or old assets."""
    root = Path(__file__).resolve().parents[3]
    allowed = (root / 'experiments/r2_bench_regional_mix_dual_v1').resolve()
    destination = Path(destination).resolve()
    if destination.exists() or not destination.is_relative_to(allowed):
        raise ValueError('Selector ledger requires a new directory under this experiment')
    destination.mkdir(parents=True)
    with (destination / 'selection.npz').open('xb') as handle:
        np.savez_compressed(handle, **{k: result[k] for k in (
            'output', 'seed', 'selected_RAW', 'region_ids', 'cluster_ids', 'distances',
            'nearest_BASE_face_ids', 'diagnostic_ALL7_seed')})
    with (destination / 'per_face_selection.csv').open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result['per_face'][0]))
        writer.writeheader()
        writer.writerows(result['per_face'])
    for name, payload in (('components.json', result['components']), ('mixed_boundary.json', result['boundary']),
                          ('selection_summary.json', result['summary'])):
        with (destination / name).open('x', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    receipt = dict(status='SELECTOR_ONLY_COMPLETE', version=VERSION, request_id=request_id,
                   utc=datetime.now(timezone.utc).isoformat(), config=CONFIG,
                   source_identities=source_identities, model_calls=0, scorer_calls=0, GEO_calls=0,
                   files={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(destination.iterdir()) if p.is_file()})
    with (destination / 'complete.json').open('x', encoding='utf-8') as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return receipt
