"""C-only geometric screening proxy, not an adoption/quality classifier."""
from __future__ import annotations
import collections, math
import numpy as np

SPEC = dict(version='bestof_C_only_geometry_rank_v1',
    terms=['CF_illegal_affected_free_fraction', 'FF_illegal_affected_free_fraction', 'one_minus_boundary_cover_when_applicable'],
    combine='arithmetic_mean_of_applicable_terms', samples_per_C_open_edge=8,
    cover_threshold=0.01, cover_scale='C_only_bbox_diagonal', distance='finite_non_degenerate_F_triangles',
    cover_weighting='source_C_open_edge_length', closed_C='NA_CLOSED_C_NO_OPEN_BOUNDARY; omit cover term',
    I_policy='report_only_with_original_status; never used for selection or NA imputation',
    tie_break='lexicographic_noise_npz_SHA256',
    qualification='finite_Nx3x3; exact_original_C; correct_N_K; at_least_one_valid_free_triangle',
    valid_free_area='area > 1e-12 * C_bbox_diagonal_squared',
    forbidden_inputs=['Y', 'source_full_mesh', 'source_reconstruction', 'visual_labels', 'POST_scores'],
    interpretation='geometric_proxy_only; proximity_does_not_certify_adoption_or_topology')


def boundary_samples(vertices, faces):
    vertices = np.asarray(vertices, np.float64)
    faces = np.asarray(faces, np.int64)
    counts = collections.Counter(tuple(sorted((int(a), int(b))))
                                 for f in faces for a, b in zip(f, np.roll(f, -1)))
    edges = sorted(e for e, count in counts.items() if count == 1)
    points, weights = [], []
    for a, b in edges:
        va, vb = vertices[[a, b]]
        length = float(np.linalg.norm(vb - va))
        if length <= 0:
            raise ValueError('Zero-length C source boundary')
        for i in range(SPEC['samples_per_C_open_edge']):
            t = (i + .5) / SPEC['samples_per_C_open_edge']
            points.append((1 - t) * va + t * vb)
            weights.append(length / SPEC['samples_per_C_open_edge'])
    return np.asarray(points).reshape(-1, 3), np.asarray(weights), edges


def nearest(points, triangles):
    # Existing finite-triangle projection; inputs contain C queries and F only.
    from r2.tools.cref_layer_audit_v1.geometry import finite_distances
    values = []
    for start in range(0, len(points), 64):
        distance, _, _ = finite_distances(points[start:start + 64], triangles)
        values.extend(np.sqrt(np.min(distance, axis=1)).tolist())
    return np.asarray(values, np.float64)


def measure(raw, C_vertices, C_faces_original, N, K, cf, ff, noise_sha256):
    raw = np.asarray(raw)
    original = np.asarray(C_vertices)[np.asarray(C_faces_original)]
    if raw.shape != (N, 3, 3) or not np.isfinite(raw).all() or not np.array_equal(raw[:K], original):
        return dict(eligible=False, status='INVALID_RAW_OR_C', rank=None, noise_sha256=noise_sha256)
    scale = float(np.linalg.norm(np.ptp(original.reshape(-1, 3).astype(np.float64), axis=0)))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid C-only scale')
    free = raw[K:].astype(np.float64)
    area = .5 * np.linalg.norm(np.cross(free[:, 1] - free[:, 0], free[:, 2] - free[:, 0]), axis=1)
    valid = area > 1e-12 * scale * scale
    if not valid.any():
        return dict(eligible=False, status='NO_VALID_FREE_TRIANGLE', rank=None, noise_sha256=noise_sha256)
    for value in (cf, ff):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Missing/invalid original collision fraction')
    points, weights, edges = boundary_samples(C_vertices, C_faces_original)
    terms = [float(cf), float(ff)]
    cover, mean_distance = None, None
    if len(points):
        distances = nearest(points, free[valid])
        cover = float(weights[distances <= .01 * scale].sum() / weights.sum())
        mean_distance = float(np.dot(distances, weights) / weights.sum())
        terms.append(1 - cover)
    return dict(eligible=True, status='PASS', rank=sum(terms) / len(terms), noise_sha256=noise_sha256,
        CF_illegal_affected_free_fraction=float(cf), FF_illegal_affected_free_fraction=float(ff),
        boundary_cover=cover, boundary_cover_status='PASS' if len(points) else 'NA_CLOSED_C_NO_OPEN_BOUNDARY',
        boundary_edges=len(edges), boundary_query_count=len(points), boundary_length=float(weights.sum()),
        boundary_distance_mean_model=mean_distance, C_bbox_diagonal=scale, threshold_model=.01 * scale,
        rank_term_count=len(terms), valid_free_count=int(valid.sum()), free_count=len(free),
        selector_degenerate_fraction=float((~valid).mean()), free_area=float(area.sum()),
        source_GT_read=False, visual_labels_read=False)


def choose(rows):
    eligible = [row for row in rows if row['eligible']]
    return min(eligible, key=lambda r: (r['rank'], r['noise_sha256'])) if eligible else None
