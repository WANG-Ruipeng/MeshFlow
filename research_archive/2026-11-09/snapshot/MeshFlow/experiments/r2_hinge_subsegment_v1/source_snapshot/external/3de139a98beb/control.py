"""Three preregistered CPU control metrics; no model or training execution.

Reference arrays and per-output arrays are returned to the caller for immutable
NPZ caching and overlays. The old verified CF classifier is reused unchanged.
All tolerances below are frozen before scoring model outputs.
"""
from __future__ import annotations
import copy
import hashlib
import itertools
from pathlib import Path
import time
import numpy as np

from .surface import surface_distance_squared
from .hashing import array_sha
from .seams import nearest_segments
from .pair_audit import pair_scan, PROPER, COPLANAR

VERSION = 'chair_boundary_attachment_control_metrics_v1r1'
TAUS = (.005, .01, .02)
THETA = 20.
RHO_CROSS = .005
RHO_DUP = .005
R_BORDER = .01
NORMAL_ANGLE = 15.
LOW_SUPPORT_N = 32
AREA_FLOOR = 1e-11  # Same normalized near-degeneracy floor as independent CF.
RAY_EPS = 1e-7
RAY_DET_REL = 1e-12
RAY_BARY_EPS = 1e-10
TANGENT_COS = .05
# Visibility: blocked, definitely visible, unresolved/coincident, excluded.
BLOCKED, VISIBLE, UNKNOWN, EXCLUDED = 0, 1, 2, 3


def directions(dense=False):
    base = sorted(v for v in itertools.product((-1, 0, 1), repeat=3) if v != (0, 0, 0))
    extra = set()
    if dense:
        for shape in ((2, 1, 0), (2, 1, 1)):
            for perm in set(itertools.permutations(shape)):
                for signs in itertools.product((-1, 1), repeat=3):
                    extra.add(tuple(x*s for x, s in zip(perm, signs)))
    integer = np.asarray(base + sorted(extra), dtype=np.float64)
    assert len(integer) == (74 if dense else 26)
    return integer / np.linalg.norm(integer, axis=1)[:, None]


def frozen_definition():
    return dict(version=VERSION, primary_names=['SeamCov', 'ConflictCov_C', 'ExposedRetain'],
        densities=dict(main=dict(C_points=2048, Gamma_points_per_edge=64, directions=26),
                       dense=dict(C_points=8192, Gamma_points_per_edge=256, directions=74)),
        C_seed="first 4 SHA256 bytes little-endian of 'CHAIR_BOUNDARY_ATTACHMENT_FACTORIAL_V1|20261007|C_AREA|'+condition_id+'|'+str(count)",
        C_sampling='MT19937 area-proportional face choice then sqrt barycentric; equal point weights; retain source C face index',
        dense_directions='Original26 then lexicographically sorted signed permutations of (2,1,0) and (2,1,1);26+24+24=74',
        main_direction_vectors=directions(False).tolist(), dense_direction_vectors=directions(True).tolist(),
        SeamCov=dict(taus=list(TAUS), theta_degrees=THETA, primary_tau=.01,
            gamma='Whole trusted GT interface only;64/256 uniform midpoints with length/count weights',
            projection='Query projection along its GT edge tangent lies inside the finite predicted endpoint interval;zero interval slack',
            CF='The edge owner free face against this Gamma edge owner C face only;old independently verified classifier;proper/coplanar-illegal/unresolved excluded',
            degeneracy_area_L0_squared=AREA_FLOOR, higher_is_better=True),
        ConflictCov_C=dict(rho_cross=RHO_CROSS, rho_dup=RHO_DUP, r_border=R_BORDER,
            normal_degrees=NORMAL_ANGLE, C_border='Exact C-only undirected coordinate-edge incidence=1;no welding',
            cross='Only proper CF intersection segments belonging to the sampled C face;strict finite-segment distance < rho_cross*L0',
            dup='Existence among all nondegenerate near-parallel free triangles;not normal of the globally nearest triangle',
            low_support_interior_samples_lt=LOW_SUPPORT_N,
            unknown='Lower bound from confirmed union;upper adds affected unresolved pairs/undefined normals;PARTIAL if unknown can change sampled coverage',
            reference='Same queries and rules on GT free;raw minus GT signed difference;do not assume GT zero'),
        ExposedRetain=dict(normalized_dtype='float64', direction_tangent_exclusion_abs_dot_lt=TANGENT_COS,
            origin='q+1e-7*d;positive intersections equivalently test unshifted ray t>1e-7',
            origin_self_hit_epsilon=RAY_EPS, relative_determinant_epsilon=RAY_DET_REL,
            barycentric_epsilon=RAY_BARY_EPS, ray_aabb_epsilon=RAY_EPS,
            two_sided=True, ignore='Own source C face only;other near-zero surface hits are not discarded',
            coincident='Near-zero finite free hits and potentially relevant coplanar/degenerate rays are UNKNOWN;face order never breaks ties',
            reference='At least one definitely visible direction in full C union GTfree;unknown reference eligibility separately reported',
            lower='At least one definitely reference-visible direction stays definitely visible',
            upper='On the same definite-reference denominator, at least one definitely reference-visible direction is visible or unresolved',
            global_bounds='Additional conservative ratio bounds allow reference-unknown eligibility/directions;invalid C query normals widen this auxiliary interval to [0,1];not an extra primary score',
            rejected_determinant='Any numerically rejected finite ray/triangle test with a forward ray-AABB possibility is UNKNOWN, without assuming the triangle is parallel',
            NA='No definitely reference-exposed samples;unknown denominator does not silently become zero'),
        limits='Tolerance coverage, numerical conflict bands, and fixed-direction exposure;not topology/watertightness/penetration volume',
        model_forwards=0, backward=0, optimizer_updates=0)


def source_identity():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}


def _tri(value):
    a = np.asarray(value)
    if a.ndim == 2 and a.shape[-1] == 9: a = a.reshape(-1, 3, 3)
    if a.ndim != 3 or a.shape[1:] != (3, 3): raise ValueError('Expected unpadded triangles [N,3,3]')
    if not np.isfinite(a).all(): raise ValueError('Nonfinite triangles')
    return a.astype(np.float64, copy=False)


def normals(triangles, L0):
    t = _tri(triangles); n = np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]); length = np.linalg.norm(n, axis=1)
    good = length*.5 > AREA_FLOOR*L0**2
    unit = np.divide(n, length[:, None], out=np.zeros_like(n), where=length[:, None]>0)
    return unit, good, .5*length


def open_edges(C):
    """C-only exact FP32-value geometry incidence, no target IDs or tolerance."""
    t = _tri(C); entries = {}
    for face, vertices in enumerate(t):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            first, second = tuple(vertices[a]), tuple(vertices[b])
            if first == second: raise ValueError('C contains a zero-length edge')
            key = tuple(sorted((first, second)))
            entries.setdefault(key, []).append((face, a, b))
    selected = [(key, owner[0]) for key, owner in sorted(entries.items()) if len(owner) == 1]
    edges = np.asarray([key for key, _ in selected], np.float64).reshape(-1, 2, 3)
    return edges, dict(open_edges=len(selected), internal_edges=sum(len(v)==2 for v in entries.values()),
                       ambiguous_edges=sum(len(v)>2 for v in entries.values()),
                       owners=[int(row[0]) for _, row in selected])


def sample_C(C, condition_id, count):
    t = _tri(C); _, _, a = normals(t, 1.)
    if not len(t) or not np.isfinite(a.sum()) or a.sum() <= 0: raise ValueError('C has no positive finite area')
    text = 'CHAIR_BOUNDARY_ATTACHMENT_FACTORIAL_V1|20261007|C_AREA|'+str(condition_id)+'|'+str(count)
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'little')
    rng = np.random.RandomState(seed); face = rng.choice(len(t), count, p=a/a.sum())
    uv = rng.rand(count, 2); root = np.sqrt(uv[:, :1])
    bary = np.concatenate((1-root, root*(1-uv[:, 1:]), root*uv[:, 1:]), axis=1)
    q = np.einsum('ni,nij->nj', bary, t[face])
    return q, face.astype(np.int64), np.full(count, 1./count), bary, seed


def _gamma(condition, count):
    edges = np.asarray(condition.get('gamma_edges', np.empty((0, 2, 3))), np.float64).reshape(-1, 2, 3)
    status = str(condition.get('gamma_status', 'PASS' if len(edges) else 'NOT_APPLICABLE'))
    if status not in ('PASS', 'READY', 'VALID', 'APPLICABLE') or not len(edges):
        return dict(status='NOT_APPLICABLE', reason='Whole Gamma unavailable/ambiguous', arrays={})
    owners = np.asarray(condition.get('gamma_known_face_indices', []), np.int64).reshape(-1)
    C = _tri(condition['C']); L0 = float(condition['L0'])
    if owners.shape != (len(edges),) or np.any(owners < 0) or np.any(owners >= len(C)):
        raise ValueError('Trusted Gamma requires one unambiguous local C owner per edge')
    lengths = np.linalg.norm(edges[:, 1]-edges[:, 0], axis=1)
    if not np.isfinite(edges).all() or np.any(lengths <= 0): raise ValueError('Invalid trusted Gamma geometry')
    for edge, owner in zip(edges, owners):
        vertices = C[owner]
        if any(not np.any(np.all(vertices == endpoint, axis=1)) for endpoint in edge):
            raise ValueError('Gamma endpoints are not exact corners of the registered C owner')
    u = (np.arange(count)+.5)/count
    points = (edges[:, :1]*(1-u[None, :, None])+edges[:, 1:]*u[None, :, None]).reshape(-1, 3)
    return dict(status='PASS', arrays=dict(gamma_edges=edges, gamma_known_face_indices=owners,
        gamma_points=points, gamma_weights=np.repeat(lengths/count, count),
        gamma_point_edge=np.repeat(np.arange(len(edges), dtype=np.int64), count)), total_length=float(lengths.sum()), L0=L0)


def trusted_cf(C, F, L0, cf_scan=None):
    scan = cf_scan if cf_scan is not None else pair_scan(C, F, L0, independent=True)
    if scan.get('tested_pairs') != len(C)*len(F) or scan.get('affected_subset') is not None:
        raise ValueError('CF must be the complete C by F pair scan')
    if not scan.get('independent_verification', False): raise ValueError('Require unchanged independently verified CF classification')
    # Codes: legal/disjoint=0, proper=1, coplanar illegal=2, unresolved=3.
    codes = np.zeros((len(C), len(F)), dtype=np.uint8)
    segments, pairs, failures = [], [], []
    for row in scan['contacts']:
        i, j = map(int, row['pair']); label = row['classification']; detector = row.get('detector', {})
        if label == PROPER:
            codes[i, j] = 1
            points = detector.get('intersection_points')
            if not points:
                points = detector.get('independent', {}).get('intersection_points')
            if not points:
                # Old cached event-only formats: ask the same verified classifier
                # for its already-defined FP64 geometric witness, never relabel it.
                from .intersections import verified_pair
                repeat = verified_pair(C[i], F[j], L0)
                if repeat.get('classification') == label: points = repeat.get('intersection_points')
            p = np.asarray(points if points is not None else [], np.float64).reshape(-1, 3)
            if len(p) >= 2:
                squared = ((p[:, None]-p[None])**2).sum(-1); a, b = np.unravel_index(np.argmax(squared), squared.shape)
                segment = p[[a, b]]
                valid = np.sqrt(squared[a, b]) > 1e-9*L0
                if valid:
                    witness = np.concatenate((segment, segment.mean(0, keepdims=True)))
                    valid = max(surface_distance_squared(witness, C[i:i+1]).max(), surface_distance_squared(witness, F[j:j+1]).max()) <= (1e-8*L0)**2
                if valid: segments.append(segment); pairs.append((i, j)); continue
            failures.append(dict(pair=[i, j], classification_retained=label, reason='Proper CF segment witness unavailable or inconsistent'))
        elif label == COPLANAR: codes[i, j] = 2
        elif 'unresolved' in label.lower() or 'degenerate' in label.lower() or detector.get('resolved') is False: codes[i, j] = 3
    return dict(scan=scan, codes=codes, proper_segments=np.asarray(segments, np.float64).reshape(-1, 2, 3),
                proper_segment_pairs=np.asarray(pairs, np.int64).reshape(-1, 2), extraction_failures=failures)


def seam_metrics(F, condition, ref, cf):
    if ref['audits']['gamma']['status'] != 'PASS':
        return dict(SeamCov=None, SeamCov_status='NOT_APPLICABLE', **{'SeamCov_'+str(t): None for t in TAUS}), {}
    a = ref['arrays']; edges = a['gamma_edges']; q = a['gamma_points']; w = a['gamma_weights']
    point_edge = a['gamma_point_edge']; owners = a['gamma_known_face_indices']; L0 = float(condition['L0'])
    endpoints = F[:, ((0, 1), (1, 2), (2, 0)), :].reshape(-1, 2, 3)
    face_owner = np.repeat(np.arange(len(F)), 3); delta = endpoints[:, 1]-endpoints[:, 0]
    length = np.linalg.norm(delta, axis=1); tg = np.divide(delta, length[:, None], out=np.zeros_like(delta), where=length[:, None]>0)
    _, good_faces, _ = normals(F, L0)
    covered = {tau: np.zeros(len(q), bool) for tau in TAUS}; possible = {tau: np.zeros(len(q), bool) for tau in TAUS}
    witness = np.full(len(q), -1, np.int64); nearest_qualified = np.full(len(q), np.inf)
    distance_only = {tau: np.zeros(len(q), bool) for tau in TAUS}
    for edge_idx, e in enumerate(edges):
        ids = np.flatnonzero(point_edge == edge_idx); te = (e[1]-e[0])/np.linalg.norm(e[1]-e[0])
        parallel = np.abs(tg @ te) >= np.cos(np.deg2rad(THETA))
        code = cf['codes'][owners[edge_idx], face_owner]
        allowed = parallel & (length>0) & good_faces[face_owner] & (code==0)
        uncertain = parallel & (length>0) & ((code==3) | ~good_faces[face_owner])
        projected = (endpoints-e[0]) @ te; low, high = projected.min(1), projected.max(1)
        for start in range(0, len(ids), 128):
            use = ids[start:start+128]; points = q[use]
            num = np.einsum('qei,ei->qe', points[:, None]-endpoints[None, :, 0], delta)
            u = np.divide(num, (length**2)[None], out=np.zeros_like(num), where=(length**2)[None]>0).clip(0, 1)
            d2 = ((points[:, None]-(endpoints[None, :, 0]+u[..., None]*delta[None]))**2).sum(-1)
            s = (points-e[0]) @ te; inside = (s[:, None]>=low[None]) & (s[:, None]<=high[None])
            eligible = inside & allowed[None]; maybe = inside & uncertain[None]
            distances = np.where(eligible, d2, np.inf); choose = distances.argmin(1)
            nearest_qualified[use] = np.sqrt(distances[np.arange(len(use)), choose])
            primary = distances[np.arange(len(use)), choose] < (.01*L0)**2
            witness[use[primary]] = choose[primary]
            for tau in TAUS:
                close = d2 < (tau*L0)**2
                covered[tau][use] = (eligible & close).any(1)
                possible[tau][use] = ((eligible | maybe) & close).any(1)
                distance_only[tau][use] = close.any(1)
    result = {}
    for tau in TAUS:
        result['SeamCov_'+str(tau)] = float(np.dot(w, covered[tau])/w.sum())
        result['SeamCov_upper_'+str(tau)] = float(np.dot(w, possible[tau])/w.sum())
        result['Gamma'+str(ref['audits']['gamma_samples_per_edge'])+'_distance_only_AttachCov_'+str(tau)] = float(np.dot(w, distance_only[tau])/w.sum())
    uncertain_points = possible[.01] & ~covered[.01]
    result.update(SeamCov=result['SeamCov_0.01'], SeamCov_upper=result['SeamCov_upper_0.01'],
        SeamCov_status='PARTIAL' if uncertain_points.any() else 'PASS',
        SeamCov_uncertain_weight_fraction=float(np.dot(w, uncertain_points)/w.sum()),
        SeamCov_gamma_points=len(q), SeamCov_gamma_total_length=float(w.sum()))
    vec = dict(seam_witness_free_edge=witness, seam_free_edges=endpoints, seam_free_edge_owner=face_owner,
               seam_nearest_qualified_distance=nearest_qualified, seam_uncertain=uncertain_points)
    for tau in TAUS: vec['seam_covered_'+str(tau)] = covered[tau]
    return result, vec


def conflict_metrics(F, condition, ref, cf):
    C = _tri(condition['C']); L0 = float(condition['L0']); a = ref['arrays']
    q, owner, weights = a['C_points'], a['C_point_faces'], a['C_weights']
    nc, good_C, _ = normals(C, L0); nf, good_F, _ = normals(F, L0)
    interior = a['C_border_distance'] >= R_BORDER*L0
    cross, dup, uncertain = np.zeros(len(q), bool), np.zeros(len(q), bool), np.zeros(len(q), bool)
    missing = {tuple(r['pair']) for r in cf['extraction_failures']}
    for i in range(len(C)):
        ids = np.flatnonzero(owner==i)
        if not len(ids): continue
        segment_ids = np.flatnonzero(cf['proper_segment_pairs'][:, 0]==i)
        if len(segment_ids):
            d2 = nearest_segments(q[ids], cf['proper_segments'][segment_ids])[0]
            cross[ids] = d2 < (RHO_CROSS*L0)**2
        aligned = good_F & (np.abs(nf @ nc[i]) >= np.cos(np.deg2rad(NORMAL_ANGLE)))
        inner = ids[interior[ids]]
        if len(inner) and good_C[i] and aligned.any():
            # Filter ALL qualifying normals first, then minimize finite-surface
            # distance: a nearer nonparallel face cannot hide a duplicate face.
            dup[inner] = surface_distance_squared(q[inner], F[aligned]) < (RHO_DUP*L0)**2
        unknown_faces = np.flatnonzero(cf['codes'][i]==3)
        unknown_faces = np.unique(np.r_[unknown_faces, [j for ci, j in missing if ci==i]]).astype(np.int64)
        if len(unknown_faces):
            uncertain[ids] |= surface_distance_squared(q[ids], F[unknown_faces]) < (max(RHO_CROSS, RHO_DUP)*L0)**2
        if not good_C[i] and len(F):
            uncertain[inner] |= surface_distance_squared(q[inner], F) < (RHO_DUP*L0)**2
    union = cross | dup; unknown_only = uncertain & ~union
    weighted = lambda mask: float(np.dot(weights, mask)/weights.sum())
    status = 'PARTIAL' if unknown_only.any() or cf['extraction_failures'] else 'LOW_SUPPORT' if interior.sum()<LOW_SUPPORT_N else 'PASS'
    result = dict(ConflictCov_C=weighted(union), ConflictCov_C_upper=weighted(union | uncertain),
        ConflictCov_C_cross=weighted(cross), ConflictCov_C_dup=weighted(dup), ConflictCov_C_cross_and_dup=weighted(cross & dup),
        ConflictCov_C_status=status, ConflictCov_C_interior_samples=int(interior.sum()),
        ConflictCov_C_interior_weight_fraction=weighted(interior), ConflictCov_C_uncertain_fraction=weighted(unknown_only),
        ConflictCov_C_unresolved_pairs=int(np.count_nonzero(cf['codes']==3)),
        ConflictCov_C_coplanar_illegal_pairs=int(np.count_nonzero(cf['codes']==2)),
        ConflictCov_C_proper_pairs=int(np.count_nonzero(cf['codes']==1)),
        ConflictCov_C_segment_extraction_failures=len(cf['extraction_failures']))
    return result, dict(conflict_cross=cross, conflict_dup=dup, conflict_union=union, conflict_unknown=unknown_only,
        conflict_C_interior=interior, proper_CF_segments=cf['proper_segments'], proper_CF_segment_pairs=cf['proper_segment_pairs'])


def _ray_aabb(q, d, lower, upper):
    lo = np.full((len(q), len(lower)), -np.inf); hi = np.full_like(lo, np.inf)
    valid = np.ones_like(lo, dtype=bool)
    for axis in range(3):
        if abs(d[axis]) < 1e-15:
            valid &= (q[:, None, axis]>=lower[None, :, axis]-RAY_EPS) & (q[:, None, axis]<=upper[None, :, axis]+RAY_EPS)
        else:
            a = (lower[None, :, axis]-RAY_EPS-q[:, None, axis])/d[axis]
            b = (upper[None, :, axis]+RAY_EPS-q[:, None, axis])/d[axis]
            lo = np.maximum(lo, np.minimum(a, b)); hi = np.minimum(hi, np.maximum(a, b))
    return valid & (hi >= np.maximum(lo, RAY_EPS))


def ray_visibility(points, point_normals, triangles, direction_vectors, *, own_faces=None, point_normal_valid=None):
    """FP64 normalized-coordinate finite triangle rays, with uncertainty bounds.

    Own C faces are ignored only when explicitly supplied. The free geometry
    never receives self-hit exemptions. Positive blockers dominate uncertainty.
    """
    q = np.asarray(points, np.float64); t = _tri(triangles); dirs = np.asarray(direction_vectors, np.float64)
    valid_norm = np.ones(len(q), bool) if point_normal_valid is None else np.asarray(point_normal_valid, bool)
    eligible = (np.abs(np.asarray(point_normals) @ dirs.T)>=TANGENT_COS) & valid_norm[:, None]
    result = np.where(eligible, VISIBLE, EXCLUDED).astype(np.uint8)
    if not len(t): return result
    e1, e2 = t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]
    cross_n = np.cross(e1, e2); nlength = np.linalg.norm(cross_n, axis=1)
    valid_tri = nlength*.5 > AREA_FLOOR
    relative_scale = np.linalg.norm(e1, axis=1)*np.linalg.norm(e2, axis=1)
    bounds_lo, bounds_hi = t.min(1), t.max(1)
    for di, d in enumerate(dirs):
        pvec = np.cross(np.broadcast_to(d, e2.shape), e2)
        det = np.einsum('ti,ti->t', e1, pvec)
        regular = valid_tri & (np.abs(det)>RAY_DET_REL*relative_scale)
        inverse = np.divide(1., det, out=np.zeros_like(det), where=regular)
        relevant = np.flatnonzero(eligible[:, di])
        for start in range(0, len(relevant), 128):
            ids = relevant[start:start+128]; x = q[ids]; tv = x[:, None]-t[None, :, 0]
            u = np.einsum('qti,ti->qt', tv, pvec)*inverse[None]
            qvec = np.cross(tv, e1[None])
            v = np.einsum('qti,i->qt', qvec, d)*inverse[None]
            distance = np.einsum('qti,ti->qt', qvec, e2)*inverse[None]
            inside = (u>=-RAY_BARY_EPS) & (v>=-RAY_BARY_EPS) & (u+v<=1+RAY_BARY_EPS) & regular[None]
            strict_inside = (u>RAY_BARY_EPS) & (v>RAY_BARY_EPS) & (u+v<1-RAY_BARY_EPS)
            ignore = np.zeros_like(inside)
            if own_faces is not None: ignore[np.arange(len(ids)), np.asarray(own_faces)[ids]] = True
            inside &= ~ignore
            positive = inside & (distance>RAY_EPS)
            blocker = (positive & strict_inside).any(1)
            near = inside & (np.abs(distance)<=RAY_EPS)
            ambiguous = near.any(1) | (positive & ~strict_inside).any(1)
            parallel_or_bad = ~regular
            if parallel_or_bad.any():
                # A rejected determinant also occurs for long, skinny triangles;
                # it does not establish parallel planes or an unobstructed ray.
                # Keep every forward AABB possibility unresolved conservatively.
                potential = _ray_aabb(x, d, bounds_lo, bounds_hi) & parallel_or_bad[None] & ~ignore
                ambiguous |= potential.any(1)
            result[ids, di] = np.where(blocker, BLOCKED, np.where(ambiguous, UNKNOWN, VISIBLE))
    return result


def _combine_visibility(first, second):
    excluded = (first==EXCLUDED) | (second==EXCLUDED)
    blocked = (first==BLOCKED) | (second==BLOCKED)
    uncertain = (first==UNKNOWN) | (second==UNKNOWN)
    return np.where(excluded, EXCLUDED, np.where(blocked, BLOCKED, np.where(uncertain, UNKNOWN, VISIBLE))).astype(np.uint8)


def exposure_metrics(F, condition, ref):
    a = ref['arrays']; L0 = float(condition['L0']); q = a['C_points']/L0
    fstate = ray_visibility(q, a['C_point_normals'], F/L0, a['directions'], point_normal_valid=a['C_point_normal_valid'])
    state = _combine_visibility(a['C_visibility'], fstate); gt = a['GT_visibility']
    reference = gt==VISIBLE; reference_possible = reference | (gt==UNKNOWN)
    eligible = reference.any(1); eligible_upper = reference_possible.any(1)
    low = (reference & (state==VISIBLE)).any(1)
    high = (reference & ((state==VISIBLE) | (state==UNKNOWN))).any(1)
    possible_all = (reference_possible & ((state==VISIBLE) | (state==UNKNOWN))).any(1)
    w = a['C_weights']; denominator = float(w[eligible].sum()); possible_denominator = float(w[eligible_upper].sum())
    total = float(w.sum()); uncertain_reference = (gt==UNKNOWN).any(1)
    # Unknown query normals are explicitly outside a resolved reference domain.
    normal_unknown = ~a['C_point_normal_valid']
    if denominator == 0:
        result = dict(ExposedRetain=None, ExposedRetain_upper=None, ExposedRetain_status='NA_NO_REFERENCE_EXPOSED_POINTS')
    else:
        uncertain = (high & ~low & eligible)
        result = dict(ExposedRetain=float(w[low & eligible].sum()/denominator),
            ExposedRetain_upper=float(w[high & eligible].sum()/denominator),
            ExposedRetain_status='PARTIAL' if uncertain.any() or uncertain_reference.any() or normal_unknown.any() else 'PASS')
    result.update(reference_exposed_fraction=denominator/total, reference_possible_exposed_fraction=possible_denominator/total,
        reference_exposed_points=int(eligible.sum()), reference_visibility_uncertain_fraction=float(w[uncertain_reference | normal_unknown].sum()/total),
        ExposedRetain_uncertain_fraction=float(w[(high & ~low) & eligible].sum()/denominator) if denominator else None,
        ExposedRetain_global_lower=float(w[low].sum()/possible_denominator) if possible_denominator else None,
        ExposedRetain_global_upper=min(1., float(w[possible_all].sum()/denominator)) if denominator else (1. if possible_denominator else None),
        ExposedRetain_directions=len(a['directions']), ExposedRetain_C_points=len(q))
    if normal_unknown.any():
        # Invalid query normals leave both direction eligibility and GT-domain
        # membership unresolved. Do not certify a narrow global interval from
        # the remaining points; the primary definite-reference value is kept.
        result.update(ExposedRetain_global_lower=0., ExposedRetain_global_upper=1.)
    return result, dict(output_visibility=state, free_visibility=fstate, exposure_reference_exposed=eligible,
        exposure_retained_lower=low & eligible, exposure_retained_upper=high & eligible,
        exposure_occluded=eligible & ~high, exposure_uncertain=(high & ~low) & eligible,
        exposure_reference_uncertain=uncertain_reference | normal_unknown)


def prepare_reference(condition, condition_id, *, density='main'):
    if density not in ('main', 'dense'): raise ValueError('Only frozen main/dense scoring densities')
    C, target = _tri(condition['C']), _tri(condition['GT_free']); L0 = float(condition['L0'])
    if not np.isfinite(L0) or L0 <= 0: raise ValueError('Positive original-parent L0 required')
    count, gamma_count = (2048, 64) if density=='main' else (8192, 256)
    q, face, weight, bary, seed = sample_C(C, condition_id, count)
    cn, good, _ = normals(C, L0); border, border_audit = open_edges(C)
    dborder = np.sqrt(nearest_segments(q, border)[0]) if len(border) else np.full(count, np.inf)
    gamma = _gamma(condition, gamma_count); dirs = directions(density=='dense')
    a = dict(C_points=q, C_point_faces=face, C_weights=weight, C_barycentric=bary,
             C_point_normals=cn[face], C_point_normal_valid=good[face],
             C_open_edges=border, C_border_distance=dborder, directions=dirs, **gamma['arrays'])
    a['C_visibility'] = ray_visibility(q/L0, cn[face], C/L0, dirs, own_faces=face, point_normal_valid=good[face])
    a['GT_free_visibility'] = ray_visibility(q/L0, cn[face], target/L0, dirs, point_normal_valid=good[face])
    a['GT_visibility'] = _combine_visibility(a['C_visibility'], a['GT_free_visibility'])
    ref = dict(identity=dict(version=VERSION, condition_id=str(condition_id), density=density, C_sha256=array_sha(np.asarray(condition['C'])),
        GT_free_sha256=array_sha(np.asarray(condition['GT_free'])), L0=L0, C_seed=seed), arrays=a,
        audits=dict(gamma={k: v for k, v in gamma.items() if k!='arrays'}, C_boundary=border_audit,
                    sample_count=count, gamma_samples_per_edge=gamma_count, direction_count=len(dirs),
                    source_identity=source_identity(), model_forwards=0))
    gtcf = trusted_cf(C, target, L0)
    metrics, gt_vectors = conflict_metrics(target, condition, ref, gtcf)
    ref['reference_metrics'] = metrics
    ref['audits']['GT_CF'] = dict(status=gtcf['scan']['status'], classification_counts=gtcf['scan']['classification_counts'], extraction_failures=gtcf['extraction_failures'])
    for key, value in gt_vectors.items(): a['GT_'+key] = value
    ref['array_sha256'] = {key: array_sha(value) for key, value in a.items()}
    return ref


def score_control(free, condition, reference, *, cf_scan=None):
    tick = time.perf_counter(); C, F = _tri(condition['C']), _tri(free); L0 = float(condition['L0'])
    if not len(F) or normals(F, L0)[2].sum() <= 0:
        raise ValueError('No positive-area free surface: generation failure, not zero conflicts')
    identity = reference['identity']
    if identity['C_sha256'] != array_sha(np.asarray(condition['C'])) or identity['GT_free_sha256'] != array_sha(np.asarray(condition['GT_free'])) or identity['L0'] != L0:
        raise ValueError('Frozen reference belongs to a different condition')
    for key, expected in reference.get('array_sha256', {}).items():
        if array_sha(reference['arrays'][key]) != expected: raise ValueError('Frozen reference array changed: '+key)
    cf = trusted_cf(C, F, L0, cf_scan); metrics, vectors = {}, {}
    for function in (seam_metrics, conflict_metrics):
        m, a = function(F, condition, reference, cf); metrics.update(m); vectors.update(a)
    m, a = exposure_metrics(F, condition, reference); metrics.update(m); vectors.update(a)
    gm = reference['reference_metrics']
    metrics.update(ConflictCov_C_GT=gm['ConflictCov_C'], ConflictCov_C_GT_upper=gm['ConflictCov_C_upper'],
        ConflictCov_C_GT_status=gm['ConflictCov_C_status'],
        ConflictCov_C_minus_GT=metrics['ConflictCov_C']-gm['ConflictCov_C'],
        ConflictCov_C_minus_GT_lower=metrics['ConflictCov_C']-gm['ConflictCov_C_upper'],
        ConflictCov_C_minus_GT_upper=metrics['ConflictCov_C_upper']-gm['ConflictCov_C'])
    statuses = {name: metrics[name+'_status'] for name in ('SeamCov', 'ConflictCov_C', 'ExposedRetain')}
    return dict(metrics=metrics, statuses=statuses, arrays=vectors,
        audit=dict(version=VERSION, density=identity['density'], reference_identity=copy.deepcopy(identity),
            free_sha256=array_sha(np.asarray(free)), CF_classification_counts=cf['scan']['classification_counts'],
            proper_segment_extraction_failures=cf['extraction_failures'], seconds=time.perf_counter()-tick,
            copied_C_in_free=False, new_model_calls=0, backwards=0, optimizer_updates=0, geometry_modified=False))
