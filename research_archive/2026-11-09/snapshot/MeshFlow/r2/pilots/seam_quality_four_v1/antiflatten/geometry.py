"""Finite-C probe risk, not an I metric or an if-and-only-if collision test."""
from __future__ import annotations
import math
import numpy as np

L = math.sqrt(3) * 2 / .3762
TAU = .02 * L
COS = math.cos(math.radians(15))
BARY_EPS = 1e-6
DIST_EPS = 1e-9 * L


def normal(tri):
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    length = float(np.linalg.norm(n))
    return None if length <= 1e-12 * L * L else n / length


def segment_nearest(p, a, b):
    d = b - a
    denominator = float(d @ d)
    if denominator == 0:
        return a.copy()
    return a + np.clip(float((p - a) @ d) / denominator, 0., 1.) * d


def finite_projection(p, tri):
    """Exact triangle distance plus orthogonal-plane projection barycentrics."""
    n = normal(tri)
    if n is None:
        return None
    h = float((p - tri[0]) @ n)
    projection = p - h * n
    e0, e1 = tri[1] - tri[0], tri[2] - tri[0]
    q = projection - tri[0]
    aa, ab, bb = float(e0 @ e0), float(e0 @ e1), float(e1 @ e1)
    denom = aa * bb - ab * ab
    if denom <= 0:
        return None
    u = (float(q @ e0) * bb - float(q @ e1) * ab) / denom
    v = (float(q @ e1) * aa - float(q @ e0) * ab) / denom
    bary = np.array([1 - u - v, u, v])
    in_closed = bool(np.min(bary) >= -BARY_EPS)
    inside = bool(np.min(bary) > BARY_EPS)
    candidates = [segment_nearest(p, tri[i], tri[j]) for i, j in ((0, 1), (1, 2), (2, 0))]
    if in_closed:
        candidates.append(projection)
    closest = min(candidates, key=lambda x: float(np.sum((p - x) ** 2)))
    return dict(distance=float(np.linalg.norm(p - closest)), height=h,
                projection=projection, closest=closest, bary=bary,
                interior=inside, normal=n)


def probes(c0, c1, tip, owner, c_triangles, fixed_faces=None):
    c0, c1, tip, owner, c_triangles = map(lambda x: np.asarray(x, np.float64),
                                       (c0, c1, tip, owner, c_triangles))
    sn, cn = normal(np.array([c1, c0, tip])), normal(owner)
    edge = c1 - c0
    if sn is None or cn is None or np.linalg.norm(edge) <= DIST_EPS:
        return [dict(probe=name, status="NA_DEGENERATE", risk=0., face_id=None)
                for name in ("CENTROID", "TIP")]
    inward = float((tip - (c0 + c1) / 2) @ np.cross(cn, edge / np.linalg.norm(edge)))
    rows = []
    for pi, (name, p) in enumerate((("CENTROID", (c0 + c1 + tip) / 3), ("TIP", tip))):
        selected = range(len(c_triangles)) if fixed_faces is None else [fixed_faces[pi]]
        choices = []
        for face_id in selected:
            if face_id is None:
                continue
            result = finite_projection(p, c_triangles[face_id])
            if result is not None:
                choices.append((result['distance'], int(face_id), result))
        if not choices:
            rows.append(dict(probe=name, status="NA_NO_VALID_C_FACE", risk=0., face_id=None))
            continue
        distance, fi, r = min(choices, key=lambda x: (x[0], x[1]))
        parallel = abs(float(sn @ r['normal']))
        on_base = float(np.linalg.norm(r['projection'] - segment_nearest(r['projection'], c0, c1))) <= DIST_EPS
        active = bool(r['interior'] and not on_base and inward > 0 and parallel > COS and distance < TAU)
        rows.append(dict(probe=name, status="PASS", face_id=fi, distance_model=distance,
                         distance_fraction_L=distance / L, signed_height_model=r['height'],
                         projection_xyz=r['projection'].tolist(), barycentric=r['bary'].tolist(),
                         interior_projection=r['interior'], projection_on_shared_base=on_base,
                         inward_model=inward, parallel_abs_cosine=parallel,
                         active=active, risk=(1 - distance / TAU) ** 2 if active else 0.,
                         face_normal=r['normal'].tolist()))
    return rows


def restore_raw_normal(c0, c1, raw_tip, pre_tip, owner, c_triangles):
    """One proposal; fixed PRE-selected face, no tangential compensation."""
    raw_tip, pre_tip = np.asarray(raw_tip, np.float64), np.asarray(pre_tip, np.float64)
    pre = probes(c0, c1, pre_tip, owner, c_triangles)
    faces = [x['face_id'] for x in pre]
    raw = probes(c0, c1, raw_tip, owner, c_triangles, faces)
    choices = []
    for index, (a, b) in enumerate(zip(raw, pre)):
        if b['status'] != 'PASS' or a['status'] != 'PASS':
            continue
        n = np.asarray(b['face_normal'])
        p = np.asarray(c_triangles[b['face_id']][0], np.float64)
        h_raw, h_pre = float((raw_tip - p) @ n), float((pre_tip - p) @ n)
        if b['risk'] > a['risk'] + 1e-12 and abs(h_pre) < abs(h_raw) - DIST_EPS:
            choices.append((-(b['risk'] - a['risk']), b['distance_model'], b['face_id'], index, h_raw, h_pre, n))
    if not choices:
        return np.asarray(pre_tip, np.float32), dict(moved=False, reason='NO_INCREASED_RISK_AND_NORMAL_COMPRESSION',
                                                   raw_probes=raw, pre_probes=pre)
    _, _, face_id, pi, h_raw, h_pre, n = min(choices, key=lambda x: x[:4])
    target = (pre_tip + (h_raw - h_pre) * n).astype(np.float32)
    delta = target.astype(np.float64) - pre_tip
    return target, dict(moved=bool(np.any(delta != 0)), reason='REJECT_D0_NORMAL_COMPRESSION',
                       selected_probe=pre[pi]['probe'], selected_C_face=face_id,
                       raw_height_model=h_raw, pre_height_model=h_pre,
                       target_height_model=float((target.astype(float) - c_triangles[face_id][0]) @ n),
                       displacement_model=float(np.linalg.norm(delta)), displacement_fraction_L=float(np.linalg.norm(delta) / L),
                       tangent_displacement_residual=float(np.linalg.norm(delta - (delta @ n) * n)),
                       raw_probes=raw, pre_probes=pre, final_probes=probes(c0, c1, target, owner, c_triangles, faces))


def selftest():
    c = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    inward = probes(c[0], c[1], [.25, .25, -.001], c, c[None])
    assert any(r['active'] for r in inward), 'Owner C must not be removed from risk set'
    vertical = probes(c[0], c[1], [.5, 0, -1], c, c[None])
    assert not any(r.get('active', False) for r in vertical)
    outward = probes(c[0], c[1], [.5, -1, 0], c, c[None])
    assert not any(r.get('active', False) for r in outward)
    target, result = restore_raw_normal(c[0], c[1], [.25, .25, -.1], [.3, .25, -.001], c, c[None])
    assert result['moved'] and np.allclose(target, [.3, .25, -.1])
    # Point on the shared segment is excluded, finite-face exclusion is not.
    edge = probes(c[0], c[1], [.5, 0, -.001], c, c[None])
    assert not any(r.get('active', False) for r in edge)
    return dict(status='PASS',cases=5,model_calls=0)
