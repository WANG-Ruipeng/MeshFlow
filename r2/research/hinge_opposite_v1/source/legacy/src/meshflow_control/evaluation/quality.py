import hashlib
import numpy as np
from .surface import sample_mesh_surface,surface_distance_squared
from .seams import nearest_segments
from .shape import shape_quality
TAUS=(.005,.01,.02,.05)
ATTACH_TAUS=(.001,.003,.005,.01,.02,.05)

def tri(value):
    a = np.asarray(value)
    return a.reshape(-1, 3, 3)


def seed(*parts):
    return int.from_bytes(hashlib.sha256(('CHAIR_CONTROL_USABILITY_QUICK_PILOT_V1|' + '|'.join(map(str, parts))).encode()).digest()[:4], 'little')


def area(t):
    t = tri(t).astype(np.float64)
    return .5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=-1)


def tkey(value):
    return format(float(value), '.3f').rstrip('0').rstrip('.')


def pr_metrics(d2_yf, d2_fy, L0, count=2048):
    d2_yf, d2_fy = np.asarray(d2_yf), np.asarray(d2_fy)
    out = {f'GT_to_F_MSE{count}_L0': float(d2_yf.mean() / L0**2),
           f'F_to_GT_MSE{count}_L0': float(d2_fy.mean() / L0**2),
           f'D_sym{count}': float((d2_yf.mean() + d2_fy.mean()) / (2 * L0**2))}
    for tau in TAUS:
        p, r = float(np.mean(d2_fy < (tau * L0)**2)), float(np.mean(d2_yf < (tau * L0)**2))
        key = tkey(tau)
        out.update({f'Precision{count}_{key}': p, f'Recall{count}_{key}': r,
                    f'Fscore{count}_{key}': 2*p*r/(p+r) if p+r else 0.})
    return out


def weighted_stats(d2, weights, L0):
    d2, w = np.asarray(d2), np.asarray(weights)
    order = np.argsort(d2, kind='stable'); cumulative = np.cumsum(w[order])
    index = min(np.searchsorted(cumulative, .95*cumulative[-1], side='left'), len(order)-1)
    return dict(RMS_L0=float(np.sqrt(np.dot(w, d2)/w.sum())/L0),
                p95_L0=float(np.sqrt(d2[order[index]])/L0))


def gamma_points(edges, count):
    edges = np.asarray(edges, np.float64).reshape(-1, 2, 3)
    lengths = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=-1)
    if not len(edges) or not np.isfinite(edges).all() or np.any(lengths <= 0): return None, None
    u = (np.arange(count, dtype=np.float64) + .5) / count
    q = (edges[:, :1] * (1-u[None, :, None]) + edges[:, 1:] * u[None, :, None]).reshape(-1, 3)
    return q, np.repeat(lengths / count, count)


def interface_metrics(free, edges, L0, count=32):
    q, w = gamma_points(edges, count)
    if q is None: return {f'Gamma{count}_status': 'NOT_APPLICABLE'}, {}
    segments = np.asarray(free)[:, ((0, 1), (1, 2), (2, 0)), :].reshape(-1, 2, 3)
    # Bounded chunks prevent all Gamma points x all free edges allocation.
    de = np.concatenate([nearest_segments(q[i:i+256], segments)[0] for i in range(0, len(q), 256)])
    ds = surface_distance_squared(q, free)
    if np.any(ds > de + 1e-10*L0**2): raise AssertionError('Surface distance exceeded finite-edge distance')
    out = {f'Gamma{count}_status': 'PASS', f'Gamma{count}_points': len(q), f'Gamma{count}_length': float(w.sum())}
    for name, distance in (('edge', de), ('surface', ds)):
        out.update({f'Gamma{count}_{name}_{k}': v for k, v in weighted_stats(distance, w, L0).items()})
        for tau in ATTACH_TAUS:
            out[f'Gamma{count}_{name}_AttachCov_{tkey(tau)}'] = float(w[distance < (tau*L0)**2].sum()/w.sum())
    return out, dict(points=q, weights=w, edge_d2=de, surface_d2=ds)


def angle_reference(target, L0):
    q = shape_quality(target, L0)
    a = tri(target).astype(np.float64)/L0; angles = []
    for i in range(3):
        u, v = a[:, (i+1)%3]-a[:, i], a[:, (i+2)%3]-a[:, i]
        den = np.linalg.norm(u, axis=-1)*np.linalg.norm(v, axis=-1)
        cosine = np.divide((u*v).sum(-1), den, out=np.ones(len(a)), where=den>0)
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    values = np.stack(angles, axis=-1)
    return dict(GT_free_minimum_angle_lt5_fraction=q['min_angle_lt5_fraction'],
                GT_free_angle_p05_degrees=q['angle_p05_degrees'],
                GT_free_exact_zero_area_count=q['exact_zero_area_count'],
                GT_free_near_degenerate_count=q['near_degenerate_area_count'],
                GT_free_angle_p50_degrees=float(np.quantile(values, .5)),
                GT_free_angle_p95_degrees=float(np.quantile(values, .95)),
                GT_free_area=float(q['total_area']))
