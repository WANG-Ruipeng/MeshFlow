"""Finite triangle distances and original area sampler; MeshFlow local provenance in source_map."""
import numpy as np
import trimesh

def surface_distance_squared(points, triangles, chunk=32):
    """Trimesh's independent triangle closest-point implementation, not G helper."""
    points, triangles = np.asarray(points, dtype=np.float64), np.asarray(triangles, dtype=np.float64)
    if not np.isfinite(triangles).all() or not len(triangles):
        return np.full(len(points), np.inf)
    values = []
    for start in range(0, len(points), chunk):
        current = points[start:start+chunk]
        repeated_triangles = np.tile(triangles, (len(current), 1, 1))
        repeated_points = np.repeat(current, len(triangles), axis=0)
        closest = trimesh.triangles.closest_point(repeated_triangles, repeated_points)
        d2 = np.sum((closest-repeated_points)**2, axis=-1).reshape(len(current), len(triangles))
        # Trimesh can produce NaN on collapsed triangles. Evaluate the finite
        # boundary segments independently and use them for those entries.
        bad = ~np.isfinite(d2)
        if bad.any():
            for qi, fi in np.argwhere(bad):
                tri, p = triangles[fi], current[qi]
                candidates = []
                for i, j in ((0,1),(1,2),(2,0)):
                    edge = tri[j]-tri[i]
                    length = np.dot(edge, edge)
                    u = np.clip(np.dot(p-tri[i], edge)/length, 0, 1) if length > 0 else 0.
                    candidates.append(np.sum((p-(tri[i]+u*edge))**2))
                d2[qi,fi] = min(candidates)
        values.extend(np.min(d2, axis=1))
    return np.asarray(values)


def sample_mesh_surface(triangles, count, rng):
    tri = np.asarray(triangles, dtype=np.float64)
    areas = np.linalg.norm(np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]),axis=-1)
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        return None
    selected = tri[rng.choice(len(tri), count, p=areas/areas.sum())]
    uv = rng.rand(count,2); root = np.sqrt(uv[:,:1])
    return ((1-root)*selected[:,0] + root*(1-uv[:,1:])*selected[:,1] + root*uv[:,1:]*selected[:,2])


def raw_quality(tokens):
    tri = np.asarray(tokens, dtype=np.float64).reshape(-1,3,3)
    finite = np.isfinite(tri).all()
    result = dict(raw_faces=len(tri), nan_count=int(np.isnan(tri).sum()), inf_count=int(np.isinf(tri).sum()),
                  self_intersection_status="NOT_RUN", watertight_status="NOT_RUN", manifold_status="NOT_RUN")
    if not finite:
        result.update(degenerate_fraction=None, min_angle_lt5_fraction=None, angles_p05=None)
        return result
    edge_lengths = np.stack([np.linalg.norm(tri[:,(i+1)%3]-tri[:,i],axis=-1) for i in range(3)],axis=-1)
    area2 = np.linalg.norm(np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]),axis=-1)
    scale = max(np.linalg.norm(np.ptp(tri.reshape(-1,3),axis=0)), 1e-12)
    degenerate = area2 <= scale**2 * 1e-12
    angles = []
    for i in range(3):
        u, v = tri[:,(i+1)%3]-tri[:,i], tri[:,(i+2)%3]-tri[:,i]
        denominator = np.linalg.norm(u,axis=-1)*np.linalg.norm(v,axis=-1)
        cosine = np.divide((u*v).sum(-1), denominator, out=np.ones(len(tri)), where=denominator>0)
        angles.append(np.degrees(np.arccos(np.clip(cosine,-1,1))))
    angles = np.stack(angles,-1)
    result.update(degenerate_fraction=float(degenerate.mean()),
                  min_angle_lt5_fraction=float((angles.min(-1)<5).mean()),
                  angles_p05=float(np.quantile(angles,.05)))
    return result
