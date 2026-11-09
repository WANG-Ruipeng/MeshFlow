"""C-only layout plus supervised, role-restricted OT for the four-C prototype.

The inference functions cannot read source Y or occupancy labels. Supervised
packing is a separate function. Null slots become ordinary bulk faces; N stays
fixed. Multi-edge owner/third-C-reference cases are explicitly unsupported here.
"""
from collections import defaultdict
import numpy as np
from meshflow_control.training.coupling import nested_ot_map, to_noise_slots

L = float(np.sqrt(3) * 2 / .3762)
FEATURE_DIM = 24
FEATURE_SCHEMA = 'C_edge24_world_chair_v1'

def boundary_from_C(vertices, faces):
    v, f = np.asarray(vertices), np.asarray(faces)
    if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all():
        raise ValueError('Invalid C vertices')
    if f.ndim != 2 or f.shape[1] != 3 or f.dtype.kind not in 'iu':
        raise ValueError('Invalid C indexed triangles')
    if not len(f) or f.min() < 0 or f.max() >= len(v):
        raise ValueError('Invalid C IDs')
    edges = defaultdict(list)
    for fi, tri in enumerate(f):
        if len(set(map(int, tri))) != 3:
            raise ValueError('Degenerate C identities')
        for i in range(3):
            u, w = int(tri[i]), int(tri[(i + 1) % 3])
            edges[tuple(sorted((u, w)))].append((fi, u, w))
    if any(len(owners) > 2 for owners in edges.values()):
        raise ValueError('Prototype does not support nonmanifold C')
    for owners in edges.values():
        if len(owners) == 2 and owners[0][1:] != owners[1][1:][::-1]:
            raise ValueError('C winding must be consistent')
    # Ordering does not enter features. Physical output is edge-permutation equivariant.
    opened = [owners[0] for edge, owners in sorted(edges.items()) if len(owners) == 1]
    if not opened:
        raise ValueError('No C open boundary')
    return np.asarray(opened, np.int64)

def edge_features(C_vertices, C_faces, total_N):
    """Only C geometry/topology and requested N. No Y, parent ID or labels."""
    v, f = np.asarray(C_vertices, np.float32), np.asarray(C_faces, np.int64)
    e = boundary_from_C(v, f)
    n, k = int(total_N), len(f)
    if not 128 <= n <= 256 or not 0 < k < n or len(e) > n-k:
        raise ValueError('N cannot hold one candidate per C boundary edge')
    center = (v.min(0) + v.max(0)) * .5
    span = np.ptp(v, axis=0)
    degree = np.bincount(e[:, 1:].reshape(-1), minlength=len(v))
    features = []
    for fi, u, w in e:
        a, b = v[u], v[w]
        delta = b-a
        normal = np.cross(v[f[fi,1]]-v[f[fi,0]], v[f[fi,2]]-v[f[fi,0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            raise ValueError('Zero-area C face')
        feature = np.r_[(a+b)*.5/L, delta/L, np.abs(delta)/L,
                        normal/norm, span/L, center/L,
                        np.linalg.norm(delta)/L, k/256., n/256.,
                        degree[u]/8., degree[w]/8., 1.]
        assert feature.shape == (FEATURE_DIM,)
        features.append(feature)
    return e, np.asarray(features, np.float32)

def inference_layout(C_vertices, C_faces_original, C_faces_model, total_N, occupied):
    """Create a fixed-N layout from C and predicted booleans only.

    C_faces_model is an internal corner permutation of C_faces_original, not a
    different surface. Original C order is saved for final export.
    """
    v = np.asarray(C_vertices, np.float32)
    original = np.asarray(C_faces_original, np.int64)
    canonical = np.asarray(C_faces_model, np.int64)
    if original.shape != canonical.shape or not np.array_equal(np.sort(original,1),np.sort(canonical,1)):
        raise ValueError('Model C changes original faces')
    edges, features = edge_features(v, original, total_N)
    o = np.asarray(occupied)
    if o.dtype != np.bool_ or o.shape != (len(edges),):
        raise ValueError('An explicit predicted bool for each C edge is required')
    n, k = int(total_N), len(original)
    anchor = np.full((n,3), -1, np.int64)
    anchor[:k] = canonical
    for ei, (_, u, w) in enumerate(edges):
        if o[ei]:
            anchor[k+ei,:2] = [w,u]
    known = np.arange(n) < k
    free_corner = anchor < 0
    return dict(anchor_ids=anchor, known_mask=known, valid_mask=np.ones(n,bool),
                free_corner_mask=free_corner, occupied=o.copy(), edge_records=edges,
                features=features, free_coordinate_count=3*int(free_corner.sum()),
                seam_slots=k+np.flatnonzero(o), bulk_slots=np.flatnonzero(~known & free_corner.all(1)),
                C_vertices=v, C_faces_original=original, C_faces_model=canonical,
                N=n, K=k)

def supervised_sample(case, raw_noise, epsilon, t, permutation_seed, arm):
    """Supervised construction only. Source occupancy never used by inference_layout callers.

    Ordinary control and seam arm share the underlying noise/epsilon/t/face
    permutation, but necessarily have different OT assignments and free DOFs.
    Bulk OT retains the original pre-OT half-scale and six corner permutations.
    """
    if arm not in ('FULL4_C200','SEAM_SLOT_C200'):
        raise ValueError('Unknown arm')
    full = np.asarray(case['full_target'], np.float32).reshape(-1,3,3)
    pre = np.asarray(case['pre_ot_full'], np.float32)
    source_vertices = np.asarray(case['source_face_vertex_ids'], np.int64)
    known_ids = np.asarray(case['known_ids'], np.int64)
    free_ids = np.asarray(case['free_ids'], np.int64)
    n, k = len(full), len(known_ids)
    if not np.array_equal(pre*np.float32(2),full):
        raise ValueError('Original pre-OT x2 contract differs')
    z, eps = np.asarray(raw_noise),np.asarray(epsilon)
    if z.shape != (n-k,3,3) or eps.shape != z.shape or not np.isfinite(z).all() or not np.isfinite(eps).all():
        raise ValueError('Common underlying random arrays differ')
    if not 0 <= float(t) <= 1:
        raise ValueError('Invalid t')
    labels = np.asarray(case['occupied_labels'],bool)
    occupancy = labels if arm == 'SEAM_SLOT_C200' else np.zeros_like(labels)
    layout = inference_layout(case['C_vertices'],case['C_faces_original'],case['C_faces_model'],n,occupancy)
    anchor = layout['anchor_ids']
    target = np.zeros((n,3,3),np.float32)
    ids = np.full(n,-1,np.int64)
    corner = np.full((n,3),-1,np.int64)
    target[:k] = full[known_ids]
    ids[:k] = known_ids
    corner[:k] = np.arange(3)
    owners = {}
    if arm == 'SEAM_SLOT_C200':
        for ei in np.flatnonzero(labels):
            fi = int(case['edge_owner_face'][ei])
            if fi in owners.values():
                raise ValueError('Multi-edge face needs alias decoder; unsupported in this minimal prototype')
            source_triple = np.asarray(case['edge_source_triple'][ei],np.int64)
            if int(source_triple[2]) in set(map(int,case['C_source_vertex_ids'])):
                raise ValueError('Third-C reference needs explicit discrete output; unsupported here')
            cp = np.asarray([np.flatnonzero(source_vertices[fi] == vi).item() for vi in source_triple],np.int64)
            slot = k+int(ei)
            owners[slot] = fi
            ids[slot],corner[slot],target[slot] = fi,cp,full[fi,cp]
    bulk_slots = layout['bulk_slots']
    bulk_faces = np.asarray([fi for fi in free_ids if fi not in set(owners.values())],np.int64)
    if len(bulk_faces) != len(bulk_slots):
        raise ValueError('Owner plus returned-Null bulk violates N')
    rng = np.random.RandomState(int(permutation_seed))
    bulk_faces = bulk_faces[rng.permutation(len(bulk_faces))]
    cp0 = np.asarray([[0,1,2],[1,2,0],[2,0,1]],np.int64)[rng.randint(0,3,len(bulk_faces))]
    face_map, corner_map = nested_ot_map(pre[bulk_faces[:,None],cp0],z[bulk_slots-k])
    ids[bulk_slots] = bulk_faces[np.argsort(face_map)]
    corner[bulk_slots] = to_noise_slots(cp0,face_map,corner_map)
    target[bulk_slots] = to_noise_slots(full[bulk_faces[:,None],cp0],face_map,corner_map)
    mixed = (np.float32(np.sqrt(.75))*z.astype(np.float32)+np.float32(.5)*eps.astype(np.float32))
    x0 = np.zeros_like(target)
    x0[k:] = mixed
    fixed = anchor >= 0
    x0[fixed] = layout['C_vertices'][anchor[fixed]]
    if not np.array_equal(target[fixed],x0[fixed]):
        raise ValueError('Supervised seam endpoints do not coincide with C')
    velocity = target-x0
    xt = (np.float32(1)-np.float32(t))*x0+np.float32(t)*target
    xt[fixed] = x0[fixed]
    order = np.random.RandomState(int(permutation_seed)^0xD00D).permutation(n)
    vals = dict(x0=x0[order],x1=target[order],xt=xt[order],u=velocity[order],
                anchor_ids=anchor[order],known_mask=layout['known_mask'][order],
                valid_mask=layout['valid_mask'][order],free_corner_mask=layout['free_corner_mask'][order],
                source_face_ids=ids[order],source_corner_permutations=corner[order],
                face_permutation=order,ot_face_map=face_map,ot_corner_map=corner_map,
                t=np.asarray(t,np.float32),C_vertices=layout['C_vertices'],
                free_coordinate_count=np.asarray(layout['free_coordinate_count']),
                occupied_labels=labels,occupied_for_FM=occupancy,
                edge_features=layout['features'],N=np.asarray(n),K=np.asarray(k),
                arm=np.asarray(arm),production_OT_calls=np.asarray(1))
    assert np.array_equal(np.sort(ids),np.arange(n))
    assert np.array_equal(full[vals['source_face_ids'][:,None],vals['source_corner_permutations']],vals['x1'])
    assert not vals['u'][vals['anchor_ids']>=0].any()
    assert int(vals['free_coordinate_count']) == 9*(n-k)-6*int(occupancy.sum())
    return vals
