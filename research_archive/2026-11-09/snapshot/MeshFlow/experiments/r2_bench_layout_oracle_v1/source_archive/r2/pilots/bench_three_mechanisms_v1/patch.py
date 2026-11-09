"""Fixed C-only double-triangle patches with live shared free vertex states.

No model construction and no source-Y access in the inference functions.
Source connectivity is used only to assign the unchanged supervised target to
the fixed patch/singleton slots. This is not a learned global topology model.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import numpy as np

VERSION = 'bench_fixed_double_triangle_patch_v1'
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_bench_three_mechanisms_v1'
L = float(np.sqrt(3.) * 2 / .3762)
# Identity and exchange of the two whole faces. Reversing only the shared
# edge while retaining the face identities flips both source normals, so is
# deliberately NOT an allowed patch symmetry.
QUAD_PERMUTATIONS = np.asarray([[0, 1, 2, 3], [1, 0, 3, 2]], np.int64)


def local_path(value):
    value = str(value)
    if os.name == 'nt' and value.startswith('/mnt/'):
        value = value[5].upper() + ':/' + value[7:]
    elif os.name != 'nt' and len(value) > 2 and value[1] == ':':
        value = '/mnt/' + value[0].lower() + value[2:].replace('\\', '/')
    return Path(value)


def array_hash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def inference_layout(C_vertices, C_faces_original, C_faces_model, total_N):
    """Only C and N; M=floor((N-K)/8), with no learned/source layout."""
    cv = np.asarray(C_vertices, np.float32)
    original = np.asarray(C_faces_original, np.int64)
    model = np.asarray(C_faces_model, np.int64)
    n, k = int(total_N), len(original)
    if cv.ndim != 2 or cv.shape[1] != 3 or not np.isfinite(cv).all():
        raise ValueError('Finite C vertices required')
    if original.shape != (k, 3) or model.shape != original.shape or not 2 <= k < n:
        raise ValueError('Valid C triangle arrays and K>=2 required')
    if original.min() < 0 or original.max() >= len(cv):
        raise ValueError('C vertex ID outside input')
    if not np.array_equal(np.sort(original, 1), np.sort(model, 1)):
        raise ValueError('Internal C order must preserve every source triangle')
    if not 128 <= n <= 799:
        raise ValueError('Frozen N128..799 contract')
    free, patches = n-k, (n-k)//8
    anchor = np.full((n, 3), -1, np.int64); anchor[:k] = model
    known = np.arange(n) < k
    states = np.full((n, 3), -1, np.int64); states[:k] = model
    patch_ids = np.full(n, -1, np.int64)
    pairs, counter = [], len(cv)
    for p in range(patches):
        a, b = k+2*p, k+2*p+1
        states[a] = [counter, counter+1, counter+2]
        states[b] = [counter+1, counter, counter+3]
        pairs.extend([[3*a, 3*b+1], [3*a+1, 3*b]])
        patch_ids[[a, b]] = p; counter += 4
    for f in range(k+2*patches, n):
        states[f] = np.arange(counter, counter+3); counter += 3
    pairs = np.asarray(pairs, np.int64).reshape(-1, 2)
    weights = np.ones((n, 3), np.float32); weights[:k] = 0
    if len(pairs): weights.reshape(-1)[pairs.reshape(-1)] = .5
    divisor = 9*free-6*patches
    if 3*int(weights.sum()) != divisor:
        raise AssertionError('Independent-coordinate denominator')
    book = defaultdict(list)
    for fi, face in enumerate(original):
        for j in range(3):
            u, v = int(face[j]), int(face[(j+1)%3])
            book[tuple(sorted((u, v)))].append((fi, u, v))
    opened = np.asarray([items[0] for _, items in sorted(book.items()) if len(items) == 1], np.int64).reshape(-1, 3)
    return dict(N=n, K=k, patch_count=patches, patch_ids=patch_ids,
                shared_corner_pairs=pairs, state_ids=states,
                corner_weights=weights, free_coordinate_count=divisor,
                corner_loss_weights=weights, expanded_free_coordinate_count=9*free,
                known_mask=known, valid_mask=np.ones(n, bool), anchor_ids=anchor,
                free_corner_mask=anchor < 0, C_vertices=cv,
                C_faces_original=original, C_faces_model=model,
                occupied=np.zeros(len(opened), bool), edge_records=opened,
                seam_slots=np.zeros(0, np.int64), bulk_slots=np.arange(k, n, dtype=np.int64),
                representation=VERSION)


def _pairs(layout):
    a = np.asarray(layout['shared_corner_pairs'], np.int64)
    if a.ndim != 2 or a.shape[1] != 2 or len(np.unique(a)) != a.size:
        raise ValueError('Disjoint pairwise free-corner groups required')
    return a


def _project(value, layout, noise=False):
    """Pair average; noise mode uses sqrt(2) rather than 2 for unit variance."""
    pairs = _pairs(layout)
    factor = np.float32(1/np.sqrt(2.) if noise else .5)
    if isinstance(value, np.ndarray):
        if value.dtype != np.float32 or value.size % 3:
            raise ValueError('FP32 corner coordinates required')
        out = value.copy(); flat = out.reshape(-1, 3)
        if pairs.size and pairs.max() >= len(flat): raise ValueError('Corner index outside state')
        mean = (flat[pairs[:, 0]] + flat[pairs[:, 1]]) * factor
        flat[pairs[:, 0]] = mean; flat[pairs[:, 1]] = mean
        return out
    import torch
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
        raise ValueError('FP32 tensor required; project after BF16 prediction conversion')
    if value.ndim == 3 and value.shape[-1] == 9 and value.shape[0] != 1:
        raise ValueError('This runtime is microbatch=1 only')
    ix = torch.as_tensor(pairs, dtype=torch.long, device=value.device)
    out = value.clone(); flat = out.reshape(-1, 3)
    # Each corner appears once; indexed assignment is deterministic and the
    # mean retains gradients from both per-face velocity predictions.
    mean = (flat[ix[:, 0]] + flat[ix[:, 1]]) * float(factor)
    flat[ix[:, 0]] = mean; flat[ix[:, 1]] = mean
    return out


def project_velocity(prediction, layout):
    return _project(prediction, layout)


def project_coordinates(coordinates, layout):
    return _project(coordinates, layout)


def initial_state(layout, raw_noise):
    """Use existing independent Gaussian corners, without any random draws."""
    n, k = int(layout['N']), int(layout['K'])
    z = np.asarray(raw_noise)
    if z.shape != (n-k, 3, 3) or z.dtype != np.float32 or not np.isfinite(z).all():
        raise ValueError('Frozen finite FP32 Gaussian array differs')
    x = np.zeros((n, 3, 3), np.float32)
    x[:k] = layout['C_vertices'][layout['C_faces_model']]; x[k:] = z
    x = _project(x, layout, noise=True)
    assert_shared(x, layout)
    return x


def assert_shared(coordinates, layout):
    flat = np.asarray(coordinates).reshape(-1, 3); pairs = _pairs(layout)
    if not np.array_equal(flat[pairs[:, 0]], flat[pairs[:, 1]]):
        raise ValueError('Shared free state split; no silent fallback')


def decode_indices(triangles, layout):
    """Exact shared state IDs, not a geometric weld; no faces are removed."""
    t = np.asarray(triangles)
    if t.dtype != np.float32 or t.shape != (int(layout['N']), 3, 3):
        raise ValueError('Expected FP32 full RAW triangles')
    assert_shared(t, layout)
    ids = np.asarray(layout['state_ids'], np.int64)
    v = np.zeros((int(ids.max())+1, 3), np.float32)
    for sid in np.unique(ids):
        values = t[ids == sid]
        if not np.array_equal(values, np.broadcast_to(values[0], values.shape)):
            raise ValueError('An explicitly shared index has unequal values')
        v[sid] = values[0]
    if not np.array_equal(v[ids], t): raise ValueError('Exact index decode changed geometry')
    return dict(vertices=v, faces=ids.copy(), triangles=t.copy(),
                shared_corner_pairs=_pairs(layout).copy(), patch_ids=layout['patch_ids'].copy())


def raw_arrays(raw, layout, logits, source):
    """Existing RAW scorer schema plus explicit generated patch identities.

    ``source`` is the C-only original-order metadata sidecar, not source Y.
    """
    n, k = int(layout['N']), int(layout['K'])
    dec = decode_indices(raw, layout)
    corner_map = np.tile(np.arange(3, dtype=np.int64), (n, 1))
    corner_map[:k] = np.argsort(source['C_source_corner_permutations'], axis=1)
    faces = dec['faces'][np.arange(n)[:, None], corner_map]
    output = dec['vertices'][faces]
    return dict(raw=raw, output=output, C=output[:k].copy(), free=output[k:].copy(),
                vertices=dec['vertices'], faces=faces, known_faces=np.asarray(k), input_faces=np.asarray(n),
                source_face_ids=np.arange(n, dtype=np.int64), raw_slot_ids=np.arange(n, dtype=np.int64),
                source_corner_ids=corner_map, identity_domain=np.asarray('generated_RAW_slot_not_GT'),
                raw_anchor_ids=layout['anchor_ids'],
                output_anchor_ids=layout['anchor_ids'][np.arange(n)[:, None], corner_map],
                raw_seam_slot_mask=np.zeros(n, bool), retained_seam_mask=np.zeros(n, bool),
                raw_slot_edge_id=np.full(n, -1, np.int64), occupied=layout['occupied'],
                occupancy_logits=np.asarray(logits, np.float32), edge_records=layout['edge_records'],
                C_vertices=layout['C_vertices'], C_faces_original=layout['C_faces_original'],
                C_faces_model=layout['C_faces_model'],
                C_original_source_vertex_ids=source['C_original_source_vertex_ids'],
                C_source_corner_permutations=source['C_source_corner_permutations'],
                C_inverse_corner_permutation=np.argsort(source['C_source_corner_permutations'], axis=1),
                raw_patch_ids=layout['patch_ids'], raw_shared_corner_pairs=layout['shared_corner_pairs'],
                raw_patch_state_ids=layout['state_ids'], patch_count=np.asarray(layout['patch_count']),
                patch_representation=np.asarray(VERSION))


def build_source_plan(case, verified_pair=None):
    """CPU gate: exact target partition into legal FF pairs and singletons.

    All original faces, including unrelated old defects, remain untouched.
    Only candidate shared-edge pairs are classified, not the entire source mesh.
    """
    import networkx as nx
    faces = np.asarray(case['source_faces'], np.int64)
    canonical = np.asarray(case['source_face_vertex_ids'], np.int64)
    vertices = np.asarray(case['model_vertices'], np.float32)
    target = np.asarray(case['full_target'], np.float32)
    known, free = np.asarray(case['known_ids'], np.int64), np.asarray(case['free_ids'], np.int64)
    n, k = len(faces), len(known); m = (n-k)//8
    if not np.array_equal(vertices[canonical], target): raise ValueError('Canonical target/vertex mapping differs')
    if not np.array_equal(np.sort(np.r_[known, free]), np.arange(n)):
        raise ValueError('C/Y do not partition complete target')
    if verified_pair is None:
        from r2.data_tools.bench_parts_v1.quality import detector
        verified_pair = detector().verified_pair
    book = defaultdict(list)
    for fi, f in enumerate(faces):
        for j in range(3):
            u, v = int(f[j]), int(f[(j+1)%3])
            book[tuple(sorted((u, v)))].append((fi, u, v))
    free_set = set(map(int, free)); candidates = []; rejection = Counter()
    for edge, rows in sorted(book.items()):
        if len(rows) != 2:
            rejection['NOT_EXACT_TWO_INCIDENT_SOURCE_FACES'] += 1; continue
        (a, u, v), (b, s, t) = rows
        if a not in free_set or b not in free_set: continue
        if (u, v) != (t, s):
            rejection['SOURCE_WINDING_NOT_OPPOSITE'] += 1; continue
        if len(set(faces[a]) | set(faces[b])) != 4:
            rejection['NOT_FOUR_UNIQUE_VERTICES'] += 1; continue
        check = verified_pair(vertices[faces[a]], vertices[faces[b]], L)
        if not check.get('verified') or check.get('classification') != 'shared_edge_contact':
            rejection['PAIR_'+str(check.get('classification', 'UNKNOWN'))] += 1; continue
        ca = next(int(z) for z in faces[a] if z not in edge)
        cb = next(int(z) for z in faces[b] if z not in edge)
        candidates.append(dict(faces=[int(a), int(b)], quad=[u, v, ca, cb], edge=list(edge)))
    # Stable integer insertion and distinct rank weights make this repeatable in
    # the frozen networkx runtime. Maximum cardinality prevents greedy shortage.
    graph = nx.Graph(); graph.add_nodes_from(sorted(free_set))
    candidates.sort(key=lambda x: (x['faces'][0], x['faces'][1], x['edge']))
    for rank, item in enumerate(candidates):
        graph.add_edge(*item['faces'], weight=len(candidates)-rank)
    match = nx.algorithms.matching.max_weight_matching(graph, maxcardinality=True, weight='weight')
    match = sorted(tuple(sorted(map(int, pair))) for pair in match)
    lookup = {tuple(sorted(x['faces'])): x for x in candidates}
    enough = len(match) >= m
    selected = [lookup[p] for p in match[:m]] if enough else []
    used = {fi for row in selected for fi in row['faces']}
    singles = [int(fi) for fi in free if int(fi) not in used]
    plan = dict(schema=VERSION, status='PASS' if enough else 'BLOCKED', N=n, K=k,
                required_patch_count=m, maximum_disjoint_legal_pairs=len(match),
                legal_candidate_pairs=len(candidates), rejected_candidates=dict(rejection),
                source_pairs=[x['faces'] for x in selected], source_quads=[x['quad'] for x in selected],
                singleton_source_faces=singles, source_faces_sha256=array_hash(faces),
                canonical_ids_sha256=array_hash(canonical), full_target_sha256=array_hash(target),
                matching_algorithm='networkx_max_weight_matching_maxcardinality_sorted_integer_graph',
                networkx_version=nx.__version__, no_target_face_removed=True,
                failure_reason=None if enough else 'INSUFFICIENT_DISJOINT_LEGAL_FF_PAIRS',
                source_only_supervision=True, inference_graph_depends_on='N,K only')
    if enough:
        assert sorted([i for p in plan['source_pairs'] for i in p]+singles) == sorted(free_set)
        assert len(used) == 2*m
    return plan


@lru_cache(maxsize=64)
def _read_plan(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_plan(case, row):
    path = local_path(row.get('patch_plan_path', OUT/'data/patch_sources'/f"{row['parent_id']}.json"))
    plan = _read_plan(str(path))
    if plan['status'] != 'PASS': raise ValueError('Source patch gate is BLOCKED')
    for key, name in [('source_faces', 'source_faces_sha256'),
                      ('source_face_vertex_ids', 'canonical_ids_sha256'), ('full_target', 'full_target_sha256')]:
        if array_hash(case[key]) != plan[name]: raise ValueError('Registered patch source changed: '+key)
    return plan


def _triangle_ot(target, noise):
    from itertools import permutations
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    p = np.asarray(list(permutations(range(3))), np.int64)
    count = len(target)
    if not count: return np.zeros(0, np.int64), np.zeros((0, 3), np.int64)
    cost = np.stack([cdist(target.reshape(count, -1), noise[:, q].reshape(count, -1)) for q in p])
    rr, cc = linear_sum_assignment(cost.min(0))
    return cc.astype(np.int64), p[cost.argmin(0)[rr, cc]]


def _quad_ot(target_quads, noise_quads):
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    count = len(target_quads)
    if not count: return np.zeros(0, np.int64), np.zeros((0, 4), np.int64)
    # target permutations explicitly preserve the same two triangle sets.
    costs = np.stack([cdist(target_quads[:, p].reshape(count, -1), noise_quads.reshape(count, -1))
                      for p in QUAD_PERMUTATIONS])
    rr, cc = linear_sum_assignment(costs.min(0))
    return cc.astype(np.int64), QUAD_PERMUTATIONS[costs.argmin(0)[rr, cc]]


def make_sample(case, row, plan=None):
    """Native-compatible source sample, with independent-coordinate weights."""
    if plan is None: plan = load_plan(case, row)
    if plan['status'] != 'PASS': raise ValueError('Cannot train blocked patch plan')
    n, k = int(plan['N']), int(plan['K']); m = (n-k)//8
    full = np.asarray(case['full_target'], np.float32).copy()
    pre = np.asarray(case['pre_ot_full'], np.float32).copy()
    vertices = np.asarray(case['model_vertices'], np.float32).copy()
    cv = np.asarray(case['C_vertices'], np.float32).copy()
    alpha = np.float32(row['alpha'])
    for a in (full, pre, vertices, cv): a[..., 0] *= alpha
    if not np.array_equal(pre*np.float32(2), full): raise ValueError('preOT half-scale changed')
    layout = inference_layout(cv, case['C_faces_original'], case['C_faces_model'], n)
    z = np.random.RandomState(int(row['noise_seed'])).standard_normal((n-k, 3, 3)).astype(np.float32)
    eps = np.random.RandomState(int(row['epsilon_seed'])).standard_normal(z.shape).astype(np.float32)
    normal = float(np.random.RandomState(int(row['time_seed'])).standard_normal())
    t = np.float32(1/(1+np.exp(-normal)))
    initial_z, initial_eps = initial_state(layout, z), initial_state(layout, eps)
    canonical = np.asarray(case['source_face_vertex_ids'], np.int64)
    known = np.asarray(case['known_ids'], np.int64)
    target = np.zeros((n, 3, 3), np.float32)
    source_ids = np.full(n, -1, np.int64); corner = np.full((n, 3), -1, np.int64)
    target[:k] = full[known]; source_ids[:k] = known; corner[:k] = np.arange(3)
    qpairs = np.asarray(plan['source_pairs'], np.int64).reshape(m, 2)
    quads = np.asarray(plan['source_quads'], np.int64).reshape(m, 4)
    noise_quads = np.asarray([initial_z[[k+2*p, k+2*p, k+2*p, k+2*p+1], [0, 1, 2, 2]] for p in range(m)])
    qmap, qperm = _quad_ot(vertices[quads]*np.float32(.5), noise_quads)
    for qi, slot in enumerate(qmap):
        q = quads[qi, qperm[qi]]
        for side, wanted in enumerate((q[[0, 1, 2]], q[[1, 0, 3]])):
            sf = next(int(f) for f in qpairs[qi] if set(canonical[f]) == set(wanted))
            cp = np.asarray([int(np.flatnonzero(canonical[sf] == vi).item()) for vi in wanted])
            dst = k+2*int(slot)+side
            source_ids[dst], corner[dst], target[dst] = sf, cp, full[sf, cp]
    singles = np.asarray(plan['singleton_source_faces'], np.int64)
    rng = np.random.RandomState(int(row['permutation_seed']))
    singles = singles[rng.permutation(len(singles))]
    cp0 = np.asarray([[0,1,2], [1,2,0], [2,0,1]], np.int64)[rng.randint(0,3,len(singles))]
    fslot = np.arange(k+2*m, n)
    fmap, cmap = _triangle_ot(pre[singles[:,None], cp0], z[fslot-k])
    for ti, si in enumerate(fmap):
        cp = cp0[ti, np.argsort(cmap[ti])]
        dst = fslot[int(si)]; source_ids[dst], corner[dst] = singles[ti], cp
        target[dst] = full[singles[ti], cp]
    mixed = np.float32(np.sqrt(.75))*initial_z + np.float32(.5)*initial_eps
    mixed[:k] = cv[layout['C_faces_model']]
    velocity = target-mixed
    xt = (np.float32(1)-t)*mixed+t*target
    xt[:k] = mixed[:k]
    if not np.array_equal(np.sort(source_ids), np.arange(n)):
        raise ValueError('Patch plus singleton target faces not a bijection')
    if not np.array_equal(full[source_ids[:,None], corner], target):
        raise ValueError('Source face/corner target roundtrip failed')
    for a in (target, mixed, velocity, xt): assert_shared(a, layout)
    order = np.random.RandomState(int(row['permutation_seed']) ^ 0xD00D).permutation(n)
    inverse = np.argsort(order)
    pairs = layout['shared_corner_pairs']
    pairs = 3*inverse[pairs//3] + pairs%3
    result = dict(x0=mixed[order], x1=target[order], xt=xt[order], u=velocity[order],
                  anchor_ids=layout['anchor_ids'][order], known_mask=layout['known_mask'][order],
                  valid_mask=layout['valid_mask'][order], free_corner_mask=layout['free_corner_mask'][order],
                  source_face_ids=source_ids[order], source_corner_permutations=corner[order],
                  face_permutation=order, ot_face_map=fmap, ot_corner_map=cmap,
                  patch_ot_map=qmap, patch_ot_permutations=qperm,
                  t=t, N=np.asarray(n), K=np.asarray(k), C_vertices=cv,
                  occupied_labels=np.asarray(case['occupied_labels'], bool),
                  occupied_for_FM=np.zeros_like(case['occupied_labels'], dtype=bool),
                  free_coordinate_count=np.asarray(layout['free_coordinate_count']),
                  expanded_free_coordinate_count=np.asarray(9*(n-k)),
                  corner_weights=layout['corner_weights'][order], state_ids=layout['state_ids'][order],
                  corner_loss_weights=layout['corner_weights'][order],
                  patch_ids=layout['patch_ids'][order], shared_corner_pairs=pairs,
                  patch_count=m, arm=np.asarray('PATCH'), production_OT_calls=np.asarray(2),
                  representation=VERSION)
    for a in ('xt', 'x0', 'x1', 'u'): assert_shared(result[a], result)
    if result['u'][result['anchor_ids'] >= 0].any(): raise ValueError('C target velocity changed')
    return result


def coordinate_weights(sample, device=None):
    import torch
    return torch.as_tensor(sample['corner_weights'], dtype=torch.float32, device=device)


def verify_sample(sample):
    n, k, m = int(sample['N']), int(sample['K']), int(sample['patch_count'])
    anchor = np.asarray(sample['anchor_ids']); known = np.asarray(sample['known_mask'])
    if not 128 <= n <= 799 or not 2 <= k < n or m != (n-k)//8:
        raise ValueError('Fixed patch budget changed')
    if known.sum() != k or not np.array_equal(known, (anchor >= 0).all(1)) or (anchor[~known] >= 0).any():
        raise ValueError('Only original C corners can be anchored')
    if int(sample['free_coordinate_count']) != 9*(n-k)-6*m:
        raise ValueError('Patch independent denominator changed')
    if int(sample['expanded_free_coordinate_count']) != 9*(n-k):
        raise ValueError('Expanded denominator changed')
    if float(np.asarray(sample['corner_weights']).sum())*3 != int(sample['free_coordinate_count']):
        raise ValueError('Shared-copy weights do not count independent coordinates')
    for key in ('x0','x1','xt','u'): assert_shared(sample[key], sample)
    if not np.array_equal(np.sort(sample['source_face_ids']), np.arange(n)):
        raise ValueError('Original target faces not a bijection')
    return dict(policy=VERSION, patch_count=m, free_coordinate_count=int(sample['free_coordinate_count']),
                expanded_free_coordinate_count=9*(n-k), shared_pairs=2*m,
                source_owner_count=int(np.asarray(sample['occupied_labels']).sum()), active_seam_slots=0)


def weighted_fm_loss(prediction, target, sample, denominator):
    """Weights .5 for each shared copy; each physical coordinate counted once."""
    import torch
    n = int(sample['N']); expected = int(sample['free_coordinate_count'])
    if denominator < expected: raise ValueError('Invalid effective-batch denominator')
    p = prediction.float().reshape(-1, 3, 3)[:n]
    y = target.float().reshape(-1, 3, 3)[:n]
    weights = torch.as_tensor(sample['corner_weights'], device=p.device)
    return ((p-y).square()*weights[..., None]).sum()/int(denominator)
