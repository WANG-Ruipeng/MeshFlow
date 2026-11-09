"""Fixed indexed RAW/D0 assembly and one bounded local edge-vector fit.

No model, GEO, proximity welding, intersection detector or score calls occur
here. Original face/corner identities, not source-Y geometry, define matching.
The two arms share one topology and the same full-slot HINGE motion reference.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve

VERSION = 'fixed_namespace_local_edge_vector_v1'
ROOT = Path(__file__).resolve().parents[3]
R = .015 / (.95 * .3762)
L = np.sqrt(3.) * 2. / .3762
CONFIG = dict(version='local_edge_vector_fit_v1', regularization_lambda=1.,
              maximum_displacement_R=2., area_ratio_floor=.8,
              normal_cosine_floor=.5, region_span_ratio_floor=.9,
              nonzero_span_epsilon_L=1e-10, line_search_halvings=10)


def _path(path):
    path = str(path).replace('\\', '/')
    if os.name == 'nt' and path.startswith('/mnt/e/'):
        path = 'E:/' + path[7:]
    elif os.name != 'nt' and path.lower().startswith('e:/'):
        path = '/mnt/e/' + path[3:]
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _arrays(path):
    with np.load(_path(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _exact(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _mesh(mesh, N, K):
    v, f = mesh['vertices'], np.asarray(mesh['faces'], np.int64)
    sf = np.asarray(mesh['source_face_ids'], np.int64)
    sc = np.asarray(mesh['source_corner_ids'], np.int64)
    if v.dtype != np.float32 or v.ndim != 2 or v.shape[1] != 3:
        raise ValueError('Expected FP32 indexed vertices')
    if f.shape != (len(sf), 3) or sc.shape != f.shape or not np.isfinite(v).all():
        raise ValueError('Malformed indexed/provenance arrays')
    if not _exact(v[f], mesh['output']) or np.any(f < 0) or np.any(f >= len(v)):
        raise ValueError('Output differs from indexed mesh')
    if len(set(sf.tolist())) != len(sf) or np.any(sf < 0) or np.any(sf >= N):
        raise ValueError('Original slot lineage must be unique')
    if not np.array_equal(sf[:K], np.arange(K)) or int(mesh['known_faces']) != K:
        raise ValueError('Source C faces must be first and in original face order')
    if not np.all(np.sort(sc, axis=1) == np.arange(3)):
        raise ValueError('Each face needs an invertible canonical corner map')
    if not _exact(mesh['C'], v[f[:K]]):
        raise ValueError('Source C export mismatch')
    return v, f, sf, sc


def assemble(d0, raw_base, raw, d0_pre, selected, regions, raw_metadata):
    """Use cached BASE indices; only original C vertex identity crosses sources.

    Returned vertex IDs are compact. Both source-to-new maps preserve internal
    sharing up to this bijective renumbering; no unreferenced vertices remain.
    """
    raw = np.asarray(raw)
    d0_pre = np.asarray(d0_pre)
    N, K = len(raw), int(d0['known_faces'])
    selected = np.asarray(selected, bool)
    if raw.dtype != np.float32 or raw.shape != (N, 3, 3) or not _exact(raw[:K], d0_pre[:K]):
        raise ValueError('Expected matching canonical FP32 input C')
    if d0_pre.dtype != np.float32 or d0_pre.shape != raw.shape or not np.isfinite(raw).all() or not np.isfinite(d0_pre).all():
        raise ValueError('Invalid canonical RAW/D0_PRE')
    if selected.shape != (N,) or selected[:K].any():
        raise ValueError('Selection must be free original slots only')
    if set(regions) != {'P1', 'P2'}:
        raise ValueError('Exactly frozen P1/P2 required')
    regions = {k: np.asarray(a, bool) for k, a in regions.items()}
    if any(a.shape != (N,) or a[:K].any() for a in regions.values()):
        raise ValueError('Invalid region domain')
    if np.any(regions['P1'] & regions['P2']) or not np.array_equal(selected, regions['P1'] | regions['P2']):
        raise ValueError('Selection must equal disjoint P1 union P2')
    dv, df, ds, dc = _mesh(d0, N, K)
    rv, rf, rs, rc = _mesh(raw_base, N, K)
    if not _exact(d0['C'], raw_base['C']) or not np.array_equal(dc[:K], rc[:K]):
        raise ValueError('Source-oriented C/corner identities disagree')
    if not _exact(raw[np.arange(K)[:, None], dc[:K]], d0['C']):
        raise ValueError('C provenance does not reconstruct original canonical C')
    raw_C_to_d0 = {}
    reverse_C = {}
    for r, d in zip(rf[:K].ravel(), df[:K].ravel()):
        r, d = int(r), int(d)
        if r in raw_C_to_d0 and raw_C_to_d0[r] != d:
            raise ValueError('C vertex identity is not consistent')
        if d in reverse_C and reverse_C[d] != r:
            raise ValueError('C topology differs across cached BASE meshes')
        if not _exact(rv[r], dv[d]):
            raise ValueError('Corresponding C vertex coordinate differs')
        raw_C_to_d0[r], reverse_C[d] = d, r

    drows = {int(s): i for i, s in enumerate(ds)}
    rrows = {int(s): i for i, s in enumerate(rs)}
    records = []
    absent = np.full(N, '', dtype='<U48')
    for s in range(N):
        use_raw = bool(selected[s])
        rows = rrows if use_raw else drows
        if s not in rows:
            absent[s] = 'ABSENT_RAW_BASE' if use_raw else 'ABSENT_D0_BASE'
            continue
        i = rows[s]
        oldface, corners = (rf[i], rc[i]) if use_raw else (df[i], dc[i])
        keys = []
        for old in oldface:
            old = int(old)
            keys.append(('D0', raw_C_to_d0[old]) if use_raw and old in raw_C_to_d0
                        else ('RAW', old) if use_raw else ('D0', old))
        records.append((s, corners.copy(), keys, use_raw, i))
    keys = sorted({k for rec in records for k in rec[2]}, key=lambda k: (k[0] == 'RAW', k[1]))
    index = {k: i for i, k in enumerate(keys)}
    v = np.asarray([dv[i] if ns == 'D0' else rv[i] for ns, i in keys], np.float32)
    f = np.asarray([[index[k] for k in rec[2]] for rec in records], np.int64)
    sf = np.asarray([rec[0] for rec in records], np.int64)
    sc = np.asarray([rec[1] for rec in records], np.int64)
    selected_rows = selected[sf]
    variable = np.asarray([ns == 'RAW' for ns, _ in keys], bool)
    cv = np.full((N, 3), -1, np.int64)
    cv[sf[:, None], sc] = f
    members = [np.flatnonzero(cv.ravel() == i) for i in range(len(v))]
    if any(not len(m) for m in members):
        raise AssertionError('Unused assembly vertex has no canonical corner member')
    targets = v.astype(np.float64)
    for i in np.flatnonzero(variable):
        targets[i] = d0_pre.reshape(-1, 3)[members[i]].astype(np.float64).mean(axis=0)
    pre = d0_pre.copy()
    pre[selected] = raw[selected]
    dmap = np.full(len(dv), -1, np.int64)
    rmap = np.full(len(rv), -1, np.int64)
    for key, i in index.items():
        (dmap if key[0] == 'D0' else rmap)[key[1]] = i
    for ri, di in raw_C_to_d0.items():
        rmap[ri] = dmap[di]
    raw_anchors = np.asarray(raw_metadata['raw_anchor_ids'], np.int64)
    raw_seam = np.asarray(raw_metadata['raw_seam_slot_mask'], bool)
    if raw_anchors.shape != (N, 3) or raw_seam.shape != (N,) or raw_seam.any() or (raw_anchors[K:] >= 0).any():
        raise ValueError('Expected original Native anchors/no seam')
    post = dict(vertices=v, faces=f, output=v[f], C=v[f[:K]], free=v[f[K:]],
                known_faces=np.asarray(K, np.int64), input_faces=np.asarray(N, np.int64),
                source_face_ids=sf, source_corner_ids=sc, raw_slot_ids=sf.copy(),
                source_face_kept=np.isin(np.arange(N), sf), source_face_reason=absent,
                duplicate_of_source_face=np.full(N, -1, np.int64),
                corner_vertex_ids=cv, input_corner_vertex_ids=cv.copy(), output_corner_vertex_ids=f.copy(),
                vertex_source_input_corner_ids=np.asarray([m[0] for m in members], np.int64),
                raw_anchor_ids=raw_anchors.copy(), output_anchor_ids=raw_anchors[sf[:, None], sc],
                raw_seam_slot_mask=raw_seam.copy(), retained_seam_mask=raw_seam[sf],
                index_domain=np.asarray('FIXED_BASE_C_ID_ONLY_CROSS_NAMESPACE_V1'),
                identity_domain=np.asarray('GENERATED_RAW_SLOT_NOT_GT'),
                lineage_semantics=np.asarray('MEMBER_IDS_ARE_PROVENANCE_NOT_EXACT_COORDINATE_CLAIMS'),
                source_winding_stage=np.asarray('CACHED_BASE_SOURCE_ORIENTED_ASSEMBLY'))
    for name in ('raw_slot_edge_id', 'occupied', 'occupancy_logits', 'edge_records',
                 'C_vertices', 'C_faces_original', 'C_faces_model'):
        if name in raw_metadata:
            post[name] = raw_metadata[name].copy()
    for name in ('C_original', 'C_original_source_vertex_ids', 'C_original_source_face_ids',
                 'C_canonical_source_vertex_ids', 'C_source_corner_permutations',
                 'C_inverse_corner_permutation', 'source_winding_schema', 'source_winding_trust',
                 'source_winding_provenance', 'source_vertex_id_status', 'source_face_id_status',
                 'source_C_vertex_ID_topology_status'):
        if name in d0:
            post[name] = d0[name].copy()
    if 'compact_C_vertex_to_post' in d0:
        post['compact_C_vertex_to_post'] = dmap[d0['compact_C_vertex_to_post']]
    post['source_face_reason'][sf] = np.where(selected_rows, 'RETAINED_RAW_BASE', 'RETAINED_D0_BASE')
    _motion_metadata(post, pre, variable)
    if not _exact(post['C'], d0['C']):
        raise AssertionError('C changed in assembly')
    unselected = ~selected_rows
    for i in np.flatnonzero(unselected):
        if not _exact(post['output'][i], d0['output'][drows[int(sf[i])]]):
            raise AssertionError('D0 unselected geometry changed')
    mapping = dict(d0_vertex_to_composite=dmap, raw_vertex_to_composite=rmap,
                   variable_vertex_mask=variable, selected_output_face_mask=selected_rows,
                   rest_vertices=v.copy(), d0_target_vertices=targets,
                   support_face_mask=selected.copy(), region_P1=regions['P1'], region_P2=regions['P2'],
                   output_namespace=np.asarray(['RAW' if x else 'D0' for x in selected_rows]),
                   vertex_namespace=np.asarray([k[0] for k in keys]),
                   vertex_namespace_source_id=np.asarray([k[1] for k in keys], np.int64))
    summary = dict(version=VERSION, original_faces=N, kept_faces=len(f), C_faces=K,
                   selected_requested_faces=int(selected.sum()), selected_kept_faces=int(selected_rows.sum()),
                   unselected_D0_kept_faces=int((~selected_rows[K:]).sum()),
                   selected_absent_slots=np.flatnonzero(selected & ~post['source_face_kept']).tolist(),
                   unselected_absent_slots=np.flatnonzero(~selected & ~post['source_face_kept']).tolist(),
                   vertices=len(v), variable_vertices=int(variable.sum()), fixed_vertices=int((~variable).sum()),
                   shared_C_vertices=len(raw_C_to_d0), no_new_welding=True, no_GEO=True,
                   topology_cost='RAW/D0 free namespaces remain distinct even at coincident coordinates; internal sharing and C identity retained',
                   exact_D0_external_BASE_geometry=True, exact_C_coordinate_winding_topology=True,
                   post_only_vertex_compaction=True, no_source_Y=True,
                   HINGE_reference='Same canonical mixed original RAW/D0_PRE for both arms, never fitted coordinates')
    return dict(post=post, pre=pre, mappings=mapping, summary=summary)


def _motion_metadata(post, pre, variable):
    cv, v = post['corner_vertex_ids'], post['vertices']
    sf, sc = post['source_face_ids'], post['source_corner_ids']
    good = cv >= 0
    displacement = np.zeros(pre.shape, np.float64)
    displacement[good] = v[cv[good]].astype(np.float64) - pre[good].astype(np.float64)
    norms = np.linalg.norm(displacement, axis=-1)
    ids = np.full(cv.shape, -1, np.int64)
    ids[good] = post['vertex_source_input_corner_ids'][cv[good]]
    status = np.full(cv.shape, 'ABSENT_OUTPUT_FACE', dtype='<U48')
    status[good] = np.where(variable[cv[good]], 'RAW_NAMESPACE_REST_OR_FITTED', 'FIXED_D0_NAMESPACE')
    # No radius-search candidate counts are fabricated: this operator never ran GEO.
    for name, value in (('displacement', norms), ('displacement_vector', displacement),
                        ('target_input_ids', ids), ('status', status)):
        post['corner_' + name] = value
        post['input_corner_' + name] = value.copy()
        post['output_corner_' + name] = value[sf[:, None], sc]


def _shape(vertices, faces):
    tri = vertices.astype(np.float64)[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(cross, axis=1)
    return norm / 2., cross / np.maximum(norm[:, None], np.finfo(np.float64).tiny)


def _objective(vertices, rest, targets, variable, edges, lam):
    u = vertices.astype(np.float64) - rest.astype(np.float64)
    data = float(np.square(vertices[variable].astype(np.float64) - targets[variable]).sum(axis=1).mean()) if variable.any() else 0.
    reg = float(np.square(u[edges[:, 0]] - u[edges[:, 1]]).sum(axis=1).mean()) if len(edges) else 0.
    return dict(data_mean=data, edge_mean=reg, total=data + lam * reg)


def fit(assembly, config=None):
    """One FP64 sparse solve, per-vertex cap, then shared-step shape guards."""
    cfg = dict(CONFIG if config is None else config)
    for key, value in CONFIG.items():
        if cfg.get(key) != value:
            raise ValueError('Operator configuration differs from fixed definition: ' + key)
    post, mapping = assembly['post'], assembly['mappings']
    rest, faces = post['vertices'].copy(), post['faces']
    variable = mapping['variable_vertex_mask']
    selected_faces = mapping['selected_output_face_mask']
    targets = mapping['d0_target_vertices']
    v_ids = np.flatnonzero(variable)
    f = faces[selected_faces]
    edges = np.unique(np.sort(np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]])), axis=1), axis=0) if len(f) else np.empty((0, 2), np.int64)
    before = _objective(rest, rest, targets, variable, edges, cfg['regularization_lambda'])
    info = dict(version=cfg['version'], calls_sparse_solve=0, free_variables=len(v_ids), unique_selected_edges=len(edges),
                objective_before=before, objective_normalization='mean variable squared Euclidean distance plus mean unique selected-edge squared vector difference',
                maximum_displacement_model=cfg['maximum_displacement_R'] * R,
                trials=[], fallback=False, accepted_alpha=0., numerical_precision='FP64 solve; all guards on delivered FP32 vertices',
                detector_calls=0, score_calls=0, parameter_retries=0)
    if not len(v_ids):
        info.update(status='NO_VARIABLES_EXACT_LOCKED_RAW', objective_after=before)
        return {**assembly, 'summary': {**assembly['summary'], 'solver': info}}
    area0, normal0 = _shape(rest, f)
    if np.any(area0 <= 0) or not np.isfinite(area0).all():
        info.update(status='INVALID_REST_EXACT_FALLBACK', fallback=True, objective_after=before,
                    failure_reason='Zero/nonfinite selected rest face; no face removed or modified')
        return {**assembly, 'summary': {**assembly['summary'], 'solver': info}}
    local = np.full(len(rest), -1, np.int64)
    local[v_ids] = np.arange(len(v_ids))
    rows, cols, vals = [], [], []
    for ei, (u, w) in enumerate(edges):
        if variable[u]:
            rows.append(ei); cols.append(int(local[u])); vals.append(1.)
        if variable[w]:
            rows.append(ei); cols.append(int(local[w])); vals.append(-1.)
    incidence = sparse.csr_matrix((vals, (rows, cols)), shape=(len(edges), len(v_ids)), dtype=np.float64)
    weight = cfg['regularization_lambda'] / len(edges) if len(edges) else 0.
    matrix = sparse.eye(len(v_ids), dtype=np.float64, format='csc') / len(v_ids) + weight * (incidence.T @ incidence).tocsc()
    rhs = (targets[v_ids] - rest[v_ids].astype(np.float64)) / len(v_ids)
    with warnings.catch_warnings():
        warnings.simplefilter('error', MatrixRankWarning)
        displacement = np.asarray(spsolve(matrix, rhs, permc_spec='NATURAL', use_umfpack=False), np.float64).reshape(len(v_ids), 3)
    info['calls_sparse_solve'] = 1
    if not np.isfinite(displacement).all():
        raise FloatingPointError('Nonfinite unique sparse solution; no retry')
    residual = matrix @ displacement - rhs
    info['linear_residual_max_abs'] = float(np.abs(residual).max())
    unconstrained = rest.astype(np.float64)
    unconstrained[v_ids] += displacement
    info['objective_unconstrained'] = _objective(unconstrained, rest, targets, variable, edges, cfg['regularization_lambda'])
    lengths = np.linalg.norm(displacement, axis=1)
    cap = cfg['maximum_displacement_R'] * R
    scale = np.minimum(1., cap / np.maximum(lengths, np.finfo(np.float64).tiny))
    displacement *= scale[:, None]
    info['vertices_capped'] = int(np.sum(lengths > cap))
    info['unconstrained_max_displacement'] = float(lengths.max())
    sf = post['source_face_ids']
    region_vids = {name: np.unique(faces[mask[sf]]) for name, mask in
                   (('P1', mapping['region_P1']), ('P2', mapping['region_P2']))}
    spans = {name: np.ptp(rest[ids].astype(np.float64), axis=0) if len(ids) else np.zeros(3)
             for name, ids in region_vids.items()}
    accepted = None
    for power in range(cfg['line_search_halvings'] + 1):
        alpha = 2. ** (-power)
        trial = rest.copy()
        trial[v_ids] = (rest[v_ids].astype(np.float64) + alpha * displacement).astype(np.float32)
        area, normal = _shape(trial, f)
        cosine = (normal * normal0).sum(axis=1)
        motion = np.linalg.norm(trial[v_ids].astype(np.float64) - rest[v_ids], axis=1)
        reasons = []
        if not np.isfinite(trial).all(): reasons.append('NONFINITE')
        if not _exact(trial[~variable], rest[~variable]): reasons.append('FIXED_VERTEX_CHANGED')
        if np.any(motion > cap): reasons.append('PER_VERTEX_DISPLACEMENT_CAP')
        if np.any(area < cfg['area_ratio_floor'] * area0): reasons.append('AREA_RATIO')
        if np.any(cosine < cfg['normal_cosine_floor']): reasons.append('NORMAL_COSINE')
        span_detail = {}
        for name, ids in region_vids.items():
            active = spans[name] > cfg['nonzero_span_epsilon_L'] * L
            after = np.ptp(trial[ids].astype(np.float64), axis=0) if len(ids) else np.zeros(3)
            if np.any(after[active] < cfg['region_span_ratio_floor'] * spans[name][active]):
                reasons.append(name + '_AXIAL_SPAN')
            span_detail[name] = dict(rest=spans[name].tolist(), trial=after.tolist(), active_axes=active.tolist())
        row = dict(alpha=alpha, reasons=reasons, passed=not reasons,
                   min_area_ratio=float((area / area0).min()), min_normal_cosine=float(cosine.min()),
                   max_actual_displacement=float(motion.max()), region_spans=span_detail)
        info['trials'].append(row)
        if not reasons:
            accepted = trial
            info['accepted_alpha'] = alpha
            break
    if accepted is None:
        accepted = rest.copy()
        info.update(status='NO_FEASIBLE_STEP_EXACT_FALLBACK', fallback=True)
    else:
        info['status'] = 'ACCEPTED' if not _exact(accepted, rest) else 'ZERO_MOTION_EXACT_LOCKED_RAW'
    out = {k: a.copy() for k, a in post.items()}
    out.update(vertices=accepted, output=accepted[faces], C=accepted[faces[:int(post['known_faces'])]],
               free=accepted[faces[int(post['known_faces']):]])
    _motion_metadata(out, assembly['pre'], variable)
    if not _exact(out['C'], post['C']) or not _exact(out['faces'], post['faces']):
        raise AssertionError('C or indexed topology changed')
    if not _exact(out['output'][~selected_faces], post['output'][~selected_faces]):
        raise AssertionError('Far D0 geometry changed during fusion')
    info['objective_after'] = _objective(accepted, rest, targets, variable, edges, cfg['regularization_lambda'])
    info['maximum_actual_displacement'] = float(np.linalg.norm(accepted.astype(float) - rest, axis=1).max())
    info['moved_vertices'] = int(np.any(accepted.view(np.uint32) != rest.view(np.uint32), axis=1).sum())
    new_mapping = {k: a.copy() for k, a in mapping.items()}
    new_mapping.update(unconstrained_vertices=unconstrained, capped_displacement=np.zeros(rest.shape, np.float64))
    new_mapping['capped_displacement'][v_ids] = displacement
    return dict(post=out, pre=assembly['pre'].copy(), mappings=new_mapping,
                summary={**assembly['summary'], 'solver': info,
                         'hard_lock_scope': 'Fusion BASE only; unchanged downstream HINGE/ALPHA/HOLE separately audited'})


def build_from_job(job):
    bundle_spec = job['input_bundle']
    path = _path(bundle_spec['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != bundle_spec['sha256']:
        raise ValueError('Frozen input bundle hash differs')
    bundle = _arrays(path)
    raw_meta = _arrays(job['paths']['raw'])
    if not _exact(raw_meta['raw'], bundle['raw']):
        raise ValueError('Canonical RAW identity differs')
    refs = job['references']
    result = assemble(_arrays(refs['D0_ALL']['base']), _arrays(refs['RAW_ALL']['base']),
                      bundle['raw'], bundle['d0_pre'], bundle['support_face_mask'],
                      {name: bundle['region_' + name] for name in ('P1', 'P2')}, raw_meta)
    label = job['label']
    if label == 'LOCAL_SHAPE':
        result = fit(result, job['operator_config'])
    elif label == 'LOCKED_RAW':
        result['summary']['solver'] = dict(status='CONTROL_NO_FIT', calls_sparse_solve=0, parameter_retries=0)
    else:
        raise ValueError('Unknown arm: ' + label)
    result['summary'].update(label=label, request_id=job['request']['request_id'],
                             no_detector_or_score_calls=True, new_GEO_calls=0, model_calls=0)
    return result
