"""Source-aware, CPU-only C restoration and fixed-C free-face orientation.

This module uses only NumPy and package-local ``winding_solver``. Source input
metadata suffices: no parent mesh, GT, experiment directory, or model is read.
The solver is a byte-for-byte copy of the reviewed two-incidence XOR solver;
SOURCE_SOLVER_PROVENANCE records its historical origin without importing it.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import numpy as np

from .winding_solver import orient_fixed_C

SCHEMA = 'meshflow_source_winding_v1'
SOURCE_SOLVER_PROVENANCE = {
    'source': 'r2/tools/c_restore_bounded_v1/orientation.py',
    'sha256': '7accffe4687d2837f83c2967e8cd64ffb6745da7131323937f1be3a16951c762',
    'port': 'Exact source bytes copied into package-local winding_solver.py; no runtime dependency on the historical path',
}
INPUT_CORNER_KEYS = (
    'corner_displacement', 'corner_displacement_vector', 'corner_target_input_ids',
    'corner_vertex_ids', 'corner_status', 'corner_C_candidate_count',
)
POST_REQUIRED_KEYS = frozenset((
    'output', 'vertices', 'faces', 'C', 'known_faces', 'input_faces',
    'source_face_ids', 'source_corner_ids', 'vertex_source_input_corner_ids',
    'source_face_kept', 'source_face_reason', 'duplicate_of_source_face',
    *INPUT_CORNER_KEYS,
))
POST_OPTIONAL_KEYS = frozenset(('free',))
REAL_VERTEX_ID_STATUSES = frozenset(('PROVIDED_SOURCE_IDS', 'LEGACY_PROVIDED_SOURCE_IDS'))
UNKNOWN_VERTEX_ID_STATUS = 'UNKNOWN_SYNTHETIC_PER_CORNER_IDS'


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _integers(value, shape, name):
    a = np.asarray(value)
    _require(a.shape == shape and np.issubdtype(a.dtype, np.integer),
             f'{name} must be an integer array with shape {shape}')
    return a


def _permutation(value, n, name='corner permutation'):
    p = _integers(value, (n, 3), name)
    _require(np.array_equal(np.sort(p, axis=1), np.tile(np.arange(3), (n, 1))),
             f'{name} must contain 0, 1, 2 exactly once in each row')
    return p


def _triangles(value, name):
    a = np.asarray(value)
    _require(a.ndim == 3 and a.shape[1:] == (3, 3) and a.dtype == np.float32
             and np.isfinite(a).all(), f'{name} must be finite FP32 [N,3,3]')
    return a


def _text(value, name):
    a = np.asarray(value)
    _require(a.shape == () and a.dtype.kind in 'US', f'{name} must be a scalar string')
    return str(a.item())


def _scalar_int(value, name):
    a = _integers(value, (), name)
    return int(a)


def permutation_parity(p):
    p = np.asarray(p)
    _require(p.ndim == 2, 'Permutation must have shape [N,3]')
    p = _permutation(p, len(p))
    return (((p[:, 0] > p[:, 1]).astype(np.int8)
             + (p[:, 0] > p[:, 2]).astype(np.int8)
             + (p[:, 1] > p[:, 2]).astype(np.int8)) % 2).astype(bool)


def permute_corner_data(data, output_to_input):
    """Reindex (N,3,...) by new-slot -> former-slot, preserving all value bits."""
    a = np.asarray(data)
    _require(a.ndim >= 2 and a.shape[1] == 3, 'Per-corner array must have shape [N,3,...]')
    p = _permutation(output_to_input, len(a))
    return a[np.arange(len(a))[:, None], p].copy()


def compose_corner_permutations(first, second):
    """After first and then second, the combined slot mapping is first[second]."""
    a = np.asarray(first)
    _require(a.ndim == 2, 'First permutation must have shape [N,3]')
    a = _permutation(a, len(a), 'first permutation')
    return permute_corner_data(a, _permutation(second, len(a), 'second permutation'))


def _consistent_vertex_ids(triangles, ids):
    """One real vertex ID cannot denote distinct coordinate bit patterns."""
    seen = {}
    for vertex_id, point in zip(ids.reshape(-1), triangles.reshape(-1, 3)):
        vertex_id = int(vertex_id)
        _require(vertex_id >= 0, 'Source vertex IDs must be nonnegative')
        value = point.tobytes()
        _require(vertex_id not in seen or seen[vertex_id] == value,
                 f'Source vertex ID {vertex_id} denotes inconsistent coordinate bytes')
        seen[vertex_id] = value


def _source_topology(ids):
    edges = defaultdict(list)
    for i, face in enumerate(ids):
        for j in range(3):
            u, v = int(face[j]), int(face[(j + 1) % 3])
            edges[tuple(sorted((u, v)))].append((i, u, v))
    two = [x for key, x in edges.items() if key[0] != key[1]
           and len(x) == 2 and x[0][0] != x[1][0]]
    return dict(two_incident_edges=len(two),
                same_direction_two_incident_edges=sum(x[0][1:] == x[1][1:] for x in two),
                nonmanifold_edges=sum(len(x) > 2 for x in edges.values()),
                boundary_edges=sum(len(x) == 1 for x in edges.values()),
                repeated_vertex_ID_faces=sum(len(set(map(int, f))) < 3 for f in ids),
                certification='Local source-ID edge observations, not outward/manifold/geometry certification')


def capture_source(C_original, source_vertex_ids=None, corner_permutation=None,
                   source_face_ids=None):
    """Explicitly capture source C before canonicalization.

    P[j] selects the original corner used for canonical corner j. Omitting P
    here explicitly captures identity; validation/export never invents P.
    Missing vertex IDs receive per-corner placeholders marked UNKNOWN, with no
    coordinate welding or claim that this recovers the input connectivity.
    The returned dictionary contains only arrays and can be saved in an NPZ.
    """
    original = _triangles(C_original, 'C_original')
    K = len(original)
    _require(K > 0, 'At least one source C face is required')
    p = (np.tile(np.arange(3, dtype=np.int64), (K, 1)) if corner_permutation is None
         else _permutation(corner_permutation, K).copy())
    if source_vertex_ids is None:
        ids = np.arange(K * 3, dtype=np.int64).reshape(K, 3)
        id_status = UNKNOWN_VERTEX_ID_STATUS
    else:
        ids = _integers(source_vertex_ids, (K, 3), 'source_vertex_ids').copy()
        _consistent_vertex_ids(original, ids)
        id_status = 'PROVIDED_SOURCE_IDS'
    if source_face_ids is None:
        face_ids = np.arange(K, dtype=np.int64)
        face_status = 'INPUT_FACE_ROW_IDS'
    else:
        face_ids = _integers(source_face_ids, (K,), 'source_face_ids').copy()
        _require(np.all(face_ids >= 0) and len(np.unique(face_ids)) == K,
                 'Source face IDs must be nonnegative and unique')
        face_status = 'PROVIDED_SOURCE_FACE_IDS'
    result = dict(
        source_winding_schema=np.asarray(SCHEMA),
        source_winding_provenance=np.asarray('CAPTURED_SOURCE_INPUT'),
        source_vertex_id_status=np.asarray(id_status), source_face_id_status=np.asarray(face_status),
        C=permute_corner_data(original, p), C_original=original.copy(),
        C_source_vertex_ids=permute_corner_data(ids, p), C_original_source_vertex_ids=ids,
        C_source_corner_permutations=p, C_original_source_face_ids=face_ids,
    )
    validate_source(result, result['C'])
    return result


def validate_source(source, C_canonical):
    """Validate supplied source arrays, distinguishing original from legacy.

    Historical C/P/canonical-source-ID metadata can be inverted without a full
    parent mesh. Its trust remains SAVED_MAPPING_ONLY; no independent original
    input authentication is implied. Missing original face IDs remain -1/NA.
    """
    _require(isinstance(source, Mapping), 'Source metadata must be a mapping of arrays')
    canonical = _triangles(C_canonical, 'C_canonical'); K = len(canonical)
    _require(K > 0, 'At least one C face is required')
    for key in ('C_source_corner_permutations', 'C_source_vertex_ids'):
        _require(key in source, f'Missing {key}; explicit source capture/mapping is required, no identity fallback')
    if 'source_winding_schema' in source:
        _require(_text(source['source_winding_schema'], 'source_winding_schema') == SCHEMA,
                 'Unsupported source winding schema')
    p = _permutation(source['C_source_corner_permutations'], K, 'C_source_corner_permutations')
    inverse = np.argsort(p, axis=1)
    ids = _integers(source['C_source_vertex_ids'], (K, 3), 'C_source_vertex_ids')
    _consistent_vertex_ids(canonical, ids)
    if 'C' in source:
        _require(_bits(source['C'], canonical), 'Saved canonical C does not match the current C bytes')
    restored_ids = permute_corner_data(ids, inverse)
    if 'C_original_source_vertex_ids' in source:
        explicit_ids = _integers(source['C_original_source_vertex_ids'], (K, 3), 'C_original_source_vertex_ids')
        _require(np.array_equal(restored_ids, explicit_ids), 'Original source vertex ID mapping mismatch')
    original = permute_corner_data(canonical, inverse)
    has_original = 'C_original' in source
    if has_original:
        explicit = _triangles(source['C_original'], 'C_original')
        _require(_bits(original, explicit), 'Original source C coordinate/order bytes do not match saved P')
        _require(_bits(permute_corner_data(explicit, p), canonical), 'Forward source corner mapping mismatch')
    id_status = (_text(source['source_vertex_id_status'], 'source_vertex_id_status')
                 if 'source_vertex_id_status' in source else 'LEGACY_PROVIDED_SOURCE_IDS')
    _require(id_status in REAL_VERTEX_ID_STATUSES or id_status == UNKNOWN_VERTEX_ID_STATUS,
             'Unsupported source_vertex_id_status')
    has_real_ids = id_status in REAL_VERTEX_ID_STATUSES
    provenance = (_text(source['source_winding_provenance'], 'source_winding_provenance')
                  if 'source_winding_provenance' in source else 'LEGACY_SAVED_CONDITION_MAPPING')
    face_keys = [k for k in ('C_original_source_face_ids', 'original_source_face_ids') if k in source]
    if face_keys:
        face_ids = _integers(source[face_keys[0]], (K,), face_keys[0]).copy()
        _require(np.all(face_ids >= 0) and len(np.unique(face_ids)) == K, 'Original source face IDs must be nonnegative and unique')
        for key in face_keys[1:]:
            _require(np.array_equal(face_ids, source[key]), 'Original source face ID aliases disagree')
        face_status = (_text(source['source_face_id_status'], 'source_face_id_status')
                       if 'source_face_id_status' in source else 'LEGACY_PROVIDED_SOURCE_FACE_IDS')
    else:
        face_ids = np.full(K, -1, np.int64); face_status = 'UNKNOWN_ORIGINAL_SOURCE_FACE_IDS'
    odd = permutation_parity(p); changed = np.any(inverse != np.arange(3), axis=1)
    trust = 'EXPLICIT_ORIGINAL_ARRAY_VALIDATED' if has_original else 'SAVED_MAPPING_ONLY'
    topology = _source_topology(restored_ids) if has_real_ids else {
        'status': 'UNKNOWN_SOURCE_VERTEX_IDS', 'reason': 'Per-corner placeholders do not encode original connectivity'}
    summary = dict(K=K, trust=trust, provenance=provenance,
        original_input_array_present=has_original, external_source_authentication='NOT_PERFORMED',
        has_real_source_vertex_ids=has_real_ids, source_vertex_id_status=id_status,
        source_face_id_status=face_status, C_rows_reordered=int(changed.sum()),
        C_winding_reversed_from_canonical=int(odd.sum()),
        C_cyclic_only_reordered=int((changed & ~odd).sum()), source_C_topology=topology)
    return dict(C=canonical.copy(), C_original=original, C_source_vertex_ids=ids.copy(),
        C_original_source_vertex_ids=restored_ids, C_original_source_face_ids=face_ids,
        C_source_corner_permutations=p.copy(), P=p.copy(), P_inverse=inverse,
        has_real_source_vertex_ids=has_real_ids, trust=trust, provenance=provenance,
        source_vertex_id_status=id_status, source_face_id_status=face_status,
        source_C_topology=topology, summary=summary)


def _validate_post(post):
    _require(isinstance(post, Mapping), 'POST arrays must be a mapping')
    unknown = set(post) - POST_REQUIRED_KEYS - POST_OPTIONAL_KEYS
    _require(not unknown, 'Unsupported POST attributes require an explicit mapping contract: ' + ', '.join(sorted(unknown)))
    missing = POST_REQUIRED_KEYS - set(post)
    _require(not missing, 'Missing standard GEO_C arrays: ' + ', '.join(sorted(missing)))
    a = {key: np.asarray(value) for key, value in post.items()}
    _require(not any(x.dtype.hasobject for x in a.values()), 'Object arrays are unsupported')
    v = a['vertices']; f = a['faces']; tri = _triangles(a['output'], 'POST output'); M = len(tri)
    _require(v.ndim == 2 and v.shape[1] == 3 and v.dtype == np.float32 and np.isfinite(v).all(), 'POST vertices must be finite FP32 [V,3]')
    _integers(f, (M, 3), 'POST faces')
    _require(np.all((f >= 0) & (f < len(v))), 'POST faces have invalid vertex IDs')
    _require(_bits(v[f], tri), 'POST indexed coordinates do not match output bytes')
    K = _scalar_int(a['known_faces'], 'known_faces'); N = _scalar_int(a['input_faces'], 'input_faces')
    _require(0 < K <= M <= N, 'Require 0 < K <= POST faces <= original input faces')
    _require(_bits(a['C'], tri[:K]), 'POST C differs from first K output faces')
    if 'free' in a:
        _require(_bits(a['free'], tri[K:]), 'POST free differs from remaining output faces')
    sf = _integers(a['source_face_ids'], (M,), 'source_face_ids')
    _require(np.all((sf >= 0) & (sf < N)) and len(np.unique(sf)) == M, 'Invalid POST source face IDs')
    _require(np.array_equal(sf[:K], np.arange(K)), 'POST C source face row order changed')
    sc = _permutation(a['source_corner_ids'], M, 'local source_corner_ids')
    _require(np.array_equal(sc[:K], np.tile(np.arange(3), (K, 1))), 'Canonical POST C source-corner map must be identity')
    shapes = {key: (N, 3, 3) if key == 'corner_displacement_vector' else (N, 3) for key in INPUT_CORNER_KEYS}
    for key, shape in shapes.items():
        _require(a[key].shape == shape, f'{key} must retain the original input-face domain {shape}')
    for key in ('corner_target_input_ids', 'corner_vertex_ids', 'corner_C_candidate_count'):
        _integers(a[key], (N, 3), key)
    _require(_bits(a['corner_vertex_ids'][sf[:, None], sc], f), 'Input corner vertex map does not reproduce indexed POST faces')
    _require(np.all((a['corner_target_input_ids'] >= 0) & (a['corner_target_input_ids'] < 3 * N)), 'Invalid original flattened input-corner target IDs')
    vertex_source = _integers(a['vertex_source_input_corner_ids'], (len(v),), 'vertex_source_input_corner_ids')
    _require(np.all((vertex_source >= 0) & (vertex_source < 3 * N)), 'Invalid representative input-corner IDs')
    keep = a['source_face_kept']
    _require(keep.shape == (N,) and keep.dtype == np.bool_, 'source_face_kept must be original-input-length Boolean array')
    _require(np.array_equal(np.sort(sf), np.flatnonzero(keep)), 'Kept source-face flags disagree with POST rows')
    _require(a['source_face_reason'].shape == (N,) and a['source_face_reason'].dtype.kind in 'US', 'Invalid source_face_reason array')
    _integers(a['duplicate_of_source_face'], (N,), 'duplicate_of_source_face')
    return a, K, N


def _C_vertex_ID_topology(original_ids, restored_C_faces, has_real_ids):
    if not has_real_ids:
        return dict(status='UNKNOWN_SOURCE_VERTEX_IDS',
                    reason='Capture did not provide actual source vertex IDs; placeholders are not a topology failure')
    source_to_post = defaultdict(set); post_to_source = defaultdict(set)
    for sid, pid in zip(original_ids.reshape(-1), restored_C_faces.reshape(-1)):
        source_to_post[int(sid)].add(int(pid)); post_to_source[int(pid)].add(int(sid))
    split = {str(s): sorted(v) for s, v in source_to_post.items() if len(v) != 1}
    merged = {str(p): sorted(v) for p, v in post_to_source.items() if len(v) != 1}
    return dict(status='CHANGED_BY_GEO_C' if split or merged else 'PASS',
                original_source_vertex_count=len(source_to_post), POST_C_vertex_count=len(post_to_source),
                split_source_ID_count=len(split), merged_POST_vertex_count=len(merged),
                split_source_IDs=split, merged_POST_vertex_source_IDs=merged,
                check='Original source vertex ID to restored indexed POST vertex is single-valued and injective; no topology repair performed')


def export_source_winding(post_arrays, source):
    """Restore C with its source P inverse, freeze it, and orient only free.

    Standard original-input-domain corner audits retain their original keys
    and values, with explicit input_* aliases. output_* projections follow the
    composed source face/corner mapping. ID *values* that refer to the original
    flattened input remain in that original domain. Unknown attributes fail
    rather than silently losing UVs, normals, colors, or their semantics.
    """
    a, K, N = _validate_post(post_arrays)
    known = validate_source(source, a['C']); f = a['faces']; v = a['vertices']; M = len(f)
    restore_map = np.tile(np.arange(3, dtype=np.int64), (M, 1)); restore_map[:K] = known['P_inverse']
    restored_faces = permute_corner_data(f, restore_map)
    _require(_bits(v[restored_faces[:K]], known['C_original']), 'Restored C differs from source input order')
    id_topology = _C_vertex_ID_topology(known['C_original_source_vertex_ids'], restored_faces[:K], known['has_real_source_vertex_ids'])

    def stage_arrays(faces, permutation, flips, stage):
        out = {key: value.copy() for key, value in a.items()}
        tri = v[faces]; sf = a['source_face_ids']; sc = permute_corner_data(a['source_corner_ids'], permutation)
        _require(_bits(faces, permute_corner_data(f, permutation)), 'Composed face permutation mismatch')
        _require(_bits(tri, permute_corner_data(a['output'], permutation)), 'Coordinate bits changed during reordering')
        _require(_bits(faces, a['corner_vertex_ids'][sf[:, None], sc]), 'Composed source corner map differs from faces')
        out.update(faces=faces.copy(), output=tri.copy(), C=tri[:K].copy(), free=tri[K:].copy(),
            source_corner_ids=sc, corner_permutation_to_MODEL_POST=permutation.copy(), free_face_flips=flips.copy(),
            C_original=known['C_original'].copy(), C_original_source_vertex_ids=known['C_original_source_vertex_ids'].copy(),
            C_original_source_face_ids=known['C_original_source_face_ids'].copy(),
            C_canonical_source_vertex_ids=known['C_source_vertex_ids'].copy(),
            C_source_corner_permutations=known['P'].copy(), C_inverse_corner_permutation=known['P_inverse'].copy(),
            source_winding_schema=np.asarray(SCHEMA), source_winding_stage=np.asarray(stage),
            source_winding_trust=np.asarray(known['trust']), source_winding_provenance=np.asarray(known['provenance']),
            source_vertex_id_status=np.asarray(known['source_vertex_id_status']),
            source_face_id_status=np.asarray(known['source_face_id_status']),
            source_C_vertex_ID_topology_status=np.asarray(id_topology['status']))
        for key in INPUT_CORNER_KEYS:
            out['input_' + key] = a[key].copy()
            out['output_' + key] = a[key][sf[:, None], sc].copy()
        return out

    restored_arrays = stage_arrays(restored_faces, restore_map, np.zeros(M, bool), 'RESTORED_C_ONLY')
    oriented = orient_fixed_C(v, restored_faces, K)
    total_map = compose_corner_permutations(restore_map, oriented['face_corner_permutation'])
    oriented_arrays = stage_arrays(oriented['faces'], total_map, oriented['flips'], 'ORIENTED')
    _require(_bits(oriented_arrays['C'], restored_arrays['C']), 'Orientation changed restored source C')
    _require(_bits(restored_arrays['free'], a['output'][K:]), 'C restoration changed free triangles')
    summary = dict(known['summary'], status='PASS', schema=SCHEMA, input_faces=N, output_faces=M,
        source_C_vertex_ID_topology=id_topology, source_C_order_fixed_after_restoration=True,
        vertices_bytewise_unchanged=True, unordered_face_vertex_IDs_unchanged=True,
        face_order_and_source_face_ids_unchanged=True, input_corner_audits_preserved=True,
        output_corner_attributes_composed=True, UV_normals_colors='NOT_ACCEPTED_WITHOUT_EXPLICIT_ATTRIBUTE_CONTRACT',
        original_flattened_input_ID_domain_preserved=True,
        canonical_C_order_changed=not _bits(a['C'], restored_arrays['C']),
        geometry_quality_status='NO_NEW_SCORING; geometric equivalence alone does not execute or replace an original scorer',
        source_solver_provenance=dict(SOURCE_SOLVER_PROVENANCE),
        model_calls=0, GEO_calls=0, quality_calls=0, orientation_solver_calls=1)
    return dict(restored_arrays=restored_arrays, oriented_arrays=oriented_arrays,
                restoration_summary=summary, orientation_result=oriented)
