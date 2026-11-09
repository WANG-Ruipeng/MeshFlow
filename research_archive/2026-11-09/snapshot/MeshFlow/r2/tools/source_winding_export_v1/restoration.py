"""Restore saved source corner order at export, without changing any geometry.

P[i,j] is the original corner selected by canonical model corner j:
    canonical[i,j] = original[i,P[i,j]].
The export therefore uses argsort(P), including inverse cyclic permutations.
No model, mesh repair, source lookup, I/O, or orientation solver is called here.
"""
from __future__ import annotations
import numpy as np

VERSION = 'saved_source_corner_inverse_export_v1'


def definition():
    return dict(version=VERSION,
        source='Explicit saved original face IDs, original_source_faces, and C_source_corner_permutations; no score-based C direction selection',
        mapping='canonical_corner[j]=original_corner[P[j]]; restored_corner[j]=canonical_corner[argsort(P)[j]]',
        changes='Only first K face rows receive their exact per-face inverse corner permutation; all free face rows unchanged at this stage',
        identity='Source face row order and source vertex IDs are explicitly checked, not inferred from coincident coordinates',
        coordinates='All vertex-array bytes unchanged; original source C order in existing model coordinates restored exactly',
        corner_attributes='Apply the same output-slot-to-input-slot permutation to every saved per-corner attribute; compose subsequent free reversals',
        limitations='Changes the old ordered-canonical-C export contract; does not move vertices, certify outward normals/manifoldness, or improve geometric attachment',
        model_calls=0, GEO_calls=0, quality_calls=0)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _integer(a, shape, name):
    a = np.asarray(a)
    _require(a.shape == shape and np.issubdtype(a.dtype, np.integer),
             name + ' must be an integer array of shape ' + str(shape))
    return a


def _bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _permutation(p, n, name='corner permutation'):
    p = _integer(p, (n, 3), name)
    _require(np.array_equal(np.sort(p, axis=1), np.tile(np.arange(3), (n, 1))),
             name + ' must contain each of 0,1,2 exactly once per face')
    return p


def permutation_parity(p):
    """Return one Boolean per face: True means a reversal, False a cyclic order."""
    p = np.asarray(p)
    _require(p.ndim == 2, 'Permutation must be N by 3')
    p = _permutation(p, len(p))
    return (((p[:, 0] > p[:, 1]).astype(np.int8)
             + (p[:, 0] > p[:, 2]).astype(np.int8)
             + (p[:, 1] > p[:, 2]).astype(np.int8)) % 2).astype(bool)


def permute_corner_data(data, output_to_input):
    """Reindex axis 1 of (N,3,...) without arithmetic, preserving exact values."""
    a = np.asarray(data)
    _require(a.ndim >= 2 and a.shape[1] == 3, 'Per-corner data must have shape (N,3,...)')
    p = _permutation(output_to_input, len(a))
    return a[np.arange(len(a))[:, None], p].copy()


def compose_corner_permutations(first, second):
    """Map final slots to original slots after applying first and then second."""
    first = np.asarray(first)
    _require(first.ndim == 2, 'First permutation must have shape (N,3)')
    first = _permutation(first, len(first), 'first permutation')
    second = _permutation(second, len(first), 'second permutation')
    return permute_corner_data(first, second)


def validate_source_lineage(condition, parent):
    """Validate every mapping against the original faces retained in the parent.

    Inputs are read-only mappings of already-loaded NPZ arrays. File identity and
    official-original-asset hashes are the caller's separate registration duty.
    """
    needed = ('C', 'source_face_ids', 'original_source_face_ids',
              'C_source_vertex_ids', 'C_source_corner_permutations')
    parent_needed = ('full_target', 'model_vertices', 'original_source_faces',
                     'retained_to_original_face', 'source_faces',
                     'source_face_vertex_ids', 'source_corner_permutations')
    _require(all(k in condition for k in needed), 'Missing condition lineage array')
    _require(all(k in parent for k in parent_needed), 'Missing parent lineage array')
    C = np.asarray(condition['C'])
    _require(C.ndim == 3 and C.shape[1:] == (3, 3) and C.dtype == np.float32
             and len(C) > 0 and np.isfinite(C).all(), 'C must be finite FP32 (K,3,3)')
    K = len(C)
    model = np.asarray(parent['full_target'])
    _require(model.ndim in (2, 3) and model.shape[1:] in ((9,), (3, 3))
             and model.dtype == np.float32, 'Parent full_target must be FP32 (N,9) or (N,3,3)')
    model = model.reshape(-1, 3, 3); n = len(model)
    pvertices = np.asarray(parent['model_vertices'])
    _require(pvertices.ndim == 2 and pvertices.shape[1] == 3 and pvertices.dtype == np.float32
             and np.isfinite(pvertices).all(), 'Parent model_vertices must be finite FP32 (V,3)')
    original = np.asarray(parent['original_source_faces'])
    _require(original.ndim == 2 and original.shape[1] == 3
             and np.issubdtype(original.dtype, np.integer)
             and np.all((original >= 0) & (original < len(pvertices))), 'Invalid original source vertex IDs')
    retained = _integer(parent['retained_to_original_face'], (n,), 'retained_to_original_face')
    _require(len(np.unique(retained)) == n and np.all((retained >= 0) & (retained < len(original))),
             'Retained original face IDs must be a valid injective map')
    source_faces = _integer(parent['source_faces'], (n, 3), 'parent source_faces')
    pids = _integer(parent['source_face_vertex_ids'], (n, 3), 'parent canonical source vertex IDs')
    pp = _permutation(parent['source_corner_permutations'], n, 'parent source corner permutations')
    _require(np.array_equal(source_faces, original[retained]), 'Parent retained face lineage mismatch')
    _require(np.array_equal(permute_corner_data(source_faces, pp), pids),
             'Parent canonical source vertex IDs disagree with saved permutation')
    _require(_bits(pvertices[pids], model), 'Parent model coordinates disagree with source vertex IDs')
    selected = _integer(condition['source_face_ids'], (K,), 'condition source_face_ids')
    _require(len(np.unique(selected)) == K and np.all((selected >= 0) & (selected < n)),
             'Condition source face IDs must be a valid injective map')
    ofids = _integer(condition['original_source_face_ids'], (K,), 'condition original face IDs')
    ids = _integer(condition['C_source_vertex_ids'], (K, 3), 'condition canonical source vertex IDs')
    P = _permutation(condition['C_source_corner_permutations'], K, 'condition source corner permutations')
    _require(np.array_equal(ofids, retained[selected]), 'Condition original face ID lineage mismatch')
    _require(np.array_equal(P, pp[selected]), 'Condition corner permutation differs from parent lineage')
    _require(np.array_equal(ids, pids[selected]), 'Condition source vertex ID mismatch')
    _require(_bits(C, model[selected]) and _bits(C, pvertices[ids]), 'Condition C bytes differ from parent model order')
    if 'K' in condition:
        _require(np.asarray(condition['K']).shape == () and int(condition['K']) == K, 'Condition K mismatch')
    if 'N' in condition:
        _require(np.asarray(condition['N']).shape == () and int(condition['N']) == n, 'Condition N mismatch')
    inverse = np.argsort(P, axis=1)
    original_ids = original[ofids]
    recovered_ids = permute_corner_data(ids, inverse)
    _require(np.array_equal(recovered_ids, original_ids), 'Inverse must recover actual original source vertex IDs')
    original_C = permute_corner_data(C, inverse)
    _require(_bits(original_C, pvertices[original_ids]), 'Recovered original source C coordinates do not match exactly')
    parity = permutation_parity(P)
    rows = [dict(C_face_row=i, parent_prepared_face_id=int(selected[i]),
                 original_source_face_id=int(ofids[i]),
                 canonical_source_vertex_ids=ids[i].tolist(), original_source_vertex_ids=original_ids[i].tolist(),
                 canonical_to_original_corner_ids=P[i].tolist(), restored_to_canonical_corner_ids=inverse[i].tolist(),
                 corner_row_changes=bool(np.any(inverse[i] != np.arange(3))),
                 winding_reversed_from_canonical=bool(parity[i])) for i in range(K)]
    return dict(C_original=original_C, C_original_source_vertex_ids=original_ids.copy(),
                C_original_source_face_ids=ofids.copy(), C_source_vertex_ids=ids.copy(),
                P=P.copy(), P_inverse=inverse, parity=parity, face_rows=rows,
                summary=dict(K=K, parent_N=n, source_face_order_preserved=True,
                    original_source_vertex_IDs_exact=True, original_C_model_coordinate_bytes_exact=True,
                    C_rows_reordered=sum(r['corner_row_changes'] for r in rows),
                    C_winding_reversed_from_canonical=int(parity.sum()),
                    C_cyclic_only_reordered=sum(r['corner_row_changes'] and not r['winding_reversed_from_canonical'] for r in rows)))


def restore_source_order(vertices, faces, K, condition, parent, *,
                         source_face_ids=None, source_corner_ids=None):
    """Restore original C order in one cached canonical POST; do not orient free.

    The canonical POST must retain C as its first K faces in the registered
    order. source_corner_ids, if supplied, maps output corners into the former
    input and is composed rather than regenerated. Other per-corner fields use
    the returned face_corner_permutation and permute_corner_data().
    """
    v, f = np.asarray(vertices), np.asarray(faces)
    _require(v.ndim == 2 and v.shape[1] == 3 and v.dtype == np.float32 and np.isfinite(v).all(),
             'POST vertices must be finite FP32 (V,3)')
    _require(f.ndim == 2 and f.shape[1] == 3 and np.issubdtype(f.dtype, np.integer),
             'POST faces must be integer (N,3)')
    _require(isinstance(K, (int, np.integer)) and not isinstance(K, (bool, np.bool_))
             and 0 < K <= len(f), 'K must be an integer in 1..N')
    K = int(K)
    _require(np.all((f >= 0) & (f < len(v))), 'POST face vertex IDs out of range')
    lineage = validate_source_lineage(condition, parent)
    _require(lineage['summary']['K'] == K, 'POST K differs from source condition')
    canonical_C = np.asarray(condition['C'])
    _require(_bits(v[f[:K]], canonical_C), 'POST C is not in exact registered canonical coordinate order')
    permutation = np.tile(np.arange(3, dtype=np.int64), (len(f), 1))
    permutation[:K] = lineage['P_inverse']
    restored = permute_corner_data(f, permutation)
    _require(_bits(restored[K:], f[K:]), 'Free rows changed during C restoration')
    _require(_bits(v[restored[:K]], lineage['C_original']), 'Restored indexed C differs from original source order')
    _require(np.array_equal(np.sort(restored, axis=1), np.sort(f, axis=1)), 'Unordered indexed triangle identity changed')
    _require(_bits(permute_corner_data(restored[:K], lineage['P']), f[:K]), 'Indexed inverse round trip failed')
    result = dict(faces=restored, C=v[restored[:K]].copy(),
                  face_corner_permutation=permutation,
                  C_original_source_vertex_ids=lineage['C_original_source_vertex_ids'],
                  C_original_source_face_ids=lineage['C_original_source_face_ids'],
                  C_canonical_source_vertex_ids=lineage['C_source_vertex_ids'],
                  P=lineage['P'], P_inverse=lineage['P_inverse'],
                  face_rows=lineage['face_rows'], summary=dict(lineage['summary'],
                    version=VERSION, total_faces=len(f), free_faces=len(f)-K,
                    vertices_bytewise_unchanged=True, free_face_rows_bytewise_unchanged=True,
                    unordered_face_vertex_IDs_unchanged=True, face_order_unchanged=True,
                    canonical_C_order_changed=not _bits(v[restored[:K]], canonical_C),
                    per_corner_mapping='new output slot -> original cached POST corner slot',
                    automatic_extra_attributes=False, model_calls=0, orientation_solver_calls=0))
    if source_face_ids is not None:
        sf = _integer(source_face_ids, (len(f),), 'POST source_face_ids')
        _require(np.array_equal(sf[:K], np.arange(K)), 'POST source face IDs do not retain original C row order')
        result['source_face_ids'] = sf.copy()
    if source_corner_ids is not None:
        sc = _permutation(source_corner_ids, len(f), 'POST local source_corner_ids')
        _require(np.array_equal(sc[:K], np.tile(np.arange(3), (K, 1))),
                 'Canonical POST C source-corner map must be identity before restoration')
        result['source_corner_ids'] = permute_corner_data(sc, permutation)
    return result
