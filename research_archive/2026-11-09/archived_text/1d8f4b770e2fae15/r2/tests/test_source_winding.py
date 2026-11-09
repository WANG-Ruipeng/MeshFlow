"""Portable CPU contracts: synthetic arrays only, no model or historical assets."""
import hashlib
import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from meshflow_control import source_winding as W
from meshflow_control import winding_solver


def bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def post_fixture(vertices, faces, K, source_face_ids=None, input_faces=None):
    v = np.asarray(vertices, np.float32); f = np.asarray(faces, np.int64); M = len(f)
    sf = np.arange(M, dtype=np.int64) if source_face_ids is None else np.asarray(source_face_ids, np.int64)
    N = M if input_faces is None else input_faces
    corners = np.full((N, 3), -1, np.int64); corners[sf] = f
    keep = np.zeros(N, bool); keep[sf] = True
    representatives = np.zeros(len(v), np.int64)
    seen = set()
    for i, source in enumerate(sf):
        for c, vertex in enumerate(f[i]):
            if int(vertex) not in seen:
                representatives[vertex] = 3 * source + c; seen.add(int(vertex))
    output = v[f]
    return dict(output=output.copy(), vertices=v.copy(), faces=f.copy(), C=output[:K].copy(),
        known_faces=np.int64(K), input_faces=np.int64(N), source_face_ids=sf,
        source_corner_ids=np.tile(np.arange(3, dtype=np.int64), (M, 1)),
        vertex_source_input_corner_ids=representatives,
        corner_displacement=np.zeros((N, 3), np.float64),
        corner_displacement_vector=np.zeros((N, 3, 3), np.float64),
        corner_target_input_ids=np.arange(3 * N, dtype=np.int64).reshape(N, 3),
        corner_vertex_ids=corners, corner_status=np.full((N, 3), 'synthetic', dtype='U32'),
        corner_C_candidate_count=np.zeros((N, 3), np.int64), source_face_kept=keep,
        source_face_reason=np.where(keep, 'kept', 'synthetic_deleted'),
        duplicate_of_source_face=np.full(N, -1, np.int64))


def square():
    v = np.asarray([[0., -0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]], np.float32)
    return v, np.asarray([[0, 1, 2], [1, 2, 3]], np.int64)


@pytest.mark.parametrize('P', list(itertools.permutations(range(3))))
def test_all_six_inverse_permutations_restore_exact_source_IDs_and_order(P):
    v, f = square(); original = v[f[:1]].copy(); permutation = np.asarray([P], np.int64)
    source = W.capture_source(original, np.asarray([[10, 20, 30]]), permutation, np.asarray([77]))
    canonical_faces = f.copy(); canonical_faces[0] = f[0, permutation[0]]
    post = post_fixture(v, canonical_faces, 1)
    before = {k: a.tobytes() for k, a in post.items()}
    result = W.export_source_winding(post, source); restored = result['restored_arrays']; final = result['oriented_arrays']
    assert bits(restored['C'], original) and bits(final['C'], original)
    assert np.array_equal(restored['C_original_source_vertex_ids'], [[10, 20, 30]])
    assert np.array_equal(restored['C_original_source_face_ids'], [77])
    assert np.array_equal(restored['C_inverse_corner_permutation'], np.argsort(permutation, axis=1))
    assert bits(restored['free'], post['output'][1:])
    assert bits(final['vertices'], post['vertices'])
    assert np.array_equal(np.sort(final['faces'], axis=1), np.sort(post['faces'], axis=1))
    assert result['restoration_summary']['source_C_vertex_ID_topology']['status'] == 'PASS'
    assert result['orientation_result']['summary']['opposite_C_F_edges_after'] == 1
    assert all(a.tobytes() == before[k] for k, a in post.items())
    assert np.signbit(final['C'][0, 0, 1])


def test_three_cycle_is_inverted_and_not_reapplied():
    p = np.asarray([[1, 2, 0], [2, 0, 1]])
    inverse = np.argsort(p, axis=1)
    assert not np.array_equal(p, inverse)
    assert np.array_equal(W.compose_corner_permutations(p, inverse), [[0, 1, 2], [0, 1, 2]])
    assert not W.permutation_parity(p).any()


def test_unknown_connectivity_stays_unknown_and_capture_identity_is_explicit():
    v, f = square(); post = post_fixture(v, f, 1)
    source = W.capture_source(post['C'])
    assert bits(source['C'], post['C'])
    validated = W.validate_source(source, post['C'])
    assert not validated['has_real_source_vertex_ids']
    assert validated['source_vertex_id_status'] == W.UNKNOWN_VERTEX_ID_STATUS
    result = W.export_source_winding(post, source)
    assert result['restoration_summary']['source_C_vertex_ID_topology']['status'] == 'UNKNOWN_SOURCE_VERTEX_IDS'


def test_legacy_saved_mapping_has_limited_trust_without_parent_or_GT():
    v, f = square(); P = np.asarray([[1, 2, 0]])
    source = W.capture_source(v[f[:1]], f[:1], P)
    legacy = {k: source[k] for k in ('C', 'C_source_corner_permutations', 'C_source_vertex_ids')}
    checked = W.validate_source(legacy, source['C'])
    assert checked['trust'] == 'SAVED_MAPPING_ONLY'
    assert checked['has_real_source_vertex_ids']
    assert checked['summary']['external_source_authentication'] == 'NOT_PERFORMED'
    assert np.array_equal(checked['C_original_source_face_ids'], [-1])
    canonical_faces = f.copy(); canonical_faces[0] = canonical_faces[0, P[0]]
    result = W.export_source_winding(post_fixture(v, canonical_faces, 1), legacy)
    assert bits(result['oriented_arrays']['C'], v[f[:1]])


@pytest.mark.parametrize('missing', ['C_source_corner_permutations', 'C_source_vertex_ids'])
def test_missing_mapping_never_gets_an_implicit_identity(missing):
    v, f = square(); source = W.capture_source(v[f[:1]], f[:1]); del source[missing]
    with pytest.raises(ValueError, match='no identity fallback'):
        W.validate_source(source, v[f[:1]])


def test_wrong_source_ID_and_bad_original_bytes_are_rejected():
    v, f = square(); source = W.capture_source(v[f[:1]], f[:1])
    # Vertex ID is wrong even if its coordinate would be an identical copy.
    source['C_original_source_vertex_ids'][0, 0] = 999
    with pytest.raises(ValueError, match='Original source vertex ID mapping mismatch'):
        W.validate_source(source, source['C'])
    source = W.capture_source(v[f[:1]], f[:1]); source['C_original'][0] = source['C_original'][0, [0, 2, 1]]
    with pytest.raises(ValueError, match='Original source C coordinate/order bytes'):
        W.validate_source(source, source['C'])


def test_one_real_vertex_ID_cannot_refer_to_two_different_coordinates():
    v, f = square()
    with pytest.raises(ValueError, match='inconsistent coordinate bytes'):
        W.capture_source(v[f[:1]], np.asarray([[0, 0, 2]]))


def test_source_original_C_conflict_is_reported_without_relaxing_C():
    v, f = square(); post = post_fixture(v, f, 2)
    source = W.capture_source(post['C'], f)
    result = W.export_source_winding(post, source)
    assert result['restoration_summary']['source_C_topology']['same_direction_two_incident_edges'] == 1
    assert result['orientation_result']['summary']['fixed_C_anchor_conflict_components'] == 1
    assert not result['orientation_result']['flips'].any()
    assert bits(result['oriented_arrays']['C'], source['C_original'])


def test_parity_cycle_conflict_is_retained_not_deleted_or_hidden():
    vertices = []
    for i in range(3):
        theta = 2 * np.pi * i / 3
        for w in (-.25, .25):
            r = 1 + w * np.cos(theta / 2)
            vertices.append([r * np.cos(theta), r * np.sin(theta), w * np.sin(theta / 2)])
    faces = []
    for i in range(3):
        a, b = 2 * i, 2 * i + 1
        an, bn = (2 * (i + 1), 2 * (i + 1) + 1) if i < 2 else (1, 0)
        faces.extend([[a, b, an], [b, bn, an]])
    post = post_fixture(vertices, faces, 1)
    result = W.export_source_winding(post, W.capture_source(post['C'], post['faces'][:1]))
    summary = result['orientation_result']['summary']
    assert summary['parity_conflict_components'] == 1
    assert summary['fixed_C_anchor_status_not_evaluated_due_to_parity_components'] == 1
    assert not result['orientation_result']['flips'].any()
    assert bits(result['oriented_arrays']['faces'], post['faces'])


def test_nonmanifold_source_edges_are_reported_not_certified_or_fixed():
    v = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]], np.float32)
    f = np.asarray([[0, 1, 2], [1, 0, 3], [0, 1, 4]], np.int64)
    post = post_fixture(v, f, 3); result = W.export_source_winding(post, W.capture_source(post['C'], f))
    assert result['restoration_summary']['source_C_topology']['nonmanifold_edges'] == 1
    assert result['orientation_result']['summary']['nonmanifold_edges'] == 1
    assert not result['orientation_result']['summary']['outward_orientation_certified']
    assert bits(result['oriented_arrays']['C'], post['C'])


def test_GEO_merged_distinct_true_source_IDs_is_explicitly_reported():
    v, _ = square(); f = np.asarray([[0, 1, 2], [1, 0, 3]], np.int64)
    post = post_fixture(v, f, 2)
    # Original source has six separate vertices with some equal coordinates.
    source = W.capture_source(post['C'], np.arange(6).reshape(2, 3))
    result = W.export_source_winding(post, source); status = result['restoration_summary']['source_C_vertex_ID_topology']
    assert status['status'] == 'CHANGED_BY_GEO_C' and status['merged_POST_vertex_count'] == 2
    assert status['split_source_ID_count'] == 0
    assert bits(result['oriented_arrays']['vertices'], post['vertices'])


def test_GEO_split_one_true_source_ID_is_reported_without_rewelding():
    v, _ = square(); v = np.concatenate((v, v[[0, 1]]))
    f = np.asarray([[0, 1, 2], [5, 4, 3]], np.int64)
    post = post_fixture(v, f, 2)
    original_ids = np.asarray([[0, 1, 2], [1, 0, 3]])
    result = W.export_source_winding(post, W.capture_source(post['C'], original_ids))
    status = result['restoration_summary']['source_C_vertex_ID_topology']
    assert status['status'] == 'CHANGED_BY_GEO_C' and status['split_source_ID_count'] == 2
    assert status['merged_POST_vertex_count'] == 0


def test_input_N_corner_audits_survive_deleted_faces_and_map_to_output_M():
    v, f = square(); v = np.concatenate((v, v + np.asarray([3, 0, 0], np.float32)))
    f = np.concatenate((f, f[:1] + 4))
    P = np.asarray([[1, 2, 0]]); source = W.capture_source(v[f[:1]], f[:1], P)
    f[0] = f[0, P[0]]
    post = post_fixture(v, f, 1, source_face_ids=[0, 2, 3], input_faces=4)
    post['corner_displacement_vector'] = np.arange(36, dtype=np.float64).reshape(4, 3, 3)
    post['corner_displacement_vector'][0, 0, 0] = -0.
    post['corner_C_candidate_count'] = np.arange(12, dtype=np.int64).reshape(4, 3)
    result = W.export_source_winding(post, source)
    for final in (result['restored_arrays'], result['oriented_arrays']):
        sf, sc = final['source_face_ids'], final['source_corner_ids']
        for key in W.INPUT_CORNER_KEYS:
            assert bits(final[key], post[key]) and bits(final['input_' + key], post[key])
            assert bits(final['output_' + key], post[key][sf[:, None], sc])
        assert bits(final['vertex_source_input_corner_ids'], post['vertex_source_input_corner_ids'])
        assert bits(final['faces'], post['corner_vertex_ids'][sf[:, None], sc])
        assert len(final['corner_status']) == 4 and len(final['output_corner_status']) == 3


@pytest.mark.parametrize('name', ['uv', 'normals', 'corner_color', 'mystery_corner_attribute'])
def test_unknown_attributes_fail_instead_of_disappearing(name):
    v, f = square(); post = post_fixture(v, f, 1); source = W.capture_source(post['C'], f[:1])
    post[name] = np.zeros((len(f), 3, 2), np.float32)
    with pytest.raises(ValueError, match='Unsupported POST attributes'):
        W.export_source_winding(post, source)


def test_wrong_post_corner_identity_is_rejected_before_orientation():
    v, f = square(); post = post_fixture(v, f, 1); source = W.capture_source(post['C'], f[:1])
    post['corner_vertex_ids'][0, 0] = 3
    with pytest.raises(ValueError, match='Input corner vertex map'):
        W.export_source_winding(post, source)


def test_solver_is_the_exact_reviewed_source_and_import_is_standalone_CPU():
    assert hashlib.sha256(Path(winding_solver.__file__).read_bytes()).hexdigest() == W.SOURCE_SOLVER_PROVENANCE['sha256']
    code = ('import sys; from meshflow_control import source_winding;'
            "assert 'torch' not in sys.modules;"
            "assert not any(x=='native_t1' or x.startswith('native_t1.') or x.startswith('r2.tools.') for x in sys.modules)")
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
