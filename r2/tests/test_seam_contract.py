"""SEAM CPU contracts: synthetic geometry/OT and toy FM autograd only.

No checkpoint, source dataset, occupancy-head forward, generator, D0, optimizer
or CUDA call is needed. These checks do not certify learned generation quality.
"""
import ast
import inspect
import sys

import numpy as np
import pytest

from meshflow_control.seam import corner_contract as contract
from meshflow_control.seam import slot_data
from meshflow_control.source_winding import capture_source, permute_corner_data


def square_C():
    vertices = np.array([[0., 0., 0.], [2., 0., 0.],
                         [2., 2., 0.], [0., 2., 0.]], np.float32)
    original = np.array([[0, 1, 2], [0, 2, 3]], np.int64)
    # Deliberately change cyclic order and reverse one model face. C boundary
    # winding must still be determined by the original input, not this table.
    model = np.array([[1, 2, 0], [3, 2, 0]], np.int64)
    return vertices, original, model


def test_boundary_and_features_depend_on_C_not_model_corner_order():
    vertices, original, model = square_C()
    edge, features = slot_data.edge_features(vertices, original, 128)
    assert edge.shape == (4, 3) and features.shape == (4, 24)
    assert {tuple(row[1:]) for row in edge} == {(0, 1), (3, 0), (1, 2), (2, 3)}
    assert not any(set(row[1:]) == {0, 2} for row in edge)
    layout = slot_data.inference_layout(vertices, original, model, 128,
                                       np.array([True, False, True, False]))
    np.testing.assert_array_equal(layout['edge_records'], edge)
    np.testing.assert_array_equal(layout['features'], features)
    np.testing.assert_array_equal(layout['C_faces_original'], original)
    np.testing.assert_array_equal(layout['anchor_ids'][:2], model)
    assert layout['N'] == 128 and layout['valid_mask'].sum() == 128
    assert len(layout['seam_slots']) == 2 and len(layout['bulk_slots']) == 124
    assert layout['free_coordinate_count'] == 9 * 126 - 6 * 2
    for ei, (_, u, v) in enumerate(edge):
        actual = layout['anchor_ids'][2 + ei]
        expected = [v, u, -1] if layout['occupied'][ei] else [-1, -1, -1]
        np.testing.assert_array_equal(actual, expected)


def test_null_candidates_remain_real_faces_and_keep_total_N():
    vertices, original, model = square_C()
    layout = slot_data.inference_layout(vertices, original, model, 128,
                                       np.zeros(4, bool))
    assert len(layout['seam_slots']) == 0
    np.testing.assert_array_equal(layout['bulk_slots'], np.arange(2, 128))
    assert layout['free_coordinate_count'] == 9 * 126
    assert layout['valid_mask'].all()
    with pytest.raises(ValueError, match='bool'):
        slot_data.inference_layout(vertices, original, model, 128, np.zeros(4))


@pytest.mark.parametrize('field', ['Y', 'parent_id', 'occupied_labels'])
def test_inference_rejects_hidden_source_fields_before_head_access(field):
    from meshflow_control.seam.inference import predict_layout
    vertices, original, model = square_C()
    bundle = dict(C_vertices=vertices, C_faces_original=original,
                  C_faces_model=model, total_N=128)
    bundle[field] = np.zeros(1)
    # None has no parameters or forward: rejection must precede any head use.
    with pytest.raises(ValueError, match='exactly C'):
        predict_layout(None, bundle)


def tiny_corner_case():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], np.float32)
    known = np.array([True, False, False, False])
    valid = np.array([True, True, True, False])
    cids = np.full((4, 3), -1, np.int64)
    cids[0] = [0, 1, 2]
    endpoints = np.array([[-1, -1], [1, 0], [2, 1], [-1, -1]], np.int64)
    ids = contract.build_anchor_ids(known, valid, cids, endpoints,
                                    np.array([False, True, False, False]))
    x = np.zeros((4, 3, 3), np.float32)
    x[0] = vertices
    x[1] = [[1., 0., 0.], [0., 0., 0.], [.5, -.5, .5]]
    x[2] = [[0., 0., 0.], [3., 1., 0.], [3., 0., 1.]]
    return vertices, known, valid, ids, x


def test_shared_decode_keeps_anchor_IDs_and_does_not_silently_weld_bulk():
    vertices, known, valid, ids, x = tiny_corner_case()
    decoded = contract.decode_shared_indices(x, known, valid, ids, vertices)
    assert len(decoded['faces']) == 3  # Null face survives; padding does not.
    np.testing.assert_array_equal(decoded['faces'][1, :2], [1, 0])
    assert decoded['faces'][2, 0] >= len(vertices)  # Same xyz, different identity.
    np.testing.assert_array_equal(decoded['vertices'][decoded['faces']], x[valid])
    assert (decoded['corner_to_vertex'][3] == -1).all()


def test_known_corner_time_is_distinct_from_seam_face_attention_time():
    vertices, known, valid, ids, x = tiny_corner_case()
    times = contract.corner_conditioning(.25, known, valid, ids)
    np.testing.assert_array_equal(times['corner_time'][1], [1., 1., .25])
    np.testing.assert_array_equal(times['corner_time'][2], [.25, .25, .25])
    assert times['face_attention_time'][1] == .25
    assert not times['corner_time'][3].any()
    velocity = np.ones_like(x)
    out = contract.euler_step(x, velocity, .02, known, valid, ids, vertices)
    free = times['free_corner_mask']
    assert out[~free].tobytes() == x[~free].tobytes()
    np.testing.assert_array_equal(out[free], x[free] + np.float32(.02))


def test_original_corner_order_roundtrip_uses_saved_input_mapping():
    vertices, original, model = square_C()
    permutation = np.array([[1, 2, 0], [2, 1, 0]], np.int64)
    saved = capture_source(vertices[original], source_vertex_ids=original,
                           corner_permutation=permutation)
    assert saved['C'].tobytes() == vertices[model].tobytes()
    inverse = np.argsort(saved['C_source_corner_permutations'], axis=1)
    restored = permute_corner_data(saved['C'], inverse)
    restored_ids = permute_corner_data(saved['C_source_vertex_ids'], inverse)
    assert restored.tobytes() == vertices[original].tobytes()
    np.testing.assert_array_equal(restored_ids, original)


def test_FM_has_only_free_scalar_gradient_and_effective_batch_denominator():
    from meshflow_control import runtime as stable  # establishes policy first
    import torch
    from meshflow_control.seam.model_adapter import corner_masked_fm_loss
    vertices, known, valid, ids, x = tiny_corner_case()
    prediction = torch.arange(1, 37, dtype=torch.float32).reshape(1, 4, 9).requires_grad_()
    target = torch.zeros_like(prediction)
    valid_t = torch.from_numpy(valid[None])
    fixed_t = torch.from_numpy((ids >= 0)[None])
    mask = (valid_t[..., None] & ~fixed_t)[..., None].expand(1, 4, 3, 3).reshape_as(prediction)
    assert int(mask.sum()) == 12  # 3 seam-tip scalars + 9 returned-Null scalars.
    loss = corner_masked_fm_loss(prediction, target, valid_t, fixed_t)
    loss.backward()
    assert torch.equal(prediction.grad[~mask], torch.zeros_like(prediction.grad[~mask]))
    torch.testing.assert_close(prediction.grad[mask], 2 * prediction.detach()[mask] / 12)
    assert torch.equal(loss.detach(), prediction.detach()[mask].square().sum() / 12)
    half = corner_masked_fm_loss(prediction, target, valid_t, fixed_t, denominator=24)
    assert torch.equal(half.detach(), loss.detach() / 2)
    with pytest.raises(ValueError, match='denominator'):
        corner_masked_fm_loss(prediction, target, valid_t, fixed_t, denominator=11)
    assert not torch.cuda.is_initialized()


def synthetic_supervised_case():
    vertices, original, model = square_C()
    rng = np.random.RandomState(19)
    full = rng.uniform(3., 8., (128, 3, 3)).astype(np.float32)
    full[:2] = vertices[model]
    # One source owner for boundary edge (0,1); its target corner order differs
    # from the seam slot, forcing explicit identity-based owner permutation.
    full[2] = np.array([[1., -.5, -.5], vertices[1], vertices[0]], np.float32)
    source_ids = np.arange(5, 5 + 128 * 3, dtype=np.int64).reshape(128, 3)
    source_ids[:2] = model
    source_ids[2] = [4, 1, 0]
    case = dict(full_target=full, pre_ot_full=full * np.float32(.5),
                source_face_vertex_ids=source_ids, known_ids=np.arange(2, dtype=np.int64),
                free_ids=np.arange(2, 128, dtype=np.int64),
                occupied_labels=np.array([True, False, False, False]),
                edge_owner_face=np.array([2, -1, -1, -1]),
                edge_source_triple=np.array([[1, 0, 4], [-1, -1, -1],
                                             [-1, -1, -1], [-1, -1, -1]], np.int64),
                C_source_vertex_ids=np.arange(4, dtype=np.int64),
                C_vertices=vertices, C_faces_original=original, C_faces_model=model)
    return case, rng.standard_normal((126, 3, 3)).astype(np.float32), rng.standard_normal((126, 3, 3)).astype(np.float32)


@pytest.mark.parametrize('arm,expected_bulk,expected_scalars', [
    ('SEAM_SLOT_C200', 125, 1128), ('FULL4_C200', 126, 1134)])
def test_restricted_OT_keeps_owner_out_of_bulk_and_preserves_every_source_face(arm, expected_bulk, expected_scalars):
    case, noise, epsilon = synthetic_supervised_case()
    data = slot_data.supervised_sample(case, noise, epsilon, .4, 42, arm)
    undo = np.argsort(data['face_permutation'])
    anchors = data['anchor_ids'][undo]
    source = data['source_face_ids'][undo]
    corners = data['source_corner_permutations'][undo]
    assert len(data['ot_face_map']) == expected_bulk
    assert int(data['free_coordinate_count']) == expected_scalars
    np.testing.assert_array_equal(np.sort(source), np.arange(128))
    np.testing.assert_array_equal(case['full_target'][source[:, None], corners], data['x1'][undo])
    fixed = data['anchor_ids'] >= 0
    assert not data['u'][fixed].any()
    np.testing.assert_array_equal(data['xt'][fixed], data['x0'][fixed])
    if arm == 'SEAM_SLOT_C200':
        assert source[2] == 2
        np.testing.assert_array_equal(corners[2], [1, 2, 0])
        np.testing.assert_array_equal(anchors[2], [1, 0, -1])
        bulk = (anchors < 0).all(axis=1)
        assert 2 not in source[bulk]
        np.testing.assert_array_equal(np.sort(source[bulk]), np.arange(3, 128))
    else:
        assert (anchors[2:] == -1).all()
    # HYBRID is applied only after matching; owner endpoints remain exact C.
    expected_mix = np.float32(np.sqrt(.75)) * noise + np.float32(.5) * epsilon
    free = anchors[2:] < 0
    np.testing.assert_array_equal(data['x0'][undo][2:][free], expected_mix[free])


def test_unsupported_source_owners_fail_instead_of_silently_changing_task():
    case, noise, epsilon = synthetic_supervised_case()
    case['occupied_labels'][1] = True
    case['edge_owner_face'][1] = 2
    with pytest.raises(ValueError, match='Multi-edge'):
        slot_data.supervised_sample(case, noise, epsilon, .4, 42, 'SEAM_SLOT_C200')
    case, noise, epsilon = synthetic_supervised_case()
    case['edge_source_triple'][0, 2] = 2
    with pytest.raises(ValueError, match='Third-C'):
        slot_data.supervised_sample(case, noise, epsilon, .4, 42, 'SEAM_SLOT_C200')


def test_portable_core_has_no_experiment_or_prototype_runtime_imports():
    from meshflow_control.seam import inference, model_adapter, occupancy
    for module in (slot_data, contract, inference, model_adapter, occupancy):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            modules = ([node.module or ''] if isinstance(node, ast.ImportFrom)
                       else [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            assert all('prototype' not in name and 'experiment' not in name
                       and not name.startswith('r2.') for name in modules)
    assert not any(name.startswith('r2.prototypes.') for name in sys.modules)
