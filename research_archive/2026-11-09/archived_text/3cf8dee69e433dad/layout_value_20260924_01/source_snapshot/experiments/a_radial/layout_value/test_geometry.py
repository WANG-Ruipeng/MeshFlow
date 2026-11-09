"""Analytic checks for surface-distance definitions and geometry failure accounting."""
import json

import numpy as np
import pytest
import trimesh

from .geometry import audit_mesh, evaluate_mesh, mesh_quality


def triangle():
    return np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]), np.array([[0, 1, 2]])


def test_identical_surface_zero_despite_interior_samples():
    v, f = triangle()
    result = evaluate_mesh(v, f, v, f, points=512)
    assert result["rms_bbox_percent"] < 1e-12
    assert result["sampled_max_is_true_hausdorff"] is False
    json.dumps(result, allow_nan=False)


def test_parallel_triangles_analytic_distance_and_common_scale():
    v, f = triangle()
    shifted = v + [0., 0., 0.2]
    result = evaluate_mesh(shifted, f, v, f, points=512)
    expected = 100 * 0.2 / np.sqrt(2)
    assert result["rms_bbox_percent"] == pytest.approx(expected, abs=1e-10)
    assert result["mesh_to_reference_rms_bbox_percent"] == pytest.approx(expected)
    assert result["reference_to_mesh_rms_bbox_percent"] == pytest.approx(expected)
    scaled = evaluate_mesh(shifted * 7 + 13, f, v * 7 + 13, f, points=512)
    assert scaled["rms_bbox_percent"] == pytest.approx(expected, abs=1e-10)


def test_same_sharp_cube_with_subdivision_is_same_surface():
    cube = trimesh.creation.box()
    sub_v, sub_f = trimesh.remesh.subdivide(cube.vertices, cube.faces)
    result = evaluate_mesh(sub_v, sub_f, cube.vertices, cube.faces, points=512)
    assert result["rms_bbox_percent"] < 1e-12
    assert mesh_quality(cube.vertices, cube.faces)["min_angle_deg"] == pytest.approx(45.)
    assert audit_mesh(cube.vertices, cube.faces)["valid"] is True


def test_quality_equilateral_and_per_face_minimum_quantile():
    eq = np.array([[0., 0., 0.], [1., 0., 0.], [0.5, np.sqrt(3) / 2, 0.]])
    eq_result = mesh_quality(eq, np.array([[0, 1, 2]]))
    assert eq_result["min_angle_deg"] == pytest.approx(60.)
    assert eq_result["shape_quality_min"] == pytest.approx(1.)
    thin = np.array([[2., 0., 0.], [3., 0., 0.], [2., 0.01, 0.]])
    v = np.vstack((eq, thin))
    f = np.array([[0, 1, 2], [3, 4, 5]])
    result = mesh_quality(v, f)
    expected_min = np.rad2deg(np.arctan(0.01))
    assert result["min_angle_deg"] == pytest.approx(expected_min)
    assert result["angle_q05_deg"] == pytest.approx(np.quantile([60., expected_min], .05))


def test_bad_triangles_are_counted_not_filtered():
    v, f = triangle()
    f = np.vstack((f, [0, 0, 1]))
    result = mesh_quality(v, f)
    assert result["degenerate_count"] == 1
    assert result["min_angle_deg"] == 0.
    assert result["shape_quality_min"] == 0.
    assert audit_mesh(v, f)["valid"] is False


def test_crossing_triangles_exactly_detected():
    v = np.array([[-1., -1., 0.], [1., -1., 0.], [0., 1., 0.],
                  [0., -.5, -1.], [0., -.5, 1.], [0., .5, 0.]])
    f = np.array([[0, 1, 2], [3, 4, 5]])
    result = audit_mesh(v, f)
    assert result["selfintersection_count"] == 1
    assert result["selfintersection_face_fraction"] == 1.
    assert result["valid"] is False
    json.dumps(result, allow_nan=False)


def test_topology_only_never_claims_intersection_pass():
    cube = trimesh.creation.box()
    result = audit_mesh(cube.vertices, cube.faces, exact=False)
    assert result["topology_pass"] is True
    assert result["valid"] is None
    assert result["selfintersection_count"] is None


def test_invalid_inputs_and_open_surface_do_not_pass():
    v, f = triangle()
    assert audit_mesh(v, f)["closed"] is False
    assert audit_mesh(v, f)["valid"] is False
    assert audit_mesh(v, np.empty((0, 3), dtype=int))["valid"] is False
    assert audit_mesh(v, np.array([[0, 1, 20]]))["valid"] is False
    with pytest.raises(ValueError):
        evaluate_mesh(v * np.nan, f, v, f)
    with pytest.raises(ValueError):
        evaluate_mesh(v, f, v, f, points=0)


def test_evaluation_is_deterministic_and_does_not_consume_global_rng():
    cube = trimesh.creation.box()
    np.random.seed(901)
    state = np.random.get_state()
    first = evaluate_mesh(cube.vertices * [1.1, 1, 1], cube.faces,
                          cube.vertices, cube.faces, points=300, seed=29)
    current = np.random.get_state()
    assert state[0] == current[0]
    assert np.array_equal(state[1], current[1])
    assert state[2:] == current[2:]
    second = evaluate_mesh(cube.vertices * [1.1, 1, 1], cube.faces,
                           cube.vertices, cube.faces, points=300, seed=29)
    assert first == second
