"""Analytic correctness checks; no 16-object results tune this optimizer."""
import numpy as np
import pytest
import torch

from ..exporter import RadialSupport
from ..legacy.geometry_task import fibonacci, hull_faces
from .optimizer import (TorchRadialField, SurfaceObjective, point_triangle_squared,
                        torch_face_quality, optimize_layout, _np_quality)


def test_point_triangle_regions_and_gradient():
    tri = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]] * 3,
                       dtype=torch.float64)
    p = torch.tensor([[.2, .3, 2.], [.8, .8, .4], [-.2, -.3, .4]],
                     dtype=torch.float64, requires_grad=True)
    d2 = point_triangle_squared(p, tri)
    np.testing.assert_allclose(d2.detach(), [4., .34, .29], atol=1e-14)
    grad = torch.autograd.grad(d2.sum(), p)[0]
    np.testing.assert_allclose(grad, [[0, 0, 4], [.6, .6, .8], [-.4, -.6, .8]],
                               atol=1e-13)
    assert torch.autograd.gradcheck(point_triangle_squared,
                                   (p, tri.clone().requires_grad_(True)),
                                   eps=1e-6, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_radial_value_and_tangent_gradient(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; not a CUDA pass")
    support = RadialSupport()
    y = .18 * support.u[:, 0] + .12 * support.u[:, 1] ** 2
    field = TorchRadialField(y, device=device, support=support)
    rng = np.random.default_rng(175)
    q = rng.normal(size=(41, 3))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    qt = torch.tensor(q, device=device, dtype=torch.float64, requires_grad=True)
    radii = field.query(qt)
    np.testing.assert_allclose(radii.detach().cpu(), support.query(y, q),
                               atol=2e-13, rtol=2e-13)
    direction = rng.normal(size=q.shape)
    direction -= (direction * q).sum(1, keepdims=True) * q
    direction *= .03
    weights = rng.normal(size=len(q))
    gradient = torch.autograd.grad(
        (radii * torch.tensor(weights, dtype=torch.float64, device=device)).sum(), qt)[0]
    analytic = float((gradient.cpu().numpy() * direction).sum())
    h = 1e-5
    qp, qm = q + h * direction, q - h * direction
    qp /= np.linalg.norm(qp, axis=1, keepdims=True)
    qm /= np.linalg.norm(qm, axis=1, keepdims=True)
    np.testing.assert_array_equal(
        field.owners(torch.tensor(qp, device=device)).cpu(),
        field.owners(torch.tensor(qm, device=device)).cpu())
    finite = float(((support.query(y, qp)-support.query(y, qm)) * weights).sum() / (2*h))
    np.testing.assert_allclose(analytic, finite, atol=2e-9, rtol=2e-7)


def test_surface_loss_value_gradient_and_on_field_vertex_trap():
    support = RadialSupport()
    y = np.zeros(len(support.u))
    rv, rf = support.mesh(y)
    u = fibonacci(18)
    faces = hull_faces(u)
    field = TorchRadialField(y)
    v = field.vertices(torch.tensor(u, dtype=torch.float64))
    objective = SurfaceObjective(rv, rf, sample_count=256)
    # Radial vertices lie exactly on target, but their triangle interiors do not.
    assert float(objective(v, torch.tensor(faces))) > 1e-5
    mesh = __import__("trimesh").Trimesh(rv, rf, process=False)
    distance = __import__("trimesh").proximity.closest_point(mesh, v.numpy())[1]
    assert distance.max() < 1e-12
    scale = torch.tensor(1.03, dtype=torch.float64, requires_grad=True)
    loss = objective(v * scale, torch.tensor(faces))
    derivative = float(torch.autograd.grad(loss, scale)[0])
    h = 1e-5
    finite = float((objective(v * (1.03+h), torch.tensor(faces)) -
                    objective(v * (1.03-h), torch.tensor(faces))) / (2*h))
    np.testing.assert_allclose(derivative, finite, atol=1e-8, rtol=2e-5)


def test_quality_matches_numpy():
    u = fibonacci(34)
    faces = hull_faces(u)
    v = u * np.array([1.4, .7, 1.])
    amin, q05 = torch_face_quality(torch.tensor(v), torch.tensor(faces))
    q = _np_quality(v, faces)
    np.testing.assert_allclose([float(amin), float(q05)],
                               [q["min_angle_deg"], q["angle_q05_deg"]], atol=1e-12)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_optimizer_guarded_update_and_final_requery(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; not a CUDA pass")
    support = RadialSupport()
    # Analytic ellipsoid radius, sampled onto the fixed support.
    radii = 1 / np.sqrt((support.u ** 2 / np.array([1.5, .8, 1.]) ** 2).sum(1))
    y = np.log(radii)
    u = fibonacci(34)
    faces = hull_faces(u)
    floor = _np_quality(u * support.query(y, u)[:, None], faces)
    out = optimize_layout(y, u, faces, floor, device=device, max_steps=3, max_seconds=90,
                          snapshot_steps=(0, 1, 3))
    stats = out["stats"]
    assert stats["status"] == "feasible"
    assert stats["accepted_updates"] >= 1
    assert stats["best_internal_loss"] <= stats["initial_internal_loss"] + 1e-14
    for key in ("min_angle_deg", "angle_q05_deg"):
        assert stats["quality"][key] >= floor[key] - stats["quality_tolerance_deg"]
    np.testing.assert_allclose(out["vertices"],
                               out["u"] * support.query(y, out["u"])[:, None],
                               atol=1e-14, rtol=0)
    assert all(x["chart_alpha"] > 0 for x in out["attempts"])
    assert stats["proposal_count"] == len(out["attempts"])
    assert len(out["trace"]) == stats["iterations"] + 1


def test_infeasible_initial_has_no_fixed_fallback():
    support = RadialSupport()
    y = np.zeros(len(support.u))
    u = fibonacci(18)
    faces = hull_faces(u)
    impossible_floor = {"min_angle_deg": 80., "angle_q05_deg": 80.}
    out = optimize_layout(y, u, faces, impossible_floor, device="cpu",
                          max_steps=1, max_seconds=90, snapshot_steps=(0, 1))
    assert out["stats"]["status"] == "infeasible"
    assert out["stats"]["best_feasible_iteration"] is None
    assert out["stats"]["first_feasible_iteration"] is None
    assert all(a["phase"] == "restoration" for a in out["attempts"])
    assert len(out["vertices"]) == len(u)
