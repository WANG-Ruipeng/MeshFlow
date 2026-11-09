"""Bounded per-object layout optimization, independent of the frozen A network.

The objective is a symmetric, area-weighted point-to-triangle squared distance.
Closest-face identities and radial chart owners are discrete active sets; gradients
through the selected exact triangle/radial formula are computed in float64. This
is a local piecewise optimizer, not a certificate of globally optimal layouts.
"""
from __future__ import annotations

import time
import numpy as np
import torch
import trimesh

from ..exporter import RadialSupport
from ..legacy.chart import dets, retract, solid_angles
from ..legacy.common import sample_surface


class TorchRadialField:
    """The exact piecewise F4096 query, with detached chart face selection."""
    def __init__(self, y, device="cpu", support=None):
        self.support = support if support is not None else RadialSupport()
        self.y = np.asarray(y, dtype=np.float64)
        # Validate without silently clipping radii.
        self.support.mesh(self.y)
        self.device = torch.device(device)
        self.inv = torch.as_tensor(self.support.inv, dtype=torch.float64,
                                   device=self.device)
        rs = np.exp(self.y)
        self.face_inverse_radii = torch.as_tensor(
            1.0 / rs[self.support.faces], dtype=torch.float64, device=self.device)

    def owners(self, q):
        qn = q.detach().cpu().numpy()
        _, cand = self.support.tree.query(qn, k=12)
        bb = np.einsum("ni,nkij->nkj", qn, self.support.inv[cand])
        ok = bb.min(-1) >= -1e-12
        fi = cand[np.arange(len(qn)), ok.argmax(1)]
        missing = ~ok.any(1)
        if missing.any():
            from ..legacy.geometry_task import ray_owner
            fi[missing] = ray_owner(self.support.u, self.support.faces, qn[missing])
            if (fi[missing] < 0).any():
                raise ValueError("support chart gap")
        return torch.as_tensor(fi, dtype=torch.long, device=self.device)

    def query(self, q):
        fi = self.owners(q)
        be = torch.einsum("ni,nij->nj", q, self.inv[fi])
        den = (be * self.face_inverse_radii[fi]).sum(-1)
        if not bool(torch.isfinite(den).all()) or not bool((den > 0).all()):
            raise FloatingPointError("invalid positive radial denominator")
        return den.reciprocal()

    def vertices(self, q):
        return q * self.query(q)[:, None]


def point_triangle_squared(points, triangles):
    """Squared Euclidean distance for paired points and nondegenerate triangles.

    All four candidates (triangle-interior projection and three closed edges) are
    exact. Discrete region selection is locally constant away from boundaries.
    """
    a, b, c = triangles.unbind(dim=1)
    e, g, w = b - a, c - a, points - a
    ee, gg, eg = (e * e).sum(-1), (g * g).sum(-1), (e * g).sum(-1)
    we, wg = (w * e).sum(-1), (w * g).sum(-1)
    determinant = ee * gg - eg.square()
    tiny = torch.finfo(points.dtype).tiny
    safe_det = determinant.clamp_min(tiny)
    beta = (we * gg - wg * eg) / safe_det
    gamma = (wg * ee - we * eg) / safe_det
    normal = torch.linalg.cross(e, g)
    plane = (w * normal).sum(-1).square() / (normal * normal).sum(-1).clamp_min(tiny)
    inside = ((beta >= 0) & (gamma >= 0) & ((beta + gamma) <= 1)
              & (determinant > 0))
    plane = torch.where(inside, plane, torch.full_like(plane, float("inf")))
    candidates = [plane]
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        t = ((points - start) * edge).sum(-1) / (edge * edge).sum(-1).clamp_min(tiny)
        closest = start + t.clamp(0, 1)[:, None] * edge
        candidates.append((points - closest).square().sum(-1))
    return torch.stack(candidates, dim=-1).min(-1).values


def torch_face_quality(vertices, faces):
    tri = vertices[faces]
    angles = []
    for i in range(3):
        a = tri[:, (i + 1) % 3] - tri[:, i]
        b = tri[:, (i + 2) % 3] - tri[:, i]
        cross = torch.linalg.vector_norm(torch.linalg.cross(a, b), dim=-1)
        angles.append(torch.atan2(cross, (a * b).sum(-1)) * (180.0 / np.pi))
    per_face = torch.stack(angles, dim=-1).min(-1).values
    return per_face.min(), torch.quantile(per_face, .05)


def _np_quality(vertices, faces):
    tri = vertices[faces]
    angles = []
    for i in range(3):
        a = tri[:, (i + 1) % 3] - tri[:, i]
        b = tri[:, (i + 2) % 3] - tri[:, i]
        angles.append(np.rad2deg(np.arctan2(
            np.linalg.norm(np.cross(a, b), axis=1), np.einsum("ij,ij->i", a, b))))
    per_face = np.min(angles, axis=0)
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                    tri[:, 2] - tri[:, 0]), axis=1)
    diag2 = float(np.ptp(vertices, axis=0) @ np.ptp(vertices, axis=0))
    return {"min_angle_deg": float(per_face.min()),
            "angle_q05_deg": float(np.quantile(per_face, .05)),
            "degenerate_count": int((area2 <= max(diag2, 1e-300) * 1e-14).sum())}


class SurfaceObjective:
    """Fixed reference MC samples + three-point output-face quadrature.

    The target MC samples are independent from final evaluation samples. Reverse
    quadrature is deterministic, degree-two, and normalized by current true area.
    """
    def __init__(self, reference_vertices, reference_faces, device="cpu",
                 seed=71310, sample_count=2048):
        self.device = torch.device(device)
        self.rv = np.asarray(reference_vertices, np.float64)
        self.rf = np.asarray(reference_faces, np.int64)
        self.reference = trimesh.Trimesh(self.rv, self.rf, process=False)
        self.reference_proximity = trimesh.proximity.ProximityQuery(self.reference)
        pts, _, _ = sample_surface(self.rv, self.rf, sample_count,
                                   np.random.default_rng(seed))
        self.points_np = pts
        self.points = torch.as_tensor(pts, device=self.device, dtype=torch.float64)
        self.target_tri = torch.as_tensor(self.rv[self.rf], device=self.device,
                                         dtype=torch.float64)
        self.bary = torch.as_tensor([[2/3, 1/6, 1/6], [1/6, 2/3, 1/6],
                                     [1/6, 1/6, 2/3]],
                                    device=self.device, dtype=torch.float64)
        self.diag2 = float(np.ptp(self.rv, axis=0) @ np.ptp(self.rv, axis=0))
        if self.diag2 <= 0:
            raise ValueError("invalid target bbox")

    def __call__(self, vertices, faces):
        vn = vertices.detach().cpu().numpy()
        fn = faces.detach().cpu().numpy()
        mesh = trimesh.Trimesh(vn, fn, process=False)
        _, _, forward_id = trimesh.proximity.closest_point(mesh, self.points_np)
        forward_id = torch.as_tensor(forward_id, device=self.device, dtype=torch.long)
        tri = vertices[faces]
        forward = point_triangle_squared(self.points, tri[forward_id]).mean()
        quadrature = torch.einsum("kj,fjd->fkd", self.bary, tri)
        quad_flat = quadrature.reshape(-1, 3)
        _, _, reverse_id = self.reference_proximity.on_surface(
            quad_flat.detach().cpu().numpy())
        reverse_id = torch.as_tensor(reverse_id, device=self.device, dtype=torch.long)
        reverse_d2 = point_triangle_squared(
            quad_flat, self.target_tri[reverse_id]).reshape(len(tri), -1).mean(1)
        areas = .5 * torch.linalg.vector_norm(
            torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), dim=-1)
        reverse = (areas * reverse_d2).sum() / areas.sum()
        return .5 * (forward + reverse) / self.diag2


def _tangent(x, u):
    return x - (x * u).sum(-1, keepdim=True) * u


def _project_constraints(delta, u, constraints, quality_values, quality_floor):
    """Alternating half-space projection near the two active quality bounds."""
    delta = _tangent(delta, u)
    for _ in range(12):
        for key, grad in constraints:
            slack = quality_values[key] - quality_floor[key]
            if slack >= .1:
                continue
            # Small interior bias avoids following a tangent into a curved violation.
            desired = min(.02, max(0., .025 - slack))
            denom = (grad * grad).sum()
            dot = (grad * delta).sum()
            if denom > 1e-24 and dot < desired:
                delta = delta + (desired - dot) / denom * grad
    return _tangent(delta, u)


def optimize_layout(y, initial_u, faces, quality_floor, device="cuda",
                    max_steps=500, max_seconds=300, seed=71310,
                    snapshot_steps=(0, 25, 100, 250, 500)):
    """Optimize directions only and return best accepted feasible training iterate.

    An infeasible initial layout receives explicit quality restoration. No Fixed
    fallback is inserted. If restoration fails the last layout is returned with
    status='infeasible' and best_feasible_iteration=None. Each attempted line-search
    proposal and every iteration are retained. Final independent auditing is the
    caller's responsibility; these guards are numerical, not formal certificates.
    """
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA is unavailable")
    if max_steps < 0 or max_seconds <= 0:
        raise ValueError("invalid optimization budget")
    required = ("min_angle_deg", "angle_q05_deg")
    floor = {k: float(quality_floor[k]) for k in required}
    if not all(np.isfinite(x) and x >= 0 for x in floor.values()):
        raise ValueError("invalid quality floor")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    support = RadialSupport()
    field = TorchRadialField(y, device=device, support=support)
    ref_v, ref_f = support.mesh(y)
    objective = SurfaceObjective(ref_v, ref_f, device=device, seed=seed)
    f_np = np.asarray(faces, np.int64)
    f_t = torch.as_tensor(f_np, device=device, dtype=torch.long)
    u_np = np.array(initial_u, dtype=np.float64, copy=True)
    if not np.isfinite(u_np).all() or not np.allclose(np.linalg.norm(u_np, axis=1), 1,
                                                   atol=1e-9, rtol=0):
        raise ValueError("initial directions must be finite unit vectors")
    if dets(u_np, f_np).min() <= 0 or abs(solid_angles(u_np, f_np).sum()-4*np.pi) > 1e-6:
        raise ValueError("initial directions must be an oriented degree-one chart")
    # This tolerance covers floating point roundoff, not a quality tradeoff.
    quality_tolerance_deg = 1e-6

    def vertices_np(directions):
        return directions * support.query(y, directions)[:, None]

    def quality(directions):
        return _np_quality(vertices_np(directions), f_np)

    def violation(q):
        return sum(max(0., floor[k] - q[k] - quality_tolerance_deg) /
                   max(floor[k], 1.) for k in required)

    def feasible(q):
        return q["degenerate_count"] == 0 and violation(q) <= 1e-14

    def training_loss(directions):
        with torch.no_grad():
            ut = torch.as_tensor(directions, device=device, dtype=torch.float64)
            return float(objective(field.vertices(ut), f_t).cpu())

    attempts, trace, snapshots = [], [], []
    current_q = quality(u_np)
    current_loss = training_loss(u_np)
    initial_loss = current_loss
    best_u = u_np.copy() if feasible(current_q) else None
    best_loss = current_loss if best_u is not None else float("inf")
    best_iteration = 0 if best_u is not None else None
    initial_feasible = feasible(current_q)
    first_feasible_iteration = 0 if initial_feasible else None
    first_feasible_seconds = None
    m = torch.zeros_like(torch.as_tensor(u_np, device=device, dtype=torch.float64))
    vv = torch.zeros_like(m)
    lr = .015
    adam_count = 0
    old_phase = None
    accepted_count = 0

    def elapsed():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter() - start

    def record(iteration, accepted=False):
        row = {"iteration": iteration, "elapsed_seconds": elapsed(),
               "internal_loss": current_loss, "feasible": feasible(current_q),
               "quality": dict(current_q), "quality_violation": violation(current_q),
               "phase": "surface" if feasible(current_q) else "restoration",
               "accepted": bool(accepted),
               "best_feasible_internal_loss": None if best_u is None else best_loss}
        trace.append(row)
        if iteration in snapshot_steps:
            snapshots.append({**row, "u": u_np.copy(), "vertices": vertices_np(u_np),
                              "faces": f_np.copy()})
        return row

    record(0)
    if initial_feasible:
        first_feasible_seconds = trace[0]["elapsed_seconds"]
    stop_reason = "max_steps"
    for iteration in range(1, max_steps + 1):
        if elapsed() >= max_seconds:
            stop_reason = "max_seconds"
            break
        phase = "surface" if feasible(current_q) else "restoration"
        if phase != old_phase:
            m.zero_()
            vv.zero_()
            adam_count = 0
            old_phase = phase
        ut = torch.tensor(u_np, device=device, dtype=torch.float64, requires_grad=True)
        vertices = field.vertices(ut)
        amin, aq05 = torch_face_quality(vertices, f_t)
        qts = {"min_angle_deg": amin, "angle_q05_deg": aq05}
        constraint_gradients = []
        if phase == "surface":
            loss = objective(vertices, f_t)
            grad = torch.autograd.grad(loss, ut, retain_graph=True)[0]
            for key in required:
                if current_q[key] - floor[key] < .1:
                    cg = torch.autograd.grad(qts[key], ut, retain_graph=True)[0]
                    constraint_gradients.append((key, _tangent(cg, ut).detach()))
        else:
            loss = sum(torch.relu(floor[k] - quality_tolerance_deg - qts[k]) /
                       max(floor[k], 1.) for k in required)
            grad = torch.autograd.grad(loss, ut)[0]
        if not bool(torch.isfinite(grad).all()):
            stop_reason = "nonfinite_gradient"
            break
        grad = _tangent(grad.detach(), ut.detach())
        adam_count += 1
        m = .9 * m + .1 * grad
        vv = .999 * vv + .001 * grad.square()
        delta = -lr * (m / (1 - .9**adam_count)) / (
            torch.sqrt(vv / (1 - .999**adam_count)) + 1e-12)
        delta = _tangent(delta, ut.detach())
        # Adam momentum can point uphill after a discrete active-set change.
        if (delta * grad).sum() >= 0:
            delta = -lr * grad / grad.abs().mean().clamp_min(1e-12)
            delta = _tangent(delta, ut.detach())
        if phase == "surface":
            delta = _project_constraints(delta, ut.detach(), constraint_gradients,
                                         current_q, floor)
        largest = float(torch.linalg.vector_norm(delta, dim=1).max().cpu())
        if largest > .05:
            delta *= .05 / largest
        delta_np = delta.cpu().numpy()
        accepted = False
        for backtrack in range(14):
            if elapsed() >= max_seconds:
                stop_reason = "max_seconds"
                break
            scale = 2.0 ** -backtrack
            proposal, chart_alpha = retract(u_np, delta_np * scale, f_np)
            pq = quality(proposal)  # Radius always re-queried at accepted candidate.
            info = {"iteration": iteration, "phase": phase,
                    "backtrack": backtrack, "line_alpha": scale,
                    "chart_alpha": float(chart_alpha),
                    "combined_alpha": float(scale * chart_alpha),
                    "quality": pq, "quality_violation": violation(pq),
                    "accepted": False, "internal_loss": None}
            if pq["degenerate_count"]:
                info["reason"] = "degenerate"
            elif phase == "surface" and not feasible(pq):
                info["reason"] = "quality"
            elif phase == "restoration" and not (
                    feasible(pq) or violation(pq) < violation(current_q) - 1e-12):
                info["reason"] = "restoration_not_improved"
            else:
                ploss = training_loss(proposal)
                info["internal_loss"] = ploss
                if not np.isfinite(ploss):
                    info["reason"] = "nonfinite_loss"
                elif phase == "surface" and ploss > current_loss + 1e-14:
                    info["reason"] = "surface_loss_not_improved"
                else:
                    u_np, current_q, current_loss = proposal, pq, ploss
                    info["accepted"] = True
                    info["reason"] = "accepted"
                    accepted = True
                    accepted_count += 1
                    if feasible(pq):
                        if first_feasible_iteration is None:
                            first_feasible_iteration = iteration
                            first_feasible_seconds = elapsed()
                        if current_loss < best_loss:
                            best_u, best_loss = u_np.copy(), current_loss
                            best_iteration = iteration
            info["elapsed_seconds"] = elapsed()
            attempts.append(info)
            if accepted:
                break
        if not accepted:
            # Momentum reset is fixed behavior, not a per-object hyperparameter search.
            m.zero_()
            vv.zero_()
            adam_count = 0
        record(iteration, accepted)
        if stop_reason == "max_seconds":
            break
    else:
        iteration = max_steps
    last_iteration = trace[-1]["iteration"]
    chosen = u_np if best_u is None else best_u
    chosen_v = vertices_np(chosen)
    final_q = _np_quality(chosen_v, f_np)
    if last_iteration not in {s["iteration"] for s in snapshots}:
        snapshots.append({**trace[-1], "u": u_np.copy(),
                          "vertices": vertices_np(u_np), "faces": f_np.copy()})
    return {
        "vertices": chosen_v, "faces": f_np.copy(), "u": chosen.copy(),
        "stats": {
            "status": "feasible" if best_u is not None else "infeasible",
            "stop_reason": stop_reason, "iterations": last_iteration,
            "seconds": elapsed(), "device": str(device), "dtype": "float64",
            "initial_internal_loss": initial_loss,
            "best_internal_loss": None if best_u is None else best_loss,
            "last_internal_loss": current_loss, "best_feasible_iteration": best_iteration,
            "initial_feasible": initial_feasible,
            "first_feasible_iteration": first_feasible_iteration,
            "first_feasible_seconds": first_feasible_seconds,
            "quality": final_q, "quality_floor": floor,
            "quality_tolerance_deg": quality_tolerance_deg,
            "accepted_updates": accepted_count, "proposal_count": len(attempts),
            "rejected_proposals": sum(not a["accepted"] for a in attempts),
            "chart_shrunk_proposals": sum(a["chart_alpha"] < 1 for a in attempts),
            "training_reference_samples": 2048, "training_seed": seed,
            "reverse_quadrature_points_per_face": 3,
            "max_steps": max_steps, "max_seconds": max_seconds,
            "learning_rate": lr, "max_tangent_step_norm": .05,
            "line_search_trials": 14,
            "selection": "minimum_training_loss_among_accepted_feasible_iterates",
            "independent_final_audit_required": True,
        },
        "snapshots": snapshots, "trace": trace, "attempts": attempts,
    }
