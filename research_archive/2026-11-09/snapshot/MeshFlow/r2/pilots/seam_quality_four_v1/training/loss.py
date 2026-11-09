"""Finite-C anti-cover pilot loss; discrete geometric gates are detached.

This is a two-point proxy, not the official I scorer or a geometric certificate.
Owner C triangles remain eligible: excluding them misses folds over that very
triangle. Strict interior projection excludes shared-edge contact. At exact
coplanarity a fixed -C-normal subgradient avoids the zero-distance dead zone.
"""
import math
from meshflow_control import runtime as _runtime
import torch

VERSION = 'finite_C_anti_cover_centroid_tip_v1'
L = math.sqrt(3.) * 2 / .3762
TAU_L = .02
LAMBDA = 1.
TIME_MIN = .5


def finite_C_anti_cover(triangles, C_triangles, owner_faces, *, denominator=None):
    """Return sum hinge / all registered seam probe points (not active count).

    triangles[E,3,3] have reverse-C-edge anchors at corners0/1; only tip2 is
    differentiable. owner_faces[E] indexes original consistently wound C.
    No GT free geometry is used by the loss. FP32 computations outside autocast.
    """
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError('Expected seam triangles [E,3,3]')
    if C_triangles.ndim != 3 or C_triangles.shape[1:] != (3, 3):
        raise ValueError('Expected finite C triangles [K,3,3]')
    E, K = len(triangles), len(C_triangles)
    count = E * 2
    divisor = count if denominator is None else int(denominator)
    if E == 0 or K == 0 or divisor < count:
        raise ValueError('Nonempty C/seams and valid point denominator required')
    with torch.autocast(device_type=triangles.device.type, enabled=False):
        s, c = triangles.float(), C_triangles.detach().float()
        a, e0, e1 = c[:, 0], c[:, 1] - c[:, 0], c[:, 2] - c[:, 0]
        normal = torch.linalg.cross(e0, e1)
        double_area = torch.linalg.vector_norm(normal, dim=-1)
        normal = normal / double_area.clamp_min(1.e-12)[:, None]
        points = torch.stack((s.mean(1), s[:, 2]), dim=1)
        diff = points[:, :, None, :] - a[None, None, :, :]
        height = (diff * normal[None, None]).sum(-1)
        d00, d01, d11 = (e0*e0).sum(-1), (e0*e1).sum(-1), (e1*e1).sum(-1)
        d20, d21 = (diff * e0[None, None]).sum(-1), (diff * e1[None, None]).sum(-1)
        determinant = d00*d11-d01*d01
        v = (d11*d20-d01*d21) / determinant.clamp_min(1.e-20)
        w = (d00*d21-d01*d20) / determinant.clamp_min(1.e-20)
        interior = ((v > 1.e-6) & (w > 1.e-6) & (v+w < 1-1.e-6)
                    & (double_area[None, None] > 1.e-12*L*L)).detach()
        # Find the nearest *finite* face before strict-interior gating. An
        # otherwise nearer edge projection must not be skipped in favour of
        # the interior of a more distant C triangle.
        closed = (v >= 0) & (w >= 0) & (v+w <= 1)
        segment_squared = []
        for j in range(3):
            ea, eb = c[:, j], c[:, (j+1) % 3]
            ev = eb-ea
            ep = points[:, :, None]-ea[None, None]
            alpha = ((ep*ev[None, None]).sum(-1)
                     / (ev*ev).sum(-1).clamp_min(1.e-20)).clamp(0, 1)
            residual = ep-alpha[..., None]*ev[None, None]
            segment_squared.append((residual*residual).sum(-1))
        edge_d2 = torch.stack(segment_squared, dim=-1).min(-1).values
        candidates = torch.where(closed, height.square(), edge_d2).detach()
        nearest = candidates.argmin(-1)  # ties use original C face order
        has_interior = interior.gather(-1, nearest[..., None]).squeeze(-1)
        h = height.gather(-1, nearest[..., None]).squeeze(-1)
        selected_normal = normal[nearest]
        seam_normal = torch.linalg.cross(s[:, 1]-s[:, 0], s[:, 2]-s[:, 0])
        seam_area = torch.linalg.vector_norm(seam_normal, dim=-1)
        seam_normal = seam_normal / seam_area.clamp_min(1.e-12)[:, None]
        parallel = (seam_normal[:, None] * selected_normal).sum(-1).abs()
        tangent = s[:, 0].detach()-s[:, 1].detach()  # original C edge order
        tangent = tangent / torch.linalg.vector_norm(tangent, dim=-1).clamp_min(1.e-12)[:, None]
        inward = torch.linalg.cross(normal[owner_faces], tangent)
        r = ((s[:, 2]-(s[:, 0]+s[:, 1])*.5)*inward).sum(-1)
        gate = (has_interior & (r[:, None] > 0) & (parallel > math.cos(math.radians(15)))
                & (seam_area[:, None] > 1.e-12*L*L)).detach()
        # The sign is a detached selected subgradient, including a nonzero
        # choice at exactly h=0. Gate/direction changes are not differentiated.
        sign = torch.where(h.detach() <= 0, -torch.ones_like(h), torch.ones_like(h))
        margin = torch.relu(TAU_L - sign*h/L)
        hinge = margin.square() * gate.to(h.dtype)
        value = hinge.sum() / divisor
        point_gradient = (-2*margin.detach()*sign/L/divisor*gate)[..., None]*selected_normal
        tip_gradient = point_gradient[:, 0]/3 + point_gradient[:, 1]
    return value, dict(point_count=count, denominator=divisor,
        active_points=int((gate & (h.detach().abs() < TAU_L*L)).sum()),
        gated_points=int(gate.sum()), degenerate_seams=int((seam_area <= 1.e-12*L*L).sum()),
        raw_hinge_sum=float(hinge.detach().sum()),
        tip_gradient_l2_analytic=float(torch.linalg.vector_norm(tip_gradient.detach())))


def training_anti_cover(prediction, sample, *, denominator):
    device = prediction.device
    x = torch.as_tensor(sample['xt'], device=device)
    p = prediction.reshape(sample['N'], 3, 3).float()
    endpoint = x + (1-float(sample['t']))*p
    slots = torch.as_tensor(sample['anti_cover_seam_slots'], device=device, dtype=torch.long)
    corners = torch.as_tensor(sample['anchor_ids'], device=device)
    cv = torch.as_tensor(sample['C_vertices'], device=device)
    fixed = cv[corners[slots, :2]]
    tri = torch.cat((fixed, endpoint[slots, 2:3]), dim=1)
    c = cv[torch.as_tensor(sample['C_faces_original'], device=device)]
    owners = torch.as_tensor(sample['anti_cover_owner_faces'], device=device, dtype=torch.long)
    value, facts = finite_C_anti_cover(tri, c, owners, denominator=denominator)
    time_active = float(sample['t']) >= TIME_MIN
    return value if time_active else value*0., dict(**facts, time_active=time_active,
        velocity_gradient_l2_analytic=(1-float(sample['t']))*facts['tip_gradient_l2_analytic'] if time_active else 0.)
