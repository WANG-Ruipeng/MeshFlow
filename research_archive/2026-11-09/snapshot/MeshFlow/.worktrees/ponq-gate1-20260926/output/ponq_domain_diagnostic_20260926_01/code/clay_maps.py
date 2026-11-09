"""Frozen analytical geometry paths and their exact pointwise Jacobians.

These maps are prescribed geometric deformations, not a material model.
Input coordinates are used as supplied: this module never normalizes,
clips, repairs, or projects points. In particular, thin_u_squeeze can have
nonpositive determinant sufficiently far outside its reference y-range;
the caller must audit the returned Jacobians for every evaluated site.
"""

import numpy as np


MAP_NAMES = ("sphere_squash", "thin_u_squeeze", "bottle_bend")
TIME_STEPS = tuple(float(t) for t in np.linspace(0., 1., 8))


def _points(value, name):
    points = np.asarray(value, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 3 or points.size == 0:
        raise ValueError(f"{name} must be a nonempty array with trailing dimension 3")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains nonfinite coordinates")
    return points


def build_map(name, reference_vertices):
    """Return a JSON-serializable frozen map specification.

    The reference must already be in the intended shared coordinate frame.
    For bottle_bend, L is the full bounding-box extent along its longest
    axis, not its half extent. The frame (a,b,s) is a right-handed cyclic
    permutation of the world axes; longest-axis ties use the lowest index.
    """
    if name not in MAP_NAMES:
        raise ValueError(f"Unknown clay map {name!r}; expected one of {MAP_NAMES}")
    reference = _points(reference_vertices, "reference_vertices")
    if reference.ndim != 2:
        raise ValueError("reference_vertices must have shape (N,3)")
    lower = reference.min(axis=0)
    upper = reference.max(axis=0)
    extent = upper - lower
    center = (lower + upper) / 2
    spec = dict(
        version="CLAY_MAPS_V1", name=name,
        source_bbox_min=lower.tolist(), source_bbox_max=upper.tolist(),
        source_bbox_extent=extent.tolist(), source_bbox_center=center.tolist(),
        time_steps=list(TIME_STEPS),
        input_normalization="none; caller supplies the shared input frame",
        point_clipping=False, physical_simulation=False,
        jacobian_convention="J[..., i, j] = derivative(output_i, input_j)",
    )
    if name == "sphere_squash":
        spec.update(
            origin=[0., 0., 0.],
            axis_scale_rates=[.2, 0., -.45],
            formula="phi(x,t)=diag(1+.2*t,1,1-.45*t) @ x",
            determinant_on_reference="(1+.2*t)*(1-.45*t)",
        )
    elif name == "thin_u_squeeze":
        if extent[1] <= 0:
            raise ValueError("thin_u_squeeze needs a positive source y extent")
        spec.update(
            y_min=float(lower[1]), y_length=float(extent[1]), squeeze_rate=.65,
            formula="r=(y-y_min)/y_length; phi=(x*(1-.65*t*r*r),y,z)",
            determinant_on_reference="1-.65*t*r*r; >= .35 for t,r in [0,1]",
            extrapolation="r is not clipped outside the source y range",
        )
    else:
        s_axis = int(np.argmax(extent))
        a_axis, b_axis = (s_axis + 1) % 3, (s_axis + 2) % 3
        if extent[s_axis] <= 0:
            raise ValueError("bottle_bend needs a positive longest source extent")
        basis = np.eye(3)[:, [a_axis, b_axis, s_axis]]
        spec.update(
            center=center.tolist(), axial_axis=s_axis, bend_axis=a_axis,
            remaining_axis=b_axis, frame_axis_order=[a_axis, b_axis, s_axis],
            frame_basis_columns=basis.tolist(), frame_handedness="right",
            longest_axis_tie_break="lowest world-axis index",
            length=float(extent[s_axis]), length_definition="full source bbox axial extent",
            axial_stretch_rate=.25, bend_rate=.22,
            formula="q=x-center; s'=s*(1+.25*t); a'=a+.22*t*L*(s/L)^2; b'=b; restore center",
            determinant_on_reference="1+.25*t (also holds outside the source bbox)",
        )
    return spec


def deform(x, t, spec):
    """Return (phi(x,t), J(x,t)) for points of shape (...,3).

    J has shape (...,3,3). t must be a scalar on the frozen path [0,1].
    Coordinates outside the reference bounds are evaluated without clipping.
    The arrays returned at t=0 preserve the supplied float64 points exactly.
    Neither the input coordinates nor the specification is modified.
    """
    points = _points(x, "x")
    time = np.asarray(t, dtype=np.float64)
    if time.ndim != 0 or not np.isfinite(time) or not 0 <= float(time) <= 1:
        raise ValueError("t must be a finite scalar in [0,1]")
    time = float(time)
    name = spec.get("name")
    if name not in MAP_NAMES:
        raise ValueError(f"Unknown clay map {name!r}")
    result = points.copy()
    jacobian = np.broadcast_to(np.eye(3), points.shape[:-1] + (3, 3)).copy()
    if time == 0.:
        return result, jacobian
    if name == "sphere_squash":
        scales = 1 + time * np.asarray(spec["axis_scale_rates"], dtype=np.float64)
        result *= scales
        jacobian *= scales[:, None]
    elif name == "thin_u_squeeze":
        length = float(spec["y_length"])
        r = (points[..., 1] - float(spec["y_min"])) / length
        coefficient = float(spec["squeeze_rate"]) * time
        scale = 1 - coefficient * r * r
        result[..., 0] = points[..., 0] * scale
        jacobian[..., 0, 0] = scale
        jacobian[..., 0, 1] = -2 * coefficient * points[..., 0] * r / length
    else:
        s_axis, a_axis = int(spec["axial_axis"]), int(spec["bend_axis"])
        center = np.asarray(spec["center"], dtype=np.float64)
        length = float(spec["length"])
        s = points[..., s_axis] - center[s_axis]
        stretch = 1 + float(spec["axial_stretch_rate"]) * time
        bend = float(spec["bend_rate"]) * time
        result[..., s_axis] = center[s_axis] + stretch * s
        result[..., a_axis] = points[..., a_axis] + bend * s * s / length
        jacobian[..., s_axis, s_axis] = stretch
        jacobian[..., a_axis, s_axis] = 2 * bend * s / length
    if not np.isfinite(result).all() or not np.isfinite(jacobian).all():
        raise ValueError("Map evaluation overflowed; no clipping or repair was applied")
    return result, jacobian
