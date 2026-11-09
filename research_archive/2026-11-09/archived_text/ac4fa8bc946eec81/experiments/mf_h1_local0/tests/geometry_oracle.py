"""Independent scalar CPU oracle used only by tests/evaluation.

This module deliberately does not import the training geometry implementation.
The triangle solution enumerates the three boundary segments and, when valid,
the unconstrained least-squares interior solution (an active-set construction).
"""

import numpy as np


def segment_closest(point, a, b):
    point, a, b = (np.asarray(value, dtype=np.float64) for value in (point, a, b))
    direction = b - a
    squared_length = float(np.dot(direction, direction))
    if squared_length == 0.0:
        return a.copy()
    parameter = np.clip(np.dot(point - a, direction) / squared_length, 0.0, 1.0)
    return a + parameter * direction


def triangle_closest(point, a, b, c):
    point, a, b, c = (np.asarray(value, dtype=np.float64) for value in (point, a, b, c))
    candidates = [segment_closest(point, a, b), segment_closest(point, b, c),
                  segment_closest(point, c, a)]
    matrix = np.column_stack((b - a, c - a))
    barycentric, _, rank, _ = np.linalg.lstsq(matrix, point - a, rcond=None)
    if rank == 2 and np.all(barycentric >= 0.0) and barycentric.sum() <= 1.0:
        candidates.append(a + matrix @ barycentric)
    distances = [np.dot(point - candidate, point - candidate) for candidate in candidates]
    return candidates[int(np.argmin(distances))]


def point_to_surface_squared(points, triangles):
    """Slow, independent scalar reference; intended for small audit fixtures."""
    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.float64)
    if len(triangles) == 0:
        raise ValueError("A surface must contain at least one triangle")
    result = []
    for point in points:
        result.append(min(float(np.dot(point - closest, point - closest))
                          for closest in (triangle_closest(point, *triangle)
                                          for triangle in triangles)))
    return np.asarray(result, dtype=np.float64)
