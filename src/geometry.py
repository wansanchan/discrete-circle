"""Closed polygons as edge vectors, and the kinematic observables of a discrete circle.

Edges ``q[k]`` live in R^2. The polygon is closed when ``sum_k q[k] = 0``.
Nothing in this module refers to an external origin: the centroid is built
from the edge vectors, and π_eff is built from the perimeter and the area.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def project_closed(q: Array) -> Array:
    """Subtract the mean edge so the vector sum vanishes."""
    return q - np.mean(q, axis=0, keepdims=True)


def regular_edges(n: int, length: float = 1.0, alpha0: float = 0.0) -> Array:
    """Edges of a regular n-gon. Consecutive headings differ by 2π/n.

    The sum of the n-th roots of unity is zero, so the polygon closes.
    ``project_closed`` only removes a floating-point residual.
    """
    if n < 3:
        raise ValueError("a polygon needs n >= 3")
    k = np.arange(n, dtype=float)
    ang = alpha0 + 2.0 * np.pi * k / n
    q = length * np.column_stack((np.cos(ang), np.sin(ang)))
    return project_closed(q)


def vertices(q: Array) -> Array:
    """Vertex positions with x_0 = 0 and x_{k+1} = x_k + q_k."""
    x = np.zeros_like(q)
    x[1:] = np.cumsum(q[:-1], axis=0)
    return x


def edge_lengths(q: Array) -> Array:
    return np.linalg.norm(q, axis=1)


def closure(q: Array) -> Array:
    return np.sum(q, axis=0)


def signed_area(q: Array) -> float:
    """(1/2) sum_k x_k × q_k. Positive for a counterclockwise boundary."""
    x = vertices(q)
    return 0.5 * float(np.sum(x[:, 0] * q[:, 1] - x[:, 1] * q[:, 0]))


def perimeter(q: Array) -> float:
    return float(np.sum(edge_lengths(q)))


def pi_eff(q: Array) -> float:
    """Discrete π from the isoperimetric quotient, P²/(4A).

    For a regular n-gon this equals n tan(π/n), which is larger than π
    and converges as π + π³/(3n²) + O(n⁻⁴).
    """
    area = signed_area(q)
    if area <= 0.0:
        return np.nan
    peri = perimeter(q)
    return peri * peri / (4.0 * area)


def regular_pi_eff(n: int) -> float:
    """Exact value on a regular n-gon: n tan(π/n)."""
    return float(n * np.tan(np.pi / n))


def regular_area(n: int, length: float = 1.0) -> float:
    """Area of a regular n-gon with side ``length``: n L² / (4 tan(π/n))."""
    return float(n * length * length / (4.0 * np.tan(np.pi / n)))


def regular_inradius(n: int, length: float = 1.0) -> float:
    """Inradius r = L / (2 tan(π/n))."""
    return float(length / (2.0 * np.tan(np.pi / n)))


def turning_angles(q: Array) -> Array:
    """Exterior angles δ_k = heading(q_k) − heading(q_{k−1}) in (−π, π]."""
    prev = np.roll(q, 1, axis=0)
    cross = prev[:, 0] * q[:, 1] - prev[:, 1] * q[:, 0]
    dot = np.sum(prev * q, axis=1)
    return np.atan2(cross, dot)


def winding(q: Array) -> float:
    """Total turning in units of 2π. A convex counterclockwise polygon has w = 1."""
    return float(np.sum(turning_angles(q)) / (2.0 * np.pi))


def inward_normals(q: Array) -> Array:
    """Unit normals obtained by rotating each edge 90° toward the interior.

    The interior is on the left of a counterclockwise boundary, so the
    positive-area case uses R_{π/2}. The sign of the area flips this when
    the boundary is clockwise.
    """
    lengths = edge_lengths(q)
    if np.any(lengths == 0.0):
        raise ValueError("zero-length edge")
    rotated = np.column_stack((-q[:, 1], q[:, 0])) / lengths[:, None]
    area = signed_area(q)
    if area < 0.0:
        rotated = -rotated
    elif area == 0.0:
        raise ValueError("degenerate area")
    return rotated


def midpoints(q: Array) -> Array:
    return vertices(q) + 0.5 * q


def centroid(q: Array) -> Array:
    return np.mean(vertices(q), axis=0)


def radial_unit(q: Array) -> Array:
    """Unit vectors from edge midpoints toward the vertex centroid."""
    delta = centroid(q) - midpoints(q)
    norm = np.linalg.norm(delta, axis=1)
    if np.any(norm == 0.0):
        raise ValueError("centroid lies on an edge midpoint")
    return delta / norm[:, None]


def alignment(q: Array) -> Array:
    """cos ψ_k = n_k · û_k. Equals 1 when the inward normal points at the centroid."""
    return np.sum(inward_normals(q) * radial_unit(q), axis=1)


def mean_alignment(q: Array) -> float:
    return float(np.mean(alignment(q)))


def angle_deviation(q: Array) -> Array:
    """η_k = δ_k − 2π/n."""
    n = q.shape[0]
    return turning_angles(q) - (2.0 * np.pi / n)


def mean_eta2(q: Array) -> float:
    eta = angle_deviation(q)
    return float(np.mean(eta * eta))


def is_convex_closed(
    q: Array,
    area_floor: float,
    closure_tol: float = 1e-8,
    turning_tol: float = 1e-7,
) -> bool:
    """Convex, positively oriented, closed, and above the area floor.

    A closed polygon whose exterior angles all lie in (0, π) and sum to 2π
    is convex. The area floor keeps P²/(4A) away from the degenerate spike.
    """
    if float(np.linalg.norm(closure(q))) > closure_tol:
        return False
    delta = turning_angles(q)
    if np.any(delta <= turning_tol) or np.any(delta >= np.pi - turning_tol):
        return False
    if abs(float(np.sum(delta)) - 2.0 * np.pi) > 1e-6:
        return False
    return signed_area(q) >= area_floor
