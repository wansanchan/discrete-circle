import numpy as np
import pytest

from geometry import (
    closure,
    mean_alignment,
    pi_eff,
    regular_area,
    regular_edges,
    regular_inradius,
    regular_pi_eff,
    signed_area,
    turning_angles,
    winding,
)
from hamiltonian import Params, forces, potential, potential_parts


def test_regular_pi_matches_formula_through_n_512():
    for n in range(3, 513):
        q = regular_edges(n, length=1.0, alpha0=0.15)
        got = pi_eff(q)
        exact = regular_pi_eff(n)
        assert abs(got - exact) < 1e-12


def test_series_coefficients_from_n_16():
    ns = np.arange(16, 513, dtype=float)
    y = np.array([regular_pi_eff(int(n)) - np.pi for n in ns])
    # The N^{-8} column keeps the x^9 term in tan from biasing c.
    design = np.column_stack((ns ** -2, ns ** -4, ns ** -6, ns ** -8))
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    a_exact = np.pi**3 / 3.0
    b_exact = 2.0 * np.pi**5 / 15.0
    c_exact = 17.0 * np.pi**7 / 315.0
    assert abs(coef[0] - a_exact) / a_exact < 1e-3
    assert abs(coef[1] - b_exact) / b_exact < 1e-2
    assert abs(coef[2] - c_exact) / c_exact < 1e-2


def test_two_term_fit_is_biased_by_the_sixth_order_but_three_term_holds():
    """The plan's two-term fit from N=16 absorbs an O(1%) sixth-order piece.

    The three-term extraction is the one that recovers π³/3 and 2π⁵/15
    inside the stated tolerances. This test pins both facts.
    """
    ns = np.arange(16, 513, dtype=float)
    y = np.array([regular_pi_eff(int(n)) - np.pi for n in ns])
    two, *_ = np.linalg.lstsq(np.column_stack((ns ** -2, ns ** -4)), y, rcond=None)
    a_exact = np.pi**3 / 3.0
    b_exact = 2.0 * np.pi**5 / 15.0
    # Two-term a stays inside 0.1%; b does not, because of the x^7 term in tan.
    assert abs(two[0] - a_exact) / a_exact < 1e-3
    assert abs(two[1] - b_exact) / b_exact > 1e-2


def test_inradius_and_area_identities():
    for n in (3, 4, 6, 17):
        length = 1.3
        area = regular_area(n, length)
        radius = regular_inradius(n, length)
        semi = 0.5 * n * length
        assert area == pytest.approx(radius * semi)
        q = regular_edges(n, length=length)
        assert signed_area(q) == pytest.approx(area, rel=1e-12, abs=1e-12)


def test_regular_alignment_and_winding():
    q = regular_edges(8, length=1.0, alpha0=-0.4)
    assert np.linalg.norm(closure(q)) < 1e-12
    assert winding(q) == pytest.approx(1.0)
    assert mean_alignment(q) == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(turning_angles(q), 2.0 * np.pi / 8)


def test_rectangle_and_rhombus_split_the_potential():
    params = Params(length=1.0)
    square = regular_edges(4)
    assert potential(square, params) == pytest.approx(0.0, abs=1e-12)

    # 2×1 rectangle: equiangular, normals meet, side lengths are not all L.
    rectangle = np.array([[2.0, 0.0], [0.0, 1.0], [-2.0, 0.0], [0.0, -1.0]])
    v_len, v_turn, v_rad = potential_parts(rectangle, params)
    assert mean_alignment(rectangle) == pytest.approx(1.0, abs=1e-12)
    assert v_rad == pytest.approx(0.0, abs=1e-12)
    assert v_turn == pytest.approx(0.0, abs=1e-12)
    assert v_len > 0.0

    # 60° rhombus: equilateral, but the touch point of the incircle is not
    # the midpoint, so the midpoint-normal misses the centroid.
    s = np.sqrt(3.0) / 2.0
    rhombus = np.array([[1.0, 0.0], [0.5, s], [-1.0, 0.0], [-0.5, -s]])
    v_len, v_turn, v_rad = potential_parts(rhombus, params)
    assert np.linalg.norm(closure(rhombus)) < 1e-12
    assert mean_alignment(rhombus) < 0.99
    assert v_len == pytest.approx(0.0, abs=1e-12)
    assert v_rad > 0.0
    assert v_turn > 0.0


def test_force_vanishes_on_the_regular_polygon():
    params = Params()
    q = regular_edges(6, alpha0=0.3)
    f = forces(q, params, step=1e-6)
    assert np.linalg.norm(f) < 1e-7


def test_force_is_a_descent_direction():
    params = Params()
    q = regular_edges(5)
    q = q.copy()
    q[0] *= 1.05
    q = q - np.mean(q, axis=0)
    f = forces(q, params, step=1e-6)
    v0 = potential(q, params)
    step = 1e-3
    moved = q + step * f / np.linalg.norm(f)
    assert potential(moved, params) < v0
