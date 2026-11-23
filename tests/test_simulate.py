import numpy as np

from geometry import is_convex_closed, regular_area, regular_edges, winding
from hamiltonian import Params
from simulate import run_chain


def test_short_chain_stays_closed_convex_and_wound_once():
    params = Params()
    n = 16
    out = run_chain(n, params, temperature=1e-3, n_steps=400, seed=2)
    assert out["closure_max"] < 1e-10
    assert out["winding_fraction"] == 1.0
    assert out["n_samples"] > 100
    q = regular_edges(n)
    assert is_convex_closed(q, area_floor=0.25 * regular_area(n))
    assert abs(winding(q) - 1.0) < 1e-12
    assert np.isfinite(out["pi_mean"])
    assert out["acceptance"] > 0.05
