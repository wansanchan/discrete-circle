"""Metropolis sampling of closed convex polygons with weight exp(−V/T).

Proposals rotate one edge by +θ and another by −θ, rescale them by e^{+s}
and e^{−s}, then subtract the mean edge. The pair of opposite moves is
equally likely, and the 2-D scaling Jacobians e^{2s} e^{−2s} cancel, so the
proposal is symmetric on the closure surface. Moves that leave the convex
component, or that drop the area below the floor, are rejected.
"""

from __future__ import annotations

import math

import numpy as np

from geometry import (
    is_convex_closed,
    mean_alignment,
    mean_eta2,
    pi_eff,
    project_closed,
    regular_area,
    regular_edges,
    winding,
)
from hamiltonian import Params, potential

Array = np.ndarray


def _rotate(v: Array, theta: float) -> Array:
    c = math.cos(theta)
    s = math.sin(theta)
    x, y = float(v[0]), float(v[1])
    return np.array([c * x - s * y, s * x + c * y])


def propose(
    q: Array,
    rng: np.random.Generator,
    step_angle: float,
    step_scale: float,
) -> Array:
    n = q.shape[0]
    i, j = rng.choice(n, size=2, replace=False)
    theta = float(rng.normal(0.0, step_angle))
    scale = float(rng.normal(0.0, step_scale))
    out = q.copy()
    out[i] = _rotate(out[i], theta) * math.exp(scale)
    out[j] = _rotate(out[j], -theta) * math.exp(-scale)
    return project_closed(out)


def _acceptance_probability(delta_v: float, temperature: float, rng: np.random.Generator) -> bool:
    if delta_v <= 0.0:
        return True
    if temperature == 0.0:
        return False
    return float(rng.random()) < math.exp(-delta_v / temperature)


def tune_step(
    n: int,
    params: Params,
    temperature: float,
    rng: np.random.Generator,
    area_floor: float,
    n_pilot: int = 400,
) -> tuple[float, float]:
    """Pick a fixed proposal width from a discarded pilot, then restart later."""
    if temperature == 0.0:
        return 0.0, 0.0
    step_angle = 0.4 * math.sqrt(temperature / params.beta)
    step_scale = 0.4 * math.sqrt(temperature / params.mu)
    q = regular_edges(n, length=params.length)
    v = potential(q, params)
    accepted = 0
    window = 0
    for t in range(n_pilot):
        trial = propose(q, rng, step_angle, step_scale)
        window += 1
        if is_convex_closed(trial, area_floor):
            v_trial = potential(trial, params)
            if _acceptance_probability(v_trial - v, temperature, rng):
                q, v = trial, v_trial
                accepted += 1
        if window == 400 and t < n_pilot - 1:
            rate = accepted / window
            if rate < 0.25:
                step_angle *= 0.7
                step_scale *= 0.7
            elif rate > 0.6:
                step_angle *= 1.4
                step_scale *= 1.4
            accepted = 0
            window = 0
    return step_angle, step_scale


def _block_stderr(values: list[float], n_blocks: int = 8) -> float:
    """Standard error of the mean from contiguous blocks, so autocorrelation counts."""
    arr = np.asarray(values, dtype=float)
    if arr.size < n_blocks * 2:
        return float("nan")
    blocks = np.array_split(arr, n_blocks)
    means = np.array([float(np.mean(block)) for block in blocks])
    return float(np.std(means, ddof=1) / np.sqrt(n_blocks))


def run_chain(
    n: int,
    params: Params,
    temperature: float,
    n_steps: int,
    seed: int,
    burn_fraction: float = 0.2,
) -> dict[str, float]:
    """Sample the convex component. Temperature 0 returns the regular polygon."""
    area_floor = 0.25 * regular_area(n, params.length)
    if temperature == 0.0:
        q = regular_edges(n, length=params.length)
        return {
            "pi_mean": float(pi_eff(q)),
            "pi_std": 0.0,
            "align_mean": mean_alignment(q),
            "winding_fraction": 1.0,
            "closure_max": 0.0,
            "eta2_mean": mean_eta2(q),
            "acceptance": 1.0,
            "n_samples": 1.0,
            "step_angle": 0.0,
            "step_scale": 0.0,
            "pi_err": 0.0,
            "eta2_err": 0.0,
        }

    rng = np.random.default_rng(seed)
    step_angle, step_scale = tune_step(n, params, temperature, rng, area_floor)
    # Production chain starts again at the minimum, with the step frozen.
    rng = np.random.default_rng(seed + 17)
    q = regular_edges(n, length=params.length)
    v = potential(q, params)
    burn = int(burn_fraction * n_steps)
    pi_samples: list[float] = []
    align_samples: list[float] = []
    eta_samples: list[float] = []
    closure_max = 0.0
    winding_ok = 0
    recorded = 0
    accepted = 0

    for t in range(n_steps):
        trial = propose(q, rng, step_angle, step_scale)
        take = False
        if is_convex_closed(trial, area_floor):
            v_trial = potential(trial, params)
            if _acceptance_probability(v_trial - v, temperature, rng):
                q, v = trial, v_trial
                take = True
        if take:
            accepted += 1
        if t < burn:
            continue
        recorded += 1
        pi_samples.append(float(pi_eff(q)))
        align_samples.append(mean_alignment(q))
        eta_samples.append(mean_eta2(q))
        closure_max = max(closure_max, float(np.linalg.norm(np.sum(q, axis=0))))
        if abs(winding(q) - 1.0) < 1e-6:
            winding_ok += 1

    pi_arr = np.asarray(pi_samples)
    return {
        "pi_mean": float(np.mean(pi_arr)),
        "pi_std": float(np.std(pi_arr, ddof=1)) if recorded > 1 else 0.0,
        "pi_err": _block_stderr(pi_samples),
        "eta2_err": _block_stderr(eta_samples),
        "align_mean": float(np.mean(align_samples)),
        "winding_fraction": winding_ok / recorded,
        "closure_max": closure_max,
        "eta2_mean": float(np.mean(eta_samples)),
        "acceptance": accepted / n_steps,
        "n_samples": float(recorded),
        "step_angle": step_angle,
        "step_scale": step_scale,
    }
