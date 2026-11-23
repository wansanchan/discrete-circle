"""Effective Hamiltonian whose unique convex minimizer is the regular polygon.

The potential depends on edge lengths, exterior angles, and the alignment of
inward normals with the ray from each edge midpoint to the centroid. Those
are rigid-motion invariants. The centroid is not a background point: it is
computed from the edges, and it becomes the common intersection of the
normal lines only at the minimum.

The simulation samples the classical Gibbs weight exp(−V/T). The commutator
[q, p] = iħ is recorded in the note as the quantization of this phase space;
this module does not diagonalize that operator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geometry import (
    alignment,
    angle_deviation,
    edge_lengths,
    pi_eff,
    regular_edges,
)

Array = np.ndarray


@dataclass(frozen=True)
class Params:
    """Couplings of V_N. ``length`` is the preferred edge length L."""

    mu: float = 1.0
    beta: float = 1.0
    kappa: float = 1.0
    length: float = 1.0

    def __post_init__(self) -> None:
        if self.mu <= 0.0 or self.beta <= 0.0 or self.kappa <= 0.0:
            raise ValueError("mu, beta, kappa must be positive")
        if self.length <= 0.0:
            raise ValueError("length must be positive")


def potential_parts(q: Array, params: Params) -> tuple[float, float, float]:
    """Return (V_len, V_turn, V_rad), each separately non-negative."""
    lengths = edge_lengths(q)
    eta = angle_deviation(q)
    # 1 − n·û is zero at perfect alignment and non-negative after squaring.
    misalignment = 1.0 - alignment(q)
    v_len = 0.5 * params.mu * float(np.sum((lengths - params.length) ** 2))
    v_turn = 0.5 * params.beta * float(np.sum(eta * eta))
    v_rad = 0.5 * params.kappa * float(np.sum(misalignment * misalignment))
    return v_len, v_turn, v_rad


def potential(q: Array, params: Params) -> float:
    v_len, v_turn, v_rad = potential_parts(q, params)
    return v_len + v_turn + v_rad


def forces(q: Array, params: Params, step: float = 1e-6) -> Array:
    """Central-difference approximation of F = −∇V in the ambient edge space.

    On a regular polygon the analytic gradient vanishes, including along the
    global rotation, so this returns a numerical zero there. The closure
    multiplier λ is not included: it does no work on the constraint surface
    and is enforced by projection in the sampler.
    """
    if step <= 0.0:
        raise ValueError("step must be positive")
    grad = np.zeros_like(q)
    for i in range(q.shape[0]):
        for a in range(2):
            plus = q.copy()
            minus = q.copy()
            plus[i, a] += step
            minus[i, a] -= step
            grad[i, a] = (potential(plus, params) - potential(minus, params)) / (2.0 * step)
    return -grad


def _unpack(v: Array, n: int) -> Array:
    """Chart on sum q = 0: the last edge is minus the sum of the others."""
    q = np.zeros((n, 2))
    q[:-1] = v.reshape(n - 1, 2)
    q[-1] = -np.sum(q[:-1], axis=0)
    return q


def _hessian(f, v0: Array, n: int, step: float) -> Array:
    """Central Hessian of f(unpack(v)) at v0."""
    dim = v0.size
    hess = np.zeros((dim, dim))
    eye = np.eye(dim)
    for i in range(dim):
        for j in range(i, dim):
            hpp = f(_unpack(v0 + step * eye[i] + step * eye[j], n))
            hpm = f(_unpack(v0 + step * eye[i] - step * eye[j], n))
            hmp = f(_unpack(v0 - step * eye[i] + step * eye[j], n))
            hmm = f(_unpack(v0 - step * eye[i] - step * eye[j], n))
            val = (hpp - hpm - hmp + hmm) / (4.0 * step * step)
            hess[i, j] = val
            hess[j, i] = val
    return hess


def _angle_jacobian(v0: Array, n: int, step: float) -> Array:
    """Jacobian J_{k a} = ∂η_k / ∂v_a at the regular polygon."""
    dim = v0.size
    base = angle_deviation(_unpack(v0, n))
    jac = np.zeros((n, dim))
    eye = np.eye(dim)
    for a in range(dim):
        plus = angle_deviation(_unpack(v0 + step * eye[a], n))
        minus = angle_deviation(_unpack(v0 - step * eye[a], n))
        jac[:, a] = (plus - minus) / (2.0 * step)
    # ``base`` is ~0; the two-sided difference does not use it.
    del base
    return jac


def rotation_chart_mode(n: int, length: float) -> Array:
    """Unit vector, in the closure chart, of an infinitesimal global rotation.

    δq = R_{π/2} q preserves lengths, angles, alignment, and closure, so it
    is a null direction of both V and π_eff.
    """
    q = regular_edges(n, length=length)
    dq = np.column_stack((-q[:, 1], q[:, 0]))
    mode = dq[:-1].ravel()
    norm = float(np.linalg.norm(mode))
    if norm == 0.0:
        raise RuntimeError("rotation mode vanished")
    return mode / norm


def fluctuation_coefficient(
    n: int,
    params: Params | None = None,
    step: float = 1e-5,
    eig_floor: float = 1e-5,
) -> dict[str, float | Array]:
    """Gaussian prediction for ⟨π_eff − π_reg⟩ / ⟨η²⟩ at small T.

    On the closure surface, V ≈ (1/2) vᵀ H_V v and
    π_eff ≈ π_reg + (1/2) vᵀ H_π v. The Gibbs covariance is T H_V⁺,
    with the rotational zero mode deflated before the inverse. Then

        ⟨Δπ⟩ = (T/2) Tr(H_π H_V⁺),
        ⟨η²⟩ = (T/n) Tr(J H_V⁺ Jᵀ),

    and C_η = ⟨Δπ⟩ / ⟨η²⟩ is independent of T inside the quadratic well.
    """
    if params is None:
        params = Params()
    q0 = regular_edges(n, length=params.length)
    v0 = q0[:-1].ravel().copy()
    pi0 = pi_eff(q0)

    def v_of(q: Array) -> float:
        return potential(q, params)

    def pi_of(q: Array) -> float:
        return pi_eff(q) - pi0

    h_v = _hessian(v_of, v0, n, step)
    h_pi = _hessian(pi_of, v0, n, step)
    jac = _angle_jacobian(v0, n, step)

    mode = rotation_chart_mode(n, params.length)
    # Deflate the exact symmetry so a finite-difference residual cannot
    # dominate the pseudo-inverse.
    projector = np.eye(mode.size) - np.outer(mode, mode)
    h_v = projector @ h_v @ projector

    evals, evecs = np.linalg.eigh(0.5 * (h_v + h_v.T))
    positive = evals > eig_floor
    if int(np.sum(positive)) < 1:
        raise RuntimeError("no positive eigenvalues of H_V")
    inv_evals = np.zeros_like(evals)
    inv_evals[positive] = 1.0 / evals[positive]
    h_inv = (evecs * inv_evals) @ evecs.T

    delta_pi_over_t = 0.5 * float(np.trace(h_pi @ h_inv))
    eta_cov_over_t = jac @ h_inv @ jac.T
    eta2_over_t = float(np.trace(eta_cov_over_t)) / n
    if eta2_over_t <= 0.0:
        raise RuntimeError("angle variance vanished")
    return {
        "C_eta": delta_pi_over_t / eta2_over_t,
        "delta_pi_over_T": delta_pi_over_t,
        "eta2_over_T": eta2_over_t,
        "n_zero_modes": float(np.sum(~positive)),
        "min_positive_eigenvalue": float(np.min(evals[positive])),
        "rotation_quadratic_form": float(mode @ h_v @ mode),
    }


def quadratic_shift(temperature: float, coeff: dict[str, float | Array]) -> tuple[float, float]:
    """Return (⟨Δπ⟩, ⟨η²⟩) predicted by the quadratic well at this T."""
    return (
        temperature * float(coeff["delta_pi_over_T"]),
        temperature * float(coeff["eta2_over_T"]),
    )
