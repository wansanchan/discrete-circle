"""Scan regular and thermal polygons and write the π convergence table.

Run from the repository root, with the project virtualenv:

    .venv/bin/python src/convergence.py

The classical curve is the exact regular value. The thermal curve is one
Metropolis chain per (N, T). Seeds are fixed.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hamiltonian import Params, fluctuation_coefficient, quadratic_shift
from simulate import run_chain

# Small-N check only. Larger grids were not run: each chain is sequential Python.
NS = (4, 6, 8)
TEMPERATURES = (0.0, 1e-4, 1e-3)
STEPS = {
    0.0: 1,
    1e-4: 4_000,
    1e-3: 4_000,
}
SLOPE_N = 6
SLOPE_STEPS = 6_000
RESULTS = ROOT / "results"


def _seed(n: int, temperature: float) -> int:
    return 10_000 * n + int(round(temperature * 1_000_000))


def fit_classical() -> dict[str, float]:
    ns = np.arange(16, 513, dtype=float)
    y = ns * np.tan(np.pi / ns) - np.pi
    design = np.column_stack((ns ** -2, ns ** -4, ns ** -6, ns ** -8))
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    a_exact = np.pi**3 / 3.0
    b_exact = 2.0 * np.pi**5 / 15.0
    c_exact = 17.0 * np.pi**7 / 315.0
    return {
        "a": float(coef[0]),
        "b": float(coef[1]),
        "c": float(coef[2]),
        "a_exact": float(a_exact),
        "b_exact": float(b_exact),
        "c_exact": float(c_exact),
        "a_rel": float(abs(coef[0] - a_exact) / a_exact),
        "b_rel": float(abs(coef[1] - b_exact) / b_exact),
        "c_rel": float(abs(coef[2] - c_exact) / c_exact),
    }


def main() -> None:
    params = Params()
    RESULTS.mkdir(parents=True, exist_ok=True)
    classical = fit_classical()
    coefficients: dict[int, dict] = {}
    for n in NS:
        print(f"hessian N={n}", flush=True)
        coefficients[n] = fluctuation_coefficient(n, params)

    rows: list[dict[str, float]] = []
    for n in NS:
        coeff = coefficients[n]
        for temperature in TEMPERATURES:
            n_steps = SLOPE_STEPS if n == SLOPE_N and temperature > 0.0 else STEPS[temperature]
            print(f"chain N={n} T={temperature} steps={n_steps}", flush=True)
            stats = run_chain(
                n,
                params,
                temperature,
                n_steps=n_steps,
                seed=_seed(n, temperature),
            )
            d_pi, eta2 = quadratic_shift(temperature, coeff)
            pi_reg = float(n * np.tan(np.pi / n))
            rows.append(
                {
                    "N": n,
                    "T": temperature,
                    "pi_mean": stats["pi_mean"],
                    "pi_std": stats["pi_std"],
                    "pi_reg": pi_reg,
                    "pi_shift": stats["pi_mean"] - pi_reg,
                    "quad_shift": d_pi,
                    "align_mean": stats["align_mean"],
                    "winding_fraction": stats["winding_fraction"],
                    "closure_max": stats["closure_max"],
                    "eta2_mean": stats["eta2_mean"],
                    "eta2_quad": eta2,
                    "C_eta": float(coeff["C_eta"]),
                    "pi_err": stats["pi_err"],
                    "eta2_err": stats["eta2_err"],
                    "acceptance": stats["acceptance"],
                    "n_samples": stats["n_samples"],
                    "n_zero_modes": float(coeff["n_zero_modes"]),
                }
            )

    fieldnames = list(rows[0].keys())
    csv_path = RESULTS / "convergence.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = RESULTS / "series_fit.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(classical.keys()))
        writer.writeheader()
        writer.writerow(classical)

    _plot(rows, classical)
    _report_slope(rows)
    print(f"wrote {csv_path}", flush=True)


def _report_slope(rows: list[dict[str, float]]) -> None:
    thermal = [r for r in rows if r["N"] == SLOPE_N and r["T"] > 0.0 and r["eta2_mean"] > 0.0]
    if len(thermal) < 2:
        print("not enough slope points")
        return
    x = np.array([r["eta2_mean"] for r in thermal])
    y = np.array([r["pi_shift"] for r in thermal])
    slope, intercept = np.polyfit(x, y, 1)
    c_eta = thermal[0]["C_eta"]
    rel = abs(slope - c_eta) / abs(c_eta)
    print(
        f"slope={slope:.6g} intercept={intercept:.6g} C_eta={c_eta:.6g} rel={rel:.3%}",
        flush=True,
    )


def _plot(rows: list[dict[str, float]], classical: dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ns = np.arange(8, 513, dtype=float)
    exact = np.abs(ns * np.tan(np.pi / ns) - np.pi)
    ax.loglog(ns, exact, color="black", lw=1.4, label=r"regular $|N\tan(\pi/N)-\pi|$")
    guide = classical["a_exact"] / ns**2
    ax.loglog(ns, guide, color="0.45", ls="--", lw=1.0, label=r"$\pi^3/(3N^2)$")

    thermal_t = 1e-3
    thermal = [r for r in rows if r["T"] == thermal_t]
    ax.loglog(
        [r["N"] for r in thermal],
        [abs(r["pi_mean"] - np.pi) for r in thermal],
        "o",
        color="C0",
        label=rf"Gibbs $T={thermal_t:g}$",
    )
    ax.set_xlabel("N")
    ax.set_ylabel(r"$|\pi_{\mathrm{eff}}-\pi|$")
    ax.set_title("Discrete circle: metric error versus refinement")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "pi_convergence.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
