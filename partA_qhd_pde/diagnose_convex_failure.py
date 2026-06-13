"""
Diagnose why the convex quartic is excluded from the Part A comparison.

The convex landscape f(x,y) = (x+y)^4/256 + (x-y)^4/128 is non-periodic on its
box, so Fourier (spectral) differentiation — which assumes periodicity — produces
Gibbs oscillations in the spectral Laplacian, magnitude up to ~1e3.  That feeds
an ill-conditioned Crank-Nicolson H2 solve and the norm blows up.  This is a
limitation of the spectral method on a non-periodic problem, NOT a failure of
the QHD algorithm.

Produces:
  figures/convex_laplacian_gibbs.png   spectral |Lap f| heatmap + 1D slice vs analytic
  figures/convex_norm_drift.png        ||psi|| vs step, convex vs Styblinski-Tang
and prints a power-iteration estimate of the CN system condition number for the
convex case vs a stable case.
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from partA_qhd_pde.qhd_sim import simulate
from partA_qhd_pde.operators import (
    precompute_landscape, k_grids, _apply_H2_to,
)
from partA_qhd_pde.test_functions import FUNCTIONS, make_grid
from config import GRAD_QHD_CONFIGS

N = 128


def convex_laplacian_analytic(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Exact Laplacian of the convex quartic: 3u^2/32 + 3v^2/16, u=x+y, v=x-y."""
    u, v = X + Y, X - Y
    return 3 * u ** 2 / 32 + 3 * v ** 2 / 16


def plot_laplacian_gibbs(outpath: str) -> float:
    """Heatmap of the spectral |Lap f| for convex + a 1D slice vs the analytic.

    Returns the max spectral |Lap f| (the Gibbs magnitude) for reporting.
    """
    reg = FUNCTIONS["convex"]
    box = reg["box"]
    X, Y = make_grid(box, N=N)
    dx = (box[1] - box[0]) / N
    dy = (box[3] - box[2]) / N
    _f, _gx, _gy, _g2, lapf = precompute_landscape(
        X, Y, reg["f"], reg["grad"], dx, dy
    )
    lap_an = convex_laplacian_analytic(X, Y)
    gibbs_max = float(np.abs(lapf).max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Heatmap of spectral |Lap f| (log scale spans the Gibbs oscillations).
    im = ax1.pcolormesh(X, Y, np.abs(lapf) + 1e-3, shading="auto", cmap="magma",
                        norm=LogNorm(vmin=1e-1, vmax=gibbs_max))
    ax1.set_title("spectral |∇²f| (convex), log scale")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_aspect("equal")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # 1D slice at y ~ 0: spectral vs analytic.
    j = N // 2
    xs = X[:, j]
    ax2.plot(xs, lapf[:, j], color="#d62728", lw=1.2,
             label="spectral (FFT)")
    ax2.plot(xs, lap_an[:, j], color="#1f77b4", lw=2.0, ls="--",
             label="analytic")
    ax2.set_title(f"slice at y≈0: spectral oscillates, analytic smooth\n"
                  f"max spectral |∇²f| = {gibbs_max:.0f}")
    ax2.set_xlabel("x")
    ax2.set_ylabel("∇²f")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved: {outpath}  (max spectral |Lap f| = {gibbs_max:.1f})")
    return gibbs_max


def _norm_trace(name: str, renorm: bool) -> np.ndarray:
    """Run gradient-based QHD and return the norm_history (||psi|| per step)."""
    reg = FUNCTIONS[name]
    cfg = GRAD_QHD_CONFIGS[name]
    res = simulate(
        reg["f"], reg["grad"], reg["box"], N=N, t0=cfg["t0"], h=cfg["h"], K=200,
        alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"],
        f_star=reg["min"], delta=1.0,
        x0=cfg["x0"], y0=cfg["y0"], sigma=cfg["sigma"],
        renormalize=renorm, verbose=False,
    )
    return res.norm_history


def plot_norm_drift(outpath: str) -> dict:
    """Two panels: raw norm (no renorm, log) and per-step norm (production)."""
    traces = {
        ("convex", False): _norm_trace("convex", False),
        ("styblinski_tang", False): _norm_trace("styblinski_tang", False),
        ("convex", True): _norm_trace("convex", True),
        ("styblinski_tang", True): _norm_trace("styblinski_tang", True),
    }
    steps = np.arange(201)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Raw norm, no renormalization (true evolution); log scale for the blowup.
    ax1.semilogy(steps, traces[("convex", False)], color="#d62728", lw=2,
                 label="convex (unstable)")
    ax1.semilogy(steps, traces[("styblinski_tang", False)], color="#1f77b4",
                 lw=2, label="Styblinski-Tang (stable)")
    ax1.axhline(1.0, color="k", lw=0.6, ls=":")
    ax1.set_title("raw ||psi|| (no renormalization)")
    ax1.set_xlabel("step")
    ax1.set_ylabel("||psi||  (log)")
    ax1.legend()
    ax1.grid(alpha=0.3, which="both")

    # Per-step norm with production renormalization (resets each step).
    ax2.plot(steps, traces[("convex", True)], color="#d62728", lw=2,
             label="convex (unstable)")
    ax2.plot(steps, traces[("styblinski_tang", True)], color="#1f77b4", lw=2,
             label="Styblinski-Tang (stable)")
    ax2.axhline(1.0, color="k", lw=0.6, ls=":")
    ax2.set_title("per-step ||psi|| (production renormalization)")
    ax2.set_xlabel("step")
    ax2.set_ylabel("||psi|| after one step")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    summary = {
        "convex_raw_final": float(traces[("convex", False)][-1]),
        "styb_raw_final": float(traces[("styblinski_tang", False)][-1]),
        "convex_perstep_max": float(traces[("convex", True)].max()),
        "styb_perstep_max": float(traces[("styblinski_tang", True)].max()),
    }
    print(f"Saved: {outpath}")
    print(f"  convex raw ||psi|| at step 200 = {summary['convex_raw_final']:.3e}")
    print(f"  Styblinski raw ||psi|| at step 200 = {summary['styb_raw_final']:.3e}")
    return summary


def estimate_cn_condition(name: str, iters: int = 60) -> dict:
    """Power-iteration estimate of the CN (Cayley) system condition number.

    The CN solve is (I - half*L) where half = -i*h*alpha/4 and L = H2 action.
    L is (near) normal, so its largest-magnitude eigenvalue mu_max (estimated by
    power iteration) gives ||half*L|| = (h|alpha|/4)*mu_max and a condition
    number kappa ~ sqrt(1 + ||half*L||^2) (eigenvalues 1 + i*(h*alpha/4)*mu lie
    on a line, min |.| = 1 at mu=0).
    """
    reg = FUNCTIONS[name]
    cfg = GRAD_QHD_CONFIGS[name]
    box = reg["box"]
    X, Y = make_grid(box, N=N)
    dx = (box[1] - box[0]) / N
    dy = (box[3] - box[2]) / N
    _f, gfx, gfy, _g2, lapf = precompute_landscape(
        X, Y, reg["f"], reg["grad"], dx, dy
    )
    KX, KY = k_grids(N, dx, dy)

    rng = np.random.default_rng(0)
    v = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))
    v /= np.linalg.norm(v)
    mu = 0.0
    for _ in range(iters):
        w = _apply_H2_to(v, KX, KY, gfx, gfy, lapf)
        mu = float(np.linalg.norm(w))
        if mu == 0.0:
            break
        v = w / mu

    half_mag = cfg["h"] * abs(cfg["alpha"]) / 4.0
    half_L_norm = half_mag * mu
    kappa = float(np.sqrt(1.0 + half_L_norm ** 2))
    return {"name": name, "mu_max": mu, "half_L_norm": half_L_norm, "kappa": kappa}


def main() -> None:
    os.makedirs("figures", exist_ok=True)
    print("=== 1. spectral Laplacian Gibbs ===")
    plot_laplacian_gibbs("figures/convex_laplacian_gibbs.png")

    print("\n=== 2. norm drift / blowup ===")
    plot_norm_drift("figures/convex_norm_drift.png")

    print("\n=== 3. CN system condition-number estimate (power iteration) ===")
    for name in ("convex", "styblinski_tang"):
        d = estimate_cn_condition(name)
        print(f"  {d['name']:16s}  ||L_H2||~{d['mu_max']:.1f}  "
              f"||half*L||~{d['half_L_norm']:.2f}  kappa(CN)~{d['kappa']:.1f}")


if __name__ == "__main__":
    main()
