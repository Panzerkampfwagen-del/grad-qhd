"""
|psi|^2 evolution figures for Part A.

Reuses the existing gradient-based QHD simulation (qhd_sim.simulate) and the
repo's stable per-function parameters (config.GRAD_QHD_CONFIGS).  Captures the
normalized density at steps {0, 50, 100, 200} and draws a 4-panel figure with
the function's contours and global minimum overlaid.

Norm convention: the simulation normalizes discretely, sum_ij |psi_ij|^2 = 1
(probability mass over grid points), NOT the continuous integral
int |psi|^2 dx dy = 1.  So the norm check below verifies sum|psi|^2 ~ 1 and
also prints the raw pre-renormalization step-norm from norm_history; it does
NOT multiply by dx*dy, which under this convention would equal dx*dy, not 1.
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from partA_qhd_pde.qhd_sim import simulate
from partA_qhd_pde.test_functions import FUNCTIONS, make_grid
from config import GRAD_QHD_CONFIGS

SNAP_STEPS = [0, 50, 100, 200]

# Known global-minimum locations for the overlay marker (not in the registry).
MINIMA_XY = {
    "styblinski_tang": (-2.903534, -2.903534),
    "rastrigin": (0.0, 0.0),
}


def run_and_plot(name: str, outpath: str) -> None:
    """Simulate gradient-based QHD on `name` and save the 4-panel density figure."""
    reg = FUNCTIONS[name]
    f_fn, grad_fn = reg["f"], reg["grad"]
    box, f_star = reg["box"], reg["min"]
    cfg = GRAD_QHD_CONFIGS[name]

    res = simulate(
        f_fn, grad_fn, box, N=128, t0=cfg["t0"], h=cfg["h"], K=200,
        alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"],
        f_star=f_star, delta=1.0,
        x0=cfg["x0"], y0=cfg["y0"], sigma=cfg["sigma"],
        snapshot_steps=SNAP_STEPS, renormalize=True, verbose=False,
    )
    snaps = dict(res.snapshots)  # {step: normalized density}

    X, Y = make_grid(box, N=128)
    F = f_fn(X, Y)

    # Norm check (repo convention: sum|psi|^2 = 1).
    print(f"\n[{name}] norm at captured steps:")
    for k in SNAP_STEPS:
        mass = float(snaps[k].sum())               # normalized prob mass (= 1)
        raw = float(res.norm_history[k])           # pre-renorm step-norm ||psi||
        finite = bool(np.isfinite(snaps[k]).all())
        print(f"  step {k:3d}:  sum|psi|^2 = {mass:.6f}   "
              f"raw step-norm ||psi|| = {raw:.6f}   finite={finite}")
        assert abs(mass - 1.0) < 1e-6, f"density not normalized at step {k}: {mass}"
        assert 0.9 < raw < 1.1, f"step-norm drifted out of bounds at {k}: {raw}"

    # Shared color scale; use a high percentile so one sharp peak does not wash
    # out the earlier, broader panels.
    vmax = float(np.percentile(np.stack([snaps[k] for k in SNAP_STEPS]), 99.7))

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.7))
    mx, my = MINIMA_XY[name]
    pcm = None
    for ax, k in zip(axes, SNAP_STEPS):
        pcm = ax.pcolormesh(X, Y, snaps[k], shading="auto", cmap="inferno",
                            vmin=0.0, vmax=vmax)
        ax.contour(X, Y, F, levels=12, colors="white", linewidths=0.4, alpha=0.5)
        ax.plot(mx, my, marker="*", color="cyan", markersize=16,
                markeredgecolor="black", markeredgewidth=0.6, label="global min")
        ax.set_title(f"step {k}")
        ax.set_xlabel("x")
        ax.set_aspect("equal")
    axes[0].set_ylabel("y")
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.7)
    fig.colorbar(pcm, ax=axes, fraction=0.012, pad=0.01,
                 label="|psi|^2 (normalized, shared scale)")
    fig.suptitle(f"{name}: |psi|^2 evolution under gradient-based QHD", y=1.03)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


def main() -> None:
    os.makedirs("figures", exist_ok=True)
    run_and_plot("styblinski_tang", "figures/psi_evolution_styblinski.png")
    run_and_plot("rastrigin", "figures/psi_evolution_rastrigin.png")


if __name__ == "__main__":
    main()
