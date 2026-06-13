"""
Reproduce Styblinski-Tang results from Leng & Shi (ICML 2025, Fig. 4).

Runs 4 methods on the Styblinski-Tang landscape and produces:
  <outdir>/styblinski_metrics.png   (E[f], E[||grad f||^2], P vs step)
  <outdir>/styblinski_density.png   (|psi|^2 density snapshots)

Usage (from project root):
  python partA_qhd_pde/run_2d.py [--N 128] [--K 200] [--n-runs 1000]
  python partA_qhd_pde/run_2d.py --N 64 --K 50 --n-runs 100  # quick smoke test
"""
from __future__ import annotations
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive; remove this line for interactive display
import matplotlib.pyplot as plt

from partA_qhd_pde.qhd_sim import simulate
from partA_qhd_pde.baselines import run_sgdm, run_nag
from partA_qhd_pde.plotting import plot_metrics, plot_density_snapshots
from partA_qhd_pde.test_functions import (
    styblinski_tang, styblinski_tang_grad,
    STYBLINSKI_BOX, STYBLINSKI_MIN, make_grid,
)
from partA_qhd_pde.operators import precompute_landscape


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QHD vs baselines — Styblinski-Tang")
    p.add_argument("--N",      type=int, default=128, help="Grid points per dim")
    p.add_argument("--K",      type=int, default=200, help="QHD time steps")
    p.add_argument("--n-runs", type=int, default=1000, help="Baseline trajectories")
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--outdir", default="figures", help="Output directory for figures")
    return p.parse_args()


def print_table(methods: dict) -> None:
    print()
    print("=" * 60)
    print("Terminal values at step K")
    print(f"{'Method':<14} {'E[f]':>10} {'E[||∇f||²]':>14} {'P[f−f*≤1]':>12}")
    print("-" * 54)
    for name, res in methods.items():
        print(f"{name:<14} {res.ef[-1]:>10.4f} {res.eg2[-1]:>14.4f} {res.prob[-1]:>12.4f}")
    print("=" * 60)


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    box = STYBLINSKI_BOX    # (-5, 5, -5, 5)
    f_star = STYBLINSKI_MIN  # ≈ -31.33
    alpha, beta, gamma = -0.05, 0.0, 5.0
    h, t0, sigma = 0.01, 1.0, 1.0
    delta = 1.0
    snap_steps = [0, args.K // 4, args.K // 2, args.K]

    print("=" * 60)
    print("Part A — Styblinski-Tang  (Leng & Shi, ICML 2025)")
    print(f"N={args.N}  K={args.K}  h={h}  α={alpha}  β={beta}  γ={gamma}")
    print("=" * 60)

    print(f"\n[1/4] Gradient-based QHD  (α={alpha}, β={beta}, γ={gamma})")
    grad_qhd = simulate(
        styblinski_tang, styblinski_tang_grad, box,
        N=args.N, t0=t0, h=h, K=args.K,
        alpha=alpha, beta=beta, gamma=gamma,
        f_star=f_star, delta=delta,
        x0=0.0, y0=0.0, sigma=sigma,
        snapshot_steps=snap_steps, verbose=True,
    )
    drift = abs(grad_qhd.norm_history[-1] - 1.0)
    print(f"  → final norm {grad_qhd.norm_history[-1]:.6f}  (drift {drift:.2e})")

    print("\n[2/4] Standard QHD  (α=β=γ=0)")
    std_qhd = simulate(
        styblinski_tang, styblinski_tang_grad, box,
        N=args.N, t0=t0, h=h, K=args.K,
        alpha=0.0, beta=0.0, gamma=0.0,
        f_star=f_star, delta=delta,
        x0=0.0, y0=0.0, sigma=sigma,
        snapshot_steps=[], verbose=True,
    )

    print(f"\n[3/4] SGDM  ({args.n_runs} runs, s0=0.01, μ=0.9)")
    sgdm = run_sgdm(
        styblinski_tang, styblinski_tang_grad, box,
        K=args.K, n_runs=args.n_runs,
        s0=0.01, mu=0.9,
        f_star=f_star, delta=delta, seed=args.seed,
    )

    print(f"\n[4/4] NAG  ({args.n_runs} runs, s=0.01)")
    nag = run_nag(
        styblinski_tang, styblinski_tang_grad, box,
        K=args.K, n_runs=args.n_runs,
        s=0.01, f_star=f_star, delta=delta, seed=args.seed,
    )

    methods = {
        "grad-QHD": grad_qhd,
        "std-QHD":  std_qhd,
        "SGDM":     sgdm,
        "NAG":      nag,
    }
    print_table(methods)

    # --- metrics figure ---
    metrics_path = os.path.join(args.outdir, "styblinski_metrics.png")
    fig_m = plot_metrics(
        methods,
        title="Styblinski-Tang: grad-QHD vs baselines (Leng & Shi, 2025)",
        output_path=metrics_path,
    )
    plt.close(fig_m)

    # --- density snapshots ---
    if grad_qhd.snapshots:
        X, Y = make_grid(box, N=args.N)
        dx = (box[1] - box[0]) / args.N
        dy = (box[3] - box[2]) / args.N
        f_grid, *_ = precompute_landscape(
            X, Y, styblinski_tang, styblinski_tang_grad, dx, dy
        )
        density_path = os.path.join(args.outdir, "styblinski_density.png")
        fig_d = plot_density_snapshots(
            grad_qhd.snapshots, X, Y, f_grid,
            title=r"$|\psi|^2$ density — grad-QHD on Styblinski-Tang",
            output_path=density_path,
        )
        plt.close(fig_d)

    print(f"\nFigures written to {args.outdir}/")


if __name__ == "__main__":
    main()
