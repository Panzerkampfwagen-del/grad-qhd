"""
Run all 5 Part A test functions and produce summary figures + tables.

Usage (from project root):
  python partA_qhd_pde/run_all.py [--N 128] [--K 200] [--n-runs 1000]
  python partA_qhd_pde/run_all.py --N 64 --K 50 --n-runs 200  # quick smoke test
"""
from __future__ import annotations
import sys
import os
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from partA_qhd_pde.qhd_sim import simulate
from partA_qhd_pde.baselines import run_sgdm, run_nag
from partA_qhd_pde.plotting import plot_metrics
from partA_qhd_pde.test_functions import FUNCTIONS, make_grid
from partA_qhd_pde.operators import precompute_landscape
from config import GRAD_QHD_CONFIGS, BASELINE_CONFIGS


DISPLAY_NAMES = {
    "convex":         "Convex",
    "styblinski_tang": "Styblinski-Tang",
    "michalewicz":    "Michalewicz",
    "cube_wave":      "Cube-Wave",
    "rastrigin":      "Rastrigin",
}


def run_one_function(
    name: str,
    N: int,
    n_runs: int,
    seed: int,
    outdir: str,
    K_override: int | None = None,
) -> dict:
    """Run grad-QHD, std-QHD, SGDM, NAG on a single landscape."""
    reg = FUNCTIONS[name]
    f_fn, grad_fn = reg["f"], reg["grad"]
    box, f_star = reg["box"], reg["min"]
    cfg = GRAD_QHD_CONFIGS[name]
    bcfg = BASELINE_CONFIGS

    alpha, beta, gamma = cfg["alpha"], cfg["beta"], cfg["gamma"]
    h, t0 = cfg["h"], cfg["t0"]
    K = K_override if K_override is not None else cfg["K"]
    x0, y0, sigma = cfg["x0"], cfg["y0"], cfg["sigma"]
    delta = 1.0

    snap_steps = [0, K // 4, K // 2, K]

    t0_wall = time.time()
    grad_qhd = simulate(
        f_fn, grad_fn, box, N=N, t0=t0, h=h, K=K,
        alpha=alpha, beta=beta, gamma=gamma,
        f_star=f_star, delta=delta,
        x0=x0, y0=y0, sigma=sigma,
        snapshot_steps=snap_steps, renormalize=True, verbose=True,
    )
    t_gqhd = time.time() - t0_wall

    std_qhd = simulate(
        f_fn, grad_fn, box, N=N, t0=t0, h=h, K=K,
        alpha=0.0, beta=0.0, gamma=0.0,
        f_star=f_star, delta=delta,
        x0=x0, y0=y0, sigma=sigma,
        snapshot_steps=[], renormalize=True, verbose=False,
    )

    sgdm = run_sgdm(
        f_fn, grad_fn, box, K=K, n_runs=n_runs,
        s0=bcfg["sgdm"]["s0"], mu=bcfg["sgdm"]["mu"],
        f_star=f_star, delta=delta, seed=seed,
    )
    nag = run_nag(
        f_fn, grad_fn, box, K=K, n_runs=n_runs,
        s=bcfg["nag"]["s"],
        f_star=f_star, delta=delta, seed=seed,
    )

    methods = {
        "grad-QHD": grad_qhd,
        "std-QHD":  std_qhd,
        "SGDM":     sgdm,
        "NAG":      nag,
    }

    # Save per-function metrics figure
    fig_path = os.path.join(outdir, f"{name}_metrics.png")
    fig = plot_metrics(
        methods,
        title=f"{DISPLAY_NAMES[name]}: grad-QHD vs baselines",
        output_path=fig_path,
    )
    plt.close(fig)

    print(f"  ✓ {DISPLAY_NAMES[name]} done in {t_gqhd:.0f}s  "
          f"(norm drift {abs(grad_qhd.norm_history[-1]-1):.2e})")
    return {"methods": methods, "name": name, "cfg": cfg}


def print_summary_table(all_results: list) -> None:
    """Print a LaTeX-style ASCII table of terminal values for all functions."""
    print()
    print("=" * 78)
    print("Part A — Terminal values at step K (N=128, K=200)")
    header = f"{'Function':<16} {'Method':<12} {'E[f]':>10} {'E[||∇f||²]':>14} {'P[near]':>10}"
    print(header)
    print("-" * 78)
    for entry in all_results:
        methods = entry["methods"]
        fname = DISPLAY_NAMES[entry["name"]]
        for i, (mname, res) in enumerate(methods.items()):
            fn_col = fname if i == 0 else ""
            print(f"{fn_col:<16} {mname:<12} {res.ef[-1]:>10.4f} "
                  f"{res.eg2[-1]:>14.4f} {res.prob[-1]:>10.4f}")
        print("-" * 78)
    print("=" * 78)


def make_summary_figure(all_results: list, outdir: str) -> None:
    """5-column × 3-row panel: one column per function, rows = E[f], P, norm."""
    n_funcs = len(all_results)
    fig, axes = plt.subplots(3, n_funcs, figsize=(4 * n_funcs, 9))
    colors = {"grad-QHD": "#1f77b4", "std-QHD": "#ff7f0e",
              "SGDM": "#2ca02c", "NAG": "#d62728"}
    lws   = {"grad-QHD": 2.5, "std-QHD": 2.0, "SGDM": 1.5, "NAG": 1.5}
    styles= {"grad-QHD": "-", "std-QHD": "--", "SGDM": "-.", "NAG": ":"}

    for col, entry in enumerate(all_results):
        methods = entry["methods"]
        fname = DISPLAY_NAMES[entry["name"]]
        ax_ef, ax_p, ax_n = axes[0, col], axes[1, col], axes[2, col]

        for mname, res in methods.items():
            steps = res.steps
            kw = dict(color=colors[mname], lw=lws[mname], ls=styles[mname], label=mname)
            ax_ef.plot(steps, res.ef, **kw)
            ax_p.plot(steps, res.prob, **kw)
            if hasattr(res, "norm_history"):
                ax_n.plot(steps, res.norm_history, **kw)

        ax_ef.set_title(fname, fontsize=10, fontweight="bold")
        ax_ef.set_ylabel("E[f]" if col == 0 else "")
        ax_p.set_ylabel("P[near opt]" if col == 0 else "")
        ax_n.set_ylabel("‖ψ‖ (pre-renorm)" if col == 0 else "")
        ax_n.set_xlabel("step")
        ax_n.axhline(1.0, color="k", lw=0.5, ls=":")

        for ax in (ax_ef, ax_p, ax_n):
            ax.tick_params(labelsize=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(outdir, "summary_all_functions.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--N",       type=int, default=128)
    p.add_argument("--K",       type=int, default=200)
    p.add_argument("--n-runs",  type=int, default=1000)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--outdir",  default="figures")
    # Convex with h=0.2 causes GMRES instability due to spectral Gibbs in lapf.
    # Include it explicitly with --functions convex if needed for reference.
    _STABLE = [k for k in GRAD_QHD_CONFIGS if k != "convex"]
    p.add_argument("--functions", nargs="+",
                   default=_STABLE,
                   choices=list(GRAD_QHD_CONFIGS.keys()))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 60)
    print(f"Part A — All functions  N={args.N}  n_runs={args.n_runs}")
    print("=" * 60)

    all_results = []
    for fname in args.functions:
        cfg = GRAD_QHD_CONFIGS[fname]
        print(f"\n── {DISPLAY_NAMES[fname]} ─────────────────────────────")
        print(f"   α={cfg['alpha']}  h={cfg['h']}  K={cfg['K']}  σ={cfg['sigma']}")
        result = run_one_function(
            fname, N=args.N, n_runs=args.n_runs,
            seed=args.seed, outdir=args.outdir,
            K_override=args.K if args.K != 200 else None,
        )
        all_results.append(result)

    print_summary_table(all_results)
    make_summary_figure(all_results, args.outdir)


if __name__ == "__main__":
    main()
