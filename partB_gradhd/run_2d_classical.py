"""
GradHD (and Adam, SGD) on the SAME 2D landscapes as Part A (Addition 4).

This closes the logical gap in the project: Part A runs gradient-based QHD on 2D
functions, Part B runs GradHD on neural nets — different problem classes.  Here
we run the classical GradHD optimizer on the exact Part A landscapes and measure
the same P[near] metric, to test directly whether the classical optimizer gets
trapped in local minima where gradient-based QHD did not.

Reuse:
  - partA_qhd_pde/test_functions.py : function values, analytic gradients, boxes,
    minima.  We feed the analytic gradient into p.grad (no torch reimplementation
    of the landscapes, no autograd) so the existing gradient code is the source.
  - partB_gradhd/train.py : build_optimizer (adam / sgd / gradhd).

Vectorization: the 1000 random inits are optimized as one (1000, 2) tensor.  This
is exactly 1000 independent 2D runs ONLY because the GradHD configs use beta=0;
the beta term is the optimizer's single cross-element operation (mean of g^2).
We assert beta==0 to keep that guarantee.
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from partA_qhd_pde.test_functions import FUNCTIONS
from partB_gradhd.train import build_optimizer
from config import RESULTS_ROOT

STABLE = ["styblinski_tang", "michalewicz", "cube_wave", "rastrigin"]
DISPLAY = {
    "styblinski_tang": "Styblinski-Tang",
    "michalewicz": "Michalewicz",
    "cube_wave": "Cube-Wave",
    "rastrigin": "Rastrigin",
}
K = 200
N_RUNS = 1000
DELTA = 1.0

# Part A PDE results (grad-QHD, std-QHD), copied from partA_qhd_pde/run_all.py
# at N=128, K=200, 1000 baseline inits.  Do NOT re-run the PDE here.
PART_A_QHD = {
    "styblinski_tang": {"grad-QHD": (-23.0563, 0.1055), "std-QHD": (-19.0229, 0.0254)},
    "michalewicz":     {"grad-QHD": (-1.2991, 0.9025),  "std-QHD": (-1.0290, 0.7111)},
    "cube_wave":       {"grad-QHD": (0.0614, 0.9832),   "std-QHD": (0.0991, 0.9555)},
    "rastrigin":       {"grad-QHD": (2.6753, 0.1526),   "std-QHD": (3.3980, 0.1228)},
}

# Classical optimizers.  Same lr for the two adaptive methods; SGD smaller.
# GradHD uses Part B's representative alpha/gamma (beta MUST be 0, see header).
CLASSICAL = {
    "Adam":   {"kind": "adam",   "lr": 0.05},
    "SGDM":   {"kind": "sgd",    "lr": 0.01, "momentum": 0.9},
    "GradHD": {"kind": "gradhd", "lr": 0.05, "alpha": -0.05, "beta": 0.0, "gamma": 1.0},
}


def _random_inits(box, n: int, seed: int) -> np.ndarray:
    """n uniform random (x, y) points in the box."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(box[0], box[1], n)
    y = rng.uniform(box[2], box[3], n)
    return np.stack([x, y], axis=1)


def optimize_batch(name: str, opt_cfg: dict, n_runs: int = N_RUNS,
                   K: int = K, seed: int = 0) -> tuple[float, float]:
    """Optimize n_runs random inits on `name`; return (E[f_final], P[near])."""
    assert opt_cfg.get("beta", 0.0) == 0.0, "vectorized runs require beta=0"
    reg = FUNCTIONS[name]
    f_fn, grad_fn, box, f_star = reg["f"], reg["grad"], reg["box"], reg["min"]

    pts = torch.tensor(_random_inits(box, n_runs, seed), dtype=torch.float64,
                       requires_grad=True)
    extra = {k: v for k, v in opt_cfg.items() if k not in ("kind", "lr")}
    opt = build_optimizer(opt_cfg["kind"], [pts], opt_cfg["lr"], **extra)

    for _ in range(K):
        opt.zero_grad(set_to_none=True)
        xy = pts.detach().numpy()
        gx, gy = grad_fn(xy[:, 0], xy[:, 1])
        pts.grad = torch.tensor(np.stack([gx, gy], axis=1), dtype=pts.dtype)
        opt.step()

    xy = pts.detach().numpy()
    fvals = f_fn(xy[:, 0], xy[:, 1])
    ef = float(np.mean(fvals))
    pnear = float(np.mean((fvals - f_star) <= DELTA))
    return ef, pnear


def gradhd_trajectories(name: str, n_paths: int, K: int,
                        seed: int) -> np.ndarray:
    """Record GradHD paths (K+1, n_paths, 2) for the trapped-trajectory figure."""
    reg = FUNCTIONS[name]
    grad_fn, box = reg["grad"], reg["box"]
    cfg = CLASSICAL["GradHD"]
    pts = torch.tensor(_random_inits(box, n_paths, seed), dtype=torch.float64,
                       requires_grad=True)
    extra = {k: v for k, v in cfg.items() if k not in ("kind", "lr")}
    opt = build_optimizer(cfg["kind"], [pts], cfg["lr"], **extra)
    traj = [pts.detach().numpy().copy()]
    for _ in range(K):
        opt.zero_grad(set_to_none=True)
        xy = pts.detach().numpy()
        gx, gy = grad_fn(xy[:, 0], xy[:, 1])
        pts.grad = torch.tensor(np.stack([gx, gy], axis=1), dtype=pts.dtype)
        opt.step()
        traj.append(pts.detach().numpy().copy())
    return np.array(traj)


def plot_trapped(name: str, outpath: str, n_paths: int = 12, seed: int = 1) -> None:
    """Overlay GradHD paths on the landscape contour; mark starts, ends, global."""
    reg = FUNCTIONS[name]
    f_fn, box, f_star = reg["f"], reg["box"], reg["min"]
    traj = gradhd_trajectories(name, n_paths, K, seed)

    gx = np.linspace(box[0], box[1], 300)
    gy = np.linspace(box[2], box[3], 300)
    GX, GY = np.meshgrid(gx, gy)
    F = f_fn(GX, GY)

    fig, ax = plt.subplots(figsize=(7, 6))
    cf = ax.contourf(GX, GY, F, levels=40, cmap="viridis", alpha=0.9)
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label="f(x, y)")
    for p in range(n_paths):
        ax.plot(traj[:, p, 0], traj[:, p, 1], color="white", lw=1.0, alpha=0.8)
        ax.plot(traj[0, p, 0], traj[0, p, 1], "o", color="cyan",
                markersize=5, markeredgecolor="k", markeredgewidth=0.4)
        ax.plot(traj[-1, p, 0], traj[-1, p, 1], "X", color="red",
                markersize=8, markeredgecolor="k", markeredgewidth=0.4)
    ax.plot(0.0, 0.0, marker="*", color="gold", markersize=20,
            markeredgecolor="k", label="global min")
    ax.set_title(f"GradHD on {DISPLAY[name]}: paths trap at local minima\n"
                 f"(cyan = start, red X = end, gold ★ = global)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved: {outpath}")


def _md_table(rows: dict, metric_idx: int, fmt: str) -> str:
    """Build a markdown table; rows[func][method] = (E[f], P[near])."""
    methods = ["grad-QHD", "std-QHD", "Adam", "SGDM", "GradHD"]
    lines = ["| Function | " + " | ".join(methods) + " |",
             "|" + "---|" * (len(methods) + 1)]
    for name in STABLE:
        cells = [format(rows[name][m][metric_idx], fmt) for m in methods]
        lines.append(f"| {DISPLAY[name]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    os.makedirs("figures", exist_ok=True)
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    # rows[func][method] = (E[f], P[near])
    rows: dict = {name: dict(PART_A_QHD[name]) for name in STABLE}
    for name in STABLE:
        print(f"\n── {DISPLAY[name]} ──")
        for mname, cfg in CLASSICAL.items():
            ef, pnear = optimize_batch(name, cfg)
            rows[name][mname] = (ef, pnear)
            print(f"  {mname:8s}  E[f]={ef:9.4f}  P[near]={pnear:.4f}")

    # CSV
    csv_path = os.path.join(RESULTS_ROOT, "gradhd_2d_classical.csv")
    with open(csv_path, "w") as fh:
        fh.write("function,method,Ef,Pnear\n")
        for name in STABLE:
            for m in ["grad-QHD", "std-QHD", "Adam", "SGDM", "GradHD"]:
                ef, p = rows[name][m]
                fh.write(f"{name},{m},{ef:.6f},{p:.6f}\n")

    print("\n\n### P[near] (fraction within delta=1 of the global minimum)\n")
    print(_md_table(rows, 1, ".3f"))
    print("\n### E[f] at final point\n")
    print(_md_table(rows, 0, ".3f"))

    plot_trapped("rastrigin", "figures/gradhd_2d_rastrigin_trapped.png")
    print(f"\nSaved CSV: {csv_path}")


if __name__ == "__main__":
    main()
