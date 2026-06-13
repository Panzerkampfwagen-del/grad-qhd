"""
Ablation sweeps for the GradHD optimizer — the core scientific content of Part B.

GradHD extends Adam with three gated coefficients (alpha, beta, gamma); with all
three at 0 it is exactly Adam (proven in tests/test_optimizer.py).  This script
sweeps each coefficient independently on a light dataset (MNIST by default) and
reports how it moves final test accuracy relative to the Adam anchor.

Because this GradHD variant carries NO Hessian-vector term, the prompt's
use_hvp ablation does not apply; the alpha/beta/gamma sweeps take its place.

Each sweep includes the value 0.0, which is the Adam reduction — a built-in
sanity check that the 0.0 point matches a separately-trained Adam baseline.

Usage (from project root):
  python -m partB_gradhd.run_ablation                       # all sweeps, MNIST
  python -m partB_gradhd.run_ablation --sweeps alpha gamma
  python -m partB_gradhd.run_ablation --dataset mnist --epochs 5 --seeds 0 1 2
"""
from __future__ import annotations
import os
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    ABLATION_SWEEPS, ABLATION_DATASET, ABLATION_EPOCHS,
    ABLATION_SEEDS, ABLATION_LR, RESULTS_ROOT,
)
from partB_gradhd.train import print_env, train_model


def run_one_config(
    dataset: str,
    label: str,
    opt_cfg: dict,
    seeds: list[int],
    epochs: int,
    device,
    num_workers: int,
    pin_memory: bool,
) -> dict:
    """Train one optimizer config over seeds; return aggregated accuracy."""
    accs = []
    for seed in seeds:
        res = train_model(
            dataset, label, opt_cfg, seed, epochs, device,
            num_workers=num_workers, pin_memory=pin_memory, verbose=False,
        )
        accs.append(res.test_acc)
    accs = np.array(accs)
    print(f"  {label:<16} test acc {accs.mean():.4f} ± {accs.std():.4f}")
    return {"label": label, "cfg": opt_cfg,
            "mean": float(accs.mean()), "std": float(accs.std()),
            "accs": accs.tolist()}


def run_sweep(
    param: str,
    values: list[float],
    dataset: str,
    seeds: list[int],
    epochs: int,
    device,
    num_workers: int,
    pin_memory: bool,
) -> list[dict]:
    """Sweep one coefficient, holding the other two at 0."""
    print(f"\n── sweep {param} ∈ {values} ───────────────")
    rows = []
    for v in values:
        cfg = {"kind": "gradhd", "lr": ABLATION_LR,
               "alpha": 0.0, "beta": 0.0, "gamma": 0.0}
        cfg[param] = v
        label = f"{param}={v:g}"
        row = run_one_config(dataset, label, cfg, seeds, epochs, device,
                             num_workers, pin_memory)
        row.update({"param": param, "value": v})
        rows.append(row)
    return rows


def adam_baseline(dataset, seeds, epochs, device, num_workers, pin_memory) -> dict:
    """Train a true Adam baseline for cross-checking the 0.0 sweep anchor."""
    print("\n── Adam baseline (cross-check for the 0.0 anchor) ──")
    cfg = {"kind": "adam", "lr": ABLATION_LR}
    return run_one_config(dataset, "adam", cfg, seeds, epochs, device,
                          num_workers, pin_memory)


def plot_sweeps(sweeps: dict[str, list[dict]], adam: dict,
                dataset: str, outdir: str) -> str:
    """One panel per swept coefficient: test acc vs value, Adam line overlaid."""
    n = len(sweeps)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    for col, (param, rows) in enumerate(sweeps.items()):
        ax = axes[0, col]
        xs = [r["value"] for r in rows]
        ys = [r["mean"] for r in rows]
        es = [r["std"] for r in rows]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=4,
                    color="#d62728", label="GradHD")
        ax.axhline(adam["mean"], color="#1f77b4", ls="--", label="Adam")
        ax.fill_between([min(xs), max(xs)],
                        adam["mean"] - adam["std"], adam["mean"] + adam["std"],
                        color="#1f77b4", alpha=0.15)
        ax.set_xlabel(param)
        ax.set_ylabel("test accuracy" if col == 0 else "")
        ax.set_title(f"{param} sweep")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"{dataset}: GradHD coefficient ablations "
                 f"(0.0 = Adam reduction)", y=1.02)
    fig.tight_layout()
    path = os.path.join(outdir, f"{dataset}_ablation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def write_ablation_csv(path: str, sweeps: dict[str, list[dict]],
                       adam: dict) -> None:
    """Write one row per config (sweeps + Adam baseline)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "value", "label", "test_acc_mean", "test_acc_std"])
        for param, rows in sweeps.items():
            for r in rows:
                w.writerow([param, r["value"], r["label"],
                            f"{r['mean']:.6f}", f"{r['std']:.6f}"])
        w.writerow(["baseline", "", "adam",
                    f"{adam['mean']:.6f}", f"{adam['std']:.6f}"])


def check_anchor(sweeps: dict[str, list[dict]], adam: dict) -> None:
    """Sanity report: the 0.0 sweep points should match the Adam baseline.

    GradHD(0,0,0) is algebraically identical to Adam, but each is trained from
    its own seeded init/order, so accuracies match closely, not bit-for-bit.
    """
    print("\nAnchor check (GradHD coeff=0 vs Adam):")
    for param, rows in sweeps.items():
        zero = next((r for r in rows if r["value"] == 0.0), None)
        if zero is None:
            continue
        d = zero["mean"] - adam["mean"]
        print(f"  {param}=0: {zero['mean']:.4f}  vs adam {adam['mean']:.4f}"
              f"  (Δ={d:+.4f})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=ABLATION_DATASET,
                   choices=["mnist", "cifar10", "mri"])
    p.add_argument("--sweeps", nargs="+", default=list(ABLATION_SWEEPS),
                   choices=list(ABLATION_SWEEPS))
    p.add_argument("--epochs", type=int, default=ABLATION_EPOCHS)
    p.add_argument("--seeds", nargs="+", type=int, default=ABLATION_SEEDS)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-pin", action="store_true")
    p.add_argument("--outdir", default="figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = print_env()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    pin = not args.no_pin

    print(f"\nAblation on {args.dataset}  epochs={args.epochs}  "
          f"seeds={args.seeds}  sweeps={args.sweeps}")

    adam = adam_baseline(args.dataset, args.seeds, args.epochs, device,
                         args.num_workers, pin)

    sweeps: dict[str, list[dict]] = {}
    for param in args.sweeps:
        sweeps[param] = run_sweep(
            param, ABLATION_SWEEPS[param], args.dataset, args.seeds,
            args.epochs, device, args.num_workers, pin,
        )

    csv_path = os.path.join(RESULTS_ROOT, f"{args.dataset}_ablation.csv")
    write_ablation_csv(csv_path, sweeps, adam)
    fig_path = plot_sweeps(sweeps, adam, args.dataset, args.outdir)
    check_anchor(sweeps, adam)
    print(f"\nSaved: {csv_path}, {fig_path}")


if __name__ == "__main__":
    main()
