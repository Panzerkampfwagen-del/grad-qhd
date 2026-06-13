"""
Head-to-head optimizer comparison for the GradHD real-data experiments.

Trains each optimizer (Adam, SGD+momentum, GradHD) on a dataset over several
seeds under a matched budget, logs per-epoch curves and a summary CSV, and
saves two figures: validation-accuracy curves (mean +/- std across seeds) and
a final test-accuracy bar chart with error bars.

Usage (from project root):
  python -m partB_gradhd.run_experiments --dataset mnist
  python -m partB_gradhd.run_experiments --dataset cifar10 --epochs 30 --seeds 0 1 2
  python -m partB_gradhd.run_experiments --dataset mnist --epochs 1 \
        --seeds 0 --max-batches 50            # fast smoke run
"""
from __future__ import annotations
import os
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import EXPERIMENT_OPTIMIZERS, SEEDS, EPOCHS, RESULTS_ROOT
from partB_gradhd.train import (
    print_env, train_model, RunResult,
    write_epochs_csv, write_summary_csv, default_results_path,
)


COLORS = {"adam": "#1f77b4", "sgd": "#2ca02c", "gradhd": "#d62728"}


def _color(name: str, idx: int) -> str:
    palette = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]
    return COLORS.get(name, palette[idx % len(palette)])


def group_by_optimizer(results: list[RunResult]) -> dict[str, list[RunResult]]:
    """Group run results by optimizer name, preserving first-seen order."""
    groups: dict[str, list[RunResult]] = defaultdict(list)
    for r in results:
        groups[r.optimizer].append(r)
    return groups


def plot_val_curves(results: list[RunResult], dataset: str, outdir: str) -> str:
    """Validation-accuracy vs epoch, mean +/- std band across seeds."""
    groups = group_by_optimizer(results)
    fig, ax = plt.subplots(figsize=(7, 5))
    for idx, (name, runs) in enumerate(groups.items()):
        # Align by epoch index (all runs share the same epoch count).
        curves = np.array([[e["val_acc"] for e in r.epochs] for r in runs])
        epochs = np.arange(1, curves.shape[1] + 1)
        mean, std = curves.mean(0), curves.std(0)
        c = _color(name, idx)
        ax.plot(epochs, mean, label=name, color=c, lw=2)
        ax.fill_between(epochs, mean - std, mean + std, color=c, alpha=0.2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy")
    ax.set_title(f"{dataset}: validation accuracy ({len(next(iter(groups.values())))} seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    path = os.path.join(outdir, f"{dataset}_val_curves.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_final_accuracy(results: list[RunResult], dataset: str, outdir: str) -> str:
    """Bar chart of final test accuracy, mean +/- std across seeds."""
    groups = group_by_optimizer(results)
    names = list(groups.keys())
    means = [np.mean([r.test_acc for r in groups[n]]) for n in names]
    stds = [np.std([r.test_acc for r in groups[n]]) for n in names]
    colors = [_color(n, i) for i, n in enumerate(names)]

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=6, color=colors, alpha=0.85)
    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m + s + 0.005, f"{m:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("test accuracy")
    ax.set_title(f"{dataset}: final test accuracy (mean +/- std)")
    ax.grid(axis="y", alpha=0.3)
    path = os.path.join(outdir, f"{dataset}_final_acc.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def print_summary_table(results: list[RunResult], dataset: str) -> None:
    """Print test acc (mean +/- std) and mean epoch time per optimizer."""
    groups = group_by_optimizer(results)
    print()
    print("=" * 64)
    print(f"{dataset} — head-to-head ({len(next(iter(groups.values())))} seeds)")
    print(f"{'optimizer':<10} {'test acc':>16} {'epoch time (s)':>16}")
    print("-" * 64)
    for name, runs in groups.items():
        accs = [r.test_acc for r in runs]
        t = np.mean([r.mean_epoch_time() for r in runs])
        print(f"{name:<10} {np.mean(accs):>10.4f} ± {np.std(accs):<4.4f} {t:>16.2f}")
    print("=" * 64)


def print_markdown_table(results: list[RunResult], dataset: str,
                         n_seeds: int) -> None:
    """Print the head-to-head as markdown (test acc + mean s/epoch over seeds)."""
    groups = group_by_optimizer(results)
    print(f"\n**{dataset} ({n_seeds} seeds)**\n")
    print("| optimizer | test acc | s/epoch |")
    print("|---|---|---|")
    for name, runs in groups.items():
        accs = np.array([r.test_acc for r in runs])
        spe = float(np.mean([r.mean_epoch_time() for r in runs]))
        print(f"| {name} | {accs.mean():.4f} ± {accs.std():.4f} | {spe:.2f} |")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10", "mri"])
    p.add_argument("--optimizers", nargs="+", default=list(EXPERIMENT_OPTIMIZERS),
                   choices=list(EXPERIMENT_OPTIMIZERS))
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--epochs", type=int, default=None,
                   help="Override per-dataset default epoch count.")
    p.add_argument("--max-batches", type=int, default=None,
                   help="Cap train/eval batches per epoch (smoke runs).")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-pin", action="store_true", help="Disable pin_memory.")
    p.add_argument("--outdir", default="figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = print_env()
    epochs = args.epochs if args.epochs is not None else EPOCHS[args.dataset]
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    print(f"\nDataset={args.dataset}  epochs={epochs}  seeds={args.seeds}  "
          f"optimizers={args.optimizers}")
    if args.max_batches:
        print(f"(smoke run: max_batches={args.max_batches})")

    results: list[RunResult] = []
    for name in args.optimizers:
        opt_cfg = EXPERIMENT_OPTIMIZERS[name]
        print(f"\n── {name}  {opt_cfg} ─────────────────")
        for seed in args.seeds:
            res = train_model(
                args.dataset, name, opt_cfg, seed, epochs, device,
                max_batches=args.max_batches,
                num_workers=args.num_workers, pin_memory=not args.no_pin,
            )
            results.append(res)

    # CSV logs
    ep_csv = default_results_path(args.dataset, "epochs")
    sum_csv = default_results_path(args.dataset, "summary")
    write_epochs_csv(ep_csv, results)
    write_summary_csv(sum_csv, results)
    print(f"\nSaved CSVs: {ep_csv}, {sum_csv}")

    # Figures
    f1 = plot_val_curves(results, args.dataset, args.outdir)
    f2 = plot_final_accuracy(results, args.dataset, args.outdir)
    print(f"Saved figures: {f1}, {f2}")

    print_summary_table(results, args.dataset)
    print_markdown_table(results, args.dataset, len(args.seeds))


if __name__ == "__main__":
    main()
