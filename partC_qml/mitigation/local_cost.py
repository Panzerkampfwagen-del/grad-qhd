"""
Mitigation 1 — Local cost function.

Replaces the global n-qubit observable Z⊗...⊗Z with the sum of
single-qubit expectation values L_local = (1/n) Σ_i ⟨Z_i⟩.

Local observables avoid exponential gradient vanishing because each
term acts on O(1) qubits (McClean et al. 2018).

Trains both the unmitigated (global) and mitigated (local) QCQ-CNN on
brain-MRI over 3 seeds and reports:
  - VQC gradient variance at epoch 1 (local vs global)
  - Test accuracy mean ± std over 3 seeds (local vs global)
"""
from __future__ import annotations
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from qmlcore.data import set_seed, get_device, get_mri_loaders
from qmlcore.model import QCQCNN
from qmlcore.train import train_model, RunResult

EPOCHS = config.EPOCHS
SEEDS = config.SEEDS
LR = 5e-4
N_QUBITS = config.N_QUBITS
N_LAYERS = config.N_LAYERS

FIGURES = os.path.join(os.path.dirname(__file__), "..", "figures")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def run_variant(cost: str, seed: int, device: torch.device) -> RunResult:
    set_seed(seed)
    train_loader, val_loader, test_loader = get_mri_loaders(
        config.MRI_ROOT, config.IMG_SIZE, config.BATCH_SIZE,
        config.VAL_FRAC, seed,
    )
    model = QCQCNN(n_qubits=N_QUBITS, n_layers=N_LAYERS, n_classes=4, cost=cost)
    print(f"\n  [{cost} seed={seed}] training...")
    result = train_model(
        model, train_loader, val_loader, test_loader,
        optimizer_name=cost, seed=seed,
        epochs=EPOCHS, lr=LR, device=device, verbose=True,
    )
    return result


def plot_training_curves(
    global_runs: list[RunResult], local_runs: list[RunResult]
) -> None:
    os.makedirs(FIGURES, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, title in [
        (axes[0], "val_acc", "Validation accuracy"),
        (axes[1], "train_loss", "Training loss"),
    ]:
        for runs, label, color in [
            (global_runs, "Global (baseline)", "#d62728"),
            (local_runs, "Local (mitigated)", "#1f77b4"),
        ]:
            epochs_arr = [h["epoch"] for h in runs[0].history]
            vals = np.array([[h[metric] for h in r.history] for r in runs])
            ax.plot(epochs_arr, vals.mean(0), label=label, color=color, lw=2)
            ax.fill_between(epochs_arr,
                            vals.mean(0) - vals.std(0),
                            vals.mean(0) + vals.std(0),
                            alpha=0.2, color=color)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Mitigation 1: Local cost vs Global cost on Brain MRI", y=1.02)
    fig.tight_layout()
    path = os.path.join(FIGURES, "local_cost_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def print_summary(global_runs: list[RunResult], local_runs: list[RunResult]) -> None:
    print("\n" + "=" * 60)
    print("Mitigation 1 — Summary")
    print("=" * 60)

    for label, runs in [("Global (baseline)", global_runs),
                        ("Local (mitigated)", local_runs)]:
        accs = np.array([r.test_acc for r in runs])
        gvs = np.array([r.grad_var_epoch1 for r in runs
                        if np.isfinite(r.grad_var_epoch1)])
        print(f"\n{label}:")
        print(f"  test acc  : {accs.mean():.4f} ± {accs.std():.4f}")
        if len(gvs):
            print(f"  grad var  : {gvs.mean():.4e}  (epoch 1, VQC weights)")
        else:
            print(f"  grad var  : n/a")

    # Gradient variance ratio
    gv_g = np.array([r.grad_var_epoch1 for r in global_runs
                     if np.isfinite(r.grad_var_epoch1)])
    gv_l = np.array([r.grad_var_epoch1 for r in local_runs
                     if np.isfinite(r.grad_var_epoch1)])
    if len(gv_g) and len(gv_l) and gv_g.mean() > 0:
        ratio = gv_l.mean() / gv_g.mean()
        print(f"\n  Local/Global grad-var ratio at epoch 1: {ratio:.2f}x")

    print()


def save_results_csv(global_runs: list[RunResult], local_runs: list[RunResult]) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "local_cost_summary.csv")
    with open(path, "w") as fh:
        fh.write("variant,seed,test_acc,grad_var_epoch1\n")
        for label, runs in [("global", global_runs), ("local", local_runs)]:
            for r in runs:
                fh.write(f"{label},{r.seed},{r.test_acc:.6f},"
                         f"{r.grad_var_epoch1:.6e}\n")
    print(f"Saved CSV: {path}")


def main() -> None:
    device = get_device()
    print("=" * 60)
    print("Mitigation 1 — Local cost function")
    print(f"Device: {device}  |  n_qubits={N_QUBITS}  n_layers={N_LAYERS}")
    print(f"Epochs: {EPOCHS}  |  LR: {LR}  |  Seeds: {SEEDS}")
    print("=" * 60)

    t0 = time.time()
    global_runs: list[RunResult] = []
    local_runs: list[RunResult] = []

    print("\n── Global cost (baseline) ──")
    for seed in SEEDS:
        global_runs.append(run_variant("global", seed, device))

    print("\n── Local cost (mitigation) ──")
    for seed in SEEDS:
        local_runs.append(run_variant("local", seed, device))

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")
    plot_training_curves(global_runs, local_runs)
    print_summary(global_runs, local_runs)
    save_results_csv(global_runs, local_runs)


if __name__ == "__main__":
    main()
