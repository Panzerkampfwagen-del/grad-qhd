"""
Mitigation 2 — Layer-by-layer training.

Trains the VQC one layer at a time:
  Stage 1: only parameters in layer 0 trainable (encoder + classifier always trainable).
  Stage 2: layers 0-1 trainable.
  ...until all layers are active.

Uses the same GLOBAL cost function as the baseline (no combination with Mitigation 1).
The encoder and classifier are always trainable throughout.

Trains on brain-MRI over 3 seeds and reports:
  - Test accuracy mean ± std
  - Training curve showing accuracy at each layer-unlock checkpoint
"""
from __future__ import annotations
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from qmlcore.data import set_seed, get_device, get_mri_loaders
from qmlcore.model import QCQCNN
from qmlcore.train import evaluate, _vqc_grad_var

SEEDS = config.SEEDS
LR = 5e-4
N_QUBITS = config.N_QUBITS
N_LAYERS = config.N_LAYERS
EPOCHS_PER_STAGE = 5  # epochs to train at each layer-unlock stage
TOTAL_STAGES = N_LAYERS  # unlock one layer per stage

FIGURES = os.path.join(os.path.dirname(__file__), "..", "figures")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def _vqc_param_names(n_layers: int) -> list[list[str]]:
    """Group VQC weight parameter names by layer index."""
    # TorchLayer registers one param 'weights' of shape (n_layers, n_qubits, 2)
    # Gradients are masked by slicing, not by parameter names.
    return [f"layer_{l}" for l in range(n_layers)]


def _freeze_vqc_above(model: QCQCNN, active_layers: int) -> None:
    """Only keep the first `active_layers` VQC weight slices trainable."""
    for name, param in model.named_parameters():
        if "vqc" in name and "weights" in name:
            param.requires_grad_(True)
            # We mask the gradient of inactive layers via a hook (see below)


class _GradMask:
    """Hook that zeros gradient entries for inactive VQC layers."""

    def __init__(self, active_layers: int, n_layers: int) -> None:
        self.active_layers = active_layers
        self.n_layers = n_layers

    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(grad)
        mask[: self.active_layers] = 1.0
        return grad * mask


def run_lbl(seed: int, device: torch.device) -> dict:
    """Train one model with layer-by-layer schedule; return history + test_acc."""
    set_seed(seed)
    train_loader, val_loader, test_loader = get_mri_loaders(
        config.MRI_ROOT, config.IMG_SIZE, config.BATCH_SIZE, config.VAL_FRAC, seed,
    )
    model = QCQCNN(n_qubits=N_QUBITS, n_layers=N_LAYERS, n_classes=4, cost="global")
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    history: list[dict] = []
    unlock_epochs: list[int] = []  # epoch at which each stage starts
    grad_var_per_stage: list[float] = []
    epoch_counter = 0

    for stage in range(1, TOTAL_STAGES + 1):
        print(f"  [seed={seed}] Stage {stage}/{TOTAL_STAGES} "
              f"(VQC layers 0..{stage-1} active)")
        unlock_epochs.append(epoch_counter + 1)

        # Attach gradient mask hook for this stage
        hooks = []
        for name, param in model.named_parameters():
            if "vqc" in name and "weights" in name:
                h = param.register_hook(_GradMask(stage, N_LAYERS))
                hooks.append(h)

        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=LR
        )

        stage_gv = float("nan")
        for ep in range(EPOCHS_PER_STAGE):
            epoch_counter += 1
            model.train()
            total_loss = correct = n = 0
            first_batch = True
            t0 = time.perf_counter()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                if first_batch and ep == 0:
                    stage_gv = _vqc_grad_var(model)
                    first_batch = False
                optimizer.step()
                total_loss += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                n += len(y)
            dt = time.perf_counter() - t0
            va_loss, va_acc = evaluate(model, val_loader, criterion, device)
            tr_loss, tr_acc = total_loss / n, correct / n
            history.append(dict(
                epoch=epoch_counter, stage=stage,
                train_loss=tr_loss, train_acc=tr_acc,
                val_loss=va_loss, val_acc=va_acc,
                epoch_time_s=dt,
            ))
            print(f"    ep {epoch_counter:2d}  "
                  f"tr {tr_loss:.4f}/{tr_acc:.3f}  "
                  f"val {va_loss:.4f}/{va_acc:.3f}  {dt:.1f}s")

        grad_var_per_stage.append(stage_gv)
        for h in hooks:
            h.remove()

    te_loss, te_acc = evaluate(model, test_loader, criterion, device)
    return {
        "seed": seed,
        "history": history,
        "test_acc": te_acc,
        "test_loss": te_loss,
        "unlock_epochs": unlock_epochs,
        "grad_var_per_stage": grad_var_per_stage,
    }


def plot_lbl_curves(all_results: list[dict]) -> None:
    os.makedirs(FIGURES, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    epochs_all = [h["epoch"] for h in all_results[0]["history"]]
    unlock_epochs = all_results[0]["unlock_epochs"]

    for ax, metric, title in [
        (axes[0], "val_acc", "Validation accuracy"),
        (axes[1], "train_loss", "Training loss"),
    ]:
        vals = np.array([[h[metric] for h in r["history"]] for r in all_results])
        ax.plot(epochs_all, vals.mean(0), "b-", lw=2, label="LBL (mean ± std)")
        ax.fill_between(epochs_all,
                        vals.mean(0) - vals.std(0),
                        vals.mean(0) + vals.std(0), alpha=0.2, color="blue")
        for i, ue in enumerate(unlock_epochs):
            ax.axvline(ue, color="gray", ls="--", lw=0.8,
                       label=f"Unlock layer {i}" if i == 0 else None)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Mitigation 2: Layer-by-layer training on Brain MRI", y=1.02)
    fig.tight_layout()
    path = os.path.join(FIGURES, "layer_by_layer_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def print_summary(all_results: list[dict]) -> None:
    accs = np.array([r["test_acc"] for r in all_results])
    print("\n" + "=" * 60)
    print("Mitigation 2 — Layer-by-layer training summary")
    print("=" * 60)
    print(f"  test acc  : {accs.mean():.4f} ± {accs.std():.4f}")
    print(f"  (seeds: {[r['seed'] for r in all_results]})")
    print("\n  Grad var per stage (first batch of stage, seed 0):")
    for i, gv in enumerate(all_results[0]["grad_var_per_stage"]):
        print(f"    stage {i+1} (layers 0..{i} active): {gv:.4e}")


def save_results_csv(all_results: list[dict]) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "layer_by_layer_summary.csv")
    with open(path, "w") as fh:
        fh.write("seed,test_acc\n")
        for r in all_results:
            fh.write(f"{r['seed']},{r['test_acc']:.6f}\n")
    print(f"Saved CSV: {path}")


def main() -> None:
    device = get_device()
    print("=" * 60)
    print("Mitigation 2 — Layer-by-layer training")
    print(f"Device: {device}  |  n_qubits={N_QUBITS}  n_layers={N_LAYERS}")
    print(f"Stages: {TOTAL_STAGES}  |  Epochs/stage: {EPOCHS_PER_STAGE}  |  Seeds: {SEEDS}")
    print(f"Total epochs per seed: {TOTAL_STAGES * EPOCHS_PER_STAGE}")
    print("=" * 60)

    t0 = time.time()
    all_results = []
    for seed in SEEDS:
        print(f"\n── Seed {seed} ──")
        all_results.append(run_lbl(seed, device))

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")
    plot_lbl_curves(all_results)
    print_summary(all_results)
    save_results_csv(all_results)


if __name__ == "__main__":
    main()
