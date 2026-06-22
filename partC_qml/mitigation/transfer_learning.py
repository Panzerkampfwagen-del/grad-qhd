"""
Mitigation 3 — Quantum transfer learning.

Architecture:
  Stage 1 (frozen): Classical CNN pre-trained on CIFAR-10 (via grad-qhd checkpoint
    or trained fresh). Output: d-dimensional feature vector (d = n_qubits ≤ 8).
    Uses the grad-qhd CNN backbone if a checkpoint exists, otherwise trains a
    small CNN on CIFAR-10 and saves it.
  Stage 2 (trainable): Shallow VQC (depth ≤ 3, n = d qubits) with angle encoding
    θ_i = π * tanh(x_i) and local Z_i measurements.
  Stage 3 (trainable): Linear(n_qubits, 4) → 4-class MRI logits.

Key idea: the frozen backbone provides discriminative features so the VQC
only needs to learn a small quantum transformation, not to extract features
from raw pixels. This keeps the effective circuit depth and qubit count
in the regime where gradients are not yet exponentially small.

Reports:
  - VQC gradient variance at epoch 1 vs unmitigated QCQ
  - Test accuracy mean ± std over 3 seeds
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
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import config
from qmlcore.data import set_seed, get_device, get_mri_loaders
from qmlcore.circuit import make_torch_layer
from qmlcore.train import train_model, RunResult

# Transfer learning uses a shallower circuit to keep gradients healthy
TL_N_QUBITS = 4
TL_N_LAYERS = 2   # depth ≤ 3 as specified
TL_LR = 5e-4
SEEDS = config.SEEDS
EPOCHS = config.EPOCHS
BACKBONE_CKPT = os.path.join(os.path.dirname(__file__), "..", "results",
                             "cifar10_backbone.pt")
FIGURES = os.path.join(os.path.dirname(__file__), "..", "figures")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


# ── Classical backbone ────────────────────────────────────────────────────────

class ClassicalBackbone(nn.Module):
    """SmallCNN for CIFAR-10 pre-training; output = TL_N_QUBITS features."""

    def __init__(self, n_out: int = TL_N_QUBITS) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.features(x).flatten(1))


def train_cifar10_backbone(device: torch.device, epochs: int = 10) -> None:
    """Train ClassicalBackbone on CIFAR-10 with a 10-class head; save weights."""
    print("  Pre-training CIFAR-10 backbone...")
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2470, 0.2435, 0.2616]),
    ])
    tf_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2470, 0.2435, 0.2616]),
    ])
    # Use the CIFAR-10 data already downloaded by the grad-qhd project.
    data_root = os.path.expanduser("~/grad-qhd/data")
    train_set = datasets.CIFAR10(data_root, train=True, download=False, transform=tf_train)
    test_set = datasets.CIFAR10(data_root, train=False, download=False, transform=tf_eval)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=2, pin_memory=True)

    backbone = ClassicalBackbone(TL_N_QUBITS)
    head = nn.Linear(TL_N_QUBITS, 10)
    model_pt = nn.Sequential(backbone, head).to(device)
    optimizer = torch.optim.Adam(model_pt.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model_pt.train()
        correct = n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model_pt(x), y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                correct += (model_pt(x).argmax(1) == y).sum().item()
                n += len(y)
        model_pt.eval()
        with torch.no_grad():
            te_correct = te_n = 0
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                te_correct += (model_pt(x).argmax(1) == y).sum().item()
                te_n += len(y)
        print(f"    CIFAR10 ep {ep+1}/{epochs}  "
              f"train {correct/n:.3f}  test {te_correct/te_n:.3f}")

    # Save only the backbone weights (discard head)
    os.makedirs(os.path.dirname(BACKBONE_CKPT), exist_ok=True)
    torch.save(backbone.state_dict(), BACKBONE_CKPT)
    print(f"  Backbone saved to {BACKBONE_CKPT}")


def get_backbone(device: torch.device) -> ClassicalBackbone:
    """Load or train the CIFAR-10 backbone."""
    backbone = ClassicalBackbone(TL_N_QUBITS)
    if os.path.exists(BACKBONE_CKPT):
        backbone.load_state_dict(torch.load(BACKBONE_CKPT, map_location="cpu"))
        print(f"  Loaded backbone from {BACKBONE_CKPT}")
    else:
        train_cifar10_backbone(device, epochs=10)
        backbone.load_state_dict(torch.load(BACKBONE_CKPT, map_location="cpu"))
    return backbone


# ── Transfer-learning QCQ-CNN ─────────────────────────────────────────────────

class TransferQCQCNN(nn.Module):
    """Frozen classical backbone → trainable shallow VQC → linear classifier.

    The backbone extracts discriminative features; the VQC only needs to learn
    a small quantum transformation, staying well away from the barren plateau.
    Angle encoding: θ_i = π * tanh(x_i) (applied inside the circuit).
    """

    def __init__(self, backbone: ClassicalBackbone, n_qubits: int, n_layers: int) -> None:
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.vqc = make_torch_layer(n_qubits, n_layers, cost="local")
        self.classifier = nn.Linear(n_qubits, 4)
        self._n_qubits = n_qubits

    def to(self, *args, **kwargs):
        self.backbone = self.backbone.to(*args, **kwargs)
        self.classifier = self.classifier.to(*args, **kwargs)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.backbone(x)               # frozen, on GPU
        # π * tanh squashes to (-π, π) before angle encoding
        encoded = torch.pi * torch.tanh(feat)
        q_out = self.vqc(encoded.cpu())           # VQC on CPU
        q_out = q_out.to(x.device)
        return self.classifier(q_out)


def run_tl(seed: int, device: torch.device, backbone: ClassicalBackbone) -> RunResult:
    set_seed(seed)
    # MRI loaders need grayscale → convert to 3-channel for CIFAR10 backbone
    tf_mri = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2470, 0.2435, 0.2616]),
    ])
    tf_mri_eval = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2470, 0.2435, 0.2616]),
    ])
    from torchvision.datasets import ImageFolder
    from torch.utils.data import random_split
    mri_train_dir = os.path.join(config.MRI_ROOT, "Training")
    mri_test_dir = os.path.join(config.MRI_ROOT, "Testing")
    full_train = ImageFolder(mri_train_dir, transform=tf_mri)
    n_val = int(len(full_train) * config.VAL_FRAC)
    n_train = len(full_train) - n_val
    gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [n_train, n_val], generator=gen)
    test_set = ImageFolder(mri_test_dir, transform=tf_mri_eval)
    train_loader = DataLoader(train_set, config.BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, config.BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, config.BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True)

    model = TransferQCQCNN(backbone, TL_N_QUBITS, TL_N_LAYERS)
    model.to(device)
    print(f"\n  [transfer seed={seed}] training...")
    result = train_model(
        model, train_loader, val_loader, test_loader,
        optimizer_name="transfer", seed=seed,
        epochs=EPOCHS, lr=TL_LR, device=device, verbose=True,
    )
    return result


def plot_tl_curves(tl_runs: list[RunResult]) -> None:
    os.makedirs(FIGURES, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, title in [
        (axes[0], "val_acc", "Validation accuracy"),
        (axes[1], "train_loss", "Training loss"),
    ]:
        epochs_arr = [h["epoch"] for h in tl_runs[0].history]
        vals = np.array([[h[metric] for h in r.history] for r in tl_runs])
        ax.plot(epochs_arr, vals.mean(0), "g-", lw=2, label="Transfer (mean ± std)")
        ax.fill_between(epochs_arr,
                        vals.mean(0) - vals.std(0),
                        vals.mean(0) + vals.std(0), alpha=0.2, color="green")
        ax.axhline(0.25, color="gray", ls=":", lw=0.8, label="random baseline")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Mitigation 3: Quantum transfer learning on Brain MRI", y=1.02)
    fig.tight_layout()
    path = os.path.join(FIGURES, "transfer_learning_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def print_summary(tl_runs: list[RunResult]) -> None:
    accs = np.array([r.test_acc for r in tl_runs])
    gvs = np.array([r.grad_var_epoch1 for r in tl_runs
                    if np.isfinite(r.grad_var_epoch1)])
    print("\n" + "=" * 60)
    print("Mitigation 3 — Quantum transfer learning summary")
    print("=" * 60)
    print(f"  test acc  : {accs.mean():.4f} ± {accs.std():.4f}")
    if len(gvs):
        print(f"  grad var  : {gvs.mean():.4e}  (epoch 1, VQC weights)")
    print(f"  n_qubits={TL_N_QUBITS}  n_layers={TL_N_LAYERS}  "
          f"(shallower than baseline {config.N_LAYERS} layers)")


def save_results_csv(tl_runs: list[RunResult]) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "transfer_learning_summary.csv")
    with open(path, "w") as fh:
        fh.write("variant,seed,test_acc,grad_var_epoch1\n")
        for r in tl_runs:
            fh.write(f"transfer,{r.seed},{r.test_acc:.6f},"
                     f"{r.grad_var_epoch1:.6e}\n")
    print(f"Saved CSV: {path}")


def main() -> None:
    device = get_device()
    print("=" * 60)
    print("Mitigation 3 — Quantum transfer learning")
    print(f"Device: {device}  |  n_qubits={TL_N_QUBITS}  n_layers={TL_N_LAYERS}")
    print(f"Epochs: {EPOCHS}  |  LR: {TL_LR}  |  Seeds: {SEEDS}")
    print("=" * 60)

    backbone = get_backbone(device)

    t0 = time.time()
    tl_runs: list[RunResult] = []
    for seed in SEEDS:
        tl_runs.append(run_tl(seed, device, backbone))

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")
    plot_tl_curves(tl_runs)
    print_summary(tl_runs)
    save_results_csv(tl_runs)


if __name__ == "__main__":
    main()
