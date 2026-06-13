"""
Training engine for the GradHD real-data experiments.

Provides the building blocks used by run_experiments.py and run_ablation.py:

  - print_env / assert_vram_fits : startup environment + VRAM checks
  - build_optimizer             : factory for adam / adamw / sgd / gradhd
  - train_model                 : train one (dataset, optimizer, seed) run
  - write_epochs_csv / write_summary_csv : CSV logging

GradHD's no-HVP step needs no closure, so training is a plain
forward / backward / step loop and GradHD is a drop-in optimizer here.
"""
from __future__ import annotations
import os
import csv
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import RESULTS_ROOT
from partB_gradhd.gradhd_optim import GradHD
from partB_gradhd.data import set_seed, get_device, get_loaders
from partB_gradhd.models import build_model, count_params


# ---- environment -----------------------------------------------------------

def print_env() -> torch.device:
    """Print torch/torchvision/CUDA info and return the active device."""
    import torchvision
    print(f"torch {torch.__version__} | torchvision {torchvision.__version__}")
    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        name = torch.cuda.get_device_name(i)
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"CUDA available | device: {name} ({total:.1f} GB)")
    else:
        print("CUDA not available | running on CPU")
    return get_device()


def assert_vram_fits(min_gb: float) -> None:
    """Raise if a CUDA device has less than min_gb total memory (no-op on CPU)."""
    if not torch.cuda.is_available():
        return
    i = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(i).total_memory / 1e9
    if total + 1e-6 < min_gb:
        raise RuntimeError(f"GPU has {total:.1f} GB < required {min_gb:.1f} GB")


# ---- optimizer factory -----------------------------------------------------

def build_optimizer(
    kind: str,
    params,
    lr: float,
    weight_decay: float = 0.0,
    **kw,
) -> torch.optim.Optimizer:
    """Return an optimizer by name.

    kind in {adam, adamw, sgd, gradhd}.  Extra keys:
      sgd:    momentum (default 0.9), nesterov (default False)
      gradhd: alpha, beta, gamma (default 0.0 -> exactly Adam)
    """
    kind = kind.lower()
    if kind == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if kind == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if kind == "sgd":
        return torch.optim.SGD(
            params, lr=lr,
            momentum=kw.get("momentum", 0.9),
            nesterov=kw.get("nesterov", False),
            weight_decay=weight_decay,
        )
    if kind == "gradhd":
        return GradHD(
            params, lr=lr,
            alpha=kw.get("alpha", 0.0),
            beta=kw.get("beta", 0.0),
            gamma=kw.get("gamma", 0.0),
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unknown optimizer kind '{kind}'.")


# ---- train / eval loops ----------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> tuple[float, float]:
    """Return (mean loss, accuracy) over a loader."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for bi, (x, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        correct += int((logits.argmax(1) == y).sum().item())
        n += y.size(0)
    return total_loss / max(n, 1), correct / max(n, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> tuple[float, float]:
    """Train for one epoch; return (mean loss, accuracy)."""
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for bi, (x, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += int((logits.argmax(1) == y).sum().item())
        n += y.size(0)
    return total_loss / max(n, 1), correct / max(n, 1)


@dataclass
class RunResult:
    """Per-run record: config, per-epoch curves, and final test metrics."""
    dataset: str
    optimizer: str
    seed: int
    lr: float
    n_params: int
    epochs: list  # list of per-epoch dicts
    test_loss: float
    test_acc: float

    def mean_epoch_time(self) -> float:
        ts = [e["epoch_time_s"] for e in self.epochs]
        return sum(ts) / len(ts) if ts else 0.0


def train_model(
    dataset: str,
    optimizer_name: str,
    opt_cfg: dict,
    seed: int,
    epochs: int,
    device: torch.device,
    max_batches: Optional[int] = None,
    num_workers: int = 2,
    pin_memory: bool = True,
    verbose: bool = True,
) -> RunResult:
    """Train one (dataset, optimizer, seed) run and return a RunResult.

    opt_cfg is an EXPERIMENT_OPTIMIZERS-style dict with a "kind" and "lr" key
    plus optimizer-specific extras (momentum / alpha / beta / gamma).
    """
    set_seed(seed)
    train_loader, val_loader, test_loader = get_loaders(
        dataset, seed=seed, num_workers=num_workers, pin_memory=pin_memory
    )
    model = build_model(dataset).to(device)
    n_params = count_params(model)
    criterion = nn.CrossEntropyLoss()

    lr = opt_cfg["lr"]
    extra = {k: v for k, v in opt_cfg.items() if k not in ("kind", "lr")}
    optimizer = build_optimizer(opt_cfg["kind"], model.parameters(), lr, **extra)

    use_cuda = device.type == "cuda"
    hist: list[dict] = []
    for ep in range(1, epochs + 1):
        # Time the train loop only (not loader setup or eval).  Sync CUDA so the
        # wall-clock reflects completed kernels, not just launches.
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, max_batches
        )
        if use_cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        va_loss, va_acc = evaluate(model, val_loader, criterion, device, max_batches)
        hist.append(dict(
            epoch=ep, train_loss=tr_loss, train_acc=tr_acc,
            val_loss=va_loss, val_acc=va_acc, epoch_time_s=dt,
        ))
        if verbose:
            print(f"  [{optimizer_name} seed{seed}] ep {ep}/{epochs}  "
                  f"train {tr_loss:.4f}/{tr_acc:.3f}  "
                  f"val {va_loss:.4f}/{va_acc:.3f}  {dt:.1f}s")

    te_loss, te_acc = evaluate(model, test_loader, criterion, device, max_batches)
    if verbose:
        print(f"  [{optimizer_name} seed{seed}] TEST  {te_loss:.4f}/{te_acc:.3f}")

    return RunResult(dataset, optimizer_name, seed, lr, n_params,
                     hist, te_loss, te_acc)


# ---- CSV logging -----------------------------------------------------------

def write_epochs_csv(path: str, results: list[RunResult]) -> None:
    """Write per-epoch curves (one row per epoch per run)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = ["dataset", "optimizer", "seed", "lr", "epoch",
              "train_loss", "train_acc", "val_loss", "val_acc", "epoch_time_s"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            for e in r.epochs:
                w.writerow(dict(
                    dataset=r.dataset, optimizer=r.optimizer, seed=r.seed,
                    lr=r.lr, epoch=e["epoch"],
                    train_loss=e["train_loss"], train_acc=e["train_acc"],
                    val_loss=e["val_loss"], val_acc=e["val_acc"],
                    epoch_time_s=e["epoch_time_s"],
                ))


def write_summary_csv(path: str, results: list[RunResult]) -> None:
    """Write one row per run with final test metrics and mean epoch time."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = ["dataset", "optimizer", "seed", "lr", "n_params",
              "test_loss", "test_acc", "mean_epoch_time_s"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(dict(
                dataset=r.dataset, optimizer=r.optimizer, seed=r.seed,
                lr=r.lr, n_params=r.n_params,
                test_loss=r.test_loss, test_acc=r.test_acc,
                mean_epoch_time_s=r.mean_epoch_time(),
            ))


def default_results_path(dataset: str, suffix: str) -> str:
    """Standard CSV path: results/<dataset>_<suffix>.csv."""
    return os.path.join(RESULTS_ROOT, f"{dataset}_{suffix}.csv")
