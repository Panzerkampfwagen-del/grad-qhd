"""Shared training utilities for the QML mitigation project."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class RunResult:
    optimizer_name: str
    seed: int
    history: list[dict] = field(default_factory=list)
    test_acc: float = 0.0
    test_loss: float = 0.0
    grad_var_epoch1: float = 0.0  # VQC gradient variance at epoch 1


def _vqc_grad_var(model: nn.Module) -> float:
    """Variance of VQC weight gradients (after first backward pass)."""
    grads = []
    for name, p in model.named_parameters():
        if "vqc" in name and p.grad is not None:
            grads.append(p.grad.detach().cpu().numpy().ravel())
    if not grads:
        return float("nan")
    return float(np.var(np.concatenate(grads)))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    record_grad_var: bool = False,
) -> tuple[float, float, float]:
    """Return (mean_loss, accuracy, vqc_grad_var)."""
    model.train()
    total_loss = correct = n = 0
    grad_var = float("nan")
    first_batch = True

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        if first_batch and record_grad_var:
            grad_var = _vqc_grad_var(model)
            first_batch = False

        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)

    return total_loss / n, correct / n, grad_var


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = correct = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    optimizer_name: str,
    seed: int,
    epochs: int,
    lr: float,
    device: torch.device,
    verbose: bool = True,
) -> RunResult:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    result = RunResult(optimizer_name=optimizer_name, seed=seed)

    for ep in range(1, epochs + 1):
        record_gv = ep == 1
        t0 = time.perf_counter()
        tr_loss, tr_acc, gv = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            record_grad_var=record_gv,
        )
        dt = time.perf_counter() - t0
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        if ep == 1:
            result.grad_var_epoch1 = gv

        result.history.append(dict(
            epoch=ep, train_loss=tr_loss, train_acc=tr_acc,
            val_loss=va_loss, val_acc=va_acc, epoch_time_s=dt,
        ))
        if verbose:
            print(f"  ep {ep:2d}/{epochs}  "
                  f"tr {tr_loss:.4f}/{tr_acc:.3f}  "
                  f"val {va_loss:.4f}/{va_acc:.3f}  {dt:.1f}s")

    te_loss, te_acc = evaluate(model, test_loader, criterion, device)
    result.test_acc = te_acc
    result.test_loss = te_loss
    return result
