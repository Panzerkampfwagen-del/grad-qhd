"""
Small models for the GradHD real-data experiments.

All models are intentionally small enough to train on a 6GB GPU (RTX 3050
Laptop, sm_86).  The point of Part B is to compare optimizers head-to-head,
not to chase state-of-the-art accuracy, so the architectures are deliberately
plain.

  - MLP:       flat baseline for MNIST.
  - SmallCNN:  configurable conv net used for MNIST, CIFAR-10, and MRI.

`build_model(dataset)` returns the recommended model for each dataset using
the shapes in config.DATASET_CONFIGS.
"""
from __future__ import annotations
from typing import Sequence

import torch
import torch.nn as nn

from config import DATASET_CONFIGS


class MLP(nn.Module):
    """Two-hidden-layer MLP for flattened images (MNIST baseline)."""

    def __init__(
        self,
        in_dim: int = 28 * 28,
        hidden: Sequence[int] = (256, 128),
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))


class SmallCNN(nn.Module):
    """Configurable conv net: N conv blocks + global average pool + linear head.

    Each block is Conv(3x3, pad 1) -> BatchNorm -> ReLU -> MaxPool(2).
    A final AdaptiveAvgPool collapses the spatial dims, so the head size is
    independent of the input resolution (28, 32, or 64 all work unchanged).
    Parameter count stays well under 2M for the default channel widths.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        channels: Sequence[int] = (32, 64, 128),
        head_dim: int = 256,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        prev = in_channels
        for c in channels:
            blocks += [
                nn.Conv2d(prev, c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            prev = c
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev, head_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)


def count_params(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(dataset: str) -> nn.Module:
    """Return the recommended model for a dataset name.

    mnist   -> SmallCNN (1 channel, 2 blocks, LeNet-scale)
    cifar10 -> SmallCNN (3 channels, 3 blocks)
    mri     -> SmallCNN (1 channel, 4 blocks, for 64x64 input)
    """
    if dataset not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Choose from {list(DATASET_CONFIGS)}."
        )
    cfg = DATASET_CONFIGS[dataset]
    in_ch, n_cls = cfg["in_channels"], cfg["num_classes"]

    if dataset == "mnist":
        return SmallCNN(in_ch, n_cls, channels=(32, 64))
    if dataset == "cifar10":
        return SmallCNN(in_ch, n_cls, channels=(32, 64, 128))
    # mri: 64x64 input, one extra block to shrink the spatial map further.
    return SmallCNN(in_ch, n_cls, channels=(32, 64, 128, 128))


if __name__ == "__main__":
    # Quick shape + param-count sanity check (no data needed).
    shapes = {
        "mnist":   (1, 1, 28, 28),
        "cifar10": (1, 3, 32, 32),
        "mri":     (1, 1, 64, 64),
    }
    for name, shp in shapes.items():
        model = build_model(name)
        out = model(torch.zeros(*shp))
        n_cls = DATASET_CONFIGS[name]["num_classes"]
        assert out.shape == (1, n_cls), out.shape
        print(f"{name:8s}  params={count_params(model):>9,d}  out={tuple(out.shape)}")
