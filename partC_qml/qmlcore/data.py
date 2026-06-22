"""MRI data loading for the QML project."""
from __future__ import annotations
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_mri_loaders(
    root: str,
    img_size: int = 64,
    batch_size: int = 32,
    val_frac: float = 0.1,
    seed: int = 0,
    num_workers: int = 2,
):
    """Return (train_loader, val_loader, test_loader) for brain-MRI."""
    mean, std = [0.1858], [0.2034]

    train_tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dir = os.path.join(root, "Training")
    test_dir = os.path.join(root, "Testing")

    full_train = datasets.ImageFolder(train_dir, transform=train_tf)
    n_val = int(len(full_train) * val_frac)
    n_train = len(full_train) - n_val
    gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [n_train, n_val], generator=gen)

    test_set = datasets.ImageFolder(test_dir, transform=eval_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
