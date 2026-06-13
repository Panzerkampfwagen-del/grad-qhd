"""
Data loaders for the GradHD real-data experiments.

  - MNIST and CIFAR-10 are fetched through torchvision (download=True).
  - The brain MRI dataset is NOT downloaded automatically.  Place the Kaggle
    "Brain Tumor MRI Dataset" under config.MRI_ROOT before calling
    get_mri_loaders; if it is missing, a clear instruction message is printed
    and FileNotFoundError is raised.

All loaders return (train_loader, val_loader, test_loader).  The validation
split is carved out of the training set with a seeded generator so the split
is reproducible across optimizers and seeds.
"""
from __future__ import annotations
import os
import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from config import DATA_ROOT, MRI_ROOT, MRI_CLASSES, DATASET_CONFIGS, VAL_FRAC


# ---- reproducibility -------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed random, numpy, and torch (CPU + CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return the CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _split_train_val(
    train_set: Dataset,
    val_frac: float,
    seed: int,
) -> Tuple[Dataset, Dataset]:
    """Split a dataset into (train, val) with a seeded generator."""
    n_total = len(train_set)
    n_val = int(round(val_frac * n_total))
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(seed)
    return random_split(train_set, [n_train, n_val], generator=gen)


def _make_loaders(
    train_set: Dataset,
    val_set: Dataset,
    test_set: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Wrap three datasets in DataLoaders with shared settings."""
    common = dict(num_workers=num_workers, pin_memory=pin_memory)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=True, **common)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


# ---- MNIST -----------------------------------------------------------------

def get_mnist_loaders(
    batch_size: int | None = None,
    data_root: str = DATA_ROOT,
    val_frac: float = VAL_FRAC,
    seed: int = 0,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """MNIST loaders (auto-download).  60k train -> train/val, 10k test."""
    bs = batch_size or DATASET_CONFIGS["mnist"]["batch_size"]
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    full_train = datasets.MNIST(data_root, train=True, download=True, transform=tf)
    test_set = datasets.MNIST(data_root, train=False, download=True, transform=tf)
    train_set, val_set = _split_train_val(full_train, val_frac, seed)
    return _make_loaders(train_set, val_set, test_set, bs, num_workers, pin_memory)


# ---- CIFAR-10 --------------------------------------------------------------

def get_cifar10_loaders(
    batch_size: int | None = None,
    data_root: str = DATA_ROOT,
    val_frac: float = VAL_FRAC,
    seed: int = 0,
    num_workers: int = 2,
    pin_memory: bool = True,
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """CIFAR-10 loaders (auto-download).  50k train -> train/val, 10k test.

    Note: the train augmentation is applied to the whole training set before
    the val split, so the held-out val images are lightly augmented too.  This
    is acceptable here because all optimizers see the identical pipeline; the
    val set is only used for relative comparison, not as a clean benchmark.
    """
    bs = batch_size or DATASET_CONFIGS["cifar10"]["batch_size"]
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    norm = transforms.Normalize(mean, std)
    eval_tf = transforms.Compose([transforms.ToTensor(), norm])
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ])
    else:
        train_tf = eval_tf

    full_train = datasets.CIFAR10(data_root, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(data_root, train=False, download=True, transform=eval_tf)
    train_set, val_set = _split_train_val(full_train, val_frac, seed)
    return _make_loaders(train_set, val_set, test_set, bs, num_workers, pin_memory)


# ---- Brain MRI (Kaggle, no auto-download) ----------------------------------

_MRI_INSTRUCTIONS = """
Brain MRI dataset not found at: {root}

This dataset is NOT downloaded automatically.  To set it up:

  1. Download the Kaggle "Brain Tumor MRI Dataset":
       https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
     (e.g. with the Kaggle CLI:
        kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset)

  2. Unzip it so the folder layout under {root} is:
       {root}/Training/glioma/*.jpg
       {root}/Training/meningioma/*.jpg
       {root}/Training/notumor/*.jpg
       {root}/Training/pituitary/*.jpg
       {root}/Testing/glioma/*.jpg   (and the other three classes)

  3. Re-run.  Override the location with the GRAD_QHD_DATA environment
     variable (MRI is expected at $GRAD_QHD_DATA/brain_mri).
""".strip()


def _check_mri_layout(root: str) -> Tuple[str, str]:
    """Validate the Kaggle folder layout; return (training_dir, testing_dir)."""
    train_dir = os.path.join(root, "Training")
    test_dir = os.path.join(root, "Testing")
    if not (os.path.isdir(train_dir) and os.path.isdir(test_dir)):
        print(_MRI_INSTRUCTIONS.format(root=root))
        raise FileNotFoundError(
            f"Expected '{train_dir}' and '{test_dir}'. See instructions above."
        )
    for split_dir in (train_dir, test_dir):
        missing = [c for c in MRI_CLASSES
                   if not os.path.isdir(os.path.join(split_dir, c))]
        if missing:
            print(_MRI_INSTRUCTIONS.format(root=root))
            raise FileNotFoundError(
                f"Missing class folders {missing} under '{split_dir}'."
            )
    return train_dir, test_dir


def get_mri_loaders(
    batch_size: int | None = None,
    mri_root: str = MRI_ROOT,
    img_size: int | None = None,
    val_frac: float = VAL_FRAC,
    seed: int = 0,
    num_workers: int = 2,
    pin_memory: bool = True,
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Brain MRI loaders from the Kaggle ImageFolder layout (no auto-download).

    Images are converted to single-channel grayscale and resized to img_size.
    The Training/ folder is split into train/val; the Testing/ folder is the
    held-out test set.  Class order follows MRI_CLASSES (alphabetical, matching
    torchvision.ImageFolder): glioma, meningioma, notumor, pituitary.
    """
    cfg = DATASET_CONFIGS["mri"]
    bs = batch_size or cfg["batch_size"]
    size = img_size or cfg["img_size"]

    train_dir, test_dir = _check_mri_layout(mri_root)

    norm = transforms.Normalize((0.5,), (0.5,))
    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        norm,
    ])
    if augment:
        train_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((size, size)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ])
    else:
        train_tf = eval_tf

    full_train = datasets.ImageFolder(train_dir, transform=train_tf)
    test_set = datasets.ImageFolder(test_dir, transform=eval_tf)

    # Sanity check: ImageFolder's class order must match MRI_CLASSES.
    if full_train.classes != MRI_CLASSES:
        raise RuntimeError(
            f"MRI class order {full_train.classes} != expected {MRI_CLASSES}."
        )

    train_set, val_set = _split_train_val(full_train, val_frac, seed)
    return _make_loaders(train_set, val_set, test_set, bs, num_workers, pin_memory)


# ---- dispatcher ------------------------------------------------------------

def get_loaders(
    dataset: str,
    batch_size: int | None = None,
    seed: int = 0,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train, val, test) loaders for 'mnist', 'cifar10', or 'mri'."""
    if dataset == "mnist":
        return get_mnist_loaders(batch_size, seed=seed,
                                 num_workers=num_workers, pin_memory=pin_memory)
    if dataset == "cifar10":
        return get_cifar10_loaders(batch_size, seed=seed,
                                   num_workers=num_workers, pin_memory=pin_memory)
    if dataset == "mri":
        return get_mri_loaders(batch_size, seed=seed,
                               num_workers=num_workers, pin_memory=pin_memory)
    raise ValueError(f"Unknown dataset '{dataset}'. Use mnist|cifar10|mri.")


if __name__ == "__main__":
    # Smoke check for MNIST only (auto-downloads; MRI requires manual setup).
    set_seed(0)
    tr, va, te = get_mnist_loaders(num_workers=0, pin_memory=False)
    xb, yb = next(iter(tr))
    print(f"mnist  train_batches={len(tr)}  val={len(va.dataset)}  "
          f"test={len(te.dataset)}  x={tuple(xb.shape)}  y={tuple(yb.shape)}")
