"""Per-function QHD parameters from Leng & Shi (ICML 2025)."""
from __future__ import annotations
import os
import numpy as np

# Paper parameters: convex uses alpha=-0.1, h=0.2; non-convex uses alpha=-0.05.
# Step sizes per function: ST/Michalewicz h=0.01, Cube-Wave h=0.02, Rastrigin h=0.005.
GRAD_QHD_CONFIGS: dict[str, dict] = {
    "convex": {
        "alpha": -0.1, "beta": 0.0, "gamma": 5.0,
        "h": 0.2, "K": 200, "t0": 1.0,
        "x0": 0.0, "y0": 0.0, "sigma": 1.0,
    },
    "styblinski_tang": {
        "alpha": -0.05, "beta": 0.0, "gamma": 5.0,
        "h": 0.01, "K": 200, "t0": 1.0,
        "x0": 0.0, "y0": 0.0, "sigma": 1.0,
    },
    "michalewicz": {
        "alpha": -0.05, "beta": 0.0, "gamma": 5.0,
        "h": 0.01, "K": 200, "t0": 1.0,
        "x0": float(np.pi / 2), "y0": float(np.pi / 2), "sigma": 0.4,
    },
    "cube_wave": {
        "alpha": -0.05, "beta": 0.0, "gamma": 5.0,
        "h": 0.02, "K": 200, "t0": 1.0,
        "x0": 0.0, "y0": 0.0, "sigma": 0.5,
    },
    "rastrigin": {
        "alpha": -0.05, "beta": 0.0, "gamma": 5.0,
        "h": 0.005, "K": 200, "t0": 1.0,
        "x0": 0.0, "y0": 0.0, "sigma": 0.7,
    },
}

BASELINE_CONFIGS: dict[str, dict] = {
    "sgdm": {"s0": 0.01, "mu": 0.9, "n_runs": 1000},
    "nag":  {"s": 0.01,  "n_runs": 1000},
}

# Default N for Part A simulations.  Reduce to 64 for quick smoke tests.
DEFAULT_N = 128


# ==========================================================================
# Part B: dataset + model defaults (GradHD real-data experiments)
# ==========================================================================
# Override the data root with the GRAD_QHD_DATA environment variable.
DATA_ROOT = os.environ.get("GRAD_QHD_DATA", "data")

# Brain Tumor MRI Dataset (Kaggle, masoudnickparvar/brain-tumor-mri-dataset).
# Expected layout under MRI_ROOT:
#   Training/{glioma,meningioma,notumor,pituitary}/*.jpg
#   Testing/{glioma,meningioma,notumor,pituitary}/*.jpg
MRI_ROOT = os.path.join(DATA_ROOT, "brain_mri")
MRI_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# Per-dataset defaults. Batch sizes chosen to fit 6GB VRAM (RTX 3050 Laptop).
DATASET_CONFIGS: dict[str, dict] = {
    "mnist":   {"num_classes": 10, "in_channels": 1, "img_size": 28, "batch_size": 128},
    "cifar10": {"num_classes": 10, "in_channels": 3, "img_size": 32, "batch_size": 128},
    "mri":     {"num_classes": 4,  "in_channels": 1, "img_size": 64, "batch_size": 64},
}

# Fraction of the training split held out for validation (MNIST/CIFAR/MRI).
VAL_FRAC = 0.1

# Optimizers compared head-to-head in run_experiments.py.
# "kind" selects the torch optimizer / GradHD; remaining keys are passed through.
# GradHD with alpha=beta=gamma=0 is exactly Adam (proven in test_optimizer.py),
# so the gradhd entry below uses a representative non-trivial correction.
EXPERIMENT_OPTIMIZERS: dict[str, dict] = {
    "adam":   {"kind": "adam",   "lr": 1e-3},
    "sgd":    {"kind": "sgd",    "lr": 1e-2, "momentum": 0.9},
    "gradhd": {"kind": "gradhd", "lr": 1e-3, "alpha": -0.05, "beta": 0.0, "gamma": 1.0},
}

# Default seeds and per-dataset epoch budgets for the head-to-head.
SEEDS = [0, 1, 2]
EPOCHS = {"mnist": 5, "cifar10": 30, "mri": 30}

# Directory for Part B CSV logs (figures go in figures/).
RESULTS_ROOT = "results"

# Ablation sweeps for run_ablation.py.  GradHD has no Hessian-vector term,
# so the paper-prompt's use_hvp ablation is replaced by alpha/beta/gamma
# sweeps.  Each sweep varies ONE coefficient and holds the others at 0; the
# value 0.0 in every sweep is GradHD reduced exactly to Adam (sanity anchor).
ABLATION_SWEEPS: dict[str, list[float]] = {
    "alpha": [-0.1, -0.05, 0.0, 0.05, 0.1],
    "beta":  [0.0, 0.05, 0.1, 0.2],
    "gamma": [0.0, 1.0, 5.0],
}
ABLATION_DATASET = "mnist"   # light + fast; matches the prompt's MNIST ablation
ABLATION_EPOCHS = 5
ABLATION_SEEDS = [0, 1, 2]
ABLATION_LR = 1e-3
