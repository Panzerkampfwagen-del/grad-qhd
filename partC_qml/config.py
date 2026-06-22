"""
Global configuration for the QML barren-plateau mitigation project.
"""
import os

PYTHON = os.path.expanduser("~/miniconda3/envs/grad_qhd/bin/python")

MRI_ROOT = os.path.expanduser("~/grad-qhd/data/brain_mri")
MRI_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

IMG_SIZE = 64
BATCH_SIZE = 32
VAL_FRAC = 0.1

N_QUBITS = 4
N_LAYERS = 4
EPOCHS = 20

SEEDS = [0, 1, 2]
RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "results")
FIGURES_ROOT = os.path.join(os.path.dirname(__file__), "figures")
