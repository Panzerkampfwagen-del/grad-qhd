"""
QCQ-CNN model: classical CNN encoder → VQC → classical output layer.

Architecture:
  CNN encoder : Conv(1→8)→BN→ReLU→Pool → Conv(8→16)→BN→ReLU→Pool
                → Conv(16→32)→BN→ReLU→Pool → GAP → Linear(32, n_qubits)
  VQC         : TorchLayer (global or local measurement, always on CPU)
  Output      : Linear(vqc_out_dim, n_classes)

Device strategy: default.qubit is CPU-only, so the VQC always runs on CPU.
The CNN encoder and output classifier are moved to the target device normally.
Gradient flows through the CPU<->device transfers via PyTorch autograd.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from qmlcore.circuit import make_torch_layer


class CNNEncoder(nn.Module):
    """Small grayscale CNN that outputs an n_qubits-dim feature vector."""

    def __init__(self, n_qubits: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(32, n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        return self.proj(h)


class QCQCNN(nn.Module):
    """QCQ-CNN: classical encoder + VQC + classical classifier.

    The VQC (TorchLayer) is intentionally kept on CPU because PennyLane's
    default.qubit is a CPU simulator.  The encoder and classifier are moved
    to whatever device is requested via .to().  Gradient flows across the
    CPU<->device boundary through PyTorch autograd.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 4,
        n_classes: int = 4,
        cost: str = "global",
    ) -> None:
        super().__init__()
        self.encoder = CNNEncoder(n_qubits)
        self.vqc = make_torch_layer(n_qubits, n_layers, cost=cost)
        vqc_out = 1 if cost == "global" else n_qubits
        self.classifier = nn.Linear(vqc_out, n_classes)

    def to(self, *args, **kwargs):
        # Move encoder and classifier to the target device/dtype.
        # Deliberately skip self.vqc so its weights stay on CPU.
        self.encoder = self.encoder.to(*args, **kwargs)
        self.classifier = self.classifier.to(*args, **kwargs)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)           # on main device (GPU if available)
        q_out = self.vqc(feat.cpu())     # VQC always on CPU
        q_out = q_out.to(x.device)      # back to main device
        if q_out.dim() == 1:
            q_out = q_out.unsqueeze(1)
        return self.classifier(q_out)
