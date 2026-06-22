"""
VQC circuit definitions for the QML barren-plateau mitigation project.

All circuits use a hardware-efficient ansatz (HEA):
  - Angle embedding of input features via RY rotations
  - Variational layers: RY + RZ on each qubit, then ring CNOT entanglement
  - Measurement: global (Z⊗...⊗Z) or local (sum of Z_i)
"""
from __future__ import annotations
import pennylane as qml
from pennylane import qnn


def _hea_layers(weights, n_qubits: int, n_layers: int) -> None:
    """Hardware-efficient ansatz variational layers (in-place circuit ops)."""
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RY(weights[l, i, 0], wires=i)
            qml.RZ(weights[l, i, 1], wires=i)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])


def build_global_qnode(n_qubits: int, n_layers: int) -> qml.QNode:
    """QNode that returns expectation of the global Z⊗...⊗Z observable.

    Output shape: scalar per sample → suitable for Linear(1, n_classes).
    This is the unmitigated baseline — gradient vanishes exponentially in n_qubits.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        _hea_layers(weights, n_qubits, n_layers)
        if n_qubits == 1:
            return qml.expval(qml.PauliZ(0))
        return qml.expval(qml.prod(*[qml.PauliZ(i) for i in range(n_qubits)]))

    return circuit


def build_local_qnode(n_qubits: int, n_layers: int) -> qml.QNode:
    """QNode that returns a list of single-qubit Z expectation values.

    Output shape: (n_qubits,) per sample → suitable for Linear(n_qubits, n_classes).
    Local observables avoid the exponential gradient vanishing.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        _hea_layers(weights, n_qubits, n_layers)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def make_torch_layer(
    n_qubits: int, n_layers: int, cost: str = "global"
) -> qnn.TorchLayer:
    """Return a PennyLane TorchLayer wrapping the chosen circuit."""
    weight_shapes = {"weights": (n_layers, n_qubits, 2)}
    if cost == "global":
        qnode = build_global_qnode(n_qubits, n_layers)
    elif cost == "local":
        qnode = build_local_qnode(n_qubits, n_layers)
    else:
        raise ValueError(f"cost must be 'global' or 'local', got {cost!r}")
    return qnn.TorchLayer(qnode, weight_shapes)
