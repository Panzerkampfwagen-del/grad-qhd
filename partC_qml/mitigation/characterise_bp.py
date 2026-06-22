"""
Part 0 — Formal barren plateau characterisation.

For each (n_qubits, depth) pair:
  - Initialise 200 random parameter sets.
  - For each, compute ∂L/∂θ_j for one randomly chosen parameter j
    using the parameter-shift rule directly (2 circuit evaluations).
  - Record Var[∂L/∂θ_j] across the 200 samples.

The cost function is the expectation of the global n-qubit observable
Z⊗Z⊗...⊗Z (the McClean et al. 2018 setting that provably exhibits
exponential gradient vanishing).

Outputs:
  figures/bp_variance_vs_n.png     log(Var) vs n for each fixed depth
  figures/bp_variance_vs_depth.png log(Var) vs depth for each fixed n
"""
from __future__ import annotations
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ── grid ────────────────────────────────────────────────────────────────────
DEPTHS = [1, 2, 3, 4, 6]
N_QUBITS_LIST = [2, 4, 6, 8]
N_SAMPLES = 200
SEED = 42

FIGURES = os.path.join(os.path.dirname(__file__), "..", "figures")


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def global_obs(n: int) -> qml.operation.Operator:
    if n == 1:
        return qml.PauliZ(0)
    return qml.prod(*[qml.PauliZ(i) for i in range(n)])


def build_circuit(n_qubits: int, n_layers: int) -> qml.QNode:
    """Hardware-efficient ansatz with global Z⊗...⊗Z observable."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(params: pnp.ndarray) -> float:
        # params shape: (n_layers * n_qubits * 2,) — flat
        idx = 0
        for l in range(n_layers):
            for i in range(n_qubits):
                qml.RY(params[idx], wires=i)
                qml.RZ(params[idx + 1], wires=i)
                idx += 2
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
        return qml.expval(global_obs(n_qubits))

    return circuit


def gradient_variance(n_qubits: int, n_layers: int,
                      n_samples: int = N_SAMPLES,
                      seed: int = SEED) -> float:
    """Estimate Var[∂L/∂θ_j] via parameter-shift on a random parameter."""
    rng = np.random.default_rng(seed)
    circuit = build_circuit(n_qubits, n_layers)
    n_params = n_layers * n_qubits * 2
    grads = np.empty(n_samples)

    for s in range(n_samples):
        theta = rng.uniform(-np.pi, np.pi, n_params)
        j = rng.integers(0, n_params)
        # Direct parameter-shift: ∂L/∂θ_j = (L(θ+π/2 e_j) - L(θ-π/2 e_j)) / 2
        tp = theta.copy(); tp[j] += np.pi / 2
        tm = theta.copy(); tm[j] -= np.pi / 2
        # Use plain numpy arrays (not pnp) — no autograd needed here
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tp_pnp = pnp.array(tp, requires_grad=False)
            tm_pnp = pnp.array(tm, requires_grad=False)
            lp = float(circuit(tp_pnp))
            lm = float(circuit(tm_pnp))
        grads[s] = (lp - lm) / 2.0

    return float(np.var(grads))


def fit_log_linear(xs: list[int], ys: list[float]) -> tuple[float, float, float]:
    """Fit log(Var) = a*x + b; return (slope, intercept, r²)."""
    log_y = np.log(np.array(ys) + 1e-30)
    slope, intercept, r, _, _ = stats.linregress(xs, log_y)
    return slope, intercept, r ** 2


def run_all() -> dict:
    """Compute gradient variance for all (n, d) pairs; return nested dict."""
    set_seed(SEED)
    results: dict = {}  # results[n][d] = variance
    for n in N_QUBITS_LIST:
        results[n] = {}
        for d in DEPTHS:
            t0 = time.time()
            var = gradient_variance(n, d)
            dt = time.time() - t0
            print(f"  n={n:2d}  d={d}  Var={var:.4e}  ({dt:.1f}s)")
            results[n][d] = var
    return results


def plot_vs_n(results: dict) -> None:
    """log(Var) vs n_qubits, one curve per depth."""
    os.makedirs(FIGURES, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(DEPTHS)))
    for col, d in zip(colors, DEPTHS):
        ys = [results[n][d] for n in N_QUBITS_LIST]
        slope, _, r2 = fit_log_linear(N_QUBITS_LIST, ys)
        ax.semilogy(N_QUBITS_LIST, ys, "o-", color=col,
                    label=f"d={d}  slope={slope:.2f}/qubit  R²={r2:.2f}")
    ax.set_xlabel("Number of qubits n")
    ax.set_ylabel("Var[∂L/∂θ]  (log scale)")
    ax.set_title("Barren plateau: gradient variance vs qubit count\n"
                 "(global observable Z⊗...⊗Z, HEA ansatz, 200 random inits)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, which="both")
    ax.set_xticks(N_QUBITS_LIST)
    fig.tight_layout()
    path = os.path.join(FIGURES, "bp_variance_vs_n.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_vs_depth(results: dict) -> None:
    """log(Var) vs depth, one curve per n_qubits."""
    os.makedirs(FIGURES, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.plasma(np.linspace(0.15, 0.9, len(N_QUBITS_LIST)))
    for col, n in zip(colors, N_QUBITS_LIST):
        ys = [results[n][d] for d in DEPTHS]
        slope, _, r2 = fit_log_linear(DEPTHS, ys)
        ax.semilogy(DEPTHS, ys, "s-", color=col,
                    label=f"n={n}  slope={slope:.2f}/layer  R²={r2:.2f}")
    ax.set_xlabel("Circuit depth d (number of variational layers)")
    ax.set_ylabel("Var[∂L/∂θ]  (log scale)")
    ax.set_title("Barren plateau: gradient variance vs circuit depth\n"
                 "(global observable Z⊗...⊗Z, HEA ansatz, 200 random inits)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, which="both")
    ax.set_xticks(DEPTHS)
    fig.tight_layout()
    path = os.path.join(FIGURES, "bp_variance_vs_depth.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def print_summary(results: dict) -> None:
    """Print decay rates and check exponential decay criterion."""
    print("\n=== Fitted decay rates ===")
    print(f"{'':30s}  slope     R²")

    print("\nlog(Var) vs n_qubits (for fixed depth):")
    all_neg = True
    for d in DEPTHS:
        ys = [results[n][d] for n in N_QUBITS_LIST]
        slope, _, r2 = fit_log_linear(N_QUBITS_LIST, ys)
        flag = "" if slope < -0.1 else " ← WEAK"
        if slope >= -0.1:
            all_neg = False
        print(f"  d={d}: slope={slope:+.3f} per qubit  R²={r2:.3f}{flag}")

    print("\nlog(Var) vs depth (for fixed n):")
    for n in N_QUBITS_LIST:
        ys = [results[n][d] for d in DEPTHS]
        slope, _, r2 = fit_log_linear(DEPTHS, ys)
        flag = "" if slope < -0.1 else " ← WEAK/FLAT"
        print(f"  n={n}: slope={slope:+.3f} per layer  R²={r2:.3f}{flag}")

    print()
    if all_neg:
        print("✓ Exponential gradient vanishing confirmed in the n-axis.")
    else:
        print("✗ WARNING: decay not clearly exponential in n — check setup.")
        print("  The downstream mitigations assume exponential vanishing.")


def main() -> None:
    print("=" * 60)
    print("Part 0 — Barren plateau characterisation")
    print(f"PennyLane {qml.__version__}")
    print(f"Depths: {DEPTHS}  |  Qubits: {N_QUBITS_LIST}  |  Samples: {N_SAMPLES}")
    print("=" * 60)

    t0 = time.time()
    results = run_all()
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")

    plot_vs_n(results)
    plot_vs_depth(results)
    print_summary(results)


if __name__ == "__main__":
    main()
