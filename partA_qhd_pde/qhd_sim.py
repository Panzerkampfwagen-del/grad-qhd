"""
QHD time-stepping loop for 2D landscapes.

This is the *real* quantum simulation: psi evolves as a wavefunction via the
Schrodinger PDE with the gradient-based Hamiltonian. Observables are computed
as expectations under |psi|^2. Feasible only in 2D due to the N^d grid.

Standard QHD is the special case alpha=beta=gamma=0 (no gradient correction).
"""
from __future__ import annotations
import numpy as np
from typing import Callable, NamedTuple, Optional, Tuple

from .operators import (
    apply_H1, apply_H2, apply_H3, dealias,
    k2_grid, k_grids,
    precompute_landscape, gaussian_wavepacket,
)
from .test_functions import make_grid


class SimResult(NamedTuple):
    """Observables at each of K+1 checkpoints (step 0 through K)."""
    steps: np.ndarray        # int[K+1]
    ef: np.ndarray           # float[K+1], E[f(X)] under |psi|^2
    eg2: np.ndarray          # float[K+1], E[||grad f(X)||^2]
    prob: np.ndarray         # float[K+1], P[f(X) - f* <= delta]
    norm_history: np.ndarray # float[K+1], ||psi|| (should stay ~1)
    snapshots: list          # [(step_idx, density_2d)]


def simulate(
    f_fn: Callable,
    grad_fn: Callable,
    box: Tuple[float, float, float, float],
    *,
    N: int = 128,
    t0: float = 1.0,
    h: float = 0.01,
    K: int = 200,
    alpha: float = -0.05,
    beta: float = 0.0,
    gamma: float = 5.0,
    f_star: float = 0.0,
    delta: float = 1.0,
    x0: float = 0.0,
    y0: float = 0.0,
    sigma: float = 1.0,
    snapshot_steps: Optional[list] = None,
    renormalize: bool = True,
    verbose: bool = True,
) -> SimResult:
    """
    Run gradient-based QHD for K steps of size h starting at time t0.

    Uses Strang splitting (2nd-order):
      H1(h/2) · H2(h/2) · H3(h) · H2(h/2) · H1(h/2)
    repeated K times. With alpha=beta=gamma=0 this reduces to standard QHD.

    All observables are expectations under |psi|^2 (which sums to 1):
      E[f]         = sum_{ij} f_{ij} |psi_{ij}|^2
      E[||grad||^2] = sum_{ij} ||grad f_{ij}||^2 |psi_{ij}|^2
      P[near]      = sum_{ij: f_{ij}-f*<=delta} |psi_{ij}|^2

    Parameters
    ----------
    f_fn, grad_fn : callables accepting (X, Y) arrays
    box : (xmin, xmax, ymin, ymax)
    N : grid resolution per dimension
    t0, h : initial time and step size
    K : number of time steps
    alpha, beta, gamma : QHD hyperparameters
    f_star : global minimum of f (for success probability)
    delta : success threshold
    x0, y0, sigma : initial Gaussian center and width
    snapshot_steps : list of step indices at which to save |psi|^2
    renormalize : re-normalize psi to unit norm after each full Strang step.
        H1 and H3 are exactly unitary.  H2 uses Crank-Nicolson, which is
        exactly norm-preserving only when psi is well-resolved on the grid.
        After many H3 steps, psi accumulates near-Nyquist Fourier modes; the
        2/3-rule dealiasing filter (applied inside the loop, after H3) prevents
        aliased spectral derivatives from breaking H2 Hermiticity.  norm_history
        always records the pre-renormalization norm to expose any residual drift.
    """
    snap_set: set = set(snapshot_steps) if snapshot_steps else set()

    X, Y = make_grid(box, N=N)
    dx = (box[1] - box[0]) / N
    dy = (box[3] - box[2]) / N

    K2 = k2_grid(N, dx, dy)
    KX, KY = k_grids(N, dx, dy)
    f_grid, gfx, gfy, gradf_sq, lapf = precompute_landscape(
        X, Y, f_fn, grad_fn, dx, dy
    )

    psi = gaussian_wavepacket(X, Y, x0=x0, y0=y0, sigma=sigma)
    near_mask = f_grid - f_star <= delta

    steps_arr = np.arange(K + 1)
    ef_arr = np.zeros(K + 1)
    eg2_arr = np.zeros(K + 1)
    prob_arr = np.zeros(K + 1)
    norm_arr = np.zeros(K + 1)
    snapshots: list = []

    def _record(k: int) -> None:
        density = np.abs(psi) ** 2
        total = float(np.sum(density))
        norm_arr[k] = float(np.sqrt(total)) if total > 0 else 0.0
        if not (total > 0):
            ef_arr[k] = eg2_arr[k] = prob_arr[k] = np.nan
            return
        # Always use normalized density so observables are proper expectations.
        norm_density = density / total
        ef_arr[k] = float(np.sum(norm_density * f_grid))
        eg2_arr[k] = float(np.sum(norm_density * gradf_sq))
        prob_arr[k] = float(np.sum(norm_density[near_mask]))
        if k in snap_set:
            snapshots.append((k, norm_density.copy()))

    _record(0)

    t = t0
    for k in range(1, K + 1):
        psi = apply_H1(psi, K2, t=t, h=h / 2)
        psi = apply_H2(psi, KX, KY, gfx, gfy, lapf, h=h / 2, alpha=alpha)
        psi = apply_H3(psi, f_grid, gradf_sq, t=t, h=h,
                       alpha=alpha, beta=beta, gamma=gamma)
        # Dealias after H3: exp(-ihV)*psi folds energy into near-Nyquist modes;
        # zeroing the top third of Fourier modes keeps spectral gradients in H2
        # aliasing-free so the CN operator remains norm-preserving.
        if alpha != 0.0:
            psi = dealias(psi)
        psi = apply_H2(psi, KX, KY, gfx, gfy, lapf, h=h / 2, alpha=alpha)
        psi = apply_H1(psi, K2, t=t, h=h / 2)
        t += h
        # Record norm before any renormalization so drift is visible.
        _record(k)
        if renormalize and norm_arr[k] > 0:
            psi /= norm_arr[k]  # norm_arr[k] already computed in _record
        if verbose and k % 20 == 0:
            print(f"  step {k:4d}/{K}  E[f]={ef_arr[k]:.4f}  "
                  f"norm(pre-renorm)={norm_arr[k]:.6f}", flush=True)

    return SimResult(
        steps=steps_arr,
        ef=ef_arr,
        eg2=eg2_arr,
        prob=prob_arr,
        norm_history=norm_arr,
        snapshots=snapshots,
    )
