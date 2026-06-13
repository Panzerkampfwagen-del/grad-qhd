"""
Classical baselines for comparison with gradient-based QHD on 2D test functions.

Each method runs n_runs parallel trajectories from uniform random initial
conditions and tracks the same three observables as the QHD simulation:
  E[f], E[||grad f||^2], P[f - f* <= delta].
"""
from __future__ import annotations
import numpy as np
from typing import Callable, NamedTuple, Tuple


class BaselineResult(NamedTuple):
    steps: np.ndarray  # int[K+1]
    ef: np.ndarray     # float[K+1], mean f over n_runs trajectories
    eg2: np.ndarray    # float[K+1], mean ||grad f||^2
    prob: np.ndarray   # float[K+1], fraction with f - f* <= delta


def _random_init(
    box: Tuple[float, float, float, float],
    n_runs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    x = rng.uniform(box[0], box[1], size=(n_runs,))
    y = rng.uniform(box[2], box[3], size=(n_runs,))
    return np.column_stack([x, y])  # (n_runs, 2)


def _clip_to_box(
    pos: np.ndarray,
    box: Tuple[float, float, float, float],
) -> np.ndarray:
    pos[:, 0] = np.clip(pos[:, 0], box[0], box[1])
    pos[:, 1] = np.clip(pos[:, 1], box[2], box[3])
    return pos


def _observe(
    f_fn: Callable,
    grad_fn: Callable,
    pos: np.ndarray,
    f_star: float,
    delta: float,
) -> Tuple[float, float, float]:
    """Return (E[f], E[||grad f||^2], P[f-f*<=delta]) for a batch of positions."""
    f_vals = np.asarray(f_fn(pos[:, 0], pos[:, 1]), dtype=float)
    gx = np.asarray(grad_fn(pos[:, 0], pos[:, 1])[0], dtype=float)
    gy = np.asarray(grad_fn(pos[:, 0], pos[:, 1])[1], dtype=float)
    return (
        float(np.mean(f_vals)),
        float(np.mean(gx ** 2 + gy ** 2)),
        float(np.mean(f_vals - f_star <= delta)),
    )


def run_sgdm(
    f_fn: Callable,
    grad_fn: Callable,
    box: Tuple[float, float, float, float],
    *,
    K: int = 200,
    n_runs: int = 1000,
    s0: float = 0.01,
    mu: float = 0.9,
    f_star: float = 0.0,
    delta: float = 1.0,
    seed: int = 0,
) -> BaselineResult:
    """
    SGD with Momentum using the paper's schedule (Leng & Shi, 2025):
      eta_k = 0.5 + 0.4 * k / K    (noise std, annealed up)
      s_k   = s0 / k               (step size, decayed)
    Noisy gradient: g_k = grad_f(x_k) + eta_k * N(0, I_2).
    Heavy-ball update: v = mu*v + s_k*g;  x = x - v.
    """
    rng = np.random.default_rng(seed)
    pos = _random_init(box, n_runs, rng)
    vel = np.zeros_like(pos)

    steps_arr = np.arange(K + 1)
    ef_arr = np.zeros(K + 1)
    eg2_arr = np.zeros(K + 1)
    prob_arr = np.zeros(K + 1)

    ef_arr[0], eg2_arr[0], prob_arr[0] = _observe(f_fn, grad_fn, pos, f_star, delta)

    for k in range(1, K + 1):
        eta_k = 0.5 + 0.4 * k / K
        s_k = s0 / k
        gx, gy = grad_fn(pos[:, 0], pos[:, 1])
        noise = rng.standard_normal((n_runs, 2))
        g = np.column_stack([np.asarray(gx, float), np.asarray(gy, float)]) + eta_k * noise
        vel = mu * vel + s_k * g
        pos = pos - vel
        pos = _clip_to_box(pos, box)
        ef_arr[k], eg2_arr[k], prob_arr[k] = _observe(f_fn, grad_fn, pos, f_star, delta)

    return BaselineResult(steps=steps_arr, ef=ef_arr, eg2=eg2_arr, prob=prob_arr)


def run_nag(
    f_fn: Callable,
    grad_fn: Callable,
    box: Tuple[float, float, float, float],
    *,
    K: int = 200,
    n_runs: int = 1000,
    s: float = 0.01,
    f_star: float = 0.0,
    delta: float = 1.0,
    seed: int = 0,
) -> BaselineResult:
    """
    Nesterov Accelerated Gradient with fixed step s=0.01.
    FISTA-style update:
      x_{k+1} = y_k - s * grad_f(y_k)
      y_{k+1} = x_{k+1} + (k-1)/(k+2) * (x_{k+1} - x_k)
    """
    rng = np.random.default_rng(seed)
    pos = _random_init(box, n_runs, rng)
    y = pos.copy()

    steps_arr = np.arange(K + 1)
    ef_arr = np.zeros(K + 1)
    eg2_arr = np.zeros(K + 1)
    prob_arr = np.zeros(K + 1)

    ef_arr[0], eg2_arr[0], prob_arr[0] = _observe(f_fn, grad_fn, pos, f_star, delta)

    for k in range(1, K + 1):
        gx, gy = grad_fn(y[:, 0], y[:, 1])
        g = np.column_stack([np.asarray(gx, float), np.asarray(gy, float)])
        pos_new = _clip_to_box(y - s * g, box)
        momentum = (k - 1) / (k + 2)
        y = _clip_to_box(pos_new + momentum * (pos_new - pos), box)
        pos = pos_new
        ef_arr[k], eg2_arr[k], prob_arr[k] = _observe(f_fn, grad_fn, pos, f_star, delta)

    return BaselineResult(steps=steps_arr, ef=ef_arr, eg2=eg2_arr, prob=prob_arr)
