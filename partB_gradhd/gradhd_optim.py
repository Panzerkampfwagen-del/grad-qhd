"""
GradHD: Gradient Descent with Adaptive Curvature Correction.

A classical first-order optimizer that extends Adam with three additional
hyperparameters inspired by the structure of the gradient-based QHD ODE
(Leng & Shi, 2025).  The optimizer contains no quantum mechanics —
the corrections arise from discretizing a classical gradient-flow ODE.

Mathematical basis
------------------
The gradient-based QHD Hamiltonian generates three additive forces on
probability mass.  In the classical particle limit each force becomes
a deterministic update:

  H1 (kinetic):    random-walk diffusion → momentum (retained as Adam m, v)
  H2 (gradient):   drift ∝ alpha * ||∇f||^2  → alpha correction below
  H3 (potential):  beta/gamma modulate gradient magnitude and schedule

Discretizing these three forces and writing them in Adam's per-coordinate
normalized form gives the GradHD update.

Reduction to Adam
-----------------
Set alpha = beta = gamma = 0.  Every correction term vanishes exactly
(they are gated by if-branches, not added unconditionally).  The remaining
update is:

    m   ← beta1 * m + (1-beta1) * g          # first moment
    v   ← beta2 * v + (1-beta2) * g^2        # second moment
    θ   ← θ - lr * (m/(1-beta1^t))
                  / (sqrt(v/(1-beta2^t)) + eps)

which is the standard Adam update (Kingma & Ba, 2015) with no
approximation, rounding, or reordering.

Correction terms (alpha != 0, beta != 0, gamma != 0)
-----------------------------------------------------
alpha:  element-wise gradient-curvature term
        delta_alpha[i] = alpha * g[i] * |g[i]| / (sqrt(v_hat[i]) + eps)
        Pushes harder in directions with large signed-squared gradient,
        down-weighted by the running variance (stabilises large-grad dims).

beta:   gradient-magnitude scaling
        delta_beta[i] = beta * mean(g^2) * g[i] / (sqrt(v_hat[i]) + eps)
        Scales the Adam direction by the average gradient energy;
        amplifies updates when the loss surface is steep.

gamma:  time-decaying gradient boost
        delta_gamma[i] = (gamma / step) * g[i]
        A raw (non-adaptive) gradient term that decays as 1/t,
        providing a stronger initial correction that fades to zero.
"""
from __future__ import annotations
import math
import torch
from torch.optim import Optimizer


class GradHD(Optimizer):
    """GradHD optimizer — Adam + adaptive curvature corrections.

    Parameters
    ----------
    params : iterable
        Model parameters.
    lr : float, default 1e-3
        Global learning rate.
    beta1 : float, default 0.9
        Exponential decay for first moment (momentum).
    beta2 : float, default 0.999
        Exponential decay for second moment (variance).
    eps : float, default 1e-8
        Numerical stability term in denominator.
    alpha : float, default 0.0
        Gradient-curvature coupling.  alpha=0 → Adam exactly.
    beta : float, default 0.0
        Gradient-magnitude scaling.  beta=0 → Adam exactly.
    gamma : float, default 0.0
        Time-decaying gradient boost.  gamma=0 → Adam exactly.
    weight_decay : float, default 0.0
        L2 regularization coefficient (applied to raw gradient, same as AdamW).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        alpha: float = 0.0,
        beta: float = 0.0,
        gamma: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2: {beta2}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        defaults = dict(
            lr=lr, beta1=beta1, beta2=beta2, eps=eps,
            alpha=alpha, beta=beta, gamma=gamma,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["beta1"], group["beta2"]
            eps = group["eps"]
            alpha = group["alpha"]
            beta_q = group["beta"]
            gamma_q = group["gamma"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                if g.is_sparse:
                    raise RuntimeError("GradHD does not support sparse gradients.")

                # Optional L2 regularization (same convention as AdamW)
                if wd != 0.0:
                    g = g.add(p, alpha=wd)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                m, v = state["m"], state["v"]

                # ── Adam first/second moment updates ────────────────────
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                # Bias correction
                bias1 = 1.0 - beta1 ** t
                bias2 = 1.0 - beta2 ** t
                m_hat = m / bias1
                v_hat = v / bias2
                v_sqrt = v_hat.sqrt().add_(eps)  # sqrt(v_hat) + eps

                # ── Base Adam direction ──────────────────────────────────
                # When alpha=beta=gamma=0 this is the entire update.
                update = m_hat.div(v_sqrt)

                # ── alpha: gradient-curvature correction ─────────────────
                # delta[i] = alpha * g[i] * |g[i]| / (sqrt(v_hat[i]) + eps)
                # Reduces to zero when alpha=0 (exact Adam).
                if alpha != 0.0:
                    update.add_(g.abs().mul(g).div(v_sqrt), alpha=alpha)

                # ── beta: gradient-magnitude scaling ─────────────────────
                # delta[i] = beta * mean(g^2) * g[i] / (sqrt(v_hat[i]) + eps)
                # Reduces to zero when beta=0 (exact Adam).
                if beta_q != 0.0:
                    g_norm_sq = float(g.pow(2).mean())
                    update.add_(g.div(v_sqrt), alpha=beta_q * g_norm_sq)

                # ── gamma: time-decaying gradient boost ──────────────────
                # delta[i] = (gamma / t) * g[i]
                # Reduces to zero when gamma=0 (exact Adam).
                if gamma_q != 0.0:
                    update.add_(g, alpha=gamma_q / t)

                p.add_(update, alpha=-lr)

        return loss
