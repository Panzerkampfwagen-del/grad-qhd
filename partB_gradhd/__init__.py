"""
Part B: GradHD — a classical PyTorch optimizer.

GradHD adds adaptive gradient-curvature corrections to Adam, controlled
by three hyperparameters (alpha, beta, gamma).  When all three are zero
the optimizer is mathematically identical to Adam.  No quantum physics,
no tunneling — the update rule is derived from a classical ODE.
"""
from .gradhd_optim import GradHD

__all__ = ["GradHD"]
