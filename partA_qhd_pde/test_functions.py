"""
2D test functions from Leng & Shi (2025) Appendix E.2.

Each function returns (f, grad_f) where grad_f = (df/dx, df/dy).
All functions accept numpy arrays for grid evaluation or scalar x,y.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


# ---- convex ----------------------------------------------------------------

def convex(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (x + y) ** 4 / 256 + (x - y) ** 4 / 128


def convex_grad(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    u, v = x + y, x - y
    dfdu = 4 * u ** 3 / 256
    dfdv = 4 * v ** 3 / 128
    return dfdu + dfdv, dfdu - dfdv  # chain rule: df/dx, df/dy


CONVEX_BOX = (-5.0, 5.0, -5.0, 5.0)
CONVEX_MIN = 0.0  # global min at x=y=0


# ---- Styblinski-Tang -------------------------------------------------------
# Paper Appendix E.2 has "16y" in the text -- a typo; the standard function
# uses 16y^2. We use the standard form: 0.2*(z^4 - 16z^2 + 5z) for each dim.
# Verified global min per dimension at z ≈ -2.9035, giving f ≈ -39.1664,
# scaled by 0.2 -> -7.8333 per dimension, sum -> ≈ -31.33.

def styblinski_tang(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    def _st(z: np.ndarray) -> np.ndarray:
        return z ** 4 - 16 * z ** 2 + 5 * z

    return 0.2 * (_st(x) + _st(y))


def styblinski_tang_grad(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    def _dst(z: np.ndarray) -> np.ndarray:
        return 4 * z ** 3 - 32 * z + 5

    return 0.2 * _dst(x), 0.2 * _dst(y)


STYBLINSKI_BOX = (-5.0, 5.0, -5.0, 5.0)
STYBLINSKI_MIN = -31.3312  # 2 * 0.2 * (-39.1664)


# ---- Michalewicz -----------------------------------------------------------

def michalewicz(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -(np.sin(x) * np.sin(x ** 2 / np.pi) ** 20
             + np.sin(y) * np.sin(2 * y ** 2 / np.pi) ** 20)


def michalewicz_grad(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # df/dx: -d/dx [sin(x) * sin(x^2/pi)^20]
    s1 = np.sin(x ** 2 / np.pi)
    c1 = np.cos(x ** 2 / np.pi)
    # derivative of sin(x) * sin(x^2/pi)^20 w.r.t. x
    dx = np.cos(x) * s1 ** 20 + np.sin(x) * 20 * s1 ** 19 * c1 * (2 * x / np.pi)

    s2 = np.sin(2 * y ** 2 / np.pi)
    c2 = np.cos(2 * y ** 2 / np.pi)
    dy = np.cos(y) * s2 ** 20 + np.sin(y) * 20 * s2 ** 19 * c2 * (4 * y / np.pi)

    return -dx, -dy


MICHALEWICZ_BOX = (0.0, np.pi, 0.0, np.pi)
MICHALEWICZ_MIN = -1.8013  # approximate 2D global min


# ---- Cube-Wave -------------------------------------------------------------

def cube_wave(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.cos(np.pi * x) ** 2 + 0.25 * x ** 4 + np.cos(np.pi * y) ** 2 + 0.25 * y ** 4


def cube_wave_grad(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dfx = -2 * np.cos(np.pi * x) * np.sin(np.pi * x) * np.pi + x ** 3
    dfy = -2 * np.cos(np.pi * y) * np.sin(np.pi * y) * np.pi + y ** 3
    return dfx, dfy


CUBE_WAVE_BOX = (-2.0, 2.0, -2.0, 2.0)
# 4 global minima near (±0.4945, ±0.4945); verified numerically on N=1000 grid
CUBE_WAVE_MIN = 0.0305


# ---- Rastrigin -------------------------------------------------------------

def rastrigin(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (x ** 2 - 10 * np.cos(2 * np.pi * x)
            + y ** 2 - 10 * np.cos(2 * np.pi * y) + 20)


def rastrigin_grad(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dfx = 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)
    dfy = 2 * y + 20 * np.pi * np.sin(2 * np.pi * y)
    return dfx, dfy


RASTRIGIN_BOX = (-3.0, 3.0, -3.0, 3.0)
RASTRIGIN_MIN = 0.0  # global min at (0,0)


# ---- registry --------------------------------------------------------------

FUNCTIONS = {
    "convex": {
        "f": convex,
        "grad": convex_grad,
        "box": CONVEX_BOX,
        "min": CONVEX_MIN,
    },
    "styblinski_tang": {
        "f": styblinski_tang,
        "grad": styblinski_tang_grad,
        "box": STYBLINSKI_BOX,
        "min": STYBLINSKI_MIN,
    },
    "michalewicz": {
        "f": michalewicz,
        "grad": michalewicz_grad,
        "box": MICHALEWICZ_BOX,
        "min": MICHALEWICZ_MIN,
    },
    "cube_wave": {
        "f": cube_wave,
        "grad": cube_wave_grad,
        "box": CUBE_WAVE_BOX,
        "min": CUBE_WAVE_MIN,
    },
    "rastrigin": {
        "f": rastrigin,
        "grad": rastrigin_grad,
        "box": RASTRIGIN_BOX,
        "min": RASTRIGIN_MIN,
    },
}


def make_grid(box: Tuple[float, float, float, float], N: int = 128
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, Y) meshgrid over box = (xmin, xmax, ymin, ymax).

    Uses endpoint=False so the grid is periodic-FFT compatible:
    dx = (xmax-xmin)/N and the last point is xmax - dx (not xmax).
    """
    x = np.linspace(box[0], box[1], N, endpoint=False)
    y = np.linspace(box[2], box[3], N, endpoint=False)
    return np.meshgrid(x, y, indexing="ij")  # shape (N, N)
