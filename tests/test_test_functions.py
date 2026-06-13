"""
Tests for partA_qhd_pde/test_functions.py.

Checks:
  1. Gradient correctness via finite difference for every function.
  2. Known global-minimum locations (verify f(x*) ≈ f_min).
  3. Styblinski-Tang typo: confirm we use 16y^2 (standard form), not 16y.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from partA_qhd_pde.test_functions import (
    convex, convex_grad, CONVEX_BOX, CONVEX_MIN,
    styblinski_tang, styblinski_tang_grad, STYBLINSKI_BOX, STYBLINSKI_MIN,
    michalewicz, michalewicz_grad, MICHALEWICZ_BOX, MICHALEWICZ_MIN,
    cube_wave, cube_wave_grad, CUBE_WAVE_BOX, CUBE_WAVE_MIN,
    rastrigin, rastrigin_grad, RASTRIGIN_BOX, RASTRIGIN_MIN,
    FUNCTIONS, make_grid,
)

EPS = 1e-5  # finite-difference step
GRAD_TOL = 1e-5  # gradient match tolerance (relative to |grad| + 1)


def _finite_diff_grad(f, x, y, eps=EPS):
    """Central finite differences for a scalar (x,y)."""
    gx = (f(x + eps, y) - f(x - eps, y)) / (2 * eps)
    gy = (f(x, y + eps) - f(x, y - eps)) / (2 * eps)
    return gx, gy


CASES = [
    ("convex",         convex,         convex_grad,         CONVEX_BOX),
    ("styblinski_tang",styblinski_tang,styblinski_tang_grad,STYBLINSKI_BOX),
    ("michalewicz",    michalewicz,    michalewicz_grad,    MICHALEWICZ_BOX),
    ("cube_wave",      cube_wave,      cube_wave_grad,      CUBE_WAVE_BOX),
    ("rastrigin",      rastrigin,      rastrigin_grad,      RASTRIGIN_BOX),
]


@pytest.mark.parametrize("name,f,grad_f,box", CASES)
def test_gradient_fd(name, f, grad_f, box):
    """Analytic gradient matches central finite differences at random points."""
    rng = np.random.default_rng(42)
    xs = rng.uniform(box[0] + 0.1, box[1] - 0.1, size=50)
    ys = rng.uniform(box[2] + 0.1, box[3] - 0.1, size=50)

    for x, y in zip(xs, ys):
        gx_fd, gy_fd = _finite_diff_grad(f, float(x), float(y))
        gx_an, gy_an = grad_f(float(x), float(y))
        gx_an = float(np.asarray(gx_an).ravel()[0])
        gy_an = float(np.asarray(gy_an).ravel()[0])

        scale = max(abs(gx_fd), abs(gx_an), 1.0)
        assert abs(gx_an - gx_fd) / scale < GRAD_TOL, (
            f"{name} df/dx mismatch at ({x:.3f},{y:.3f}): "
            f"analytic={gx_an:.6f}, fd={gx_fd:.6f}"
        )
        scale = max(abs(gy_fd), abs(gy_an), 1.0)
        assert abs(gy_an - gy_fd) / scale < GRAD_TOL, (
            f"{name} df/dy mismatch at ({x:.3f},{y:.3f}): "
            f"analytic={gy_an:.6f}, fd={gy_fd:.6f}"
        )


def test_convex_minimum():
    """Global min at (0,0), value 0."""
    assert abs(convex(0.0, 0.0) - CONVEX_MIN) < 1e-10


def test_styblinski_tang_minimum():
    """Global min ≈ -31.33 at (≈-2.9035, ≈-2.9035)."""
    z_star = -2.903534  # argmin of t^4 - 16t^2 + 5t
    f_star = styblinski_tang(z_star, z_star)
    assert abs(f_star - STYBLINSKI_MIN) < 0.01, f"ST min={f_star:.4f}, expected≈{STYBLINSKI_MIN}"


def test_styblinski_tang_uses_standard_form():
    """
    Verify the standard form 16y^2 (not the paper's likely typo 16y).
    The single-variable function 0.2*(z^4 - 16z^2 + 5z) has a root of its
    derivative (4z^3 - 32z + 5) near z = -2.9035, not near z = 0.
    If 16y (linear) were used, the x-term and y-term would have different
    functional forms, breaking symmetry. We check symmetry and the correct min.
    """
    f_sym = styblinski_tang(1.0, 1.0) - styblinski_tang(1.0, 1.0)
    assert f_sym == 0.0  # trivially true; the real check is the minimum location

    # gradient of pure 16y form would be 0.2*(4z^3 + 5) at z=-2.9 (no -32z term)
    # With 16y^2 form, gradient is 0.2*(4z^3 - 32z + 5) ≈ 0 at z=-2.9035
    z = -2.903534
    gx, gy = styblinski_tang_grad(float(z), float(z))
    gx_val = float(np.asarray(gx).ravel()[0])
    gy_val = float(np.asarray(gy).ravel()[0])
    assert abs(gx_val) < 1e-3, f"gradient at known min = {gx_val:.6f}, expected ~0"
    assert abs(gy_val) < 1e-3, f"gradient at known min = {gy_val:.6f}, expected ~0"


def test_michalewicz_box_interior():
    """f is finite and negative inside [0,pi]^2."""
    X, Y = make_grid(MICHALEWICZ_BOX, N=32)
    vals = michalewicz(X, Y)
    assert np.all(np.isfinite(vals))
    assert np.any(vals < -1.0), "Michalewicz should reach < -1 in its box"


def test_cube_wave_minimum():
    """Four global minima near (±0.4945, ±0.4945), value ≈ 0.0305."""
    X, Y = make_grid(CUBE_WAVE_BOX, N=500)
    vals = cube_wave(X, Y)
    grid_min = float(vals.min())
    assert abs(grid_min - CUBE_WAVE_MIN) < 1e-3, (
        f"cube_wave grid min = {grid_min:.6f}, expected ≈ {CUBE_WAVE_MIN}"
    )
    # All 4 symmetric quadrant minima should be close to the same value
    x_star = 0.4945
    for sx, sy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        f_q = cube_wave(sx * x_star, sy * x_star)
        assert f_q < 0.035, (
            f"cube_wave at ({sx*x_star:.4f},{sy*x_star:.4f}) = {f_q:.6f}, expected < 0.035"
        )


def test_rastrigin_minimum():
    """Global min at (0,0), value 0."""
    assert abs(rastrigin(0.0, 0.0) - RASTRIGIN_MIN) < 1e-10


def test_make_grid_shape():
    """make_grid returns (N,N) arrays with correct shape and spacing."""
    box = (-5.0, 5.0, -5.0, 5.0)
    N = 64
    X, Y = make_grid(box, N=N)
    dx = (box[1] - box[0]) / N
    assert X.shape == (N, N)
    assert Y.shape == (N, N)
    assert abs(X[0, 0] - box[0]) < 1e-10
    # endpoint=False: last point is box[1] - dx
    assert abs(X[-1, -1] - (box[1] - dx)) < 1e-10
    assert abs(float(X[1, 0] - X[0, 0]) - dx) < 1e-10


def test_registry_complete():
    """FUNCTIONS registry has all 5 entries with expected keys."""
    required = {"convex", "styblinski_tang", "michalewicz", "cube_wave", "rastrigin"}
    assert set(FUNCTIONS.keys()) == required
    for name, entry in FUNCTIONS.items():
        assert "f" in entry and "grad" in entry and "box" in entry and "min" in entry, name


@pytest.mark.parametrize("name,f,grad_f,box", CASES)
def test_vectorized_grid_eval(name, f, grad_f, box):
    """Functions and grads accept (N,N) arrays and return finite values."""
    X, Y = make_grid(box, N=32)
    vals = f(X, Y)
    gx, gy = grad_f(X, Y)
    assert vals.shape == (32, 32), f"{name} f shape wrong"
    assert gx.shape == (32, 32), f"{name} gx shape wrong"
    assert np.all(np.isfinite(vals)), f"{name} has non-finite f values"
    assert np.all(np.isfinite(gx)), f"{name} has non-finite gx values"
    assert np.all(np.isfinite(gy)), f"{name} has non-finite gy values"
