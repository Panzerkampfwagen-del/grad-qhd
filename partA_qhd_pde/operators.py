"""
Split-operator pieces for gradient-based QHD (Leng & Shi, ICML 2025).

The Hamiltonian is H = H1 + H2 + H3 where:
  H1 = -(1/(2t^3)) Laplacian                        (kinetic, FFT-diagonal)
  H2 = (alpha/2) {-i grad, grad f}                  (gradient correction, Hermitian)
  H3 = ((alpha^2+beta)/2) t^3 ||grad f||^2
       + (t^3 + gamma t^2) f                         (potential, position-diagonal)

Each apply_* function propagates psi by one time step h under its sub-Hamiltonian.
All functions operate on numpy complex128 arrays of shape (N, N).
"""

from __future__ import annotations
import numpy as np
from typing import Tuple
from scipy.sparse.linalg import LinearOperator, gmres as _gmres


# ---- momentum / wavenumber grids -------------------------------------------

def wavenumber_grid(N: int, dx: float) -> np.ndarray:
    """1-D wavenumber array k_n = 2π n / (N dx).

    For even N the Nyquist bin (n = N//2) is set to zero so that the array
    is strictly anti-symmetric (k_{-n} = -k_n), which is required for the
    discrete IBP identity  Σ a_j (∂_x b)_j = -Σ (∂_x a)_j b_j  to hold
    exactly and make the H2 operator discretely Hermitian.
    """
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    if N % 2 == 0:
        k[N // 2] = 0.0  # zero Nyquist bin
    return k


def k2_grid(N: int, dx: float, dy: float) -> np.ndarray:
    """2-D |k|^2 grid, shape (N,N)."""
    kx = wavenumber_grid(N, dx)
    ky = wavenumber_grid(N, dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    return KX ** 2 + KY ** 2


def k_grids(N: int, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (KX, KY) wavenumber grids, each shape (N,N)."""
    kx = wavenumber_grid(N, dx)
    ky = wavenumber_grid(N, dy)
    return np.meshgrid(kx, ky, indexing="ij")


# ---- H1: kinetic (Laplacian), diagonal in Fourier space --------------------

def apply_H1(
    psi: np.ndarray,
    K2: np.ndarray,
    t: float,
    h: float,
) -> np.ndarray:
    """
    exp(-i h H1) psi  where  H1 = -(1/(2t^3)) Laplacian.

    In Fourier space: H1 diagonal with eigenvalue (1/(2t^3)) |k|^2.
    Phase: exp(-i h * (1/(2t^3)) * |k|^2).
    """
    psi_k = np.fft.fft2(psi)
    phase = np.exp(-1j * h * K2 / (2.0 * t ** 3))
    return np.fft.ifft2(phase * psi_k)


# ---- H3: potential, diagonal in position space -----------------------------

def apply_H3(
    psi: np.ndarray,
    f_grid: np.ndarray,
    gradf_sq: np.ndarray,
    t: float,
    h: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """
    exp(-i h H3) psi  where
    H3 = ((alpha^2+beta)/2) t^3 ||grad f||^2 + (t^3 + gamma t^2) f.

    This is a purely real, position-diagonal operator, so it acts as pointwise
    phase multiplication.
    """
    V = 0.5 * (alpha ** 2 + beta) * t ** 3 * gradf_sq + (t ** 3 + gamma * t ** 2) * f_grid
    return psi * np.exp(-1j * h * V)


# ---- H2: gradient-correction term ------------------------------------------
#
# H2 = (alpha/2) {-i grad, grad f}  (anticommutator of -i grad and grad f)
#
# The anticommutator {A, B} = AB + BA for operators A = -i grad (momentum)
# and B = grad f (a multiplicative operator):
#   (-i grad)(grad_f * psi) + grad_f * (-i grad psi)
#   = -i (psi * Delta_f + grad_f . grad_psi) - i grad_f . grad_psi
#   = -i psi * Delta_f - 2i (grad_f . grad_psi)
#
# So H2 psi = (alpha/2)[-i psi * Delta_f - 2i (grad_f . grad_psi)]
#           = -(i alpha / 2)[psi * Delta_f + 2 grad_f . grad_psi]
#
# For exact time-evolution exp(-i h H2) we use first-order Lie-Trotter
# approximation: exp(-i h H2) ~ I - i h H2 to 1st order, or we exponentiate
# the full operator using a pseudo-spectral approach.
#
# Practical approach: since H2 is linear in alpha, we compute H2*psi explicitly
# and use a matrix-free Crank-Nicolson (midpoint) integrator for this sub-step,
# which is norm-preserving by construction.
#
# We use: psi_new = (I + i(h/2)H2)^{-1} (I - i(h/2)H2) psi
# Both (I ± i(h/2)H2) are applied via a single explicit application since H2
# involves only spectral derivatives and pointwise multiplications.
# For small h the CN approximation matches exp(-ihH2) to O(h^3).

def dealias(psi: np.ndarray) -> np.ndarray:
    """2/3-rule spectral dealiasing: zero Fourier modes with |k_index| > N//3.

    After each H3 step, exp(-ihV)*psi creates modes up to k_eff = k_psi + k_V.
    Once k_eff > k_Nyquist the spectral derivative in _apply_H2_to aliases,
    breaking the anti-Hermitian property of the operator and causing norm blowup.
    Zeroing the top third of modes prevents this aliasing so that H2 remains
    norm-preserving throughout the simulation.
    """
    N = psi.shape[0]
    # Keep indices 0..c-1 and N-c+1..N-1, i.e. freqs {0..+(c-1), -(c-1)..-1}.
    # Zeroing N-c as well ensures the band is conjugate-symmetric: both +c and -c
    # are excluded, so the dealiased subspace has an antisymmetric derivative KX.
    c = N // 3
    psi_k = np.fft.fft2(psi)
    psi_k[c : N - c + 1, :] = 0.0
    psi_k[:, c : N - c + 1] = 0.0
    return np.fft.ifft2(psi_k)


def _zero_nyquist(f_k: np.ndarray) -> np.ndarray:
    """Zero the Nyquist mode(s) so KX is antisymmetric for even N.

    Without this, discrete IBP sum_j a*(d_x b)_j = -sum_j (d_x a)*_j b_j fails
    at the Nyquist bin, breaking Hermiticity of H2.
    """
    N = f_k.shape[0]
    out = f_k.copy()
    if N % 2 == 0:
        out[N // 2, :] = 0
        out[:, N // 2] = 0
    return out


def _apply_H2_to(
    psi: np.ndarray,
    KX: np.ndarray,
    KY: np.ndarray,
    gfx: np.ndarray,
    gfy: np.ndarray,
    lapf: np.ndarray,
) -> np.ndarray:
    """H2 action: -i [Δf·ψ + 2 ∇f·∇ψ]  (the alpha/2 factor is NOT included here).

    The zero-Nyquist condition is carried implicitly by KX and KY (each has
    k[N//2]=0 from wavenumber_grid), so iKX*psi_k and iKY*psi_k have the
    correct Nyquist rows/columns zeroed without any explicit projection on psi_k.
    Projecting psi_k via _zero_nyquist would make the matvec non-skew-Hermitian
    and cause systematic norm drift in expm_multiply.
    """
    psi_k = np.fft.fft2(psi)
    dpsi_dx = np.fft.ifft2(1j * KX * psi_k)
    dpsi_dy = np.fft.ifft2(1j * KY * psi_k)
    return -1j * (psi * lapf + 2.0 * (gfx * dpsi_dx + gfy * dpsi_dy))


def apply_H2(
    psi: np.ndarray,
    KX: np.ndarray,
    KY: np.ndarray,
    gfx: np.ndarray,
    gfy: np.ndarray,
    lapf: np.ndarray,
    h: float,
    alpha: float,
    tol: float = 1e-10,
) -> np.ndarray:
    """
    Apply exp(-i h H2) psi via Crank-Nicolson (Cayley transform).

    CN discretization of exp(A) where A = -i h H2 (anti-Hermitian):
        (I - A/2) psi_new = (I + A/2) psi

    The Cayley transform (I - A/2)^{-1}(I + A/2) is EXACTLY unitary for any
    anti-Hermitian A, regardless of boundary conditions or state oscillations.
    This prevents the norm runaway that occurs with Krylov methods (expm_multiply)
    when psi accumulates high-frequency content from H3 phase rotations.

    The O(h^2) splitting error in the evolution direction is acceptable for
    small h.  GMRES converges in O(1) iterations since ||(A/2)|| << 1 in practice.
    """
    if alpha == 0.0:
        return psi

    N = psi.shape[0]
    flat = N * N
    # A = coeff * L_H2 is anti-Hermitian (coeff is purely imaginary, L_H2 Hermitian)
    coeff = -1j * h * 0.5 * alpha
    half = coeff / 2.0

    def _L(v_flat: np.ndarray) -> np.ndarray:
        return _apply_H2_to(v_flat.reshape(N, N), KX, KY, gfx, gfy, lapf).ravel()

    psi_flat = psi.ravel()
    b = psi_flat + half * _L(psi_flat)  # (I + A/2) psi

    def _matvec_lhs(v: np.ndarray) -> np.ndarray:
        return v - half * _L(v)  # (I - A/2) v

    A_lhs = LinearOperator((flat, flat), matvec=_matvec_lhs, dtype=np.complex128)
    # maxiter caps outer restart cycles (default is N²; each cycle costs restart=20
    # matvecs, so the default is catastrophically slow for large N or large ||h*L||).
    # For a well-conditioned system (||half*L|| < 30) 20 restart cycles is plenty.
    psi_new, _info = _gmres(A_lhs, b, x0=psi_flat, rtol=tol, atol=0, maxiter=20)
    if _info != 0:
        import warnings
        warnings.warn(
            f"apply_H2 GMRES did not converge (info={_info}); "
            "the CN step is non-unitary and psi may be physically wrong. "
            "Reduce h or alpha to keep ||half*L|| small.",
            RuntimeWarning, stacklevel=2,
        )
    return psi_new.reshape(N, N)


# ---- Precompute landscape quantities on the grid ---------------------------

def precompute_landscape(
    X: np.ndarray,
    Y: np.ndarray,
    f_fn,
    grad_fn,
    dx: float,
    dy: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate f, grad f, |grad f|^2, and Laplacian f on the grid.

    gfx, gfy, and lapf are computed as SPECTRAL derivatives of f_grid so
    that they are mutually consistent: discrete integration by parts holds
    exactly, making H2 = (alpha/2){-i grad, grad f} discretely Hermitian.

    Using spectral (rather than analytic) gradients introduces Gibbs
    wrap-around artifacts at box boundaries, but the wavefunction psi decays
    to ~0 there, so physical observables are unaffected.

    Returns (f_grid, gfx, gfy, gradf_sq, lapf), all shape (N,N).
    """
    N = X.shape[0]
    f_grid = f_fn(X, Y).astype(np.float64)

    kx = wavenumber_grid(N, dx)
    ky = wavenumber_grid(N, dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K2 = KX ** 2 + KY ** 2

    # Zero the Nyquist mode so discrete IBP holds exactly in H2.
    f_k = _zero_nyquist(np.fft.fft2(f_grid))
    gfx = np.real(np.fft.ifft2(1j * KX * f_k))
    gfy = np.real(np.fft.ifft2(1j * KY * f_k))
    gradf_sq = gfx ** 2 + gfy ** 2
    lapf = np.real(np.fft.ifft2(-K2 * f_k))  # consistent: Δf = ∂_x gfx + ∂_y gfy

    return f_grid, gfx, gfy, gradf_sq, lapf


# ---- Initial wavepacket ----------------------------------------------------

def gaussian_wavepacket(
    X: np.ndarray,
    Y: np.ndarray,
    x0: float,
    y0: float,
    sigma: float,
) -> np.ndarray:
    """Normalized broad Gaussian wavepacket centered at (x0, y0)."""
    psi = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (4 * sigma ** 2)).astype(
        np.complex128
    )
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2))
    return psi
