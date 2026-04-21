"""
observables.py
──────────────
Physical observables computed from the wave function ψ.
All quantities in atomic units: ℏ = m = 1.

The functions use explicit volume elements (dx or dx·dy) so they
work correctly for both 1-D and 2-D cases — the caller supplies the
appropriate grid arrays and FFT callable.

Public API  (1-D)
-----------------
norm_1d(psi, dx)
expect_x_1d(psi, x, dx)
expect_p_1d(psi, k, dx)
energy_1d(psi, k, V, dx)
transmission_1d(psi, x, dx)

Public API  (2-D)
-----------------
norm_2d(psi, dx, dy)
expect_x_2d(psi, X, dx, dy)
expect_y_2d(psi, Y, dx, dy)
expect_px_2d(psi, KX, dx, dy)
energy_2d(psi, KX, KY, V, dx, dy)
transmitted_prob_2d(psi, x, dx, dy)
"""

import numpy as np
from scipy import fft as _fft


# ════════════════════════════════════════════════════════════════════
#  1-D observables
# ════════════════════════════════════════════════════════════════════

def norm_1d(psi: np.ndarray, dx: float) -> float:
    """
    Total probability  ∫ |ψ|² dx.
    Should remain ≈ 1 for the Strang / Yoshida integrators
    (exact in the absence of the absorbing boundary).
    """
    return float(np.sum(np.abs(psi) ** 2) * dx)


def expect_x_1d(psi: np.ndarray, x: np.ndarray, dx: float) -> float:
    """⟨x⟩ = ∫ x |ψ|² dx — mean position."""
    return float(np.real(np.sum(x * np.abs(psi) ** 2) * dx))


def expect_p_1d(psi: np.ndarray, k: np.ndarray, dx: float) -> float:
    """
    ⟨p⟩ = ⟨ℏk⟩ = ∫ k |ψ̃(k)|² dk   (ℏ = 1).

    The discrete FFT convention used by scipy.fft gives
        ψ̃(k) ≈ (dx/√2π) Σⱼ ψ(xⱼ) e^{−ikxⱼ}
    so the momentum-space probability density is |ψ̃|² · dk
    with dk = 2π / (N dx).
    """
    N      = len(psi)
    psi_k  = _fft.fft(psi) * dx / np.sqrt(2 * np.pi)
    dk     = 2 * np.pi / (N * dx)
    return float(np.real(np.sum(k * np.abs(psi_k) ** 2) * dk))


def energy_1d(psi: np.ndarray, k: np.ndarray,
              V: np.ndarray, dx: float) -> float:
    """
    Total energy  ⟨H⟩ = ⟨T⟩ + ⟨V⟩.

        ⟨T⟩ = ∫ (k²/2) |ψ̃(k)|² dk
        ⟨V⟩ = ∫ V(x) |ψ(x)|²  dx
    """
    N      = len(psi)
    psi_k  = _fft.fft(psi) * dx / np.sqrt(2 * np.pi)
    dk     = 2 * np.pi / (N * dx)
    KE = float(np.real(np.sum((k ** 2 / 2) * np.abs(psi_k) ** 2) * dk))
    PE = float(np.real(np.sum(V * np.abs(psi) ** 2) * dx))
    return KE + PE


def transmission_1d(psi: np.ndarray, x: np.ndarray, dx: float) -> float:
    """
    Transmission coefficient  T = ∫_{x>0} |ψ|² dx.

    Valid for a barrier centred at x = 0.
    Reflection coefficient R = norm_1d(psi, dx) − T  (≈ 1 − T
    when norm is close to 1, i.e. before significant absorption).
    """
    return float(np.sum(np.abs(psi[x > 0]) ** 2) * dx)


# ════════════════════════════════════════════════════════════════════
#  2-D observables
# ════════════════════════════════════════════════════════════════════

def norm_2d(psi: np.ndarray, dx: float, dy: float) -> float:
    """Total probability  ∬ |ψ|² dx dy."""
    return float(np.sum(np.abs(psi) ** 2) * dx * dy)


def expect_x_2d(psi: np.ndarray, X: np.ndarray,
                dx: float, dy: float) -> float:
    """⟨x⟩ = ∬ x |ψ|² dx dy."""
    return float(np.real(np.sum(X * np.abs(psi) ** 2) * dx * dy))


def expect_y_2d(psi: np.ndarray, Y: np.ndarray,
                dx: float, dy: float) -> float:
    """⟨y⟩ = ∬ y |ψ|² dx dy."""
    return float(np.real(np.sum(Y * np.abs(psi) ** 2) * dx * dy))


def expect_px_2d(psi: np.ndarray, KX: np.ndarray,
                 dx: float, dy: float) -> float:
    """
    ⟨pₓ⟩ = ∬ kₓ |ψ̃(k)|² dkₓ dkᵧ   (ℏ = 1).

    Uses the 2-D FFT convention:
        ψ̃(k) ≈ (dx dy / 2π) Σ ψ(r) e^{−ik·r}
    """
    Nx, Ny = psi.shape
    norm_k = dx * dy / (2 * np.pi)
    psi_k  = _fft.fft2(psi) * norm_k
    dkx    = 2 * np.pi / (Nx * dx)
    dky    = 2 * np.pi / (Ny * dy)
    return float(np.real(np.sum(KX * np.abs(psi_k) ** 2) * dkx * dky))


def energy_2d(psi: np.ndarray,
              KX: np.ndarray, KY: np.ndarray,
              V: np.ndarray,
              dx: float, dy: float) -> float:
    """
    Total energy  ⟨H⟩ = ⟨T⟩ + ⟨V⟩  in 2-D.

        ⟨T⟩ = ∬ (kₓ²+kᵧ²)/2 · |ψ̃|² dkₓ dkᵧ
        ⟨V⟩ = ∬ V(x,y) |ψ|²   dx dy
    """
    Nx, Ny = psi.shape
    norm_k = dx * dy / (2 * np.pi)
    psi_k  = _fft.fft2(psi) * norm_k
    dkx    = 2 * np.pi / (Nx * dx)
    dky    = 2 * np.pi / (Ny * dy)
    KE = float(np.real(
        np.sum((KX ** 2 + KY ** 2) / 2 * np.abs(psi_k) ** 2) * dkx * dky
    ))
    PE = float(np.real(np.sum(V * np.abs(psi) ** 2) * dx * dy))
    return KE + PE


def transmitted_prob_2d(psi: np.ndarray, x: np.ndarray,
                        dx: float, dy: float) -> float:
    """
    Fraction of probability with x > 0.

    For a double-slit wall at x = 0, this measures the fraction of
    the wave packet that has passed through the barrier.

    Parameters
    ----------
    psi : shape (Nx, Ny)
    x   : 1-D x-grid, shape (Nx,)
    """
    return float(np.sum(np.abs(psi[x > 0, :]) ** 2) * dx * dy)
