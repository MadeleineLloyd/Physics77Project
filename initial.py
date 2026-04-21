"""
initial.py
──────────
Gaussian wave-packet constructors for 1-D and 2-D simulations.
All quantities in atomic units: ℏ = m = 1.

Public API
----------
gaussian_packet_1d(x, p)     → ψ₀  shape (N,),      complex128
gaussian_packet_2d(X, Y, p)  → ψ₀  shape (Nx, Ny),  complex128

Both functions produce normalised states:
    ∫ |ψ₀|² dx = 1   (1-D)
    ∬ |ψ₀|² dx dy = 1   (2-D)
"""

import numpy as np


def gaussian_packet_1d(x: np.ndarray, p) -> np.ndarray:
    """
    Normalised 1-D Gaussian coherent state.

        ψ(x, 0) = (2πσ²)^{-1/4} exp(−(x−x₀)²/4σ²) exp(ik₀x)

    This is a minimum-uncertainty state: Δx · Δp = ℏ/2.

    Parameters
    ----------
    x : 1-D position grid, shape (N,)
    p : Params1D instance  (uses p.x0, p.sigma, p.k0)

    Returns
    -------
    psi : shape (N,), complex128
    """
    norm     = (2 * np.pi * p.sigma ** 2) ** (-0.25)
    envelope = np.exp(-(x - p.x0) ** 2 / (4 * p.sigma ** 2))
    phase    = np.exp(1j * p.k0 * x)
    return (norm * envelope * phase).astype(np.complex128)


def gaussian_packet_2d(X: np.ndarray, Y: np.ndarray, p) -> np.ndarray:
    """
    Normalised 2-D isotropic Gaussian coherent state.

        ψ(x, y, 0) = (2πσ²)^{-1/2}
                     · exp(−((x−x₀)²+(y−y₀)²) / 4σ²)
                     · exp(i(k₀ₓx + k₀ᵧy))

    The 2-D normalisation factor (2πσ²)^{-1/2} = [(2πσ²)^{-1/4}]²
    ensures ∬|ψ|² dx dy = 1.

    Parameters
    ----------
    X, Y : 2-D coordinate arrays, shape (Nx, Ny), indexing='ij'
    p    : Params2D instance  (uses p.x0, p.y0, p.sigma, p.k0x, p.k0y)

    Returns
    -------
    psi : shape (Nx, Ny), complex128
    """
    norm     = (2 * np.pi * p.sigma ** 2) ** (-0.5)
    envelope = np.exp(-((X - p.x0) ** 2 + (Y - p.y0) ** 2) / (4 * p.sigma ** 2))
    phase    = np.exp(1j * (p.k0x * X + p.k0y * Y))
    return (norm * envelope * phase).astype(np.complex128)
