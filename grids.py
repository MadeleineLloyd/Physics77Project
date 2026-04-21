"""
grids.py
────────
Grid construction and absorbing boundary masks.
All routines work for both 1D (arrays) and 2D (meshgrid arrays).

Public API
----------
make_grid_1d(p)          → x, k
make_grid_2d(p)          → x, y, X, Y, KX, KY
absorbing_mask_1d(x, p)  → mask  shape (N,)
absorbing_mask_2d(x, y, p) → mask  shape (Nx, Ny)
"""

import numpy as np
from scipy import fft


# ── Internal helper ────────────────────────────────────────────────────────
def _cos_cap_1d(coords: np.ndarray, L: float,
                width: float, strength: float) -> np.ndarray:
    """
    1-D cosine absorbing mask on [-L/2, L/2].

    Within `width` a.u. of each edge the mask tapers from 1 → 0 as
        mask = cos(π/2 · (1 − dist/width))^strength
    Outside the absorption layer mask = 1 (no attenuation).

    Parameters
    ----------
    coords   : 1-D position array
    L        : box length
    width    : CAP layer thickness [a.u.]
    strength : cosine exponent (larger → stronger absorption)
    """
    mask = np.ones(len(coords))
    for edge in [-L / 2, L / 2]:
        dist   = np.abs(coords - edge)
        absorb = dist < width
        mask[absorb] *= np.cos(np.pi * (1 - dist[absorb] / width) / 2) ** strength
    return mask


# ── 1-D grid ───────────────────────────────────────────────────────────────
def make_grid_1d(p) -> tuple[np.ndarray, np.ndarray]:
    """
    Build 1-D position and momentum grids from a Params1D object.

    Returns
    -------
    x : shape (N,)   position grid  [-L/2, L/2)
    k : shape (N,)   momentum grid  (rad a.u.⁻¹)
    """
    x = np.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False)
    k = 2 * np.pi * fft.fftfreq(p.N, d=p.dx)
    return x, k


def absorbing_mask_1d(x: np.ndarray, p) -> np.ndarray:
    """
    Cosine CAP mask for the 1-D box.

    Returns
    -------
    mask : shape (N,),  real-valued in (0, 1]
    """
    return _cos_cap_1d(x, p.L, p.cap_width, p.cap_strength)


# ── 2-D grid ───────────────────────────────────────────────────────────────
def make_grid_2d(p) -> tuple:
    """
    Build 2-D position and momentum grids from a Params2D object.

    Returns
    -------
    x   : shape (Nx,)
    y   : shape (Ny,)
    X   : shape (Nx, Ny)  — meshgrid, indexing='ij'
    Y   : shape (Nx, Ny)
    KX  : shape (Nx, Ny)  — momentum meshgrid
    KY  : shape (Nx, Ny)
    """
    x = np.linspace(-p.Lx / 2, p.Lx / 2, p.Nx, endpoint=False)
    y = np.linspace(-p.Ly / 2, p.Ly / 2, p.Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    kx = 2 * np.pi * fft.fftfreq(p.Nx, d=p.dx)
    ky = 2 * np.pi * fft.fftfreq(p.Ny, d=p.dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    return x, y, X, Y, KX, KY


def absorbing_mask_2d(x: np.ndarray, y: np.ndarray, p) -> np.ndarray:
    """
    Separable cosine CAP mask for the 2-D box.

    The 2-D mask is the outer product of two independent 1-D masks:
        cap(x, y) = cap_x(x) · cap_y(y)
    This is computationally efficient and physically correct because
    the absorption acts independently along each axis.

    Returns
    -------
    mask : shape (Nx, Ny), real-valued in (0, 1]
    """
    cap_x = _cos_cap_1d(x, p.Lx, p.cap_width, p.cap_strength)
    cap_y = _cos_cap_1d(y, p.Ly, p.cap_width, p.cap_strength)
    return np.outer(cap_x, cap_y)
