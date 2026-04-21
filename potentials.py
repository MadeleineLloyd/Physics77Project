"""
potentials.py
─────────────
Static potential energy arrays V(x) and V(x, y).
All quantities in atomic units: ℏ = m = 1.

Public API
----------
make_potential_1d(x, p)     → V  shape (N,)
make_potential_2d(X, Y, p)  → V  shape (Nx, Ny)

Both dispatchers read p.potential (str) and construct the corresponding
array. Unknown names raise ValueError.

1-D potentials
--------------
'free'        : V = 0
'box'         : hard walls at box edges (V = 1e6)
'barrier'     : rectangular barrier at x = 0
'harmonic'    : isotropic SHO  V = ½ω²x²
'double_well' : quartic  V = a x⁴ − b x²

2-D potentials
--------------
'free'        : V = 0
'double_slit' : thin wall at x = 0 with two openings at y = ±slit_sep/2
'harmonic_2d' : isotropic 2D SHO  V = ½ω²(x²+y²)
'ring'        : circular hard wall, confines packet to a ring of radius R₀
"""

import numpy as np


# ── 1-D ───────────────────────────────────────────────────────────────────
def make_potential_1d(x: np.ndarray, p) -> np.ndarray:
    """
    Return V(x) for the potential selected in Params1D.

    Parameters
    ----------
    x : 1-D position grid, shape (N,)
    p : Params1D instance

    Returns
    -------
    V : shape (N,), float64
    """
    V    = np.zeros_like(x, dtype=float)
    name = p.potential

    if name == 'free':
        pass                                            # V = 0 everywhere

    elif name == 'box':
        V[x < -p.L / 2.2] = 1e6                       # hard left wall
        V[x >  p.L / 2.2] = 1e6                       # hard right wall

    elif name == 'barrier':
        # Rectangular barrier centred at x = 0
        V[np.abs(x) < p.barrier_width / 2] = p.V0

    elif name == 'harmonic':
        V = 0.5 * p.omega ** 2 * x ** 2               # ½mω²x²,  m = 1

    elif name == 'double_well':
        # Quartic double-well: V = a x⁴ − b x²
        # Minima at x = ±√(b/2a), barrier height = b²/4a
        a, b = 0.05, 2.0
        V = a * x ** 4 - b * x ** 2

    else:
        raise ValueError(
            f"Unknown 1-D potential '{name}'. "
            "Choose from: 'free', 'box', 'barrier', 'harmonic', 'double_well'."
        )

    return V


# ── 2-D ───────────────────────────────────────────────────────────────────
def make_potential_2d(X: np.ndarray, Y: np.ndarray, p) -> np.ndarray:
    """
    Return V(x, y) for the potential selected in Params2D.

    Parameters
    ----------
    X, Y : 2-D coordinate arrays, shape (Nx, Ny), indexing='ij'
    p    : Params2D instance

    Returns
    -------
    V : shape (Nx, Ny), float64
    """
    V    = np.zeros_like(X, dtype=float)
    name = p.potential

    if name == 'free':
        pass                                            # V = 0 everywhere

    elif name == 'double_slit':
        # Thin wall at x = 0, two openings centred at y = ±slit_sep/2
        in_wall = np.abs(X) < p.wall_thick / 2
        slit1   = np.abs(Y - p.slit_sep / 2) < p.slit_width / 2
        slit2   = np.abs(Y + p.slit_sep / 2) < p.slit_width / 2
        V[in_wall & ~slit1 & ~slit2] = p.V0

    elif name == 'harmonic_2d':
        V = 0.5 * p.omega ** 2 * (X ** 2 + Y ** 2)    # isotropic 2D SHO

    elif name == 'ring':
        # Hard circular wall: confines the packet to radius R₀ ± dR
        R0, dR = 5.0, 0.5
        r = np.sqrt(X ** 2 + Y ** 2)
        V[np.abs(r - R0) > dR] = 1e4

    else:
        raise ValueError(
            f"Unknown 2-D potential '{name}'. "
            "Choose from: 'free', 'double_slit', 'harmonic_2d', 'ring'."
        )

    return V
