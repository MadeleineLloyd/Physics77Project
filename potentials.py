import numpy as np


def make_potential_1d(x, p):
    name = p.potential
    V = np.zeros_like(x, dtype=float)
    if name == 'free':
        pass
    elif name == 'box':
        V[x < -p.L / 2.2] = 1e6
        V[x >  p.L / 2.2] = 1e6
    elif name == 'barrier':
        V[np.abs(x) < p.barrier_width / 2] = p.V0
    elif name == 'harmonic':
        V = 0.5 * p.omega ** 2 * x ** 2
    elif name == 'double_well':
        V = 0.05 * x ** 4 - 2.0 * x ** 2
    else:
        raise ValueError(f"Unknown 1-D potential '{name}'.")
    return V


def make_potential_2d(X, Y, p):
    name = p.potential
    V = np.zeros_like(X, dtype=float)
    if name == 'free':
        pass
    elif name == 'double_slit':
        in_wall = np.abs(X) < p.wall_thick / 2
        slit1   = np.abs(Y - p.slit_sep / 2) < p.slit_width / 2
        slit2   = np.abs(Y + p.slit_sep / 2) < p.slit_width / 2
        V[in_wall & ~slit1 & ~slit2] = p.V0
    elif name == 'harmonic_2d':
        V = 0.5 * p.omega ** 2 * (X ** 2 + Y ** 2)
    elif name == 'ring':
        r = np.sqrt(X ** 2 + Y ** 2)
        V[np.abs(r - 5.0) > 0.5] = 1e4
    else:
        raise ValueError(f"Unknown 2-D potential '{name}'.")
    return V
