import numpy as np
from scipy.special import erf

def _smooth_barrier(x, width, smooth):
    if smooth == 0.0:
        return np.where(np.abs(x) < width / 2, 1.0, 0.0)
    return 0.5 * (1 + erf((width / 2 - abs(x)) / (np.sqrt(2) * smooth)))


def make_potential_1d(x, p, t=0.0):
    name = p.potential
    V = np.zeros_like(x, dtype=float)
    if name == 'free':
        pass
    elif name == 'box':
        V = p.V0 * (1 - _smooth_barrier(x, 2 * p.A, p.smooth))
    elif name == 'barrier':
        V = p.V0 * _smooth_barrier(x, p.barrier_width, p.smooth)
    elif name == 'bragg':
        V = p.V0 * np.cos(2 * np.pi * x / p.a) * _smooth_barrier(x, p.barrier_width, 0.0)
    elif name == 'harmonic':
        V = 0.5 * p.omega ** 2 * x ** 2
    elif name == 'driven_harmonic':
        V = 0.5 * p.omega ** 2 * x ** 2 - p.A * p.omega ** 2 * np.cos(p.Omega * t) * x
    else:
        raise ValueError(f"Unknown 1-D potential '{name}'.")
    return V


def make_potential_2d(X, Y, p, t=0.0):
    name = p.potential
    V = np.zeros_like(X, dtype=float)
    if name == 'free':
        pass
    elif name == 'ring':
        V = p.V0 * (1 - _smooth_barrier(X ** 2 + Y ** 2, 2 * p.A ** 2, p.smooth))
    elif name == 'barrier':
        V = p.V0 * _smooth_barrier(X, p.barrier_width, p.smooth)
    elif name == 'double_slit':
        wall = _smooth_barrier(X, p.barrier_width, p.smooth)
        slit1 = _smooth_barrier(Y - p.slit_sep / 2, p.slit_width, p.smooth)
        slit2 = _smooth_barrier(Y + p.slit_sep / 2, p.slit_width, p.smooth)
        V = p.V0 * wall * (1.0 - slit1 - slit2)
    elif name == 'harmonic':
        V = 0.5 * p.omega ** 2 * (X ** 2 + p.epsilon * Y ** 2)
    elif name == 'driven_harmonic':
        center_x = p.A * np.cos(p.Omega * t)
        center_y = p.A * np.sin(p.Omega * t)
        V = 0.5 * p.omega ** 2 * ((X - center_x) ** 2 + p.epsilon * (Y - center_y) ** 2)
    else:
        raise ValueError(f"Unknown 2-D potential '{name}'.")
    return V
