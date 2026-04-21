import numpy as np
from scipy import fft


def _cos_cap(coords, L, width, strength):
    mask = np.ones(len(coords))
    for edge in [-L / 2, L / 2]:
        dist = np.abs(coords - edge)
        ab = dist < width
        mask[ab] *= np.cos(np.pi * (1 - dist[ab] / width) / 2) ** strength
    return mask


def make_grid_1d(p):
    x = np.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False)
    k = 2 * np.pi * fft.fftfreq(p.N, d=p.dx)
    return x, k


def absorbing_mask_1d(x, p):
    return _cos_cap(x, p.L, p.cap_width, p.cap_strength)


def make_grid_2d(p):
    x  = np.linspace(-p.Lx / 2, p.Lx / 2, p.Nx, endpoint=False)
    y  = np.linspace(-p.Ly / 2, p.Ly / 2, p.Ny, endpoint=False)
    X, Y   = np.meshgrid(x, y, indexing='ij')
    kx = 2 * np.pi * fft.fftfreq(p.Nx, d=p.dx)
    ky = 2 * np.pi * fft.fftfreq(p.Ny, d=p.dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    return x, y, X, Y, KX, KY


def absorbing_mask_2d(x, y, p):
    return np.outer(_cos_cap(x, p.Lx, p.cap_width, p.cap_strength),
                    _cos_cap(y, p.Ly, p.cap_width, p.cap_strength))
