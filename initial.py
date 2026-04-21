import numpy as np


def gaussian_packet_1d(x, p):
    norm = (2 * np.pi * p.sigma ** 2) ** (-0.25)
    return (norm * np.exp(-(x - p.x0) ** 2 / (4 * p.sigma ** 2))
                 * np.exp(1j * p.k0 * x)).astype(np.complex128)


def gaussian_packet_2d(X, Y, p):
    norm = (2 * np.pi * p.sigma ** 2) ** (-0.5)
    return (norm * np.exp(-((X - p.x0) ** 2 + (Y - p.y0) ** 2) / (4 * p.sigma ** 2))
                 * np.exp(1j * (p.k0x * X + p.k0y * Y))).astype(np.complex128)
