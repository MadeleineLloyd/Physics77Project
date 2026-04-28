import numpy as np


def make_initial_1d(x, p):
    psi = np.zeros_like(x, dtype=np.complex128)
    if p.initial == 'gaussian':
        for x0, k0, sigma0 in zip(p.x0, p.k0, p.sigma0):
            norm = (2 * np.pi * sigma0 ** 2) ** (-0.25)
            envelope = np.exp(-(x - x0) ** 2 / (4 * sigma0 ** 2))
            psi += norm * envelope * np.exp(1j * k0 * x)
        psi /= np.sqrt(float(np.sum(np.abs(psi) ** 2) * p.dx))
    elif p.initial == 'plane':
        for k0 in p.k0:
            psi += np.exp(1j * k0 * x)
        psi /= np.sqrt(float(np.sum(np.abs(psi) ** 2) * p.dx))
    elif p.initial == 'eigenstate':
        if p.potential == 'harmonic':
            from scipy.special import factorial, hermite
            norm = (p.omega / np.pi) ** 0.25 / np.sqrt(2**p.n * factorial(p.n))
            psi = norm * hermite(p.n)(np.sqrt(p.omega) * x) * np.exp(-0.5 * p.omega * x**2)
        elif p.potential == 'box':
            norm = 1 / np.sqrt(p.A)
            if p.n % 2 == 1:
                psi = norm * np.cos(p.n * np.pi * x / (2 * p.A)) * np.where(np.abs(x) < p.A, 1.0, 0.0)
            else:
                psi = norm * np.sin(p.n * np.pi * x / (2 * p.A)) * np.where(np.abs(x) < p.A, 1.0, 0.0)
        else:
            raise ValueError(f"Unknown eigenstate for potential '{p.potential}'.")
    else:
        raise ValueError(f"Unknown 1-D initial '{p.initial}'.")
    return psi


def make_initial_2d(X, Y, p):
    psi = np.zeros_like(X, dtype=np.complex128)
    if p.initial == 'gaussian':
        for x0, y0, k0x, k0y, sigma0 in zip(p.x0, p.y0, p.k0x, p.k0y, p.sigma0):
            norm = (2 * np.pi * sigma0 ** 2) ** (-0.5)
            envelope = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (4 * sigma0 ** 2))
            psi += norm * envelope * np.exp(1j * (k0x * X + k0y * Y))
        psi /= np.sqrt(float(np.sum(np.abs(psi) ** 2) * p.dx * p.dy))
    elif p.initial == 'plane':
        for k0x, k0y in zip(p.k0x, p.k0y):
            psi += np.exp(1j * (k0x * X + k0y * Y))
        psi /= np.sqrt(float(np.sum(np.abs(psi) ** 2) * p.dx * p.dy))
    else:
        raise ValueError(f"Unknown 2-D initial '{p.initial}'.")
    return psi
