import numpy as np
from scipy import fft as _fft


# ── 1-D ───────────────────────────────────────────────────────────────────

def norm_1d(psi, dx):
    return float(np.sum(np.abs(psi) ** 2) * dx)

def expect_x_1d(psi, x, dx):
    return float(np.real(np.sum(x * np.abs(psi) ** 2) * dx))

def expect_p_1d(psi, k, dx):
    psi_k = _fft.fft(psi) * dx / np.sqrt(2 * np.pi)
    dk    = 2 * np.pi / (len(psi) * dx)
    return float(np.real(np.sum(k * np.abs(psi_k) ** 2) * dk))

def energy_1d(psi, k, V, dx):
    psi_k = _fft.fft(psi) * dx / np.sqrt(2 * np.pi)
    dk    = 2 * np.pi / (len(psi) * dx)
    KE = float(np.real(np.sum(k ** 2 / 2 * np.abs(psi_k) ** 2) * dk))
    PE = float(np.real(np.sum(V * np.abs(psi) ** 2) * dx))
    return KE + PE

def transmission_1d(psi, x, dx):
    return float(np.sum(np.abs(psi[x > 0]) ** 2) * dx)


# ── 2-D ───────────────────────────────────────────────────────────────────

def norm_2d(psi, dx, dy):
    return float(np.sum(np.abs(psi) ** 2) * dx * dy)

def expect_x_2d(psi, X, dx, dy):
    return float(np.real(np.sum(X * np.abs(psi) ** 2) * dx * dy))

def expect_y_2d(psi, Y, dx, dy):
    return float(np.real(np.sum(Y * np.abs(psi) ** 2) * dx * dy))

def expect_px_2d(psi, KX, dx, dy):
    Nx, Ny = psi.shape
    psi_k  = _fft.fft2(psi) * dx * dy / (2 * np.pi)
    dkx, dky = 2 * np.pi / (Nx * dx), 2 * np.pi / (Ny * dy)
    return float(np.real(np.sum(KX * np.abs(psi_k) ** 2) * dkx * dky))

def energy_2d(psi, KX, KY, V, dx, dy):
    Nx, Ny = psi.shape
    psi_k  = _fft.fft2(psi) * dx * dy / (2 * np.pi)
    dkx, dky = 2 * np.pi / (Nx * dx), 2 * np.pi / (Ny * dy)
    KE = float(np.real(np.sum((KX**2 + KY**2) / 2 * np.abs(psi_k)**2) * dkx * dky))
    PE = float(np.real(np.sum(V * np.abs(psi) ** 2) * dx * dy))
    return KE + PE

def transmitted_prob_2d(psi, x, dx, dy):
    return float(np.sum(np.abs(psi[x > 0, :]) ** 2) * dx * dy)
