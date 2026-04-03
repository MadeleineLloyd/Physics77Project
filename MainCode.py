import numpy as np
from scipy import fft
import matplotlib.pyplot as plt

# Position space (1D)
nx = 256
xmin, xmax = -10, 10
dx = (xmax - xmin) / nx
x = np.linspace(xmin, xmax, nx, endpoint=False)

# Momentum space (1D)
dk = 2 * np.pi / (nx * dx)
k = fft.fftfreq(nx, dx) * 2 * np.pi

# Time evolution
nt = 100
T = 10.0
dt = T / nt
t = np.linspace(0, T, nt)
