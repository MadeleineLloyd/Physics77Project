import numpy as np
from scipy import fft
import matplotlib.pyplot as plt

# Position space (1D)
# takes number of points, xmin and xmax and returns x list, fourier transformed frequiences as k, and the spacing 
def assemble_grid(nx, xmin,xmax):
    dx = (xmax - xmin) / nx
    x = np.linspace(xmin, xmax, nx, endpoint=False)
    k = fft.fftfreq(nx,dx) * 2 *np.pi
    dk = 2 * np.pi/(nx*dx)
    return x, k, dx, dk

# Time evolution - takes total number of times and final time, returns full list of t to use and delta t
def set_times(nt,tmax):
    dt = tmax/nt
    t = np.linspace(0,tmax,nt)
    return t, dt

#Define constants (For right now, setting hbar and m, which is mass of e, to 1)
hbar = 1
m = 1

#Initial Wave function
def Gaussian_Wavepacket(x0,x,sigma,k0): #Takes basic initial conditions and returns UNNORMALIZED initial wave function
    psi = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x) # e^-2(x-x0)/2(sig^2) * e^i factor = Gaussian wave packet * phase
    return psi

def normalize(psi,dx): #takes a wavefunction and delta x to properly normalize wf
    return psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx) # to normalize psi


# Different potentials
V_free = np.zeros_like(x) #free particle potential

U_half = np.exp(-1j * V_free * dt / (2 * hbar)) # potential energy evolution operator
T = (hbar * k)**2 / (2 * m) # kinetic operator (hk)^2/2m 
T_full = np.exp(-1j * T * dt / hbar) # unitary time evolution operator e^-i * KE * time ev/ hbar
psi_t = []

for _ in range(nt):
    psi = U_half * psi # psi multiplied by half poten. energy operator
    psi_k = fft.fft(psi) # transform x to k 
    psi_k = T_full * psi_k # psi (now U_half * psi) multiplied by KE Unitary operator
    psi = fft.ifft(psi_k) # transform back to position space
    psi = U_half * psi # the second half of the potential
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx) # retain normalization of psi
    psi_t.append(np.abs(psi)**2) # inside parantheses is the probability amp. , stores for latter graphing

plt.plot(x, psi_t[0], label="t = 0", color = 'Red')
plt.plot(x, psi_t[-1], label=f"Final State (t = {tmax}) with initial momentum (k ={k0})",color='Blue')

plt.xlabel("x")
plt.ylabel("|ψ|²")
plt.title("Wavepacket Evolution (Free Particle)")
plt.xlim(-10,10)
plt.grid(alpha=0.3)
plt.legend()
plt.show()
