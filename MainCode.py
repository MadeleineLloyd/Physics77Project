import numpy as np
from scipy import fft
import matplotlib.pyplot as plt

# Position space (1D)
# takes number of points, xmin and xmax and returns x list, fourier transformed frequiences as k, and the spacing 
def assemble_grid (nx, xmin,xmax):
    dx = (xmax - xmin) / nx
    x = np.linspace(xmin, xmax, nx, endpoint=False)
    k = fft.fftfreq(nx,dx) * 2 *np.pi
    dk = 2 * np.pi/(nx*dx)
    return x, k, dx, dk

# Time evolution - takes total number of times and final time, returns full list of t to use and delta t
def set_times (nt,tmax):
    dt = tmax/nt
    t = np.linspace(0,tmax,nt)
    return t, dt

#Define constants (For right now, setting hbar and m, which is mass of e, to 1)
hbar = 1
m = 1

#Initial Wave function
def Gaussian_Wavepacket (x0,x,sigma,k0): #Takes basic initial conditions and returns UNNORMALIZED initial wave function
    psi = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x) # e^-2(x-x0)/2(sig^2) * e^i factor = Gaussian wave packet * phase
    return psi

def normalize (psi,dx): #takes a wavefunction and delta x to properly normalize wf
    return psi / np.sqrt(np.sum(np.abs(psi)**2) * dx) # to normalize psi


# Different potentials
def free_potential(x):
    return np.zeros_like(x) #free particle potential

def barrier_potential(x, height=10,width=1): #square potential well - tunneling / transmission
    return height * (np.abs(x)< width)

def quantum_harmonic_oscillator(x,k=1.0): #everyone's favorite and the classic :D
    return 0.5*k*(x**2)

def gaussian_potential(x,height,sigma=1):
    return height * np.exp(-x**2 / (2*sigma**2))


def evolve(psi, U_half, T_full, nt, dx):
    psi_t = []

    for _ in range(nt):
        psi = U_half * psi # psi multiplied by half poten. energy operator
        psi_k = fft.fft(psi) # transform x to k 
        psi_k = T_full * psi_k # psi (now U_half * psi) multiplied by KE Unitary operator
        psi = fft.ifft(psi_k) # transform back to position space
        psi = U_half * psi # the second half of the potential

        psi = normalize(psi, dx)  # retain normalization of psi
        psi_t.append(np.abs(psi)**2) # inside parantheses is the probability amp. , stores for latter graphing

    return psi_t

###############

# Grid
nx = 256
xmin, xmax = -10, 10
x, k, dx, dk = assemble_grid(nx, xmin, xmax)

# Time
nt = 100
tmax = 5.0
t, dt = set_times(nt, tmax)

# Initial wavefunction
x0 = -5.0
sigma = 1.0
k0 = 3.0

psi = Gaussian_Wavepacket(x0, x, sigma, k0)
psi = normalize(psi, dx)

#Choose V
#V = barrier_potential(x, height=5, width=0.5)
#V = free_potential(x)
V = quantum_harmonic_oscillator(x)

U_half = np.exp(-1j * V * dt / (2 * hbar)) # potential energy evolution operator
T = (hbar * k)**2 / (2 * m) # kinetic operator (hk)^2/2m 
T_full = np.exp(-1j * T * dt / hbar) # unitary time evolution operator e^-i * KE * time ev/ hbar

# Run simulation
psi_t = evolve(psi, U_half, T_full, nt, dx)
print("Normalization is ", (np.sum(np.abs(psi)**2) * dx))
plt.plot(x, psi_t[0], label="t = 0")
#plt.plot(x,V,label='Potential')
plt.plot(x, psi_t[-1], label=f"Final State (t = {tmax})")

plt.xlabel("x")
plt.ylabel("|ψ|²")
plt.title("Wavepacket Evolution")
plt.xlim(-10,10)
plt.grid(alpha=0.3)
plt.legend()
plt.show()