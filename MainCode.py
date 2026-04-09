import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import fft

# ─────────────────────────────────────────
#  §1  PARAMETERS  (atomic units: ℏ = m = 1)
# ─────────────────────────────────────────
class Params:
    # Spatial grid
    N             = 512               # number of grid points (power of 2 for FFT)
    L             = 20.0              # simulation box length [a.u.] (absorbing boundaries outside ±L/2)
    dx            = L / N             # spatial step

    # Time integration
    Nsteps        = 100               # total number of steps
    T             = 5.0               # maximum simulation time [a.u.]
    dt            = T / Nsteps        # time step

    # Initial Gaussian wave packet  ψ₀ = A·exp(-(x-x0)²/4σ²)·exp(ik₀x)
    x0            = 0.0              # initial position
    sigma         = 1.0               # spatial width (position uncertainty)
    k0            = 3.0               # central wave vector (mean momentum)

    # Potential selection: 'free' | 'barrier' | 'box' | 'harmonic' | 'double_well'
    potential     = 'harmonic'

    # Barrier
    V0            = 10.0              # barrier height [a.u.]
    barrier_width = 1.0               # barrier width [a.u.] (centered at origin)

    # Box
    box_width     = 18.0              # box potential width (hard walls at ±box_width/2)

    # Harmonic oscillator  V = ½mω²x² (m=1)
    omega         = 1.0               # angular frequency

    # Quartic double-well  V = a x⁴ - b x²
    a             = 0.05
    b             = 2.0


p = Params()


# ─────────────────────────────────────────
#  §2  GRID SETUP
# ─────────────────────────────────────────
x = np.linspace(-p.L/2, p.L/2, p.N, endpoint=False)   # position grid
# Wave-vector grid (fftfreq gives frequencies in cycles/sample; multiply by 2π/dx)
k = 2 * np.pi * fft.fftfreq(p.N, d=p.dx)              # momentum grid (rad/a.u.)


# ─────────────────────────────────────────
#  §3  POTENTIAL DEFINITIONS
# ─────────────────────────────────────────
def make_potential(x, params):
    """Return V(x) array for the selected potential type."""
    V = np.zeros_like(x)
    name = params.potential

    if name == 'free':
        pass                                               # V = 0 everywhere

    elif name == 'barrier':
        mask = np.abs(x) < params.barrier_width / 2        # rectangular barrier centred at x=0
        V[mask] = params.V0

    elif name == 'box':
        V[x < -params.box_width/2] = 1e6                   # hard walls
        V[x >  params.box_width/2] = 1e6

    elif name == 'harmonic':
        V = 0.5 * params.omega**2 * x**2                   # ½mω²x² (m=1)

    elif name == 'double_well':
        V = params.a * x**4 - params.b * x**2              # V = a x⁴ - b x²

    else:
        raise ValueError(f"Unknown potential: {name}")

    return V


V = make_potential(x, p)


# ─────────────────────────────────────────
#  §4  INITIAL WAVE PACKET  (Gaussian coherent state)
# ─────────────────────────────────────────
def gaussian_packet(x, x0, sigma, k0):
    # Normalized Gaussian wave packet  ψ(x,0) = (2πσ²)^{-1/4} exp(-(x-x0)²/4σ²) exp(ik₀x)
    norm = (2 * np.pi * sigma**2)**(-0.25)
    envelope = np.exp(-(x - x0)**2 / (4 * sigma**2))
    phase = np.exp(1j * k0 * x)
    return norm * envelope * phase


psi = gaussian_packet(x, p.x0, p.sigma, p.k0).astype(np.complex128)


# Absorbing boundary layer (complex absorbing potential / CAP)
# Prevents reflections from the edges of the simulation box
def absorbing_mask(x, L, width=3.0, strength=0.02):
    """Cosine-shaped absorbing mask near box edges."""
    mask = np.ones(len(x))
    for edge in [-L/2, L/2]:
        dist = np.abs(x - edge)
        absorb = dist < width
        mask[absorb] *= np.cos(np.pi * (1 - dist[absorb]/width) / 2) ** strength
    return mask


cap = absorbing_mask(x, p.L, width=0.0)


# ─────────────────────────────────────────
#  §5  PRECOMPUTE PROPAGATORS  (split-step phases)
# ─────────────────────────────────────────
# Half-step potential propagator  exp(-i V Δt/2)  [position space]
exp_V_half = np.exp(-1j * V * p.dt / 2)               # ℏ = 1

# Full-step kinetic propagator  exp(-i ℏk²Δt/2m)  [momentum space, ℏ=m=1]
exp_T_full = np.exp(-1j * (k**2 / 2) * p.dt)


# ─────────────────────────────────────────
#  §6  SPLIT-STEP FFT PROPAGATION  (Strang splitting)
#
#   ψ(t+Δt) ≈ exp(-iV·Δt/2) · IFFT[exp(-iK·Δt) · FFT[exp(-iV·Δt/2) · ψ(t)]]
# ─────────────────────────────────────────
def step(psi, exp_V_half, exp_T_full, cap):
    """One Strang split-step propagation."""
    psi *= exp_V_half             # ① Half-step potential (position space)
    psi_k = fft.fft(psi)          # ② FFT → momentum space
    psi_k *= exp_T_full           # ③ Full kinetic step (momentum space)
    psi = fft.ifft(psi_k)         # ④ IFFT → position space
    psi *= exp_V_half             # ⑤ Half-step potential (position space)
    psi *= cap                    # ⑥ Apply absorbing boundary
    return psi


# ─────────────────────────────────────────
#  §7  OBSERVABLES
# ─────────────────────────────────────────
def norm(psi, dx):
    """Total probability (should remain ≈ 1)."""
    return np.sum(np.abs(psi)**2) * dx

def expect_x(psi, x, dx):
    """⟨x⟩ — mean position."""
    return np.real(np.sum(x * np.abs(psi)**2) * dx)

def expect_p(psi, k, dx):
    """⟨p⟩ — mean momentum (in ℏk space, ℏ=1)."""
    psi_k = fft.fft(psi) * dx / np.sqrt(2 * np.pi)
    dk = 2 * np.pi / (len(k) * dx)
    return np.real(np.sum(k * np.abs(psi_k)**2) * dk)

def energy(psi, k, V, dx):
    """Total energy ⟨H⟩ = ⟨T⟩ + ⟨V⟩."""
    psi_k = fft.fft(psi) * dx / np.sqrt(2 * np.pi)
    dk = 2 * np.pi / (len(k) * dx)
    KE = np.real(np.sum((k**2/2) * np.abs(psi_k)**2) * dk)
    PE = np.real(np.sum(V * np.abs(psi)**2) * dx)
    return KE + PE

def transmission(psi, x, dx):
    """Transmission coefficient T = ∫_{x>0} |ψ|² dx."""
    return np.sum(np.abs(psi[x > 0])**2) * dx


# ─────────────────────────────────────────
#  §8  MAIN SIMULATION LOOP
# ─────────────────────────────────────────
t_arr, norm_arr, x_arr, p_arr, E_arr = [], [], [], [], []
snapshots = []
save_every = 5                     # save a frame every N steps

t = 0.0
E0 = energy(psi, k, V, p.dx)       # reference energy at t=0

for n in range(p.Nsteps):
    psi = step(psi, exp_V_half, exp_T_full, cap)
    t += p.dt

    if n % save_every == 0:
        t_arr.append(t)
        norm_arr.append(norm(psi, p.dx))
        x_arr.append(expect_x(psi, x, p.dx))
        p_arr.append(expect_p(psi, k, p.dx))
        E_arr.append(energy(psi, k, V, p.dx))
        snapshots.append(np.abs(psi)**2)

# Final transmission / reflection
T_coeff = transmission(psi, x, p.dx)
R_coeff = 1.0 - T_coeff
print(f"T = {T_coeff:.4f},  R = {R_coeff:.4f},  T/R = {T_coeff/R_coeff:.6f}")
print(f"Energy drift: ΔE/E₀ = {abs(E_arr[-1]-E0)/abs(E0):.2e}")


# ─────────────────────────────────────────
#  §9  VISUALIZATION
# ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Panel A: Probability density animation
ax = axes[0, 0]
ax.set_title('Probability Density |ψ(x,t)|²')
ax.set_xlabel('x [a.u.]')
prob_line, = ax.plot(x, snapshots[0], lw=1.5)
ax.fill_between(x, snapshots[0], alpha=0.2)
ax2 = ax.twinx()
ax2.fill_between(x, V / V.max() * np.max(snapshots[0]), alpha=0.25)
ax2.set_yticks([])
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.set_xlim(x[0], x[-1])

# Panel B: Expectation values ⟨x⟩ and ⟨p⟩
ax = axes[0, 1]
ax.set_title('Expectation Values')
ax.plot(t_arr, x_arr, lw=1.5, label='⟨x⟩')
ax.plot(t_arr, p_arr, lw=1.5, label='⟨p⟩', linestyle='--')
ax.set_xlabel('t [a.u.]')
ax.legend()

# Panel C: Normalization drift
ax = axes[1, 0]
ax.set_title('Normalization ∫|ψ|²dx')
ax.plot(t_arr, norm_arr, lw=1.5)
ax.axhline(1.0, lw=0.8, linestyle='--', label='exact')
ax.set_xlabel('t [a.u.]')
ax.legend()

# Panel D: Energy conservation
ax = axes[1, 1]
ax.set_title('Total Energy ⟨H⟩')
ax.plot(t_arr, E_arr, lw=1.5)
ax.axhline(E0, lw=0.8, linestyle='--', label='E₀')
ax.set_xlabel('t [a.u.]')
ax.legend()

plt.tight_layout(pad=2.0)

# Animation update function
def update(frame):
    prob_line.set_ydata(snapshots[frame])
    time_text.set_text(f't = {t_arr[frame]:.2f} a.u.')
    return prob_line, time_text

ani = animation.FuncAnimation(
    fig, update, frames=len(snapshots), interval=40, blit=True
)
ani.save('wavepacket_evolution.gif', writer='pillow', fps=25, dpi=120)
plt.show()
