import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import fft

# ─────────────────────────────────────────
#  §1  PARAMETERS  (atomic units: ℏ = m = 1)
# ─────────────────────────────────────────
class Params2D:
    # Spatial grid
    Nx, Ny        = 512, 512            # number of grid points (power of 2 for FFT)
    Lx, Ly        = 20.0, 20.0          # simulation box length [a.u.] (absorbing boundaries outside ±L/2)
    dx, dy        = Lx / Nx, Ly / Ny    # spatial step

    # Time integration
    Nsteps        = 100                 # total number of steps
    T             = 5.0                 # maximum simulation time [a.u.]
    dt            = T / Nsteps          # time step

    # Initial 2D Gaussian wave packet  ψ₀ = A·exp(-((x-x0)²+(y-y0)²)/4σ²)·exp(i(k0x·x + k0y·y))
    x0, y0        = -8.0, 0.0           # initial position
    sigma         = 1.5                 # isotropic spatial width
    k0x,k0y       = 6.0, 0.0            # central wave vector


    # Potential selection: 'free' | 'double_slit' | 'harmonic_2d'
    potential     = 'harmonic_2d'

    # Double-slit
    V0            = 50.0                # barrier height [a.u.]
    wall_thick    = 0.3                 # barrier thickness in x [a.u.]
    slit_sep      = 2.0                 # centre-to-centre slit separation [a.u.]
    slit_width    = 0.4                 # half-width of each slit opening [a.u.]

    # Harmonic oscillator (2D isotropic)
    omega         = 1.0                 # angular frequency


p = Params2D()


# ─────────────────────────────────────────
#  §2  GRID SETUP
# ─────────────────────────────────────────
x = np.linspace(-p.Lx/2, p.Lx/2, p.Nx, endpoint=False)      # x position grid
y = np.linspace(-p.Ly/2, p.Ly/2, p.Ny, endpoint=False)      # y position grid
X, Y = np.meshgrid(x, y, indexing='ij')                     # 2D coordinate arrays

# 2D momentum grids (fftfreq × 2π/step)
kx = 2 * np.pi * fft.fftfreq(p.Nx, d=p.dx)                  
ky = 2 * np.pi * fft.fftfreq(p.Nx, d=p.dy)                  
KX, KY = np.meshgrid(kx, ky, indexing='ij')                 # 2D momentum arrays


# ─────────────────────────────────────────
#  §3  POTENTIAL DEFINITIONS
# ─────────────────────────────────────────
def make_potential(X, Y, params):
    """Return V(x, y) array for the selected potential type."""
    V = np.zeros_like(X)
    name = params.potential

    if name == 'free':
        pass                                                # V = 0 everywhere

    elif name == 'double_slit':
        # Thin wall at x=0, two openings at y = ±slit_sep/2
        in_wall = np.abs(X) < params.wall_thick / 2
        slit1   = np.abs(Y -  params.slit_sep / 2) < params.slit_width / 2
        slit2   = np.abs(Y +  params.slit_sep / 2) < params.slit_width / 2
        V[in_wall & ~slit1 & ~slit2] = params.V0

    elif name == 'harmonic_2d':
        V = 0.5 * params.omega**2 * (X**2 + Y**2)           # isotropic SHO

    else:
        raise ValueError(f"Unknown potential: {name}")

    return V

V = make_potential(X, Y, p)


# ─────────────────────────────────────────
#  §4  INITIAL WAVE PACKET  (Gaussian coherent state)
# ─────────────────────────────────────────
def gaussian_packet(X, Y, x0, y0, sigma, k0x, k0y):
    # Normalised 2D Gaussian wave packet  ψ(x,y,0) = (2πσ²)^{-1/2} exp(-((x-x0)²+(y-y0)²)/4σ²) exp(i(k0x·x+k0y·y))
    norm     = (2 * np.pi * sigma**2)**(-0.5)
    envelope = np.exp(-((X - x0)**2 + (Y - y0)**2) / (4 * sigma**2))
    phase    = np.exp(1j * (k0x * X + k0y * Y))
    return norm * envelope * phase


psi = gaussian_packet(X, Y, p.x0, p.y0, p.sigma, p.k0x, p.k0y).astype(np.complex128)


# Absorbing boundary layer (complex absorbing potential / CAP)
# Prevents reflections from the edges of the simulation box
def absorbing_mask(x, y, Lx, Ly, width=2.0, strength=0.02):
    """Separable cosine absorbing mask: cap(x,y) = cap_x(x) · cap_y(y)."""
    def _1d_mask(coords, L):
        mask = np.ones(len(coords))
        for edge in [-L/2, L/2]:
            dist   = np.abs(coords - edge)
            absorb = dist < width
            mask[absorb] *= np.cos(np.pi * (1 - dist[absorb]/width) / 2) ** strength
        return mask
    cap_x = _1d_mask(x, Lx)
    cap_y = _1d_mask(y, Ly)
    return np.outer(cap_x, cap_y)  


# ─────────────────────────────────────────
#  §5  PRECOMPUTE PROPAGATORS  (split-step phases)
# ─────────────────────────────────────────
# Half-step potential propagator  exp(-i V Δt/2)  [position space, shape Nx×Ny]
exp_V_half = np.exp(-1j * V * p.dt / 2)

# Full-step kinetic propagator  exp(-i ℏk²Δt/2m)  [momentum space, ℏ=m=1]
exp_T_full = np.exp(-1j * (KX**2 + KY**2) / 2 * p.dt)


# ─────────────────────────────────────────
#  §6  SPLIT-STEP FFT PROPAGATION  (Strang splitting)
#
#   ψ(t+Δt) ≈ exp(-iV·Δt/2) · IFFT[exp(-iK·Δt) · FFT[exp(-iV·Δt/2) · ψ(t)]]
# ─────────────────────────────────────────
def step(psi, exp_V_half, exp_T_full, cap=1):
    """One Strang split-step propagation in 2D."""
    psi *= exp_V_half               # ① Half-step potential (position space)
    psi_k = fft.fft2(psi)           # ② FFT → momentum space
    psi_k *= exp_T_full             # ③ Full kinetic step (momentum space)
    psi = fft.ifft2(psi_k)          # ④ IFFT → position space
    psi *= exp_V_half               # ⑤ Half-step potential (position space)
    psi *= cap                      # ⑥ Apply absorbing boundary
    return psi


# ─────────────────────────────────────────
#  §7  OBSERVABLES
# ─────────────────────────────────────────
def norm(psi, dx, dy):
    """Total probability ∬|ψ|² dx dy  (should remain ≈ 1)."""
    return np.sum(np.abs(psi)**2) * dx * dy

def expect_x(psi, X, dx, dy):
    """⟨x⟩ — mean x-position."""
    return np.real(np.sum(X * np.abs(psi)**2) * dx * dy)

def expect_y(psi, Y, dx, dy):
    """⟨y⟩ — mean y-position."""
    return np.real(np.sum(Y * np.abs(psi)**2) * dx * dy)

def expect_px(psi, KX, dx, dy):
    """⟨px⟩ — mean x-momentum (ℏ=1)."""
    norm_k = dx * dy / (2 * np.pi)
    psi_k  = fft.fft2(psi) * norm_k
    dkx    = 2 * np.pi / (psi.shape[0] * dx)
    dky    = 2 * np.pi / (psi.shape[1] * dy)
    return np.real(np.sum(KX * np.abs(psi_k)**2) * dkx * dky)

def energy(psi, KX, KY, V, dx, dy):
    """Total energy ⟨H⟩ = ⟨T⟩ + ⟨V⟩."""
    norm_k = dx * dy / (2 * np.pi)
    psi_k  = fft.fft2(psi) * norm_k
    dkx    = 2 * np.pi / (psi.shape[0] * dx)
    dky    = 2 * np.pi / (psi.shape[1] * dy)
    KE = np.real(np.sum((KX**2 + KY**2) / 2 * np.abs(psi_k)**2) * dkx * dky)
    PE = np.real(np.sum(V * np.abs(psi)**2) * dx * dy)
    return KE + PE

def transmitted_prob(psi, x, dx, dy):
    """Fraction of probability with x > 0  (transmission through y-aligned wall)."""
    return np.sum(np.abs(psi[x > 0, :])**2) * dx * dy


# ─────────────────────────────────────────
#  §8  MAIN SIMULATION LOOP
# ─────────────────────────────────────────
t_arr, norm_arr, x_arr, y_arr, E_arr = [], [], [], [], []
snapshots  = []
save_every = 5                      # save a frame every N steps

t  = 0.0
E0 = energy(psi, KX, KY, V, p.dx, p.dy)     # reference energy at t=0

for n in range(p.Nsteps):
    psi = step(psi, exp_V_half, exp_T_full)
    t  += p.dt

    if n % save_every == 0:
        t_arr.append(t)
        norm_arr.append(norm(psi, p.dx, p.dy))
        x_arr.append(expect_x(psi, X, p.dx, p.dy))
        y_arr.append(expect_y(psi, Y, p.dx, p.dy))
        E_arr.append(energy(psi, KX, KY, V, p.dx, p.dy))
        snapshots.append(np.abs(psi)**2)

# Final diagnostics
T_coeff = transmitted_prob(psi, x, p.dx, p.dy)
print(f"Transmitted fraction  = {T_coeff:.4f}")
print(f"Energy drift: ΔE/E₀  = {abs(E_arr[-1]-E0)/abs(E0):.2e}")


# ─────────────────────────────────────────
#  §9  VISUALIZATION
# ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Panel A: Probability density animation
ax = axes[0, 0]
ax.set_title('Probability Density |ψ(x,t)|²')
ax.set_xlabel('x [a.u.]')
ax.set_ylabel('y [a.u.]')
im = ax.imshow(snapshots[0].T, origin='lower', cmap='inferno', aspect='equal', animated=True)
ax.contour(x, y, V.T, levels=[p.V0 / 2], linewidths=0.8)
time_text = ax.text(0.02, 0.96, '', transform=ax.transAxes, va='top')

# Panel B: Expectation values ⟨x⟩ and ⟨p⟩
ax = axes[0, 1]
ax.set_title('Expectation Values')
ax.plot(t_arr, x_arr, lw=1.5, label='⟨x⟩')
ax.plot(t_arr, y_arr, lw=1.5, label='⟨y⟩', linestyle='--')
ax.set_xlabel('t [a.u.]')
ax.legend()

# Panel C: Normalization drift
ax = axes[1, 0]
ax.set_title('Normalization ∬|ψ|² dx dy')
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
    im.set_data(snapshots[frame].T)
    im.set_clim(vmax=np.max(snapshots[frame]))          # rescale colour range each frame
    time_text.set_text(f't = {t_arr[frame]:.2f} a.u.')
    return im, time_text

ani = animation.FuncAnimation(
    fig, update, frames=len(snapshots), interval=40, blit=True
)
ani.save('wavepacket_evolution.gif', writer='pillow', fps=25, dpi=120)
plt.show()
