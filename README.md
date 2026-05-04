# Quantum Wavepacket Simulator

A Python toolkit for simulating quantum wavepacket dynamics in 1D and 2D using the **split-step Fourier (SSF)** method. Supports multiple potentials, split-step integrators of order 1/2/4, absorbing boundary conditions, and observable tracking.

---

## Features

- **1D and 2D** split-step FFT propagation
- **Three integrator orders**: Lie–Trotter (1st), Strang splitting (2nd), Yoshida (4th)
- **Multiple potentials**: free, harmonic, barrier, box, Bragg grating, double slit, driven harmonic, ring
- **Absorbing boundary layer** (cosine-CAP) to suppress reflections
- **Observable tracking**: norm, ⟨x⟩, ⟨p⟩, ⟨H⟩, transmission coefficient
- **Exact reference solutions** for free Gaussian and coherent-state (harmonic) packets
- **Convergence analysis** in Δt and Δx with automatic slope fitting
- **Animation export** to GIF, summary plots to PNG
- **Berry phase extraction** for cyclic driven-harmonic evolution

---

## Project Structure

```
.
├── params.py          # Dataclass parameter containers (Params1D, Params2D)
├── grids.py           # Coordinate and k-space grid construction, CAP mask
├── potentials.py      # All 1D/2D potential builders (including time-dependent)
├── initial.py         # Initial wavefunction factories (Gaussian, plane wave, eigenstate)
├── propagator.py      # Lie, Strang, and Yoshida split-step kernels
├── observables.py     # Norm, expectation values, energy, transmission
├── exact.py           # Analytic solutions: free packet, coherent state
├── convergence.py     # Δt and Δx convergence studies with plots
├── run_1d.py          # 1D simulation driver + plotting/animation
├── run_2d.py          # 2D simulation driver + plotting/animation
└── __init__.py        # Package-level imports
```

---

## Installation

Requires Python ≥ 3.9.

```bash
pip install numpy scipy matplotlib
```

No other dependencies are needed.

---

## Quick Start

### 1D — Gaussian packet tunnelling through a barrier

```bash
python run_1d.py --T=7.5 --Nsteps=1600 --N=4096 --L=50.0 --x0=-10.0 --k0=3.0 --sigma=3.0 --potential=barrier --V0=5.0 --barrier_width=1.0 --smooth=0.05
```

### 1D — Coherent state in a harmonic trap

```bash
python run_1d.py --T=6.2832 --Nsteps=1200 --x0=-5.0 --k0=0.0 --sigma=0.5 --potential=harmonic --omega=2.0
```

### 1D — Multi-packet Superposition

```bash
python run_1d.py --T=5.0 --x0=-5.0,5.0 --k0=1.5,-1.5 --sigma0=1.0,1.0 --potential=free
```

### 2D — Double-slit diffraction

```bash
python run_2d.py --T=4. --Nsteps=800 --Lx=100. --Ly=100. --Nx=1024 --Ny=1024 --x0=-12. --y0=0. --k0x=6. --k0y=0. --sigma0=3. --potential=double_slit --V0=50 --barrier_width=2. --smooth=0.05 --slit_sep=3. --slit_width=0.4
```

### Convergence study

```bash
python convergence.py
```

Produces log–log and semilogy plots of L² error vs Δt and Δx for orders 1, 2, and 4, and prints fitted convergence slopes.

---

## Parameters

All parameters are exposed as CLI flags (via `argparse`) and as fields of `Params1D` / `Params2D`.

### Shared parameters

| Parameter | Default | Description |
|---|---|---|
| `T` | `5.0` | Total simulation time [a.u.] |
| `Nsteps` | `1000` | Number of time steps |
| `save_every` | `8` | Save snapshot every N steps |
| `order` | `2` | Integrator order: `1`, `2`, or `4` |
| `potential` | `free` | Potential name (see below) |
| `V0` | `2.0` | Potential barrier height |
| `barrier_width` | `1.0` | Potential barrier width |
| `smooth` | `0.0` | Potential smoothness (0 = sharp edges) |
| `omega` | `1.0` | Harmonic frequency |
| `sigma0` | `(1.0,)` | Initial packet width(s) |
| `cap_width` | `0.0` | CAP absorber width (0 = disabled) |
| `cap_strength` | `0.0` | CAP cosine exponent (0 = disabled) |
| `save_anim` | `True` | Export animation GIF |

### 1D-specific (`Params1D`)

| Parameter | Default | Description |
|---|---|---|
| `N` | `256` | Number of grid points |
| `L` | `20.0` | Domain length |
| `x0` | `(-5.0,)` | Initial packet centre(s) |
| `k0` | `(1.5,)` | Initial momentum/momenta |

### 2D-specific (`Params2D`)

| Parameter | Default | Description |
|---|---|---|
| `Nx`, `Ny` | `256, 256` | Grid points along x, y |
| `Lx`, `Ly` | `20.0, 20.0` | Domain extents |
| `x0`, `y0` | `(-5.0,)`, `(0.0,)` | Packet centre(s) |
| `k0x`, `k0y` | `(1.5,)`, `(0.0,)` | Initial momenta |
| `slit_sep` | `2.0` | Double-slit separation |
| `slit_width` | `0.4` | Single slit opening width |
| `epsilon` | `1.0` | Anisotropy of harmonic trap (ωy² = ε·ωx²) |

---

## Available Potentials

| Name | Dimensions | Description |
|---|---|---|
| `free` | 1D / 2D | No potential |
| `harmonic` | 1D / 2D | Quadratic trap ½ω²r² |
| `driven_harmonic` | 1D / 2D | Trap centre orbits at amplitude A, frequency Ω |
| `barrier` | 1D / 2D | Smooth rectangular barrier of height V₀ |
| `box` | 1D | Infinite-well approximation |
| `bragg` | 1D | Cosine Bragg grating inside a barrier |
| `double_slit` | 2D | Two Gaussian-edge slits in a barrier wall |
| `ring` | 2D | Circular hard-wall confinement |

---

## Integrators

| Order | Scheme | Global error |
|---|---|---|
| 1 | Lie–Trotter | O(Δt) |
| 2 | Strang splitting | O(Δt²) |
| 4 | Yoshida (triple-Strang) | O(Δt⁴) |

All three conserve unitarity exactly (up to floating-point precision) for time-independent potentials without the CAP.

---

## Outputs

| File | Content |
|---|---|
| `summary_1d.png` | 2×2 panel: probability density animation frame, expectation values, norm, energy |
| `wavepacket_1d.gif` | Animated probability density with overlaid potential |
| `summary_2d.png` | 2×2 panel: 2D density heat-map, expectation values, norm, energy |
| `wavepacket_2d.gif` | Animated 2D density with potential contour overlay |

---

## Berry Phase (Driven Harmonic)

For a cyclic evolution of the driven harmonic trap (one full orbit, `T = 2π/Ω`), the code prints the geometric (Berry) phase extracted from the final overlap:

```
Geometric phase γ = +X.XXXX rad  (theory: ±2π A² ω rad)
```

> **Note on energy conservation**: the lab-frame energy ⟨H(t)⟩ oscillates in this experiment because the Hamiltonian is explicitly time-dependent — the external drive continuously exchanges energy with the system. The quantity that is adiabatically conserved is the *instantaneous* eigenenergy ħω(n + ½) in the co-moving frame, not the lab-frame expectation value. Use the analytic eigenvalue rather than the integrated ⟨H(t)⟩ when computing the dynamical phase, to avoid accumulated error over the long adiabatic period T = 2π/Ω.
