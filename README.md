# Split-Step FFT Schrödinger Solver

A compact Python implementation of split-step Fourier methods for simulating quantum wavepacket dynamics in 1D and 2D.

## Features

- 1D time-dependent Schrödinger equation solver (`run_1d.py`)
- 2D time-dependent solver (`run_2d.py`)
- Exact 1D Gaussian wavepacket evolution for validation (`exact.py`)
- Convergence analysis for time-stepping and spatial resolution (`convergence.py`)
- Automatic CLI argument parsing from parameter dataclasses

## Requirements

- Python 3.10+ (or compatible)
- NumPy
- SciPy
- Matplotlib
- Pillow (for GIF output)

Install dependencies with pip:

```bash
python -m pip install numpy scipy matplotlib pillow
```

## Quick Start

From the repository root, run one of the main scripts:

```bash
python run_1d.py
python run_2d.py
python run_double_slit.py
python exact.py
python convergence.py
```

## Example Usage

Run a 1D driven harmonic simulation with a larger grid and no animation output:

```bash
python run_1d.py \
  --T=314.1593 \
  --Nsteps=300000 \
  --save_every=1000 \
  --L=10. \
  --N=2048 \
  --potential=driven_harmonic \
  --A=1. \
  --Omega=0.02 \
  --omega=2. \
  --x0=1. \
  --k0=0. \
  --sigma=0.5 \
  --no-save_anim
```

Run a 2D solver with a Gaussian wavepacket and a double-slit potential:

```bash
python run_2d.py \
  --potential=double_slit \
  --Nx=256 --Ny=256 \
  --Lx=20.0 --Ly=20.0 \
  --x0=-5.0 --y0=0.0 \
  --k0x=1.5 --k0y=0.0 \
  --sigma0=1.0 \
  --save_every=10
```

## CLI Reference

The scripts derive arguments from the `Params1D` and `Params2D` dataclasses in `params.py`. There are many supported options, including:

- `--T`, `--Nsteps`, `--save_every`
- `--potential`, `--V0`, `--barrier_width`, `--omega`, `--A`, `--Omega`
- `--initial`, `--sigma0`, `--x0`, `--k0`, `--n`
- `--cap_width`, `--cap_strength`
- `--order`
- `--save_anim` / `--no-save_anim`
- 1D-specific: `--N`, `--L`, `--a`
- 2D-specific: `--Nx`, `--Ny`, `--Lx`, `--Ly`, `--y0`, `--k0x`, `--k0y`, `--slit_sep`, `--slit_width`, `--epsilon`

For a full list of supported arguments, use:

```bash
python run_1d.py --help
python run_2d.py --help
```

## Output

The solvers generate plots and save summary images. When animation is enabled, they also save `wavepacket_1d.gif` or `wavepacket_2d.gif`.
