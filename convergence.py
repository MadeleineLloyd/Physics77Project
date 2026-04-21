"""
convergence.py
──────────────
Numerical convergence study for the three split-step FFT integrators.

Demonstrates that the global temporal error of each scheme decreases
with the expected power of Δt:

    order 1  (Lie)     →  error ∝ Δt¹
    order 2  (Strang)  →  error ∝ Δt²
    order 4  (Yoshida) →  error ∝ Δt⁴

Physical setup
--------------
1-D harmonic oscillator, Gaussian coherent-state initial condition.
The analytic solution is:

    ψ_exact(x, t) = (2πσ²)^{-1/4}
                    · exp(−(x − x_c(t))² / 4σ²)
                    · exp(i (k_c(t) x − φ(t)))

where the centre follows the classical trajectory

    x_c(t) = x₀ cos(ωt) + (k₀/ω) sin(ωt)
    k_c(t) = k₀ cos(ωt) − x₀ ω  sin(ωt)

and φ(t) is the accumulated dynamical phase.
The coherent-state width σ = 1/√(2ω) is constant in time.

Error metric
------------
We use the phase-corrected L² norm:

    err(Δt) = min_θ ‖ψ_num(T) − e^{iθ} ψ_exact(T)‖₂ · √dx
            = √( ∫ |ψ_num/e^{iθ*} − ψ_exact|² dx )

where the optimal phase θ* = arg⟨ψ_exact|ψ_num⟩ is computed analytically.
This removes the trivial global phase ambiguity while preserving full
sensitivity to local (physically meaningful) errors in the wave packet.

Usage
-----
    python convergence.py                     # standard run
    python convergence.py --T_end 2.0         # longer horizon
    python convergence.py --omega 2.0         # faster oscillator
    python convergence.py --no-plot           # table only
    python convergence.py --dt_coarse 0.10 --dt_fine 0.002 --n_dt 9
"""

import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import fft

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from params      import Params1D
from grids       import make_grid_1d, absorbing_mask_1d
from potentials  import make_potential_1d
from propagator  import make_stepper


# ══════════════════════════════════════════════════════════════════════
#  Analytic solution — SHO coherent state
# ══════════════════════════════════════════════════════════════════════

def psi_exact_sho(x: np.ndarray, t: float,
                  x0: float, k0: float,
                  sigma: float, omega: float) -> np.ndarray:
    """
    Exact wave function of a Gaussian coherent state in V = ½ω²x²
    (ℏ = m = 1).

    Parameters
    ----------
    x        : position grid
    t        : evaluation time
    x0, k0   : initial centre position and wave vector
    sigma    : coherent-state width  (= 1/√(2ω) for the ground state)
    omega    : oscillator frequency
    """
    x_c = x0 * np.cos(omega * t) + (k0 / omega) * np.sin(omega * t)
    k_c = k0 * np.cos(omega * t) - x0 * omega  * np.sin(omega * t)

    # Accumulated dynamical phase (energy + geometric Berry contribution)
    E   = 0.5 * (k0 ** 2 + omega ** 2 * x0 ** 2) + 0.5 * omega
    phi = E * t - 0.5 * (x_c * k_c - x0 * k0)

    norm     = (2 * np.pi * sigma ** 2) ** (-0.25)
    envelope = np.exp(-(x - x_c) ** 2 / (4 * sigma ** 2))
    phase    = np.exp(1j * (k_c * x - phi))
    return (norm * envelope * phase).astype(np.complex128)


# ══════════════════════════════════════════════════════════════════════
#  Single-run error measurement
# ══════════════════════════════════════════════════════════════════════

def measure_error(order: int, dt: float, T_end: float,
                  N: int, L: float,
                  x0: float, k0: float,
                  sigma: float, omega: float,
                  cap_width: float) -> float:
    """
    Propagate the SHO coherent state to T_end and return the
    phase-corrected L² error against the analytic solution.

    Parameters
    ----------
    order     : integrator order (1, 2, or 4)
    dt        : time step
    T_end     : target integration time (actual time = Nsteps × dt)
    N, L      : grid size and box length
    x0, k0    : initial wave-packet centre and wave vector
    sigma     : wave-packet width
    omega     : SHO frequency
    cap_width : absorbing layer width (keep narrow to avoid interfering)
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = Params1D(
            N=N, L=L, dt=dt, Nsteps=int(round(T_end / dt)),
            x0=x0, sigma=sigma, k0=k0,
            potential='harmonic', omega=omega,
            order=order,
            cap_width=cap_width, cap_strength=0.003,
        )

    x, k   = make_grid_1d(p)
    V      = make_potential_1d(x, p)
    cap    = absorbing_mask_1d(x, p)
    K2     = k ** 2

    psi  = psi_exact_sho(x, 0.0, x0, k0, sigma, omega)
    step = make_stepper(p, V, K2, cap, fft.fft, fft.ifft)

    for _ in range(p.Nsteps):
        psi = step(psi)

    t_actual = p.Nsteps * dt
    ref = psi_exact_sho(x, t_actual, x0, k0, sigma, omega)

    # Phase-corrected L² error:  remove the optimal global phase e^{iθ*}
    # that minimises ‖ψ_num/e^{iθ} − ψ_ref‖ before computing the norm.
    overlap = np.sum(np.conj(ref) * psi) * p.dx   # ⟨ψ_ref | ψ_num⟩
    phase_factor = overlap / abs(overlap)          # e^{iθ*}
    return float(np.sqrt(np.sum(np.abs(psi / phase_factor - ref) ** 2) * p.dx))


# ══════════════════════════════════════════════════════════════════════
#  Convergence sweep
# ══════════════════════════════════════════════════════════════════════

def run_convergence(
    T_end:     float = 0.5,
    omega:     float = 1.0,
    x0:        float = 3.0,
    k0:        float = 2.0,    # non-zero so the packet moves, richer phase
    N:         int   = 2048,
    L:         float = 30.0,
    n_dt:      int   = 8,
    dt_coarse: float = 0.10,
    dt_fine:   float = 0.003,
) -> dict:
    """
    Sweep Δt for orders 1, 2, 4 and collect phase-corrected L² errors.

    Returns
    -------
    dict with keys:
        dt_vals : (n_dt,) array of time steps
        errors  : (3, n_dt) array, rows = [order1, order2, order4]
        slopes  : list of 3 fitted log-log slopes
        orders  : [1, 2, 4]
        T_end, omega : for labelling
    """
    sigma   = 1.0 / np.sqrt(2 * omega)
    cap_w   = min(2.0, L / 12)
    orders  = [1, 2, 4]
    dt_vals = np.geomspace(dt_coarse, dt_fine, n_dt)
    errors  = np.zeros((len(orders), n_dt))

    for i, order in enumerate(orders):
        label = {1: 'Lie    (order 1)', 2: 'Strang (order 2)',
                 4: 'Yoshida(order 4)'}[order]
        print(f"  {label}: ", end='', flush=True)
        for j, dt in enumerate(dt_vals):
            errors[i, j] = measure_error(
                order, dt, T_end, N, L, x0, k0, sigma, omega, cap_w)
            print('.', end='', flush=True)
        print()

    # Fit slopes on the finer half of the Δt range
    mid = n_dt // 2
    slopes = []
    for i in range(len(orders)):
        ldt   = np.log10(dt_vals[mid:])
        ler   = np.log10(errors[i, mid:])
        valid = np.isfinite(ler)
        if valid.sum() >= 2:
            slope, _ = np.polyfit(ldt[valid], ler[valid], 1)
        else:
            slope = float('nan')
        slopes.append(slope)

    return dict(dt_vals=dt_vals, errors=errors, slopes=slopes,
                orders=orders, T_end=T_end, omega=omega, k0=k0)


# ══════════════════════════════════════════════════════════════════════
#  Pretty table
# ══════════════════════════════════════════════════════════════════════

def print_table(res: dict) -> None:
    dt_vals = res['dt_vals']
    errors  = res['errors']
    orders  = res['orders']
    theory  = {1: 1, 2: 2, 4: 4}

    col = 18
    header = f"{'Δt':>10}" + "".join(f"{'order '+str(o):>{col}}" for o in orders)
    sep    = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for j, dt in enumerate(dt_vals):
        row = f"{dt:10.5f}" + "".join(f"{errors[i,j]:>{col}.4e}"
                                       for i in range(len(orders)))
        print(row)
    print(sep)
    print("Fitted log-log slopes (finer half of Δt range):")
    for order, slope in zip(orders, res['slopes']):
        th = theory[order]
        flag = '' if abs(slope - th) < 0.15 else '  ← check grid/CAP settings'
        print(f"  order {order}: slope = {slope:+.3f}  (theory {th:.0f}){flag}")
    print()


# ══════════════════════════════════════════════════════════════════════
#  Visualisation — two-panel figure
# ══════════════════════════════════════════════════════════════════════

def plot_convergence(res: dict) -> None:
    """
    Left panel  — log-log error vs Δt with reference slope guide-lines.
    Right panel — error / Δt^p, should plateau to a constant C when the
                  asymptotic regime is reached, confirming the order.
    """
    dt_vals = res['dt_vals']
    errors  = res['errors']
    slopes  = res['slopes']
    orders  = res['orders']

    COLORS  = ['#c87a5a', '#e8a840', '#7ab8a0']
    NAMES   = {1: 'Lie  (order 1)', 2: 'Strang  (order 2)',
               4: 'Yoshida  (order 4)'}
    THEORY  = {1: 1, 2: 2, 4: 4}
    DARK    = '#0e0c08'
    PAN     = '#141008'
    SP      = '#3a3020'
    TK      = '#7a6e5a'
    TX      = '#d4c9a8'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(DARK)
    fig.suptitle(
        f'Split-Step FFT  —  Convergence in Δt\n'
        f'SHO coherent state,  ω = {res["omega"]:.1f},  '
        f'k₀ = {res["k0"]:.1f},  T = {res["T_end"]:.2f} a.u.',
        color=TX, fontsize=12, y=1.01,
    )

    for ax in (ax1, ax2):
        ax.set_facecolor(PAN)
        ax.tick_params(colors=TK, which='both')
        for sp in ax.spines.values():
            sp.set_edgecolor(SP)
        ax.xaxis.label.set_color(TK)
        ax.yaxis.label.set_color(TK)

    # ── Panel A: error vs Δt ─────────────────────────────────────────
    ax1.set_xlabel('Δt  [a.u.]')
    ax1.set_ylabel('Phase-corrected L² error')
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_title('Error vs time step', color=TX, fontsize=11)

    for i, (order, color) in enumerate(zip(orders, COLORS)):
        th    = THEORY[order]
        slope = slopes[i]
        ax1.plot(dt_vals, errors[i], 'o-', color=color, lw=2, ms=5,
                 label=f'{NAMES[order]}  (fitted slope {slope:+.2f})')

        # Reference guide-line anchored at the coarsest point
        dt0, e0 = dt_vals[0], errors[i, 0]
        dt_line = np.array([dt_vals[-1], dt_vals[0]])
        ax1.plot(dt_line, e0 * (dt_line / dt0) ** th,
                 '--', color=color, lw=0.9, alpha=0.45,
                 label=f'  ∝ Δt^{th}  (theory)')

    ax1.legend(facecolor='#1e1a12', edgecolor=SP,
               labelcolor=TX, fontsize=9)
    ax1.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax1.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

    # ── Panel B: error / Δt^p  (pre-asymptotic constant C) ───────────
    ax2.set_xlabel('Δt  [a.u.]')
    ax2.set_ylabel('error / Δt^p')
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.set_title('Pre-asymptotic constant  C  in  error ≈ C · Δt^p',
                  color=TX, fontsize=11)

    for i, (order, color) in enumerate(zip(orders, COLORS)):
        th     = THEORY[order]
        normed = errors[i] / dt_vals ** th
        ax2.plot(dt_vals, normed, 'o-', color=color, lw=2, ms=5,
                 label=NAMES[order])

    ax2.legend(facecolor='#1e1a12', edgecolor=SP,
               labelcolor=TX, fontsize=9)
    ax2.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax2.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

    plt.tight_layout()
    plt.savefig('convergence.png', dpi=150, facecolor=DARK,
                bbox_inches='tight')
    print("Saved: convergence.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════════
#  CLI entry point
# ══════════════════════════════════════════════════════════════════════

def _parse():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--T_end',     type=float, default=0.5,
                    help='Integration horizon [a.u.] (default 0.5)')
    ap.add_argument('--omega',     type=float, default=1.0,
                    help='SHO frequency ω (default 1.0)')
    ap.add_argument('--x0',        type=float, default=3.0,
                    help='Initial packet centre (default 3.0)')
    ap.add_argument('--k0',        type=float, default=2.0,
                    help='Initial wave vector k₀ (default 2.0)')
    ap.add_argument('--N',         type=int,   default=2048,
                    help='Grid points N (default 2048)')
    ap.add_argument('--L',         type=float, default=30.0,
                    help='Box length [a.u.] (default 30.0)')
    ap.add_argument('--n_dt',      type=int,   default=8,
                    help='Number of Δt values per order (default 8)')
    ap.add_argument('--dt_coarse', type=float, default=0.10,
                    help='Largest Δt (default 0.10)')
    ap.add_argument('--dt_fine',   type=float, default=0.003,
                    help='Smallest Δt (default 0.003)')
    ap.add_argument('--no-plot',   action='store_true',
                    help='Print table only, skip figure')
    return ap.parse_args()


if __name__ == '__main__':
    args = _parse()
    print(f"Convergence study:  T_end={args.T_end},  ω={args.omega},  "
          f"k₀={args.k0},  N={args.N},  L={args.L}")
    print(f"Δt range: [{args.dt_fine}, {args.dt_coarse}],  {args.n_dt} points\n")

    res = run_convergence(
        T_end=args.T_end, omega=args.omega,
        x0=args.x0, k0=args.k0,
        N=args.N, L=args.L,
        n_dt=args.n_dt,
        dt_coarse=args.dt_coarse,
        dt_fine=args.dt_fine,
    )

    print_table(res)
    if not getattr(args, 'no_plot'):
        plot_convergence(res)
