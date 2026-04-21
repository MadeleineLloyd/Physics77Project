"""
run_1d.py
─────────
1-D Split-Step FFT driver.

Usage
-----
    python run_1d.py                        # default: barrier scattering
    python run_1d.py --potential harmonic --Nsteps 6000
    python run_1d.py --potential double_well --V0 5.0 --order 4
    python run_1d.py --help                 # shows all Params1D fields
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import fft

from params      import Params1D, auto_parser
from grids       import make_grid_1d, absorbing_mask_1d
from potentials  import make_potential_1d
from initial     import gaussian_packet_1d
from propagator  import make_stepper
from observables import (norm_1d, expect_x_1d, expect_p_1d,
                         energy_1d, transmission_1d)


# ── CLI ────────────────────────────────────────────────────────────────────
def _parse_args():
    """
    Build parser automatically from Params1D fields, then add the two
    extra flags that are not physics parameters.
    """
    return (
        auto_parser(Params1D, description='1-D Split-Step FFT solver')
        .add_argument('--no-anim', action='store_true',
                      help='skip saving the GIF animation')
        .parse_into()
    )


# ── simulation ─────────────────────────────────────────────────────────────
def run(p: Params1D) -> dict:
    x, k = make_grid_1d(p)
    V    = make_potential_1d(x, p)
    psi  = gaussian_packet_1d(x, p)
    cap  = absorbing_mask_1d(x, p)

    K2         = k ** 2

    step = make_stepper(p, V, K2, cap, fft.fft, fft.ifft)

    E0 = energy_1d(psi, k, V, p.dx)
    t_arr, norm_arr, x_arr, p_arr, E_arr, snapshots = [], [], [], [], [], []
    t = 0.0

    for n in range(p.Nsteps):
        psi  = step(psi)
        t   += p.dt
        if n % p.save_every == 0:
            t_arr.append(t)
            norm_arr.append(norm_1d(psi, p.dx))
            x_arr.append(expect_x_1d(psi, x, p.dx))
            p_arr.append(expect_p_1d(psi, k, p.dx))
            E_arr.append(energy_1d(psi, k, V, p.dx))
            snapshots.append(np.abs(psi) ** 2)

    T_coeff = transmission_1d(psi, x, p.dx)
    R_coeff = norm_1d(psi, p.dx) - T_coeff
    print(f"T = {T_coeff:.4f},  R = {R_coeff:.4f},  T+R = {T_coeff+R_coeff:.6f}")
    print(f"Energy drift  dE/E0 = {abs(E_arr[-1] - E0) / abs(E0):.2e}")

    return dict(x=x, k=k, V=V, psi_final=psi,
                t_arr=t_arr, norm_arr=norm_arr, x_arr=x_arr,
                p_arr=p_arr, E_arr=E_arr, snapshots=snapshots, E0=E0)


# ── visualisation ───────────────────────────────────────────────────────────
def plot(p: Params1D, res: dict, save_anim: bool = True) -> None:
    x, V, t_arr, snapshots = res['x'], res['V'], res['t_arr'], res['snapshots']
    BG, PB, SP, TC = '#0e0c08', '#141008', '#3a3020', '#7a6e5a'

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor(BG)
    for ax in axes.flat:
        ax.set_facecolor(PB)
        ax.tick_params(colors=TC)
        for sp in ax.spines.values():
            sp.set_edgecolor(SP)

    ax = axes[0, 0]
    ax.set_title('Probability Density |ψ(x,t)|²', color='#d4c9a8', fontsize=11)
    ax.set_xlabel('x [a.u.]', color=TC)
    prob_line, = ax.plot(x, snapshots[0], color='#e8a840', lw=1.5)
    ax.fill_between(x, snapshots[0], alpha=0.15, color='#e8a840')
    ax2 = ax.twinx()
    if V.max() > 0:
        ax2.fill_between(x, V / V.max() * np.max(snapshots[0]),
                         alpha=0.2, color='#c87a5a')
    ax2.set_yticks([])
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                        color='#a09080', fontsize=9, va='top')
    ax.set_xlim(x[0], x[-1])

    ax = axes[0, 1]
    ax.set_title('Expectation Values', color='#d4c9a8', fontsize=11)
    ax.plot(t_arr, res['x_arr'], color='#7ab8a0', lw=1.5, label='<x>')
    ax.plot(t_arr, res['p_arr'], color='#e8a840', lw=1.5, label='<p>', ls='--')
    ax.set_xlabel('t [a.u.]', color=TC)
    ax.legend(facecolor='#1e1a12', edgecolor=SP, labelcolor='#d4c9a8')

    ax = axes[1, 0]
    ax.set_title('Normalization int|psi|^2 dx', color='#d4c9a8', fontsize=11)
    ax.plot(t_arr, res['norm_arr'], color='#a8c8e8', lw=1.5)
    ax.axhline(1.0, color='#6a5a3a', lw=0.8, ls='--', label='exact')
    ax.set_xlabel('t [a.u.]', color=TC)
    ax.legend(facecolor='#1e1a12', edgecolor=SP, labelcolor='#d4c9a8')

    ax = axes[1, 1]
    ax.set_title('Total Energy <H>', color='#d4c9a8', fontsize=11)
    ax.plot(t_arr, res['E_arr'], color='#c87a5a', lw=1.5)
    ax.axhline(res['E0'], color='#6a5a3a', lw=0.8, ls='--', label='E0')
    ax.set_xlabel('t [a.u.]', color=TC)
    ax.legend(facecolor='#1e1a12', edgecolor=SP, labelcolor='#d4c9a8')

    plt.tight_layout(pad=2.0)

    if save_anim:
        def update(frame):
            prob_line.set_ydata(snapshots[frame])
            time_text.set_text(f't = {t_arr[frame]:.2f} a.u.')
            return prob_line, time_text
        ani = animation.FuncAnimation(
            fig, update, frames=len(snapshots), interval=40, blit=True)
        ani.save('wavepacket_1d.gif', writer='pillow', fps=25, dpi=120)
        print("Saved: wavepacket_1d.gif")

    plt.savefig('summary_1d.png', dpi=150, facecolor=BG)
    print("Saved: summary_1d.png")
    plt.show()


# ── entry point ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p, extras = _parse_args()
    res = run(p)
    plot(p, res, save_anim=not extras.no_anim)
