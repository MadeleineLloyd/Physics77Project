"""
run_2d.py
─────────
2-D Split-Step FFT driver.

Usage
-----
    python run_2d.py                        # default: double-slit
    python run_2d.py --potential harmonic_2d --Nsteps 2000
    python run_2d.py --slit_sep 3.0 --order 4
    python run_2d.py --help                 # shows all Params2D fields
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import fft

from params      import Params2D, auto_parser
from grids       import make_grid_2d, absorbing_mask_2d
from potentials  import make_potential_2d
from initial     import gaussian_packet_2d
from propagator  import make_stepper
from observables import (norm_2d, expect_x_2d, expect_y_2d,
                         energy_2d, transmitted_prob_2d)


# ── CLI ────────────────────────────────────────────────────────────────────
def _parse_args():
    return (
        auto_parser(Params2D, description='2-D Split-Step FFT solver')
        .add_argument('--no-anim', action='store_true',
                      help='skip saving the GIF animation')
        .parse_into()
    )


# ── simulation ─────────────────────────────────────────────────────────────
def run(p: Params2D) -> dict:
    x, y, X, Y, KX, KY = make_grid_2d(p)
    V   = make_potential_2d(X, Y, p)
    psi = gaussian_packet_2d(X, Y, p)
    cap = absorbing_mask_2d(x, y, p)

    K2         = KX ** 2 + KY ** 2

    step = make_stepper(p, V, K2, cap, fft.fft2, fft.ifft2)

    E0 = energy_2d(psi, KX, KY, V, p.dx, p.dy)
    t_arr, norm_arr, x_arr, y_arr, E_arr, snapshots = [], [], [], [], [], []
    t = 0.0

    for n in range(p.Nsteps):
        psi  = step(psi)
        t   += p.dt
        if n % p.save_every == 0:
            t_arr.append(t)
            norm_arr.append(norm_2d(psi, p.dx, p.dy))
            x_arr.append(expect_x_2d(psi, X, p.dx, p.dy))
            y_arr.append(expect_y_2d(psi, Y, p.dx, p.dy))
            E_arr.append(energy_2d(psi, KX, KY, V, p.dx, p.dy))
            snapshots.append(np.abs(psi) ** 2)

    T_frac = transmitted_prob_2d(psi, x, p.dx, p.dy)
    print(f"Transmitted fraction = {T_frac:.4f}")
    print(f"Energy drift  dE/E0 = {abs(E_arr[-1] - E0) / abs(E0):.2e}")

    return dict(x=x, y=y, X=X, Y=Y, KX=KX, KY=KY, V=V, psi_final=psi,
                t_arr=t_arr, norm_arr=norm_arr, x_arr=x_arr,
                y_arr=y_arr, E_arr=E_arr, snapshots=snapshots, E0=E0)


# ── visualisation ───────────────────────────────────────────────────────────
def plot(p: Params2D, res: dict, save_anim: bool = True) -> None:
    x, y, V    = res['x'], res['y'], res['V']
    t_arr      = res['t_arr']
    snapshots  = res['snapshots']
    extent     = [-p.Lx / 2, p.Lx / 2, -p.Ly / 2, p.Ly / 2]
    BG, PB, SP, TC = '#0e0c08', '#141008', '#3a3020', '#7a6e5a'

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.patch.set_facecolor(BG)
    for ax in axes.flat:
        ax.set_facecolor(PB)
        ax.tick_params(colors=TC)
        for sp in ax.spines.values():
            sp.set_edgecolor(SP)

    ax = axes[0, 0]
    ax.set_title('Probability Density |psi(x,y,t)|^2', color='#d4c9a8', fontsize=11)
    ax.set_xlabel('x [a.u.]', color=TC)
    ax.set_ylabel('y [a.u.]', color=TC)
    im = ax.imshow(snapshots[0].T, origin='lower', extent=extent,
                   cmap='inferno', aspect='equal', animated=True)
    if V.max() > 0:
        ax.contour(x, y, V.T, levels=[p.V0 / 2],
                   colors=['#4a9acc'], linewidths=0.8)
    time_text = ax.text(0.02, 0.96, '', transform=ax.transAxes,
                        color='#a09080', fontsize=9, va='top')

    ax = axes[0, 1]
    ax.set_title('Expectation Values', color='#d4c9a8', fontsize=11)
    ax.plot(t_arr, res['x_arr'], color='#7ab8a0', lw=1.5, label='<x>')
    ax.plot(t_arr, res['y_arr'], color='#e8a840', lw=1.5, label='<y>', ls='--')
    ax.set_xlabel('t [a.u.]', color=TC)
    ax.legend(facecolor='#1e1a12', edgecolor=SP, labelcolor='#d4c9a8')

    ax = axes[1, 0]
    ax.set_title('Normalization iint|psi|^2 dx dy', color='#d4c9a8', fontsize=11)
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
            im.set_data(snapshots[frame].T)
            im.set_clim(vmax=np.max(snapshots[frame]))
            time_text.set_text(f't = {t_arr[frame]:.2f} a.u.')
            return im, time_text
        ani = animation.FuncAnimation(
            fig, update, frames=len(snapshots), interval=40, blit=True)
        ani.save('wavepacket_2d.gif', writer='pillow', fps=25, dpi=120)
        print("Saved: wavepacket_2d.gif")

    plt.savefig('summary_2d.png', dpi=150, facecolor=BG)
    print("Saved: summary_2d.png")
    plt.show()


# ── entry point ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p, extras = _parse_args()
    res = run(p)
    plot(p, res, save_anim=not extras.no_anim)
