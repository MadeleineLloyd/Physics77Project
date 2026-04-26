import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import fft

from params      import Params1D, auto_parser, parse_into
from grids       import make_grid_1d, absorbing_mask_1d
from potentials  import make_potential_1d
from initial     import gaussian_packet_1d
from propagator  import step
from observables import norm_1d, expect_x_1d, expect_p_1d, energy_1d, transmission_1d


def run(p):
    x, k = make_grid_1d(p)
    V    = make_potential_1d(x, p)
    psi  = gaussian_packet_1d(x, p)
    cap  = absorbing_mask_1d(x, p)

    E0 = energy_1d(psi, k, V, p.dx)
    t_arr, norm_arr, x_arr, p_arr, E_arr, snapshots = [], [], [], [], [], []
    t = 0.0

    for n in range(p.Nsteps):
        psi = step(psi, V, k ** 2, p.dt, cap, fft.fft, fft.ifft, p.order)
        t  += p.dt
        if n % p.save_every == 0:
            t_arr.append(t)
            norm_arr.append(norm_1d(psi, p.dx))
            x_arr.append(expect_x_1d(psi, x, p.dx))
            p_arr.append(expect_p_1d(psi, k, p.dx))
            E_arr.append(energy_1d(psi, k, V, p.dx))
            snapshots.append(np.abs(psi) ** 2)

    T = transmission_1d(psi, x, p.dx)
    print(f"T = {T:.4f}")
    print(f"Energy drift  ΔE/E0 = {abs(E_arr[-1] - E0) / abs(E0):.2e}")
    return dict(x=x, k=k, V=V, psi_final=psi, t_arr=t_arr, norm_arr=norm_arr,
                x_arr=x_arr, p_arr=p_arr, E_arr=E_arr, snapshots=snapshots, E0=E0)


def plot(p, res, save_anim=True):
    x, V, t_arr, snapshots = res['x'], res['V'], res['t_arr'], res['snapshots']

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    ax.set_title(r'Probability Density $|\psi(x,t)|^2$', fontsize=11)
    ax.set_xlabel(r'$x$ [a.u.]')
    prob_line, = ax.plot(x, snapshots[0], lw=1.5)
    if V.max() > 0:
        axes[0, 0].twinx().fill_between(x, V / V.max() * np.max(snapshots[0]), alpha=0.2)
        axes[0, 0].get_figure().axes[-1].set_yticks([])
    time_text = ax.text(0.02, 0.96, '', transform=ax.transAxes, fontsize=9, va='top')
    ax.set_xlim(-p.L/2, p.L/2)

    axes[0, 1].set_title('Expectation Values', fontsize=11)
    axes[0, 1].plot(t_arr, res['x_arr'], lw=1.5, label=r'$\langle x \rangle$')
    axes[0, 1].plot(t_arr, res['p_arr'], color='red', lw=1.5, label=r'$\langle p \rangle$')
    axes[0,1].axhline(y=0,color='black',ls='dotted')
    axes[0, 1].set_xlabel('t [a.u.]')
    axes[0, 1].legend()

    axes[1, 0].set_title(r'Normalization $\int|\psi|^2 dx$', fontsize=11)
    axes[1, 0].plot(t_arr, res['norm_arr'], lw=1.5)
    axes[1, 0].axhline(1.0, lw=0.8, ls='--', label='exact')
    axes[1, 0].set_xlabel('t [a.u.]')
    axes[1, 0].legend()

    axes[1, 1].set_title(r'Total Energy $\langle H \rangle$', fontsize=11)
    axes[1, 1].plot(t_arr, res['E_arr'], lw=1.5)
    axes[1, 1].axhline(res['E0'], lw=0.8, ls='--', label=r'$E_0$')
    axes[1, 1].set_xlabel('t [a.u.]')
    axes[1, 1].legend()

    plt.tight_layout(pad=2.0)

    if save_anim:
        def update(frame):
            prob_line.set_ydata(snapshots[frame])
            time_text.set_text(f't = {t_arr[frame]:.2f} a.u.')
            return prob_line, time_text
        animation.FuncAnimation(fig, update, frames=len(snapshots), interval=40, blit=True
                                 ).save('wavepacket_1d.gif', writer='pillow', fps=25, dpi=120)
        print("Saved: wavepacket_1d.gif")

    plt.savefig('summary_1d.png', dpi=150)
    print("Saved: summary_1d.png")
    plt.show()


if __name__ == '__main__':
    parser = auto_parser(Params1D, description='1-D Split-Step FFT solver')
    parser.add_argument('--no-anim', action='store_true')
    p, extras = parse_into(parser, Params1D)
    res = run(p)
    plot(p, res, save_anim=not extras.no_anim)
