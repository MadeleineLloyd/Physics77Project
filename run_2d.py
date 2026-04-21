import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import fft

from params      import Params2D, auto_parser, parse_into
from grids       import make_grid_2d, absorbing_mask_2d
from potentials  import make_potential_2d
from initial     import gaussian_packet_2d
from propagator  import step
from observables import norm_2d, expect_x_2d, expect_y_2d, energy_2d, transmitted_prob_2d


def run(p):
    x, y, X, Y, KX, KY = make_grid_2d(p)
    V   = make_potential_2d(X, Y, p)
    psi = gaussian_packet_2d(X, Y, p)
    cap = absorbing_mask_2d(x, y, p)

    E0 = energy_2d(psi, KX, KY, V, p.dx, p.dy)
    t_arr, norm_arr, x_arr, y_arr, E_arr, snapshots = [], [], [], [], [], []
    t = 0.0

    for n in range(p.Nsteps):
        psi = step(psi, V, KX ** 2 + KY ** 2, p.dt, cap, fft.fft2, fft.ifft2, p.order)
        t  += p.dt
        if n % p.save_every == 0:
            t_arr.append(t)
            norm_arr.append(norm_2d(psi, p.dx, p.dy))
            x_arr.append(expect_x_2d(psi, X, p.dx, p.dy))
            y_arr.append(expect_y_2d(psi, Y, p.dx, p.dy))
            E_arr.append(energy_2d(psi, KX, KY, V, p.dx, p.dy))
            snapshots.append(np.abs(psi) ** 2)

    T = transmitted_prob_2d(psi, x, p.dx, p.dy)
    print(f"Transmitted fraction = {T:.4f}")
    print(f"Energy drift  ΔE/E0 = {abs(E_arr[-1] - E0) / abs(E0):.2e}")
    return dict(x=x, y=y, X=X, Y=Y, KX=KX, KY=KY, V=V, psi_final=psi,
                t_arr=t_arr, norm_arr=norm_arr, x_arr=x_arr,
                y_arr=y_arr, E_arr=E_arr, snapshots=snapshots, E0=E0)


def plot(p, res, save_anim=True):
    x, y, V    = res['x'], res['y'], res['V']
    t_arr      = res['t_arr']
    snapshots  = res['snapshots']
    extent     = [-p.Lx / 2, p.Lx / 2, -p.Ly / 2, p.Ly / 2]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    ax = axes[0, 0]
    ax.set_title(r'Probability Density $|\psi(x,y,t)|^2$', fontsize=11)
    ax.set_xlabel(r'$x$ [a.u.]')
    ax.set_ylabel(r'$y$ [a.u.]')
    im = ax.imshow(snapshots[0].T, origin='lower', extent=extent, 
                   cmap='inferno', aspect='equal', animated=True, 
                   vmin=0, vmax=np.percentile(snapshots[0], 99.5))
    ax.imshow(V.T, origin='lower', extent=extent, 
              cmap='Greys_r', alpha=0.3, aspect='equal', 
              vmin=np.min(V), vmax=np.max(V))
    if V.max() > 0:
        ax.contour(x, y, V.T, levels=[(np.max(V) + np.min(V)) / 2], 
                   colors='white', linewidths=0.8)
    time_text = ax.text(0.02, 0.96, '', transform=ax.transAxes, 
                        color='white', fontsize=9, va='top')

    axes[0, 1].set_title('Expectation Values', fontsize=11)
    axes[0, 1].plot(t_arr, res['x_arr'], lw=1.5, label=r'$\langle x \rangle$')
    axes[0, 1].plot(t_arr, res['y_arr'], lw=1.5, label=r'$\langle y \rangle$', ls='--')
    axes[0, 1].set_xlabel('t [a.u.]')
    axes[0, 1].legend()

    axes[1, 0].set_title(r'Normalization $\iint|\psi|^2 dx dy$', fontsize=11)
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
            im.set_data(snapshots[frame].T)
            im.set_clim(vmax=np.percentile(snapshots[frame], 99.5))
            time_text.set_text(f't = {t_arr[frame]:.2f} a.u.')
            return im, time_text
        animation.FuncAnimation(fig, update, frames=len(snapshots), interval=40, blit=True
                                ).save('wavepacket_2d.gif', writer='pillow', fps=25, dpi=120)
        print("Saved: wavepacket_2d.gif")

    plt.savefig('summary_2d.png', dpi=150)
    print("Saved: summary_2d.png")
    plt.show()


if __name__ == '__main__':
    parser = auto_parser(Params2D, description='2-D Split-Step FFT solver')
    parser.add_argument('--no-anim', action='store_true')
    p, extras = parse_into(parser, Params2D)
    res = run(p)
    plot(p, res, save_anim=not extras.no_anim)
