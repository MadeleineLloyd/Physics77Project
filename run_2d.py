import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from scipy import fft

from params         import Params2D, auto_parser, parse_into
from grids          import make_grid_2d, absorbing_mask_2d
from potentials     import make_potential_2d
from initial        import make_initial_2d
from propagator     import step
from observables    import (
    norm_2d, expect_x_2d, expect_y_2d, expect_px_2d, expect_py_2d,
    energy_2d, transmission_2d
)


colors = [(0, 0, 0, 0), (255, 255, 255, 0.5)]
new_cmap = LinearSegmentedColormap.from_list('V_alpha', colors, N=256)


def run(p):
    x, y, X, Y, KX, KY = make_grid_2d(p)
    V   = make_potential_2d(X, Y, p, 0.0)
    psi = make_initial_2d(X, Y, p)
    cap = absorbing_mask_2d(x, y, p)

    E0 = energy_2d(psi, KX, KY, V, p.dx, p.dy)
    psi0 = psi.copy()
    t_arr, V_arr, norm_arr, x_arr, y_arr, px_arr, py_arr, E_arr, snapshots = [], [], [], [], [], [], [], [], []
    t = 0.0

    for n in range(p.Nsteps):
        V = make_potential_2d(X, Y, p, t)
        psi = step(psi, V, KX ** 2 + KY ** 2, p.dt, cap, fft.fft2, fft.ifft2, p.order)
        t += p.dt
        if n % p.save_every == 0:
            t_arr.append(t)
            V_arr.append(V)
            norm_arr.append(norm_2d(psi, p.dx, p.dy))
            x_arr.append(expect_x_2d(psi, X, p.dx, p.dy))
            y_arr.append(expect_y_2d(psi, Y, p.dx, p.dy))
            px_arr.append(expect_px_2d(psi, KX, p.dx, p.dy))
            py_arr.append(expect_py_2d(psi, KY, p.dx, p.dy))
            E_arr.append(energy_2d(psi, KX, KY, V, p.dx, p.dy))
            snapshots.append(np.abs(psi) ** 2)

    clean_E = [E for E, nm in zip(E_arr, norm_arr) if abs(nm - 1) < 0.01 and not (E != E)]
    if clean_E:
        drift = max(abs(E - E0) / abs(E0) for E in clean_E)
        print(f"Integrator drift (0.99<norm<1.01)  ΔE/E0 = {drift:.2e}")
    if p.potential == 'barrier' or p.potential == 'double_slit':
        T = transmission_2d(psi, x, p.barrier_width / 2, p.dx, p.dy)
        print(f"Transmitted fraction = {T:.4f}")
    elif p.potential == 'berry':
        overlap = np.sum(np.conj(psi0) * psi) * p.dx * p.dy
        gamma = np.angle(overlap) + np.trapezoid(E_arr, dx=p.dt)
        gamma = np.arctan2(np.sin(gamma), np.cos(gamma))
        print(f"Geometric phase γ = {gamma:+.4f} rad  (theory: {2 * np.pi * p.A ** 2 * np.omega:+.4f} rad)")
    return dict(x=x, y=y, X=X, Y=Y, psi_final=psi, t_arr=t_arr, V_arr=V_arr, norm_arr=norm_arr,
                x_arr=x_arr, y_arr=y_arr, px_arr=px_arr, py_arr=py_arr,
                E_arr=E_arr, snapshots=snapshots, E0=E0)


def plot(p, res):
    x, y        = res['x'], res['y']
    t_arr       = res['t_arr']
    V_arr       = res['V_arr']
    snapshots   = res['snapshots']
    extent      = [-p.Lx / 2, p.Lx / 2, -p.Ly / 2, p.Ly / 2]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    ax = axes[0, 0]
    ax.set_title(r'Probability Density $|\psi(x,y,t)|^2$', fontsize=11)
    ax.set_xlabel(r'$x$ [a.u.]')
    ax.set_ylabel(r'$y$ [a.u.]')
    im = ax.imshow(snapshots[0].T, origin='lower', extent=extent, 
                   cmap='inferno', aspect='equal', animated=True, 
                   vmin=0, vmax=np.max(snapshots))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    draw_V = p.potential != 'free'
    if draw_V:
        V_min_global, V_max_global = np.min(V_arr), np.max(V_arr)
        level = (V_min_global + V_max_global) / 2
        ax.text(0.02, 0.02, f'V = {level:.0f}', transform=ax.transAxes, 
                color='white', fontsize=7, va='bottom')
        im_V = ax.imshow(V_arr[0].T, origin='lower', extent=extent, 
                         cmap=new_cmap, aspect='equal', animated=True, 
                         vmin=V_min_global, vmax=V_max_global)
        contours = [None]
        def redraw_contours(frame):
            if contours[0] is not None:
                contours[0].remove()
            contours[0] = ax.contour(x, y, V_arr[frame].T, levels=[level],
                                     colors=['white'], linewidths=0.8)
        redraw_contours(0)
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                        color='white', fontsize=9, va='top')

    axes[0, 1].set_title('Expectation Values', fontsize=11)
    axes[0, 1].plot(t_arr, res['x_arr'], lw=1.5, label=r'$\langle x \rangle$', color='tab:blue')
    axes[0, 1].plot(t_arr, res['y_arr'], lw=1.5, label=r'$\langle y \rangle$', ls='--', color='tab:blue')
    axes[0, 1].plot(t_arr, res['px_arr'], lw=1.5, label=r'$\langle p_x \rangle$', color='tab:orange')
    axes[0, 1].plot(t_arr, res['py_arr'], lw=1.5, label=r'$\langle p_y \rangle$', ls='--', color='tab:orange')
    axes[0, 1].axhline(0.0, lw=0.8, ls=':', color='gray')
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

    if p.save_anim:
        def update(frame):
            im.set_data(snapshots[frame].T)
            time_text.set_text(f't = {t_arr[frame]:.2f} a.u.')
            if draw_V:
                im_V.set_data(V_arr[frame].T)
                redraw_contours(frame)
                return im, im_V, time_text
            else:
                return im, time_text
        animation.FuncAnimation(fig, update, frames=len(snapshots), interval=40, blit=True
                                ).save('wavepacket_2d.gif', writer='pillow', fps=25, dpi=100)
        print("Saved: wavepacket_2d.gif")

    plt.savefig('summary_2d.png', dpi=150)
    print("Saved: summary_2d.png")
    plt.show()


if __name__ == '__main__':
    parser = auto_parser(Params2D, description='2-D Split-Step FFT solver')
    p, _ = parse_into(parser, Params2D)
    res = run(p)
    plot(p, res)
