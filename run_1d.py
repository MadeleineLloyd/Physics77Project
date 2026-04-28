import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import fft

from params         import Params1D, auto_parser, parse_into
from grids          import make_grid_1d, absorbing_mask_1d
from potentials     import make_potential_1d
from initial        import make_initial_1d
from propagator     import step
from observables    import norm_1d, expect_x_1d, expect_p_1d, energy_1d, transmission_1d


def run(p):
    x, k = make_grid_1d(p)
    V    = make_potential_1d(x, p, 0.0)
    psi  = make_initial_1d(x, p)
    cap  = absorbing_mask_1d(x, p)

    E0 = energy_1d(psi, k, V, p.dx)
    psi0 = psi.copy()
    t_arr, V_arr, norm_arr, x_arr, p_arr, E_arr, snapshots = [], [], [], [], [], [], []
    t = 0.0

    for n in range(p.Nsteps):
        V = make_potential_1d(x, p, t)
        psi = step(psi, V, k ** 2, p.dt, cap, fft.fft, fft.ifft, p.order)
        t += p.dt
        if n % p.save_every == 0:
            t_arr.append(t)
            V_arr.append(V)
            norm_arr.append(norm_1d(psi, p.dx))
            x_arr.append(expect_x_1d(psi, x, p.dx))
            p_arr.append(expect_p_1d(psi, k, p.dx))
            E_arr.append(energy_1d(psi, k, V, p.dx))
            snapshots.append(np.abs(psi) ** 2)

    clean_E = [E for E, nm in zip(E_arr, norm_arr) if abs(nm - 1) < 0.01 and not (E != E)]
    if clean_E:
        drift = max(abs(E - E0) / abs(E0) for E in clean_E)
        print(f"Integrator drift (0.99<norm<1.01)  ΔE/E0 = {drift:.2e}")
    if p.potential == 'barrier':
        T = transmission_1d(psi, x, p.barrier_width / 2, p.dx)
        print(f"T = {T:.4f}")
    elif p.potential == 'berry':
        overlap = np.sum(np.conj(psi0) * psi) * p.dx
        gamma = np.angle(overlap) + np.trapezoid(E_arr, t_arr)
        gamma = np.arctan2(np.sin(gamma), np.cos(gamma))
        print(f"Fidelity |⟨ψ0|ψ⟩| = {abs(overlap):.6f}  (theory: 1)")
        print(f"Geometric phase γ = {gamma:.5f} rad  (theory: {np.pi * p.A ** 2 * p.omega:.5f} rad)")
    if p.initial == 'eigenstate':
        overlap = np.sum(np.conj(psi0) * psi) * p.dx
        gamma = np.angle(overlap) + np.trapezoid(E_arr, t_arr)
        gamma = np.arctan2(np.sin(gamma), np.cos(gamma))
        print(f"Fidelity |⟨ψ0|ψ⟩| = {abs(overlap):.6f}  (theory: 1)")
        print(f"Dynamic phase γ = {gamma:.5f} rad  (theory: 0 rad)")

    return dict(x=x, psi_final=psi, t_arr=t_arr, V_arr=V_arr, norm_arr=norm_arr,
                x_arr=x_arr, p_arr=p_arr, E_arr=E_arr, snapshots=snapshots, E0=E0)


def plot(p, res):
    x           = res['x']
    t_arr       = res['t_arr'] 
    V_arr       = res['V_arr']
    snapshots   = res['snapshots']

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    ax.set_title(r'Probability Density $|\psi(x,t)|^2$', fontsize=11)
    prob_line, = ax.plot(x, snapshots[0], lw=1.5)
    ax.set_xlabel(r'$x$ [a.u.]')
    ax.set_ylabel(r'$|\psi|^2$')
    ax.set_xlim(-p.L/2, p.L/2)
    ax.set_ylim(0.0, np.max(snapshots) * 1.1)
    draw_V = p.potential != 'free'
    if draw_V:
        ax_V = ax.twinx()
        ax_V.set_ylabel(r'$V$')
        ax_V.set_ylim(np.min(V_arr) * 1.1, np.max(V_arr) * 1.1)
    V_fill = [None]
    def redraw_V(frame):
        if draw_V:
            if V_fill[0] is not None:
                V_fill[0].remove()
            V_fill[0] = ax_V.fill_between(x, V_arr[frame], alpha=0.2, color='tab:blue')
    redraw_V(0)
    time_text = ax.text(0.02, 0.96, '', transform=ax.transAxes, fontsize=9, va='top')

    axes[0, 1].set_title('Expectation Values', fontsize=11)
    axes[0, 1].plot(t_arr, res['x_arr'], lw=1.5, label=r'$\langle x \rangle$')
    axes[0, 1].plot(t_arr, res['p_arr'], lw=1.5, label=r'$\langle p \rangle$')
    axes[0, 1].axhline(0.0, lw=0.8, ls=':', color='gray')
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

    if p.save_anim:
        def update(frame):
            prob_line.set_ydata(snapshots[frame])
            redraw_V(frame)
            time_text.set_text(f't = {t_arr[frame]:.2f} a.u.')
            return prob_line, time_text
        animation.FuncAnimation(fig, update, frames=len(snapshots), interval=40, blit=True
                                ).save('wavepacket_1d.gif', writer='pillow', fps=25, dpi=100)
        print("Saved: wavepacket_1d.gif")

    plt.savefig('summary_1d.png', dpi=150)
    print("Saved: summary_1d.png")
    plt.show()


if __name__ == '__main__':
    parser = auto_parser(Params1D, description='1-D Split-Step FFT solver')
    p, _ = parse_into(parser, Params1D)
    res = run(p)
    plot(p, res)
