import numpy as np


def free_gaussian_packet_1d(x, t, p):
    psi = np.zeros_like(x, dtype=np.complex128)
    for x0, k0, sigma0 in zip(p.x0, p.k0, p.sigma0):
        x_c = x0 + k0 * t
        norm = (2 * np.pi * sigma0 ** 2) ** (-0.25) / np.sqrt(1 + 1j * t / (2 * sigma0 ** 2))
        envelope = np.exp(-(x - x_c) ** 2 / (4 * sigma0 ** 2 * (1 + 1j * t / (2 * sigma0 ** 2))))
        psi += norm * envelope * np.exp(1j * (k0 * x - 0.5 * k0 ** 2 * t))
    psi /= np.sqrt(float(np.sum(np.abs(psi) ** 2) * p.dx))
    return psi

def coherent_gaussian_packet_1d(x, t, p):
    if len(set(p.sigma0)) != 1:
        raise ValueError('sigma0 must be the same for all packets')
    psi = np.zeros_like(x, dtype=np.complex128)
    omega = p.omega
    sigma0 = 1 / np.sqrt(2 * omega)
    for x0, k0 in zip(p.x0, p.k0):
        x_c = x0 * np.cos(omega * t) + (k0 / omega) * np.sin(omega * t)
        k_c = k0 * np.cos(omega * t) - x0 * omega * np.sin(omega * t)
        E   = 0.5 * (k0 ** 2 + omega ** 2 * x0 ** 2) + 0.5 * omega
        norm = (2 * np.pi * sigma0 ** 2) ** (-0.25)
        envelope = np.exp(-(x - x_c)**2 / (4 * sigma0**2))
        psi += norm * envelope * np.exp(1j * (k_c * x - E * t + 0.5 * (x_c * k_c - x0 * k0)))
    psi /= np.sqrt(float(np.sum(np.abs(psi) ** 2) * p.dx))
    return psi


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from params         import Params1D, auto_parser, parse_into
    from grids          import make_grid_1d

    parser = auto_parser(Params1D, description='1-D Exact Wavepacket')
    p, _ = parse_into(parser, Params1D)

    x, k = make_grid_1d(p)
    t_arr, snapshots = [], []
    t = 0.0
    for n in range(p.Nsteps):
        psi = coherent_gaussian_packet_1d(x, t, p)
        # psi = free_gaussian_packet_1d(x, t, p)
        t += p.dt
        if n % p.save_every == 0:
            t_arr.append(t)
            snapshots.append(np.abs(psi) ** 2)

    fig, ax = plt.subplots()
    prob_line, = ax.plot(x, snapshots[0], lw=1.5)
    time_text = ax.text(0.02, 0.96, '', transform=ax.transAxes, fontsize=9, va='top')
    def update(frame):
        prob_line.set_ydata(snapshots[frame])
        time_text.set_text(f't = {t_arr[frame]:.2f} a.u.')
        return prob_line, time_text
    ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=40, blit=True)

    plt.show()