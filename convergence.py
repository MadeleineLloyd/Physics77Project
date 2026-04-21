import argparse
import numpy as np
from scipy import fft

from params      import Params1D
from grids       import make_grid_1d, absorbing_mask_1d
from potentials  import make_potential_1d
from propagator  import step


def psi_exact_sho(x, t, x0, k0, sigma, omega):
    x_c = x0 * np.cos(omega * t) + (k0 / omega) * np.sin(omega * t)
    k_c = k0 * np.cos(omega * t) - x0 * omega * np.sin(omega * t)
    E   = 0.5 * (k0**2 + omega**2 * x0**2) + 0.5 * omega
    phi = E * t - 0.5 * (x_c * k_c - x0 * k0)
    env = np.exp(-(x - x_c)**2 / (4 * sigma**2))
    return (2 * np.pi * sigma**2)**(-0.25) * env * np.exp(1j * (k_c * x - phi))


def measure_error(order, T, Nsteps, N, L, x0, k0, sigma, omega, cap_width):
    p = Params1D(T=T, Nsteps=Nsteps, N=N, L=L, x0=x0, sigma=sigma, k0=k0,
                 potential='harmonic', omega=omega, order=order,
                 cap_width=cap_width, cap_strength=0.000)
    x, k = make_grid_1d(p)
    V    = make_potential_1d(x, p)
    cap  = absorbing_mask_1d(x, p)
    psi  = psi_exact_sho(x, 0.0, x0, k0, sigma, omega)

    for _ in range(Nsteps):
        psi = step(psi, V, k**2, p.dt, cap, fft.fft, fft.ifft, p.order)

    ref = psi_exact_sho(x, Nsteps * p.dt, x0, k0, sigma, omega)
    overlap = np.sum(np.conj(ref) * psi) * p.dx
    phase = overlap / abs(overlap)
    return float(np.sqrt(np.sum(np.abs(psi / phase - ref)**2) * p.dx))


def run_convergence(T=5.0, omega=1.0, x0=-5.0, k0=0.0,
                    N=2048, L=20.0, n_dt=10, dt_coarse=1.0, dt_fine=0.03):
    sigma = 1.0 / np.sqrt(2 * omega)
    cap_w = min(2.0, L / 12)
    orders = [1, 2, 4]
    dt_vals = np.geomspace(dt_coarse, dt_fine, n_dt)
    errors = np.zeros((len(orders), n_dt))

    for i, order in enumerate(orders):
        print(f"order {order}:", end=' ', flush=True)
        for j, dt in enumerate(dt_vals):
            Nsteps = int(round(T / dt))
            errors[i, j] = measure_error(order, T, Nsteps, N, L, x0, k0, sigma, omega, cap_w)
            print('.', end='', flush=True)
        print()

    mid = n_dt // 2
    slopes = [np.polyfit(np.log10(dt_vals[mid:]), np.log10(errors[i, mid:]), 1)[0]
              for i in range(len(orders))]
    return dt_vals, errors, slopes, orders


def print_table(dt_vals, errors, slopes, orders):
    print("\nΔt".ljust(10) + "order 1".rjust(14) + "order 2".rjust(14) + "order 4".rjust(14))
    print("-" * 52)
    for j, dt in enumerate(dt_vals):
        print(f"{dt:<10.5f}{errors[0,j]:>14.4e}{errors[1,j]:>14.4e}{errors[2,j]:>14.4e}")
    print("\nFitted slopes (theory: 1, 2, 4):")
    for o, s in zip(orders, slopes):
        print(f"  order {o}: {s:+.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    dt_vals, errors, slopes, orders = run_convergence()
    print_table(dt_vals, errors, slopes, orders)
    if not args.no_plot:
        import matplotlib.pyplot as plt
        plt.loglog(dt_vals, errors[0], 'o-', label='Lie')
        plt.loglog(dt_vals, errors[1], 'o-', label='Strang')
        plt.loglog(dt_vals, errors[2], 'o-', label='Yoshida')
        plt.xlabel('Δt'); plt.ylabel(r'$L^2$ error'); plt.legend()
        plt.title('Split-step FFT convergence'); plt.show()