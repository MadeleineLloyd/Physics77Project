import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft

from params     import Params1D, auto_parser, parse_into
from grids      import make_grid_1d, absorbing_mask_1d
from potentials import make_potential_1d
from propagator import step
from exact      import coherent_gaussian_packet_1d, free_gaussian_packet_1d


def measure_error(p, order):
    x, k = make_grid_1d(p)
    V    = make_potential_1d(x, p)
    cap  = absorbing_mask_1d(x, p)

    if p.potential == 'harmonic':
        psi0 = coherent_gaussian_packet_1d(x, 0.0, p)
        psi_exact = coherent_gaussian_packet_1d(x, p.T, p)
    elif p.potential == 'free':
        psi0 = free_gaussian_packet_1d(x, 0.0, p)
        psi_exact = free_gaussian_packet_1d(x, p.T, p)
    else:
        raise ValueError("Error measurement only supports 'free' or 'harmonic' potentials.")

    psi = psi0
    for _ in range(p.Nsteps):
        psi = step(psi, V, k**2, p.dt, cap, fft.fft, fft.ifft, order)

    overlap = np.sum(np.conj(psi_exact) * psi) * p.dx
    phase = overlap / abs(overlap)
    return float(np.sqrt(np.sum(np.abs(psi / phase - psi_exact) ** 2) * p.dx))


def run_convergence_dt(p, T=5.0, orders=(1, 2, 4), n_dt=10, dt_coarse=0.5, dt_fine=0.05):
    p_base = copy.deepcopy(p)
    p_base.T = T

    dt_targets = np.geomspace(dt_coarse, dt_fine, n_dt)
    dt_vals = np.empty(n_dt)
    errors = np.zeros((len(orders), n_dt))

    for j, dt_target in enumerate(dt_targets):
        p_test = copy.deepcopy(p_base)
        p_test.Nsteps = max(1, int(round(T / dt_target)))
        dt_vals[j] = p_test.dt
        for i, order in enumerate(orders):
            errors[i, j] = measure_error(p_test, order)

    mid = max(1, n_dt // 2)
    slopes = [np.polyfit(np.log10(dt_vals[mid:]), np.log10(errors[i, mid:]), 1)[0]
              for i in range(len(orders))]
    return dt_vals, errors, slopes


def run_convergence_N(p, T=5.0, orders=(1, 2, 4), N_values=None, dt_fixed=0.001):
    if N_values is None:
        N_values = [16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 192, 256]

    p_base = copy.deepcopy(p)
    p_base.T = T
    p_base.Nsteps = max(1, int(round(T / dt_fixed)))

    errors = np.zeros((len(orders), len(N_values)))
    dx_values = np.empty(len(N_values))
    for j, N in enumerate(N_values):
        p_test = copy.deepcopy(p_base)
        p_test.N = N
        dx_values[j] = p_test.L / N
        for i, order in enumerate(orders):
            errors[i, j] = measure_error(p_test, order)
    return dx_values, errors


def plot_dt(dt_vals, errors, slopes, orders):
    print("\nΔt".ljust(10) + "order 1".rjust(14) + "order 2".rjust(14) + "order 4".rjust(14))
    print("-" * 52)
    for j, dt in enumerate(dt_vals):
        print(f"{dt:<10.5f}{errors[0,j]:>14.4e}{errors[1,j]:>14.4e}{errors[2,j]:>14.4e}")
    print("\nFitted slopes (theory: 1, 2, 4):")
    for o, s in zip(orders, slopes):
        print(f"  order {o}: {s:+.3f}")

    plt.figure()
    colors = ['C0', 'C1', 'C2']
    for i, order in enumerate(orders):
        plt.loglog(dt_vals, errors[i], 'o-', color=colors[i], label=f'order {order}')
        coeffs = np.polyfit(np.log10(dt_vals), np.log10(errors[i]), 1)
        fit_line = 10**coeffs[1] * dt_vals**coeffs[0]
        plt.loglog(dt_vals, fit_line, '--', color=colors[i], alpha=0.7,
                   label=f'fit order {order} (slope {coeffs[0]:.2f})')
    plt.xlabel('Δt')
    plt.ylabel(r'$L^2$ error')
    plt.legend()
    plt.title('Error vs Δt')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()


def plot_dx(dx_values, errors, orders):
    print("\ndx".ljust(10) + "order 1".rjust(14) + "order 2".rjust(14) + "order 4".rjust(14))
    print("-" * 52)
    for j, dx in enumerate(dx_values):
        print(f"{dx:<10.5e}{errors[0,j]:>14.4e}{errors[1,j]:>14.4e}{errors[2,j]:>14.4e}")

    plt.figure()
    plt.loglog(dx_values, errors[0], 'o-', label='order 1')
    plt.loglog(dx_values, errors[1], 'o-', label='order 2')
    plt.loglog(dx_values, errors[2], 'o-', label='order 4')
    plt.xlabel(r'$\Delta x$')
    plt.ylabel(r'$L^2$ error')
    plt.legend()
    plt.title('Error vs Δx')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()


if __name__ == '__main__':
    p = Params1D(potential='harmonic', N=256, L=20.0, x0=(-5.0,), k0=(0.0,), sigma0=(1 / np.sqrt(2),), omega=1.0)

    orders = [1, 2, 4]
    dt_vals, dt_errors, slopes = run_convergence_dt(p)
    plot_dt(dt_vals, dt_errors, slopes, orders)

    dx_values, dx_errors = run_convergence_N(p)
    plot_dx(dx_values, dx_errors, orders)

    plt.show()