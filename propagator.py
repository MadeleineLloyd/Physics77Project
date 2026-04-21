import numpy as np

_W1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_W0 = 1.0 - 2.0 * _W1


def lie_step(psi, V, K2, dt, cap, fwd, inv):
    exp_V_full = np.exp(-1j * V * dt)
    exp_T_full = np.exp(-1j * K2 * (dt / 2))

    psi = inv(fwd(psi) * exp_T_full)
    psi = psi * exp_V_full
    return psi * cap


def strang_step(psi, V, K2, dt, cap, fwd, inv):
    exp_V_half = np.exp(-1j * V  * (dt / 2))
    exp_T_full = np.exp(-1j * K2 * (dt / 2))

    psi = psi * exp_V_half
    psi = inv(fwd(psi) * exp_T_full)
    psi = psi * exp_V_half
    return psi * cap


def yoshida_step(psi, V, K2, dt, cap, fwd, inv):
    for w in (_W1, _W0, _W1):
        psi = strang_step(psi, V, K2, w * dt, cap, fwd, inv)
    return psi


def step(psi, V, K2, dt, cap, fwd, inv, order=2):
    if order == 1:
        return lie_step(psi, V, K2, dt, cap, fwd, inv)
    elif order == 2:
        return strang_step(psi, V, K2, dt, cap, fwd, inv)
    elif order == 4:
        return yoshida_step(psi, V, K2, dt, cap, fwd, inv)
    else:
        raise ValueError(f"Unsupported order {p.order!r}. Choose 1, 2, or 4.")