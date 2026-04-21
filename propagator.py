"""
propagator.py
─────────────
Split-step FFT propagation kernels — dimension-agnostic.

All three integrators accept plain NumPy arrays and FFT/IFFT callables,
so the exact same functions serve both 1-D and 2-D simulations.

Public API
----------
lie_step(psi, exp_V_full, exp_T_full, cap, fwd, inv)
    1st-order Lie (T·V) splitting.  Global error O(Δt).

strang_step(psi, exp_V_half, exp_T_full, cap, fwd, inv)
    2nd-order Strang (V/2·T·V/2) splitting.  Global error O(Δt²).

yoshida_step(psi, V, K2, dt, cap, fwd, inv)
    4th-order Yoshida composition of three Strang sub-steps.
    Global error O(Δt⁴).

make_stepper(p, V, K2, cap, fwd, inv)
    Factory: reads p.order and returns a single-argument step(psi)→psi
    closure, so callers need no dispatch logic of their own.

Mathematical summary
--------------------
Lie (1st order, T·V):
    U(Δt) ≈ exp(−iT Δt) · exp(−iV Δt)               local error O(Δt²)

Strang (2nd order, V/2·T·V/2):
    U(Δt) ≈ exp(−iV Δt/2) · exp(−iT Δt) · exp(−iV Δt/2)
                                                        local error O(Δt³)

Yoshida (4th order):
    U⁽⁴⁾(Δt) = U⁽²⁾(w₁Δt) · U⁽²⁾(w₀Δt) · U⁽²⁾(w₁Δt)
    w₁ = 1/(2 − 2^{1/3}),  w₀ = 1 − 2w₁  ≈ −1.70    local error O(Δt⁵)

    IMPORTANT: w₀ is negative. Each sub-step U⁽²⁾(wᵢΔt) must scale
    *both* the kinetic phase exp(−iK² wᵢΔt/2) *and* the potential
    phase exp(−iV wᵢΔt/2) by the same weight wᵢ. Reusing a pre-built
    exp_V_half from the full Δt breaks the 4th-order cancellation.
"""

import numpy as np
from typing import Callable

Array   = np.ndarray
FFTFunc = Callable[[Array], Array]

# ── Yoshida weights (Yoshida 1990, Phys. Lett. A 150 262) ─────────────────
_W1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))   # ≈ +1.3512
_W0 = 1.0 - 2.0 * _W1                      # ≈ −1.7024  (negative!)


# ── 1st-order Lie step ─────────────────────────────────────────────────────
def lie_step(
    psi:        Array,
    exp_V_full: Array,   # exp(−i V Δt)
    exp_T_full: Array,   # exp(−i K² Δt/2)
    cap:        Array,
    fwd:        FFTFunc,
    inv:        FFTFunc,
) -> Array:
    """
    One 1st-order Lie (T·V) split step.

    Applies the full kinetic propagator first (in momentum space),
    then the full potential step (in position space).  Asymmetric,
    so the local truncation error is O(Δt²) and the global error O(Δt).

        ψ_{n+1} = exp(−iV Δt) · IFFT[exp(−iK² Δt/2) · FFT[ψₙ]]

    Parameters
    ----------
    psi        : current wave function (any shape)
    exp_V_full : exp(−i V Δt),    same shape as psi        (position space)
    exp_T_full : exp(−i K² Δt/2), same shape as psi        (momentum space)
    cap        : absorbing boundary mask
    fwd / inv  : FFT / IFFT callables  (fft/ifft for 1-D, fft2/ifft2 for 2-D)
    """
    psi = inv(fwd(psi) * exp_T_full)   # ① T full step  (momentum space)
    psi = psi * exp_V_full             # ② V full step  (position space)
    return psi * cap                   # ③ absorbing boundary


# ── 2nd-order Strang step ──────────────────────────────────────────────────
def strang_step(
    psi:        Array,
    exp_V_half: Array,   # exp(−i V Δt/2)
    exp_T_full: Array,   # exp(−i K² Δt/2)
    cap:        Array,
    fwd:        FFTFunc,
    inv:        FFTFunc,
) -> Array:
    """
    One 2nd-order Strang (V/2·T·V/2) split step.

    The symmetric arrangement cancels the leading commutator error term,
    giving a local truncation error O(Δt³) and global error O(Δt²).
    The scheme is time-reversible and symplectic.

        ψ_{n+1} = exp(−iV Δt/2) · IFFT[exp(−iK² Δt/2) · FFT[exp(−iV Δt/2) · ψₙ]]

    Parameters
    ----------
    psi        : current wave function (any shape)
    exp_V_half : exp(−i V Δt/2),   same shape as psi  (position space)
    exp_T_full : exp(−i K² Δt/2),  same shape as psi  (momentum space)
    cap        : absorbing boundary mask
    fwd / inv  : FFT / IFFT callables
    """
    psi = psi * exp_V_half             # ① V half-step  (position space)
    psi = inv(fwd(psi) * exp_T_full)   # ② T full step  (momentum space)
    psi = psi * exp_V_half             # ③ V half-step  (position space)
    return psi * cap                   # ④ absorbing boundary


# ── 4th-order Yoshida step ─────────────────────────────────────────────────
def yoshida_step(
    psi: Array,
    V:   Array,   # potential energy array  V(x)  or  V(x,y)
    K2:  Array,   # K² = k² (1-D) or kx²+ky² (2-D), momentum-space array
    dt:  float,
    cap: Array,
    fwd: FFTFunc,
    inv: FFTFunc,
) -> Array:
    """
    One 4th-order Yoshida composite step.

    Each of the three Strang sub-steps uses its own scaled weight wᵢ,
    and *both* the potential phase exp(−iV wᵢΔt/2) and the kinetic
    phase exp(−iK² wᵢΔt/2) are recomputed with that weight.
    This is essential because w₀ ≈ −1.70 is negative: reusing a
    pre-built exp_V_half from the full Δt would give the wrong sign
    and destroy the 4th-order cancellation.

    The absorbing boundary is applied only once per full step (after
    all three sub-steps) to avoid over-damping.

    Parameters
    ----------
    psi  : current wave function
    V    : potential array      (same spatial shape as psi)
    K2   : K² momentum array   (same shape as psi in momentum space)
    dt   : full time step Δt
    cap  : absorbing boundary mask
    fwd / inv : FFT / IFFT callables
    """
    # Dummy cap (no absorption between sub-steps)
    no_cap = np.ones_like(psi, dtype=float)

    def _sub(psi: Array, w: float) -> Array:
        ev = np.exp(-1j * V  * (w * dt / 2))   # scaled V half-step
        et = np.exp(-1j * K2 * (w * dt / 2))   # scaled T full step
        return strang_step(psi, ev, et, no_cap, fwd, inv)

    psi = _sub(psi, _W1)
    psi = _sub(psi, _W0)
    psi = _sub(psi, _W1)
    return psi * cap                            # absorb once at end


# ── Factory ────────────────────────────────────────────────────────────────
def make_stepper(
    p,
    V:   Array,
    K2:  Array,
    cap: Array,
    fwd: FFTFunc,
    inv: FFTFunc,
) -> Callable[[Array], Array]:
    """
    Return a single-argument  step(psi) -> psi  closure for the
    integrator order specified in p.order.

    Pre-computes exp_V_half, exp_V_full, exp_T_full from V, K2 and p.dt
    so the returned closure only does array multiplications per call.

    Yoshida uses V and K2 directly (not pre-computed phases) because
    each sub-step needs independently scaled phases.

    Parameters
    ----------
    p   : Params1D or Params2D  (reads p.order and p.dt)
    V   : potential array   (position space, any shape)
    K2  : K² array          (momentum space, same shape as V)
    cap : absorbing mask
    fwd / inv : FFT / IFFT callables
    """
    dt = p.dt
    exp_V_half = np.exp(-1j * V  * (dt / 2))
    exp_V_full = np.exp(-1j * V  *  dt)
    exp_T_full = np.exp(-1j * K2 * (dt / 2))

    if p.order == 1:
        def step(psi: Array) -> Array:
            return lie_step(psi, exp_V_full, exp_T_full, cap, fwd, inv)

    elif p.order == 2:
        def step(psi: Array) -> Array:
            return strang_step(psi, exp_V_half, exp_T_full, cap, fwd, inv)

    elif p.order == 4:
        def step(psi: Array) -> Array:
            return yoshida_step(psi, V, K2, dt, cap, fwd, inv)

    else:
        raise ValueError(f"Unsupported order {p.order!r}. Choose 1, 2, or 4.")

    return step
