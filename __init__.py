"""
solver
──────
Split-Step FFT solver package for the 1-D and 2-D
Time-Dependent Schrödinger Equation.

Module layout
─────────────
params      — Params1D, Params2D  dataclasses (all user settings)
grids       — make_grid_1d / 2d,  absorbing_mask_1d / 2d
potentials  — make_potential_1d / 2d  (dispatches on p.potential)
initial     — gaussian_packet_1d / 2d
propagator  — strang_step, yoshida_step  (dimension-agnostic)
observables — norm, expect_x/y/p, energy, transmission  (1-D and 2-D)
run_1d      — assemble + simulate + visualise  (1-D, CLI entry point)
run_2d      — assemble + simulate + visualise  (2-D, CLI entry point)
"""

from .params      import Params1D, Params2D
from .grids       import make_grid_1d, make_grid_2d
from .potentials  import make_potential_1d, make_potential_2d
from .initial     import gaussian_packet_1d, gaussian_packet_2d
from .propagator  import lie_step, strang_step, yoshida_step, make_stepper
from .observables import (
    norm_1d, expect_x_1d, expect_p_1d, energy_1d, transmission_1d,
    norm_2d, expect_x_2d, expect_y_2d, expect_px_2d,
    energy_2d, transmitted_prob_2d,
)

__all__ = [
    'Params1D', 'Params2D',
    'make_grid_1d', 'make_grid_2d',
    'make_potential_1d', 'make_potential_2d',
    'gaussian_packet_1d', 'gaussian_packet_2d',
    'lie_step', 'strang_step', 'yoshida_step', 'make_stepper',
    'norm_1d', 'expect_x_1d', 'expect_p_1d', 'energy_1d', 'transmission_1d',
    'norm_2d', 'expect_x_2d', 'expect_y_2d', 'expect_px_2d',
    'energy_2d', 'transmitted_prob_2d',
]
