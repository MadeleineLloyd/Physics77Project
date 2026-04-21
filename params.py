"""
params.py
─────────
Configuration dataclasses for 1-D and 2-D split-step FFT solvers.
All quantities in atomic units:  ℏ = m = 1.

Design
------
Each dataclass is the single source of truth for every parameter —
its field names, types, and defaults are defined exactly once here.

`auto_parser(cls)` reads those fields via dataclasses.fields() and
builds an argparse.ArgumentParser automatically, so run_1d.py and
run_2d.py never repeat the field list.

The `__post_init__` method validates physical constraints and warns
about numerically problematic settings.

Usage
-----
    # In a script or notebook — construct directly:
    from params import Params1D, Params2D
    p = Params1D(potential='harmonic', k0=0.0)

    # As a CLI helper — let run_*.py call:
    from params import auto_parser
    p, extras = auto_parser(Params1D).parse_into()
"""

import dataclasses
import argparse
import warnings
from dataclasses import dataclass

# ── internal type → argparse type map ─────────────────────────────────────
_ARGTYPE = {int: int, float: float, str: str, bool: bool}


class _AutoParser:
    """
    Thin wrapper around ArgumentParser built automatically from a dataclass.
    Extra flags (--order, --no-anim) not in the dataclass are added with
    .add_argument() before calling .parse_into().
    """
    def __init__(self, parser: argparse.ArgumentParser, cls):
        self._parser = parser
        self._cls    = cls

    def add_argument(self, *args, **kwargs):
        """Delegate extra arguments (e.g. --order, --no-anim) to the parser."""
        self._parser.add_argument(*args, **kwargs)
        return self

    def parse_into(self, cls=None):
        """
        Parse sys.argv, split into dataclass fields vs extra flags, and
        return (params_instance, extras_namespace).

        Returns
        -------
        params : instance of the dataclass built from CLI values
        extras : argparse.Namespace for non-dataclass flags
        """
        cls = cls or self._cls
        args = self._parser.parse_args()
        field_names = {f.name for f in dataclasses.fields(cls)}

        dc_kwargs, extra_dict = {}, {}
        for key, val in vars(args).items():
            clean = key.replace('-', '_')      # --no-anim  →  no_anim
            (dc_kwargs if clean in field_names else extra_dict)[clean] = val

        return cls(**dc_kwargs), argparse.Namespace(**extra_dict)


def auto_parser(cls, description: str = '') -> _AutoParser:
    """
    Build an ArgumentParser from *all* dataclass fields of `cls`
    (including inherited ones), using each field's type and default.

    Fields whose type is not in {int, float, str, bool} are skipped
    (e.g. @property computed fields like dx, dy).
    """
    parser = argparse.ArgumentParser(description=description or cls.__doc__ or '')

    for f in dataclasses.fields(cls):
        argtype = _ARGTYPE.get(f.type)
        if argtype is None:
            continue                        # skip computed / non-primitive fields

        default = (f.default
                   if f.default is not dataclasses.MISSING
                   else f.default_factory())

        if f.type is bool:
            action = 'store_false' if default else 'store_true'
            parser.add_argument(f'--{f.name}', action=action,
                                help=f'(default: {default})')
        else:
            parser.add_argument(f'--{f.name}', type=argtype,
                                default=default,
                                help=f'(default: {default})')

    return _AutoParser(parser, cls)


# ── shared base ────────────────────────────────────────────────────────────
@dataclass
class _BaseParams:
    """Shared time-integration, wave-packet, and CAP settings."""

    # ── Time integration ──────────────────────────────────────────────────
    dt:         float = 0.005
    Nsteps:     int   = 4000
    save_every: int   = 20

    # ── Wave packet ───────────────────────────────────────────────────────
    sigma: float = 1.0

    # ── Potential ─────────────────────────────────────────────────────────
    potential:     str   = 'barrier'
    V0:            float = 20.0
    barrier_width: float = 1.0
    omega:         float = 1.0

    # ── Integrator order ──────────────────────────────────────────────────
    # 1 = Lie (1st-order, T·V),  2 = Strang (2nd-order, V/2·T·V/2),
    # 4 = Yoshida (4th-order, composition of three Strang sub-steps)
    # Global error scales as  O(Δt^order).
    order: int = 2

    # ── Absorbing boundary (CAP) ──────────────────────────────────────────
    cap_width:    float = 3.0
    cap_strength: float = 0.02


# ── 1-D ────────────────────────────────────────────────────────────────────
@dataclass
class Params1D(_BaseParams):
    """Parameters for the 1-D split-step FFT solver (hbar = m = 1)."""

    N:  int   = 2048
    L:  float = 40.0
    x0: float = -8.0
    k0: float =  5.0

    @property
    def dx(self) -> float:
        return self.L / self.N

    def __post_init__(self):
        if self.N & (self.N - 1):
            warnings.warn(f"N={self.N} is not a power of 2; FFT may be slow.")
        dt_cfl_T = self.dx ** 2 / 2
        dt_cfl_V = 1.0 / self.V0 if self.V0 else float('inf')
        if self.dt > 0.5 * dt_cfl_T:
            warnings.warn(
                f"dt={self.dt} may violate kinetic CFL (dx^2/2 = {dt_cfl_T:.4f}).")
        if self.dt > 0.5 * dt_cfl_V:
            warnings.warn(
                f"dt={self.dt} may violate potential CFL (1/V0 = {dt_cfl_V:.4f}).")
        if self.cap_width > self.L / 6:
            warnings.warn(
                f"cap_width={self.cap_width} is large relative to L={self.L}.")


# ── 2-D ────────────────────────────────────────────────────────────────────
@dataclass
class Params2D(_BaseParams):
    """Parameters for the 2-D split-step FFT solver (hbar = m = 1)."""

    Nx: int   = 512
    Ny: int   = 512
    Lx: float = 30.0
    Ly: float = 30.0
    x0:    float = -8.0
    y0:    float =  0.0
    sigma: float =  1.5
    k0x:   float =  6.0
    k0y:   float =  0.0
    wall_thick:  float = 0.3
    slit_sep:    float = 2.0
    slit_width:  float = 0.4

    @property
    def dx(self) -> float:
        return self.Lx / self.Nx

    @property
    def dy(self) -> float:
        return self.Ly / self.Ny

    def __post_init__(self):
        for n, label in [(self.Nx, 'Nx'), (self.Ny, 'Ny')]:
            if n & (n - 1):
                warnings.warn(f"{label}={n} is not a power of 2; FFT may be slow.")
        dt_cfl_T = min(self.dx, self.dy) ** 2 / 2
        dt_cfl_V = 1.0 / self.V0 if self.V0 else float('inf')
        if self.dt > 0.5 * dt_cfl_T:
            warnings.warn(
                f"dt={self.dt} may violate kinetic CFL (min(dx,dy)^2/2 = {dt_cfl_T:.4f}).")
        if self.dt > 0.5 * dt_cfl_V:
            warnings.warn(
                f"dt={self.dt} may violate potential CFL (1/V0 = {dt_cfl_V:.4f}).")
