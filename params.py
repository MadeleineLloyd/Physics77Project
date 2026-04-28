import argparse
import dataclasses
import numpy as np
from dataclasses import dataclass
from typing import get_args, get_origin

_SCALAR = {int: int, float: float, str: str, bool: bool}


def _parse_tuple(s):
    return tuple(float(x) for x in s.split(','))


def auto_parser(cls, description=''):
    parser = argparse.ArgumentParser(description=description)
    for f in dataclasses.fields(cls):
        default = f.default if f.default is not dataclasses.MISSING else f.default_factory()
        if f.type in _SCALAR:
            if f.type is bool:
                parser.add_argument(f'--{f.name}', action=argparse.BooleanOptionalAction, default=default)
            else:
                parser.add_argument(f'--{f.name}', type=_SCALAR[f.type], default=default)
        elif f.type is tuple:
            parser.add_argument(f'--{f.name}', type=_parse_tuple,
                                default=default,
                                metavar='a,b,...')
    return parser


def parse_into(parser, cls):
    args = parser.parse_args()
    field_names = {f.name for f in dataclasses.fields(cls)}
    dc, extra = {}, {}
    for k, v in vars(args).items():
        (dc if k in field_names else extra)[k] = v
    return cls(**dc), argparse.Namespace(**extra)


@dataclass
class _BaseParams:
    # Time integration
    T:              float = 5.0
    Nsteps:         int   = 1000
    save_every:     int   = 8

    # Potential
    potential:      str   = 'free'
    V0:             float = 2.0
    smooth:         float = 0.1
    barrier_width:  float = 1.0
    omega:          float = 1.0
    A:              float = 1.0
    Omega:          float = 0.02

    # Initial condition
    initial:        str   = 'gaussian'
    sigma0:         tuple = (1.0,)
    n:              int   = 1

    # Absorbing boundary (CAP)
    cap_width:      float = 0.0
    cap_strength:   float = 0.0

    # Integration order
    order:          int   = 2

    # Output
    save_anim:      bool  = True

    @property
    def dt(self): return self.T / self.Nsteps

@dataclass
class Params1D(_BaseParams):
    N:      int   = 256
    L:      float = 20.0

    x0:     tuple = (-5.0,)
    k0:     tuple = (1.5,)

    a:      float = 0.5

    @property
    def dx(self): return self.L / self.N

    def __post_init__(self):
        if not (len(self.x0) == len(self.k0) == len(self.sigma0)):
            raise ValueError("x0, k0 and σ0 must have the same length.")


@dataclass
class Params2D(_BaseParams):
    Nx:         int   = 256
    Ny:         int   = 256
    Lx:         float = 20.0
    Ly:         float = 20.0

    x0:         tuple = (-5.0,)
    y0:         tuple = (0.0,)
    k0x:        tuple = (1.5,)
    k0y:        tuple = (0.0,)
    
    slit_sep:   float = 2.0
    slit_width: float = 0.4
    epsilon:    float = 1.0

    @property
    def dx(self): return self.Lx / self.Nx

    @property
    def dy(self): return self.Ly / self.Ny
    
    def __post_init__(self):
        if not (len(self.x0) == len(self.y0) == len(self.k0x) == len(self.k0y) == len(self.sigma0)):
            raise ValueError("x0, y0, k0x, k0y and σ0 must have the same length.")
