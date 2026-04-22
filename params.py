import argparse
import dataclasses
from dataclasses import dataclass

_ARGTYPE = {int: int, float: float, str: str, bool: bool}


def auto_parser(cls, description=''):
    parser = argparse.ArgumentParser(description=description)
    for f in dataclasses.fields(cls):
        if f.type not in _ARGTYPE:
            continue
        default = f.default if f.default is not dataclasses.MISSING else f.default_factory()
        if f.type is bool:
            parser.add_argument(f'--{f.name}', action='store_false' if default else 'store_true', default=default)
        else:
            parser.add_argument(f'--{f.name}', type=_ARGTYPE[f.type], default=default)
    return parser


def parse_into(parser, cls, extra_args=()):
    for name, *opts in extra_args:
        parser.add_argument(name, **opts[0] if opts else {})
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
    Nsteps:         int   = 100

    # Potential
    potential:      str   = 'free'
    V0:             float = 20.0
    barrier_width:  float = 1.0
    omega:          float = 1.0

    # Absorbing boundary (CAP)
    cap_width:      float = 1.0
    cap_strength:   float = 0.0

    # Integration order
    order:          int   = 2

    @property
    def dt(self): return self.T / self.Nsteps

    @property
    def save_every(self): return max(1, int(round(0.04 / self.dt)))

@dataclass
class Params1D(_BaseParams):
    N:      int   = 512
    L:      float = 20.0

    x0:     float = -5.0
    k0:     float = 1.5
    sigma:  float = 1.0

    @property
    def dx(self): return self.L / self.N


@dataclass
class Params2D(_BaseParams):
    Nx:         int   = 512
    Ny:         int   = 512
    Lx:         float = 20.0
    Ly:         float = 20.0

    x0:         float = -5.0
    y0:         float = 0.0
    sigma:      float = 1.0
    k0x:        float = 1.5
    k0y:        float = 0.0
    
    wall_thick: float = 0.3
    slit_sep:   float = 2.0
    slit_width: float = 0.4

    @property
    def dx(self): return self.Lx / self.Nx

    @property
    def dy(self): return self.Ly / self.Ny
