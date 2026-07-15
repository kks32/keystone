"""Every tolerance in keystone lives in this dataclass.

A tolerance constant defined anywhere else in the package is a bug
(CLAUDE.md charter item 5). Geometric tolerances are relative to L,
the assembly bounding-box diagonal. Feasibility and certification
tolerances apply to nondimensional quantities and need no scaling.

Solver knobs (qpax solver_tol, max_iter) are not tolerances. They live
in keystone.solve.SolverOptions.

Changing a default requires a CHANGELOG entry and a rerun of the
benchmark suite.
"""

import math
from dataclasses import dataclass, fields

# Certification tolerances that are not set explicitly are derived from
# tol_feas by these factors. tol_dual and tol_gap carry certificate
# acceptance slack: a Farkas dual residual is accepted up to 10 * tol_feas
# and a primal-dual gap up to 100 * tol_feas, both tied to tol_feas so a
# single knob scales the whole certification band.
_TOL_DUAL_FACTOR = 10.0
_TOL_GAP_FACTOR = 100.0


@dataclass(frozen=True)
class Tolerances:
    """Explicit tolerances, passed to every geometry and solver call.

    Geometry:
    theta_tol: coplanarity angle for opposing faces, radians.
    g_tol:     contact gap, relative to L.
    w_tol:     vertex weld distance, relative to L.
    A_min:     minimum patch area, relative to L squared.
               In 2D this bounds patch length, relative to L.

    Solve, primal:
    eps_reg:   Tikhonov regularization on the P4 force block,
               dimensionless (system is nondimensionalized first).
    tol_feas:  the base feasibility scale, dimensionless. The other
               certification tolerances default to multiples of it.

    Certification (verified verdicts):
    tol_eq:    equilibrium residual ||A f + w|| / ||w|| below which a
               primal force state certifies feasibility. Default tol_feas.
    tol_cone:  cone violation max(0, max(G f)) below which the force state
               is accepted as cone-admissible. Default tol_feas.
    tol_dual:  dual/certificate residual ||A^T y + G^T z|| below which a
               Farkas ray is accepted. Default 10 * tol_feas (certificate
               acceptance slack).
    tol_power: minimum load power y . w for a Farkas ray to certify
               infeasibility. Default tol_feas.
    tol_gap:   primal-dual complementarity gap below which the reported P4
               margin is certified optimal. Default 100 * tol_feas.

    Unset certification fields (None) derive from tol_feas in
    __post_init__. All eleven values must be finite and strictly positive.
    """

    theta_tol: float = 1e-3
    g_tol: float = 1e-4
    w_tol: float = 1e-9
    A_min: float = 1e-8
    eps_reg: float = 1e-12
    tol_feas: float = 1e-8
    tol_eq: float | None = None
    tol_cone: float | None = None
    tol_dual: float | None = None
    tol_power: float | None = None
    tol_gap: float | None = None

    def __post_init__(self):
        # Derive certification tolerances from tol_feas when left unset.
        if self.tol_eq is None:
            object.__setattr__(self, "tol_eq", self.tol_feas)
        if self.tol_cone is None:
            object.__setattr__(self, "tol_cone", self.tol_feas)
        if self.tol_dual is None:
            object.__setattr__(self, "tol_dual", _TOL_DUAL_FACTOR * self.tol_feas)
        if self.tol_power is None:
            object.__setattr__(self, "tol_power", self.tol_feas)
        if self.tol_gap is None:
            object.__setattr__(self, "tol_gap", _TOL_GAP_FACTOR * self.tol_feas)
        # Every tolerance must be a finite, strictly positive number.
        for fld in fields(self):
            val = getattr(self, fld.name)
            if not isinstance(val, (int, float)) or not math.isfinite(val) or val <= 0.0:
                raise ValueError(
                    f"Tolerances.{fld.name} must be finite and > 0, got {val!r}"
                )
