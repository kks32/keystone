"""The Result type. Every solver call returns one (charter item 2).

Verdict semantics (charter item 3). Every verdict here is verified, not
read off a solver flag:

- FEASIBLE means a primal force state was recomputed to satisfy
  equilibrium (||A f + w|| / ||w|| <= tol_eq) and the friction cone
  (max(G f) <= tol_cone). It is the associative verdict.
- INFEASIBLE means a Farkas ray y was found and validated: A^T y lies in
  the dual cone K* (dual residual <= tol_dual) and the load power
  y . w > tol_power. y is the collapse mechanism.
- NO_CONVERGE means neither certificate held.

Bound directions against true Coulomb capacity are structured fields, not
prose. The associative relaxation is not, on its own, exact; a linear 2D
cone overestimates true capacity, an inscribed pyramid is not ordered
against it, and a critical friction is a lower estimate of the true
requirement. Fields carrying associative quantities say so in their names.

physical_bound_direction is populated only when a capacity factor is
reported (lambda_assoc for P2, mu_critical_assoc for P3). P0 and P4 report
no capacity factor, so they leave it None. A P3 inscribed pyramid and a
censored P2 factor are unordered against true capacity, so both carry
"unknown"; only a linear 2D uncensored P2 factor carries "upper" and only
a linear 2D P3 critical friction carries "lower".
"""

from dataclasses import dataclass, field

import numpy as np

FEASIBLE = "feasible"
INFEASIBLE = "infeasible"
NO_CONVERGE = "no_converge"


@dataclass(frozen=True)
class Result:
    """Outcome of a solver call.

    status:  "feasible" | "infeasible" | "no_converge", each verified as
             described in the module docstring.
    margin:  P4 elastic margin ||A f + w|| / ||w||, dimensionless,
             recomputed from the returned force by matvec (not read from
             the solver slack). Zero (below tol_eq) means equilibrated.
             Bound on the optimum: a cone-admissible iterate (cone
             violation <= tol_cone) is P4-feasible, so its margin is an
             upper bound on the optimal P4 margin. An iterate that violates
             the cone is not P4-feasible and its margin need not bound the
             optimum in either direction. info["margin_certified"] is True
             only when the full KKT residual (stationarity, equality,
             inequality, complementarity) certifies the iterate optimal;
             see batch_jax.margin_core.
    forces:  (nf,) newtons, layout of EquilibriumSystem, or None. For P2
             and P3 this is the force state on the certified-feasible side
             of the bracket.
    lambda_assoc: associative P2 load factor under the stated cone model,
             or None. See physical_bound_direction for how it relates to
             true Coulomb capacity.
    mu_critical_assoc: associative P3 critical friction, or None. A lower
             estimate: the true required friction can be higher.
    mechanism: (N, 3) in 2D (vx, vz, wy) or (N, 6) in 3D, virtual twist
             per block on infeasibility, the validated Farkas ray
             normalized, or None.
    cone_model: the cone model used, "linear2d" | "pyramid" | "socp".
    constitutive_model: "associative" for every solver in this library.
    physical_bound_direction: the reported capacity factor relative to true
             Coulomb capacity, or None when no capacity factor is reported.
             "upper" for a linear 2D uncensored associative load factor
             (overestimate of capacity). "unknown" for an inscribed pyramid
             load factor (associative overestimate combined with a cone
             underestimate; no ordering) and for a censored P2 factor (a
             cap that lower-bounds associative capacity but is unordered vs
             true even in 2D). "lower" for a linear 2D critical friction
             (mu_critical_assoc; true required friction can be higher); a
             pyramid critical friction is "unknown" for the same
             cone-vs-associative reason. P0 and P4 report no capacity
             factor and set None.
    trajectory: per-iteration record for P5, or None.
    info:    solver diagnostics (iterations, convergence flags, certificate
             summaries, bound tag).
    """

    status: str
    margin: float
    forces: np.ndarray | None = None
    lambda_assoc: float | None = None
    mu_critical_assoc: float | None = None
    mechanism: np.ndarray | None = None
    cone_model: str | None = None
    constitutive_model: str = "associative"
    physical_bound_direction: str | None = None
    trajectory: np.ndarray | None = None
    info: dict = field(default_factory=dict)

    @property
    def feasible_assoc(self) -> bool:
        """Associative verdict. See the module docstring for bound directions."""
        return self.status == FEASIBLE
