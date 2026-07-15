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
             When the solve did not converge the margin is only an upper
             bound on the optimal P4 margin; info["margin_certified"]
             records whether the primal-dual gap certifies it optimal.
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
    physical_bound_direction: the reported factor relative to true Coulomb
             capacity. "upper" for a linear 2D associative load factor
             (overestimate of capacity). "unknown" for an inscribed
             pyramid load factor (associative overestimate combined with a
             cone underestimate; no ordering). "lower" for a critical
             friction (mu_critical_assoc; true required friction can be
             higher). None when no such factor is reported.
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
