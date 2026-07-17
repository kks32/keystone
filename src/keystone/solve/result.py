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

A bisection reports two bracket endpoints. The headline factor
(lambda_assoc for P2, mu_critical_assoc for P3) is the verified-feasible
bound: it is achievable in the associative model but is not ordered against
true Coulomb capacity. The verified endpoint on the other side carries the
true-model ordering: lambda_upper_verified is verified infeasible and
upper-bounds true capacity in 2D, and mu_lower_verified is verified
infeasible and lower-bounds the true required friction in 2D.

physical_bound_direction describes that verified endpoint, not the headline
factor. It is None when no verified endpoint is reported. P0 and P4 report
no capacity factor, so they leave it None. A P3 inscribed pyramid and a
censored or uncertified P2 factor are unordered against true capacity, so
both carry "unknown"; only a linear 2D uncensored P2 upper endpoint carries
"upper" and only a linear 2D P3 lower endpoint carries "lower".
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
             Objective bound on the optimum: the P4 program minimizes
             0.5 ||A f + w||^2 + 0.5 eps_reg ||f||^2 over the cone, so a
             cone-admissible iterate (cone violation <= tol_cone) is
             P4-feasible and its regularized objective upper-bounds the
             optimal objective. The margin alone does not bound the optimal
             margin: the regularizer trades residual against force size. An
             iterate that violates the cone is not P4-feasible and bounds
             nothing. info["margin_certified"] is True only when the full
             KKT check (stationarity, equality, inequality-slack, and
             complementarity residuals below tol_gap, and the slack and dual
             minima above -tol_gap) certifies the iterate optimal; see
             batch_jax.margin_core.
    forces:  (nf,) newtons, layout of EquilibriumSystem, or None. For P2
             and P3 this is the force state on the verified-feasible side
             of the bracket.
    lambda_assoc: associative P2 load factor under the stated cone model,
             or None. Kept for backward compatibility; it equals
             lambda_achievable, the verified-feasible bracket bound. This
             lower bracket bound is achievable in the associative model but
             is NOT ordered against true Coulomb capacity; the value that
             upper-bounds true capacity is lambda_upper_verified. See
             physical_bound_direction.
    lambda_achievable: the largest load factor verified feasible in the
             associative model (the lower bracket bound, equal to
             lambda_assoc). Achievable, not ordered against true capacity.
             None when no capacity factor is reported.
    lambda_upper_verified: a load factor verified infeasible in the
             associative model (the upper bracket bound), or None when no
             infeasible endpoint was verified. For a 2D exact cone it obeys
             lambda_upper_verified >= lambda*_assoc >= lambda_true, so it
             upper-bounds true Coulomb capacity. This is the endpoint
             physical_bound_direction describes.
    mu_critical_assoc: associative P3 critical friction, or None. Kept for
             backward compatibility; it equals mu_achievable, the
             verified-feasible bracket bound (the smallest friction at which
             feasibility was verified). Achievable, not ordered against the
             true required friction.
    mu_achievable: the smallest friction verified feasible in the
             associative model (equal to mu_critical_assoc). None when no
             critical friction is reported.
    mu_lower_verified: the largest friction verified infeasible in the
             associative model (the lower bracket bound), or None. For a 2D
             exact cone it obeys mu_lower_verified <= mu*_assoc <= mu_true,
             so it lower-bounds the true required friction. This is the
             endpoint physical_bound_direction describes.
    mechanism: (N, 3) in 2D (vx, vz, wy) or (N, 6) in 3D, virtual twist
             per block on infeasibility, the validated Farkas ray
             normalized, or None.
    cone_model: the cone model used, "linear2d" | "pyramid" | "socp".
    constitutive_model: "associative" for every solver in this library.
    physical_bound_direction: the direction, relative to true Coulomb
             capacity, of the VERIFIED bracket endpoint (lambda_upper_verified
             for P2, mu_lower_verified for P3), not of the achievable
             headline factor. None when no such verified endpoint is
             reported. "upper" for a linear 2D uncensored P2 upper endpoint
             (lambda_upper_verified >= lambda_true). "lower" for a linear 2D
             P3 lower endpoint (mu_lower_verified <= mu_true). "unknown" for
             an inscribed pyramid (associative overestimate combined with a
             cone underestimate; no ordering) and for a censored or
             uncertified P2 factor (only the achievable value is reported,
             which is unordered vs true even in 2D). P0 and P4 report no
             capacity factor and set None.
    trajectory: per-iteration record for P5, or None.
    info:    solver diagnostics (iterations, convergence flags, certificate
             summaries, bound tag).
    """

    status: str
    margin: float
    forces: np.ndarray | None = None
    lambda_assoc: float | None = None
    lambda_achievable: float | None = None
    lambda_upper_verified: float | None = None
    mu_critical_assoc: float | None = None
    mu_achievable: float | None = None
    mu_lower_verified: float | None = None
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
