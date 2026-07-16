"""Interior-point solver knobs, separate from feasibility tolerances.

solver_tol and max_iter tune the qpax primal-dual interior point. They
are not feasibility tolerances (those live in geometry.Tolerances) and
they never decide a verdict. Verdicts always come from recomputed
residuals compared against Tolerances (CLAUDE.md charter item 5).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SolverOptions:
    """qpax knobs.

    solver_tol:   KKT residual target for the interior point.
    max_iter:     interior-point iteration cap.
    target_kappa: complementarity smoothing for the differentiable path.
                  qpax computes gradients at a relaxed KKT point where the
                  complementarity product s . z is driven to target_kappa
                  rather than to zero (see qpax.implicit.diff_qp). The
                  default matches qpax (1e-3). A smaller value sharpens the
                  gradient toward the true (unrelaxed) sensitivity at the
                  cost of conditioning; it does not affect any verdict.
    """

    solver_tol: float = 1e-9
    max_iter: int = 100
    target_kappa: float = 1e-3

    def __post_init__(self):
        if not (self.solver_tol > 0.0):
            raise ValueError(f"solver_tol must be > 0, got {self.solver_tol!r}")
        if not (isinstance(self.max_iter, int) and self.max_iter >= 1):
            raise ValueError(f"max_iter must be an int >= 1, got {self.max_iter!r}")
        if not (self.target_kappa > 0.0):
            raise ValueError(f"target_kappa must be > 0, got {self.target_kappa!r}")
