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

    solver_tol: KKT residual target for the interior point.
    max_iter:   interior-point iteration cap.
    """

    solver_tol: float = 1e-9
    max_iter: int = 100

    def __post_init__(self):
        if not (self.solver_tol > 0.0):
            raise ValueError(f"solver_tol must be > 0, got {self.solver_tol!r}")
        if not (isinstance(self.max_iter, int) and self.max_iter >= 1):
            raise ValueError(f"max_iter must be an int >= 1, got {self.max_iter!r}")
