"""Rigid-block equilibrium for 2D and 3D masonry assemblies.

Correctness baseline is float64. JAX x64 mode is enabled at import,
before any array is created. float32 is an opt-in throughput mode,
allowed only where the agreement suite passes at that precision.
"""

import jax

jax.config.update("jax_enable_x64", True)

from .geometry import Assembly, Box, Tolerances, bbox_diagonal, box_2d, build_assembly  # noqa: E402
from .mechanics import EquilibriumSystem, assemble  # noqa: E402
from .solve import (  # noqa: E402
    FEASIBLE,
    INFEASIBLE,
    NO_CONVERGE,
    Result,
    SolverOptions,
    margin_batch,
    solve_p0,
    solve_p0_exact,
    solve_p2,
    solve_p2_exact,
    solve_p3,
    solve_p4,
    solve_p4_batch,
)

__version__ = "0.0.1"

__all__ = [
    "Assembly",
    "Box",
    "EquilibriumSystem",
    "FEASIBLE",
    "INFEASIBLE",
    "NO_CONVERGE",
    "Result",
    "SolverOptions",
    "Tolerances",
    "assemble",
    "bbox_diagonal",
    "box_2d",
    "build_assembly",
    "margin_batch",
    "solve_p0",
    "solve_p0_exact",
    "solve_p2",
    "solve_p2_exact",
    "solve_p3",
    "solve_p4",
    "solve_p4_batch",
]
