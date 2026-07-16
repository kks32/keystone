"""Replayable regression for the 1.25-block-width overhang discovery.

Provenance. Found by
examples/search_overhang_fast.py --n 6 --sims 50000 --seed 0 --batch 64
(dx = 1/24) on TACC Vista GH200, 2026-07-15. The counterweighted six-cube
stack reaches a rightmost cube edge of 1.25 block widths beyond the pedestal
edge, beating the harmonic six-block baseline sum_{k=1..6} 1/(2k) = 49/40 =
1.225. This test freezes the exact placements so the result stays replayable
without a GPU or a search run.

Scene (CLAUDE.md Section 4, 2D xz plane, gravity along -z). A pedestal
box_2d(6, 1, -3, 0.5) puts its right edge at x = 0 and its top at z = 1.
Unit cubes box_2d(1, 1, x, 1.5 + layer) stack above it. Friction mu = 0.7,
Tolerances() defaults, cone linear2d.

Two facts are pinned. First, prefix feasibility: every prefix of the build
order is statically feasible under P0. Second, ordering is load bearing:
placing the reacher (1, 0.75) directly on the lone base cube (0, 0.0),
skipping the counterweights, gives an infeasible three-box structure.
"""

import numpy as np

from keystone import (
    FEASIBLE,
    INFEASIBLE,
    Tolerances,
    assemble,
    box_2d,
    build_assembly,
    solve_p0,
)

TOL = Tolerances()
MU = 0.7

# Placements in build order as (layer, center_x). This is the order the
# search emitted; each prefix must be feasible.
PLACEMENTS = [
    (0, 0.0),
    (0, -2.125),
    (1, -0.5),
    (2, -5.0 / 24.0),
    (0, -13.0 / 12.0),
    (1, 0.75),
]


def pedestal():
    """Wide base. Right edge at x = 0, top face at z = 1."""
    return box_2d(6.0, 1.0, -3.0, 0.5)


def cube(layer, cx):
    """Unit cube at the given layer, center x = cx, center z = 1.5 + layer."""
    return box_2d(1.0, 1.0, cx, 1.5 + layer)


def solve_scene(cubes):
    """Solve P0 for the pedestal plus the given (layer, x) cubes."""
    boxes = [pedestal()] + [cube(layer, cx) for (layer, cx) in cubes]
    asm = build_assembly(boxes, mu=MU, tol=TOL, dim=2)
    system = assemble(asm, TOL, cone="linear2d")
    return solve_p0(system, TOL)


def test_every_prefix_is_feasible():
    # Each prefix of the build order must satisfy P0. This is the property
    # the sequence checker relies on: the structure can be built one cube at
    # a time without an intermediate collapse.
    for k in range(1, len(PLACEMENTS) + 1):
        r = solve_scene(PLACEMENTS[:k])
        assert r.status == FEASIBLE, (k, r.status, r.margin)


def test_full_structure_reaches_1p25():
    # The rightmost cube edge is max(center_x) + 0.5 = 0.75 + 0.5 = 1.25
    # block widths beyond the pedestal edge at x = 0.
    xs = [cx for (_, cx) in PLACEMENTS]
    rightmost_edge = max(xs) + 0.5
    assert rightmost_edge == 1.25
    # It beats the harmonic six-block overhang 49/40 = 1.225.
    harmonic6 = sum(1.0 / (2.0 * k) for k in range(1, 7))
    assert abs(harmonic6 - 49.0 / 40.0) < 1e-12
    assert rightmost_edge > harmonic6
    # The full structure is feasible with a margin inside the band.
    r = solve_scene(PLACEMENTS)
    assert r.status == FEASIBLE
    assert r.margin <= TOL.tol_feas


def test_greedy_order_counterexample_is_infeasible():
    # Ordering is load bearing. Placing the reacher (1, 0.75) directly on the
    # lone base cube (0, 0.0), without the counterweights the search laid
    # down first, gives an infeasible structure: the layer-1 cube overhangs
    # its single support and topples. The same set of cubes reached in the
    # full build order is feasible, so the sequence, not just the final set,
    # decides admissibility.
    r = solve_scene([(0, 0.0), (1, 0.75)])
    assert r.status == INFEASIBLE, (r.status, r.margin)
