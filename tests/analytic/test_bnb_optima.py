"""Regression tests for branch-and-bound certified grid optima.

Provenance. Each optimum here was certified by examples/certify_overhang.py
(keystone.search.bnb), which runs best-first branch and bound over the
prefix-feasible cube-stacking lattice with an admissible bound and reports a
proven grid optimum when the frontier maximum bound drops to the incumbent.
See src/keystone/search/bnb.py for the bound derivation and the closed-list
transposition argument.

These tests do not re-run branch and bound. They replay each certified
optimal build order through the host pipeline (build_assembly + assemble +
solve_p0) and pin two facts: every prefix is statically feasible, and the
rightmost cube edge equals the certified optimum. Only certified optima are
archived here; intervals from budget-limited runs are recorded in
out/search only.

Scene (CLAUDE.md Section 4, 2D xz plane, gravity along -z). A pedestal
box_2d(6, 1, -3, 0.5) puts its right edge at x = 0 and its top at z = 1.
Unit cubes box_2d(1, 1, j*dx, 1.5 + layer) stack above it on a dx grid.
Friction mu = 0.7, Tolerances() defaults, cone linear2d.

Literature note. The unconstrained Paterson-Zwick maximum overhang grows
like a small multiple of the block width and is not the quantity certified
here. This scene adds a wide pedestal, a fixed grid, prefix feasibility, and
strict static feasibility, so its optimum sits below the idealized harmonic
stack sum_{k=1..n} 1/(2k): the harmonic stack is only marginally stable and
fails strict P0. The comparison values below are context, not the assertion.
"""

import numpy as np

from keystone import FEASIBLE, Tolerances, assemble, box_2d, build_assembly, solve_p0

TOL = Tolerances()
MU = 0.7


# Certified grid optima as (n, dx, optimum, sequence). sequence is the
# certified prefix-feasible build order as (layer, grid_index_j); center x is
# j * dx. Each entry was proven optimal, not merely found.
CERTIFIED = [
    # n=3, dx=1/12: optimum 5/6 = 0.8333.., one grid step below harmonic(3)
    # = 11/12. Counterweighting does not beat the harmonic overhang at n=3 on
    # this grid; the harmonic stack itself is only marginally stable.
    (3, 1.0 / 12.0, 5.0 / 6.0, [(0, -4), (1, 0), (2, 4)]),
    # n=4, dx=1/12: optimum 5/4 = 1.25. A counterweighted stack: a base cube,
    # two counterweights up and left, and a reacher resting on the base with
    # the minimum 2*dx overlap. It beats harmonic(4) = 25/24 by 2.5 grid
    # steps and beats the batched MCTS result 7/6 by exactly one 1/12 step,
    # so 7/6 was not optimal.
    (4, 1.0 / 12.0, 5.0 / 4.0, [(0, -1), (1, -7), (2, -2), (1, 9)]),
    # n=4, dx=1/24: optimum 31/24 = 1.2917. The same counterweighted shape as
    # the 1/12 optimum, with the reacher pushed one finer step further right;
    # refining the grid gains exactly one 1/24 step over 5/4 = 30/24.
    (4, 1.0 / 24.0, 31.0 / 24.0, [(0, -2), (1, -14), (2, -4), (1, 19)]),
]


def pedestal():
    """Wide base. Right edge at x = 0, top face at z = 1."""
    return box_2d(6.0, 1.0, -3.0, 0.5)


def cube(layer, cx):
    """Unit cube at the given layer, center x = cx, center z = 1.5 + layer."""
    return box_2d(1.0, 1.0, cx, 1.5 + layer)


def solve_scene(cubes):
    """Solve P0 for the pedestal plus the given (layer, center_x) cubes."""
    boxes = [pedestal()] + [cube(layer, cx) for (layer, cx) in cubes]
    asm = build_assembly(boxes, mu=MU, tol=TOL, dim=2)
    system = assemble(asm, TOL, cone="linear2d")
    return solve_p0(system, TOL)


def replay(sequence, dx):
    """Prefixes of a build order as lists of (layer, center_x)."""
    placed = []
    for (L, j) in sequence:
        placed.append((int(L), j * dx))
        yield list(placed)


def test_certified_optima_are_prefix_feasible():
    # Every prefix of every certified build order satisfies P0. This is the
    # property branch and bound relied on: the structure builds one cube at a
    # time with no intermediate collapse.
    for (n, dx, optimum, sequence) in CERTIFIED:
        assert len(sequence) <= n
        for prefix in replay(sequence, dx):
            r = solve_scene(prefix)
            assert r.status == FEASIBLE, (n, dx, len(prefix), r.status, r.margin)


def test_certified_optima_reach_their_value():
    # The rightmost cube edge equals the certified optimum, and the full
    # structure is feasible with a margin inside the band.
    for (n, dx, optimum, sequence) in CERTIFIED:
        rightmost_edge = max(j * dx + 0.5 for (_, j) in sequence)
        assert abs(rightmost_edge - optimum) < 1e-9, (n, dx, rightmost_edge, optimum)
        r = solve_scene([(int(L), j * dx) for (L, j) in sequence])
        assert r.status == FEASIBLE, (n, dx, r.status, r.margin)
        assert r.margin <= TOL.tol_feas


def test_n3_dx12_optimum_is_five_sixths():
    # Pin the anchor value exactly. 5/6 sits one 1/12 grid step below the
    # idealized harmonic overhang 11/12 = sum_{k=1..3} 1/(2k).
    n, dx, optimum, sequence = CERTIFIED[0]
    assert (n, optimum) == (3, 5.0 / 6.0)
    harmonic3 = sum(1.0 / (2.0 * k) for k in range(1, 4))
    assert abs(harmonic3 - 11.0 / 12.0) < 1e-12
    assert optimum < harmonic3
    assert abs((harmonic3 - optimum) - dx) < 1e-9
