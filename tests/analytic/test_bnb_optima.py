"""Regression tests for branch-and-bound certified grid optima.

Provenance. Each optimum here was certified by examples/certify_overhang.py
(keystone.search.bnb), which runs best-first branch and bound over the
prefix-feasible cube-stacking lattice with an admissible bound and reports a
proven grid optimum when the frontier maximum bound drops to the incumbent.
See src/keystone/search/bnb.py for the bound derivation and the closed-list
transposition argument.

These tests replay each certified optimal build order through the host
pipeline (build_assembly + assemble + solve_p0) and pin two facts: every
prefix is statically feasible, and the rightmost cube edge equals the
certified optimum. Placement-mode optima (drop, slide) are additionally
re-checked step by step with bnb.sequence_reachable in their recorded
order. One short live n=3 static run pins the pre-reachability node counts
bitwise; everything else is a replay. Only certified optima are archived
here; intervals from budget-limited runs are recorded in out/search only.

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

from keystone import (
    FEASIBLE,
    SolverOptions,
    Tolerances,
    assemble,
    box_2d,
    build_assembly,
    solve_p0,
)
from keystone.search import bnb

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


# Certified optima under placement-reachability modes, as
# (n, dx, placement, optimum, sequence). Same provenance as CERTIFIED
# (examples/certify_overhang.py with --placement), same replay discipline,
# plus a per-step reachability re-check in the recorded order. The static
# optima above are the placement="static" baselines.
CERTIFIED_PLACEMENT = [
    # n=3, dx=1/12, drop: the static optimum 5/6 survives. Its staircase
    # build is top-down clear at every step, so the crane constraint is
    # free at n=3 on this grid.
    (3, 1.0 / 12.0, "drop", 5.0 / 6.0, [(0, -4), (1, 0), (2, 4)]),
    # n=4, dx=1/12, drop: optimum 1. The 5/4 clamp needs its reacher slid
    # under the layer-2 bridge, which a crane cannot do, and no drop-legal
    # order rescues 5/4: the reacher placed early is statically infeasible.
    # The certified price of drop-only construction is 1/4 block width,
    # 3 grid steps. The optimal drop design is two towers of two: the
    # right tower's top cube reaches edge 1 and the left pair counters it.
    (4, 1.0 / 12.0, "drop", 1.0, [(0, 0), (0, -21), (1, -11), (1, 6)]),
    # n=4, dx=1/12, slide: the static optimum 5/4 survives with the same
    # clamp sequence. The final reacher enters from the right at layer 1;
    # the layer-2 bridge sits above its slide plane and never blocks it.
    (4, 1.0 / 12.0, "slide", 5.0 / 4.0, [(0, -1), (1, -7), (2, -2), (1, 9)]),
    # n=4, dx=1/24, drop: optimum 1, unchanged from the coarser grid, now
    # reached by a four-high staircase instead of two towers of two. The
    # crane price at this grid is 31/24 - 1 = 7/24 of a block width.
    (4, 1.0 / 24.0, "drop", 1.0, [(0, -8), (1, -5), (2, 1), (3, 12)]),
    # n=4, dx=1/12, slide_clear: optimum 1, equal to drop, same design.
    # Banning the zero-clearance under-bridge pass (which jams in rigid
    # body simulation) costs the whole corridor advantage: the executable
    # slide is worth no more than the crane on this grid. In particular
    # no slide_clear order reaches the MCTS 7/6 design's overhang.
    (4, 1.0 / 12.0, "slide_clear", 1.0, [(0, 0), (0, -21), (1, -11), (1, 6)]),
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


def test_placement_optima_replay():
    # Each placement-mode optimum is statically prefix-feasible on the host
    # pipeline, reachable step by step in its recorded order under its own
    # mode, and reaches exactly the certified overhang.
    for (n, dx, placement, optimum, sequence) in CERTIFIED_PLACEMENT:
        assert len(sequence) <= n
        for prefix in replay(sequence, dx):
            r = solve_scene(prefix)
            assert r.status == FEASIBLE, (n, dx, placement, len(prefix), r.status)
        ok, flags = bnb.sequence_reachable(n, dx, sequence, placement)
        assert ok, (n, dx, placement, flags)
        rightmost_edge = max(j * dx + 0.5 for (_, j) in sequence)
        assert abs(rightmost_edge - optimum) < 1e-9, (n, dx, placement)


def test_drop_price_at_n4():
    # The headline numbers: crane (drop-only) construction certifiably
    # costs 1/4 block width at n=4 on the 1/12 grid and 7/24 on the 1/24
    # grid; the drop optimum is exactly 1 on both. Slide costs nothing:
    # the static optimum's build order is slide-reachable. The executable
    # slide (slide_clear) is worth exactly the crane: the certified ladder
    # at dx=1/12 is static 5/4 = slide 5/4 > slide_clear 1 = drop 1.
    def opt(table, n, dx, placement=None):
        for row in table:
            if placement is None:
                (rn, rdx, ro, _s) = row
                if rn == n and abs(rdx - dx) < 1e-12:
                    return ro
            else:
                (rn, rdx, rp, ro, _s) = row
                if rn == n and rp == placement and abs(rdx - dx) < 1e-12:
                    return ro
        raise KeyError((n, dx, placement))

    dx12, dx24 = 1.0 / 12.0, 1.0 / 24.0
    assert abs(opt(CERTIFIED, 4, dx12) - 5.0 / 4.0) < 1e-12
    assert abs(opt(CERTIFIED_PLACEMENT, 4, dx12, "slide") - 5.0 / 4.0) < 1e-12
    assert opt(CERTIFIED_PLACEMENT, 4, dx12, "slide_clear") == 1.0
    assert opt(CERTIFIED_PLACEMENT, 4, dx12, "drop") == 1.0
    assert opt(CERTIFIED_PLACEMENT, 4, dx24, "drop") == 1.0
    price12 = opt(CERTIFIED, 4, dx12) - opt(CERTIFIED_PLACEMENT, 4, dx12, "drop")
    price24 = opt(CERTIFIED, 4, dx24) - opt(CERTIFIED_PLACEMENT, 4, dx24, "drop")
    assert abs(price12 - 1.0 / 4.0) < 1e-12
    assert abs(price24 - 7.0 / 24.0) < 1e-9


def test_static_clamp_final_step_blocked_under_drop():
    # The archived static optima at n=4 end with an under-bridge slide, so
    # their recorded orders fail the drop and slide_clear re-checks exactly
    # at the last step and pass the slide re-check at every step.
    for (n, dx, _optimum, sequence) in CERTIFIED:
        if n != 4:
            continue
        ok, flags = bnb.sequence_reachable(n, dx, sequence, "slide")
        assert ok, (n, dx, flags)
        for mode in ("drop", "slide_clear"):
            ok, flags = bnb.sequence_reachable(n, dx, sequence, mode)
            assert not ok, (n, dx, mode)
            assert flags[:-1] == [True] * (len(sequence) - 1), (n, dx, mode)


def test_bnb_static_short_run_matches_prechange_counts():
    # One live n=3 static run. The reachability modes were added behind a
    # compile-time flag; this pins the default path to the run recorded
    # before the flag existed, node for node and solve for solve.
    res = bnb.certify(3, 1.0 / 12.0, TOL, opts=SolverOptions(), progress=False)
    assert res.placement == "static"
    assert res.certified and res.stop_reason == "optimal"
    assert abs(res.optimum - 5.0 / 6.0) < 1e-9
    assert [tuple(a) for a in res.sequence] == [(0, -4), (1, 0), (2, 4)]
    assert res.nodes_expanded == 254
    assert res.nodes_generated == 1085
    assert res.qp_solves == 2002
    assert res.closed_size == 254
    assert res.host_verifications == 5
    assert res.host_verified
