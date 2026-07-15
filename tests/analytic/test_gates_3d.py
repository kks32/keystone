"""Analytic gate tests for the 3D solve path (CLAUDE.md M2, PLAN.md C2).

Boxes only, pyramid friction cones with k = 8 inscribed facets. The cone
is conservative: capacity along a polygon vertex direction is exact,
capacity along a facet-mid direction is short by cos(pi/k). The ground
patch frame puts t1 on +x, so a +x live load lands on a polygon vertex
and the slide capacity is exact. Rotating the load to 22.5 degrees lands
it on a facet mid and exposes the cos(pi/8) underestimate.

Conventions are CLAUDE.md Section 4. lambda_assoc is the associative
load factor, an upper estimate of the true Coulomb collapse factor.
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from keystone import (
    FEASIBLE,
    Box,
    Tolerances,
    assemble,
    box_2d,
    build_assembly,
    solve_p0,
    solve_p0_exact,
    solve_p2,
    solve_p2_exact,
)

TOL = Tolerances()
# The default eps_reg = 1e-12 keeps the P2 bisection in agreement with the
# exact LP. See tests/analytic/test_gates_2d.py for the reasoning.

CUBE = np.array([0.5, 0.5, 0.5])


def quat_z(angle):
    """Unit quaternion (w, x, y, z) for a rotation about the z axis."""
    return np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])


def ground_system(half, mu):
    """One cuboid resting on the ground, base centered at the origin."""
    boxes = [Box(np.asarray(half, dtype=float), np.array([0.0, 0.0, half[2]]))]
    a = build_assembly(boxes, mu=mu, tol=TOL, dim=3)
    return a, assemble(a, TOL, cone="pyramid", k=8)


def test_ground_patch_frame_is_x():
    # The frame rule sends t1 to +x and t2 to +y on a horizontal patch, so
    # a +x live load aligns with a k = 8 polygon vertex. This underpins the
    # exact-capacity claim in the tilt test below.
    a, _ = ground_system(CUBE, 0.5)
    p = int(np.flatnonzero(np.asarray(a.patch_mask))[0])
    assert np.allclose(a.normal[p], [0.0, 0.0, 1.0], atol=1e-12)
    assert np.allclose(a.t1[p], [1.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(a.t2[p], [0.0, 1.0, 0.0], atol=1e-12)


def test_cuboid_tilt_axis_aligned():
    # Under +x load a cuboid slides at lambda = mu and topples at
    # lambda = hx / hz, so lambda* = min(hx/hz, mu). The +x load is a
    # polygon-vertex direction, so the pyramid slide capacity is exact.
    # Cube: hx/hz = 1. Slab (2 x 1 x 1): hx/hz = 1.0 / 0.5 = 2.
    cases = [("cube", CUBE, 1.0), ("slab", np.array([1.0, 0.5, 0.5]), 2.0)]
    for name, half, bh in cases:
        for mu in [0.3, 1.5]:
            _, s = ground_system(half, mu)
            r = solve_p2(s, TOL, n_iter=50, lam_hi=8.0)
            expected = min(bh, mu)
            assert r.status == FEASIBLE
            assert abs(r.lambda_assoc - expected) < 1e-5, (name, mu)


def test_cuboid_slide_facet_mid_is_conservative():
    # Rotating the box about z does not rotate the cone, because t1 comes
    # from the global-x frame rule, not the box orientation. So rotate the
    # live load itself to 22.5 degrees in the (t1, t2) plane, a facet mid
    # for k = 8. The inscribed pyramid underestimates slide capacity there
    # by cos(pi/8), so the slide-governed factor drops to mu cos(pi/8).
    mu = 0.3
    _, s = ground_system(CUBE, mu)
    angle = np.pi / 8.0
    w_live = np.asarray(s.w_live).copy()
    n_blocks = w_live.shape[0] // 6  # 6 rows per block in 3D
    for bi in range(n_blocks):
        fx = w_live[6 * bi + 0]  # +x load per unit weight
        w_live[6 * bi + 0] = fx * np.cos(angle)  # Fx row
        w_live[6 * bi + 1] = fx * np.sin(angle)  # Fy row
    s_rot = dataclasses.replace(s, w_live=jnp.asarray(w_live))
    r = solve_p2(s_rot, TOL, n_iter=50, lam_hi=8.0)
    expected = mu * np.cos(np.pi / 8.0)
    assert r.status == FEASIBLE
    assert abs(r.lambda_assoc - expected) < 1e-5


def _stack_positions(heights):
    """Face-to-face z centers for a vertical stack of the given heights."""
    zc = []
    prev_h = None
    z = 0.0
    for k, h in enumerate(heights):
        z = h / 2.0 if k == 0 else z + prev_h / 2.0 + h / 2.0
        zc.append(z)
        prev_h = h
    return zc


def test_extrusion_matches_2d():
    # A 2D stack and its unit-depth 3D extrusion answer the same question.
    # The axes stay aligned, so the pyramid conservatism never bites and the
    # verdicts match. lambda agrees within 2e-3 relative. The residual gap
    # (order 1e-4 relative) comes from the differing nondimensional length
    # L, the 4-vertex versus 2-vertex contact discretization, and the
    # elastic-margin bisection offset, not from the cone model.
    configs = [
        ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], 0.6),
        ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.2, 0.35], 0.6),
        ([1.0, 1.0], [1.0, 1.0], [0.0, 0.4], 0.3),
    ]
    for widths, heights, offsets, mu in configs:
        zc = _stack_positions(heights)
        boxes2 = [
            box_2d(w, h, x, z)
            for w, h, x, z in zip(widths, heights, offsets, zc)
        ]
        s2 = assemble(
            build_assembly(boxes2, mu=mu, tol=TOL, dim=2), TOL, cone="linear2d"
        )
        boxes3 = [
            Box(np.array([w / 2.0, 0.5, h / 2.0]), np.array([x, 0.0, z]))
            for w, h, x, z in zip(widths, heights, offsets, zc)
        ]
        s3 = assemble(
            build_assembly(boxes3, mu=mu, tol=TOL, dim=3), TOL, cone="pyramid", k=8
        )
        v2 = solve_p0(s2, TOL).status
        v3 = solve_p0(s3, TOL).status
        assert v2 == v3, offsets
        lam2 = solve_p2(s2, TOL, n_iter=50, lam_hi=8.0).lambda_assoc
        lam3 = solve_p2(s3, TOL, n_iter=50, lam_hi=8.0).lambda_assoc
        assert abs(lam2 - lam3) / max(lam2, 1e-9) < 2e-3, offsets


def test_three_cube_tower_topple():
    # Three unit cubes, mu = 0.84. The governing collapse is the minimum
    # over joints of (half width) / (com height above the joint). Joint at
    # z = 0 carries three cubes with com at z = 1.5: ratio 0.5 / 1.5 = 1/3.
    # Joint at z = 1 carries two cubes, com 1.0 above it: 0.5 / 1.0 = 1/2.
    # Top cube: 0.5 / 0.5 = 1. Minimum is 1/3. Slide would need mu <= 1/3,
    # but mu = 0.84 > 1/3, so toppling governs and lambda* = 1/3.
    boxes = [Box(CUBE, np.array([0.0, 0.0, 0.5 + k])) for k in range(3)]
    s = assemble(
        build_assembly(boxes, mu=0.84, tol=TOL, dim=3), TOL, cone="pyramid", k=8
    )
    r = solve_p2(s, TOL, n_iter=50, lam_hi=8.0)
    assert r.status == FEASIBLE
    assert abs(r.lambda_assoc - 1.0 / 3.0) < 1e-5


def test_rotated_tower_solves():
    # A three-cube tower with the middle cube turned 45 degrees about z. Its
    # shared faces clip to octagonal patches (8 vertices). The tower stands
    # under gravity and carries a strictly positive load factor.
    boxes = [
        Box(CUBE, np.array([0.0, 0.0, 0.5])),
        Box(CUBE, np.array([0.0, 0.0, 1.5]), quat_z(np.pi / 4.0)),
        Box(CUBE, np.array([0.0, 0.0, 2.5])),
    ]
    a = build_assembly(boxes, mu=0.6, tol=TOL, dim=3)
    s = assemble(a, TOL, cone="pyramid", k=8)
    # The 45-degree joints really are octagons.
    vert_counts = sorted(
        int(a.vert_mask[p].sum())
        for p in range(a.n_patches)
        if a.patch_mask[p]
    )
    assert vert_counts == [4, 8, 8]
    r0 = solve_p0(s, TOL)
    assert r0.status == FEASIBLE
    r2 = solve_p2(s, TOL, n_iter=40, lam_hi=8.0)
    assert r2.lambda_assoc > 0.0


def cube_tower_3d(rng):
    """A random 2 to 4 cube tower with random x and y shear per level."""
    n = int(rng.integers(2, 5))
    mu = float(rng.uniform(0.3, 1.0))
    boxes = []
    x = 0.0
    y = 0.0
    for k in range(n):
        if k > 0:
            x = x + float(rng.uniform(-0.6, 0.6))
            y = y + float(rng.uniform(-0.6, 0.6))
        boxes.append(Box(CUBE, np.array([x, y, 0.5 + k])))
    return boxes, mu


def test_oracle_agreement_3d():
    # 60 random cube towers. The JAX P0 verdict matches the HiGHS oracle
    # away from the feasibility band, and the JAX P2 load factor matches the
    # exact LP on feasible cases. Both cones are the same pyramid, so this
    # is a solver-versus-solver check at fixed cone model.
    rng = np.random.default_rng(1)
    band = 0
    disagreements = 0
    lam_max_err = 0.0
    for _ in range(60):
        boxes, mu = cube_tower_3d(rng)
        a = build_assembly(
            boxes, mu=mu, tol=TOL, dim=3,
            pad_blocks=4, pad_patches=4, pad_verts=8,
        )
        s = assemble(a, TOL, cone="pyramid", k=8)
        rj = solve_p0(s, TOL)
        re = solve_p0_exact(s, TOL)
        m = rj.margin
        if not np.isfinite(m) or (TOL.tol_feas < m < 10.0 * TOL.tol_feas):
            band += 1
            continue
        if rj.status != re.status:
            disagreements += 1
        if rj.status == FEASIBLE and re.status == FEASIBLE:
            r2 = solve_p2(s, TOL, n_iter=45, lam_hi=16.0)
            r2e = solve_p2_exact(s, TOL, lam_hi=16.0)
            if (
                r2.status == FEASIBLE
                and r2e.status == FEASIBLE
                and r2e.lambda_assoc < 16.0 - 1e-6
            ):
                lam_max_err = max(
                    lam_max_err,
                    abs(r2.lambda_assoc - r2e.lambda_assoc),
                )
    assert disagreements == 0
    assert band < 10  # empirically zero for this seed
    assert lam_max_err < 1e-4


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
