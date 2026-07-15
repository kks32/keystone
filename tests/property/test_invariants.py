"""Property invariants for the solve path (CLAUDE.md Section 5, PLAN.md C2).

These check the symmetries the charter names: reciprocity under block
relabeling, invariance under rigid motion that preserves gravity, scale
invariance of the dimensionless factors, subdivision monotonicity, and
reorder determinism. Seeds are fixed through a committed hypothesis
profile so the suite is reproducible bit for bit.

Rigid motion is restricted to transforms that leave gravity along -z and
the ground plane at z = 0. In 2D that means x translation only, since a
y rotation tilts the block. In 3D it means x and y translation plus a
rotation about z.
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from keystone import (
    FEASIBLE,
    Box,
    Tolerances,
    assemble,
    box_2d,
    build_assembly,
    solve_p0,
    solve_p2,
    solve_p2_exact,
    solve_p3,
)
from keystone.mechanics.cones import cone_matrix_2d

TOL = Tolerances()  # default eps_reg = 1e-12, matching the analytic gates

# Committed profile: deterministic examples, no per-example deadline.
settings.register_profile("keystone", deadline=None, derandomize=True)
settings.load_profile("keystone")

MAX_2D = 25
MAX_3D = 10

# Common padded shapes so each test compiles the JAX kernel once.
PAD_2D = dict(pad_blocks=3, pad_patches=3, pad_verts=2)
PAD_3D = dict(pad_blocks=3, pad_patches=3, pad_verts=8)
CUBE = np.array([0.5, 0.5, 0.5])


# --- scene strategies -------------------------------------------------------

@st.composite
def scene_2d(draw):
    """A 2 to 3 block vertical stack in the xz plane with random shear."""
    n = draw(st.integers(2, 3))
    widths = [draw(st.floats(0.6, 1.6)) for _ in range(n)]
    heights = [draw(st.floats(0.6, 1.6)) for _ in range(n)]
    mu = draw(st.floats(0.4, 1.0))
    boxes = []
    z = 0.0
    prev_h = None
    x = 0.0
    for k in range(n):
        z = heights[k] / 2.0 if k == 0 else z + prev_h / 2.0 + heights[k] / 2.0
        if k > 0:
            x = x + draw(st.floats(-0.3, 0.3)) * widths[k]
        boxes.append(box_2d(widths[k], heights[k], x, z))
        prev_h = heights[k]
    return boxes, mu


@st.composite
def scene_3d(draw):
    """A 2 to 3 unit-cube tower with random x and y shear per level."""
    n = draw(st.integers(2, 3))
    mu = draw(st.floats(0.4, 1.0))
    boxes = []
    x = 0.0
    y = 0.0
    for k in range(n):
        if k > 0:
            x = x + draw(st.floats(-0.2, 0.2))
            y = y + draw(st.floats(-0.2, 0.2))
        boxes.append(Box(CUBE, np.array([x, y, 0.5 + k])))
    return boxes, mu


def system_2d(boxes, mu):
    a = build_assembly(boxes, mu=mu, tol=TOL, dim=2, **PAD_2D)
    return a, assemble(a, TOL, cone="linear2d")


def system_3d(boxes, mu):
    a = build_assembly(boxes, mu=mu, tol=TOL, dim=3, **PAD_3D)
    return a, assemble(a, TOL, cone="pyramid", k=8)


def quat_mul(q1, q0):
    w1, x1, y1, z1 = q1
    w0, x0, y0, z0 = q0
    return np.array([
        w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0,
        w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0,
        w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0,
        w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0,
    ])


# --- reciprocity ------------------------------------------------------------

@settings(max_examples=MAX_2D)
@given(scene_2d())
def test_reciprocity_2d(scene):
    # Reversing the block list relabels node ids and flips the affected
    # normals. Every verdict and margin is unchanged bit for bit at the
    # JAX P0 level. The load factor is invariant too: the JAX bisection
    # carries a threshold-crossing residual of order 1e-7 near flat-margin
    # regions, so the load-factor invariant is checked on the deterministic
    # LP oracle, which reproduces lambda to LP precision under relabeling.
    boxes, mu = scene
    _, s = system_2d(boxes, mu)
    _, sr = system_2d(list(reversed(boxes)), mu)
    r = solve_p0(s, TOL)
    rr = solve_p0(sr, TOL)
    assert r.status == rr.status
    assert abs(r.margin - rr.margin) < 1e-9
    if r.status == FEASIBLE:
        lam = solve_p2_exact(s, TOL).lambda_assoc
        lam_r = solve_p2_exact(sr, TOL).lambda_assoc
        assert abs(lam - lam_r) < 1e-9


@settings(max_examples=MAX_3D)
@given(scene_3d())
def test_reciprocity_3d(scene):
    boxes, mu = scene
    _, s = system_3d(boxes, mu)
    _, sr = system_3d(list(reversed(boxes)), mu)
    r = solve_p0(s, TOL)
    rr = solve_p0(sr, TOL)
    assert r.status == rr.status
    assert abs(r.margin - rr.margin) < 1e-9
    if r.status == FEASIBLE:
        lam = solve_p2_exact(s, TOL).lambda_assoc
        lam_r = solve_p2_exact(sr, TOL).lambda_assoc
        assert abs(lam - lam_r) < 1e-9


# --- rigid transform --------------------------------------------------------

@settings(max_examples=MAX_2D)
@given(scene_2d(), st.floats(-5.0, 5.0))
def test_rigid_translation_2d(scene, dx):
    # A pure x translation moves the assembly along the ground without
    # changing gravity or the contact geometry. Verdict, margin, and load
    # factor are unchanged.
    boxes, mu = scene
    _, s = system_2d(boxes, mu)
    moved = [box_2d(b.half_extents[0] * 2.0, b.half_extents[2] * 2.0,
                    b.position[0] + dx, b.position[2]) for b in boxes]
    _, sm = system_2d(moved, mu)
    r = solve_p0(s, TOL)
    rm = solve_p0(sm, TOL)
    assert r.status == rm.status
    assert abs(r.margin - rm.margin) < 1e-9
    if r.status == FEASIBLE:
        lam = solve_p2(s, TOL, n_iter=42, lam_hi=16.0).lambda_assoc
        lam_m = solve_p2(sm, TOL, n_iter=42, lam_hi=16.0).lambda_assoc
        assert abs(lam - lam_m) < 1e-7


@settings(max_examples=MAX_3D)
@given(scene_3d(), st.floats(-4.0, 4.0), st.floats(-4.0, 4.0), st.floats(0.0, 3.0))
def test_rigid_motion_3d(scene, dx, dy, theta):
    # x and y translation plus a rotation about z preserves gravity and the
    # ground plane. The P0 verdict and margin are unchanged. The load factor
    # is not tested here: the +x live load direction is fixed in world space,
    # so rotating the structure changes the load relative to the structure.
    boxes, mu = scene
    _, s = system_3d(boxes, mu)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    qz = np.array([np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)])
    moved = [
        Box(b.half_extents, R @ b.position + np.array([dx, dy, 0.0]),
            quat_mul(qz, b.quat))
        for b in boxes
    ]
    _, sm = system_3d(moved, mu)
    r = solve_p0(s, TOL)
    rm = solve_p0(sm, TOL)
    assert r.status == rm.status
    assert abs(r.margin - rm.margin) < 1e-9


# --- uniform scaling --------------------------------------------------------

@settings(max_examples=MAX_2D)
@given(scene_2d(), st.floats(0.3, 3.0))
def test_scale_invariance_2d(scene, scale):
    # Uniform scaling of every length leaves the nondimensional system, and
    # therefore lambda* and mu*, unchanged. mu* is exercised with a permanent
    # 0.3 W horizontal load folded into the dead load so it is nonzero.
    boxes, mu = scene
    a, s = system_2d(boxes, mu)
    scaled_boxes = [
        box_2d(b.half_extents[0] * 2.0 * scale, b.half_extents[2] * 2.0 * scale,
               b.position[0] * scale, b.position[2] * scale)
        for b in boxes
    ]
    ascaled, ss = system_2d(scaled_boxes, mu)

    lam = solve_p2(s, TOL, n_iter=45, lam_hi=16.0)
    lam_s = solve_p2(ss, TOL, n_iter=45, lam_hi=16.0)
    assert lam.status == lam_s.status
    if lam.status == FEASIBLE:
        assert abs(lam.lambda_assoc - lam_s.lambda_assoc) < 1e-7

    P, V = a.n_patches, a.verts_per_patch
    g_of_mu = lambda m: cone_matrix_2d(jnp.full(P, m), P, V)  # noqa: E731
    s_load = dataclasses.replace(s, w_dead=s.w_dead + 0.3 * s.w_live)
    ss_load = dataclasses.replace(ss, w_dead=ss.w_dead + 0.3 * ss.w_live)
    mu_star = solve_p3(s_load, g_of_mu, TOL, n_iter=40)
    mu_star_s = solve_p3(ss_load, g_of_mu, TOL, n_iter=40)
    assert mu_star.status == mu_star_s.status
    if mu_star.mu_critical_assoc is not None and mu_star_s.mu_critical_assoc is not None:
        assert abs(mu_star.mu_critical_assoc - mu_star_s.mu_critical_assoc) < 1e-7


@settings(max_examples=MAX_3D)
@given(scene_3d(), st.floats(0.3, 3.0))
def test_scale_invariance_3d(scene, scale):
    boxes, mu = scene
    _, s = system_3d(boxes, mu)
    scaled_boxes = [
        Box(b.half_extents * scale, b.position * scale, b.quat) for b in boxes
    ]
    _, ss = system_3d(scaled_boxes, mu)
    lam = solve_p2(s, TOL, n_iter=45, lam_hi=16.0)
    lam_s = solve_p2(ss, TOL, n_iter=45, lam_hi=16.0)
    assert lam.status == lam_s.status
    if lam.status == FEASIBLE:
        assert abs(lam.lambda_assoc - lam_s.lambda_assoc) < 1e-7


# --- subdivision monotonicity ----------------------------------------------

@settings(max_examples=MAX_2D)
@given(
    st.floats(0.6, 1.4), st.floats(0.6, 1.4), st.floats(-0.25, 0.25), st.floats(0.5, 1.0)
)
def test_subdivision_monotone_2d(w, h, e, mu):
    # Splitting each contact segment at its midpoint adds vertices without
    # changing the frames. A larger vertex set can only enlarge the
    # representable force set, so a feasible verdict stays feasible and the
    # elastic margin does not grow.
    boxes = [box_2d(1.0, h, 0.0, h / 2.0), box_2d(w, h, e * w, 1.5 * h)]
    a = build_assembly(boxes, mu=mu, tol=TOL, dim=2)
    s = assemble(a, TOL, cone="linear2d")
    r = solve_p0(s, TOL)
    assume(r.status == FEASIBLE)

    V = a.verts_per_patch
    assert V == 2
    blocks, norm, t1, t2, verts, mu_arr = [], [], [], [], [], []
    for p in range(a.n_patches):
        if not a.patch_mask[p]:
            continue
        v0, v1 = a.verts[p, 0], a.verts[p, 1]
        mid = 0.5 * (v0 + v1)
        for seg in ((v0, mid), (mid, v1)):
            blocks.append(a.patch_blocks[p])
            norm.append(a.normal[p])
            t1.append(a.t1[p])
            t2.append(a.t2[p])
            verts.append(np.stack(seg))
            mu_arr.append(a.mu[p])
    pn = len(blocks)
    a_sub = dataclasses.replace(
        a,
        patch_mask=np.ones(pn, dtype=bool),
        patch_blocks=np.array(blocks, dtype=np.int32),
        normal=np.array(norm),
        t1=np.array(t1),
        t2=np.array(t2),
        verts=np.array(verts),
        vert_mask=np.ones((pn, 2), dtype=bool),
        mu=np.array(mu_arr),
    )
    s_sub = assemble(a_sub, TOL, cone="linear2d")
    r_sub = solve_p0(s_sub, TOL)
    assert r_sub.status == FEASIBLE
    assert r_sub.margin <= r.margin + 1e-9


# --- determinism ------------------------------------------------------------

@settings(max_examples=MAX_2D)
@given(scene_2d())
def test_determinism_2d(scene):
    # The same scene built twice yields bit-identical A, w, and G.
    boxes, mu = scene
    _, s1 = system_2d(boxes, mu)
    _, s2 = system_2d(boxes, mu)
    assert np.array_equal(np.asarray(s1.A), np.asarray(s2.A))
    assert np.array_equal(np.asarray(s1.w_dead), np.asarray(s2.w_dead))
    assert np.array_equal(np.asarray(s1.w_live), np.asarray(s2.w_live))
    assert np.array_equal(np.asarray(s1.G), np.asarray(s2.G))


@settings(max_examples=MAX_3D)
@given(scene_3d())
def test_determinism_3d(scene):
    boxes, mu = scene
    _, s1 = system_3d(boxes, mu)
    _, s2 = system_3d(boxes, mu)
    assert np.array_equal(np.asarray(s1.A), np.asarray(s2.A))
    assert np.array_equal(np.asarray(s1.w_dead), np.asarray(s2.w_dead))
    assert np.array_equal(np.asarray(s1.G), np.asarray(s2.G))
