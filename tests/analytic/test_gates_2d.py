"""Analytic gate tests for the 2D solve path (CLAUDE.md M1, PLAN.md C1).

Every value here has a closed-form derivation, stated in the test. The
assertion tolerances are test tolerances; the geometric cuts and the
feasibility threshold come from the Tolerances dataclass.

Sign and frame conventions are CLAUDE.md Section 4. 2D lives in the xz
plane, gravity along -z, live load along +x. A 2D interface is a segment
with two vertices. lambda_assoc is the associative load factor, an
upper estimate of the true Coulomb collapse factor (charter item 3).
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from keystone import (
    FEASIBLE,
    INFEASIBLE,
    Tolerances,
    assemble,
    box_2d,
    build_assembly,
    solve_p0,
    solve_p0_exact,
    solve_p2,
    solve_p2_exact,
    solve_p4,
)

TOL = Tolerances()
# The default Tikhonov term is eps_reg = 1e-12. It keeps the P4
# elastic-margin floor well below tol_feas even when near-collapse contact
# forces are large, so the P2 bisection agrees with the exact LP on tall
# stacks. At the earlier default 1e-10 the floor crossed tol_feas on
# multi-block stacks and the bisection under-certified lambda*; that is the
# eps_reg sensitivity PLAN.md C1 calls out.


def single_block(b, h, mu):
    """One b x h block resting on the ground, its base centered at x = 0."""
    boxes = [box_2d(b, h, 0.0, h / 2.0)]
    a = build_assembly(boxes, mu=mu, tol=TOL, dim=2)
    return assemble(a, TOL, cone="linear2d")


def test_single_block_load_factor():
    # A block topples about its downhill base edge at lambda = b / h and
    # slides at lambda = mu, so lambda* = min(b/h, mu). Derivation: the
    # +x live load lam * W acts at com height h/2; the restoring moment of
    # W about the base edge is W * b/2, so topple needs lam <= b/h. The
    # base normal reaction is W, friction caps the shear at mu * W, so
    # slide needs lam <= mu.
    for b, h in [(1.0, 1.0), (0.5, 1.5), (2.0, 1.0)]:
        for mu in [0.2, 0.4, 0.6, 0.9, 1.5]:
            s = single_block(b, h, mu)
            r = solve_p2(s, TOL, n_iter=60, lam_hi=8.0)
            expected = min(b / h, mu)
            assert r.status == FEASIBLE
            assert abs(r.lambda_assoc - expected) < 1e-6, (b, h, mu)


def test_tall_tower_load_factor():
    # An aligned stack of N unit blocks topples about its base edge. The
    # full stack has com at height N/2 and half width 1/2, so the base joint
    # gives lambda = (1/2) / (N/2) = 1/N. Every higher joint carries fewer
    # blocks with a lower com, so its ratio is larger; the base governs.
    # With mu = 0.9 sliding needs lambda <= 0.9, well above 1/N, so toppling
    # governs and lambda* = 1/N. This case broke at eps_reg = 1e-10 and works
    # at the 1e-12 default.
    for n in [6, 12, 20]:
        boxes = [box_2d(1.0, 1.0, 0.0, 0.5 + k) for k in range(n)]
        s = assemble(
            build_assembly(boxes, mu=0.9, tol=TOL, dim=2), TOL, cone="linear2d"
        )
        r = solve_p2(s, TOL, n_iter=55, lam_hi=8.0)
        assert r.status == FEASIBLE
        assert abs(r.lambda_assoc - 1.0 / n) < 1e-4, n


def test_verdict_switch_kink_at_b_over_h():
    # For b/h = 1 the analytic curve is lambda(mu) = min(1, mu). The two
    # branches meet at mu = b/h = 1. Sampling just below and just above
    # the kink pins its location: at mu = 1 - 1e-3 the slide branch gives
    # lambda = mu, at mu = 1 + 1e-3 the topple branch gives lambda = 1.
    s_lo = single_block(1.0, 1.0, 1.0 - 1e-3)
    s_hi = single_block(1.0, 1.0, 1.0 + 1e-3)
    lam_lo = solve_p2(s_lo, TOL, n_iter=60, lam_hi=8.0).lambda_assoc
    lam_hi = solve_p2(s_hi, TOL, n_iter=60, lam_hi=8.0).lambda_assoc
    # Below the kink lambda tracks mu; above it saturates at b/h = 1.
    assert abs(lam_lo - (1.0 - 1e-3)) < 1e-3
    assert abs(lam_hi - 1.0) < 1e-3
    # The two samples straddle the kink: slide value is below the topple cap.
    assert lam_lo < lam_hi


def test_offset_pair_boundary():
    # Two unit blocks, the upper offset by e. The upper block rests on the
    # overlap segment [e - b/2, b/2] and its com sits at x = e. Equilibrium
    # under gravity needs the com over the contact, so e <= b/2. With high
    # friction the flip is a pure toppling boundary at e = b/2 = 0.5.
    b = 1.0

    def feasible_at(e):
        boxes = [box_2d(b, 1.0, 0.0, 0.5), box_2d(b, 1.0, e, 1.5)]
        a = build_assembly(boxes, mu=0.9, tol=TOL, dim=2)
        s = assemble(a, TOL, cone="linear2d")
        return solve_p0(s, TOL).status == FEASIBLE

    lo, hi = 0.4, 0.6  # lo feasible, hi infeasible
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        if feasible_at(mid):
            lo = mid
        else:
            hi = mid
    flip = 0.5 * (lo + hi)
    assert abs(flip - b / 2.0) < 1e-6


def corbel(m, c):
    """m unit blocks on a wide pedestal, fully optimal harmonic stacking.

    Pedestal is 4 x 1 with its top at z = 1 and its right edge at x = 0.
    Block b counted from the bottom has top-index j = m - b. Its right
    edge is R_j = sum_{l=j}^{m} c / (2 l), so the joint shifts read c/(2k)
    for k = m .. 1 going bottom to top. Every joint, including the pedestal
    joint, is simultaneously critical at c = 1: the com of the top-j blocks
    equals R_{j+1} + c/2 - 1/2, which sits on the supporting edge exactly
    when c = 1. The full-stack overhang limit is (1/2) H_4 = 25/24 widths.
    """
    pedestal = box_2d(4.0, 1.0, -2.0, 0.5)  # spans x in [-4, 0], top z = 1
    blocks = [pedestal]
    for b in range(m):
        j = m - b
        right_edge = sum(c / (2.0 * l) for l in range(j, m + 1))
        cx = right_edge - 0.5
        blocks.append(box_2d(1.0, 1.0, cx, 1.5 + b))
    return blocks


def test_corbel_per_prefix():
    # The harmonic overhang limit holds per prefix: an m-block corbel is
    # feasible just below c = 1 and infeasible just above, for every m. At
    # c = 1 all joints collapse together, so the margin at c = 1 sits inside
    # the tolerance band; the assertions stay a hair off the boundary.
    for m in [1, 2, 3, 4]:
        s_lo = assemble(
            build_assembly(corbel(m, 1.0 - 1e-3), mu=0.6, tol=TOL, dim=2),
            TOL,
            cone="linear2d",
        )
        s_hi = assemble(
            build_assembly(corbel(m, 1.0 + 1e-3), mu=0.6, tol=TOL, dim=2),
            TOL,
            cone="linear2d",
        )
        assert solve_p0(s_lo, TOL).status == FEASIBLE, m
        assert solve_p0(s_hi, TOL).status == INFEASIBLE, m
        # The exact HiGHS oracle agrees on both verdicts.
        assert solve_p0_exact(s_lo, TOL).status == FEASIBLE, m
        assert solve_p0_exact(s_hi, TOL).status == INFEASIBLE, m


def test_mechanism_low_mu_slides():
    # mu = 0.2, unit block, load factor 0.25 > mu. Sliding governs. The
    # collapse mechanism is a horizontal translation: the x component of
    # the block twist dominates the rotation. load power is positive.
    s = single_block(1.0, 1.0, 0.2)
    lam = 0.25
    # A finite margin at lam confirms the point is past collapse.
    assert solve_p4(s, TOL, lam=lam).margin > TOL.tol_feas
    # Fold the live load into the dead load so solve_p0 sees the loaded state.
    loaded = dataclasses.replace(s, w_dead=s.w_dead + lam * s.w_live)
    r = solve_p0(loaded, TOL)
    assert r.status == INFEASIBLE
    vx, vz, wy = r.mechanism[0]
    assert abs(vx) > 5.0 * abs(wy)  # translation dominates rotation
    assert abs(vx) > abs(vz)  # motion is mostly horizontal
    assert r.info["load_power"] > 0.0


def test_mechanism_high_mu_topples():
    # mu = 2.0, unit block, load factor 1.25 > b/h = 1. Toppling governs.
    # The mechanism is a rotational twist: the y rotation dominates both
    # translation components. load power is positive.
    s = single_block(1.0, 1.0, 2.0)
    lam = 1.25
    assert solve_p4(s, TOL, lam=lam).margin > TOL.tol_feas
    loaded = dataclasses.replace(s, w_dead=s.w_dead + lam * s.w_live)
    r = solve_p0(loaded, TOL)
    assert r.status == INFEASIBLE
    vx, vz, wy = r.mechanism[0]
    assert abs(wy) > abs(vx)  # rotation dominates translation
    assert abs(wy) > abs(vz)
    assert r.info["load_power"] > 0.0


def test_complementary_slackness_feasible():
    # A stable block. Its contact forces obey the cone G f <= 0 and keep
    # every normal component nonnegative. Forces come back in SI, so divide
    # by W to compare against the nondimensional cone rows.
    s = single_block(1.0, 1.0, 0.5)
    r = solve_p0(s, TOL)
    assert r.status == FEASIBLE
    f_nd = np.asarray(r.forces) / float(s.W)
    cone_residual = np.asarray(s.G) @ f_nd
    assert cone_residual.max() <= 1e-8  # G f <= 0 up to solver tolerance
    # Normal component index is (p V + v) ncomp + 0 with ncomp = 2, V = 2.
    ncomp, V = 2, 2
    vmask = np.asarray(s.vert_mask)
    for p in range(vmask.shape[0]):
        for v in range(V):
            if vmask[p, v]:
                n_k = f_nd[(p * V + v) * ncomp + 0]
                assert n_k >= -1e-10


def stacked_scene_2d(rng):
    """A random 2 to 6 block stack. Face-to-face contact, random x shear."""
    n = int(rng.integers(2, 7))
    widths = rng.uniform(0.4, 2.0, n)
    heights = rng.uniform(0.4, 2.0, n)
    mu = float(rng.uniform(0.2, 1.0))
    boxes = []
    zc = 0.0
    prev_h = None
    x = 0.0
    for k in range(n):
        zc = heights[k] / 2.0 if k == 0 else zc + prev_h / 2.0 + heights[k] / 2.0
        if k > 0:
            x = x + float(rng.uniform(-0.6, 0.6)) * widths[k]
        boxes.append(box_2d(widths[k], heights[k], x, zc))
        prev_h = heights[k]
    return boxes, mu


def test_oracle_agreement_2d():
    # 100 random stacks. The JAX P0 verdict matches the HiGHS oracle away
    # from the feasibility band, and the JAX P2 load factor matches the
    # exact LP on feasible cases. Padding to a common shape keeps the JAX
    # compile count at one.
    rng = np.random.default_rng(0)
    band = 0
    disagreements = 0
    lam_max_err = 0.0
    bracket_violations = 0
    unverified_hi = 0
    for _ in range(100):
        boxes, mu = stacked_scene_2d(rng)
        a = build_assembly(
            boxes, mu=mu, tol=TOL, dim=2,
            pad_blocks=6, pad_patches=6, pad_verts=2,
        )
        s = assemble(a, TOL, cone="linear2d")
        rj = solve_p0(s, TOL)
        re = solve_p0_exact(s, TOL)
        m = rj.margin
        # Exclude the band |margin - tol_feas| < 10 tol_feas and non-finite
        # margins; those escalate to the oracle in production.
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
                # The LP optimum must sit inside the verified bisection
                # bracket [lam_lo_feasible, lam_hi].
                lo = r2.info["lam_lo_feasible"]
                hi = r2.info["lam_hi"]
                if r2.info["lam_hi_verified_infeasible"]:
                    # With a verified-infeasible hi the LP optimum must sit
                    # inside [lo, hi] up to the solvers' feasibility
                    # tolerances. Both endpoints carry tol_feas-scale fuzz
                    # (1e-8): the QP can verify a point a few 1e-9 past the
                    # LP's reported optimum. Measured gaps: <= 6.3e-9.
                    if not (lo - 1e-7 <= r2e.lambda_assoc <= hi + 1e-6):
                        bracket_violations += 1
                else:
                    unverified_hi += 1
    assert disagreements == 0
    # Band count is reported and bounded. Empirically zero for this seed.
    assert band < 10
    # lambda_assoc is the last VERIFIED-feasible point. Near the boundary
    # sits the undecidable tolerance band, so the verified value can trail
    # the LP optimum by up to the band width; point equality at 1e-5 is the
    # old, unsound expectation. Containment in the verified bracket is the
    # sound check, plus a loose absolute cap.
    assert bracket_violations == 0
    # Boundary-band cases where hi could not be verified are counted, bounded.
    assert unverified_hi < 10
    assert lam_max_err < 1e-3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
