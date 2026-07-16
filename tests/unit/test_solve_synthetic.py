"""Synthetic solver tests on hand-built EquilibriumSystem instances.

These do not touch the geometry or mechanics builders. Every system is
assembled directly from numpy arrays so the solver layer is tested in
isolation against analytic answers.

Toy single 2D block, width b = 1, height h = 1, com at (0, 0, 0.5).
Ground patch (nodes ground=0, block=1), n_hat = +z, t1_hat = +x. Two
vertices at x = -0.5 and x = +0.5, z = 0. The block receives -F_k =
+n n_hat - u t1_hat per vertex, so compression pushes the block up.

Unknown layout (p V + v) ncomp + c gives f = [n0, u0, n1, u1]. Row layout
[Fx, Fz, Ty] about the block com. Uniform compression n0 = n1 = 0.5
balances gravity, which the first test verifies.

Analytic collapse for the toy block under horizontal live load:
slide at lambda = mu, topple at lambda = b/h = 1, so
lambda* = min(mu, b/h) and the governing mode switches at mu = b/h.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from keystone.geometry.tolerances import Tolerances
from keystone.mechanics.assemble import EquilibriumSystem
from keystone.solve import (
    FEASIBLE,
    INFEASIBLE,
    NO_CONVERGE,
    Result,
    SolverOptions,
    margin_and_grad,
    margin_batch,
    solve_p0,
    solve_p0_exact,
    solve_p2,
    solve_p2_exact,
    solve_p3,
    solve_p4,
)

TOL = Tolerances()

# Toy single block, centered support.
A_BLOCK = np.array(
    [
        [0.0, -1.0, 0.0, -1.0],  # Fx
        [1.0, 0.0, 1.0, 0.0],  # Fz
        [0.5, 0.5, -0.5, 0.5],  # Ty about com (0, 0, 0.5)
    ]
)
# Support strictly right of the com: vertices at x = 0.2 and x = 0.5.
A_OFFSET = np.array(
    [
        [0.0, -1.0, 0.0, -1.0],
        [1.0, 0.0, 1.0, 0.0],
        [-0.2, 0.5, -0.5, 0.5],
    ]
)
W_DEAD = np.array([0.0, -1.0, 0.0])  # gravity down, no torque about com
W_LIVE = np.array([1.0, 0.0, 0.0])  # unit horizontal pseudo-static load


def cone_2d(mu, nvert=2):
    """Exact 2D Coulomb cone rows G f <= 0 for nvert vertices.

    Per vertex k with n at column 2k and u at column 2k+1:
    -n <= 0, u - mu n <= 0, -u - mu n <= 0.
    """
    rows = []
    for v in range(nvert):
        b = 2 * v
        r = np.zeros(2 * nvert)
        r[b] = -1.0
        rows.append(r)
        r = np.zeros(2 * nvert)
        r[b] = -mu
        r[b + 1] = 1.0
        rows.append(r)
        r = np.zeros(2 * nvert)
        r[b] = -mu
        r[b + 1] = -1.0
        rows.append(r)
    return np.array(rows)


def make_system(A, w_dead, w_live, G, mu, *, dim=2, L=1.0, W=9.81):
    """Build an EquilibriumSystem from raw arrays. mu and vert_mask are
    dummies here; the solvers read only A, w, G, dim, W."""
    nvert = A.shape[1] // (2 if dim == 2 else 3)
    return EquilibriumSystem(
        A=jnp.asarray(A, dtype=jnp.float64),
        w_dead=jnp.asarray(w_dead, dtype=jnp.float64),
        w_live=jnp.asarray(w_live, dtype=jnp.float64),
        G=jnp.asarray(G, dtype=jnp.float64),
        mu=jnp.asarray(np.atleast_1d(mu), dtype=jnp.float64),
        vert_mask=jnp.ones((1, max(nvert, 1)), dtype=bool),
        L=jnp.asarray(L, dtype=jnp.float64),
        W=jnp.asarray(W, dtype=jnp.float64),
        dim=dim,
        cone="linear2d",
        k=0,
    )


def block_system(mu):
    return make_system(A_BLOCK, W_DEAD, W_LIVE, cone_2d(mu), mu)


def test_hand_built_A_balances_gravity():
    # Uniform compression n0 = n1 = 0.5 equilibrates self-weight exactly.
    f = np.array([0.5, 0.0, 0.5, 0.0])
    assert np.allclose(A_BLOCK @ f + W_DEAD, 0.0, atol=1e-12)


def test_p2_slide_governed():
    # mu = 0.3 < b/h, so sliding governs: lambda* = mu = 0.3.
    sys = block_system(0.3)
    r = solve_p2(sys, TOL)
    re = solve_p2_exact(sys, TOL)
    assert r.status == FEASIBLE
    assert abs(r.lambda_assoc - 0.3) < 1e-6
    assert abs(re.lambda_assoc - 0.3) < 1e-6


def test_p2_topple_governed():
    # mu = 2.0 > b/h, so toppling governs: lambda* = b/h = 1.0.
    sys = block_system(2.0)
    r = solve_p2(sys, TOL)
    re = solve_p2_exact(sys, TOL)
    assert r.status == FEASIBLE
    assert abs(r.lambda_assoc - 1.0) < 1e-6
    assert abs(re.lambda_assoc - 1.0) < 1e-6


def test_verdict_switch_at_b_over_h():
    # At the critical load lambda = b/h = 1, feasibility flips at mu = b/h.
    def feasible_at(mu, lam):
        sys = make_system(A_BLOCK, W_DEAD, W_LIVE, cone_2d(mu), mu)
        return float(solve_p4(sys, TOL, lam=lam).margin) <= TOL.tol_feas

    lo, hi = 0.5, 1.5
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if feasible_at(mid, 1.0):
            hi = mid
        else:
            lo = mid
    switch = 0.5 * (lo + hi)
    assert abs(switch - 1.0) < 1e-3


def test_stable_block_feasible():
    sys = block_system(0.5)
    r4 = solve_p4(sys, TOL)
    assert r4.margin <= TOL.tol_feas
    r0 = solve_p0(sys, TOL)
    assert r0.status == FEASIBLE
    # Verdict is deterministic under repeat calls.
    assert solve_p0(sys, TOL).status == FEASIBLE


def test_floating_block_mechanism():
    # No real support: one masked patch gives zero A columns.
    sys = make_system(np.zeros((3, 4)), W_DEAD, W_LIVE, cone_2d(0.5), 0.5)
    r = solve_p0(sys, TOL)
    assert r.status == INFEASIBLE
    assert r.mechanism is not None
    assert np.linalg.norm(r.mechanism) > 0.0
    assert r.info["load_power"] > 0.0
    # Row layout (vx, vz, wy): the block falls in -z.
    vx, vz, wy = r.mechanism[0]
    assert vz < 0.0
    assert abs(vx) < 1e-6
    assert abs(wy) < 1e-6


def test_offset_support_topples():
    # Support entirely right of the com with low friction: infeasible.
    sys = make_system(A_OFFSET, W_DEAD, W_LIVE, cone_2d(0.1), 0.1)
    r = solve_p0(sys, TOL)
    re = solve_p0_exact(sys, TOL)
    assert r.status == INFEASIBLE
    assert re.status == INFEASIBLE
    assert r.info["load_power"] > 0.0
    # Rotation component nonzero, the toppling sense.
    assert abs(r.mechanism[0][2]) > 1e-3


def test_p3_critical_friction():
    # Permanent horizontal load 0.2 W. Slide-governed since b/h = 1 > 0.2,
    # so the block is feasible iff mu >= 0.2. mu* = 0.2.
    w_dead2 = W_DEAD + 0.2 * W_LIVE
    sys = make_system(A_BLOCK, w_dead2, W_LIVE, cone_2d(0.5), 0.5)

    def g_of_mu(mu):
        return jnp.asarray(cone_2d(mu), dtype=jnp.float64)

    r = solve_p3(sys, g_of_mu, TOL)
    assert r.mu_critical_assoc is not None
    assert abs(r.mu_critical_assoc - 0.2) < 1e-3


def test_margin_batch_split_and_determinism():
    mu = 0.3
    A_b = jnp.broadcast_to(jnp.asarray(A_BLOCK), (8, 3, 4))
    G_b = jnp.broadcast_to(jnp.asarray(cone_2d(mu)), (8, 6, 4))
    lams = np.linspace(0.0, 2.0, 8)
    w_b = jnp.stack([jnp.asarray(W_DEAD + lam * W_LIVE) for lam in lams])
    margins, certified = margin_batch(A_b, w_b, G_b, TOL.eps_reg)
    margins = np.asarray(margins)
    # certified is a per-element cone-admissible-and-finite flag.
    assert np.all(np.asarray(certified))
    # Margin is nondecreasing in the load factor.
    assert np.all(np.diff(margins) >= -1e-9)
    feasible = margins <= TOL.tol_feas
    # lambda* = 0.3. lams[0]=0 and lams[1]~0.286 feasible; lams[2]~0.571 not.
    assert feasible[0] and feasible[1]
    assert not feasible[2]
    # Bitwise determinism across identical calls.
    margins2, _ = margin_batch(A_b, w_b, G_b, TOL.eps_reg)
    assert np.array_equal(margins, np.asarray(margins2))


def test_margin_and_grad_sign():
    # Infeasible point: lambda = 2 > lambda* = 0.3 for mu = 0.3.
    A = jnp.asarray(A_BLOCK)
    G = jnp.asarray(cone_2d(0.3))
    w = jnp.asarray(W_DEAD + 2.0 * W_LIVE)
    value, grad = margin_and_grad(A, w, G, TOL.eps_reg)
    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(grad)))
    assert float(value) > TOL.tol_feas
    # More horizontal live load cannot lower the margin at an infeasible point.
    dderiv = float(jnp.asarray(W_LIVE) @ grad)
    assert dderiv >= -1e-6


def test_random_lp_agreement():
    # JAX P4 verdict agrees with the HiGHS oracle away from the feasibility
    # band. Cases inside the band or where the interior point did not resolve
    # (non-finite margin) are escalated to the oracle and excluded here.
    rng = np.random.default_rng(0)
    excluded = 0
    disagreements = 0
    for _ in range(20):
        A = rng.standard_normal((6, 8))
        w = rng.standard_normal(6)
        mu = float(rng.uniform(0.1, 1.0))
        G = cone_2d(mu, nvert=4)
        sys = make_system(A, w, np.zeros(6), G, mu, dim=2)
        rj = solve_p0(sys, TOL)
        re = solve_p0_exact(sys, TOL)
        m = rj.margin
        band = TOL.tol_feas < m < 10.0 * TOL.tol_feas
        if not np.isfinite(m) or band:
            excluded += 1
            continue
        if rj.status != re.status:
            disagreements += 1
    assert excluded < 5
    assert disagreements == 0


def g_of_mu_2d(mu):
    return jnp.asarray(cone_2d(mu), dtype=jnp.float64)


def test_p2_p3_escalation_forces_provenance():
    # With max_iter=2 the interior point never converges, so every verdict is
    # decided by the exact oracle and the returned forces are the oracle's
    # (finding 1a). info["forces_certified_by"] names the deciding backend.
    opts = SolverOptions(max_iter=2)
    r2 = solve_p2(block_system(0.3), TOL, opts=opts)
    assert r2.status == FEASIBLE
    assert r2.info["forces_certified_by"] == "exact"
    assert abs(r2.lambda_assoc - 0.3) < 1e-4

    wd = W_DEAD + 0.2 * W_LIVE
    sys = make_system(A_BLOCK, wd, W_LIVE, cone_2d(0.5), 0.5)
    r3 = solve_p3(sys, g_of_mu_2d, TOL, opts=opts)
    assert r3.status == FEASIBLE
    assert r3.info["forces_certified_by"] == "exact"
    assert abs(r3.mu_critical_assoc - 0.2) < 1e-3


def test_p2_uncertified_band_recording(monkeypatch):
    # Force the hi endpoint uncertified: max_iter = 2 stalls the QP, the QP
    # Farkas check is suppressed so no midpoint self-certifies infeasible, and
    # the patched oracle certifies only the dead-load state. solve_p2 must then
    # report the last certified-feasible lambda (0) with an uncertified band,
    # never lam = hi with uncertified forces (finding 1b).
    import keystone.solve.batch_jax as bj

    real_exact = bj.solve_p0_exact
    dead = np.asarray(W_DEAD, dtype=float)

    def fake_exact(system, tol):
        if np.allclose(np.asarray(system.w_dead), dead):
            return real_exact(system, tol)
        return Result(status=NO_CONVERGE, margin=float("inf"))

    monkeypatch.setattr(bj, "solve_p0_exact", fake_exact)
    monkeypatch.setattr(
        bj, "_verify_farkas",
        lambda y, A, G, w, tol: {
            "load_power": 0.0, "dual_residual": float("inf"), "certified": False
        },
    )
    sys = block_system(0.3)
    r = solve_p2(sys, TOL, lam_hi=16.0, opts=SolverOptions(max_iter=2))
    assert r.status == FEASIBLE
    assert r.lambda_assoc == 0.0
    assert r.info.get("bracket_found") is False
    lo_band, hi_band = r.info["uncertified_band"]
    assert lo_band == 0.0 and hi_band == 16.0
    # The reported factor is a certified-feasible lower value, unordered vs true.
    assert r.physical_bound_direction == "unknown"
    # The reported forces certify equilibrium at the dead-load state.
    f_nd = np.asarray(r.forces) / float(sys.W)
    resid = A_BLOCK @ f_nd + W_DEAD
    assert np.linalg.norm(resid) / np.linalg.norm(W_DEAD) <= TOL.tol_eq


def test_p3_no_converge_at_mu_hi_not_infeasible(monkeypatch):
    # NO_CONVERGE at mu_hi must never be reported as INFEASIBLE (finding 1c).
    # Patch the oracle to always fail and stall the QP with max_iter = 2.
    import keystone.solve.batch_jax as bj

    monkeypatch.setattr(
        bj, "solve_p0_exact",
        lambda system, tol: Result(status=NO_CONVERGE, margin=float("inf")),
    )
    r = solve_p3(block_system(0.5), g_of_mu_2d, TOL, opts=SolverOptions(max_iter=2))
    assert r.status == NO_CONVERGE
    assert r.info.get("uncertified_at_mu_lo") is True
    assert r.mu_critical_assoc is None
    assert r.physical_bound_direction is None


def test_p3_feasible_at_mu_lo_uses_low_forces():
    # Feasible even at mu_lo: the returned forces come from the low endpoint
    # and are admissible for the reported low friction (finding 1c).
    wd = W_DEAD + 0.1 * W_LIVE
    sys = make_system(A_BLOCK, wd, W_LIVE, cone_2d(0.5), 0.5)
    r = solve_p3(sys, g_of_mu_2d, TOL, mu_lo=0.15, mu_hi=4.0)
    assert r.status == FEASIBLE
    assert abs(r.mu_critical_assoc - 0.15) < 1e-12
    f_nd = np.asarray(r.forces) / float(sys.W)
    assert np.max(cone_2d(0.15) @ f_nd) <= TOL.tol_cone
    assert np.linalg.norm(A_BLOCK @ f_nd + wd) / np.linalg.norm(wd) <= TOL.tol_eq


def test_margin_and_grad_finite_difference():
    # Central-difference check at a smooth infeasible point: mu = 0.2,
    # lam = 0.5 (above lam* = 0.2). The gradient is taken at a relaxed KKT
    # point; a target_kappa tighter than the qpax default sharpens it. At
    # target_kappa = 1e-5 the measured relative error is ~3e-4, well below
    # 1e-3; the qpax default 1e-3 gives ~2.4e-2 (recorded, not asserted).
    A = jnp.asarray(A_BLOCK)
    G = jnp.asarray(cone_2d(0.2))
    w = jnp.asarray(W_DEAD + 0.5 * W_LIVE)
    tk = 1e-5
    value, grad = margin_and_grad(A, w, G, TOL.eps_reg, target_kappa=tk)
    grad = np.asarray(grad)
    assert np.isfinite(float(value)) and float(value) > TOL.tol_feas
    assert np.all(np.isfinite(grad))

    rng = np.random.default_rng(1)
    rand = rng.standard_normal(3)
    rand /= np.linalg.norm(rand)
    base = float(np.linalg.norm(np.asarray(w)))
    eps = 1e-5 * base
    worst = 0.0
    for d in (np.asarray(W_LIVE, dtype=float), rand):
        mp, _ = margin_and_grad(
            A, jnp.asarray(np.asarray(w) + eps * d), G, TOL.eps_reg, target_kappa=tk
        )
        mm, _ = margin_and_grad(
            A, jnp.asarray(np.asarray(w) - eps * d), G, TOL.eps_reg, target_kappa=tk
        )
        fd = (float(mp) - float(mm)) / (2.0 * eps)
        ana = float(grad @ d)
        rel = abs(fd - ana) / max(abs(fd), 1e-12)
        worst = max(worst, rel)
    assert worst < 1e-3, worst


def test_margin_and_grad_value_matches_solve_p4():
    # margin_and_grad now reports the recomputed-residual margin, the same
    # quantity solve_p4 reports, so the two agree on the converged iterate
    # (finding 4).
    A = jnp.asarray(A_BLOCK)
    G = jnp.asarray(cone_2d(0.2))
    w = jnp.asarray(W_DEAD + 0.5 * W_LIVE)
    value, _ = margin_and_grad(A, w, G, TOL.eps_reg)
    sys = make_system(A_BLOCK, W_DEAD, W_LIVE, cone_2d(0.2), 0.2)
    r = solve_p4(sys, TOL, lam=0.5)
    assert abs(float(value) - float(r.margin)) < 1e-6


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
