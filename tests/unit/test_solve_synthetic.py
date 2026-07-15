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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
