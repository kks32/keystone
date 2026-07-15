"""Certificate and validation tests for the solve layer.

Verdicts are verified, never read off a solver flag. FEASIBLE needs a
primal certificate (recomputed residual and cone violation both small).
INFEASIBLE needs a validated Farkas ray (load power positive, A^T y in the
dual cone). Everything else is NO_CONVERGE. This file also pins the input
validation on Tolerances, SolverOptions, and the pyramid cone, and the
load-factor field rename to lambda_assoc.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from keystone.geometry.tolerances import Tolerances
from keystone.mechanics.assemble import EquilibriumSystem
from keystone.mechanics.cones import cone_matrix_2d, cone_matrix_pyramid
from keystone.solve import (
    FEASIBLE,
    INFEASIBLE,
    NO_CONVERGE,
    Result,
    SolverOptions,
    solve_p0,
    solve_p0_exact,
    solve_p2_exact,
    solve_p4,
)
from keystone.solve.batch_jax import (
    _dual_cone_violation,
    _verify_farkas,
    margin_core,
)

TOL = Tolerances()

# Toy single 2D block, centered support (feasible) and offset support (topples).
A_BLOCK = np.array([[0.0, -1.0, 0.0, -1.0], [1.0, 0.0, 1.0, 0.0], [0.5, 0.5, -0.5, 0.5]])
A_OFFSET = np.array([[0.0, -1.0, 0.0, -1.0], [1.0, 0.0, 1.0, 0.0], [-0.2, 0.5, -0.5, 0.5]])
W_DEAD = np.array([0.0, -1.0, 0.0])
W_LIVE = np.array([1.0, 0.0, 0.0])


def cone_2d(mu, nvert=2):
    rows = []
    for v in range(nvert):
        b = 2 * v
        r = np.zeros(2 * nvert); r[b] = -1.0; rows.append(r)
        r = np.zeros(2 * nvert); r[b] = -mu; r[b + 1] = 1.0; rows.append(r)
        r = np.zeros(2 * nvert); r[b] = -mu; r[b + 1] = -1.0; rows.append(r)
    return np.array(rows)


def make_system(A, wd, wl, G, mu, dim=2, L=1.0, W=9.81):
    nvert = A.shape[1] // (2 if dim == 2 else 3)
    return EquilibriumSystem(
        A=jnp.asarray(A, dtype=jnp.float64),
        w_dead=jnp.asarray(wd, dtype=jnp.float64),
        w_live=jnp.asarray(wl, dtype=jnp.float64),
        G=jnp.asarray(G, dtype=jnp.float64),
        mu=jnp.asarray(np.atleast_1d(mu), dtype=jnp.float64),
        vert_mask=jnp.ones((1, max(nvert, 1)), dtype=bool),
        L=jnp.asarray(L, dtype=jnp.float64),
        W=jnp.asarray(W, dtype=jnp.float64),
        dim=dim, cone="linear2d", k=0,
    )


def block(mu):
    return make_system(A_BLOCK, W_DEAD, W_LIVE, cone_2d(mu), mu)


def floating():
    return make_system(np.zeros((3, 4)), W_DEAD, W_LIVE, cone_2d(0.5), 0.5)


def offset():
    return make_system(A_OFFSET, W_DEAD, W_LIVE, cone_2d(0.1), 0.1)


# --- margin_core signature --------------------------------------------------

def test_margin_core_returns_seven():
    out = margin_core(
        jnp.asarray(A_BLOCK), jnp.asarray(W_DEAD), jnp.asarray(cone_2d(0.5)), TOL.eps_reg
    )
    assert len(out) == 7
    margin, f, r, viol, gap, converged, iters = out
    # Recomputed residual margin is tiny for the balanced block.
    assert float(margin) <= TOL.tol_eq
    assert float(viol) <= TOL.tol_cone
    # r is the recomputed residual A f + w.
    assert np.allclose(np.asarray(r), np.asarray(A_BLOCK) @ np.asarray(f) + W_DEAD)


# --- verified verdicts ------------------------------------------------------

def test_no_converge_without_certificate():
    # Three interior-point iterations cannot resolve the balanced block. The
    # solver must not claim FEASIBLE or INFEASIBLE without a real certificate.
    r = solve_p0(block(0.5), TOL, opts=SolverOptions(max_iter=3))
    assert r.status == NO_CONVERGE
    for mi in (1, 2, 3):
        rr = solve_p0(block(0.5), TOL, opts=SolverOptions(max_iter=mi))
        if rr.status == FEASIBLE:
            assert rr.margin <= TOL.tol_eq and rr.info["viol"] <= TOL.tol_cone
        elif rr.status == INFEASIBLE:
            assert rr.info["farkas"]["certified"]
        else:
            assert rr.status == NO_CONVERGE


def test_farkas_floating_and_offset():
    for sys in (floating(), offset()):
        r = solve_p0(sys, TOL)
        assert r.status == INFEASIBLE
        fk = r.info["farkas"]
        assert fk["certified"]
        assert fk["load_power"] > 0.0
        assert fk["dual_residual"] <= TOL.tol_dual
        assert r.mechanism is not None
        assert np.all(np.isfinite(r.mechanism))


def test_verify_farkas_direct():
    # Hand ray for the floating block: the block falls in -z.
    y = np.array([0.0, -1.0, 0.0])
    fk = _verify_farkas(y, np.zeros((3, 4)), cone_2d(0.5), W_DEAD, TOL)
    assert fk["certified"]
    assert fk["load_power"] > 0.0
    assert fk["dual_residual"] <= TOL.tol_dual
    # A ray with no load power (orthogonal to w) does not certify.
    fk2 = _verify_farkas(np.array([1.0, 0.0, 0.0]), np.zeros((3, 4)), cone_2d(0.5), W_DEAD, TOL)
    assert not fk2["certified"]


# --- dual cone closed-form cross-check --------------------------------------

def test_dual_cone_2d():
    # 2D wedge dual for mu = 0.5: z_n >= mu |z_u|.
    assert _dual_cone_violation([1.0, 0.0], [0.5], 2, 0) < 0.0  # inside
    assert _dual_cone_violation([0.0, 1.0], [0.5], 2, 0) > 0.0  # outside
    assert abs(_dual_cone_violation([1.0, 2.0], [0.5], 2, 0)) < 1e-12  # boundary


def test_dual_cone_pyramid_k4():
    # Inscribed pyramid dual, k = 4, mu = 0.5.
    assert _dual_cone_violation([1.0, 0.0, 0.0], [0.5], 3, 4) < 0.0  # inside
    assert _dual_cone_violation([0.0, 5.0, 0.0], [0.5], 3, 4) > 0.0  # outside


# --- exact backend mechanism ------------------------------------------------

def test_exact_mechanism_infeasible():
    re = solve_p0_exact(offset(), TOL)
    assert re.status == INFEASIBLE
    assert re.mechanism is not None
    y = np.asarray(re.mechanism).reshape(-1)
    assert float(y @ W_DEAD) > 0.0  # load power positive
    assert re.info["farkas"]["certified"]
    assert re.info["farkas"]["load_power"] > 0.0
    assert re.info["farkas"]["dual_residual"] <= TOL.tol_dual


def test_exact_censored_lam_hi():
    # The mu = 2 block topples at lambda = 1. A cap of 0.5 censors the answer.
    r = solve_p2_exact(block(2.0), TOL, lam_hi=0.5)
    assert r.status == FEASIBLE
    assert r.info.get("censored") is True
    assert abs(r.lambda_assoc - 0.5) < 1e-6


# --- structured bound fields and rename -------------------------------------

def test_result_rename_and_bound_fields():
    fields = Result.__dataclass_fields__
    assert "lambda_assoc" in fields
    assert "lambda_upper_assoc" not in fields
    for name in ("cone_model", "constitutive_model", "physical_bound_direction"):
        assert name in fields
    r = Result(status=FEASIBLE, margin=0.0)
    assert r.constitutive_model == "associative"


def test_bound_direction_linear2d():
    r = solve_p0(block(0.5), TOL)
    assert r.cone_model == "linear2d"
    assert r.physical_bound_direction == "upper"
    assert r.info["bound"] == "upper-of-true-assoc-exact"


# --- input validation -------------------------------------------------------

def test_tolerances_reject_nonpositive():
    with pytest.raises(ValueError):
        Tolerances(tol_feas=-1.0)
    with pytest.raises(ValueError):
        Tolerances(eps_reg=0.0)
    with pytest.raises(ValueError):
        Tolerances(g_tol=float("inf"))
    with pytest.raises(ValueError):
        Tolerances(A_min=-1e-9)


def test_tolerances_derived_defaults():
    t = Tolerances()
    assert t.tol_eq == t.tol_feas
    assert t.tol_cone == t.tol_feas
    assert t.tol_power == t.tol_feas
    assert t.tol_dual == 10.0 * t.tol_feas
    assert t.tol_gap == 100.0 * t.tol_feas
    # An explicit override is honored.
    t2 = Tolerances(tol_dual=1e-5)
    assert t2.tol_dual == 1e-5


def test_solver_options_validation():
    with pytest.raises(ValueError):
        SolverOptions(max_iter=0)
    with pytest.raises(ValueError):
        SolverOptions(solver_tol=0.0)


def test_cone_pyramid_k_validation():
    mu = jnp.array([0.5])
    with pytest.raises(ValueError):
        cone_matrix_pyramid(mu, 1, 4, 2)  # k < 3
    with pytest.raises(ValueError):
        cone_matrix_pyramid(mu, 1, 4, 3)  # odd, standard API
    with pytest.raises(ValueError):
        cone_matrix_pyramid(mu, 1, 4, 4.5)  # non-integer
    # Even k >= 4 is fine; odd k is allowed only when explicitly asymmetric.
    cone_matrix_pyramid(mu, 1, 4, 4)
    cone_matrix_pyramid(mu, 1, 4, 3, allow_asymmetric=True)


def test_cone_mu_validation():
    with pytest.raises(ValueError):
        cone_matrix_2d(jnp.array([-0.5]), 1, 2)
    with pytest.raises(ValueError):
        cone_matrix_pyramid(jnp.array([-0.5]), 1, 4, 4)
    with pytest.raises(ValueError):
        cone_matrix_2d(jnp.array([np.inf]), 1, 2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
