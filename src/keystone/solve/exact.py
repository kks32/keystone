"""CPU oracle solvers through scipy HiGHS (scipy.optimize.linprog).

These prove the JAX path. Used by tests and as an escalation path for
verdicts inside the feasibility band. LP-shaped problems only:
P0 as an LP feasibility problem, P2 as max lambda.

The exact path has no elastic margin. It reports margin 0.0 on a feasible
verdict and infinity on an infeasible one. Use the JAX P4 path for a
graded margin.

On infeasibility the mechanism is a validated Farkas ray. scipy's
eqlin.marginals is not a Farkas ray for an infeasible LP, so the ray comes
from an explicit normalized certificate LP and is checked numerically
before it is returned (see _farkas_certificate).

At a P2 optimum eqlin.marginals is valid: the equality duals of the
maximize-lambda LP are the kinematic collapse mechanism. solve_p2_exact
extracts them at an uncensored optimum, orients and normalizes the ray, and
records the complementary-slackness residual in info. Every verdict is
independently rechecked: a HiGHS-feasible primal is accepted only after its
recomputed equilibrium residual and cone violation clear tol_eq / tol_cone,
and a HiGHS-infeasible verdict only after the Farkas certificate validates;
either check failing yields NO_CONVERGE, never a false FEASIBLE/INFEASIBLE.
"""

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from ..geometry.tolerances import Tolerances
from ..mechanics.assemble import EquilibriumSystem
from .result import FEASIBLE, INFEASIBLE, NO_CONVERGE, Result

# scipy status 2 is "problem is infeasible".
_SCIPY_INFEASIBLE = 2


def _highs_opts(tol: Tolerances) -> dict:
    """Map keystone tolerances onto HiGHS feasibility options.

    tol_feas sets both the primal and dual feasibility tolerances. scipy's
    method="highs" accepts these keys.
    """
    return {
        "primal_feasibility_tolerance": tol.tol_feas,
        "dual_feasibility_tolerance": tol.tol_feas,
    }


def _p2_bound(cone: str, censored: bool):
    """(physical_bound_direction, info bound tag) for a P2 load factor.

    A linear 2D uncensored associative load factor overestimates true
    Coulomb capacity ("upper"). An inscribed pyramid combines an
    associative overestimate with a cone underestimate, so it is unordered
    vs true ("unknown"). A censored factor is only a cap that lower-bounds
    associative capacity; it is unordered vs true even in 2D ("unknown").
    """
    if censored:
        return "unknown", "censored-lower-of-assoc-unordered-vs-true"
    if cone == "linear2d":
        return "upper", "upper-of-true-assoc-exact"
    return "unknown", "lower-of-assoc-exact-unordered-vs-true"


def _primal_residual(A, w, G, f):
    """Recomputed equilibrium residual margin and cone violation for f.

    margin = ||A f + w|| / ||w||, viol = max(0, max(G f)). Same rule as the
    JAX path so the two backends certify feasibility identically.
    """
    r = A @ f + w
    norm_w = float(np.linalg.norm(w))
    denom = max(norm_w, float(np.finfo(float).tiny))
    margin = float(np.linalg.norm(r) / denom)
    viol = float(max(0.0, float(np.max(G @ f)))) if G.shape[0] else 0.0
    return margin, viol


def _complementary_slackness(G, f, z):
    """Scaled max complementary-slackness violation for G f <= 0, duals z.

    slack_i = -(G f)_i >= 0 at a feasible optimum, z_i the inequality dual.
    Complementary slackness asks z_i * slack_i = 0 for every row. Report the
    max product normalized by the dual and slack magnitudes so the threshold
    is dimensionless and comparable to tol_dual.
    """
    if G.shape[0] == 0:
        return 0.0
    slack = -(G @ f)
    zscale = max(1.0, float(np.max(np.abs(z))))
    sscale = max(1.0, float(np.max(np.abs(slack))))
    return float(np.max(np.abs(z * slack)) / (zscale * sscale))


def _farkas_certificate(A, w, G, dim, tol):
    """Validated Farkas ray for an infeasible {A f = -w, G f <= 0}.

    Solve the normalized certificate LP:
        maximize y . w  s.t.  A^T y + G^T z = 0,  z >= 0,  -1 <= y <= 1.
    A positive optimum with the equality satisfied is a Farkas ray: A^T y
    lies in the dual cone K* (witnessed by z >= 0) and y . w > 0, so no
    primal f in the cone can equilibrate -w. The box -1 <= y <= 1
    normalizes the scale-invariant certificate cone. The ray is validated
    numerically (dual residual and load power recomputed) before return;
    solver sign conventions are never trusted.

    Returns (mechanism, farkas). mechanism is the normalized ray reshaped
    to per-block twists when validated, else None.
    """
    A = np.asarray(A, dtype=float)
    w = np.asarray(w, dtype=float)
    G = np.asarray(G, dtype=float)
    m = A.shape[0]
    nf = A.shape[1]
    ncone = G.shape[0]
    # Variables [y (m), z (ncone)]; maximize y.w is minimize -w.y.
    c = np.concatenate([-w, np.zeros(ncone)])
    A_eq = sp.csr_matrix(np.hstack([A.T, G.T]))
    b_eq = np.zeros(nf)
    bounds = [(-1.0, 1.0)] * m + [(0.0, None)] * ncone
    res = linprog(
        c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs",
        options=_highs_opts(tol),
    )
    if not res.success:
        return None, {"load_power": 0.0, "dual_residual": float("inf"),
                      "certified": False}
    y = np.asarray(res.x[:m])
    z = np.asarray(res.x[m:])
    dual_res = float(np.linalg.norm(A.T @ y + G.T @ z))
    load_power = float(y @ w)
    certified = bool(
        np.isfinite(load_power)
        and load_power > tol.tol_power
        and dual_res <= tol.tol_dual
    )
    mechanism = None
    if certified:
        norm_y = np.linalg.norm(y)
        yn = y / norm_y if norm_y > 0.0 else y
        rpb = 3 if dim == 2 else 6
        mechanism = yn.reshape(m // rpb, rpb)
        load_power = float(yn @ w)
    return mechanism, {"load_power": load_power, "dual_residual": dual_res,
                       "certified": certified}


def solve_p0_exact(system: EquilibriumSystem, tol: Tolerances) -> Result:
    """P0 feasibility as an LP. minimize 0 s.t. A f = -w_dead, G f <= 0.

    Forces are free; the cone rows in G carry n >= 0. tol wires HiGHS
    feasibility options and the certificate tolerances. P0 reports no
    capacity factor, so physical_bound_direction is None.

    A HiGHS-feasible primal is accepted only after the recomputed
    equilibrium residual and cone violation clear tol_eq / tol_cone;
    otherwise NO_CONVERGE. A HiGHS-infeasible verdict is accepted only after
    the explicit Farkas certificate validates; otherwise NO_CONVERGE.
    """
    A = np.asarray(system.A)
    w = np.asarray(system.w_dead)
    G = np.asarray(system.G)
    nf = A.shape[1]

    c = np.zeros(nf)
    A_eq = sp.csr_matrix(A)
    b_eq = -w
    A_ub = sp.csr_matrix(G)
    b_ub = np.zeros(G.shape[0])
    bounds = [(None, None)] * nf

    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
        method="highs", options=_highs_opts(tol),
    )
    info = {
        "scipy_status": int(res.status),
        "message": str(res.message),
        "note": "exact LP feasibility; no elastic margin",
    }
    # P0 reports no capacity factor (charter: bound direction only on a factor).
    common = dict(cone_model=system.cone, physical_bound_direction=None)
    if res.success:
        f = np.asarray(res.x)
        margin, viol = _primal_residual(A, w, G, f)
        info["recomputed_margin"] = margin
        info["recomputed_viol"] = viol
        if np.all(np.isfinite(f)) and margin <= tol.tol_eq and viol <= tol.tol_cone:
            return Result(
                status=FEASIBLE, margin=0.0, forces=f * float(system.W),
                info=info, **common,
            )
        info["note"] = "HiGHS feasible but recomputed residual/cone check failed"
        return Result(status=NO_CONVERGE, margin=float("inf"), info=info, **common)
    if res.status == _SCIPY_INFEASIBLE:
        mechanism, farkas = _farkas_certificate(A, w, G, system.dim, tol)
        info["farkas"] = farkas
        info["load_power"] = farkas["load_power"]
        if not farkas["certified"]:
            info["note"] = "HiGHS infeasible but Farkas certificate failed validation"
            return Result(status=NO_CONVERGE, margin=float("inf"), info=info, **common)
        return Result(
            status=INFEASIBLE, margin=float("inf"), mechanism=mechanism,
            info=info, **common,
        )
    return Result(status=NO_CONVERGE, margin=float("inf"), info=info, **common)


def solve_p2_exact(
    system: EquilibriumSystem,
    tol: Tolerances,
    *,
    lam_hi: float = 16.0,
) -> Result:
    """P2 load factor as one LP. maximize lam over [f, lam].

    A f + lam w_live = -w_dead, G f <= 0, lam in [0, lam_hi], f free.
    lambda_assoc is the optimum. On infeasibility at lam = 0 the mechanism
    is a validated Farkas ray. If lam saturates lam_hi the result is still
    feasible but flagged censored.
    """
    if not (lam_hi > 0.0):
        raise ValueError(f"lam_hi must be > 0, got {lam_hi!r}")
    A = np.asarray(system.A)
    w_dead = np.asarray(system.w_dead)
    w_live = np.asarray(system.w_live)
    G = np.asarray(system.G)
    nrows, nf = A.shape
    ncone = G.shape[0]
    rpb = 3 if system.dim == 2 else 6

    # Variables [f, lam]; maximize lam is minimize -lam.
    c = np.zeros(nf + 1)
    c[-1] = -1.0
    A_eq = sp.csr_matrix(np.hstack([A, w_live.reshape(-1, 1)]))
    b_eq = -w_dead
    A_ub = sp.csr_matrix(np.hstack([G, np.zeros((ncone, 1))]))
    b_ub = np.zeros(ncone)
    bounds = [(None, None)] * nf + [(0.0, float(lam_hi))]

    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
        method="highs", options=_highs_opts(tol),
    )
    info = {
        "scipy_status": int(res.status),
        "message": str(res.message),
        "lam_hi": float(lam_hi),
    }
    if res.success:
        lam = float(res.x[-1])
        f = np.asarray(res.x[:nf])
        censored = lam >= float(lam_hi) - tol.tol_feas
        # Recompute the residual and cone violation on the loaded state.
        w_total = w_dead + lam * w_live
        margin, viol = _primal_residual(A, w_total, G, f)
        info["recomputed_margin"] = margin
        info["recomputed_viol"] = viol
        direction, tag = _p2_bound(system.cone, censored)
        info["bound"] = tag
        common = dict(cone_model=system.cone, physical_bound_direction=direction)
        if not (np.all(np.isfinite(f)) and margin <= tol.tol_eq
                and viol <= tol.tol_cone):
            info["note"] = "HiGHS feasible but recomputed residual/cone check failed"
            return Result(status=NO_CONVERGE, margin=float("inf"), info=info, **common)
        if censored:
            info["censored"] = True
            info["note"] = "lambda hit lam_hi cap; lambda_assoc is a cap"
        # Mechanism from the equality duals, valid at an LP optimum. Orient so
        # gravity does nonnegative power, then normalize. Only at an uncensored
        # optimum is this a physical collapse mechanism; a censored optimum is
        # the capped problem, so no mechanism is attached.
        mechanism = None
        y = np.asarray(res.eqlin.marginals, dtype=float)
        if float(y @ w_dead) < 0.0:
            y = -y
        norm_y = float(np.linalg.norm(y))
        yn = y / norm_y if norm_y > 0.0 else y
        z = np.asarray(res.ineqlin.marginals, dtype=float)
        cs = _complementary_slackness(G, f, z)
        info["complementary_slackness"] = cs
        info["complementary_slackness_ok"] = bool(cs <= tol.tol_dual)
        if not censored:
            mechanism = yn.reshape(nrows // rpb, rpb)
        return Result(
            status=FEASIBLE, margin=0.0, forces=f * float(system.W),
            lambda_assoc=lam, mechanism=mechanism, info=info, **common,
        )
    # lam = 0 is fixed geometry, so an uncensored bound tag applies here.
    direction, tag = _p2_bound(system.cone, False)
    info["bound"] = tag
    common = dict(cone_model=system.cone, physical_bound_direction=direction)
    if res.status == _SCIPY_INFEASIBLE:
        # Infeasible even at lam = 0. No positive load factor exists.
        mechanism, farkas = _farkas_certificate(A, w_dead, G, system.dim, tol)
        info["farkas"] = farkas
        info["load_power"] = farkas["load_power"]
        if not farkas["certified"]:
            info["note"] = "HiGHS infeasible but Farkas certificate failed validation"
            return Result(status=NO_CONVERGE, margin=float("inf"), info=info, **common)
        return Result(
            status=INFEASIBLE, margin=float("inf"), lambda_assoc=0.0,
            mechanism=mechanism, info=info, **common,
        )
    return Result(status=NO_CONVERGE, margin=float("inf"), info=info, **common)
