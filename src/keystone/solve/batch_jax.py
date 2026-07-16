"""JAX solvers: P4 elastic margin, P0 feasibility, P2 load factor,
P3 critical friction. qpax primal-dual interior point under jit/vmap.

P4 formulation (frozen contract):
    variables x = [f, s], nx = nf + nrows
    minimize 0.5 * s.s + 0.5 * eps_reg * f.f
    subject to A f - s = -w_total          (equality)
               G f <= 0                    (inequality)
    margin = ||A f + w_total|| / ||w_total||
The slack formulation keeps Q block-diagonal and well conditioned and
avoids squaring the condition number of A. The reported margin is
recomputed from the returned force by matvec, not read from the slack s,
so it never depends on the solver's internal slack floor.

Verdict rule (verified, independent of solver flags):
- FEASIBLE requires a primal certificate: the recomputed equilibrium
  residual margin <= tol_eq, the cone violation max(0, max(G f)) <=
  tol_cone, and all quantities finite.
- INFEASIBLE requires a validated Farkas ray. The normalized residual
  direction y = (A f + w) / ||A f + w|| is a candidate. It certifies when
  the load power y . w > tol_power and A^T y lies in the dual cone K*,
  checked generally by finding z >= 0 with A^T y + G^T z = 0 to a dual
  residual <= tol_dual (Farkas for the polyhedral cone {f : G f <= 0}).
- Anything else is NO_CONVERGE.

Margin bound on the optimum: a cone-admissible iterate (cone violation
<= tol_cone) is P4-feasible, so its recomputed margin upper-bounds the
optimal P4 margin. An iterate that violates the cone is not P4-feasible
and its margin need not bound the optimum in either direction.
info["margin_certified"] is True only when the full KKT residual of the
slack QP (stationarity, equality, inequality violation, complementarity;
each an inf-norm) sits below tol_gap, certifying the iterate optimal. The
four residuals are recorded in info["kkt"]. The raw converged flag and
iteration count are always in info.

P2 and P3 bisect on this verified verdict. A NO_CONVERGE midpoint is not
treated as infeasible: it escalates to the exact LP oracle on the same
system with the loaded weight swapped in. When the oracle certifies
feasibility the returned force state is the oracle's, not the uncertified
QP iterate (info["certified_by"] names the deciding backend). If the
oracle also fails, the bisection stops and records an uncertified band.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import qpax
from scipy.optimize import nnls

from ..geometry.tolerances import Tolerances
from ..mechanics.assemble import EquilibriumSystem
from .exact import solve_p0_exact
from .options import SolverOptions
from .result import FEASIBLE, INFEASIBLE, NO_CONVERGE, Result

_DEFAULT_OPTS = SolverOptions()
# Default cone-violation tolerance for the low-level batch kernel, read
# from Tolerances so no tolerance constant lives outside that dataclass.
_DEFAULT_TOL_CONE = Tolerances().tol_cone


def margin_core(
    A,
    w_total,
    G,
    eps_reg,
    *,
    solver_tol=_DEFAULT_OPTS.solver_tol,
    max_iter=_DEFAULT_OPTS.max_iter,
):
    """P4 elastic margin as a slack QP, dimension agnostic.

    Variables x = [f, s]. Minimize 0.5 s.s + 0.5 eps_reg f.f subject to
    [A, -I] x = -w_total and [G, 0] x <= 0. Q is block diagonal:
    eps_reg I on the force block, I on the slack block.

    Pure and jit-able. Consumes matrices only, so 2D and 3D share it.

    Returns (margin, f, r, viol, gap, converged, iters, res_stat, res_eq,
    res_ineq, res_comp):
      margin    ||A f + w_total|| / ||w_total||, recomputed by matvec.
      f         the force block of the solution.
      r         the recomputed equilibrium residual A f + w_total.
      viol      cone violation max(0, max(G f)); 0 means cone-admissible.
      gap       primal-dual complementarity gap s_ineq . z from qpax.
      converged raw qpax flag (not the verdict; see the module docstring).
      iters     raw qpax iteration count.
      res_stat  inf-norm of the KKT stationarity residual
                ||Q x + q + Aeq^T y + Gineq^T z||_inf.
      res_eq    inf-norm of the equality residual ||Aeq x - beq||_inf.
      res_ineq  inequality violation max(0, max(Gineq x - hineq)).
      res_comp  inf-norm of the complementarity residual ||s_ineq * z||_inf.
    The four residuals mirror qpax's own convergence test and let a consumer
    certify the iterate optimal (all four below tol_gap) rather than trust
    the raw converged flag.
    """
    nf = A.shape[1]
    nrows = A.shape[0]
    ncone = G.shape[0]
    # Q = block diag(eps_reg I_nf, I_nrows), q = 0.
    Q = jnp.zeros((nf + nrows, nf + nrows))
    Q = Q.at[:nf, :nf].set(eps_reg * jnp.eye(nf))
    Q = Q.at[nf:, nf:].set(jnp.eye(nrows))
    q = jnp.zeros(nf + nrows)
    # Equality [A, -I] x = -w_total ties s to the equilibrium residual.
    Aeq = jnp.concatenate([A, -jnp.eye(nrows)], axis=1)
    beq = -w_total
    # Inequality [G, 0] x <= 0 keeps the force block in the friction cone.
    Gineq = jnp.concatenate([G, jnp.zeros((ncone, nrows))], axis=1)
    hineq = jnp.zeros(ncone)
    x, s_ineq, z, y, converged, iters = qpax.solve_qp(
        Q, q, Aeq, beq, Gineq, hineq, solver_tol=solver_tol, max_iter=max_iter
    )
    f = x[:nf]
    # Recompute the residual from the force, independent of the slack.
    r = A @ f + w_total
    norm_w = jnp.linalg.norm(w_total)
    denom = jnp.maximum(norm_w, jnp.finfo(w_total.dtype).tiny)
    margin = jnp.linalg.norm(r) / denom
    viol = jnp.maximum(0.0, jnp.max(G @ f))
    gap = s_ineq @ z
    # Full KKT residuals on the augmented slack system, mirroring qpax's own
    # convergence test (see refs pdip.solve_qp): a consumer decides optimality
    # from these, not from the raw converged flag.
    res_stat = jnp.linalg.norm(Q @ x + q + Aeq.T @ y + Gineq.T @ z, ord=jnp.inf)
    res_eq = jnp.linalg.norm(Aeq @ x - beq, ord=jnp.inf)
    res_ineq = jnp.maximum(0.0, jnp.max(Gineq @ x - hineq))
    res_comp = jnp.linalg.norm(s_ineq * z, ord=jnp.inf)
    return (margin, f, r, viol, gap, converged, iters,
            res_stat, res_eq, res_ineq, res_comp)


@functools.lru_cache(maxsize=None)
def _jit_margin(solver_tol, max_iter):
    """One jitted margin_core per (solver_tol, max_iter), shapes flexible."""
    core = functools.partial(margin_core, solver_tol=solver_tol, max_iter=max_iter)
    return jax.jit(core)


@functools.lru_cache(maxsize=None)
def _jit_margin_vmap(solver_tol, max_iter):
    """vmap of margin_core over the batch axis, jitted once per option pair."""
    core = functools.partial(margin_core, solver_tol=solver_tol, max_iter=max_iter)
    return jax.jit(jax.vmap(core, in_axes=(0, 0, 0, None)))


def _reshape_twist(y, dim, nrows):
    """Normalized ray reshaped to per-block virtual twists.

    Row layout is [Fx, Fz, Ty] in 2D, [Fx, Fy, Fz, Tx, Ty, Tz] in 3D, so
    y reshapes to (N, 3) or (N, 6).
    """
    rpb = 3 if dim == 2 else 6
    n_blocks = nrows // rpb
    return np.asarray(y, dtype=float).reshape(n_blocks, rpb)


def _dual_cone_violation(z, mu_per_vertex, dim, k):
    """Per-vertex violation of z in the friction dual cone K*.

    z is A^T y reshaped per vertex, components (n, u) in 2D or (n, u, v)
    in 3D. Returns the largest constraint violation over vertices; <= 0
    means z is in K*. Exact closed form:
      2D wedge dual:            z_n >= mu |z_u|.
      3D inscribed-pyramid dual: z_n + mu (z_u cos a_j + z_v sin a_j) >= 0
        for every polygon ray a_j = 2 pi j / k, j = 0..k-1.
    This is the exact specialization for our G. solve_p0/solve_p2 use the
    general NNLS check in _verify_farkas; this stays as a cross-check and
    is unit tested directly.
    """
    ncomp = 2 if dim == 2 else 3
    zc = np.asarray(z, dtype=float).reshape(-1, ncomp)
    nvert = zc.shape[0]
    mu_v = np.asarray(mu_per_vertex, dtype=float).reshape(-1)
    if mu_v.size == 1:
        mu_v = np.full(nvert, mu_v[0])
    if dim == 2:
        zn, zu = zc[:, 0], zc[:, 1]
        return float(np.max(mu_v * np.abs(zu) - zn))
    zn, zu, zv = zc[:, 0], zc[:, 1], zc[:, 2]
    j = np.arange(k)
    a = 2.0 * np.pi * j / k
    vals = zn[:, None] + mu_v[:, None] * (
        np.outer(zu, np.cos(a)) + np.outer(zv, np.sin(a))
    )
    return float(np.max(-vals))


def _verify_farkas(y, A, G, w, tol):
    """Validate a candidate Farkas ray y for {A f = -w, G f <= 0}.

    A^T y lies in the dual cone K* = {c : c^T f >= 0 for all f with
    G f <= 0} iff there is z >= 0 with A^T y + G^T z = 0 (Farkas for the
    polyhedral cone). Solve that by NNLS: min over z >= 0 of
    ||G^T z + A^T y||; the residual is the dual gap. The certificate holds
    when the load power y . w > tol_power and the dual residual <=
    tol_dual. Works for both the 2D linear cone and the 3D pyramid because
    it reads G directly.

    Returns {load_power, dual_residual, certified}.
    """
    y = np.asarray(y, dtype=float)
    A = np.asarray(A, dtype=float)
    G = np.asarray(G, dtype=float)
    w = np.asarray(w, dtype=float)
    load_power = float(y @ w)
    target = -(A.T @ y)
    try:
        _z, rnorm = nnls(G.T, target)
        dual_res = float(rnorm)
    except RuntimeError:
        # NNLS hit its iteration cap; treat as an unproven ray.
        dual_res = float("inf")
    certified = bool(
        np.isfinite(load_power)
        and load_power > tol.tol_power
        and dual_res <= tol.tol_dual
    )
    return {"load_power": load_power, "dual_residual": dual_res, "certified": certified}


@dataclasses.dataclass
class _CoreVerdict:
    """Verified verdict and diagnostics from one margin_core solve."""

    status: str
    margin: float
    viol: float
    gap: float
    converged: bool
    iters: int
    finite: bool
    margin_certified: bool
    f: np.ndarray
    r: np.ndarray
    mechanism: np.ndarray | None
    farkas: dict | None
    kkt: dict = dataclasses.field(default_factory=dict)
    certified_by: str | None = None


def _classify_core(
    margin, f, r, viol, gap, converged, iters,
    res_stat, res_eq, res_ineq, res_comp,
    A, G, w_total, dim, tol, *, want_mechanism,
):
    """Turn raw margin_core outputs into a verified verdict.

    margin_certified is a full KKT check: the iterate certifies the optimal
    P4 margin only when the stationarity, equality, inequality, and
    complementarity residuals of the slack QP are each below tol_gap, matching
    qpax's own convergence criterion. tol_gap defaults to 100 * tol_feas, so
    the certificate band sits above the interior-point solver_tol target.
    """
    margin = float(margin)
    f = np.asarray(f)
    r = np.asarray(r)
    viol = float(viol)
    gap = float(gap)
    converged = bool(converged)
    iters = int(iters)
    kkt = {
        "stationarity": float(res_stat),
        "equality": float(res_eq),
        "inequality": float(res_ineq),
        "complementarity": float(res_comp),
    }
    finite = bool(
        np.isfinite(margin) and np.all(np.isfinite(f)) and np.all(np.isfinite(r))
        and all(np.isfinite(v) for v in kkt.values())
    )
    kkt_ok = all(v <= tol.tol_gap for v in kkt.values())
    margin_certified = bool(finite and kkt_ok)
    primal_ok = bool(finite and margin <= tol.tol_eq and viol <= tol.tol_cone)
    if primal_ok:
        return _CoreVerdict(
            FEASIBLE, margin, viol, gap, converged, iters, finite,
            margin_certified, f, r, None, None, kkt=kkt, certified_by="qp",
        )
    nrows = np.asarray(A).shape[0]
    norm_r = float(np.linalg.norm(r))
    farkas = None
    mechanism = None
    status = NO_CONVERGE
    certified_by = None
    if finite and norm_r > 0.0:
        y = r / norm_r
        farkas = _verify_farkas(y, A, G, w_total, tol)
        if farkas["certified"]:
            status = INFEASIBLE
            certified_by = "qp"
            if want_mechanism:
                mechanism = _reshape_twist(y, dim, nrows)
    return _CoreVerdict(
        status, margin, viol, gap, converged, iters, finite,
        margin_certified, f, r, mechanism, farkas,
        kkt=kkt, certified_by=certified_by,
    )


def _core_verdict(A, G, w_total, dim, tol, jitfn, *, want_mechanism):
    """Solve margin_core once and classify the verdict."""
    (margin, f, r, viol, gap, converged, iters,
     res_stat, res_eq, res_ineq, res_comp) = jitfn(A, w_total, G, tol.eps_reg)
    return _classify_core(
        margin, f, r, viol, gap, converged, iters,
        res_stat, res_eq, res_ineq, res_comp,
        A, G, w_total, dim, tol, want_mechanism=want_mechanism,
    )


def _p2_bound(cone, censored):
    """(physical_bound_direction, info bound tag) for a P2 load factor.

    A linear 2D uncensored associative capacity overestimates true Coulomb
    capacity, so the factor is an upper estimate. An inscribed pyramid
    combines an associative overestimate with a cone underestimate, so it is
    not ordered against true Coulomb capacity. A censored factor is only a
    cap that lower-bounds associative capacity; it is unordered vs true even
    in 2D.
    """
    if censored:
        return "unknown", "censored-lower-of-assoc-unordered-vs-true"
    if cone == "linear2d":
        return "upper", "upper-of-true-assoc-exact"
    return "unknown", "lower-of-assoc-exact-unordered-vs-true"


def _p3_bound(cone):
    """(physical_bound_direction, info bound tag) for a P3 critical friction.

    Linear 2D associative critical friction is a lower estimate of the true
    requirement. An inscribed pyramid needs more friction than exact
    associative while true Coulomb also needs more than exact associative,
    so the pyramid critical friction is unordered against true.
    """
    if cone == "linear2d":
        return "lower", "mu-lower-of-true"
    return "unknown", "mu-pyramid-unordered-vs-true"


def _base_info(v: _CoreVerdict):
    """Common info fields from a core verdict."""
    info = {
        "iters": v.iters,
        "converged": v.converged,
        "finite": v.finite,
        "viol": v.viol,
        "gap": v.gap,
        "margin_certified": v.margin_certified,
        "kkt": v.kkt,
        "formulation": "p4-slack-qpax",
    }
    if v.certified_by is not None:
        info["certified_by"] = v.certified_by
    if v.farkas is not None:
        info["farkas"] = v.farkas
        info["load_power"] = v.farkas["load_power"]
    return info


def solve_p4(
    system: EquilibriumSystem,
    tol: Tolerances,
    lam: float = 0.0,
    *,
    opts: SolverOptions = _DEFAULT_OPTS,
) -> Result:
    """P4 elastic margin at load factor lam. Always returns a Result.

    w_total = w_dead + lam * w_live. The margin is the recomputed
    equilibrium residual. Status is the verified verdict.
    """
    w_total = system.w_dead + lam * system.w_live
    jitfn = _jit_margin(opts.solver_tol, opts.max_iter)
    v = _core_verdict(system.A, system.G, w_total, system.dim, tol, jitfn,
                      want_mechanism=True)
    info = _base_info(v)
    info["lam"] = float(lam)
    forces = v.f * float(system.W)
    # P4 reports no capacity factor, so physical_bound_direction is None.
    return Result(
        status=v.status,
        margin=v.margin,
        forces=forces,
        mechanism=v.mechanism,
        cone_model=system.cone,
        physical_bound_direction=None,
        info=info,
    )


def solve_p0(
    system: EquilibriumSystem,
    tol: Tolerances,
    *,
    opts: SolverOptions = _DEFAULT_OPTS,
) -> Result:
    """P0 feasibility: P4 plus a verified verdict and a collapse mechanism.

    Feasible when the recomputed force state certifies equilibrium and the
    cone. Infeasible when the residual direction is a validated Farkas ray;
    that ray, normalized and reshaped to per-block twists, is the
    mechanism, and its load power y . w is recorded in info.
    """
    jitfn = _jit_margin(opts.solver_tol, opts.max_iter)
    v = _core_verdict(system.A, system.G, system.w_dead, system.dim, tol, jitfn,
                      want_mechanism=True)
    info = _base_info(v)
    forces = v.f * float(system.W)
    # P0 reports no capacity factor, so physical_bound_direction is None.
    return Result(
        status=v.status,
        margin=v.margin,
        forces=forces,
        mechanism=v.mechanism,
        cone_model=system.cone,
        physical_bound_direction=None,
        info=info,
    )


def _resolve_verdict(system, w_total, tol, jitfn, *, want_mechanism):
    """Certified verdict for a loaded state, escalating NO_CONVERGE.

    Returns (status, core_verdict). A NO_CONVERGE core verdict is escalated
    to the exact LP oracle on the same system with w_dead swapped for
    w_total. The returned status is then FEASIBLE or INFEASIBLE from the
    oracle, or NO_CONVERGE if the oracle also fails to decide.

    On an escalated FEASIBLE the QP iterate is uncertified, so the core
    verdict's force block is replaced by the oracle's recomputed-residual
    checked forces (nondimensional) and certified_by is set to "exact". On
    an escalated INFEASIBLE the oracle's validated Farkas ray is borrowed.
    """
    v = _core_verdict(system.A, system.G, w_total, system.dim, tol, jitfn,
                      want_mechanism=want_mechanism)
    if v.status != NO_CONVERGE:
        return v.status, v
    loaded = dataclasses.replace(system, w_dead=w_total)
    rex = solve_p0_exact(loaded, tol)
    if rex.status == INFEASIBLE:
        # Borrow the oracle's validated Farkas ray for the mechanism.
        if want_mechanism and v.mechanism is None:
            v.mechanism = rex.mechanism
        if v.farkas is None:
            v.farkas = rex.info.get("farkas")
        v.certified_by = "exact"
        return INFEASIBLE, v
    if rex.status == FEASIBLE:
        # The QP iterate is uncertified. Use the oracle's certified forces,
        # returned in SI, converted back to nondimensional for the caller.
        v.f = np.asarray(rex.forces, dtype=float) / float(system.W)
        v.certified_by = "exact"
        return FEASIBLE, v
    return NO_CONVERGE, v


def solve_p2(
    system: EquilibriumSystem,
    tol: Tolerances,
    *,
    lam_hi: float = 16.0,
    n_iter: int = 60,
    opts: SolverOptions = _DEFAULT_OPTS,
) -> Result:
    """P2 associative load factor by bisection on lambda.

    Feasibility of P4 is monotone in lambda, so the feasible set is an
    interval [0, lambda*]. lambda_assoc is the certified-feasible lower
    bracket bound. The verdict at each midpoint is verified; a NO_CONVERGE
    midpoint escalates to the exact oracle (tri-state bisection). The
    mechanism comes from the infeasible side of the final bracket, the
    force state from the feasible side.
    """
    if not (lam_hi > 0.0):
        raise ValueError(f"lam_hi must be > 0, got {lam_hi!r}")
    if not (isinstance(n_iter, int) and n_iter >= 1):
        raise ValueError(f"n_iter must be an int >= 1, got {n_iter!r}")

    jitfn = _jit_margin(opts.solver_tol, opts.max_iter)
    w_dead, w_live = system.w_dead, system.w_live
    direction, tag = _p2_bound(system.cone, False)
    # Censored / uncertified factors are unordered vs true capacity.
    cen_direction, cen_tag = _p2_bound(system.cone, True)

    def resolve(lam):
        return _resolve_verdict(system, w_dead + lam * w_live, tol, jitfn,
                                want_mechanism=True)

    info = {"n_iter": int(n_iter), "bound": tag}

    # Verdict and reported margin come from lam = 0.
    s0, v0 = resolve(0.0)
    info["iters"] = v0.iters
    info["converged"] = v0.converged
    info["margin_certified"] = v0.margin_certified

    def result(status, lam_assoc, mechanism, forces, extra, *, direction=direction):
        merged = {**info, **extra}
        return Result(
            status=status,
            margin=v0.margin,
            forces=forces,
            lambda_assoc=lam_assoc,
            mechanism=mechanism,
            cone_model=system.cone,
            physical_bound_direction=direction,
            info=merged,
        )

    if s0 == NO_CONVERGE:
        # No certified factor at all, so no bound direction is reported.
        return result(NO_CONVERGE, None, None, v0.f * float(system.W),
                      {"note": "lam=0 verdict uncertified"}, direction=None)

    if s0 == INFEASIBLE:
        # Already collapsing under dead load. lambda* is zero.
        return result(
            INFEASIBLE, 0.0, v0.mechanism, None,
            {"bracket_width": 0.0, "load_power": (v0.farkas or {}).get("load_power"),
             "forces_certified_by": v0.certified_by},
        )

    # Push lam_hi until the top of the bracket is certified infeasible,
    # tracking the largest certified-feasible lambda seen (>= 0 always,
    # since lam = 0 is certified feasible here).
    hi = float(lam_hi)
    s_hi, v_hi = resolve(hi)
    last_feas_lam = 0.0
    last_feas_v = v0
    if s_hi == FEASIBLE:
        last_feas_lam, last_feas_v = hi, v_hi
    doublings = 0
    while s_hi == FEASIBLE and doublings < 4:
        hi = min(hi * 2.0, 256.0)
        s_hi, v_hi = resolve(hi)
        if s_hi == FEASIBLE:
            last_feas_lam, last_feas_v = hi, v_hi
        doublings += 1
    info["lam_hi_used"] = float(hi)
    info["doublings"] = int(doublings)

    if s_hi == NO_CONVERGE:
        # The hi endpoint is uncertified. Never report it as the factor.
        # Fall back to the last certified-feasible lambda and mark the band
        # above it uncertified.
        return result(
            FEASIBLE, last_feas_lam, None, last_feas_v.f * float(system.W),
            {"bracket_found": False,
             "uncertified_band": (last_feas_lam, float(hi)),
             "bound": cen_tag,
             "forces_certified_by": last_feas_v.certified_by,
             "note": "hi endpoint uncertified; lambda_assoc is the last "
                     "certified-feasible lambda"},
            direction=cen_direction,
        )

    if s_hi == FEASIBLE:
        # Certified feasible all the way to the cap. lambda_assoc is a cap.
        return result(
            FEASIBLE, hi, None, v_hi.f * float(system.W),
            {"bracket_found": False, "censored": True, "bracket_width": 0.0,
             "bound": cen_tag, "forces_certified_by": v_hi.certified_by,
             "note": "feasible to cap; lambda_assoc is a cap"},
            direction=cen_direction,
        )

    # Bisect the certified interval [lo, hi] with lo certified feasible and
    # hi certified infeasible.
    lo = last_feas_lam
    feas_v = last_feas_v
    infeas_v = v_hi
    uncertified = None
    for _ in range(int(n_iter)):
        mid = 0.5 * (lo + hi)
        s_mid, v_mid = resolve(mid)
        if s_mid == FEASIBLE:
            lo = mid
            feas_v = v_mid
        elif s_mid == INFEASIBLE:
            hi = mid
            infeas_v = v_mid
        else:
            uncertified = (float(lo), float(hi))
            break

    extra = {
        "bracket_found": True,
        "bracket_width": float(hi - lo),
        "load_power": (infeas_v.farkas or {}).get("load_power"),
        "forces_certified_by": feas_v.certified_by,
    }
    if uncertified is not None:
        extra["uncertified_band"] = uncertified
        extra["note"] = "bisection stopped on an uncertified midpoint"
    return result(FEASIBLE, float(lo), infeas_v.mechanism,
                  feas_v.f * float(system.W), extra)


def solve_p3(
    system: EquilibriumSystem,
    g_of_mu,
    tol: Tolerances,
    *,
    mu_lo: float = 0.0,
    mu_hi: float = 4.0,
    n_iter: int = 60,
    opts: SolverOptions = _DEFAULT_OPTS,
) -> Result:
    """P3 associative critical friction by bisection on mu.

    g_of_mu(mu) -> G rebuilds the cone matrix for a uniform friction mu, so
    the solver stays decoupled from the mechanics cone builder. Feasibility
    of P0 is monotone nondecreasing in mu, so the feasible set is
    [mu*, inf). mu_critical_assoc is the smallest certified-feasible mu.
    The true required friction can be higher (lower estimate). Verdicts are
    verified; a NO_CONVERGE midpoint escalates to the exact oracle.
    """
    if not (isinstance(n_iter, int) and n_iter >= 1):
        raise ValueError(f"n_iter must be an int >= 1, got {n_iter!r}")

    jitfn = _jit_margin(opts.solver_tol, opts.max_iter)
    w = system.w_dead

    def resolve(mu):
        # Escalate a NO_CONVERGE QP verdict to the exact oracle on the same
        # G. On FEASIBLE take the oracle's certified forces; on INFEASIBLE
        # borrow its validated mechanism. Returns (status, core_verdict).
        loaded = dataclasses.replace(system, G=g_of_mu(mu))
        v = _core_verdict(loaded.A, loaded.G, w, loaded.dim, tol, jitfn,
                          want_mechanism=True)
        if v.status != NO_CONVERGE:
            return v.status, v
        rex = solve_p0_exact(loaded, tol)
        if rex.status == INFEASIBLE:
            if v.mechanism is None:
                v.mechanism = rex.mechanism
            if v.farkas is None:
                v.farkas = rex.info.get("farkas")
            v.certified_by = "exact"
            return INFEASIBLE, v
        if rex.status == FEASIBLE:
            v.f = np.asarray(rex.forces, dtype=float) / float(system.W)
            v.certified_by = "exact"
            return FEASIBLE, v
        return NO_CONVERGE, v

    lo = float(mu_lo)  # expected infeasible side
    hi = float(mu_hi)  # expected feasible side
    s_lo, v_lo = resolve(lo)
    s_hi, v_hi = resolve(hi)
    p3_direction, p3_tag = _p3_bound(system.cone)
    info = {
        "n_iter": int(n_iter),
        "mu_lo": lo,
        "mu_hi": hi,
        "status_at_mu_lo": s_lo,
        "status_at_mu_hi": s_hi,
        "bound": p3_tag,
    }
    if s_lo == NO_CONVERGE:
        # Unknown at mu_lo is not infeasible-at-lo. Record it and keep the
        # certified bracket; mu_critical stays the smallest certified feasible.
        info["uncertified_at_mu_lo"] = True

    def result(status, mu_crit, forces, extra, *, mechanism=None,
               direction=p3_direction):
        return Result(
            status=status,
            margin=0.0 if status == FEASIBLE else float("inf"),
            forces=forces,
            mu_critical_assoc=mu_crit,
            mechanism=mechanism,
            cone_model=system.cone,
            physical_bound_direction=direction,
            info={**info, **extra},
        )

    if s_lo == FEASIBLE:
        # Feasible even at the lowest mu. Forces come from the LOW endpoint.
        return result(FEASIBLE, lo, v_lo.f * float(system.W),
                      {"note": "feasible at mu_lo; mu_critical_assoc is a lower cap",
                       "resolution": 0.0,
                       "forces_certified_by": v_lo.certified_by})
    if s_hi == NO_CONVERGE:
        # Could not certify the top of the range. Never report as infeasible.
        return result(NO_CONVERGE, None, None,
                      {"note": "mu_hi endpoint uncertified; no certified "
                               "feasible mu found", "resolution": 0.0},
                      direction=None)
    if s_hi == INFEASIBLE:
        # Infeasible even at the highest mu (friction-independent collapse).
        # No critical friction exists; carry the mechanism from mu_hi.
        return result(INFEASIBLE, None, None,
                      {"note": "no certified feasible mu in range",
                       "resolution": 0.0},
                      mechanism=v_hi.mechanism, direction=None)

    feas_v = v_hi
    uncertified = None
    for _ in range(int(n_iter)):
        mid = 0.5 * (lo + hi)
        s_mid, v_mid = resolve(mid)
        if s_mid == FEASIBLE:
            hi = mid
            feas_v = v_mid
        elif s_mid == INFEASIBLE:
            lo = mid
        else:
            uncertified = (float(lo), float(hi))
            break
    info["resolution"] = float(hi - lo)
    if uncertified is not None:
        info["uncertified_band"] = uncertified
    return result(FEASIBLE, float(hi), feas_v.f * float(system.W),
                  {"forces_certified_by": feas_v.certified_by})


def margin_batch(
    A_batch,
    w_batch,
    G_batch,
    eps_reg,
    *,
    tol_cone: float = _DEFAULT_TOL_CONE,
    solver_tol: float = _DEFAULT_OPTS.solver_tol,
    max_iter: int = _DEFAULT_OPTS.max_iter,
):
    """Throughput kernel: recomputed margins over a padded batch.

    A_batch (B, nrows, nf), w_batch (B, nrows), G_batch (B, ncone, nf).
    Returns raw arrays (margins, certified), each (B,). margins is the
    recomputed residual margin ||A f + w|| / ||w|| per element. certified
    is True where the force is cone-admissible (viol <= tol_cone) and the
    margin is finite; it is a primal-admissibility flag, not the full P0
    verdict. The Result-typed API is solve_p4 / solve_p4_batch.
    """
    (margins, _f, _r, viol, _gap, _c, _i,
     _rs, _re, _ri, _rc) = _jit_margin_vmap(solver_tol, max_iter)(
        A_batch, w_batch, G_batch, eps_reg
    )
    certified = jnp.logical_and(viol <= tol_cone, jnp.isfinite(margins))
    return margins, certified


def solve_p4_batch(
    systems: list[EquilibriumSystem],
    tol: Tolerances,
    *,
    opts: SolverOptions = _DEFAULT_OPTS,
) -> list[Result]:
    """P4 over a list of identically shaped systems, certified per element.

    Stacks A, w_dead, G, runs the vmap kernel once, then applies the
    verified P4 verdict to each element and returns a list of Results.
    Every system must share (blocks, patches, vertices) shape.
    """
    if not systems:
        return []
    A_batch = jnp.stack([s.A for s in systems])
    w_batch = jnp.stack([s.w_dead for s in systems])
    G_batch = jnp.stack([s.G for s in systems])
    (margins, fs, rs, viols, gaps, convs, iters,
     rstat, req, rineq, rcomp) = _jit_margin_vmap(
        opts.solver_tol, opts.max_iter
    )(A_batch, w_batch, G_batch, tol.eps_reg)
    results = []
    for i, system in enumerate(systems):
        v = _classify_core(
            margins[i], fs[i], rs[i], viols[i], gaps[i], convs[i], iters[i],
            rstat[i], req[i], rineq[i], rcomp[i],
            system.A, system.G, system.w_dead, system.dim, tol,
            want_mechanism=True,
        )
        info = _base_info(v)
        info["lam"] = 0.0
        # P4 reports no capacity factor, so physical_bound_direction is None.
        results.append(
            Result(
                status=v.status,
                margin=v.margin,
                forces=v.f * float(system.W),
                mechanism=v.mechanism,
                cone_model=system.cone,
                physical_bound_direction=None,
                info=info,
            )
        )
    return results


def margin_and_grad(
    A,
    w_total,
    G,
    eps_reg,
    *,
    solver_tol: float = _DEFAULT_OPTS.solver_tol,
    max_iter: int = _DEFAULT_OPTS.max_iter,
    target_kappa: float = _DEFAULT_OPTS.target_kappa,
):
    """Value and gradient of the P4 margin with respect to w_total.

    Uses qpax.solve_qp_primal, which carries a custom vjp, so the margin
    is differentiable through the equality right-hand side beq = -w_total.
    A differentiability hook for the diff backend.

    The value is recomputed from the force block, margin = ||A f + w|| /
    ||w||, the same quantity solve_p4 reports, so value and grad refer to one
    function on any iterate (not the internal slack norm). The gradient is
    taken at a relaxed KKT point where qpax drives the complementarity
    product to target_kappa; a smaller target_kappa sharpens the gradient
    toward the true sensitivity. target_kappa never affects a verdict.
    """
    nf = A.shape[1]
    nrows = A.shape[0]
    ncone = G.shape[0]
    Q = jnp.zeros((nf + nrows, nf + nrows))
    Q = Q.at[:nf, :nf].set(eps_reg * jnp.eye(nf))
    Q = Q.at[nf:, nf:].set(jnp.eye(nrows))
    q = jnp.zeros(nf + nrows)
    Aeq = jnp.concatenate([A, -jnp.eye(nrows)], axis=1)
    Gineq = jnp.concatenate([G, jnp.zeros((ncone, nrows))], axis=1)
    hineq = jnp.zeros(ncone)

    def margin_of_w(w):
        beq = -w
        x = qpax.solve_qp_primal(
            Q, q, Aeq, beq, Gineq, hineq, solver_tol=solver_tol,
            target_kappa=target_kappa, max_iter=max_iter,
        )
        f = x[:nf]
        # Recompute from the force so value matches solve_p4 on any iterate.
        r = A @ f + w
        denom = jnp.maximum(jnp.linalg.norm(w), jnp.finfo(w.dtype).tiny)
        return jnp.linalg.norm(r) / denom

    value, grad = jax.value_and_grad(margin_of_w)(w_total)
    return value, grad
