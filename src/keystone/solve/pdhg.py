"""First-order P4 screening by a fixed-iteration primal-dual kernel.

This is a screener, not a certifier. It answers the same P4 question as
solve.batch_jax.margin_core (minimize the equilibrium residual over the
friction cone) but with a matvec-only Condat-Vu iteration instead of an
interior-point QP. Matvecs are cheap, warm-startable, and vmap cleanly,
so the search can screen many candidates per second. FEASIBLE claims that
reach a user still come from the certified path; see docs/KNOWN_LIMITS.md
for the screening-semantics entry and the empirical error direction.

Problem (folded P4, same objective as margin_core, cone kept explicit):

    minimize   0.5 ||A f + w||^2 + 0.5 eps_reg ||f||^2
    subject to G f <= 0.

Write it as min_f phi(f) + h(G f) with phi smooth and h the indicator of
the nonpositive orthant. Then

    grad phi(f) = A^T (A f + w) + eps_reg f,
    h*(y)       = indicator of {y >= 0},  so prox_{sigma h*} = max(0, .).

Condat-Vu (Condat 2013, Vu 2013) primal-dual step, one iteration:

    f_new = f - tau (grad phi(f) + G^T y),
    y_new = max(0, y + sigma G (2 f_new - f)).

The 2 f_new - f term is the primal over-relaxation the dual step reads.

Step sizes (deterministic, jit-friendly, no power iteration, no
randomness). Condat 2013 converges when

    tau (Lf / 2 + sigma ||G||_2^2) <= 1,

where Lf is the Lipschitz constant of grad phi. We use safe upper bounds
from Frobenius norms: ||A||_2 <= ||A||_F and ||G||_2 <= ||G||_F, so

    Lf <= ||A||_F^2 + eps_reg   and   ||G||_2^2 <= ||G||_F^2.

We split the budget by a fixed primal fraction c in (0, 1):

    tau   = 2 c / Lf,                       so tau Lf / 2      = c,
    sigma = (1 - c) / (tau ||G||_F^2),      so tau sigma ||G||_F^2 = 1 - c,

which meets the condition with equality, tau (Lf/2 + sigma ||G||_F^2) = 1.
c = _PRIMAL_FRAC = 0.9 favors the primal step. The Frobenius bound
overestimates ||A||_2, so tau = 1.8 / Lf stays below the plain
gradient-descent limit 2 / ||A||_2^2 and the whole rule is conservative.
c is a step-size split, not a feasibility tolerance, so it is a fixed
algorithm constant here and not a Tolerances field.

Acceleration. On this problem family the equilibrium operator is
ill-conditioned (cond(A^T A) reaches a few thousand), so the plain
iteration needs thousands of steps to drive the residual to tol_feas. A
Nesterov extrapolation on the primal iterate with adaptive gradient
restart (accel=True, default) wraps the same Condat-Vu step and cuts
that to a few hundred. The extrapolation is primal-only on purpose: an
infeasible instance has no saddle point and its dual iterate grows
without bound, so dual momentum compounds that growth geometrically
(observed overshoot by 1e5 on infeasible lattice states), while the
plain dual update stays bounded-growth and the reported margin then
tracks the true infimum. Restart follows O'Donoghue and Candes 2015:
momentum resets when the extrapolation direction opposes the step. The
wrapper adds no dependency, no randomness, and no per-iteration linear
algebra beyond the base step. accel=False recovers the plain scheme
exactly. The base step sizes above still satisfy the Condat-Vu
condition; the extrapolation is a heuristic on top and carries no
separate guarantee.

margin and viol are recomputed from the final force by matvec, with the
same definitions as margin_core:

    margin = ||A f + w|| / max(||w||, tiny),
    viol   = max(0, max(G f)).
"""

import functools

import jax
import jax.numpy as jnp

# Primal fraction of the Condat-Vu step-size budget. A fixed step-size
# split (see the module docstring), not a feasibility tolerance.
_PRIMAL_FRAC = 0.9


def _t_next(t):
    """FISTA momentum sequence update t_{k+1} = (1 + sqrt(1 + 4 t_k^2)) / 2."""
    return 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t * t))


def pdhg_margin(A, w, G, eps_reg, *, iters, f0=None, y0=None, accel=True):
    """Fixed-iteration Condat-Vu screen of the folded P4 problem.

    Pure jnp, jit and vmap safe, float64. Consumes matrices only, so 2D
    and 3D share it. iters is a fixed loop count (a solver option, like
    SolverOptions fields), so the cost is deterministic.

    Warm start from (f0, y0) when given, zeros otherwise. accel enables the
    Nesterov extrapolation with adaptive restart (default); accel=False is
    the plain scheme.

    Returns (margin, f, y, viol):
      margin  ||A f + w|| / max(||w||, tiny), recomputed by matvec.
      f       the final primal force iterate.
      y       the final dual iterate (nonnegative), a warm start for reuse.
      viol    cone violation max(0, max(G f)); 0 means cone-admissible.
    """
    nf = A.shape[1]
    ncone = G.shape[0]
    tiny = jnp.finfo(A.dtype).tiny

    # Safe upper bounds from Frobenius norms. No power iteration.
    normA2 = jnp.sum(A * A)  # >= ||A||_2^2
    normG2 = jnp.sum(G * G)  # >= ||G||_2^2
    Lf = normA2 + eps_reg  # >= Lipschitz constant of grad phi
    tau = 2.0 * _PRIMAL_FRAC / Lf
    sigma = (1.0 - _PRIMAL_FRAC) / (tau * jnp.maximum(normG2, tiny))

    f = jnp.zeros(nf) if f0 is None else f0
    y = jnp.zeros(ncone) if y0 is None else y0
    f_prev = f
    t = jnp.asarray(1.0, dtype=A.dtype)

    def step(fe, y):
        # One Condat-Vu iteration from the (possibly extrapolated) primal.
        grad_phi = A.T @ (A @ fe + w) + eps_reg * fe
        f_new = fe - tau * (grad_phi + G.T @ y)
        y_new = jnp.maximum(0.0, y + sigma * (G @ (2.0 * f_new - fe)))
        return f_new, y_new

    def body(_i, carry):
        f, f_prev, y, t = carry
        if accel:
            # Primal-only Nesterov extrapolation, then the Condat-Vu step.
            beta = (t - 1.0) / _t_next(t)
            fe = f + beta * (f - f_prev)
            f_new, y_new = step(fe, y)
            t_new = _t_next(t)
            # Adaptive gradient restart: reset momentum when the step and
            # the extrapolation disagree (O'Donoghue and Candes 2015).
            restart = jnp.sum((fe - f_new) * (f_new - f)) > 0.0
            t_new = jnp.where(restart, 1.0, t_new)
        else:
            f_new, y_new = step(f, y)
            t_new = t
        return (f_new, f, y_new, t_new)

    f, f_prev, y, t = jax.lax.fori_loop(0, iters, body, (f, f_prev, y, t))

    r = A @ f + w
    denom = jnp.maximum(jnp.linalg.norm(w), tiny)
    margin = jnp.linalg.norm(r) / denom
    viol = jnp.maximum(0.0, jnp.max(G @ f))
    return margin, f, y, viol


@functools.lru_cache(maxsize=None)
def _jit_batch(iters, accel):
    """vmap of pdhg_margin over the batch axis, jitted once per option pair.

    f0 and y0 are batched (leading axis) or None (broadcast cold start).
    """

    def core(A, w, G, eps_reg, f0, y0):
        return pdhg_margin(A, w, G, eps_reg, iters=iters, f0=f0, y0=y0, accel=accel)

    def with_warm(A, w, G, eps_reg, f0, y0):
        fn = jax.vmap(core, in_axes=(0, 0, 0, None, 0, 0))
        return fn(A, w, G, eps_reg, f0, y0)

    def cold(A, w, G, eps_reg):
        fn = jax.vmap(
            lambda A, w, G: pdhg_margin(A, w, G, eps_reg, iters=iters, accel=accel),
            in_axes=(0, 0, 0),
        )
        return fn(A, w, G)

    return jax.jit(with_warm), jax.jit(cold)


def pdhg_margin_batch(
    A_batch, w_batch, G_batch, eps_reg, *, iters, f0=None, y0=None, accel=True
):
    """Screen a padded batch of identically shaped systems.

    A_batch (B, nrows, nf), w_batch (B, nrows), G_batch (B, ncone, nf).
    Warm start from f0 (B, nf) and y0 (B, ncone) when both are given, else
    cold. Mirrors the batched margin evaluation the search uses (see
    keystone.search.lattice), so swapping it in for the qpax kernel is a
    function swap.

    Returns (margins, fs, ys, viols):
      margins (B,)         recomputed residual margins.
      fs      (B, nf)      final force iterates, reusable warm starts.
      ys      (B, ncone)   final dual iterates, reusable warm starts.
      viols   (B,)         cone violations.
    """
    warm_fn, cold_fn = _jit_batch(int(iters), bool(accel))
    if f0 is not None and y0 is not None:
        return warm_fn(A_batch, w_batch, G_batch, eps_reg, f0, y0)
    return cold_fn(A_batch, w_batch, G_batch, eps_reg)
