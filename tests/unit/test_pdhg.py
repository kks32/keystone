"""Unit tests and validation study for the first-order P4 screener.

The kernel tests pin pdhg_margin against margin_core on hand-built and
lattice systems: same margin and viol definitions, determinism, warm
starts, and the batch wrapper.

The validation study is the screening-semantics gate. On 500 random
reachable lattice states (250 each from the n=4 and n=6 specs, legal
rollouts, seed 0) plus all constructible states within 2 grid steps of
the analytic boundaries (stacked pair around e = b/2, harmonic corbel
around c = 1), it compares pdhg verdicts at iters in {50, 100, 200, 400}
against certified margin_core verdicts, cold and warm started. The
asserted properties:

- Zero false-feasibles at every iteration count, cold and warm. This is
  the empirical one-sided error direction of the screen on this problem
  family, not a theorem (docs/KNOWN_LIMITS.md).
- The n=4 sims=500 seed=0 search returns the same best overhang with
  screener="qpax" and screener="pdhg" at the default pdhg_iters.

False-infeasible rates are printed, not asserted: they stay far above
2 percent at every tested iteration count (boundary-adjacent feasible
states have certified margins within 10x of tol_feas, unresolvable by a
few hundred first-order iterations). The search compensates by gating
every best-overhang update on a certified qpax re-verification, which is
what the same-best assertion exercises end to end.
"""

import numpy as np
import pytest

import jax.numpy as jnp

from keystone import Tolerances
from keystone.search import Search
from keystone.search import lattice as LT
from keystone.solve.batch_jax import margin_core
from keystone.solve.pdhg import pdhg_margin, pdhg_margin_batch

TOL = Tolerances()
DX = 1.0 / 24.0
DEFAULT_ITERS = 400
STUDY_ITERS = (50, 100, 200, 400)

# Toy single 2D block on a centered ground patch, same fixture as
# test_solve_synthetic. f = [n0, u0, n1, u1], rows [Fx, Fz, Ty].
A_BLOCK = np.array(
    [
        [0.0, -1.0, 0.0, -1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.5, 0.5, -0.5, 0.5],
    ]
)
W_DEAD = np.array([0.0, -1.0, 0.0])
W_LIVE = np.array([1.0, 0.0, 0.0])


def cone_2d(mu, nvert=2):
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


def lattice_system(spec, key):
    state = LT.state_from_placements(spec, key)
    return LT.build_system(spec, state)


# =========================================================================
# Kernel unit tests.
# =========================================================================


class TestKernel:
    def test_feasible_block_converges(self):
        # Stable block: margin must reach the feasibility tolerance.
        A = jnp.asarray(A_BLOCK)
        w = jnp.asarray(W_DEAD)
        G = jnp.asarray(cone_2d(0.5))
        m, f, y, v = pdhg_margin(A, w, G, TOL.eps_reg, iters=DEFAULT_ITERS)
        assert float(m) <= TOL.tol_feas
        assert float(v) <= TOL.tol_cone
        # The returned force equilibrates gravity: uniform compression 0.5.
        r = np.asarray(A) @ np.asarray(f) + np.asarray(w)
        assert np.linalg.norm(r) < 1e-8

    def test_infeasible_load_stays_infeasible(self):
        # lambda = 2 > lambda* = 0.3 for mu = 0.3: margin must stay large.
        A = jnp.asarray(A_BLOCK)
        w = jnp.asarray(W_DEAD + 2.0 * W_LIVE)
        G = jnp.asarray(cone_2d(0.3))
        m_ref = float(
            margin_core(A, w, G, TOL.eps_reg, solver_tol=1e-9, max_iter=100)[0]
        )
        m, f, y, v = pdhg_margin(A, w, G, TOL.eps_reg, iters=DEFAULT_ITERS)
        assert float(m) > TOL.tol_feas
        # Screened margin is an upper estimate of the optimal residual.
        assert float(m) >= m_ref - 1e-9

    def test_margin_and_viol_match_margin_core_definitions(self):
        # Same recomputed definitions: feed margin_core's own force back.
        spec = LT.LatticeSpec(n_max=4, dx=DX)
        A, w, G, L, W = lattice_system(spec, ((0, -12),))
        mq, fq, rq, vq = margin_core(
            A, w, G, TOL.eps_reg, solver_tol=1e-9, max_iter=100
        )[:4]
        # Zero-iteration pdhg from margin_core's iterate returns exactly the
        # recomputed margin and viol of that force.
        m0, f0, y0, v0 = pdhg_margin(
            A, w, G, TOL.eps_reg, iters=0, f0=fq, y0=jnp.zeros(G.shape[0])
        )
        assert float(m0) == pytest.approx(float(mq), abs=1e-15)
        assert float(v0) == pytest.approx(float(vq), abs=1e-15)

    def test_deterministic(self):
        spec = LT.LatticeSpec(n_max=4, dx=DX)
        A, w, G, L, W = lattice_system(spec, ((0, -12), (1, -6)))
        m1, f1, y1, v1 = pdhg_margin(A, w, G, TOL.eps_reg, iters=200)
        m2, f2, y2, v2 = pdhg_margin(A, w, G, TOL.eps_reg, iters=200)
        assert np.array_equal(np.asarray(f1), np.asarray(f2))
        assert float(m1) == float(m2)

    def test_float64(self):
        spec = LT.LatticeSpec(n_max=4, dx=DX)
        A, w, G, L, W = lattice_system(spec, ((0, -12),))
        m, f, y, v = pdhg_margin(A, w, G, TOL.eps_reg, iters=50)
        assert f.dtype == jnp.float64
        assert y.dtype == jnp.float64

    def test_warm_start_improves(self):
        # Chaining two 200-iteration solves beats one cold 200-iteration
        # solve: the warm start is read and helps.
        spec = LT.LatticeSpec(n_max=4, dx=DX)
        A, w, G, L, W = lattice_system(spec, ((0, -12), (1, -6)))
        m_cold, f1, y1, v1 = pdhg_margin(A, w, G, TOL.eps_reg, iters=200)
        m_warm, f2, y2, v2 = pdhg_margin(
            A, w, G, TOL.eps_reg, iters=200, f0=f1, y0=y1
        )
        assert float(m_warm) < float(m_cold)

    def test_plain_scheme_monotone_option(self):
        # accel=False runs the plain Condat-Vu scheme and stays finite and
        # conservative on an infeasible state.
        spec = LT.LatticeSpec(n_max=4, dx=DX)
        A, w, G, L, W = lattice_system(
            spec, ((0, 10), (1, 22), (2, 34), (3, 46))
        )
        m, f, y, v = pdhg_margin(A, w, G, TOL.eps_reg, iters=400, accel=False)
        assert np.isfinite(float(m))
        assert float(m) > TOL.tol_feas

    def test_batch_matches_single(self):
        spec = LT.LatticeSpec(n_max=4, dx=DX)
        keys = [((0, -12),), ((0, -6), (1, 0)), ((0, 10), (1, 22))]
        As, ws, Gs = [], [], []
        for k in keys:
            A, w, G, L, W = lattice_system(spec, k)
            As.append(A)
            ws.append(w)
            Gs.append(G)
        Ab, wb, Gb = jnp.stack(As), jnp.stack(ws), jnp.stack(Gs)
        mb, fb, yb, vb = pdhg_margin_batch(Ab, wb, Gb, TOL.eps_reg, iters=200)
        for i, k in enumerate(keys):
            m, f, y, v = pdhg_margin(As[i], ws[i], Gs[i], TOL.eps_reg, iters=200)
            assert float(mb[i]) == pytest.approx(float(m), rel=0, abs=1e-14)
            np.testing.assert_allclose(np.asarray(fb[i]), np.asarray(f), atol=1e-14)

    def test_batch_warm_start(self):
        spec = LT.LatticeSpec(n_max=4, dx=DX)
        A, w, G, L, W = lattice_system(spec, ((0, -12),))
        Ab = jnp.stack([A, A])
        wb = jnp.stack([w, w])
        Gb = jnp.stack([G, G])
        m1, f1, y1, v1 = pdhg_margin_batch(Ab, wb, Gb, TOL.eps_reg, iters=100)
        m2, f2, y2, v2 = pdhg_margin_batch(
            Ab, wb, Gb, TOL.eps_reg, iters=100, f0=f1, y0=y1
        )
        assert float(m2[0]) < float(m1[0])


# =========================================================================
# Validation study: screening error direction, cold and warm.
# =========================================================================


def rollout_chains(spec, n_chains, seed=0):
    """Random legal rollout chains of prefix keys, fixed seed."""
    rng = np.random.default_rng(seed)
    cand_L_j, cand_J_j = LT.action_grid(spec)
    cand_L = np.asarray(cand_L_j)
    cand_J = np.asarray(cand_J_j)
    chains = []
    while len(chains) < n_chains:
        placements = []
        chain = []
        for _ in range(spec.n_max):
            legal = np.asarray(
                LT.legal_grid(
                    spec,
                    LT.batch_states(spec, [tuple(placements)]),
                    cand_L_j,
                    cand_J_j,
                )
            )[0]
            idx = np.nonzero(legal)[0]
            if idx.size == 0:
                break
            pick = int(rng.choice(idx))
            placements.append((int(cand_L[pick]), int(cand_J[pick])))
            chain.append(tuple(sorted(placements)))
        if chain:
            chains.append(chain)
    return chains


def unique_states(chains, cap):
    keys = []
    seen = set()
    for ch in chains:
        for k in ch:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    return keys[:cap]


def boundary_states():
    """States within 2 grid steps of constructible analytic boundaries.

    Stacked pair: base cube at (0, -48), top cube at layer 1 with center
    offset e; the boundary is e = b/2, which is 12 grid steps at dx = 1/24.
    Harmonic corbel (n = 4): consecutive center offsets 4, 6, 12 grid steps
    over a base cube whose edge overhangs the pedestal by 3 steps put the
    total overhang at the analytic limit 25/24 exactly; sweeping the whole
    stack by -2..2 steps crosses the boundary.
    """
    states = []
    for off in range(10, 15):
        states.append(((0, -48), (1, -48 + off)))
    for s in range(-2, 3):
        states.append(
            tuple(sorted([(0, -9 + s), (1, -5 + s), (2, 1 + s), (3, 13 + s)]))
        )
    return states


def certified_verdicts(spec, keys):
    states = LT.batch_states(spec, keys)
    m, c = LT.margins_of_states(
        spec, states, TOL.eps_reg, TOL.tol_cone, solver_tol=1e-9, max_iter=100
    )
    m = np.asarray(m)
    c = np.asarray(c)
    return m, (m <= TOL.tol_feas) & c


def stacked_systems(spec, keys):
    As, ws, Gs = [], [], []
    for k in keys:
        A, w, G, L, W = lattice_system(spec, k)
        As.append(A)
        ws.append(w)
        Gs.append(G)
    return jnp.stack(As), jnp.stack(ws), jnp.stack(Gs)


def screen_cold(spec, keys, iters):
    Ab, wb, Gb = stacked_systems(spec, keys)
    m, f, y, v = pdhg_margin_batch(Ab, wb, Gb, TOL.eps_reg, iters=iters)
    m = np.asarray(m)
    v = np.asarray(v)
    feas = (m <= TOL.tol_feas) & (v <= TOL.tol_cone) & np.isfinite(m)
    return m, feas


def screen_warm(spec, chains, keys, iters):
    """Screen every chain with parent warm starts, the search regime."""
    results = {}
    for ch in chains:
        f_prev = None
        y_prev = None
        for k in ch:
            A, w, G, L, W = lattice_system(spec, k)
            m, f, y, v = pdhg_margin(
                A, w, G, TOL.eps_reg, iters=iters, f0=f_prev, y0=y_prev
            )
            f_prev, y_prev = f, y
            if k not in results:
                results[k] = (float(m), float(v))
    m = np.array([results[k][0] for k in keys])
    v = np.array([results[k][1] for k in keys])
    feas = (m <= TOL.tol_feas) & (v <= TOL.tol_cone) & np.isfinite(m)
    return m, feas


class TestValidationStudy:
    """500 rollout states plus boundary states, fixed seed, printed report."""

    spec4 = LT.LatticeSpec(n_max=4, dx=DX)
    spec6 = LT.LatticeSpec(n_max=6, dx=DX)

    @classmethod
    def setup_class(cls):
        cls.chains4 = rollout_chains(cls.spec4, 80, seed=0)
        cls.chains6 = rollout_chains(cls.spec6, 60, seed=0)
        cls.keys4 = unique_states(cls.chains4, 250)
        cls.keys6 = unique_states(cls.chains6, 250)
        cls.bnd = boundary_states()
        cls.gt = {}
        for tag, spec, keys in [
            ("n4", cls.spec4, cls.keys4),
            ("n6", cls.spec6, cls.keys6),
            ("bnd", cls.spec4, cls.bnd),
        ]:
            cls.gt[tag] = certified_verdicts(spec, keys)

    def test_study_size(self):
        assert len(self.keys4) == 250
        assert len(self.keys6) == 250
        assert len(self.bnd) == 10
        # The sets span both verdicts.
        for tag in ("n4", "n6", "bnd"):
            gm, gf = self.gt[tag]
            assert 0 < int(gf.sum()) < len(gf)

    def test_zero_false_feasible_cold_all_iters(self):
        # The asserted screening property: the screen never calls a
        # certified-infeasible state feasible, at any studied iteration
        # count. False-infeasible rates are reported, not asserted.
        print("\ncold screen vs certified, per iteration count:")
        for iters in STUDY_ITERS:
            for tag, spec, keys in [
                ("n4", self.spec4, self.keys4),
                ("n6", self.spec6, self.keys6),
                ("bnd", self.spec4, self.bnd),
            ]:
                gm, gf = self.gt[tag]
                pm, pf = screen_cold(spec, keys, iters)
                ff = int(np.sum(pf & ~gf))
                fi = int(np.sum(~pf & gf))
                nfeas = int(gf.sum())
                err = np.abs(pm - gm)
                print(
                    f"  iters={iters:4d} {tag:4s} false_feas={ff} "
                    f"false_infeas={fi}/{nfeas} "
                    f"err med={np.median(err):.1e} "
                    f"p90={np.quantile(err, 0.9):.1e} "
                    f"max={np.max(err):.1e}"
                )
                assert ff == 0, (iters, tag)

    def test_zero_false_feasible_warm_default_iters(self):
        # Warm starts (the search regime) keep the error direction.
        print("\nwarm screen vs certified at the default iteration count:")
        for tag, spec, chains, keys in [
            ("n4", self.spec4, self.chains4, self.keys4),
            ("n6", self.spec6, self.chains6, self.keys6),
        ]:
            gm, gf = self.gt[tag]
            pm, pf = screen_warm(spec, chains, keys, DEFAULT_ITERS)
            ff = int(np.sum(pf & ~gf))
            fi = int(np.sum(~pf & gf))
            nfeas = int(gf.sum())
            print(
                f"  iters={DEFAULT_ITERS} {tag:4s} false_feas={ff} "
                f"false_infeas={fi}/{nfeas}"
            )
            assert ff == 0, tag

    def test_screen_margins_finite_at_default(self):
        # No blowup: the primal-only momentum keeps every screened margin
        # finite, including on infeasible states.
        for tag, spec, keys in [
            ("n4", self.spec4, self.keys4),
            ("n6", self.spec6, self.keys6),
        ]:
            pm, pf = screen_cold(spec, keys, DEFAULT_ITERS)
            assert np.all(np.isfinite(pm)), tag
            assert np.max(pm) < 10.0, tag


# =========================================================================
# Search integration: same best overhang with both screeners.
# =========================================================================


class TestSearchIntegration:
    def test_same_best_overhang_n4_sims500_seed0(self):
        tol = Tolerances()
        s_qpax = Search(4, DX, tol, seed=0, search_iter=50, screener="qpax")
        best_qpax = s_qpax.run(500)
        s_pdhg = Search(
            4, DX, tol, seed=0, search_iter=50, screener="pdhg",
            pdhg_iters=DEFAULT_ITERS,
        )
        best_pdhg = s_pdhg.run(500)
        print(
            f"\nqpax best={best_qpax:.6f} ({s_qpax.n_qp} QPs), "
            f"pdhg best={best_pdhg:.6f} ({s_pdhg.n_qp} screens, "
            f"{s_pdhg.n_reverify} certified re-verifies)"
        )
        assert best_pdhg == pytest.approx(best_qpax, abs=1e-12)
        assert s_pdhg.best_sequence() == s_qpax.best_sequence()
        # Re-verification volume stays a small fraction of the screens.
        assert s_pdhg.n_reverify < 0.15 * s_pdhg.n_qp

    def test_best_update_gated_on_certified_verdict(self):
        # Every best_key the pdhg search reports has a certified-feasible
        # entry in its certification cache.
        tol = Tolerances()
        s = Search(3, DX, tol, seed=1, search_iter=50, screener="pdhg",
                   pdhg_iters=DEFAULT_ITERS)
        s.run(200)
        assert s.best_key is not None
        assert s._cert_cache.get(s.best_key) is True

    def test_screener_option_validation(self):
        with pytest.raises(ValueError):
            Search(3, DX, Tolerances(), screener="bogus")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
