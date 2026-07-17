"""Batched PUCT tree search over the jittable lattice environment.

Same search as examples/search_overhang.py: PUCT with a uniform prior, the
same c_puct and leaf-value heuristic, a transposition table keyed by the
sorted placement set, solved-subtree pruning, and a batched feasibility
oracle. The difference is the oracle. The naive script rebuilds a numpy
Assembly per candidate on the host and then solves a CPU batch QP. Here
the whole geometry-to-margin path is the pure-JNP lattice environment, so
the legal frontier of many leaves is built and solved in jitted, vmapped
calls with no host-side geometry.

Per iteration the search runs K selections with virtual loss so the K
paths diverge, collects the K leaf states, decides feasibility of every
legal child in batched kernel calls, updates the tree, removes the virtual
losses, and backs up the real leaf values. K defaults to 16.

Feasibility for the search is margin <= tol.tol_feas AND certified, where
certified is the margin_core meaning: cone-admissible and finite. The best
overhang is tracked over every certified-feasible state the search visits,
and the reported best sequence is re-verified with the host pipeline.

screener="pdhg" swaps the expansion oracle for the first-order screen in
keystone.solve.pdhg, warm started from each candidate's parent. Screened
verdicts decide expansion admissibility only. Any state that would improve
the best overhang is first re-verified with the certified qpax kernel
(counted in n_reverify), so a screener false-feasible can never corrupt
the best-overhang tracking. The final host-pipeline re-verification of the
best sequence is unchanged. See docs/KNOWN_LIMITS.md, screening semantics.

Stochasticity and determinism: one numpy Generator seeds the run and now
actually drives exploration. PUCT ties break by a seeded epsilon jitter
(tie_eps, relative to the O(1) score scale) instead of always by the first
sorted action, so two different seeds explore genuinely different trees while
one seed reproduces its own run bit for bit. Optional root Dirichlet noise
(root_noise, the AlphaZero root prior perturbation) and optional
temperature action sampling (sample_action) also draw from the same
Generator. All three default off nowhere they matter for reproducibility:
tie_eps is on so seeds diverge, root_noise defaults off (self-play data
collection turns it on, pure evaluation leaves it off), and temperature
defaults to argmax.

Best-overhang tracking is reported two ways. best_overhang is the best over
every certified-feasible state the search visits at any depth (best-at-most-n).
best_by_count[c] is the best over states using exactly c cubes, so a size-n
run can report its exact-n design separately from a shorter design that
happens to reach further. The two differ whenever the deepest feasible design
is not the widest one.
"""

import math

import jax.numpy as jnp
import numpy as np

from ..geometry.tolerances import Tolerances
from ..solve.options import SolverOptions
from . import lattice as LT

# Batch shaping for the certified solve. The frontier is padded up to a
# power of two (capped at CHUNK) so the number of compiled shapes stays
# small. Padding slots repeat one real state and their results are dropped.
CHUNK = 256


class Search:
    """PUCT search with virtual-loss batching and a jitted lattice oracle."""

    def __init__(
        self,
        n,
        dx,
        tol: Tolerances,
        c_puct=1.4,
        seed=0,
        batch=16,
        search_iter=50,
        opts: SolverOptions = SolverOptions(),
        screener="qpax",
        pdhg_iters=400,
        pdhg_accel=True,
        prior_fn=None,
        value_fn=None,
        lam_min=None,
        tie_eps=1e-9,
        root_noise=False,
        dirichlet_alpha=0.3,
        dirichlet_frac=0.25,
        temperature=0.0,
    ):
        self.n = n
        self.dx = dx
        self.tol = tol
        self.c_puct = c_puct
        self.K = int(batch)
        self.rng = np.random.default_rng(seed)
        # Real stochasticity, all off the same seeded Generator so a seed
        # reproduces its run and distinct seeds genuinely diverge.
        # tie_eps: uniform jitter added to each PUCT score, on the O(1) score
        #   scale (default 1e-9), so exact ties break randomly per seed instead
        #   of always by the first sorted action. It never overrides a real
        #   score gap. Set 0.0 to recover first-action tie-breaking.
        # root_noise: mix Dirichlet(dirichlet_alpha) noise into the root prior
        #   at weight dirichlet_frac. Default off; self-play data collection
        #   turns it on, pure evaluation leaves it off.
        # temperature: softmax temperature for sample_action over visit counts.
        #   Default 0.0 is argmax (eval). Positive values sample self-play moves.
        self.tie_eps = float(tie_eps)
        self.root_noise = bool(root_noise)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_frac = float(dirichlet_frac)
        self.temperature = float(temperature)
        # Root Dirichlet draw over the root's admissible actions, filled once
        # when the root expands (only when root_noise is on).
        self._root_noise = None
        # Optional learned-search hooks. prior_fn(state_key, action_indices)
        # returns a prior probability vector over those actions, replacing the
        # uniform PUCT prior. value_fn(state_key) returns a scalar in [0, 1]
        # replacing the heuristic value at a non-terminal leaf. Both default to
        # None, which reproduces the uniform-prior heuristic-value search
        # exactly: every new code path is guarded on `is not None`, so the None
        # path runs the same arithmetic as before. Certification is untouched;
        # the hooks only shape exploration.
        self.prior_fn = prior_fn
        self.value_fn = value_fn
        # qpax iteration cap for the search oracle. Matches the naive default.
        # Fewer iterations only flip a near-boundary state from feasible to
        # infeasible (conservative), so a state called feasible here is
        # feasible at the library default too. The best sequence is
        # re-verified with the default cap.
        self.search_iter = int(search_iter)
        self.solver_tol = opts.solver_tol

        # Feasibility screener. "qpax" is the certified interior-point path
        # and the default, so recorded behavior is unchanged. "pdhg" is the
        # first-order screener: it decides expansion admissibility, warm
        # starts each candidate from its parent, and never updates the best
        # overhang without a certified qpax re-verification (see _expand_batch).
        if screener not in ("qpax", "pdhg"):
            raise ValueError(f"screener must be 'qpax' or 'pdhg', got {screener!r}")
        self.screener = screener
        self.pdhg_iters = int(pdhg_iters)
        self.pdhg_accel = bool(pdhg_accel)
        # Reserve mode. lam_min not None makes feasibility mean lam-robustness
        # (P4 feasible under +-lam_min lateral load), so both the frontier solve
        # and the best-overhang re-verify certify reserve. It rides the
        # certified qpax path only. None (the default) keeps the plain
        # feasibility oracle and traces exactly as before.
        self.robust = lam_min is not None
        self.lam_min = None if lam_min is None else float(lam_min)
        if self.robust and screener != "qpax":
            raise ValueError("lam_min reserve mode requires screener='qpax'")
        # Screened (f, y) iterates per state key, for warm starting children.
        self.screened_f = {}
        self.screened_y = {}
        # Parent key per child key, so the pdhg frontier can warm start.
        self._parent_of = {}
        # Certified re-verifications of best-improving screened states.
        self._cert_cache = {}
        self.n_reverify = 0
        self.t_reverify = 0.0

        self.spec = LT.LatticeSpec(n_max=n, dx=dx)
        cand_L, cand_J = LT.action_grid(self.spec)
        self.cand_L = cand_L
        self.cand_J = cand_J
        self.cand_L_np = np.asarray(cand_L)
        self.cand_J_np = np.asarray(cand_J)
        self.M = int(cand_L.shape[0])

        self.value_norm = 2.0 * LT.harmonic(n)  # beating harmonic maps Q near 0.5

        self.tree = {}
        self.feas_cache = {}  # child key -> (feasible, margin, certified)

        self.best_overhang = float("-inf")
        self.best_key = None
        self.best_parent = None
        self.best_action = None
        # Per-block-count bests, so exact-n and best-at-most-n are reported
        # separately. best_by_count[c] = (overhang, key, parent_key, action)
        # for the best certified-feasible state that uses exactly c cubes.
        self.best_by_count = {}

        # Full DAG child edges recorded at expansion. A state (canonical key)
        # can be reached from several parents in the transposition table; the
        # single node["parent"] field keeps only the first. Every expanded node
        # enumerates its complete legal-child frontier, so recording those
        # edges here captures every parent->child relation. Subtree-value
        # targets (az._subtree_targets) aggregate over these, not one parent.
        self._children_of = {}
        self._parents_of = {}

        # Instrumentation.
        self.t_legal = 0.0  # legality passes, seconds
        self.t_solve = 0.0  # certified batch QP, seconds
        self.n_qp = 0  # real (unpadded) QP feasibility solves
        self.n_cache_hits = 0
        self.n_expansions = 0
        self.n_revisits = 0

        self.root = self._node((), None, None)

    # --- node bookkeeping -------------------------------------------------

    def _node(self, key, parent, action_in):
        node = {
            "key": key,
            "parent": parent,
            "action_in": action_in,
            "expanded": False,
            "terminal": False,
            "solved": False,
            "actions": [],
            "N": 0,
            "N_a": {},
            "W_a": {},
            "P_a": {},  # learned priors per action, filled only when prior_fn set
        }
        self.tree[key] = node
        return node

    @staticmethod
    def _canonical(placements):
        return tuple(sorted(placements))

    def _child_key(self, key, action):
        return self._canonical(key + (action,))

    def _action_index(self, action):
        """Layer-major grid index of an (layer, xidx) action.

        Matches lattice.action_grid: index = layer * n_pos + (xidx - j_lo).
        The index is the same integer a learned model uses for this action,
        because the model shares the layer-major convention and the same n_pos.
        """
        layer, xidx = action
        return layer * self.spec.n_pos + (xidx - self.spec.j_lo)

    # --- feasibility oracle ----------------------------------------------

    def _solve_frontier(self, child_keys):
        """Feasibility for a list of child state keys, cache-aware and batched.

        Returns nothing; fills self.feas_cache. Only uncached states hit the
        solver. The screener chooses the kernel: qpax is the certified
        interior point, pdhg is the first-order screen with warm starts.
        """
        pending = [k for k in child_keys if k not in self.feas_cache]
        # Dedupe while keeping order.
        seen = set()
        pending = [k for k in pending if not (k in seen or seen.add(k))]
        if not pending:
            return
        self.n_qp += len(pending)
        if self.screener == "pdhg":
            self._solve_frontier_pdhg(pending)
        else:
            self._solve_frontier_qpax(pending)

    def _solve_frontier_qpax(self, pending):
        """Certified interior-point feasibility over power-of-two chunks."""
        import time

        i = 0
        while i < len(pending):
            chunk = pending[i : i + CHUNK]
            c = len(chunk)
            b = min(1 << (c - 1).bit_length(), CHUNK) if c > 1 else 1
            padded = chunk + [chunk[-1]] * (b - c)
            t0 = time.perf_counter()
            states = LT.batch_states(self.spec, padded)
            margins, cert = self._kernel_margins(states)
            margins = np.asarray(margins)
            cert = np.asarray(cert)
            self.t_solve += time.perf_counter() - t0
            for k, mg, cf in zip(chunk, margins[:c], cert[:c]):
                mg = float(mg)
                cf = bool(cf)
                feasible = (mg <= self.tol.tol_feas) and cf
                self.feas_cache[k] = (feasible, mg, cf)
            i += CHUNK

    def _kernel_margins(self, states):
        """Certified margins and flags for a batch: reserve or plain feasibility.

        In reserve mode this returns the two-sided lateral verdict, otherwise
        the plain P4 verdict, both from the certified qpax kernel and both
        tested by the identical margin <= tol_feas and cert rule downstream.
        """
        if self.robust:
            return LT.robust_margins_of_states(
                self.spec,
                states,
                self.tol.eps_reg,
                self.tol.tol_cone,
                self.lam_min,
                solver_tol=self.solver_tol,
                max_iter=self.search_iter,
            )
        return LT.margins_of_states(
            self.spec,
            states,
            self.tol.eps_reg,
            self.tol.tol_cone,
            solver_tol=self.solver_tol,
            max_iter=self.search_iter,
        )

    def _solve_frontier_pdhg(self, pending):
        """First-order screen over chunks, warm started from each parent.

        Every candidate warm starts from its parent's screened (f, y),
        padded with zeros when the parent has none (root or unscreened). The
        final (f, y) of each child is stored for its own descendants. The
        margin and cert are the same recomputed quantities as the certified
        path, so the only change is the kernel.
        """
        import time

        nf = self.spec.nf
        ncone = self.spec.ncone
        zero_f = np.zeros(nf)
        zero_y = np.zeros(ncone)
        i = 0
        while i < len(pending):
            chunk = pending[i : i + CHUNK]
            c = len(chunk)
            b = min(1 << (c - 1).bit_length(), CHUNK) if c > 1 else 1
            padded = chunk + [chunk[-1]] * (b - c)
            f0 = np.zeros((b, nf))
            y0 = np.zeros((b, ncone))
            for r, k in enumerate(padded):
                parent = self._parent_of.get(k)
                if parent is not None and parent in self.screened_f:
                    f0[r] = self.screened_f[parent]
                    y0[r] = self.screened_y[parent]
            t0 = time.perf_counter()
            states = LT.batch_states(self.spec, padded)
            margins, cert, fs, ys = LT.margins_of_states_pdhg(
                self.spec,
                states,
                self.tol.eps_reg,
                self.tol.tol_cone,
                iters=self.pdhg_iters,
                accel=self.pdhg_accel,
                f0=jnp.asarray(f0),
                y0=jnp.asarray(y0),
            )
            margins = np.asarray(margins)
            cert = np.asarray(cert)
            fs = np.asarray(fs)
            ys = np.asarray(ys)
            self.t_solve += time.perf_counter() - t0
            for j, (k, mg, cf) in enumerate(zip(chunk, margins[:c], cert[:c])):
                mg = float(mg)
                cf = bool(cf)
                feasible = (mg <= self.tol.tol_feas) and cf
                self.feas_cache[k] = (feasible, mg, cf)
                self.screened_f[k] = fs[j]
                self.screened_y[k] = ys[j]
            i += CHUNK

    def _certify_batch(self, keys):
        """Certified qpax verdicts for a list of state keys, batched.

        Fills self._cert_cache and counts every solved key in n_reverify.
        Chunked and padded exactly like the qpax frontier solve so the
        compiled shapes stay shared with that path.
        """
        import time

        pending = [k for k in keys if k not in self._cert_cache]
        seen = set()
        pending = [k for k in pending if not (k in seen or seen.add(k))]
        if not pending:
            return
        self.n_reverify += len(pending)
        i = 0
        while i < len(pending):
            chunk = pending[i : i + CHUNK]
            c = len(chunk)
            b = min(1 << (c - 1).bit_length(), CHUNK) if c > 1 else 1
            padded = chunk + [chunk[-1]] * (b - c)
            t0 = time.perf_counter()
            states = LT.batch_states(self.spec, padded)
            margins, cert = self._kernel_margins(states)
            margins = np.asarray(margins)
            cert = np.asarray(cert)
            self.t_reverify += time.perf_counter() - t0
            for k, mg, cf in zip(chunk, margins[:c], cert[:c]):
                self._cert_cache[k] = (float(mg) <= self.tol.tol_feas) and bool(cf)
            i += CHUNK

    def _certified_feasible(self, key):
        """Certified qpax verdict for one state, cached per key.

        The pdhg screener consults this before any best-overhang update, so
        a screener error in either direction can only cost extra certified
        solves, never a wrong best. Counted in n_reverify.
        """
        hit = self._cert_cache.get(key)
        if hit is None:
            self._certify_batch([key])
            hit = self._cert_cache[key]
        return hit

    # --- expansion --------------------------------------------------------

    def _expand_batch(self, leaves):
        """Expand a set of unexpanded, non-terminal leaf nodes together.

        One legality pass over the full (leaves, M) action grid, then one
        certified solve over the union of the legal, uncached children.
        """
        import time

        # Terminal by depth: no room for another cube.
        pending = []
        for node in leaves:
            node["expanded"] = True
            self.n_expansions += 1
            if len(node["key"]) >= self.n:
                node["terminal"] = True
                node["solved"] = True
            else:
                pending.append(node)
        if not pending:
            return

        # Legality over the full grid for every pending leaf, padded to K so
        # the legality kernel compiles once.
        keys = [node["key"] for node in pending]
        padded_keys = keys + [keys[-1]] * (self.K - len(keys)) if len(keys) < self.K else keys
        t0 = time.perf_counter()
        states = LT.batch_states(self.spec, padded_keys)
        legal = np.asarray(LT.legal_grid(self.spec, states, self.cand_L, self.cand_J))
        self.t_legal += time.perf_counter() - t0
        legal = legal[: len(pending)]

        # Gather the legal actions and children per leaf.
        per_leaf_actions = []
        all_children = []
        for b, node in enumerate(pending):
            idx = np.nonzero(legal[b])[0]
            actions = [
                (int(self.cand_L_np[i]), int(self.cand_J_np[i])) for i in idx
            ]
            actions.sort()  # stable order for deterministic tie-breaking
            per_leaf_actions.append(actions)
            for a in actions:
                ck = self._child_key(node["key"], a)
                all_children.append(ck)
                # First-seen parent, for pdhg warm starts. Transpositions can
                # give several parents; any screened parent is a valid start.
                self._parent_of.setdefault(ck, node["key"])
                # Full DAG edge, for DAG-correct subtree-value targets.
                self._children_of.setdefault(node["key"], set()).add(ck)
                self._parents_of.setdefault(ck, set()).add(node["key"])

        self._solve_frontier(all_children)

        # Under the pdhg screener, every child that could improve the best
        # overhang gets a certified verdict before the best update, in one
        # batched call. Both screen directions are re-verified: a screened
        # FEASIBLE could be false (would corrupt the best), and a screened
        # INFEASIBLE at the boundary could be a false prune of the very
        # state the search is looking for. Admissibility stays screened.
        if self.screener == "pdhg":
            improving = []
            for node, actions in zip(pending, per_leaf_actions):
                for a in actions:
                    ck = self._child_key(node["key"], a)
                    if ck in self._cert_cache:
                        continue
                    ov = LT.overhang(ck, self.dx)
                    prev = self.best_by_count.get(len(ck))
                    if ov > self.best_overhang or prev is None or ov > prev[0]:
                        improving.append(ck)
            self._certify_batch(improving)

        # Admissible children and best-overhang bookkeeping.
        for node, actions in zip(pending, per_leaf_actions):
            admissible = []
            for a in actions:
                ck = self._child_key(node["key"], a)
                feasible, _mg, _cf = self.feas_cache[ck]
                if feasible:
                    admissible.append(a)
                    node["N_a"][a] = 0
                    node["W_a"][a] = 0.0
                count = len(ck)
                ov = LT.overhang(ck, self.dx)
                prev_count = self.best_by_count.get(count)
                improves_count = prev_count is None or ov > prev_count[0]
                improves_global = ov > self.best_overhang
                if improves_count or improves_global:
                    # The best update is gated on the certified verdict when
                    # screening, on the (already certified) qpax verdict
                    # otherwise. One certification serves both the global and
                    # the per-count update.
                    if self.screener == "pdhg":
                        best_ok = self._certified_feasible(ck)
                    else:
                        best_ok = feasible
                    if best_ok:
                        if improves_count:
                            self.best_by_count[count] = (ov, ck, node["key"], a)
                        if improves_global:
                            self.best_overhang = ov
                            self.best_key = ck
                            self.best_parent = node["key"]
                            self.best_action = a
            node["actions"] = admissible
            # Root Dirichlet noise, sampled once when the root's admissible set
            # is known. Drawn from the seeded Generator so it is reproducible
            # per seed and different across seeds.
            if (self.root_noise and node is self.root and admissible
                    and self._root_noise is None):
                draw = self.rng.dirichlet(
                    np.full(len(admissible), self.dirichlet_alpha)
                )
                self._root_noise = {
                    a: float(x) for a, x in zip(admissible, draw)
                }
            # Learned priors over the admissible actions. Computed once at
            # expansion and frozen, as in AlphaZero. When prior_fn is None the
            # dict stays empty and _puct_action falls back to the uniform prior.
            if self.prior_fn is not None and admissible:
                idxs = [self._action_index(a) for a in admissible]
                pri = self.prior_fn(node["key"], idxs)
                node["P_a"] = {a: float(p) for a, p in zip(admissible, pri)}
            node["terminal"] = len(node["key"]) >= self.n or not admissible
            if node["terminal"]:
                node["solved"] = True
            elif len(node["key"]) == self.n - 1:
                # Every admissible child places the last cube, so all children
                # are terminal and were fully valued in this expansion. Nothing
                # remains to learn below this node.
                node["solved"] = True

    # --- selection, backup, solved propagation ----------------------------

    def _puct_action(self, node):
        """Argmax of Q + c_puct P sqrt(N) / (1 + N_a), uniform prior.

        Skips children whose subtree is solved. Returns None when every child
        is solved (this node is then solved too). Ties break by a seeded
        epsilon jitter (tie_eps), so distinct seeds explore different trees;
        the jitter is on the O(1) score scale and never overrides a real gap.
        When root_noise is on, the root prior is mixed with Dirichlet noise.
        """
        actions = node["actions"]
        prior = 1.0 / len(actions)
        sqrt_n = math.sqrt(node["N"]) if node["N"] > 0 else 0.0
        noise = self._root_noise if node is self.root else None
        frac = self.dirichlet_frac
        best_a = None
        best_score = float("-inf")
        for a in actions:
            child = self.tree.get(self._child_key(node["key"], a))
            if child is not None and child["solved"]:
                continue
            na = node["N_a"][a]
            q = node["W_a"][a] / na if na > 0 else 0.0
            # Uniform prior by default; the learned prior when prior_fn is set.
            p = prior if self.prior_fn is None else node["P_a"][a]
            if noise is not None:
                p = (1.0 - frac) * p + frac * noise.get(a, 0.0)
            u = self.c_puct * p * sqrt_n / (1.0 + na)
            score = q + u
            if self.tie_eps > 0.0:
                score += self.tie_eps * self.rng.random()
            if score > best_score:
                best_score = score
                best_a = a
        return best_a

    def _select_one(self):
        """Descend from the root under PUCT, applying virtual loss on the way.

        Virtual loss adds a pending visit with zero reward to each edge and to
        the leaf, so the next of the K selections in this iteration sees the
        pending visits and diverges. The real value is added at backup.
        """
        node = self.root
        path = []
        while node["expanded"] and not node["terminal"] and node["actions"]:
            a = self._puct_action(node)
            if a is None:
                break
            path.append((node, a))
            ck = self._child_key(node["key"], a)
            child = self.tree.get(ck)
            if child is None:
                child = self._node(ck, node["key"], a)
            node = child
        node["N"] += 1  # virtual-loss visit at the leaf
        for (nd, a) in path:
            nd["N"] += 1
            nd["N_a"][a] += 1
            nd["W_a"][a] += 0.0  # virtual loss: visit, no reward yet
        return node, path

    def _backup(self, path, value):
        """Add the real leaf value to every edge on the path.

        Visits were already counted by the virtual loss in _select_one, so
        only the reward is added here.
        """
        for (nd, a) in path:
            nd["W_a"][a] += value

    def _leaf_value(self, node):
        """Leaf value in [0, 1], normalized so overhang == harmonic maps to 0.5.

        A non-terminal leaf gets an optimistic 0.25 per unplaced cube, a proxy
        for the rollout it is not running. A terminal leaf is valued by its
        overhang alone.
        """
        ov = LT.overhang(node["key"], self.dx)
        if ov == float("-inf"):
            ov = 0.0
        if node["terminal"]:
            # A terminal leaf has an exact overhang value; never a heuristic,
            # so value_fn does not apply here.
            v = ov / self.value_norm
        elif self.value_fn is not None:
            # Learned value replaces the optimistic heuristic at a leaf.
            v = self.value_fn(node["key"])
        else:
            remaining = self.n - len(node["key"])
            v = (ov + 0.25 * remaining) / self.value_norm
        return min(max(v, 0.0), 1.0)

    def _propagate_solved(self, key):
        """Mark nodes solved bottom up: terminal, or all children solved."""
        k = key
        while k is not None:
            node = self.tree[k]
            if node["solved"]:
                k = node["parent"]
                continue
            if not node["expanded"]:
                break
            all_solved = bool(node["actions"])
            for a in node["actions"]:
                child = self.tree.get(self._child_key(node["key"], a))
                if child is None or not child["solved"]:
                    all_solved = False
                    break
            if not all_solved:
                break
            node["solved"] = True
            k = node["parent"]

    # --- driver -----------------------------------------------------------

    def run_iteration(self):
        """One batched PUCT iteration: K selections, one expand, K backups."""
        selections = [self._select_one() for _ in range(self.K)]

        to_expand = {}
        for (leaf, _p) in selections:
            if not leaf["expanded"] and not leaf["terminal"]:
                to_expand[leaf["key"]] = leaf
            elif leaf["expanded"]:
                self.n_revisits += 1
        if to_expand:
            self._expand_batch(list(to_expand.values()))

        for (leaf, path) in selections:
            self._backup(path, self._leaf_value(leaf))
        for (leaf, _p) in selections:
            self._propagate_solved(leaf["key"])

    def run(self, sims, progress=None):
        """Run about `sims` simulations in ceil(sims / K) batched iterations."""
        import time

        n_iter = max(1, math.ceil(sims / self.K))
        t0 = time.perf_counter()
        done = 0
        for it in range(1, n_iter + 1):
            self.run_iteration()
            done += self.K
            if progress is not None and (it % progress == 0 or it == n_iter):
                dt = time.perf_counter() - t0
                rate = done / dt if dt > 0 else 0.0
                print(
                    f"  sims={done:6d}  nodes={len(self.tree):7d}  "
                    f"best_overhang={self.best_overhang:.4f}  {rate:7.1f} sims/s",
                    flush=True,
                )
        self.wall = time.perf_counter() - t0
        self.sims_done = done
        return self.best_overhang

    # --- reporting --------------------------------------------------------

    def margin_of(self, key):
        """Cached certified P4 margin for a state, or None if not solved.

        The frontier solve stores (feasible, margin, certified) per child key
        in feas_cache. A data collector reads margins from here to supervise a
        margin head. Read-only; the search never depends on this method, and
        the empty root has no cached margin (returns None).
        """
        hit = self.feas_cache.get(key)
        return None if hit is None else hit[1]

    def history_of(self, key):
        """A prefix-feasible action order that builds the state at key."""
        seq = []
        k = key
        while k is not None:
            node = self.tree.get(k)
            if node is None or node["action_in"] is None:
                break
            seq.append(node["action_in"])
            k = node["parent"]
        return list(reversed(seq))

    def best_sequence(self):
        """Action sequence (list of (L, j)) that builds the best state.

        This is the best-at-most-n state (best_overhang), which may use fewer
        than n cubes. For the exact-n design use exact_sequence(self.n).
        """
        if self.best_key is None:
            return []
        if self.best_parent is not None:
            return self.history_of(self.best_parent) + [self.best_action]
        if self.best_key in self.tree:
            return self.history_of(self.best_key)
        return sorted(self.best_key, key=lambda lj: (lj[0], lj[1]))

    def exact_overhang(self, count):
        """Best certified-feasible overhang over states using exactly `count`
        cubes, or -inf if none was found. best_overhang is the max over all
        counts (best-at-most-n); this isolates one block count."""
        rec = self.best_by_count.get(count)
        return float("-inf") if rec is None else rec[0]

    def exact_sequence(self, count):
        """Build order (list of (L, j)) for the best exact-`count` state."""
        rec = self.best_by_count.get(count)
        if rec is None:
            return []
        _ov, key, parent, action = rec
        if parent is not None:
            return self.history_of(parent) + [action]
        return sorted(key, key=lambda lj: (lj[0], lj[1]))

    def sample_action(self, node, temperature=None):
        """Pick a child action from visit counts with a softmax temperature.

        temperature == 0 (default) is argmax over visit counts, the
        deterministic eval move. temperature > 0 samples proportional to
        N_a ** (1 / temperature) using the constructor Generator, the
        AlphaZero self-play move rule. Returns None when the node has no
        admissible action. This is the episodic move primitive; the
        distillation pipeline supervises on the full visit distribution
        (az.selfplay_samples) rather than sampled single moves.
        """
        if temperature is None:
            temperature = self.temperature
        actions = node["actions"]
        if not actions:
            return None
        visits = np.array([node["N_a"].get(a, 0) for a in actions],
                          dtype=np.float64)
        if temperature <= 0.0 or visits.sum() <= 0.0:
            return actions[int(np.argmax(visits))]
        logits = np.log(np.maximum(visits, 1e-12)) / temperature
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        return actions[int(self.rng.choice(len(actions), p=p))]

    def tree_report(self):
        """Per-depth node and terminal counts with the best overhang seen."""
        by_depth = {}
        for node in self.tree.values():
            d = len(node["key"])
            rec = by_depth.setdefault(d, [0, 0, float("-inf")])
            rec[0] += 1
            if node["expanded"] and node["terminal"]:
                rec[1] += 1
            ov = LT.overhang(node["key"], self.dx)
            if ov > rec[2]:
                rec[2] = ov
        lines = [
            f"tree: {len(self.tree)} nodes, {self.n_expansions} expansions, "
            f"{self.n_revisits} revisits"
        ]
        for d in sorted(by_depth):
            nodes, term, best = by_depth[d]
            best_s = f"{best:.4f}" if best != float("-inf") else "n/a"
            lines.append(
                f"  depth {d}: nodes={nodes:6d} terminal={term:6d} "
                f"best_overhang={best_s}"
            )
        return "\n".join(lines)
