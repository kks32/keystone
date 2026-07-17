"""Branch-and-bound optimality certifier for prefix-feasible max overhang.

This module proves the true grid optimum of the maximum-overhang cube
stacking problem for small n, something a heuristic search cannot do. It
searches the same lattice environment as keystone.search.mcts (imported
read only), but instead of a bandit it runs best-first branch and bound
with an admissible bound, so a finished run reports either a certified
optimum or an honest interval.

Scene. Identical physics to examples/search_overhang_fast.py and
keystone.search.lattice: 2D xz plane, gravity along -z, mu = 0.7, a
pedestal box_2d(6, 1, -3, 0.5) with right edge at x = 0, unit cubes
box_2d(1, 1, j*dx, 1.5 + L) stacked above it on a dx grid.

Admissible upper bound.

Overhang is the rightmost cube edge measured from the pedestal right edge
at x = 0. A unit cube at grid index j has center j*dx and right edge
j*dx + 0.5.

A new cube rests on a support: the pedestal top for layer 0, a layer-(L-1)
cube for layer L >= 1. Legality requires the footprint overlap with that
support to reach 2*dx (the grid form of the >= 2 dx rule; overlaps are
multiples of dx, so the > 1.5*dx test in lattice.is_legal means >= 2*dx).
Let the support have right edge S. The new cube footprint is
[c - 0.5, c + 0.5]. Pushing the cube right, the overlap is S - (c - 0.5).
The overlap floor 2*dx gives c <= S + 0.5 - 2*dx, so the new cube right
edge c + 0.5 <= S + 1 - 2*dx.

Let E be the largest right edge over the pedestal (edge 0) and every placed
cube. Any support has right edge at most E, so any newly placed cube has
right edge at most E + (1 - 2*dx). Placing a cube can only raise E, and by
at most (1 - 2*dx). With r cubes remaining, r placements raise E by at most
r*(1 - 2*dx). Therefore

    bound(state) = E + r*(1 - 2*dx),  r = n - placed,  E = max(0, overhang)

is an upper bound on the overhang of every completion. It is admissible: no
legal completion, feasible or not, can exceed it. The domain cap x_hi + 0.5
(no cube center exceeds x_hi) tightens it:

    bound(state) = min(E + r*(1 - 2*dx), x_hi + 0.5).

The bound is monotone: adding a cube lowers r by one (subtracts 1 - 2*dx)
and raises E by at most 1 - 2*dx, so a child bound never exceeds its parent
bound. A monotone admissible bound makes best-first branch and bound exact.
The first time the frontier maximum bound drops to the incumbent, the
incumbent is the true optimum.

The bound is geometry only. It counts grid reach and never reads density
or friction, so it stays admissible under heterogeneous materials: no
feasible completion can exceed it whatever the cube masses or frictions.
Materials only change which completions are feasible, which the QP verdict
below decides, never the reach ceiling the bound enforces.

All bounds and overhangs lie on the grid {k*dx + 0.5}, so any two differ by
an integer multiple of dx. Pruning compares with a half-step margin 0.5*dx,
which separates "no improvement possible" (difference <= 0) from "at least
one more grid step" (difference >= dx) exactly. No feasibility tolerance
enters the prune test, so no tolerance constant lives here.

Closed list and transpositions.

A state is the set of placed (layer, index) cells. The same set can be
reached by different placement orders. Static feasibility of a set is a
property of the assembly, not of the order that built it, so the feasible
completions of a set are the same whichever feasible order first reached
it. Once a set has been reached prefix-feasibly and expanded, re-expanding
it under another order finds the same completions. The closed list drops
repeat expansions of a set without losing any reachable completion.
Reaching a set prefix-feasibly by one order does not make every order
feasible, and the closed list never claims that: it only asserts that one
feasible arrival is enough to enumerate the set's future.

Heterogeneous materials keep this property. Each cube's density and
friction is assigned by the sorted-cell position of the set, not by the
order the set was built in, so the assembly of a set is still a function of
the set alone. The i-th cube in sorted (layer, index) order takes
densities[i] and mu_by_slot[i]. Inventory order is therefore fixed to that
sorted-cell convention; a caller who wants a different assignment sorts the
inventory before the run. The dedup and the transposition argument are
unchanged.

The argument survives the placement-reachability modes. A drop or slide
check reads the set of already-placed cubes and the candidate cell, never
the order that built the set, so the legal children of a set are still a
function of the set alone. One legal, prefix-feasible arrival at a set is
therefore still enough to enumerate its future under any mode.

Placement modes.

placement selects the reachability rule of lattice.is_legal: "static"
(no motion check, today's behavior), "drop" (clear vertical column above
the target), "slide" (drop column or a lateral corridor at the target
layer), "slide_clear" (slide with under-bridge corridors forbidden).
Reachability only removes actions, so every completion under a mode is
also a static completion and the admissible bound below is unchanged: it
never undercounts what a restricted mode can reach.

Certification and the three-state screen.

A fast solve can fail to converge, so its verdict alone cannot justify a
discard. Every candidate child is screened three ways by the qpax kernel at
the library default iteration cap (SolverOptions max_iter = 100):

- verified-feasible: margin <= tol.tol_eq and the cone-admissible/finite
  flag both hold. This is a genuine feasible witness, the same rule the rest
  of the library uses.
- candidate-infeasible: the margin is finite and exceeds tol.tol_feas.
- unknown: the margin is non-finite, or finite but not cone-admissible.

A child is permanently discarded only after a verified infeasibility. A
child whose admissible bound is already at or below the incumbent is dropped
on the bound alone, which is sound. Every candidate-infeasible or unknown
child whose bound still beats the incumbent escalates to the exact LP path
(solve_p0_exact with its validated Farkas construction, cached per set). The
exact verdict decides: infeasible discards the child, feasible keeps it, and
an abstain (no convergence on either path) leaves the child live. A live
child is expanded like any other; if it is a full stack that cannot be
expanded, it is recorded as unresolved.

The reported optimum's build order is re-verified prefix by prefix through
the host pipeline (build_assembly + assemble + solve_p0), so the incumbent
is always a host-verified structure.

Claim. The run reports "proved optimal" only when the frontier is exhausted
with every discard justified by bound-domination or a verified
infeasibility, and no unresolved state has an overhang above the incumbent.
Otherwise it reports the interval [incumbent, best remaining bound] with a
count of unresolved states, and never claims optimality. A node or time
budget that stops the run early yields the same interval.

Determinism. One fixed candidate ordering, sorted (layer, index) actions,
and a monotone push counter break every tie, so a run is reproducible.
"""

import heapq
import time
from dataclasses import dataclass, field, replace
from fractions import Fraction

import numpy as np

from ..geometry import Tolerances, box_2d, build_assembly
from ..mechanics import assemble
from ..solve import (
    FEASIBLE,
    INFEASIBLE,
    NO_CONVERGE,
    SolverOptions,
    solve_p0,
    solve_p0_exact,
)
from . import lattice as LT

# Three-state screening classes for a candidate child. VERIFIED and
# CAND_INFEASIBLE and UNKNOWN partition every screened set; only VERIFIED is a
# feasible witness, only a verified infeasibility (exact path) may discard.
VERIFIED = "feasible"
CAND_INFEASIBLE = "candidate_infeasible"
UNKNOWN = "unknown"

# Frontier padding for the batched certified solve, matching mcts. Uncached
# children are solved in power-of-two chunks capped at CHUNK so the compiled
# shapes stay few. Padding slots repeat one real state and are dropped.
CHUNK = 256


def parse_dx(value) -> float:
    """Parse a grid step given as a float or a fraction string like '1/12'."""
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if "/" in s:
        return float(Fraction(s))
    return float(s)


def canonical(placements):
    """Canonical set key: the sorted tuple of (layer, index) cells."""
    return tuple(sorted(placements))


def bound_of(key, n: int, dx: float, x_hi: float) -> float:
    """Admissible upper bound on the overhang of any completion of key.

    See the module docstring for the derivation. E is the largest right edge
    over the pedestal (0) and the placed cubes; r is the number of cubes
    still available; each remaining cube extends the reach by at most
    1 - 2*dx. The domain cap x_hi + 0.5 tightens the raw bound.
    """
    ov = LT.overhang(key, dx)
    e_supp = max(0.0, ov)
    r = n - len(key)
    raw = e_supp + r * (1.0 - 2.0 * dx)
    return min(raw, x_hi + 0.5)


def _classify_screen(margin, cert, tol):
    """Three-state class of a screened set from its margin and cert flag.

    cert is the kernel's cone-admissible-and-finite flag. VERIFIED needs a
    finite margin at or below tol_eq with cert set, a feasible witness.
    CAND_INFEASIBLE needs a finite margin above tol_feas. Everything else (a
    non-finite margin, or a finite margin at or below tol_feas that is not
    cone-admissible) is UNKNOWN: unverifiable in either direction.
    """
    finite = bool(np.isfinite(margin))
    if finite and margin <= tol.tol_eq and cert:
        return VERIFIED
    if finite and margin > tol.tol_feas:
        return CAND_INFEASIBLE
    return UNKNOWN


@dataclass
class CertifyResult:
    """Outcome of a certification run.

    certified True means best-first branch and bound closed the gap and the
    optimum is proven: the frontier was exhausted with every discard justified
    by bound-domination or a verified infeasibility, and no unresolved state
    outranks the incumbent. False means a budget stopped the run, or a state
    could not be resolved, and only the interval [lower, upper] holds. optimum
    is the incumbent overhang (the best host-verified structure found); it
    equals lower. placement records the reachability mode the run enforced
    ("static", "drop", or "slide"); the optimum is proven within that mode's
    legal orders. exact_escalations counts sets sent to the exact LP path.
    unresolved counts states the run could not decide whose overhang still
    outranks the incumbent (each one blocks the optimality claim).
    """

    n: int
    dx: float
    placement: str
    certified: bool
    optimum: float
    lower: float
    upper: float
    sequence: list
    prefix_trace: list
    host_verified: bool
    stop_reason: str
    nodes_expanded: int
    nodes_generated: int
    frontier_size: int
    max_frontier: int
    qp_solves: int
    host_verifications: int
    exact_escalations: int
    unresolved: int
    closed_size: int
    wall_time: float
    info: dict = field(default_factory=dict)


class Certifier:
    """Best-first branch and bound over prefix-feasible lattice states."""

    def __init__(
        self,
        n: int,
        dx: float,
        tol: Tolerances,
        *,
        opts: SolverOptions = SolverOptions(),
        mu: float = LT.MU,
        placement: str = "static",
        densities=None,
        mu_ground=None,
        mu_by_slot=None,
        robust: bool = False,
        lam_min: float = LT.LAM_MIN,
    ):
        self.n = int(n)
        self.dx = float(dx)
        self.tol = tol
        self.opts = opts
        self.mu = float(mu)
        self.placement = str(placement)
        # Reserve mode. When robust is set, the feasibility oracle certifies
        # lam-robustness (P4 feasible under +-lam_min lateral load) instead of
        # plain static feasibility. robust-feasible is a subset of feasible, so
        # the admissible bound below is unchanged and stays valid. Off by
        # default, so the static path traces exactly as before.
        self.robust = bool(robust)
        self.lam_min = float(lam_min)
        # Heterogeneous materials, one value per cube. densities and
        # mu_by_slot are indexed by sorted-cell position of the set, because
        # the search keys states on their canonical (sorted) placement set;
        # see the module docstring on the admissible bound. None on all three
        # is the homogeneous scene, evaluated exactly as before.
        self.densities = None if densities is None else tuple(float(d) for d in densities)
        self.mu_ground = None if mu_ground is None else float(mu_ground)
        self.mu_by_slot = (
            None if mu_by_slot is None else tuple(float(m) for m in mu_by_slot)
        )
        self._homogeneous = (
            self.densities is None
            and self.mu_by_slot is None
            and self.mu_ground is None
        )

        self.spec = LT.LatticeSpec(
            n_max=self.n,
            dx=self.dx,
            mode=self.placement,
            mu=self.mu,
            densities=self.densities,
            mu_ground=self.mu_ground,
            mu_by_slot=self.mu_by_slot,
        )
        self.x_hi = self.spec.x_hi
        cand_L, cand_J = LT.action_grid(self.spec)
        self.cand_L = cand_L
        self.cand_J = cand_J
        self.cand_L_np = np.asarray(cand_L)
        self.cand_J_np = np.asarray(cand_J)
        self.M = int(cand_L.shape[0])

        # Half a grid step. Bounds and overhangs are grid values, so their
        # difference is a multiple of dx; this margin separates a real
        # one-step improvement from none exactly. It is a grid quantity, not
        # a feasibility tolerance.
        self.prune_eps = 0.5 * self.dx

        # Screen class of a set, cached by canonical key: one of VERIFIED,
        # CAND_INFEASIBLE, UNKNOWN.
        self.feas_cache = {}
        # Exact-LP verdict of a set, cached by canonical key: one of FEASIBLE,
        # INFEASIBLE, NO_CONVERGE. Only sets that screen non-feasible and beat
        # the incumbent reach it.
        self.exact_cache = {}
        # Sets that neither the screen nor the exact path could decide, mapped
        # to their own overhang. A state here with overhang above the final
        # incumbent blocks the optimality claim.
        self.unresolved = {}

        self.incumbent = float("-inf")
        self.incumbent_seq = []
        self.incumbent_trace = []

        # Instrumentation.
        self.nodes_expanded = 0
        self.nodes_generated = 0
        self.max_frontier = 0
        self.qp_solves = 0
        self.host_verifications = 0
        self.exact_escalations = 0

    # --- feasibility oracle ----------------------------------------------

    def _solve_feasibility(self, keys):
        """Three-state qpax screen for a list of set keys, batched.

        Fills feas_cache with one of VERIFIED, CAND_INFEASIBLE, UNKNOWN per
        key. Only uncached keys hit the solver. Uses the full library
        iteration cap so the screen is the reference, not a reduced search
        screen. A non-finite margin, or a finite margin that is not
        cone-admissible, is UNKNOWN (unverifiable either way); a finite
        cone-admissible margin at or below tol_eq is VERIFIED; a finite margin
        above tol_feas is CAND_INFEASIBLE.
        """
        pending = [k for k in keys if k not in self.feas_cache]
        seen = set()
        pending = [k for k in pending if not (k in seen or seen.add(k))]
        if not pending:
            return
        self.qp_solves += len(pending)
        i = 0
        while i < len(pending):
            chunk = pending[i : i + CHUNK]
            c = len(chunk)
            b = min(1 << (c - 1).bit_length(), CHUNK) if c > 1 else 1
            padded = chunk + [chunk[-1]] * (b - c)
            states = LT.batch_states(self.spec, padded)
            if self.robust:
                margins, cert = LT.robust_margins_of_states(
                    self.spec,
                    states,
                    self.tol.eps_reg,
                    self.tol.tol_cone,
                    self.lam_min,
                    solver_tol=self.opts.solver_tol,
                    max_iter=self.opts.max_iter,
                )
            else:
                margins, cert = LT.margins_of_states(
                    self.spec,
                    states,
                    self.tol.eps_reg,
                    self.tol.tol_cone,
                    solver_tol=self.opts.solver_tol,
                    max_iter=self.opts.max_iter,
                )
            margins = np.asarray(margins)
            cert = np.asarray(cert)
            for k, mg, cf in zip(chunk, margins[:c], cert[:c]):
                self.feas_cache[k] = _classify_screen(
                    float(mg), bool(cf), self.tol
                )
            i += CHUNK

    # --- exact-LP escalation ---------------------------------------------

    def _exact_verdict(self, key):
        """Exact-LP verdict for a set, cached. FEASIBLE/INFEASIBLE/NO_CONVERGE.

        The escalation path for the three-state screen: a candidate-infeasible
        or unknown set is decided here on the validated exact LP oracle before
        it can be discarded, so an unconverged fast solve never deletes a live
        branch. In robust mode the verdict is lam-robust: robust-feasible
        means feasible at both +lam_min and -lam_min, so the verdict is
        FEASIBLE only when both loaded states are exact-feasible, INFEASIBLE
        when either is exact-infeasible, and NO_CONVERGE otherwise.
        """
        cached = self.exact_cache.get(key)
        if cached is not None:
            return cached
        self.exact_escalations += 1
        placed = list(key)  # key is already sorted
        if self._homogeneous:
            boxes = [box_2d(6.0, 1.0, -3.0, 0.5)]
            for (L, j) in placed:
                boxes.append(box_2d(1.0, 1.0, j * self.dx, 1.5 + L))
            asm = build_assembly(boxes, mu=self.mu, tol=self.tol, dim=2)
        else:
            asm = self._host_assembly(placed)
        system = assemble(asm, self.tol, cone="linear2d")
        if not self.robust:
            status = solve_p0_exact(system, self.tol).status
        else:
            w_live = system.w_live
            sp = solve_p0_exact(
                replace(system, w_dead=system.w_dead + self.lam_min * w_live),
                self.tol,
            ).status
            sm = solve_p0_exact(
                replace(system, w_dead=system.w_dead - self.lam_min * w_live),
                self.tol,
            ).status
            if sp == INFEASIBLE or sm == INFEASIBLE:
                status = INFEASIBLE
            elif sp == FEASIBLE and sm == FEASIBLE:
                status = FEASIBLE
            else:
                status = NO_CONVERGE
        self.exact_cache[key] = status
        return status

    # --- host re-verification --------------------------------------------

    def host_prefix_feasible(self, seq):
        """Replay a build order on the certified host pipeline.

        Builds the pedestal plus one cube per step, solves P0 at every
        prefix, and stops at the first non-feasible prefix. Returns
        (all_feasible, trace) with trace a list of
        (layer, index, x, margin, status).

        Heterogeneous runs assign each prefix its densities and frictions by
        sorted-cell position, exactly how the fast oracle builds the set from
        its canonical key, so the host verdict matches the oracle. The
        homogeneous path is byte for byte the old code.
        """
        if self._homogeneous:
            boxes = [box_2d(6.0, 1.0, -3.0, 0.5)]
            trace = []
            for (L, j) in seq:
                x = j * self.dx
                boxes.append(box_2d(1.0, 1.0, x, 1.5 + L))
                asm = build_assembly(boxes, mu=self.mu, tol=self.tol, dim=2)
                system = assemble(asm, self.tol, cone="linear2d")
                r = solve_p0(system, self.tol)
                trace.append((int(L), int(j), float(x), float(r.margin), r.status))
                if r.status != FEASIBLE:
                    return False, trace
            return True, trace

        trace = []
        placed = []
        for (L, j) in seq:
            placed.append((int(L), int(j)))
            asm = self._host_assembly(placed)
            system = assemble(asm, self.tol, cone="linear2d")
            r = solve_p0(system, self.tol)
            x = j * self.dx
            trace.append((int(L), int(j), float(x), float(r.margin), r.status))
            if r.status != FEASIBLE:
                return False, trace
        return True, trace

    def _host_assembly(self, placed):
        """Host Assembly for a set of placed cells under heterogeneous materials.

        placed is a list of (layer, index) cells. Sort by (layer, index) and
        give the i-th sorted cube densities[i] and mu_by_slot[i], matching
        build_system's per-slot read on a canonical key. The pedestal keeps
        spec.density; friction combines by min via lattice.host_mu_fn.
        """
        order = sorted(range(len(placed)), key=lambda i: placed[i])
        boxes = [box_2d(6.0, 1.0, -3.0, 0.5, density=self.spec.density)]
        cube_materials = []
        for si, oi in enumerate(order):
            (L, j) = placed[oi]
            dens = LT.DENSITY if self.densities is None else self.densities[si]
            boxes.append(box_2d(1.0, 1.0, j * self.dx, 1.5 + L, density=dens))
            mat = self.mu if self.mu_by_slot is None else self.mu_by_slot[si]
            cube_materials.append(mat)
        mu_fn = LT.host_mu_fn(self.mu, self.mu_ground, cube_materials)
        return build_assembly(boxes, mu=self.mu, tol=self.tol, dim=2, mu_fn=mu_fn)

    # --- expansion --------------------------------------------------------

    def _expand_batch(self, nodes):
        """Expand a group of sets together and return feasible push records.

        nodes is a list of (key, seq). One legality pass over the whole group,
        one batched feasibility solve over the union of their candidate
        children, then per-node incumbent updates. Batching amortizes the
        vmapped kernel over many candidates at once, the throughput path the
        lattice kernel is built for.

        A child whose bound cannot beat the incumbent is dropped before the
        feasibility solve: its overhang is at most its bound, so it can
        neither raise the incumbent nor seed a better completion. This is the
        same admissible-bound prune applied one step earlier, and it removes
        the feasibility solve entirely for those children.

        Screened children are then handled three ways. A verified-feasible
        child is kept. A candidate-infeasible or unknown child that still beats
        the incumbent escalates to the exact LP path: infeasible discards it,
        feasible keeps it, and an abstain leaves it live and records it as
        unresolved. Only a verified infeasibility (or bound-domination) ever
        discards a child, so an unconverged fast solve cannot delete a branch.

        Returns a list of (child_key, child_seq, overhang, depth, bound) for
        every surviving child (feasible or live), across all nodes.
        """
        keys = [key for (key, _seq) in nodes]
        states = LT.batch_states(self.spec, keys)
        legal = np.asarray(LT.legal_grid(self.spec, states, self.cand_L, self.cand_J))

        # Candidate children per node with the pre-solve bound filter.
        per_node = []
        all_ck = []
        for bi, (key, seq) in enumerate(nodes):
            idx = np.nonzero(legal[bi])[0]
            actions = sorted(
                (int(self.cand_L_np[i]), int(self.cand_J_np[i])) for i in idx
            )
            cand = []
            for a in actions:
                ck = canonical(key + (a,))
                cb = bound_of(ck, self.n, self.dx, self.x_hi)
                if cb <= self.incumbent + self.prune_eps:
                    continue
                cand.append((a, ck, cb))
                all_ck.append(ck)
            per_node.append((seq, cand))
        self._solve_feasibility(all_ck)

        out = []
        for (seq, cand) in per_node:
            # Split the screen: genuine feasible witnesses versus the rest.
            feasible = []
            pending = []
            for (a, ck, cb) in cand:
                if self.feas_cache[ck] == VERIFIED:
                    child_seq = seq + [a]
                    ov = LT.overhang(ck, self.dx)
                    feasible.append((ck, child_seq, ov, len(ck), cb, a))
                else:
                    pending.append((a, ck, cb))

            # Raise the incumbent on the fast-feasible pool first, so the
            # escalation below sees the tightest incumbent and skips more
            # bound-dominated children without an exact solve.
            self._incumbent_update(feasible)

            # Escalate the non-feasible children that still beat the incumbent.
            # Drop the bound-dominated ones on the bound alone (sound). The
            # exact path is the only thing that may permanently discard a
            # child, via a verified infeasibility.
            exact_feasible = []
            live = []
            for (a, ck, cb) in pending:
                if cb <= self.incumbent + self.prune_eps:
                    continue
                ev = self._exact_verdict(ck)
                if ev == INFEASIBLE:
                    continue
                child_seq = seq + [a]
                ov = LT.overhang(ck, self.dx)
                rec = (ck, child_seq, ov, len(ck), cb, a)
                if ev == FEASIBLE:
                    exact_feasible.append(rec)
                else:
                    # Neither path decided. Keep the child live and record it;
                    # a live full stack whose overhang beats the incumbent
                    # blocks the optimality claim.
                    self.unresolved[ck] = ov
                    live.append(rec)

            # Exact-feasible children can also raise the incumbent.
            self._incumbent_update(exact_feasible)

            out.extend(
                (ck, child_seq, ov, depth, cb)
                for (ck, child_seq, ov, depth, cb, a)
                in feasible + exact_feasible + live
            )
        return out

    def _incumbent_update(self, feasible):
        """Raise the incumbent on the first host-verified feasible child.

        Verify improving children in descending overhang order and stop at the
        first the host pipeline confirms; the rest have lower overhang and
        cannot beat it. Every child here is a feasible set (a fast screen
        witness or an exact-LP witness), so it is a valid answer. When the host
        pipeline cannot confirm its build order, we can neither promote it to
        the incumbent nor rule it out, so it is recorded as unresolved and
        blocks the optimality claim. The incumbent stays the best structure the
        host pipeline itself verifies.
        """
        for (ck, child_seq, ov, depth, cb, a) in sorted(
            feasible, key=lambda t: (-t[2], t[5])
        ):
            if ov <= self.incumbent + self.prune_eps:
                break
            ok, trace = self.host_prefix_feasible(child_seq)
            self.host_verifications += 1
            if ok:
                self.incumbent = ov
                self.incumbent_seq = list(child_seq)
                self.incumbent_trace = trace
                break
            # Feasible set, but the host pipeline could not confirm the build
            # order. Record it: if its overhang outranks the final incumbent it
            # degrades the claim to an interval.
            self.unresolved[ck] = ov

    # --- driver -----------------------------------------------------------

    def run(self, *, max_nodes=None, time_limit=None, progress=True,
            progress_every=5.0, batch=64):
        """Run best-first branch and bound and return a CertifyResult.

        Each round pops up to `batch` frontier nodes of highest bound and
        expands them together, so their children hit the vmapped feasibility
        kernel in one batched solve. max_nodes caps expansions, time_limit
        caps wall seconds. When either stops the run before the frontier
        proves optimality, the result is an interval, not a claimed optimum.
        """
        t0 = time.perf_counter()
        counter = 0
        closed = set()

        # Heap ordered by bound descending, then overhang descending, then a
        # monotone counter. The counter is unique, so heap entries never
        # compare beyond it, and the order is fully deterministic.
        root_bound = bound_of((), self.n, self.dx, self.x_hi)
        heap = [(-root_bound, 0.0, counter, (), tuple())]
        counter += 1
        self.max_frontier = 1

        stop_reason = "empty"
        last_print = t0

        while heap:
            # Pop up to `batch` highest-bound live nodes for this round. A
            # popped node with bound <= incumbent + eps is prunable: it holds
            # the maximum bound left in the heap, so every remaining node is
            # prunable too. Discard it and stop popping. The bound only ever
            # falls and the incumbent only ever rises, so it can never revive.
            group = []
            while heap and len(group) < batch:
                neg_b, _neg_ov, _c, key, seq = heapq.heappop(heap)
                b = -neg_b
                if b <= self.incumbent + self.prune_eps:
                    break
                if key in closed:
                    continue
                closed.add(key)
                group.append((key, list(seq)))

            if not group:
                # The frontier maximum is prunable, or the heap is exhausted.
                # Every non-expanded node is dominated, so the incumbent is
                # the proven optimum.
                stop_reason = "optimal"
                break

            self.nodes_expanded += len(group)
            children = self._expand_batch(group)
            for (ck, child_seq, ov, depth, cb) in children:
                if depth >= self.n:
                    # A full stack. Its overhang was already weighed for the
                    # incumbent; there is nothing left to expand.
                    continue
                if cb <= self.incumbent + self.prune_eps:
                    continue
                if ck in closed:
                    continue
                heapq.heappush(heap, (-cb, -ov, counter, ck, tuple(child_seq)))
                counter += 1
                self.nodes_generated += 1
            self.max_frontier = max(self.max_frontier, len(heap))

            now = time.perf_counter()
            if progress and (now - last_print) >= progress_every:
                top = -heap[0][0] if heap else self.incumbent
                gap = top - self.incumbent
                inc = self.incumbent if self.incumbent > float("-inf") else float("nan")
                print(
                    f"  expanded={self.nodes_expanded:8d} frontier={len(heap):8d} "
                    f"incumbent={inc:.6f} top_bound={top:.6f} gap={gap:.6f} "
                    f"qp={self.qp_solves} t={now - t0:6.1f}s",
                    flush=True,
                )
                last_print = now

            if max_nodes is not None and self.nodes_expanded >= max_nodes:
                stop_reason = "max_nodes"
                break
            if time_limit is not None and (now - t0) >= time_limit:
                stop_reason = "time_limit"
                break

        wall = time.perf_counter() - t0

        # Best remaining bound over the live frontier (entries not yet closed).
        remaining = [(-e[0]) for e in heap if e[3] not in closed]
        if stop_reason in ("optimal", "empty") or not remaining:
            upper = self.incumbent
            certified = True
        else:
            upper = max(remaining)
            # If the frontier maximum cannot beat the incumbent, the optimum
            # is still proven even though a budget fired.
            if upper <= self.incumbent + self.prune_eps:
                upper = self.incumbent
                certified = True
                stop_reason = "optimal"
            else:
                certified = False

        # Unresolved states block the optimality claim. A state the run could
        # not decide, whose own overhang still beats the incumbent, might be a
        # feasible structure better than the incumbent, and nothing ruled it
        # out. Its bound is a valid contribution to the interval upper.
        blocking = {
            k: ov for k, ov in self.unresolved.items()
            if ov > self.incumbent + self.prune_eps
        }
        if blocking:
            certified = False
            upper = max(
                [upper]
                + [bound_of(k, self.n, self.dx, self.x_hi) for k in blocking]
            )
            if stop_reason == "optimal":
                stop_reason = "unresolved"
        n_unresolved = len(blocking)

        # Host re-verify the reported optimum's build order one more time so
        # the emitted record stands on the certified pipeline alone. Under a
        # reachability mode, also re-check the build order step by step; the
        # search only generates reachable orders, so this is a cross-check.
        host_verified = False
        reach_verified = True
        trace = self.incumbent_trace
        if self.incumbent_seq:
            host_verified, trace = self.host_prefix_feasible(self.incumbent_seq)
            reach_verified, _ = sequence_reachable(
                self.n, self.dx, self.incumbent_seq, self.placement
            )
            host_verified = host_verified and reach_verified

        return CertifyResult(
            n=self.n,
            dx=self.dx,
            placement=self.placement,
            certified=bool(certified),
            optimum=self.incumbent,
            lower=self.incumbent,
            upper=upper,
            sequence=list(self.incumbent_seq),
            prefix_trace=list(trace),
            host_verified=bool(host_verified),
            stop_reason=stop_reason,
            nodes_expanded=self.nodes_expanded,
            nodes_generated=self.nodes_generated,
            frontier_size=len(heap),
            max_frontier=self.max_frontier,
            qp_solves=self.qp_solves,
            host_verifications=self.host_verifications,
            exact_escalations=self.exact_escalations,
            unresolved=int(n_unresolved),
            closed_size=len(closed),
            wall_time=wall,
            info={
                "reach_verified": bool(reach_verified),
                "robust": self.robust,
                "lam_min": self.lam_min if self.robust else None,
                "unresolved_states": sorted(blocking),
            },
        )


def sequence_reachable(n: int, dx: float, sequence, placement: str):
    """Re-check a build order step by step under a placement mode.

    Replays the sequence through lattice.is_legal on a spec with the given
    placement mode, evaluating each step against the state before it, which
    is exactly how the search gated it. Returns (all_legal, flags) with one
    bool per step. Used to verify archived optima in their recorded order.
    """
    spec = LT.LatticeSpec(n_max=int(n), dx=float(dx), mode=placement)
    state = LT.empty_state(spec)
    flags = []
    for (L, j) in sequence:
        flags.append(bool(LT.is_legal(spec, state, int(L), int(j))))
        state = LT.place(spec, state, int(L), int(j))
    return all(flags), flags


def certify(
    n: int,
    dx: float,
    tol: Tolerances,
    *,
    opts: SolverOptions = SolverOptions(),
    max_nodes=None,
    time_limit=None,
    progress=True,
    placement: str = "static",
    mu: float = LT.MU,
    densities=None,
    mu_ground=None,
    mu_by_slot=None,
    robust: bool = False,
    lam_min: float = LT.LAM_MIN,
) -> CertifyResult:
    """Certify the grid optimum (or an interval) for n cubes at step dx.

    mu is the base friction (pedestal material and the default cube
    friction). densities and mu_by_slot are optional per-cube arrays of
    length n, indexed by sorted-cell position; mu_ground is the optional
    pedestal-ground friction. All default to the homogeneous scene.

    robust switches the feasibility oracle to lam-robustness at lam_min: a
    state counts feasible only when it certifies under +-lam_min lateral load.
    The certified optimum is then the largest overhang with that lateral
    reserve. The admissible bound is unchanged (robust-feasible is a subset of
    feasible). Off by default.
    """
    engine = Certifier(
        n,
        dx,
        tol,
        opts=opts,
        placement=placement,
        mu=mu,
        densities=densities,
        mu_ground=mu_ground,
        mu_by_slot=mu_by_slot,
        robust=robust,
        lam_min=lam_min,
    )
    return engine.run(max_nodes=max_nodes, time_limit=time_limit, progress=progress)
