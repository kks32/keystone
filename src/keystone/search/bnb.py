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

The argument survives the placement-reachability modes. A drop or slide
check reads the set of already-placed cubes and the candidate cell, never
the order that built the set, so the legal children of a set are still a
function of the set alone. One legal, prefix-feasible arrival at a set is
therefore still enough to enumerate its future under any mode.

Placement modes.

placement selects the reachability rule of lattice.is_legal: "static"
(no motion check, today's behavior), "drop" (clear vertical column above
the target), "slide" (drop column or a lateral corridor at the target
layer). Reachability only removes actions, so every completion under drop
or slide is also a static completion and the admissible bound below is
unchanged: it never undercounts what a restricted mode can reach.

Certification.

Expansion feasibility uses the certified qpax path (the lattice
margins_of_states kernel) at the library default iteration cap
(SolverOptions max_iter = 100), not the reduced search screen. Feasibility
is margin <= tol.tol_feas and the cone-admissible/finite certified flag,
the same rule the rest of the library uses. This is the reference verdict
for infeasibility pruning. The reported optimum's build order is then
re-verified prefix by prefix through the host pipeline (build_assembly +
assemble + solve_p0), so the incumbent that pruning trusts is always a
host-certified structure. When a node budget or a time limit is reached
before the frontier proves optimality, the run reports the certified
interval [incumbent, best remaining bound] and never claims optimality.

Determinism. One fixed candidate ordering, sorted (layer, index) actions,
and a monotone push counter break every tie, so a run is reproducible.
"""

import heapq
import time
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from ..geometry import Tolerances, box_2d, build_assembly
from ..mechanics import assemble
from ..solve import FEASIBLE, SolverOptions, solve_p0
from . import lattice as LT

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


@dataclass
class CertifyResult:
    """Outcome of a certification run.

    certified True means best-first branch and bound closed the gap and the
    optimum is proven. False means a budget stopped the run and only the
    interval [lower, upper] is certified. optimum is the incumbent overhang
    (the best host-certified structure found); it equals lower. placement
    records the reachability mode the run enforced ("static", "drop", or
    "slide"); the optimum is proven within that mode's legal orders.
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
    ):
        self.n = int(n)
        self.dx = float(dx)
        self.tol = tol
        self.opts = opts
        self.mu = float(mu)
        self.placement = str(placement)

        self.spec = LT.LatticeSpec(n_max=self.n, dx=self.dx, mode=self.placement)
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

        # Feasibility of a set, cached by canonical key.
        self.feas_cache = {}

        self.incumbent = float("-inf")
        self.incumbent_seq = []
        self.incumbent_trace = []

        # Instrumentation.
        self.nodes_expanded = 0
        self.nodes_generated = 0
        self.max_frontier = 0
        self.qp_solves = 0
        self.host_verifications = 0

    # --- feasibility oracle ----------------------------------------------

    def _solve_feasibility(self, keys):
        """Certified qpax feasibility for a list of set keys, batched.

        Fills feas_cache. Only uncached keys hit the solver. Uses the full
        library iteration cap so the verdict is the reference, not a screen.
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
                feasible = (float(mg) <= self.tol.tol_feas) and bool(cf)
                self.feas_cache[k] = feasible
            i += CHUNK

    # --- host re-verification --------------------------------------------

    def host_prefix_feasible(self, seq):
        """Replay a build order on the certified host pipeline.

        Builds the pedestal plus one cube per step, solves P0 at every
        prefix, and stops at the first non-feasible prefix. Returns
        (all_feasible, trace) with trace a list of
        (layer, index, x, margin, status).
        """
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

        Returns a list of (child_key, child_seq, overhang, depth, bound) for
        every feasible child, across all nodes.
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
            feasible = []
            for (a, ck, cb) in cand:
                if self.feas_cache[ck]:
                    child_seq = seq + [a]
                    ov = LT.overhang(ck, self.dx)
                    feasible.append((ck, child_seq, ov, len(ck), cb, a))

            # Incumbent update. Verify improving children in descending
            # overhang order and stop at the first host-certified one; the
            # rest have lower overhang and cannot beat it.
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
                # Host disagrees with the fast oracle. Do not raise the
                # incumbent; keep exploring lower-overhang siblings.

            out.extend(
                (ck, child_seq, ov, depth, cb)
                for (ck, child_seq, ov, depth, cb, a) in feasible
            )
        return out

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
            closed_size=len(closed),
            wall_time=wall,
            info={"reach_verified": bool(reach_verified)},
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
) -> CertifyResult:
    """Certify the grid optimum (or an interval) for n cubes at step dx."""
    engine = Certifier(n, dx, tol, opts=opts, placement=placement)
    return engine.run(max_nodes=max_nodes, time_limit=time_limit, progress=progress)
