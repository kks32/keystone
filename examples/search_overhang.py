"""Monte-Carlo tree search for maximum overhang, cube stacking on keystone.

This is the slow reference baseline. It builds and assembles every child
state on the host and calls the batched oracle one node at a time. The fast
implementation, which keeps the whole frontier on the device, is
examples/search_overhang_fast.py; use that for large runs.

Measures how far pure PUCT search reaches on the maximum-overhang cube
stacking problem, with no learning and a prefix-feasibility oracle. Every
construction step must be statically feasible: after each cube is placed
the whole assembly must satisfy the P4 elastic margin test
(margin <= tol.tol_feas) AND the force state must be cone-certified, the
same admission the batched kernel reports. Feasibility of all legal
children of a node is decided in one batched qpax call.

Ground truth anchor. The harmonic (simple) stack reaches an overhang of
sum_{k=1..n} 1/(2k) block widths beyond the support edge. Counterweighted
designs can exceed it (Paterson and Zwick). No optimum beyond the harmonic
formula is hardcoded; the harmonic sum is used only as a reported baseline
and to normalize the search value.

Task geometry (frozen). 2D, xz plane. A pedestal box_2d(width=6, height=1,
x=-3, z=0.5) puts its right edge at x = 0 and its top at z = 1. Unit cubes
box_2d(1, 1, x, z) sit at layer L with center z = 1.5 + L and center x on
the grid x = j * dx. Friction mu = 0.7 everywhere, high enough that toppling,
not sliding, governs. Overhang of a state is max over placed cubes of
(x_center + 0.5), the rightmost cube edge beyond the pedestal edge at 0.

Run: python examples/search_overhang.py --n 4 --sims 2000
"""

import argparse
import math
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")

import jax.numpy as jnp
import numpy as np

from keystone import Tolerances, assemble, box_2d, build_assembly, solve_p4
from keystone.solve import margin_batch
from keystone.viz import plot_assembly_2d, save_fig

# Frozen task constants. These describe the problem, not tolerances.
PEDESTAL_W = 6.0
PEDESTAL_H = 1.0
PEDESTAL_X = -3.0  # right edge at x = 0, top at z = 1
MU = 0.7
X_MIN = -3.0
X_MAX = 4.0

# Batch shaping. margin_batch jits per batch dimension, so batches are padded
# up to a power of two (capped at CHUNK) to keep the number of compiled shapes
# small. Padding slots repeat one real system and their margins are discarded.
CHUNK = 256


def harmonic(n):
    """Overhang of the simple stack in block widths: sum_{k=1..n} 1/(2k)."""
    return sum(1.0 / (2.0 * k) for k in range(1, n + 1))


def pedestal_box():
    """The wide base. Right edge at x = 0, top face at z = 1."""
    return box_2d(width=PEDESTAL_W, height=PEDESTAL_H, x=PEDESTAL_X, z=0.5)


def cube_box(L, j, dx):
    """Unit cube at layer L, center x = j * dx, center z = 1.5 + L."""
    return box_2d(1.0, 1.0, j * dx, 1.5 + L)


def boxes_of(placements, dx):
    """Pedestal plus one unit cube per placement (L, j)."""
    boxes = [pedestal_box()]
    for (L, j) in placements:
        boxes.append(cube_box(L, j, dx))
    return boxes


def overhang(placements, dx):
    """Rightmost cube edge beyond the pedestal edge, or -inf if no cubes."""
    if not placements:
        return float("-inf")
    return max(j * dx + 0.5 for (_, j) in placements)


def legal_actions(placements, dx, n):
    """Geometry-only legal moves from a state. No oracle here.

    A move is (layer L, grid index j) with x = j * dx in [X_MIN, X_MAX]. Rules:
    - Support. A layer-0 cube must overlap the pedestal top [-6, 0] by at least
      2 * dx. A layer L >= 1 cube must overlap the top face of at least one
      already-placed cube at layer L - 1 by at least 2 * dx, which means
      |x - x_c| <= 1 - 2 * dx.
    - No same-layer overlap. For every placed cube at layer L, |x - x_c| >= 1.
    Layers above the current top plus one are unreachable and skipped.
    """
    if len(placements) >= n:
        return []
    by_layer = {}
    for (L, j) in placements:
        by_layer.setdefault(L, []).append(j)
    max_L = max(by_layer) if by_layer else -1

    j_lo = int(math.ceil(X_MIN / dx - 1e-9))
    j_hi = int(math.floor(X_MAX / dx + 1e-9))
    support_reach = 1.0 - 2.0 * dx  # max |x - x_c| that still overlaps by 2 dx

    actions = []
    top_layer = min(max_L + 1, n - 1)
    for L in range(top_layer + 1):
        same_x = [jj * dx for jj in by_layer.get(L, [])]
        if L == 0:
            candidates = range(j_lo, j_hi + 1)
        else:
            below = by_layer.get(L - 1, [])
            if not below:
                continue
            cand = set()
            for jc in below:
                xc = jc * dx
                a = int(math.ceil((xc - support_reach) / dx - 1e-9))
                b = int(math.floor((xc + support_reach) / dx + 1e-9))
                for j in range(max(a, j_lo), min(b, j_hi) + 1):
                    cand.add(j)
            candidates = sorted(cand)
        for j in candidates:
            x = j * dx
            if L == 0:
                ov = min(x + 0.5, 0.0) - max(x - 0.5, -PEDESTAL_W)
                if ov < 2.0 * dx - 1e-9:
                    continue
            if any(abs(x - sx) < 1.0 - 1e-9 for sx in same_x):
                continue
            actions.append((L, j))
    return actions


def canonical(placements):
    """Order-independent state key. Sound because prefix-feasible histories
    reaching the same set of placements have identical futures."""
    return tuple(sorted(placements))


class Search:
    """PUCT tree search over cube placements with a batched feasibility oracle."""

    def __init__(self, n, dx, tol, c_puct, rollouts, rng, search_iter):
        self.n = n
        self.dx = dx
        self.tol = tol
        self.c_puct = c_puct
        self.rollouts = rollouts
        self.rng = rng
        # qpax iteration cap for the search feasibility oracle. Fewer iterations
        # only ever flip a near-boundary state from feasible to infeasible (a
        # measured, one-directional effect), so the verdict stays conservative:
        # a state called feasible here is feasible at the library default too.
        # The reported best sequence is re-verified with solve_p4 (default 100).
        self.search_iter = search_iter

        self.NB = n + 1                 # pedestal plus n cubes
        self.NP = 3 * (n + 1)           # provably >= max contacts (3 n)
        self.value_norm = 2.0 * harmonic(n)  # beating harmonic maps Q near 0.5

        # Tree and caches. Nodes keyed by canonical placement tuple.
        self.tree = {}
        self.feas_cache = {}            # state key -> (feasible, margin)

        # Global bests over every feasible state ever evaluated.
        self.best_overhang = float("-inf")
        self.best_key = None
        self.best_parent = None
        self.best_action = None

        # Instrumentation.
        self.t_host = 0.0               # build_assembly + assemble, seconds
        self.t_solve = 0.0              # stack + margin_batch + fetch, seconds
        self.n_qp = 0                   # real (unpadded) QP feasibility solves
        self.n_cache_hits = 0
        self.n_revisits = 0             # sims that reached an already-expanded leaf

        root = self._node((), None, None)
        self.root = root

    # --- node bookkeeping -------------------------------------------------

    def _node(self, key, parent, action_in):
        node = {
            "key": key,
            "parent": parent,
            "action_in": action_in,
            "expanded": False,
            "terminal": False,
            "solved": False,   # subtree fully explored (self or all children solved)
            "actions": [],
            "N": 0,
            "N_a": {},
            "W_a": {},
        }
        self.tree[key] = node
        return node

    # --- feasibility oracle ----------------------------------------------

    def _system(self, key):
        """Build the nondimensional equilibrium system for a state.

        Padding (NB, NP, pad_verts=2) is identical for every state in the run,
        so all systems share one array shape and the batch jits once per shape.
        """
        boxes = boxes_of(key, self.dx)
        asm = build_assembly(
            boxes, mu=MU, tol=self.tol, dim=2,
            pad_blocks=self.NB, pad_patches=self.NP, pad_verts=2,
        )
        return assemble(asm, self.tol, cone="linear2d")

    def _batch_margins(self, systems):
        """Margins and certified flags for a list of systems.

        Uses power-of-two padded batches. certified is margin_batch's
        primal cone-admissibility flag; admission requires it (see
        _evaluate).
        """
        out_m = np.empty(len(systems), dtype=np.float64)
        out_c = np.empty(len(systems), dtype=bool)
        i = 0
        while i < len(systems):
            chunk = systems[i : i + CHUNK]
            c = len(chunk)
            b = min(1 << (c - 1).bit_length(), CHUNK) if c > 0 else 1
            sel = chunk + [chunk[-1]] * (b - c)
            A_b = jnp.stack([s.A for s in sel])
            w_b = jnp.stack([s.w_dead for s in sel])
            G_b = jnp.stack([s.G for s in sel])
            margins, certified = margin_batch(
                A_b, w_b, G_b, self.tol.eps_reg, max_iter=self.search_iter
            )
            margins = np.asarray(margins)
            certified = np.asarray(certified)
            out_m[i : i + c] = margins[:c]
            out_c[i : i + c] = certified[:c]
            i += CHUNK
        return out_m, out_c

    def _evaluate(self, child_keys):
        """Feasibility for a list of state keys, cache-aware and batched.

        Returns a dict key -> feasible bool. Feasible requires both
        margin <= tol.tol_feas AND the cone-certified flag from the batched
        kernel; a small margin on an uncertified force is not admitted. Only
        uncached states hit the solver. Margins are kept in feas_cache for
        reporting.
        """
        result = {}
        pending = []
        for key in child_keys:
            if key in self.feas_cache:
                self.n_cache_hits += 1
                result[key] = self.feas_cache[key][0]
            else:
                pending.append(key)
        if pending:
            t0 = time.perf_counter()
            systems = [self._system(key) for key in pending]
            self.t_host += time.perf_counter() - t0
            t0 = time.perf_counter()
            margins, certified = self._batch_margins(systems)
            self.t_solve += time.perf_counter() - t0
            self.n_qp += len(pending)
            for key, mg, cert in zip(pending, margins, certified):
                mg = float(mg)
                feasible = bool(mg <= self.tol.tol_feas and cert)
                self.feas_cache[key] = (feasible, mg)
                result[key] = feasible
        return result

    # --- expansion, selection, backup ------------------------------------

    def _expand(self, node):
        """First visit to a node: enumerate legal moves, decide feasibility of
        all children in one batch, keep admissible ones, record any new best."""
        node["expanded"] = True
        legal = legal_actions(node["key"], self.dx, self.n)
        child_keys = {a: canonical(node["key"] + (a,)) for a in legal}
        feasible = self._evaluate(list(child_keys.values()))

        admissible = []
        for a in legal:
            ck = child_keys[a]
            if feasible[ck]:
                admissible.append(a)
                node["N_a"][a] = 0
                node["W_a"][a] = 0.0
                # Every feasible state counts toward the global best.
                ov = overhang(ck, self.dx)
                if ov > self.best_overhang:
                    self.best_overhang = ov
                    self.best_key = ck
                    self.best_parent = node["key"]
                    self.best_action = a
        node["actions"] = admissible
        node["terminal"] = len(node["key"]) >= self.n or not admissible
        # A terminal leaf has no frontier below it, so its subtree is solved.
        if node["terminal"]:
            node["solved"] = True
        # If every admissible child places the last cube, all children are
        # terminal and were fully evaluated in this one expansion (their best
        # overhang is already recorded). Nothing remains to learn below, so the
        # node is solved. This keeps sims on unexplored nodes instead of
        # descending into terminal children that carry no new information.
        elif len(node["key"]) == self.n - 1:
            node["solved"] = True

    def _puct_action(self, node):
        """Argmax of Q + c_puct * P * sqrt(N(s)) / (1 + N(s,a)), uniform prior.

        Children whose subtree is already solved are skipped, so sims are not
        wasted re-walking fully explored terminal branches. Returns None when
        every child is solved, which means this node is solved too. The prior
        stays uniform over all admissible actions (the frozen spec); pruning
        only affects which action is selected, not the prior mass.
        """
        actions = node["actions"]
        prior = 1.0 / len(actions)
        sqrt_n = math.sqrt(node["N"]) if node["N"] > 0 else 0.0
        best_a = None
        best_score = float("-inf")
        for a in actions:
            child = self.tree.get(canonical(node["key"] + (a,)))
            if child is not None and child["solved"]:
                continue
            na = node["N_a"][a]
            q = node["W_a"][a] / na if na > 0 else 0.0
            u = self.c_puct * prior * sqrt_n / (1.0 + na)
            score = q + u
            if score > best_score:
                best_score = score
                best_a = a
        return best_a

    def _select(self):
        """Descend from the root, following PUCT into unsolved subtrees."""
        node = self.root
        path = []
        while node["expanded"] and not node["terminal"] and node["actions"]:
            a = self._puct_action(node)
            if a is None:  # all children solved: this node is solved
                break
            path.append((node, a))
            ck = canonical(node["key"] + (a,))
            child = self.tree.get(ck)
            if child is None:
                child = self._node(ck, node["key"], a)
            node = child
        return node, path

    def _propagate_solved(self, key):
        """Mark nodes solved bottom up. A node is solved when it is terminal or
        every admissible child exists and is solved."""
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
                child = self.tree.get(canonical(node["key"] + (a,)))
                if child is None or not child["solved"]:
                    all_solved = False
                    break
            if not all_solved:
                break
            node["solved"] = True
            k = node["parent"]

    def _leaf_value(self, node):
        """Leaf value in [0, 1], normalized so overhang == harmonic maps to 0.5.

        A non-terminal leaf gets an optimistic room-to-grow bonus of 0.25 per
        unplaced cube, a proxy for the rollout it is not running. A terminal
        leaf has no future, so it is valued by its overhang alone, the same way
        the rollout path values a terminal state. Without this split a stuck
        shallow dead-end would collect a phantom bonus for cubes it can never
        place, which lures the search into early terminals.
        """
        ov = overhang(node["key"], self.dx)
        if ov == float("-inf"):
            ov = 0.0
        if node["terminal"]:
            v = ov / self.value_norm
        else:
            remaining = self.n - len(node["key"])
            v = (ov + 0.25 * remaining) / self.value_norm
        return min(max(v, 0.0), 1.0)

    def _rollout_value(self, node):
        """Mean terminal overhang over K random admissible rollouts, / norm.

        Each rollout step evaluates its legal children in a batched call, so
        rollouts are slow. Off by default.
        """
        totals = []
        for _ in range(self.rollouts):
            placements = node["key"]
            while len(placements) < self.n:
                legal = legal_actions(placements, self.dx, self.n)
                if not legal:
                    break
                keys = {a: canonical(placements + (a,)) for a in legal}
                feasible = self._evaluate(list(keys.values()))
                adm = [a for a in legal if feasible[keys[a]]]
                if not adm:
                    break
                a = adm[int(self.rng.integers(len(adm)))]
                placements = keys[a]
                ov = overhang(placements, self.dx)
                if ov > self.best_overhang:
                    self.best_overhang = ov
                    self.best_key = placements
                    # Rollout states are re-derivable; store a build order later.
                    self.best_parent = None
                    self.best_action = None
            ov = overhang(placements, self.dx)
            totals.append(ov if ov != float("-inf") else 0.0)
        mean_ov = sum(totals) / len(totals) if totals else 0.0
        return min(max(mean_ov / self.value_norm, 0.0), 1.0)

    def _backup(self, node, path, value):
        node["N"] += 1
        for (nd, a) in path:
            nd["N"] += 1
            nd["N_a"][a] += 1
            nd["W_a"][a] += value

    def run_sim(self):
        leaf, path = self._select()
        if not leaf["expanded"]:
            self._expand(leaf)
        else:
            self.n_revisits += 1
        if self.rollouts > 0 and not leaf["terminal"]:
            value = self._rollout_value(leaf)
        else:
            value = self._leaf_value(leaf)
        self._backup(leaf, path, value)
        self._propagate_solved(leaf["key"])

    # --- reporting --------------------------------------------------------

    def history_of(self, key):
        """A prefix-feasible action order that builds the state at key.

        Walks first-creation parent pointers to the root. Each edge was created
        from a feasible parent by a feasible action, so every prefix is feasible.
        """
        seq = []
        k = key
        while k is not None:
            node = self.tree.get(k)
            if node is None or node["action_in"] is None:
                break
            seq.append(node["action_in"])
            k = node["parent"]
        return list(reversed(seq))

    def tree_report(self):
        """Compact tree-shape summary: per-depth node count, terminal count,
        and best overhang, plus the frontier revisit count."""
        by_depth = {}
        for node in self.tree.values():
            d = len(node["key"])
            rec = by_depth.setdefault(d, [0, 0, float("-inf")])
            rec[0] += 1
            if node["expanded"] and node["terminal"]:
                rec[1] += 1
            ov = overhang(node["key"], self.dx)
            if ov > rec[2]:
                rec[2] = ov
        lines = [f"tree: {len(self.tree)} nodes, {self.n_revisits} frontier revisits"]
        for d in sorted(by_depth):
            nodes, term, best = by_depth[d]
            best_s = f"{best:.4f}" if best != float("-inf") else "n/a"
            lines.append(
                f"  depth {d}: nodes={nodes:5d} terminal={term:5d} best_overhang={best_s}"
            )
        return "\n".join(lines)

    def best_sequence(self):
        """Action sequence (list of (L, j)) that builds the best state."""
        if self.best_key is None:
            return []
        if self.best_parent is not None:
            return self.history_of(self.best_parent) + [self.best_action]
        if self.best_key in self.tree:
            return self.history_of(self.best_key)
        # Rollout-only best without a parent pointer: build bottom up.
        return sorted(self.best_key, key=lambda lj: (lj[0], lj[1]))


def verify_sequence(seq, dx, tol):
    """Replay a build order, solve P4 at each step, return the margin trace.

    Returns list of (L, j, x_center, margin, status). Prefix feasibility holds
    when every status is 'feasible'.
    """
    trace = []
    placed = []
    for (L, j) in seq:
        placed.append((L, j))
        boxes = boxes_of(tuple(placed), dx)
        n = len(placed)
        asm = build_assembly(
            boxes, mu=MU, tol=tol, dim=2,
            pad_blocks=n + 1, pad_patches=3 * (n + 1), pad_verts=2,
        )
        system = assemble(asm, tol, cone="linear2d")
        r = solve_p4(system, tol)
        trace.append((L, j, j * dx, r.margin, r.status))
    return trace


def render_best(key, dx, tol, path):
    """Draw the best structure with its P4 result and save it."""
    import matplotlib.pyplot as plt

    boxes = boxes_of(key, dx)
    n = len(key)
    asm = build_assembly(
        boxes, mu=MU, tol=tol, dim=2,
        pad_blocks=n + 1, pad_patches=3 * (n + 1), pad_verts=2,
    )
    system = assemble(asm, tol, cone="linear2d")
    r = solve_p4(system, tol)
    ov = overhang(key, dx)
    title = f"best overhang = {ov:.4f} (n={n} cubes)"
    fig = plot_assembly_2d(asm, r, boxes=boxes, title=title)
    save_fig(fig, path)
    plt.close(fig)
    return r


def main():
    parser = argparse.ArgumentParser(
        description="MCTS for maximum overhang cube stacking on keystone."
    )
    parser.add_argument("--n", type=int, default=4, help="number of cubes")
    parser.add_argument("--sims", type=int, default=2000, help="MCTS simulations")
    parser.add_argument("--dx", type=float, default=1.0 / 24.0, help="x grid step")
    parser.add_argument("--cpuct", type=float, default=1.4, help="PUCT constant")
    parser.add_argument("--rollouts", type=int, default=0,
                        help="random rollouts per leaf (0 uses the leaf formula)")
    parser.add_argument("--seed", type=int, default=0, help="numpy seed")
    parser.add_argument("--search_iter", type=int, default=50,
                        help="qpax iteration cap for the search oracle "
                             "(conservative below 100; final check uses 100)")
    parser.add_argument("--out", default="out/search", help="output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tol = Tolerances()
    rng = np.random.default_rng(args.seed)

    search = Search(
        args.n, args.dx, tol, args.cpuct, args.rollouts, rng, args.search_iter
    )
    base = harmonic(args.n)

    print(
        f"search: n={args.n} sims={args.sims} dx=1/{round(1 / args.dx)} "
        f"c_puct={args.cpuct} rollouts={args.rollouts} seed={args.seed} "
        f"search_iter={args.search_iter}",
        flush=True,
    )
    print(f"harmonic baseline sum(1/2k) = {base:.6f} block widths", flush=True)

    t_start = time.perf_counter()
    for sim in range(1, args.sims + 1):
        search.run_sim()
        if sim % 500 == 0 or sim == args.sims:
            dt = time.perf_counter() - t_start
            rate = sim / dt if dt > 0 else 0.0
            print(
                f"  sims={sim:6d}  nodes={len(search.tree):7d}  "
                f"best_overhang={search.best_overhang:.4f}  "
                f"{rate:6.1f} sims/s",
                flush=True,
            )
    wall = time.perf_counter() - t_start

    seq = search.best_sequence()
    best = search.best_overhang
    ratio = best / base if base > 0 else float("nan")

    print("")
    print(f"best overhang        = {best:.6f} block widths")
    print(f"harmonic(n)          = {base:.6f}")
    print(f"ratio best/harmonic  = {ratio:.4f}")
    print(f"gap below harmonic   = {base - best:.6f} "
          f"({(base - best) / args.dx:.2f} grid steps)")
    print(f"best sequence (L, j, x): "
          + ", ".join(f"({L},{j},{j * args.dx:+.4f})" for (L, j) in seq))
    print(f"wall time            = {wall:.1f} s")
    print(f"time split           = host {search.t_host:.1f} s, "
          f"solver {search.t_solve:.1f} s")
    print(f"qp feasibility solves= {search.n_qp} "
          f"(cache hits {search.n_cache_hits})")
    if search.n_qp > 0:
        print(f"solver per qp        = {1000 * search.t_solve / search.n_qp:.2f} ms")
    print(search.tree_report(), flush=True)

    # Re-check prefix feasibility of the best sequence, step by step.
    print("")
    print("prefix margin trace of the best sequence (step by step):")
    trace = verify_sequence(seq, args.dx, tol)
    all_feasible = True
    for step, (L, j, x, margin, status) in enumerate(trace, start=1):
        ov = x + 0.5
        print(f"  step {step:2d}: place (L={L}, x={x:+.4f}) "
              f"cube_edge={ov:+.4f}  margin={margin:.3e}  {status}")
        if status != "feasible":
            all_feasible = False
    print(f"prefix feasible throughout: {all_feasible}")

    # Renders.
    png = os.path.join(args.out, f"best_n{args.n}.png")
    if search.best_key is not None:
        render_best(search.best_key, args.dx, tol, png)
        print(f"wrote {png}")
    else:
        print("no feasible cube placement found")

    return 0


if __name__ == "__main__":
    sys.exit(main())
