"""AlphaZero-style learned priors and values for the lattice search.

The tree search in mcts.py explores placements with a uniform PUCT prior and
an optimistic heuristic leaf value. This module trains one small network that
shapes that exploration: a policy head that biases the prior toward promising
placements and a value head that scores a partial stack. Nothing about
certification changes. Feasibility is still decided by the certified qpax
kernel; the network only decides where the search looks first.

Design choices that make one network cover several stack sizes:

- Action space is the full layer-major grid of a canonical spec with
  max_layers layers at a fixed dx. The grid index of a placement is
  layer * n_pos + (xidx - j_lo), the same integer mcts.Search uses, so a model
  index and a search action index are identical. A size-n search uses layers
  0..n-1, which are the first n * n_pos indices of the canonical grid, so a
  size-n action maps into the canonical space with no remapping.
- The feature vector is the occupancy grid over (layer, xidx) plus five
  scalars: placed fraction count/n, current rightmost cube edge over two,
  remaining fraction (n - count)/n, and two absolute-scale horizon features,
  n / n_max_global and log(n) / log(n_max_global). The first three are ratios
  to n and so collapse to the same values at the empty root for every horizon;
  the two absolute features distinguish horizons at the empty root, so the net
  can condition its root prior and value on n. n_max_global is the largest
  stack the shared model covers, MAX_LAYERS.
- Legality is never learned. The search supplies the legal or admissible
  action indices; the network only ranks them. Priors are masked to those
  indices and mixed with a uniform floor so every legal action keeps support.

Determinism: flax init is seeded by a PRNGKey, minibatch order is a seeded
numpy permutation, and the adapters are pure functions of the current params.
"""

import functools
import json
import math
import os
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from . import lattice as LT
from .lattice import DX, LatticeSpec, harmonic, overhang

# Canonical layer count. It bounds the largest stack the shared model covers.
# n = 4, 5, 6 all fit inside six layers at dx = 1/12.
MAX_LAYERS = 6

# Global horizon normalizer for the absolute-scale root features. It is the
# largest stack size the shared model is meant to serve, equal to MAX_LAYERS.
# n / N_MAX_GLOBAL and log(n) / log(N_MAX_GLOBAL) sit in (0, 1] for n in 1..6
# and give the empty root a distinct feature vector per horizon, which the
# ratio-to-n scalars cannot (they are all 0/0/1 at count 0 for every n).
N_MAX_GLOBAL = MAX_LAYERS

# Default mix toward the uniform prior. PUCT weights the exploration bonus by
# the prior, so an action with a zero prior gets a zero bonus and, before it is
# ever visited, a zero Q. It could then never be selected. The floor keeps
# every legal action reachable.
EPS_UNIFORM = 0.1

# Fixed range for the margin head, in log10 of the certified P4 margin. The
# search records margins in feas_cache; a small sweep on n = 4 saw log10
# margins from about -11.7 (near equilibrium) to -1.9 (clearly infeasible),
# so [-12, 0] brackets the observed range with headroom. The target maps this
# window linearly onto [0, 1]: log10 == -12 -> 0, log10 == 0 -> 1. A floor of
# 1e-12 inside the log keeps the transform finite when a margin is exactly 0.
MARGIN_LOG_LO = -12.0
MARGIN_LOG_HI = 0.0
MARGIN_FLOOR = 1e-12

# Default weight on the auxiliary loss in the combined objective. The total is
# policy CE + value MSE + MARGIN_WEIGHT * aux MSE. Exposed as a train argument;
# this is only the default.
MARGIN_WEIGHT = 0.5

# Lateral-reserve auxiliary target. The new default auxiliary head regresses
# the reserve capacity min(|lam+|, |lam-|): the largest symmetric lateral load
# factor, as a fraction of self-weight, under which the state stays P4-feasible
# in both +x and -x. This is a genuine distance-to-collapse (it is the quantity
# the LAM_MIN calibration in lattice.py is tied to, separating stood from
# toppled with a clean gap), unlike the old margin head, which regressed the
# near-zero P4 equilibrium residual and so had almost no signal to learn.
#
# The reserve is found by sweeping lattice.robust_margins_of_states (the
# imported reserve kernel; not edited) over a lam grid and taking, per state,
# the largest grid lam still certified-robust from lam = 0 up. RESERVE_LAM_HI
# tops the sweep, RESERVE_GRID sets both resolution and per-state solve cost
# (2 * RESERVE_GRID certified directional P4 solves per distinct state).
# RESERVE_CLIP maps the capacity linearly onto [0, 1]: reserve == 0 -> 0,
# reserve >= RESERVE_CLIP -> 1.
RESERVE_LAM_HI = 0.5
RESERVE_GRID = 16
RESERVE_CLIP = 0.25

# Known-good dx = 1/12 build orders from real uniform searches, kept as seed
# imitation data. Provenance: out/search/run_n4_dx12.log and
# out/search/run_n6_and_determinism.log (examples/search_overhang.py, seed 0
# and seed 7, 2026-07-15). Each is prefix-feasible on the host pipeline.
KNOWN_SEQUENCES_DX12 = (
    {"n": 4, "dx": 1.0 / 12.0, "seq": [(0, -1), (1, -6), (2, -3), (1, 8)]},
    {"n": 6, "dx": 1.0 / 12.0,
     "seq": [(0, -11), (0, -35), (0, -23), (0, 1), (1, -4), (1, 8)]},
    {"n": 4, "dx": 1.0 / 12.0, "seq": [(0, -1), (0, -25), (0, -13), (1, 4)]},
)


# --- feature and action encoding ------------------------------------------


@dataclass(frozen=True)
class FeatureSpec:
    """Static shape of the shared model input and action space.

    n_pos and j_lo come from a canonical LatticeSpec and depend only on dx,
    not on the stack size. M is the canonical action-grid size and F is the
    feature length: the occupancy grid plus three scalars.
    """

    dx: float
    max_layers: int
    n_pos: int
    j_lo: int
    n_max_global: int = N_MAX_GLOBAL

    @property
    def M(self) -> int:
        return self.max_layers * self.n_pos

    @property
    def F(self) -> int:
        # Occupancy grid plus five scalars (three ratio-to-n, two absolute
        # horizon features). See encode_state.
        return self.M + 5


def make_feature_spec(dx: float = DX, max_layers: int = MAX_LAYERS) -> FeatureSpec:
    """Feature spec for a fixed dx. n_pos and j_lo are read from the lattice."""
    ls = LatticeSpec(n_max=max_layers, dx=dx)
    return FeatureSpec(dx=dx, max_layers=max_layers, n_pos=ls.n_pos,
                       j_lo=ls.j_lo, n_max_global=N_MAX_GLOBAL)


def action_index(fs: FeatureSpec, layer: int, xidx: int) -> int:
    """Layer-major grid index of a placement, the shared model index."""
    return int(layer) * fs.n_pos + (int(xidx) - fs.j_lo)


def encode_state(fs: FeatureSpec, key, n: int) -> np.ndarray:
    """Feature vector for a state, as float32 of length F.

    key is a tuple of placed (layer, xidx) pairs. n is the target stack size
    of the episode. Positions outside the canonical grid are skipped; on this
    scene every legal placement is inside it.
    """
    feat = np.zeros(fs.F, dtype=np.float32)
    for (L, j) in key:
        if 0 <= L < fs.max_layers and fs.j_lo <= j <= fs.j_lo + fs.n_pos - 1:
            feat[L * fs.n_pos + (j - fs.j_lo)] = 1.0
    count = len(key)
    ov = overhang(key, fs.dx)
    edge = 0.0 if ov == float("-inf") else ov
    # Ratio-to-n scalars: identical at the empty root for every horizon.
    feat[fs.M + 0] = count / n
    feat[fs.M + 1] = edge / 2.0
    feat[fs.M + 2] = (n - count) / n
    # Absolute-scale horizon features: distinct per n at the empty root, so the
    # net can condition on the target size before any cube is placed.
    ng = fs.n_max_global
    feat[fs.M + 3] = n / ng
    feat[fs.M + 4] = math.log(n) / math.log(ng) if ng > 1 else 0.0
    return feat


def legal_masks(fs: FeatureSpec, keys, n: int) -> np.ndarray:
    """Legal-action masks for a list of size-n states, shape (len(keys), M).

    Legality is the pure-geometry lattice rule at size n. A size-n legal index
    equals its canonical index, so the size-n mask fills the first n * n_pos
    columns and the higher layers stay illegal.
    """
    if not keys:
        return np.zeros((0, fs.M), dtype=bool)
    spec = LatticeSpec(n_max=n, dx=fs.dx)
    states = LT.batch_states(spec, list(keys))
    cl, cj = LT.action_grid(spec)
    lg = np.asarray(LT.legal_grid(spec, states, cl, cj))  # (len, n * n_pos)
    masks = np.zeros((len(keys), fs.M), dtype=bool)
    masks[:, : lg.shape[1]] = lg
    return masks


# --- network ---------------------------------------------------------------


class AZNet(nn.Module):
    """Two-hidden-layer MLP with policy, value, and auxiliary heads.

    The policy head emits logits over the full action grid; the caller masks
    them to the legal set. The value head is a sigmoid in [0, 1], matching the
    search value scale where overhang == harmonic(n) maps near 0.5. The third
    sigmoid output is the auxiliary head. Its default target is the normalized
    lateral-reserve capacity min(|lam+|, |lam-|) clipped to [0, RESERVE_CLIP]
    (see _reserve_target), the quantity calibrated to predict physical
    survival. assemble_arrays(aux_mode="margin") reproduces the legacy target,
    the normalized log10 P4 equilibrium residual (MARGIN_LOG_LO /
    MARGIN_LOG_HI), for comparison runs. The head is auxiliary supervision
    only; the search never reads it, so it stays backward compatible with
    callers that unpack the first two outputs by position.
    """

    m: int
    hidden: int = 256

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(self.hidden)(x))
        x = nn.relu(nn.Dense(self.hidden)(x))
        logits = nn.Dense(self.m)(x)
        value = nn.sigmoid(nn.Dense(1)(x))
        margin = nn.sigmoid(nn.Dense(1)(x))
        return logits, value[..., 0], margin[..., 0]


class AZModel:
    """A network plus its parameters and a jitted forward pass.

    One model serves every stack size for a given feature spec. init_seed
    fixes the flax parameter initialization, so two models built with the same
    seed are identical.
    """

    def __init__(self, fs: FeatureSpec, init_seed: int = 0, hidden: int = 256,
                 eps: float = EPS_UNIFORM):
        self.fs = fs
        self.eps = float(eps)
        self.net = AZNet(m=fs.M, hidden=hidden)
        dummy = jnp.zeros((1, fs.F), dtype=jnp.float32)
        self.params = self.net.init(jax.random.PRNGKey(int(init_seed)), dummy)
        self._apply = jax.jit(lambda p, x: self.net.apply(p, x))

    def forward(self, feats: np.ndarray):
        """Logits (B, M), values (B,), and margins (B,) for a feature batch.

        The third output is the normalized margin head. Callers that want only
        priors or values index [0] or [1] and are unaffected by its presence.
        """
        return self._apply(self.params, jnp.asarray(feats, dtype=jnp.float32))


# --- search adapters --------------------------------------------------------


def make_prior_fn(model: AZModel, n: int):
    """prior_fn(state_key, action_indices) -> prior over those actions.

    Softmax of the policy logits gathered at the given indices, mixed with a
    uniform floor. Results are cached per state key because the params are
    fixed during a search.
    """
    fs = model.fs
    eps = model.eps
    cache: dict = {}

    def prior_fn(key, action_indices):
        idx = np.asarray(action_indices, dtype=np.int64)
        logits = cache.get(key)
        if logits is None:
            feat = encode_state(fs, key, n)[None, :]
            logits = np.asarray(model.forward(feat)[0])[0]
            cache[key] = logits
        sub = logits[idx]
        sub = sub - sub.max()
        p = np.exp(sub)
        p = p / p.sum()
        u = np.ones_like(p) / len(p)
        return (1.0 - eps) * p + eps * u

    return prior_fn


def make_value_fn(model: AZModel, n: int):
    """value_fn(state_key) -> scalar value in [0, 1], cached per key."""
    fs = model.fs
    cache: dict = {}

    def value_fn(key):
        v = cache.get(key)
        if v is None:
            feat = encode_state(fs, key, n)[None, :]
            v = float(np.asarray(model.forward(feat)[1])[0])
            cache[key] = v
        return v

    return value_fn


# --- training data ----------------------------------------------------------


@dataclass
class Sample:
    """One training row before batching.

    Exactly one of `taken` (imitation, an action index) and `pol` (self-play,
    an action-index -> probability map) is set. value is the target for the
    value head. margin is the certified P4 margin of this state (the old
    auxiliary target, kept for comparison), or None when no certified margin is
    known. reserve is the lateral-reserve capacity min(|lam+|, |lam-|) of the
    state (the new auxiliary target), or None when it was not computed. Which
    of the two feeds the auxiliary head is chosen by assemble_arrays(aux_mode).
    """

    key: tuple
    n: int
    taken: int
    pol: dict
    value: float
    margin: float = None
    reserve: float = None


def _value_target(overhang_val: float, n: int) -> float:
    """Overhang normalized to [0, 1] by 2 * harmonic(n), then clipped."""
    v = overhang_val / (2.0 * harmonic(n))
    return float(min(max(v, 0.0), 1.0))


def _margin_target(margin_val: float) -> float:
    """Certified margin mapped to [0, 1] through normalized log10.

    log10(margin + MARGIN_FLOOR) is placed on the fixed window
    [MARGIN_LOG_LO, MARGIN_LOG_HI] and clipped. A near-zero margin (near
    equilibrium) maps near 0; a large margin (clearly infeasible) maps near 1.
    """
    lm = math.log10(max(float(margin_val), 0.0) + MARGIN_FLOOR)
    x = (lm - MARGIN_LOG_LO) / (MARGIN_LOG_HI - MARGIN_LOG_LO)
    return float(min(max(x, 0.0), 1.0))


def _reserve_target(reserve_val: float) -> float:
    """Lateral-reserve capacity mapped to [0, 1] by a linear clip.

    reserve == 0 -> 0, reserve >= RESERVE_CLIP -> 1, linear in between. The
    reserve is min(|lam+|, |lam-|), a load factor as a fraction of self-weight.
    """
    x = float(reserve_val) / RESERVE_CLIP
    return float(min(max(x, 0.0), 1.0))


def reserve_capacities(spec, keys, tol, *, solver_tol, max_iter,
                       lam_hi=RESERVE_LAM_HI, grid=RESERVE_GRID):
    """Lateral-reserve capacity min(|lam+|, |lam-|) for a list of state keys.

    Sweeps the imported reserve kernel lattice.robust_margins_of_states over
    `grid` lam values in (0, lam_hi]. At each lam the kernel returns, per state,
    the worse of the +lam and -lam P4 margins and the AND of the two cone
    certificates; the state is robust at lam when that worst margin certifies
    feasible. The reserve is the largest lam that is robust with every smaller
    grid lam also robust (the monotone first-failure boundary), which is exactly
    min(|lam+|, |lam-|) because the worst-direction margin binds first.

    Fully batched over keys: per power-of-two chunk (capped at 256, the same
    shaping as the search frontier, so compiled shapes are shared) one vmapped
    kernel call per lam value, each call a 2-direction certified solve per
    state. Returns (caps, n_solves) where caps is a float array aligned with
    keys and n_solves is the certified directional P4 solve count spent,
    2 * grid * (number of distinct keys).
    """
    if not keys:
        return np.zeros((0,), dtype=np.float64), 0
    uniq = list(dict.fromkeys(keys))  # dedupe, keep order; solve each once
    lams = np.linspace(lam_hi / grid, lam_hi, grid)
    robust = np.zeros((grid, len(uniq)), dtype=bool)
    chunk_cap = 256
    i = 0
    while i < len(uniq):
        chunk = uniq[i : i + chunk_cap]
        c = len(chunk)
        b = min(1 << (c - 1).bit_length(), chunk_cap) if c > 1 else 1
        padded = chunk + [chunk[-1]] * (b - c)
        states = LT.batch_states(spec, padded)
        for gi, lam in enumerate(lams):
            margins, cert = LT.robust_margins_of_states(
                spec, states, tol.eps_reg, tol.tol_cone, float(lam),
                solver_tol=solver_tol, max_iter=max_iter,
            )
            margins = np.asarray(margins)[:c]
            cert = np.asarray(cert)[:c]
            robust[gi, i : i + c] = (margins <= tol.tol_feas) & cert
        i += chunk_cap
    caps_uniq = np.zeros(len(uniq), dtype=np.float64)
    for s in range(len(uniq)):
        cap = 0.0
        for gi in range(grid):
            if robust[gi, s]:
                cap = float(lams[gi])
            else:
                break
        caps_uniq[s] = cap
    cap_by_key = {k: caps_uniq[i] for i, k in enumerate(uniq)}
    caps = np.array([cap_by_key[k] for k in keys], dtype=np.float64)
    return caps, 2 * grid * len(uniq)


def attach_reserve_targets(samples, tol, *, dx, solver_tol, max_iter,
                           lam_hi=RESERVE_LAM_HI, grid=RESERVE_GRID,
                           max_states=None):
    """Fill smp.reserve for a list of samples in place.

    Groups samples by stack size (each size needs its own LatticeSpec for the
    padded solve shape), computes reserve capacities batched per size, and
    writes the capacity onto each sample. max_states caps the number of
    distinct states solved across all sizes so the extra-solve budget stays
    bounded; samples past the cap keep reserve == None and contribute no
    auxiliary target. Returns (n_solves, wall_seconds) for cost reporting.
    """
    import time

    t0 = time.perf_counter()
    by_n = {}
    for smp in samples:
        by_n.setdefault(smp.n, []).append(smp)
    total_solves = 0
    solved_states = 0
    for n, group in sorted(by_n.items()):
        spec = LatticeSpec(n_max=max(int(n), 1), dx=dx)
        keys = [smp.key for smp in group]
        if max_states is not None:
            uniq = list(dict.fromkeys(keys))
            room = max_states - solved_states
            if room <= 0:
                continue
            allowed = set(uniq[:room]) if len(uniq) > room else set(uniq)
            idx = [i for i, k in enumerate(keys) if k in allowed]
        else:
            idx = list(range(len(keys)))
        sub_keys = [keys[i] for i in idx]
        caps, nsolve = reserve_capacities(
            spec, sub_keys, tol, solver_tol=solver_tol, max_iter=max_iter,
            lam_hi=lam_hi, grid=grid,
        )
        total_solves += nsolve
        solved_states += len(set(sub_keys))
        for j, i in enumerate(idx):
            group[i].reserve = float(caps[j])
    return total_solves, time.perf_counter() - t0


def _suffix_max_overhangs(seq, dx):
    """Suffix-max overhang for each prefix state along a build order.

    seq is a list of (layer, xidx) placements. State s_k is the prefix with k
    cubes placed, for k = 0..len(seq). Returns a list r of length len(seq)
    where r[k] = max over j >= k of overhang(s_j), the best overhang reachable
    from state s_k by continuing this order. Overhang is non-decreasing along a
    build order, so this equals the final overhang here; it is computed as a
    true suffix maximum so the target stays correct for any ordering.
    """
    ovs = [overhang(tuple(seq[:j]), dx) for j in range(len(seq) + 1)]
    suffix = [float("-inf")] * (len(seq) + 1)
    run = float("-inf")
    for k in range(len(seq), -1, -1):
        run = max(run, ovs[k])
        suffix[k] = run
    return suffix[: len(seq)]


def _sequence_is_lattice_legal(fs: FeatureSpec, seq, n: int) -> bool:
    """True when every placement in seq is legal on the lattice at size n.

    Sequences from the naive host script can place same-layer cubes exactly one
    block width apart, which the lattice rejects. Imitating such a placement
    would put target mass on an action the search never offers, so those
    records are dropped.
    """
    spec = LatticeSpec(n_max=n, dx=fs.dx)
    state = LT.empty_state(spec)
    for (L, j) in seq:
        if not bool(LT.is_legal(spec, state, L, j)):
            return False
        state = LT.place(spec, state, L, j)
    return True


def imitation_samples(fs: FeatureSpec, records):
    """Prefix imitation rows from good build orders, plus a drop count.

    records is an iterable of dicts with keys n, dx, seq. Only records whose dx
    matches the feature spec and whose every placement is lattice-legal are
    used. For each prefix, the policy target is the taken action and the value
    target is the final overhang normalized. Returns (samples, n_dropped).
    """
    out = []
    dropped = 0
    for rec in records:
        if abs(rec["dx"] - fs.dx) > 1e-12:
            dropped += 1
            continue
        n = int(rec["n"])
        seq = [(int(L), int(j)) for (L, j) in rec["seq"]]
        if not _sequence_is_lattice_legal(fs, seq, n):
            dropped += 1
            continue
        # Per-state value: the best overhang reachable from each prefix state,
        # not one constant final value shared by the whole trajectory.
        suffix = _suffix_max_overhangs(seq, fs.dx)
        for k in range(len(seq)):
            prefix = tuple(sorted(seq[:k]))
            vt = _value_target(suffix[k], n)
            out.append(Sample(key=prefix, n=n, taken=action_index(fs, *seq[k]),
                              pol=None, value=vt, margin=None))
    return out, dropped


def _subtree_targets(search, robust: bool):
    """Best reachable overhang per node, and whether its best path is knife-edge.

    submax[key] is the maximum overhang over the node and all its feasible
    descendants in the search DAG: the best overhang the search proved
    reachable from that state. knife[key] is True when every best path from
    the node down to that overhang runs through a knife-edge state, a
    certified-feasible state whose margin sits within a factor of ten of
    tol_feas. knife is meaningful only when robust is True; otherwise it is
    all False.

    DAG correctness. The transposition table stores one node per canonical
    placement set, so a state can have several parents; the single
    node["parent"] field keeps only the first one seen. Walking first-parent
    edges alone drops every other parent's subtree from its target, so a
    parent whose good descendant was first reached through a different order
    was silently under-credited. The fix aggregates over the full edge set:
    mcts.Search records every parent->child edge at expansion in
    _children_of, and the targets here are computed by one reverse
    topological pass over that DAG (children have exactly one more cube than
    their parent, so descending key length is a topological order; each node
    then reads the finished values of all its children, however many parents
    those children have). When the edge set is absent (hand-built stubs) the
    pass falls back to the tree's first-parent edges.

    Two smaller semantic notes. First, enumerated frontier children that were
    solved certified-feasible at expansion but never selected also count:
    their overhang is search evidence and it is exactly what
    Search.best_overhang credits. Infeasible children never count. Second,
    with several best paths to the same overhang, knife is True only when all
    of them are knife-edge; a safe path to the same overhang clears the flag.
    """
    tree = search.tree
    dx = search.dx
    tol_feas = search.tol.tol_feas
    knife_lo = tol_feas / 10.0

    # Full DAG edges when the search recorded them, first-parent tree edges
    # as the fallback (and for stub searches in tests).
    children = {}
    for k, node in tree.items():
        p = node["parent"]
        if p is not None:
            children.setdefault(p, set()).add(k)
    for p, kids in getattr(search, "_children_of", {}).items():
        children.setdefault(p, set()).update(kids)

    feas = getattr(search, "feas_cache", {})

    def feasible_state(k):
        if k in tree:
            # Tree nodes exist only for admissible (certified-feasible)
            # actions, plus the root.
            return True
        hit = feas.get(k)
        return hit is not None and bool(hit[0])

    def self_knife(k):
        if not robust:
            return False
        mg = search.margin_of(k)
        # Knife-edge: certified feasible but within one order of tol_feas.
        return mg is not None and knife_lo <= mg <= tol_feas

    nodes = set(tree)
    for p, kids in children.items():
        nodes.add(p)
        nodes.update(kids)
    nodes = {k for k in nodes if feasible_state(k)}

    submax = {}
    knife = {}
    # Reverse topological order: a child key always has one more placement
    # than its parent, so descending length visits all children first.
    for k in sorted(nodes, key=len, reverse=True):
        best = overhang(k, dx)
        best_knife = self_knife(k)
        for c in children.get(k, ()):
            if c not in nodes:
                continue  # infeasible frontier child: not reachable evidence
            cmax = submax[c]
            cknife = self_knife(k) or knife[c]
            if cmax > best:
                best = cmax
                best_knife = cknife
            elif cmax == best:
                # A tie keeps knife only if both options are knife-edge.
                best_knife = best_knife and cknife
        submax[k] = best
        knife[k] = best_knife
    return submax, knife


def selfplay_samples(fs: FeatureSpec, search, robust: bool = False,
                     robust_penalty: float = 0.05) -> list:
    """Distillation rows from a finished search tree.

    For every expanded, non-terminal node with visits, the policy target is the
    child visit distribution. The value target is the best overhang reachable
    from that node, its DAG subtree maximum aggregated over all stored parents
    (see _subtree_targets), normalized. Each node then gets credit for what its
    own subtree can achieve rather than one constant episode return shared by
    the whole tree. The margin field is the node's certified P4 margin,
    threaded from the search feasibility cache; it backs the legacy
    aux_mode="margin" target. The reserve field is left None here; call
    attach_reserve_targets to fill it for aux_mode="reserve".

    robust subtracts robust_penalty from the value target of any node whose
    best path runs through a knife-edge state (see _subtree_targets). Default
    off. robust_penalty is in normalized value units, about 0.1 block widths of
    overhang at n = 4.
    """
    out = []
    n = search.n
    submax, knife = _subtree_targets(search, robust)
    for node in search.tree.values():
        if not node["expanded"] or node["terminal"]:
            continue
        acts = node["actions"]
        visits = np.array([node["N_a"][a] for a in acts], dtype=np.float64)
        tot = visits.sum()
        if tot <= 0:
            continue
        pol = {}
        for a, v in zip(acts, visits):
            if v > 0:
                pol[action_index(fs, *a)] = float(v / tot)
        key = node["key"]
        best_ov = submax.get(key, float("-inf"))
        vt = _value_target(0.0 if best_ov == float("-inf") else best_ov, n)
        if robust and knife.get(key, False):
            vt = float(min(max(vt - robust_penalty, 0.0), 1.0))
        out.append(Sample(key=key, n=n, taken=None, pol=pol, value=vt,
                          margin=search.margin_of(key)))
    return out


def assemble_arrays(fs: FeatureSpec, samples, smooth: float = 0.9,
                    aux_mode: str = "reserve"):
    """Stack samples into arrays for training.

    Returns (feats, pols, vals, masks, aux, aux_mask). Legal masks are
    computed once per stack size. Imitation policy targets are a smoothed
    one-hot: `smooth` on the taken action and the rest spread over all legal
    actions. Self-play targets are the visit distribution as given.

    aux_mode picks the auxiliary-head target per row:
      "reserve" (default): the normalized lateral-reserve capacity
        _reserve_target(smp.reserve), the calibrated distance-to-collapse
        quantity. Rows fill only when attach_reserve_targets set smp.reserve.
      "margin": the legacy normalized log10 P4 equilibrium-residual target
        _margin_target(smp.margin), kept to reproduce the old experiments.
    aux_mask is True on rows carrying the chosen target, so the auxiliary loss
    trains only on rows the target was computed for.
    """
    if aux_mode not in ("reserve", "margin"):
        raise ValueError(f"aux_mode must be 'reserve' or 'margin', "
                         f"got {aux_mode!r}")
    s = len(samples)
    feats = np.zeros((s, fs.F), dtype=np.float32)
    pols = np.zeros((s, fs.M), dtype=np.float32)
    vals = np.zeros((s,), dtype=np.float32)
    masks = np.zeros((s, fs.M), dtype=bool)
    aux = np.zeros((s,), dtype=np.float32)
    aux_mask = np.zeros((s,), dtype=bool)

    # Group row indices by stack size so legal masks batch per size.
    by_n = {}
    for i, smp in enumerate(samples):
        by_n.setdefault(smp.n, []).append(i)

    for n, rows in by_n.items():
        keys = [samples[i].key for i in rows]
        m = legal_masks(fs, keys, n)
        for r, i in enumerate(rows):
            smp = samples[i]
            feats[i] = encode_state(fs, smp.key, n)
            masks[i] = m[r]
            vals[i] = smp.value
            if aux_mode == "reserve":
                if smp.reserve is not None:
                    aux[i] = _reserve_target(smp.reserve)
                    aux_mask[i] = True
            elif smp.margin is not None:
                aux[i] = _margin_target(smp.margin)
                aux_mask[i] = True
            if smp.pol is not None:
                for idx, prob in smp.pol.items():
                    pols[i, idx] = prob
            else:
                nleg = int(m[r].sum())
                base = (1.0 - smooth) / max(nleg, 1)
                pols[i][m[r]] = base
                pols[i, smp.taken] += smooth
            tot = pols[i].sum()
            if tot > 0:
                pols[i] /= tot
    return feats, pols, vals, masks, aux, aux_mask


# --- training loop ----------------------------------------------------------


def _loss(net, params, f, p, v, m, mg, mgm, margin_weight):
    """Masked policy CE, value MSE, and weighted masked margin MSE.

    mg is the normalized margin target and mgm is a 0/1 mask marking rows that
    carry a certified margin. The margin term is a masked mean, so rows without
    a margin contribute nothing and never bias the head. When no row in the
    batch has a margin the term is zero.
    """
    logits, val, marg = net.apply(params, f)
    masked = jnp.where(m, logits, jnp.float32(-1e9))
    logp = jax.nn.log_softmax(masked, axis=-1)
    pol_loss = jnp.mean(-jnp.sum(p * logp, axis=-1))
    val_loss = jnp.mean((val - v) ** 2)
    denom = jnp.maximum(jnp.sum(mgm), 1.0)
    margin_loss = jnp.sum(mgm * (marg - mg) ** 2) / denom
    total = pol_loss + val_loss + margin_weight * margin_loss
    return total, (pol_loss, val_loss, margin_loss)


def train(model: AZModel, feats, pols, vals, masks, margins=None,
          margin_mask=None, steps: int = 1000, batch: int = 256,
          lr: float = 3e-4, margin_weight: float = MARGIN_WEIGHT,
          seed: int = 0):
    """Train the model in place. Returns the loss history as a list of dicts.

    Adam at the given learning rate. Minibatches are a seeded permutation of
    the rows, cycled until `steps` updates run. Deterministic for a fixed seed
    and dataset. margins and margin_mask carry whichever auxiliary target
    assemble_arrays produced (reserve by default, legacy margin on request);
    they default to no auxiliary supervision, and passing them with
    margin_weight > 0 trains the auxiliary head. Each history row records the
    policy, value, and auxiliary ("margin") components.
    """
    net = model.net
    opt = optax.adam(lr)
    opt_state = opt.init(model.params)

    s = feats.shape[0]
    if margins is None:
        margins = np.zeros((s,), dtype=np.float32)
    if margin_mask is None:
        margin_mask = np.zeros((s,), dtype=bool)

    grad_fn = jax.jit(jax.value_and_grad(
        functools.partial(_loss, net), has_aux=True))

    @jax.jit
    def step(params, opt_state, f, p, v, m, mg, mgm):
        (total, (pl, vl, ml)), grads = grad_fn(
            params, f, p, v, m, mg, mgm, jnp.float32(margin_weight))
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, total, pl, vl, ml

    fj = jnp.asarray(feats)
    pj = jnp.asarray(pols)
    vj = jnp.asarray(vals)
    mj = jnp.asarray(masks)
    mgj = jnp.asarray(margins)
    mgmj = jnp.asarray(margin_mask.astype(np.float32))

    rng = np.random.default_rng(seed)
    order = rng.permutation(s)
    pos = 0
    params = model.params
    history = []
    for t in range(steps):
        if pos + batch > s:
            order = rng.permutation(s)
            pos = 0
        sel = order[pos : pos + batch]
        pos += batch
        sj = jnp.asarray(sel)
        params, opt_state, total, pl, vl, ml = step(
            params, opt_state, fj[sj], pj[sj], vj[sj], mj[sj], mgj[sj], mgmj[sj]
        )
        if t == 0 or (t + 1) % 100 == 0 or t == steps - 1:
            history.append({
                "step": t + 1,
                "loss": float(total),
                "policy": float(pl),
                "value": float(vl),
                "margin": float(ml),
            })
    model.params = params
    return history


# --- checkpoints ------------------------------------------------------------


def save_params(model: AZModel, path: str):
    """Write model params with flax serialization."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(model.params))


def load_params(model: AZModel, path: str):
    """Load params into the model in place.

    Raises ValueError when the checkpoint's parameter shapes do not match the
    model (a checkpoint from an older feature encoding, for example).
    serialization.from_bytes restores stale shapes silently and the failure
    would otherwise surface only at the first forward pass, past the callers'
    load-time error handling.
    """
    with open(path, "rb") as f:
        data = f.read()
    restored = serialization.from_bytes(model.params, data)
    want = jax.tree_util.tree_leaves(model.params)
    got = jax.tree_util.tree_leaves(restored)
    for w, g in zip(want, got):
        if jnp.shape(w) != jnp.shape(g):
            raise ValueError(
                f"checkpoint {path} has parameter shape {jnp.shape(g)} "
                f"where the model expects {jnp.shape(w)}; it was written "
                f"by an incompatible feature encoding"
            )
    model.params = restored
    return model


# --- external data sources --------------------------------------------------


def _seq_from_json_record(rec):
    """Pull (n, dx, seq) from a search or bnb JSON record, or None.

    Accepts the fast-search record shape (sequence is a list of dicts with
    layer and j) and a plain list of [layer, j] pairs. Skips records that are
    not marked prefix feasible when that flag is present.
    """
    if rec.get("prefix_feasible") is False:
        return None
    seq_raw = rec.get("sequence") or rec.get("seq")
    if not seq_raw:
        return None
    seq = []
    for item in seq_raw:
        if isinstance(item, dict):
            seq.append((int(item["layer"]), int(item["j"])))
        else:
            seq.append((int(item[0]), int(item[1])))
    n = int(rec.get("n", len(seq)))
    dx = float(rec.get("dx", DX))
    return {"n": n, "dx": dx, "seq": seq}


def load_external_records(paths):
    """Read imitation records from a list of JSON file paths that exist.

    Each file is either one record or a list of records. Missing files are
    skipped. Used to consume bnb_optima.json and fast-search JSON records.
    """
    records = []
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            with open(p) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        items = data if isinstance(data, list) else [data]
        for rec in items:
            parsed = _seq_from_json_record(rec)
            if parsed is not None:
                records.append(parsed)
    return records
