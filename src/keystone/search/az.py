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
- The feature vector is the occupancy grid over (layer, xidx) plus three
  scalars: placed fraction count/n, current rightmost cube edge over two, and
  remaining fraction (n - count)/n. The scalars carry the horizon n, so one
  network reads states from any n through the same input.
- Legality is never learned. The search supplies the legal or admissible
  action indices; the network only ranks them. Priors are masked to those
  indices and mixed with a uniform floor so every legal action keeps support.

Determinism: flax init is seeded by a PRNGKey, minibatch order is a seeded
numpy permutation, and the adapters are pure functions of the current params.
"""

import functools
import json
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

# Default mix toward the uniform prior. PUCT weights the exploration bonus by
# the prior, so an action with a zero prior gets a zero bonus and, before it is
# ever visited, a zero Q. It could then never be selected. The floor keeps
# every legal action reachable.
EPS_UNIFORM = 0.1

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

    @property
    def M(self) -> int:
        return self.max_layers * self.n_pos

    @property
    def F(self) -> int:
        return self.M + 3


def make_feature_spec(dx: float = DX, max_layers: int = MAX_LAYERS) -> FeatureSpec:
    """Feature spec for a fixed dx. n_pos and j_lo are read from the lattice."""
    ls = LatticeSpec(n_max=max_layers, dx=dx)
    return FeatureSpec(dx=dx, max_layers=max_layers, n_pos=ls.n_pos, j_lo=ls.j_lo)


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
    feat[fs.M + 0] = count / n
    feat[fs.M + 1] = edge / 2.0
    feat[fs.M + 2] = (n - count) / n
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
    """Two-hidden-layer MLP with a policy head and a value head.

    The policy head emits logits over the full action grid; the caller masks
    them to the legal set. The value head is a sigmoid in [0, 1], matching the
    search value scale where overhang == harmonic(n) maps near 0.5.
    """

    m: int
    hidden: int = 256

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(self.hidden)(x))
        x = nn.relu(nn.Dense(self.hidden)(x))
        logits = nn.Dense(self.m)(x)
        value = nn.sigmoid(nn.Dense(1)(x))
        return logits, value[..., 0]


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
        """Logits (B, M) and values (B,) for a batch of feature rows."""
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
    value head.
    """

    key: tuple
    n: int
    taken: int
    pol: dict
    value: float


def _value_target(overhang_val: float, n: int) -> float:
    """Final overhang normalized to [0, 1] by 2 * harmonic(n), then clipped."""
    v = overhang_val / (2.0 * harmonic(n))
    return float(min(max(v, 0.0), 1.0))


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
        ov = overhang(tuple(seq), fs.dx)
        vt = _value_target(ov, n)
        for k in range(len(seq)):
            prefix = tuple(sorted(seq[:k]))
            out.append(Sample(key=prefix, n=n, taken=action_index(fs, *seq[k]),
                              pol=None, value=vt))
    return out, dropped


def selfplay_samples(fs: FeatureSpec, search) -> list:
    """Distillation rows from a finished search tree.

    For every expanded, non-terminal node with visits, the policy target is the
    child visit distribution and the value target is the episode's best
    overhang normalized. This is the standard MCTS-visit distillation target.
    """
    out = []
    n = search.n
    vt = _value_target(
        0.0 if search.best_overhang == float("-inf") else search.best_overhang, n
    )
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
        out.append(Sample(key=node["key"], n=n, taken=None, pol=pol, value=vt))
    return out


def assemble_arrays(fs: FeatureSpec, samples, smooth: float = 0.9):
    """Stack samples into (feats, pols, vals, masks) arrays for training.

    Legal masks are computed once per stack size. Imitation policy targets are
    a smoothed one-hot: `smooth` on the taken action and the rest spread over
    all legal actions. Self-play targets are the visit distribution as given.
    """
    s = len(samples)
    feats = np.zeros((s, fs.F), dtype=np.float32)
    pols = np.zeros((s, fs.M), dtype=np.float32)
    vals = np.zeros((s,), dtype=np.float32)
    masks = np.zeros((s, fs.M), dtype=bool)

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
    return feats, pols, vals, masks


# --- training loop ----------------------------------------------------------


def _loss(net, params, f, p, v, m):
    """Masked policy cross-entropy plus value mean squared error."""
    logits, val = net.apply(params, f)
    masked = jnp.where(m, logits, jnp.float32(-1e9))
    logp = jax.nn.log_softmax(masked, axis=-1)
    pol_loss = -jnp.sum(p * logp, axis=-1)
    val_loss = (val - v) ** 2
    total = jnp.mean(pol_loss) + jnp.mean(val_loss)
    return total, (jnp.mean(pol_loss), jnp.mean(val_loss))


def train(model: AZModel, feats, pols, vals, masks, steps: int,
          batch: int = 256, lr: float = 3e-4, seed: int = 0):
    """Train the model in place. Returns the loss history as a list of dicts.

    Adam at the given learning rate. Minibatches are a seeded permutation of
    the rows, cycled until `steps` updates run. Deterministic for a fixed seed
    and dataset.
    """
    net = model.net
    opt = optax.adam(lr)
    opt_state = opt.init(model.params)

    grad_fn = jax.jit(jax.value_and_grad(
        functools.partial(_loss, net), has_aux=True))

    @jax.jit
    def step(params, opt_state, f, p, v, m):
        (total, (pl, vl)), grads = grad_fn(params, f, p, v, m)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, total, pl, vl

    fj = jnp.asarray(feats)
    pj = jnp.asarray(pols)
    vj = jnp.asarray(vals)
    mj = jnp.asarray(masks)

    s = feats.shape[0]
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
        params, opt_state, total, pl, vl = step(
            params, opt_state, fj[sj], pj[sj], vj[sj], mj[sj]
        )
        if t == 0 or (t + 1) % 100 == 0 or t == steps - 1:
            history.append({
                "step": t + 1,
                "loss": float(total),
                "policy": float(pl),
                "value": float(vl),
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
    """Load params into the model in place."""
    with open(path, "rb") as f:
        data = f.read()
    model.params = serialization.from_bytes(model.params, data)
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
