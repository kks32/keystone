"""Unit tests for the learned-prior search stage (keystone.search.az).

These check the encoding, the network, the search adapters, and that the
learned-search hooks default to the exact uniform-prior search. Kept small so
the whole file runs in well under two minutes on one CPU core.

The flax and optax imports are done inside the test bodies, not at module top.
pytest imports every test module at collection, before any test runs. A
top-level flax import would then load before the geometry property tests and
shift the process numerics enough to tip a borderline tolerance in one
numerically fragile 3D invariant test. Deferring the import keeps that test
untouched, since these unit tests run after the property tests.
"""

import numpy as np


def _feature_spec():
    from keystone.search import az

    return az.make_feature_spec(dx=1.0 / 12.0, max_layers=6)


def test_encoding_shapes_and_action_index():
    from keystone.search import az

    fs = _feature_spec()
    assert fs.n_pos == 85
    assert fs.M == 6 * 85
    assert fs.F == fs.M + 3

    # A layer-major index matches the lattice convention.
    assert az.action_index(fs, 0, fs.j_lo) == 0
    assert az.action_index(fs, 1, fs.j_lo) == fs.n_pos

    key = ((0, -1), (1, -6))
    feat = az.encode_state(fs, key, n=4)
    assert feat.shape == (fs.F,)
    # Occupancy marks exactly the placed cells.
    assert feat[az.action_index(fs, 0, -1)] == 1.0
    assert feat[az.action_index(fs, 1, -6)] == 1.0
    assert feat[: fs.M].sum() == 2.0
    # Scalars: count fraction, edge over two, remaining fraction.
    assert np.isclose(feat[fs.M + 0], 2.0 / 4.0)
    assert np.isclose(feat[fs.M + 2], 2.0 / 4.0)


def test_legality_masking():
    from keystone.search import az

    fs = _feature_spec()
    # Empty state at n = 4: a layer-0 placement over the pedestal top is legal;
    # a layer-2 placement with nothing under it is not.
    masks = az.legal_masks(fs, [()], n=4)
    assert masks.shape == (1, fs.M)
    assert masks[0, az.action_index(fs, 0, -1)]  # over pedestal, supported
    assert not masks[0, az.action_index(fs, 2, 0)]  # no support below
    # Layers at or above n are never legal (they lie past the size-n grid).
    assert not masks[0, az.action_index(fs, 4, 0)]
    assert not masks[0, az.action_index(fs, 5, 0)]


def test_net_forward_determinism():
    from keystone.search import az

    fs = _feature_spec()
    a = az.AZModel(fs, init_seed=0)
    b = az.AZModel(fs, init_seed=0)
    x = az.encode_state(fs, ((0, -1),), n=4)[None, :]
    la, va, ma = a.forward(x)
    lb, vb, mb = b.forward(x)
    assert np.array_equal(np.asarray(la), np.asarray(lb))
    assert np.array_equal(np.asarray(va), np.asarray(vb))
    assert np.array_equal(np.asarray(ma), np.asarray(mb))
    # A different seed gives different parameters.
    c = az.AZModel(fs, init_seed=1)
    lc, _, _ = c.forward(x)
    assert not np.array_equal(np.asarray(la), np.asarray(lc))


def test_prior_fn_epsilon_mix_sums_to_one():
    from keystone.search import az

    fs = _feature_spec()
    model = az.AZModel(fs, init_seed=0, eps=0.1)
    prior_fn = az.make_prior_fn(model, n=4)
    idx = [az.action_index(fs, 0, -1), az.action_index(fs, 0, 0),
           az.action_index(fs, 0, 5)]
    p = prior_fn(((0, -2),), idx)
    assert p.shape == (3,)
    assert np.isclose(p.sum(), 1.0)
    # The uniform floor keeps every legal action strictly positive.
    assert np.all(p > 0.0)
    assert np.all(p >= 0.1 / len(idx) - 1e-9)


def test_value_fn_in_unit_interval():
    from keystone.search import az

    fs = _feature_spec()
    model = az.AZModel(fs, init_seed=0)
    value_fn = az.make_value_fn(model, n=4)
    v = value_fn(((0, -1), (1, -6)))
    assert 0.0 <= v <= 1.0


def test_none_hooks_reproduce_uniform_search():
    # The learned-search hooks default to None, which must run the same search
    # as before: same best overhang and same best sequence, bit for bit.
    from keystone.geometry.tolerances import Tolerances
    from keystone.search.mcts import Search

    tol = Tolerances()
    kw = dict(n=3, dx=1.0 / 6.0, tol=tol, seed=0, batch=8, search_iter=40)

    base = Search(**kw)
    base.run(100)

    hooked = Search(prior_fn=None, value_fn=None, **kw)
    hooked.run(100)

    assert hooked.best_overhang == base.best_overhang
    assert hooked.best_sequence() == base.best_sequence()


def test_imitation_dropped_illegal_records():
    from keystone.search import az

    # Records from the naive host script can violate lattice legality and must
    # be dropped, not imitated. At least one known sequence is lattice-legal.
    fs = _feature_spec()
    samples, dropped = az.imitation_samples(fs, az.KNOWN_SEQUENCES_DX12)
    assert len(samples) > 0
    assert dropped >= 1
    # Every imitated prefix is lattice-legal, so its target sits on a legal cell.
    feats, pols, vals, masks, margins, margin_mask = az.assemble_arrays(fs, samples)
    for i, smp in enumerate(samples):
        assert masks[i, smp.taken]
    # Imitation prefixes carry no certified margin.
    assert not margin_mask.any()


def test_imitation_training_decreases_loss():
    from keystone.search import az

    fs = _feature_spec()
    samples, _dropped = az.imitation_samples(fs, az.KNOWN_SEQUENCES_DX12)
    assert len(samples) > 0
    feats, pols, vals, masks, margins, margin_mask = az.assemble_arrays(fs, samples)
    # Policy targets are proper distributions over the legal set.
    assert np.allclose(pols.sum(axis=1), 1.0, atol=1e-5)
    assert np.all((vals >= 0.0) & (vals <= 1.0))

    model = az.AZModel(fs, init_seed=0)
    history = az.train(model, feats, pols, vals, masks, margins, margin_mask,
                       steps=50, batch=256, seed=0)
    assert history[-1]["loss"] < history[0]["loss"]


def test_suffix_max_value_targets():
    from keystone.search import az

    fs = _feature_spec()
    dx = 1.0 / 12.0
    # A hand-built episode. Overhang is the rightmost cube edge, so it is
    # non-decreasing along a build order; the suffix-max at every prefix equals
    # the maximum overhang over the remaining prefixes. The helper must
    # reproduce that suffix maximum exactly, even when the last cube placed is
    # not the rightmost (here the final placement at j = 0 does not extend the
    # edge, so the suffix maximum sits at an interior prefix).
    seq = [(0, 6), (1, 3), (2, 0)]
    got = az._suffix_max_overhangs(seq, dx)
    assert len(got) == len(seq)
    ovs = [az.overhang(tuple(seq[:j]), dx) for j in range(len(seq) + 1)]
    want = [max(ovs[k:]) for k in range(len(seq))]
    assert np.allclose(got, want)
    # The rightmost edge is set by the first cube, 6/12 + 0.5 = 1.0, and no
    # later cube exceeds it, so every prefix suffix-max is 1.0.
    assert np.allclose(got, 1.0)

    # imitation_samples turns per-prefix suffix-maxima into value targets on a
    # lattice-legal record. This known n = 4 order is legal on the grid.
    lseq = [(0, -1), (1, -6), (2, -3), (1, 8)]
    rec = {"n": 4, "dx": dx, "seq": lseq}
    samples, dropped = az.imitation_samples(fs, [rec])
    assert dropped == 0 and len(samples) == len(lseq)
    lwant = az._suffix_max_overhangs(lseq, dx)
    for smp, ov in zip(samples, lwant):
        assert np.isclose(smp.value, az._value_target(ov, 4))
        assert smp.margin is None


def test_selfplay_subtree_max_gives_per_state_credit():
    # The old target was one constant per episode; the value head learned
    # nothing. Self-play now labels each node with its own subtree-max overhang.
    from keystone.geometry.tolerances import Tolerances
    from keystone.search import az
    from keystone.search.mcts import Search

    dx = 1.0 / 6.0
    fs = az.make_feature_spec(dx=dx, max_layers=6)
    tol = Tolerances()
    s = Search(n=3, dx=dx, tol=tol, seed=0, batch=8, search_iter=40)
    s.run(240)

    samples = az.selfplay_samples(fs, s)
    assert len(samples) > 0
    # Certified margins are threaded onto the solved states.
    assert any(smp.margin is not None for smp in samples)
    # Per-state credit: the value targets are not one shared constant.
    values = np.array([smp.value for smp in samples])
    assert values.std() > 0.0
    # Correctness: each value equals its own subtree-max overhang normalized.
    submax, _knife = az._subtree_targets(s, robust=False)
    for smp in samples:
        best = submax[smp.key]
        want = az._value_target(0.0 if best == float("-inf") else best, 3)
        assert np.isclose(smp.value, want)


def test_robust_value_knife_edge_mechanism():
    # The robustness variant flags nodes whose best path passes through a
    # near-boundary (knife-edge) state. A small stub tree with known margins
    # exercises the mechanism deterministically. Default off leaves it clear.
    import types

    from keystone.search import az

    dx = 1.0 / 6.0
    tol_feas = 1e-8
    ka = ((0, 3),)
    kb = ((0, 3), (1, 6))
    tree = {
        (): {"key": (), "parent": None},
        ka: {"key": ka, "parent": ()},
        kb: {"key": kb, "parent": ka},
    }
    # kb is knife-edge (margin within a factor of ten of tol_feas); ka is safe.
    margins = {ka: 1e-11, kb: 5e-9}

    class _Stub:
        def __init__(self):
            self.tree = tree
            self.dx = dx
            self.tol = types.SimpleNamespace(tol_feas=tol_feas)

        def margin_of(self, key):
            return margins.get(key)

    stub = _Stub()
    submax, knife = az._subtree_targets(stub, robust=True)
    # The best overhang, set by kb, is 6/6 + 0.5 = 1.5 and is shared up the path.
    assert np.isclose(submax[()], 1.5)
    assert np.isclose(submax[ka], 1.5)
    # The best path from every node runs through the knife-edge kb.
    assert knife[()] and knife[ka] and knife[kb]
    # Robust off flags nothing.
    _sub2, knife_off = az._subtree_targets(stub, robust=False)
    assert not any(knife_off.values())


def test_robust_value_never_raises_targets():
    # On a real search, the penalty is one-directional: no robust target ever
    # exceeds its plain counterpart, whether or not any knife edge is present.
    from keystone.geometry.tolerances import Tolerances
    from keystone.search import az
    from keystone.search.mcts import Search

    dx = 1.0 / 6.0
    fs = az.make_feature_spec(dx=dx, max_layers=6)
    tol = Tolerances()
    s = Search(n=3, dx=dx, tol=tol, seed=0, batch=8, search_iter=40)
    s.run(240)

    plain = {smp.key: smp.value for smp in az.selfplay_samples(fs, s)}
    robust = {smp.key: smp.value
              for smp in az.selfplay_samples(fs, s, robust=True,
                                             robust_penalty=0.05)}
    assert all(robust[k] <= plain[k] + 1e-9 for k in plain)


def test_margin_normalization_bounds():
    from keystone.search import az

    # A margin at or below the floor maps to 0; a large margin saturates at 1.
    assert az._margin_target(0.0) == 0.0
    assert np.isclose(az._margin_target(1.0), 1.0)
    assert az._margin_target(1e6) == 1.0
    # A mid-range margin lands inside the window: log10(1e-8) = -8 over [-12, 0]
    # is (−8 + 12) / 12 = 1/3.
    assert np.isclose(az._margin_target(1e-8), 1.0 / 3.0, atol=1e-3)
    # Monotone and bounded across the observed range.
    xs = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0]
    ys = [az._margin_target(x) for x in xs]
    assert all(0.0 <= y <= 1.0 for y in ys)
    assert all(ys[i] <= ys[i + 1] + 1e-9 for i in range(len(ys) - 1))


def test_third_head_shape_and_bounds():
    from keystone.search import az

    fs = _feature_spec()
    model = az.AZModel(fs, init_seed=0)
    keys = [((0, -1),), ((0, -1), (1, -6)), ()]
    x = np.stack([az.encode_state(fs, k, n=4) for k in keys])
    out = model.forward(x)
    # Three heads now: logits, value, margin.
    assert len(out) == 3
    logits, value, margin = out
    logits = np.asarray(logits)
    value = np.asarray(value)
    margin = np.asarray(margin)
    assert logits.shape == (3, fs.M)
    assert value.shape == (3,)
    assert margin.shape == (3,)
    assert np.all((margin >= 0.0) & (margin <= 1.0))


def test_value_fn_interface_unchanged():
    # The value_fn adapter must still take a state key and return a scalar in
    # [0, 1] read from the value head, unaffected by the new margin head.
    from keystone.search import az

    fs = _feature_spec()
    model = az.AZModel(fs, init_seed=0)
    value_fn = az.make_value_fn(model, n=4)
    v = value_fn(((0, -1), (1, -6)))
    assert isinstance(v, float)
    assert 0.0 <= v <= 1.0
    # It reads output [1] of forward, the value head, exactly as before.
    feat = az.encode_state(fs, ((0, -1), (1, -6)), n=4)[None, :]
    assert np.isclose(v, float(np.asarray(model.forward(feat)[1])[0]))


def test_combined_loss_decreases_with_margin_supervision():
    from keystone.search import az

    fs = _feature_spec()
    samples, _dropped = az.imitation_samples(fs, az.KNOWN_SEQUENCES_DX12)
    assert len(samples) > 0
    # Attach synthetic certified margins so the margin head has a target on
    # every row. Values span the normalized window.
    for i, smp in enumerate(samples):
        smp.margin = 10.0 ** (-(float(i % 10) + 1.0))
    feats, pols, vals, masks, margins, margin_mask = az.assemble_arrays(fs, samples)
    assert margin_mask.all()

    model = az.AZModel(fs, init_seed=0)
    history = az.train(model, feats, pols, vals, masks, margins, margin_mask,
                       steps=50, batch=256, lr=3e-4, margin_weight=0.5, seed=0)
    # The combined objective and its margin component both fall.
    assert history[-1]["loss"] < history[0]["loss"]
    assert history[-1]["margin"] < history[0]["margin"]
    assert "margin" in history[0]
