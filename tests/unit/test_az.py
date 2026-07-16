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
    la, va = a.forward(x)
    lb, vb = b.forward(x)
    assert np.array_equal(np.asarray(la), np.asarray(lb))
    assert np.array_equal(np.asarray(va), np.asarray(vb))
    # A different seed gives different parameters.
    c = az.AZModel(fs, init_seed=1)
    lc, _ = c.forward(x)
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
    feats, pols, vals, masks = az.assemble_arrays(fs, samples)
    for i, smp in enumerate(samples):
        assert masks[i, smp.taken]


def test_imitation_training_decreases_loss():
    from keystone.search import az

    fs = _feature_spec()
    samples, _dropped = az.imitation_samples(fs, az.KNOWN_SEQUENCES_DX12)
    assert len(samples) > 0
    feats, pols, vals, masks = az.assemble_arrays(fs, samples)
    # Policy targets are proper distributions over the legal set.
    assert np.allclose(pols.sum(axis=1), 1.0, atol=1e-5)
    assert np.all((vals >= 0.0) & (vals <= 1.0))

    model = az.AZModel(fs, init_seed=0)
    history = az.train(model, feats, pols, vals, masks, steps=50, batch=256, seed=0)
    assert history[-1]["loss"] < history[0]["loss"]
