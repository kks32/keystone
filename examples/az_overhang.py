"""Learned priors and values for the maximum-overhang lattice search (v3).

This trains an AlphaZero-style distillation network that shapes the PUCT
search in keystone.search and measures whether it reaches a target overhang in
fewer simulations than the uniform-prior search. "AlphaZero-style
distillation" is the precise claim: PUCT with learned policy and value heads,
supervised on tree visit distributions. There are no independent stochastic
self-play games; data comes from search trees, and prior to v3 those trees
were fully deterministic. Certification is unchanged: the network only steers
exploration, and every reported best sequence is re-verified step by step on
the certified host pipeline.

v3 fixes five experimental-validity defects of the v2 protocol:

1. Seeds are real. The search RNG now drives PUCT tie-breaking jitter, and
   data-collection searches add root Dirichlet noise. v2's five "seeds" were
   five identical deterministic runs, so each v2 median was one measurement.
2. Exact-n reporting. best-at-any-depth conflated block counts (v2's "n=6
   reached 1.25" was a four-block design). v3 reports best_exact_n and
   best_at_most_n separately and labels every row; headline claims use
   exact-n.
3. Root features carry the horizon. The v2 empty-root feature vector was
   identical for every n, so the net could not condition on target size at
   the root. v3 adds n / N_MAX_GLOBAL and log-scale block count and retrains
   from scratch.
4. DAG-correct value targets. Subtree-max targets aggregate over all stored
   parents of each transposition, not the first one.
5. The auxiliary head regresses the lateral-reserve capacity
   min(|lam+|, |lam-|) (the quantity calibrated to predict physical
   survival) instead of the near-constant P4 equilibrium residual.
   --aux margin reproduces the old target for comparison.

Two modes:

  python examples/az_overhang.py --train
      Collect imitation and distillation data on the lattice, train two
      shared networks (per-state value, and value plus the reserve auxiliary
      head), and checkpoint them to out/search/az_params_v3*.msgpack.

  python examples/az_overhang.py --eval
      For n = 4 and n = 6 at dx = 1/12, report simulations-to-target as
      median and min..max over five genuinely different seeds under four
      conditions: uniform priors, learned priors, learned priors plus value
      head, and that plus the reserve auxiliary head. Targets are exact-n
      (a target for n cubes means best among states using exactly n cubes);
      the at-most-n crossing is reported alongside for comparison with v2.
      A transfer row at n = 5 is included with its leakage caveat stated.

The network covers several stack sizes through one canonical layer-major
action grid (see keystone.search.az). Training uses n in {4, 6}; n = 5 is
never trained on directly, but see the eval banner for what the n = 5 row
does and does not show. dx = 1/12 keeps the per-simulation cost low enough
to run on one CPU core; dx = 1/24 evaluation is out of scope here.
"""

import argparse
import math
import os
import statistics
import sys
import time

from keystone import (
    Tolerances,
    assemble,
    box_2d,
    build_assembly,
    solve_p0,
    solve_p4,
)
from keystone.search import Search
from keystone.search import az
from keystone.search import lattice as LT

DX = 1.0 / 12.0
MAX_LAYERS = 6
K_BATCH = 16
SEARCH_ITER = 50
MU = LT.MU

OUT_DIR = "out/search"
# v3 checkpoints. The value model uses per-state value targets only; the
# reserve model adds the auxiliary lateral-reserve head. Both back the eval
# conditions. --aux margin writes the legacy-target comparison checkpoint.
PARAMS_V3_VALUE = os.path.join(OUT_DIR, "az_params_v3_value.msgpack")
PARAMS_V3 = os.path.join(OUT_DIR, "az_params_v3.msgpack")
PARAMS_V3_MARGIN = os.path.join(OUT_DIR, "az_params_v3_margin.msgpack")

# Bootstrap simulation budgets per training size. Small: enough to surface a
# few good sequences and a tree to distill, not a full solve.
BOOTSTRAP_SIMS = {4: 1200, 5: 1500, 6: 1500}
TRAIN_NS = (4, 6)  # n = 5 held out for the transfer row

# Cap on distinct states given reserve targets per training run, bounding the
# extra certified solves at 2 * az.RESERVE_GRID * RESERVE_MAX_STATES.
RESERVE_MAX_STATES = 1200


# --- certified host re-verification (identical to the fast example) --------


def boxes_of(key, dx):
    """Pedestal plus one unit cube per placement (L, j), sorted (L, j)."""
    boxes = [box_2d(6, 1, -3, 0.5)]
    for (L, j) in sorted(key):
        boxes.append(box_2d(1, 1, j * dx, 1.5 + L))
    return boxes


def host_system(key, dx, tol):
    """Certified host equilibrium system for a state, matching the search."""
    n = len(key)
    asm = build_assembly(
        boxes_of(key, dx), mu=MU, tol=tol, dim=2,
        pad_blocks=n + 1, pad_patches=2 * n + 2, pad_verts=2,
    )
    return asm, assemble(asm, tol, cone="linear2d")


def verify_sequence(seq, dx, tol):
    """Replay a build order and solve P4 at each prefix. Returns the trace."""
    trace = []
    placed = []
    for (L, j) in seq:
        placed.append((L, j))
        _asm, system = host_system(tuple(placed), dx, tol)
        r = solve_p4(system, tol)
        trace.append((L, j, j * dx, r.margin, r.status))
    return trace


def print_verification(label, seq, dx, tol):
    """Re-verify a sequence on the host pipeline and print the verdict."""
    if not seq:
        print(f"  {label}: no sequence")
        return
    trace = verify_sequence(seq, dx, tol)
    ok = all(status == "feasible" for (_L, _j, _x, _m, status) in trace)
    edge = max(x + 0.5 for (_L, _j, x, _m, _s) in trace)
    print(f"  {label}: prefix_feasible={ok} rightmost_edge={edge:.4f} "
          f"steps={len(seq)}")


# --- data collection --------------------------------------------------------


def run_uniform_search(n, sims, seed, tol):
    """A uniform-prior search used to bootstrap imitation and distillation data.

    Data collection turns root Dirichlet noise on so different seeds explore
    genuinely different root actions. Evaluation searches never set it.
    """
    s = Search(n, DX, tol, seed=seed, batch=K_BATCH, search_iter=SEARCH_ITER,
               root_noise=True)
    s.run(sims)
    return s


def run_learned_search(n, sims, seed, tol, model):
    """A search steered by the model, used to collect fresh distillation data.

    Root noise on, as in run_uniform_search: collection explores, eval does not.
    """
    s = Search(
        n, DX, tol, seed=seed, batch=K_BATCH, search_iter=SEARCH_ITER,
        prior_fn=az.make_prior_fn(model, n),
        value_fn=az.make_value_fn(model, n),
        root_noise=True,
    )
    s.run(sims)
    return s


def attach_reserves(samples, tol, budget_state):
    """Attach lateral-reserve targets to rows that lack one, tracking cost.

    budget_state is a dict carrying solves, wall, and states_done across
    calls, so the total extra-solve cost of reserve supervision is bounded by
    RESERVE_MAX_STATES distinct states per training run and reported once.
    """
    pending = [smp for smp in samples if smp.reserve is None]
    room = RESERVE_MAX_STATES - budget_state["states_done"]
    if not pending or room <= 0:
        return
    n_solves, wall = az.attach_reserve_targets(
        pending, tol, dx=DX, solver_tol=1e-9, max_iter=SEARCH_ITER,
        max_states=room,
    )
    filled = sum(1 for smp in pending if smp.reserve is not None)
    budget_state["solves"] += n_solves
    budget_state["wall"] += wall
    budget_state["states_done"] += n_solves // (2 * az.RESERVE_GRID)
    print(f"  reserve targets: +{filled} rows, {n_solves} directional solves, "
          f"{wall:.1f}s (cumulative {budget_state['solves']} solves, "
          f"{budget_state['wall']:.1f}s)")


def _value_loss_trajectory(hist):
    """Compact value-loss trajectory string from a training history."""
    return " ".join(f"{h['step']}:{h['value']:.2e}" for h in hist)


def train_variant(name, fs, replay_seed, phase1, imit, selfplay_boot, tol,
                  model, margin_weight, robust, args, aux_mode="reserve",
                  budget_state=None):
    """Train one model through phase 1 and the self-play rounds.

    margin_weight scales the auxiliary loss; 0.0 disables the head's
    supervision. aux_mode picks the auxiliary target: "reserve" (the lateral
    reserve capacity, the v3 default) or "margin" (the legacy equilibrium
    residual, kept for comparison). robust switches the value target to the
    knife-edge-penalized variant. Returns the phase-1 and last self-play
    histories for reporting.
    """
    needs_reserve = margin_weight > 0.0 and aux_mode == "reserve"
    if needs_reserve:
        attach_reserves(phase1, tol, budget_state)
    f1, p1, v1, m1, mg1, mgm1 = az.assemble_arrays(fs, phase1,
                                                   aux_mode=aux_mode)
    print(f"  [{name}] phase-1 value-target std={float(v1.std()):.4f} "
          f"aux[{aux_mode}] rows={int(mgm1.sum())}/{len(phase1)} "
          f"target std={float(mg1[mgm1].std()) if mgm1.any() else 0.0:.4f}")
    h1 = az.train(model, f1, p1, v1, m1, mg1, mgm1,
                  steps=args.imit_steps, batch=256, lr=3e-4,
                  margin_weight=margin_weight, seed=0)
    print(f"  [{name}] phase-1 train: {args.imit_steps} steps "
          f"loss {h1[0]['loss']:.4f} -> {h1[-1]['loss']:.4f} "
          f"(policy {h1[-1]['policy']:.4f} value {h1[-1]['value']:.2e} "
          f"aux {h1[-1]['margin']:.2e})")
    print(f"  [{name}] phase-1 value loss: {_value_loss_trajectory(h1)}")

    # Self-play rounds: run the learned search under this model's own hooks,
    # distil visit distributions with per-state DAG subtree-max value targets,
    # add to the replay buffer, and retrain.
    replay = list(imit) + list(selfplay_boot)
    h2 = h1
    for r in range(args.sp_rounds):
        for n in TRAIN_NS:
            s = run_learned_search(n, args.sp_sims, seed=100 + r, tol=tol,
                                   model=model)
            new = az.selfplay_samples(fs, s, robust=robust,
                                      robust_penalty=args.robust_penalty)
            replay += new
            print(f"  [{name}] selfplay round {r + 1} n={n}: "
                  f"best_exact={s.exact_overhang(n):.4f} "
                  f"best_any={s.best_overhang:.4f} rows+={len(new)} "
                  f"buffer={len(replay)}")
        if needs_reserve:
            attach_reserves(replay, tol, budget_state)
        f2, p2, v2, m2, mg2, mgm2 = az.assemble_arrays(fs, replay,
                                                       aux_mode=aux_mode)
        h2 = az.train(model, f2, p2, v2, m2, mg2, mgm2,
                      steps=args.sp_steps, batch=256, lr=3e-4,
                      margin_weight=margin_weight, seed=r + 1)
        print(f"  [{name}] selfplay round {r + 1} train: {args.sp_steps} steps "
              f"loss {h2[0]['loss']:.4f} -> {h2[-1]['loss']:.4f} "
              f"(value {h2[-1]['value']:.2e} aux {h2[-1]['margin']:.2e})")
    print(f"  [{name}] final self-play value loss: {_value_loss_trajectory(h2)}")
    return h1, h2


def do_train(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    tol = Tolerances()
    fs = az.make_feature_spec(dx=DX, max_layers=MAX_LAYERS)
    print(f"train v3: dx=1/{round(1 / DX)} max_layers={MAX_LAYERS} "
          f"M={fs.M} F={fs.F} train_ns={list(TRAIN_NS)} (n=5 held out)")
    print(f"  per-state value targets (suffix/DAG-subtree-max), aux="
          f"{args.aux} weight={args.margin_weight}, robust={bool(args.robust)}")
    print(f"  root features carry n / {az.N_MAX_GLOBAL} and log-scale count; "
          f"data-collection searches use tie jitter + root Dirichlet noise")

    # Imitation records: the known-good seeds, any external optima, and the
    # best sequence of a uniform bootstrap search per training size.
    records = list(az.KNOWN_SEQUENCES_DX12)
    ext_paths = [os.path.join(OUT_DIR, "bnb_optima.json")]
    ext_paths += [os.path.join(OUT_DIR, f"fast_best_n{n}_dx12.json")
                  for n in (4, 5, 6)]
    ext = az.load_external_records(ext_paths)
    records += ext
    print(f"  external records loaded: {len(ext)} "
          f"(bnb_optima.json present: "
          f"{os.path.exists(os.path.join(OUT_DIR, 'bnb_optima.json'))})")

    # Hold-out hygiene. External optima files carry records labeled n=5 (and
    # n=6 rows whose sequence is a 4-cube design). Now that the root features
    # carry the horizon, an n=5-labeled imitation row would train exactly the
    # conditioning the transfer row claims is held out, so every record whose
    # n is not a training size is dropped. Records whose sequence uses fewer
    # cubes than n are dropped too: imitating a 4-cube build as an "n=6"
    # episode is the block-count conflation this revision removes.
    before = len(records)
    records = [r for r in records
               if r["n"] in TRAIN_NS and len(r["seq"]) == r["n"]]
    print(f"  hold-out filter: kept {len(records)}/{before} records "
          f"(train sizes {list(TRAIN_NS)}, exact-length sequences only)")

    # Bootstrap uniform searches. These are the shared data source for both
    # variants: same searches, same seeds, so the only difference downstream is
    # the training objective.
    selfplay_boot = []
    t0 = time.perf_counter()
    for n in TRAIN_NS:
        sims = BOOTSTRAP_SIMS[n]
        s = run_uniform_search(n, sims, seed=n, tol=tol)
        # Imitate the best exact-n build, not the best-at-any-depth design;
        # a shorter design would smuggle the block-count conflation back in.
        seq = s.exact_sequence(n)
        if seq:
            records.append({"n": n, "dx": DX, "seq": seq})
        sp = az.selfplay_samples(fs, s, robust=bool(args.robust),
                                 robust_penalty=args.robust_penalty)
        selfplay_boot += sp
        print(f"  bootstrap n={n}: sims={s.sims_done} "
              f"best_exact={s.exact_overhang(n):.4f} "
              f"best_any={s.best_overhang:.4f} selfplay_rows={len(sp)} "
              f"({time.perf_counter() - t0:.0f}s elapsed)")

    imit, dropped = az.imitation_samples(fs, records)
    print(f"  imitation rows: {len(imit)} (dropped {dropped} records: "
          f"wrong dx or lattice-illegal)")
    print(f"  bootstrap distillation rows: {len(selfplay_boot)}")

    phase1 = list(imit) + list(selfplay_boot)
    print(f"  phase-1 dataset: {len(phase1)} rows")

    # Reserve-solve budget, shared across the aux variant's attach calls.
    budget = {"solves": 0, "wall": 0.0, "states_done": 0}

    # Value model: per-state value targets, no auxiliary supervision.
    value_model = az.AZModel(fs, init_seed=args.init_seed)
    train_variant("value", fs, 0, phase1, imit, selfplay_boot, tol,
                  value_model, margin_weight=0.0, robust=bool(args.robust),
                  args=args)
    az.save_params(value_model, PARAMS_V3_VALUE)
    print(f"  wrote {PARAMS_V3_VALUE}")

    # Aux model: same per-state value targets plus the auxiliary head. The
    # default target is the lateral-reserve capacity computed by the batched
    # reserve kernel on buffer states (cost reported); --aux margin reproduces
    # the legacy equilibrium-residual target for comparison.
    aux_name = "reserve-aux" if args.aux == "reserve" else "margin-aux(legacy)"
    aux_path = PARAMS_V3 if args.aux == "reserve" else PARAMS_V3_MARGIN
    aux_model = az.AZModel(fs, init_seed=args.init_seed)
    train_variant(aux_name, fs, 0, phase1, imit, selfplay_boot, tol,
                  aux_model, margin_weight=args.margin_weight,
                  robust=bool(args.robust), args=args, aux_mode=args.aux,
                  budget_state=budget)
    az.save_params(aux_model, aux_path)
    print(f"  wrote {aux_path}")

    if args.aux == "reserve":
        print(f"  reserve supervision cost: {budget['solves']} directional "
              f"certified solves, {budget['wall']:.1f}s wall "
              f"(cap {RESERVE_MAX_STATES} states, grid {az.RESERVE_GRID})")
    print(f"  total train wall: {time.perf_counter() - t0:.0f}s")
    return 0


# --- evaluation -------------------------------------------------------------


def sims_to_targets(n, targets, cap, seed, tol, prior_fn=None, value_fn=None):
    """Simulations until the exact-n and at-most-n bests reach each target.

    Runs one batched iteration at a time so the first crossing is caught at
    K-simulation resolution, until both crossing sets are complete or the cap
    is hit. Evaluation searches use the seeded tie jitter only: no root noise,
    no sampling temperature.

    Returns (reached_exact, reached_atmost, best_exact, best_atmost, search).
    reached_exact[t] is the sim count at which the best overhang among states
    using exactly n cubes first reached t (None if never); reached_atmost[t]
    is the same for the best at any block count <= n, the old v2 semantics.
    """
    s = Search(n, DX, tol, seed=seed, batch=K_BATCH, search_iter=SEARCH_ITER,
               prior_fn=prior_fn, value_fn=value_fn)
    n_iter = max(1, math.ceil(cap / K_BATCH))
    reached_exact = {t: None for t in targets}
    reached_atmost = {t: None for t in targets}
    it = 0
    for it in range(n_iter):
        s.run_iteration()
        sims = (it + 1) * K_BATCH
        be = s.exact_overhang(n)
        ba = s.best_overhang
        for t in targets:
            if reached_exact[t] is None and be >= t - 1e-9:
                reached_exact[t] = sims
            if reached_atmost[t] is None and ba >= t - 1e-9:
                reached_atmost[t] = sims
        if (all(v is not None for v in reached_exact.values())
                and all(v is not None for v in reached_atmost.values())):
            break
        if s.root["solved"]:
            break  # tree exhausted; nothing further can be reached
    s.sims_done = (it + 1) * K_BATCH
    best_exact = s.exact_overhang(n)
    best_exact = 0.0 if best_exact == float("-inf") else best_exact
    best_atmost = s.best_overhang if s.best_overhang != float("-inf") else 0.0
    return reached_exact, reached_atmost, best_exact, best_atmost, s


def eval_condition(n, targets, cap, seeds, tol, make_hooks):
    """Run one condition over seeds.

    make_hooks(n) -> (prior_fn, value_fn). Returns a dict of aligned per-seed
    lists: reached_exact, reached_atmost, best_exact, best_atmost, searches.
    """
    out = {"reached_exact": [], "reached_atmost": [], "best_exact": [],
           "best_atmost": [], "searches": []}
    for seed in seeds:
        prior_fn, value_fn = make_hooks(n)
        re_, ra, be, ba, s = sims_to_targets(n, targets, cap, seed, tol,
                                             prior_fn=prior_fn,
                                             value_fn=value_fn)
        out["reached_exact"].append(re_)
        out["reached_atmost"].append(ra)
        out["best_exact"].append(be)
        out["best_atmost"].append(ba)
        out["searches"].append(s)
    return out


def summarize_target(per_seed, target, cap):
    """(median, min, max, reached-count) of sims-to-target across seeds.

    Not-reached counts as cap in the median and the spread, so a cell with
    reached < len(seeds) understates the true cost; the reached count is
    printed next to it.
    """
    vals = [d[target] for d in per_seed]
    filled = [v if v is not None else cap for v in vals]
    reached = sum(1 for v in vals if v is not None)
    return statistics.median(filled), min(filled), max(filled), reached


def spread(vals):
    """(median, min, max) of a list of floats."""
    return statistics.median(vals), min(vals), max(vals)


def do_eval(args):
    tol = Tolerances()
    fs = az.make_feature_spec(dx=DX, max_layers=MAX_LAYERS)
    seeds = list(range(args.seeds))

    # Load the two v3 models. The value model uses per-state value targets
    # only; the reserve model adds the auxiliary lateral-reserve head during
    # training. The auxiliary head never feeds the search, so any difference
    # between the two is the auxiliary task reshaping the shared prior and
    # value heads.
    value_model = az.AZModel(fs, init_seed=args.init_seed)
    az.load_params(value_model, PARAMS_V3_VALUE)
    aux_model = az.AZModel(fs, init_seed=args.init_seed)
    az.load_params(aux_model, PARAMS_V3)

    # Conditions. Each maps a size n to (prior_fn, value_fn).
    def uniform_hooks(n):
        return None, None

    def prior_only_hooks(n):
        return az.make_prior_fn(value_model, n), None

    def prior_value_hooks(n):
        return az.make_prior_fn(value_model, n), az.make_value_fn(value_model, n)

    def prior_value_reserve_hooks(n):
        return (az.make_prior_fn(aux_model, n),
                az.make_value_fn(aux_model, n))

    conditions = [
        ("uniform", uniform_hooks),
        ("prior", prior_only_hooks),
        ("prior+value", prior_value_hooks),
        ("prior+value+reserve-aux", prior_value_reserve_hooks),
    ]

    # Targets per size, on the dx = 1/12 grid (edges are j/12 + 0.5). All
    # headline rows are exact-n: a target for n cubes means best among states
    # using exactly n cubes. The at-most-n crossing (v2's semantics) is
    # printed alongside for comparison.
    # n = 4: 7/6 (the task target) and 1.25, the bnb-certified exact-4
    #   optimum and current best-known.
    # n = 6: 1.25 and 4/3, both beating 1.225 with exactly six cubes. v2's
    #   "n=6 reached 1.25" was a four-block design; these rows cannot be
    #   satisfied that way.
    # n = 5 (transfer): 1.0833 (j = 7).
    tasks = [
        (4, [7.0 / 6.0, 1.25], args.cap_n4, False),
        (6, [1.25, 4.0 / 3.0], args.cap_n6, False),
        (5, [1.0 + 1.0 / 12.0], args.cap_n5, True),
    ]

    print(f"eval v3: seeds={seeds} caps n4={args.cap_n4} n6={args.cap_n6} "
          f"n5={args.cap_n5}")
    print(f"grid dx=1/{round(1 / DX)}; n=4 exact-4 optimum 1.25 is "
          f"bnb-certified; headline rows are exact-n")
    print("seeds are genuinely different runs (seeded PUCT tie jitter); "
          "eval uses no root noise and argmax moves")
    print("n=5 transfer caveat: n=5 was never a search size during training, "
          "but training data included n=4 and n=6 sequences over the same "
          "shared action grid, and 4- and 5-cube prefixes of n=6 episodes "
          "are in the buffer. The n=5 row therefore shows generalization "
          "across the horizon input, not performance on unseen states.")
    print("")
    header = (f"{'n':>3} {'target':>8} {'scope':>8} {'condition':>24} "
              f"{'med_sims':>9} {'min..max':>13} {'reached':>8} "
              f"{'med_best':>9} {'best min..max':>15}")
    print(header)
    print("-" * len(header))

    verify_jobs = []  # (label, n, sequence, best) for host re-verification
    for (n, targets, cap, transfer) in tasks:
        tag = "transfer" if transfer else "target"
        for (name, hooks) in conditions:
            res = eval_condition(n, targets, cap, seeds, tol, hooks)
            med_be, lo_be, hi_be = spread(res["best_exact"])
            med_ba, lo_ba, hi_ba = spread(res["best_atmost"])
            for t in targets:
                for scope, per_seed, mb, lo_b, hi_b in (
                    ("exact-n", res["reached_exact"], med_be, lo_be, hi_be),
                    ("<=n", res["reached_atmost"], med_ba, lo_ba, hi_ba),
                ):
                    med, lo, hi, reached = summarize_target(per_seed, t, cap)
                    print(f"{n:>3} {t:>8.4f} {scope:>8} {name:>24} "
                          f"{med:>9.0f} {f'{lo:.0f}..{hi:.0f}':>13} "
                          f"{reached:>4}/{len(seeds):<3} {mb:>9.4f} "
                          f"{f'{lo_b:.4f}..{hi_b:.4f}':>15}")
            # Re-verify the exact-n best sequence of the best seed.
            bests = res["best_exact"]
            pick = res["searches"][
                int(max(range(len(bests)), key=lambda i: bests[i]))]
            verify_jobs.append((f"n={n} {name} {tag} exact-{n}", n,
                                pick.exact_sequence(n), pick.exact_overhang(n)))
        print("")

    print("certified host re-verification of reported exact-n best sequences:")
    for (label, n, seq, best) in verify_jobs:
        best_s = "none" if best == float("-inf") else f"{best:.4f}"
        print(f"  [{label}] search_best={best_s}")
        print_verification("    host", seq, DX, tol)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Learned priors and values for overhang lattice search.")
    parser.add_argument("--train", action="store_true", help="train the model")
    parser.add_argument("--eval", action="store_true",
                        help="run the sims-to-target comparison")
    parser.add_argument("--init_seed", type=int, default=0,
                        help="flax parameter init seed")
    parser.add_argument("--imit_steps", type=int, default=3000,
                        help="imitation training steps")
    parser.add_argument("--sp_rounds", type=int, default=0,
                        help="self-play rounds after imitation")
    parser.add_argument("--sp_sims", type=int, default=1200,
                        help="simulations per self-play search")
    parser.add_argument("--sp_steps", type=int, default=1500,
                        help="training steps per self-play round")
    parser.add_argument("--seeds", type=int, default=5,
                        help="number of eval seeds")
    parser.add_argument("--margin_weight", type=float, default=0.5,
                        help="weight on the auxiliary loss")
    parser.add_argument("--aux", choices=["reserve", "margin"],
                        default="reserve",
                        help="auxiliary target: lateral reserve (v3 default) "
                             "or the legacy P4 residual margin")
    parser.add_argument("--robust", action="store_true",
                        help="use the knife-edge-penalized value target")
    parser.add_argument("--robust_penalty", type=float, default=0.05,
                        help="value penalty for knife-edge best paths")
    parser.add_argument("--cap_n4", type=int, default=3000)
    parser.add_argument("--cap_n5", type=int, default=2500)
    parser.add_argument("--cap_n6", type=int, default=4000)
    args = parser.parse_args()

    if not args.train and not args.eval:
        parser.error("pass --train, --eval, or both")
    rc = 0
    if args.train:
        rc = do_train(args)
    if args.eval:
        rc = do_eval(args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
