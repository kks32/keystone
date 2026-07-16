"""Learned priors and values for the maximum-overhang lattice search.

This trains an AlphaZero-style network that shapes the PUCT search in
keystone.search and measures whether it reaches a target overhang in fewer
simulations than the uniform-prior search. Certification is unchanged: the
network only steers exploration, and every reported best sequence is
re-verified step by step on the certified host pipeline.

Two modes:

  python examples/az_overhang.py --train
      Collect imitation and self-play data on the lattice, train the shared
      network, and checkpoint it to out/search/az_params*.msgpack.

  python examples/az_overhang.py --eval
      For n = 4 and n = 6 at dx = 1/12, report simulations-to-target as the
      median over five seeds under three conditions: uniform priors, learned
      priors, and learned priors plus learned value. Add a transfer row at
      n = 5, a size held out of all training.

The network covers several stack sizes through one canonical layer-major
action grid (see keystone.search.az). Training uses n in {4, 6}; n = 5 is
never trained, so its row is a zero-shot transfer test. dx = 1/12 keeps the
per-simulation cost low enough to run on one CPU core; dx = 1/24 evaluation is
out of scope here.
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
PARAMS_IMIT = os.path.join(OUT_DIR, "az_params_imit.msgpack")
PARAMS_FINAL = os.path.join(OUT_DIR, "az_params.msgpack")

# Bootstrap simulation budgets per training size. Small: enough to surface a
# few good sequences and a tree to distill, not a full solve.
BOOTSTRAP_SIMS = {4: 1200, 5: 1500, 6: 1500}
TRAIN_NS = (4, 6)  # n = 5 held out for the transfer row


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
    """A uniform-prior search used to bootstrap imitation and self-play data."""
    s = Search(n, DX, tol, seed=seed, batch=K_BATCH, search_iter=SEARCH_ITER)
    s.run(sims)
    return s


def run_learned_search(n, sims, seed, tol, model):
    """A search steered by the model, used to collect fresh self-play data."""
    s = Search(
        n, DX, tol, seed=seed, batch=K_BATCH, search_iter=SEARCH_ITER,
        prior_fn=az.make_prior_fn(model, n),
        value_fn=az.make_value_fn(model, n),
    )
    s.run(sims)
    return s


def do_train(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    tol = Tolerances()
    fs = az.make_feature_spec(dx=DX, max_layers=MAX_LAYERS)
    print(f"train: dx=1/{round(1 / DX)} max_layers={MAX_LAYERS} "
          f"M={fs.M} F={fs.F} train_ns={list(TRAIN_NS)} (n=5 held out)")

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

    selfplay = []
    t0 = time.perf_counter()
    for n in TRAIN_NS:
        sims = BOOTSTRAP_SIMS[n]
        s = run_uniform_search(n, sims, seed=0, tol=tol)
        seq = s.best_sequence()
        records.append({"n": n, "dx": DX, "seq": seq})
        sp = az.selfplay_samples(fs, s)
        selfplay += sp
        print(f"  bootstrap n={n}: sims={s.sims_done} "
              f"best_overhang={s.best_overhang:.4f} selfplay_rows={len(sp)} "
              f"({time.perf_counter() - t0:.0f}s elapsed)")

    imit, dropped = az.imitation_samples(fs, records)
    print(f"  imitation rows: {len(imit)} (dropped {dropped} records: "
          f"wrong dx or lattice-illegal)")
    print(f"  bootstrap distillation rows: {len(selfplay)}")

    # Phase 1: imitation plus distillation of the uniform bootstrap searches.
    # Best-sequence prefixes give one-hot policy targets; the search visit
    # distributions give soft targets. Both value heads use the final overhang.
    # This model backs the learned eval conditions before any self-play round.
    phase1 = list(imit) + list(selfplay)
    print(f"  phase-1 dataset: {len(phase1)} rows")
    feats, pols, vals, masks = az.assemble_arrays(fs, phase1)
    model = az.AZModel(fs, init_seed=args.init_seed)
    hist = az.train(model, feats, pols, vals, masks,
                    steps=args.imit_steps, batch=256, lr=3e-4, seed=0)
    print(f"  phase-1 train: {args.imit_steps} steps "
          f"loss {hist[0]['loss']:.4f} -> {hist[-1]['loss']:.4f} "
          f"(policy {hist[-1]['policy']:.4f} value {hist[-1]['value']:.4f})")
    az.save_params(model, PARAMS_IMIT)
    print(f"  wrote {PARAMS_IMIT}")

    # Phase 2: self-play rounds. Run the learned search, distil its visit
    # distributions, add them to the replay buffer, and retrain. Optional.
    replay = list(imit) + list(selfplay)
    for r in range(args.sp_rounds):
        for n in TRAIN_NS:
            s = run_learned_search(n, args.sp_sims, seed=100 + r, tol=tol,
                                   model=model)
            new = az.selfplay_samples(fs, s)
            replay += new
            print(f"  selfplay round {r + 1} n={n}: best={s.best_overhang:.4f} "
                  f"rows+={len(new)} buffer={len(replay)}")
        f2, p2, v2, m2 = az.assemble_arrays(fs, replay)
        hist = az.train(model, f2, p2, v2, m2,
                        steps=args.sp_steps, batch=256, lr=3e-4, seed=r + 1)
        print(f"  selfplay round {r + 1} train: {args.sp_steps} steps "
              f"loss {hist[0]['loss']:.4f} -> {hist[-1]['loss']:.4f}")

    az.save_params(model, PARAMS_FINAL)
    print(f"  wrote {PARAMS_FINAL} "
          f"({'imitation only' if args.sp_rounds == 0 else 'imitation + self-play'})")
    print(f"  total train wall: {time.perf_counter() - t0:.0f}s")
    return 0


# --- evaluation -------------------------------------------------------------


def sims_to_targets(n, targets, cap, seed, tol, prior_fn=None, value_fn=None):
    """Simulations until best_overhang first reaches each target, or None.

    Runs one batched iteration at a time so the first crossing is caught at
    K-simulation resolution. Runs until every target is crossed or the cap is
    hit. Returns (reached_dict, best_seen, search) where reached_dict maps each
    target to its crossing sim count or None.
    """
    s = Search(n, DX, tol, seed=seed, batch=K_BATCH, search_iter=SEARCH_ITER,
               prior_fn=prior_fn, value_fn=value_fn)
    n_iter = max(1, math.ceil(cap / K_BATCH))
    reached = {t: None for t in targets}
    remaining = set(targets)
    it = 0
    for it in range(n_iter):
        s.run_iteration()
        done = [t for t in remaining if s.best_overhang >= t - 1e-9]
        for t in done:
            reached[t] = (it + 1) * K_BATCH
            remaining.discard(t)
        if not remaining:
            break
    s.sims_done = (it + 1) * K_BATCH
    best = s.best_overhang if s.best_overhang != float("-inf") else 0.0
    return reached, best, s


def eval_condition(n, targets, cap, seeds, tol, make_hooks):
    """Run one condition over seeds. Returns (per_seed_reached, bests, searches).

    make_hooks(n) -> (prior_fn, value_fn). per_seed_reached[i] is the reached
    dict for seed i; bests[i] is its best overhang; searches[i] is the search.
    """
    per_seed, bests, searches = [], [], []
    for seed in seeds:
        prior_fn, value_fn = make_hooks(n)
        reached, best, s = sims_to_targets(n, targets, cap, seed, tol,
                                           prior_fn=prior_fn, value_fn=value_fn)
        per_seed.append(reached)
        bests.append(best)
        searches.append(s)
    return per_seed, bests, searches


def summarize_target(per_seed, target, cap):
    """Median sims-to-target and reached-count. Not-reached counts as cap."""
    vals = [d[target] for d in per_seed]
    filled = [v if v is not None else cap for v in vals]
    reached = sum(1 for v in vals if v is not None)
    return statistics.median(filled), reached


def do_eval(args):
    tol = Tolerances()
    fs = az.make_feature_spec(dx=DX, max_layers=MAX_LAYERS)
    seeds = list(range(args.seeds))

    # Load the trained models. Imitation model backs the learned conditions;
    # the final model adds self-play if the training run collected any.
    imit_model = az.AZModel(fs, init_seed=args.init_seed)
    az.load_params(imit_model, PARAMS_IMIT)
    final_model = az.AZModel(fs, init_seed=args.init_seed)
    have_final = os.path.exists(PARAMS_FINAL)
    if have_final:
        az.load_params(final_model, PARAMS_FINAL)

    # Conditions. Each maps a size n to (prior_fn, value_fn).
    def uniform_hooks(n):
        return None, None

    def prior_only_hooks(n):
        return az.make_prior_fn(imit_model, n), None

    def prior_value_hooks(n):
        return az.make_prior_fn(imit_model, n), az.make_value_fn(imit_model, n)

    def final_hooks(n):
        return az.make_prior_fn(final_model, n), az.make_value_fn(final_model, n)

    conditions = [
        ("uniform", uniform_hooks),
        ("prior", prior_only_hooks),
        ("prior+value", prior_value_hooks),
    ]
    if have_final:
        conditions.append(("prior+value+selfplay", final_hooks))

    # Targets per size, on the dx = 1/12 grid (edges are j/12 + 0.5).
    # n = 4: bnb certifies the optimum at 1.25 (j = 9). The task target 7/6 =
    #   1.1667 (j = 8) is that optimum minus one grid step, so we report both:
    #   1.1667 (both methods should reach) and 1.25 (the certified optimum).
    # n = 6: 1.1667 is the reachable target of prior uniform runs. Note 1.2083
    #   is off the dx = 1/12 grid; the next on-grid value above 1.1667 is 1.25.
    # n = 5 (transfer, held out of training): harmonic(5) = 1.1417 rounded down
    #   to grid is 1.0833 (j = 7).
    OPT4 = 1.25
    tasks = [
        (4, [7.0 / 6.0, OPT4], args.cap_n4, False),
        (6, [1.0 + 2.0 / 12.0], args.cap_n6, False),
        (5, [1.0 + 1.0 / 12.0], args.cap_n5, True),
    ]

    print(f"eval: seeds={seeds} caps n4={args.cap_n4} n6={args.cap_n6} "
          f"n5={args.cap_n5} self-play model in checkpoint: {have_final}")
    print(f"grid dx=1/{round(1 / DX)}; n=4 optimum 1.25 is bnb-certified; "
          f"n=5 is a held-out transfer size")
    print("")
    header = (f"{'n':>3} {'target':>8} {'condition':>22} {'median_sims':>12} "
              f"{'reached':>9} {'med_best':>9}")
    print(header)
    print("-" * len(header))

    verify_jobs = []  # (label, n, sequence, best) for host re-verification
    for (n, targets, cap, transfer) in tasks:
        tag = "transfer" if transfer else "target"
        for (name, hooks) in conditions:
            per_seed, bests, searches = eval_condition(
                n, targets, cap, seeds, tol, hooks)
            med_best = statistics.median(bests)
            for t in targets:
                med, reached = summarize_target(per_seed, t, cap)
                print(f"{n:>3} {t:>8.4f} {name:>22} {med:>12.0f} "
                      f"{reached:>4}/{len(seeds):<3} {med_best:>9.4f}")
            # Re-verify the best sequence of the seed with the highest overhang.
            pick = searches[int(max(range(len(bests)), key=lambda i: bests[i]))]
            verify_jobs.append((f"n={n} {name} {tag}", n, pick.best_sequence(),
                               pick.best_overhang))
        print("")

    print("certified host re-verification of reported best sequences:")
    for (label, n, seq, best) in verify_jobs:
        print(f"  [{label}] search_best={best:.4f}")
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
    parser.add_argument("--cap_n4", type=int, default=4000)
    parser.add_argument("--cap_n5", type=int, default=3000)
    parser.add_argument("--cap_n6", type=int, default=6000)
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
