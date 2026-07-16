"""Learned priors and values for the maximum-overhang lattice search.

This trains an AlphaZero-style network that shapes the PUCT search in
keystone.search and measures whether it reaches a target overhang in fewer
simulations than the uniform-prior search. Certification is unchanged: the
network only steers exploration, and every reported best sequence is
re-verified step by step on the certified host pipeline.

Value targets are per-state, not one constant per episode. Imitation prefixes
carry the suffix-max overhang of their trajectory; self-play nodes carry the
subtree-max overhang reachable from that node. A third network head regresses
the certified P4 margin the search already computed, an auxiliary signal that
never feeds the search. See keystone.search.az.

Two modes:

  python examples/az_overhang.py --train
      Collect imitation and self-play data on the lattice, train two shared
      networks (per-state value, and value plus the margin head), and
      checkpoint them to out/search/az_params_v2*.msgpack.

  python examples/az_overhang.py --eval
      For n = 4 and n = 6 at dx = 1/12, report simulations-to-target as the
      median over five seeds under four conditions: uniform priors, learned
      priors, learned priors plus the per-state value head, and that plus the
      margin auxiliary head. Add a transfer row at n = 5, a size held out of
      all training, and an n = 6 overshoot probe.

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
# v2 checkpoints. The value model uses per-state value targets only; the
# margin model adds the auxiliary margin head. Both back the eval conditions.
PARAMS_V2_VALUE = os.path.join(OUT_DIR, "az_params_v2_value.msgpack")
PARAMS_V2 = os.path.join(OUT_DIR, "az_params_v2.msgpack")

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


def _value_loss_trajectory(hist):
    """Compact value-loss trajectory string from a training history."""
    return " ".join(f"{h['step']}:{h['value']:.2e}" for h in hist)


def train_variant(name, fs, replay_seed, phase1, imit, selfplay_boot, tol,
                  model, margin_weight, robust, args):
    """Train one model through phase 1 and the self-play rounds.

    margin_weight scales the auxiliary margin loss; 0.0 disables the head's
    supervision. robust switches the value target to the knife-edge-penalized
    variant. Returns the phase-1 and last self-play histories for reporting.
    """
    f1, p1, v1, m1, mg1, mgm1 = az.assemble_arrays(fs, phase1)
    print(f"  [{name}] phase-1 value-target std={float(v1.std()):.4f} "
          f"margin rows={int(mgm1.sum())}/{len(phase1)}")
    h1 = az.train(model, f1, p1, v1, m1, mg1, mgm1,
                  steps=args.imit_steps, batch=256, lr=3e-4,
                  margin_weight=margin_weight, seed=0)
    print(f"  [{name}] phase-1 train: {args.imit_steps} steps "
          f"loss {h1[0]['loss']:.4f} -> {h1[-1]['loss']:.4f} "
          f"(policy {h1[-1]['policy']:.4f} value {h1[-1]['value']:.2e} "
          f"margin {h1[-1]['margin']:.2e})")
    print(f"  [{name}] phase-1 value loss: {_value_loss_trajectory(h1)}")

    # Self-play rounds: run the learned search under this model's own hooks,
    # distil visit distributions with per-state subtree-max value targets and
    # threaded margins, add to the replay buffer, and retrain.
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
                  f"best={s.best_overhang:.4f} rows+={len(new)} "
                  f"buffer={len(replay)}")
        f2, p2, v2, m2, mg2, mgm2 = az.assemble_arrays(fs, replay)
        h2 = az.train(model, f2, p2, v2, m2, mg2, mgm2,
                      steps=args.sp_steps, batch=256, lr=3e-4,
                      margin_weight=margin_weight, seed=r + 1)
        print(f"  [{name}] selfplay round {r + 1} train: {args.sp_steps} steps "
              f"loss {h2[0]['loss']:.4f} -> {h2[-1]['loss']:.4f} "
              f"(value {h2[-1]['value']:.2e} margin {h2[-1]['margin']:.2e})")
    print(f"  [{name}] final self-play value loss: {_value_loss_trajectory(h2)}")
    return h1, h2


def do_train(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    tol = Tolerances()
    fs = az.make_feature_spec(dx=DX, max_layers=MAX_LAYERS)
    print(f"train v2: dx=1/{round(1 / DX)} max_layers={MAX_LAYERS} "
          f"M={fs.M} F={fs.F} train_ns={list(TRAIN_NS)} (n=5 held out)")
    print(f"  per-state value targets (suffix/subtree-max), margin_weight="
          f"{args.margin_weight}, robust={bool(args.robust)}")

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

    # Bootstrap uniform searches. These are the shared data source for both
    # variants: same searches, same seeds, so the only difference downstream is
    # the training objective.
    selfplay_boot = []
    t0 = time.perf_counter()
    for n in TRAIN_NS:
        sims = BOOTSTRAP_SIMS[n]
        s = run_uniform_search(n, sims, seed=0, tol=tol)
        seq = s.best_sequence()
        records.append({"n": n, "dx": DX, "seq": seq})
        sp = az.selfplay_samples(fs, s, robust=bool(args.robust),
                                 robust_penalty=args.robust_penalty)
        selfplay_boot += sp
        print(f"  bootstrap n={n}: sims={s.sims_done} "
              f"best_overhang={s.best_overhang:.4f} selfplay_rows={len(sp)} "
              f"({time.perf_counter() - t0:.0f}s elapsed)")

    imit, dropped = az.imitation_samples(fs, records)
    print(f"  imitation rows: {len(imit)} (dropped {dropped} records: "
          f"wrong dx or lattice-illegal)")
    print(f"  bootstrap distillation rows: {len(selfplay_boot)}")

    phase1 = list(imit) + list(selfplay_boot)
    print(f"  phase-1 dataset: {len(phase1)} rows")

    # Value model: per-state value targets, no margin supervision.
    value_model = az.AZModel(fs, init_seed=args.init_seed)
    train_variant("value", fs, 0, phase1, imit, selfplay_boot, tol,
                  value_model, margin_weight=0.0, robust=bool(args.robust),
                  args=args)
    az.save_params(value_model, PARAMS_V2_VALUE)
    print(f"  wrote {PARAMS_V2_VALUE}")

    # Margin-aux model: same per-state value targets plus the auxiliary margin
    # head trained on the certified margins the search already computed.
    margin_model = az.AZModel(fs, init_seed=args.init_seed)
    train_variant("margin-aux", fs, 0, phase1, imit, selfplay_boot, tol,
                  margin_model, margin_weight=args.margin_weight,
                  robust=bool(args.robust), args=args)
    az.save_params(margin_model, PARAMS_V2)
    print(f"  wrote {PARAMS_V2}")

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


def overshoot_probe(conditions, tol, cap, seeds):
    """n = 6 overshoot probe: sims to reach 1.25 and 1.3333 past the 1.1667 target.

    Learned conditions only. Mirrors the supplementary probe in the prior run
    so the two runs stay comparable.
    """
    print(f"=== supplementary n=6 overshoot probe (learned only, cap {cap}, "
          f"seeds {seeds}) ===")
    print(f"n=6 overshoot probe, cap {cap} seeds {seeds}")
    targets = [1.25, 4.0 / 3.0]
    header = (f"{'condition':>22}   {'->1.25':>10} {'->1.3333':>10} "
              f"{'med_best':>9}")
    print(header)
    verify = []
    for (name, hooks) in conditions:
        if name == "uniform":
            continue
        per_seed, bests, searches = eval_condition(6, targets, cap, seeds, tol,
                                                   hooks)
        med_best = statistics.median(bests)
        cells = []
        for t in targets:
            med, reached = summarize_target(per_seed, t, cap)
            cells.append(f"{med:.0f}({reached}/{len(seeds)})")
        print(f"{name:>22}   {cells[0]:>10} {cells[1]:>10} {med_best:>9.4f}")
        pick = searches[int(max(range(len(bests)), key=lambda i: bests[i]))]
        verify.append((name, pick.best_sequence(), pick.best_overhang))
    print("host re-verification of best n=6 sequences:")
    for (name, seq, best) in verify:
        if not seq:
            print(f"  {name} best={best:.4f}: no sequence")
            continue
        trace = verify_sequence(seq, DX, tol)
        ok = all(status == "feasible" for (_L, _j, _x, _m, status) in trace)
        edge = max(x + 0.5 for (_L, _j, x, _m, _s) in trace)
        print(f"  {name} best={best:.4f}: prefix_feasible={ok} "
              f"rightmost_edge={edge:.4f} steps={len(seq)}")


def do_eval(args):
    tol = Tolerances()
    fs = az.make_feature_spec(dx=DX, max_layers=MAX_LAYERS)
    seeds = list(range(args.seeds))

    # Load the two v2 models. The value model uses per-state value targets
    # only; the margin model adds the auxiliary margin head during training.
    # The margin head never feeds the search, so any difference between the two
    # is the auxiliary task reshaping the shared prior and value heads.
    value_model = az.AZModel(fs, init_seed=args.init_seed)
    az.load_params(value_model, PARAMS_V2_VALUE)
    margin_model = az.AZModel(fs, init_seed=args.init_seed)
    az.load_params(margin_model, PARAMS_V2)

    # Conditions. Each maps a size n to (prior_fn, value_fn).
    def uniform_hooks(n):
        return None, None

    def prior_only_hooks(n):
        return az.make_prior_fn(value_model, n), None

    def prior_value_hooks(n):
        return az.make_prior_fn(value_model, n), az.make_value_fn(value_model, n)

    def prior_value_margin_hooks(n):
        return (az.make_prior_fn(margin_model, n),
                az.make_value_fn(margin_model, n))

    conditions = [
        ("uniform", uniform_hooks),
        ("prior", prior_only_hooks),
        ("prior+value(new)", prior_value_hooks),
        ("prior+value+margin-aux", prior_value_margin_hooks),
    ]

    # Targets per size, on the dx = 1/12 grid (edges are j/12 + 0.5).
    # n = 4: bnb certifies the optimum at 1.25 (j = 9). The task target 7/6 =
    #   1.1667 (j = 8) is that optimum minus one grid step, so we report both.
    # n = 6: 1.1667 is the reachable target of prior uniform runs.
    # n = 5 (transfer, held out of training): 1.0833 (j = 7).
    OPT4 = 1.25
    tasks = [
        (4, [7.0 / 6.0, OPT4], args.cap_n4, False),
        (6, [1.0 + 2.0 / 12.0], args.cap_n6, False),
        (5, [1.0 + 1.0 / 12.0], args.cap_n5, True),
    ]

    print(f"eval v2: seeds={seeds} caps n4={args.cap_n4} n6={args.cap_n6} "
          f"n5={args.cap_n5}")
    print(f"grid dx=1/{round(1 / DX)}; n=4 optimum 1.25 is bnb-certified; "
          f"n=5 is a held-out transfer size")
    print("conditions compare per-state value (new) and the margin auxiliary "
          "head against uniform and prior-only")
    print("")
    header = (f"{'n':>3} {'target':>8} {'condition':>24} {'median_sims':>12} "
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
                print(f"{n:>3} {t:>8.4f} {name:>24} {med:>12.0f} "
                      f"{reached:>4}/{len(seeds):<3} {med_best:>9.4f}")
            # Re-verify the best sequence of the seed with the highest overhang.
            pick = searches[int(max(range(len(bests)), key=lambda i: bests[i]))]
            verify_jobs.append((f"n={n} {name} {tag}", n, pick.best_sequence(),
                               pick.best_overhang))
        print("")

    overshoot_probe(conditions, tol, args.cap_overshoot, list(range(3)))
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
    parser.add_argument("--margin_weight", type=float, default=0.5,
                        help="weight on the auxiliary margin loss")
    parser.add_argument("--robust", action="store_true",
                        help="use the knife-edge-penalized value target")
    parser.add_argument("--robust_penalty", type=float, default=0.05,
                        help="value penalty for knife-edge best paths")
    parser.add_argument("--cap_n4", type=int, default=4000)
    parser.add_argument("--cap_n5", type=int, default=3000)
    parser.add_argument("--cap_n6", type=int, default=6000)
    parser.add_argument("--cap_overshoot", type=int, default=1200,
                        help="cap for the n=6 overshoot probe")
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
