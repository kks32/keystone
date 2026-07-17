"""Certified price of lateral reserve, and a reserve-mode learned search.

Two studies of "lam-robust" certification. A state is lam-robust when its P4
margin certifies feasible under a pseudo-static lateral load of +lam_min and
-lam_min times self-weight (both directions). The threshold lam_min is
calibrated against MuJoCo compliant-physics outcomes; see
docs/KNOWN_LIMITS.md and keystone.search.lattice.LAM_MIN.

1. --sweep (default): branch and bound with reserve feasibility at n=4,
   dx=1/12, mu=0.7, for several lam_min values. As lam_min grows the certified
   optimum walks down from the static 5/4: the price of reserve. The admissible
   reach bound is unchanged and stays valid, because a robust-feasible
   completion is also a feasible completion, so no reserve run can exceed the
   static reach ceiling; only the certified feasible set shrinks.

2. --mcts: the learned PUCT search (keystone.search.Search) with reserve
   feasibility at the calibrated lam_min, n=6, then one driver execution of the
   design it finds. The execution records a movie and reports the settle
   verdict, closing the loop from reserve certificate to compliant physics.

Run:
  python examples/robust_sweep.py --sweep --time-limit 900
  python examples/robust_sweep.py --mcts --sims 4000
"""

import argparse
import json
import os
import sys

from keystone import SolverOptions, Tolerances
from keystone.search import bnb
from keystone.search import lattice as LT


def run_sweep(n, dx, mu, lam_mins, time_limit, out):
    """Branch-and-bound optimum vs lam_min: the certified price of reserve."""
    tol = Tolerances()
    opts = SolverOptions()

    print(f"\nreserve price sweep: n={n} dx={dx:.5f} mu={mu} "
          f"time_limit={time_limit}s per run", flush=True)
    print(f"static optimum is the lam_min=0 baseline; "
          f"harmonic(n)={LT.harmonic(n):.4f}\n", flush=True)

    rows = []
    static = bnb.certify(n, dx, tol, opts=opts, mu=mu, progress=False,
                         time_limit=time_limit)
    rows.append({
        "lam_min": None,
        "optimum": static.optimum,
        "certified": static.certified,
        "upper": static.upper,
        "wall_s": static.wall_time,
        "sequence": [[int(L), int(j)] for (L, j) in static.sequence],
    })
    print(f"  lam_min=static  optimum={static.optimum:.4f}  "
          f"certified={static.certified}  wall={static.wall_time:.1f}s", flush=True)

    for lam in lam_mins:
        res = bnb.certify(n, dx, tol, opts=opts, mu=mu, robust=True, lam_min=lam,
                          progress=False, time_limit=time_limit)
        rows.append({
            "lam_min": lam,
            "optimum": res.optimum,
            "certified": res.certified,
            "upper": res.upper,
            "wall_s": res.wall_time,
            "sequence": [[int(L), int(j)] for (L, j) in res.sequence],
        })
        tag = "certified" if res.certified else f"interval<= {res.upper:.4f}"
        print(f"  lam_min={lam:<6.3f}  optimum={res.optimum:.4f}  "
              f"{tag}  wall={res.wall_time:.1f}s", flush=True)

    print("\nreserve price curve (optimum in block widths):")
    base = rows[0]["optimum"]
    for r in rows:
        lm = "static" if r["lam_min"] is None else f"{r['lam_min']:.3f}"
        drop = r["optimum"] - base
        print(f"  lam_min={lm:>7s}  optimum={r['optimum']:.4f}  "
              f"price={drop:+.4f} ({drop / dx:+.2f} grid steps)")

    record = {"n": n, "dx": dx, "mu": mu, "time_limit": time_limit, "rows": rows}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(record, f, indent=1)
    print(f"\nwrote {out}")
    return record


def _reserve_search(n, dx, sims, seed, lam_min, checkpoint, tol):
    """Learned PUCT search with reserve feasibility. Returns (best, sequence).

    Mirrors keystone.pipeline._run_search, loading the learned prior and value
    heads when the checkpoint is present, but constructs the Search with
    lam_min so every feasibility verdict is the two-sided lateral reserve test.
    """
    from keystone.search import az
    from keystone.search.mcts import Search

    prior_fn = value_fn = None
    prior = "uniform"
    if n <= az.MAX_LAYERS and checkpoint and os.path.exists(checkpoint):
        fs = az.make_feature_spec(dx=dx, max_layers=az.MAX_LAYERS)
        model = az.AZModel(fs, init_seed=0)
        try:
            az.load_params(model, checkpoint)
            prior_fn = az.make_prior_fn(model, n)
            value_fn = az.make_value_fn(model, n)
            prior = "learned"
        except Exception as exc:  # noqa: BLE001
            print(f"  checkpoint load failed ({type(exc).__name__}); uniform prior")

    search = Search(n, dx, tol, seed=seed, prior_fn=prior_fn, value_fn=value_fn,
                    lam_min=lam_min)
    best = search.run(sims, progress=max(1, sims // (10 * search.K)))
    seq = [(int(L), int(j)) for (L, j) in search.best_sequence()]
    return search, float(best), seq, prior


def run_mcts(n, dx, sims, seed, lam_min, checkpoint, out_dir):
    """Reserve-mode learned search at n, then one driver execution of its find."""
    from keystone.pipeline import evaluate_stacking

    tol = Tolerances()
    print(f"\nreserve MCTS: n={n} dx={dx:.5f} sims={sims} lam_min={lam_min} "
          f"seed={seed}", flush=True)
    _search, best, seq, prior = _reserve_search(
        n, dx, sims, seed, lam_min, checkpoint, tol
    )
    print(f"  prior={prior}  reserve-best overhang={best:.4f}  "
          f"harmonic(n)={LT.harmonic(n):.4f}", flush=True)
    print(f"  sequence: {[(L, j) for (L, j) in seq]}", flush=True)
    if not seq:
        print("  reserve search found no lam-robust design", flush=True)
        return None

    # One driver execution of the reserve design, with a movie.
    print("\n  executing the reserve design in MuJoCo (driver, record=True)...",
          flush=True)
    rec = evaluate_stacking(
        n=n, dx=dx, sequence=seq, seed=seed, out_dir=out_dir, record=True,
        executor="driver", tag="_robust", verbose=True,
    )
    ag = rec.get("agreement", {})
    print(f"\n  reserve design executed: predicted={rec.get('predicted_physics')} "
          f"physics_stands={ag.get('physics')} verdict={ag.get('three_way')}",
          flush=True)
    vids = rec.get("videos", {})
    print(f"  video mp4={vids.get('mp4')} gif={vids.get('gif')}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", action="store_true",
                    help="run the branch-and-bound reserve price sweep (default)")
    ap.add_argument("--mcts", action="store_true",
                    help="run the reserve learned search plus one execution")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--dx", type=str, default=None, help="grid step, e.g. 1/12")
    ap.add_argument("--mu", type=float, default=LT.MU)
    ap.add_argument("--lam-mins", type=str, default="0.01,0.02,0.05,0.1",
                    help="comma list of lam_min for the sweep")
    ap.add_argument("--lam-min", type=float, default=LT.LAM_MIN,
                    help="calibrated reserve threshold for the mcts run")
    ap.add_argument("--time-limit", type=float, default=900.0,
                    help="wall-clock budget per branch-and-bound run, seconds")
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", type=str,
                    default="out/search/az_params_v2.msgpack")
    ap.add_argument("--out-dir", type=str, default="out/robust")
    args = ap.parse_args()

    # Default to the sweep when neither is named.
    do_sweep = args.sweep or not args.mcts
    os.makedirs(args.out_dir, exist_ok=True)

    if do_sweep:
        n = args.n if args.n is not None else 4
        dx = bnb.parse_dx(args.dx) if args.dx else 1.0 / 12.0
        lam_mins = [float(x) for x in args.lam_mins.split(",")]
        out = os.path.join(args.out_dir, f"reserve_sweep_n{n}.json")
        run_sweep(n, dx, args.mu, lam_mins, args.time_limit, out)

    if args.mcts:
        n = args.n if args.n is not None else 6
        dx = bnb.parse_dx(args.dx) if args.dx else 1.0 / 12.0
        run_mcts(n, dx, args.sims, args.seed, args.lam_min, args.checkpoint,
                 args.out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
