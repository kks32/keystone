"""CLI for the end-to-end stacking pipeline (keystone.pipeline).

Run one stack size, or several with --batch, through search, certification,
build planning, MuJoCo execution, and a movie.

    python examples/pipeline_stack.py --n 4 --sims 2000
    python examples/pipeline_stack.py --n 4 --sims 2000 --executor franka
    python examples/pipeline_stack.py --batch 4,6 --sims 4000
    python examples/pipeline_stack.py --n 4 --sims 50 --no-video
"""

import argparse
import sys

from keystone.pipeline import evaluate_stacking


def _fmt_row(rec):
    s = rec.get("search", {})
    ag = rec.get("agreement", {})
    plan = rec.get("plan", {})
    counts = plan.get("protocol_counts", {})
    protos = "/".join(
        f"{counts.get(k, 0)}{k[0]}" for k in ("drop", "ride_under", "prop")
    )
    ex = rec.get("execute", {})
    return (
        f"{rec['n']:>3d} "
        f"{s.get('best_overhang', float('nan')):>8.4f} "
        f"{rec.get('prior', '?'):>8s} "
        f"{str(ag.get('certificate')):>6s} "
        f"{protos:>10s} "
        f"{str(ex.get('verdict', '?')):>26s} "
        f"{str(ag.get('three_way', '?')):>26s}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=4, help="number of cubes")
    ap.add_argument(
        "--batch",
        type=str,
        default=None,
        help='comma-separated n values, e.g. "4,6"; overrides --n',
    )
    ap.add_argument("--sims", type=int, default=4000, help="MCTS simulations")
    ap.add_argument("--dx", type=float, default=1.0 / 12.0, help="x grid step")
    ap.add_argument("--seed", type=int, default=0, help="search seed")
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="out/search/az_params_v2.msgpack",
        help="AlphaZero params; uniform fallback when missing",
    )
    ap.add_argument("--out", type=str, default="out/pipeline", help="output dir")
    ap.add_argument(
        "--executor",
        type=str,
        default="driver",
        choices=["driver", "franka"],
        help="EXECUTE stage: capped impedance driver (default) or Franka arm",
    )
    ap.add_argument(
        "--no-video", action="store_true", help="skip the movie (record=False)"
    )
    ap.add_argument(
        "--progress",
        type=int,
        default=None,
        help="print search progress every this many iterations",
    )
    args = ap.parse_args()

    ns = (
        [int(x) for x in args.batch.split(",") if x.strip()]
        if args.batch
        else [args.n]
    )

    records = []
    for n in ns:
        rec = evaluate_stacking(
            n,
            dx=args.dx,
            sims=args.sims,
            seed=args.seed,
            checkpoint=args.checkpoint,
            out_dir=args.out,
            record=not args.no_video,
            executor=args.executor,
            progress=args.progress,
        )
        records.append(rec)

    if len(records) > 1:
        print()
        print("=== batch summary ===")
        print(
            f"{'n':>3s} {'overhang':>8s} {'prior':>8s} {'cert':>6s} "
            f"{'plan':>10s} {'exec_verdict':>26s} {'three_way':>26s}"
        )
        for rec in records:
            print(_fmt_row(rec))

    return 0


if __name__ == "__main__":
    sys.exit(main())
