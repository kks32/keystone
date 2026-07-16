"""Certify the true grid optimum of prefix-feasible maximum overhang.

Runs best-first branch and bound (keystone.search.bnb) over the lattice
cube-stacking scene and reports either a certified grid optimum or an honest
interval when a budget stops the search first. The optimum's build order is
re-verified prefix by prefix on the host pipeline before it is emitted.

--placement selects the reachability rule the search enforces per step:
static (no motion check, the default), drop (clear vertical column above
the target, a crane or top-grasp build), or slide (drop column or a clear
lateral corridor at the target layer).

Two JSON files are written. --out holds the full record of this run (node
counts, wall time, the prefix margin trace). The accumulating optima file
(--optima, default out/search/bnb_optima.json) keeps one simple record per
(n, dx, placement) for downstream learning: n, dx, placement, optimum,
certified, and the optimal build sequence. A record without a placement
field is a static run from before the field existed.

Run:
  python examples/certify_overhang.py --n 4 --dx 1/12
  python examples/certify_overhang.py --n 4 --dx 1/24 --time-limit 1800
  python examples/certify_overhang.py --n 4 --dx 1/12 --placement drop
"""

import argparse
import json
import os
import sys

from keystone import SolverOptions, Tolerances
from keystone.search import bnb
from keystone.search import lattice as LT


def dx_denom(dx: float) -> int:
    """Grid-step denominator for file naming, e.g. 1/12 -> 12."""
    return round(1.0 / dx)


def seq_records(sequence, dx):
    """Sequence as simple JSON records: layer, grid index j, center x."""
    return [{"layer": int(L), "j": int(j), "x": j * dx} for (L, j) in sequence]


def update_optima(path, res):
    """Accumulate one optima record per (n, dx, placement).

    Replaces any stale record with the same key. A record with no
    placement field predates the field and means placement "static".
    """
    records = []
    if os.path.exists(path):
        with open(path) as f:
            records = json.load(f)
    key = (res.n, dx_denom(res.dx), res.placement)
    records = [
        r
        for r in records
        if (r["n"], dx_denom(r["dx"]), r.get("placement", "static")) != key
    ]
    records.append(
        {
            "n": res.n,
            "dx": res.dx,
            "placement": res.placement,
            "optimum": res.optimum,
            "certified": res.certified,
            "sequence": seq_records(res.sequence, res.dx),
        }
    )
    records.sort(
        key=lambda r: (r["n"], dx_denom(r["dx"]), r.get("placement", "static"))
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=1)


def main():
    parser = argparse.ArgumentParser(
        description="Certify the grid optimum of prefix-feasible max overhang."
    )
    parser.add_argument("--n", type=int, default=4, help="number of cubes")
    parser.add_argument("--dx", type=str, default="1/12",
                        help="grid step, float or fraction like 1/12")
    parser.add_argument("--placement", choices=["static", "slide", "drop"],
                        default="static",
                        help="per-step reachability rule enforced by the search")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="wall-clock budget in seconds")
    parser.add_argument("--max-nodes", type=int, default=None,
                        help="expansion budget")
    parser.add_argument("--out", type=str, default=None,
                        help="per-run JSON record path")
    parser.add_argument("--optima", type=str, default="out/search/bnb_optima.json",
                        help="accumulating optima JSON for downstream learning")
    parser.add_argument("--solver-tol", type=float, default=SolverOptions().solver_tol)
    parser.add_argument("--max-iter", type=int, default=SolverOptions().max_iter,
                        help="qpax iteration cap for the reference verdict")
    args = parser.parse_args()

    dx = bnb.parse_dx(args.dx)
    den = dx_denom(dx)
    tol = Tolerances()
    opts = SolverOptions(solver_tol=args.solver_tol, max_iter=args.max_iter)
    base = LT.harmonic(args.n)

    tag = "" if args.placement == "static" else f"_{args.placement}"
    out = args.out or f"out/search/certify_n{args.n}_dx{den}{tag}.json"

    print(
        f"certify: n={args.n} dx=1/{den} placement={args.placement} "
        f"time_limit={args.time_limit} max_nodes={args.max_nodes} "
        f"max_iter={opts.max_iter}",
        flush=True,
    )
    print(f"harmonic(n) sum(1/2k) = {base:.6f} block widths", flush=True)

    engine = bnb.Certifier(args.n, dx, tol, opts=opts, placement=args.placement)
    res = engine.run(max_nodes=args.max_nodes, time_limit=args.time_limit)

    print("")
    seq_str = ", ".join(f"({L},{j},{j * dx:+.4f})" for (L, j) in res.sequence)
    if res.certified:
        print(f"certified grid optimum = {res.optimum:.6f} with sequence {seq_str}")
    else:
        print(f"certified interval [{res.lower:.6f}, {res.upper:.6f}] "
              f"(stopped: {res.stop_reason}); best sequence {seq_str}")
    print(f"harmonic(n)            = {base:.6f}")
    if res.optimum > float("-inf"):
        print(f"optimum - harmonic     = {res.optimum - base:+.6f} "
              f"({(res.optimum - base) / dx:+.2f} grid steps)")
    print(f"host re-verified       = {res.host_verified}")
    print(f"stop reason            = {res.stop_reason}")
    print(f"nodes expanded         = {res.nodes_expanded}")
    print(f"nodes generated        = {res.nodes_generated}")
    print(f"frontier at stop       = {res.frontier_size} (max {res.max_frontier})")
    print(f"closed set size        = {res.closed_size}")
    print(f"qp feasibility solves  = {res.qp_solves}")
    print(f"host verifications     = {res.host_verifications}")
    print(f"wall time              = {res.wall_time:.1f} s")

    print("")
    print("prefix margin trace of the reported structure (host pipeline):")
    for step, (L, j, x, margin, status) in enumerate(res.prefix_trace, start=1):
        print(f"  step {step:2d}: place (L={L}, x={x:+.4f}) "
              f"cube_edge={x + 0.5:+.4f}  margin={margin:.3e}  {status}")

    record = {
        "n": res.n,
        "dx": res.dx,
        "placement": res.placement,
        "certified": res.certified,
        "optimum": res.optimum,
        "lower": res.lower,
        "upper": res.upper,
        "stop_reason": res.stop_reason,
        "host_verified": res.host_verified,
        "harmonic": base,
        "sequence": seq_records(res.sequence, dx),
        "prefix_trace": [
            {"layer": L, "j": j, "x": x, "margin": margin, "status": status}
            for (L, j, x, margin, status) in res.prefix_trace
        ],
        "nodes_expanded": res.nodes_expanded,
        "nodes_generated": res.nodes_generated,
        "frontier_size": res.frontier_size,
        "max_frontier": res.max_frontier,
        "qp_solves": res.qp_solves,
        "host_verifications": res.host_verifications,
        "closed_size": res.closed_size,
        "wall_time": res.wall_time,
        "time_limit": args.time_limit,
        "max_nodes": args.max_nodes,
        "max_iter": opts.max_iter,
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(record, f, indent=1)
    print("")
    print(f"wrote {out}")

    update_optima(args.optima, res)
    print(f"updated {args.optima}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
