"""Fast maximum-overhang cube-stacking search on the jittable lattice.

Same problem and same PUCT search as examples/search_overhang.py, with the
inner loop replaced by keystone.search: a pure-JNP environment
(build_system) and batched PUCT with virtual loss. The naive script
rebuilds a numpy Assembly per candidate on the host, which dominates its
runtime. Here the geometry-to-margin path is jitted and vmapped, so the
per-simulation cost drops to the batched QP alone.

The reported best structure is re-verified step by step with the certified
host pipeline (build_assembly + assemble + solve_p0/solve_p4) and rendered
through keystone.viz from that same certified path, so the picture and the
prefix-feasibility check never depend on the fast environment.

Run: python examples/search_overhang_fast.py --n 4 --sims 2000 --batch 16
"""

import argparse
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")

from keystone import (
    Tolerances,
    assemble,
    box_2d,
    build_assembly,
    solve_p0,
    solve_p4,
)
from keystone.search import Search
from keystone.search import lattice as LT
from keystone.viz import plot_assembly_2d, save_fig

MU = LT.MU


def boxes_of(key, dx):
    """Pedestal plus one unit cube per placement (L, j), sorted (L, j)."""
    boxes = [box_2d(6, 1, -3, 0.5)]
    for (L, j) in sorted(key):
        boxes.append(box_2d(1, 1, j * dx, 1.5 + L))
    return boxes


def host_system(key, dx, tol):
    """Certified host equilibrium system for a state, matching the search."""
    n = len(key)
    boxes = boxes_of(key, dx)
    asm = build_assembly(
        boxes, mu=MU, tol=tol, dim=2,
        pad_blocks=n + 1, pad_patches=2 * n + 2, pad_verts=2,
    )
    return asm, assemble(asm, tol, cone="linear2d")


def verify_sequence(seq, dx, tol):
    """Replay a build order and solve P4 at each prefix on the host pipeline.

    Returns [(L, j, x, margin, status)]. Prefix feasibility holds when every
    status is feasible.
    """
    trace = []
    placed = []
    for (L, j) in seq:
        placed.append((L, j))
        _asm, system = host_system(tuple(placed), dx, tol)
        r = solve_p4(system, tol)
        trace.append((L, j, j * dx, r.margin, r.status))
    return trace


def render_best(key, dx, tol, path):
    """Draw the certified best structure with its P0 result and save it."""
    import matplotlib.pyplot as plt

    asm, system = host_system(key, dx, tol)
    r = solve_p0(system, tol)
    ov = LT.overhang(key, dx)
    title = f"best overhang = {ov:.4f} ({len(key)} cubes), status {r.status}"
    fig = plot_assembly_2d(asm, r, boxes=boxes_of(key, dx), title=title)
    save_fig(fig, path)
    plt.close(fig)
    return r


def main():
    parser = argparse.ArgumentParser(
        description="Fast batched MCTS for maximum-overhang cube stacking."
    )
    parser.add_argument("--n", type=int, default=4, help="number of cubes")
    parser.add_argument("--sims", type=int, default=2000, help="MCTS simulations")
    parser.add_argument("--dx", type=float, default=1.0 / 24.0, help="x grid step")
    parser.add_argument("--cpuct", type=float, default=1.4, help="PUCT constant")
    parser.add_argument("--batch", type=int, default=16,
                        help="K: selections per batched iteration (virtual loss)")
    parser.add_argument("--seed", type=int, default=0, help="numpy seed")
    parser.add_argument("--search_iter", type=int, default=50,
                        help="qpax iteration cap for the search oracle "
                             "(conservative below 100; final check uses 100)")
    parser.add_argument("--out", default="out/search", help="output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tol = Tolerances()
    base = LT.harmonic(args.n)

    search = Search(
        args.n, args.dx, tol, c_puct=args.cpuct, seed=args.seed,
        batch=args.batch, search_iter=args.search_iter,
    )

    print(
        f"search-fast: n={args.n} sims={args.sims} dx=1/{round(1 / args.dx)} "
        f"c_puct={args.cpuct} batch(K)={args.batch} seed={args.seed} "
        f"search_iter={args.search_iter}",
        flush=True,
    )
    print(f"harmonic baseline sum(1/2k) = {base:.6f} block widths", flush=True)

    # progress print roughly every 500 sims.
    progress = max(1, round(500 / max(1, args.batch)))
    best = search.run(args.sims, progress=progress)
    wall = search.wall

    seq = search.best_sequence()
    ratio = best / base if base > 0 else float("nan")

    print("")
    print(f"best overhang        = {best:.6f} block widths")
    print(f"harmonic(n)          = {base:.6f}")
    print(f"ratio best/harmonic  = {ratio:.4f}")
    print(f"gap below harmonic   = {base - best:.6f} "
          f"({(base - best) / args.dx:.2f} grid steps)")
    print(f"best sequence (L, j, x): "
          + ", ".join(f"({L},{j},{j * args.dx:+.4f})" for (L, j) in seq))
    print(f"wall time            = {wall:.1f} s")
    print(f"rate                 = {search.sims_done / wall:.1f} sims/s "
          f"({search.sims_done} sims over {wall:.1f} s)")
    print(f"time split           = legality {search.t_legal:.2f} s, "
          f"solver {search.t_solve:.1f} s")
    print(f"qp feasibility solves= {search.n_qp} "
          f"(cache size {len(search.feas_cache)})")
    if search.n_qp > 0 and search.t_solve > 0:
        print(f"solver per qp        = {1000 * search.t_solve / search.n_qp:.2f} ms")
    print(search.tree_report(), flush=True)

    # Re-verify prefix feasibility of the best sequence on the host pipeline.
    print("")
    print("prefix margin trace of the best sequence (host pipeline):")
    trace = verify_sequence(seq, args.dx, tol)
    all_feasible = True
    for step, (L, j, x, margin, status) in enumerate(trace, start=1):
        print(f"  step {step:2d}: place (L={L}, x={x:+.4f}) "
              f"cube_edge={x + 0.5:+.4f}  margin={margin:.3e}  {status}")
        if status != "feasible":
            all_feasible = False
    print(f"prefix feasible throughout: {all_feasible}")

    # Render the certified best structure.
    png = os.path.join(args.out, f"fast_best_n{args.n}_dx{round(1 / args.dx)}.png")
    if search.best_key is not None:
        render_best(search.best_key, args.dx, tol, png)
        print(f"wrote {png}")
    else:
        print("no feasible cube placement found")

    return 0


if __name__ == "__main__":
    sys.exit(main())
