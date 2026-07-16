"""Render every construction step of a placement sequence.

Replays a cube placement sequence on the overhang pedestal, solves each
prefix through the certified host pipeline, and renders one panel per
step with its verdict and margin. Shows that every intermediate state
stands, not only the finished structure.

Default sequence is the n=4 counterweight design found by the search
(overhang 1.1667, past the 25/24 harmonic limit).

Run: python examples/replay_steps.py --out out/search
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from keystone import Tolerances, assemble, box_2d, build_assembly, solve_p4
from keystone.viz import plot_assembly_2d

# (layer, center x) per placement, in build order. Layer L sits at
# center z = 1.5 + L on the pedestal (6 wide, right edge at x = 0).
N4_COUNTERWEIGHT = [
    (0, -1.0 / 12.0),
    (1, -0.5),
    (2, -0.25),
    (1, 2.0 / 3.0),
]


def pedestal():
    return box_2d(6.0, 1.0, -3.0, 0.5)


def cube(layer, x):
    return box_2d(1.0, 1.0, x, 1.5 + layer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/search")
    ap.add_argument("--mu", type=float, default=0.7)
    ap.add_argument(
        "--seq", default=None,
        help="JSON record written by search_overhang_fast.py; "
             "defaults to the built-in n=4 counterweight sequence",
    )
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tol = Tolerances()
    if args.seq is not None:
        with open(args.seq) as f:
            record = json.load(f)
        seq = [(p["layer"], p["x"]) for p in record["sequence"]]
    else:
        seq = N4_COUNTERWEIGHT
    n = len(seq)

    fig, axes = plt.subplots(
        1, n, figsize=(4.2 * n, 4.6), sharey=True, squeeze=False
    )
    for k, ax in enumerate(axes[0]):
        boxes = [pedestal()] + [cube(L, x) for L, x in seq[: k + 1]]
        a = build_assembly(boxes, mu=args.mu, tol=tol, dim=2)
        s = assemble(a, tol, cone="linear2d")
        r = solve_p4(s, tol)
        edge = max(x + 0.5 for _, x in seq[: k + 1])
        plot_assembly_2d(a, r, boxes=boxes, ax=ax, show_forces=True)
        ax.set_title(
            f"step {k + 1}: {r.status}\nmargin {r.margin:.1e}, edge {edge:+.3f}",
            fontsize=9,
        )
        print(f"step {k + 1}: {r.status}  margin={r.margin:.3e}  edge={edge:+.4f}")

    path = os.path.join(args.out, f"steps_n{n}.png")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
