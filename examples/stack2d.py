"""Render 2D box-stack scenes with keystone and its visualization layer.

Scenes: an aligned tower, an offset stacked pair at and past the b/2
limit, and a four-block corbel at and past the 25/24 overhang limit.
Each scene is built with the real public API, solved for P0 and P2, and
saved as a PNG. A one-line summary prints per scene.

The geometry, mechanics, and solve layers are filled by other agents.
While any of them is still a stub (NotImplementedError), this script
reports which piece is missing and exits 1, so the integrator can run
it piecemeal.

Run: python examples/stack2d.py --out ./out
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from keystone import viz
from keystone.geometry import Tolerances, box_2d, build_assembly
from keystone.mechanics import assemble

# Assumed solver API in keystone.solve.batch_jax, owned by another agent.
# If names or signatures differ, adjust the two resolvers below and
# _call_solver; nothing else in this file changes.
P0_NAMES = ("solve_p0", "p0", "solve_feasibility", "feasibility")
P2_NAMES = ("solve_p2", "p2", "solve_load_factor", "load_factor")

MU = 0.6


def _resolve(module, names):
    for name in names:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    raise NotImplementedError(
        f"no solver among {names} on {module.__name__}"
    )


def _get_solvers():
    from keystone.solve import batch_jax as backend

    return _resolve(backend, P0_NAMES), _resolve(backend, P2_NAMES)


def _call_solver(fn, system, tol):
    try:
        return fn(system, tol)
    except TypeError:
        return fn(system)


def tower_boxes():
    """Five aligned unit blocks."""
    return [box_2d(1.0, 1.0, 0.0, 0.5 + k) for k in range(5)]


def offset_boxes(e):
    """Two unit blocks, the upper one offset by e in x."""
    return [box_2d(1.0, 1.0, 0.0, 0.5), box_2d(1.0, 1.0, e, 1.5)]


def corbel_boxes(c):
    """Wide pedestal plus four unit blocks with harmonic overhangs.

    Consecutive shifts from the bottom block upward are c/(2k) with
    k = 4, 3, 2, 1, so c/8, c/6, c/4, c/2. The bottom block right edge
    sits at the pedestal right edge (x = 0) plus c/8. The top block
    right edge is then at c * (1/8 + 1/6 + 1/4 + 1/2) = c * 25/24 beyond
    x = 0. At c = 1 that is 25/24 block widths, the analytic overhang
    limit, so c < 1 is feasible and c > 1 is not.
    """
    b = 1.0
    pedestal = box_2d(3.0, 1.0, -1.5, 0.5)  # spans [-3, 0], right edge x = 0
    x = [0.0, 0.0, 0.0, 0.0]
    # Bottom block centered so its right edge is at 0 + c/8.
    x[0] = -0.5 + c * b / 8.0
    x[1] = x[0] + c * b / 6.0
    x[2] = x[1] + c * b / 4.0
    x[3] = x[2] + c * b / 2.0
    # At c = 1: x[3] = -0.5 + 1/8 + 1/6 + 1/4 + 1/2 = 0.541666...,
    # top right edge = x[3] + 0.5 = 1.041666... = 25/24 beyond x = 0.
    units = [box_2d(1.0, 1.0, x[k], 1.5 + k) for k in range(4)]
    return [pedestal] + units


def render_scene(name, filename, boxes, tol, solvers, out_dir):
    """Build, solve P0 and P2, save the figure, print a summary line."""
    p0, p2 = solvers
    assembly = build_assembly(boxes, mu=MU, tol=tol, dim=2)
    system = assemble(assembly, tol, cone="linear2d")
    r0 = _call_solver(p0, system, tol)
    r2 = _call_solver(p2, system, tol)

    fig = viz.plot_assembly_2d(assembly, r0, boxes=boxes, title=name)
    path = os.path.join(out_dir, filename)
    viz.save_fig(fig, path)
    plt.close(fig)

    lam = r2.lambda_assoc if r2 is not None else None
    lam_str = f"{lam:.4g}" if lam is not None else "n/a"
    print(
        f"{name}: status={r0.status} margin={r0.margin:.4g} "
        f"lambda_assoc={lam_str} -> {filename}"
    )


def run_all(out_dir, tol):
    solvers = _get_solvers()
    render_scene("tower (5 aligned)", "tower.png", tower_boxes(), tol, solvers, out_dir)
    render_scene("offset e=0.45 (feasible)", "offset_ok.png", offset_boxes(0.45), tol, solvers, out_dir)
    render_scene("offset e=0.55 (infeasible)", "offset_fail.png", offset_boxes(0.55), tol, solvers, out_dir)
    render_scene("corbel c=0.98 (feasible)", "corbel_ok.png", corbel_boxes(0.98), tol, solvers, out_dir)
    render_scene("corbel c=1.02 (infeasible)", "corbel_fail.png", corbel_boxes(1.02), tol, solvers, out_dir)


def main():
    parser = argparse.ArgumentParser(description="Render 2D box-stack scenes.")
    parser.add_argument("--out", default="./out", help="output directory")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tol = Tolerances()
    try:
        run_all(args.out, tol)
    except NotImplementedError as exc:
        print(f"[stack2d] a required piece is not implemented yet: {exc}")
        print("[stack2d] fill the geometry, mechanics, and solve stubs, then rerun.")
        return 1
    print(f"[stack2d] wrote scenes to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
