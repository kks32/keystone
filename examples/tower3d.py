"""Render a 3D box tower with keystone and its visualization layer.

Scene: three unit cubes stacked, the middle one rotated 45 degrees about
z (its shared faces clip to an octagonal patch), plus a 2x1x1 slab on
top offset in x. The assembly uses an inscribed 8-facet pyramid cone.
Two views are saved: the default oblique view and a higher top-ish view.

As with stack2d.py, this reports which piece is missing and exits 1
while any of the geometry, mechanics, or solve layers is still a stub.

Run: python examples/tower3d.py --out ./out
"""

import argparse
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from keystone import viz
from keystone.geometry import Box, Tolerances, build_assembly
from keystone.mechanics import assemble

# Assumed solver API in keystone.solve.batch_jax, owned by another agent.
# Adjust the resolvers and _call_solver if names or signatures differ.
P0_NAMES = ("solve_p0", "p0", "solve_feasibility", "feasibility")
P2_NAMES = ("solve_p2", "p2", "solve_load_factor", "load_factor")

MU = 0.6
PYRAMID_K = 8


def _resolve(module, names):
    for name in names:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    raise NotImplementedError(f"no solver among {names} on {module.__name__}")


def _get_solvers():
    from keystone.solve import batch_jax as backend

    return _resolve(backend, P0_NAMES), _resolve(backend, P2_NAMES)


def _call_solver(fn, system, tol):
    try:
        return fn(system, tol)
    except TypeError:
        return fn(system)


def _quat_z(angle):
    """Unit quaternion (w, x, y, z) for a rotation about the z axis."""
    return np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])


def tower_boxes():
    """Three unit cubes (middle rotated 45 deg) plus an offset slab."""
    cube = np.array([0.5, 0.5, 0.5])
    return [
        Box(cube, np.array([0.0, 0.0, 0.5])),
        Box(cube, np.array([0.0, 0.0, 1.5]), _quat_z(np.pi / 4.0)),
        Box(cube, np.array([0.0, 0.0, 2.5])),
        Box(np.array([1.0, 0.5, 0.5]), np.array([0.3, 0.0, 3.5])),
    ]


def run_all(out_dir, tol):
    p0, p2 = _get_solvers()
    boxes = tower_boxes()
    assembly = build_assembly(boxes, mu=MU, tol=tol, dim=3)
    system = assemble(assembly, tol, cone="pyramid", k=PYRAMID_K)
    r0 = _call_solver(p0, system, tol)
    r2 = _call_solver(p2, system, tol)

    lam = r2.lambda_assoc if r2 is not None else None
    lam_str = f"{lam:.4g}" if lam is not None else "n/a"
    print(
        f"tower3d: status={r0.status} margin={r0.margin:.4g} "
        f"lambda_assoc={lam_str}"
    )

    fig = viz.plot_assembly_3d(assembly, r0, boxes=boxes, title="tower3d")
    viz.save_fig(fig, os.path.join(out_dir, "tower3d.png"))
    plt.close(fig)

    fig = viz.plot_assembly_3d(
        assembly, r0, boxes=boxes, elev=60, azim=-60, title="tower3d (top-ish)"
    )
    viz.save_fig(fig, os.path.join(out_dir, "tower3d_top.png"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render a 3D box tower.")
    parser.add_argument("--out", default="./out", help="output directory")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tol = Tolerances()
    try:
        run_all(args.out, tol)
    except NotImplementedError as exc:
        print(f"[tower3d] a required piece is not implemented yet: {exc}")
        print("[tower3d] fill the geometry, mechanics, and solve stubs, then rerun.")
        return 1
    print(f"[tower3d] wrote scenes to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
