"""Three hand-made 3D lattice scenes, solved by the env and the host.

No search. This checks that keystone.search.lattice3d.build_system agrees
with the certified host pipeline (build_assembly + assemble + margin_core)
on concrete 3D cube stacks, and renders each scene with keystone.viz.

For every scene the script:
  1. builds the pedestal-plus-cubes box list (the host scene),
  2. solves the P4 elastic margin through the host pipeline,
  3. builds the same state in the lattice3d environment and solves its P4
     margin through build_system + margin_core,
  4. prints both margins and asserts they agree to 1e-6,
  5. renders the host assembly with the P4 forces to out/search3d/.

Scenes:
  tower  -- four unit cubes stacked in a straight column.
  slab   -- a 2x2 layer-0 slab (cubes strictly separated) plus one
            layer-1 cube offset in +x for a small overhang.
  corbel -- a two-step overhang stepping half a block in +x per layer.

Run: python examples/overhang3d_demo.py --out out/search3d
"""

import argparse
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from keystone import Tolerances, build_assembly, solve_p4
from keystone.geometry import Box
from keystone.mechanics import assemble
from keystone.solve.batch_jax import margin_core
from keystone.search import lattice3d as L3
from keystone.viz import plot_assembly_3d, save_fig

DX = 1.0 / 12.0
MU = 0.7
SOLVER_TOL = 1e-9
MAX_ITER = 100

# Scenes as lists of (layer, ix, iy) placements on the default grid.
SCENES = {
    "tower": [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)],
    "slab": [
        (0, -30, -18),
        (0, -6, -18),
        (0, -30, 18),
        (0, -6, 18),
        (1, -3, 18),
    ],
    "corbel": [(0, -12, 0), (1, -6, 0), (2, 0, 0)],
}


def host_boxes(key):
    """Pedestal plus one unit cube per placement, in sorted (L, ix, iy) order."""
    boxes = [Box(np.array([3.0, 3.0, 0.5]), np.array([-3.0, 0.0, 0.5]), density=2000.0)]
    for (L, ix, iy) in sorted(key):
        boxes.append(
            Box(
                np.array([0.5, 0.5, 0.5]),
                np.array([ix * DX, iy * DX, 1.5 + L]),
                density=2000.0,
            )
        )
    return boxes


def solve_scene(name, placements, tol, out_dir):
    """Solve one scene by both pipelines, print margins, render, return agreement."""
    key = tuple(sorted(placements))
    spec = L3.LatticeSpec3D(n_max=len(key), dx=DX)

    # Host pipeline: assembly, pyramid cone, P4 elastic margin.
    boxes = host_boxes(key)
    asm = build_assembly(
        boxes, mu=MU, tol=tol, dim=3,
        pad_blocks=spec.n_blocks, pad_patches=spec.P_max, pad_verts=spec.V,
    )
    hsys = assemble(asm, tol, cone="pyramid", k=8)
    host_margin = float(
        margin_core(
            hsys.A, hsys.w_dead, hsys.G, tol.eps_reg,
            solver_tol=SOLVER_TOL, max_iter=MAX_ITER,
        )[0]
    )
    result = solve_p4(hsys, tol)

    # Env pipeline: build_system on the same state, same P4 margin.
    state = L3.state_from_placements(spec, key)
    A, w, G, L, W = L3.build_system(spec, state)
    env_margin = float(
        margin_core(A, w, G, tol.eps_reg, solver_tol=SOLVER_TOL, max_iter=MAX_ITER)[0]
    )

    diff = abs(env_margin - host_margin)
    overhang = L3.overhang(key, DX)
    print(
        f"{name:7s}: cubes={len(key)} overhang={overhang:+.4f}  "
        f"status={result.status}  env_margin={env_margin:.3e}  "
        f"host_margin={host_margin:.3e}  |diff|={diff:.2e}"
    )

    # Render the host assembly with the P4 force state.
    fig = plot_assembly_3d(
        asm, result, boxes=boxes,
        title=f"overhang3d: {name} (overhang {overhang:+.3f})",
    )
    png = os.path.join(out_dir, f"{name}.png")
    save_fig(fig, png)
    plt.close(fig)
    print(f"         wrote {png}")
    return diff


def main():
    parser = argparse.ArgumentParser(description="3D lattice env-vs-host demo.")
    parser.add_argument("--out", default="out/search3d", help="output directory")
    parser.add_argument(
        "--tol", type=float, default=1e-6, help="env-host margin agreement tolerance"
    )
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    tol = Tolerances()

    print("keystone 3D lattice environment vs certified host pipeline")
    print(f"pyramid cone k=8, mu={MU}, dx=1/{round(1 / DX)}")
    print("")

    max_diff = 0.0
    for name, placements in SCENES.items():
        max_diff = max(max_diff, solve_scene(name, placements, tol, out_dir))

    print("")
    print(f"max env-host margin difference over all scenes: {max_diff:.2e}")
    if max_diff < args.tol:
        print(f"AGREEMENT OK (< {args.tol:.0e})")
        return 0
    print(f"AGREEMENT FAILED (>= {args.tol:.0e})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
