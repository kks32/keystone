"""Throughput and latency benchmark for the keystone JAX backend.

Runs in .venv (py3.12: keystone + jax + qpax). Three scenarios:

a. Single-solve latency. 2D tower of 12 unit blocks and 3D tower of 5
   cubes (pyramid k=8). P4 cold (first call, includes jit compile), P4
   warm (median of 50 after 5 warmups), P2 warm (n_iter=40).
b. Batched P4 throughput. The CLAUDE.md Section 6 target row: 2D, <= 12
   blocks. Batches of a 12-block stack family with small random offsets
   (seed 0). solves/s = B / median wall of 5 calls after one warmup.
c. Scaling. Warm P4 per-solve time versus block count N.

Writes the '## Throughput (recorded run)' section of bench/RESULTS.md.
No number here is quoted anywhere without this script reproducing it.

Run:
    /Users/krishna/research/keystone/.venv/bin/python bench/throughput.py
"""

import os
import statistics
import sys
import time

import numpy as np

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

import keystone  # noqa: E402
from keystone import (  # noqa: E402
    Box,
    Tolerances,
    assemble,
    box_2d,
    build_assembly,
    solve_p2,
    solve_p4,
)
from keystone.solve import margin_batch  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RESULTS.md")
TOL = Tolerances()
SEED = 0

# Timing knobs, recorded in the output so the numbers are reproducible.
WARM_REPS_P4 = 50
WARM_WARMUP_P4 = 5
WARM_REPS_P2 = 5
WARM_WARMUP_P2 = 2
SCALE_REPS = 20
SCALE_WARMUP = 3
BATCH_REPS = 5
BATCH_SIZES = [64, 256, 1024, 4096]
SCALE_NS = [2, 4, 8, 12, 16, 24]


def _median_ms(fn, reps, warmup):
    """Median wall of fn over reps calls, after warmup calls. Milliseconds."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def tower_2d(n):
    """n aligned unit blocks in a column on the ground."""
    return [box_2d(1.0, 1.0, 0.0, 0.5 + k) for k in range(n)]


def tower_3d(n):
    """n aligned unit cubes in a column on the ground."""
    cube = np.array([0.5, 0.5, 0.5])
    return [Box(cube, np.array([0.0, 0.0, 0.5 + k])) for k in range(n)]


def system_2d(n, mu=0.84):
    a = build_assembly(tower_2d(n), mu=mu, tol=TOL, dim=2)
    return assemble(a, TOL, cone="linear2d")


def system_3d(n, mu=0.84, k=8):
    a = build_assembly(tower_3d(n), mu=mu, tol=TOL, dim=3)
    return assemble(a, TOL, cone="pyramid", k=k)


def latency_rows():
    """Cold and warm single-solve latency for the 2D-12 and 3D-5 towers.

    Cold P4 must be the first solve of each shape in this process, so the
    jit compile lands in the cold number. This scenario runs before the
    scaling scenario for that reason.
    """
    rows = []

    # 2D tower of 12: cold P4 first, before any other 2D-12 solve.
    s2 = system_2d(12)
    t0 = time.perf_counter()
    solve_p4(s2, TOL)
    cold2 = (time.perf_counter() - t0) * 1e3
    warm2 = _median_ms(lambda: solve_p4(s2, TOL), WARM_REPS_P4, WARM_WARMUP_P4)
    p2_2 = _median_ms(
        lambda: solve_p2(s2, TOL, n_iter=40), WARM_REPS_P2, WARM_WARMUP_P2
    )

    # 3D tower of 5: cold P4 first for this shape.
    s3 = system_3d(5)
    t0 = time.perf_counter()
    solve_p4(s3, TOL)
    cold3 = (time.perf_counter() - t0) * 1e3
    warm3 = _median_ms(lambda: solve_p4(s3, TOL), WARM_REPS_P4, WARM_WARMUP_P4)
    p2_3 = _median_ms(
        lambda: solve_p2(s3, TOL, n_iter=40), WARM_REPS_P2, WARM_WARMUP_P2
    )

    rows.append(["2D tower, 12 blocks", "P4", "cold (incl. jit)", f"{cold2:.1f}"])
    rows.append(
        ["2D tower, 12 blocks", "P4", f"warm median ({WARM_REPS_P4})", f"{warm2:.2f}"]
    )
    rows.append(
        [
            "2D tower, 12 blocks",
            "P2 (n_iter=40)",
            f"warm median ({WARM_REPS_P2})",
            f"{p2_2:.1f}",
        ]
    )
    rows.append(["3D tower, 5 cubes (k=8)", "P4", "cold (incl. jit)", f"{cold3:.1f}"])
    rows.append(
        [
            "3D tower, 5 cubes (k=8)",
            "P4",
            f"warm median ({WARM_REPS_P4})",
            f"{warm3:.2f}",
        ]
    )
    rows.append(
        [
            "3D tower, 5 cubes (k=8)",
            "P2 (n_iter=40)",
            f"warm median ({WARM_REPS_P2})",
            f"{p2_3:.1f}",
        ]
    )
    return rows


def _batch_arrays(B, n=12):
    """A batch of B twelve-block 2D stacks with per-element random offsets.

    Identical pad counts so every element shares shapes and can be stacked.
    Offsets are small (+-0.05) to keep contact topology fixed; magnitudes do
    not affect timing.
    """
    rng = np.random.default_rng(SEED)
    As, ws, Gs = [], [], []
    for _ in range(B):
        off = rng.uniform(-0.05, 0.05, size=n - 1)
        boxes = [box_2d(1.0, 1.0, 0.0, 0.5)]
        x = 0.0
        for k in range(1, n):
            x += off[k - 1]
            boxes.append(box_2d(1.0, 1.0, x, 0.5 + k))
        a = build_assembly(
            boxes, mu=0.84, tol=TOL, dim=2, pad_blocks=n, pad_patches=n, pad_verts=2
        )
        s = assemble(a, TOL, cone="linear2d")
        As.append(s.A)
        ws.append(s.w_dead)
        Gs.append(s.G)
    return jnp.stack(As), jnp.stack(ws), jnp.stack(Gs)


def batch_rows():
    """Batched P4 throughput at each batch size."""
    rows = []
    for B in BATCH_SIZES:
        Ab, wb, Gb = _batch_arrays(B)
        # One warmup call carries the jit compile.
        t0 = time.perf_counter()
        m, _c = margin_batch(Ab, wb, Gb, TOL.eps_reg)
        m.block_until_ready()
        jit_warm = time.perf_counter() - t0
        reps = []
        for _ in range(BATCH_REPS):
            t0 = time.perf_counter()
            m, _c = margin_batch(Ab, wb, Gb, TOL.eps_reg)
            m.block_until_ready()
            reps.append(time.perf_counter() - t0)
        med = statistics.median(reps)
        rows.append(
            [str(B), f"{jit_warm:.2f}", f"{med:.3f}", f"{B / med:.1f}"]
        )
    return rows


def scaling_rows():
    """Warm P4 per-solve time versus block count."""
    rows = []
    for n in SCALE_NS:
        s = system_2d(n)
        ms = _median_ms(lambda: solve_p4(s, TOL), SCALE_REPS, SCALE_WARMUP)
        rows.append(
            [
                str(n),
                str(int(s.A.shape[0])),
                str(int(s.A.shape[1])),
                str(int(s.G.shape[0])),
                f"{ms:.2f}",
            ]
        )
    return rows


def build_section(lat, batch, scale, jinfo):
    parts = []
    parts.append("## Throughput (recorded run)")
    parts.append("")
    parts.append(
        f"Recorded {common.today()}. python {sys.version.split()[0]}, "
        f"keystone {keystone.__version__}, jax {jinfo['version']}, "
        f"qpax (P4 QP engine). Backend device: {jinfo['devices']}. "
        f"Precision float64 (jax_enable_x64)."
    )
    parts.append("")
    parts.append(
        "Command: "
        "`/Users/krishna/research/keystone/.venv/bin/python bench/throughput.py`"
    )
    parts.append("")

    parts.append("### a. Single-solve latency")
    parts.append("")
    parts.append(
        common.md_table(["scene", "problem", "metric", "time (ms)"], lat)
    )
    parts.append("")
    parts.append(
        "Cold includes JAX tracing and XLA compile for that problem shape. "
        "Warm reuses the cached executable."
    )
    parts.append("")

    parts.append("### b. Batched P4 throughput (2D, 12 blocks)")
    parts.append("")
    parts.append(
        "Family of twelve-block 2D stacks, per-element random x offsets "
        f"(seed {SEED}, +-0.05). One jit warmup call, then the median wall "
        f"of {BATCH_REPS} calls. solves/s = B / median wall."
    )
    parts.append("")
    parts.append(
        common.md_table(
            ["batch B", "jit+warmup (s)", "warm median (s)", "solves/s"], batch
        )
    )
    parts.append("")
    parts.append(
        "This is an Apple-silicon CPU. The CLAUDE.md Section 6 target of "
        ">= 5000 P4 solves/s applies to one A100 or 4090 and is NOT VALIDATED "
        "here (see Not validated). vmap on CPU runs the batch elements in "
        "near-lockstep, so wall grows about linearly with B and solves/s is "
        "roughly flat. GPU is expected to change this shape."
    )
    parts.append("")

    parts.append("### c. Scaling: warm P4 versus block count (2D)")
    parts.append("")
    parts.append(
        f"Aligned 2D towers. Warm median of {SCALE_REPS} solves after "
        f"{SCALE_WARMUP} warmups. rows = equilibrium rows, nf = force "
        "unknowns, cone = inequality rows."
    )
    parts.append("")
    parts.append(
        common.md_table(
            ["N blocks", "rows", "nf", "cone rows", "warm P4 (ms)"], scale
        )
    )
    parts.append("")
    return "\n".join(parts)


def main():
    jinfo = common.jax_info(jax)
    print(f"jax {jinfo['version']} devices={jinfo['devices']}")

    print("scenario a: single-solve latency ...")
    lat = latency_rows()
    for r in lat:
        print("  ", r)

    print("scenario c: scaling ...")
    scale = scaling_rows()
    for r in scale:
        print("  ", r)

    print("scenario b: batched throughput ...")
    batch = batch_rows()
    for r in batch:
        print("  ", r)

    common.ensure_scaffold(RESULTS)
    common.upsert_section(
        RESULTS, "throughput", build_section(lat, batch, scale, jinfo)
    )
    print(f"wrote throughput section to {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
