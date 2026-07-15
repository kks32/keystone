"""Parity of keystone against compas_cra on tilt-table scenes.

Runs in .venv-cra (py3.11: keystone AND compas_cra 0.5.0 + pyomo 6.9.1 +
ipopt). Tilt-table towers on a fixed base, tilted about the y axis. Three
scenes:

  A  3-cube tower: fixed base + 2 free cubes. Analytic topple arctan(1/2)
     = 26.565 deg. Slide variant at mu=0.30: arctan(0.30) = 16.699 deg.
  B  5-cube tower: fixed base + 4 free cubes. Analytic arctan(1/4)
     = 14.036 deg.
  C  3-cube tower with the middle (lower free) cube offset +0.2 in x. No
     closed form; keystone is the reference.

Equivalence of the two models (charter reciprocity of the scene, not of
labels): keystone uses an infinite ground plane as the fixed base, so the
lowest cube contacts the ground over its full base rectangle. compas_cra
defines contact by the explicit interface mesh, so the base-to-first-cube
interface is set to that same full base rectangle. Interfaces between free
cubes are the true face overlap. Without this, an offset cube overhangs a
unit support and the toppling edge moves, a scene difference, not a solver
difference.

keystone tilt equivalence: horizontal pseudo-static load factor lambda
maps to a gravity tilt of arctan(lambda). keystone angle = degrees(arctan(
solve_p2.lambda_assoc)). compas_cra angle: bisect the tilt degree on
the cra_solve verdict to 0.05 deg.

Writes the '## compas_cra parity (recorded run)' section of
bench/RESULTS.md.

Run:
    /Users/krishna/research/keystone/.venv-cra/bin/python bench/parity_cra.py
"""

import contextlib
import io
import os
import subprocess
import sys
import time

import numpy as np

import jax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

import keystone  # noqa: E402
from keystone import Box, Tolerances, assemble, build_assembly, solve_p2, solve_p4  # noqa: E402
from keystone.solve import margin_batch  # noqa: E402

from compas.datastructures import Mesh  # noqa: E402
from compas.geometry import Box as CBox  # noqa: E402
from compas.geometry import Frame, Translation  # noqa: E402
from compas_assembly.datastructures import Block  # noqa: E402
from compas_cra.datastructures import CRA_Assembly  # noqa: E402
from compas_cra.equilibrium import cra_solve  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RESULTS.md")
TOL = Tolerances()
BISECT_RES = 0.05  # degrees
TILT_MAX = 60.0

# Scenes: (name, x offsets of free cubes, mu, analytic angle or None).
SCENES = [
    ("A topple (mu=0.84)", [0.0, 0.0], 0.84, np.degrees(np.arctan(0.5))),
    ("A slide (mu=0.30)", [0.0, 0.0], 0.30, np.degrees(np.arctan(0.30))),
    ("B topple (mu=0.84)", [0.0, 0.0, 0.0, 0.0], 0.84, np.degrees(np.arctan(0.25))),
    ("C offset (mu=0.84)", [0.2, 0.0], 0.84, None),
]


# ---------------------------------------------------------------------------
# compas_cra side
# ---------------------------------------------------------------------------
def _quad(corners):
    m = Mesh()
    for i, c in enumerate(corners):
        m.add_vertex(key=i, x=c[0], y=c[1], z=c[2])
    m.add_face([0, 1, 2, 3])
    return m


def _rect_at_z(xlo, xhi, z, half_y=0.5):
    return [[xhi, half_y, z], [xlo, half_y, z], [xlo, -half_y, z], [xhi, -half_y, z]]


def cra_build(xs):
    """Fixed base (node 0) plus free cubes at z centers 1, 2, ... with x
    offsets xs. The base-to-first-cube interface is the first cube's full
    base rectangle (the ground-plane analog). Inter-cube interfaces are the
    face overlap. A base wide enough to contain every interface keeps the
    contacts physical.
    """
    a = CRA_Assembly()
    a.add_block(Block.from_shape(CBox(3, 3, 1)), node=0)  # fixed wide base
    for k, x in enumerate(xs):
        fr = Frame.worldXY().transformed(Translation.from_vector([x, 0, k + 1]))
        a.add_block(Block.from_shape(CBox(1, 1, 1, frame=fr)), node=k + 1)
    a.set_boundary_conditions([0])
    for k in range(len(xs)):
        z = 0.5 + k
        if k == 0:
            xlo, xhi = xs[0] - 0.5, xs[0] + 0.5  # full base of first cube
        else:
            xlo = max(xs[k - 1] - 0.5, xs[k] - 0.5)
            xhi = min(xs[k - 1] + 0.5, xs[k] + 0.5)  # face overlap
        a.add_interfaces_from_meshes([_quad(_rect_at_z(xlo, xhi, z))], k, k + 1)
    return a


def cra_verdict(xs, deg, mu):
    """True iff cra_solve finds an admissible state at this tilt."""
    a = cra_build(xs)
    a.rotate_assembly([0, 0, 0], [0, 1, 0], deg)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            cra_solve(a, mu=mu, density=1.0)
        return True
    except ValueError:
        return False


def cra_angle(xs, mu):
    """Collapse tilt in degrees, bisected on the cra_solve verdict."""
    lo, hi = 0.0, TILT_MAX
    while hi - lo > BISECT_RES:
        mid = 0.5 * (lo + hi)
        if cra_verdict(xs, mid, mu):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# keystone side
# ---------------------------------------------------------------------------
def ks_system(xs, mu, pad_blocks=None, pad_patches=None):
    """Ground plane (node 0) plus free cubes at z centers 0.5, 1.5, ...

    The cubes sit 0.5 lower than the compas_cra free cubes because keystone
    has no base block below them; the scene is a rigid vertical shift, so
    the collapse tilt is unchanged.
    """
    boxes = [
        Box(np.array([0.5, 0.5, 0.5]), np.array([x, 0.0, 0.5 + k]))
        for k, x in enumerate(xs)
    ]
    a = build_assembly(
        boxes, mu=mu, tol=TOL, dim=3, pad_blocks=pad_blocks, pad_patches=pad_patches,
        pad_verts=8,
    )
    return assemble(a, TOL, cone="pyramid", k=8)


def ks_angle(xs, mu):
    s = ks_system(xs, mu)
    r = solve_p2(s, TOL, n_iter=50)
    return float(np.degrees(np.arctan(r.lambda_assoc))), r.lambda_assoc


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------
def cra_solve_walls(xs, mu, degs):
    """Wall of one cra_solve per tilt angle, milliseconds."""
    out = []
    for d in degs:
        t0 = time.perf_counter()
        cra_verdict(xs, d, mu)
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def ks_p4_walls(xs, mu, degs):
    """Warm solve_p4 wall per tilt (lam = tan(theta)), milliseconds."""
    s = ks_system(xs, mu)
    for _ in range(5):  # warmup, carries jit compile
        solve_p4(s, TOL, lam=0.1)
    out = []
    for d in degs:
        lam = float(np.tan(np.radians(d)))
        t0 = time.perf_counter()
        solve_p4(s, TOL, lam=lam)
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def ks_batch_all_scenes():
    """One batch: 3 mu=0.84 scenes x 61 tilt angles (0..60 step 1).

    All scenes padded to common shapes (max blocks and patches) so the
    padded arrays stack. Returns (total_wall_s, per_verdict_ms, count).
    """
    import jax.numpy as jnp

    scenes = [("A", [0.0, 0.0]), ("B", [0.0, 0.0, 0.0, 0.0]), ("C", [0.2, 0.0])]
    pad_b = max(len(xs) for _, xs in scenes)
    pad_p = pad_b  # ground + (n-1) inter-cube = n patches for an n-cube column
    systems = {name: ks_system(xs, 0.84, pad_b, pad_p) for name, xs in scenes}
    degs = list(range(0, 61))
    As, ws, Gs = [], [], []
    for name, _xs in scenes:
        s = systems[name]
        for d in degs:
            lam = float(np.tan(np.radians(d)))
            As.append(s.A)
            ws.append(s.w_dead + lam * s.w_live)
            Gs.append(s.G)
    Ab, wb, Gb = jnp.stack(As), jnp.stack(ws), jnp.stack(Gs)
    count = Ab.shape[0]
    # Warmup carries the jit compile.
    m, _c = margin_batch(Ab, wb, Gb, TOL.eps_reg)
    m.block_until_ready()
    t0 = time.perf_counter()
    m, _c = margin_batch(Ab, wb, Gb, TOL.eps_reg)
    m.block_until_ready()
    total = time.perf_counter() - t0
    return total, total / count * 1e3, count


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def ipopt_version():
    try:
        out = subprocess.check_output(
            ["/opt/homebrew/bin/ipopt", "--version"], stderr=subprocess.STDOUT
        )
        return out.decode().splitlines()[0].strip()
    except Exception:
        return "unknown"


def pkg_versions():
    import compas
    import compas_assembly
    import compas_cra
    import pyomo

    return {
        "compas": compas.__version__,
        "compas_assembly": compas_assembly.__version__,
        "compas_cra": getattr(compas_cra, "__version__", "0.5.0"),
        "pyomo": pyomo.version.version,
        "ipopt": ipopt_version(),
    }


def build_section(parity_rows, timing_rows, batch_summary, vers, jinfo, warnings):
    parts = []
    parts.append("## compas_cra parity (recorded run)")
    parts.append("")
    parts.append(
        f"Recorded {common.today()}. python {sys.version.split()[0]}, "
        f"keystone {keystone.__version__}, jax {jinfo['version']}, "
        f"compas {vers['compas']}, compas_assembly {vers['compas_assembly']}, "
        f"compas_cra {vers['compas_cra']}, pyomo {vers['pyomo']}, "
        f"{vers['ipopt']}."
    )
    parts.append("")
    parts.append(
        "Command: "
        "`/Users/krishna/research/keystone/.venv-cra/bin/python "
        "bench/parity_cra.py`"
    )
    parts.append("")
    parts.append(
        "Tilt-table towers on a fixed base, tilted about y. keystone angle = "
        "arctan(lambda_assoc) from P2 (n_iter=50). compas_cra angle "
        f"bisected on the cra_solve verdict to {BISECT_RES} deg. keystone is "
        "the associative upper estimate, so keystone angle >= compas_cra "
        f"angle minus the {BISECT_RES} deg bisection resolution is expected."
    )
    parts.append("")
    parts.append("### Collapse tilt angles (degrees)")
    parts.append("")
    parts.append(
        common.md_table(
            ["scene", "analytic", "keystone", "compas_cra", "abs diff (deg)"],
            parity_rows,
        )
    )
    parts.append("")

    parts.append("### Per-solve timing")
    parts.append("")
    parts.append(
        "Scene A (mu=0.84) at three tilt angles. One cra_solve (IPOPT NLP "
        "rebuilt per call) versus one warm keystone solve_p4 (qpax QP)."
    )
    parts.append("")
    parts.append(
        common.md_table(
            ["tilt (deg)", "cra_solve (ms)", "keystone P4 warm (ms)"], timing_rows
        )
    )
    parts.append("")
    parts.append(
        "keystone batched: margin_batch over 3 scenes x 61 tilt angles "
        f"(0..60 deg) as one batch of {batch_summary[2]} elements. Total wall "
        f"{batch_summary[0] * 1e3:.1f} ms, {batch_summary[1]:.3f} ms per "
        "verdict (Apple-silicon CPU, one jit warmup excluded)."
    )
    parts.append("")

    if warnings:
        parts.append("### Notes")
        parts.append("")
        for w in warnings:
            parts.append(f"- {w}")
        parts.append("")
    return "\n".join(parts)


def main():
    jinfo = common.jax_info(jax)
    vers = pkg_versions()
    print("versions:", vers)

    parity_rows = []
    warnings = []
    for name, xs, mu, analytic in SCENES:
        ks_deg, lam = ks_angle(xs, mu)
        cra_deg = cra_angle(xs, mu)
        diff = abs(ks_deg - cra_deg)
        parity_rows.append(
            [
                name,
                f"{analytic:.3f}" if analytic is not None else "n/a",
                f"{ks_deg:.3f}",
                f"{cra_deg:.3f}",
                f"{diff:.3f}",
            ]
        )
        print(
            f"{name}: keystone {ks_deg:.3f} (lam {lam:.4f}), "
            f"cra {cra_deg:.3f}, diff {diff:.3f}"
        )
        # Ordering check: keystone (upper estimate) >= cra minus resolution.
        if ks_deg < cra_deg - BISECT_RES:
            warnings.append(
                f"{name}: keystone {ks_deg:.3f} deg below compas_cra "
                f"{cra_deg:.3f} deg by more than the bisection resolution; "
                "unexpected for an associative upper estimate."
            )
        if diff > 0.5:
            warnings.append(
                f"{name}: |keystone - cra| = {diff:.3f} deg exceeds 0.5 deg."
            )

    # Timing head to head on scene A (mu=0.84).
    degs = [10.0, 25.0, 40.0]
    cra_ms = cra_solve_walls([0.0, 0.0], 0.84, degs)
    ks_ms = ks_p4_walls([0.0, 0.0], 0.84, degs)
    timing_rows = [
        [f"{d:.0f}", f"{c:.1f}", f"{k:.2f}"]
        for d, c, k in zip(degs, cra_ms, ks_ms)
    ]
    print("timing (deg, cra_ms, ks_ms):", timing_rows)

    batch_summary = ks_batch_all_scenes()
    print(
        f"batched: {batch_summary[2]} verdicts in "
        f"{batch_summary[0] * 1e3:.1f} ms, {batch_summary[1]:.3f} ms each"
    )

    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print("  ", w)

    common.ensure_scaffold(RESULTS)
    common.upsert_section(
        RESULTS,
        "parity",
        build_section(parity_rows, timing_rows, batch_summary, vers, jinfo, warnings),
    )
    print(f"wrote parity section to {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
