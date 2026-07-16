"""MuJoCo scene exchange and the settle-test oracle (PLAN.md Section 8.1, 8.2).

mujoco is imported lazily inside each function. The core package never needs it.

Scene convention. keystone boxes are 3D rigid boxes; the 2D cube-stacking slice
is a set of unit-depth boxes in the xz plane, which drop into MuJoCo directly as
3D bodies. Node 0 (the ground) maps to a MuJoCo plane at z = 0. Every block is a
free body unless declared static. Gravity is MuJoCo's default (0, 0, -9.81),
which matches gravity along -z.

Contacts are explicit `<pair>` elements only. Default geom collision is turned
off (contype = conaffinity = 0) so MuJoCo's friction-combination rules never
apply; each pair carries our mu directly with condim = 3 (sliding friction, the
point-friction vertex model). Newton solver, small timestep.

solref/solimp policy (documented once, here). Pairs use MuJoCo defaults unless
the `solref` and `solimp` kwargs are set. Defaults are soft-ish contacts; a
knife-edge assembly (keystone margin near zero) can sag and topple under them.
That is a margin finding, not a solver bug: the certified optima are exact
limit states with no margin to spend on compliance. Pass stiffer solref (for
example (0.001, 1.0)) to probe sensitivity.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ..geometry.boxes import Box

_GROUND = "ground"


def _fmt(x: float) -> str:
    """Shortest string that round-trips the float64 exactly. repr gives the
    minimal exact representation, so poses reproduce to machine precision while
    the XML stays readable."""
    return repr(float(x))


def _vec(a: Iterable[float]) -> str:
    return " ".join(_fmt(v) for v in a)


def assembly_diagonal(boxes: Sequence[Box]) -> float:
    """L: bounding-box diagonal over all block corners. Meters."""
    corners = np.concatenate([b.corners() for b in boxes], axis=0)
    diag = float(np.linalg.norm(corners.max(axis=0) - corners.min(axis=0)))
    if diag <= 0.0:
        raise ValueError("degenerate assembly: zero bounding-box diagonal")
    return diag


def _aabb(box: Box) -> tuple[np.ndarray, np.ndarray]:
    """World axis-aligned bounding box (min, max) of a box's corners."""
    c = box.corners()
    return c.min(axis=0), c.max(axis=0)


def aabb_adjacent_pairs(
    boxes: Sequence[Box], gap: float
) -> list[tuple[int, int]]:
    """Index pairs (i, j), i < j, whose AABBs overlap when each is inflated by
    gap on every axis. Stacked boxes touch face-to-face (zero true gap), so any
    positive inflation catches every real contact. Over-inclusion is safe:
    MuJoCo's narrow phase drops pairs that are not in contact."""
    aabbs = [_aabb(b) for b in boxes]
    pairs: list[tuple[int, int]] = []
    for i in range(len(boxes)):
        lo_i, hi_i = aabbs[i]
        for j in range(i + 1, len(boxes)):
            lo_j, hi_j = aabbs[j]
            if np.all(hi_i + gap >= lo_j) and np.all(hi_j + gap >= lo_i):
                pairs.append((i, j))
    return pairs


def to_mjcf(
    boxes: Sequence[Box],
    mu: float,
    *,
    free: Iterable[int] | None = None,
    timestep: float = 5e-4,
    aabb_gap: float = 1e-3,
    all_pairs: bool = False,
    solref: tuple[float, float] | None = None,
    solimp: tuple[float, ...] | None = None,
    condim: int = 3,
    extra_worldbody: str = "",
    extra_equality: str = "",
    model_name: str = "keystone",
) -> str:
    """Export boxes to an MJCF string (PLAN.md Section 8.1).

    boxes: the blocks, indexed 0..N-1. Body `block{i}`, geom `geom{i}`.
    mu: friction on every pair.
    free: indices that get a free joint. None means all boxes free. Indices not
          listed become static (welded to the world, no joint).
    timestep: MuJoCo integrator step, seconds.
    aabb_gap: AABB inflation for adjacency detection, meters.
    all_pairs: if True, emit a pair for every geom-geom combination (used by the
          insertion demo where the moving block's contacts are not known from the
          start pose). Default emits AABB-adjacent pairs only.
    solref, solimp: contact stiffness overrides. None keeps MuJoCo defaults.
    condim: contact dimensionality, 3 = sliding friction only.
    extra_worldbody, extra_equality: raw XML appended inside <worldbody> and
          <equality> (mocap bodies and weld constraints for the insertion demo).
    """
    boxes = list(boxes)
    n = len(boxes)
    if n == 0:
        raise ValueError("to_mjcf needs at least one box")
    if free is None:
        free_set = set(range(n))
    else:
        free_set = set(int(i) for i in free)
        for i in free_set:
            if not 0 <= i < n:
                raise ValueError(f"free index {i} out of range 0..{n - 1}")

    # A ground plane sized to the scene so contacts always have support under
    # every block.
    L = assembly_diagonal(boxes)
    plane_half = max(2.0 * L, 5.0)

    solref_attr = f' solref="{_vec(solref)}"' if solref is not None else ""
    solimp_attr = f' solimp="{_vec(solimp)}"' if solimp is not None else ""
    fric = f"{_fmt(mu)} {_fmt(mu)} 0.005 0.0001 0.0001"

    lines: list[str] = []
    lines.append(f'<mujoco model="{model_name}">')
    lines.append(
        f'  <option timestep="{_fmt(timestep)}" solver="Newton" '
        f'integrator="implicitfast"/>'
    )
    lines.append("  <worldbody>")
    lines.append(
        f'    <geom name="{_GROUND}" type="plane" '
        f'size="{_fmt(plane_half)} {_fmt(plane_half)} 0.1" '
        f'contype="0" conaffinity="0"/>'
    )
    for i, b in enumerate(boxes):
        joint = "      <freejoint/>\n" if i in free_set else ""
        lines.append(
            f'    <body name="block{i}" pos="{_vec(b.position)}" '
            f'quat="{_vec(b.quat)}">'
        )
        if joint:
            lines.append(joint.rstrip("\n"))
        lines.append(
            f'      <geom name="geom{i}" type="box" '
            f'size="{_vec(b.half_extents)}" density="{_fmt(b.density)}" '
            f'contype="0" conaffinity="0"/>'
        )
        lines.append("    </body>")
    if extra_worldbody:
        lines.append(extra_worldbody)
    lines.append("  </worldbody>")

    if extra_equality:
        lines.append("  <equality>")
        lines.append(extra_equality)
        lines.append("  </equality>")

    lines.append("  <contact>")
    # Ground pair for every block. A block that does not touch the ground simply
    # generates no contact.
    for i in range(n):
        lines.append(
            f'    <pair geom1="{_GROUND}" geom2="geom{i}" '
            f'condim="{condim}" friction="{fric}"{solref_attr}{solimp_attr}/>'
        )
    if all_pairs:
        bb = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        bb = aabb_adjacent_pairs(boxes, aabb_gap)
    for i, j in bb:
        lines.append(
            f'    <pair geom1="geom{i}" geom2="geom{j}" '
            f'condim="{condim}" friction="{fric}"{solref_attr}{solimp_attr}/>'
        )
    lines.append("  </contact>")
    lines.append("</mujoco>")
    return "\n".join(lines)


def _mat_to_quat(mat9: np.ndarray) -> np.ndarray:
    """Row-major 3x3 rotation (as length-9) to unit quaternion (w, x, y, z)."""
    import mujoco

    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(mat9, dtype=np.float64).ravel())
    n = np.linalg.norm(quat)
    return quat / n if n > 0 else quat


def from_mjcf(xml_or_model) -> list[Box]:
    """Read box geoms from an MJCF source into keystone Boxes (PLAN.md 8.1).

    Accepts an MJCF string, a path to an .xml file, or an mujoco.MjModel.
    Plane geoms (the ground, node 0) are skipped. Any other non-box geom raises
    a ValueError. Box world pose comes from the forward-kinematics geom frame,
    so the round trip to_mjcf -> from_mjcf reproduces poses to machine precision.
    Density is recovered as body mass over box volume (one geom per body).
    """
    import mujoco

    if isinstance(xml_or_model, mujoco.MjModel):
        model = xml_or_model
    elif isinstance(xml_or_model, str):
        if "<mujoco" in xml_or_model:
            model = mujoco.MjModel.from_xml_string(xml_or_model)
        else:
            model = mujoco.MjModel.from_xml_path(xml_or_model)
    else:
        raise TypeError(
            "from_mjcf expects an MJCF string, a path, or an mujoco.MjModel, "
            f"got {type(xml_or_model)!r}"
        )

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    boxes: list[Box] = []
    for g in range(model.ngeom):
        gtype = int(model.geom_type[g])
        if gtype == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        if gtype != int(mujoco.mjtGeom.mjGEOM_BOX):
            name = model.geom(g).name or f"geom{g}"
            raise ValueError(
                f"from_mjcf supports box geoms only; geom {name!r} has type "
                f"{mujoco.mjtGeom(gtype).name}"
            )
        half = np.asarray(model.geom_size[g], dtype=np.float64).copy()
        pos = np.asarray(data.geom_xpos[g], dtype=np.float64).copy()
        quat = _mat_to_quat(data.geom_xmat[g])
        bid = int(model.geom_bodyid[g])
        volume = 8.0 * float(np.prod(half))
        density = float(model.body_mass[bid]) / volume
        boxes.append(Box(half, pos, quat, density))
    return boxes


def _quat_angle(q0: np.ndarray, q1: np.ndarray) -> float:
    """Geodesic angle in radians between two unit quaternions."""
    dot = abs(float(np.dot(q0, q1)))
    dot = min(1.0, dot)
    return 2.0 * float(np.arccos(dot))


def settle_test(
    boxes: Sequence[Box],
    mu: float,
    *,
    duration: float = 2.0,
    disp_tol_rel: float = 0.005,
    rot_tol: float = 0.01,
    timestep: float = 5e-4,
    free: Iterable[int] | None = None,
    solref: tuple[float, float] | None = None,
    solimp: tuple[float, ...] | None = None,
    sample_every: int = 20,
) -> dict:
    """Dynamic sanity oracle (PLAN.md Section 8.2).

    Build the scene, place blocks at their exact poses, run forward dynamics for
    `duration` seconds with no noise, and report how far the blocks moved.

    Stable iff max block displacement < disp_tol_rel * L and max rotation
    < rot_tol, both measured from the initial pose to the settled pose. L is the
    assembly bounding-box diagonal. The trajectory extrema (the running maxima
    over the whole run) are reported too, so a structure that wobbles and returns
    is distinguishable from one that drifts.

    Deterministic: same inputs, same output. MuJoCo's contact model is soft and
    regularized; a keystone knife-edge optimum (margin near zero) may topple here
    even though it certifies as a feasible limit state. That gap is the finding
    (PLAN.md Section 8.4), reported, never tuned away.
    """
    import mujoco

    boxes = list(boxes)
    n = len(boxes)
    L = assembly_diagonal(boxes)
    xml = to_mjcf(
        boxes, mu, free=free, timestep=timestep, solref=solref, solimp=solimp
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    body_ids = [int(model.body(f"block{i}").id) for i in range(n)]
    pos0 = np.array([data.xpos[b].copy() for b in body_ids])
    quat0 = np.array([data.xquat[b].copy() for b in body_ids])

    n_steps = max(1, int(round(duration / timestep)))
    traj_max_disp = 0.0
    traj_max_rot = 0.0

    def snapshot() -> tuple[float, float]:
        d_max = 0.0
        r_max = 0.0
        for k, b in enumerate(body_ids):
            d = float(np.linalg.norm(data.xpos[b] - pos0[k]))
            r = _quat_angle(quat0[k], data.xquat[b])
            d_max = max(d_max, d)
            r_max = max(r_max, r)
        return d_max, r_max

    for step in range(n_steps):
        mujoco.mj_step(model, data)
        if step % sample_every == 0:
            d_max, r_max = snapshot()
            traj_max_disp = max(traj_max_disp, d_max)
            traj_max_rot = max(traj_max_rot, r_max)

    # Final state.
    per_disp = []
    per_rot = []
    for k, b in enumerate(body_ids):
        per_disp.append(float(np.linalg.norm(data.xpos[b] - pos0[k])))
        per_rot.append(_quat_angle(quat0[k], data.xquat[b]))
    max_disp = max(per_disp)
    max_rot = max(per_rot)
    traj_max_disp = max(traj_max_disp, max_disp)
    traj_max_rot = max(traj_max_rot, max_rot)

    max_disp_rel = max_disp / L
    stable = bool(max_disp_rel < disp_tol_rel and max_rot < rot_tol)

    return {
        "stable": stable,
        "verdict": "stable" if stable else "unstable",
        "max_disp": max_disp,
        "max_disp_rel": max_disp_rel,
        "max_rot": max_rot,
        "L": L,
        "duration": duration,
        "timestep": timestep,
        "n_steps": n_steps,
        "traj_max_disp": traj_max_disp,
        "traj_max_disp_rel": traj_max_disp / L,
        "traj_max_rot": traj_max_rot,
        "per_block_disp": per_disp,
        "per_block_disp_rel": [d / L for d in per_disp],
        "per_block_rot": per_rot,
        "disp_tol_rel": disp_tol_rel,
        "rot_tol": rot_tol,
    }
