"""Placement-motion demo: can the build sequence be executed? (PLAN.md 8.3b).

A static certificate says a block SET stands. It says nothing about the motion
that puts each block in place. This demo executes a build sequence one block at
a time in MuJoCo. Each already-placed block is a free body. The next block is
welded to a mocap body and driven along a straight-line path at constant slow
speed, then released on arrival (weld deactivated), then the structure settles
before the next block.

Two path modes:
- drop: descend vertically from above the target.
- slide: approach laterally at the target layer height from the open (+x) side.

The clamp optimum's final block slides in under an existing bridge; every other
placement drops. The demo reports, per step, whether the block reached its
target, how much the already-placed structure was disturbed during the motion,
the peak contact force, and the post-release settle verdict.

Contacts use a stiff solref so the structures stand (the certified optima are
knife-edge; see mujoco_validate.py). That isolates insertion-motion disturbance
from the structures' own near-zero margin. Run:

    python examples/mujoco_insert.py --out out/mujoco
"""

import argparse
import json
import os

import numpy as np

from keystone import (
    Box,
    Tolerances,
    assemble,
    box_2d,
    build_assembly,
    solve_p0,
    solve_p4,
)
from keystone.interop.mujoco_io import (
    assembly_diagonal,
    capped_impedance_wrench,
    restacked_cubes,
    to_mjcf,
)

DROP = "drop"
SLIDE = "slide"

# Stiff contacts so the knife-edge structures stand while we test the motion.
INSERT_SOLREF = (0.002, 1.0)

# Gravity used for the force cap (matches MuJoCo default (0, 0, -9.81)).
G = 9.81

TOL = Tolerances()

# Impedance-driver defaults that reach the minimum insertion tolerance in the
# sweep. Reported by run_route_a; see docs/KNOWN_LIMITS.md for the finding.
IMPEDANCE = dict(
    kp=6.0e5,
    kd=1.8e5,
    kp_rot=1.2e6,
    kd_rot=3.0e5,
    max_push_x=5.0,   # cap on the driver push, in block weights
    speed=0.15,       # lateral/vertical approach speed, m/s
    z_bias=-0.003,    # bias the target down so the block rides the lower face
)

# The two counterweighted reacher designs, in certified build order. cells are
# (layer, grid_index_j); center x is j * dx. The last cell is the reacher, which
# must enter under a clamping block one layer above it.
DESIGNS = {
    "clamp_31_24": dict(
        dx=1.0 / 24.0,
        cells=[(0, -2), (1, -14), (2, -4), (1, 19)],
        reacher=3,
        overhang=31.0 / 24.0,
        mu=0.7,
    ),
    "n6_4_3": dict(
        dx=1.0 / 12.0,
        cells=[(0, 0), (1, -6), (0, -36), (2, -1), (3, -4), (1, 10)],
        reacher=5,
        overhang=4.0 / 3.0,
        mu=0.7,
    ),
}


def _fmt(x):
    return "%.17g" % float(x)


def _vec(a):
    return " ".join(_fmt(v) for v in a)


def pedestal6():
    return box_2d(6.0, 1.0, -3.0, 0.5)


def cube(layer, x):
    return box_2d(1.0, 1.0, x, 1.5 + layer)


def insert_step(
    placed_boxes,
    target_box,
    mu,
    mode,
    *,
    solref=INSERT_SOLREF,
    timestep=1e-3,
    speed=0.5,
    drop_height=0.8,
    slide_offset=1.0,
    hold_time=0.4,
    settle_time=1.0,
    disturb_tol_rel=0.01,
):
    """Drive one block into place, release, and settle. Returns a report dict
    and the settled Box list (placed + new) for the next step."""
    import mujoco

    k = len(placed_boxes)
    if mode == DROP:
        start_pos = target_box.position + np.array([0.0, 0.0, drop_height])
    elif mode == SLIDE:
        start_pos = target_box.position + np.array([slide_offset, 0.0, 0.0])
    else:
        raise ValueError(f"mode must be {DROP!r} or {SLIDE!r}, got {mode!r}")

    new_start = Box(
        target_box.half_extents, start_pos, target_box.quat, target_box.density
    )
    boxes = list(placed_boxes) + [new_start]
    L = assembly_diagonal(list(placed_boxes) + [target_box])

    mocap = (
        f'    <body name="mocap" pos="{_vec(start_pos)}" '
        f'quat="{_vec(target_box.quat)}" mocap="true"/>'
    )
    weld = (
        f'    <weld name="insertweld" body1="mocap" '
        f'body2="block{k}" active="true"/>'
    )
    xml = to_mjcf(
        boxes,
        mu,
        timestep=timestep,
        all_pairs=True,
        solref=solref,
        extra_worldbody=mocap,
        extra_equality=weld,
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    ids = [int(model.body(f"block{i}").id) for i in range(k + 1)]
    placed_ids = ids[:k]
    new_id = ids[k]
    new_geom = int(model.geom(f"geom{k}").id)
    placed_geoms = {int(model.geom(f"geom{i}").id) for i in range(k)}
    mocap_id = int(model.body_mocapid[int(model.body("mocap").id)])
    pos0 = {i: data.xpos[i].copy() for i in placed_ids}

    def structure_disp():
        if not placed_ids:
            return 0.0
        return max(
            float(np.linalg.norm(data.xpos[i] - pos0[i])) for i in placed_ids
        )

    def total_contact_force():
        buf = np.zeros(6)
        tot = 0.0
        for c in range(data.ncon):
            mujoco.mj_contactForce(model, data, c, buf)
            tot += float(np.linalg.norm(buf[:3]))
        return tot

    def start_penetration():
        # Deepest penetration between the moving block and any placed block at
        # the current (start) pose. A drop from above whose column is blocked by
        # an overhanging placed block starts already inside it.
        pen = 0.0
        for c in range(data.ncon):
            con = data.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            if new_geom not in (g1, g2):
                continue
            other = g2 if g1 == new_geom else g1
            if other in placed_geoms and con.dist < 0.0:
                pen = max(pen, -float(con.dist))
        return pen

    # Obstruction check. If the approach start already penetrates a placed
    # block, the straight-line path is blocked and no rigid drive can seat the
    # block without gross overlap. Report it rather than driving through it.
    pen0 = start_penetration()
    if pen0 > 1e-3:
        settled = list(placed_boxes) + [target_box]
        report = {
            "mode": mode,
            "obstructed": True,
            "start_penetration": pen0,
            "reach_err": float(np.linalg.norm(start_pos - target_box.position)),
            "reached": False,
            "disturb_max": 0.0,
            "disturb_max_rel": 0.0,
            "undisturbed": True,
            "peak_contact_force": 0.0,
            "settle_disp": 0.0,
            "settle_disp_rel": 0.0,
            "settle_stable": True,
            "L": L,
            "note": "approach column blocked by an overhanging placed block; "
            "not reachable by a rigid straight-line path",
        }
        return report, settled

    dist = float(np.linalg.norm(target_box.position - start_pos))
    n_motion = max(1, int(round(dist / speed / timestep)))
    max_disturb = 0.0
    max_force = 0.0

    # Motion phase: drive the mocap along the straight-line path.
    for t in range(n_motion):
        a = (t + 1) / n_motion
        data.mocap_pos[mocap_id] = start_pos + a * (target_box.position - start_pos)
        mujoco.mj_step(model, data)
        max_disturb = max(max_disturb, structure_disp())
        max_force = max(max_force, total_contact_force())

    # Hold phase: keep the mocap at the target so the weld converges the block.
    n_hold = max(1, int(round(hold_time / timestep)))
    for _ in range(n_hold):
        data.mocap_pos[mocap_id] = target_box.position.copy()
        mujoco.mj_step(model, data)
        max_disturb = max(max_disturb, structure_disp())
        max_force = max(max_force, total_contact_force())

    reach_err = float(np.linalg.norm(data.xpos[new_id] - target_box.position))

    # Release the weld and settle.
    data.eq_active[0] = 0
    rel_pos = {i: data.xpos[i].copy() for i in ids}
    n_settle = max(1, int(round(settle_time / timestep)))
    settle_disp = 0.0
    for _ in range(n_settle):
        mujoco.mj_step(model, data)
        settle_disp = max(
            settle_disp,
            max(float(np.linalg.norm(data.xpos[i] - rel_pos[i])) for i in ids),
        )

    # Final settled poses carried to the next step.
    settled = []
    for idx, i in enumerate(ids):
        src = boxes[idx]
        pos = np.asarray(data.xpos[i]).copy()
        quat = np.asarray(data.xquat[i]).copy()
        settled.append(Box(src.half_extents, pos, quat, src.density))

    report = {
        "mode": mode,
        "obstructed": False,
        "start_penetration": pen0,
        "reach_err": reach_err,
        "reached": bool(reach_err < 0.02),
        "disturb_max": max_disturb,
        "disturb_max_rel": max_disturb / L,
        "undisturbed": bool(max_disturb / L < disturb_tol_rel),
        "peak_contact_force": max_force,
        "settle_disp": settle_disp,
        "settle_disp_rel": settle_disp / L,
        "settle_stable": bool(settle_disp / L < disturb_tol_rel),
        "L": L,
        "note": "",
    }
    return report, settled


def run_sequence(name, initial, sequence, mu, args):
    """initial: list of Box already present (the foundation). sequence: list of
    (target_box, mode) to insert in order."""
    placed = list(initial)
    steps = []
    print()
    print(f"=== {name} (mu={mu}) ===")
    for step_i, (target, mode) in enumerate(sequence):
        rep, placed = insert_step(
            placed,
            target,
            mu,
            mode,
            timestep=args.timestep,
            speed=args.speed,
        )
        rep["step"] = step_i
        steps.append(rep)
        if rep["obstructed"]:
            print(
                f"  step {step_i} [{mode:5s}] OBSTRUCTED: column blocked "
                f"(start penetration {rep['start_penetration']:.3f} m); "
                f"not reachable by a rigid straight-line drop"
            )
            continue
        flag = "ok" if rep["undisturbed"] else "DISTURBED"
        print(
            f"  step {step_i} [{mode:5s}] reached={rep['reached']} "
            f"(err {rep['reach_err']:.4f})  structure {flag} "
            f"(disp {rep['disturb_max']:.4f} m, {rep['disturb_max_rel']:.4f} L)  "
            f"peak force {rep['peak_contact_force']:.3e} N  "
            f"settle {'stable' if rep['settle_stable'] else 'moved'} "
            f"({rep['settle_disp']:.4f} m)"
        )
    return {"name": name, "mu": mu, "steps": steps, "final_boxes": len(placed)}, placed


def render_final(name, boxes, mu, out_dir):
    """Offscreen render of the final settled structure. Returns the path or a
    reason string on failure. GL is not fought: any error is reported and
    skipped."""
    try:
        import mujoco

        xml = to_mjcf(boxes, mu, solref=INSERT_SOLREF)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, height=480, width=640)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.lookat[:] = [0.0, 0.0, 2.0]
        cam.distance = 12.0
        cam.azimuth = 90.0
        cam.elevation = -10.0
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        renderer.close()
        path = os.path.join(out_dir, f"insert_{name}.png")
        try:
            import matplotlib.image as mpimg

            mpimg.imsave(path, img)
        except ImportError:
            return "no PNG writer (matplotlib) available; render skipped"
        return path
    except Exception as e:  # noqa: BLE001
        return f"render skipped ({type(e).__name__}: {e})"


# --------------------------------------------------------------------------
# Route A: block tolerance + compliant insertion.
#
# The reacher of a counterweighted clamp enters under a block one layer above
# it. A `size_tol` block tolerance shrinks each cube and re-stacks the shrunk
# cubes so vertical contacts still meet (interop.mujoco_io.restacked_cubes).
# The rigid weld drive of the demo above is replaced by a capped impedance
# driver so a wedged block stalls at `max_push` instead of destroying the
# structure.
# --------------------------------------------------------------------------


def certify(boxes, mu):
    """Certified P0 verdict and P4 margin from the host pipeline."""
    a = build_assembly(boxes, mu=mu, tol=TOL, dim=2)
    s = assemble(a, TOL, cone="linear2d")
    return solve_p0(s, TOL).status, float(solve_p4(s, TOL).margin)


def recertify_shrunk(spec, size_tol):
    """Re-certify the shrunk, re-stacked design and every build-order prefix.

    Reports the full-design margin and the status of each prefix, so an
    intermediate prefix that a tolerance turns infeasible is visible."""
    cubes = restacked_cubes(spec["cells"], size_tol, spec["dx"])
    full = [pedestal6()] + cubes
    fs, fm = certify(full, spec["mu"])
    prefixes = []
    for i in range(1, len(cubes) + 1):
        st, mg = certify([pedestal6()] + cubes[:i], spec["mu"])
        prefixes.append(
            {"n": i, "cell": list(spec["cells"][i - 1]), "status": st, "margin": mg}
        )
    return {"full_status": fs, "full_margin": fm, "prefixes": prefixes}


def clearance_geometry(spec, size_tol):
    """Static check of the clearance geometry: keep every block at its nominal
    unit height and lower only the reacher to height 1 - size_tol, resting on
    its base. That opens a `size_tol` gap under the clamping block, the exact
    clearance a rigid slide-in needs. Certify it."""
    cells, dx, ri = spec["cells"], spec["dx"], spec["reacher"]
    boxes = [pedestal6()]
    for i, (layer, j) in enumerate(cells):
        if i == ri:
            hr = 1.0 - size_tol
            base_top = 1.0 + layer  # nominal unit layer below the reacher top
            boxes.append(box_2d(1.0, hr, j * dx, base_top + hr / 2.0))
        else:
            boxes.append(box_2d(1.0, 1.0, j * dx, 1.5 + layer))
    st, mg = certify(boxes, spec["mu"])
    return {"gap": size_tol, "status": st, "margin": mg}


def _drive(model, data, body_id, imp, weight, max_torque):
    """One impedance step for the moving body. imp holds the target pose and
    the gains. Returns the applied push magnitude."""
    import mujoco

    dof = int(model.body_dofadr[body_id])
    R = data.xmat[body_id].reshape(3, 3)
    linvel = data.qvel[dof : dof + 3].copy()          # world frame
    angvel = R @ data.qvel[dof + 3 : dof + 6]         # local -> world
    force, torque = capped_impedance_wrench(
        data.xpos[body_id],
        data.xquat[body_id],
        imp["target_pos"],
        imp["target_quat"],
        linvel,
        angvel,
        kp=imp["kp"],
        kd=imp["kd"],
        kp_rot=imp["kp_rot"],
        kd_rot=imp["kd_rot"],
        max_push=imp["max_push"],
        max_torque=max_torque,
    )
    data.xfrc_applied[body_id, :3] = force
    data.xfrc_applied[body_id, 3:] = torque
    mujoco.mj_step(model, data)
    return float(np.linalg.norm(force))


def compliant_insert(
    placed_boxes,
    target_box,
    mode,
    mu,
    *,
    params=IMPEDANCE,
    static_boxes=(),
    timestep=5e-4,
    slide_off=0.9,
    drop_h=0.3,
    hold=0.8,
    settle_time=1.5,
    solref=INSERT_SOLREF,
    stand_disp_rel=0.01,
    stand_rot=0.05,
):
    """Drive one block into place with the capped impedance driver.

    placed_boxes: free bodies already in place (exact or settled poses).
    static_boxes: extra static support bodies (a falsework prop), welded.
    mode: DROP (descend from above) or SLIDE (approach laterally from +x).
    During transit the target z carries params["z_bias"] so the block rides the
    lower face; the hold phase removes the bias and seats the block.

    Returns (report, settled_boxes). settled_boxes are the placed bodies plus
    the moving block at their post-settle poses (supports excluded)."""
    import mujoco

    k = len(placed_boxes)
    tgt = target_box.position
    if mode == DROP:
        start = tgt + np.array([0.0, 0.0, drop_h])
    elif mode == SLIDE:
        start = tgt + np.array([slide_off, 0.0, 0.0])
    else:
        raise ValueError(f"mode must be {DROP!r} or {SLIDE!r}, got {mode!r}")

    moving = Box(target_box.half_extents, start, target_box.quat, target_box.density)
    boxes = list(placed_boxes) + [moving] + list(static_boxes)
    free = list(range(k + 1))  # placed + moving are free; supports are static
    L = assembly_diagonal(list(placed_boxes) + [target_box])

    xml = to_mjcf(
        boxes, mu, timestep=timestep, all_pairs=True, solref=solref, free=free
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    ids = [int(model.body(f"block{i}").id) for i in range(k + 1)]
    new_id = ids[k]
    placed_ids = ids[:k]
    support_geoms = {
        int(model.geom(f"geom{k + 1 + s}").id) for s in range(len(static_boxes))
    }
    weight = float(model.body_mass[new_id]) * G
    max_push = params["max_push_x"] * weight
    max_torque = 0.5 * max_push
    imp = dict(
        kp=params["kp"],
        kd=params["kd"],
        kp_rot=params["kp_rot"],
        kd_rot=params["kd_rot"],
        max_push=max_push,
        target_quat=target_box.quat.copy(),
    )
    pos0 = {i: data.xpos[i].copy() for i in placed_ids}
    quat0 = {i: data.xquat[i].copy() for i in placed_ids}

    def struct_disp():
        if not placed_ids:
            return 0.0
        return max(float(np.linalg.norm(data.xpos[i] - pos0[i])) for i in placed_ids)

    def struct_rot():
        if not placed_ids:
            return 0.0
        out = 0.0
        for i in placed_ids:
            dot = min(1.0, abs(float(np.dot(quat0[i], data.xquat[i]))))
            out = max(out, 2.0 * float(np.arccos(dot)))
        return out

    def contact_force():
        buf = np.zeros(6)
        tot = 0.0
        for c in range(data.ncon):
            mujoco.mj_contactForce(model, data, c, buf)
            tot += float(np.linalg.norm(buf[:3]))
        return tot

    def support_force():
        if not support_geoms:
            return 0.0
        buf = np.zeros(6)
        tot = 0.0
        for c in range(data.ncon):
            con = data.contact[c]
            if int(con.geom1) in support_geoms or int(con.geom2) in support_geoms:
                mujoco.mj_contactForce(model, data, c, buf)
                tot += float(np.linalg.norm(buf[:3]))
        return tot

    dist = float(np.linalg.norm(tgt - start))
    n_motion = max(1, int(round(dist / params["speed"] / timestep)))
    peak_push = peak_contact = max_disturb = peak_support = 0.0
    z_bias = params["z_bias"]

    for t in range(n_motion):
        a = (t + 1) / n_motion
        moving_target = start + a * (tgt - start)
        moving_target = moving_target + np.array([0.0, 0.0, z_bias])
        imp["target_pos"] = moving_target
        peak_push = max(peak_push, _drive(model, data, new_id, imp, weight, max_torque))
        peak_contact = max(peak_contact, contact_force())
        max_disturb = max(max_disturb, struct_disp())
        peak_support = max(peak_support, support_force())

    imp["target_pos"] = tgt.copy()  # remove the bias and seat
    for _ in range(max(1, int(round(hold / timestep)))):
        peak_push = max(peak_push, _drive(model, data, new_id, imp, weight, max_torque))
        peak_contact = max(peak_contact, contact_force())
        max_disturb = max(max_disturb, struct_disp())
        peak_support = max(peak_support, support_force())

    reach_err = float(np.linalg.norm(data.xpos[new_id] - tgt))

    data.xfrc_applied[:] = 0.0
    rel = {i: data.xpos[i].copy() for i in ids}
    settle_disp = 0.0
    for _ in range(max(1, int(round(settle_time / timestep)))):
        mujoco.mj_step(model, data)
        settle_disp = max(
            settle_disp,
            max(float(np.linalg.norm(data.xpos[i] - rel[i])) for i in ids),
        )
        peak_support = max(peak_support, support_force())

    disturb_rel = struct_disp() / L
    rot = struct_rot()
    reached = bool(reach_err < 0.03)
    stands = bool(disturb_rel < stand_disp_rel and rot < stand_rot)

    if reached and stands:
        outcome = "seated"
    elif not reached and stands:
        outcome = "stall"          # block stopped short, structure intact
    elif reached and not stands:
        outcome = "disturbed"      # block reached but structure moved
    else:
        outcome = "collapse"       # block stalled and structure toppled

    settled = []
    for idx, i in enumerate(ids):
        src = boxes[idx]
        settled.append(
            Box(
                src.half_extents,
                np.asarray(data.xpos[i]).copy(),
                np.asarray(data.xquat[i]).copy(),
                src.density,
            )
        )

    report = {
        "mode": mode,
        "reach_err": reach_err,
        "reached": reached,
        "peak_push": peak_push,
        "max_push": max_push,
        "peak_contact_force": peak_contact,
        "peak_support_force": peak_support,
        "struct_disturb": struct_disp(),
        "struct_disturb_rel": disturb_rel,
        "struct_rot": rot,
        "settle_disp": settle_disp,
        "stands": stands,
        "outcome": outcome,
        "L": L,
    }
    return report, settled


def compliant_reacher_slide(spec, size_tol, params=IMPEDANCE):
    """Place the pre-reacher structure at exact re-stacked poses, then slide the
    reacher in with the compliant driver. The pre-reacher structure is
    re-certified stable first; this isolates the reacher insertion from any
    drop disturbance."""
    cubes = restacked_cubes(spec["cells"], size_tol, spec["dx"])
    ri = spec["reacher"]
    pre = [pedestal6()] + cubes[:ri]
    reacher = cubes[ri]
    rep, _ = compliant_insert(pre, reacher, SLIDE, spec["mu"], params=params)
    return rep


def run_route_a(name, args, size_tols=(0.0, 0.005, 0.0075, 0.01, 0.02)):
    """Sweep block tolerance for one design. Re-certify statics, run the
    compliant reacher slide, and find the minimum tolerance that seats the
    reacher and leaves the structure standing."""
    spec = DESIGNS[name]
    print()
    print(f"=== Route A: {name} (overhang {spec['overhang']:.4f}, mu={spec['mu']}) ===")
    print(
        f"{'size_tol':>8s} {'full_cert':>10s} {'full_marg':>10s} "
        f"{'cw_prefix':>10s} {'clr_geom':>10s} {'slide':>10s} "
        f"{'reach':>7s} {'push/cap':>16s} {'disturb':>9s} {'rot':>7s}"
    )
    nominal = recertify_shrunk(spec, 0.0)["full_margin"]
    rows = []
    min_tol = None
    for st in size_tols:
        rc = recertify_shrunk(spec, st)
        # The knife-edge counterweight/support prefix is prefix n=2 in build order.
        cw = next((p for p in rc["prefixes"] if p["n"] == 2), None)
        clr = clearance_geometry(spec, st)
        slide = compliant_reacher_slide(spec, st)
        if slide["outcome"] == "seated" and min_tol is None:
            min_tol = st
        rows.append(
            {
                "size_tol": st,
                "recertify": rc,
                "margin_shift": rc["full_margin"] - nominal,
                "clearance_geometry": clr,
                "slide": slide,
            }
        )
        print(
            f"{st:8.4f} {rc['full_status']:>10s} {rc['full_margin']:10.2e} "
            f"{cw['status']:>10s} {clr['status']:>10s} {slide['outcome']:>10s} "
            f"{slide['reach_err']:7.4f} "
            f"{slide['peak_push']:7.1e}/{slide['max_push']:.1e} "
            f"{slide['struct_disturb_rel']:9.4f} {slide['struct_rot']:7.4f}"
        )
    print(
        f"  minimum size_tol that seats the reacher and stands: "
        f"{min_tol if min_tol is not None else 'NONE in swept range'}"
    )
    return {
        "design": name,
        "overhang": spec["overhang"],
        "mu": spec["mu"],
        "nominal_margin": nominal,
        "min_clean_size_tol": min_tol,
        "rows": rows,
    }


def cap_sweep(name, size_tol=0.005, caps=(1.0, 2.0, 5.0, 10.0)):
    """Vary the force cap at one tolerance. Shows the cap trades collapse
    severity for reach: it never buys both."""
    spec = DESIGNS[name]
    print()
    print(f"force-cap sweep: {name} at size_tol={size_tol}")
    entries = []
    for cx in caps:
        params = dict(IMPEDANCE, max_push_x=cx)
        rep = compliant_reacher_slide(spec, size_tol, params)
        entries.append(
            {
                "max_push_x": cx,
                "outcome": rep["outcome"],
                "reach_err": rep["reach_err"],
                "struct_disturb_rel": rep["struct_disturb_rel"],
                "struct_rot": rep["struct_rot"],
            }
        )
        print(
            f"  cap {cx:4.1f}x weight -> {rep['outcome']:10s} "
            f"reach={rep['reach_err']:.4f} disturb={rep['struct_disturb_rel']:.4f} "
            f"rot={rep['struct_rot']:.4f}"
        )
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/mujoco")
    ap.add_argument("--timestep", type=float, default=1e-3)
    ap.add_argument("--speed", type=float, default=0.5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("mujoco not installed; install keystone[mujoco] to run this demo.")
        return

    results = {}
    renders = {}

    # n=4 MCTS design 7/6: every placement drops.
    n4_seq = [
        (cube(0, -1.0 / 12.0), DROP),
        (cube(1, -0.5), DROP),
        (cube(2, -0.25), DROP),
        (cube(1, 2.0 / 3.0), DROP),
    ]
    res, final = run_sequence("n4_mcts_7_6", [pedestal6()], n4_seq, 0.7, args)
    results["n4_mcts_7_6"] = res
    renders["n4_mcts_7_6"] = render_final("n4_mcts_7_6", final, 0.7, args.out)

    # n=4 clamp 31/24: base, two counterweights, then the reacher slides in.
    dx = 1.0 / 24.0
    clamp_seq = [
        (cube(0, -2 * dx), DROP),
        (cube(1, -14 * dx), DROP),
        (cube(2, -4 * dx), DROP),
        (cube(1, 19 * dx), SLIDE),
    ]
    res, final = run_sequence("clamp_31_24", [pedestal6()], clamp_seq, 0.7, args)
    results["clamp_31_24"] = res
    renders["clamp_31_24"] = render_final("clamp_31_24", final, 0.7, args.out)

    # Route A: block tolerance + compliant insertion, both reacher designs.
    route_a = {}
    for name in ("clamp_31_24", "n6_4_3"):
        route_a[name] = run_route_a(name, args)
    route_a["clamp_31_24"]["cap_sweep"] = cap_sweep("clamp_31_24")

    # Render the failed clamp insertion at the realistic tolerance.
    spec = DESIGNS["clamp_31_24"]
    cubes = restacked_cubes(spec["cells"], 0.005, spec["dx"])
    _, settled = compliant_insert(
        [pedestal6()] + cubes[: spec["reacher"]], cubes[spec["reacher"]], SLIDE, spec["mu"]
    )
    renders["routeA_clamp_s0005"] = render_final(
        "routeA_clamp_s0005", settled, spec["mu"], args.out
    )

    print()
    for name, path in renders.items():
        print(f"render {name}: {path}")

    out = {
        "meta": {
            "timestep": args.timestep,
            "speed": args.speed,
            "solref": list(INSERT_SOLREF),
            "note": "rigid baseline jams; Route A adds block tolerance + a capped "
            "impedance driver",
            "impedance": IMPEDANCE,
        },
        "rigid_baseline": results,
        "route_a": route_a,
        "renders": renders,
    }
    path = os.path.join(args.out, "mujoco_insert.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
