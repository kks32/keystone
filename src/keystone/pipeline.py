"""End-to-end stacking evaluation: search, certify, plan, execute, report.

evaluate_stacking answers one question end to end: given n cubes on a dx grid,
what is the best overhang stack the learned AlphaZero search finds, does it
certify on the exact host pipeline, how would a robot build it, and does the
build stand in MuJoCo. Five stages thread one JSON record:

1. SEARCH   run keystone.search.Search with the learned prior and value heads
            (uniform fallback when the checkpoint is missing), take the best
            sequence and overhang.
2. CERTIFY  replay every prefix through build_assembly + assemble + solve_p0,
            recording per-step P0 margins and the full-state verdict.
3. PLAN     classify each placement at its build time with the lattice
            reachability rule: drop, ride_under, or prop.
4. EXECUTE  drive the build in MuJoCo with a capped impedance driver (drops,
            nose-down ride-under pushes, and falsework props), instrument each
            block, and finish with the stiff-contact settle verdict.
5. REPORT   write the JSON record and an HD movie, and print a one-screen
            summary with the three-way agreement (search claim, certificate,
            physics).

The driver executor is v1. An arm executor is a later plug-in; the record
carries an "executor" field and enough per-block target and protocol detail to
drive one. This module never imports mujoco or flax at import time; both load
lazily inside the stage that needs them.
"""

import json
import os

import numpy as np

from keystone import (
    FEASIBLE,
    Tolerances,
    assemble,
    box_2d,
    build_assembly,
    solve_p0,
)
from keystone.geometry.boxes import Box
from keystone.interop.movies import FrameRecorder

# Frozen scene, unit scale, matching keystone.search.lattice and the mujoco
# examples: pedestal 6 x 1 with right edge at x = 0, unit cubes on the dx grid.
MU = 0.7
G = 9.81

# Capped impedance driver gains, ported from examples/mujoco_insert.IMPEDANCE.
# max_push_x is the push cap in block weights; speed is the approach speed.
IMPEDANCE = dict(
    kp=6.0e5,
    kd=1.8e5,
    kp_rot=1.2e6,
    kd_rot=3.0e5,
    max_push_x=5.0,
    speed=0.35,
    z_bias=-0.003,
)

# Stiff contacts so a certified structure stands while the driver works; the
# same stiffness is the settle verdict. Soft defaults let knife-edge optima sag.
INSERT_SOLREF = (0.002, 1.0)
STIFF_SOLREF = (0.002, 1.0)

# Drop geometry.
DROP_H = 0.15

# Ride-under protocol, from the rideunder study: a small nose-down tilt clears
# the leading top corner under the bridge lip, and the push is capped at a few
# block weights.
RIDE_TILT_DEG = 4.0
RIDE_CAP_X = 4.0
RIDE_SLIDE_OFF = 0.7

# Falsework prop: a slender vertical column under the com-side underside, with
# a catcher relief so the block seats on the structure and tips onto the prop.
PROP_W = 0.2
PROP_RELIEF = 1e-3

EXECUTOR = "impedance_driver"


# --------------------------------------------------------------------------
# Scene geometry.
# --------------------------------------------------------------------------


def pedestal():
    """The pedestal block: 6 wide, 1 tall, right edge at x = 0."""
    return box_2d(6.0, 1.0, -3.0, 0.5)


def cube(layer, x):
    """A unit cube centered at x on the given layer (center z = 1.5 + layer)."""
    return box_2d(1.0, 1.0, x, 1.5 + layer)


# --------------------------------------------------------------------------
# Stage 1: search.
# --------------------------------------------------------------------------


def _run_search(n, dx, sims, seed, checkpoint, tol, progress=None):
    """Run the PUCT search with the learned heads, or uniform on a miss.

    Returns (search, best_overhang, sequence, info). info records whether the
    checkpoint loaded and which prior the run used.
    """
    from keystone.search import az
    from keystone.search.mcts import Search

    info = {"checkpoint": checkpoint, "checkpoint_loaded": False, "prior": "uniform"}
    prior_fn = value_fn = None

    use_learned = n <= az.MAX_LAYERS
    if not use_learned:
        info["note"] = f"n={n} exceeds MAX_LAYERS={az.MAX_LAYERS}; uniform prior"
    elif checkpoint and os.path.exists(checkpoint):
        fs = az.make_feature_spec(dx=dx, max_layers=az.MAX_LAYERS)
        model = az.AZModel(fs, init_seed=0)
        try:
            az.load_params(model, checkpoint)
            prior_fn = az.make_prior_fn(model, n)
            value_fn = az.make_value_fn(model, n)
            info["checkpoint_loaded"] = True
            info["prior"] = "learned"
        except Exception as exc:  # noqa: BLE001
            info["note"] = (
                f"checkpoint load failed ({type(exc).__name__}: {exc}); "
                "uniform prior"
            )
    else:
        info["note"] = "checkpoint missing; uniform prior"

    search = Search(
        n, dx, tol, seed=seed, prior_fn=prior_fn, value_fn=value_fn
    )
    best = search.run(sims, progress=progress)
    seq = [(int(L), int(j)) for (L, j) in search.best_sequence()]
    return search, float(best), seq, info


# --------------------------------------------------------------------------
# Stage 2: certify.
# --------------------------------------------------------------------------


def certify_prefixes(seq, dx, tol, mu=MU):
    """Replay each prefix through the host pipeline and solve P0.

    Returns a dict with per-step (layer, j, x, margin, status), the full-state
    status and margin, prefix_feasible, and on a failure the failing step.
    """
    steps = []
    placed = []
    prefix_feasible = True
    fail_step = None
    for (L, j) in seq:
        placed.append((L, j))
        boxes = [pedestal()] + [cube(pl, pj * dx) for (pl, pj) in placed]
        asm = build_assembly(boxes, mu=mu, tol=tol, dim=2)
        system = assemble(asm, tol, cone="linear2d")
        r = solve_p0(system, tol)
        steps.append(
            {
                "n": len(placed),
                "layer": int(L),
                "j": int(j),
                "x": float(j * dx),
                "margin": float(r.margin),
                "status": r.status,
            }
        )
        if r.status != FEASIBLE:
            prefix_feasible = False
            fail_step = len(placed) - 1
            break
    full = steps[-1] if steps else None
    return {
        "steps": steps,
        "prefix_feasible": bool(prefix_feasible),
        "full_status": full["status"] if full else None,
        "full_margin": full["margin"] if full else None,
        "fail_step": fail_step,
        "margins": [s["margin"] for s in steps],
    }


# --------------------------------------------------------------------------
# Stage 3: plan the build.
# --------------------------------------------------------------------------


def _corridor_state(L, j, dx, placed):
    """Right/left corridor blockage and the overhead blocker at (L, j).

    Mirrors keystone.search.lattice._reach_ok on the placed set. A same-layer
    cube to the right blocks the right corridor, one to the left blocks the
    left corridor, and any cube on a higher layer whose footprint overlaps the
    target column blocks the drop. Returns (right_blocked, left_blocked,
    blocker) where blocker is the (layer, j) of the deepest overhead cube or
    None.
    """
    x = j * dx
    right_blocked = False
    left_blocked = False
    blocker = None
    for (pl, pj) in placed:
        px = pj * dx
        if pl == L:
            d = px - x
            if d > -1.0 + 0.5 * dx:
                right_blocked = True
            if d < 1.0 - 0.5 * dx:
                left_blocked = True
        if pl > L:
            ov = min(x + 0.5, px + 0.5) - max(x - 0.5, px - 0.5)
            if ov > 0.5 * dx:
                if blocker is None or pl < blocker[0]:
                    blocker = (int(pl), int(pj))
    return right_blocked, left_blocked, blocker


def _support_center(L, j, dx, placed):
    """Center x of the surface the target at (L, j) rests on.

    Layer 0 rests on the pedestal (overlap midpoint with [-6, 0]); layer L
    rests on the best-overlapping layer-(L-1) cube. Used only to pick the
    com-side for a prop.
    """
    x = j * dx
    if L == 0:
        lo = max(x - 0.5, -6.0)
        hi = min(x + 0.5, 0.0)
        return 0.5 * (lo + hi) if hi > lo else x
    best_ov = 0.0
    center = x
    for (pl, pj) in placed:
        if pl == L - 1:
            px = pj * dx
            ov = min(x + 0.5, px + 0.5) - max(x - 0.5, px - 0.5)
            if ov > best_ov:
                best_ov = ov
                center = px
    return center


def classify_build(n, dx, seq):
    """Classify each placement into a build protocol at its build time.

    drop when the vertical column above the target is clear (drop-legal);
    else ride_under when a lateral corridor at the target layer is clear and
    the blocker is overhead (slide-legal but not drop-legal); else prop. Uses
    keystone.search.lattice.is_legal with modes "drop" and "slide" so the rule
    matches the search environment exactly. Returns a list of block records.
    """
    from keystone.search import lattice as LT

    spec_drop = LT.LatticeSpec(n_max=n, dx=dx, mode="drop")
    spec_slide = LT.LatticeSpec(n_max=n, dx=dx, mode="slide")

    blocks = []
    placed = []
    for step, (L, j) in enumerate(seq):
        state = LT.state_from_placements(spec_drop, placed)
        drop_ok = bool(LT.is_legal(spec_drop, state, L, j))
        slide_ok = bool(LT.is_legal(spec_slide, state, L, j))
        if drop_ok:
            protocol = "drop"
        elif slide_ok:
            protocol = "ride_under"
        else:
            protocol = "prop"
        params = _protocol_params(protocol, L, j, dx, placed, step)
        blocks.append(
            {
                "step": step,
                "layer": int(L),
                "j": int(j),
                "x": float(j * dx),
                "protocol": protocol,
                "params": params,
            }
        )
        placed.append((L, j))
    return blocks


def _protocol_params(protocol, L, j, dx, placed, step):
    """Protocol parameters recorded in the plan (also read by an arm executor)."""
    x = j * dx
    z = 1.5 + L
    underside = 1.0 + L
    target = {"x": float(x), "z": float(z)}
    if protocol == "drop":
        return {"target": target, "descent_h": DROP_H, "start_z": float(z + DROP_H)}
    if protocol == "ride_under":
        right_blocked, left_blocked, blocker = _corridor_state(L, j, dx, placed)
        # Approach from the clear side and push toward the seat.
        side = "+x" if not right_blocked else "-x"
        return {
            "target": target,
            "tilt_deg": RIDE_TILT_DEG,
            "cap_x": RIDE_CAP_X,
            "approach_side": side,
            "slide_off": RIDE_SLIDE_OFF,
            "blocker": list(blocker) if blocker else None,
        }
    # prop: a column under the com-side underside.
    support_center = _support_center(L, j, dx, placed)
    com_side = "+x" if x >= support_center else "-x"
    sign = 1.0 if com_side == "+x" else -1.0
    prop_x = x + sign * (0.5 - PROP_W / 2.0)
    return {
        "target": target,
        "com_side": com_side,
        "prop": {
            "x": float(prop_x),
            "width": PROP_W,
            "half_h": float(underside / 2.0),
            "z": float(underside / 2.0),
            "axis": "z",
            "retract_disp": float(-(underside + 0.4)),
            "supports_step": step,
        },
    }


# --------------------------------------------------------------------------
# Stage 4: execute in MuJoCo.
# --------------------------------------------------------------------------


def _tilt_quat_about_y(deg):
    """Quaternion for a rotation of deg degrees about world +y (lowers +x)."""
    a = np.radians(deg) / 2.0
    return np.array([np.cos(a), 0.0, np.sin(a), 0.0])


def _prop_box(meta, relief=0.0):
    """Build the prop column Box, optionally lowered by a catcher relief."""
    half_h = meta["half_h"] - relief / 2.0
    z = meta["z"] - relief / 2.0
    return box_2d(meta["width"], 2.0 * half_h, meta["x"], z)


def _impedance_place(
    placed_boxes,
    target_box,
    *,
    mu,
    start_pos,
    start_quat,
    traj,
    hold_pos,
    hold_quat,
    static_boxes=(),
    recorder=None,
    timestep=5e-4,
    params=IMPEDANCE,
    hold_time=0.4,
    settle_time=0.8,
    solref=INSERT_SOLREF,
    reach_tol=0.03,
    stand_disp_rel=0.01,
    stand_rot=0.05,
):
    """Drive one block along a trajectory with the capped impedance driver.

    placed_boxes are free bodies already in place; static_boxes are welded
    support columns (props). traj(a) returns (target_pos, target_quat) for the
    motion fraction a in (0, 1]; hold_pos/hold_quat seat the block. Returns a
    report dict and the settled boxes (placed + moving, props excluded).
    """
    import mujoco

    k = len(placed_boxes)
    moving = Box(
        target_box.half_extents.copy(),
        np.asarray(start_pos, dtype=np.float64),
        np.asarray(start_quat, dtype=np.float64),
        target_box.density,
    )
    boxes = list(placed_boxes) + [moving] + list(static_boxes)
    free = list(range(k + 1))
    L = _assembly_diag(list(placed_boxes) + [target_box])

    from keystone.interop.mujoco_io import to_mjcf

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
    pos0 = {i: data.xpos[i].copy() for i in placed_ids}
    quat0 = {i: data.xquat[i].copy() for i in placed_ids}

    def struct_disp():
        if not placed_ids:
            return 0.0
        return max(float(np.linalg.norm(data.xpos[i] - pos0[i])) for i in placed_ids)

    def struct_rot():
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

    def drive_step(tpos, tquat):
        dof = int(model.body_dofadr[new_id])
        R = data.xmat[new_id].reshape(3, 3)
        linvel = data.qvel[dof : dof + 3].copy()
        angvel = R @ data.qvel[dof + 3 : dof + 6]
        force, torque = _wrench(
            data.xpos[new_id],
            data.xquat[new_id],
            tpos,
            tquat,
            linvel,
            angvel,
            params,
            max_push,
            max_torque,
        )
        data.xfrc_applied[new_id, :3] = force
        data.xfrc_applied[new_id, 3:] = torque
        mujoco.mj_step(model, data)
        if recorder is not None:
            recorder.capture(model, data)
        return float(np.linalg.norm(force))

    dist = float(np.linalg.norm(target_box.position - np.asarray(start_pos)))
    n_motion = max(1, int(round(dist / params["speed"] / timestep)))
    peak_push = peak_contact = max_disturb = peak_support = 0.0

    for t in range(n_motion):
        a = (t + 1) / n_motion
        tpos, tquat = traj(a)
        peak_push = max(peak_push, drive_step(tpos, tquat))
        peak_contact = max(peak_contact, contact_force())
        max_disturb = max(max_disturb, struct_disp())
        peak_support = max(peak_support, support_force())

    for _ in range(max(1, int(round(hold_time / timestep)))):
        peak_push = max(peak_push, drive_step(hold_pos, hold_quat))
        peak_contact = max(peak_contact, contact_force())
        max_disturb = max(max_disturb, struct_disp())
        peak_support = max(peak_support, support_force())

    reach_err = float(np.linalg.norm(data.xpos[new_id] - target_box.position))

    data.xfrc_applied[:] = 0.0
    rel = {i: data.xpos[i].copy() for i in ids}
    settle_disp = 0.0
    for _ in range(max(1, int(round(settle_time / timestep)))):
        mujoco.mj_step(model, data)
        if recorder is not None:
            recorder.capture(model, data)
        settle_disp = max(
            settle_disp,
            max(float(np.linalg.norm(data.xpos[i] - rel[i])) for i in ids),
        )
        peak_support = max(peak_support, support_force())

    disturb_rel = struct_disp() / L
    rot = struct_rot()
    reached = bool(reach_err < reach_tol)
    stands = bool(disturb_rel < stand_disp_rel and rot < stand_rot)
    if reached and stands:
        outcome = "seated"
    elif not reached and stands:
        outcome = "stall"
    elif reached and not stands:
        outcome = "disturbed"
    else:
        outcome = "collapse"

    settled = []
    for idx, i in enumerate(ids):
        src = boxes[idx]
        settled.append(
            Box(
                src.half_extents.copy(),
                np.asarray(data.xpos[i]).copy(),
                np.asarray(data.xquat[i]).copy(),
                src.density,
            )
        )

    report = {
        "reach_err": reach_err,
        "reached": reached,
        "peak_push": peak_push,
        "max_push": max_push,
        "peak_contact_force": peak_contact,
        "peak_support_force": peak_support,
        "struct_disturb_rel": disturb_rel,
        "struct_rot": rot,
        "settle_disp": settle_disp,
        "stands": stands,
        "outcome": outcome,
        "L": L,
    }
    return report, settled


def _wrench(pos, quat, tpos, tquat, linvel, angvel, params, max_push, max_torque):
    from keystone.interop.mujoco_io import capped_impedance_wrench

    return capped_impedance_wrench(
        pos,
        quat,
        tpos,
        tquat,
        linvel,
        angvel,
        kp=params["kp"],
        kd=params["kd"],
        kp_rot=params["kp_rot"],
        kd_rot=params["kd_rot"],
        max_push=max_push,
        max_torque=max_torque,
    )


def _assembly_diag(boxes):
    from keystone.interop.mujoco_io import assembly_diagonal

    return assembly_diagonal(boxes)


def _drop_step(placed_boxes, target_box, static_boxes, mu, recorder, timestep):
    """Drop the block from above and seat it (short descent + release)."""
    tgt = target_box.position
    start = tgt + np.array([0.0, 0.0, DROP_H])
    z_bias = IMPEDANCE["z_bias"]

    def traj(a):
        pos = start + a * (tgt - start) + np.array([0.0, 0.0, z_bias])
        return pos, target_box.quat

    return _impedance_place(
        placed_boxes,
        target_box,
        mu=mu,
        start_pos=start,
        start_quat=target_box.quat.copy(),
        traj=traj,
        hold_pos=tgt.copy(),
        hold_quat=target_box.quat.copy(),
        static_boxes=static_boxes,
        recorder=recorder,
        timestep=timestep,
    )


def _ride_under_step(placed_boxes, target_box, params, static_boxes, mu, recorder,
                     timestep):
    """Push the block in nose-down under an overhead blocker, then flatten.

    The block starts tilted on the clear corridor side at seat height, riding
    on the support top. A small nose-down tilt drops the leading top corner
    below the blocker lip. The push drives it to the seat while flattening the
    tilt, so the blocker rides over and settles back on top.
    """
    tilt_deg = params["tilt_deg"]
    side = params["approach_side"]
    slide_off = params["slide_off"]
    sign = 1.0 if side == "+x" else -1.0
    tgt = target_box.position
    x_seat = float(tgt[0])
    z_seat = float(tgt[2])
    z_bias = IMPEDANCE["z_bias"]

    # Nose (leading edge in the push direction) down. Push moves toward -sign*x,
    # so the leading edge is on the -sign side; lower it.
    tilt_deg_signed = -sign * tilt_deg
    start = np.array([x_seat + sign * slide_off, 0.0, z_seat])
    start_quat = _tilt_quat_about_y(tilt_deg_signed)

    ride_params = dict(IMPEDANCE, max_push_x=params["cap_x"])

    def traj(a):
        x = start[0] + a * (x_seat - start[0])
        # Flatten the tilt over the last two thirds of the push.
        flat = min(1.0, a / 0.66) if a > 0.34 else 0.0
        deg = tilt_deg_signed * (1.0 - flat)
        pos = np.array([x, 0.0, z_seat + z_bias])
        return pos, _tilt_quat_about_y(deg)

    return _impedance_place(
        placed_boxes,
        target_box,
        mu=mu,
        start_pos=start,
        start_quat=start_quat,
        traj=traj,
        hold_pos=tgt.copy(),
        hold_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        static_boxes=static_boxes,
        recorder=recorder,
        timestep=timestep,
        params=ride_params,
        hold_time=0.5,
        reach_tol=0.05,
    )


def _retract_props(boxes, props, mu, recorder, timestep, *, ramp_time=1.0,
                   stage_settle=0.5, final_settle=1.0):
    """Retract the props one at a time on position-actuated sliders.

    boxes are the pedestal and built blocks at settled poses (all free). Each
    prop rides one slider so the other axes are held rigidly; the prop top is
    set from its supported block minus a 1 mm relief. Ported from
    examples/mujoco_falsework.retract_props. Returns a report and the final
    block poses.
    """
    import mujoco

    from keystone.interop.mujoco_io import to_mjcf

    relief = 1e-3
    axis_vec = {"x": "1 0 0", "z": "0 0 1"}
    world_parts = []
    act_parts = []
    for idx, p in enumerate(props):
        meta = p["meta"]
        blk = boxes[p["placed_index"]]
        top = float(blk.position[2] - blk.half_extents[2]) - relief
        half_h = meta["half_h"]
        zc = top - half_h
        world_parts.append(
            f'    <body name="prop{idx}" pos="{_f(meta["x"])} 0.0 {_f(zc)}">\n'
            f'      <joint name="prop{idx}slide" type="slide" '
            f'axis="{axis_vec[meta["axis"]]}" range="-3.0 1.0" '
            f'damping="20000.0"/>\n'
            f'      <geom name="prop{idx}geom" type="box" '
            f'size="{_f(meta["width"] / 2.0)} 0.5 {_f(half_h)}" '
            f'density="2000.0" contype="0" conaffinity="0"/>\n'
            f"    </body>"
        )
        act_parts.append(
            f'    <position name="prop{idx}act" joint="prop{idx}slide" '
            f'kp="2000000.0" kv="200000.0" ctrlrange="-3.0 1.0"/>'
        )
    fric = "0.7 0.7 0.005 0.0001 0.0001"
    sr = f"{_f(INSERT_SOLREF[0])} {_f(INSERT_SOLREF[1])}"
    pairs = [
        f'    <pair geom1="prop{idx}geom" geom2="geom{i}" condim="3" '
        f'friction="{fric}" solref="{sr}"/>'
        for idx in range(len(props))
        for i in range(1, len(boxes))
    ]
    xml = to_mjcf(
        boxes,
        mu,
        timestep=timestep,
        all_pairs=True,
        solref=INSERT_SOLREF,
        extra_worldbody="\n".join(world_parts),
        extra_pairs="\n".join(pairs),
        extra_actuator="\n".join(act_parts),
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    n = len(boxes)
    ids = [int(model.body(f"block{i}").id) for i in range(n)]
    pos0 = [data.xpos[i].copy() for i in ids]
    quat0 = [data.xquat[i].copy() for i in ids]
    pgid = [int(model.geom(f"prop{idx}geom").id) for idx in range(len(props))]
    L = _assembly_diag(boxes)

    def prop_load(idx):
        buf = np.zeros(6)
        tot = 0.0
        for c in range(data.ncon):
            con = data.contact[c]
            if pgid[idx] in (int(con.geom1), int(con.geom2)):
                mujoco.mj_contactForce(model, data, c, buf)
                tot += float(np.linalg.norm(buf[:3]))
        return tot

    n_ramp = max(1, int(round(ramp_time / timestep)))
    n_stage = max(1, int(round(stage_settle / timestep)))

    for _ in range(n_stage):
        mujoco.mj_step(model, data)
        if recorder is not None:
            recorder.capture(model, data)

    stages = []
    for idx, p in enumerate(props):
        load_before = prop_load(idx)
        peak = load_before
        for t in range(n_ramp):
            data.ctrl[idx] = p["meta"]["retract_disp"] * (t + 1) / n_ramp
            mujoco.mj_step(model, data)
            if recorder is not None:
                recorder.capture(model, data)
            peak = max(peak, prop_load(idx))
        for _ in range(n_stage):
            mujoco.mj_step(model, data)
            if recorder is not None:
                recorder.capture(model, data)
        stages.append(
            {
                "prop": p["meta"].get("name", f"prop{idx}"),
                "axis": p["meta"]["axis"],
                "load_before": load_before,
                "peak_during_retract": peak,
                "load_after": prop_load(idx),
            }
        )

    for _ in range(max(1, int(round(final_settle / timestep)))):
        mujoco.mj_step(model, data)
        if recorder is not None:
            recorder.capture(model, data)

    disp = max(float(np.linalg.norm(data.xpos[i] - pos0[k])) for k, i in enumerate(ids))
    rot = max(
        2.0 * float(np.arccos(min(1.0, abs(float(np.dot(quat0[k], data.xquat[i]))))))
        for k, i in enumerate(ids)
    )
    stands = bool(disp / L < 0.01 and rot < 0.05)
    final = [
        Box(
            boxes[k].half_extents.copy(),
            np.asarray(data.xpos[i]).copy(),
            np.asarray(data.xquat[i]).copy(),
            boxes[k].density,
        )
        for k, i in enumerate(ids)
    ]
    report = {
        "stages": stages,
        "disp_rel": disp / L,
        "rot": rot,
        "verdict": "stands" if stands else "collapsed",
    }
    return report, final


def _f(x):
    return "%.17g" % float(x)


def _execute(plan_blocks, dx, mu, recorder, timestep, settle_duration=2.0):
    """Run the build in MuJoCo and settle. Returns the execute-stage record."""
    from keystone.interop.mujoco_io import settle_test

    placed_boxes = [pedestal()]
    props = []
    step_reports = []
    failed_protocol = None
    failure_detail = None

    for blk in plan_blocks:
        L, j = blk["layer"], blk["j"]
        protocol = blk["protocol"]
        target = cube(L, j * dx)

        if protocol == "prop":
            meta = dict(blk["params"]["prop"])
            props.append(
                {
                    "meta": meta,
                    "placed_index": len(placed_boxes),  # this block's index
                }
            )

        static = [_prop_box(p["meta"], relief=PROP_RELIEF) for p in props]

        if protocol == "ride_under":
            rep, placed_boxes = _ride_under_step(
                placed_boxes, target, blk["params"], static, mu, recorder, timestep
            )
        else:
            # drop and prop both seat with the descent driver; the prop is the
            # support that makes an otherwise unsupported drop stand.
            rep, placed_boxes = _drop_step(
                placed_boxes, target, static, mu, recorder, timestep
            )
        rep["step"] = blk["step"]
        rep["protocol"] = protocol
        step_reports.append(rep)

        if rep["outcome"] == "collapse":
            failed_protocol = protocol
            failure_detail = (
                f"step {blk['step']} ({protocol}) collapsed: reach_err="
                f"{rep['reach_err']:.4f}, struct_rot={rep['struct_rot']:.4f}"
            )
            break

    retraction = None
    if props and failed_protocol is None:
        retraction, placed_boxes = _retract_props(
            placed_boxes, props, mu, recorder, timestep
        )

    settle = settle_test(
        placed_boxes, mu, duration=settle_duration, solref=STIFF_SOLREF
    )

    if failed_protocol is not None:
        verdict = "execution_failed"
    elif retraction is not None and retraction["verdict"] != "stands":
        verdict = "prop_retraction_failed"
        failed_protocol = "prop"
        failure_detail = (
            f"structure moved after prop retraction: disp_rel="
            f"{retraction['disp_rel']:.4f}, rot={retraction['rot']:.4f}"
        )
    elif settle["stable"]:
        verdict = "stands"
    else:
        verdict = "certified_but_dynamic_fail"

    return {
        "executor": EXECUTOR,
        "steps": step_reports,
        "retraction": retraction,
        "settle": {
            "verdict": settle["verdict"],
            "stable": bool(settle["stable"]),
            "max_disp_rel": settle["max_disp_rel"],
            "max_rot": settle["max_rot"],
            "traj_max_disp_rel": settle["traj_max_disp_rel"],
            "traj_max_rot": settle["traj_max_rot"],
        },
        "verdict": verdict,
        "failed_protocol": failed_protocol,
        "failure_detail": failure_detail,
    }


# --------------------------------------------------------------------------
# Orchestration.
# --------------------------------------------------------------------------


def evaluate_stacking(
    n,
    dx=1.0 / 12.0,
    sims=4000,
    seed=0,
    checkpoint="out/search/az_params_v2.msgpack",
    out_dir="out/pipeline",
    record=True,
    *,
    timestep=5e-4,
    tol=None,
    progress=None,
    settle_duration=2.0,
    verbose=True,
):
    """Evaluate a stacking design end to end and return the JSON record.

    n cubes on a dx grid. Runs search, host certification, build planning,
    MuJoCo execution, and a movie, writing {out_dir}/pipeline_n{n}_seed{seed}
    .json/.mp4/.gif/_final.png. record=False skips the movie (the test path).
    """
    tol = tol or Tolerances()
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"pipeline_n{n}_seed{seed}")

    from keystone.search import lattice as LT

    record_out = {
        "n": int(n),
        "dx": float(dx),
        "sims": int(sims),
        "seed": int(seed),
        "executor": EXECUTOR,
        "harmonic": LT.harmonic(n),
        "notes": [],
    }

    # Stage 1: search.
    search, best, seq, search_info = _run_search(
        n, dx, sims, seed, checkpoint, tol, progress=progress
    )
    record_out.update(search_info)
    record_out["search"] = {
        "best_overhang": best,
        "harmonic": LT.harmonic(n),
        "ratio": best / LT.harmonic(n) if LT.harmonic(n) > 0 else None,
        "sequence": [{"layer": L, "j": j, "x": j * dx} for (L, j) in seq],
        "best_key": [[L, j] for (L, j) in sorted(seq)],
        "wall_s": getattr(search, "wall", None),
        "sims_done": getattr(search, "sims_done", None),
    }
    search_claim = bool(seq) and best > float("-inf")

    if not seq:
        record_out["notes"].append("search found no feasible placement")
        record_out["agreement"] = {
            "search_claim": False,
            "certificate": None,
            "physics": None,
            "three_way": "no_design",
        }
        _write_json(base + ".json", record_out)
        if verbose:
            _print_summary(record_out)
        return record_out

    # Stage 2: certify.
    certify = certify_prefixes(seq, dx, tol, mu=MU)
    record_out["certify"] = certify
    certificate = bool(certify["prefix_feasible"])

    if not certificate:
        record_out["notes"].append(
            f"certify aborted: prefix {certify['fail_step']} not feasible "
            "(unexpected; the search re-verifies)"
        )
        record_out["agreement"] = {
            "search_claim": search_claim,
            "certificate": False,
            "physics": None,
            "three_way": "certify_disagree",
        }
        _write_json(base + ".json", record_out)
        if verbose:
            _print_summary(record_out)
        return record_out

    # Stage 3: plan.
    plan_blocks = classify_build(n, dx, seq)
    counts = {"drop": 0, "ride_under": 0, "prop": 0}
    for b in plan_blocks:
        counts[b["protocol"]] += 1
    record_out["plan"] = {"blocks": plan_blocks, "protocol_counts": counts}

    # Stage 4: execute (with the movie recorder across the whole build).
    recorder = FrameRecorder(stride=40, record=record)
    execute = None
    try:
        execute = _execute(
            plan_blocks, dx, MU, recorder, timestep, settle_duration=settle_duration
        )
    except Exception as exc:  # noqa: BLE001
        record_out["notes"].append(
            f"execution raised {type(exc).__name__}: {exc}"
        )
        execute = {
            "executor": EXECUTOR,
            "steps": [],
            "verdict": "execution_error",
            "failed_protocol": None,
            "failure_detail": f"{type(exc).__name__}: {exc}",
            "settle": None,
        }
    record_out["execute"] = execute

    # Stage 5: movie and report.
    videos = recorder.finalize(base)
    videos["json"] = base + ".json"
    record_out["videos"] = videos

    physics = execute.get("verdict") == "stands"
    if not certificate:
        three_way = "certify_disagree"
    elif search_claim and physics:
        three_way = "agree_stands"
    elif execute.get("verdict") == "certified_but_dynamic_fail":
        three_way = "certified_but_dynamic_fail"
    else:
        three_way = execute.get("verdict", "unknown")
    record_out["agreement"] = {
        "search_claim": search_claim,
        "certificate": certificate,
        "physics": bool(physics),
        "three_way": three_way,
    }

    _write_json(base + ".json", record_out)
    if verbose:
        _print_summary(record_out)
    return record_out


def _write_json(path, record):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(record, f, indent=1, default=_json_default)


def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _print_summary(record):
    """Print a one-screen summary of the five stages."""
    n = record["n"]
    dx = record["dx"]
    print()
    print(f"=== keystone stacking pipeline: n={n} dx={dx:.5f} seed={record['seed']} ===")
    s = record.get("search", {})
    print(
        f"SEARCH   prior={record.get('prior', '?')} "
        f"best_overhang={s.get('best_overhang', float('nan')):.4f} "
        f"harmonic={record['harmonic']:.4f} "
        f"ratio={s.get('ratio') if s.get('ratio') is not None else float('nan'):.3f}"
    )
    seq = s.get("sequence", [])
    print(
        "         sequence: "
        + ", ".join(f"(L{b['layer']},x{b['x']:+.3f})" for b in seq)
    )
    cert = record.get("certify")
    if cert:
        margins = cert["margins"]
        rng = (min(margins), max(margins)) if margins else (float("nan"),) * 2
        print(
            f"CERTIFY  prefix_feasible={cert['prefix_feasible']} "
            f"full_status={cert['full_status']} "
            f"margins[{rng[0]:.2e}, {rng[1]:.2e}]"
        )
    plan = record.get("plan")
    if plan:
        protos = ", ".join(
            f"{b['protocol']}" for b in plan["blocks"]
        )
        print(f"PLAN     {plan['protocol_counts']}  per-block: [{protos}]")
    ex = record.get("execute")
    if ex:
        print(
            f"EXECUTE  executor={ex['executor']} verdict={ex['verdict']}"
            + (
                f"  failed={ex['failed_protocol']} ({ex['failure_detail']})"
                if ex.get("failed_protocol")
                else ""
            )
        )
        if ex.get("settle"):
            st = ex["settle"]
            print(
                f"         settle {st['verdict']} "
                f"max_disp_rel={st['max_disp_rel']:.4f} max_rot={st['max_rot']:.4f}"
            )
    ag = record.get("agreement", {})
    print(
        f"AGREEMENT search_claim={ag.get('search_claim')} "
        f"certificate={ag.get('certificate')} physics={ag.get('physics')} "
        f"-> {ag.get('three_way')}"
    )
    vids = record.get("videos", {})
    if vids:
        print(
            f"VIDEO    mp4={vids.get('mp4')} gif={vids.get('gif')} "
            f"still={vids.get('still')}"
        )
