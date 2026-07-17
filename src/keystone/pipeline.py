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

The driver executor is the default. The Franka arm executor (executor="franka")
is the selectable alternative: a menagerie Panda based at the overhang end
executes the exact search sequence, drops with the closed-loop press, and
threads the ride-under with the admittance push plus a seat correction. The
record carries an "executor" field and enough per-block target and protocol
detail to drive either. This module never imports mujoco or flax at import
time; both load lazily inside the stage that needs them.
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
# Stage 4b: execute with the Franka arm (executor="franka").
#
# The unit lattice is a uniform scaling of the Franka table-top scene (cube
# side 0.05 m, 0.25 kg). keystone margins are scale-invariant, so the unit
# certificate carries to this scale. The base sits at the overhang end (max
# target x) with a y offset, yaw facing the structure, so every approach comes
# through the empty y-corridor and no link arches over the wall. Drops use the
# franka_build closed-loop press (hover correction, iterated alignment, seat
# press). Ride-under uses the phase-2 admittance push (rideunder) plus a new
# force-capped seat correction. Props stay scene sliders and retract at the end.
# --------------------------------------------------------------------------

FRANKA_EXECUTOR = "franka"
FRANKA_S = 0.05          # cube side, meters
FRANKA_G = 9.81

# Base placement.
FRANKA_BASE_Y = -0.40    # y offset into the free corridor (task range -.35..-.45)
FRANKA_BASE_YAW = np.pi / 2.0   # face +y, toward the structure plane at y = 0
FRANKA_BASE_SLIDE = 0.03        # m, x steps tried when the overhang end fails reach
FRANKA_BASE_SLIDE_MAX = 8       # slide attempts each direction

# Staging row on the arm side (-y), past the pedestal right edge in +x.
FRANKA_STAGE_Y = -0.10
FRANKA_STAGE_DX = 0.06

# Kinematic reach acceptance for the pre-check.
FRANKA_REACH_POS_TOL = 2e-3     # m
FRANKA_REACH_ROT_TOL = 3e-2     # rad

# Drop recipe (franka_build), speeds m/s and offsets meters.
F_SPEED_TRANSIT = 0.10
F_SPEED_DESCEND = 0.03
F_PRE_GRASP_DZ = 0.12
F_LIFT_DZ = 0.05
F_TRANSIT_Z = 0.30
F_HOVER_DZ = 0.020
F_ALIGN_DZ = 0.002
F_ALIGN_TOL = 4e-4
F_ALIGN_ITERS = 3
F_PRESS_STEP = 0.001
F_PRESS_ITERS = 5
F_SEAT_TOL = 4e-4
F_RETREAT_DZ = 0.10
F_GRASP_TRACK_TOL = 0.010
F_T_CLOSE = 0.6
F_T_OPEN = 0.5
F_T_SETTLE_BLOCK = 0.6

# Ride-under admittance (rideunder phase 2).
F_RU_SLIDE_OFF = 0.30 * FRANKA_S
F_RU_GRASP_DX = 0.20 * FRANKA_S
F_RU_Z_END_MARGIN = 0.0025
F_RU_OVER = 0.004
F_RU_LIFT_CAP = 2.5             # N, bridge-lift cap (leveling torque stand-in)
F_RU_THREAD_T = 4.5             # s, nominal thread duration
F_RU_KMAX_T = 6.0              # s, thread step budget
# New seat correction after the push.
F_SEAT_CORR_ITERS = 6
F_SEAT_CORR_TOL = 1.0e-3        # m
F_SEAT_CORR_SPEED = 0.01
F_SEAT_CORR_GAIN = 0.6         # gentle integral gain on the reacher-to-seat error

# Verdict thresholds. A large structure disturbance during a manipulation that
# coincides with arm contact or drag is the arm knocking things over.
F_TOPPLE_DISTURB = 0.5 * FRANKA_S   # m
F_RU_DRAG_FAIL = 0.030              # m, bridge dragged this far = the arm rammed
# A block more than half a cube from its seat means the intended structure was
# not produced (a block slid or fell), separate from settle motion.
F_BUILD_TOL = 0.5 * FRANKA_S        # m

# Franka movie camera (frames the tiny structure and the arm above it).
FRANKA_CAMERA = {
    "lookat": [0.45, -0.06, 0.13],
    "distance": 1.05,
    "azimuth": 135.0,
    "elevation": -18.0,
}


def _qmul(a, b):
    """Hamilton product of two (w, x, y, z) quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _tilt_grasp_quat(deg):
    """Top-grasp wrist quaternion (tool z down) pitched nose-down by deg about
    world y. deg = 0 is the plain top grasp used for the flat drops."""
    a = -np.radians(deg) / 2.0
    return _qmul(np.array([np.cos(a), 0.0, 0.0, np.sin(a)]),
                 np.array([0.0, 0.0, 1.0, 0.0]))


def _grasp_world(cx, cz, deg, grasp_dx):
    """World grasp point near the trailing end of a reacher centered at (cx, cz)
    and pitched nose-down by deg about world y."""
    a = -np.radians(deg)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    return np.array([cx, 0.0, cz]) + R @ np.array([grasp_dx, 0.0, 0.0])


def _y_angle(q):
    """Signed rotation about +y in a (w, 0, y, 0) quaternion, radians."""
    return 2.0 * float(np.arctan2(q[2], q[0]))


def _ride_geometry(params, tgt, layer, dx, scale, base_offset):
    """Ride-under thread geometry at the Franka scale, from the plan params.

    tgt is the world target cube center. Returns a dict of the push waypoints:
    the tilted start, the engage point under the blocker lip, and the seat, plus
    the sign of the approach (+1 for a +x approach, -1 for -x)."""
    tilt_deg = float(params["tilt_deg"])
    sign = 1.0 if params["approach_side"] == "+x" else -1.0
    thr = np.radians(tilt_deg)
    span = 0.5 * scale * (np.cos(thr) + np.sin(thr))
    x_seat = float(tgt[0])
    z_seat = float(tgt[2])
    support_top = (1.0 + layer) * scale         # top of the layer below the seat
    z_ride = support_top + span + 0.0005
    blocker = params["blocker"]                  # [layer, j] of the bridge
    bridge_cx = blocker[1] * dx * scale + base_offset[0]
    bridge_edge = bridge_cx + sign * 0.5 * scale  # edge on the approach side
    x_engage = bridge_edge + sign * span
    x_start = x_seat + sign * F_RU_SLIDE_OFF
    x_deep = x_seat - sign * F_RU_OVER           # thread a hair past the seat
    grasp_dx = sign * F_RU_GRASP_DX              # grasp near the trailing end
    return dict(tilt_deg=tilt_deg, cap_x=float(params["cap_x"]), sign=sign,
                x_seat=x_seat, z_seat=z_seat, z_ride=z_ride, x_start=x_start,
                x_engage=x_engage, x_deep=x_deep, grasp_dx=grasp_dx)


def _franka_waypoints(plan_blocks, info, scale, base_offset):
    """Key reach waypoints (name, pos, quat) an arm must reach for this plan:
    every staging pick, every drop target, and the ride-under start, engage, and
    seat. Finer executor waypoints interpolate within these extremes."""
    from keystone.interop.franka_scene import GRASP_QUAT

    wps = []
    for i, p in enumerate(info.staging_world):
        wps.append((f"pick_hover_{i}", np.asarray(p) + [0, 0, F_PRE_GRASP_DZ],
                    GRASP_QUAT))
        wps.append((f"pick_{i}", np.asarray(p), GRASP_QUAT))
    for blk in plan_blocks:
        i = blk["step"]
        tgt = info.target_world[i]
        if blk["protocol"] == "ride_under":
            g = _ride_geometry(blk["params"], tgt, blk["layer"], info.dx, scale,
                               base_offset)
            wps.append((f"ru_start_{i}",
                        _grasp_world(g["x_start"], g["z_ride"], g["tilt_deg"],
                                     g["grasp_dx"]), _tilt_grasp_quat(g["tilt_deg"])))
            wps.append((f"ru_engage_{i}",
                        _grasp_world(g["x_engage"], g["z_seat"] + F_RU_Z_END_MARGIN,
                                     g["tilt_deg"], g["grasp_dx"]),
                        _tilt_grasp_quat(g["tilt_deg"])))
            wps.append((f"ru_seat_{i}",
                        _grasp_world(g["x_seat"], g["z_seat"] + F_RU_Z_END_MARGIN,
                                     0.0, g["grasp_dx"]), _tilt_grasp_quat(0.0)))
        else:
            wps.append((f"tgt_hover_{i}", np.asarray(tgt) + [0, 0, F_HOVER_DZ + 0.08],
                        GRASP_QUAT))
            wps.append((f"seat_{i}", np.asarray(tgt) + [0, 0, F_ALIGN_DZ], GRASP_QUAT))
    return wps


def _reach_scan(model, info, waypoints):
    """Run warm-started IK to every waypoint from the home seed. Returns the
    worst position error, worst rotation error, and the per-waypoint table."""
    import mujoco

    from keystone.interop.franka_scene import dls_ik, reset_home

    scratch = mujoco.MjData(model)
    reset_home(model, scratch)
    seed = scratch.qpos.copy()
    rows = []
    worst_pos = worst_rot = 0.0
    for name, pos, quat in waypoints:
        scratch.qpos[:] = seed
        _q, pe, re = dls_ik(model, scratch, np.asarray(pos, dtype=np.float64), quat)
        rows.append({"waypoint": name, "pos_err_mm": pe * 1000.0, "rot_err": re,
                     "reachable": bool(pe < FRANKA_REACH_POS_TOL
                                       and re < FRANKA_REACH_ROT_TOL)})
        worst_pos = max(worst_pos, pe)
        worst_rot = max(worst_rot, re)
    return worst_pos, worst_rot, rows


def _place_franka_base(plan_blocks, cells, cell_names, dx, scale, base_offset,
                       prop_specs):
    """Place the arm base at the overhang end and verify reach for every
    waypoint, sliding along x by the minimum needed if the exact end fails.

    Returns (base_pos, base_yaw, staging, reach_rows, moved_note, worst_pos)."""
    from keystone.interop.franka_scene import compose_scene

    targets = np.array([np.array([j * dx * scale, 0.0, (1.5 + layer) * scale])
                        + base_offset for (layer, j) in cells])
    x_over = float(np.max(targets[:, 0]))          # overhang end, world x
    x_right = float(np.max(targets[:, 0])) + 0.5 * scale
    ped_right = base_offset[0]                      # pedestal right edge at x = 0
    stage_x0 = max(x_right + 0.5 * scale, ped_right + 0.5 * scale)
    staging = np.array([[stage_x0 + i * FRANKA_STAGE_DX, FRANKA_STAGE_Y, 0.5 * scale]
                        for i in range(len(cells))])

    # Try the exact overhang end first, then slide by +-FRANKA_BASE_SLIDE.
    offsets = [0.0]
    for k in range(1, FRANKA_BASE_SLIDE_MAX + 1):
        offsets += [-k * FRANKA_BASE_SLIDE, k * FRANKA_BASE_SLIDE]
    best = None
    for off in offsets:
        base_pos = np.array([x_over + off, FRANKA_BASE_Y, 0.0])
        spec, info = compose_scene(scale=scale, base_offset=base_offset, dx=dx,
                                   cells=cells, cell_names=cell_names,
                                   prop_specs=prop_specs, staging=staging,
                                   arm_base_pos=base_pos, arm_base_yaw=FRANKA_BASE_YAW)
        model = spec.compile()
        wps = _franka_waypoints(plan_blocks, info, scale, base_offset)
        worst_pos, worst_rot, rows = _reach_scan(model, info, wps)
        ok = worst_pos < FRANKA_REACH_POS_TOL and worst_rot < FRANKA_REACH_ROT_TOL
        if best is None or worst_pos < best[5]:
            best = (base_pos, FRANKA_BASE_YAW, staging, rows, off, worst_pos)
        if ok:
            moved = ("at the overhang end" if off == 0.0
                     else f"slid {off * 1000.0:+.0f} mm in x to clear reach")
            return base_pos, FRANKA_BASE_YAW, staging, rows, moved, worst_pos
    # No fully-reachable base found; return the best and flag it.
    base_pos, yaw, staging, rows, off, worst_pos = best
    moved = (f"best effort, slid {off * 1000.0:+.0f} mm; worst reach "
             f"{worst_pos * 1000.0:.1f} mm still exceeds tolerance")
    return base_pos, yaw, staging, rows, moved, worst_pos


def _plan_to_prop_specs(plan_blocks, dx):
    """Translate prop-protocol blocks into unit-scale prop specs for the scene.
    supports indexes into the build order (the cube on the prop)."""
    specs = []
    for blk in plan_blocks:
        if blk["protocol"] != "prop":
            continue
        pm = blk["params"]["prop"]
        # The plan prop x/width/half_h are in keystone units on the dx grid.
        specs.append(dict(
            name=f"prop{blk['step']}",
            x=float(pm["x"]),
            half_h=float(pm["half_h"]),
            z=float(pm["z"]),
            axis=pm["axis"],
            retract_disp=float(pm["retract_disp"]),
            supports=int(blk["step"]),
        ))
    return specs


class _FrankaDriver:
    """Waypoint controller for the arm build. Uses one MuJoCo model for the
    whole build and captures frames through the shared FrameRecorder."""

    def __init__(self, model, info, timestep, recorder, mu, scale, base_offset):
        import mujoco

        self.mj = mujoco
        self.model = model
        self.info = info
        self.data = mujoco.MjData(model)
        self.scratch = mujoco.MjData(model)
        self.timestep = timestep
        self.recorder = recorder
        self.mu = mu
        self.scale = scale
        self.base_offset = np.asarray(base_offset, dtype=np.float64)
        self.sid = model.site(info.tcp_site).id
        self.grip_aid = model.actuator(info.gripper_actuator).id
        self.cube_bids = [model.body(b).id for b in info.cube_bodies]
        self.cube_gids = [model.geom(g).id for g in info.cube_geoms]
        self.pad_gids = [model.geom(g).id for g in info.finger_pads]
        self.prop_aids = [model.actuator(p["act"]).id for p in info.props]
        self.prop_gids = [model.geom(p["geom"]).id for p in info.props]
        hand_bids = {model.body(b).id
                     for b in ("hand", "left_finger", "right_finger")}
        self.hand_gids = [g for g in range(model.ngeom)
                          if int(model.geom_bodyid[g]) in hand_bids
                          and int(model.geom_group[g]) == 3]
        corners = np.concatenate(
            [self._cube_corners(i) for i in range(len(self.cube_bids))]
            + [np.array([[base_offset[0] - 3 * scale, 0, 0],
                         [base_offset[0], 0, scale]])], axis=0)
        self.L = float(np.linalg.norm(corners.max(axis=0) - corners.min(axis=0)))

    def _cube_corners(self, i):
        c = self.info.target_world[i]
        h = 0.5 * self.scale
        return np.array([[c[0] - h, 0, c[2] - h], [c[0] + h, 0, c[2] + h]])

    # -- stepping and motion ------------------------------------------------

    def step(self):
        self.mj.mj_step(self.model, self.data)
        if self.recorder is not None:
            self.recorder.capture(self.model, self.data)

    def hold(self, t):
        for _ in range(max(1, int(round(t / self.timestep)))):
            self.step()

    def tcp(self):
        return self.data.site_xpos[self.sid].copy()

    def move_to(self, pos, quat, speed=F_SPEED_DESCEND, min_steps=100, iters=200):
        from keystone.interop.franka_scene import dls_ik

        q0 = np.array(self.data.ctrl[:7])
        self.scratch.qpos[:] = self.data.qpos
        qg, _, _ = dls_ik(self.model, self.scratch, np.asarray(pos, np.float64),
                          quat, iters=iters)
        dist = float(np.linalg.norm(np.asarray(pos) - self.tcp()))
        n = max(min_steps, int(round(dist / speed / self.timestep)))
        for t in range(n):
            a = (t + 1) / n
            self.data.ctrl[:7] = q0 + a * (qg - q0)
            self.step()
        return qg

    def grip(self, ctrl, t):
        self.data.ctrl[self.grip_aid] = ctrl
        self.hold(t)

    # -- measurements -------------------------------------------------------

    def contact_force(self, ga, gb):
        buf = np.zeros(6)
        tot = np.zeros(3)
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            if (g1 == ga and g2 == gb) or (g2 == ga and g1 == gb):
                self.mj.mj_contactForce(self.model, self.data, c, buf)
                tot += con.frame.reshape(3, 3).T @ buf[:3]
        return tot

    def arm_touch(self, cube_idxs):
        """Peak arm-geom (pad or hand) normal force on any of the given placed
        cubes this step. Nonzero means the arm is in contact with the structure."""
        buf = np.zeros(6)
        arm = set(self.pad_gids) | set(self.hand_gids)
        gids = {self.cube_gids[i] for i in cube_idxs}
        f = 0.0
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            hit = (g1 in arm and g2 in gids) or (g2 in arm and g1 in gids)
            if hit:
                self.mj.mj_contactForce(self.model, self.data, c, buf)
                f = max(f, float(np.linalg.norm(buf[:3])))
        return f

    def struct_disturb(self, baseline):
        """Max displacement of the baselined placed cubes from their baseline."""
        if not baseline:
            return 0.0
        return max(float(np.linalg.norm(self.data.xpos[b] - p0))
                   for b, p0 in baseline.items())

    # -- drop with the closed-loop press (franka_build recipe) --------------

    def place_drop(self, i, placed_idxs):
        info = self.info
        stage = np.asarray(info.staging_world[i], np.float64)
        target = np.asarray(info.target_world[i], np.float64)
        bid = self.cube_bids[i]
        base = {self.cube_bids[k]: self.data.xpos[self.cube_bids[k]].copy()
                for k in placed_idxs}
        peak_arm = 0.0
        max_disturb = 0.0

        def track():
            nonlocal peak_arm, max_disturb
            peak_arm = max(peak_arm, self.arm_touch(placed_idxs))
            max_disturb = max(max_disturb, self.struct_disturb(base))

        from keystone.interop.franka_scene import GRASP_QUAT, GRIPPER_CLOSE, \
            GRIPPER_OPEN

        # Pick.
        self.move_to(stage + [0, 0, F_PRE_GRASP_DZ], GRASP_QUAT, F_SPEED_TRANSIT)
        self.grip(GRIPPER_OPEN, 0.2)
        self.move_to(stage, GRASP_QUAT, F_SPEED_DESCEND)
        self.grip(GRIPPER_CLOSE, F_T_CLOSE)
        # Grasp check.
        self.move_to(stage + [0, 0, F_LIFT_DZ], GRASP_QUAT, F_SPEED_DESCEND)
        track()
        gtrack = float(np.linalg.norm(self.data.xpos[bid] - self.tcp()))
        rose = float(self.data.xpos[bid][2] - stage[2])
        grasped = bool(gtrack < F_GRASP_TRACK_TOL and rose > 0.5 * F_LIFT_DZ)
        if not grasped:
            return self._drop_report(i, target, grasped, None, None, peak_arm,
                                     max_disturb, seated=False)
        # Transport.
        self.move_to([stage[0], stage[1], F_TRANSIT_Z], GRASP_QUAT, F_SPEED_TRANSIT)
        self.move_to([target[0], stage[1], F_TRANSIT_Z], GRASP_QUAT, F_SPEED_TRANSIT)
        self.move_to([target[0], target[1], F_TRANSIT_Z], GRASP_QUAT, F_SPEED_TRANSIT)
        self.hold(0.3)
        # Hover with the in-grasp offset corrected so the CUBE arrives above target.
        offset = self.data.xpos[bid] - self.tcp()
        cube_goal = target + [0, 0, F_ALIGN_DZ]
        self.move_to(target + [0, 0, F_HOVER_DZ] - offset, GRASP_QUAT, F_SPEED_DESCEND)
        self.hold(0.3)
        track()
        err_before = float(np.linalg.norm(self.data.xpos[bid] - target))
        # Fine alignment, closed loop on the measured cube.
        offset = self.data.xpos[bid] - self.tcp()
        cmd = cube_goal - offset
        self.move_to(cmd, GRASP_QUAT, F_SPEED_DESCEND)
        self.hold(0.3)
        align_err = None
        for _ in range(F_ALIGN_ITERS):
            err = self.data.xpos[bid] - cube_goal
            align_err = float(np.linalg.norm(err[:2]))
            if align_err < F_ALIGN_TOL:
                break
            cmd = cmd - err
            self.move_to(cmd, GRASP_QUAT, F_SPEED_DESCEND, min_steps=400)
            self.hold(0.25)
            track()
        # Seat press, closed loop until the cube height stops at the seat.
        seated_z = False
        for _ in range(F_PRESS_ITERS):
            if self.data.xpos[bid][2] - target[2] < F_SEAT_TOL:
                seated_z = True
                break
            cmd = cmd - [0, 0, F_PRESS_STEP]
            self.move_to(cmd, GRASP_QUAT, F_SPEED_DESCEND, min_steps=400)
            self.hold(0.25)
            track()
        self.hold(0.3)
        track()
        # Error at release (in the gripper, on the seat) before the fingers open.
        err_at_release = float(np.linalg.norm(self.data.xpos[bid] - target))
        # Release and retreat.
        self.grip(GRIPPER_OPEN, F_T_OPEN)
        self.move_to(target + [0, 0, F_RETREAT_DZ], GRASP_QUAT, F_SPEED_DESCEND)
        self.hold(F_T_SETTLE_BLOCK)
        track()
        err_after = float(np.linalg.norm(self.data.xpos[bid] - target))
        return self._drop_report(i, target, grasped, err_before, err_after,
                                 peak_arm, max_disturb, seated=seated_z,
                                 align_err=align_err, err_at_release=err_at_release)

    def _drop_report(self, i, target, grasped, err_before, err_after, peak_arm,
                     max_disturb, *, seated, align_err=None, err_at_release=None):
        bid = self.cube_bids[i]
        rot = _quat_angle_np(np.array([1.0, 0, 0, 0]), self.data.xquat[bid].copy())
        # Arm knocked the structure over if a big disturbance came with contact.
        knocked = bool(max_disturb > F_TOPPLE_DISTURB and peak_arm > 0.0)
        if not grasped:
            outcome = "grasp_failed"
        elif knocked:
            outcome = "executor_knocked"
        elif seated:
            outcome = "seated"
        else:
            outcome = "placed"
        return {
            "step": i,
            "protocol": "drop",
            "grasped": bool(grasped),
            "placement_error_before_press_mm":
                None if err_before is None else err_before * 1000.0,
            "placement_error_at_release_mm":
                None if err_at_release is None else err_at_release * 1000.0,
            "placement_error_after_press_mm":
                None if err_after is None else err_after * 1000.0,
            "align_err_mm": None if align_err is None else align_err * 1000.0,
            "press_seated": bool(seated),
            "peak_arm_contact_N": float(peak_arm),
            "struct_disturb_m": float(max_disturb),
            "struct_disturb_rel": float(max_disturb) / self.L,
            "struct_rot": float(rot),
            "peak_push": float(peak_arm),
            "outcome": outcome,
            "knocked": knocked,
        }

    # -- ride-under: admittance push then a force-capped seat correction ----

    def place_ride_under(self, i, params, layer, placed_idxs, bridge_idx):
        from keystone.interop.franka_scene import GRIPPER_CLOSE, GRIPPER_OPEN

        info = self.info
        g = _ride_geometry(params, info.target_world[i], layer, info.dx,
                           self.scale, self.base_offset)
        sign = g["sign"]
        tilt_deg = g["tilt_deg"]
        x_seat, z_seat = g["x_seat"], g["z_seat"]
        z_ride, x_start = g["z_ride"], g["x_start"]
        x_engage, x_deep = g["x_engage"], g["x_deep"]
        grasp_dx = g["grasp_dx"]
        rid = self.cube_bids[i]
        rgeom = self.cube_gids[i]
        bid = self.cube_bids[bridge_idx] if bridge_idx is not None else None
        # The support the reacher rides on is the layer-(L-1) cube (base).
        support_idxs = [k for k in placed_idxs
                        if info.cells[k][0] == layer - 1]
        basegeom = self.cube_gids[support_idxs[0]] if support_idxs else rgeom
        rw = float(self.model.body_mass[rid]) * FRANKA_G
        max_push = g["cap_x"] * rw

        base = {self.cube_bids[k]: self.data.xpos[self.cube_bids[k]].copy()
                for k in placed_idxs}
        stage = np.asarray(info.staging_world[i], np.float64)

        # Pick flat near the trailing end.
        gp = stage + [grasp_dx, 0.0, 0.0]
        self.move_to(gp + [0, 0, F_PRE_GRASP_DZ], _tilt_grasp_quat(0.0),
                     F_SPEED_TRANSIT)
        self.grip(GRIPPER_OPEN, 0.2)
        self.move_to(gp, _tilt_grasp_quat(0.0), 0.02)
        self.grip(GRIPPER_CLOSE, F_T_CLOSE)
        self.move_to(gp + [0, 0, F_PRE_GRASP_DZ], _tilt_grasp_quat(0.0), 0.03)
        grasped = bool(self.data.xpos[rid][2] - stage[2] > 0.03)

        # Carry tilted to the pre-push pose over the support top.
        self.move_to([x_start, FRANKA_STAGE_Y, F_TRANSIT_Z + 0.02],
                     _tilt_grasp_quat(0.0), F_SPEED_TRANSIT)
        self.move_to([_grasp_world(x_start, z_ride + 0.05, tilt_deg, grasp_dx)[0],
                      0.0, F_TRANSIT_Z + 0.02], _tilt_grasp_quat(tilt_deg),
                     F_SPEED_TRANSIT)
        self.move_to(_grasp_world(x_start, z_ride, tilt_deg, grasp_dx),
                     _tilt_grasp_quat(tilt_deg), 0.02, min_steps=400)
        self.hold(0.3)

        bpos0 = self.data.xpos[bid].copy() if bid is not None else None
        bang0 = _y_angle(self.data.xquat[bid]) if bid is not None else 0.0

        # Admittance thread: advance x (gated by push) and lower z (gated by
        # bridge lift) while pitching to flat; the bridge presses the reacher flat.
        peak_push = peak_lift = peak_drag = 0.0
        max_bdx = max_pivot = max_disturb = peak_arm = 0.0
        push_profile = []
        xcur, zcur = x_start, z_ride
        vx = (x_seat - x_start) / (F_RU_THREAD_T / self.timestep)
        vz = (z_seat - z_ride) / (F_RU_THREAD_T / self.timestep)
        kmax = int(round(F_RU_KMAX_T / self.timestep))
        kk = 0
        while sign * (xcur - x_deep) > 0 and kk < kmax:
            kk += 1
            lift = self.contact_force(rgeom, bid) if bid is not None else np.zeros(3)
            drag = self.contact_force(rgeom, basegeom)
            push_r = abs(drag[0]) + abs(lift[0])
            pry = abs(lift[2])
            step_x = vx if push_r <= max_push else -abs(vx) * 6.0
            xcur = (max(x_deep, xcur + step_x) if sign > 0
                    else min(x_deep, xcur + step_x))
            step_z = vz if pry <= F_RU_LIFT_CAP else -abs(vz) * 6.0
            zcur = max(z_seat + F_RU_Z_END_MARGIN, zcur + step_z)
            f = (0.0 if sign * (xcur - x_engage) >= 0
                 else min(1.0, sign * (x_engage - xcur) / max(1e-9,
                          abs(x_engage - x_seat))))
            deg = tilt_deg * (1.0 - f)
            self.scratch.qpos[:] = self.data.qpos
            from keystone.interop.franka_scene import dls_ik
            qg, _, _ = dls_ik(self.model, self.scratch,
                              _grasp_world(xcur, zcur, deg, grasp_dx),
                              _tilt_grasp_quat(deg), iters=40)
            self.data.ctrl[:7] = qg
            self.step()
            peak_push = max(peak_push, push_r)
            peak_lift = max(peak_lift, pry)
            peak_drag = max(peak_drag, abs(drag[0]))
            peak_arm = max(peak_arm, self.arm_touch(placed_idxs))
            max_disturb = max(max_disturb, self.struct_disturb(base))
            if bid is not None:
                bdx = abs(self.data.xpos[bid][0] - bpos0[0])
                max_bdx = max(max_bdx, bdx)
                max_pivot = max(max_pivot, -(_y_angle(self.data.xquat[bid]) - bang0))
            if kk % 40 == 0:
                push_profile.append({
                    "t": round(float(self.data.time), 3),
                    "reacher_x": float(self.data.xpos[rid][0]),
                    "push_N": round(push_r, 4), "lift_N": round(float(lift[2]), 4),
                    "bridge_dx_mm": round(max_bdx * 1000.0, 3)})

        # Hold at the seat so the bridge settles onto the reacher.
        for _ in range(int(round(0.4 / self.timestep))):
            lift = self.contact_force(rgeom, bid) if bid is not None else np.zeros(3)
            self.step()
            peak_lift = max(peak_lift, abs(lift[2]))
            if bid is not None:
                max_bdx = max(max_bdx, abs(self.data.xpos[bid][0] - bpos0[0]))
            max_disturb = max(max_disturb, self.struct_disturb(base))

        seat_target = np.array([x_seat, 0.0, z_seat])
        seat_err_before = float(np.linalg.norm(self.data.xpos[rid] - seat_target))
        reach_tilt = float(np.degrees(_y_angle(self.data.xquat[rid])))

        # New seat correction: a gentle force-capped press toward the certified
        # seat while still gripping. Closed loop on the measured reacher: nudge
        # the TCP command by the reacher-to-seat error (franka_build press form),
        # backing the gain off when the push reaction exceeds the cap so the
        # press stays gentle. The push-offset is along the thread axis, so the
        # correction closes the horizontal offset and never presses the reacher
        # down onto its exact seat (that over-seats the zero-margin clamp).
        seat_iters_used = 0
        cmd = self.tcp()
        for _ in range(F_SEAT_CORR_ITERS):
            err = self.data.xpos[rid] - seat_target
            err[2] = 0.0                              # horizontal correction only
            if float(np.linalg.norm(err)) < F_SEAT_CORR_TOL:
                break
            lift = self.contact_force(rgeom, bid) if bid is not None else np.zeros(3)
            drag = self.contact_force(rgeom, basegeom)
            push_r = abs(drag[0]) + abs(lift[0])
            gain = F_SEAT_CORR_GAIN if push_r <= max_push else 0.2 * F_SEAT_CORR_GAIN
            cmd = cmd - gain * err
            self.move_to(cmd, _tilt_grasp_quat(0.0), F_SEAT_CORR_SPEED, min_steps=300)
            self.hold(0.2)
            peak_push = max(peak_push, push_r)
            peak_lift = max(peak_lift, abs(lift[2]))
            peak_arm = max(peak_arm, self.arm_touch(placed_idxs))
            if bid is not None:
                max_bdx = max(max_bdx, abs(self.data.xpos[bid][0] - bpos0[0]))
            max_disturb = max(max_disturb, self.struct_disturb(base))
            seat_iters_used += 1
        seat_err_after = float(np.linalg.norm(self.data.xpos[rid] - seat_target))

        # Dwell, open, dwell so the fingers clear, then retreat.
        self.hold(0.3)
        self.grip(GRIPPER_OPEN, 0.1)
        self.hold(0.5)
        self.move_to([x_seat + sign * 0.02, FRANKA_STAGE_Y * 0.5, z_ride + 0.12],
                     _tilt_grasp_quat(0.0), 0.06)
        self.hold(0.3)

        reach_err = float(np.linalg.norm(self.data.xpos[rid] - seat_target))
        rammed = bool(max_bdx > F_RU_DRAG_FAIL)
        knocked = bool(max_disturb > F_TOPPLE_DISTURB and (peak_arm > 0.0 or rammed))
        if not grasped:
            outcome = "grasp_failed"
        elif rammed or knocked:
            outcome = "executor_rammed"
        elif seat_err_after < 3.0e-3 and abs(reach_tilt) < 5.0:
            outcome = "seated"
        else:
            outcome = "settle_marginal"
        return {
            "step": i,
            "protocol": "ride_under",
            "grasped": bool(grasped),
            "reach_err_mm": reach_err * 1000.0,
            "reach_tilt_deg": reach_tilt,
            "seat_err_before_correction_mm": seat_err_before * 1000.0,
            "seat_err_after_correction_mm": seat_err_after * 1000.0,
            "seat_correction_iters": seat_iters_used,
            "peak_push_N": float(peak_push),
            "push_cap_N": float(max_push),
            "peak_lift_N": float(peak_lift),
            "peak_drag_N": float(peak_drag),
            "max_bridge_dx_mm": float(max_bdx) * 1000.0,
            "max_pivot_deg": float(np.degrees(max_pivot)),
            "peak_arm_contact_N": float(peak_arm),
            "struct_disturb_m": float(max_disturb),
            "struct_disturb_rel": float(max_disturb) / self.L,
            "struct_rot": 0.0,
            "peak_push": float(peak_push),
            "push_profile": push_profile,
            "outcome": outcome,
            "knocked": bool(rammed or knocked),
        }

    def retract_props(self):
        """Retract the props one at a time on their sliders (falsework protocol).
        A no-op with an empty prop list."""
        if not self.prop_aids:
            return None
        self.hold(0.75)
        pos0 = {b: self.data.xpos[b].copy() for b in self.cube_bids}
        stages = []
        n_ramp = max(1, int(round(1.5 / self.timestep)))
        for k, p in enumerate(self.info.props):

            def load():
                buf = np.zeros(6)
                tot = 0.0
                for c in range(self.data.ncon):
                    con = self.data.contact[c]
                    if self.prop_gids[k] in (int(con.geom1), int(con.geom2)):
                        self.mj.mj_contactForce(self.model, self.data, c, buf)
                        tot += float(np.linalg.norm(buf[:3]))
                return tot

            load_before = load()
            peak = load_before
            for t in range(n_ramp):
                self.data.ctrl[self.prop_aids[k]] = p["retract_disp"] * (t + 1) / n_ramp
                self.step()
                peak = max(peak, load())
            self.hold(0.75)
            stages.append({"prop": p["name"], "axis": p["axis"],
                           "load_before_N": load_before,
                           "peak_during_retract_N": peak, "load_after_N": load()})
        self.hold(1.0)
        disp = max(float(np.linalg.norm(self.data.xpos[b] - pos0[b]))
                   for b in self.cube_bids)
        stands = bool(disp / self.L < 0.01)
        return {"stages": stages, "disp_rel": disp / self.L,
                "verdict": "stands" if stands else "collapsed"}

    def extract_boxes(self):
        """As-built cube poses as Franka-scale keystone Boxes (pedestal-relative),
        for the stiff-contact settle verdict."""
        boxes = [box_2d(6.0 * self.scale, 1.0 * self.scale, -3.0 * self.scale,
                        0.5 * self.scale, depth=self.scale)]
        h = 0.5 * self.scale
        for b in self.cube_bids:
            pos = self.data.xpos[b].copy() - self.base_offset
            quat = self.data.xquat[b].copy()
            boxes.append(Box(np.array([h, h, h]), pos, quat, 2000.0))
        return boxes


def _quat_angle_np(q0, q1):
    dot = min(1.0, abs(float(np.dot(q0, q1))))
    return 2.0 * float(np.arccos(dot))


def _execute_franka(plan_blocks, seq, dx, mu, recorder, timestep,
                    settle_duration=2.0, scale=FRANKA_S):
    """Drive the exact plan with the Franka arm at the table-top scale.

    Composes the scene with the base at the overhang end, verifies reach for
    every waypoint, restages the cubes on the arm side, executes each block by
    its protocol, retracts any props, and returns the execute-stage record with
    the arm-specific fields and the settle verdict."""
    import mujoco

    from keystone.interop.franka_scene import (
        BASE_OFFSET, compose_scene, reset_home,
    )
    from keystone.interop.mujoco_io import settle_test

    base_offset = np.asarray(BASE_OFFSET, dtype=np.float64)
    cells = [(int(L), int(j)) for (L, j) in seq]
    cell_names = [b["protocol"] + f"_{b['step']}" for b in plan_blocks]
    prop_specs = _plan_to_prop_specs(plan_blocks, dx)

    # Base placement and reach verification.
    base_pos, base_yaw, staging, reach_rows, moved, worst_pos = _place_franka_base(
        plan_blocks, cells, cell_names, dx, scale, base_offset, prop_specs)
    reach_ok = worst_pos < FRANKA_REACH_POS_TOL

    spec, info = compose_scene(scale=scale, base_offset=base_offset, dx=dx,
                               cells=cells, cell_names=cell_names,
                               prop_specs=prop_specs, staging=staging,
                               timestep=timestep, arm_base_pos=base_pos,
                               arm_base_yaw=base_yaw)
    model = spec.compile()
    driver = _FrankaDriver(model, info, timestep, recorder, mu, scale, base_offset)
    reset_home(model, driver.data)
    if recorder is not None:
        recorder.bind(model, driver.data)
    driver.hold(0.3)

    step_reports = []
    failed_protocol = None
    failure_detail = None
    placed = []
    for blk in plan_blocks:
        i = blk["step"]
        protocol = blk["protocol"]
        if protocol == "ride_under":
            blocker = blk["params"].get("blocker")
            bridge_idx = None
            if blocker is not None:
                bridge_idx = next((k for k, c in enumerate(cells)
                                   if [c[0], c[1]] == list(blocker)), None)
            rep = driver.place_ride_under(i, blk["params"], blk["layer"], placed,
                                          bridge_idx)
        else:
            # drop and prop both seat with the press; the prop is the scene
            # support that catches an otherwise unsupported drop.
            rep = driver.place_drop(i, placed)
        step_reports.append(rep)
        placed.append(i)
        if rep.get("knocked"):
            failed_protocol = protocol
            failure_detail = (
                f"step {i} ({protocol}): arm knocked the structure "
                f"(disturb {rep['struct_disturb_m'] * 1000.0:.1f} mm"
                + (f", bridge {rep.get('max_bridge_dx_mm', 0.0):.1f} mm"
                   if protocol == "ride_under" else "")
                + ")")
            break
        if not rep.get("grasped", True):
            failed_protocol = protocol
            failure_detail = f"step {i} ({protocol}): grasp failed"
            break

    retraction = None
    if failed_protocol is None:
        retraction = driver.retract_props()

    # As-built correctness: how far each cube sits from its world target in the
    # live sim. A block grossly off target means the intended structure was not
    # produced (it slid or fell), even if the extracted pile then settles still.
    as_built = []
    as_built_max = 0.0
    for k, b in enumerate(driver.cube_bids):
        err = float(np.linalg.norm(driver.data.xpos[b] - info.target_world[k]))
        as_built.append({"block": info.cube_names[k], "as_built_error_mm": err * 1000.0})
        as_built_max = max(as_built_max, err)
    built_ok = as_built_max < F_BUILD_TOL

    settled = driver.extract_boxes()
    settle = settle_test(settled, mu, duration=settle_duration, solref=STIFF_SOLREF)
    # A knife-edge optimum can creep slowly; a longer settle is the honest check.
    settle6 = settle_test(settled, mu, duration=6.0, solref=STIFF_SOLREF)

    if failed_protocol is not None:
        verdict = "executor_failed"
    elif retraction is not None and retraction["verdict"] != "stands":
        verdict = "executor_failed"
        failed_protocol = "prop"
        failure_detail = (f"structure moved after prop retraction: disp_rel="
                          f"{retraction['disp_rel']:.4f}")
    elif not built_ok:
        # The arm placed every block without knocking the structure over, but a
        # block did not hold on the seat prop-free. Dynamic, not an arm fault.
        verdict = "certified_but_dynamic_fail"
        failure_detail = (f"as-built structure off target: worst block "
                          f"{as_built_max * 1000.0:.1f} mm from its seat")
    elif settle["stable"] and settle6["stable"]:
        verdict = "stands"
    else:
        verdict = "certified_but_dynamic_fail"

    if recorder is not None:
        recorder.capture(force=True)

    return {
        "executor": FRANKA_EXECUTOR,
        "scale_m": scale,
        "cube_mass_kg": 2000.0 * scale ** 3,
        "base_pose": {"x": float(base_pos[0]), "y": float(base_pos[1]),
                      "z": float(base_pos[2]), "yaw_rad": float(base_yaw)},
        "base_placement": moved,
        "reach_ok": bool(reach_ok),
        "worst_reach_mm": float(worst_pos) * 1000.0,
        "reach_waypoints": reach_rows,
        "staging_world": [list(map(float, s)) for s in staging],
        "steps": step_reports,
        "as_built": as_built,
        "as_built_max_err_mm": as_built_max * 1000.0,
        "built_ok": bool(built_ok),
        "retraction": retraction,
        "settle": {
            "verdict": settle["verdict"],
            "stable": bool(settle["stable"]),
            "max_disp_rel": settle["max_disp_rel"],
            "max_rot": settle["max_rot"],
            "traj_max_disp_rel": settle["traj_max_disp_rel"],
            "traj_max_rot": settle["traj_max_rot"],
            "duration": settle_duration,
        },
        "settle_6s": {
            "verdict": settle6["verdict"],
            "stable": bool(settle6["stable"]),
            "max_disp_rel": settle6["max_disp_rel"],
            "max_rot": settle6["max_rot"],
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
    executor="driver",
    sequence=None,
    prop_steps=None,
    tag="",
    timestep=5e-4,
    tol=None,
    progress=None,
    settle_duration=2.0,
    verbose=True,
):
    """Evaluate a stacking design end to end and return the JSON record.

    n cubes on a dx grid. Runs search, host certification, build planning,
    MuJoCo execution, and a movie, writing {out_dir}/pipeline_n{n}_seed{seed}
    {tag}.json/.mp4/.gif/_final.png. record=False skips the movie (the test path).

    executor selects the EXECUTE stage: "driver" (default) is the capped
    impedance driver, "franka" is the menagerie Panda based at the overhang
    end. sequence, when given, replaces the search with a fixed (layer, j) list
    (a known control build); the other stages are unchanged. prop_steps, a list
    of build steps, forces those placements onto a transient falsework prop
    (retracted at the end), a build aid for a knife-edge cube whose prop-free
    drop is dynamically fragile; the certificate is unchanged.
    """
    tol = tol or Tolerances()
    os.makedirs(out_dir, exist_ok=True)
    suffix = ("" if executor == "driver" else f"_{executor}") + tag
    base = os.path.join(out_dir, f"pipeline_n{n}_seed{seed}{suffix}")

    from keystone.search import lattice as LT

    record_out = {
        "n": int(n),
        "dx": float(dx),
        "sims": int(sims),
        "seed": int(seed),
        "executor": EXECUTOR if executor == "driver" else FRANKA_EXECUTOR,
        "harmonic": LT.harmonic(n),
        "notes": [],
    }

    # Stage 1: search, or a fixed control sequence.
    if sequence is not None:
        seq = [(int(L), int(j)) for (L, j) in sequence]
        best = LT.overhang(seq, dx)
        search = None
        search_info = {"prior": "fixed_sequence", "checkpoint_loaded": False,
                       "note": "search skipped; fixed control sequence executed"}
    else:
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
    if prop_steps:
        # Force the named build steps onto a transient falsework prop. A knife-
        # edge cube whose column is clear classifies as a drop, but its prop-free
        # arm-drop is dynamically fragile; the prop catches it until the clamp
        # closes, then retracts. The static certificate is unaffected.
        placed = []
        for blk in plan_blocks:
            if blk["step"] in prop_steps and blk["protocol"] != "prop":
                blk["protocol"] = "prop"
                blk["params"] = _protocol_params(
                    "prop", blk["layer"], blk["j"], dx, placed, blk["step"])
                record_out["notes"].append(
                    f"step {blk['step']} forced to a transient prop (prop_steps)")
            placed.append((blk["layer"], blk["j"]))
    counts = {"drop": 0, "ride_under": 0, "prop": 0}
    for b in plan_blocks:
        counts[b["protocol"]] += 1
    record_out["plan"] = {"blocks": plan_blocks, "protocol_counts": counts}

    # Stage 4: execute (with the movie recorder across the whole build).
    if executor == "franka":
        # The arm build is long (tens of thousands of steps), so a big stride
        # and a modest frame size keep the GIF small; the MP4 stays smooth.
        recorder = FrameRecorder(stride=250, record=record, height=360, width=640,
                                 fps=30, camera_overrides=FRANKA_CAMERA,
                                 brighten=1.4)
    else:
        recorder = FrameRecorder(stride=40, record=record)
    execute = None
    try:
        if executor == "franka":
            execute = _execute_franka(
                plan_blocks, seq, dx, MU, recorder, timestep,
                settle_duration=settle_duration
            )
        else:
            execute = _execute(
                plan_blocks, dx, MU, recorder, timestep,
                settle_duration=settle_duration
            )
    except Exception as exc:  # noqa: BLE001
        record_out["notes"].append(
            f"execution raised {type(exc).__name__}: {exc}"
        )
        execute = {
            "executor": EXECUTOR if executor == "driver" else FRANKA_EXECUTOR,
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
