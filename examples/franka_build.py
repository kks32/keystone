"""Franka Panda builds the clamp 29/24 falsework design, end to end (M5 story).

This replaces the invisible-hand impedance driver of examples/mujoco_falsework.py
with a real manipulator: a menagerie Franka Panda with its two-finger gripper
picks each cube from a staging row, places it by top grasp, the falsework props
retract on their sliders, and the structure settles. It records what
manipulation costs that the disembodied driver never paid: grasp forces,
placement error through a real kinematic chain, and structure disturbance from
finger contact.

The scale trick. keystone verdicts and dimensionless margins are scale-invariant
(property-tested in tests/property/test_invariants.py). The unit-cube lattice is
rescaled to cube side 0.05 m: pedestal 0.30 x 0.30 x 0.05 m, cubes 0.25 kg at
density 2000, inside the Panda payload and reach. The scaled design is
re-certified through the host pipeline before the arm moves, and the
dimensionless margins are asserted to match the unit-scale ones to float
precision. Gravity is standard.

Control. Damped-least-squares differential IK on the TCP site jacobian drives
the menagerie position actuators (seven arm servos plus one tendon servo for
the fingers). Waypoints per block: home, pre-grasp above the staging cube
(y-pinch: fingers straddle the cube across y, which the planar-in-xz design
keeps free), descend, close force-limited, lift 5 cm and verify the cube came
along, transport above the target, then a closed-loop endgame that measures
the carried cube's pose (a vision stand-in): corrected hover, iterated
alignment to sub-half-millimeter, and a press onto the seat before the fingers
open. The open-loop arm alone misses by 3 to 27 mm (in-grasp slip, servo sag,
finger-opening drag); the numbers are in KNOWN_LIMITS.md. End-effector speed
is capped; everything is deterministic.

Build order is the certified falsework sequence for clamp 29/24: base, reacher
onto its prop, counterweight onto its prop, bridge, then staged prop retraction
(counterweight prop slides out along x, reacher prop lowers like a jack) and a
final stiff-contact settle verdict.

Run:

    python examples/franka_build.py --out out/mujoco
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

from keystone import (
    Tolerances,
    assemble,
    build_assembly,
    solve_p0,
    solve_p4,
)
from keystone.interop.franka_scene import (
    CELL_NAMES,
    DESIGN_MU,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    HOME_Q,
    OVERHANG,
    PROPS_UNIT,
    S,
    compose_scene,
    design_cert_boxes,
    dls_ik,
    pedestal_cert,
    prop_cert_boxes,
    reset_home,
)

TOL = Tolerances()

# Speed caps, m/s at the end effector. Transit is free space; descend and lift
# are near-contact moves.
SPEED_TRANSIT = 0.10
SPEED_DESCEND = 0.03

# Waypoint offsets, meters.
PRE_GRASP_DZ = 0.12       # hover above the staging cube
LIFT_DZ = 0.05            # grasp-check lift
TRANSIT_Z = 0.30          # cruise height, clear of the whole structure
HOVER_DZ = 0.020          # fine-alignment hover above the target
ALIGN_DZ = 0.002          # aligned hold height above the target center
ALIGN_TOL = 4e-4          # in-plane alignment tolerance before the press
ALIGN_ITERS = 3           # closed-loop alignment iterations
PRESS_STEP = 0.001        # closed-loop press increment
PRESS_ITERS = 5           # press iterations (seat check each step)
SEAT_TOL = 4e-4           # cube z above target that counts as seated
RETREAT_DZ = 0.10         # rise after release

# Grasp-check thresholds. The cube must follow the lift.
GRASP_TRACK_TOL = 0.010   # m, cube-to-TCP tracking error after the lift

# Phase timings, seconds.
T_CLOSE = 0.6
T_OPEN = 0.5
T_SETTLE_BLOCK = 0.8

# Retraction protocol (scaled mujoco_falsework values).
T_RELAX = 0.75
T_RAMP = 1.5
T_STAGE = 0.75
T_FINAL = 1.5

# Stands verdict thresholds (mujoco_falsework convention).
STAND_DISP_REL = 0.01
STAND_ROT = 0.05


def certify(boxes, mu):
    """Certified P0 verdict and P4 margin from the host pipeline."""
    a = build_assembly(boxes, mu=mu, tol=TOL, dim=2)
    s = assemble(a, TOL, cone="linear2d")
    return solve_p0(s, TOL).status, float(solve_p4(s, TOL).margin)


def recertify_scaled():
    """Certify the design and every propped build state at unit scale and at
    the Panda scale. Dimensionless margins must match to float precision."""
    from keystone.interop.franka_scene import CELLS, cube_cert

    rows = []
    for label, scale in (("unit", 1.0), ("scaled", S)):
        ped = pedestal_cert(scale)
        props = prop_cert_boxes(scale)
        states = []
        placed = []
        for (layer, j), name in zip(CELLS, CELL_NAMES):
            placed.append(cube_cert(layer, j, scale))
            st, m = certify([ped] + placed + props, DESIGN_MU)
            states.append((f"place {name} (propped)", st, m))
        remaining = list(props)
        for p in PROPS_UNIT:
            remaining = remaining[1:]
            st, m = certify([ped] + placed + remaining, DESIGN_MU)
            states.append((f"retract {p['name']} prop", st, m))
        rows.append((label, scale, states))

    print(f"=== scale re-certification: clamp 29/24 (mu={DESIGN_MU}) ===")
    print(f"  {'state':28s} {'unit margin':>14s} {'scaled margin':>14s} {'|diff|':>10s}")
    table = []
    (_, _, unit_states), (_, _, scaled_states) = rows
    for (name, st_u, m_u), (_n, st_s, m_s) in zip(unit_states, scaled_states):
        assert st_u == st_s, (name, st_u, st_s)
        diff = abs(m_u - m_s)
        assert diff < 1e-15, (name, m_u, m_s, diff)
        assert st_u == "feasible", (name, st_u)
        print(f"  {name:28s} {m_u:14.6e} {m_s:14.6e} {diff:10.1e}")
        table.append({"state": name, "status": st_u,
                      "margin_unit": m_u, "margin_scaled": m_s, "abs_diff": diff})
    print("  all states feasible; margins match to float precision")
    return table


def _quat_angle(q0, q1):
    d = min(1.0, abs(float(np.dot(q0, q1))))
    return 2.0 * float(np.arccos(d))


class BuildDriver:
    """Waypoint controller plus metric and frame recording."""

    def __init__(self, model, info, timestep, frame_stride=0, frame_size=(300, 420)):
        import mujoco

        self.mujoco = mujoco
        self.model = model
        self.info = info
        self.data = mujoco.MjData(model)
        self.scratch = mujoco.MjData(model)
        self.timestep = timestep
        self.sid = model.site(info.tcp_site).id
        self.cube_bids = [model.body(b).id for b in info.cube_bodies]
        self.cube_gids = [model.geom(g).id for g in info.cube_geoms]
        self.pad_gids = [model.geom(g).id for g in info.finger_pads]
        self.grip_aid = model.actuator(info.gripper_actuator).id
        self.prop_aids = [model.actuator(p["act"]).id for p in info.props]
        self.prop_gids = [model.geom(p["geom"]).id for p in info.props]
        # Structure diagonal for dimensionless disturbance (certified geometry).
        corners = np.concatenate(
            [b.corners() for b in design_cert_boxes(info.scale)], axis=0)
        self.L = float(np.linalg.norm(corners.max(axis=0) - corners.min(axis=0)))
        # Frame capture.
        self.frame_stride = frame_stride
        self.frames = []
        self.renderer = None
        self.cam = None
        if frame_stride > 0:
            self.renderer = mujoco.Renderer(model, height=frame_size[0],
                                            width=frame_size[1])
            self.cam = self._camera()
        self.nstep = 0
        # Per-phase metric state.
        self.carried = None            # cube index being manipulated
        self.placed = []               # cube indices already placed
        self.baseline = {}             # placed cube id -> (pos, quat) baseline
        self.metrics = None

    def _camera(self):
        cam = self.mujoco.MjvCamera()
        self.mujoco.mjv_defaultFreeCamera(self.model, cam)
        cam.lookat[:] = [0.33, -0.05, 0.12]
        cam.distance = 1.15
        cam.azimuth = 150.0
        cam.elevation = -18.0
        return cam

    # -- low-level stepping with metrics --------------------------------

    def _pad_forces(self):
        """(carried_force, clearance_events) from pad contacts this step.
        carried_force: total normal force between the pads and the carried
        cube. clearance_events: list of (pad, other cube index, force) for pad
        contact with any cube that is not the carried one."""
        mj = self.mujoco
        d = self.data
        buf = np.zeros(6)
        carried_gid = (self.cube_gids[self.carried]
                       if self.carried is not None else -1)
        f_carried = 0.0
        events = []
        for c in range(d.ncon):
            con = d.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            pad = g1 in self.pad_gids or g2 in self.pad_gids
            if not pad:
                continue
            other = g2 if g1 in self.pad_gids else g1
            if other not in self.cube_gids:
                continue
            mj.mj_contactForce(self.model, d, c, buf)
            fn = abs(float(buf[0]))
            if other == carried_gid:
                f_carried += fn
            else:
                events.append((self.cube_gids.index(other), fn))
        return f_carried, events

    def step(self):
        self.mujoco.mj_step(self.model, self.data)
        self.nstep += 1
        m = self.metrics
        if m is not None:
            f_carried, events = self._pad_forces()
            m["peak_finger_force"] = max(m["peak_finger_force"], f_carried)
            for idx, fn in events:
                key = self.info.cube_names[idx]
                m["clearance_contacts"][key] = max(
                    m["clearance_contacts"].get(key, 0.0), fn)
            if self.baseline:
                disturb = max(
                    float(np.linalg.norm(
                        self.data.xpos[b] - self.baseline[b][0]))
                    for b in self.baseline)
                m["disturb"] = max(m["disturb"], disturb)
        if self.renderer is not None and self.nstep % self.frame_stride == 0:
            self.renderer.update_scene(self.data, camera=self.cam)
            img = self.renderer.render().copy()
            self.frames.append(
                np.clip(img.astype(np.float32) * 1.4, 0, 255).astype(np.uint8))

    def hold(self, t):
        for _ in range(max(1, int(round(t / self.timestep)))):
            self.step()

    # -- waypoint motion --------------------------------------------------

    def tcp(self):
        return self.data.site_xpos[self.sid].copy()

    def move_to(self, target_pos, speed, min_steps=100):
        """IK to the target TCP pose, then linear joint-space ramp with the
        end-effector speed capped by the segment length."""
        q0 = np.array(self.data.ctrl[:7])
        self.scratch.qpos[:] = self.data.qpos
        q_goal, ep, er = dls_ik(
            self.model, self.scratch, target_pos, self.info.grasp_quat)
        dist = float(np.linalg.norm(np.asarray(target_pos) - self.tcp()))
        n = max(min_steps, int(round(dist / speed / self.timestep)))
        for t in range(n):
            a = (t + 1) / n
            self.data.ctrl[:7] = q0 + a * (q_goal - q0)
            self.step()
        return ep, er

    def grip(self, ctrl, t):
        self.data.ctrl[self.grip_aid] = ctrl
        self.hold(t)

    # -- per-block build --------------------------------------------------

    def place_block(self, i):
        """Pick cube i from staging and place it at its target. Returns the
        report dict; report["grasped"] False aborts the build."""
        info = self.info
        stage = info.staging_world[i]
        target = info.target_world[i]
        self.carried = i
        self.baseline = {
            self.cube_bids[k]: (self.data.xpos[self.cube_bids[k]].copy(),
                                self.data.xquat[self.cube_bids[k]].copy())
            for k in self.placed
        }
        self.metrics = {
            "peak_finger_force": 0.0,
            "disturb": 0.0,
            "clearance_contacts": {},
        }
        bid = self.cube_bids[i]

        # Pick.
        self.move_to(stage + [0.0, 0.0, PRE_GRASP_DZ], SPEED_TRANSIT)
        self.grip(GRIPPER_OPEN, 0.2)
        self.move_to(stage, SPEED_DESCEND)
        self.grip(GRIPPER_CLOSE, T_CLOSE)

        # Grasp check: lift and verify the cube came along.
        self.move_to(stage + [0.0, 0.0, LIFT_DZ], SPEED_DESCEND)
        track = float(np.linalg.norm(self.data.xpos[bid] - self.tcp()))
        rose = float(self.data.xpos[bid][2] - stage[2])
        grasped = bool(track < GRASP_TRACK_TOL and rose > 0.5 * LIFT_DZ)
        if not grasped:
            self.metrics.update(grasped=False, grasp_track=track,
                                grasp_rise=rose, phase="grasp-check")
            return self._finish_block(i, target, seated=False)

        # Transport: up, across at cruise height.
        self.move_to([stage[0], stage[1], TRANSIT_Z], SPEED_TRANSIT)
        self.move_to([target[0], target[1], TRANSIT_Z], SPEED_TRANSIT)
        self.hold(0.3)

        # In-grasp offset, measured at cruise (a vision system in a real
        # cell; the simulator pose here). The y-pinch constrains x only by
        # friction and the pads creep slightly, so the cube does not hang
        # exactly at the TCP. The hover descent is corrected by the offset so
        # the CUBE, not the TCP, arrives above the target with the full hover
        # clearance.
        offset = self.data.xpos[bid] - self.tcp()
        self.metrics["grasp_slip_mm"] = [float(v) * 1000.0 for v in offset]
        cube_goal = target + [0.0, 0.0, ALIGN_DZ]
        self.move_to(target + [0.0, 0.0, HOVER_DZ] - offset, SPEED_DESCEND)
        self.hold(0.3)

        # Fine alignment at the hover, closed loop: the residual after the
        # feed-forward correction is the arm servo's gravity lag, which the
        # loop removes because it measures the cube, not the joints.
        offset = self.data.xpos[bid] - self.tcp()
        cmd = cube_goal - offset
        self.move_to(cmd, SPEED_DESCEND)
        self.hold(0.3)
        align_err = None
        for _ in range(ALIGN_ITERS):
            err = self.data.xpos[bid] - cube_goal
            align_err = float(np.linalg.norm(err[:2]))
            if align_err < ALIGN_TOL:
                break
            cmd = cmd - err
            self.move_to(cmd, SPEED_DESCEND, min_steps=400)
            self.hold(0.25)
        self.metrics["align_err_mm"] = align_err * 1000.0

        # Press the cube onto its seat before opening: the seat contact then
        # holds it against the finger-opening drag (measured 1.7 mm when
        # released hanging free). Closed loop: step down until the measured
        # cube height stops at the seat (the arm servo lags a commanded press,
        # so an open-loop press can leave the cube hanging).
        seated_z = False
        for _ in range(PRESS_ITERS):
            if self.data.xpos[bid][2] - target[2] < SEAT_TOL:
                seated_z = True
                break
            cmd = cmd - [0.0, 0.0, PRESS_STEP]
            self.move_to(cmd, SPEED_DESCEND, min_steps=400)
            self.hold(0.25)
        self.metrics["press_seated"] = seated_z
        self.hold(0.3)

        # Release and retreat.
        self.grip(GRIPPER_OPEN, T_OPEN)
        self.move_to(target + [0.0, 0.0, RETREAT_DZ], SPEED_DESCEND)
        self.hold(T_SETTLE_BLOCK)
        self.metrics.update(grasped=True, grasp_track=track, grasp_rise=rose,
                            phase="done")
        return self._finish_block(i, target, seated=True)

    def _finish_block(self, i, target, seated):
        bid = self.cube_bids[i]
        pos = self.data.xpos[bid].copy()
        quat = self.data.xquat[bid].copy()
        err = float(np.linalg.norm(pos - target))
        m = self.metrics
        report = {
            "block": self.info.cube_names[i],
            "cell": i,
            "grasped": bool(m.get("grasped", False)),
            "grasp_track_m": float(m.get("grasp_track", np.nan)),
            "grasp_rise_m": float(m.get("grasp_rise", np.nan)),
            "grasp_slip_mm": m.get("grasp_slip_mm"),
            "align_err_mm": m.get("align_err_mm"),
            "press_seated": m.get("press_seated"),
            "peak_finger_force_N": float(m["peak_finger_force"]),
            "placement_error_mm": err * 1000.0,
            "placement_error_xyz_mm": [
                float(v) * 1000.0 for v in (pos - target)],
            "rot_error_rad": _quat_angle(np.array([1.0, 0, 0, 0]), quat),
            "struct_disturb_m": float(m["disturb"]),
            "struct_disturb_rel": float(m["disturb"]) / self.L,
            "clearance_contacts": {
                k: float(v) for k, v in m["clearance_contacts"].items()},
            "seated": bool(seated),
        }
        self.carried = None
        self.metrics = None
        if seated:
            self.placed.append(i)
        return report

    # -- prop retraction ---------------------------------------------------

    def prop_load(self, k):
        mj = self.mujoco
        buf = np.zeros(6)
        tot = 0.0
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            if self.prop_gids[k] in (int(con.geom1), int(con.geom2)):
                mj.mj_contactForce(self.model, self.data, c, buf)
                tot += float(np.linalg.norm(buf[:3]))
        return tot

    def retract_props(self):
        """Retract the props one at a time on their position-actuated sliders,
        in plan order, with ramps and settles (mujoco_falsework protocol)."""
        self.metrics = None
        self.hold(T_RELAX)
        pos0 = {b: self.data.xpos[b].copy() for b in self.cube_bids}
        quat0 = {b: self.data.xquat[b].copy() for b in self.cube_bids}
        stages = []
        n_ramp = max(1, int(round(T_RAMP / self.timestep)))
        for k, p in enumerate(self.info.props):
            load_before = self.prop_load(k)
            peak = load_before
            for t in range(n_ramp):
                self.data.ctrl[self.prop_aids[k]] = (
                    p["retract_disp"] * (t + 1) / n_ramp)
                self.step()
                peak = max(peak, self.prop_load(k))
            self.hold(T_STAGE)
            stages.append({
                "prop": p["name"],
                "axis": p["axis"],
                "load_before_N": load_before,
                "peak_during_retract_N": peak,
                "load_after_N": self.prop_load(k),
            })
        self.hold(T_FINAL)
        disp = max(float(np.linalg.norm(self.data.xpos[b] - pos0[b]))
                   for b in self.cube_bids)
        rot = max(_quat_angle(quat0[b], self.data.xquat[b])
                  for b in self.cube_bids)
        stands = bool(disp / self.L < STAND_DISP_REL and rot < STAND_ROT)
        return {
            "stages": stages,
            "disp_m": disp,
            "disp_rel": disp / self.L,
            "rot_rad": rot,
            "stands": stands,
            "verdict": "stands" if stands else "collapsed",
        }

    def final_state(self):
        """As-built cube poses against targets after everything settles."""
        rows = []
        for i, b in enumerate(self.cube_bids):
            err = float(np.linalg.norm(
                self.data.xpos[b] - self.info.target_world[i]))
            rows.append({
                "block": self.info.cube_names[i],
                "final_error_mm": err * 1000.0,
                "final_rot_rad": _quat_angle(
                    np.array([1.0, 0, 0, 0]), self.data.xquat[b]),
            })
        return rows

    def snapshot(self, path):
        """Full-resolution render of the current state."""
        mj = self.mujoco
        r = mj.Renderer(self.model, height=480, width=640)
        r.update_scene(self.data, camera=self.cam or self._camera())
        img = np.clip(r.render().astype(np.float32) * 1.4, 0, 255).astype(np.uint8)
        r.close()
        try:
            from PIL import Image

            Image.fromarray(img).save(path)
            return path
        except ImportError:
            return "no PNG writer (PIL) available"

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def gripper_clearance_note(info):
    """Where gripper clearance mattered, computed from the design geometry.

    The fingers straddle the carried cube across y (pads on its y faces), so
    the grasp needs the cube's y faces free; the design is planar in xz
    (every cube at y = 0), so they always are. The pads are not free in y
    though: pinched, their inner faces sit exactly on the structure's y =
    +-0.025 planes, overlapping the structure's y band by the pad thickness
    sliver, so the vertical clearances below are real contact questions, not
    hypothetical ones. The pads grip the cube's top half and protrude
    12.4 mm below its center, so they clear the seat plane of a lower
    neighbor by half the cube side minus that, 12.6 mm.
    Reported per placement: the minimum vertical clearance
    between the fingertip bottoms at seat and any placed cube or prop top
    inside the pad's x footprint."""
    from keystone.interop.franka_scene import CELLS, DX_GRID

    s = info.scale
    fingertip_below = 0.0124   # pad bottoms below the TCP (menagerie geometry)
    pad_half_x = 0.0085
    rows = []
    placed = []
    for i, (layer, j) in enumerate(CELLS):
        cx = j * DX_GRID * s + info.base_offset[0]
        cz = (1.5 + layer) * s
        tip_z = cz - fingertip_below
        sweep = (cx - pad_half_x, cx + pad_half_x)
        nearest = None
        for k in placed:
            (kl, kj) = CELLS[k]
            kx = kj * DX_GRID * s + info.base_offset[0]
            if kx - s / 2.0 < sweep[1] and kx + s / 2.0 > sweep[0]:
                gap = tip_z - (2.0 + kl) * s
                entry = {"over": info.cube_names[k],
                         "clearance_mm": gap * 1000.0}
                if nearest is None or gap < nearest["clearance_mm"] / 1000.0:
                    nearest = entry
        for p in info.props:
            plo, phi = p["x"] - p["half_w"], p["x"] + p["half_w"]
            if plo < sweep[1] and phi > sweep[0]:
                gap = tip_z - (p["zc"] + p["half_h"])
                entry = {"over": f"{p['name']} prop",
                         "clearance_mm": gap * 1000.0}
                if nearest is None or gap < nearest["clearance_mm"] / 1000.0:
                    nearest = entry
        rows.append({
            "block": info.cube_names[i],
            "min_fingertip_clearance": nearest,
            "y_faces_free": True,
        })
        placed.append(i)
    return rows


def write_movie(frames, out_dir, fps=25):
    """GIF via PIL (mujoco_shim.build_movie pattern) plus an mp4 when ffmpeg
    exists. Returns (gif_path_or_reason, mp4_path_or_reason)."""
    if not frames:
        return "no frames captured", "no frames captured"
    try:
        from PIL import Image
    except ImportError:
        return "movie skipped (PIL missing)", "mp4 skipped (PIL missing)"
    imgs = [Image.fromarray(f) for f in frames]
    gif = os.path.join(out_dir, "franka_build.gif")
    imgs[0].save(gif, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)
    ffmpeg = shutil.which("ffmpeg") or (
        "/opt/homebrew/bin/ffmpeg"
        if os.path.exists("/opt/homebrew/bin/ffmpeg") else None)
    if ffmpeg is None:
        return gif, "mp4 skipped (ffmpeg not found)"
    tmp = os.path.join(out_dir, "_franka_frames")
    os.makedirs(tmp, exist_ok=True)
    for k, im in enumerate(imgs):
        im.save(os.path.join(tmp, f"f{k:05d}.png"))
    mp4 = os.path.join(out_dir, "franka_build.mp4")
    proc = subprocess.run(
        [ffmpeg, "-y", "-framerate", str(fps), "-i",
         os.path.join(tmp, "f%05d.png"), "-pix_fmt", "yuv420p", mp4],
        capture_output=True)
    shutil.rmtree(tmp, ignore_errors=True)
    if proc.returncode != 0:
        return gif, f"mp4 failed (ffmpeg rc={proc.returncode})"
    return gif, mp4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/mujoco")
    ap.add_argument("--timestep", type=float, default=5e-4)
    ap.add_argument("--frame-stride", type=int, default=400,
                    help="capture a movie frame every N steps; 0 disables")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("mujoco not installed; install keystone[mujoco] to run this demo.")
        return

    # Step 0: certify the scaled design before the arm moves.
    cert_table = recertify_scaled()

    # Step 1: compose and reset the scene.
    spec, info = compose_scene(timestep=args.timestep)
    model = spec.compile()
    driver = BuildDriver(model, info, args.timestep,
                         frame_stride=args.frame_stride)
    reset_home(model, driver.data)
    driver.hold(0.3)

    clearance = gripper_clearance_note(info)

    # Step 2: pick and place each block in the certified falsework order.
    print()
    print(f"=== franka build: clamp 29/24 scaled to s={S} m "
          f"(overhang {OVERHANG:.4f}) ===")
    blocks = []
    renders = {}
    aborted = None
    for i, name in enumerate(info.cube_names):
        rep = driver.place_block(i)
        blocks.append(rep)
        print(
            f"  block {i} {name:14s}: "
            f"grasp={'ok' if rep['grasped'] else 'FAIL'} "
            f"finger={rep['peak_finger_force_N']:6.1f} N "
            f"err={rep['placement_error_mm']:6.2f} mm "
            f"disturb={rep['struct_disturb_rel']:.5f} L "
            f"clearance={rep['clearance_contacts'] or 'none'}"
        )
        renders[f"place_{i}_{name}"] = driver.snapshot(
            os.path.join(args.out, f"franka_place_{i}_{name}.png"))
        if not rep["grasped"]:
            aborted = {"block": name, "phase": "grasp-check",
                       "track_m": rep["grasp_track_m"],
                       "rise_m": rep["grasp_rise_m"]}
            print(f"  ABORT: grasp failed on {name} "
                  f"(track {rep['grasp_track_m'] * 1000:.1f} mm, "
                  f"rise {rep['grasp_rise_m'] * 1000:.1f} mm)")
            break

    # Step 3: retract the props and settle, if the build completed.
    retract = None
    finals = None
    if aborted is None:
        # Park the arm away from the structure before retraction.
        driver.move_to([0.30, -0.25, 0.35], SPEED_TRANSIT)
        retract = driver.retract_props()
        for st in retract["stages"]:
            print(
                f"  retract {st['prop']:8s} ({st['axis']}): "
                f"load {st['load_before_N']:.3f} N, "
                f"peak {st['peak_during_retract_N']:.3f}, "
                f"after {st['load_after_N']:.3f}"
            )
        print(
            f"  post-retraction: structure {retract['verdict']} "
            f"(disp {retract['disp_rel']:.5f} L, rot {retract['rot_rad']:.4f})"
        )
        finals = driver.final_state()
        for row in finals:
            print(f"  final {row['block']:14s}: "
                  f"err={row['final_error_mm']:6.2f} mm "
                  f"rot={row['final_rot_rad']:.4f}")
    renders["final"] = driver.snapshot(
        os.path.join(args.out, "franka_final.png"))

    gif, mp4 = write_movie(driver.frames, args.out)
    driver.close()
    print(f"  movie: {gif}")
    print(f"  mp4:   {mp4}")

    success = bool(aborted is None and retract is not None
                   and retract["stands"])
    print(f"  franka falsework build "
          f"{'SUCCEEDS' if success else 'FAILS'} for clamp 29/24")

    out = {
        "meta": {
            "design": "clamp_29_24",
            "overhang": OVERHANG,
            "mu": DESIGN_MU,
            "scale_m": S,
            "cube_mass_kg": 2000.0 * S ** 3,
            "timestep": args.timestep,
            "speed_transit": SPEED_TRANSIT,
            "speed_descend": SPEED_DESCEND,
            "grasp": "top grasp, y-pinch, grip servo stiffened to 1000 N/m "
                     "in memory (see franka_scene.compose_scene)",
            "note": "Franka Panda executes the certified clamp 29/24 "
                    "falsework build at 1/20 scale; margins are "
                    "scale-invariant and re-certified above.",
        },
        "recertification": cert_table,
        "blocks": blocks,
        "gripper_clearance": clearance,
        "retraction": retract,
        "final_state": finals,
        "aborted": aborted,
        "success": success,
        "renders": renders,
        "movie": {"gif": gif, "mp4": mp4},
    }
    path = os.path.join(args.out, "franka_build.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"wrote {path}")
    return out


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
