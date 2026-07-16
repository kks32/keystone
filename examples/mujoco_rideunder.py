"""Ride-under insertion of the full-height reacher (prop-free, one-handed push).

The counterweighted clamp's reacher must enter a slot one block high under a
clamping bridge. A rigid straight slide jams at zero clearance
(examples/mujoco_insert.py). Hold-and-shim splits the reacher into a short block
plus a shim and needs two hands (examples/mujoco_shim.py). Ride-under keeps the
reacher whole and uses it as its own tool: push it in nose-first with a slight
nose-down tilt so its leading top corner slips under the bridge lip; the bridge
pivots up and rides over the reacher as it slides through, then settles down on
top. One capped push replaces the two-handed hold-and-shim.

Geometry of the trick. A unit reacher tilted nose-down by theta, riding on its
leading bottom corner at the base top, carries its leading top corner a height
S*(1 - cos theta) below the bridge lip. Any positive tilt lifts that corner
clear; a flat reacher (theta = 0) has it exactly at the lip, the jam. The whole
tilted block spans S*(cos theta + sin theta) > S vertically, so as it advances
its top face acts as a ramp that lifts the bridge by the excess span. Leveling
the reacher once its corner is under the lip lays the bridge back down.

Scale. Run at the Franka table-top scale, cube side 0.05 m at density 2000, so a
block is 0.25 kg and push forces are robot-sized (a few newtons). keystone
verdicts and dimensionless margins are scale-invariant
(tests/property/test_invariants.py), so the static story is scale-free; MuJoCo
contact dynamics are not (fixed solref time constant), which is itself a finding
here (the seated 29/24 knife-edge stands at unit scale but creep-topples at this
scale).

Protocol per (design, tilt, cap):
1. Place the certified pre-reacher stack (pedestal, base, counterweight, bridge)
   at exact poses. It stands prop-free.
2. Start the tilted reacher clear of the bridge on the base top, steadied by a
   capped impedance driver.
3. Drive left. Stay tilted until the leading top corner passes under the lip,
   then flatten over the remaining travel so the bridge settles back.
4. Release and settle. Verdict under stiff contacts.

Instrumented: leading-corner catch versus slip-under, bridge pivot over time,
push force profile and peak, bridge return-to-pose error, ballast disturbance,
reacher seated error, and a force decomposition (measured reacher-bridge lift and
reacher-base drag against the analytic edge-lift force for a 0.25 kg bridge).
Intermediate states are re-certified arm-free through the host pipeline (P4
margin versus push force versus time). An HD movie frames the slot so the pivot
is visible. Run:

    python examples/mujoco_rideunder.py --out out/mujoco
"""

import argparse
import json
import os
import shutil
import subprocess

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
    settle_test,
    to_mjcf,
)

# Franka table-top scale. Cube side 0.05 m, density 2000 -> 0.25 kg per block.
S = 0.05
DX = 1.0 / 24.0
DEPTH = S
DENSITY = 2000.0
G = 9.81
MU = 0.7
TOL = Tolerances()

# Contacts. The drive is moderately soft so hard corner impacts on light blocks
# stay a bounded penetration transient instead of a numerical explosion (the
# shim demo's policy). The verdict is a stiff settle.
DRIVE_SOLREF = (0.003, 1.0)
STIFF_SOLREF = (0.002, 1.0)

# Impedance gains for the 0.25 kg reacher: (kp, kd, kp_rot, kd_rot). Overdamped
# so the push is quasi-static and the cap governs the force.
GAINS = (1000.0, 40.0, 1.2, 0.04)

# Pre-reacher stack, shared by every clamp design: base, counterweight, bridge.
# cells are (layer, grid_index_j); center x is j * DX.
PRE_CELLS = [(0, -2), (1, -14), (2, -4)]
PRE_NAMES = ["base", "counterweight", "bridge"]

# Clamp designs differ only in the reacher grid index (how deep it threads under
# the bridge). Bridge right edge sits at 8/24; reacher left edge at seat is
# (j - 12)/24, so thread = 8/24 - (j - 12)/24 = (20 - j)/24.
DESIGNS = {
    "clamp_29_24": dict(reacher_j=17, overhang=29.0 / 24.0),  # thread 3/24
    "clamp_31_24": dict(reacher_j=19, overhang=31.0 / 24.0),  # thread 1/24
    "clamp_26_24": dict(reacher_j=14, overhang=26.0 / 24.0),  # thread 6/24
}

BRIDGE_RIGHT = 8.0 / 24.0 * S


# --------------------------------------------------------------------------
# Geometry.
# --------------------------------------------------------------------------


def pedestal():
    return box_2d(6.0 * S, 1.0 * S, -3.0 * S, 0.5 * S, density=DENSITY, depth=DEPTH)


def cube(layer, j):
    return box_2d(1.0 * S, 1.0 * S, j * DX * S, (1.5 + layer) * S,
                  density=DENSITY, depth=DEPTH)


def pre_reacher_boxes():
    return [pedestal()] + [cube(l, j) for (l, j) in PRE_CELLS]


def tilt_quat(deg):
    """Nose (leading, -x) down: a negative rotation about world +y."""
    a = -np.radians(deg) / 2.0
    return np.array([np.cos(a), 0.0, np.sin(a), 0.0])


def y_angle(q):
    """Signed rotation about +y encoded in a (w, 0, y, 0) quaternion, radians."""
    return 2.0 * float(np.arctan2(q[2], q[0]))


def box2d_from_box(b):
    """Rebuild an oriented 2D box from a world-pose Box (extract the y tilt)."""
    ay = y_angle(b.quat)
    w = 2.0 * float(b.half_extents[0])
    h = 2.0 * float(b.half_extents[2])
    dep = 2.0 * float(b.half_extents[1])
    return box_2d(w, h, float(b.position[0]), float(b.position[2]),
                  angle_y=ay, density=b.density, depth=dep)


def offplane(q):
    """Out-of-plane quaternion content (should be ~0 for a planar maneuver)."""
    return float(np.hypot(q[1], q[3]))


# --------------------------------------------------------------------------
# Host-pipeline certification (scale-free).
# --------------------------------------------------------------------------


def certify_p4(boxes):
    """P0 verdict and P4 margin, or an error tag if the state is not certifiable
    (a dynamic overlap trips the interpenetration guard)."""
    try:
        a = build_assembly(boxes, mu=MU, tol=TOL, dim=2)
        s = assemble(a, TOL, cone="linear2d")
        return solve_p0(s, TOL).status, float(solve_p4(s, TOL).margin), None
    except Exception as e:  # noqa: BLE001
        return None, None, f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------
# The maneuver.
# --------------------------------------------------------------------------


def _wrench(m, d, bid, tgt, quat, maxf, maxt):
    import mujoco  # noqa: F401

    dof = int(m.body_dofadr[bid])
    R = d.xmat[bid].reshape(3, 3)
    lv = d.qvel[dof:dof + 3].copy()
    av = R @ d.qvel[dof + 3:dof + 6]
    return capped_impedance_wrench(
        d.xpos[bid], d.xquat[bid], tgt, quat, lv, av,
        kp=GAINS[0], kd=GAINS[1], kp_rot=GAINS[2], kd_rot=GAINS[3],
        max_push=maxf, max_torque=maxt)


def ride_under(reacher_j, tilt_deg, cap_x, *, drive_sr=DRIVE_SOLREF,
               slide_off=0.30 * S, z_press=0.0004, dwell=0.0,
               t_start=0.6, t_push=2.5, t_hold=0.5, t_settle=1.5,
               ts=5e-4, on_step=None):
    """Push the full-height reacher in under the bridge and settle it.

    Returns a report dict with the instrumentation, the settled Box list, the
    time series, and pose snapshots for arm-free re-certification. on_step, if
    given, is called (model, data, nstep, phase) every step for frame capture.
    """
    import mujoco

    pre = pre_reacher_boxes()
    r_seat = cube(1, reacher_j)
    P = len(pre)
    x_seat = float(r_seat.position[0])
    z_seat = float(r_seat.position[2])
    thr = np.radians(tilt_deg)
    z_ride = 2.0 * S + 0.5 * S * (np.cos(thr) + np.sin(thr)) + 0.0005
    # Stay tilted until the leading top corner clears the lip, then flatten.
    x_engage = BRIDGE_RIGHT + 0.5 * S * (np.cos(thr) + np.sin(thr))
    x_start = x_seat + slide_off
    q_tilt = tilt_quat(tilt_deg)
    r0 = Box(r_seat.half_extents, np.array([x_start, 0.0, z_ride]),
             q_tilt, r_seat.density)
    boxes = pre + [r0]
    L = assembly_diagonal(pre + [r_seat])

    m = mujoco.MjModel.from_xml_string(
        to_mjcf(boxes, MU, timestep=ts, all_pairs=True, solref=drive_sr))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    rid = int(m.body(f"block{P}").id)
    base_id = int(m.body("block1").id)
    cw_id = int(m.body("block2").id)
    bid = int(m.body("block3").id)
    rgeom = int(m.geom(f"geom{P}").id)
    bgeom = int(m.geom("geom3").id)
    basegeom = int(m.geom("geom1").id)

    bpos0 = d.xpos[bid].copy()
    bang0 = y_angle(d.xquat[bid])
    pos0 = {b: d.xpos[b].copy() for b in (base_id, cw_id)}
    q0 = {b: d.xquat[b].copy() for b in (base_id, cw_id)}
    rw = float(m.body_mass[rid]) * G
    bw = float(m.body_mass[bid]) * G
    rff = np.array([0.0, 0.0, rw])
    maxf = cap_x * rw
    maxt = maxf * 0.5 * S

    def contact_force(ga, gb):
        buf = np.zeros(6)
        tot = np.zeros(3)
        for c in range(d.ncon):
            con = d.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            if (g1 == ga and g2 == gb) or (g2 == ga and g1 == gb):
                mujoco.mj_contactForce(m, d, c, buf)
                fr = con.frame.reshape(3, 3)
                tot += fr.T @ buf[:3]
        return tot

    peak_push = peak_lift = peak_drag = 0.0
    max_pivot = 0.0
    ballast_disp = ballast_rot = 0.0
    series = []
    nstep = [0]

    def record(push, phase):
        nonlocal peak_push, peak_lift, peak_drag, max_pivot, ballast_disp, ballast_rot
        lift = contact_force(rgeom, bgeom)
        drag = contact_force(rgeom, basegeom)
        pivot = -(y_angle(d.xquat[bid]) - bang0)  # +ve = bridge right side up
        peak_push = max(peak_push, push)
        peak_lift = max(peak_lift, abs(lift[2]))
        peak_drag = max(peak_drag, abs(drag[0]))
        max_pivot = max(max_pivot, pivot)
        bd = 0.0
        br = 0.0
        for b in (base_id, cw_id):
            bd = max(bd, float(np.linalg.norm(d.xpos[b] - pos0[b])))
            dq = min(1.0, abs(float(np.dot(q0[b], d.xquat[b]))))
            br = max(br, 2.0 * float(np.arccos(dq)))
        ballast_disp = max(ballast_disp, bd)
        ballast_rot = max(ballast_rot, br)
        if nstep[0] % 20 == 0:
            series.append(dict(
                t=float(d.time), push=push,
                reacher_x=float(d.xpos[rid][0]),
                reacher_tilt_deg=np.degrees(y_angle(d.xquat[rid])),
                bridge_pivot_deg=np.degrees(pivot),
                bridge_dx_mm=float(d.xpos[bid][0] - bpos0[0]) * 1000.0,
                bridge_dz_mm=float(d.xpos[bid][2] - bpos0[2]) * 1000.0,
                lift_N=float(lift[2]), drag_N=float(drag[0]),
                ballast_disp_mm=bd * 1000.0, ballast_rot_deg=np.degrees(br),
                phase=phase))

    def snapshot():
        return [pedestal()] + [
            Box(boxes[1 + i].half_extents, np.asarray(d.xpos[b]).copy(),
                np.asarray(d.xquat[b]).copy(), boxes[1 + i].density)
            for i, b in enumerate((base_id, cw_id, bid))
        ] + [Box(r_seat.half_extents, np.asarray(d.xpos[rid]).copy(),
                 np.asarray(d.xquat[rid]).copy(), r_seat.density)]

    # Split the push time by distance: approach (start -> engage, tilted) and
    # thread (engage -> seat, flattening). z_low keeps the reacher pressed on the
    # base so its top corner stays just under the lip.
    z_low = z_ride - z_press
    d_app = max(1e-9, x_start - x_engage)
    d_thr = max(1e-9, x_engage - x_seat)
    t_app = t_push * d_app / (d_app + d_thr)
    t_thr = t_push - t_app
    snaps = []
    state = {"push": 0.0}
    ns = lambda t: max(1, int(round(t / ts)))  # noqa: E731

    def drive_to(n, xf, zf, degf, phase, drive=True):
        for k in range(max(1, n)):
            a = (k + 1) / max(1, n)
            if drive:
                x, z, deg = xf(a), zf(a), degf(a)
                fr, tr = _wrench(m, d, rid, np.array([x, 0.0, z]), tilt_quat(deg),
                                 maxf, maxt)
                d.xfrc_applied[rid, :3] = fr + rff
                d.xfrc_applied[rid, 3:] = tr
                state["push"] = float(np.linalg.norm(fr))
            else:
                state["push"] = 0.0
            mujoco.mj_step(m, d)
            nstep[0] += 1
            record(state["push"], phase)
            if on_step is not None:
                on_step(m, d, nstep[0], phase)

    def add_snap(name):
        snaps.append(dict(name=name, t=float(d.time), push=state["push"],
                          poses=snapshot()))

    # Phase: settle the tilted reacher at the start pose, clear of the bridge.
    drive_to(ns(t_start), lambda a: x_start, lambda a: z_ride,
             lambda a: tilt_deg, "start")
    add_snap("start")
    # Phase: approach the lip, tilted.
    drive_to(ns(t_app), lambda a: x_start + a * (x_engage - x_start),
             lambda a: z_low, lambda a: tilt_deg, "approach")
    add_snap("engage")
    # Phase: dwell tilted under the lip. Lifts the bridge; makes the pivot visible.
    if dwell > 0.0:
        drive_to(ns(dwell), lambda a: x_engage, lambda a: z_low,
                 lambda a: tilt_deg, "dwell")
    add_snap("under")
    # Phase: thread under and flatten from engagement to seat.
    drive_to(ns(t_thr), lambda a: x_engage + a * (x_seat - x_engage),
             lambda a: z_low + a * (z_seat - z_low),
             lambda a: (1.0 - a) * tilt_deg, "thread")
    # Phase: hold at the seat, flat.
    drive_to(ns(t_hold), lambda a: x_seat, lambda a: z_seat, lambda a: 0.0, "hold")
    reach_err = float(np.linalg.norm(d.xpos[rid] - np.array([x_seat, 0.0, z_seat])))
    reach_tilt = np.degrees(y_angle(d.xquat[rid]))
    add_snap("seated")
    # Phase: release and settle under the drive contacts.
    d.xfrc_applied[:] = 0.0
    drive_to(ns(t_settle), None, None, None, "settle", drive=False)
    add_snap("released")

    settled = snapshot()
    bridge_dx = float(d.xpos[bid][0] - bpos0[0])
    bridge_dz = float(d.xpos[bid][2] - bpos0[2])
    bridge_return = float(np.linalg.norm(d.xpos[bid] - bpos0))
    bridge_pivot_final = np.degrees(y_angle(d.xquat[bid]) - bang0)
    # Clamp overlap: bridge right edge minus reacher left edge (both near x=0.015).
    overlap = (float(d.xpos[bid][0]) + 0.5 * S) - (float(d.xpos[rid][0]) - 0.5 * S)

    v2 = settle_test(settled, MU, duration=2.0, solref=STIFF_SOLREF)
    v6 = settle_test(settled, MU, duration=6.0, solref=STIFF_SOLREF)

    seated = bool(reach_err < 1.5e-3 and abs(reach_tilt) < 1.5)
    clamp_intact = bool(overlap > 0.5e-3 and bridge_return < 6.0e-3
                        and np.degrees(ballast_rot) < 5.0)
    if seated and clamp_intact:
        outcome = "slip_under"
    elif np.degrees(ballast_rot) >= 5.0 or bridge_return >= 6.0e-3:
        outcome = "topple"        # over-lift threw the bridge or ballast
    else:
        outcome = "catch"         # reacher stalled or clamp not formed

    report = dict(
        reacher_j=reacher_j, tilt_deg=tilt_deg, cap_x=cap_x, drive_sr=list(drive_sr),
        reach_err_mm=reach_err * 1000.0, reach_tilt_deg=reach_tilt,
        peak_push_N=peak_push, peak_push_rw=peak_push / rw,
        peak_lift_N=peak_lift, peak_drag_N=peak_drag,
        reacher_weight_N=rw, bridge_weight_N=bw,
        max_pivot_deg=np.degrees(max_pivot),
        bridge_pivot_final_deg=bridge_pivot_final,
        bridge_return_mm=bridge_return * 1000.0,
        bridge_dx_mm=bridge_dx * 1000.0, bridge_dz_mm=bridge_dz * 1000.0,
        overlap_mm=overlap * 1000.0,
        ballast_disp_mm=ballast_disp * 1000.0,
        ballast_rot_deg=np.degrees(ballast_rot),
        seated=seated, clamp_intact=clamp_intact, outcome=outcome,
        verdict_2s=v2["verdict"], rot_2s=v2["max_rot"],
        verdict_6s=v6["verdict"], rot_6s=v6["max_rot"],
        L=L,
    )
    return report, settled, series, snaps


def downsample(series, every=4):
    """Thin the time series for JSON persistence, keeping phase transitions."""
    if not series:
        return []
    out = [series[0]]
    for i in range(1, len(series)):
        if i % every == 0 or series[i]["phase"] != series[i - 1]["phase"]:
            out.append(series[i])
    if out[-1] is not series[-1]:
        out.append(series[-1])
    return out


# --------------------------------------------------------------------------
# Force decomposition against the analytic edge-lift reference.
# --------------------------------------------------------------------------


def force_decomposition(report):
    """Split the peak push into an edge-lift share and a friction-drag share and
    compare the measured reacher-bridge lift to the analytic edge-lift force.

    The bridge pivots about its far (left) support edge at x = -16/24. Its center
    of mass sits at x = -4/24, lever 12/24 from the pivot; the reacher lifts near
    the bridge right edge x = 8/24, lever 24/24. Torque balance gives the vertical
    lift force W_bridge * (12/24) / (24/24) = 0.5 * W_bridge."""
    wb = report["bridge_weight_N"]
    edge_lift_ref = 0.5 * wb
    theta = np.radians(report["tilt_deg"])
    # Horizontal push to raise the edge-lift load through a ramp of angle theta,
    # frictionless: F_h = F_v * tan(theta).
    lift_horizontal = edge_lift_ref * np.tan(theta) if theta > 0 else 0.0
    peak = report["peak_push_N"]
    drag_share = max(0.0, peak - lift_horizontal)
    return dict(
        analytic_edge_lift_N=edge_lift_ref,
        measured_peak_lift_N=report["peak_lift_N"],
        lift_over_ref=report["peak_lift_N"] / edge_lift_ref if edge_lift_ref else None,
        peak_push_N=peak,
        edge_lift_horizontal_share_N=lift_horizontal,
        friction_drag_share_N=drag_share,
        measured_reacher_base_drag_N=report["peak_drag_N"],
    )


# --------------------------------------------------------------------------
# Intermediate arm-free certification.
# --------------------------------------------------------------------------


def certify_snapshots(snaps):
    """Re-certify each maneuver snapshot arm-free through the host pipeline. The
    dangling reacher is unsupported (needs the arm); a tilted transient overlaps
    the bridge and trips the interpenetration guard; the seated state is
    feasible. Reports P4 margin (external help needed) versus push and time."""
    rows = []
    for s in snaps:
        boxes = [box2d_from_box(b) for b in s["poses"]]
        offp = max(offplane(b.quat) for b in s["poses"])
        status, margin, err = certify_p4(boxes)
        rows.append(dict(
            snapshot=s["name"], t=round(s["t"], 4),
            push_N=round(s["push"], 4),
            status=status, margin=margin, error=err,
            max_offplane=offp))
    return rows


# --------------------------------------------------------------------------
# Movie.
# --------------------------------------------------------------------------


def _slot_camera(mujoco):
    """Free camera framed on the clamp so the reacher slides in under the bridge
    and the bridge lift reads. Side view of the xz plane (azimuth 90)."""
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(mujoco.MjModel.from_xml_string(
        to_mjcf([cube(1, 17)], MU)), cam)
    cam.lookat[:] = [0.05 * S, 0.0, 2.85 * S]
    cam.distance = 6.0 * S
    cam.azimuth = 90.0
    cam.elevation = -4.0
    return cam


def _brighten(img, gain=1.5):
    return np.clip(img.astype(np.float32) * gain, 0.0, 255.0).astype(np.uint8)


def build_movie(reacher_j, tilt_deg, cap_x, out_dir, *, dwell, height=720,
                width=1280, stride=60, fps=25, tag=""):
    """Render the ride-under to an HD movie (GIF plus MP4 when ffmpeg exists).
    Frames are captured live during the drive with a slot-framing camera."""
    try:
        import mujoco
        from PIL import Image
    except Exception as e:  # noqa: BLE001
        return {"gif": f"movie skipped ({type(e).__name__}: {e})", "mp4": None}

    frames = []
    renderer = {"r": None}
    cam = _slot_camera(mujoco)

    def on_step(model, data, nstep, phase):
        if renderer["r"] is None:
            renderer["r"] = mujoco.Renderer(model, height=height, width=width)
        if nstep % stride == 0:
            renderer["r"].update_scene(data, camera=cam)
            frames.append(_brighten(renderer["r"].render().copy()))

    ride_under(reacher_j, tilt_deg, cap_x, dwell=dwell, on_step=on_step)
    if renderer["r"] is not None:
        renderer["r"].close()
    if not frames:
        return {"gif": "movie skipped (no frames)", "mp4": None}

    imgs = [Image.fromarray(f) for f in frames]
    gif = os.path.join(out_dir, f"rideunder{tag}.gif")
    imgs[0].save(gif, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)

    ffmpeg = shutil.which("ffmpeg") or (
        "/opt/homebrew/bin/ffmpeg" if os.path.exists("/opt/homebrew/bin/ffmpeg")
        else None)
    if ffmpeg is None:
        return {"gif": gif, "mp4": "mp4 skipped (ffmpeg not found)"}
    tmp = os.path.join(out_dir, "_ru_frames")
    os.makedirs(tmp, exist_ok=True)
    for k, im in enumerate(imgs):
        im.save(os.path.join(tmp, f"f{k:05d}.png"))
    mp4 = os.path.join(out_dir, f"rideunder{tag}.mp4")
    proc = subprocess.run(
        [ffmpeg, "-y", "-framerate", str(fps), "-i",
         os.path.join(tmp, "f%05d.png"), "-pix_fmt", "yuv420p", mp4],
        capture_output=True)
    shutil.rmtree(tmp, ignore_errors=True)
    if proc.returncode != 0:
        return {"gif": gif, "mp4": f"mp4 failed (rc={proc.returncode})"}
    return {"gif": gif, "mp4": mp4}


def _encode_frames(frames, out_dir, name, fps=25, gif=True):
    """MP4 (when ffmpeg exists) and optionally a GIF from a captured frame list.
    Long HD sequences skip the GIF (it balloons); the MP4 is the HD deliverable."""
    try:
        from PIL import Image
    except Exception:  # noqa: BLE001
        return {"gif": "PIL missing", "mp4": None}
    if not frames:
        return {"gif": "no frames", "mp4": None}
    imgs = [Image.fromarray(f) for f in frames]
    gif_path = None
    if gif:
        gif_path = os.path.join(out_dir, f"{name}.gif")
        imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / fps), loop=0)
    ffmpeg = shutil.which("ffmpeg") or (
        "/opt/homebrew/bin/ffmpeg" if os.path.exists("/opt/homebrew/bin/ffmpeg")
        else None)
    if ffmpeg is None:
        return {"gif": gif_path, "mp4": "ffmpeg not found"}
    tmp = os.path.join(out_dir, f"_{name}_frames")
    os.makedirs(tmp, exist_ok=True)
    for k, im in enumerate(imgs):
        im.save(os.path.join(tmp, f"f{k:05d}.png"))
    mp4 = os.path.join(out_dir, f"{name}.mp4")
    proc = subprocess.run(
        [ffmpeg, "-y", "-framerate", str(fps), "-i",
         os.path.join(tmp, "f%05d.png"), "-pix_fmt", "yuv420p", mp4],
        capture_output=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return {"gif": gif_path, "mp4": mp4 if proc.returncode == 0 else "mp4 failed"}


# --------------------------------------------------------------------------
# Phase 2: Franka execution (place-then-push with a real arm).
# --------------------------------------------------------------------------


def phase2_clearance():
    """Geometric clearance of the finger grasp against the bridge, computed from
    the design geometry at the Franka scale. The reacher is gripped near its
    trailing (+x) end; the bridge overlaps only the reacher's leading (-x) tail,
    so the fingers sit well clear of the bridge and its cantilever hangs over
    empty space right of the base. Returns per-design clearances in mm."""
    bridge_right = 8.0 / 24.0  # grid units
    base_right = 10.0 / 24.0
    rows = {}
    for name, spec in DESIGNS.items():
        r_right = spec["reacher_j"] / 24.0 + 0.5           # reacher trailing edge
        r_left = spec["reacher_j"] / 24.0 - 0.5            # reacher leading edge
        rows[name] = dict(
            trailing_to_bridge_mm=(r_right - bridge_right) * S * 1000.0,
            trailing_past_base_mm=(r_right - base_right) * S * 1000.0,
            leading_under_bridge_mm=(bridge_right - r_left) * S * 1000.0,
        )
    return rows


def _set_free(model, data, body, pos, quat):
    jadr = model.jnt_qposadr[model.body(body).jntadr[0]]
    data.qpos[jadr:jadr + 3] = pos
    data.qpos[jadr + 3:jadr + 7] = quat


def _tilt_grasp_quat(deg):
    """Top-grasp quaternion (tool z down) pitched nose-down by deg about world y."""
    a = -np.radians(deg) / 2.0
    yq = np.array([np.cos(a), 0.0, np.sin(a), 0.0])
    b = np.array([0.0, 0.0, 1.0, 0.0])
    return np.array([yq[0] * b[0] - yq[2] * b[2], yq[0] * b[1] + yq[2] * b[3],
                     yq[0] * b[2] + yq[2] * b[0], yq[0] * b[3] - yq[2] * b[1]])


def phase2_execution(reacher_j, tilt_deg, out_dir, *, ts=5e-4, movie=True):
    """Attempt the ride-under with the menagerie Franka: pick the reacher from
    staging, carry it in clear of the structure, and push it under the bridge
    with a wrist pitch, then release and settle. Reuses franka_scene composition
    and IK; the falsework props are retracted (the pre-stack stands prop-free).

    Instruments the pad-bridge clearance and the bridge and ballast motion, and
    returns the outcome. The pre-stack is teleported to its certified poses (this
    is the reacher-insertion step; the pre-stack build is the falsework demo)."""
    import mujoco
    from keystone.interop.franka_scene import (
        BASE_OFFSET, DX_GRID, GRIPPER_CLOSE, GRIPPER_OPEN, compose_scene,
        dls_ik, reset_home,
    )

    spec, info = compose_scene(timestep=ts)
    model = spec.compile()
    d = mujoco.MjData(model)
    scr = mujoco.MjData(model)
    reset_home(model, d)

    tgt = info.target_world
    _set_free(model, d, "cube0", tgt[0], [1, 0, 0, 0])   # base
    _set_free(model, d, "cube2", tgt[2], [1, 0, 0, 0])   # counterweight
    _set_free(model, d, "cube3", tgt[3], [1, 0, 0, 0])   # bridge
    for p in info.props:
        d.ctrl[model.actuator(p["act"]).id] = p["retract_disp"]
    mujoco.mj_forward(model, d)

    sid = model.site(info.tcp_site).id
    grip_aid = model.actuator(info.gripper_actuator).id
    rid = model.body("cube1").id
    bid = model.body("cube3").id
    base_bid = model.body("cube0").id
    cw_bid = model.body("cube2").id
    pad_g = [model.geom(g).id for g in info.finger_pads]
    bgeom = model.geom("cube3_geom").id
    frames = []
    cam = None
    renderer = {"r": None}
    if movie:
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.lookat[:] = [0.44, -0.02, 0.12]
        cam.distance = 0.62
        cam.azimuth = 138.0
        cam.elevation = -14.0
    metrics = {"pad_bridge": 0.0, "nstep": 0}

    def step():
        mujoco.mj_step(model, d)
        metrics["nstep"] += 1
        buf = np.zeros(6)
        for c in range(d.ncon):
            con = d.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            if ((g1 in pad_g and g2 == bgeom) or (g2 in pad_g and g1 == bgeom)):
                mujoco.mj_contactForce(model, d, c, buf)
                metrics["pad_bridge"] = max(metrics["pad_bridge"],
                                            float(np.linalg.norm(buf[:3])))
        if movie and metrics["nstep"] % 200 == 0:
            if renderer["r"] is None:
                renderer["r"] = mujoco.Renderer(model, height=720, width=1280)
            renderer["r"].update_scene(d, camera=cam)
            frames.append(_brighten(renderer["r"].render().copy(), gain=1.4))

    def tcp():
        return d.site_xpos[sid].copy()

    def move_to(pos, quat, speed=0.05, min_steps=100):
        q0 = np.array(d.ctrl[:7])
        scr.qpos[:] = d.qpos
        qg, _, _ = dls_ik(model, scr, pos, quat)
        n = max(min_steps, int(round(float(np.linalg.norm(np.asarray(pos) - tcp()))
                                     / speed / ts)))
        for t in range(n):
            a = (t + 1) / n
            d.ctrl[:7] = q0 + a * (qg - q0)
            step()

    def grip(c, t):
        d.ctrl[grip_aid] = c
        for _ in range(max(1, int(round(t / ts)))):
            step()

    thr = np.radians(tilt_deg)
    z_ride = BASE_OFFSET[2] + 2.0 * S + 0.5 * S * (np.cos(thr) + np.sin(thr)) + 0.0005
    z_seat = tgt[1][2]
    x_seat = reacher_j * DX_GRID * S + BASE_OFFSET[0]
    bridge_right = 8.0 / 24.0 * S + BASE_OFFSET[0]
    x_engage = bridge_right + 0.5 * S * (np.cos(thr) + np.sin(thr))
    x_start = x_seat + 0.60 * S
    grasp_dx = 0.20 * S
    stage = info.staging_world[1]

    def grasp_world(cx, cz, deg):
        a = -np.radians(deg)
        R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        return np.array([cx, 0.0, cz]) + R @ np.array([grasp_dx, 0.0, 0.0])

    # Pick from staging (offset toward the trailing end).
    gp = stage + np.array([grasp_dx, 0.0, 0.0])
    move_to(gp + [0, 0, 0.12], info.grasp_quat, speed=0.1)
    grip(GRIPPER_OPEN, 0.2)
    move_to(gp, info.grasp_quat, speed=0.03)
    grip(GRIPPER_CLOSE, 0.6)
    move_to(gp + [0, 0, 0.10], info.grasp_quat, speed=0.03)
    grasped = bool(d.xpos[rid][2] - stage[2] > 0.03)

    # Carry clear: straight up above the structure, across high, then descend on
    # the open (+x) side, right of the bridge.
    move_to([stage[0], stage[1], 0.34], info.grasp_quat, speed=0.1)
    move_to([x_start, 0.0, 0.34], _tilt_grasp_quat(0.0), speed=0.1)
    move_to(grasp_world(x_start, z_ride, tilt_deg), _tilt_grasp_quat(tilt_deg), speed=0.03)
    bridge_after_carry = float(np.linalg.norm(d.xpos[bid] - tgt[3]))

    # Ride-under push: left with a wrist pitch, flattening from engage to seat.
    N = 40
    for k in range(N):
        a = (k + 1) / N
        x = x_start + a * (x_seat - x_start)
        f = 0.0 if x >= x_engage else min(1.0, (x_engage - x) / max(1e-9, x_engage - x_seat))
        deg = (1.0 - f) * tilt_deg
        z = z_ride + f * (z_seat - z_ride)
        move_to(grasp_world(x, z, deg), _tilt_grasp_quat(deg), speed=0.02, min_steps=50)
    for _ in range(int(round(0.4 / ts))):
        step()
    reach_err = float(np.linalg.norm(d.xpos[rid] - np.array([x_seat, 0.0, z_seat])))

    grip(GRIPPER_OPEN, 0.5)
    move_to([x_seat + 0.06, 0.0, z_seat + 0.12], _tilt_grasp_quat(0.0), speed=0.05)
    for _ in range(int(round(1.0 / ts))):
        step()

    if renderer["r"] is not None:
        renderer["r"].close()
    bridge_return = float(np.linalg.norm(d.xpos[bid] - tgt[3]))
    ballast_drop = max(float(tgt[0][2] - d.xpos[base_bid][2]),
                       float(tgt[2][2] - d.xpos[cw_bid][2]))
    mv = (_encode_frames(frames, out_dir, f"rideunder_franka_{reacher_j}", gif=False)
          if movie else None)
    return dict(
        reacher_j=reacher_j, tilt_deg=tilt_deg, grasped=grasped,
        bridge_after_carry_mm=bridge_after_carry * 1000.0,
        reach_err_mm=reach_err * 1000.0,
        bridge_return_mm=bridge_return * 1000.0,
        ballast_drop_mm=ballast_drop * 1000.0,
        pad_bridge_contact_N=metrics["pad_bridge"],
        movie=mv,
    )


# --------------------------------------------------------------------------
# Context: static certification and settle of the standing states.
# --------------------------------------------------------------------------


def standing_context():
    """Certify the pre-reacher stack and each seated clamp, and settle them at
    the Franka scale (stiff) and at unit scale, so the scale non-invariance of
    the MuJoCo settle is visible next to the scale-free static verdict."""
    pre = pre_reacher_boxes()
    st, mg, _ = certify_p4(pre)
    v = settle_test(pre, MU, duration=3.0, solref=STIFF_SOLREF)
    rows = {"pre_reacher": dict(status=st, margin=mg, verdict=v["verdict"],
                                rot=v["max_rot"])}
    for name, spec in DESIGNS.items():
        seated = pre + [cube(1, spec["reacher_j"])]
        st, mg, _ = certify_p4(seated)
        vf = settle_test(seated, MU, duration=6.0, solref=STIFF_SOLREF)
        # Unit-scale settle of the exact clamp (scale cross-check).
        unit = [box_2d(6.0, 1.0, -3.0, 0.5)] + [
            box_2d(1.0, 1.0, j / 24.0, 1.5 + l) for (l, j) in PRE_CELLS
        ] + [box_2d(1.0, 1.0, spec["reacher_j"] / 24.0, 2.5)]
        vu = settle_test(unit, MU, duration=6.0, solref=STIFF_SOLREF)
        rows[name] = dict(
            overhang=spec["overhang"], status=st, margin=mg,
            franka_verdict=vf["verdict"], franka_rot=vf["max_rot"],
            unit_verdict=vu["verdict"], unit_rot=vu["max_rot"])
    return rows


# --------------------------------------------------------------------------
# Driver.
# --------------------------------------------------------------------------


def run_tilt_sweep(reacher_j, cap_x, tilts=(0.0, 1.0, 2.0, 4.0, 8.0)):
    print(f"=== tilt sweep: reacher_j={reacher_j} cap={cap_x}x ===")
    print(f"  {'tilt':>4s} {'outcome':>10s} {'reach':>7s} {'push(rw)':>9s} "
          f"{'lift':>6s} {'drag':>6s} {'pivot':>6s} {'b_ret':>6s} {'bal_rot':>7s} "
          f"{'2s':>9s} {'6s':>9s}")
    rows = []
    for t in tilts:
        rep, _, _, _ = ride_under(reacher_j, t, cap_x)
        rows.append(rep)
        print(f"  {t:4.0f} {rep['outcome']:>10s} {rep['reach_err_mm']:6.2f}m "
              f"{rep['peak_push_rw']:8.2f}x {rep['peak_lift_N']:6.2f} "
              f"{rep['peak_drag_N']:6.2f} {rep['max_pivot_deg']:5.1f}d "
              f"{rep['bridge_return_mm']:5.1f}m {rep['ballast_rot_deg']:6.2f}d "
              f"{rep['verdict_2s']:>9s} {rep['verdict_6s']:>9s}")
    return rows


def run_cap_sweep(reacher_j, tilt, caps=(0.5, 0.75, 1.0, 2.0, 4.0)):
    print(f"=== force-cap sweep: reacher_j={reacher_j} tilt={tilt} ===")
    print(f"  {'cap(rw)':>7s} {'outcome':>10s} {'reach':>7s} {'push(N)':>8s} "
          f"{'b_ret':>6s}")
    rows = []
    min_cap = None
    for c in caps:
        rep, _, _, _ = ride_under(reacher_j, tilt, c)
        rows.append(rep)
        if rep["outcome"] == "slip_under" and min_cap is None:
            min_cap = c
        print(f"  {c:7.2f} {rep['outcome']:>10s} {rep['reach_err_mm']:6.2f}m "
              f"{rep['peak_push_N']:7.3f}N {rep['bridge_return_mm']:5.1f}m")
    print(f"  minimum cap that slips under: "
          f"{min_cap if min_cap is not None else 'none in swept range'}")
    return rows, min_cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/mujoco")
    ap.add_argument("--movie-stride", type=int, default=60)
    ap.add_argument("--no-phase2", action="store_true",
                    help="skip the Franka execution (phase 2)")
    ap.add_argument("--no-movie", action="store_true", help="skip movie rendering")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("mujoco not installed; install keystone[mujoco] to run this demo.")
        return

    best_tilt = 4.0

    print()
    ctx = standing_context()
    print("=== standing context (statics scale-free, settle scale-dependent) ===")
    for k, r in ctx.items():
        print(f"  {k}: {r}")

    print()
    tilt29 = run_tilt_sweep(DESIGNS["clamp_29_24"]["reacher_j"], 4.0)

    print()
    cap29, min_cap = run_cap_sweep(DESIGNS["clamp_29_24"]["reacher_j"], best_tilt)
    decomp = None
    for rep in cap29:
        if rep["outcome"] == "slip_under":
            decomp = force_decomposition(rep)
            break
    if decomp is None:
        decomp = force_decomposition(cap29[-1])
    print("  force decomposition (best slip-under):")
    for k, v in decomp.items():
        print(f"    {k}: {v}")

    print()
    print("=== per-design outcomes (tilt 4, cap 4) ===")
    designs = {}
    for name, spec in DESIGNS.items():
        rep, settled, series, snaps = ride_under(spec["reacher_j"], best_tilt, 4.0)
        cert = certify_snapshots(snaps)
        # As-built margin: the seated (held, flat) snapshot certifies cleanly; the
        # post-release soft-settle leaves a sub-mm reacher-bridge penetration that
        # trips the interpenetration guard (a soft-contact artifact).
        seated_row = next((r for r in cert if r["snapshot"] == "seated"), None)
        designs[name] = dict(report=rep,
                             asbuilt_seated_margin=(seated_row or {}).get("margin"),
                             asbuilt_seated_status=(seated_row or {}).get("status"),
                             intermediate_cert=cert,
                             series=downsample(series))
        print(f"  {name} (overhang {spec['overhang']:.4f}): {rep['outcome']} "
              f"reach={rep['reach_err_mm']:.2f}mm push={rep['peak_push_N']:.2f}N "
              f"overlap={rep['overlap_mm']:.1f}mm bridge_ret={rep['bridge_return_mm']:.1f}mm "
              f"pivot_max={rep['max_pivot_deg']:.1f}deg "
              f"seated_P4={(seated_row or {}).get('margin')} | "
              f"2s={rep['verdict_2s']} 6s={rep['verdict_6s']}")
        print("    arm-free P4 margin vs push vs time (external help needed):")
        for row in cert:
            print(f"      {row['snapshot']:>9s} t={row['t']:.3f}s push={row['push_N']:.3f}N "
                  f"-> {row['status']} margin={row['margin']} "
                  f"{'(' + row['error'] + ')' if row['error'] else ''}")

    print()
    print("=== HD movie: clamp_26_24 (clean standing ride-under) ===")
    if args.no_movie:
        movie = "skipped (--no-movie)"
    else:
        movie = build_movie(DESIGNS["clamp_26_24"]["reacher_j"], 6.0, 4.0, args.out,
                            dwell=0.4, stride=args.movie_stride, tag="_clamp_26_24")
    print(f"  {movie}")

    print()
    print("=== phase 2: Franka execution ===")
    clr = phase2_clearance()
    print("  finger-grasp clearance (geometric, mm):")
    for name, r in clr.items():
        print(f"    {name}: {r}")
    phase2 = {"clearance": clr, "execution": None}
    if not args.no_phase2:
        try:
            ex = phase2_execution(DESIGNS["clamp_26_24"]["reacher_j"], 6.0, args.out,
                                  movie=not args.no_movie)
            phase2["execution"] = ex
            print(f"  Franka push (clamp_26_24): grasped={ex['grasped']} "
                  f"bridge_after_carry={ex['bridge_after_carry_mm']:.1f}mm "
                  f"reach={ex['reach_err_mm']:.1f}mm "
                  f"pad_bridge_contact={ex['pad_bridge_contact_N']:.3f}N "
                  f"bridge_return={ex['bridge_return_mm']:.1f}mm "
                  f"ballast_drop={ex['ballast_drop_mm']:.1f}mm")
            print(f"  movie: {ex['movie']}")
        except Exception as e:  # noqa: BLE001
            phase2["execution"] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  phase 2 execution skipped ({type(e).__name__}: {e})")

    out = {
        "meta": {
            "scale_m": S, "cube_mass_kg": DENSITY * S ** 3,
            "drive_solref": list(DRIVE_SOLREF), "stiff_solref": list(STIFF_SOLREF),
            "gains": {"kp": GAINS[0], "kd": GAINS[1], "kp_rot": GAINS[2],
                      "kd_rot": GAINS[3]},
            "best_tilt": best_tilt,
            "note": "Ride-under: push the full-height reacher in nose-down, the "
                    "bridge pivots up and rides over, then settles. Drive under "
                    "soft contacts (bounded transient), verdict under stiff "
                    "contacts. Statics scale-free; MuJoCo settle scale-dependent.",
        },
        "standing_context": ctx,
        "tilt_sweep_29_24": tilt29,
        "cap_sweep_29_24": {"rows": cap29, "min_cap": min_cap,
                            "decomposition": decomp},
        "designs": designs,
        "movie": movie,
        "phase2": phase2,
    }
    path = os.path.join(args.out, "mujoco_rideunder.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
