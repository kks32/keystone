"""Hold-and-shim build of the counterweighted reacher designs (no falsework).

The blocker for the counterweighted clamp is the reacher: it must enter a slot
exactly one block high under a clamping bridge, and a rigid straight slide jams
at zero clearance (examples/mujoco_insert.py, Route A). Tilt cannot help: a
tilted unit square spans cos(theta) + sin(theta) > 1 vertically, worse than one
(the PART 1 negative control in mujoco_insert.py records this).

Hold-and-shim splits insertion from statics. Replace the reacher with two blocks:
a SHORT reacher of height 1 - eps and a SHIM plate of thickness eps that fills
the gap above the reacher's tail. The short reacher slides into the one-block
slot with eps clearance while HELD by an impedance driver (a held block need not
stand). A second impedance driver then drives the shim into the gap. Both release
and the structure settles. The final state is all boxes, so keystone certifies it
exactly.

Protocol per (design, eps):
1. STATIC CERTIFICATION FIRST. Certify the split state (short reacher + shim)
   through the host pipeline for both shim footprints (full plate, tail plate).
   Certify the short reacher alone too: it must be infeasible (the eps gap above
   the reacher un-certifies the clamp), which is exactly why the shim is needed.
2. BUILD. Drop the pre-reacher blocks (rigid weld drive, as in mujoco_insert).
   Slide the short reacher in while held. Drive the shim in with its own capped
   impedance driver (descend onto the reacher clear of the bridge, then slide
   horizontally under it). Release the shim, then the reacher. Contacts are soft
   during the drive so the eps-into-eps shim seat is a bounded penetration
   transient, the honest elasticity story. Settle stiff for the verdict.
3. Sweep eps and report the minimum eps at which the full build succeeds.

Finding recorded in docs/KNOWN_LIMITS.md: the shim seam is a slip and compliance
path that lowers the clamp's dynamic margin, so the buildable-and-standing
overhang backs off below the monolithic falsework value. Run:

    python examples/mujoco_shim.py --out out/mujoco
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mujoco_insert import (  # noqa: E402
    DROP,
    INSERT_SOLREF,
    certify,
    cube,
    insert_step,
    pedestal6,
)

from keystone import Box  # noqa: E402
from keystone.interop.mujoco_io import (  # noqa: E402
    assembly_diagonal,
    capped_impedance_wrench,
    settle_test,
    split_reacher,
    to_mjcf,
)

# Gravity for the force scales (matches MuJoCo default (0, 0, -9.81)).
G = 9.81
DENSITY = 2000.0
CUBE_WEIGHT = DENSITY * 1.0 * G  # unit block weight, N

# Build designs. cells are (layer, grid_index_j); center x is j * dx. The reacher
# is the last cell and gets split. The clamp backs off along its grid; the split
# needs more back-off than the monolith to stand (KNOWN_LIMITS.md).
DESIGNS = {
    # Full certified clamp optimum. Reacher threads 1/24 under the bridge, so it
    # inserts cleanly, but the split creep-topples (the shim seam lowers margin).
    "clamp_31_24": dict(dx=1.0 / 24.0, reacher=3, mu=0.7, overhang=31.0 / 24.0,
                        cells=[(0, -2), (1, -14), (2, -4), (1, 19)]),
    # The monolith's falsework-standing back-off. Inserts cleanly (thread 3/24)
    # but still topples as a split: the split needs a deeper back-off to stand.
    "clamp_29_24": dict(dx=1.0 / 24.0, reacher=3, mu=0.7, overhang=29.0 / 24.0,
                        cells=[(0, -2), (1, -14), (2, -4), (1, 17)]),
    # The sweet spot. Reacher threads 6/24: still inserts (just below the ram
    # threshold) and the split stands. Full hold-and-shim success at eps=0.02.
    "clamp_26_24": dict(dx=1.0 / 24.0, reacher=3, mu=0.7, overhang=26.0 / 24.0,
                        cells=[(0, -2), (1, -14), (2, -4), (1, 14)]),
    # One grid step deeper. The split stands (exact poses) but the reacher now
    # threads 7/24 and rams the cantilevered bridge during the slide.
    "clamp_25_24": dict(dx=1.0 / 24.0, reacher=3, mu=0.7, overhang=25.0 / 24.0,
                        cells=[(0, -2), (1, -14), (2, -4), (1, 13)]),
    # n=6 4/3 primary candidate: its base cube is a knife-edge on the pedestal
    # rim that the reacher slide tips, so the reacher stalls short.
    "n6_4_3": dict(dx=1.0 / 12.0, reacher=5, mu=0.7, overhang=4.0 / 3.0,
                   cells=[(0, 0), (1, -6), (0, -36), (2, -1), (3, -4), (1, 10)]),
}

# Which designs get a full eps sweep vs a single representative eps in main().
SWEEP_DESIGNS = ("clamp_31_24", "clamp_26_24", "n6_4_3")
SINGLE_DESIGNS = {"clamp_29_24": 0.02, "clamp_25_24": 0.02}

# Reacher back-off scan for the standing-threshold table, per family. Grid steps
# j for the reacher cell, descending overhang.
BACKOFF = {
    "clamp": dict(dx=1.0 / 24.0, mu=0.7, others=[(0, -2), (1, -14), (2, -4)],
                  reacher_layer=1, js=[19, 17, 15, 13, 11]),
    "n6": dict(dx=1.0 / 12.0, mu=0.7,
               others=[(0, 0), (1, -6), (0, -36), (2, -1), (3, -4)],
               reacher_layer=1, js=[10, 9, 8, 7]),
}

EPS_SET = (0.01, 0.02, 0.04)

# Impedance gains. The reacher is a full block (about 2000 kg); the shim is thin
# and light, so its gains are scaled down to stay near-critically damped. Both
# drivers carry the held block's weight as a feedforward (a gripper holds the
# block up), so the block does not sag below its target and ram a neighbor.
REACHER_GAINS = dict(kp=6.0e5, kd=1.8e5, kp_rot=5.0e6, kd_rot=8.0e5)
SHIM_GAINS = dict(kp=4.0e5, kd=5.0e4, kp_rot=2.0e5, kd_rot=3.0e4)


def nominal_boxes(spec):
    """Pedestal plus the design's cubes at nominal poses (monolithic reacher)."""
    return [pedestal6()] + [cube(l, j * spec["dx"]) for (l, j) in spec["cells"]]


def clamp_overlap(spec):
    """x interval (lo, hi) where the clamping bridge overlaps the reacher. The
    bridge is the block one layer above the reacher that overlaps it in x."""
    boxes = nominal_boxes(spec)
    ri = spec["reacher"]
    r = boxes[1 + ri]
    r_layer = spec["cells"][ri][0]
    rlo, rhi = r.position[0] - r.half_extents[0], r.position[0] + r.half_extents[0]
    for i, (layer, _j) in enumerate(spec["cells"]):
        if layer == r_layer + 1:
            b = boxes[1 + i]
            lo = max(rlo, b.position[0] - b.half_extents[0])
            hi = min(rhi, b.position[0] + b.half_extents[0])
            if hi > lo:
                return (float(lo), float(hi))
    raise ValueError(f"no clamping block found for {spec}")


def split_state(spec, eps, footprint):
    """Full box list of the split design: pre-reacher blocks plus short reacher
    plus shim. The reacher (last cell) is replaced."""
    boxes = nominal_boxes(spec)
    ri = spec["reacher"]
    short, shim = split_reacher(
        boxes[1 + ri], eps, footprint=footprint, tail_x=clamp_overlap(spec)
    )
    pre = boxes[: 1 + ri] + boxes[1 + ri + 1 :]
    return pre, short, shim


# --------------------------------------------------------------------------
# Step 1: static certification.
# --------------------------------------------------------------------------


def certify_split(spec, eps, footprint):
    """Certify the split state and the short-reacher-only state through the host
    pipeline. The split must be feasible; the short reacher alone must not, since
    the eps gap above it un-certifies the clamp (that is why the shim exists)."""
    pre, short, shim = split_state(spec, eps, footprint)
    st, mg = certify(pre + [short, shim], spec["mu"])
    st0, mg0 = certify(pre + [short], spec["mu"])
    return {
        "eps": eps,
        "footprint": footprint,
        "status": st,
        "margin": mg,
        "short_only_status": st0,
        "short_only_margin": mg0,
    }


def run_certification(name):
    spec = DESIGNS[name]
    print()
    print(f"=== certification: {name} (overhang {spec['overhang']:.4f}) ===")
    nom_status, nom_margin = certify(nominal_boxes(spec), spec["mu"])
    print(f"  nominal (monolithic reacher): {nom_status} margin={nom_margin:.3e}")
    print(
        f"  {'eps':>6s} {'footprint':>10s} {'split':>10s} {'margin':>11s} "
        f"{'short-only':>11s} {'gap margin':>11s}"
    )
    rows = []
    for eps in EPS_SET:
        for fp in ("full", "tail"):
            r = certify_split(spec, eps, fp)
            rows.append(r)
            print(
                f"  {eps:6.2f} {fp:>10s} {r['status']:>10s} {r['margin']:11.3e} "
                f"{r['short_only_status']:>11s} {r['short_only_margin']:11.3e}"
            )
    # Best footprint by margin at each eps.
    best = {}
    for eps in EPS_SET:
        cand = [r for r in rows if r["eps"] == eps and r["status"] == "feasible"]
        if cand:
            best[eps] = max(cand, key=lambda r: r["margin"])["footprint"]
    print(f"  best footprint by margin: {best}")
    return {"design": name, "overhang": spec["overhang"],
            "nominal_status": nom_status, "nominal_margin": nom_margin,
            "rows": rows, "best_footprint": best}


# --------------------------------------------------------------------------
# Standing-threshold scan: settle the split at exact poses across reacher
# back-off positions. Locates the overhang at which the split first stands.
# --------------------------------------------------------------------------


def standing_threshold(family, eps=0.02, footprint="full", duration=6.0):
    cfg = BACKOFF[family]
    print()
    print(f"=== standing threshold: {family} split (eps={eps}, {footprint}) ===")
    base = [pedestal6()] + [cube(l, j * cfg["dx"]) for (l, j) in cfg["others"]]
    ov = None
    rows = []
    first_stand = None
    for j in cfg["js"]:
        r = cube(cfg["reacher_layer"], j * cfg["dx"])
        rlo = float(r.position[0] - r.half_extents[0])
        # bridge for this family: the block one layer above overlapping in x.
        if ov is None:
            for (bl, bj) in cfg["others"]:
                if bl == cfg["reacher_layer"] + 1:
                    b = cube(bl, bj * cfg["dx"])
                    lo = max(rlo, float(b.position[0] - b.half_extents[0]))
                    hi = min(
                        float(r.position[0] + r.half_extents[0]),
                        float(b.position[0] + b.half_extents[0]),
                    )
                    ov = (lo, hi)
        short, shim = split_reacher(r, eps, footprint=footprint, tail_x=ov)
        split = base + [short, shim]
        v = settle_test(split, cfg["mu"], duration=duration, solref=INSERT_SOLREF)
        overhang = j * cfg["dx"] + 0.5
        rows.append({"j": j, "overhang": overhang, "verdict": v["verdict"],
                     "max_rot": v["max_rot"], "max_disp_rel": v["max_disp_rel"]})
        if v["stable"] and first_stand is None:
            first_stand = overhang
        print(f"  j={j:3d} overhang={overhang:.4f}: {v['verdict']:8s} "
              f"rot={v['max_rot']:.4f} disp_rel={v['max_disp_rel']:.5f}")
    print(f"  first back-off that stands ({duration:.0f}s stiff): "
          f"{first_stand if first_stand is not None else 'none in scan'}")
    return {"family": family, "eps": eps, "footprint": footprint,
            "duration": duration, "first_standing_overhang": first_stand,
            "rows": rows}


# --------------------------------------------------------------------------
# Step 2: build simulation.
# --------------------------------------------------------------------------


def drop_pre_reacher(spec, timestep=1e-3):
    """Drop every non-reacher block with the rigid weld drive (mujoco_insert.
    insert_step). Returns the settled boxes (pedestal first)."""
    placed = [pedestal6()]
    reports = []
    for i, (layer, j) in enumerate(spec["cells"]):
        if i == spec["reacher"]:
            continue
        rep, placed = insert_step(
            placed, cube(layer, j * spec["dx"]), spec["mu"], DROP, timestep=timestep
        )
        reports.append({"cell": [layer, j], "reached": rep["reached"],
                        "reach_err": rep["reach_err"],
                        "settle_disp_rel": rep["settle_disp_rel"]})
    return placed, reports


def _y_identity():
    return np.array([1.0, 0.0, 0.0, 0.0])


def _wrench(model, data, bid, tgt, quat, gains, max_push):
    import mujoco  # noqa: F401

    dof = int(model.body_dofadr[bid])
    R = data.xmat[bid].reshape(3, 3)
    linvel = data.qvel[dof : dof + 3].copy()
    angvel = R @ data.qvel[dof + 3 : dof + 6]
    return capped_impedance_wrench(
        data.xpos[bid], data.xquat[bid], tgt, quat, linvel, angvel,
        kp=gains["kp"], kd=gains["kd"], kp_rot=gains["kp_rot"],
        kd_rot=gains["kd_rot"], max_push=max_push, max_torque=max_push,
    )


def hold_and_shim(
    pre_boxes, short, shim, mu, *,
    reacher_tilt=0.0, shim_tilt=0.0,
    reacher_cap_x=8.0, shim_cap_block=1.0,
    slide_off=0.9, ready_off=0.12, timestep=5e-4, solref=None,
    t_reach=2.5, hold_a=1.5, t_descend=0.8, hold_ready=0.4,
    t_shim=1.5, hold_b=0.8, settle_shim=0.6, settle_reacher=1.5,
    on_step=None,
):
    """Slide the short reacher in while held, then drive the shim in with its own
    capped impedance driver, then release both.

    reacher_cap_x: reacher push cap in reacher weights. shim_cap_block: shim push
    cap in unit-block weights (the shim is light; this is the honest force scale).
    Both held blocks carry a gravity feedforward so they do not sag. Soft contacts
    (solref None) admit the bounded penetration transient of the eps-into-eps shim
    seat. Returns (report, settled_boxes, mid_shim_snapshot)."""
    import mujoco

    q = _y_identity()
    P = len(pre_boxes)
    r_seat = short.position.copy()
    s_seat = shim.position.copy()
    r_start = r_seat + np.array([slide_off, 0.0, 0.0])
    s_park = s_seat + np.array([ready_off, 0.0, 0.5])
    r_quat = _tilt(q, reacher_tilt)
    s_quat = _tilt(q, shim_tilt)
    boxes = list(pre_boxes) + [
        Box(short.half_extents, r_start, r_quat, short.density),
        Box(shim.half_extents, s_park, s_quat, shim.density),
    ]
    L = assembly_diagonal(list(pre_boxes) + [short, shim])
    model = mujoco.MjModel.from_xml_string(
        to_mjcf(boxes, mu, timestep=timestep, all_pairs=True, solref=solref)
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    rid = int(model.body(f"block{P}").id)
    sid = int(model.body(f"block{P + 1}").id)
    pids = [int(model.body(f"block{i}").id) for i in range(P)]
    pos0 = {i: data.xpos[i].copy() for i in pids}
    quat0 = {i: data.xquat[i].copy() for i in pids}
    rw = float(model.body_mass[rid]) * G
    sw = float(model.body_mass[sid]) * G
    rff = np.array([0.0, 0.0, rw])
    sff = np.array([0.0, 0.0, sw])
    r_max = reacher_cap_x * rw
    s_max = shim_cap_block * CUBE_WEIGHT

    def disturb():
        return max(float(np.linalg.norm(data.xpos[i] - pos0[i])) for i in pids)

    def struct_rot():
        out = 0.0
        for i in pids:
            d = min(1.0, abs(float(np.dot(quat0[i], data.xquat[i]))))
            out = max(out, 2.0 * float(np.arccos(d)))
        return out

    metrics = dict(peak_reacher=0.0, peak_shim=0.0,
                   disturb_reacher=0.0, disturb_shim=0.0, nstep=0)

    def step(r_tgt, s_tgt, drive_shim, phase):
        fr, tr = _wrench(model, data, rid, r_tgt, r_quat, REACHER_GAINS, r_max)
        data.xfrc_applied[rid, :3] = fr + rff
        data.xfrc_applied[rid, 3:] = tr
        metrics["peak_reacher"] = max(metrics["peak_reacher"], float(np.linalg.norm(fr)))
        if drive_shim and s_tgt is not None:
            fs, ts = _wrench(model, data, sid, s_tgt, s_quat, SHIM_GAINS, s_max)
            data.xfrc_applied[sid, :3] = fs + sff
            data.xfrc_applied[sid, 3:] = ts
            metrics["peak_shim"] = max(metrics["peak_shim"], float(np.linalg.norm(fs)))
        mujoco.mj_step(model, data)
        d = disturb()
        if phase == "reacher":
            metrics["disturb_reacher"] = max(metrics["disturb_reacher"], d)
        else:
            metrics["disturb_shim"] = max(metrics["disturb_shim"], d)
        metrics["nstep"] += 1
        if on_step is not None:
            on_step(model, data, metrics["nstep"], phase)

    def ramp(n, r_of, s_of, drive_shim, phase):
        for k in range(max(1, n)):
            a = (k + 1) / max(1, n)
            step(r_of(a), None if s_of is None else s_of(a), drive_shim, phase)

    ns = lambda t: int(round(t / timestep))  # noqa: E731

    # Phase A: reacher slides in from +x, shim parked clear.
    ramp(ns(t_reach), lambda a: r_start + a * (r_seat - r_start),
         lambda a: s_park, True, "reacher")
    ramp(ns(hold_a), lambda a: r_seat, lambda a: s_park, True, "reacher")
    r_reach = float(np.linalg.norm(data.xpos[rid] - r_seat))

    # Phase B: shim descends onto the reacher clear of the bridge, then slides
    # horizontally under the bridge to its seat.
    ready = np.array([s_seat[0] + ready_off, 0.0, s_seat[2]])
    ramp(ns(t_descend), lambda a: r_seat, lambda a: s_park + a * (ready - s_park),
         True, "shim")
    ramp(ns(hold_ready), lambda a: r_seat, lambda a: ready, True, "shim")
    half = ns(t_shim) // 2
    mid_snapshot = None
    cur = 0
    for k in range(ns(t_shim)):
        a = (k + 1) / ns(t_shim)
        step(r_seat, np.array([ready[0] + a * (s_seat[0] - ready[0]), 0.0, s_seat[2]]),
             True, "shim")
        cur += 1
        if cur == half:
            mid_snapshot = _snapshot(model, data, boxes, pids + [rid, sid])
    ramp(ns(hold_b), lambda a: r_seat, lambda a: s_seat, True, "shim")
    shim_held_dx = abs(float(data.xpos[sid][0] - s_seat[0]))

    # Phase C: release the shim (hold the reacher), then release the reacher.
    data.xfrc_applied[sid] = 0.0
    ramp(ns(settle_shim), lambda a: r_seat, None, False, "shim")
    shim_seated_dx = abs(float(data.xpos[sid][0] - s_seat[0]))
    data.xfrc_applied[:] = 0.0
    for _ in range(ns(settle_reacher)):
        mujoco.mj_step(model, data)
        metrics["nstep"] += 1
        if on_step is not None:
            on_step(model, data, metrics["nstep"], "settle")

    settled = _snapshot(model, data, boxes, pids + [rid, sid])
    report = dict(
        r_reach=r_reach,
        reacher_seated=bool(r_reach < 0.003),
        shim_held_dx=shim_held_dx,
        shim_seated_dx=shim_seated_dx,
        shim_seated=bool(shim_seated_dx < 0.002),
        peak_reacher_push=metrics["peak_reacher"],
        peak_shim_push=metrics["peak_shim"],
        reacher_cap=r_max,
        shim_cap=s_max,
        shim_peak_block=metrics["peak_shim"] / CUBE_WEIGHT,
        disturb_reacher_rel=metrics["disturb_reacher"] / L,
        disturb_shim_rel=metrics["disturb_shim"] / L,
        struct_rot=struct_rot(),
        L=L,
    )
    return report, settled, mid_snapshot


def _tilt(base_quat, deg):
    t = np.radians(deg) / 2.0
    yq = np.array([np.cos(t), 0.0, np.sin(t), 0.0])
    b = np.asarray(base_quat, dtype=np.float64)
    return np.array([
        yq[0] * b[0] - yq[2] * b[2],
        yq[0] * b[1] + yq[2] * b[3],
        yq[0] * b[2] + yq[2] * b[0],
        yq[0] * b[3] - yq[2] * b[1],
    ])


def _snapshot(model, data, boxes, ids):
    out = []
    for idx, b in enumerate(ids):
        out.append(Box(boxes[idx].half_extents,
                       np.asarray(data.xpos[b]).copy(),
                       np.asarray(data.xquat[b]).copy(),
                       boxes[idx].density))
    return out


def run_build(name, eps, footprint="full", *, reacher_tilt=0.0, shim_tilt=0.0):
    spec = DESIGNS[name]
    pre, short, shim = split_state(spec, eps, footprint)
    pre_boxes, drop_reports = drop_pre_reacher(spec)
    pre_ok = all(r["reached"] for r in drop_reports)
    rep, settled, mid = hold_and_shim(
        pre_boxes, short, shim, spec["mu"],
        reacher_tilt=reacher_tilt, shim_tilt=shim_tilt,
    )
    v2 = settle_test(settled, spec["mu"], duration=2.0, solref=INSERT_SOLREF)
    v6 = settle_test(settled, spec["mu"], duration=6.0, solref=INSERT_SOLREF)
    success = bool(pre_ok and rep["reacher_seated"] and rep["shim_seated"]
                   and v2["stable"])
    rep.update(
        design=name, eps=eps, footprint=footprint, pre_drops_ok=pre_ok,
        verdict_2s=v2["verdict"], rot_2s=v2["max_rot"],
        verdict_6s=v6["verdict"], rot_6s=v6["max_rot"], success=success,
    )
    print(
        f"  eps={eps:.2f} {footprint:>4s}: pre_drops={'ok' if pre_ok else 'BAD'} "
        f"reacher={'seated' if rep['reacher_seated'] else 'STALL'}"
        f"({rep['r_reach'] * 1000:.1f}mm) "
        f"shim={'seated' if rep['shim_seated'] else 'short'}"
        f"({rep['shim_seated_dx'] * 1000:.2f}mm) "
        f"shim_push={rep['shim_peak_block']:.2f}blk "
        f"disturb(R/S)={rep['disturb_reacher_rel']:.4f}/{rep['disturb_shim_rel']:.4f} "
        f"| 2s={v2['verdict']}(rot{v2['max_rot']:.4f}) 6s={v6['verdict']} "
        f"=> {'SUCCESS' if success else 'no'}"
    )
    return rep, settled, mid


def render(tag, boxes, mu, out_dir):
    """Offscreen render of a box list. Returns the path or a reason string."""
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
        cam.distance = 10.0
        cam.azimuth = 90.0
        cam.elevation = -8.0
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        renderer.close()
        path = os.path.join(out_dir, f"shim_{tag}.png")
        try:
            import matplotlib.image as mpimg

            mpimg.imsave(path, img)
        except ImportError:
            return "no PNG writer (matplotlib) available; render skipped"
        return path
    except Exception as e:  # noqa: BLE001
        return f"render skipped ({type(e).__name__}: {e})"


def _movie_camera(mujoco):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 2.0]
    cam.distance = 10.0
    cam.azimuth = 90.0
    cam.elevation = -8.0
    return cam


def _brighten(img, gain=1.7):
    """Lift the default-headlight render out of the dark for the movie."""
    return np.clip(img.astype(np.float32) * gain, 0.0, 255.0).astype(np.uint8)


def _static_frames(boxes, mu, n, renderer, cam, mujoco):
    """Render one static frame of a box list, repeated n times to pause on it."""
    model = mujoco.MjModel.from_xml_string(to_mjcf(boxes, mu, solref=INSERT_SOLREF))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    img = _brighten(renderer.render().copy())
    return [img] * n


def build_movie(name, eps, out_dir, footprint="full", height=300, width=420,
                stride=140, fps=25):
    """Render the whole hold-and-shim build to an animated GIF: the pre-reacher
    drops as growing-structure frames, then the live reacher slide, shim drive,
    and release-and-settle. Returns the GIF path or a reason string."""
    try:
        import mujoco
        from PIL import Image
    except Exception as e:  # noqa: BLE001
        return f"movie skipped ({type(e).__name__}: {e})"
    try:
        spec = DESIGNS[name]
        mu = spec["mu"]
        pre, short, shim = split_state(spec, eps, footprint)
        pre_boxes, _ = drop_pre_reacher(spec)
        renderer = mujoco.Renderer(mujoco.MjModel.from_xml_string(
            to_mjcf(pre_boxes + [short, shim], mu, solref=INSERT_SOLREF)),
            height=height, width=width)
        cam = _movie_camera(mujoco)

        frames = []
        # Growing structure: pedestal, then each dropped block, as held frames.
        grow = [pedestal6()]
        frames += _static_frames(grow, mu, 6, renderer, cam, mujoco)
        for i, (layer, j) in enumerate(spec["cells"]):
            if i == spec["reacher"]:
                continue
            grow = grow + [cube(layer, j * spec["dx"])]
            frames += _static_frames(grow, mu, 10, renderer, cam, mujoco)

        # Live reacher slide, shim drive, and settle.
        live = []
        live_renderer = None

        def on_step(model, data, nstep, phase):
            nonlocal live_renderer
            if live_renderer is None:
                live_renderer = mujoco.Renderer(model, height=height, width=width)
            if nstep % stride == 0:
                live_renderer.update_scene(data, camera=cam)
                live.append(_brighten(live_renderer.render().copy()))

        hold_and_shim(pre_boxes, short, shim, mu, on_step=on_step)
        frames += live
        if live_renderer is not None:
            live_renderer.close()
        renderer.close()

        if not frames:
            return "movie skipped (no frames captured)"
        imgs = [Image.fromarray(f) for f in frames]
        path = os.path.join(out_dir, f"shim_movie_{name}_eps{eps}.gif")
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / fps), loop=0)
        return path
    except Exception as e:  # noqa: BLE001
        return f"movie skipped ({type(e).__name__}: {e})"


def run_design(name, args, eps_list):
    print()
    print(f"########## {name} ##########")
    cert = run_certification(name)
    print()
    print(f"=== build: {name} (eps {eps_list}) ===")
    builds = []
    best = None
    for eps in eps_list:
        rep, settled, mid = run_build(name, eps, footprint="full")
        builds.append(rep)
        if rep["success"] and best is None:
            best = (eps, settled, mid)
    min_eps = next((b["eps"] for b in builds if b["success"]), None)
    print(f"  minimum eps for a full hold-and-shim success: "
          f"{min_eps if min_eps is not None else 'none in {}'.format(tuple(eps_list))}")
    renders = {}
    if best is not None:
        eps, settled, mid = best
        if mid is not None:
            renders["mid_shim"] = render(f"{name}_mid_eps{eps}", mid, DESIGNS[name]["mu"], args.out)
        renders["final"] = render(f"{name}_final_eps{eps}", settled, DESIGNS[name]["mu"], args.out)
        for k, p in renders.items():
            print(f"  render {k}: {p}")
    return {"design": name, "certification": cert, "builds": builds,
            "min_success_eps": min_eps, "renders": renders}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/mujoco")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("mujoco not installed; install keystone[mujoco] to run this demo.")
        return

    thresholds = {
        "clamp": standing_threshold("clamp"),
        "n6": standing_threshold("n6"),
    }
    designs = {}
    for name in SWEEP_DESIGNS:
        designs[name] = run_design(name, args, list(EPS_SET))
    for name, eps in SINGLE_DESIGNS.items():
        designs[name] = run_design(name, args, [eps])

    # Movie of the whole build for the first design that fully succeeds.
    movie = None
    for name, d in designs.items():
        if d["min_success_eps"] is not None:
            print()
            print(f"=== rendering build movie: {name} (eps {d['min_success_eps']}) ===")
            movie = build_movie(name, d["min_success_eps"], args.out)
            print(f"  movie: {movie}")
            break

    out = {
        "meta": {
            "solref": list(INSERT_SOLREF),
            "eps_set": list(EPS_SET),
            "reacher_gains": REACHER_GAINS,
            "shim_gains": SHIM_GAINS,
            "cube_weight_N": CUBE_WEIGHT,
            "note": "hold-and-shim: short held reacher slides in, shim fills the "
            "eps gap, both release. Soft contacts during the drive admit bounded "
            "penetration transients; verdict is a stiff settle.",
        },
        "standing_thresholds": thresholds,
        "designs": designs,
        "movie": movie,
    }
    path = os.path.join(args.out, "mujoco_shim.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
