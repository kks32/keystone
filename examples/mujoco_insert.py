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

from keystone import Box, box_2d
from keystone.interop.mujoco_io import assembly_diagonal, to_mjcf

DROP = "drop"
SLIDE = "slide"

# Stiff contacts so the knife-edge structures stand while we test the motion.
INSERT_SOLREF = (0.002, 1.0)


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

    print()
    for name, path in renders.items():
        print(f"render {name}: {path}")

    out = {
        "meta": {
            "timestep": args.timestep,
            "speed": args.speed,
            "solref": list(INSERT_SOLREF),
            "note": "stiff contacts isolate insertion motion from knife-edge collapse",
        },
        "sequences": results,
        "renders": renders,
    }
    path = os.path.join(args.out, "mujoco_insert.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
