"""Falsework build of the counterweighted reacher designs (Route B, M5 story).

No build order for the clamp set is both statically prefix-feasible and
executable through clear space (drop and slide_clear optima are 1.0), and
Route A shows block tolerance cannot open the reacher slot: the clearance a
rigid slide needs is the clamp contact that holds the reacher up
(examples/mujoco_insert.py). The remaining physical route is falsework:
declare temporary supports, certify the propped prefixes, execute drops only,
retract the supports, settle.

Props are slender columns, one under every overhang that is unstable before
its counterweight or clamp arrives:
- a reacher prop on the ground under the reacher's overhang. It lets the
  reacher be dropped before its clamp exists.
- a counterweight prop on the pedestal under the counterweight's overhang.
  Without it the counterweight prefix is a knife-edge (certified margin 5e-9)
  that MuJoCo topples during the bridge's approach; the bridge then lands on
  the tipped corner and seats 0.13 m off pose.
- n6 only: a base prop under the base cube, which sits centered exactly on
  the pedestal's edge (another knife-edge prefix); the reacher landing on the
  unpropped base tips it 0.64 rad.

Protocol per design:
1. Certify every propped prefix through the host pipeline (props are real
   blocks in the assembly) plus the staged retraction states.
2. Execute in MuJoCo: all placements are compliant capped-impedance drops
   (mujoco_insert.compliant_insert) with the props as static bodies.
3. Retract one prop at a time on position-actuated sliders, in plan order.
   Pedestal-borne props slide out horizontally (the pedestal is under them);
   ground-borne props lower vertically like jacks, with no ground pair so the
   column can pass below the plane, the standard lowering abstraction. Slow
   ramps, settle between stages, settle verdict.
4. Report prefix margins, insertion forces, prop peak loads, retraction
   verdict, and a creep control: the exact certified design settled for the
   same duration with no props. The zero-margin optima creep-rotate under
   MuJoCo's compliant contacts (KNOWN_LIMITS.md); the control separates that
   model gap from anything the falsework route adds. Renders (mid-build and
   final) go to out/mujoco/.

Run:

    python examples/mujoco_falsework.py --out out/mujoco
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mujoco_insert import (  # noqa: E402
    DROP,
    IMPEDANCE,
    INSERT_SOLREF,
    certify,
    compliant_insert,
    cube,
    pedestal6,
)

from keystone import Box, box_2d  # noqa: E402
from keystone.interop.mujoco_io import assembly_diagonal, to_mjcf  # noqa: E402


def _fmt(x):
    return "%.17g" % float(x)


PROP_W = 0.2

# Falsework build plans. cells in placement order (layer, grid_index_j); the
# reacher goes in right after the base so no placement ever needs the
# under-bridge slide. Each prop: x, half_h and nominal center z (column
# geometry), axis ("x" slides out horizontally, "z" lowers vertically),
# retract_disp (signed slider displacement), and supports (index into cells
# of the block whose underside the prop top meets). Retraction runs in list
# order.
PLANS = {
    "clamp_31_24": dict(
        dx=1.0 / 24.0,
        # base, reacher (onto its prop), counterweight (onto its prop), bridge
        cells=[(0, -2), (1, 19), (1, -14), (2, -4)],
        props=[
            dict(name="cw", x=-0.85, half_h=0.5, z=1.5, axis="x",
                 retract_disp=-1.45, supports=2),
            dict(name="reacher", x=1.0, half_h=1.0, z=1.0, axis="z",
                 retract_disp=-1.4, supports=1),
        ],
        mu=0.7,
        overhang=31.0 / 24.0,
    ),
    # The certified clamp optimum creeps in MuJoCo at every tested contact
    # stiffness (zero margin, KNOWN_LIMITS.md); this variant backs the
    # reacher off two grid steps, which the 6 s settle control shows is
    # enough dynamic margin to stand.
    "clamp_29_24": dict(
        dx=1.0 / 24.0,
        cells=[(0, -2), (1, 17), (1, -14), (2, -4)],
        props=[
            dict(name="cw", x=-0.85, half_h=0.5, z=1.5, axis="x",
                 retract_disp=-1.45, supports=2),
            dict(name="reacher", x=1.0, half_h=1.0, z=1.0, axis="z",
                 retract_disp=-1.4, supports=1),
        ],
        mu=0.7,
        overhang=29.0 / 24.0,
    ),
    "n6_4_3": dict(
        dx=1.0 / 12.0,
        # base, reacher (onto its prop), counterweight (onto its prop),
        # ballast, clamp bridge, top block. The base cube sits centered on
        # the pedestal edge, so it gets a prop too.
        cells=[(0, 0), (1, 10), (1, -6), (0, -36), (2, -1), (3, -4)],
        props=[
            dict(name="cw", x=-0.8, half_h=0.5, z=1.5, axis="x",
                 retract_disp=-1.4, supports=2),
            dict(name="base", x=0.3, half_h=0.5, z=0.5, axis="z",
                 retract_disp=-1.2, supports=0),
            dict(name="reacher", x=1.1, half_h=1.0, z=1.0, axis="z",
                 retract_disp=-1.4, supports=1),
        ],
        mu=0.7,
        overhang=4.0 / 3.0,
    ),
}


def props_for(plan):
    """The props as real blocks (certification and static build bodies)."""
    return [
        box_2d(PROP_W, 2.0 * p["half_h"], p["x"], p["z"]) for p in plan["props"]
    ]


def certify_prefixes(plan):
    """Host-pipeline certification of every propped prefix and the staged
    retraction states. Props are ordinary blocks in the assembly."""
    prop_boxes = props_for(plan)
    dx = plan["dx"]
    rows = []
    placed = []
    for (layer, j) in plan["cells"]:
        placed.append(cube(layer, j * dx))
        status, margin = certify([pedestal6()] + placed + prop_boxes, plan["mu"])
        rows.append(
            {"n": len(placed), "cell": [layer, j], "status": status, "margin": margin}
        )
    remaining = list(prop_boxes)
    for p in plan["props"]:
        remaining = remaining[1:]
        status, margin = certify([pedestal6()] + placed + remaining, plan["mu"])
        rows.append(
            {
                "n": f"{p['name']} prop out",
                "cell": None,
                "status": status,
                "margin": margin,
            }
        )
    return rows


def build_with_props(plan, timestep=5e-4, relief=1e-3):
    """Execute the drops with all props static. Returns the settled boxes
    (pedestal first), per-step reports, and the mid-build snapshot.

    The static props are catchers: each top sits `relief` below the supported
    block's nominal underside. A prop top flush with the underside makes the
    drop an over-constrained press between two rigid surfaces; the driver's
    3 mm seating bias then wedges the block and pops it out laterally
    (observed 9e7 N and a 2.7 m ejection on the n6 base). With the relief the
    block seats on the structure and tips onto the prop by at most
    relief / lever, a few milliradians, which is where the certified propped
    equilibrium takes over."""
    prop_boxes = []
    for b in props_for(plan):
        he = b.half_extents.copy()
        he[2] -= relief / 2.0
        pos = b.position.copy()
        pos[2] -= relief / 2.0
        prop_boxes.append(Box(he, pos, b.quat, b.density))
    dx = plan["dx"]
    placed = [pedestal6()]
    steps = []
    mid_snapshot = None
    for i, (layer, j) in enumerate(plan["cells"]):
        target = cube(layer, j * dx)
        rep, settled = compliant_insert(
            placed,
            target,
            DROP,
            plan["mu"],
            static_boxes=prop_boxes,
            timestep=timestep,
        )
        rep["step"] = i
        rep["cell"] = [layer, j]
        steps.append(rep)
        placed = settled
        if i == 1:
            mid_snapshot = list(placed)  # reacher resting on its prop
        print(
            f"  step {i} drop {tuple((layer, j))}: {rep['outcome']:9s} "
            f"reach_err={rep['reach_err']:.4f} "
            f"push={rep['peak_push']:.2e}/{rep['max_push']:.2e} "
            f"prop_load={rep['peak_support_force']:.3e} "
            f"disturb={rep['struct_disturb_rel']:.4f} rot={rep['struct_rot']:.4f}"
        )
    return placed, steps, mid_snapshot


def retract_props(
    boxes,
    plan,
    *,
    timestep=5e-4,
    ramp_time=1.5,
    stage_settle=0.75,
    final_settle=1.5,
    solref=INSERT_SOLREF,
):
    """Retract the props one at a time on position-actuated sliders.

    boxes: pedestal plus built blocks at settled poses (all free). Each prop
    rides one slider, so the joint carries the other axes rigidly. Prop tops
    are set from the settled poses of the blocks they support, minus a 1 mm
    relief. Without the relief the rigid slider pins the prop top fractionally
    inside the sagged block and a horizontal slide-out drags the block through
    stiff-contact friction (observed 3.6e5 N spike), an artifact of the joint,
    not a structural load. Returns a report and the final block poses."""
    import mujoco

    relief = 1e-3
    axis_vec = {"x": "1 0 0", "z": "0 0 1"}
    world_parts = []
    act_parts = []
    for idx, p in enumerate(plan["props"]):
        blk = boxes[p["supports"] + 1]  # +1: pedestal is boxes[0]
        top = float(blk.position[2] - blk.half_extents[2]) - relief
        zc = top - p["half_h"]
        world_parts.append(
            f'    <body name="prop{idx}" pos="{_fmt(p["x"])} 0.0 {_fmt(zc)}">\n'
            f'      <joint name="prop{idx}slide" type="slide" '
            f'axis="{axis_vec[p["axis"]]}" range="-3.0 1.0" '
            f'damping="20000.0"/>\n'
            f'      <geom name="prop{idx}geom" type="box" '
            f'size="{_fmt(PROP_W / 2)} 0.5 {_fmt(p["half_h"])}" '
            f'density="2000.0" contype="0" conaffinity="0"/>\n'
            f"    </body>"
        )
        act_parts.append(
            f'    <position name="prop{idx}act" joint="prop{idx}slide" '
            f'kp="2000000.0" kv="200000.0" ctrlrange="-3.0 1.0"/>'
        )
    fric = "0.7 0.7 0.005 0.0001 0.0001"
    sr = f"{_fmt(solref[0])} {_fmt(solref[1])}"
    # Pairs with the placed blocks only. The sliders carry the props, so no
    # pedestal or ground pair exists: a joint-held prop overlapping the
    # pedestal by the relief would otherwise generate a huge phantom load.
    pairs = [
        f'    <pair geom1="prop{idx}geom" geom2="geom{i}" condim="3" '
        f'friction="{fric}" solref="{sr}"/>'
        for idx in range(len(plan["props"]))
        for i in range(1, len(boxes))
    ]
    xml = to_mjcf(
        boxes,
        plan["mu"],
        timestep=timestep,
        all_pairs=True,
        solref=solref,
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
    pgid = [int(model.geom(f"prop{idx}geom").id) for idx in range(len(plan["props"]))]
    L = assembly_diagonal(boxes)

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

    # Relax onto the props.
    for _ in range(n_stage):
        mujoco.mj_step(model, data)

    stages = []
    for idx, p in enumerate(plan["props"]):
        load_before = prop_load(idx)
        peak = load_before
        for t in range(n_ramp):
            data.ctrl[idx] = p["retract_disp"] * (t + 1) / n_ramp
            mujoco.mj_step(model, data)
            peak = max(peak, prop_load(idx))
        for _ in range(n_stage):
            mujoco.mj_step(model, data)
        stages.append(
            {
                "prop": p["name"],
                "axis": p["axis"],
                "load_before": load_before,
                "peak_during_retract": peak,
                "load_after": prop_load(idx),
            }
        )

    for _ in range(max(1, int(round(final_settle / timestep)))):
        mujoco.mj_step(model, data)

    disp = max(float(np.linalg.norm(data.xpos[i] - pos0[k])) for k, i in enumerate(ids))
    rot = max(
        2.0 * float(np.arccos(min(1.0, abs(float(np.dot(quat0[k], data.xquat[i]))))))
        for k, i in enumerate(ids)
    )
    stands = bool(disp / L < 0.01 and rot < 0.05)
    final = [
        Box(
            boxes[k].half_extents,
            np.asarray(data.xpos[i]).copy(),
            np.asarray(data.xquat[i]).copy(),
            boxes[k].density,
        )
        for k, i in enumerate(ids)
    ]
    report = {
        "stages": stages,
        "disp": disp,
        "disp_rel": disp / L,
        "rot": rot,
        "stands": stands,
        "verdict": "stands" if stands else "collapsed",
    }
    return report, final


def creep_control(plan, duration):
    """Settle the exact certified design (no props, no build) for the same
    duration as the retraction protocol."""
    from keystone.interop import settle_test

    dx = plan["dx"]
    boxes = [pedestal6()] + [cube(l, j * dx) for (l, j) in plan["cells"]]
    r = settle_test(boxes, plan["mu"], duration=duration, solref=INSERT_SOLREF)
    return {"verdict": r["verdict"], "disp_rel": r["max_disp_rel"], "rot": r["max_rot"]}


def render(name, boxes, mu, out_dir, extra_boxes=()):
    """Offscreen render. extra_boxes (the props) are drawn static. Returns the
    path or a reason string."""
    try:
        import mujoco

        all_boxes = list(boxes) + list(extra_boxes)
        free = list(range(len(boxes)))
        xml = to_mjcf(all_boxes, mu, solref=INSERT_SOLREF, free=free)
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
        path = os.path.join(out_dir, f"falsework_{name}.png")
        try:
            import matplotlib.image as mpimg

            mpimg.imsave(path, img)
        except ImportError:
            return "no PNG writer (matplotlib) available; render skipped"
        return path
    except Exception as e:  # noqa: BLE001
        return f"render skipped ({type(e).__name__}: {e})"


def run_design(name, args):
    plan = PLANS[name]
    print()
    print(f"=== falsework build: {name} (overhang {plan['overhang']:.4f}) ===")

    print("  propped prefix certification (props as real blocks):")
    prefixes = certify_prefixes(plan)
    for row in prefixes:
        print(
            f"    prefix {str(row['n']):>14s} cell={str(row['cell']):>10s} "
            f"{row['status']:>10s} margin={row['margin']:.2e}"
        )
    all_certified = all(r["status"] == "feasible" for r in prefixes)
    if not all_certified:
        print("  a propped prefix is infeasible; execution would be uncertified")

    built, steps, mid = build_with_props(plan, timestep=args.timestep)
    renders = {}
    if mid is not None:
        renders["mid"] = render(
            f"{name}_mid", mid, plan["mu"], args.out, props_for(plan)
        )

    retract, final = retract_props(built, plan, timestep=args.timestep)
    control = creep_control(plan, duration=6.0)
    for st in retract["stages"]:
        print(
            f"  retract {st['prop']:8s} ({st['axis']}): load "
            f"{st['load_before']:.3e} N, peak {st['peak_during_retract']:.3e}, "
            f"after {st['load_after']:.3e}"
        )
    print(
        f"  post-retraction: structure {retract['verdict']} "
        f"(disp {retract['disp_rel']:.4f} L, rot {retract['rot']:.4f}); "
        f"creep control (exact poses, no props, 6 s): {control['verdict']} "
        f"rot={control['rot']:.4f}"
    )
    renders["final"] = render(f"{name}_final", final, plan["mu"], args.out)

    all_seated = all(s["outcome"] == "seated" for s in steps)
    executed = bool(all_certified and all_seated)
    success = bool(executed and retract["stands"])
    if executed and not retract["stands"] and control["verdict"] == "unstable":
        note = (
            "execution clean; final creep-topple matches the exact-pose "
            "control, the knife-edge model gap of KNOWN_LIMITS.md, not a "
            "falsework failure"
        )
        print(f"  note: {note}")
    else:
        note = ""
    print(f"  falsework build {'SUCCEEDS' if success else 'FAILS'} for {name}")
    return {
        "design": name,
        "overhang": plan["overhang"],
        "mu": plan["mu"],
        "props": [
            {k: p[k] for k in ("name", "x", "axis", "retract_disp")}
            for p in plan["props"]
        ],
        "prefixes": prefixes,
        "steps": steps,
        "retraction": retract,
        "creep_control": control,
        "executed": executed,
        "success": success,
        "note": note,
        "renders": renders,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/mujoco")
    ap.add_argument("--timestep", type=float, default=5e-4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("mujoco not installed; install keystone[mujoco] to run this demo.")
        return

    results = {}
    results["clamp_31_24"] = run_design("clamp_31_24", args)
    if results["clamp_31_24"]["executed"]:
        results["clamp_29_24"] = run_design("clamp_29_24", args)
        results["n6_4_3"] = run_design("n6_4_3", args)
    else:
        print("clamp falsework execution failed; skipping variants per protocol")

    out = {
        "meta": {
            "timestep": args.timestep,
            "solref": list(INSERT_SOLREF),
            "impedance": IMPEDANCE,
            "note": "falsework route: certified propped prefixes, compliant "
            "drops, staged position-actuated prop retraction, creep control",
        },
        "designs": results,
    }
    path = os.path.join(args.out, "mujoco_falsework.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
