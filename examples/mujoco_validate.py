"""Dynamic validation of certified structures against MuJoCo (PLAN.md 8.3b).

keystone certifies static equilibrium. MuJoCo validates what a static
certificate cannot: whether the structure survives settle dynamics under a soft,
regularized contact model. The two answers should agree away from the margin
boundary. They are expected to disagree exactly at knife-edge optima, where the
certified margin is a rounding error above zero and the structure is an exact
limit state with no margin to spend on contact compliance. That disagreement
is the finding, not a bug in either system (PLAN.md Section 8.4).

Scenes (all 2D lattice designs with unit depth, dropped into MuJoCo as 3D
unit-cube stacks):
- corbel c = 0.98 (certified feasible, knife-edge) and c = 1.02 (infeasible).
- a 5-block aligned tower and the offset pair e = 0.45 / 0.55.
- the n=4 clamp optimum 31/24 at mu = 0.7 (knife-edge) and mu = 0.3 (infeasible),
  plus backed-off reacher variants.
- the n=6 4/3 design at mu = 0.7, plus backed-off variants.

For every certified-feasible knife-edge scene the harness also sweeps contact
stiffness (solref time constant) to record how rigid the contacts must be for
the exact limit state to stand. Run:

    python examples/mujoco_validate.py --out out/mujoco
"""

import argparse
import json
import os

import numpy as np

from keystone import (
    FEASIBLE,
    INFEASIBLE,
    Tolerances,
    assemble,
    box_2d,
    build_assembly,
    solve_p0,
    solve_p4,
)

TOL = Tolerances()

# Contact-stiffness sweep. None is the MuJoCo default (time constant 0.02 s).
# Smaller time constants are stiffer, closer to a rigid contact.
SOLREF_SWEEP = [None, (0.01, 1.0), (0.005, 1.0), (0.002, 1.0)]


def pedestal6():
    """Wide base for the lattice scenes. Right edge at x = 0, top at z = 1."""
    return box_2d(6.0, 1.0, -3.0, 0.5)


def cube(layer, x):
    """Unit cube at the given layer, center x, center z = 1.5 + layer."""
    return box_2d(1.0, 1.0, x, 1.5 + layer)


def corbel(m, c):
    """m unit blocks harmonically stacked on a 4x1 pedestal (test_gates_2d)."""
    ped = box_2d(4.0, 1.0, -2.0, 0.5)
    blocks = [ped]
    for b in range(m):
        j = m - b
        right_edge = sum(c / (2.0 * l) for l in range(j, m + 1))
        blocks.append(box_2d(1.0, 1.0, right_edge - 0.5, 1.5 + b))
    return blocks


def clamp_31_24(reacher_j=19):
    """The n=4 clamp optimum on a dx = 1/24 grid (test_bnb_optima).

    Sequence (layer, grid index j): (0,-2), (1,-14), (2,-4), (1, reacher_j).
    The layer-2 block is a bridge; the reacher sits on the base with a small
    overlap and is clamped from above by the bridge. reacher_j = 19 gives the
    certified optimum, right edge 31/24. Lower reacher_j pulls the reacher left.
    """
    dx = 1.0 / 24.0
    return [
        pedestal6(),
        cube(0, -2 * dx),
        cube(1, -14 * dx),
        cube(2, -4 * dx),
        cube(1, reacher_j * dx),
    ]


def design_n6(reacher_j=10):
    """The n=6 4/3 design from out/search/az_best_n6_seq.json (dx = 1/12)."""
    dx = 1.0 / 12.0
    return [
        pedestal6(),
        cube(0, 0.0),
        cube(1, -6 * dx),
        cube(0, -36 * dx),
        cube(2, -1 * dx),
        cube(3, -4 * dx),
        cube(1, reacher_j * dx),
    ]


def certify(boxes, mu):
    """Certified verdict and P4 margin from the host pipeline."""
    a = build_assembly(boxes, mu=mu, tol=TOL, dim=2)
    s = assemble(a, TOL, cone="linear2d")
    r0 = solve_p0(s, TOL)
    r4 = solve_p4(s, TOL)
    return r0.status, float(r4.margin)


def classify(cert_status, margin, mj_stable, mj_rot, rot_tol):
    """Agreement band between the certificate and the settle test."""
    if cert_status == INFEASIBLE:
        if not mj_stable:
            return "agree", ""
        return "DISAGREE", "certified infeasible but settled (would be a bug)"
    # certified feasible
    if mj_stable:
        return "agree", ""
    # feasible but toppled: knife-edge (zero-margin limit state).
    if mj_rot < rot_tol:
        note = "compliance sag (no rotation); recovers under stiffer contacts"
    else:
        note = "knife-edge topple; recovers under stiffer contacts"
    band = "knife-edge" if margin < 10.0 * TOL.tol_feas else "DISAGREE"
    return band, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/mujoco")
    ap.add_argument("--duration", type=float, default=2.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    try:
        from keystone.interop import settle_test
        import mujoco  # noqa: F401
    except ImportError:
        print("mujoco not installed; install keystone[mujoco] to run this demo.")
        return

    # Fixture ladder, all at MuJoCo default contact stiffness.
    fixtures = [
        ("corbel c=0.98", corbel(4, 0.98), 0.6),
        ("corbel c=1.02", corbel(4, 1.02), 0.6),
        ("tower 5-block", [box_2d(1, 1, 0, 0.5 + k) for k in range(5)], 0.7),
        ("pair e=0.45", [box_2d(1, 1, 0, 0.5), box_2d(1, 1, 0.45, 1.5)], 0.9),
        ("pair e=0.55", [box_2d(1, 1, 0, 0.5), box_2d(1, 1, 0.55, 1.5)], 0.9),
        ("clamp 31/24 mu=0.7", clamp_31_24(), 0.7),
        ("clamp 31/24 mu=0.3", clamp_31_24(), 0.3),
        ("n6 4/3 mu=0.7", design_n6(), 0.7),
    ]

    rows = []
    print()
    hdr = (
        f"{'scene':22s} {'certified':11s} {'margin':>11s}  "
        f"{'mujoco':9s} {'disp_rel':>9s} {'rot':>8s}  {'agreement':11s} note"
    )
    print(hdr)
    print("-" * len(hdr))
    for name, boxes, mu in fixtures:
        cert, margin = certify(boxes, mu)
        res = settle_test(boxes, mu, duration=args.duration)
        band, note = classify(
            cert, margin, res["stable"], res["max_rot"], res["rot_tol"]
        )
        rows.append(
            {
                "scene": name,
                "mu": mu,
                "certified": cert,
                "margin": margin,
                "mujoco": res["verdict"],
                "disp_rel": res["max_disp_rel"],
                "rot": res["max_rot"],
                "agreement": band,
                "note": note,
            }
        )
        print(
            f"{name:22s} {cert:11s} {margin:11.3e}  "
            f"{res['verdict']:9s} {res['max_disp_rel']:9.4f} "
            f"{res['max_rot']:8.4f}  {band:11s} {note}"
        )

    # Stiffness sweep for the certified-feasible knife-edge scenes. How rigid
    # must contacts be for the exact limit state to stand.
    knife = [
        ("corbel c=0.98", corbel(4, 0.98), 0.6),
        ("tower 5-block", [box_2d(1, 1, 0, 0.5 + k) for k in range(5)], 0.7),
        ("clamp 31/24 mu=0.7", clamp_31_24(), 0.7),
        ("n6 4/3 mu=0.7", design_n6(), 0.7),
    ]
    sweep = {}
    print()
    print("stiffness sweep (solref time constant; smaller = stiffer)")
    print(f"{'scene':22s} " + "  ".join(f"{str(s):>16s}" for s in SOLREF_SWEEP))
    for name, boxes, mu in knife:
        entries = []
        cells = []
        for solref in SOLREF_SWEEP:
            res = settle_test(boxes, mu, duration=args.duration, solref=solref)
            entries.append(
                {
                    "solref": solref,
                    "verdict": res["verdict"],
                    "disp_rel": res["max_disp_rel"],
                    "rot": res["max_rot"],
                }
            )
            cells.append(f"{res['verdict']}({res['max_rot']:.3f})")
        sweep[name] = entries
        print(f"{name:22s} " + "  ".join(f"{c:>16s}" for c in cells))

    # Backed-off reacher variants at default softness. Report the first that
    # settles.
    backoff = {}
    for label, builder, key_j, dx, steps in [
        ("clamp", clamp_31_24, 19, 1.0 / 24.0, [0, 1, 2]),
        ("n6", design_n6, 10, 1.0 / 12.0, [0, 1, 2]),
    ]:
        mu = 0.7
        entries = []
        first_settle = None
        print()
        print(f"backed-off {label} (reacher pulled left, mu={mu}, default soft)")
        for back in steps:
            j = key_j - back
            boxes = builder(j)
            cert, margin = certify(boxes, mu)
            res = settle_test(boxes, mu, duration=args.duration)
            edge = j * dx + 0.5
            e = {
                "back": back,
                "reacher_j": j,
                "edge": edge,
                "certified": cert,
                "margin": margin,
                "mujoco": res["verdict"],
                "disp_rel": res["max_disp_rel"],
                "rot": res["max_rot"],
            }
            entries.append(e)
            if res["stable"] and first_settle is None:
                first_settle = back
            print(
                f"  back {back} edge={edge:.4f} cert={cert:10s} "
                f"margin={margin:.3e} mj={res['verdict']:9s} "
                f"disp_rel={res['max_disp_rel']:.4f} rot={res['max_rot']:.4f}"
            )
        backoff[label] = {"first_settle": first_settle, "steps": entries}
        if first_settle is None:
            print(
                f"  none of the {label} back-off steps 0..{steps[-1]} settle "
                f"at default softness (fragile topology)"
            )
        else:
            print(f"  first {label} back-off to settle: {first_settle} step(s)")

    out = {
        "meta": {
            "duration": args.duration,
            "disp_tol_rel": 0.005,
            "rot_tol": 0.01,
            "tol_feas": TOL.tol_feas,
            "note": "MuJoCo soft-contact settle vs keystone static certificate",
        },
        "fixtures": rows,
        "stiffness_sweep": {
            k: [
                {**e, "solref": list(e["solref"]) if e["solref"] else None}
                for e in v
            ]
            for k, v in sweep.items()
        },
        "backoff": backoff,
    }
    path = os.path.join(args.out, "mujoco_validate.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
