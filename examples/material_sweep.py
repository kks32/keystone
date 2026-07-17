"""Certified material sweeps for the cube-stacking overhang problem.

The optimum overhang is a function of material. This script certifies it
across friction and density with keystone.search.bnb (best-first branch and
bound, admissible geometry bound, host-verified incumbent). Every run reports
a proven grid optimum or an honest interval when a budget stops it first.

Three sweeps, all at n=4 on the 1/12 grid, static placement:

A. Friction curve, homogeneous density. mu in {0.3, 0.4, 0.5, 0.6, 0.7,
   0.85, 1.0}. The 5/4 clamp needs its reacher held by friction, so low mu
   should knock out the clamp-class designs: expect a staircase in mu.

B. Density mix, mu = 0.7. Four length-4 inventories of cube densities,
   indexed by sorted-cell position (layer then x). Two share a multiset in
   opposite orders, which tests where the heavy cubes want to sit.

C. One combined point: the best inventory from B, now at mu = 0.5. Does
   heterogeneity rescue overhang where uniform-density friction cannot?

Densities are per sorted-cell position: the i-th cube in sorted (layer,
index) order takes densities[i]. See src/keystone/search/bnb.py.

Run:
  python examples/material_sweep.py
  python examples/material_sweep.py --budget-a 600 --budget-b 900
"""

import argparse
import json
import os
import time

from keystone import SolverOptions, Tolerances
from keystone.search import bnb
from keystone.search import lattice as LT

N = 4
DX = 1.0 / 12.0
OUT_DIR = "out/search/material"

FRICTIONS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]

INVENTORIES = [
    ("uniform_2000", (2000.0, 2000.0, 2000.0, 2000.0)),
    ("desc_4_2_2_1", (4000.0, 2000.0, 2000.0, 1000.0)),
    ("heavy_low_4_4_1_1", (4000.0, 4000.0, 1000.0, 1000.0)),
    ("heavy_high_1_1_4_4", (1000.0, 1000.0, 4000.0, 4000.0)),
]


def sorted_cells(sequence):
    """Cells of a build sequence sorted by (layer, index): the slot order."""
    return sorted((int(L), int(j)) for (L, j) in sequence)


def design_rows(sequence, densities, dx):
    """One row per cube: sorted-cell position, layer, x, right edge, density."""
    cells = sorted_cells(sequence)
    rows = []
    for i, (L, j) in enumerate(cells):
        d = None if densities is None else float(densities[i])
        rows.append(
            {
                "slot": i,
                "layer": L,
                "j": j,
                "x": j * dx,
                "edge": j * dx + 0.5,
                "density": d,
            }
        )
    return rows


def run_one(label, *, mu, densities, budget, tol, opts):
    """Certify one material point and return a plot-ready record."""
    print(
        f"\n=== {label}: mu={mu} densities={densities} budget={budget}s ===",
        flush=True,
    )
    t0 = time.perf_counter()
    res = bnb.certify(
        N, DX, tol, opts=opts, progress=False,
        mu=mu, densities=densities, time_limit=budget,
    )
    wall = time.perf_counter() - t0
    rows = design_rows(res.sequence, densities, DX)
    record = {
        "label": label,
        "n": N,
        "dx": DX,
        "mu": mu,
        "densities": None if densities is None else [float(d) for d in densities],
        "certified": bool(res.certified),
        "optimum": res.optimum,
        "lower": res.lower,
        "upper": res.upper,
        "stop_reason": res.stop_reason,
        "host_verified": bool(res.host_verified),
        "sequence": [[int(L), int(j)] for (L, j) in res.sequence],
        "design": rows,
        "nodes_expanded": res.nodes_expanded,
        "nodes_generated": res.nodes_generated,
        "qp_solves": res.qp_solves,
        "host_verifications": res.host_verifications,
        "wall_time": wall,
    }
    verdict = (
        f"optimum {res.optimum:.6f}"
        if res.certified
        else f"interval [{res.lower:.6f}, {res.upper:.6f}] ({res.stop_reason})"
    )
    seq_str = ", ".join(f"({L},{j})" for (L, j) in res.sequence)
    print(
        f"  {verdict}  seq [{seq_str}]  nodes={res.nodes_expanded} "
        f"host_verified={res.host_verified} wall={wall:.1f}s",
        flush=True,
    )
    return record


def fmt_optimum(rec):
    """Certified optimum or interval, as a short string."""
    if rec["certified"]:
        return f"{rec['optimum']:.4f}"
    return f"[{rec['lower']:.4f},{rec['upper']:.4f}]"


def friction_table(records):
    """Text staircase table for sweep A."""
    lines = ["", "Friction curve (n=4, dx=1/12, homogeneous density):", ""]
    lines.append(f"  {'mu':>5}  {'optimum':>16}  {'reacher_edge':>12}  "
                 f"{'nodes':>7}  {'wall_s':>7}  stop")
    for r in records:
        edge = max((row["edge"] for row in r["design"]), default=float("nan"))
        lines.append(
            f"  {r['mu']:>5.2f}  {fmt_optimum(r):>16}  {edge:>12.4f}  "
            f"{r['nodes_expanded']:>7d}  {r['wall_time']:>7.1f}  {r['stop_reason']}"
        )
    return "\n".join(lines)


def density_table(records):
    """Text table for sweep B, with the density placed at each sorted cell."""
    lines = ["", "Density mix (n=4, dx=1/12, mu=0.7):", ""]
    lines.append(f"  {'inventory':>20}  {'optimum':>16}  {'nodes':>7}  "
                 f"{'wall_s':>7}  design (cell:density)")
    for r in records:
        design = " ".join(
            f"(L{row['layer']},x{row['x']:+.3f}):{int(row['density'])}"
            for row in r["design"]
        )
        lines.append(
            f"  {r['label']:>20}  {fmt_optimum(r):>16}  "
            f"{r['nodes_expanded']:>7d}  {r['wall_time']:>7.1f}  {design}"
        )
    return "\n".join(lines)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print(f"wrote {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Certified material sweeps.")
    parser.add_argument("--budget-a", type=float, default=600.0,
                        help="per-point wall budget for the friction curve")
    parser.add_argument("--budget-b", type=float, default=900.0,
                        help="per-point wall budget for the density mix and C")
    parser.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = parser.parse_args()

    tol = Tolerances()
    opts = SolverOptions()

    # --- A. Friction curve, homogeneous density. ---
    a_records = []
    for mu in FRICTIONS:
        a_records.append(
            run_one(f"mu_{mu}", mu=mu, densities=None, budget=args.budget_a,
                    tol=tol, opts=opts)
        )
    write_json(os.path.join(args.out_dir, "friction_curve.json"), a_records)

    # --- B. Density mix at mu = 0.7. ---
    b_records = []
    for label, inv in INVENTORIES:
        b_records.append(
            run_one(label, mu=0.7, densities=inv, budget=args.budget_b,
                    tol=tol, opts=opts)
        )
    write_json(os.path.join(args.out_dir, "density_mix.json"), b_records)

    # Best inventory from B: highest certified optimum, then lower bound.
    best = max(b_records, key=lambda r: (r["optimum"], r["certified"]))
    best_inv = best["densities"]

    # --- C. Best inventory at mu = 0.5. ---
    c_record = run_one(
        f"{best['label']}_mu0.5", mu=0.5, densities=tuple(best_inv),
        budget=args.budget_b, tol=tol, opts=opts,
    )
    # The uniform baseline at mu=0.5 for the rescue comparison.
    uniform_c = run_one(
        "uniform_2000_mu0.5", mu=0.5, densities=None, budget=args.budget_b,
        tol=tol, opts=opts,
    )
    write_json(
        os.path.join(args.out_dir, "combined.json"),
        {"best_inventory": best["label"], "heterogeneous": c_record,
         "uniform": uniform_c},
    )

    # --- Tables. ---
    report = [
        friction_table(a_records),
        density_table(b_records),
        "",
        f"Combined point (best inventory {best['label']} at mu=0.5):",
        f"  heterogeneous: {fmt_optimum(c_record)} "
        f"(stop {c_record['stop_reason']}, wall {c_record['wall_time']:.1f}s)",
        f"  uniform 2000:  {fmt_optimum(uniform_c)} "
        f"(stop {uniform_c['stop_reason']}, wall {uniform_c['wall_time']:.1f}s)",
    ]
    text = "\n".join(report)
    print("\n" + text, flush=True)
    txt_path = os.path.join(args.out_dir, "summary.txt")
    os.makedirs(os.path.dirname(txt_path) or ".", exist_ok=True)
    with open(txt_path, "w") as f:
        f.write(text + "\n")
    print(f"\nwrote {txt_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
