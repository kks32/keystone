"""Search throughput benchmark: naive baseline vs the batched lattice.

Records, on this CPU:
- the naive baseline rate, quoted from the committed out/search/*.log
  artifacts of examples/search_overhang.py (parsed, never from memory),
- the batched search rate at K=1 and K=16 for n=4 and n=6,
- the pure expand_kernel_batch rate at (K=16, M=full grid): wall time per
  call and per candidate.

Numbers are honest about the device. CPU vmap runs the K * M candidate QPs
in sequence, so batching does not speed the wall clock here; the batched
kernel is written for a GPU, where the same call fills the device. No GPU
number is produced until this runs on a GPU.

The table is written to <out>/search_perf.txt and printed. A marked
section is also upserted into bench/SEARCH_RESULTS.md (a separate file from
bench/RESULTS.md) through the shared bench.common helpers.

Run: python bench/search_perf.py --out out/search
"""

import argparse
import glob
import os
import re
import sys
import time

import jax
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

from keystone import Tolerances  # noqa: E402
from keystone.search import Search  # noqa: E402
from keystone.search import lattice as LT  # noqa: E402

SEARCH_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SEARCH_RESULTS.md")


def parse_naive_logs(search_dir):
    """Parse committed naive logs for (config, best_rate_sims_per_s, wall).

    Reads only logs written by examples/search_overhang.py (header line
    starts with 'search:'), skipping this benchmark's own fast_* logs.
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(search_dir, "*.log"))):
        base = os.path.basename(path)
        if base.startswith("fast_"):
            continue
        try:
            with open(path) as fh:
                text = fh.read()
        except OSError:
            continue
        header = re.search(r"^search: (n=\d+ sims=\d+ dx=1/\d+.*)$", text, re.M)
        if not header:
            continue
        rates = [float(x) for x in re.findall(r"([\d.]+)\s*sims/s", text)]
        wall = re.search(r"wall time\s*=\s*([\d.]+)\s*s", text)
        perqp = re.search(r"solver per qp\s*=\s*([\d.]+)\s*ms", text)
        rows.append(
            {
                "file": base,
                "config": header.group(1).strip(),
                "rate": max(rates) if rates else None,
                "wall": float(wall.group(1)) if wall else None,
                "per_qp_ms": float(perqp.group(1)) if perqp else None,
            }
        )
    return rows


def measure_search_rate(n, dx, batch, sims, tol, seed=0, search_iter=50):
    """Run the batched search for a capped sim budget; return sims/s and stats."""
    s = Search(n=n, dx=dx, tol=tol, batch=batch, seed=seed, search_iter=search_iter)
    s.run(sims, progress=None)
    return {
        "n": n,
        "K": batch,
        "sims": s.sims_done,
        "wall": s.wall,
        "rate": s.sims_done / s.wall if s.wall > 0 else float("nan"),
        "n_qp": s.n_qp,
        "t_solve": s.t_solve,
        "t_legal": s.t_legal,
        "best": s.best_overhang,
    }


def measure_kernel_rate(n, dx, tol, K, search_iter=50, reps=3):
    """Time expand_kernel_batch at (K leaves, M full grid). Returns dict."""
    spec = LT.LatticeSpec(n_max=n, dx=dx)
    cand_L, cand_J = LT.action_grid(spec)
    M = spec.M
    # K distinct small reachable leaf states: single layer-0 cubes spread out.
    rng = np.random.default_rng(0)
    js = rng.integers(spec.j_lo, 1, size=K)  # inside the pedestal
    keys = [((0, int(j)),) for j in js]
    states = LT.batch_states(spec, keys)
    opts_tol = 1e-9

    # Compile.
    legal, margins, cert = LT.expand_kernel_batch(
        spec, states, cand_L, cand_J, tol.eps_reg, tol.tol_cone,
        solver_tol=opts_tol, max_iter=search_iter,
    )
    legal.block_until_ready()
    # Timed.
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        legal, margins, cert = LT.expand_kernel_batch(
            spec, states, cand_L, cand_J, tol.eps_reg, tol.tol_cone,
            solver_tol=opts_tol, max_iter=search_iter,
        )
        margins.block_until_ready()
        best = min(best, time.perf_counter() - t0)
    n_cand = K * M
    return {
        "n": n,
        "K": K,
        "M": M,
        "n_cand": n_cand,
        "call_s": best,
        "per_cand_ms": 1000.0 * best / n_cand,
        "rate_cand_s": n_cand / best,
    }


def main():
    ap = argparse.ArgumentParser(description="Search throughput benchmark.")
    ap.add_argument("--out", default="out/search", help="output directory")
    ap.add_argument("--rate_sims", type=int, default=320,
                    help="capped sim budget for each rate measurement")
    ap.add_argument("--search_iter", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tol = Tolerances()
    dx = 1.0 / 24.0

    machine = common.machine_info()
    jinfo = common.jax_info(jax)
    dtype = "float64" if jax.config.jax_enable_x64 else "float32"

    lines = []
    lines.append("keystone search throughput (CPU)")
    lines.append("")
    lines.append(f"machine: {machine['cpu']}, {machine['platform']}, "
                 f"{machine['logical_cores']} logical cores")
    lines.append(f"jax: {jinfo['version']}, devices: {jinfo['devices']}, dtype: {dtype}")
    lines.append(f"date: {common.today()}")
    lines.append(f"grid: dx=1/{round(1/dx)}, search_iter={args.search_iter}, "
                 f"rate over {args.rate_sims} sims")
    lines.append("")

    # Naive baseline from committed artifacts.
    naive = parse_naive_logs(args.out)
    lines.append("naive baseline (examples/search_overhang.py, from out/search/*.log):")
    if naive:
        for r in naive:
            wall = f"{r['wall']:.0f}s" if r["wall"] else "partial"
            perqp = f"{r['per_qp_ms']:.2f}ms/qp" if r["per_qp_ms"] else "-"
            lines.append(f"  {r['config']}: {r['rate']} sims/s ({wall}, {perqp}) "
                         f"[{r['file']}]")
    else:
        lines.append("  (no naive logs found in out/search)")
    lines.append("")

    # Batched search rate at K=1 and K=16 for n=4 and n=6.
    lines.append(f"batched lattice search (this work), rate over {args.rate_sims} sims:")
    search_rows = []
    for n in (4, 6):
        for K in (1, 16):
            r = measure_search_rate(n, dx, K, args.rate_sims, tol,
                                     search_iter=args.search_iter)
            search_rows.append(r)
            lines.append(
                f"  n={r['n']} K={r['K']:2d}: {r['rate']:6.1f} sims/s  "
                f"({r['sims']} sims / {r['wall']:.1f}s, {r['n_qp']} qp, "
                f"solve {r['t_solve']:.1f}s legality {r['t_legal']:.2f}s)"
            )
    lines.append("")

    # Pure kernel rate at (K=16, M=full grid).
    lines.append("pure expand_kernel_batch at (K=16, M=full grid):")
    kernel_rows = []
    for n in (4, 6):
        k = measure_kernel_rate(n, dx, tol, K=16, search_iter=args.search_iter)
        kernel_rows.append(k)
        lines.append(
            f"  n={k['n']}: M={k['M']} candidates/leaf, K*M={k['n_cand']} solves/call; "
            f"{k['call_s']*1000:.0f} ms/call, {k['per_cand_ms']:.3f} ms/candidate, "
            f"{k['rate_cand_s']:.0f} candidate-solves/s"
        )
    lines.append("")
    lines.append("Note: CPU vmap serializes the K*M candidate QPs, so batching does "
                 "not cut wall time here. The kernel is device agnostic; a GPU runs "
                 "the K*M solves concurrently. No GPU number exists until this runs "
                 "on a GPU.")

    table = "\n".join(lines)
    print(table)
    out_txt = os.path.join(args.out, "search_perf.txt")
    with open(out_txt, "w") as fh:
        fh.write(table + "\n")
    print(f"\nwrote {out_txt}")

    # Upsert a marked section into the separate bench/SEARCH_RESULTS.md.
    body = (
        "# keystone search benchmark results\n\n"
        "Numbers come only from this committed script "
        "(bench/search_perf.py). Do not quote without rerunning.\n\n"
        f"Machine: {machine['cpu']}, {machine['platform']}, "
        f"{machine['logical_cores']} logical cores.\n"
        f"jax {jinfo['version']}, devices {jinfo['devices']}, dtype {dtype}. "
        f"Last updated: {common.today()}.\n\n"
        "Command:\n\n"
        "    /Users/krishna/research/keystone/.venv/bin/python bench/search_perf.py\n\n"
        "```\n" + table + "\n```\n"
    )
    common.upsert_section(SEARCH_RESULTS, "search", body)
    print(f"wrote {SEARCH_RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
