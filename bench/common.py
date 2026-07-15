"""Shared helpers for the keystone benchmark scripts.

Stdlib only, so it imports the same under py3.11 (.venv-cra) and py3.12
(.venv). Two jobs:

- gather machine and interpreter facts for the RESULTS.md header,
- upsert marked sections into RESULTS.md so each script writes its own
  section idempotently and the file keeps a fixed section order no matter
  which script runs first.

Charter (CLAUDE.md Section 2, item item on performance numbers): a number
is quoted only from a recorded run of a committed script, and its section
records machine, python, package versions, exact command, and date.
"""

import datetime
import os
import platform
import re
import subprocess
import sys

# Fixed section order in RESULTS.md, independent of run order.
CANONICAL_ORDER = ["header", "throughput", "parity", "notvalidated"]


def _begin(key):
    return f"<!-- keystone-bench: {key} -->"


def _end(key):
    return f"<!-- /keystone-bench: {key} -->"


def cpu_brand():
    """CPU brand string. sysctl on darwin, platform.processor elsewhere."""
    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL,
            )
            return out.decode().strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


def machine_info():
    """Machine facts common to every run."""
    info = {
        "cpu": cpu_brand(),
        "platform": platform.platform(),
        "logical_cores": os.cpu_count(),
    }
    try:
        import psutil  # optional

        info["total_ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        info["total_ram_gb"] = None
    return info


def today():
    return datetime.date.today().isoformat()


def jax_info(jax):
    """jax version and device string for the running interpreter."""
    return {
        "version": jax.__version__,
        "devices": ", ".join(str(d) for d in jax.devices()),
    }


def md_table(headers, rows):
    """A GitHub markdown table. rows is a list of lists of str."""
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join([line, sep] + body)


def header_block():
    """The stable top banner. Identical from either script on a given day."""
    m = machine_info()
    ram = f", {m['total_ram_gb']} GB RAM" if m["total_ram_gb"] else ""
    return (
        "# keystone benchmark results\n\n"
        "Numbers here come only from committed scripts. Do not quote a number "
        "without rerunning its script. Each recorded-run section states the "
        "machine, python, package versions, exact command, and date of that "
        "run.\n\n"
        f"Machine: {m['cpu']}, {m['platform']}, "
        f"{m['logical_cores']} logical cores{ram}.\n"
        f"Last updated: {today()}.\n\n"
        "Commands:\n\n"
        "    /Users/krishna/research/keystone/.venv/bin/python bench/throughput.py\n"
        "    /Users/krishna/research/keystone/.venv-cra/bin/python bench/parity_cra.py\n"
    )


def not_validated_block():
    """Targets from CLAUDE.md Section 6 that this hardware cannot check."""
    return (
        "## Not validated\n\n"
        "These CLAUDE.md Section 6 targets need GPU hardware not present on "
        "this machine (Apple-silicon CPU only). They are pending, not failed.\n\n"
        + md_table(
            ["target", "status", "reason"],
            [
                [
                    "2D, <=12 blocks, >= 5000 P4 solves/s at batch 4096 on one "
                    "A100 or 4090",
                    "PENDING",
                    "no A100/4090 available; CPU throughput recorded above",
                ],
                [
                    "3D, 100 blocks ~300 patches, exact cold solve < 500 ms",
                    "PENDING",
                    "exact CPU backend (HiGHS/Clarabel) not in this slice; "
                    "JAX/qpax path only",
                ],
                [
                    "3D, warm one-block-appended solve < 50 ms",
                    "PENDING",
                    "sequence warm-start API (M5/C5) not built in this slice",
                ],
            ],
        )
        + "\n"
    )


def _parse_blocks(text):
    """Return {key: full_block_text} for every marked block found."""
    blocks = {}
    for key in set(re.findall(r"<!-- keystone-bench: ([\w-]+) -->", text)):
        pat = re.compile(
            re.escape(_begin(key)) + r".*?" + re.escape(_end(key)),
            re.DOTALL,
        )
        m = pat.search(text)
        if m:
            blocks[key] = m.group(0)
    return blocks


def upsert_section(path, key, body_md):
    """Write body_md as the marked section `key`, ordered canonically.

    Idempotent: rerunning replaces the section in place. Sections keep
    CANONICAL_ORDER regardless of which script wrote them or when.
    """
    text = ""
    if os.path.exists(path):
        with open(path, "r") as fh:
            text = fh.read()
    blocks = _parse_blocks(text)
    wrapped = _begin(key) + "\n" + body_md.rstrip("\n") + "\n" + _end(key)
    blocks[key] = wrapped
    ordered_keys = [k for k in CANONICAL_ORDER if k in blocks]
    ordered_keys += [k for k in blocks if k not in CANONICAL_ORDER]
    out = "\n\n".join(blocks[k] for k in ordered_keys) + "\n"
    with open(path, "w") as fh:
        fh.write(out)


def ensure_scaffold(path):
    """Write the header and Not-validated sections if absent (idempotent)."""
    upsert_section(path, "header", header_block())
    upsert_section(path, "notvalidated", not_validated_block())
