"""Unit tests for the end-to-end stacking pipeline (keystone.pipeline).

Plan classification is pure and fast. The movie recorder and the full pipeline
smoke need mujoco and skip cleanly without it. Kept small: a size-3 uniform
search, short MuJoCo drives, no movie in the smoke.

flax/jax load only when evaluate_stacking runs, not at import, so collection
stays light and does not perturb the numerically fragile 3D property test.
"""

import os

import numpy as np
import pytest

from keystone.pipeline import classify_build, evaluate_stacking


# --------------------------------------------------------------------------
# Stage 3: plan classification on hand-built sequences.
# --------------------------------------------------------------------------


def test_classify_drop_tower():
    # A straight vertical tower: every cube drops onto the one below with a
    # clear column above, so every placement is a drop.
    dx = 1.0 / 6.0
    seq = [(0, 0), (1, 0), (2, 0)]
    blocks = classify_build(n=3, dx=dx, seq=seq)
    assert [b["protocol"] for b in blocks] == ["drop", "drop", "drop"]
    # Drop params carry the descent geometry an executor needs.
    for b in blocks:
        assert "descent_h" in b["params"]
        assert b["params"]["target"]["z"] == 1.5 + b["layer"]


def test_classify_ride_under():
    # The counterweighted clamp: base, counterweight, bridge, then the reacher
    # threads in under the bridge from the open side. The reacher's column is
    # blocked overhead but a lateral corridor is clear, so it is a ride_under.
    dx = 1.0 / 24.0
    seq = [(0, -2), (1, -14), (2, -4), (1, 19)]
    blocks = classify_build(n=4, dx=dx, seq=seq)
    assert [b["protocol"] for b in blocks[:3]] == ["drop", "drop", "drop"]
    reacher = blocks[3]
    assert reacher["protocol"] == "ride_under"
    p = reacher["params"]
    assert p["tilt_deg"] > 0.0
    assert p["approach_side"] in ("+x", "-x")
    assert p["blocker"] is not None  # the overhead bridge


def test_classify_prop():
    # A cube boxed in on both sides at its layer and covered above: the column
    # is blocked (drop out) and both lateral corridors are blocked (ride_under
    # out), so it falls through to a prop. The cube is otherwise base-legal, so
    # the prop classification comes from reachability alone, not illegality.
    dx = 1.0 / 6.0
    seq = [(0, 0), (1, -7), (1, 7), (2, -3), (1, 0)]
    blocks = classify_build(n=6, dx=dx, seq=seq)
    boxed = blocks[4]
    assert boxed["protocol"] == "prop"
    prop = boxed["params"]["prop"]
    # A ground-borne column with a retract slider and the supported step.
    assert prop["axis"] == "z"
    assert prop["retract_disp"] < 0.0
    assert prop["supports_step"] == 4
    assert boxed["params"]["com_side"] in ("+x", "-x")


# --------------------------------------------------------------------------
# Movie recorder smoke.
# --------------------------------------------------------------------------


def test_movie_recorder_smoke(tmp_path):
    pytest.importorskip("mujoco")
    import mujoco

    from keystone import box_2d
    from keystone.interop.movies import FrameRecorder
    from keystone.interop.mujoco_io import to_mjcf

    boxes = [box_2d(6.0, 1.0, -3.0, 0.5), box_2d(1.0, 1.0, 0.0, 1.5)]
    model = mujoco.MjModel.from_xml_string(to_mjcf(boxes, 0.7))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    rec = FrameRecorder(model, data, height=120, width=160, stride=1, record=True)
    for _ in range(6):
        mujoco.mj_step(model, data)
        rec.capture(model, data)
    base = str(tmp_path / "smoke")
    out = rec.finalize(base)

    assert out["n_frames"] >= 5
    # The GIF (PIL) is always written; assert it is nonempty.
    assert out["gif"] and os.path.exists(out["gif"])
    assert os.path.getsize(out["gif"]) > 0
    assert out["still"] and os.path.exists(out["still"])


def test_movie_recorder_off_is_noop(tmp_path):
    # record=False imports nothing and writes nothing.
    from keystone.interop.movies import FrameRecorder

    rec = FrameRecorder(record=False)
    rec.capture()  # no model, no crash
    out = rec.finalize(str(tmp_path / "off"))
    assert out["mp4"] is None and out["gif"] is None
    assert out["skipped"] == "record=False"


# --------------------------------------------------------------------------
# Full pipeline smoke and record-schema completeness.
# --------------------------------------------------------------------------


def test_pipeline_smoke_and_schema(tmp_path):
    pytest.importorskip("mujoco")

    # Tiny uniform search (checkpoint intentionally missing), short drives, no
    # movie. Exercises all five stages and checks the record schema an arm
    # executor plug-in reads.
    rec = evaluate_stacking(
        n=3,
        dx=1.0 / 6.0,
        sims=50,
        seed=0,
        checkpoint="does/not/exist.msgpack",
        out_dir=str(tmp_path),
        record=False,
        settle_duration=0.5,
        verbose=False,
    )

    # Top-level schema.
    for key in ("n", "dx", "sims", "seed", "executor", "harmonic", "search",
                "agreement", "notes"):
        assert key in rec, f"missing top-level key {key}"
    assert rec["executor"] == "impedance_driver"
    assert rec["prior"] == "uniform"  # checkpoint missing -> uniform fallback
    assert rec["checkpoint_loaded"] is False

    # Search stage.
    s = rec["search"]
    for key in ("best_overhang", "sequence", "best_key", "ratio"):
        assert key in s
    assert isinstance(s["sequence"], list) and len(s["sequence"]) > 0

    # Certify stage.
    cert = rec["certify"]
    for key in ("steps", "prefix_feasible", "full_status", "margins"):
        assert key in cert
    assert cert["prefix_feasible"] is True
    assert len(cert["steps"]) == len(s["sequence"])

    # Plan stage: every block carries a protocol and the params an arm needs.
    plan = rec["plan"]
    assert set(plan["protocol_counts"]) == {"drop", "ride_under", "prop"}
    for b in plan["blocks"]:
        for key in ("step", "layer", "j", "x", "protocol", "params"):
            assert key in b
        assert b["protocol"] in ("drop", "ride_under", "prop")
        assert "target" in b["params"]

    # Execute stage.
    ex = rec["execute"]
    for key in ("executor", "steps", "verdict", "settle"):
        assert key in ex
    assert ex["executor"] == "impedance_driver"
    for st in ex["steps"]:
        for key in ("step", "protocol", "outcome", "peak_push",
                    "struct_disturb_rel", "struct_rot"):
            assert key in st

    # Agreement: the three-way verdict.
    ag = rec["agreement"]
    for key in ("search_claim", "certificate", "physics", "three_way"):
        assert key in ag
    assert ag["search_claim"] is True
    assert ag["certificate"] is True

    # The JSON record was written to disk.
    js = os.path.join(str(tmp_path), "pipeline_n3_seed0.json")
    assert os.path.exists(js)
