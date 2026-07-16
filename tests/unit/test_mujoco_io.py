"""Unit tests for the MuJoCo bridge (PLAN.md Section 8.1, 8.2).

mujoco is an optional extra. These tests skip cleanly when it is absent. They
stay well under a minute: short settle durations, small scenes.
"""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from keystone import Box, box_2d
from keystone.interop import from_mjcf, settle_test, to_mjcf


def _rot_close(qa, qb, atol=1e-11):
    """Quaternions match up to sign (q and -q are the same rotation)."""
    return abs(abs(float(np.dot(qa, qb))) - 1.0) < atol


def test_round_trip_poses_to_1e_12():
    # to_mjcf then from_mjcf reproduces box poses to machine precision. Full
    # float64 formatting in the MJCF is what makes this exact.
    boxes = [
        box_2d(6.0, 1.0, -3.0, 0.5),
        box_2d(1.0, 1.0, 0.13, 1.5, angle_y=0.3),
        Box(
            np.array([0.3, 0.4, 0.5]),
            np.array([2.0, 0.0, 0.75]),
            np.array([0.9238795325112867, 0.0, 0.3826834323650898, 0.0]),
            density=1500.0,
        ),
    ]
    xml = to_mjcf(boxes, 0.7)
    back = from_mjcf(xml)
    assert len(back) == len(boxes)
    for a, b in zip(boxes, back):
        assert np.allclose(a.position, b.position, rtol=0, atol=1e-12)
        assert np.allclose(a.half_extents, b.half_extents, rtol=0, atol=1e-12)
        assert _rot_close(a.quat, b.quat)
        assert abs(a.density - b.density) < 1e-9 * a.density


def test_round_trip_accepts_compiled_model():
    # from_mjcf also takes an already-compiled MjModel.
    boxes = [box_2d(1.0, 1.0, 0.0, 0.5)]
    model = mujoco.MjModel.from_xml_string(to_mjcf(boxes, 0.5))
    back = from_mjcf(model)
    assert len(back) == 1
    assert np.allclose(back[0].position, boxes[0].position, atol=1e-12)


def test_from_mjcf_rejects_non_box_geom():
    # A sphere geom is not supported in v1 and must raise, not silently drop.
    xml = """
    <mujoco>
      <worldbody>
        <geom name="ground" type="plane" size="5 5 0.1"/>
        <body pos="0 0 1"><freejoint/>
          <geom name="ball" type="sphere" size="0.5" density="1000"/>
        </body>
      </worldbody>
    </mujoco>
    """
    with pytest.raises(ValueError, match="box geoms only"):
        from_mjcf(xml)


def test_settle_resting_cube_is_stable():
    # A unit cube centered on the ground stays put.
    r = settle_test([box_2d(1.0, 1.0, 0.0, 0.5)], 0.7, duration=0.5)
    assert r["stable"] is True
    assert r["verdict"] == "stable"
    assert r["max_disp_rel"] < r["disp_tol_rel"]
    assert r["max_rot"] < r["rot_tol"]


def test_settle_floating_cube_falls():
    # A cube released above the ground falls: displacement far exceeds the band.
    r = settle_test([box_2d(1.0, 1.0, 0.0, 3.0)], 0.7, duration=0.6)
    assert r["stable"] is False
    assert r["verdict"] == "unstable"
    assert r["max_disp_rel"] > r["disp_tol_rel"]


def test_settle_offset_pair_055_collapses():
    # Two unit blocks, the upper offset by 0.55 > b/2 = 0.5. The top block's
    # center of mass sits past the contact, so it topples. keystone certifies
    # this state infeasible; MuJoCo agrees by toppling it.
    boxes = [box_2d(1.0, 1.0, 0.0, 0.5), box_2d(1.0, 1.0, 0.55, 1.5)]
    r = settle_test(boxes, 0.9, duration=1.0)
    assert r["stable"] is False
    assert r["max_rot"] > r["rot_tol"]


def test_settle_offset_pair_045_stands():
    # The same pair at offset 0.45 < 0.5 is stable, the agreeing feasible case.
    boxes = [box_2d(1.0, 1.0, 0.0, 0.5), box_2d(1.0, 1.0, 0.45, 1.5)]
    r = settle_test(boxes, 0.9, duration=1.0)
    assert r["stable"] is True


def test_to_mjcf_pairs_and_static():
    # Explicit pairs: one ground pair per block plus AABB-adjacent block pairs.
    # A stacked pair gives 2 ground pairs and 1 block-block pair. A static index
    # emits no free joint for that body.
    boxes = [box_2d(1.0, 1.0, 0.0, 0.5), box_2d(1.0, 1.0, 0.0, 1.5)]
    xml = to_mjcf(boxes, 0.6, free=[1])
    assert xml.count("<pair") == 3
    assert xml.count("freejoint") == 1  # only block 1 is free
    # The model compiles.
    model = mujoco.MjModel.from_xml_string(xml)
    assert model.npair == 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
