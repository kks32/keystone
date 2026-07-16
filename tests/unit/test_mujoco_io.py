"""Unit tests for the MuJoCo bridge (PLAN.md Section 8.1, 8.2).

mujoco is an optional extra. These tests skip cleanly when it is absent. They
stay well under a minute: short settle durations, small scenes.
"""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from keystone import Box, box_2d
from keystone.interop import (
    capped_impedance_wrench,
    from_mjcf,
    orientation_error,
    restacked_cubes,
    settle_test,
    to_mjcf,
)


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


def test_restacked_cubes_contact_planes_meet():
    # Shrinking by size_tol and re-stacking keeps every vertical contact
    # closed: layer L's top face equals layer L+1's bottom face, and layer 0
    # rests exactly on the pedestal top. Horizontal same-layer gaps open by
    # size_tol; the vertical slot between a block and the layer above stays
    # at zero clearance because its walls shrink with it.
    s = 0.02
    dx = 1.0 / 12.0
    tower = restacked_cubes([(0, 0), (1, 0), (2, 0)], s, dx)
    assert abs(tower[0].position[2] - tower[0].half_extents[2] - 1.0) < 1e-12
    for lower, upper in zip(tower, tower[1:]):
        top = lower.position[2] + lower.half_extents[2]
        bottom = upper.position[2] - upper.half_extents[2]
        assert abs(top - bottom) < 1e-12
    # In-plane side shrinks to 1 - s; depth stays 1.
    assert abs(2.0 * tower[0].half_extents[0] - (1.0 - s)) < 1e-12
    assert abs(2.0 * tower[0].half_extents[2] - (1.0 - s)) < 1e-12
    assert abs(2.0 * tower[0].half_extents[1] - 1.0) < 1e-12
    # Same-layer neighbors on adjacent grid cells (24 steps of 1/24 = one
    # nominal width apart) open a gap of exactly size_tol.
    pair = restacked_cubes([(0, 0), (0, 24)], s, 1.0 / 24.0)
    gap = (pair[1].position[0] - pair[1].half_extents[0]) - (
        pair[0].position[0] + pair[0].half_extents[0]
    )
    assert abs(gap - s) < 1e-12
    with pytest.raises(ValueError, match="size_tol"):
        restacked_cubes([(0, 0)], 1.0, dx)


def test_orientation_error_axis_angle():
    # Identity to a 0.2 rad rotation about y gives the rotation vector
    # (0, 0.2, 0); the identity pair gives zero.
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    qy = np.array([np.cos(0.1), 0.0, np.sin(0.1), 0.0])
    assert np.allclose(orientation_error(q0, qy), [0.0, 0.2, 0.0], atol=1e-12)
    assert np.allclose(orientation_error(qy, qy), 0.0, atol=1e-12)


def test_capped_impedance_wrench_clips_force():
    # The commanded force never exceeds max_push regardless of the position
    # error; direction is preserved.
    q = np.array([1.0, 0.0, 0.0, 0.0])
    f, t = capped_impedance_wrench(
        [0.0, 0.0, 0.0], q, [10.0, 0.0, 0.0], q, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        kp=1e8, kd=0.0, kp_rot=1e8, kd_rot=0.0, max_push=500.0, max_torque=100.0,
    )
    assert abs(np.linalg.norm(f) - 500.0) < 1e-9
    assert f[0] > 0.0 and abs(f[1]) < 1e-9 and abs(f[2]) < 1e-9


def test_compliant_driver_stalls_at_cap_two_block_smoke():
    # Two blocks: a static wall cube and a driven cube commanded straight
    # into it. The capped driver must stall at the wall (never reach the
    # target) while the applied force stays at or under the cap. This is the
    # safety property that replaces the rigid weld drive of the insertion
    # demo, which jammed at about 700x block weight.
    wall = box_2d(1.0, 1.0, 0.0, 0.5)
    mover = box_2d(1.0, 1.0, 1.2, 0.5)
    # all_pairs: the mover starts away from the wall, outside AABB adjacency.
    xml = to_mjcf([wall, mover], 0.5, timestep=1e-3, free=[1], all_pairs=True)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body = int(model.body("block1").id)
    dof = int(model.body_dofadr[body])
    weight = float(model.body_mass[body]) * 9.81
    cap = 2.0 * weight
    target_pos = np.array([0.0, 0.0, 0.5])  # inside the wall: unreachable
    target_quat = np.array([1.0, 0.0, 0.0, 0.0])
    peak = 0.0
    for _ in range(400):
        R = data.xmat[body].reshape(3, 3)
        linvel = data.qvel[dof : dof + 3].copy()
        angvel = R @ data.qvel[dof + 3 : dof + 6]
        f, t = capped_impedance_wrench(
            data.xpos[body], data.xquat[body], target_pos, target_quat,
            linvel, angvel,
            kp=6e5, kd=1.8e5, kp_rot=1.2e6, kd_rot=3e5,
            max_push=cap, max_torque=0.5 * cap,
        )
        data.xfrc_applied[body, :3] = f
        data.xfrc_applied[body, 3:] = t
        mujoco.mj_step(model, data)
        peak = max(peak, float(np.linalg.norm(f)))
    assert peak <= cap + 1e-6
    # Stalled against the wall: the mover cannot pass x = 1.0 (face contact).
    assert data.xpos[body][0] > 0.95
    # It did press up against the wall rather than hovering at the start.
    assert data.xpos[body][0] < 1.2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
