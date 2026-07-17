"""Unit tests for the Franka Panda scene composition (franka_scene.py).

mujoco is an optional extra; these tests skip cleanly when it is absent. They
stay fast: the scene compiles once, IK runs kinematically (no dynamics), and
the certification checks reuse the 2D pipeline on a 7-block scene.
"""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from keystone import Tolerances, assemble, build_assembly, solve_p0, solve_p4
from keystone.interop.franka_scene import (
    CELL_NAMES,
    CELLS,
    DESIGN_MU,
    GRASP_QUAT,
    S,
    compose_scene,
    design_cert_boxes,
    dls_ik,
    prop_cert_boxes,
    reset_home,
    staging_world,
    target_world,
)

TOL = Tolerances()


@pytest.fixture(scope="module")
def scene():
    spec, info = compose_scene()
    model = spec.compile()
    return model, info


def test_scene_compiles_with_expected_structure(scene):
    model, info = scene
    # Panda (8 actuators) plus one position servo per prop.
    names = [model.actuator(i).name for i in range(model.nu)]
    assert names[:8] == [f"actuator{i}" for i in range(1, 9)]
    assert names[8:] == [p["act"] for p in info.props]
    # Four free cubes at 0.25 kg each (0.05 m side, density 2000).
    for b in info.cube_bodies:
        bid = model.body(b).id
        assert abs(float(model.body_mass[bid]) - 0.25) < 1e-9
    # Explicit pairs only: floor (4) + pedestal (4) + cube-cube (6)
    # + prop-cube (2 * 4) + pad-cube (2 * 4).
    assert model.npair == 4 + 4 + 6 + 8 + 8
    # The two fingertip pads exist and got our names.
    for g in info.finger_pads:
        assert model.geom(g).id >= 0


def test_scene_free_bodies_start_at_staging(scene):
    model, info = scene
    data = mujoco.MjData(model)
    reset_home(model, data)
    for i, b in enumerate(info.cube_bodies):
        bid = model.body(b).id
        assert np.allclose(data.xpos[bid], info.staging_world[i], atol=1e-12)


def test_arm_reaches_key_waypoints_kinematically(scene):
    # Differential IK reaches every staging pick pose and every placement pose
    # (plus the hover waypoints) with sub-millimeter position error and the
    # top-grasp orientation. Pure kinematics: no stepping.
    model, info = scene
    scratch = mujoco.MjData(model)
    reset_home(model, scratch)
    seed = scratch.qpos.copy()
    waypoints = []
    for p in info.staging_world:
        waypoints += [p + [0.0, 0.0, 0.12], p]
    for p in info.target_world:
        waypoints += [p + [0.0, 0.0, 0.10], p + [0.0, 0.0, 0.002]]
    for w in waypoints:
        scratch.qpos[:] = seed
        _q, pos_err, rot_err = dls_ik(model, scratch, w, GRASP_QUAT)
        assert pos_err < 1e-3, (w, pos_err)
        assert rot_err < 1e-2, (w, rot_err)


def test_gripper_aperture_fits_cube(scene):
    model, info = scene
    # Full finger travel: two slide joints, 0.04 m each, 0.08 m aperture.
    aperture = 0.0
    for name in ("finger_joint1", "finger_joint2"):
        j = model.joint(name)
        aperture += float(model.jnt_range[j.id][1])
    assert aperture > info.cube_side + 0.02
    # The pinch at cube width must hold the cube: servo force at the closed
    # error times the grasp friction beats the cube weight with margin.
    grip = model.actuator("actuator8")
    kp = -float(model.actuator_biasprm[grip.id][1])
    tendon_err = (aperture - info.cube_side) / 2.0  # tendon is the average
    pinch = kp * tendon_err
    weight = 0.25 * 9.81
    assert pinch * 1.0 > 3.0 * weight  # mu_grasp = 1.0, factor-3 margin


def test_menagerie_file_untouched_on_disk():
    # Composition is in memory. The menagerie XML must not contain any of the
    # names we add (pads, props, tcp).
    from keystone.interop.franka_scene import PANDA_XML

    with open(PANDA_XML) as f:
        text = f.read()
    for token in ("left_pad", "right_pad", "tcp", "prop0", "cube0"):
        assert token not in text


def test_scaled_design_recertifies_like_unit_scale():
    # The scale trick: uniform scaling to cube side 0.05 m leaves the
    # dimensionless P4 margin unchanged to float precision.
    def certify(boxes):
        a = build_assembly(boxes, mu=DESIGN_MU, tol=TOL, dim=2)
        s = assemble(a, TOL, cone="linear2d")
        return solve_p0(s, TOL).status, float(solve_p4(s, TOL).margin)

    st_u, m_u = certify(design_cert_boxes(1.0) + prop_cert_boxes(1.0))
    st_s, m_s = certify(design_cert_boxes(S) + prop_cert_boxes(S))
    assert st_u == st_s == "feasible"
    assert abs(m_u - m_s) < 1e-15


def test_targets_reproduce_certified_cells():
    # World targets are the certified lattice cells scaled and offset.
    tw = target_world()
    sw = staging_world()
    assert tw.shape == (len(CELLS), 3)
    assert sw.shape == (len(CELLS), 3)
    assert len(CELL_NAMES) == len(CELLS)
    for (layer, j), t in zip(CELLS, tw):
        assert abs(t[2] - (1.5 + layer) * S) < 1e-12
        assert abs(t[1]) < 1e-12  # planar in xz: y faces free for the pinch
    # Staging cubes rest on the floor.
    assert np.allclose(sw[:, 2], S / 2.0, atol=1e-12)


def test_resolve_panda_xml_default(monkeypatch):
    # No env var, no argument: the bundled refs/ default, which exists here.
    from keystone.interop.franka_scene import PANDA_XML, resolve_panda_xml

    monkeypatch.delenv("KEYSTONE_MENAGERIE", raising=False)
    assert resolve_panda_xml() == PANDA_XML


def test_resolve_panda_xml_env_var_wins(monkeypatch, tmp_path):
    # The env var takes precedence, over both the argument and the refs/ default.
    from keystone.interop import franka_scene as fs

    root = tmp_path / "menagerie"
    (root / "franka_emika_panda").mkdir(parents=True)
    panda = root / "franka_emika_panda" / "panda.xml"
    panda.write_text("<mujoco/>")
    monkeypatch.setenv("KEYSTONE_MENAGERIE", str(root))
    assert fs.resolve_panda_xml() == str(panda)
    # Env var beats an explicit argument.
    assert fs.resolve_panda_xml(menagerie="/no/such/root") == str(panda)


def test_resolve_panda_xml_argument(monkeypatch, tmp_path):
    # With no env var the argument is used, over the refs/ default.
    from keystone.interop import franka_scene as fs

    monkeypatch.delenv("KEYSTONE_MENAGERIE", raising=False)
    root = tmp_path / "arg_menagerie"
    (root / "franka_emika_panda").mkdir(parents=True)
    panda = root / "franka_emika_panda" / "panda.xml"
    panda.write_text("<mujoco/>")
    assert fs.resolve_panda_xml(menagerie=str(root)) == str(panda)


def test_resolve_panda_xml_missing_raises_listing_options(monkeypatch, tmp_path):
    # A missing file raises with the source and all three options listed.
    from keystone.interop import franka_scene as fs

    monkeypatch.setenv("KEYSTONE_MENAGERIE", str(tmp_path / "does_not_exist"))
    with pytest.raises(FileNotFoundError, match="KEYSTONE_MENAGERIE"):
        fs.resolve_panda_xml()
    monkeypatch.delenv("KEYSTONE_MENAGERIE", raising=False)
    try:
        fs.resolve_panda_xml(menagerie=str(tmp_path / "nope"))
    except FileNotFoundError as e:
        msg = str(e)
        assert "KEYSTONE_MENAGERIE" in msg
        assert "menagerie" in msg
        assert "refs/" in msg
    else:
        raise AssertionError("expected FileNotFoundError")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
