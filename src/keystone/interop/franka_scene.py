"""Scene composition for the Franka Panda falsework build (examples/franka_build.py).

The robotic-arm demo builds the certified clamp 29/24 falsework design (the
counterweighted reacher of tests/analytic/test_bnb_optima.py, backed off two
grid steps so it stands under MuJoCo's compliant contacts; see
examples/mujoco_falsework.py) with a real two-finger gripper instead of the
invisible-hand impedance driver.

Scale trick. keystone verdicts and dimensionless margins are scale-invariant
(property-tested in tests/property/test_invariants.py: test_scale_invariance_2d
holds lambda* and mu* fixed under uniform scaling). The unit-cube lattice is
therefore rescaled to a table-top size the Panda can lift and reach: cube side
s = 0.05 m at density 2000 gives 0.25 kg per cube, inside the Panda payload, and
the finished structure fits a 0.30 m pedestal beside the arm. The scaled design
is re-certified through the host pipeline (see examples/franka_build.py); its
dimensionless margins match the unit-scale ones to float precision.

This module composes the scene in memory with MjSpec: it loads the unmodified
menagerie panda.xml (with hand), keeps its meshes, and adds a ground plane, a
static pedestal, the design cubes as free bodies in a staging row, the two
falsework props on position-actuated sliders, finger-cube grasp pairs, and a
TCP site for differential inverse kinematics. The menagerie file on disk is
never edited.

Conventions kept from interop.mujoco_io: contacts are explicit <pair> elements
with condim 3 and our friction; structure and floor pairs carry mu = 0.7, the
finger-cube grasp pairs carry a higher mu (1.0) so the pinch holds a 0.25 kg
cube. keystone box geoms stay contype = conaffinity = 0, so only the declared
pairs generate contact; the panda's own collision geoms keep the menagerie
defaults. Gravity is MuJoCo's default (0, 0, -9.81).

Menagerie location. compose_scene loads panda.xml from a mujoco_menagerie root
resolved in this precedence order: the KEYSTONE_MENAGERIE environment variable,
then the compose_scene `menagerie` argument, then the bundled refs/ default
(refs/mujoco_menagerie beside the repo root). panda.xml is expected at
<root>/franka_emika_panda/panda.xml. If it is not found, resolve_panda_xml
raises with the searched path and all three options listed. The menagerie file
on disk is never edited.

mujoco is imported lazily. The core package never needs it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from ..geometry.boxes import Box, box_2d

# Menagerie root and the panda.xml default (kept read-only on disk).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
_REFS_MENAGERIE = os.path.join(_REPO_ROOT, "refs", "mujoco_menagerie")
# The bundled default path. resolve_panda_xml applies the env var and argument
# overrides; this constant is the refs/ fallback.
PANDA_XML = os.path.join(_REFS_MENAGERIE, "franka_emika_panda", "panda.xml")


def resolve_panda_xml(menagerie: str | None = None) -> str:
    """Resolve the menagerie panda.xml path, checking the options in order.

    Precedence: the KEYSTONE_MENAGERIE environment variable, then the
    `menagerie` argument, then the bundled refs/ default. Each names a
    mujoco_menagerie root; panda.xml sits at
    <root>/franka_emika_panda/panda.xml. Raise a clear error listing every
    option when the file is absent.
    """
    env_root = os.environ.get("KEYSTONE_MENAGERIE")
    if env_root:
        root, source = env_root, "KEYSTONE_MENAGERIE"
    elif menagerie is not None:
        root, source = str(menagerie), "menagerie argument"
    else:
        root, source = _REFS_MENAGERIE, "refs/ default"
    panda = os.path.join(root, "franka_emika_panda", "panda.xml")
    if not os.path.isfile(panda):
        raise FileNotFoundError(
            f"menagerie panda.xml not found at {panda!r} (from {source}). "
            "Provide a mujoco_menagerie root by one of, in precedence order: "
            "set the KEYSTONE_MENAGERIE environment variable, pass "
            "compose_scene(menagerie=...), or place the repository under refs/ "
            f"(expected {_REFS_MENAGERIE!r}). Clone it from "
            "https://github.com/google-deepmind/mujoco_menagerie."
        )
    return panda

# Physical scale. Cube side 0.05 m, density 2000 -> 0.25 kg per cube.
S = 0.05
DENSITY = 2000.0
MU_STRUCT = 0.7
MU_GRASP = 1.0

# Keystone origin placed 0.45 m in +x from the panda base (at the world origin),
# so the structure sits beside the arm within a comfortable top-down reach.
BASE_OFFSET = np.array([0.45, 0.0, 0.0])

# clamp 29/24 build plan, read from examples/mujoco_falsework.py PLANS.
# cells in placement order (layer, grid_index_j); center x is j * DX_GRID.
DX_GRID = 1.0 / 24.0
CELLS = [(0, -2), (1, 17), (1, -14), (2, -4)]
CELL_NAMES = ["base", "reacher", "counterweight", "bridge"]
DESIGN_MU = 0.7
OVERHANG = 29.0 / 24.0

# Falsework props (examples/mujoco_falsework.py, clamp_29_24), in keystone units.
# supports: index into CELLS of the block whose underside the prop top meets.
# axis "x" slides the prop out horizontally, "z" lowers it like a jack.
PROP_W = 0.2
PROPS_UNIT = [
    dict(name="cw", x=-0.85, half_h=0.5, z=1.5, axis="x",
         retract_disp=-1.45, supports=2),
    dict(name="reacher", x=1.0, half_h=1.0, z=1.0, axis="z",
         retract_disp=-1.4, supports=1),
]

# Top-down grasp: tool z points to world -z, finger slide axis (hand local y)
# points along world +y so the fingers straddle the cube across y. The designs
# are planar in xz, so the y faces are always clear (see docs/KNOWN_LIMITS.md).
# Quaternion is a 180 deg rotation about world y, (w, x, y, z).
GRASP_QUAT = np.array([0.0, 0.0, 1.0, 0.0])

# TCP site offset in the hand frame: the grasp point between the fingertip pads.
TCP_OFFSET = np.array([0.0, 0.0, 0.10])
TCP_SITE = "tcp"

# Staging row: the design cubes wait on the floor, spaced along x at a fixed y
# offset from the structure plane, in build order.
STAGING_X = np.array([0.30, 0.40, 0.50, 0.60])
STAGING_Y = -0.20

# Gripper actuator (menagerie actuator8) ctrl: 255 opens (0.08 m aperture),
# 0 closes. A 0.05 m cube fits the open gripper with 0.03 m of margin.
GRIPPER_OPEN = 255.0
GRIPPER_CLOSE = 0.0

# Arm home configuration (menagerie "home" keyframe arm angles). The compiled
# scene has extra free joints, so the keyframe itself is not used.
HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["finger_joint1", "finger_joint2"]

# Prop slider prop tops are set a hair below the supported cube underside so the
# placed cube rests on its structural neighbor and tips onto the prop by a few
# tenths of a millimeter, matching the falsework catcher model.
PROP_RELIEF = 3e-4


@dataclass
class SceneInfo:
    """Everything the build driver needs to address the composed model."""

    scale: float
    base_offset: np.ndarray
    grasp_quat: np.ndarray
    tcp_site: str
    cube_bodies: list
    cube_geoms: list
    cube_names: list
    staging_world: np.ndarray            # (N, 3) staging cube centers, world
    target_world: np.ndarray             # (N, 3) target cube centers, world
    finger_pads: tuple
    props: list                          # scaled prop metadata dicts
    floor: str
    pedestal_body: str
    cube_side: float
    arm_actuators: list = field(default_factory=lambda: [
        f"actuator{i}" for i in range(1, 8)])
    gripper_actuator: str = "actuator8"
    cells: list = field(default_factory=lambda: list(CELLS))
    dx: float = DX_GRID


# --------------------------------------------------------------------------
# Design geometry for host-pipeline certification (keystone coordinates).
# Uniform scaling of the unit lattice, so scale = 1.0 reproduces the archived
# clamp 29/24 boxes exactly and scale = 0.05 is a pure uniform scale of them.
# Depth is set to the scale too, so the comparison is a clean uniform scaling.
# --------------------------------------------------------------------------


def pedestal_cert(scale: float = S) -> Box:
    """Pedestal for certification. Right edge at keystone x = 0, top at z."""
    return box_2d(6.0 * scale, 1.0 * scale, -3.0 * scale, 0.5 * scale,
                  density=DENSITY, depth=1.0 * scale)


def cube_cert(layer: int, j: int, scale: float = S) -> Box:
    """A design cube for certification at (layer, grid index j)."""
    return box_2d(1.0 * scale, 1.0 * scale, j * DX_GRID * scale,
                  (1.5 + layer) * scale, density=DENSITY, depth=1.0 * scale)


def design_cert_boxes(scale: float = S) -> list:
    """Pedestal plus the four design cubes, keystone coordinates."""
    return [pedestal_cert(scale)] + [
        cube_cert(layer, j, scale) for (layer, j) in CELLS]


def prop_cert_boxes(scale: float = S) -> list:
    """The two props as ordinary blocks for propped-prefix certification."""
    return [
        box_2d(PROP_W * scale, 2.0 * p["half_h"] * scale, p["x"] * scale,
               p["z"] * scale, density=DENSITY, depth=1.0 * scale)
        for p in PROPS_UNIT
    ]


# --------------------------------------------------------------------------
# World placement (scaled keystone coordinates plus the base offset).
# --------------------------------------------------------------------------


def target_world(scale: float = S, base_offset: np.ndarray = BASE_OFFSET,
                 cells=None, dx: float = DX_GRID) -> np.ndarray:
    """World cube-center targets for the finished structure, build order.

    cells is a list of (layer, grid_index_j); None uses the archived CELLS.
    dx is the grid step in keystone units (DX_GRID = 1/24 by default). The two
    defaults reproduce the clamp 29/24 targets exactly."""
    cells = CELLS if cells is None else list(cells)
    out = []
    for (layer, j) in cells:
        out.append(np.array([j * dx * scale, 0.0, (1.5 + layer) * scale])
                   + base_offset)
    return np.array(out)


def staging_world(scale: float = S, cells=None, staging_x=None,
                  staging_y: float = STAGING_Y) -> np.ndarray:
    """World cube-center poses in the staging row (on the floor).

    cells sets the cube count; None uses CELLS. staging_x is the per-cube x row
    (world meters); None uses the archived STAGING_X. Defaults reproduce the
    clamp 29/24 staging exactly."""
    cells = CELLS if cells is None else list(cells)
    xs = STAGING_X if staging_x is None else np.asarray(staging_x, dtype=np.float64)
    half = 0.5 * scale
    return np.array([[xs[i], staging_y, half] for i in range(len(cells))])


def scaled_props(scale: float = S, base_offset: np.ndarray = BASE_OFFSET,
                 relief: float = PROP_RELIEF, prop_specs=None, cells=None,
                 dx: float = DX_GRID) -> list:
    """Prop metadata in world coordinates, tops set from the cube targets.

    prop_specs is a list of unit-scale prop dicts (PROPS_UNIT format: name, x,
    half_h, z, axis, retract_disp, supports); None uses PROPS_UNIT, and [] means
    no props. supports indexes into cells. Defaults reproduce clamp 29/24."""
    prop_specs = PROPS_UNIT if prop_specs is None else list(prop_specs)
    tgt = target_world(scale, base_offset, cells=cells, dx=dx)
    props = []
    for k, p in enumerate(prop_specs):
        sup = p["supports"]
        cube_bottom = float(tgt[sup][2] - 0.5 * scale)  # underside of the cube
        half_h = p["half_h"] * scale
        top = cube_bottom - relief
        zc = top - half_h
        props.append(dict(
            name=p["name"],
            body=f"prop{k}",
            geom=f"prop{k}_geom",
            joint=f"prop{k}_slide",
            act=f"prop{k}_act",
            axis=p["axis"],
            supports=sup,
            x=float(p["x"] * scale + base_offset[0]),
            zc=float(zc),
            half_h=float(half_h),
            half_w=float(PROP_W * scale / 2.0),
            half_d=float(0.5 * scale),
            retract_disp=float(p["retract_disp"] * scale),
        ))
    return props


# --------------------------------------------------------------------------
# MjSpec composition.
# --------------------------------------------------------------------------


def _friction(mu: float) -> list:
    return [mu, mu, 0.005, 0.0001, 0.0001]


def compose_scene(
    *,
    scale: float = S,
    base_offset: np.ndarray = BASE_OFFSET,
    mu_struct: float = MU_STRUCT,
    mu_grasp: float = MU_GRASP,
    timestep: float = 1e-3,
    struct_solref: tuple = (0.002, 1.0),
    grasp_solref: tuple = (0.005, 1.0),
    impratio: float = 10.0,
    relief: float = PROP_RELIEF,
    prop_kp: float = 1.0e5,
    prop_kv: float = 1.0e3,
    grip_kp: float = 1000.0,
    grip_kv: float = 30.0,
    arm_base_pos: np.ndarray | None = None,
    arm_base_yaw: float = 0.0,
    cells=None,
    cell_names=None,
    dx: float = DX_GRID,
    prop_specs=None,
    staging=None,
    menagerie: str | None = None,
):
    """Compose the full build scene in memory. Returns (spec, SceneInfo).

    The panda (with hand) loads from the unmodified menagerie file and keeps its
    meshes. Added: a ground plane, a static pedestal, the design cubes as free
    bodies in the staging row, the two falsework props on position-actuated
    sliders, a TCP site on the hand, and the explicit contact pairs. Compile the
    returned spec with spec.compile().

    arm_base_pos, arm_base_yaw place the panda base body (link0) beside the
    structure for a side approach. Default (None, 0.0) keeps link0 at the world
    origin, reproducing the top-down front build of examples/franka_build.py. A
    non-None arm_base_pos moves the base to that world point; arm_base_yaw
    rotates it about world z (radians) so the arm faces the structure. The
    structure, staging, and props are unaffected; only the arm is repositioned.

    cells, cell_names, dx, prop_specs, staging generalize the structure to an
    arbitrary build plan (the pipeline Franka executor). Defaults (all None, dx
    = DX_GRID) reproduce the archived clamp 29/24 scene bit for bit. cells is a
    list of (layer, grid_index_j) in build order; cell_names names them; dx is
    the grid step in keystone units; prop_specs is a list of unit-scale prop
    dicts ([] means no props); staging is an (N, 3) world staging row (None uses
    the default staging_world).

    menagerie sets the mujoco_menagerie root for panda.xml. None (default)
    resolves through resolve_panda_xml: KEYSTONE_MENAGERIE, then this argument,
    then the refs/ default."""
    import mujoco

    base_offset = np.asarray(base_offset, dtype=np.float64)
    cells = list(CELLS) if cells is None else list(cells)
    cell_names = list(CELL_NAMES) if cell_names is None else list(cell_names)
    prop_specs = list(PROPS_UNIT) if prop_specs is None else list(prop_specs)
    spec = mujoco.MjSpec.from_file(resolve_panda_xml(menagerie))
    if arm_base_pos is not None:
        link0 = spec.body("link0")
        link0.pos = np.asarray(arm_base_pos, dtype=np.float64).tolist()
        a = 0.5 * float(arm_base_yaw)
        link0.quat = [np.cos(a), 0.0, 0.0, np.sin(a)]
    spec.option.timestep = timestep
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    # HD offscreen framebuffer with multisampling for movie rendering.
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.visual.quality.offsamples = 8
    # Elliptic cones with a high impratio: MuJoCo's regularized pyramidal
    # friction lets a pinched cube creep tangentially under gravity (measured
    # 2.0 mm/s at a 23 N pinch, mu 1.0; the cube slid 9 mm down the fingers
    # during one transport). Elliptic + impratio 10 cuts the creep to
    # 0.05 mm/s. This is the documented MuJoCo grasping recipe, applied to the
    # whole scene, structural contacts included.
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = impratio

    wb = spec.worldbody

    # Ground plane, keystone convention (explicit pairs only).
    floor = wb.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [5.0, 5.0, 0.1]
    floor.contype = 0
    floor.conaffinity = 0
    floor.rgba = [0.7, 0.7, 0.72, 1.0]

    # A fill light so the offscreen render is not dark.
    lt = wb.add_light()
    lt.pos = [0.4, -0.3, 1.2]
    lt.dir = [-0.2, 0.2, -1.0]

    half = 0.5 * scale

    # Static pedestal. 0.30 x 0.30 x 0.05 m: width and depth 6 * scale, height
    # 1 * scale. The wide y depth is a physical footprint and does not enter the
    # 2D certification (which zeroes the out-of-plane axis).
    ped_center = np.array([-3.0 * scale, 0.0, 0.5 * scale]) + base_offset
    pedestal = wb.add_body()
    pedestal.name = "pedestal"
    pedestal.pos = ped_center.tolist()
    pg = pedestal.add_geom()
    pg.name = "pedestal_geom"
    pg.type = mujoco.mjtGeom.mjGEOM_BOX
    pg.size = [3.0 * scale, 3.0 * scale, 0.5 * scale]
    pg.density = DENSITY
    pg.contype = 0
    pg.conaffinity = 0
    pg.rgba = [0.55, 0.57, 0.60, 1.0]

    # Design cubes as free bodies, starting in the staging row.
    if staging is None:
        staging = staging_world(scale, cells=cells)
    else:
        staging = np.asarray(staging, dtype=np.float64)
    cube_bodies, cube_geoms = [], []
    for i, name in enumerate(cell_names):
        b = wb.add_body()
        b.name = f"cube{i}"
        b.pos = staging[i].tolist()
        b.add_freejoint()
        g = b.add_geom()
        g.name = f"cube{i}_geom"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [half, half, half]
        g.density = DENSITY
        g.contype = 0
        g.conaffinity = 0
        # Distinct colors so the render reads the build order.
        hue = 0.2 + 0.6 * i / max(1, len(cell_names) - 1)
        g.rgba = [hue, 0.45, 1.0 - hue, 1.0]
        cube_bodies.append(b.name)
        cube_geoms.append(g.name)

    # Falsework props on sliders with position actuators.
    props = scaled_props(scale, base_offset, relief, prop_specs=prop_specs,
                         cells=cells, dx=dx)
    axis_vec = {"x": [1.0, 0.0, 0.0], "z": [0.0, 0.0, 1.0]}
    for p in props:
        pb = wb.add_body()
        pb.name = p["body"]
        pb.pos = [p["x"], base_offset[1], p["zc"]]
        j = pb.add_joint()
        j.name = p["joint"]
        j.type = mujoco.mjtJoint.mjJNT_SLIDE
        j.axis = axis_vec[p["axis"]]
        j.range = [-4.0 * scale, 2.0 * scale]
        g = pb.add_geom()
        g.name = p["geom"]
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [p["half_w"], p["half_d"], p["half_h"]]
        g.density = DENSITY
        g.contype = 0
        g.conaffinity = 0
        g.rgba = [0.85, 0.75, 0.2, 1.0]
        act = spec.add_actuator()
        act.name = p["act"]
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.target = p["joint"]
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.gainprm[0] = prop_kp
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.biasprm[0] = 0.0
        act.biasprm[1] = -prop_kp
        act.biasprm[2] = -prop_kv
        act.ctrlrange = [-4.0 * scale, 2.0 * scale]

    # TCP site on the hand for differential IK.
    hand = spec.body("hand")
    site = hand.add_site()
    site.name = TCP_SITE
    site.pos = TCP_OFFSET.tolist()
    site.quat = [1.0, 0.0, 0.0, 0.0]

    # Name the fingertip pad boxes (menagerie leaves them unnamed) so we can pair
    # them with the cubes. geoms[3] is the large inner pad on each finger.
    left_pad = spec.body("left_finger").geoms[3]
    left_pad.name = "left_pad"
    right_pad = spec.body("right_finger").geoms[3]
    right_pad.name = "right_pad"
    finger_pads = ("left_pad", "right_pad")

    # Stiffen the gripper tendon servo in memory. The menagerie actuator8 is a
    # position servo with kp = 100 N/m on the split tendon; pinching a 0.05 m
    # cube leaves a 0.028 m tendon error, so the pinch saturates at 2.8 N,
    # under the 2.45 N cube weight times mu = 1.0 (measured: the cube slips out
    # during the lift). grip_kp = 1000 N/m gives a 28 N pinch on this cube,
    # inside the real Panda's 70 N grasp-force spec. ctrl semantics are kept:
    # ctrl in [0, 255] maps to a tendon length target in [0, 0.04] m.
    grip = spec.actuator("actuator8")
    grip.gainprm[0] = grip_kp * 0.04 / 255.0
    grip.biasprm[1] = -grip_kp
    grip.biasprm[2] = -grip_kv

    # Explicit contact pairs (keystone convention).
    def add_pair(g1, g2, mu, solref):
        pr = spec.add_pair()
        pr.geomname1 = g1
        pr.geomname2 = g2
        pr.condim = 3
        pr.friction = _friction(mu)
        pr.solref = list(solref)
        return pr

    n = len(cube_geoms)
    # floor and pedestal support every cube.
    for i in range(n):
        add_pair("floor", cube_geoms[i], mu_struct, struct_solref)
        add_pair("pedestal_geom", cube_geoms[i], mu_struct, struct_solref)
    # cube-cube.
    for i in range(n):
        for k in range(i + 1, n):
            add_pair(cube_geoms[i], cube_geoms[k], mu_struct, struct_solref)
    # props catch every cube.
    for p in props:
        for i in range(n):
            add_pair(p["geom"], cube_geoms[i], mu_struct, struct_solref)
    # finger pads grip every cube.
    for pad in finger_pads:
        for i in range(n):
            add_pair(pad, cube_geoms[i], mu_grasp, grasp_solref)

    info = SceneInfo(
        scale=scale,
        base_offset=base_offset,
        grasp_quat=GRASP_QUAT.copy(),
        tcp_site=TCP_SITE,
        cube_bodies=cube_bodies,
        cube_geoms=cube_geoms,
        cube_names=list(cell_names),
        staging_world=staging,
        target_world=target_world(scale, base_offset, cells=cells, dx=dx),
        finger_pads=finger_pads,
        props=props,
        floor="floor",
        pedestal_body="pedestal",
        cube_side=scale,
        cells=list(cells),
        dx=dx,
    )
    return spec, info


# --------------------------------------------------------------------------
# Kinematics helpers shared by the build driver and the unit tests.
# --------------------------------------------------------------------------


def reset_home(model, data) -> None:
    """Reset the scene: arm at HOME_Q, gripper open, free bodies at their
    body-frame poses (staging row). The menagerie 'home' keyframe is not used
    because the composed model has extra free joints the keyframe zeroes."""
    import mujoco

    mujoco.mj_resetData(model, data)
    for k, name in enumerate(ARM_JOINTS):
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = HOME_Q[k]
    for name in FINGER_JOINTS:
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = 0.04
    data.ctrl[:7] = HOME_Q
    data.ctrl[7] = GRIPPER_OPEN
    mujoco.mj_forward(model, data)


def dls_ik(
    model,
    scratch,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    *,
    site: str = TCP_SITE,
    iters: int = 200,
    damping: float = 0.08,
    step: float = 0.6,
    pos_tol: float = 5e-5,
    rot_tol: float = 5e-4,
):
    """Damped-least-squares differential IK on the TCP site jacobian.

    scratch: an MjData used as workspace; seed its qpos before calling (the
    current sim qpos gives warm-started, continuous solutions). Only the seven
    arm joints move. Returns (q_arm, pos_err, rot_err)."""
    import mujoco

    from .mujoco_io import orientation_error

    target_pos = np.asarray(target_pos, dtype=np.float64)
    sid = model.site(site).id
    qadr = [model.jnt_qposadr[model.joint(n).id] for n in ARM_JOINTS]
    dadr = [model.jnt_dofadr[model.joint(n).id] for n in ARM_JOINTS]
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    quat = np.zeros(4)
    scratch.qvel[:] = 0.0
    for _ in range(iters):
        mujoco.mj_forward(model, scratch)
        pos = scratch.site_xpos[sid]
        mujoco.mju_mat2Quat(quat, scratch.site_xmat[sid])
        ep = target_pos - pos
        er = orientation_error(quat, target_quat)
        if np.linalg.norm(ep) < pos_tol and np.linalg.norm(er) < rot_tol:
            break
        mujoco.mj_jacSite(model, scratch, jacp, jacr, sid)
        J = np.vstack([jacp[:, dadr], jacr[:, dadr]])
        err = np.concatenate([ep, er])
        dq = J.T @ np.linalg.solve(J @ J.T + damping * damping * np.eye(6), err)
        for k in range(7):
            lo, hi = model.jnt_range[model.joint(ARM_JOINTS[k]).id]
            scratch.qpos[qadr[k]] = float(
                np.clip(scratch.qpos[qadr[k]] + step * dq[k], lo, hi)
            )
    mujoco.mj_forward(model, scratch)
    pos = scratch.site_xpos[sid]
    mujoco.mju_mat2Quat(quat, scratch.site_xmat[sid])
    q_arm = np.array([scratch.qpos[a] for a in qadr])
    pos_err = float(np.linalg.norm(target_pos - pos))
    rot_err = float(np.linalg.norm(orientation_error(quat, target_quat)))
    return q_arm, pos_err, rot_err
