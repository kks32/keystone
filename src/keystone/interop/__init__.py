"""MuJoCo interop for keystone.

mujoco is an optional extra (install `keystone[mujoco]`). Every function in
`mujoco_io` imports mujoco lazily inside its body, so the core package works
without it. Import this subpackage freely; a missing mujoco surfaces only when
a bridge function is called.

Scope (PLAN.md Section 8.1, 8.2): scene exchange (to_mjcf/from_mjcf) and the
settle-test dynamic sanity oracle. MuJoCo is a soft-constraint, regularized
contact model. It is not ground truth for limit analysis, and the associative
verdict is an upper estimate by construction (PLAN.md Section 8.4). The bridge
measures and reports gaps; it never tunes keystone to match MuJoCo.
"""

from .franka_scene import (
    SceneInfo,
    compose_scene,
    design_cert_boxes,
    dls_ik,
    prop_cert_boxes,
    reset_home,
    resolve_panda_xml,
    staging_world,
    target_world,
)
from .mujoco_io import (
    aabb_adjacent_pairs,
    assembly_diagonal,
    capped_impedance_wrench,
    from_mjcf,
    orientation_error,
    restacked_cubes,
    settle_test,
    split_reacher,
    to_mjcf,
)

__all__ = [
    "SceneInfo",
    "aabb_adjacent_pairs",
    "assembly_diagonal",
    "capped_impedance_wrench",
    "compose_scene",
    "design_cert_boxes",
    "dls_ik",
    "from_mjcf",
    "orientation_error",
    "prop_cert_boxes",
    "reset_home",
    "resolve_panda_xml",
    "restacked_cubes",
    "settle_test",
    "split_reacher",
    "staging_world",
    "target_world",
    "to_mjcf",
]
