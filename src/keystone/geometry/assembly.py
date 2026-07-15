"""The padded Assembly structure and its builder.

This is the single data contract between geometry, mechanics, and the
solvers. All arrays are padded to static shapes with boolean masks so
the downstream code can jit and vmap.

Conventions (CLAUDE.md Section 4):
- Node 0 is the ground. Blocks are nodes 1..N. block index in the
  arrays below is node_id - 1.
- A patch joins nodes (i, j) with i < j. n_hat points from i into j.
- Patch vertices are ordered counterclockwise about n_hat. A 2D patch
  is a segment with two vertices.
- 2D frames: t1_hat = normalize(cross(y_hat, n_hat)), t2_hat is zero.
  This is the 2D degeneration of the t1 rule in CLAUDE.md; the patch
  plane of a segment contains global y, and t1 must be in the xz plane.
- 3D frames: t1_hat is the normalized projection of global x onto the
  patch plane unless abs(n_hat . x) > 0.9, in which case project
  global y. t2_hat = cross(n_hat, t1_hat).

Determinism: blocks keep list order (ids 1..N). Patches are sorted by
(i, j, centroid x, centroid y, centroid z) before padding.
"""

from dataclasses import dataclass
from typing import Sequence

import jax
import numpy as np

from .boxes import Box
from .interfaces import detect_patches_2d, detect_patches_3d
from .tolerances import Tolerances


@dataclass(frozen=True)
class Assembly:
    """Padded rigid-block assembly.

    Shapes: N padded blocks, P padded patches, V padded vertices per
    patch (V = 2 in 2D, V = 8 in 3D).

    block_mask:   (N,) bool, True for real blocks.
    mass:         (N,) kg. Zero on masked entries.
    com:          (N, 3) m, world center of mass. Zero on masked entries.
    patch_mask:   (P,) bool.
    patch_blocks: (P, 2) int32, node ids (i, j) with i < j, 0 = ground.
                  (0, 0) on masked entries.
    normal:       (P, 3) unit n_hat, from i into j.
    t1:           (P, 3) unit tangent.
    t2:           (P, 3) unit tangent (zero rows in 2D).
    verts:        (P, V, 3) m, world vertex coordinates.
    vert_mask:    (P, V) bool.
    mu:           (P,) friction coefficient per patch.
    dim:          2 or 3, static metadata.
    """

    block_mask: np.ndarray
    mass: np.ndarray
    com: np.ndarray
    patch_mask: np.ndarray
    patch_blocks: np.ndarray
    normal: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    verts: np.ndarray
    vert_mask: np.ndarray
    mu: np.ndarray
    dim: int

    @property
    def n_blocks(self) -> int:
        return int(self.block_mask.shape[0])

    @property
    def n_patches(self) -> int:
        return int(self.patch_mask.shape[0])

    @property
    def verts_per_patch(self) -> int:
        return int(self.verts.shape[1])


jax.tree_util.register_dataclass(
    Assembly,
    data_fields=[
        "block_mask",
        "mass",
        "com",
        "patch_mask",
        "patch_blocks",
        "normal",
        "t1",
        "t2",
        "verts",
        "vert_mask",
        "mu",
    ],
    meta_fields=["dim"],
)


def bbox_diagonal(assembly: Assembly) -> float:
    """L: diagonal of the bounding box of all active patch vertices and coms."""
    pts = [assembly.com[assembly.block_mask]]
    vm = assembly.vert_mask
    pts.append(assembly.verts[vm])
    pts = np.concatenate(pts, axis=0)
    if pts.shape[0] == 0:
        raise ValueError("empty assembly: no active blocks and no patch vertices")
    diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    # Representability floor, not a physics tolerance. A single floating
    # block with no patches collapses to one point and gives diag = 0,
    # which would make every L-scaled quantity blow up downstream.
    if diag < 1e2 * np.finfo(np.float64).tiny:
        raise ValueError(
            "degenerate length scale: assembly has no spatial extent "
            "(single floating block with no patches). Add the ground or a "
            "second block so the bounding box has nonzero size."
        )
    return diag


def build_assembly(
    boxes: Sequence[Box],
    *,
    mu: float,
    tol: Tolerances,
    dim: int,
    ground: bool = True,
    pad_blocks: int | None = None,
    pad_patches: int | None = None,
    pad_verts: int | None = None,
) -> Assembly:
    """Detect interfaces between boxes (and the ground plane z = 0) and
    build the padded Assembly. Host-mode numpy; deterministic.

    pad_* of None means no padding beyond the detected counts.
    """
    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {dim}")
    boxes = list(boxes)
    n_real = len(boxes)
    if n_real == 0:
        raise ValueError("build_assembly needs at least one box")

    # L for tolerance scaling: bounding box diagonal of all block corners.
    # In 2D the assembly lives in the xz plane and the out-of-plane depth
    # along y is arbitrary (box_2d defaults to 1 m). Zero the y column so
    # the detection length scale, and therefore g_tol scaling, does not
    # move when the depth changes.
    all_corners = np.concatenate([b.corners() for b in boxes], axis=0)
    if dim == 2:
        all_corners = all_corners.copy()
        all_corners[:, 1] = 0.0
    L = float(np.linalg.norm(all_corners.max(axis=0) - all_corners.min(axis=0)))

    if dim == 2:
        records = detect_patches_2d(boxes, ground, L, tol)
        v_default = 2
    else:
        records = detect_patches_3d(boxes, ground, L, tol)
        v_default = 8

    p_real = len(records)
    max_v = max((rec[4].shape[0] for rec in records), default=0)

    n_pad = n_real if pad_blocks is None else pad_blocks
    p_pad = p_real if pad_patches is None else pad_patches
    v_pad = v_default if pad_verts is None else pad_verts
    if n_pad < n_real:
        raise ValueError("pad_blocks smaller than detected block count")
    if p_pad < p_real:
        raise ValueError("pad_patches smaller than detected patch count")
    if v_pad < max_v:
        raise ValueError("pad_verts smaller than detected vertex count")

    block_mask = np.zeros(n_pad, dtype=bool)
    block_mask[:n_real] = True
    mass = np.zeros(n_pad, dtype=np.float64)
    com = np.zeros((n_pad, 3), dtype=np.float64)
    for bi, box in enumerate(boxes):
        mass[bi] = box.mass
        com[bi] = box.com

    patch_mask = np.zeros(p_pad, dtype=bool)
    patch_mask[:p_real] = True
    patch_blocks = np.zeros((p_pad, 2), dtype=np.int32)
    normal = np.zeros((p_pad, 3), dtype=np.float64)
    t1 = np.zeros((p_pad, 3), dtype=np.float64)
    t2 = np.zeros((p_pad, 3), dtype=np.float64)
    verts = np.zeros((p_pad, v_pad, 3), dtype=np.float64)
    vert_mask = np.zeros((p_pad, v_pad), dtype=bool)
    mu_arr = np.zeros(p_pad, dtype=np.float64)

    for p, rec in enumerate(records):
        i, j, n_hat, tan1, pverts = rec
        patch_blocks[p] = (i, j)
        normal[p] = n_hat
        t1[p] = tan1
        if dim == 3:
            t2[p] = np.cross(n_hat, tan1)
        nv = pverts.shape[0]
        verts[p, :nv] = pverts
        vert_mask[p, :nv] = True
        mu_arr[p] = mu

    return Assembly(
        block_mask=block_mask,
        mass=mass,
        com=com,
        patch_mask=patch_mask,
        patch_blocks=patch_blocks,
        normal=normal,
        t1=t1,
        t2=t2,
        verts=verts,
        vert_mask=vert_mask,
        mu=mu_arr,
        dim=dim,
    )
