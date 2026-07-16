"""Jittable 3D lattice environment for cube-stacking search.

This is the 3D-native twin of keystone.search.lattice. Unit cubes sit on
a (layer, ix, iy) grid over a rectangular pedestal. The whole
geometry-to-system path is one pure function of the state, so a
placement, an equilibrium system, and a P4 margin all run inside a
single jitted, vmapped kernel. Fixed padded shapes keep the kernel
device agnostic: the same code runs on CPU now and on a GPU later.

Contacts on this lattice are axis-aligned rectangle overlaps, so no
clipping is needed. A cube's bottom face meets the pedestal top or a
layer-below cube in a rectangle with +z normal; t1 = +x and t2 = +y
follow the frame rule for a +z normal (n_hat . x = 0, so t1 is the
projection of global x, which is +x). Every support patch is that
overlap rectangle with four vertices ordered counterclockwise about +z.
The cone model is the inscribed pyramid with k = 8 facets, matching
mechanics.assemble(dim=3, cone="pyramid", k=8).

Scene (frozen physics):
- 3D, gravity along -z, mu = 0.7 everywhere, density 2000 kg/m^3.
- Pedestal box 6 x 6 x 1 (x, y, z), top at z = 1, its +x edge at x = 0,
  so its center is (-3, 0, 0.5). Node 1. Its bottom face is one ground
  patch (node pair (0, 1)).
- Unit cubes at layer L have center z = 1.5 + L and center
  (ix * dx, iy * dy), dx = dy = 1/12 by default. Default grid x in
  [-3, 4], y in [-3, 3]. Cubes take nodes 2.. in sorted (L, ix, iy)
  order, matching the host box list of overhang3d_demo.py.

Legality (geometry only):
- Support. The bottom face must overlap a support below by a rectangle
  with both side lengths at least 2 dx. A layer-0 cube overlaps the
  pedestal top [-6, 0] x [-3, 3]; a layer-L (L >= 1) cube overlaps at
  least one layer-(L-1) cube. On this grid every overlap is a multiple
  of dx (or dy), so 0.5 dx separates a real overlap (>= dx) from a
  touching edge (0) and 1.5 dx separates the 2 dx floor from dx.
- Same-layer clearance. Two same-layer cubes must be strictly separated
  in at least one axis: center distance greater than one block width in
  x or in y. This is the open-overlap rule tightened at the touching
  boundary. Touching same-layer cubes share a vertical contact face
  that the support-only patch budget does not model, so they are
  excluded, exactly as the 2D module excludes same-layer touching.
  Excluding them keeps P_max tight and keeps build_system an exact
  match of the certified host pipeline. Touching never helps overhang.
- In bounds and at most one layer above the current highest layer.
- LatticeSpec3D.mode adds a placement-reachability conjunct. "static"
  (default) checks static support only. "drop" additionally requires a
  clear vertical column above the target: any higher-layer cube whose
  footprint openly overlaps the target footprint in both x and y blocks.
  Slide modes are deferred in 3D v1: a lateral corridor in the plane
  needs a direction parameter (a slide can go +x, -x, +y, -y or any
  diagonal), and 3D corridor semantics are out of scope for this slice.
  See KNOWN_LIMITS.md.

Support-count bound (sizes P_max). A cube's footprint is a unit square
T. Every support is a layer-below cube (or the pedestal) whose footprint
openly overlaps T, so its center lies in the open box
(-0.5, 1.5) x (-0.5, 1.5) around T. Supports are mutually separated:
each pair has center distance greater than one block width in x or in y.
Split the x range into halves L = (-0.5, 0.5] and R = (0.5, 1.5), each
of width one. Two supports both in L have x distance strictly less than
one, so they are not x-separated and must be y-separated; the y range
has width two, so at most two points are pairwise more than one apart.
So L holds at most two supports and R holds at most two, giving at most
four supports, and the four corners of T realize it. So the true bound
is four supports per cube. P_max = 1 ground patch + 4 * n_max support
slots + 1 slack = 4 * n_max + 2, with V = 4 vertices per patch.

Node ordering matches the host box list. The host builds
boxes = [pedestal] + [cube per placement in sorted (L, ix, iy) order],
so the pedestal is node 1 and cubes take nodes 2.. in that order.
patch_table sorts the active placements the same way, so block indices,
the (i, j) node pairs on every patch, and the load vector all line up
with keystone.mechanics.assemble on the same scene.
"""

import functools
import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..mechanics.loads import DEFAULT_G
from ..solve.batch_jax import margin_core

# Frozen task constants. These describe the scene, not tolerances.
PEDESTAL_W = 6.0
PEDESTAL_D = 6.0  # depth along y
PEDESTAL_H = 1.0
PEDESTAL_X = -3.0  # +x edge at x = 0, top at z = 1
PEDESTAL_Y = 0.0
PED_LEFT = PEDESTAL_X - PEDESTAL_W / 2.0  # -6.0
PED_RIGHT = PEDESTAL_X + PEDESTAL_W / 2.0  # 0.0
PED_BOT = PEDESTAL_Y - PEDESTAL_D / 2.0  # -3.0
PED_TOP = PEDESTAL_Y + PEDESTAL_D / 2.0  # 3.0
PED_CZ = PEDESTAL_H / 2.0  # 0.5
DENSITY = 2000.0  # kg/m^3, pedestal and cubes
MU = 0.7
X_LO = -3.0
X_HI = 4.0
Y_LO = -3.0
Y_HI = 3.0
DX = 1.0 / 12.0
PYRAMID_K = 8


@dataclass(frozen=True)
class LatticeSpec3D:
    """Static description of the 3D lattice scene. Hashable, so it keys jit.

    All fields are python scalars, so a LatticeSpec3D is a static
    argument to the kernels: build_system reads n_max, dx, dy, the
    pedestal geometry, and the derived paddings as compile-time
    constants.

    Grid bounds x_lo, x_hi, y_lo, y_hi are exposed so demos can shrink
    the action grid. The full grid is large: M = n_layers * n_pos_x *
    n_pos_y is about n_max * 85 * 73 on the default bounds.

    mode selects the placement-reachability rule enforced by is_legal:
    "static" (default) checks static-support legality only, "drop"
    additionally requires a clear vertical column above the target. Slide
    modes are deferred in 3D v1 (see the module docstring). mode is a
    compile-time constant, so the "static" path traces the same graph as
    before this field existed.
    """

    n_max: int
    dx: float = DX
    dy: float = DX
    mode: str = "static"
    x_lo: float = X_LO
    x_hi: float = X_HI
    y_lo: float = Y_LO
    y_hi: float = Y_HI
    mu: float = MU
    density: float = DENSITY
    g: float = DEFAULT_G
    k: int = PYRAMID_K
    ped_left: float = PED_LEFT
    ped_right: float = PED_RIGHT
    ped_bot: float = PED_BOT
    ped_top: float = PED_TOP
    ped_cx: float = PEDESTAL_X
    ped_cy: float = PEDESTAL_Y
    ped_cz: float = PED_CZ

    def __post_init__(self):
        if self.mode not in ("static", "drop"):
            raise ValueError(
                "mode must be 'static' or 'drop' (slide is deferred in 3D "
                f"v1), got {self.mode!r}"
            )

    @property
    def ix_lo(self) -> int:
        """Lowest x grid index with x = ix dx >= x_lo."""
        return int(math.ceil(self.x_lo / self.dx - 1e-9))

    @property
    def ix_hi(self) -> int:
        """Highest x grid index with x = ix dx <= x_hi."""
        return int(math.floor(self.x_hi / self.dx + 1e-9))

    @property
    def iy_lo(self) -> int:
        return int(math.ceil(self.y_lo / self.dy - 1e-9))

    @property
    def iy_hi(self) -> int:
        return int(math.floor(self.y_hi / self.dy + 1e-9))

    @property
    def n_pos_x(self) -> int:
        return self.ix_hi - self.ix_lo + 1

    @property
    def n_pos_y(self) -> int:
        return self.iy_hi - self.iy_lo + 1

    @property
    def n_layers(self) -> int:
        return self.n_max

    @property
    def n_blocks(self) -> int:
        """Pedestal plus n_max cube slots."""
        return self.n_max + 1

    @property
    def P_max(self) -> int:
        """Ground patch, at most four supports per cube, plus one slack."""
        return 4 * self.n_max + 2

    @property
    def V(self) -> int:
        return 4

    @property
    def ncomp(self) -> int:
        return 3

    @property
    def rows(self) -> int:
        return 6 * self.n_blocks

    @property
    def nf(self) -> int:
        return self.ncomp * self.P_max * self.V

    @property
    def ncone(self) -> int:
        return (1 + self.k) * self.P_max * self.V

    @property
    def M(self) -> int:
        """Full action-grid size: layers times x positions times y positions."""
        return self.n_layers * self.n_pos_x * self.n_pos_y

    @property
    def ped_mass(self) -> float:
        return self.density * PEDESTAL_W * PEDESTAL_D * PEDESTAL_H

    @property
    def cube_mass(self) -> float:
        return self.density * 1.0 * 1.0 * 1.0  # unit cube


@dataclass(frozen=True)
class State:
    """Placed cubes as fixed-shape arrays. count real slots are used.

    placed_layer, placed_ix, placed_iy, placed_mask have length n_max.
    Slot order is arbitrary; build_system sorts the active slots by
    (layer, ix, iy) to fix node ids, so the order a rollout fills slots
    in does not matter.
    """

    placed_layer: jnp.ndarray
    placed_ix: jnp.ndarray
    placed_iy: jnp.ndarray
    placed_mask: jnp.ndarray
    count: jnp.ndarray


jax.tree_util.register_dataclass(
    State,
    data_fields=["placed_layer", "placed_ix", "placed_iy", "placed_mask", "count"],
    meta_fields=[],
)


def empty_state(spec: LatticeSpec3D) -> State:
    """A state with no cubes placed."""
    n = spec.n_max
    return State(
        placed_layer=jnp.zeros(n, dtype=jnp.int32),
        placed_ix=jnp.zeros(n, dtype=jnp.int32),
        placed_iy=jnp.zeros(n, dtype=jnp.int32),
        placed_mask=jnp.zeros(n, dtype=bool),
        count=jnp.asarray(0, dtype=jnp.int32),
    )


def state_from_placements(spec: LatticeSpec3D, placements) -> State:
    """Build a State from a python iterable of (layer, ix, iy) triples."""
    placements = list(placements)
    n = spec.n_max
    if len(placements) > n:
        raise ValueError(f"{len(placements)} placements exceed n_max={n}")
    lay = jnp.zeros(n, dtype=jnp.int32)
    xid = jnp.zeros(n, dtype=jnp.int32)
    yid = jnp.zeros(n, dtype=jnp.int32)
    msk = jnp.zeros(n, dtype=bool)
    for i, (L, ix, iy) in enumerate(placements):
        lay = lay.at[i].set(int(L))
        xid = xid.at[i].set(int(ix))
        yid = yid.at[i].set(int(iy))
        msk = msk.at[i].set(True)
    return State(lay, xid, yid, msk, jnp.asarray(len(placements), dtype=jnp.int32))


def place(spec: LatticeSpec3D, state: State, layer, ix, iy) -> State:
    """Append one cube at (layer, ix, iy). Pure and fixed-shape.

    The write index is clamped to the last slot so an out-of-room
    placement never writes out of bounds. Out-of-room placements are
    illegal and get masked to +inf margin by expand_kernel, so the
    clamped write is never read as a real result.
    """
    idx = jnp.minimum(state.count, spec.n_max - 1)
    return State(
        placed_layer=state.placed_layer.at[idx].set(jnp.asarray(layer, jnp.int32)),
        placed_ix=state.placed_ix.at[idx].set(jnp.asarray(ix, jnp.int32)),
        placed_iy=state.placed_iy.at[idx].set(jnp.asarray(iy, jnp.int32)),
        placed_mask=state.placed_mask.at[idx].set(True),
        count=state.count + 1,
    )


def action_grid(spec: LatticeSpec3D):
    """Full (cand_layers, cand_ix, cand_iy), each (M,).

    Row-major over (layer, ix, iy). Deterministic and fixed.
    """
    import numpy as np

    layers = np.arange(spec.n_layers, dtype=np.int32)
    xs = np.arange(spec.ix_lo, spec.ix_hi + 1, dtype=np.int32)
    ys = np.arange(spec.iy_lo, spec.iy_hi + 1, dtype=np.int32)
    LL, XX, YY = np.meshgrid(layers, xs, ys, indexing="ij")
    return (
        jnp.asarray(LL.reshape(-1)),
        jnp.asarray(XX.reshape(-1)),
        jnp.asarray(YY.reshape(-1)),
    )


def _cone_pyramid(mu_vec: jnp.ndarray, n_patches: int, verts_per: int, k: int):
    """3D inscribed pyramid cone rows G, pure JNP twin of
    cones.cone_matrix_pyramid.

    Rows per vertex: one non-penetration row [-n <= 0] then k facet rows
        u cos(th_j) + v sin(th_j) - mu cos(pi/k) n <= 0,
    th_j = (2j + 1) pi / k. Block diagonal with (1 + k) x 3 vertex
    blocks; row (p V + v) (1 + k) + r, column (p V + v) 3 + c. Reproduces
    the frozen cone layout exactly, without the host-side mu and k checks
    of cones.cone_matrix_pyramid, so it traces under jit.
    """
    pv = n_patches * verts_per
    mu_v = jnp.repeat(mu_vec, verts_per)
    j = jnp.arange(k)
    th = (2.0 * j + 1.0) * jnp.pi / k
    cos_th = jnp.cos(th)
    sin_th = jnp.sin(th)
    cos_pik = jnp.cos(jnp.pi / k)

    row0 = jnp.stack(
        [-jnp.ones(pv), jnp.zeros(pv), jnp.zeros(pv)], axis=-1
    )[:, None, :]  # (pv, 1, 3)
    n_coeff = -(mu_v[:, None] * cos_pik) * jnp.ones((pv, k))
    u_coeff = jnp.broadcast_to(cos_th, (pv, k))
    v_coeff = jnp.broadcast_to(sin_th, (pv, k))
    facets = jnp.stack([n_coeff, u_coeff, v_coeff], axis=-1)  # (pv, k, 3)
    blocks = jnp.concatenate([row0, facets], axis=1)  # (pv, 1 + k, 3)
    full = jnp.einsum("vrc,vw->vrwc", blocks, jnp.eye(pv))
    return full.reshape((1 + k) * pv, 3 * pv)


@dataclass(frozen=True)
class PatchTable:
    """Block and patch metadata for one state, before nondimensionalization.

    Blocks are indexed 0 (pedestal) then 1..n_max (cube slots); node id
    is the block index plus one. Patch slot 0 is the pedestal-ground
    contact, slots 1 + 4 c + s hold support s of cube c (s in 0..3), and
    the last slot is slack. Vertices are the four corners of the overlap
    rectangle, ordered counterclockwise about +z. Every field is a jnp
    array so PatchTable is a pytree that flows through jit.
    """

    com: jnp.ndarray  # (nb, 3) world block coms
    block_mask: jnp.ndarray  # (nb,)
    mass: jnp.ndarray  # (nb,)
    patch_i: jnp.ndarray  # (P,) lower node id, 0 for ground
    patch_j: jnp.ndarray  # (P,) upper node id
    patch_active: jnp.ndarray  # (P,)
    verts: jnp.ndarray  # (P, 4, 3) world patch vertices
    vert_mask: jnp.ndarray  # (P, 4)
    mu_vec: jnp.ndarray  # (P,) friction, 0 on masked patches


jax.tree_util.register_dataclass(
    PatchTable,
    data_fields=[
        "com",
        "block_mask",
        "mass",
        "patch_i",
        "patch_j",
        "patch_active",
        "verts",
        "vert_mask",
        "mu_vec",
    ],
    meta_fields=[],
)


def patch_table(spec: LatticeSpec3D, state: State) -> PatchTable:
    """Blocks and contact patches of a state, pure JNP. No dynamic loops.

    Sorts active placements by (layer, ix, iy) so node ids match the
    host, then enumerates the ground patch and up to four supports per
    cube by masked reductions over the static n_max and P_max ranges.
    This is the single source of the scene geometry; build_system
    consumes it.
    """
    n = spec.n_max
    dx = spec.dx
    dy = spec.dy
    P = spec.P_max
    nx = spec.n_pos_x
    ny = spec.n_pos_y

    # 1. Sort active placements by (layer, ix, iy) so node ids match the host.
    lay = state.placed_layer
    xi = state.placed_ix
    yi = state.placed_iy
    msk = state.placed_mask
    enc = (
        lay.astype(jnp.int64) * (nx * ny)
        + (xi.astype(jnp.int64) - spec.ix_lo) * ny
        + (yi.astype(jnp.int64) - spec.iy_lo)
    )
    key = jnp.where(msk, enc, jnp.int64(n) * (nx * ny) + 1)  # masked slots sort last
    order = jnp.argsort(key)
    s_ix = xi[order]
    s_iy = yi[order]
    s_msk = msk[order]
    s_layf = lay[order].astype(jnp.float64)
    s_x = s_ix.astype(jnp.float64) * dx
    s_y = s_iy.astype(jnp.float64) * dy

    # 2. Block coms, masses, and mask. Block 0 pedestal, blocks 1..n cubes.
    ped_com = jnp.array([spec.ped_cx, spec.ped_cy, spec.ped_cz])
    cube_com = jnp.stack([s_x, s_y, 1.5 + s_layf], axis=1)
    com = jnp.concatenate([ped_com[None, :], cube_com], axis=0)  # (nb, 3)
    block_mask = jnp.concatenate([jnp.array([True]), s_msk])  # (nb,)
    mass = jnp.where(
        block_mask,
        jnp.concatenate([jnp.array([spec.ped_mass]), jnp.full(n, spec.cube_mass)]),
        0.0,
    )

    # 3. Supports per cube. Lower slots b in 0..n (0 = pedestal treated as
    #    layer -1 so the uniform rule lower.layer == cube.layer - 1 covers
    #    both the pedestal and cube-cube supports).
    low_xl = jnp.concatenate([jnp.array([spec.ped_left]), s_x - 0.5])
    low_xr = jnp.concatenate([jnp.array([spec.ped_right]), s_x + 0.5])
    low_yl = jnp.concatenate([jnp.array([spec.ped_bot]), s_y - 0.5])
    low_yr = jnp.concatenate([jnp.array([spec.ped_top]), s_y + 0.5])
    low_node = jnp.concatenate([jnp.array([1.0]), jnp.arange(n) + 2.0])
    low_layer = jnp.concatenate([jnp.array([-1.0]), s_layf])
    low_placed = jnp.concatenate([jnp.array([True]), s_msk])

    c_x = s_x  # (n,)
    c_y = s_y
    c_layer = s_layf
    c_placed = s_msk

    # overlap[c, b] of cube c footprint with lower b footprint, per axis.
    ov_x = jnp.minimum(low_xr[None, :], (c_x + 0.5)[:, None]) - jnp.maximum(
        low_xl[None, :], (c_x - 0.5)[:, None]
    )
    ov_y = jnp.minimum(low_yr[None, :], (c_y + 0.5)[:, None]) - jnp.maximum(
        low_yl[None, :], (c_y - 0.5)[:, None]
    )
    layer_ok = low_layer[None, :] == (c_layer[:, None] - 1.0)
    # A contact exists when both overlaps clear the patch-area floor. On this
    # grid every overlap is a multiple of dx (or dy), so 0.5 dx separates a
    # real contact (>= dx) from a touching edge (0), matching the host A_min
    # classification for the tiny lattice areas.
    valid = (
        c_placed[:, None]
        & low_placed[None, :]
        & layer_ok
        & (ov_x > 0.5 * dx)
        & (ov_y > 0.5 * dy)
    )  # (n, nb)

    # Rank valid supports by lower-slot order; keep at most four (the bound).
    rank = jnp.cumsum(valid.astype(jnp.int32), axis=1) - 1  # (n, nb)
    s_slot = jnp.arange(4)
    sel = valid[:, None, :] & (rank[:, None, :] == s_slot[None, :, None])  # (n, 4, nb)
    active_cs = jnp.any(sel, axis=2)  # (n, 4)
    selg = sel.astype(jnp.float64)
    g_xl = jnp.sum(selg * low_xl[None, None, :], axis=2)  # (n, 4)
    g_xr = jnp.sum(selg * low_xr[None, None, :], axis=2)
    g_yl = jnp.sum(selg * low_yl[None, None, :], axis=2)
    g_yr = jnp.sum(selg * low_yr[None, None, :], axis=2)
    g_node = jnp.sum(selg * low_node[None, None, :], axis=2)

    # Overlap rectangle at the contact plane z = 1 + layer.
    xl = jnp.maximum(g_xl, (c_x - 0.5)[:, None])  # (n, 4)
    xr = jnp.minimum(g_xr, (c_x + 0.5)[:, None])
    yl = jnp.maximum(g_yl, (c_y - 0.5)[:, None])
    yr = jnp.minimum(g_yr, (c_y + 0.5)[:, None])
    zc = (1.0 + c_layer)[:, None] * jnp.ones((1, 4))
    up_node = ((jnp.arange(n) + 2.0)[:, None]) * jnp.ones((1, 4))

    # 4. Full patch arrays. Slot 0 ground, slots 1..4n cube supports (slot
    #    1 + 4 c + s), final slot slack (inactive).
    ci = g_node.reshape(-1)
    cj = up_node.reshape(-1)
    ca = active_cs.reshape(-1)
    cxl = xl.reshape(-1)
    cxr = xr.reshape(-1)
    cyl = yl.reshape(-1)
    cyr = yr.reshape(-1)
    cz = zc.reshape(-1)
    patch_i = jnp.concatenate([jnp.array([0.0]), ci, jnp.array([0.0])])
    patch_j = jnp.concatenate([jnp.array([1.0]), cj, jnp.array([0.0])])
    patch_active = jnp.concatenate([jnp.array([True]), ca, jnp.array([False])])
    patch_xl = jnp.concatenate([jnp.array([spec.ped_left]), cxl, jnp.array([0.0])])
    patch_xr = jnp.concatenate([jnp.array([spec.ped_right]), cxr, jnp.array([0.0])])
    patch_yl = jnp.concatenate([jnp.array([spec.ped_bot]), cyl, jnp.array([0.0])])
    patch_yr = jnp.concatenate([jnp.array([spec.ped_top]), cyr, jnp.array([0.0])])
    patch_z = jnp.concatenate([jnp.array([0.0]), cz, jnp.array([0.0])])

    # Four corners CCW about +z: (xl,yl) -> (xr,yl) -> (xr,yr) -> (xl,yr).
    vx = jnp.stack([patch_xl, patch_xr, patch_xr, patch_xl], axis=1)  # (P, 4)
    vy = jnp.stack([patch_yl, patch_yl, patch_yr, patch_yr], axis=1)
    vz = patch_z[:, None] * jnp.ones((1, 4))
    verts = jnp.stack([vx, vy, vz], axis=2)  # (P, 4, 3)
    vert_mask = jnp.broadcast_to(patch_active[:, None], (P, 4))

    # Masked patches carry mu = 0 (their columns are zero and the P4
    # regularizer drives their variables to zero).
    mu_vec = jnp.where(patch_active, spec.mu, 0.0)
    return PatchTable(
        com=com,
        block_mask=block_mask,
        mass=mass,
        patch_i=patch_i,
        patch_j=patch_j,
        patch_active=patch_active,
        verts=verts,
        vert_mask=vert_mask,
        mu_vec=mu_vec,
    )


def build_system(spec: LatticeSpec3D, state: State):
    """Nondimensional (A, w_dead, G, L, W) for a lattice state, pure JNP.

    Reproduces keystone.mechanics.assemble for this scene family exactly:
    same sign convention, same [Fx, Fy, Fz, Tx, Ty, Tz] row layout about
    each block's own com, same (p V + v) ncomp + c column layout, the
    same nondimensionalization (L is the bounding-box diagonal of active
    patch vertices and coms, W is the total active weight with DEFAULT_G),
    and the same inscribed pyramid cone rows with k = 8 facets.
    """
    nb = spec.n_blocks
    P = spec.P_max
    V = spec.V
    ncomp = spec.ncomp
    pt = patch_table(spec, state)
    com = pt.com
    block_mask = pt.block_mask
    mass = pt.mass
    patch_i = pt.patch_i
    patch_j = pt.patch_j
    patch_active = pt.patch_active
    verts = pt.verts
    vert_mask = pt.vert_mask
    mu_vec = pt.mu_vec

    # 5. Nondimensional length: bbox diagonal of active coms and patch verts.
    big = 1e30
    all_pts = jnp.concatenate([com, verts.reshape(-1, 3)], axis=0)
    all_valid = jnp.concatenate(
        [block_mask, vert_mask.reshape(-1)], axis=0
    )[:, None]
    pmin = jnp.min(jnp.where(all_valid, all_pts, big), axis=0)
    pmax = jnp.max(jnp.where(all_valid, all_pts, -big), axis=0)
    L = jnp.linalg.norm(pmax - pmin)

    # 6. Total weight and dead load, nondimensional. 6 rows per block, Fz is
    #    row index 2 in [Fx, Fy, Fz, Tx, Ty, Tz].
    W = jnp.sum(jnp.where(block_mask, mass, 0.0)) * spec.g
    w_dead = jnp.zeros(6 * nb)
    fz_rows = 6 * jnp.arange(nb) + 2
    w_dead = w_dead.at[fz_rows].set(jnp.where(block_mask, -mass * spec.g / W, 0.0))

    # 7. Equilibrium map A. Force on block i at a vertex is
    #    F_i = -n n_hat + u t1 + v t2; block j gets the negative. Every patch
    #    has n_hat = +z, t1 = +x, t2 = +y.
    verts_nd = verts / L
    com_nd = com / L
    node_of_block = jnp.arange(nb) + 1  # node id per block index
    is_i = (patch_i[None, :] == node_of_block[:, None].astype(jnp.float64)) & (
        patch_i[None, :] >= 1.0
    )
    is_j = patch_j[None, :] == node_of_block[:, None].astype(jnp.float64)
    sgn = jnp.where(is_i, 1.0, jnp.where(is_j, -1.0, 0.0))  # (nb, P)
    act_bp = (is_i | is_j) & patch_active[None, :]  # (nb, P)
    # Force basis on block i per component: c=0 is -n_hat = (0,0,-1),
    # c=1 is t1 = (1,0,0), c=2 is t2 = (0,1,0).
    fi = jnp.array(
        [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )  # (ncomp, 3)
    fvec = sgn[:, :, None, None] * fi[None, None, :, :]  # (nb, P, ncomp, 3)
    d = verts_nd[None, :, :, :] - com_nd[:, None, None, :]  # (nb, P, V, 3)
    d_exp = d[:, :, :, None, :]  # (nb, P, V, 1, 3)
    f_exp = fvec[:, :, None, :, :]  # (nb, P, 1, ncomp, 3)
    torque = jnp.cross(
        jnp.broadcast_to(d_exp, (nb, P, V, ncomp, 3)),
        jnp.broadcast_to(f_exp, (nb, P, V, ncomp, 3)),
        axis=-1,
    )  # (nb, P, V, ncomp, 3)
    force = jnp.broadcast_to(f_exp, (nb, P, V, ncomp, 3))
    entry = jnp.concatenate([force, torque], axis=-1)  # (nb, P, V, ncomp, 6)
    entry = entry * act_bp[:, :, None, None, None]
    # Rows: block b contributes 6 rows 6b..6b+5; columns (p V + v) ncomp + c.
    A = jnp.transpose(entry, (0, 4, 1, 2, 3)).reshape(6 * nb, P * V * ncomp)

    # 8. Cone rows from the per-patch friction of the table.
    G = _cone_pyramid(mu_vec, P, V, spec.k)
    return A, w_dead, G, L, W


def _reach_ok(spec: LatticeSpec3D, state: State, layer, ix, iy):
    """Placement reachability for mode "drop". Pure JNP bool.

    Reachability is evaluated against the state BEFORE the placement, so
    the same final set can be reachable in one build order and not in
    another.

    Drop: the vertical column above the target cell must be clear. A
    placed cube at a layer strictly above the target blocks when its
    footprint openly overlaps the target footprint in both x and y
    (open overlap; touching edges do not block). The pedestal never
    blocks; it sits below layer 0.

    Slide is deferred in 3D v1: a lateral corridor needs a direction
    parameter and 3D corridor semantics are out of scope for this slice
    (see the module docstring and KNOWN_LIMITS.md).
    """
    dx = spec.dx
    dy = spec.dy
    x = jnp.asarray(ix, jnp.float64) * dx
    y = jnp.asarray(iy, jnp.float64) * dy
    layer = jnp.asarray(layer, jnp.int32)
    px = state.placed_ix.astype(jnp.float64) * dx
    py = state.placed_iy.astype(jnp.float64) * dy
    pl = state.placed_layer
    msk = state.placed_mask

    # Overlaps on this grid are multiples of dx (or dy), so 0.5 dx separates
    # touching (0) from a real overlap (>= dx).
    ov_x = jnp.minimum(x + 0.5, px + 0.5) - jnp.maximum(x - 0.5, px - 0.5)
    ov_y = jnp.minimum(y + 0.5, py + 0.5) - jnp.maximum(y - 0.5, py - 0.5)
    above = msk & (pl > layer)
    blocks = above & (ov_x > 0.5 * dx) & (ov_y > 0.5 * dy)
    return ~jnp.any(blocks)


def is_legal(spec: LatticeSpec3D, state: State, layer, ix, iy):
    """Geometry-only legality of placing a cube at (layer, ix, iy).

    Pure JNP, returns a scalar bool. Mirrors the frozen legality rules:
    room, layer reachability, support overlap rectangle both sides
    >= 2 dx, same-layer strict separation in at least one axis, and in
    bounds.

    spec.mode adds a placement-reachability conjunct: "drop" requires a
    clear vertical column above the target (see _reach_ok). The mode is a
    compile-time constant; "static" (the default) skips the check
    entirely, so the default trace is the pre-mode code path unchanged.
    """
    dx = spec.dx
    dy = spec.dy
    x = jnp.asarray(ix, jnp.float64) * dx
    y = jnp.asarray(iy, jnp.float64) * dy
    layer = jnp.asarray(layer, jnp.int32)

    room = state.count < spec.n_max
    max_L = jnp.max(jnp.where(state.placed_mask, state.placed_layer, -1))
    layer_ok = (layer >= 0) & (layer <= spec.n_max - 1) & (layer <= max_L + 1)

    # Support. Layer 0 rests on the pedestal top; layer L rests on a
    # layer-(L-1) cube. Overlap rectangle must reach 2 dx on both sides.
    ped_ov_x = jnp.minimum(x + 0.5, spec.ped_right) - jnp.maximum(
        x - 0.5, spec.ped_left
    )
    ped_ov_y = jnp.minimum(y + 0.5, spec.ped_top) - jnp.maximum(
        y - 0.5, spec.ped_bot
    )
    support0 = (ped_ov_x > 1.5 * dx) & (ped_ov_y > 1.5 * dy)  # >= 2 dx both sides
    below = state.placed_mask & (state.placed_layer == layer - 1)
    bx = state.placed_ix.astype(jnp.float64) * dx
    by = state.placed_iy.astype(jnp.float64) * dy
    below_ov_x = jnp.minimum(x + 0.5, bx + 0.5) - jnp.maximum(x - 0.5, bx - 0.5)
    below_ov_y = jnp.minimum(y + 0.5, by + 0.5) - jnp.maximum(y - 0.5, by - 0.5)
    supportL = jnp.any(
        below & (below_ov_x > 1.5 * dx) & (below_ov_y > 1.5 * dy)
    )
    support_ok = jnp.where(layer == 0, support0, supportL)

    # Same-layer clearance: strictly separated in at least one axis, so no
    # footprint overlap and no shared vertical face. A pair conflicts when it
    # is not separated in x (x distance <= one block width) and not separated
    # in y. On the grid, distance <= 1 tests as distance < 1 + 0.5 dx.
    same = state.placed_mask & (state.placed_layer == layer)
    sx = state.placed_ix.astype(jnp.float64) * dx
    sy = state.placed_iy.astype(jnp.float64) * dy
    near_x = jnp.abs(x - sx) < 1.0 + 0.5 * dx
    near_y = jnp.abs(y - sy) < 1.0 + 0.5 * dy
    clear_ok = ~jnp.any(same & near_x & near_y)

    bounds_ok = (
        (ix >= spec.ix_lo)
        & (ix <= spec.ix_hi)
        & (iy >= spec.iy_lo)
        & (iy <= spec.iy_hi)
    )
    base = room & layer_ok & support_ok & clear_ok & bounds_ok
    if spec.mode == "static":
        return base
    return base & _reach_ok(spec, state, layer, ix, iy)


def expand_kernel(
    spec: LatticeSpec3D,
    state: State,
    cand_layers,
    cand_ix,
    cand_iy,
    eps_reg,
    tol_cone,
    *,
    solver_tol,
    max_iter,
):
    """Legality, P4 margin, and a certified flag for each candidate.

    vmaps [place -> build_system -> margin_core] over the candidate
    arrays. Illegal candidates are still solved for shape uniformity,
    then their margin is set to +inf and their certified flag to False.
    certified is the margin_batch meaning: cone-admissible
    (viol <= tol_cone) and the margin finite. Returns
    (legal (M,), margins (M,), certified (M,)).

    jit once per (spec, len(cand), solver_tol, max_iter) through the
    _expand_jit cache below.
    """
    fn = _expand_jit(
        spec, int(cand_layers.shape[0]), float(solver_tol), int(max_iter)
    )
    return fn(state, cand_layers, cand_ix, cand_iy, eps_reg, tol_cone)


@functools.lru_cache(maxsize=None)
def _expand_jit(spec: LatticeSpec3D, ncand: int, solver_tol: float, max_iter: int):
    """Build and cache the jitted, vmapped expansion for a fixed shape."""

    def one(state, cl, cx, cy, eps_reg, tol_cone):
        legal = is_legal(spec, state, cl, cx, cy)
        ns = place(spec, state, cl, cx, cy)
        A, w_dead, G, L, W = build_system(spec, ns)
        # margin_core returns (margin, f, r, viol, ...); slice the first
        # four so appended diagnostics never break this consumer.
        margin, f, r, viol = margin_core(
            A, w_dead, G, eps_reg, solver_tol=solver_tol, max_iter=max_iter
        )[:4]
        cert = (viol <= tol_cone) & jnp.isfinite(margin)
        margin = jnp.where(legal, margin, jnp.inf)
        cert = jnp.where(legal, cert, False)
        return legal, margin, cert

    vmapped = jax.vmap(one, in_axes=(None, 0, 0, 0, None, None))
    return jax.jit(vmapped)


@functools.lru_cache(maxsize=None)
def _legal_grid_jit(spec: LatticeSpec3D, ncand: int):
    """Cached jit of legality over a fixed candidate grid, batched over leaves."""

    def one(state, cl, cx, cy):
        return is_legal(spec, state, cl, cx, cy)

    over_cand = jax.vmap(one, in_axes=(None, 0, 0, 0))
    over_leaf = jax.vmap(over_cand, in_axes=(0, None, None, None))
    return jax.jit(over_leaf)


def legal_grid(spec: LatticeSpec3D, states: State, cand_layers, cand_ix, cand_iy):
    """(B, M) legality for a batch of B states over the fixed candidate grid.

    Pure geometry, no QP, so this pass is cheap. The search uses it to
    find the legal frontier before spending any solve on it.
    """
    fn = _legal_grid_jit(spec, int(cand_layers.shape[0]))
    return fn(states, cand_layers, cand_ix, cand_iy)


@functools.lru_cache(maxsize=None)
def _solve_states_jit(spec: LatticeSpec3D, B: int, solver_tol: float, max_iter: int):
    """Cached jit of [build_system -> margin_core] over a batch of B states."""

    def one(state, eps_reg, tol_cone):
        A, w_dead, G, L, W = build_system(spec, state)
        # margin_core returns (margin, f, r, viol, ...); slice the first
        # four so appended diagnostics never break this consumer.
        margin, f, r, viol = margin_core(
            A, w_dead, G, eps_reg, solver_tol=solver_tol, max_iter=max_iter
        )[:4]
        cert = (viol <= tol_cone) & jnp.isfinite(margin)
        return margin, cert

    return jax.jit(jax.vmap(one, in_axes=(0, None, None)))


def margins_of_states(
    spec: LatticeSpec3D, states: State, eps_reg, tol_cone, *, solver_tol, max_iter
):
    """P4 margins and certified flags for a batch of B lattice states.

    states is a State pytree with a leading batch axis. Returns
    (margins (B,), certified (B,)). This is the certified-path solve the
    search runs on its legal, uncached frontier.
    """
    b = int(states.count.shape[0])
    fn = _solve_states_jit(spec, b, float(solver_tol), int(max_iter))
    return fn(states, eps_reg, tol_cone)


def stack_states(states_list):
    """Stack a python list of State pytrees along a new leading axis."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states_list)


def batch_states(spec: LatticeSpec3D, keys):
    """Build a batched State pytree from a list of placement-tuple keys.

    keys is a list of tuples of (layer, ix, iy) triples. Uses numpy to
    fill the padded arrays in one shot, so a large frontier is cheap to
    assemble.
    """
    import numpy as np

    b = len(keys)
    n = spec.n_max
    lay = np.zeros((b, n), dtype=np.int32)
    xid = np.zeros((b, n), dtype=np.int32)
    yid = np.zeros((b, n), dtype=np.int32)
    msk = np.zeros((b, n), dtype=bool)
    cnt = np.zeros(b, dtype=np.int32)
    for r, key in enumerate(keys):
        for c, (L, ix, iy) in enumerate(key):
            lay[r, c] = int(L)
            xid[r, c] = int(ix)
            yid[r, c] = int(iy)
            msk[r, c] = True
        cnt[r] = len(key)
    return State(
        placed_layer=jnp.asarray(lay),
        placed_ix=jnp.asarray(xid),
        placed_iy=jnp.asarray(yid),
        placed_mask=jnp.asarray(msk),
        count=jnp.asarray(cnt),
    )


def overhang(placements, dx: float = DX) -> float:
    """Rightmost cube edge beyond the pedestal edge at x = 0, or -inf."""
    if not placements:
        return float("-inf")
    return max(ix * dx + 0.5 for (_, ix, _iy) in placements)
