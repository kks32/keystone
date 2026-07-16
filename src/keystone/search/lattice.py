"""Jittable 2D lattice environment for cube-stacking search.

This module rebuilds the frozen cube-stacking scene of examples/
search_overhang.py as a pure-JNP environment. The naive script rebuilds
a host-side numpy Assembly per candidate and then runs a CPU batch QP.
Here the whole geometry-to-system path is one pure function of the state,
so a placement, an equilibrium system, and a P4 margin all run inside a
single jitted, vmapped kernel. Fixed padded shapes keep the kernel device
agnostic: the same code runs on CPU now and on a GPU later.

Scene (frozen, identical physics to the naive script):
- 2D, xz plane, gravity along -z, mu = 0.7 everywhere.
- Pedestal box 6 wide, 1 tall, right edge at x = 0, on the ground z = 0.
  Node 1. Its bottom face is one ground patch (node pair (0, 1)).
- Unit cubes at layer L have center z = 1.5 + L and center x = j * dx,
  dx = 1/24, with j in the integer range for x in [-3, 4].
- Legality: a layer-0 cube overlaps the pedestal top [-6, 0] by >= 2 dx;
  a layer-L (L >= 1) cube overlaps at least one layer-(L-1) cube by
  >= 2 dx; same-layer cubes are separated by more than one block width
  (no footprint overlap and no shared vertical face); in bounds; and the
  layer is at most one above the current highest layer. LatticeSpec.mode
  can add a placement-reachability conjunct (drop column or slide
  corridor, see _reach_ok); the default "static" keeps exactly the rules
  above.

Node ordering matches the naive script exactly. The host builds
boxes_of(key) = [pedestal] + [cube per placement in sorted (L, j) order],
so pedestal is node 1 and cubes take nodes 2.. in sorted (L, j) order.
build_system sorts the active placements the same way, so block indices,
the (i, j) node pairs on every patch, and the load vector all line up
with keystone.mechanics.assemble on the same scene.

Contacts on this lattice are horizontal interval overlaps. Every patch
normal is +z and t1 is +x (the 2D frame rule gives exactly this for a
horizontal contact), so there is no clipping. A cube sits on at most two
supports (same-layer centers are more than one apart), so patches per
cube are at most two. With the ground patch that gives at most
2 * n_max + 1 patches; P_max = 2 * n_max + 2 leaves one padded slot.

Same-layer touching (center distance exactly one block width) is excluded
here. The naive script allows it, but a shared vertical face adds a patch
the two-supports bound does not budget for, and touching never helps
overhang. Excluding it keeps P_max tight and keeps build_system an exact
match of the certified host pipeline.
"""

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..mechanics.loads import DEFAULT_G
from ..solve.batch_jax import margin_core
from ..solve.pdhg import pdhg_margin

# Frozen task constants. These describe the scene, not tolerances.
PEDESTAL_W = 6.0
PEDESTAL_H = 1.0
PEDESTAL_X = -3.0  # right edge at x = 0, top at z = 1
PED_LEFT = PEDESTAL_X - PEDESTAL_W / 2.0  # -6.0
PED_RIGHT = PEDESTAL_X + PEDESTAL_W / 2.0  # 0.0
PED_CZ = PEDESTAL_H / 2.0  # 0.5
DENSITY = 2000.0  # kg/m^3, both pedestal and cubes (matches the naive script)
MU = 0.7
X_LO = -3.0
X_HI = 4.0
DX = 1.0 / 24.0


@dataclass(frozen=True)
class LatticeSpec:
    """Static description of the lattice scene. Hashable, so it keys jit.

    All fields are python scalars, so a LatticeSpec is a static argument
    to the kernel: build_system reads n_max, dx, the pedestal geometry,
    and the derived paddings as compile-time constants.

    mode selects the placement-reachability rule enforced by is_legal:
    "static" (default) checks static-support legality only, "drop"
    additionally requires a clear vertical column above the target, and
    "slide" additionally requires either a clear column or a clear lateral
    corridor at the target layer. mode is a compile-time constant, so the
    "static" path traces the same graph as before this field existed.
    """

    n_max: int
    dx: float = DX
    mode: str = "static"
    x_lo: float = X_LO
    x_hi: float = X_HI
    mu: float = MU
    density: float = DENSITY
    g: float = DEFAULT_G
    ped_left: float = PED_LEFT
    ped_right: float = PED_RIGHT
    ped_cx: float = PEDESTAL_X
    ped_cz: float = PED_CZ

    def __post_init__(self):
        if self.mode not in ("static", "drop", "slide"):
            raise ValueError(
                f"mode must be 'static', 'drop', or 'slide', got {self.mode!r}"
            )

    @property
    def j_lo(self) -> int:
        """Lowest grid index with x = j dx >= x_lo."""
        import math

        return int(math.ceil(self.x_lo / self.dx - 1e-9))

    @property
    def j_hi(self) -> int:
        """Highest grid index with x = j dx <= x_hi."""
        import math

        return int(math.floor(self.x_hi / self.dx + 1e-9))

    @property
    def n_pos(self) -> int:
        return self.j_hi - self.j_lo + 1

    @property
    def n_layers(self) -> int:
        return self.n_max

    @property
    def n_blocks(self) -> int:
        """Pedestal plus n_max cube slots."""
        return self.n_max + 1

    @property
    def P_max(self) -> int:
        """Ground patch, at most two supports per cube, plus one slack."""
        return 2 * self.n_max + 2

    @property
    def V(self) -> int:
        return 2

    @property
    def ncomp(self) -> int:
        return 2

    @property
    def rows(self) -> int:
        return 3 * self.n_blocks

    @property
    def nf(self) -> int:
        return self.ncomp * self.P_max * self.V

    @property
    def ncone(self) -> int:
        return 3 * self.P_max * self.V

    @property
    def M(self) -> int:
        """Full action-grid size: layers times positions."""
        return self.n_layers * self.n_pos

    @property
    def ped_mass(self) -> float:
        return self.density * PEDESTAL_W * PEDESTAL_H  # unit depth

    @property
    def cube_mass(self) -> float:
        return self.density * 1.0 * 1.0  # unit cube, unit depth


@dataclass(frozen=True)
class State:
    """Placed cubes as fixed-shape arrays. count real slots are used.

    placed_layer, placed_xidx, placed_mask have length n_max. Slot order
    is arbitrary; build_system sorts the active slots by (layer, xidx) to
    fix node ids, so the order a rollout fills slots in does not matter.
    """

    placed_layer: jnp.ndarray
    placed_xidx: jnp.ndarray
    placed_mask: jnp.ndarray
    count: jnp.ndarray


jax.tree_util.register_dataclass(
    State,
    data_fields=["placed_layer", "placed_xidx", "placed_mask", "count"],
    meta_fields=[],
)


def empty_state(spec: LatticeSpec) -> State:
    """A state with no cubes placed."""
    n = spec.n_max
    return State(
        placed_layer=jnp.zeros(n, dtype=jnp.int32),
        placed_xidx=jnp.zeros(n, dtype=jnp.int32),
        placed_mask=jnp.zeros(n, dtype=bool),
        count=jnp.asarray(0, dtype=jnp.int32),
    )


def state_from_placements(spec: LatticeSpec, placements) -> State:
    """Build a State from a python iterable of (layer, xidx) pairs."""
    placements = list(placements)
    n = spec.n_max
    if len(placements) > n:
        raise ValueError(f"{len(placements)} placements exceed n_max={n}")
    lay = jnp.zeros(n, dtype=jnp.int32)
    xid = jnp.zeros(n, dtype=jnp.int32)
    msk = jnp.zeros(n, dtype=bool)
    for i, (L, j) in enumerate(placements):
        lay = lay.at[i].set(int(L))
        xid = xid.at[i].set(int(j))
        msk = msk.at[i].set(True)
    return State(lay, xid, msk, jnp.asarray(len(placements), dtype=jnp.int32))


def place(spec: LatticeSpec, state: State, layer, xidx) -> State:
    """Append one cube at (layer, xidx). Pure and fixed-shape.

    The write index is clamped to the last slot so an out-of-room
    placement never writes out of bounds. Out-of-room placements are
    illegal and get masked to +inf margin by expand_kernel, so the
    clamped write is never read as a real result.
    """
    idx = jnp.minimum(state.count, spec.n_max - 1)
    return State(
        placed_layer=state.placed_layer.at[idx].set(jnp.asarray(layer, jnp.int32)),
        placed_xidx=state.placed_xidx.at[idx].set(jnp.asarray(xidx, jnp.int32)),
        placed_mask=state.placed_mask.at[idx].set(True),
        count=state.count + 1,
    )


def action_grid(spec: LatticeSpec):
    """Full (cand_layers, cand_xidx), each (M,), over layers times positions.

    Row-major over (layer, position). Deterministic and fixed.
    """
    import numpy as np

    layers = np.arange(spec.n_layers, dtype=np.int32)
    xs = np.arange(spec.j_lo, spec.j_hi + 1, dtype=np.int32)
    LL, JJ = np.meshgrid(layers, xs, indexing="ij")
    return jnp.asarray(LL.reshape(-1)), jnp.asarray(JJ.reshape(-1))


def _cone_2d(mu_vec: jnp.ndarray, n_patches: int, verts_per: int) -> jnp.ndarray:
    """2D friction cone rows G, pure JNP twin of cones.cone_matrix_2d.

    Rows per vertex, in order: [-n <= 0], [u - mu n <= 0], [-u - mu n <= 0].
    Block diagonal with 3x2 vertex blocks, row (p V + v) 3 + r, column
    (p V + v) 2 + c. Reproduces the frozen cone layout exactly.
    """
    pv = n_patches * verts_per
    mu_v = jnp.repeat(mu_vec, verts_per)
    zero = jnp.zeros(pv)
    one = jnp.ones(pv)
    blocks = jnp.stack(
        [
            jnp.stack([-one, zero], axis=-1),
            jnp.stack([-mu_v, one], axis=-1),
            jnp.stack([-mu_v, -one], axis=-1),
        ],
        axis=1,
    )
    full = jnp.einsum("vrc,vw->vrwc", blocks, jnp.eye(pv))
    return full.reshape(3 * pv, 2 * pv)


@dataclass(frozen=True)
class PatchTable:
    """Block and patch metadata for one state, before nondimensionalization.

    Blocks are indexed 0 (pedestal) then 1..n_max (cube slots); node id is
    the block index plus one. Patch slot 0 is the pedestal-ground contact,
    slots 1 + 2 c + s hold support s of cube c, and the last slot is slack.
    Vertices are ordered v0 left, v1 right (t1 = +x). Every field is a jnp
    array so PatchTable is a pytree that flows through jit.
    """

    com: jnp.ndarray  # (nb, 3) world block coms
    block_mask: jnp.ndarray  # (nb,)
    mass: jnp.ndarray  # (nb,)
    patch_i: jnp.ndarray  # (P,) lower node id, 0 for ground
    patch_j: jnp.ndarray  # (P,) upper node id
    patch_active: jnp.ndarray  # (P,)
    verts: jnp.ndarray  # (P, 2, 3) world patch vertices
    vert_mask: jnp.ndarray  # (P, 2)
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


def patch_table(spec: LatticeSpec, state: State) -> PatchTable:
    """Blocks and contact patches of a state, pure JNP. No dynamic loops.

    Sorts active placements by (layer, xidx) so node ids match the host,
    then enumerates the ground patch and up to two supports per cube by
    masked reductions over the static n_max and P_max ranges. This is the
    single source of the scene geometry; build_system consumes it.
    """
    n = spec.n_max
    dx = spec.dx
    P = spec.P_max
    V = spec.V

    # 1. Sort active placements by (layer, xidx) so node ids match the host.
    lay = state.placed_layer
    pos = state.placed_xidx
    msk = state.placed_mask
    key = jnp.where(
        msk,
        lay.astype(jnp.int64) * spec.n_pos + (pos.astype(jnp.int64) - spec.j_lo),
        jnp.int64(n) * spec.n_pos + 1,  # masked slots sort last
    )
    order = jnp.argsort(key)
    s_lay = lay[order]
    s_pos = pos[order]
    s_msk = msk[order]
    s_x = s_pos.astype(jnp.float64) * dx
    s_layf = s_lay.astype(jnp.float64)

    # 2. Block coms, masses, and mask. Block 0 pedestal, blocks 1..n cubes.
    ped_com = jnp.array([spec.ped_cx, 0.0, spec.ped_cz])
    cube_com = jnp.stack([s_x, jnp.zeros(n), 1.5 + s_layf], axis=1)
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
    low_left = jnp.concatenate([jnp.array([spec.ped_left]), s_x - 0.5])
    low_right = jnp.concatenate([jnp.array([spec.ped_right]), s_x + 0.5])
    low_node = jnp.concatenate([jnp.array([1.0]), jnp.arange(n) + 2.0])
    low_layer = jnp.concatenate([jnp.array([-1.0]), s_layf])
    low_placed = jnp.concatenate([jnp.array([True]), s_msk])

    c_x = s_x  # (n,)
    c_layer = s_layf
    c_placed = s_msk

    # overlap[c, b] of cube c footprint with lower b footprint.
    ov = jnp.minimum(low_right[None, :], (c_x + 0.5)[:, None]) - jnp.maximum(
        low_left[None, :], (c_x - 0.5)[:, None]
    )
    layer_ok = low_layer[None, :] == (c_layer[:, None] - 1.0)
    # A contact exists when the overlap clears the patch-area floor. On this
    # grid every overlap is a multiple of dx, so 0.5 dx separates a real
    # contact (>= dx) from none (0), matching the host A_min classification.
    valid = (
        c_placed[:, None]
        & low_placed[None, :]
        & layer_ok
        & (ov > 0.5 * dx)
    )  # (n, nb)

    # Rank valid supports by lower-slot order; keep at most two.
    rank = jnp.cumsum(valid.astype(jnp.int32), axis=1) - 1  # (n, nb)
    s_ix = jnp.arange(2)
    sel = valid[:, None, :] & (rank[:, None, :] == s_ix[None, :, None])  # (n, 2, nb)
    active_cs = jnp.any(sel, axis=2)  # (n, 2)
    selg = sel.astype(jnp.float64)
    g_left = jnp.sum(selg * low_left[None, None, :], axis=2)
    g_right = jnp.sum(selg * low_right[None, None, :], axis=2)
    g_node = jnp.sum(selg * low_node[None, None, :], axis=2)

    # Patch vertices: overlap interval at the contact plane z = 1 + layer.
    xl = jnp.maximum(g_left, (c_x - 0.5)[:, None])  # (n, 2)
    xr = jnp.minimum(g_right, (c_x + 0.5)[:, None])
    zc = (1.0 + c_layer)[:, None] * jnp.ones((1, 2))
    up_node = ((jnp.arange(n) + 2.0)[:, None]) * jnp.ones((1, 2))

    # 4. Full patch arrays. Slot 0 ground, slots 1..2n cube supports (slot
    #    1 + 2 c + s), final slot slack (inactive).
    ci = g_node.reshape(-1)
    cj = up_node.reshape(-1)
    ca = active_cs.reshape(-1)
    cxl = xl.reshape(-1)
    cxr = xr.reshape(-1)
    cz = zc.reshape(-1)
    patch_i = jnp.concatenate([jnp.array([0.0]), ci, jnp.array([0.0])])
    patch_j = jnp.concatenate([jnp.array([1.0]), cj, jnp.array([0.0])])
    patch_active = jnp.concatenate([jnp.array([True]), ca, jnp.array([False])])
    patch_xl = jnp.concatenate([jnp.array([spec.ped_left]), cxl, jnp.array([0.0])])
    patch_xr = jnp.concatenate([jnp.array([spec.ped_right]), cxr, jnp.array([0.0])])
    patch_z = jnp.concatenate([jnp.array([0.0]), cz, jnp.array([0.0])])

    vx = jnp.stack([patch_xl, patch_xr], axis=1)  # (P, 2), t1=+x so v0 is left
    verts = jnp.stack(
        [vx, jnp.zeros((P, 2)), patch_z[:, None] * jnp.ones((1, 2))], axis=2
    )  # (P, 2, 3)
    vert_mask = jnp.stack([patch_active, patch_active], axis=1)  # (P, 2)

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


def build_system(spec: LatticeSpec, state: State):
    """Nondimensional (A, w_dead, G, L, W) for a lattice state, pure JNP.

    Reproduces keystone.mechanics.assemble for this scene family exactly:
    same sign convention, same [Fx, Fz, Ty] row layout about each block's
    own com, same (p V + v) ncomp + c column layout, the same
    nondimensionalization (L is the bounding-box diagonal of active patch
    vertices and coms, W is the total active weight with DEFAULT_G), and
    the same 2D cone rows.
    """
    nb = spec.n_blocks
    P = spec.P_max
    V = spec.V
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

    # 6. Total weight and dead load, nondimensional.
    W = jnp.sum(jnp.where(block_mask, mass, 0.0)) * spec.g
    w_dead = jnp.zeros(3 * nb)
    fz_rows = 3 * jnp.arange(nb) + 1
    w_dead = w_dead.at[fz_rows].set(jnp.where(block_mask, -mass * spec.g / W, 0.0))

    # 7. Equilibrium map A. Force on block i at a vertex is -n n_hat + u t1;
    #    block j gets the negative. n_hat = +z and t1 = +x for every patch.
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
    # c=1 is t1 = (1,0,0).
    fi = jnp.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])  # (ncomp, 3)
    fvec = sgn[:, :, None, None] * fi[None, None, :, :]  # (nb, P, ncomp, 3)
    d = verts_nd[None, :, :, :] - com_nd[:, None, None, :]  # (nb, P, V, 3)
    Fx = fvec[:, :, None, :, 0]  # (nb, P, 1, ncomp)
    Fz = fvec[:, :, None, :, 2]
    dz = d[:, :, :, None, 2]  # (nb, P, V, 1)
    dxc = d[:, :, :, None, 0]
    Ty = dz * Fx - dxc * Fz  # (nb, P, V, ncomp)
    Fxb = jnp.broadcast_to(Fx, Ty.shape)
    Fzb = jnp.broadcast_to(Fz, Ty.shape)
    entry = jnp.stack([Fxb, Fzb, Ty], axis=-1)  # (nb, P, V, ncomp, 3)
    entry = entry * act_bp[:, :, None, None, None]
    A = jnp.transpose(entry, (0, 4, 1, 2, 3)).reshape(3 * nb, P * V * spec.ncomp)

    # 8. Cone rows from the per-patch friction of the table.
    G = _cone_2d(mu_vec, P, V)
    return A, w_dead, G, L, W


def _reach_ok(spec: LatticeSpec, state: State, layer, xidx):
    """Placement reachability for modes "drop" and "slide". Pure JNP bool.

    Reachability is evaluated against the state BEFORE the placement, so
    the same final set can be reachable in one build order and not in
    another. That order dependence is the point of these modes.

    Drop: the vertical column above the target cell must be clear. A
    placed cube at a layer strictly above the target blocks when its
    x-interval overlaps the target's x-interval with nonzero width (open
    overlap; touching edges do not block). The pedestal never blocks, it
    sits below layer 0.

    Slide: legal when the drop column is clear OR a lateral corridor
    exists at the target layer, to the right or to the left. The moving
    cube is one unit tall and slides at its own layer: layer heights are
    exact, so a layer-(L+1) bridge clears it by construction and cubes
    above or below the target layer never block a slide. A corridor is
    blocked when any placed layer-L cube's interval meets, with nonzero
    width, the interval the cube sweeps from the target out past every
    placed extent on that side.

    Modeling assumptions, recorded in KNOWN_LIMITS.md: clearances are the
    exact unit-cell gaps of the lattice, straight-line axis-aligned
    approach motions only, and no gripper or tool clearance is modeled.
    """
    dx = spec.dx
    x = jnp.asarray(xidx, jnp.float64) * dx
    layer = jnp.asarray(layer, jnp.int32)
    px = state.placed_xidx.astype(jnp.float64) * dx
    pl = state.placed_layer
    msk = state.placed_mask

    # Drop column. Overlaps on this grid are multiples of dx, so 0.5 dx
    # separates touching (0) from a real overlap (>= dx).
    ov = jnp.minimum(x + 0.5, px + 0.5) - jnp.maximum(x - 0.5, px - 0.5)
    above = msk & (pl > layer)
    drop_clear = ~jnp.any(above & (ov > 0.5 * dx))
    if spec.mode == "drop":
        return drop_clear

    # Slide corridors at the target layer. Going right the cube sweeps
    # (x - 0.5, +inf); a same-layer cube at px meets that sweep with
    # nonzero width iff px - x > -1. Mirror for the left sweep.
    same = msk & (pl == layer)
    d = px - x
    right_blocked = jnp.any(same & (d > -1.0 + 0.5 * dx))
    left_blocked = jnp.any(same & (d < 1.0 - 0.5 * dx))
    return drop_clear | ~right_blocked | ~left_blocked


def is_legal(spec: LatticeSpec, state: State, layer, xidx):
    """Geometry-only legality of placing a cube at (layer, xidx).

    Pure JNP, returns a scalar bool. Mirrors the frozen legality rules:
    room, layer reachability, support overlap >= 2 dx, same-layer
    clearance greater than one block width, and in bounds.

    spec.mode adds a placement-reachability conjunct: "drop" requires a
    clear vertical column above the target, "slide" requires the column
    or a lateral corridor at the target layer (see _reach_ok). The mode
    is a compile-time constant; "static" (the default) skips the check
    entirely, so the default trace is the pre-mode code path unchanged.
    """
    dx = spec.dx
    x = jnp.asarray(xidx, jnp.float64) * dx
    layer = jnp.asarray(layer, jnp.int32)

    room = state.count < spec.n_max
    max_L = jnp.max(jnp.where(state.placed_mask, state.placed_layer, -1))
    layer_ok = (layer >= 0) & (layer <= spec.n_max - 1) & (layer <= max_L + 1)

    # Support. Layer 0 rests on the pedestal top [-6, 0]; layer L rests on a
    # layer-(L-1) cube. Overlap must reach 2 dx.
    ped_ov = jnp.minimum(x + 0.5, spec.ped_right) - jnp.maximum(
        x - 0.5, spec.ped_left
    )
    support0 = ped_ov > 1.5 * dx  # >= 2 dx on the grid
    below = state.placed_mask & (state.placed_layer == layer - 1)
    bx = state.placed_xidx.astype(jnp.float64) * dx
    below_ov = jnp.minimum(x + 0.5, bx + 0.5) - jnp.maximum(x - 0.5, bx - 0.5)
    supportL = jnp.any(below & (below_ov > 1.5 * dx))
    support_ok = jnp.where(layer == 0, support0, supportL)

    # Same-layer clearance: strictly more than one block width apart, so no
    # footprint overlap and no shared vertical face.
    same = state.placed_mask & (state.placed_layer == layer)
    sx = state.placed_xidx.astype(jnp.float64) * dx
    gaps = jnp.where(same, jnp.abs(x - sx), 1e30)
    clear_ok = jnp.min(gaps) > 1.0 + 0.5 * dx

    bounds_ok = (xidx >= spec.j_lo) & (xidx <= spec.j_hi)
    base = room & layer_ok & support_ok & clear_ok & bounds_ok
    if spec.mode == "static":
        return base
    return base & _reach_ok(spec, state, layer, xidx)


def expand_kernel(
    spec: LatticeSpec,
    state: State,
    cand_layers,
    cand_xidx,
    eps_reg,
    tol_cone,
    *,
    solver_tol,
    max_iter,
):
    """Legality, P4 margin, and a certified flag for each candidate.

    vmaps [place -> build_system -> margin_core] over the candidate arrays.
    Illegal candidates are still solved for shape uniformity, then their
    margin is set to +inf and their certified flag to False. certified is
    the margin_batch meaning: cone-admissible (viol <= tol_cone) and the
    margin finite. Returns (legal (M,), margins (M,), certified (M,)).

    jit once per (spec, len(cand), solver_tol, max_iter) through the
    _expand_jit cache below.
    """
    fn = _expand_jit(spec, int(cand_layers.shape[0]), float(solver_tol), int(max_iter))
    return fn(state, cand_layers, cand_xidx, eps_reg, tol_cone)


@functools.lru_cache(maxsize=None)
def _expand_jit(spec: LatticeSpec, ncand: int, solver_tol: float, max_iter: int):
    """Build and cache the jitted, vmapped expansion for a fixed shape."""

    def one(state, cl, cx, eps_reg, tol_cone):
        legal = is_legal(spec, state, cl, cx)
        ns = place(spec, state, cl, cx)
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

    vmapped = jax.vmap(one, in_axes=(None, 0, 0, None, None))
    return jax.jit(vmapped)


@functools.lru_cache(maxsize=None)
def _expand_batch_jit(spec: LatticeSpec, ncand: int, solver_tol: float, max_iter: int):
    """Doubly-vmapped expand: over leaves then candidates, shapes (K, M)."""

    def one(state, cl, cx, eps_reg, tol_cone):
        legal = is_legal(spec, state, cl, cx)
        ns = place(spec, state, cl, cx)
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

    over_cand = jax.vmap(one, in_axes=(None, 0, 0, None, None))
    over_leaf = jax.vmap(over_cand, in_axes=(0, None, None, None, None))
    return jax.jit(over_leaf)


def expand_kernel_batch(
    spec: LatticeSpec,
    states: State,
    cand_layers,
    cand_xidx,
    eps_reg,
    tol_cone,
    *,
    solver_tol,
    max_iter,
):
    """expand_kernel batched over K leaf states: the literal (K, M) primitive.

    states is a State pytree with a leading axis of K. Returns
    (legal, margins, certified), each (K, M). This is the GPU-ready
    full-grid expand the throughput benchmark times. On a GPU the K * M
    candidate solves fill the device in one call; on CPU the vmap runs them
    in sequence, so the wall time scales with K * M.
    """
    fn = _expand_batch_jit(
        spec, int(cand_layers.shape[0]), float(solver_tol), int(max_iter)
    )
    return fn(states, cand_layers, cand_xidx, eps_reg, tol_cone)


@functools.lru_cache(maxsize=None)
def _legal_grid_jit(spec: LatticeSpec, ncand: int):
    """Cached jit of legality over a fixed candidate grid, batched over leaves."""

    def one(state, cl, cx):
        return is_legal(spec, state, cl, cx)

    over_cand = jax.vmap(one, in_axes=(None, 0, 0))
    over_leaf = jax.vmap(over_cand, in_axes=(0, None, None))
    return jax.jit(over_leaf)


def legal_grid(spec: LatticeSpec, states: State, cand_layers, cand_xidx):
    """(B, M) legality for a batch of B states over the fixed candidate grid.

    Pure geometry, no QP, so this pass is cheap. The search uses it to find
    the legal frontier before spending any solve on it.
    """
    fn = _legal_grid_jit(spec, int(cand_layers.shape[0]))
    return fn(states, cand_layers, cand_xidx)


@functools.lru_cache(maxsize=None)
def _solve_states_jit(spec: LatticeSpec, B: int, solver_tol: float, max_iter: int):
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
    spec: LatticeSpec, states: State, eps_reg, tol_cone, *, solver_tol, max_iter
):
    """P4 margins and certified flags for a batch of B lattice states.

    states is a State pytree with a leading batch axis. Returns
    (margins (B,), certified (B,)). This is the certified-path solve the
    search runs on its legal, uncached frontier.
    """
    b = int(states.count.shape[0])
    fn = _solve_states_jit(spec, b, float(solver_tol), int(max_iter))
    return fn(states, eps_reg, tol_cone)


@functools.lru_cache(maxsize=None)
def _solve_states_pdhg_jit(spec: LatticeSpec, B: int, iters: int, accel: bool):
    """Cached jit of [build_system -> pdhg_margin] over a batch of B states.

    Warm starts are passed as arrays (zeros for a cold start), so the trace
    is uniform and jit caches on (spec, B, iters, accel) only.
    """

    def one(state, eps_reg, tol_cone, f0, y0):
        A, w_dead, G, L, W = build_system(spec, state)
        margin, f, y, viol = pdhg_margin(
            A, w_dead, G, eps_reg, iters=iters, f0=f0, y0=y0, accel=accel
        )
        cert = (viol <= tol_cone) & jnp.isfinite(margin)
        return margin, cert, f, y

    return jax.jit(jax.vmap(one, in_axes=(0, None, None, 0, 0)))


def margins_of_states_pdhg(
    spec: LatticeSpec, states: State, eps_reg, tol_cone, *, iters, accel, f0, y0
):
    """First-order P4 screen margins and cert flags for a batch of states.

    Mirrors margins_of_states but uses the pdhg screener and also returns
    the final (f, y) iterates so descendants can warm start from them. f0
    and y0 are (B, nf) and (B, ncone) warm starts; pass zeros for cold.
    Returns (margins (B,), certified (B,), fs (B, nf), ys (B, ncone)).
    """
    b = int(states.count.shape[0])
    fn = _solve_states_pdhg_jit(spec, b, int(iters), bool(accel))
    return fn(states, eps_reg, tol_cone, f0, y0)


def stack_states(states_list):
    """Stack a python list of State pytrees along a new leading axis."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states_list)


def batch_states(spec: LatticeSpec, keys):
    """Build a batched State pytree from a list of placement-tuple keys.

    keys is a list of tuples of (layer, xidx) pairs. Uses numpy to fill the
    padded arrays in one shot, so a large frontier is cheap to assemble.
    """
    import numpy as np

    b = len(keys)
    n = spec.n_max
    lay = np.zeros((b, n), dtype=np.int32)
    xid = np.zeros((b, n), dtype=np.int32)
    msk = np.zeros((b, n), dtype=bool)
    cnt = np.zeros(b, dtype=np.int32)
    for r, key in enumerate(keys):
        for c, (L, j) in enumerate(key):
            lay[r, c] = int(L)
            xid[r, c] = int(j)
            msk[r, c] = True
        cnt[r] = len(key)
    return State(
        placed_layer=jnp.asarray(lay),
        placed_xidx=jnp.asarray(xid),
        placed_mask=jnp.asarray(msk),
        count=jnp.asarray(cnt),
    )


def harmonic(n: int) -> float:
    """Overhang of the simple stack in block widths: sum_{k=1..n} 1/(2k)."""
    return sum(1.0 / (2.0 * k) for k in range(1, n + 1))


def overhang(placements, dx: float = DX) -> float:
    """Rightmost cube edge beyond the pedestal edge at x = 0, or -inf."""
    if not placements:
        return float("-inf")
    return max(j * dx + 0.5 for (_, j) in placements)
