"""Assemble the nondimensional equilibrium system (A, w, G) from an Assembly.

Sign convention (CLAUDE.md Section 3, enforced by tests):
For vertex k on a patch joining nodes (i, j), the force exerted by
block j on block i is

    F_k = -n_k * n_hat + u_k * t1_hat + v_k * t2_hat,   n_k >= 0.

Compression is positive n_k. Block i receives F_k, block j receives
-F_k. The ground (node 0) has no equilibrium rows.

Unknown layout (frozen contract):
    ncomp = 2 in 2D (n, u), 3 in 3D (n, u, v)
    index of component c of vertex v of patch p:  (p * V + v) * ncomp + c

Row layout (frozen contract):
    2D: 3 rows per block, node id b -> rows [3(b-1) : 3b] = [Fx, Fz, Ty]
    3D: 6 rows per block, [Fx, Fy, Fz, Tx, Ty, Tz]
Torques are taken about each block's own center of mass, so gravity
contributes no torque row entries.

Nondimensionalization (charter item 7): lengths divide by L (bounding
box diagonal), forces by W (total weight of active blocks). w_dead has
total magnitude 1 by construction. Solvers report SI by scaling back
with the stored L and W.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..geometry.assembly import Assembly, bbox_diagonal
from ..geometry.tolerances import Tolerances
from .cones import cone_matrix_2d, cone_matrix_pyramid
from .loads import DEFAULT_G, dead_and_live_loads


@dataclass(frozen=True)
class EquilibriumSystem:
    """Nondimensional equilibrium problem, ready for any solver backend.

    A:      (nrows, nf) equilibrium map, nrows = rows_per_block * N,
            nf = ncomp * P * V. Masked blocks give zero rows, masked
            vertices give zero columns.
    w_dead: (nrows,) self-weight, nondimensional.
    w_live: (nrows,) unit horizontal pseudo-static load (+x per unit
            weight of each block), nondimensional.
    G:      (ncone, nf) friction cone rows, G f <= 0.
    mu:     (P,) per-patch friction, kept for P3 cone rebuilds.
    vert_mask: (P, V) bool, kept for cone rebuilds.
    L:      scalar, meters. W: scalar, newtons.
    dim:    2 or 3, static metadata.
    """

    A: jnp.ndarray
    w_dead: jnp.ndarray
    w_live: jnp.ndarray
    G: jnp.ndarray
    mu: jnp.ndarray
    vert_mask: jnp.ndarray
    L: jnp.ndarray
    W: jnp.ndarray
    dim: int
    cone: str
    k: int


jax.tree_util.register_dataclass(
    EquilibriumSystem,
    data_fields=["A", "w_dead", "w_live", "G", "mu", "vert_mask", "L", "W"],
    meta_fields=["dim", "cone", "k"],
)


def assemble(
    assembly: Assembly,
    tol: Tolerances,
    *,
    cone: str = "linear2d",
    k: int = 8,
) -> EquilibriumSystem:
    """Build the nondimensional (A, w_dead, w_live, G) for an Assembly.

    The cone model is explicit at the call site (charter item 4):
    cone="linear2d" for 2D (exact), cone="pyramid" with k facets for 3D
    (inscribed, conservative). There is no silent fallback. cone="socp"
    raises NotImplementedError until a conic backend lands.
    """
    dim = assembly.dim
    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {dim!r}")
    if cone == "socp":
        raise NotImplementedError(
            "cone='socp' needs a conic backend (Clarabel); not available in "
            "this slice. Use cone='linear2d' (2D) or cone='pyramid' (3D)."
        )
    if dim == 2 and cone != "linear2d":
        raise ValueError(
            f"dim=2 requires cone='linear2d', got cone={cone!r}"
        )
    if dim == 3 and cone != "pyramid":
        raise ValueError(
            f"dim=3 requires cone='pyramid', got cone={cone!r}"
        )

    ncomp = 2 if dim == 2 else 3
    rpb = 3 if dim == 2 else 6
    n_blocks = assembly.n_blocks
    n_patches = assembly.n_patches
    verts_per = assembly.verts_per_patch

    L = bbox_diagonal(assembly)
    active = np.asarray(assembly.block_mask)
    mass = np.asarray(assembly.mass)
    W = float(mass[active].sum()) * DEFAULT_G

    verts = np.asarray(assembly.verts) / L
    com = np.asarray(assembly.com) / L
    normal = np.asarray(assembly.normal)
    t1 = np.asarray(assembly.t1)
    t2 = np.asarray(assembly.t2)
    pblocks = np.asarray(assembly.patch_blocks)
    pmask = np.asarray(assembly.patch_mask)
    vmask = np.asarray(assembly.vert_mask)

    nf = ncomp * n_patches * verts_per
    nrows = rpb * n_blocks
    A = np.zeros((nrows, nf), dtype=np.float64)
    # Rows kept from the full 6-vector [Fx, Fy, Fz, Tx, Ty, Tz].
    sel = [0, 2, 4] if dim == 2 else [0, 1, 2, 3, 4, 5]

    for p in range(n_patches):
        if not pmask[p]:
            continue
        i, j = int(pblocks[p, 0]), int(pblocks[p, 1])
        n_hat = normal[p]
        # Force basis on block i per unit of each component (n, u, v):
        # F_i = -n n_hat + u t1 + v t2.
        basis = [-n_hat, t1[p]]
        if dim == 3:
            basis.append(t2[p])
        for v in range(verts_per):
            if not vmask[p, v]:
                continue
            r = verts[p, v]
            base_col = (p * verts_per + v) * ncomp
            for c, f_i in enumerate(basis):
                col = base_col + c
                if i != 0:
                    bi = i - 1
                    torque = np.cross(r - com[bi], f_i)
                    A[rpb * bi : rpb * bi + rpb, col] += np.concatenate(
                        [f_i, torque]
                    )[sel]
                bj = j - 1
                f_j = -f_i
                torque = np.cross(r - com[bj], f_j)
                A[rpb * bj : rpb * bj + rpb, col] += np.concatenate(
                    [f_j, torque]
                )[sel]

    w_dead, w_live = dead_and_live_loads(assembly.mass, assembly.block_mask, W, dim)

    mu = jnp.asarray(assembly.mu)
    if dim == 2:
        G = cone_matrix_2d(mu, n_patches, verts_per)
    else:
        G = cone_matrix_pyramid(mu, n_patches, verts_per, k)

    return EquilibriumSystem(
        A=jnp.asarray(A),
        w_dead=jnp.asarray(w_dead),
        w_live=jnp.asarray(w_live),
        G=G,
        mu=mu,
        vert_mask=jnp.asarray(assembly.vert_mask),
        L=jnp.asarray(L),
        W=jnp.asarray(W),
        dim=dim,
        cone=cone,
        k=k,
    )
