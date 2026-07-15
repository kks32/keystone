"""Friction cone rows. G f <= 0 with the layout of assemble.py.

2D (exact, linear), per vertex with friction mu:
    -n <= 0
     u - mu * n <= 0
    -u - mu * n <= 0
Three rows per vertex, in this order, vertex-major then patch-major:
row index of rule r of vertex v of patch p is (p * V + v) * 3 + r.

Masked vertices keep their rows. Their A columns are zero and the P4
regularizer drives their variables to zero, which satisfies the cone.

3D pyramid cones (inscribed k-gon) share the layout: rows per vertex
are one non-penetration row (-n <= 0) followed by k facet rows. The
cone model is explicit at the call site; there is no silent fallback.
"""

import jax.numpy as jnp
import numpy as np


def _check_mu(mu):
    """mu must be finite and nonnegative. Arrays: every entry.

    mu is concrete at every call site (assemble time, or the P3 bisection
    which passes concrete floats), so a host-side check is safe and never
    runs under a jit trace.
    """
    mu_arr = np.asarray(mu, dtype=float)
    if not np.all(np.isfinite(mu_arr)) or np.any(mu_arr < 0.0):
        raise ValueError(f"mu must be finite and >= 0, got {mu!r}")


def _check_pyramid_k(k, allow_asymmetric):
    """Validate the pyramid facet count k.

    Standard API requires even k >= 4: an even facet count keeps the
    inscribed polygon symmetric under t -> -t, so the cone represents
    isotropic Coulomb friction. Odd k >= 3 breaks that symmetry (the
    friction limit differs between +t and -t) and is allowed only as an
    explicit experimental anisotropic option via allow_asymmetric=True.
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError(f"pyramid k must be an integer, got {k!r}")
    k = int(k)
    if k < 3:
        raise ValueError(f"pyramid k must be >= 3, got {k}")
    if not allow_asymmetric and (k < 4 or k % 2 != 0):
        raise ValueError(
            f"standard pyramid API requires even k >= 4, got k={k}; set "
            "allow_asymmetric=True for odd k (experimental anisotropic friction)"
        )
    return k


def _block_diagonal(blocks: jnp.ndarray) -> jnp.ndarray:
    """Lay per-vertex (PV, R, C) blocks on the block diagonal.

    Returns (R * PV, C * PV). Vertex vv occupies rows [R vv : R vv + R]
    and cols [C vv : C vv + C], matching the (p * V + v) ordering.
    """
    pv, r, c = blocks.shape
    full = jnp.einsum("vrc,vw->vrwc", blocks, jnp.eye(pv))
    return full.reshape(r * pv, c * pv)


def cone_matrix_2d(mu: jnp.ndarray, n_patches: int, verts_per_patch: int) -> jnp.ndarray:
    """(3 P V, 2 P V) cone matrix for 2D. mu has shape (P,).

    Rows per vertex, in order: [-n <= 0], [u - mu n <= 0], [-u - mu n <= 0].
    Columns per vertex are (n, u).
    """
    _check_mu(mu)
    mu = jnp.asarray(mu)
    pv = n_patches * verts_per_patch
    mu_v = jnp.repeat(mu, verts_per_patch)
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
    return _block_diagonal(blocks)


def cone_matrix_pyramid(
    mu: jnp.ndarray,
    n_patches: int,
    verts_per_patch: int,
    k: int,
    *,
    allow_asymmetric: bool = False,
) -> jnp.ndarray:
    """((1 + k) P V, 3 P V) inscribed k-gon cone matrix for 3D.

    Rows per vertex: [-n <= 0] then k facet rows
        u cos(th_j) + v sin(th_j) - mu cos(pi/k) n <= 0,
    with th_j = (2j + 1) pi / k, j = 0..k-1. Polygon vertices sit at
    angles 2 pi j / k on the true cone, so capacity along +t1 is exact
    and facet-mid directions are conservative by cos(pi/k). Columns per
    vertex are (n, u, v).

    k must be an even integer >= 4 (isotropic Coulomb). allow_asymmetric
    permits odd k >= 3 as an experimental anisotropic option; see
    _check_pyramid_k.
    """
    _check_mu(mu)
    k = _check_pyramid_k(k, allow_asymmetric)
    mu = jnp.asarray(mu)
    pv = n_patches * verts_per_patch
    mu_v = jnp.repeat(mu, verts_per_patch)
    j = jnp.arange(k)
    th = (2.0 * j + 1.0) * jnp.pi / k
    cos_th = jnp.cos(th)
    sin_th = jnp.sin(th)
    cos_pik = jnp.cos(jnp.pi / k)

    row0 = jnp.stack(
        [-jnp.ones(pv), jnp.zeros(pv), jnp.zeros(pv)], axis=-1
    )[:, None, :]
    n_coeff = -(mu_v[:, None] * cos_pik) * jnp.ones((pv, k))
    u_coeff = jnp.broadcast_to(cos_th, (pv, k))
    v_coeff = jnp.broadcast_to(sin_th, (pv, k))
    facets = jnp.stack([n_coeff, u_coeff, v_coeff], axis=-1)
    blocks = jnp.concatenate([row0, facets], axis=1)
    return _block_diagonal(blocks)
