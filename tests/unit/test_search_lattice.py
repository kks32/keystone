"""Agreement and unit tests for the jittable lattice environment.

The agreement suite is the correctness gate: for 200 random reachable
lattice states it pins keystone.search.lattice.build_system to the
certified host pipeline (build_assembly + assemble + margin_core). The
margins agree to 1e-9, the active-patch masks agree exactly, and (A, w)
agree to 1e-12 after aligning patches by (node pair, vertex positions).

The rest of the file checks the legality rules against an independent
reference, the fixed padded shapes, expand_kernel masking, and node-order
invariance under placement order.
"""

import numpy as np
import pytest

from keystone import Tolerances, assemble, box_2d, build_assembly
from keystone.solve.batch_jax import margin_batch, margin_core
from keystone.search import lattice as LT

TOL = Tolerances()
DX = 1.0 / 24.0
MU = 0.7
SOLVER_TOL = 1e-9
MAX_ITER = 100


# --- host pipeline helpers ------------------------------------------------


def host_boxes(key):
    """Pedestal plus one unit cube per placement, in sorted (L, j) order."""
    boxes = [box_2d(6, 1, -3, 0.5)]
    for (L, j) in sorted(key):
        boxes.append(box_2d(1, 1, j * DX, 1.5 + L))
    return boxes


def host_system(spec, key):
    boxes = host_boxes(key)
    asm = build_assembly(
        boxes,
        mu=MU,
        tol=TOL,
        dim=2,
        pad_blocks=spec.n_blocks,
        pad_patches=spec.P_max,
        pad_verts=2,
    )
    return asm, assemble(asm, TOL, cone="linear2d")


def host_active_patches(asm):
    """Set of active host patches as (i, j, (v0x, v1x), z), rounded."""
    out = []
    for p in range(asm.n_patches):
        if not asm.patch_mask[p]:
            continue
        i, j = int(asm.patch_blocks[p, 0]), int(asm.patch_blocks[p, 1])
        vx = tuple(round(float(asm.verts[p, v, 0]), 7) for v in range(2))
        z = round(float(asm.verts[p, 0, 2]), 7)
        out.append((i, j, vx, z))
    return out


def host_patch_col_index(asm):
    """Map (i, j) -> host column block start (4 columns per patch)."""
    m = {}
    for p in range(asm.n_patches):
        if not asm.patch_mask[p]:
            continue
        i, j = int(asm.patch_blocks[p, 0]), int(asm.patch_blocks[p, 1])
        m[(i, j)] = 4 * p  # ncomp * V = 4 columns per patch
    return m


# --- mine helpers ---------------------------------------------------------


def mine_active_patches(spec, key):
    """Set of active patches from patch_table, matching host_active_patches."""
    pt = LT.patch_table(spec, LT.state_from_placements(spec, key))
    pi = np.asarray(pt.patch_i)
    pj = np.asarray(pt.patch_j)
    pa = np.asarray(pt.patch_active)
    verts = np.asarray(pt.verts)
    out = []
    colmap = {}
    for p in range(pi.shape[0]):
        if not pa[p]:
            continue
        i, j = int(round(pi[p])), int(round(pj[p]))
        vx = tuple(round(float(verts[p, v, 0]), 7) for v in range(2))
        z = round(float(verts[p, 0, 2]), 7)
        out.append((i, j, vx, z))
        colmap[(i, j)] = 4 * p
    return out, colmap


# --- reachable-state generator -------------------------------------------


def random_reachable_states(spec, n_states, seed=0):
    """Build reachable states by random legal rollouts using the env itself.

    Records every intermediate prefix, so the set spans all depths. Returns
    a list of placement-tuple keys, including the empty state and full
    stacks.
    """
    rng = np.random.default_rng(seed)
    cand_L_j, cand_J_j = LT.action_grid(spec)
    cand_L = np.asarray(cand_L_j)
    cand_J = np.asarray(cand_J_j)
    keys = set()
    keys.add(())
    while len(keys) < n_states:
        placements = []
        for _ in range(spec.n_max):
            legal = np.asarray(
                LT.legal_grid(
                    spec, LT.batch_states(spec, [tuple(placements)]),
                    cand_L_j, cand_J_j,
                )
            )[0]
            idx = np.nonzero(legal)[0]
            if idx.size == 0:
                break
            pick = int(rng.choice(idx))
            placements.append((int(cand_L[pick]), int(cand_J[pick])))
            keys.add(tuple(sorted(placements)))
            if len(keys) >= n_states:
                break
    return list(keys)[:n_states]


# --- independent legality reference ---------------------------------------


def ref_legal(spec, key, layer, xidx):
    """Frozen legality rules, plain python, independent of is_legal."""
    dx = spec.dx
    x = xidx * dx
    if len(key) >= spec.n_max:
        return False
    if not (spec.j_lo <= xidx <= spec.j_hi):
        return False
    layers = [L for (L, _) in key]
    max_L = max(layers) if layers else -1
    if not (0 <= layer <= spec.n_max - 1 and layer <= max_L + 1):
        return False
    # support
    if layer == 0:
        ov = min(x + 0.5, spec.ped_right) - max(x - 0.5, spec.ped_left)
        support = ov > 1.5 * dx
    else:
        support = False
        for (L, j) in key:
            if L == layer - 1:
                bx = j * dx
                ov = min(x + 0.5, bx + 0.5) - max(x - 0.5, bx - 0.5)
                if ov > 1.5 * dx:
                    support = True
                    break
    if not support:
        return False
    # same-layer clearance, strictly more than one block width
    for (L, j) in key:
        if L == layer and abs(x - j * dx) <= 1.0 + 0.5 * dx:
            return False
    return True


# =========================================================================
# Agreement suite: the correctness gate.
# =========================================================================


class TestAgreement:
    spec = LT.LatticeSpec(n_max=5, dx=DX)
    keys = None

    @classmethod
    def setup_class(cls):
        cls.keys = random_reachable_states(cls.spec, 200, seed=0)

    def test_two_hundred_reachable_states(self):
        assert len(self.keys) == 200
        # Depth spread: not all the same size.
        depths = {len(k) for k in self.keys}
        assert len(depths) >= 3

    def test_margins_agree(self):
        spec = self.spec
        # Mine, batched through the certified path.
        states = LT.batch_states(spec, self.keys)
        m_mine, cert_mine = LT.margins_of_states(
            spec, states, TOL.eps_reg, TOL.tol_cone,
            solver_tol=SOLVER_TOL, max_iter=MAX_ITER,
        )
        m_mine = np.asarray(m_mine)
        cert_mine = np.asarray(cert_mine)
        # Host, batched through margin_batch on the same systems.
        As, ws, Gs = [], [], []
        for key in self.keys:
            _asm, hsys = host_system(spec, key)
            As.append(hsys.A)
            ws.append(hsys.w_dead)
            Gs.append(hsys.G)
        import jax.numpy as jnp

        m_host, cert_host = margin_batch(
            jnp.stack(As), jnp.stack(ws), jnp.stack(Gs),
            TOL.eps_reg, tol_cone=TOL.tol_cone, max_iter=MAX_ITER,
        )
        m_host = np.asarray(m_host)
        cert_host = np.asarray(cert_host)
        assert np.max(np.abs(m_mine - m_host)) < 1e-9
        # Certified (cone-admissible and finite) flags match exactly.
        assert np.array_equal(cert_mine, cert_host)

    def test_masks_and_matrices_agree(self):
        spec = self.spec
        max_dA = 0.0
        max_dw = 0.0
        for key in self.keys:
            asm, hsys = host_system(spec, key)
            state = LT.state_from_placements(spec, key)
            A, w, G, L, W = LT.build_system(spec, state)
            A = np.asarray(A)
            w = np.asarray(w)
            hA = np.asarray(hsys.A)
            hw = np.asarray(hsys.w_dead)

            # Active-patch masks agree exactly (as sets of (i, j, verts, z)).
            hpatch = host_active_patches(asm)
            mpatch, mcol = mine_active_patches(spec, key)
            assert sorted(hpatch) == sorted(mpatch), key

            # Block count and dead load agree directly (block order matches).
            assert int(asm.block_mask.sum()) == 1 + len(key)
            max_dw = max(max_dw, float(np.max(np.abs(w - hw))))

            # Align A columns patch by patch (matched by node pair).
            hcol = host_patch_col_index(asm)
            for (i, j, _vx, _z) in hpatch:
                hc = hcol[(i, j)]
                mc = mcol[(i, j)]
                block_h = hA[:, hc : hc + 4]
                block_m = A[:, mc : mc + 4]
                max_dA = max(max_dA, float(np.max(np.abs(block_h - block_m))))
            # L and W agree.
            assert abs(float(L) - float(np.asarray(hsys.L))) < 1e-9
            assert abs(float(W) - float(np.asarray(hsys.W))) < 1e-6
        assert max_dw < 1e-12, max_dw
        assert max_dA < 1e-12, max_dA


# =========================================================================
# Legality unit tests.
# =========================================================================


class TestLegality:
    spec = LT.LatticeSpec(n_max=6, dx=DX)

    def test_matches_reference_on_reachable_states(self):
        spec = self.spec
        keys = random_reachable_states(spec, 60, seed=1)
        cand_L, cand_J = LT.action_grid(spec)
        cand_L = np.asarray(cand_L)
        cand_J = np.asarray(cand_J)
        for key in keys:
            legal = np.asarray(
                LT.legal_grid(spec, LT.batch_states(spec, [key]), *LT.action_grid(spec))
            )[0]
            ref = np.array(
                [ref_legal(spec, key, int(cand_L[i]), int(cand_J[i]))
                 for i in range(cand_L.shape[0])]
            )
            assert np.array_equal(legal, ref), key

    def test_layer0_support_needs_pedestal_overlap(self):
        spec = self.spec
        empty = LT.empty_state(spec)
        # x = 0.5 spans [0, 1], no overlap with pedestal top [-6, 0].
        j_off = round(0.5 / DX)
        assert not bool(LT.is_legal(spec, empty, 0, j_off))
        # x = -0.5 spans [-1, 0], overlaps by 0.5 >= 2 dx.
        assert bool(LT.is_legal(spec, empty, 0, -j_off))

    def test_same_layer_touching_forbidden(self):
        spec = self.spec
        # Base cube at x = -2.0, well inside the pedestal top [-6, 0].
        st = LT.place(spec, LT.empty_state(spec), 0, -48)  # x = -2.0
        # 24 grid steps away is exactly one block width: touching, forbidden.
        assert not bool(LT.is_legal(spec, st, 0, -24))  # x = -1.0, distance 1.0
        # 25 grid steps away is more than one block width: allowed.
        assert bool(LT.is_legal(spec, st, 0, -23))  # x = -0.9583, distance 1.0417

    def test_layer_reachability(self):
        spec = self.spec
        empty = LT.empty_state(spec)
        # Cannot place at layer 1 with nothing below.
        assert not bool(LT.is_legal(spec, empty, 1, 0))
        st = LT.place(spec, empty, 0, 0)
        assert bool(LT.is_legal(spec, st, 1, 0))
        # Layer 2 still unreachable (top layer is 1).
        assert not bool(LT.is_legal(spec, st, 2, 0))

    def test_node_order_invariant_to_placement_order(self):
        spec = LT.LatticeSpec(n_max=5, dx=DX)
        key = [(0, -6), (1, 6), (0, 18)]
        A1, w1, G1, L1, W1 = LT.build_system(
            spec, LT.state_from_placements(spec, key)
        )
        A2, w2, G2, L2, W2 = LT.build_system(
            spec, LT.state_from_placements(spec, list(reversed(key)))
        )
        # Internal sort makes the system independent of insertion order.
        assert np.max(np.abs(np.asarray(A1) - np.asarray(A2))) < 1e-14
        assert np.max(np.abs(np.asarray(w1) - np.asarray(w2))) < 1e-14


# =========================================================================
# Shapes and kernel masking.
# =========================================================================


class TestShapesAndKernel:
    def test_derived_shapes(self):
        spec = LT.LatticeSpec(n_max=4, dx=DX)
        assert spec.V == 2 and spec.ncomp == 2
        assert spec.P_max == 2 * 4 + 2
        assert spec.rows == 3 * (4 + 1)
        assert spec.nf == spec.ncomp * spec.P_max * spec.V
        assert spec.ncone == 3 * spec.P_max * spec.V
        assert spec.M == spec.n_layers * spec.n_pos
        state = LT.empty_state(spec)
        A, w, G, L, W = LT.build_system(spec, state)
        assert A.shape == (spec.rows, spec.nf)
        assert w.shape == (spec.rows,)
        assert G.shape == (spec.ncone, spec.nf)

    def test_expand_kernel_masks_illegal(self):
        spec = LT.LatticeSpec(n_max=4, dx=DX)
        state = LT.empty_state(spec)
        cand_L, cand_J = LT.action_grid(spec)
        legal, margins, cert = LT.expand_kernel(
            spec, state, cand_L, cand_J, TOL.eps_reg, TOL.tol_cone,
            solver_tol=SOLVER_TOL, max_iter=50,
        )
        legal = np.asarray(legal)
        margins = np.asarray(margins)
        cert = np.asarray(cert)
        # Illegal candidates carry +inf margin and are not certified.
        assert np.all(np.isinf(margins[~legal]))
        assert not np.any(cert[~legal])
        # Legal layer-0 candidates exist and some are feasible.
        assert legal.sum() > 0
        feasible = (margins <= TOL.tol_feas) & cert
        assert feasible.sum() > 0
        # A feasible layer-0 cube well inside the pedestal has a small margin.
        cand_L = np.asarray(cand_L)
        cand_J = np.asarray(cand_J)
        centered = np.where((cand_L == 0) & (cand_J == -72 // 2))[0]
        if centered.size:
            assert margins[centered[0]] <= TOL.tol_feas

    def test_full_grid_expand_matches_gathered_solve(self):
        # The full-grid kernel and the per-state solve agree on legal states.
        spec = LT.LatticeSpec(n_max=4, dx=DX)
        state = LT.place(spec, LT.empty_state(spec), 0, -12)
        cand_L, cand_J = LT.action_grid(spec)
        legal, margins, cert = LT.expand_kernel(
            spec, state, cand_L, cand_J, TOL.eps_reg, TOL.tol_cone,
            solver_tol=SOLVER_TOL, max_iter=MAX_ITER,
        )
        legal = np.asarray(legal)
        margins = np.asarray(margins)
        cand_L = np.asarray(cand_L)
        cand_J = np.asarray(cand_J)
        # Rebuild each legal child directly and compare margins.
        idx = np.nonzero(legal)[0][:20]
        keys = [((0, -12), (int(cand_L[i]), int(cand_J[i]))) for i in idx]
        states = LT.batch_states(spec, [tuple(sorted(k)) for k in keys])
        m2, _c2 = LT.margins_of_states(
            spec, states, TOL.eps_reg, TOL.tol_cone,
            solver_tol=SOLVER_TOL, max_iter=MAX_ITER,
        )
        assert np.max(np.abs(margins[idx] - np.asarray(m2))) < 1e-9
