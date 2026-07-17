"""Agreement and unit tests for the jittable lattice environment.

The agreement suite is the correctness gate: for 200 random reachable
lattice states it pins keystone.search.lattice.build_system to the
certified host pipeline (build_assembly + assemble + margin_core). The
margins agree to 1e-9, the active-patch masks agree exactly, and (A, w)
agree to 1e-12 after aligning patches by (node pair, vertex positions).

The rest of the file checks the legality rules against an independent
reference, the placement-reachability modes (drop, slide) against a second
independent reference, the fixed padded shapes, expand_kernel masking, and
node-order invariance under placement order.
"""

import numpy as np
import pytest

from keystone import Tolerances, assemble, box_2d, build_assembly
from keystone.solve.batch_jax import margin_batch, margin_core
from keystone.search import bnb
from keystone.search import lattice as LT

TOL = Tolerances()
DX = 1.0 / 24.0
MU = 0.7
DENSITY = LT.DENSITY
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


def ref_reach(spec, key, layer, xidx, mode):
    """Frozen reachability rules, plain python, independent of _reach_ok."""
    dx = spec.dx
    x = xidx * dx
    if mode == "static":
        return True
    # Drop column: a cube strictly above the target blocks when its
    # x-interval overlaps the target's with nonzero width (open overlap).
    drop_clear = True
    for (L, j) in key:
        if L > layer:
            px = j * dx
            ov = min(x + 0.5, px + 0.5) - max(x - 0.5, px - 0.5)
            if ov > 0.5 * dx:
                drop_clear = False
    if mode == "drop":
        return drop_clear
    # Slide corridors at the target layer; slide_clear also treats a
    # layer-(L+1) cube over the sweep as a blocker.
    layers = (layer,) if mode == "slide" else (layer, layer + 1)
    block = [j * dx for (L, j) in key if L in layers]
    right_blocked = any(px - x > -1.0 + 0.5 * dx for px in block)
    left_blocked = any(px - x < 1.0 - 0.5 * dx for px in block)
    return drop_clear or (not right_blocked) or (not left_blocked)


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
# Placement reachability (LatticeSpec.mode).
# =========================================================================


class TestReachability:
    DX12 = 1.0 / 12.0
    DX24 = 1.0 / 24.0

    def test_default_mode_is_static(self):
        assert LT.LatticeSpec(n_max=4).mode == "static"

    def test_bad_mode_rejected(self):
        with pytest.raises(ValueError, match="mode"):
            LT.LatticeSpec(n_max=4, mode="teleport")

    def test_matches_reference_under_modes(self):
        # Full legality under each mode equals the independent python
        # reference (static rules AND reachability) on random reachable
        # states generated under the same mode.
        for mode in ("drop", "slide", "slide_clear"):
            spec = LT.LatticeSpec(n_max=5, dx=self.DX12, mode=mode)
            keys = random_reachable_states(spec, 40, seed=2)
            cand_L, cand_J = LT.action_grid(spec)
            cand_L_np = np.asarray(cand_L)
            cand_J_np = np.asarray(cand_J)
            for key in keys:
                legal = np.asarray(
                    LT.legal_grid(spec, LT.batch_states(spec, [key]),
                                  cand_L, cand_J)
                )[0]
                ref = np.array(
                    [ref_legal(spec, key, int(cand_L_np[i]), int(cand_J_np[i]))
                     and ref_reach(spec, key, int(cand_L_np[i]),
                                   int(cand_J_np[i]), mode)
                     for i in range(cand_L_np.shape[0])]
                )
                assert np.array_equal(legal, ref), (mode, key)

    def test_modes_only_remove_actions(self):
        # drop implies slide_clear implies slide implies static, legality
        # elementwise on every tested state. Reachability is a pure
        # restriction and the modes are nested.
        specs = {
            m: LT.LatticeSpec(n_max=5, dx=self.DX12, mode=m)
            for m in ("static", "slide", "slide_clear", "drop")
        }
        keys = random_reachable_states(specs["static"], 40, seed=3)
        cand = LT.action_grid(specs["static"])
        for key in keys:
            grids = {
                m: np.asarray(
                    LT.legal_grid(s, LT.batch_states(s, [key]), *cand)
                )[0]
                for m, s in specs.items()
            }
            assert not np.any(grids["drop"] & ~grids["slide_clear"]), key
            assert not np.any(grids["slide_clear"] & ~grids["slide"]), key
            assert not np.any(grids["slide"] & ~grids["static"]), key

    def test_drop_blocks_under_bridge_clamp(self):
        # The certified static 31/24 clamp at n=4, dx=1/24 ends by sliding
        # the reacher (1, 19) under the layer-2 bridge at (2, -4). That
        # final placement must be illegal under drop and slide_clear (the
        # under-bridge press fit) and legal under slide.
        prefix = [(0, -2), (1, -14), (2, -4)]
        cases = (("static", True), ("slide", True),
                 ("slide_clear", False), ("drop", False))
        for mode, want in cases:
            spec = LT.LatticeSpec(n_max=4, dx=self.DX24, mode=mode)
            st = LT.state_from_placements(spec, prefix)
            assert bool(LT.is_legal(spec, st, 1, 19)) is want, mode

    def test_clamp_order_slide_legal_every_step(self):
        # The full certified clamp build order passes the per-step
        # reachability re-check under slide and fails only at the final
        # under-bridge step under drop.
        clamp = [(0, -2), (1, -14), (2, -4), (1, 19)]
        ok, flags = bnb.sequence_reachable(4, self.DX24, clamp, "slide")
        assert ok and flags == [True, True, True, True]
        ok, flags = bnb.sequence_reachable(4, self.DX24, clamp, "drop")
        assert not ok and flags == [True, True, True, False]

    def test_reachability_depends_on_order(self):
        # The same final set is drop-reachable when the reacher is placed
        # before the bridge above it. Order matters; sets do not carry it.
        reordered = [(0, -2), (1, 19), (1, -14), (2, -4)]
        ok, flags = bnb.sequence_reachable(4, self.DX24, reordered, "drop")
        assert ok and all(flags)

    def test_slide_blocked_on_both_sides(self):
        # Same-layer cubes left and right of the target plus a bridge over
        # it: no drop column, no corridor, so slide is illegal while the
        # static rules still accept the placement.
        spec_by_mode = {
            m: LT.LatticeSpec(n_max=6, dx=self.DX12, mode=m)
            for m in ("static", "slide", "drop")
        }
        key = [(0, -30), (0, 0), (1, -24)]  # x = -2.5, 0.0, bridge at -2.0
        for mode, want in (("static", True), ("slide", False), ("drop", False)):
            spec = spec_by_mode[mode]
            st = LT.state_from_placements(spec, key)
            assert bool(LT.is_legal(spec, st, 0, -15)) is want, mode

    def test_slide_clear_blocks_corridor_under_bridge(self):
        # A layer-1 bridge sits over the only open corridor to a layer-0
        # target and over part of the target itself. slide passes under it
        # (the bridge is above the slide plane); slide_clear treats the
        # zero-clearance pass as blocked; drop is blocked by the overlap.
        key = [(0, -30), (1, -24)]  # x = -2.5 and a bridge at x = -2.0
        cases = (("static", True), ("slide", True),
                 ("slide_clear", False), ("drop", False))
        for mode, want in cases:
            spec = LT.LatticeSpec(n_max=6, dx=self.DX12, mode=mode)
            st = LT.state_from_placements(spec, key)
            assert bool(LT.is_legal(spec, st, 0, -15)) is want, mode

    def test_touching_column_does_not_block_drop(self):
        # Open overlap: a cube above whose interval exactly touches the
        # target's does not block the drop; one grid step closer does.
        spec = LT.LatticeSpec(n_max=6, dx=self.DX12, mode="drop")
        st = LT.state_from_placements(spec, [(0, -30), (1, -24)])
        # Bridge spans [-2.5, -1.5]. Target (0, -12) spans [-1.5, -0.5].
        assert bool(LT.is_legal(spec, st, 0, -12))
        # Target (0, -13) spans [-1.5833, -0.5833]: overlap dx, blocked.
        assert not bool(LT.is_legal(spec, st, 0, -13))

    def test_legal_grid_matches_is_legal_under_drop(self):
        # The batched grid and the scalar path agree under a mode.
        spec = LT.LatticeSpec(n_max=4, dx=self.DX24, mode="drop")
        key = ((0, -2), (1, -14), (2, -4))
        cand_L, cand_J = LT.action_grid(spec)
        grid = np.asarray(
            LT.legal_grid(spec, LT.batch_states(spec, [key]), cand_L, cand_J)
        )[0]
        st = LT.state_from_placements(spec, key)
        cand_L_np = np.asarray(cand_L)
        cand_J_np = np.asarray(cand_J)
        for i in range(0, cand_L_np.shape[0], 7):
            one = bool(LT.is_legal(spec, st, int(cand_L_np[i]), int(cand_J_np[i])))
            assert one == bool(grid[i])


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


# =========================================================================
# Heterogeneous materials: per-slot density and friction.
# =========================================================================


def host_system_hetero(spec, placements, densities, mu, mu_ground, mu_by_slot):
    """Host (A, w, G) for a state under heterogeneous materials.

    placements is the slot-order list of (layer, index) cells. densities and
    mu_by_slot are per placement slot. The cubes are sorted by (layer, index)
    to fix node ids, and each cube carries its own slot's density and
    friction through that sort, matching build_system. Friction combines by
    the min rule via LT.host_mu_fn. This is the independent reference the
    lattice per-slot path is pinned against.
    """
    order = sorted(range(len(placements)), key=lambda i: placements[i])
    boxes = [box_2d(6, 1, -3, 0.5, density=spec.density)]
    cube_materials = []
    for oi in order:
        (L, j) = placements[oi]
        boxes.append(box_2d(1, 1, j * DX, 1.5 + L, density=densities[oi]))
        cube_materials.append(mu_by_slot[oi])
    mu_fn = LT.host_mu_fn(mu, mu_ground, cube_materials)
    asm = build_assembly(
        boxes, mu=mu, tol=TOL, dim=2, mu_fn=mu_fn,
        pad_blocks=spec.n_blocks, pad_patches=spec.P_max, pad_verts=2,
    )
    return asm, assemble(asm, TOL, cone="linear2d")


class TestHeterogeneousAgreement:
    """Correctness gate for per-slot density and friction.

    For 50 random reachable states, each built in a shuffled placement order
    so the sort gather is exercised, the lattice per-slot build_system margins
    match the certified host pipeline to 1e-9, and the certified flags match
    exactly. densities are random in [500, 4000], per-slot mu random in
    [0.3, 1.0], with a separate ground friction.
    """

    def test_margins_agree_hetero(self):
        import jax.numpy as jnp

        n_max = 5
        rng = np.random.default_rng(7)
        densities = tuple(float(d) for d in rng.uniform(500.0, 4000.0, n_max))
        mu_by_slot = tuple(float(m) for m in rng.uniform(0.3, 1.0, n_max))
        mu = 0.7
        mu_ground = float(rng.uniform(0.3, 1.0))
        spec = LT.LatticeSpec(
            n_max=n_max, dx=DX, mu=mu, densities=densities,
            mu_ground=mu_ground, mu_by_slot=mu_by_slot,
        )

        # Generate reachable sets, then shuffle each into a placement order.
        keys = random_reachable_states(spec, 50, seed=0)
        placements_list = []
        for key in keys:
            cells = list(key)
            rng.shuffle(cells)
            placements_list.append(cells)

        # Mine, batched through the certified lattice path.
        states = LT.stack_states(
            [LT.state_from_placements(spec, p) for p in placements_list]
        )
        m_mine, cert_mine = LT.margins_of_states(
            spec, states, TOL.eps_reg, TOL.tol_cone,
            solver_tol=SOLVER_TOL, max_iter=MAX_ITER,
        )
        m_mine = np.asarray(m_mine)
        cert_mine = np.asarray(cert_mine)

        # Host, batched through margin_batch on the same systems.
        As, ws, Gs = [], [], []
        for placements in placements_list:
            _asm, hsys = host_system_hetero(
                spec, placements, densities, mu, mu_ground, mu_by_slot
            )
            As.append(hsys.A)
            ws.append(hsys.w_dead)
            Gs.append(hsys.G)
        m_host, cert_host = margin_batch(
            jnp.stack(As), jnp.stack(ws), jnp.stack(Gs),
            TOL.eps_reg, tol_cone=TOL.tol_cone, max_iter=MAX_ITER,
        )
        m_host = np.asarray(m_host)
        cert_host = np.asarray(cert_host)

        assert np.max(np.abs(m_mine - m_host)) < 1e-9
        assert np.array_equal(cert_mine, cert_host)

    def test_equal_materials_match_homogeneous_bitwise(self):
        # Explicit all-equal materials degenerate to the homogeneous scene
        # bit for bit: same (A, w, G, L, W) as the default (None) spec.
        n_max = 4
        spec0 = LT.LatticeSpec(n_max=n_max, dx=DX)
        spec1 = LT.LatticeSpec(
            n_max=n_max, dx=DX, densities=(DENSITY,) * n_max,
            mu_by_slot=(MU,) * n_max, mu_ground=MU,
        )
        key = [(0, -2), (1, -14), (2, -4), (1, 19)]
        A0, w0, G0, L0, W0 = LT.build_system(spec0, LT.state_from_placements(spec0, key))
        A1, w1, G1, L1, W1 = LT.build_system(spec1, LT.state_from_placements(spec1, key))
        assert np.array_equal(np.asarray(A0), np.asarray(A1))
        assert np.array_equal(np.asarray(w0), np.asarray(w1))
        assert np.array_equal(np.asarray(G0), np.asarray(G1))
        assert float(L0) == float(L1) and float(W0) == float(W1)


class TestBallastDirection:
    """Hand-check: heavier ballast raises the P4 feasibility of a reacher.

    A reacher cube overhangs the pedestal right edge and would tip; a
    counterweight cube sits on it, shifted left. As the counterweight density
    rises the combined center of mass moves back over the support, so the P4
    margin falls monotonically and the state flips from infeasible to
    feasible. The margin ordering is the assertion.
    """

    def test_heavier_ballast_lowers_margin(self):
        # Reacher (0, 7) at x = +0.2917 on the pedestal; ballast (1, -5) at
        # x = -0.2083 resting on it. Sorted-cell order puts the reacher at
        # slot 0 and the ballast at slot 1, so densities[1] is the ballast.
        key = [(0, 7), (1, -5)]
        cw_densities = [500.0, 1000.0, 2000.0, 4000.0, 8000.0]
        margins = []
        feasibles = []
        for d_cw in cw_densities:
            spec = LT.LatticeSpec(n_max=2, dx=1.0 / 24.0, densities=(2000.0, d_cw))
            st = LT.state_from_placements(spec, key)
            m, c = LT.margins_of_states(
                spec, LT.stack_states([st]), TOL.eps_reg, TOL.tol_cone,
                solver_tol=SOLVER_TOL, max_iter=MAX_ITER,
            )
            m = float(np.asarray(m)[0])
            c = bool(np.asarray(c)[0])
            margins.append(m)
            feasibles.append((m <= TOL.tol_feas) and c)

        # Margin is strictly decreasing in ballast mass.
        for a, b in zip(margins, margins[1:]):
            assert b < a + 1e-12, margins
        # The light ballast cannot hold the reacher; the heavy ballast can.
        assert not feasibles[0]
        assert feasibles[-1]
