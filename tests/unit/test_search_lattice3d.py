"""Agreement and unit tests for the jittable 3D lattice environment.

The agreement suite is the correctness gate: for 100 random reachable 3D
lattice states it pins keystone.search.lattice3d.build_system to the
certified host pipeline (build_assembly dim=3 + assemble cone="pyramid"
k=8 + margin_core). The margins agree to 1e-9, the active-patch masks
agree exactly, (A, w) agree to 1e-12 after aligning patches by (node
pair, vertex positions), L agrees to 1e-9, and W agrees to 1e-6.

The rest of the file checks the legality rules against an independent
reference, the drop reachability mode against a second independent
reference, the support-count bound of four, the fixed padded shapes,
expand_kernel masking, and node-order invariance under placement order.
"""

import numpy as np
import pytest

from keystone import Tolerances, build_assembly
from keystone.geometry import Box
from keystone.mechanics import assemble
from keystone.solve.batch_jax import margin_batch, margin_core
from keystone.search import lattice3d as L3

TOL = Tolerances()
DX = 1.0 / 12.0
MU = 0.7
SOLVER_TOL = 1e-9
MAX_ITER = 100

# Reduced demo grid used by the agreement suite. The full default grid is
# ~n_max * 85 * 73 candidates; this box keeps the random rollouts and the
# host parity loop cheap while still spanning multiple layers and depths.
DEMO = dict(x_lo=-1.5, x_hi=2.0, y_lo=-1.5, y_hi=1.5)


# --- host pipeline helpers ------------------------------------------------


def host_boxes(key):
    """Pedestal plus one unit cube per placement, in sorted (L, ix, iy) order."""
    boxes = [Box(np.array([3.0, 3.0, 0.5]), np.array([-3.0, 0.0, 0.5]), density=2000.0)]
    for (L, ix, iy) in sorted(key):
        boxes.append(
            Box(
                np.array([0.5, 0.5, 0.5]),
                np.array([ix * DX, iy * DX, 1.5 + L]),
                density=2000.0,
            )
        )
    return boxes


def host_system(spec, key):
    boxes = host_boxes(key)
    asm = build_assembly(
        boxes,
        mu=MU,
        tol=TOL,
        dim=3,
        pad_blocks=spec.n_blocks,
        pad_patches=spec.P_max,
        pad_verts=spec.V,
    )
    return asm, assemble(asm, TOL, cone="pyramid", k=8)


def _sorted_verts(varr, p):
    """The four vertices of patch p as a sorted tuple of rounded triples."""
    return tuple(
        sorted(tuple(round(float(x), 6) for x in varr[p, v]) for v in range(4))
    )


def host_active_patches(asm):
    """Set of active host patches as (i, j, sorted vertex triples)."""
    verts = np.asarray(asm.verts)
    out = []
    for p in range(asm.n_patches):
        if not asm.patch_mask[p]:
            continue
        i, j = int(asm.patch_blocks[p, 0]), int(asm.patch_blocks[p, 1])
        out.append((i, j, _sorted_verts(verts, p)))
    return out


def host_patch_vertorder(asm):
    """Map (i, j) -> (host patch column base, vertex order sorting positions)."""
    verts = np.asarray(asm.verts)
    m = {}
    for p in range(asm.n_patches):
        if not asm.patch_mask[p]:
            continue
        i, j = int(asm.patch_blocks[p, 0]), int(asm.patch_blocks[p, 1])
        order = sorted(
            range(4), key=lambda v: tuple(round(float(x), 6) for x in verts[p, v])
        )
        # ncomp * V = 12 columns per patch; three columns per vertex.
        m[(i, j)] = [12 * p + 3 * v for v in order]
    return m


# --- mine helpers ---------------------------------------------------------


def mine_active_patches(pt):
    """Set of active patches from patch_table, matching host_active_patches."""
    pi = np.asarray(pt.patch_i)
    pj = np.asarray(pt.patch_j)
    pa = np.asarray(pt.patch_active)
    verts = np.asarray(pt.verts)
    out = []
    for p in range(pi.shape[0]):
        if not pa[p]:
            continue
        i, j = int(round(pi[p])), int(round(pj[p]))
        out.append((i, j, _sorted_verts(verts, p)))
    return out


def mine_patch_vertorder(pt):
    pi = np.asarray(pt.patch_i)
    pj = np.asarray(pt.patch_j)
    pa = np.asarray(pt.patch_active)
    verts = np.asarray(pt.verts)
    m = {}
    for p in range(pi.shape[0]):
        if not pa[p]:
            continue
        i, j = int(round(pi[p])), int(round(pj[p]))
        order = sorted(
            range(4), key=lambda v: tuple(round(float(x), 6) for x in verts[p, v])
        )
        m[(i, j)] = [12 * p + 3 * v for v in order]
    return m


# --- reachable-state generator -------------------------------------------


def random_reachable_states(spec, n_states, seed=0):
    """Build reachable states by random legal rollouts using the env itself.

    Records every intermediate prefix, so the set spans all depths. Returns
    a list of placement-tuple keys, including the empty state and full stacks.
    """
    rng = np.random.default_rng(seed)
    cand = L3.action_grid(spec)
    cand_L = np.asarray(cand[0])
    cand_X = np.asarray(cand[1])
    cand_Y = np.asarray(cand[2])
    keys = set()
    keys.add(())
    while len(keys) < n_states:
        placements = []
        for _ in range(spec.n_max):
            legal = np.asarray(
                L3.legal_grid(spec, L3.batch_states(spec, [tuple(placements)]), *cand)
            )[0]
            idx = np.nonzero(legal)[0]
            if idx.size == 0:
                break
            pick = int(rng.choice(idx))
            placements.append((int(cand_L[pick]), int(cand_X[pick]), int(cand_Y[pick])))
            keys.add(tuple(sorted(placements)))
            if len(keys) >= n_states:
                break
    return list(keys)[:n_states]


# --- independent legality and reachability references ---------------------


def ref_legal(spec, key, layer, ix, iy):
    """Frozen legality rules, plain python, independent of is_legal."""
    dx, dy = spec.dx, spec.dy
    x, y = ix * dx, iy * dy
    if len(key) >= spec.n_max:
        return False
    if not (spec.ix_lo <= ix <= spec.ix_hi and spec.iy_lo <= iy <= spec.iy_hi):
        return False
    layers = [L for (L, _, _) in key]
    max_L = max(layers) if layers else -1
    if not (0 <= layer <= spec.n_max - 1 and layer <= max_L + 1):
        return False
    # support: overlap rectangle both sides >= 2 dx
    if layer == 0:
        ovx = min(x + 0.5, spec.ped_right) - max(x - 0.5, spec.ped_left)
        ovy = min(y + 0.5, spec.ped_top) - max(y - 0.5, spec.ped_bot)
        support = (ovx > 1.5 * dx) and (ovy > 1.5 * dy)
    else:
        support = False
        for (L, jx, jy) in key:
            if L == layer - 1:
                ovx = min(x + 0.5, jx * dx + 0.5) - max(x - 0.5, jx * dx - 0.5)
                ovy = min(y + 0.5, jy * dy + 0.5) - max(y - 0.5, jy * dy - 0.5)
                if ovx > 1.5 * dx and ovy > 1.5 * dy:
                    support = True
                    break
    if not support:
        return False
    # same-layer strict separation in at least one axis
    for (L, jx, jy) in key:
        if L == layer:
            near_x = abs(x - jx * dx) < 1.0 + 0.5 * dx
            near_y = abs(y - jy * dy) < 1.0 + 0.5 * dy
            if near_x and near_y:
                return False
    return True


def ref_reach(spec, key, layer, ix, iy):
    """Frozen reachability rules, plain python, independent of _reach_ok."""
    if spec.mode == "static":
        return True
    dx, dy = spec.dx, spec.dy
    x, y = ix * dx, iy * dy
    # Drop column: a cube strictly above the target blocks when its
    # footprint openly overlaps the target footprint in both x and y.
    for (L, jx, jy) in key:
        if L > layer:
            ovx = min(x + 0.5, jx * dx + 0.5) - max(x - 0.5, jx * dx - 0.5)
            ovy = min(y + 0.5, jy * dy + 0.5) - max(y - 0.5, jy * dy - 0.5)
            if ovx > 0.5 * dx and ovy > 0.5 * dy:
                return False
    return True


# =========================================================================
# Agreement suite: the correctness gate.
# =========================================================================


class TestAgreement:
    spec = L3.LatticeSpec3D(n_max=4, dx=DX, **DEMO)
    keys = None

    @classmethod
    def setup_class(cls):
        cls.keys = random_reachable_states(cls.spec, 100, seed=0)

    def test_one_hundred_reachable_states(self):
        assert len(self.keys) == 100
        depths = {len(k) for k in self.keys}
        assert len(depths) >= 3

    def test_margins_agree(self):
        spec = self.spec
        states = L3.batch_states(spec, self.keys)
        m_mine, cert_mine = L3.margins_of_states(
            spec, states, TOL.eps_reg, TOL.tol_cone,
            solver_tol=SOLVER_TOL, max_iter=MAX_ITER,
        )
        m_mine = np.asarray(m_mine)
        cert_mine = np.asarray(cert_mine)
        import jax.numpy as jnp

        As, ws, Gs = [], [], []
        for key in self.keys:
            _asm, hsys = host_system(spec, key)
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

    def test_masks_and_matrices_agree(self):
        spec = self.spec
        max_dA = 0.0
        max_dw = 0.0
        for key in self.keys:
            asm, hsys = host_system(spec, key)
            state = L3.state_from_placements(spec, key)
            A, w, G, L, W = L3.build_system(spec, state)
            A = np.asarray(A)
            w = np.asarray(w)
            hA = np.asarray(hsys.A)
            hw = np.asarray(hsys.w_dead)
            pt = L3.patch_table(spec, state)

            # Active-patch masks agree exactly as sets of (i, j, sorted verts).
            hpatch = host_active_patches(asm)
            mpatch = mine_active_patches(pt)
            assert sorted(hpatch) == sorted(mpatch), key

            # Block count and dead load agree directly (block order matches).
            assert int(asm.block_mask.sum()) == 1 + len(key)
            max_dw = max(max_dw, float(np.max(np.abs(w - hw))))

            # Align A columns patch by patch (matched by node pair), and within
            # a patch align the three-column vertex groups by sorted position.
            hcol = host_patch_vertorder(asm)
            mcol = mine_patch_vertorder(pt)
            for (i, j, _vs) in hpatch:
                for hc, mc in zip(hcol[(i, j)], mcol[(i, j)]):
                    dblock = np.abs(hA[:, hc : hc + 3] - A[:, mc : mc + 3])
                    max_dA = max(max_dA, float(np.max(dblock)))
            assert abs(float(L) - float(np.asarray(hsys.L))) < 1e-9
            assert abs(float(W) - float(np.asarray(hsys.W))) < 1e-6
        assert max_dw < 1e-12, max_dw
        assert max_dA < 1e-12, max_dA


# =========================================================================
# Legality unit tests.
# =========================================================================


class TestLegality:
    # The default full grid gives the scalar edge tests room near the
    # pedestal edges. The reference sweep uses the reduced grid for speed.
    spec = L3.LatticeSpec3D(n_max=6, dx=DX)

    def test_matches_reference_on_reachable_states(self):
        spec = L3.LatticeSpec3D(n_max=5, dx=DX, **DEMO)
        keys = random_reachable_states(spec, 40, seed=1)
        cand = L3.action_grid(spec)
        cand_L = np.asarray(cand[0])
        cand_X = np.asarray(cand[1])
        cand_Y = np.asarray(cand[2])
        for key in keys:
            legal = np.asarray(
                L3.legal_grid(spec, L3.batch_states(spec, [key]), *cand)
            )[0]
            ref = np.array(
                [
                    ref_legal(spec, key, int(cand_L[i]), int(cand_X[i]), int(cand_Y[i]))
                    for i in range(cand_L.shape[0])
                ]
            )
            assert np.array_equal(legal, ref), key

    def test_layer0_support_needs_pedestal_overlap(self):
        spec = self.spec
        empty = L3.empty_state(spec)
        # x = 0.5 spans [0, 1], no x overlap with pedestal top [-6, 0].
        j_off = round(0.5 / DX)
        assert not bool(L3.is_legal(spec, empty, 0, j_off, 0))
        # x = -0.5 spans [-1, 0], overlaps by 0.5 >= 2 dx, y centred.
        assert bool(L3.is_legal(spec, empty, 0, -j_off, 0))

    def test_support_needs_two_dx_on_both_axes(self):
        spec = self.spec
        # A layer-0 base at (x, y) = (-1, 0). Its top face lets a layer-1 cube
        # test the 2 dx support floor on the y axis, well inside the grid.
        base = L3.place(spec, L3.empty_state(spec), 0, -12, 0)
        # Shifted 10 dx in y: y overlap 2 dx, at the floor, supported.
        assert bool(L3.is_legal(spec, base, 1, -12, 10))
        # Shifted 11 dx in y: y overlap only dx, below the floor, unsupported.
        assert not bool(L3.is_legal(spec, base, 1, -12, 11))

    def test_same_layer_touching_forbidden(self):
        spec = self.spec
        # Base cube at (x, y) = (-2.0, 0.0), well inside the pedestal top.
        st = L3.place(spec, L3.empty_state(spec), 0, -24, 0)
        # 12 grid steps away in x is exactly one block width: touching in x
        # and overlapping in y, forbidden (shared vertical face).
        assert not bool(L3.is_legal(spec, st, 0, -12, 0))  # x = -1.0
        # 13 grid steps in x is more than one block width: separated, allowed.
        assert bool(L3.is_legal(spec, st, 0, -11, 0))  # x = -0.9167
        # Touching in x but separated in y (distance 2 > 1): allowed.
        assert bool(L3.is_legal(spec, st, 0, -12, 24))

    def test_layer_reachability(self):
        spec = self.spec
        empty = L3.empty_state(spec)
        assert not bool(L3.is_legal(spec, empty, 1, -24, 0))
        st = L3.place(spec, empty, 0, -24, 0)
        assert bool(L3.is_legal(spec, st, 1, -24, 0))
        assert not bool(L3.is_legal(spec, st, 2, -24, 0))

    def test_node_order_invariant_to_placement_order(self):
        spec = L3.LatticeSpec3D(n_max=5, dx=DX, **DEMO)
        key = [(0, -18, -6), (1, -12, 0), (0, -6, 6)]
        A1, w1, G1, L1, W1 = L3.build_system(
            spec, L3.state_from_placements(spec, key)
        )
        A2, w2, G2, L2, W2 = L3.build_system(
            spec, L3.state_from_placements(spec, list(reversed(key)))
        )
        assert np.max(np.abs(np.asarray(A1) - np.asarray(A2))) < 1e-14
        assert np.max(np.abs(np.asarray(w1) - np.asarray(w2))) < 1e-14
        assert np.max(np.abs(np.asarray(G1) - np.asarray(G2))) < 1e-14


# =========================================================================
# Support-count bound: four supports per cube, realized by a corner tetrad.
# =========================================================================


class TestSupportBound:
    def test_four_supports_realized(self):
        # Four layer-0 cubes at the corners of a target footprint, mutually
        # separated (distance 4/3 > 1 in x and y), all supporting one layer-1
        # cube centred over them. The layer-1 cube contacts all four.
        spec = L3.LatticeSpec3D(n_max=5, dx=DX)
        a = 8  # offset in grid steps; 2 a dx = 4/3 > 1 separates the corners
        cx, cy = -24, -24  # target at (-2, -2), inside the pedestal top
        corners = [
            (0, cx - a, cy - a),
            (0, cx + a, cy - a),
            (0, cx - a, cy + a),
            (0, cx + a, cy + a),
        ]
        # Each corner overlaps the target by 1 - 8/12 = 1/3 = 4 dx in each
        # axis, above the 2 dx floor, so each is a real support, and the four
        # are pairwise separated by 4/3 > 1, so all four coexist legally.
        key = tuple(sorted(corners + [(1, cx, cy)]))
        state = L3.state_from_placements(spec, key)
        pt = L3.patch_table(spec, state)
        pi = np.asarray(pt.patch_i)
        pj = np.asarray(pt.patch_j)
        pa = np.asarray(pt.patch_active)
        # Node of the layer-1 cube (sorted last: highest layer).
        top_node = 6  # pedestal 1, four layer-0 cubes 2..5, layer-1 cube 6
        supports = [
            (int(round(pi[p])), int(round(pj[p])))
            for p in range(pi.shape[0])
            if pa[p] and int(round(pj[p])) == top_node
        ]
        assert len(supports) == 4, supports
        # All four lower nodes 2..5 support the top cube.
        assert sorted(i for (i, _j) in supports) == [2, 3, 4, 5]


# =========================================================================
# Placement reachability (drop mode).
# =========================================================================


class TestReachability:
    def test_default_mode_is_static(self):
        assert L3.LatticeSpec3D(n_max=4).mode == "static"

    def test_bad_mode_rejected(self):
        with pytest.raises(ValueError, match="mode"):
            L3.LatticeSpec3D(n_max=4, mode="teleport")

    def test_slide_deferred(self):
        with pytest.raises(ValueError, match="slide"):
            L3.LatticeSpec3D(n_max=4, mode="slide")

    def test_matches_reference_under_drop(self):
        spec = L3.LatticeSpec3D(n_max=5, dx=DX, mode="drop", **DEMO)
        keys = random_reachable_states(spec, 30, seed=2)
        cand = L3.action_grid(spec)
        cand_L = np.asarray(cand[0])
        cand_X = np.asarray(cand[1])
        cand_Y = np.asarray(cand[2])
        for key in keys:
            legal = np.asarray(
                L3.legal_grid(spec, L3.batch_states(spec, [key]), *cand)
            )[0]
            ref = np.array(
                [
                    ref_legal(spec, key, int(cand_L[i]), int(cand_X[i]), int(cand_Y[i]))
                    and ref_reach(
                        spec, key, int(cand_L[i]), int(cand_X[i]), int(cand_Y[i])
                    )
                    for i in range(cand_L.shape[0])
                ]
            )
            assert np.array_equal(legal, ref), key

    def test_drop_only_removes_actions(self):
        # drop legality is a subset of static legality on every tested state.
        static = L3.LatticeSpec3D(n_max=5, dx=DX, mode="static", **DEMO)
        drop = L3.LatticeSpec3D(n_max=5, dx=DX, mode="drop", **DEMO)
        keys = random_reachable_states(static, 30, seed=3)
        cand = L3.action_grid(static)
        for key in keys:
            g_static = np.asarray(
                L3.legal_grid(static, L3.batch_states(static, [key]), *cand)
            )[0]
            g_drop = np.asarray(
                L3.legal_grid(drop, L3.batch_states(drop, [key]), *cand)
            )[0]
            assert not np.any(g_drop & ~g_static), key

    def test_drop_blocked_by_column_above(self):
        # A layer-1 bridge hanging over an empty layer-0 cell blocks the drop
        # into that cell while the static rules still accept the placement.
        static = L3.LatticeSpec3D(n_max=4, dx=DX)
        drop = L3.LatticeSpec3D(n_max=4, dx=DX, mode="drop")
        # Support at x = -1, bridge at x = -1.5 resting on it (overlap 0.5).
        key = ((0, -12, 0), (1, -18, 0))
        st = L3.state_from_placements(static, key)
        # Target layer-0 at ix = -29 (x = -2.417): on the pedestal, separated
        # from the layer-0 support by 1.417 > 1, but its column is blocked by
        # the bridge above (footprint overlap 1 dx > 0).
        assert bool(L3.is_legal(static, st, 0, -29, 0))
        assert not bool(L3.is_legal(drop, st, 0, -29, 0))


# =========================================================================
# Shapes and kernel masking.
# =========================================================================


class TestShapesAndKernel:
    def test_derived_shapes(self):
        spec = L3.LatticeSpec3D(n_max=4, dx=DX)
        assert spec.V == 4 and spec.ncomp == 3
        assert spec.P_max == 4 * 4 + 2
        assert spec.rows == 6 * (4 + 1)
        assert spec.nf == spec.ncomp * spec.P_max * spec.V
        assert spec.ncone == (1 + spec.k) * spec.P_max * spec.V
        assert spec.M == spec.n_layers * spec.n_pos_x * spec.n_pos_y
        state = L3.empty_state(spec)
        A, w, G, L, W = L3.build_system(spec, state)
        assert A.shape == (spec.rows, spec.nf)
        assert w.shape == (spec.rows,)
        assert G.shape == (spec.ncone, spec.nf)

    def test_expand_kernel_masks_illegal(self):
        # A deliberately tiny grid keeps the vmapped QP solve fast.
        spec = L3.LatticeSpec3D(
            n_max=2, dx=DX, x_lo=-1.5, x_hi=-0.5, y_lo=-0.25, y_hi=0.25
        )
        state = L3.empty_state(spec)
        cand = L3.action_grid(spec)
        legal, margins, cert = L3.expand_kernel(
            spec, state, *cand, TOL.eps_reg, TOL.tol_cone,
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

    def test_full_grid_expand_matches_gathered_solve(self):
        # The kernel and the per-state solve agree on legal states.
        spec = L3.LatticeSpec3D(
            n_max=3, dx=DX, x_lo=-1.5, x_hi=-0.5, y_lo=-0.25, y_hi=0.25
        )
        state = L3.place(spec, L3.empty_state(spec), 0, -12, 0)
        cand = L3.action_grid(spec)
        legal, margins, cert = L3.expand_kernel(
            spec, state, *cand, TOL.eps_reg, TOL.tol_cone,
            solver_tol=SOLVER_TOL, max_iter=MAX_ITER,
        )
        legal = np.asarray(legal)
        margins = np.asarray(margins)
        cand_L = np.asarray(cand[0])
        cand_X = np.asarray(cand[1])
        cand_Y = np.asarray(cand[2])
        idx = np.nonzero(legal)[0][:12]
        keys = [
            ((0, -12, 0), (int(cand_L[i]), int(cand_X[i]), int(cand_Y[i])))
            for i in idx
        ]
        states = L3.batch_states(spec, [tuple(sorted(k)) for k in keys])
        m2, _c2 = L3.margins_of_states(
            spec, states, TOL.eps_reg, TOL.tol_cone,
            solver_tol=SOLVER_TOL, max_iter=MAX_ITER,
        )
        assert np.max(np.abs(margins[idx] - np.asarray(m2))) < 1e-9


# =========================================================================
# Spec validation and the bitwise default regression (fix 4).
# =========================================================================


class TestSpecValidation:
    def test_ped_mass_matches_module_constants_bitwise(self):
        # ped_mass now derives from the pedestal bounds on the spec. The
        # default bounds reproduce the old module-constant product bit for bit.
        spec = L3.LatticeSpec3D(n_max=4, dx=DX)
        assert spec.ped_mass == (
            L3.DENSITY * L3.PEDESTAL_W * L3.PEDESTAL_D * L3.PEDESTAL_H
        )
        assert spec.cube_mass == L3.DENSITY * 1.0 * 1.0 * 1.0

    def test_default_build_system_weight_matches_constants_bitwise(self):
        # Total weight of a default-spec state through build_system is bitwise
        # the constant-based hand value: the bounds-derived ped_mass path did
        # not move the default scene.
        from keystone.mechanics.loads import DEFAULT_G

        spec = L3.LatticeSpec3D(n_max=4, dx=DX)
        state = L3.state_from_placements(spec, [(0, -6, 0)])
        _A, _w, _G, _L, W = L3.build_system(spec, state)
        expected = (
            L3.DENSITY * L3.PEDESTAL_W * L3.PEDESTAL_D * L3.PEDESTAL_H
            + L3.DENSITY
        ) * DEFAULT_G
        assert float(W) == expected

    def test_invalid_scalars_raise(self):
        with pytest.raises(ValueError, match="n_max"):
            L3.LatticeSpec3D(n_max=0, dx=DX)
        with pytest.raises(ValueError, match="dx"):
            L3.LatticeSpec3D(n_max=4, dx=0.0)
        with pytest.raises(ValueError, match="dy"):
            L3.LatticeSpec3D(n_max=4, dx=DX, dy=-1.0)
        with pytest.raises(ValueError, match="dx"):
            L3.LatticeSpec3D(n_max=4, dx=float("inf"))
        with pytest.raises(ValueError, match="x_lo < x_hi"):
            L3.LatticeSpec3D(n_max=4, dx=DX, x_lo=2.0, x_hi=-3.0)
        with pytest.raises(ValueError, match="y_lo < y_hi"):
            L3.LatticeSpec3D(n_max=4, dx=DX, y_lo=2.0, y_hi=-3.0)
        with pytest.raises(ValueError, match="mu"):
            L3.LatticeSpec3D(n_max=4, dx=DX, mu=-0.1)
        with pytest.raises(ValueError, match="density"):
            L3.LatticeSpec3D(n_max=4, dx=DX, density=0.0)
        with pytest.raises(ValueError, match="g"):
            L3.LatticeSpec3D(n_max=4, dx=DX, g=0.0)

    def test_invalid_k_raises(self):
        # k must be an even integer >= 4 (inscribed pyramid facet count).
        with pytest.raises(ValueError, match="k"):
            L3.LatticeSpec3D(n_max=4, dx=DX, k=3)  # odd
        with pytest.raises(ValueError, match="k"):
            L3.LatticeSpec3D(n_max=4, dx=DX, k=2)  # below 4
        with pytest.raises(ValueError, match="k"):
            L3.LatticeSpec3D(n_max=4, dx=DX, k=5)  # odd

    def test_valid_k_accepted(self):
        # Even k >= 4 is accepted; the default is 8.
        assert L3.LatticeSpec3D(n_max=4, dx=DX, k=4).k == 4
        assert L3.LatticeSpec3D(n_max=4, dx=DX, k=6).k == 6
        assert L3.LatticeSpec3D(n_max=4, dx=DX).k == 8
