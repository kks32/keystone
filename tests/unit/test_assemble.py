"""Equilibrium assembly, load, and cone matrix tests.

Sign convention under test (CLAUDE.md Section 3): the force exerted by
block j on block i at vertex k is F_k = -n_k n_hat + u_k t1 + v_k t2,
compression positive. Assertion tolerances here are test tolerances.
"""

import numpy as np
import pytest

from keystone.geometry.assembly import bbox_diagonal, build_assembly
from keystone.geometry.boxes import Box, box_2d
from keystone.geometry.tolerances import Tolerances
from keystone.mechanics.assemble import assemble
from keystone.mechanics.cones import cone_matrix_2d, cone_matrix_pyramid
from keystone.mechanics.loads import DEFAULT_G, dead_and_live_loads

TOL = Tolerances()


def cube(cx, cy, cz, half=0.5, density=2000.0):
    return Box(
        np.array([half, half, half]), np.array([cx, cy, cz]), density=density
    )


def force_rows(dim, b):
    """Force row indices of zero-based block b."""
    if dim == 2:
        return [3 * b + 0, 3 * b + 1]
    return [6 * b + 0, 6 * b + 1, 6 * b + 2]


class TestLoads:
    def test_2d_values(self):
        mass = np.array([2.0, 3.0])
        mask = np.array([True, True])
        W = 5.0 * DEFAULT_G
        w_dead, w_live = dead_and_live_loads(mass, mask, W, dim=2)
        assert w_dead.shape == (6,)
        assert np.allclose(w_dead, [0, -0.4, 0, 0, -0.6, 0], atol=1e-15)
        assert np.allclose(w_live, [0.4, 0, 0, 0.6, 0, 0], atol=1e-15)

    def test_3d_values_and_masked_block(self):
        mass = np.array([2.0, 3.0, 7.0])
        mask = np.array([True, True, False])
        W = 5.0 * DEFAULT_G
        w_dead, w_live = dead_and_live_loads(mass, mask, W, dim=3)
        assert w_dead.shape == (18,)
        assert w_dead[2] == pytest.approx(-0.4)
        assert w_dead[8] == pytest.approx(-0.6)
        assert np.all(w_dead[12:] == 0.0)
        assert w_live[0] == pytest.approx(0.4)
        assert w_live[6] == pytest.approx(0.6)
        assert np.all(w_live[12:] == 0.0)
        # Torque rows are zero for both loads.
        for b in range(3):
            assert np.all(w_dead[6 * b + 3 : 6 * b + 6] == 0.0)
            assert np.all(w_live[6 * b + 3 : 6 * b + 6] == 0.0)


class TestActionReaction:
    def check_columns(self, asm, sys):
        dim = asm.dim
        ncomp = 2 if dim == 2 else 3
        A = np.asarray(sys.A)
        V = asm.verts_per_patch
        for p in range(asm.n_patches):
            if not asm.patch_mask[p]:
                continue
            i, j = int(asm.patch_blocks[p, 0]), int(asm.patch_blocks[p, 1])
            basis_i = [-asm.normal[p], asm.t1[p]]
            if dim == 3:
                basis_i.append(asm.t2[p])
            comps = [0, 2] if dim == 2 else [0, 1, 2]
            for v in range(V):
                if not asm.vert_mask[p, v]:
                    continue
                for c in range(ncomp):
                    col = (p * V + v) * ncomp + c
                    fj = A[force_rows(dim, j - 1), col]
                    expect_j = -basis_i[c][comps]
                    assert np.allclose(fj, expect_j, atol=1e-12)
                    if i != 0:
                        fi = A[force_rows(dim, i - 1), col]
                        # Both sides real: net force is zero.
                        assert np.allclose(fi + fj, 0.0, atol=1e-12)
                        assert np.allclose(fi, basis_i[c][comps], atol=1e-12)

    def test_2d_stack(self):
        boxes = [box_2d(1, 1, 0, 0.5), box_2d(1, 1, 0.2, 1.5)]
        asm = build_assembly(boxes, mu=0.5, tol=TOL, dim=2)
        sys = assemble(asm, TOL, cone="linear2d")
        self.check_columns(asm, sys)

    def test_3d_stack(self):
        boxes = [cube(0, 0, 0.5), cube(0.2, 0.1, 1.5)]
        asm = build_assembly(boxes, mu=0.5, tol=TOL, dim=3)
        sys = assemble(asm, TOL, cone="pyramid", k=8)
        self.check_columns(asm, sys)


class TestTorqueConsistency:
    def test_2d_centered_block_uniform_normal(self):
        asm = build_assembly([box_2d(1, 1, 0, 0.5)], mu=0.5, tol=TOL, dim=2)
        sys = assemble(asm, TOL, cone="linear2d")
        # One ground patch, two verts. Each carries n = m g / (W * 2)
        # = 1/2 nondimensional, u = 0. Symmetry balances the torque.
        f = np.array([0.5, 0.0, 0.5, 0.0])
        resid = np.asarray(sys.A) @ f + np.asarray(sys.w_dead)
        assert np.allclose(resid, 0.0, atol=1e-12)

    def test_2d_torque_entry_sign(self):
        # Lock the torque sign: vertex at x = -1/2, com above at
        # (0, 0, 1/2). Unit n pushes +z on the block, T_y about the com
        # is (r - com)_z Fx - (r - com)_x Fz = +1/2 / L.
        asm = build_assembly([box_2d(1, 1, 0, 0.5)], mu=0.5, tol=TOL, dim=2)
        sys = assemble(asm, TOL, cone="linear2d")
        L = bbox_diagonal(asm)
        A = np.asarray(sys.A)
        assert asm.verts[0, 0, 0] == pytest.approx(-0.5)
        assert A[0, 0] == pytest.approx(0.0, abs=1e-15)
        assert A[1, 0] == pytest.approx(1.0)
        assert A[2, 0] == pytest.approx(0.5 / L)

    def test_3d_centered_block_uniform_normal(self):
        asm = build_assembly([cube(0, 0, 0.5)], mu=0.5, tol=TOL, dim=3)
        sys = assemble(asm, TOL, cone="pyramid", k=8)
        # One ground patch, four verts, n = 1/4 each by symmetry.
        f = np.zeros(3 * asm.n_patches * asm.verts_per_patch)
        for v in range(4):
            f[3 * v] = 0.25
        resid = np.asarray(sys.A) @ f + np.asarray(sys.w_dead)
        assert np.allclose(resid, 0.0, atol=1e-12)

    def test_2d_two_block_stack_uniform_normal(self):
        # Equal geometry, different density. Ground patch carries the
        # total weight, inter patch the top block's weight, split
        # evenly over the two verts of each patch by symmetry.
        bottom = box_2d(1, 1, 0, 0.5, density=2000.0)
        top = box_2d(1, 1, 0, 1.5, density=1000.0)
        for boxes in ([bottom, top], [top, bottom]):
            asm = build_assembly(boxes, mu=0.5, tol=TOL, dim=2)
            sys = assemble(asm, TOL, cone="linear2d")
            m_top = top.mass
            m_tot = bottom.mass + top.mass
            f = np.zeros(2 * asm.n_patches * asm.verts_per_patch)
            for p in range(asm.n_patches):
                i = int(asm.patch_blocks[p, 0])
                share = 0.5 if i == 0 else 0.5 * m_top / m_tot
                for v in range(2):
                    f[2 * (2 * p + v)] = share
            resid = np.asarray(sys.A) @ f + np.asarray(sys.w_dead)
            assert np.allclose(resid, 0.0, atol=1e-12)

    def test_reciprocity_w_permutes_with_blocks(self):
        bottom = box_2d(1, 1, 0, 0.5, density=2000.0)
        top = box_2d(1, 1, 0, 1.5, density=1000.0)
        asm_a = build_assembly([bottom, top], mu=0.5, tol=TOL, dim=2)
        asm_b = build_assembly([top, bottom], mu=0.5, tol=TOL, dim=2)
        sys_a = assemble(asm_a, TOL, cone="linear2d")
        sys_b = assemble(asm_b, TOL, cone="linear2d")
        wa = np.asarray(sys_a.w_dead).reshape(2, 3)
        wb = np.asarray(sys_b.w_dead).reshape(2, 3)
        assert np.allclose(wa, wb[::-1], atol=1e-15)
        assert float(sys_a.W) == pytest.approx(float(sys_b.W))
        # i < j on every active patch in both orderings.
        for asm in (asm_a, asm_b):
            pb = asm.patch_blocks[asm.patch_mask]
            assert np.all(pb[:, 0] < pb[:, 1])


class TestMasking:
    def test_masked_blocks_and_verts_zero(self):
        boxes = [box_2d(1, 1, 0, 0.5), box_2d(1, 1, 0, 1.5)]
        asm = build_assembly(
            boxes, mu=0.5, tol=TOL, dim=2, pad_blocks=4, pad_patches=5
        )
        sys = assemble(asm, TOL, cone="linear2d")
        A = np.asarray(sys.A)
        assert A.shape == (12, 20)
        # Masked block rows are zero.
        assert np.all(A[6:, :] == 0.0)
        # Masked patch columns are zero.
        assert np.all(A[:, 8:] == 0.0)
        # Masked block load entries are zero.
        assert np.all(np.asarray(sys.w_dead)[6:] == 0.0)
        assert np.all(np.asarray(sys.w_live)[6:] == 0.0)

    def test_cone_validation(self):
        asm2 = build_assembly([box_2d(1, 1, 0, 0.5)], mu=0.5, tol=TOL, dim=2)
        asm3 = build_assembly([cube(0, 0, 0.5)], mu=0.5, tol=TOL, dim=3)
        with pytest.raises(ValueError):
            assemble(asm2, TOL, cone="pyramid")
        with pytest.raises(ValueError):
            assemble(asm3, TOL, cone="linear2d")
        with pytest.raises(NotImplementedError):
            assemble(asm3, TOL, cone="socp")


class TestCone2D:
    def test_hand_check_p1_v2(self):
        mu = np.array([0.7])
        G = np.asarray(cone_matrix_2d(mu, 1, 2))
        assert G.shape == (6, 4)
        expect_v0 = np.array(
            [
                [-1.0, 0.0],
                [-0.7, 1.0],
                [-0.7, -1.0],
            ]
        )
        assert np.allclose(G[0:3, 0:2], expect_v0, atol=1e-15)
        assert np.all(G[0:3, 2:4] == 0.0)
        assert np.allclose(G[3:6, 2:4], expect_v0, atol=1e-15)
        assert np.all(G[3:6, 0:2] == 0.0)

    def test_per_patch_mu(self):
        mu = np.array([0.3, 0.9])
        G = np.asarray(cone_matrix_2d(mu, 2, 2))
        assert G.shape == (12, 8)
        # Vertex order p0v0, p0v1, p1v0, p1v1: mu rows follow the patch.
        for vv, m in enumerate([0.3, 0.3, 0.9, 0.9]):
            assert G[3 * vv + 1, 2 * vv] == pytest.approx(-m)
            assert G[3 * vv + 2, 2 * vv] == pytest.approx(-m)

    def test_semantics(self):
        mu = np.array([0.5])
        G = np.asarray(cone_matrix_2d(mu, 1, 2))
        inside = np.array([1.0, 0.4, 1.0, -0.4])
        boundary = np.array([1.0, 0.5, 1.0, -0.5])
        outside = np.array([1.0, 0.6, 1.0, 0.0])
        tension = np.array([-1.0, 0.0, 1.0, 0.0])
        assert np.all(G @ inside < 0.0)
        assert np.max(G @ boundary) == pytest.approx(0.0, abs=1e-15)
        assert np.max(G @ outside) > 0.0
        assert np.max(G @ tension) > 0.0


class TestConePyramid:
    def test_hand_check_p1_v1(self):
        mu = np.array([0.6])
        k = 8
        G = np.asarray(cone_matrix_pyramid(mu, 1, 1, k))
        assert G.shape == (9, 3)
        assert np.allclose(G[0], [-1.0, 0.0, 0.0], atol=1e-15)
        for j in range(k):
            th = (2 * j + 1) * np.pi / k
            expect = [-0.6 * np.cos(np.pi / k), np.cos(th), np.sin(th)]
            assert np.allclose(G[1 + j], expect, atol=1e-15)

    def test_block_structure_p1_v4(self):
        mu = np.array([0.6])
        k = 8
        G = np.asarray(cone_matrix_pyramid(mu, 1, 4, k))
        assert G.shape == (36, 12)
        for vv in range(4):
            block = G[9 * vv : 9 * vv + 9, 3 * vv : 3 * vv + 3]
            assert np.allclose(
                block, np.asarray(cone_matrix_pyramid(mu, 1, 1, k)), atol=1e-15
            )
        off = G.copy()
        for vv in range(4):
            off[9 * vv : 9 * vv + 9, 3 * vv : 3 * vv + 3] = 0.0
        assert np.all(off == 0.0)

    def test_boundary_points(self):
        mu_val = 0.6
        k = 8
        G = np.asarray(cone_matrix_pyramid(np.array([mu_val]), 1, 1, k))
        n = 1.0
        # Polygon vertices sit on the exact cone: angles 2 pi j / k at
        # radius mu n. All rows hold to 1e-12.
        for j in range(k):
            ang = 2.0 * np.pi * j / k
            f = np.array(
                [n, mu_val * n * np.cos(ang), mu_val * n * np.sin(ang)]
            )
            assert np.max(G @ f) <= 1e-12
        # Facet-mid directions just outside the exact cone violate a row.
        for j in range(k):
            th = (2 * j + 1) * np.pi / k
            r = mu_val * n * (1.0 + 1e-6)
            f = np.array([n, r * np.cos(th), r * np.sin(th)])
            assert np.max(G @ f) > 0.0
        # The incircle radius mu n cos(pi/k) is inside for all angles.
        rng = np.random.default_rng(0)
        for ang in np.concatenate(
            [np.linspace(0, 2 * np.pi, 17), rng.uniform(0, 2 * np.pi, 16)]
        ):
            r = mu_val * n * np.cos(np.pi / k) * (1.0 - 1e-9)
            f = np.array([n, r * np.cos(ang), r * np.sin(ang)])
            assert np.max(G @ f) <= 0.0

    def test_tension_rejected(self):
        G = np.asarray(cone_matrix_pyramid(np.array([0.6]), 1, 1, 8))
        f = np.array([-1.0, 0.0, 0.0])
        assert np.max(G @ f) > 0.0


class TestSystemShapes:
    def test_2d_shapes_and_nondimensional_w(self):
        boxes = [box_2d(1, 1, 0, 0.5), box_2d(1, 1, 0, 1.5)]
        asm = build_assembly(boxes, mu=0.5, tol=TOL, dim=2)
        sys = assemble(asm, TOL, cone="linear2d")
        P, V = asm.n_patches, asm.verts_per_patch
        assert np.asarray(sys.A).shape == (6, 2 * P * V)
        assert np.asarray(sys.G).shape == (3 * P * V, 2 * P * V)
        # w_dead has total magnitude 1 by construction.
        assert np.sum(np.asarray(sys.w_dead)) == pytest.approx(-1.0)
        assert np.allclose(
            np.asarray(sys.w_dead).reshape(2, 3)[:, 1], [-0.5, -0.5]
        )
        assert float(sys.W) == pytest.approx(2 * box_2d(1, 1, 0, 0.5).mass * DEFAULT_G)

    def test_3d_shapes(self):
        boxes = [cube(0, 0, 0.5), cube(0, 0, 1.5)]
        asm = build_assembly(boxes, mu=0.5, tol=TOL, dim=3)
        sys = assemble(asm, TOL, cone="pyramid", k=8)
        P, V = asm.n_patches, asm.verts_per_patch
        assert np.asarray(sys.A).shape == (12, 3 * P * V)
        assert np.asarray(sys.G).shape == (9 * P * V, 3 * P * V)
        assert sys.dim == 3
        assert sys.cone == "pyramid"
        assert sys.k == 8
