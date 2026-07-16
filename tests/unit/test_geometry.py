"""Interface detection and Assembly construction tests.

Analytic scenes with exact expected patches. Assertion tolerances here
are test tolerances, not geometric cuts; the geometric cuts all come
from the Tolerances dataclass.
"""

import numpy as np
import pytest

from keystone.geometry.assembly import bbox_diagonal, build_assembly
from keystone.geometry.boxes import Box, box_2d
from keystone.geometry.interfaces import detect_patches_2d, detect_patches_3d
from keystone.geometry.tolerances import Tolerances

TOL = Tolerances()


def cube(cx, cy, cz, half=0.5, angle_z=0.0, density=2000.0):
    """Axis-aligned cube of side 2 * half, optionally rotated about z."""
    quat = np.array([np.cos(angle_z / 2.0), 0.0, 0.0, np.sin(angle_z / 2.0)])
    return Box(np.array([half, half, half]), np.array([cx, cy, cz]), quat, density)


def polygon_area(verts, n_hat):
    """Area of a planar 3D polygon with normal n_hat, shoelace form."""
    m = verts.mean(axis=0)
    rel = verts - m
    cross_sum = np.zeros(3)
    for a in range(len(verts)):
        b = (a + 1) % len(verts)
        cross_sum += np.cross(rel[a], rel[b])
    return 0.5 * float(np.dot(cross_sum, n_hat))


def vert_set(verts, decimals=9):
    """Order-free comparable form of a vertex array."""
    return frozenset(tuple(np.round(v, decimals)) for v in verts)


class TestDetect2D:
    def test_two_stacked_boxes_on_ground(self):
        b1 = box_2d(1.0, 1.0, 0.0, 0.5)
        b2 = box_2d(1.0, 1.0, 0.0, 1.5)
        recs = detect_patches_2d([b1, b2], True, 2.0, TOL)
        assert len(recs) == 2

        pairs = [(r[0], r[1]) for r in recs]
        assert pairs == [(0, 1), (1, 2)]

        for i, j, n_hat, t1, verts in recs:
            assert np.allclose(n_hat, [0.0, 0.0, 1.0], atol=1e-12)
            assert np.allclose(t1, [1.0, 0.0, 0.0], atol=1e-12)
            assert verts.shape == (2, 3)
            # v0 -> v1 runs along t1: deterministic CCW rule.
            assert np.dot(verts[1] - verts[0], t1) > 0.0

        ground = recs[0][4]
        inter = recs[1][4]
        assert np.allclose(ground, [[-0.5, 0, 0], [0.5, 0, 0]], atol=1e-12)
        assert np.allclose(inter, [[-0.5, 0, 1], [0.5, 0, 1]], atol=1e-12)

    def test_offset_pair_overlap_length(self):
        b = 1.0
        e = 0.3
        b1 = box_2d(b, 1.0, 0.0, 0.5)
        b2 = box_2d(b, 1.0, e, 1.5)
        recs = detect_patches_2d([b1, b2], False, 2.0, TOL)
        assert len(recs) == 1
        i, j, n_hat, t1, verts = recs[0]
        assert (i, j) == (1, 2)
        length = np.linalg.norm(verts[1] - verts[0])
        assert np.isclose(length, b - e, atol=1e-12)
        # Overlap interval is [e - b/2, b/2] on the shared line z = 1.
        assert np.allclose(verts[0], [e - b / 2, 0, 1], atol=1e-12)
        assert np.allclose(verts[1], [b / 2, 0, 1], atol=1e-12)

    def test_non_touching_boxes_no_patch(self):
        b1 = box_2d(1.0, 1.0, 0.0, 0.5)
        b2 = box_2d(1.0, 1.0, 3.0, 0.5)
        recs = detect_patches_2d([b1, b2], False, 4.0, TOL)
        assert recs == []
        # Vertical gap above g_tol * L also yields no inter patch.
        b3 = box_2d(1.0, 1.0, 0.0, 1.6)
        recs = detect_patches_2d([b1, b3], False, 3.0, TOL)
        assert recs == []

    def test_detection_deterministic_and_sorted(self):
        boxes = [
            box_2d(1.0, 1.0, 0.0, 0.5),
            box_2d(1.0, 1.0, 0.0, 1.5),
            box_2d(1.0, 1.0, 1.0, 0.5),
        ]
        recs_a = detect_patches_2d(boxes, True, 3.0, TOL)
        recs_b = detect_patches_2d(boxes, True, 3.0, TOL)
        assert len(recs_a) == len(recs_b)
        for ra, rb in zip(recs_a, recs_b):
            assert (ra[0], ra[1]) == (rb[0], rb[1])
            assert np.array_equal(ra[2], rb[2])
            assert np.array_equal(ra[3], rb[3])
            assert np.array_equal(ra[4], rb[4])
        pairs = [(r[0], r[1]) for r in recs_a]
        assert pairs == sorted(pairs)
        assert all(i < j for i, j in pairs)


class TestDetect3D:
    def test_two_stacked_cubes_on_ground(self):
        c1 = cube(0, 0, 0.5)
        c2 = cube(0, 0, 1.5)
        recs = detect_patches_3d([c1, c2], True, 2.0, TOL)
        assert len(recs) == 2
        pairs = [(r[0], r[1]) for r in recs]
        assert pairs == [(0, 1), (1, 2)]

        for i, j, n_hat, t1, verts in recs:
            assert np.allclose(n_hat, [0, 0, 1], atol=1e-12)
            assert verts.shape == (4, 3)

        ground = recs[0][4]
        inter = recs[1][4]
        expect_xy = {(0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)}
        assert vert_set(ground) == frozenset((x, y, 0.0) for x, y in expect_xy)
        assert vert_set(inter) == frozenset((x, y, 1.0) for x, y in expect_xy)
        assert np.isclose(polygon_area(inter, recs[1][2]), 1.0, atol=1e-12)
        # CCW about n_hat: signed area is positive.
        assert polygon_area(ground, recs[0][2]) > 0.0

    def test_offset_cube_clipped_rectangle(self):
        b = 1.0
        d = 1.0
        e = 0.3
        c1 = cube(0, 0, 0.5)
        c2 = cube(e, 0, 1.5)
        recs = detect_patches_3d([c1, c2], False, 2.0, TOL)
        assert len(recs) == 1
        i, j, n_hat, t1, verts = recs[0]
        assert (i, j) == (1, 2)
        assert verts.shape == (4, 3)
        assert np.isclose(polygon_area(verts, n_hat), (b - e) * d, atol=1e-12)
        expect = frozenset(
            (x, y, 1.0)
            for x in (e - b / 2, b / 2)
            for y in (-d / 2, d / 2)
        )
        assert vert_set(verts) == expect

    def test_rotated_45_cube_octagon(self):
        # Unit square [-1/2, 1/2]^2 clipped with the same square rotated
        # 45 degrees. The rotated square is |x + y| <= sqrt(2)/2 and
        # |x - y| <= sqrt(2)/2. Each corner of the axis square is cut by
        # one diagonal edge, leaving a right triangle with legs
        # 1 - sqrt(2)/2, area (1 - sqrt(2)/2)^2 / 2 each. Octagon area
        # = 1 - 4 * (1 - sqrt(2)/2)^2 / 2 = 1 - (3 - 2 sqrt(2))
        # = 2 (sqrt(2) - 1).
        c1 = cube(0, 0, 0.5)
        c2 = cube(0, 0, 1.5, angle_z=np.pi / 4)
        recs = detect_patches_3d([c1, c2], False, 2.0, TOL)
        assert len(recs) == 1
        i, j, n_hat, t1, verts = recs[0]
        assert (i, j) == (1, 2)
        assert verts.shape == (8, 3)
        area = polygon_area(verts, n_hat)
        assert np.isclose(area, 2.0 * (np.sqrt(2.0) - 1.0), atol=1e-12)
        assert np.allclose(verts[:, 2], 1.0, atol=1e-12)

    def test_non_touching_cubes_no_patch(self):
        c1 = cube(0, 0, 0.5)
        c2 = cube(3, 0, 0.5)
        recs = detect_patches_3d([c1, c2], False, 4.0, TOL)
        assert recs == []


class TestBuildAssembly:
    def test_padding_and_masks_2d(self):
        b1 = box_2d(1.0, 1.0, 0.0, 0.5)
        b2 = box_2d(1.0, 1.0, 0.0, 1.5)
        asm = build_assembly(
            [b1, b2], mu=0.5, tol=TOL, dim=2,
            pad_blocks=4, pad_patches=5, pad_verts=2,
        )
        assert asm.n_blocks == 4
        assert asm.n_patches == 5
        assert asm.verts_per_patch == 2
        assert asm.block_mask.tolist() == [True, True, False, False]
        assert asm.patch_mask.tolist() == [True, True, False, False, False]
        assert asm.mass[2] == 0.0
        assert np.all(asm.patch_blocks[2:] == 0)
        assert np.all(~asm.vert_mask[2:])
        assert np.allclose(asm.mu[:2], 0.5)
        assert asm.dim == 2

    def test_no_padding_exact_counts_3d(self):
        c1 = cube(0, 0, 0.5)
        c2 = cube(0, 0, 1.5)
        asm = build_assembly([c1, c2], mu=0.5, tol=TOL, dim=3)
        assert asm.n_blocks == 2
        assert asm.n_patches == 2
        assert asm.verts_per_patch == 8
        assert asm.vert_mask[:, :4].all()
        assert not asm.vert_mask[:, 4:].any()
        # t2 = cross(n, t1) on active patches.
        for p in range(2):
            t2 = np.cross(asm.normal[p], asm.t1[p])
            assert np.allclose(asm.t2[p], t2, atol=1e-12)

    def test_2d_t2_is_zero(self):
        b1 = box_2d(1.0, 1.0, 0.0, 0.5)
        asm = build_assembly([b1], mu=0.5, tol=TOL, dim=2)
        assert np.all(asm.t2 == 0.0)


class TestReciprocityPrecheck:
    """Swapping block list order relabels ids and flips n_hat. The
    patch set and every downstream verdict must be unchanged."""

    def make(self, order):
        bottom = box_2d(1.0, 1.0, 0.0, 0.5)
        top = box_2d(0.8, 0.6, 0.1, 1.3)
        boxes = [bottom, top] if order == "bt" else [top, bottom]
        return build_assembly(boxes, mu=0.5, tol=TOL, dim=2)

    def test_same_patch_set_up_to_relabel(self):
        asm_a = self.make("bt")
        asm_b = self.make("tb")
        # Permutation of node ids: list order swap maps 1 <-> 2.
        relabel = {0: 0, 1: 2, 2: 1}

        def canon(asm, mapping):
            out = set()
            for p in range(asm.n_patches):
                if not asm.patch_mask[p]:
                    continue
                i, j = int(asm.patch_blocks[p, 0]), int(asm.patch_blocks[p, 1])
                assert i < j
                mi, mj = mapping[i], mapping[j]
                verts = asm.verts[p][asm.vert_mask[p]]
                n = asm.normal[p]
                # Normalize to (min, max) id order; flip n if ids swap.
                if mi > mj:
                    mi, mj = mj, mi
                    n = -n
                out.add((mi, mj, tuple(np.round(n, 9)), vert_set(verts)))
            return out

        assert canon(asm_a, relabel) == canon(asm_b, {0: 0, 1: 1, 2: 2})


class TestDepthIndependence2D:
    """2D detection must ignore the arbitrary out-of-plane depth. The
    length scale L for tolerance scaling comes from the xz projection
    only, so g_tol scaling does not move when the y depth changes."""

    def test_touching_blocks_identical_across_depth(self):
        def build(depth):
            b1 = box_2d(1.0, 1.0, 0.0, 0.5, depth=depth)
            b2 = box_2d(1.0, 1.0, 0.0, 1.5, depth=depth)
            return build_assembly([b1, b2], mu=0.5, tol=TOL, dim=2)

        a1 = build(1.0)
        a100 = build(100.0)
        assert int(a1.patch_mask.sum()) == int(a100.patch_mask.sum()) == 2
        assert np.array_equal(a1.patch_blocks, a100.patch_blocks)
        assert np.allclose(a1.normal, a100.normal, atol=1e-12)
        assert np.allclose(
            a1.verts[a1.vert_mask], a100.verts[a100.vert_mask], atol=1e-12
        )

    def test_small_gap_not_bridged_by_depth(self):
        # A vertical gap wider than g_tol * L (xz) must be rejected at any
        # depth. Before the fix, depth = 100 inflated L a hundredfold and
        # g_tol * L bridged the gap, detecting a spurious patch.
        gap = 1e-3  # exceeds g_tol * L_xz (about 2.2e-4) for these blocks.

        def build(depth):
            b1 = box_2d(1.0, 1.0, 0.0, 0.5, depth=depth)
            b2 = box_2d(1.0, 1.0, 0.0, 1.5 + gap, depth=depth)
            return build_assembly(
                [b1, b2], mu=0.5, tol=TOL, dim=2, ground=False
            )

        assert int(build(1.0).patch_mask.sum()) == 0
        assert int(build(100.0).patch_mask.sum()) == 0


class TestFaceGapAtCorners:
    """Face pairs must be close at every corner, not only at the center.
    Block-block corner deviation from the mid-plane must stay below
    g_tol * L / 2, so the total face separation stays within g_tol * L, the
    same budget the ground path spends measuring against z = 0. A face
    tilted within theta_tol can have a small center gap yet corners past
    that budget; those must not flatten into a patch."""

    def slabs(self, tilt):
        # 4 x 4 x 1 slabs, half_extents [2, 2, 0.5]. Bottom axis-aligned,
        # top tilted about y by `tilt` and centered one unit above so the
        # two contact-face centers meet near z = 1 (center gap ~ 0). A tilt
        # about y moves the x = +-2 corners in z by 2 * sin(tilt), lifting
        # one edge and dipping the opposite edge by the same amount.
        half = np.array([2.0, 2.0, 0.5])
        bottom = Box(half, np.array([0.0, 0.0, 0.5]))
        q = np.array([np.cos(tilt / 2.0), 0.0, np.sin(tilt / 2.0), 0.0])
        top = Box(half, np.array([0.0, 0.0, 1.5]), q)
        return [bottom, top]

    def scale(self, boxes):
        allc = np.concatenate([b.corners() for b in boxes], axis=0)
        return float(np.linalg.norm(allc.max(axis=0) - allc.min(axis=0)))

    def test_small_tilt_detects_patch(self):
        # tilt = theta_tol / 10. Corner deviation 2 * sin(tilt) ~ 2.0e-4 is
        # below the halved budget g_tol * L / 2 ~ 3.0e-4, so the patch is
        # kept. The dipped edge penetrates by only 2.0e-4, far under the
        # 2 * g_tol * L interpenetration guard, so detection does not raise.
        tilt = 0.10 * TOL.theta_tol
        boxes = self.slabs(tilt)
        L = self.scale(boxes)
        gap_half = 0.5 * TOL.g_tol * L
        corner_dev = 2.0 * np.sin(tilt)
        assert corner_dev < gap_half
        recs = detect_patches_3d(boxes, False, L, TOL)
        assert len(recs) == 1
        assert (recs[0][0], recs[0][1]) == (1, 2)

    def test_within_orientation_but_corners_far_no_patch(self):
        # tilt = 0.4 * theta_tol passes the orientation filter (relative
        # normal angle below theta_tol), but corner deviation 2 * sin(tilt)
        # ~ 8.0e-4 exceeds the halved budget g_tol * L / 2 ~ 3.0e-4. The old
        # center-only gap check accepted this; the corner check rejects it.
        # The dip is 8.0e-4, still below the 2 * g_tol * L ~ 1.2e-3
        # interpenetration guard, so the gap check is what rejects it.
        tilt = 0.4 * TOL.theta_tol
        boxes = self.slabs(tilt)
        L = self.scale(boxes)
        gap_half = 0.5 * TOL.g_tol * L
        corner_dev = 2.0 * np.sin(tilt)
        assert tilt < TOL.theta_tol        # orientation filter passes
        assert corner_dev > gap_half       # corner gap check rejects
        assert corner_dev < 2.0 * TOL.g_tol * L  # below interpenetration guard
        recs = detect_patches_3d(boxes, False, L, TOL)
        assert recs == []

    def test_large_tilt_interpenetrates(self):
        # tilt = 2 * theta_tol fails the orientation filter, and with the
        # face centers coincident the dipped edge sinks 2 * sin(tilt) ~ 4e-3
        # into the bottom slab, past the 2 * g_tol * L ~ 1.2e-3
        # interpenetration guard. Detection raises rather than silently
        # returning no patch for an overlapping pair.
        tilt = 2.0 * TOL.theta_tol
        boxes = self.slabs(tilt)
        L = self.scale(boxes)
        assert 2.0 * np.sin(tilt) > 2.0 * TOL.g_tol * L
        with pytest.raises(ValueError, match="interpenetrate"):
            detect_patches_3d(boxes, False, L, TOL)


class TestDegenerateLengthScale:
    """bbox_diagonal must fail loudly on an assembly with no spatial
    extent rather than return L = 0 and blow up every scaled quantity."""

    def test_single_floating_block_raises(self):
        asm = build_assembly(
            [box_2d(1.0, 1.0, 0.0, 5.0)],
            mu=0.5, tol=TOL, dim=2, ground=False,
        )
        assert int(asm.patch_mask.sum()) == 0
        with pytest.raises(ValueError, match="degenerate length scale"):
            bbox_diagonal(asm)


class TestBoxValidation:
    """Box rejects nonpositive or nonfinite geometry and material."""

    def test_nonpositive_half_extents(self):
        with pytest.raises(ValueError, match="half_extents must be positive"):
            Box(np.array([1.0, 0.0, 1.0]), np.zeros(3))
        with pytest.raises(ValueError, match="half_extents must be positive"):
            Box(np.array([1.0, -1.0, 1.0]), np.zeros(3))

    def test_nonfinite_half_extents(self):
        with pytest.raises(ValueError, match="half_extents must be finite"):
            Box(np.array([1.0, np.inf, 1.0]), np.zeros(3))
        with pytest.raises(ValueError, match="half_extents must be finite"):
            Box(np.array([1.0, np.nan, 1.0]), np.zeros(3))

    def test_nonpositive_density(self):
        with pytest.raises(ValueError, match="density"):
            Box(np.ones(3), np.zeros(3), density=0.0)
        with pytest.raises(ValueError, match="density"):
            Box(np.ones(3), np.zeros(3), density=-5.0)

    def test_nonfinite_density(self):
        with pytest.raises(ValueError, match="density"):
            Box(np.ones(3), np.zeros(3), density=np.inf)
        with pytest.raises(ValueError, match="density"):
            Box(np.ones(3), np.zeros(3), density=float("nan"))

    def test_nonfinite_position(self):
        with pytest.raises(ValueError, match="position must be finite"):
            Box(np.ones(3), np.array([0.0, np.inf, 0.0]))
        with pytest.raises(ValueError, match="position must be finite"):
            Box(np.ones(3), np.array([np.nan, 0.0, 0.0]))

    def test_bad_half_extents_shape(self):
        # A 2-vector cannot be a 3D half-extent.
        with pytest.raises(ValueError, match=r"half_extents must have shape \(3,\)"):
            Box(np.array([1.0, 1.0]), np.zeros(3))
        with pytest.raises(ValueError, match=r"half_extents must have shape \(3,\)"):
            Box(np.array([1.0, 1.0, 1.0, 1.0]), np.zeros(3))

    def test_bad_position_shape(self):
        with pytest.raises(ValueError, match=r"position must have shape \(3,\)"):
            Box(np.ones(3), np.array([0.0, 0.0]))
        with pytest.raises(ValueError, match=r"position must have shape \(3,\)"):
            Box(np.ones(3), np.zeros((3, 1)))

    def test_bad_quat_shape(self):
        # A 3-element quaternion must be rejected at construction, not later
        # when rotation unpacks four components. Note [1, 0, 0] has norm 1,
        # so the norm check alone would let it through; the shape check
        # catches it first.
        with pytest.raises(ValueError, match=r"quat must have shape \(4,\)"):
            Box(np.ones(3), np.zeros(3), quat=np.array([1.0, 0.0, 0.0]))
        with pytest.raises(ValueError, match=r"quat must have shape \(4,\)"):
            Box(np.ones(3), np.zeros(3), quat=np.array([1.0, 0.0, 0.0, 0.0, 0.0]))


class TestInterpenetration:
    """build_assembly's detection path rejects grossly overlapping boxes
    with a conservative AABB test at 2 * g_tol * L per axis. Legitimate
    contacts, whose overlap on the contact axis is near zero, pass."""

    def test_coincident_cubes_raise(self):
        c1 = cube(0, 0, 0.5)
        c2 = cube(0, 0, 0.5)
        with pytest.raises(ValueError, match="interpenetrate"):
            detect_patches_3d([c1, c2], False, 2.0, TOL)
        with pytest.raises(ValueError, match="interpenetrate"):
            build_assembly([c1, c2], mu=0.5, tol=TOL, dim=3, ground=False)

    def test_stacked_pair_ok(self):
        # Faces touch, so the contact-axis overlap is zero. Not flagged.
        c1 = cube(0, 0, 0.5)
        c2 = cube(0, 0, 1.5)
        recs = detect_patches_3d([c1, c2], False, 2.0, TOL)
        assert len(recs) == 1

    def test_offset_pair_ok(self):
        # Offset horizontally and stacked: the z overlap is zero.
        c1 = cube(0, 0, 0.5)
        c2 = cube(0.3, 0, 1.5)
        recs = detect_patches_3d([c1, c2], False, 2.0, TOL)
        assert len(recs) == 1

    def test_coincident_2d_blocks_raise(self):
        b1 = box_2d(1.0, 1.0, 0.0, 0.5)
        b2 = box_2d(1.0, 1.0, 0.0, 0.5)
        with pytest.raises(ValueError, match="interpenetrate"):
            detect_patches_2d([b1, b2], False, 2.0, TOL)


class TestGapConvention2D:
    """g_tol is the maximum TOTAL face separation. Two stacked 2D blocks
    with a vertical gap detect a patch only while the gap stays within
    g_tol * L, matching the ground path's direct g_tol * L budget."""

    def build(self, gap):
        b1 = box_2d(1.0, 1.0, 0.0, 0.5)
        b2 = box_2d(1.0, 1.0, 0.0, 1.5 + gap)
        allc = np.concatenate([b1.corners(), b2.corners()], axis=0)
        allc[:, 1] = 0.0  # 2D scale ignores the out-of-plane depth
        L = float(np.linalg.norm(allc.max(axis=0) - allc.min(axis=0)))
        recs = detect_patches_2d([b1, b2], False, L, TOL)
        return L, recs

    def test_total_separation_just_below_budget_detects(self):
        # The two faces sit `gap` apart, so `gap` is the total separation.
        # Just below g_tol * L it is kept.
        L0 = float(np.linalg.norm(np.array([1.0, 2.0])))  # sqrt(5), gap ~ 0
        gap = 0.9 * TOL.g_tol * L0
        L, recs = self.build(gap)
        assert gap < TOL.g_tol * L        # inside the total-separation budget
        assert len(recs) == 1

    def test_total_separation_just_above_budget_rejects(self):
        L0 = float(np.linalg.norm(np.array([1.0, 2.0])))
        gap = 1.1 * TOL.g_tol * L0
        L, recs = self.build(gap)
        assert gap > TOL.g_tol * L        # past the total-separation budget
        assert recs == []
