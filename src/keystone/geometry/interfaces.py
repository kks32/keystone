"""Interface detection primitives for box blocks.

2D: contact between oriented rectangles in the xz plane, and between
rectangles and the ground line z = 0. A contact is a segment patch
with two vertices.

3D: face-face contact between oriented boxes and the ground plane,
clipped with Sutherland-Hodgman to at most 8 vertices (arrives in C2).

All functions take an explicit Tolerances argument. Geometric cuts use
tolerances scaled by the caller-supplied L.

A patch record is the tuple (i, j, normal(3,), t1(3,), verts(V, 3)):
node ids i < j with ground = 0, n_hat pointing from i into j, t1 the
2D or 3D tangent from the frame rule, and V world vertices ordered
counterclockwise about n_hat. t2 is recomputed by the assembly builder
as cross(n_hat, t1); a 2D patch has t2 = 0.
"""

import numpy as np

from .tolerances import Tolerances

_Y_HAT = np.array([0.0, 1.0, 0.0])
_Z_HAT = np.array([0.0, 0.0, 1.0])
# Frame selection threshold from CLAUDE.md Section 4. This picks the
# reference axis for t1 and is a fixed convention, not a tolerance.
_FRAME_AXIS_CUTOFF = 0.9


def _t1_2d(n_hat: np.ndarray) -> np.ndarray:
    """2D tangent: normalize(cross(y_hat, n_hat)), in the xz plane."""
    t = np.cross(_Y_HAT, n_hat)
    return t / np.linalg.norm(t)


def _frame_3d(n_hat: np.ndarray):
    """3D frame tangents (t1, t2) from the CLAUDE.md Section 4 rule."""
    x = np.array([1.0, 0.0, 0.0])
    ref = _Y_HAT if abs(np.dot(n_hat, x)) > _FRAME_AXIS_CUTOFF else x
    t1 = ref - np.dot(ref, n_hat) * n_hat
    t1 = t1 / np.linalg.norm(t1)
    t2 = np.cross(n_hat, t1)
    return t1, t2


def _edges_2d(box):
    """Four world edges of a 2D box in the xz plane.

    Each edge is (outward_normal(3,), p0(3,), p1(3,)), all with y = 0.
    """
    R = box.rotation
    c = box.position
    hx = box.half_extents[0]
    hz = box.half_extents[2]
    ex = R[:, 0]
    ez = R[:, 2]
    edges = []
    cx = c + hx * ex
    edges.append((ex, cx - hz * ez, cx + hz * ez))
    cx = c - hx * ex
    edges.append((-ex, cx - hz * ez, cx + hz * ez))
    cz = c + hz * ez
    edges.append((ez, cz - hx * ex, cz + hx * ex))
    cz = c - hz * ez
    edges.append((-ez, cz - hx * ex, cz + hx * ex))
    return edges


def _faces_3d(box):
    """Six world faces of a box.

    Each face is (outward_normal(3,), center(3,), corners(4, 3)).
    Corner winding is arbitrary; callers reorder in the patch frame.
    """
    R = box.rotation
    c = box.position
    h = box.half_extents
    axes = [R[:, 0], R[:, 1], R[:, 2]]
    faces = []
    for a in range(3):
        others = [d for d in range(3) if d != a]
        u = axes[others[0]]
        w = axes[others[1]]
        hu = h[others[0]]
        hw = h[others[1]]
        for sgn in (1.0, -1.0):
            n = sgn * axes[a]
            center = c + sgn * h[a] * axes[a]
            corners = np.array(
                [
                    center - hu * u - hw * w,
                    center + hu * u - hw * w,
                    center + hu * u + hw * w,
                    center - hu * u + hw * w,
                ]
            )
            faces.append((n, center, corners))
    return faces


def _order_ccw(pts2d: np.ndarray) -> np.ndarray:
    """Order 2D points counterclockwise, starting from the smallest
    angle about the centroid measured from the +x (t1) axis."""
    centroid = pts2d.mean(axis=0)
    rel = pts2d - centroid
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    ang = np.mod(ang, 2.0 * np.pi)
    order = np.argsort(ang, kind="stable")
    return pts2d[order]


def _polygon_area(pts2d: np.ndarray) -> float:
    """Signed shoelace area of an ordered 2D polygon."""
    x = pts2d[:, 0]
    y = pts2d[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clip_2d(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clip of subject polygon by convex CCW clip.

    Both are (M, 2). Returns the clipped polygon vertices.
    """
    output = list(subject)
    n = len(clip)
    for e in range(n):
        a = clip[e]
        b = clip[(e + 1) % n]
        edge = b - a
        if not output:
            break
        inp = output
        output = []
        s = inp[-1]
        s_in = edge[0] * (s[1] - a[1]) - edge[1] * (s[0] - a[0]) >= 0.0
        for pt in inp:
            pt_in = edge[0] * (pt[1] - a[1]) - edge[1] * (pt[0] - a[0]) >= 0.0
            if pt_in:
                if not s_in:
                    output.append(_line_intersect(s, pt, a, b))
                output.append(pt)
            elif s_in:
                output.append(_line_intersect(s, pt, a, b))
            s = pt
            s_in = pt_in
    return np.array(output) if output else np.zeros((0, 2))


def _line_intersect(p0, p1, a, b):
    """Intersection of segment p0->p1 with the infinite line a->b."""
    r = p1 - p0
    sdir = b - a
    denom = r[0] * sdir[1] - r[1] * sdir[0]
    t = ((a[0] - p0[0]) * sdir[1] - (a[1] - p0[1]) * sdir[0]) / denom
    return p0 + t * r


def _weld(pts: np.ndarray, tol_dist: float) -> np.ndarray:
    """Drop points that repeat a neighbor within tol_dist (cyclic)."""
    if pts.shape[0] == 0:
        return pts
    kept = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - kept[-1]) > tol_dist:
            kept.append(p)
    if len(kept) > 1 and np.linalg.norm(kept[-1] - kept[0]) <= tol_dist:
        kept.pop()
    return np.array(kept)


def detect_patches_2d(boxes, ground: bool, L: float, tol: Tolerances):
    """Return a deterministic list of patch records for 2D boxes.

    Each record: (i, j, normal(3,), t1(3,), verts(2, 3)) with i < j node
    ids, ground = node 0. n_hat points from i into j.
    """
    cos_theta = np.cos(tol.theta_tol)
    gap_max = tol.g_tol * L
    len_min = tol.A_min * L
    records = []

    edges = [_edges_2d(b) for b in boxes]

    # Ground contacts: block bottom face against the line z = 0.
    if ground:
        for jdx, box in enumerate(boxes):
            for n_e, p0, p1 in edges[jdx]:
                if np.dot(n_e, -_Z_HAT) <= cos_theta:
                    continue
                center = 0.5 * (p0 + p1)
                # Both edge endpoints must lie within gap_max of z = 0, not
                # just the midpoint. An edge tilted within theta_tol can
                # have its center near 0 and an endpoint far off.
                if max(abs(float(p0[2])), abs(float(p1[2]))) > gap_max:
                    continue
                n_hat = _Z_HAT
                t1 = _t1_2d(n_hat)
                mid_z = 0.5 * center[2]
                v0 = p0.copy()
                v1 = p1.copy()
                v0[2] = mid_z
                v1[2] = mid_z
                if np.dot(v1 - v0, t1) < 0.0:
                    v0, v1 = v1, v0
                records.append((0, jdx + 1, n_hat, t1, np.array([v0, v1])))

    # Block-block contacts.
    n = len(boxes)
    for i in range(n):
        for j in range(i + 1, n):
            for n_a, a0, a1 in edges[i]:
                for n_b, b0, b1 in edges[j]:
                    if np.dot(n_a, n_b) >= -cos_theta:
                        continue
                    c_a = 0.5 * (a0 + a1)
                    c_b = 0.5 * (b0 + b1)
                    m = 0.5 * (c_a + c_b)
                    # All four endpoints must lie within gap_max of the
                    # mid-plane. The center gap alone lets edges tilted
                    # within theta_tol pass with far corners.
                    ends = np.array([a0, a1, b0, b1])
                    if float(np.max(np.abs((ends - m) @ n_a))) > gap_max:
                        continue
                    t = a1 - a0
                    t = t / np.linalg.norm(t)
                    ca = sorted([np.dot(a0 - c_a, t), np.dot(a1 - c_a, t)])
                    cb = sorted([np.dot(b0 - c_a, t), np.dot(b1 - c_a, t)])
                    lo = max(ca[0], cb[0])
                    hi = min(ca[1], cb[1])
                    if hi - lo <= len_min:
                        continue
                    shift = 0.5 * np.dot(c_b - c_a, n_a) * n_a
                    v0 = c_a + lo * t + shift
                    v1 = c_a + hi * t + shift
                    n_hat = n_a
                    t1 = _t1_2d(n_hat)
                    if np.dot(v1 - v0, t1) < 0.0:
                        v0, v1 = v1, v0
                    records.append(
                        (i + 1, j + 1, n_hat, t1, np.array([v0, v1]))
                    )

    return _sort_records(records, tol, L)


def detect_patches_3d(boxes, ground: bool, L: float, tol: Tolerances):
    """Return a deterministic list of patch records for 3D boxes.

    Each record: (i, j, normal(3,), t1(3,), verts(V, 3)) with V up to 8,
    ordered CCW about n_hat, i < j node ids, ground = node 0.
    """
    cos_theta = np.cos(tol.theta_tol)
    gap_max = tol.g_tol * L
    area_min = tol.A_min * L * L
    weld_dist = tol.w_tol * L
    records = []

    faces = [_faces_3d(b) for b in boxes]

    # Ground contacts: block faces with outward normal near -z at z ~ 0.
    if ground:
        for jdx, box in enumerate(boxes):
            for n_f, center, corners in faces[jdx]:
                if np.dot(n_f, -_Z_HAT) <= cos_theta:
                    continue
                # Every face corner must lie within gap_max of the ground
                # plane z = 0, not just the center. A face tilted within
                # theta_tol can have a center near 0 and corners far off.
                if float(np.max(np.abs(corners[:, 2]))) > gap_max:
                    continue
                n_hat = _Z_HAT
                t1, t2 = _frame_3d(n_hat)
                mid_z = 0.5 * center[2]
                verts = corners.copy()
                verts[:, 2] = mid_z
                rec = _finish_patch_3d(
                    0, jdx + 1, n_hat, t1, t2, verts, area_min, weld_dist
                )
                if rec is not None:
                    records.append(rec)

    # Block-block contacts with an inflated AABB broadphase.
    n = len(boxes)
    aabb = [_aabb(b, gap_max) for b in boxes]
    for i in range(n):
        for j in range(i + 1, n):
            if not _aabb_overlap(aabb[i], aabb[j]):
                continue
            for n_a, c_a, corners_a in faces[i]:
                for n_b, c_b, corners_b in faces[j]:
                    if np.dot(n_a, n_b) >= -cos_theta:
                        continue
                    m = 0.5 * (c_a + c_b)
                    # All corners of both faces must lie within gap_max of
                    # the mid-plane. Checking the center gap alone lets
                    # faces tilted within theta_tol flatten into a patch
                    # while their corners sit far outside the gap.
                    d_a = np.abs((corners_a - m) @ n_a)
                    d_b = np.abs((corners_b - m) @ n_a)
                    if max(float(d_a.max()), float(d_b.max())) > gap_max:
                        continue
                    n_hat = n_a
                    t1, t2 = _frame_3d(n_hat)
                    pa = _project(corners_a, m, t1, t2)
                    pb = _project(corners_b, m, t1, t2)
                    clip = _order_ccw(pa)
                    subj = _order_ccw(pb)
                    poly = _clip_2d(subj, clip)
                    if poly.shape[0] < 3:
                        continue
                    verts = m + poly[:, [0]] * t1 + poly[:, [1]] * t2
                    rec = _finish_patch_3d(
                        i + 1, j + 1, n_hat, t1, t2, verts, area_min, weld_dist
                    )
                    if rec is not None:
                        records.append(rec)

    return _sort_records(records, tol, L)


def _finish_patch_3d(i, j, n_hat, t1, t2, verts, area_min, weld_dist):
    """Weld, drop small, and order a 3D patch. None if too small."""
    m = verts.mean(axis=0)
    coords = np.stack(
        [np.dot(verts - m, t1), np.dot(verts - m, t2)], axis=1
    )
    coords = _order_ccw(coords)
    coords = _weld(coords, weld_dist)
    if coords.shape[0] < 3:
        return None
    if abs(_polygon_area(coords)) < area_min:
        return None
    world = m + coords[:, [0]] * t1 + coords[:, [1]] * t2
    return (i, j, n_hat, t1, world)


def _project(pts: np.ndarray, origin: np.ndarray, t1: np.ndarray, t2: np.ndarray):
    """Project 3D points to the (t1, t2) mid-plane coordinates."""
    rel = pts - origin
    return np.stack([rel @ t1, rel @ t2], axis=1)


def _aabb(box, pad: float):
    corners = box.corners()
    return corners.min(axis=0) - pad, corners.max(axis=0) + pad


def _aabb_overlap(a, b) -> bool:
    return bool(np.all(a[0] <= b[1]) and np.all(b[0] <= a[1]))


def _sort_records(records, tol: Tolerances, L: float):
    """Sort by (i, j, centroid rounded to w_tol * L)."""
    scale = tol.w_tol * L
    if scale <= 0.0:
        scale = 1.0

    def key(rec):
        centroid = rec[4].mean(axis=0)
        grid = np.round(centroid / scale).astype(np.int64)
        return (rec[0], rec[1], int(grid[0]), int(grid[1]), int(grid[2]))

    return sorted(records, key=key)
