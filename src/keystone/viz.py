"""Matplotlib rendering of keystone assemblies and solver results.

Display only. This module reads the frozen data contracts
(keystone.geometry.assembly.Assembly and keystone.solve.result.Result)
and draws them. It never runs a solver and never touches physics
tolerances. Every color, alpha, and scale is a display constant named
STYLE_* or a keyword argument.

2D uses the xz plane (gravity along -z). 3D uses mpl_toolkits.mplot3d.

Backend policy: no backend is set here. Import matplotlib.pyplot lazily
inside the functions so a caller may pick "Agg" (or any backend) before
the first pyplot import. The functions never call plt.show(), so they
work headless.

Sign convention for force arrows (from mechanics/assemble.py): for a
patch joining nodes (i, j) with i < j, the contact force on the j-side
block at vertex k is

    F_j_k = n_k * n_hat - u_k * t1_hat - v_k * t2_hat,   n_k >= 0.

The force on the i-side block is the negative. Ground patches have
i = 0, so the drawn arrow uses the j-side sign (the reaction pushing on
the resting block). Arrows are anchored at the vertex.

Force layout in Result.forces: ncomp = 2 in 2D (n, u), 3 in 3D
(n, u, v). The component c of vertex v of patch p sits at index
(p * V + v) * ncomp + c, where V is the padded vertex count per patch.

Mechanism twists (Result.mechanism) are normalized dual components, not
physical displacements or angles. The 2D wy value and the 3D |omega|
annotation label a direction and relative magnitude, not radians. Labels
use "~" and the legend says "normalized twist" to keep this clear.
"""

import numpy as np

# Display palette. Not physics. Hex strings and alphas only.
STYLE_BLOCK_FACE = "#7d98b3"      # soft steel blue
STYLE_BLOCK_EDGE = "#2f3b47"      # dark slate
STYLE_COM = "#1b2229"             # near black com dot
STYLE_PATCH = "#20272e"           # dark contact segment
STYLE_FORCE_NORMAL = "#2a7f8f"    # blue green, normal component
STYLE_FORCE_RESULTANT = "#c66a1e" # dark orange, full resultant
STYLE_MECHANISM = "#c0304a"       # crimson, virtual displacement
STYLE_GROUND = "#3a3a3a"          # ground line and hatch
STYLE_GROUND_FILL = "#d9d9d9"     # light gray 3D ground quad

STYLE_BLOCK_ALPHA_2D = 0.35
STYLE_BLOCK_ALPHA_3D = 0.6
STYLE_PATCH_ALPHA_3D = 0.25
STYLE_GROUND_ALPHA = 0.3

STYLE_FORCE_FRACTION = 0.15       # arrow length as a fraction of extent
STYLE_MECH_FRACTION = 0.15
STYLE_MECH_ROT_FRACTION = 0.6     # rotation-axis segment length as a
                                  # fraction of the mechanism arrow length
STYLE_MECH_ROT_CUTOFF = 1e-3      # draw a block rotation axis when its
                                  # |omega| exceeds this fraction of the
                                  # largest block twist norm
STYLE_FONT_SIZE = 9

# Per-face lightness for 3D box shading, order matches _BOX_FACES below.
STYLE_FACE_SHADE = (0.72, 1.0, 0.88, 0.82, 0.9, 0.85)

# Box corner order is fixed by geometry/boxes.py Box.corners(). Faces as
# index loops: bottom, top, y-minus, y-plus, x-minus, x-plus.
_BOX_FACES = (
    (0, 1, 3, 2),
    (4, 5, 7, 6),
    (0, 1, 5, 4),
    (2, 3, 7, 6),
    (0, 2, 6, 4),
    (1, 3, 7, 5),
)


def _sig3(x):
    """Format a float to 3 significant digits, or pass through None."""
    if x is None:
        return None
    return f"{float(x):.3g}"


def _compose_title(result, user_title):
    """Build the title from status, margin, and any associative factors."""
    lines = []
    if user_title:
        lines.append(str(user_title))
    if result is not None:
        parts = [str(result.status), f"margin = {_sig3(result.margin)}"]
        if result.lambda_assoc is not None:
            bound = getattr(result, "physical_bound_direction", None)
            tag = f" ({bound} est.)" if bound else ""
            parts.append(f"lambda_assoc{tag} = {_sig3(result.lambda_assoc)}")
        if result.mu_critical_assoc is not None:
            parts.append(f"mu_crit (assoc) = {_sig3(result.mu_critical_assoc)}")
        lines.append("  |  ".join(parts))
    return "\n".join(lines) if lines else None


def _active_patches(assembly):
    """Indices of real patches, in stored (sorted) order."""
    return np.flatnonzero(np.asarray(assembly.patch_mask))


def _active_blocks(assembly):
    """Indices of real blocks. Block index equals node id minus one."""
    return np.flatnonzero(np.asarray(assembly.block_mask))


def _ncomp(assembly):
    return 2 if assembly.dim == 2 else 3


def _force_at(forces, ncomp, p, v, verts_per_patch):
    """Return the (n, u, v) tuple for vertex v of patch p, v is 0 in 2D."""
    base = (p * verts_per_patch + v) * ncomp
    if base + ncomp > forces.shape[0]:
        return None
    if ncomp == 2:
        return forces[base + 0], forces[base + 1], 0.0
    return forces[base + 0], forces[base + 1], forces[base + 2]


def _resultant_vectors(assembly, forces):
    """Per active vertex: anchor point, normal-only vector, full resultant.

    Returns arrays anchors (m, 3), normals (m, 3), resultants (m, 3) in
    world coordinates. m is the count of active vertices with force data.
    """
    ncomp = _ncomp(assembly)
    V = assembly.verts_per_patch
    normal = np.asarray(assembly.normal)
    t1 = np.asarray(assembly.t1)
    t2 = np.asarray(assembly.t2)
    verts = np.asarray(assembly.verts)
    vmask = np.asarray(assembly.vert_mask)
    forces = np.asarray(forces).ravel()

    anchors, nrm, res = [], [], []
    for p in _active_patches(assembly):
        n_hat, t1v, t2v = normal[p], t1[p], t2[p]
        for v in range(V):
            if not vmask[p, v]:
                continue
            comp = _force_at(forces, ncomp, p, v, V)
            if comp is None:
                continue
            nk, uk, vk = comp
            normal_vec = nk * n_hat
            resultant = nk * n_hat - uk * t1v - vk * t2v
            anchors.append(verts[p, v])
            nrm.append(normal_vec)
            res.append(resultant)
    if not anchors:
        empty = np.zeros((0, 3))
        return empty, empty, empty
    return np.array(anchors), np.array(nrm), np.array(res)


def _extent(points_2d):
    """Characteristic in-plane size: the larger bounding-box side."""
    if points_2d.shape[0] == 0:
        return 1.0
    span = points_2d.max(axis=0) - points_2d.min(axis=0)
    return float(max(span.max(), 1e-9))


def _dedupe_order_xz(corners):
    """Project 8 box corners to xz, drop duplicates, order around center."""
    xz = np.column_stack([corners[:, 0], corners[:, 2]])
    seen = {}
    for pt in xz:
        seen[(round(float(pt[0]), 9), round(float(pt[1]), 9))] = pt
    pts = np.array(list(seen.values()))
    center = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(ang)]


def _shade(hex_color, factor):
    """Multiply an RGB hex color by a lightness factor, clip to [0, 1]."""
    h = hex_color.lstrip("#")
    rgb = np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)]) / 255.0
    return tuple(np.clip(rgb * factor, 0.0, 1.0))


def _finish_2d(ax, xmin, xmax, zmin, zmax, extent, title):
    """Common 2D cosmetics: limits, aspect, spines, title."""
    pad = 0.12 * extent
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(min(zmin, 0.0) - 0.10 * extent, zmax + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=STYLE_FONT_SIZE)
    ax.set_ylabel("z (m)", fontsize=STYLE_FONT_SIZE)
    ax.tick_params(labelsize=STYLE_FONT_SIZE - 1)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, fontsize=STYLE_FONT_SIZE)


def plot_assembly_2d(
    assembly,
    result=None,
    *,
    boxes=None,
    ax=None,
    force_scale=None,
    mechanism_scale=None,
    show_forces=True,
    show_mechanism=True,
    title=None,
):
    """Draw a 2D assembly in the xz plane and return the Figure.

    assembly:   keystone.geometry.assembly.Assembly with dim == 2.
    result:     keystone.solve.result.Result or None. Forces draw when
                result.forces is not None; the mechanism draws when the
                result carries one and the status is infeasible.
    boxes:      optional Sequence[keystone.geometry.boxes.Box] used to
                build the assembly. When given, true block outlines are
                drawn from Box.corners() projected to xz. When absent,
                only patches and coms are drawn.
    ax:         existing Axes to draw into, or None for a new figure.
    force_scale, mechanism_scale: arrow lengths in data units per unit
                force or displacement. None means auto from the extent.
    show_forces, show_mechanism: display toggles.
    title:      extra title line placed above the status line.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Arc, Polygon, Rectangle

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 5.0))
    else:
        fig = ax.figure

    verts = np.asarray(assembly.verts)
    vmask = np.asarray(assembly.vert_mask)
    active_verts = verts[vmask]

    # Collect points for extent and axis limits.
    pt_sets = []
    if active_verts.shape[0]:
        pt_sets.append(active_verts[:, [0, 2]])
    if boxes is not None:
        for box in boxes:
            pt_sets.append(box.corners()[:, [0, 2]])
    else:
        com = np.asarray(assembly.com)[np.asarray(assembly.block_mask)]
        if com.shape[0]:
            pt_sets.append(com[:, [0, 2]])
    all_xz = np.vstack(pt_sets) if pt_sets else np.zeros((1, 2))
    extent = _extent(all_xz)
    xmin, zmin = all_xz.min(axis=0)
    xmax, zmax = all_xz.max(axis=0)

    # Ground: hatched band below z = 0 plus a thick line at z = 0.
    band = 0.08 * extent
    gx0, gx1 = xmin - 0.12 * extent, xmax + 0.12 * extent
    ax.add_patch(
        Rectangle(
            (gx0, -band),
            gx1 - gx0,
            band,
            facecolor="none",
            edgecolor=STYLE_GROUND,
            hatch="////",
            linewidth=0.0,
            zorder=0,
        )
    )
    ax.plot([gx0, gx1], [0.0, 0.0], color=STYLE_GROUND, linewidth=2.2, zorder=1)

    # Blocks: true outlines from boxes, else just com dots.
    if boxes is not None:
        for box in boxes:
            quad = _dedupe_order_xz(box.corners())
            ax.add_patch(
                Polygon(
                    quad,
                    closed=True,
                    facecolor=STYLE_BLOCK_FACE,
                    edgecolor=STYLE_BLOCK_EDGE,
                    alpha=STYLE_BLOCK_ALPHA_2D,
                    linewidth=1.3,
                    zorder=2,
                )
            )
            ax.add_patch(
                Polygon(
                    quad,
                    closed=True,
                    fill=False,
                    edgecolor=STYLE_BLOCK_EDGE,
                    linewidth=1.3,
                    zorder=3,
                )
            )
            c = box.com
            ax.scatter([c[0]], [c[2]], s=14, color=STYLE_COM, zorder=6)
    else:
        com = np.asarray(assembly.com)[np.asarray(assembly.block_mask)]
        if com.shape[0]:
            ax.scatter(com[:, 0], com[:, 2], s=14, color=STYLE_COM, zorder=6)

    # Patches: dark segments, slightly offset along the projected normal.
    normal = np.asarray(assembly.normal)
    off = 0.015 * extent
    for p in _active_patches(assembly):
        vs = verts[p][vmask[p]]
        if vs.shape[0] < 2:
            continue
        n_xz = np.array([normal[p, 0], normal[p, 2]])
        seg = vs[:, [0, 2]] + off * n_xz
        ax.plot(seg[:, 0], seg[:, 1], color=STYLE_PATCH, linewidth=2.4, zorder=4)

    legend_handles = []

    # Forces: normal component and full resultant per active vertex.
    if show_forces and result is not None and result.forces is not None:
        anchors, nrm, res = _resultant_vectors(assembly, result.forces)
        if anchors.shape[0]:
            mag = np.linalg.norm(res[:, [0, 2]], axis=1)
            max_force = float(mag.max()) if mag.size else 0.0
            if force_scale is None:
                fs = (
                    STYLE_FORCE_FRACTION * extent / max_force
                    if max_force > 0.0
                    else 0.0
                )
            else:
                fs = force_scale
            ax_xz = anchors[:, 0]
            az_xz = anchors[:, 2]
            nu, nw = nrm[:, 0] * fs, nrm[:, 2] * fs
            ru, rw = res[:, 0] * fs, res[:, 2] * fs
            keep_n = (np.abs(nu) + np.abs(nw)) > 0
            keep_r = (np.abs(ru) + np.abs(rw)) > 0
            if keep_n.any():
                ax.quiver(
                    ax_xz[keep_n], az_xz[keep_n], nu[keep_n], nw[keep_n],
                    angles="xy", scale_units="xy", scale=1.0,
                    color=STYLE_FORCE_NORMAL, width=0.006, zorder=7,
                )
                legend_handles.append(
                    Line2D([0], [0], color=STYLE_FORCE_NORMAL, lw=2,
                           label="normal force")
                )
            if keep_r.any():
                ax.quiver(
                    ax_xz[keep_r], az_xz[keep_r], ru[keep_r], rw[keep_r],
                    angles="xy", scale_units="xy", scale=1.0,
                    color=STYLE_FORCE_RESULTANT, width=0.004, zorder=8,
                )
                legend_handles.append(
                    Line2D([0], [0], color=STYLE_FORCE_RESULTANT, lw=2,
                           label="resultant force")
                )

    # Mechanism: per-block dashed displacement arrow plus rotation marker.
    draw_mech = (
        show_mechanism
        and result is not None
        and result.mechanism is not None
        and result.status == "infeasible"
    )
    if draw_mech:
        mech = np.asarray(result.mechanism)
        com = np.asarray(assembly.com)
        blocks = _active_blocks(assembly)
        trans = mech[blocks][:, [0, 1]]  # (vx, vz) for 2D twist [vx, vz, wy]
        disp_mag = np.linalg.norm(trans, axis=1)
        max_disp = float(disp_mag.max()) if disp_mag.size else 0.0
        if mechanism_scale is None:
            ms = (
                STYLE_MECH_FRACTION * extent / max_disp
                if max_disp > 0.0
                else 0.0
            )
        else:
            ms = mechanism_scale
        drew_any = False
        for b in blocks:
            cx, cz = com[b, 0], com[b, 2]
            vx, wy = mech[b, 0], mech[b, 2]
            vz = mech[b, 1]
            if ms > 0.0 and (abs(vx) + abs(vz)) > 0:
                ax.annotate(
                    "",
                    xy=(cx + vx * ms, cz + vz * ms),
                    xytext=(cx, cz),
                    arrowprops=dict(
                        arrowstyle="->",
                        linestyle="--",
                        color=STYLE_MECHANISM,
                        linewidth=1.6,
                    ),
                    zorder=9,
                )
                drew_any = True
            if abs(wy) > 0:
                r = 0.07 * extent
                sweep = float(np.clip(np.degrees(wy) * 4.0, -150.0, 150.0))
                t1a, t2a = (0.0, sweep) if sweep >= 0 else (sweep, 0.0)
                ax.add_patch(
                    Arc(
                        (cx, cz), 2 * r, 2 * r, angle=0.0,
                        theta1=t1a, theta2=t2a,
                        edgecolor=STYLE_MECHANISM, linewidth=1.4, zorder=9,
                    )
                )
                ax.annotate(
                    f"wy~{_sig3(wy)}",
                    xy=(cx + r, cz + r),
                    fontsize=STYLE_FONT_SIZE - 2,
                    color=STYLE_MECHANISM,
                    zorder=9,
                )
                drew_any = True
        if drew_any:
            legend_handles.append(
                Line2D([0], [0], color=STYLE_MECHANISM, lw=2, ls="--",
                       label="mechanism (normalized twist)")
            )

    if legend_handles:
        # Outside the axes so it never covers blocks or arrows.
        ax.legend(
            handles=legend_handles, fontsize=STYLE_FONT_SIZE - 1,
            loc="upper left", bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0, framealpha=0.8,
        )

    _finish_2d(ax, xmin, xmax, zmin, zmax, extent, _compose_title(result, title))
    fig.tight_layout()
    return fig


def plot_assembly_3d(
    assembly,
    result=None,
    *,
    boxes=None,
    ax=None,
    elev=18,
    azim=-60,
    force_scale=None,
    mechanism_scale=None,
    show_forces=True,
    show_mechanism=True,
    title=None,
):
    """Draw a 3D assembly with mplot3d and return the Figure.

    Arguments match plot_assembly_2d. elev and azim set the view. Box
    faces come from Box.corners() when boxes is given; patches are drawn
    as outlined translucent polygons; forces and mechanism as quivers.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if ax is None:
        fig = plt.figure(figsize=(6.0, 5.0))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    verts = np.asarray(assembly.verts)
    vmask = np.asarray(assembly.vert_mask)
    active_verts = verts[vmask]

    pt_sets = []
    if active_verts.shape[0]:
        pt_sets.append(active_verts)
    if boxes is not None:
        for box in boxes:
            pt_sets.append(box.corners())
    else:
        com = np.asarray(assembly.com)[np.asarray(assembly.block_mask)]
        if com.shape[0]:
            pt_sets.append(com)
    all_pts = np.vstack(pt_sets) if pt_sets else np.zeros((1, 3))
    lo = all_pts.min(axis=0)
    hi = all_pts.max(axis=0)
    extent = float(max((hi - lo).max(), 1e-9))

    # Ground: a light quad at z = 0 spanning the data footprint.
    gx0, gy0 = lo[0] - 0.15 * extent, lo[1] - 0.15 * extent
    gx1, gy1 = hi[0] + 0.15 * extent, hi[1] + 0.15 * extent
    ground = [[
        (gx0, gy0, 0.0), (gx1, gy0, 0.0), (gx1, gy1, 0.0), (gx0, gy1, 0.0),
    ]]
    ax.add_collection3d(
        Poly3DCollection(
            ground, facecolor=STYLE_GROUND_FILL, edgecolor=STYLE_GROUND,
            alpha=STYLE_GROUND_ALPHA, linewidths=0.6,
        )
    )

    # Blocks: 6 shaded quads per box.
    if boxes is not None:
        for box in boxes:
            corners = box.corners()
            for face_idx, loop in enumerate(_BOX_FACES):
                quad = [corners[i] for i in loop]
                ax.add_collection3d(
                    Poly3DCollection(
                        [quad],
                        facecolor=_shade(
                            STYLE_BLOCK_FACE, STYLE_FACE_SHADE[face_idx]
                        ),
                        edgecolor=STYLE_BLOCK_EDGE,
                        alpha=STYLE_BLOCK_ALPHA_3D,
                        linewidths=0.9,
                    )
                )
            c = box.com
            ax.scatter([c[0]], [c[1]], [c[2]], s=16, color=STYLE_COM)
    else:
        com = np.asarray(assembly.com)[np.asarray(assembly.block_mask)]
        if com.shape[0]:
            ax.scatter(com[:, 0], com[:, 1], com[:, 2], s=16, color=STYLE_COM)

    # Patches: outlined translucent dark polygons.
    for p in _active_patches(assembly):
        poly = verts[p][vmask[p]]
        if poly.shape[0] < 3:
            continue
        ax.add_collection3d(
            Poly3DCollection(
                [list(map(tuple, poly))],
                facecolor=STYLE_PATCH, edgecolor=STYLE_PATCH,
                alpha=STYLE_PATCH_ALPHA_3D, linewidths=1.1,
            )
        )

    legend_handles = []

    # Forces: normal and resultant quivers at each active vertex.
    if show_forces and result is not None and result.forces is not None:
        anchors, nrm, res = _resultant_vectors(assembly, result.forces)
        if anchors.shape[0]:
            mag = np.linalg.norm(res, axis=1)
            max_force = float(mag.max()) if mag.size else 0.0
            fs = force_scale
            if fs is None:
                fs = (
                    STYLE_FORCE_FRACTION * extent / max_force
                    if max_force > 0.0
                    else 0.0
                )
            if fs > 0.0:
                ax.quiver(
                    anchors[:, 0], anchors[:, 1], anchors[:, 2],
                    nrm[:, 0] * fs, nrm[:, 1] * fs, nrm[:, 2] * fs,
                    color=STYLE_FORCE_NORMAL, length=1.0, normalize=False,
                    arrow_length_ratio=0.25, linewidth=1.2,
                )
                ax.quiver(
                    anchors[:, 0], anchors[:, 1], anchors[:, 2],
                    res[:, 0] * fs, res[:, 1] * fs, res[:, 2] * fs,
                    color=STYLE_FORCE_RESULTANT, length=1.0, normalize=False,
                    arrow_length_ratio=0.25, linewidth=1.0,
                )
                legend_handles.append(
                    Line2D([0], [0], color=STYLE_FORCE_NORMAL, lw=2,
                           label="normal force")
                )
                legend_handles.append(
                    Line2D([0], [0], color=STYLE_FORCE_RESULTANT, lw=2,
                           label="resultant force")
                )

    # Mechanism: displacement quivers from block coms.
    draw_mech = (
        show_mechanism
        and result is not None
        and result.mechanism is not None
        and result.status == "infeasible"
    )
    if draw_mech:
        mech = np.asarray(result.mechanism)
        com = np.asarray(assembly.com)
        blocks = _active_blocks(assembly)
        block_twist = mech[blocks]
        trans = block_twist[:, :3]
        omega = block_twist[:, 3:6]
        twist_mag = np.linalg.norm(block_twist, axis=1)
        max_twist = float(twist_mag.max()) if twist_mag.size else 0.0
        disp_mag = np.linalg.norm(trans, axis=1)
        max_disp = float(disp_mag.max()) if disp_mag.size else 0.0
        ms = mechanism_scale
        if ms is None:
            ms = (
                STYLE_MECH_FRACTION * extent / max_disp
                if max_disp > 0.0
                else 0.0
            )
        drew_mech = False
        # Translation: displacement quivers from block coms.
        if ms > 0.0:
            ax.quiver(
                com[blocks, 0], com[blocks, 1], com[blocks, 2],
                trans[:, 0] * ms, trans[:, 1] * ms, trans[:, 2] * ms,
                color=STYLE_MECHANISM, length=1.0, normalize=False,
                arrow_length_ratio=0.3, linewidth=1.6,
            )
            drew_mech = True
        # Rotation: a short segment through each com along the omega axis.
        # Sized from the extent, not ms, so a pure spin (zero translation,
        # ms = 0) still draws.
        rot_len = STYLE_MECH_ROT_FRACTION * STYLE_MECH_FRACTION * extent
        omega_mag = np.linalg.norm(omega, axis=1)
        for bi, b in enumerate(blocks):
            wmag = float(omega_mag[bi])
            if max_twist <= 0.0 or wmag <= STYLE_MECH_ROT_CUTOFF * max_twist:
                continue
            axis_hat = omega[bi] / wmag
            p0 = com[b] - 0.5 * rot_len * axis_hat
            p1 = com[b] + 0.5 * rot_len * axis_hat
            ax.plot(
                [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                color=STYLE_MECHANISM, linewidth=1.4, zorder=5,
            )
            ax.text(
                com[b, 0], com[b, 1], com[b, 2], f"|w|~{wmag:.2g}",
                color=STYLE_MECHANISM, fontsize=STYLE_FONT_SIZE - 2,
            )
            drew_mech = True
        if drew_mech:
            legend_handles.append(
                Line2D([0], [0], color=STYLE_MECHANISM, lw=2,
                       label="mechanism (normalized twist)")
            )

    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=STYLE_FONT_SIZE - 1,
                  loc="upper left")

    # Aspect from data ranges, with a small floor to avoid zero spans.
    spans = np.maximum(hi - lo, 1e-6)
    ax.set_box_aspect(tuple(spans))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x (m)", fontsize=STYLE_FONT_SIZE, labelpad=12)
    ax.set_ylabel("y (m)", fontsize=STYLE_FONT_SIZE, labelpad=12)
    ax.set_zlabel("z (m)", fontsize=STYLE_FONT_SIZE, labelpad=8)
    ax.tick_params(labelsize=STYLE_FONT_SIZE - 2)

    # White panes, light gray edges.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor((0.8, 0.8, 0.8, 1.0))
    ax.grid(True, color=(0.9, 0.9, 0.9))

    ttl = _compose_title(result, title)
    if ttl:
        ax.set_title(ttl, fontsize=STYLE_FONT_SIZE)
    return fig


def save_fig(fig, path, dpi=200):
    """Write fig to path as a tight, white-background raster.

    pad_inches keeps mplot3d axis labels inside the tight bbox; the
    tight-bbox computation misses them otherwise.
    """
    fig.savefig(
        path, dpi=dpi, bbox_inches="tight", pad_inches=0.35, facecolor="white"
    )
    return path
