"""Smoke tests for keystone.viz.

These build mock Assembly and Result objects by hand from numpy arrays,
per the frozen data contracts, so the visualization layer is testable
without any geometry, mechanics, or solver code. matplotlib runs under
the Agg backend, set before pyplot is imported anywhere.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from keystone import viz
from keystone.geometry.assembly import Assembly
from keystone.geometry.boxes import Box, box_2d
from keystone.solve.result import FEASIBLE, INFEASIBLE, Result


def _mock_assembly_2d():
    """Two stacked unit blocks, ground patch plus one shared patch."""
    block_mask = np.array([True, True])
    mass = np.array([1.0, 1.0])
    com = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 1.5]])
    patch_mask = np.array([True, True])
    patch_blocks = np.array([[0, 1], [1, 2]], dtype=np.int32)
    normal = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    t1 = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    t2 = np.zeros((2, 3))
    verts = np.array(
        [
            [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
            [[-0.5, 0.0, 1.0], [0.5, 0.0, 1.0]],
        ]
    )
    vert_mask = np.array([[True, True], [True, True]])
    mu = np.array([0.6, 0.6])
    return Assembly(
        block_mask, mass, com, patch_mask, patch_blocks,
        normal, t1, t2, verts, vert_mask, mu, 2,
    )


def _mock_assembly_3d():
    """Two stacked unit cubes with quad patches padded to V = 8."""
    block_mask = np.array([True, True])
    mass = np.array([1.0, 1.0])
    com = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 1.5]])
    patch_mask = np.array([True, True])
    patch_blocks = np.array([[0, 1], [1, 2]], dtype=np.int32)
    normal = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    t1 = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    t2 = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    verts = np.zeros((2, 8, 3))
    square = np.array(
        [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
    )
    for p, zval in enumerate((0.0, 1.0)):
        verts[p, :4, 0] = square[:, 0]
        verts[p, :4, 1] = square[:, 1]
        verts[p, :4, 2] = zval
    vert_mask = np.zeros((2, 8), dtype=bool)
    vert_mask[:, :4] = True
    mu = np.array([0.6, 0.6])
    return Assembly(
        block_mask, mass, com, patch_mask, patch_blocks,
        normal, t1, t2, verts, vert_mask, mu, 3,
    )


def _feasible_result(nf, rng):
    forces = rng.random(nf) + 0.1
    return Result(
        status=FEASIBLE, margin=1e-10, forces=forces,
        lambda_assoc=0.42, mu_critical_assoc=0.31,
    )


def _infeasible_result_2d(nf, rng):
    mech = rng.standard_normal((2, 3))
    return Result(
        status=INFEASIBLE, margin=0.8, forces=rng.random(nf),
        mechanism=mech, lambda_assoc=0.05,
    )


def _infeasible_result_3d(nf, rng):
    mech = rng.standard_normal((2, 6))
    return Result(
        status=INFEASIBLE, margin=0.7, forces=rng.random(nf),
        mechanism=mech, lambda_assoc=0.04,
    )


def _infeasible_result_3d_pure_rotation(nf, rng):
    # Zero translation, nonzero rotation per block: a pure spin.
    mech = np.zeros((2, 6))
    mech[:, 3:6] = rng.standard_normal((2, 3))
    return Result(
        status=INFEASIBLE, margin=0.6, forces=rng.random(nf),
        mechanism=mech, lambda_assoc=0.03,
    )


def _has_content(ax):
    return (len(ax.patches) + len(ax.collections) + len(ax.lines)) > 0


def test_plot_2d_no_result_no_boxes():
    fig = viz.plot_assembly_2d(_mock_assembly_2d())
    assert isinstance(fig, Figure)
    assert _has_content(fig.axes[0])
    plt.close(fig)


def test_plot_2d_feasible_with_boxes(tmp_path):
    rng = np.random.default_rng(0)
    a = _mock_assembly_2d()
    boxes = [box_2d(1.0, 1.0, 0.0, 0.5), box_2d(1.0, 1.0, 0.0, 1.5)]
    r = _feasible_result(2 * 2 * 2, rng)
    fig = viz.plot_assembly_2d(a, r, boxes=boxes, title="stacked pair")
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert _has_content(ax)
    assert len(ax.patches) > 0  # box outlines and ground band
    out = tmp_path / "stack2d.png"
    viz.save_fig(fig, str(out))
    assert out.stat().st_size > 0
    plt.close(fig)


def test_plot_2d_infeasible_shows_mechanism():
    rng = np.random.default_rng(1)
    a = _mock_assembly_2d()
    r = _infeasible_result_2d(2 * 2 * 2, rng)
    fig = viz.plot_assembly_2d(a, r)
    assert isinstance(fig, Figure)
    assert _has_content(fig.axes[0])
    assert INFEASIBLE in fig.axes[0].get_title()
    plt.close(fig)


def test_plot_3d_no_result_no_boxes():
    fig = viz.plot_assembly_3d(_mock_assembly_3d())
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].collections) > 0
    plt.close(fig)


def test_plot_3d_feasible_with_boxes(tmp_path):
    rng = np.random.default_rng(2)
    a = _mock_assembly_3d()
    boxes = [
        Box(np.array([0.5, 0.5, 0.5]), np.array([0.0, 0.0, 0.5])),
        Box(np.array([0.5, 0.5, 0.5]), np.array([0.0, 0.0, 1.5])),
    ]
    r = _feasible_result(3 * 2 * 8, rng)
    fig = viz.plot_assembly_3d(a, r, boxes=boxes, title="two cubes")
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].collections) > 0
    out = tmp_path / "tower3d.png"
    viz.save_fig(fig, str(out))
    assert out.stat().st_size > 0
    plt.close(fig)


def test_plot_3d_infeasible_shows_mechanism():
    rng = np.random.default_rng(3)
    a = _mock_assembly_3d()
    r = _infeasible_result_3d(3 * 2 * 8, rng)
    fig = viz.plot_assembly_3d(a, r)
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].collections) > 0
    plt.close(fig)


def test_plot_3d_pure_rotation_draws_axis_line():
    # A pure spin has zero translation, so the arrow scale collapses and no
    # displacement quiver is drawn. The rotation axis must still render as a
    # Line3D per rotating block. Line artists land in ax.lines; the base 3D
    # plot uses none, so any line here is new.
    rng = np.random.default_rng(7)
    a = _mock_assembly_3d()
    base = viz.plot_assembly_3d(a)
    assert len(base.axes[0].lines) == 0
    plt.close(base)

    r = _infeasible_result_3d_pure_rotation(3 * 2 * 8, rng)
    fig = viz.plot_assembly_3d(a, r)
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].lines) >= 1
    plt.close(fig)


def test_save_fig_writes_png(tmp_path):
    fig = viz.plot_assembly_2d(_mock_assembly_2d())
    out = tmp_path / "plain.png"
    viz.save_fig(fig, out)
    assert out.exists()
    assert out.stat().st_size > 0
    plt.close(fig)
