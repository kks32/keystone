"""Box blocks. The only block shape in the cube-stacking slice.

Conventions (CLAUDE.md Section 4):
- 2D lives in the xz plane, gravity along -z. 2D boxes rotate about y only.
- Quaternions are unit, scalar first (w, x, y, z).
- The centroid coincides with the box center, so com equals position.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Box:
    """A rigid box block.

    half_extents: (3,) meters, half sizes along the local x, y, z axes.
    position:     (3,) meters, world coordinates of the center.
    quat:         (4,) unit quaternion (w, x, y, z), local to world.
    density:      kg/m^3.
    """

    half_extents: np.ndarray
    position: np.ndarray
    quat: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0])
    )
    density: float = 2000.0

    def __post_init__(self):
        he = np.asarray(self.half_extents, dtype=np.float64)
        if he.shape != (3,):
            raise ValueError(f"half_extents must have shape (3,), got {he.shape}")
        if not np.all(np.isfinite(he)):
            raise ValueError(f"half_extents must be finite, got {he}")
        if np.any(he <= 0.0):
            raise ValueError(f"half_extents must be positive, got {he}")
        object.__setattr__(self, "half_extents", he)
        pos = np.asarray(self.position, dtype=np.float64)
        if pos.shape != (3,):
            raise ValueError(f"position must have shape (3,), got {pos.shape}")
        if not np.all(np.isfinite(pos)):
            raise ValueError(f"position must be finite, got {pos}")
        object.__setattr__(self, "position", pos)
        if not np.isfinite(self.density) or self.density <= 0.0:
            raise ValueError(f"density must be finite and positive, got {self.density}")
        q = np.asarray(self.quat, dtype=np.float64)
        if q.shape != (4,):
            raise ValueError(f"quat must have shape (4,), got {q.shape}")
        n = np.linalg.norm(q)
        if not np.isclose(n, 1.0, rtol=0, atol=1e-9):
            raise ValueError(f"quaternion norm {n} is not 1")
        object.__setattr__(self, "quat", q / n)

    @property
    def rotation(self) -> np.ndarray:
        """(3, 3) rotation matrix, local to world."""
        w, x, y, z = self.quat
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )

    @property
    def volume(self) -> float:
        return float(8.0 * np.prod(self.half_extents))

    @property
    def mass(self) -> float:
        return self.density * self.volume

    @property
    def com(self) -> np.ndarray:
        """World center of mass. Boxes are uniform, so this is position."""
        return self.position

    def corners(self) -> np.ndarray:
        """(8, 3) world corner coordinates, fixed sign order."""
        signs = np.array(
            [
                [-1, -1, -1],
                [+1, -1, -1],
                [-1, +1, -1],
                [+1, +1, -1],
                [-1, -1, +1],
                [+1, -1, +1],
                [-1, +1, +1],
                [+1, +1, +1],
            ],
            dtype=np.float64,
        )
        local = signs * self.half_extents
        return local @ self.rotation.T + self.position


def box_2d(
    width: float,
    height: float,
    x: float,
    z: float,
    angle_y: float = 0.0,
    density: float = 2000.0,
    depth: float = 1.0,
) -> Box:
    """A 2D block: a box in the xz plane with unit depth along y.

    width spans x, height spans z. angle_y is the rotation about the
    world y axis in radians. Mass uses the full 3D volume, so a 2D
    block of depth 1 m weighs width * height * 1 * density * g.
    """
    half = np.array([width / 2.0, depth / 2.0, height / 2.0])
    quat = np.array(
        [np.cos(angle_y / 2.0), 0.0, np.sin(angle_y / 2.0), 0.0]
    )
    return Box(half, np.array([x, 0.0, z]), quat, density)
