"""External load vectors.

Dead load is self-weight, gravity along -z. The live load is the
default P2 loading: horizontal pseudo-static gravity along +x, scaled
per block by its weight. Both are assembled nondimensionally (forces
divide by W). Torques about each block's own com vanish for both.
"""

import numpy as np

DEFAULT_G = 9.81  # m/s^2, standard gravity. Not a tolerance.


def dead_and_live_loads(mass, block_mask, W, dim: int):
    """Return (w_dead, w_live), each (rows_per_block * N,), nondimensional.

    Dead load: self-weight along -z, force row Fz gets -m_i g / W. Live
    load: horizontal pseudo-static gravity along +x, force row Fx gets
    +m_i g / W. Torque rows are zero for both (torques taken about each
    block's own com). Masked blocks contribute zero rows.
    """
    mass = np.asarray(mass, dtype=np.float64)
    block_mask = np.asarray(block_mask, dtype=bool)
    n = mass.shape[0]
    rpb = 3 if dim == 2 else 6
    w_dead = np.zeros(rpb * n, dtype=np.float64)
    w_live = np.zeros(rpb * n, dtype=np.float64)
    # Row offsets of Fx and Fz within a block: 2D rows are [Fx, Fz, Ty],
    # 3D rows are [Fx, Fy, Fz, Tx, Ty, Tz].
    fx = 0
    fz = 1 if dim == 2 else 2
    for bi in range(n):
        if not block_mask[bi]:
            continue
        weight = mass[bi] * DEFAULT_G / W
        w_dead[rpb * bi + fz] = -weight
        w_live[rpb * bi + fx] = weight
    return w_dead, w_live
