# keystone

Static equilibrium for assemblies of rigid blocks with dry frictional joints.
Given a stack of blocks, keystone answers four questions: can it stand, by what
margin, how does it collapse, and how much extra load it takes before it does.
The solvers run in JAX with fixed array shapes, so thousands of assemblies
solve in one batched call and the stability margin is differentiable with
respect to the applied loads.

## Install

    uv venv --python 3.12
    uv pip install -e ".[dev,viz]"

## Run

    python -m pytest tests/ -q                          # the test suite, about 250 tests
    python examples/stack2d.py --out out                # 2D stacks, rendered
    python examples/tower3d.py --out out                # 3D tower, rendered
    python examples/certify_overhang.py --n 4 --dx 1/12 # prove the max-overhang optimum

This example places unit cubes one at a time on a pedestal, requires
every intermediate stack to verify as a static equilibrium, and searches for
the farthest reach past the pedestal edge. At 4 cubes on a $1/12$ grid with
friction 0.7 it proves the grid optimum is 5/4 block widths, past the 25/24
harmonic limit of a simple book stack, and reports the winning build order. The
proof is a best-first branch and bound with an admissible geometric bound;
`docs/MATH_REFERENCE.md` derives the bound and the optimality argument.

## GPU

The solvers and the search kernel are pure JAX with fixed shapes, so the same
code runs on CUDA. Install a CUDA build of jax through the matching extra:

    uv pip install -e ".[cuda12]"     # or .[cuda13]
    python -c "import jax; print(jax.devices())"    # expect CudaDevice

Then run the same commands. Batched solves are the payoff. On CPU the QP batch
runs serially; a data-center GPU runs the batch concurrently. keystone computes
in float64 by default for correctness, which is fast on A100 and H100 class
hardware and slow on consumer cards. Recorded performance numbers live in
`bench/` and come only from committed scripts.

## Positioning

The engineering claims are speed and batching. The whole geometry-to-margin
path is one jittable, vmappable function of fixed shape, so a large family of
assemblies solves in a single device call and the margin carries a gradient with
respect to the loads.

Two correctness properties hold at the size of a single assembly. First, an
infeasible verdict carries a checked collapse mechanism: a virtual twist $y$
with $y^\top w > 0$ and $A^\top y + G^\top z = 0$ for some $z \ge 0$, the dual
ray of the equilibrium problem. Tools in the rigid-block-equilibrium lineage
report a solver status here and no motion. Second, the solver verifies or
abstains. Every verdict is one of feasible, infeasible, or no_converge, and each
decided value is rechecked on recomputed quantities before it is returned rather
than read off a solver flag. This matters at scale, because one optimality proof
consumes on the order of $10^5$ verdicts and a silent misread would corrupt the
proof. A feasible force state is a recheckable witness, and that is not a new
idea: a zero-tension rigid-block-equilibrium solution is the same kind of
witness, so keystone claims nothing novel on the feasible side.

One caveat is load-bearing. Every solver uses the associative friction model,
which overestimates true Coulomb capacity. A falls-verdict from keystone
transfers to the true frictional model, because the true model has less
capacity. A stands-verdict does not transfer: the associative model can stand
where true Coulomb friction slides. The `assoc` fields and the
`physical_bound_direction` field on every `Result` state which way the reported
number bounds the true value; `docs/MATH_REFERENCE.md` Section 2 derives it.

## Layout

    src/keystone/geometry/    boxes, interface detection, tolerances
    src/keystone/mechanics/   equilibrium matrix, loads, friction cones
    src/keystone/solve/       qpax QP solvers, LP oracles, verdict checks
    src/keystone/search/      jittable lattice environment, MCTS, branch and bound
    docs/                     MATH_REFERENCE.md, KNOWN_LIMITS.md

Scope today is box-shaped blocks. General meshes, the non-associative bracket,
and compas interop are planned; see `CHANGELOG.md` for what changed and
`docs/KNOWN_LIMITS.md` for sharp edges.

## A note on the name

"keystone" is a placeholder, an open decision in the project charter. The
system today handles box assemblies only. Vault geometry, the setting a real
keystone belongs to, is the planned M2 milestone and does not exist yet. The
name looks ahead to that milestone.

## License

MIT. See LICENSE.
