# keystone

Static equilibrium for assemblies of rigid blocks with dry frictional joints.
Given a stack of blocks, keystone answers: can it stand, by what margin, how
does it collapse, and how much extra load it takes before it does. Solvers run
in JAX, so thousands of assemblies solve in one batched call and the margin is
differentiable with respect to the applied loads.

Verdicts are certificates, not solver flags. A feasible answer comes with a
force state you can check against equilibrium and the friction cone. An
infeasible answer comes with a validated collapse mechanism (a Farkas ray).
Anything the solver cannot certify is reported as no_converge, never guessed.
Every result is cross-checked against LP oracles (scipy HiGHS) in the tests,
and against compas_cra on tilt benchmarks.

## Install

    uv venv --python 3.12
    uv pip install -e ".[dev,viz]"

## Run

    python -m pytest tests/ -q                      # 113 tests
    python examples/stack2d.py --out out            # 2D stacks, rendered
    python examples/tower3d.py --out out            # 3D tower, rendered
    python examples/search_overhang_fast.py --n 4 --sims 2000   # MCTS overhang search

The search example places cubes one at a time, requires every intermediate
structure to be certifiably stable, and maximizes how far past a table edge
the stack can reach. With 4 cubes it finds a counterweighted design reaching
1.167 block widths, past the 25/24 harmonic limit of simple stacks and within
0.1% of the known unconstrained optimum.

## GPU

The solvers and the search kernel are pure JAX with fixed shapes, so the same
code runs on CUDA. Install the CUDA build of jax through the extra:

    uv pip install -e ".[cuda12]"     # or .[cuda13]
    python -c "import jax; print(jax.devices())"    # expect CudaDevice

Then run the same commands as above. Batched solves are the payoff: on CPU the
QP batch runs serially (about 800 solves/s); a data-center GPU runs the batch
concurrently. keystone computes in float64 by default for correctness, which
is fast on A100/H100 class hardware and slow on consumer cards. Recorded
performance numbers live in bench/ and only come from committed scripts.

## Layout

    src/keystone/geometry/    boxes, interface detection, tolerances
    src/keystone/mechanics/   equilibrium matrix, loads, friction cones
    src/keystone/solve/       qpax QP solvers, LP oracles, certificates
    src/keystone/search/      jittable lattice environment, batched MCTS
    docs/                     MATH_REFERENCE.md, KNOWN_LIMITS.md

Scope today: box-shaped blocks in 2D and 3D. General meshes, the
non-associative friction bracket, and MuJoCo interop are planned; see
CHANGELOG.md for what changed and docs/KNOWN_LIMITS.md for sharp edges.

## License

MIT. See LICENSE.
