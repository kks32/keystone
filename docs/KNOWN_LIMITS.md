# KNOWN_LIMITS

Recorded gaps and sharp edges. Each entry states the limit, the impact, and the
planned resolution. Dates are when the entry was recorded.

## Prefix feasibility does not model placement kinematics (2026-07-15, updated 2026-07-16)

Sequence feasibility certifies each intermediate block SET as a static
equilibrium. The motion between configurations is not checked by default: a
block may be "placed" into a pocket that a physical manipulator could only
reach by sliding in from the side, or not at all. This became load-bearing
with the certified n=4 clamp design (overhang 1.25), whose final cube slides
in under an existing bridge.

Update: the lattice search now offers placement-reachability modes on
`LatticeSpec.mode` and `--placement` in `examples/certify_overhang.py`.
"static" is the default and checks nothing kinematic (bitwise the old
behavior). "drop" requires a clear vertical column above the target cell at
placement time (a crane or top-grasp build). "slide" accepts a clear column
or a clear straight lateral corridor at the target layer, entered from either
side past every placed cube. "slide_clear" is the executable slide: it also
forbids corridors that pass under a layer-(L+1) bridge, target cell included,
because the lattice's exact unit clearance makes that pass a zero-clearance
press fit (rigid-body simulation of the certified clamp shows the slide-in
jamming at about 700x block weight and collapsing the structure).
Reachability is evaluated against the state before each placement, so it
depends on build order, not just the final set.

What the modes model, and what they do not. Clearances are the exact unit
gaps of the lattice: a layer-(L+1) bridge clears a layer-L slide by
construction because layer heights are exact, and interval overlaps are open
(touching faces do not block). Motions are single-axis and straight. No
gripper, tool, or finger clearance is modeled; a physical end effector needs
side or top room the lattice does not budget for. No swept-volume physics:
the moving cube is assumed massless in transit and the structure is only
checked statically before and after.

Certified ladder at n=4, dx=1/12 (branch and bound,
tests/analytic/test_bnb_optima.py): static 5/4 = slide 5/4 > slide_clear 1 =
drop 1. The idealized slide costs nothing, the 5/4 clamp build order is
slide-legal at every step, but every step of that value rides on the
unexecutable under-bridge pass: banning it (slide_clear) collapses the
optimum to the crane value 1, so no executable-slide order reaches even the
MCTS 7/6 design. Drop-only pins the optimum at exactly 1 on both grids:
price 1/4 block width against 5/4 static at dx=1/12 (two towers of two) and
7/24 against 31/24 at dx=1/24 (a four-high staircase). At n=3, dx=1/12 drop
is free (5/6 either way). Resolution for real hardware: a sweep-collision
check with end-effector geometry belongs to the robot planner, not this
lattice.

## Two L definitions (2026-07-15, updated 2026-07-15)

Interface detection scales tolerances by the bounding-box diagonal of all block
corners. Nondimensionalization in `assemble` uses `bbox_diagonal(assembly)`, which
measures active patch vertices and centers of mass only. The two values differ
(example: a single unit block gives sqrt(3) vs sqrt(1.25)). Both are deterministic
and each is used consistently, so verdicts are unaffected. Resolution: unify on the
corner-based diagonal when the Assembly structure grows a corners record (M2).

Update: the detection L in `build_assembly` now zeroes the y column before the
diagonal in 2D, so the arbitrary out-of-plane depth (box_2d defaults to 1 m) no
longer moves g_tol scaling. Two touching 2D blocks detect identically at depth 1 m
and depth 100 m. The two-L design still stands; only the 2D depth dependence is
removed. `bbox_diagonal` now raises a clear ValueError on a single floating block
with no patches (zero spatial extent) instead of returning L = 0.

## Gap convention: g_tol is the maximum total face separation (2026-07-15)

`g_tol` means the largest total separation of a contact, the distance between
the two opposing faces. The ground path measures a block face directly against
z = 0 and accepts up to `g_tol * L`. The block-block path measures each corner
deviation from the mid-plane and accepts up to `g_tol * L / 2`, so the total
separation (twice the mid-plane deviation for parallel faces) also stays within
`g_tol * L`. Before this change the block-block path spent `g_tol * L` on the
mid-plane deviation, which admitted total separations up to `2 * g_tol * L`,
twice the ground budget. The two paths now share one meaning of `g_tol`.
Borderline detection tests were recomputed against the halved block-block
threshold.

## Interpenetration checking is AABB-conservative only (2026-07-15)

`check_no_interpenetration` in `interfaces.py` runs at the top of both 2D and
3D detection. It flags a pair only when the two world axis-aligned bounding
boxes overlap by more than `2 * g_tol * L` on every axis at once, and raises a
ValueError naming the pair. This is a cheap guard against gross modeling
errors (for example two coincident blocks), not full collision rejection. It
uses AABBs, so it can miss oriented-box overlaps that no axis-aligned box
reveals, and it never flags a legitimate tight contact because the overlap on
the contact-normal axis stays near zero (contact penetration is bounded by
`g_tol * L`). A pair tilted so one corner dips more than `2 * g_tol * L` into
the neighbor is flagged even though the faces nearly meet at their centers;
that is the intended catch for an overlapping input. Resolution: a proper
narrow-phase separating-axis test lands with the mesh pipeline (M2 onward).

## Verdicts at exact analytic boundaries (2026-07-15)

At a mathematically exact limit state (corbel scale c = 1.0), the P4 margin lands
within the tolerance band around `tol_feas` and the verdict is band-sensitive.
This is inherent to floating point limit analysis, not a bug. Gates therefore
assert strictly inside and outside the boundary (c = 1 +- 1e-3). Callers should
treat `|margin - tol_feas| < 10 * tol_feas` as an escalation band and re-verify
with `solve_p0_exact`.

## P4 feasible-side margin equals the Tikhonov bias (2026-07-15)

At a feasible state the elastic QP optimum does not reach s = 0; it stops at a
bias linear in `eps_reg` and proportional to the squared contact force norm
(observed: margin = 560 * eps_reg on a 12-block 2D tower near collapse; stable
from eps_reg 1e-10 down to 1e-14). Infeasibility residuals are eps-independent.
Consequence: the feasibility verdict fails when the bias crosses `tol_feas`,
which happened at the old default eps_reg = 1e-10 for towers of 8 or more
blocks. Default is now 1e-12 (CHANGELOG). Escalation for very large or badly
conditioned assemblies: re-solve with eps_reg / 100; a margin that drops about
100x is bias (feasible), a margin that stays put is real (infeasible).

## qpax converged flag is pessimistic (2026-07-15)

With `eps_reg = 1e-10` and `solver_tol = 1e-9`, qpax's internal slack floor
(sqrt machine epsilon, about 1.5e-8 in float64) keeps the complementarity block of
the KKT residual above the tolerance whenever a cone facet is active. The flag then
reads 0 even for fully resolved solves. Verdicts are margin-primary; the raw flag
and iteration count are recorded in `Result.info`. Resolution: revisit if qpax
exposes a component-wise convergence test.

## qpax divergence at extreme load factors (2026-07-15)

At lambda far beyond collapse (for example the lam_hi = 16 bracket cap on a unit
block), the interior point can emit NaN. `solve_p2` sanitizes non-finite margins to
infeasible, which keeps bisection correct. Mechanisms are read from the finite side
of the bracket.

## Pyramid cone conservatism (2026-07-15)

`cone="pyramid", k` underestimates tangential capacity by up to `1 - cos(pi/k)`
(about 7.6 % at k = 8) in facet-mid directions. Polygon vertices are aligned so the
+t1 direction is exact. This is the documented inscribed (conservative) choice, per
charter item 4. `cone="socp"` raises NotImplementedError until a conic backend
lands (Clarabel on CPU, or a JAX SOCP).

## Gradients at contact topology switches (2026-07-15)

Patch vertices are piecewise smooth functions of pose. Gradients are undefined
exactly where the contact topology changes (faces gaining or losing overlap,
vertex count changes in clipping). Differentiation tests sample away from
switches. Users of `margin_and_grad` should expect subgradients near these loci.

## 2D live load direction (2026-07-15)

`w_live` is horizontal +x per unit block weight. Other live loadings (point loads,
applied wrenches, arbitrary directions) need `dataclasses.replace` on the
`EquilibriumSystem` for now. A loads API arrives with M5 falsework support.

## Patch holes (from charter Section 5)

Not reachable in the cube slice (box faces clip to convex polygons). The v1 policy
for the mesh pipeline remains: drop polygons with holes and log a warning.

## PDHG screening semantics (2026-07-15)

`solve.pdhg.pdhg_margin` is a fixed-iteration first-order screen of the P4
problem, not a certifier. FEASIBLE claims that reach a user always come from the
certified path (`margin_core` plus the verified verdict, or `solve_p0_exact`).
The search (`search.mcts`, `screener="pdhg"`) uses screened margins for expansion
admissibility only and gates every best-overhang update on a certified qpax
re-verification.

Error direction, measured, not proven: on the fixed-seed validation study in
`tests/unit/test_pdhg.py` (500 random reachable lattice states from the n=4 and
n=6 specs plus 10 states within 2 grid steps of the stacked-pair e = b/2 and
harmonic-corbel boundaries), the screen produced zero false FEASIBLE verdicts at
every tested iteration count (50 to 800, cold and warm started). A feasible
state's screened margin decreases toward zero from above, so truncation reads
conservative. This is an empirical property of this problem family, asserted in
CI, and not a theorem; a new problem family needs its own study.

False INFEASIBLE is common and expected: at the default 400 iterations about half
of the certified-feasible study states still screen infeasible, because feasible
states adjacent to an analytic boundary have certified margins within 10x of
`tol_feas` (the escalation band above) and a few hundred first-order iterations
cannot resolve a residual to 1e-8 on systems with cond(A^T A) in the thousands.
The target of a sub-2 % false-infeasible rate was not reachable at any tested
iteration count (still 4.8 % on n=6 at 6400 iterations, where the screen is
already slower than qpax). In the search a false INFEASIBLE only prunes
exploration; the certified re-verification of improving states recovers the
boundary states that matter for the reported best. The n=4 sims=500 seed=0
search returns the same best overhang and sequence under both screeners, which
CI asserts. At larger budgets the two searches diverge: at sims=2000 the qpax
search finds deeper counterweighted structures (best 0.958 at n=4, 1.000 at
n=6) that the screened tree has permanently pruned (best 0.708, unchanged even
with 6x more simulations; see `out/search/search_perf_pdhg.txt`). The pdhg
screener trades search quality at equal simulations for a 6x wall-time
speedup on CPU; on this objective, which rewards exactly the boundary states
the screen resolves worst, that trade is currently unfavorable at large
budgets.

Acceleration on infeasible instances: the primal Nesterov wrapper is restarted
adaptively and extrapolates the primal only. Dual momentum is deliberately off;
an infeasible instance has no saddle point, its dual iterate grows without
bound, and momentum on it compounds the growth geometrically (observed 1e5
margin overshoot). With primal-only momentum the screened margin of an
infeasible state stays finite and tracks the true residual infimum from above.

## MuJoCo soft contacts topple knife-edge certified optima (2026-07-16)

The certified overhang optima are exact limit states. Their P4 elastic margins
are near machine zero (corbel c=0.98: 3e-11; clamp 31/24: 2e-11; n=6 4/3: 3e-11),
which certifies static equilibrium but signals zero margin to perturbation.
MuJoCo's default contact model is soft (solref time constant 0.02 s). Under it
these structures topple in the 2 s settle test. The infeasible controls (corbel
c=1.02, offset pair e=0.55, clamp at mu=0.3) also fall, so those agree.
Stiffening the contacts (solref time constant toward 0.002 s) recovers stability
in fragility order: corbel first, then n=6, then the clamp, which needs
near-rigid contacts to stand (rotation 0.005 rad at solref 0.002). This is not a
bug in either system. It is the model gap of PLAN.md Section 8.4 measured:
associative limit analysis ranks capacity above MuJoCo's regularized, compliant
model, and a zero-margin optimum has nothing to spend on compliance. keystone
reports the exact verdict and is never tuned toward the simulator. Numbers:
examples/mujoco_validate.py, out/mujoco/mujoco_validate.json. Downstream search
that wants a dynamic safety margin should back off the reacher (next entry) or
add a margin floor to the objective.

## Default-softness settling reads as displacement on tall aligned stacks (2026-07-16)

A 5-block aligned tower is certified feasible and physically stable, but at
MuJoCo's default contact softness the settle test flags it unstable: the stack
sags vertically 0.037 m (0.0072 L), just past the 0.005 L displacement band,
while its rotation stays near zero (2e-4 rad). The signature (displacement
exceeded, rotation near zero) separates compliance sag from a topple. The sag is
soft-contact compression accumulating over stacked interfaces and vanishes under
stiffer contacts (0.0072 L at default down to 5e-5 L at solref 0.002). The settle
verdict is therefore contact-stiffness dependent near the band. The harness
reports both the default-softness result and a stiffness sweep. Resolution: read
the rotation channel alongside displacement, or run the sweep, when a stack fails
on displacement alone.

## Placement motions are obstructed for both reacher designs (2026-07-16, updated 2026-07-16)

The first entry in this file records the lattice placement-reachability modes
(static, drop, slide) added to the search layer. Those modes reason about clear
columns and corridors with exact unit gaps and open interval overlaps, and they
assume the moving cube is massless in transit. examples/mujoco_insert.py runs the
missing swept-volume physics check in MuJoCo and finds both counterweighted
reachers obstructed under rigid-body contact:

- n=4 MCTS 7/6 reacher (drop path). This design was found under the static mode,
  which checks nothing kinematic. The layer-2 counterweight overhangs the
  reacher's target column by 1/12 of a block width, so the column is not clear. A
  vertical drop starts 0.083 m inside that counterweight (measured as start-pose
  contact penetration) and cannot seat the block. MuJoCo agrees with the stricter
  lattice drop mode, which also rejects this placement: the "drop-legal" label
  attached to 7/6 came from the permissive static search, not from a drop check.
- n=4 clamp 31/24 reacher (slide path). The lattice slide mode accepts this
  placement: the layer-2 bridge clears a layer-1 slide by construction because
  layer heights are exact and touching faces do not block. Rigid-body physics does
  not have that luxury. The bridge and the base cube form a slot exactly one block
  high (bridge bottom z=3.0, base top z=2.0, reacher height 1.0). Sliding a unit
  cube into a zero-clearance slot jams: peak contact force 1.4e7 N (about 700x a
  block weight), the already-placed structure is shoved 0.19 m (0.023 L) during
  the slide, and after release the structure collapses (2.0 m settle
  displacement). The idealized slide is not executable quasi-statically at this
  grid.

The three counterweight drops in each design (base and both counterweights) place
cleanly: structure displacement below 0.01 m, no disturbance. Impact: a sequence
that passes the lattice reachability modes, is prefix-feasible, and even stands
under stiff contacts can still be non-constructible because no collision-free
rigid placement path exists for the reacher. The lattice modes are necessary but
not sufficient; they need a swept-volume and end-effector check for real hardware.
Resolution: that check belongs to the robot planner, or the grid needs sub-unit
clearance; the solver certifies statics only. Numbers: out/mujoco/mujoco_insert.json.

Update, Route A: block tolerance plus compliant insertion does not rescue the
reacher slide, and the reason is structural, not a tuning gap. Shrinking every
cube by size_tol (in-plane side 1 - size_tol) and re-stacking so vertical
contacts meet keeps the full designs certified feasible at every tested
tolerance (clamp margin 1.9e-11 to 1.8e-11, n6 2.7e-11 across size_tol 0 to
0.02), but the reacher slot never opens: its floor and ceiling are blocks that
shrink with the reacher, so the slot height stays exactly one reacher height.
The opposite assignment, nominal slot with a shortened reacher, opens a
size_tol gap under the clamping block and is certified infeasible for any
gap > 0 (clamp margin 7.7e-3, n6 6.2e-3): the clearance a rigid slide needs is
the clamp contact that holds the reacher up. Replacing the rigid weld drive
with a capped impedance driver (spring-damper on the free block, force capped
at max_push) converts the 700x jam into a bounded outcome, but the slide still
fails at every tolerance: at cap 1x block weight the reacher stalls 0.61 m
short with the structure intact and retrievable; at 2x and above it wedges at
the bridge edge (stops 0.05 to 0.07 m short) and the shove topples the
knife-edge structure during insertion, not after release. Two shrunk-geometry
side effects are worth recording: uniform in-plane shrink turns the certified
counterweight prefix infeasible (its center of mass moves past the shrunk
support edge, margin 2.4e-4 at size_tol 0.005), so the shrunk designs are not
prefix-buildable in the certified order either. Route A is closed for this
topology: no (design, size_tol) pair in {clamp 31/24, n6 4/3} x {0, 0.005,
0.0075, 0.01, 0.02} inserts cleanly and stands. Numbers and the full table:
out/mujoco/mujoco_insert.json (route_a).

Update, Route B: falsework executes both designs with drops only,
examples/mujoco_falsework.py. Slender prop columns are declared under every
overhang that is unstable before its counterweight or clamp arrives (reacher
prop on the ground, counterweight prop on the pedestal, plus a base prop for
n6 whose base cube sits centered on the pedestal edge), each propped prefix is
certified through the host pipeline as an ordinary block assembly (all
feasible, margins 4e-12 to 1.4e-11), the build runs as capped-impedance drops
onto the propped structure, and the props retract one at a time on
position-actuated sliders. Execution details that mattered: props are set as
catchers 1 mm shy of the underside (a flush static prop over-constrains the
drop press and ejects the block), and prop tops in the retraction model are
set from settled poses with the same relief (a flush slider-held prop drags
the block during slide-out). All drops seat to within 2.3 mm, prop peak loads
stay under 4.3x block weight during placement impacts and read zero before
retraction (the finished clamp carries itself). Outcome: the n6 4/3 certified
optimum builds and stands after retraction (rotation 0.010); the clamp 31/24
executes cleanly but creep-topples after retraction exactly like its
exact-pose control (the zero-margin knife-edge creeps about 0.0026 rad/s under
MuJoCo's compliant contacts at solref 0.002 and is unstable by 6 s with no
build involved), and backing the reacher off two grid steps (29/24) gives a
falsework build that stands. The practical ladder at n=4, dx=1/24 is
therefore: statically certified 31/24, falsework-buildable-and-standing 29/24,
clear-space (drop) 1. Numbers: out/mujoco/mujoco_falsework.json.
