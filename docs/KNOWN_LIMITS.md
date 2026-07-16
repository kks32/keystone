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

Update, tilt negative control: tilting the reacher on approach does not open the
zero-clearance slot, as the geometry demands. examples/mujoco_insert.py adds a
`tilt_deg` mode to the compliant slide. At size_tol = 0 (nominal geometry) the
clamp reacher never seats: straight (0 deg) collapses the structure (peak contact
9.2e5 N), 2 deg collapses it (8.8e5 N), 5 deg wedges so hard it stalls 3 cm short
without toppling (7.5e5 N). No tilt reaches the target. A tilted unit square spans
cos(theta) + sin(theta) > 1 vertically, so a tilt can only make a one-block slot
tighter. Numbers: out/mujoco/mujoco_insert.json (tilt_control).

Update, hold-and-shim: split insertion from statics. Replace the reacher with a
short reacher of height 1 - eps plus a shim plate of thickness eps that fills the
gap above the reacher's tail. examples/mujoco_shim.py. The final state is all
boxes, so keystone certifies it exactly; the split certifies feasible for eps in
{0.01, 0.02, 0.04} and both shim footprints (full plate, tail plate), full
footprint carrying the better margin (about 1.8e-11 to 3.3e-11 across designs; the
full plate reproduces the reacher's mass and center of mass exactly). The short
reacher alone is infeasible (gap margins 2.6e-3 to 7.7e-3): the eps gap above the
reacher un-certifies the clamp, which is why the shim exists.

Two findings. First, hold-and-shim solves the insertion. The short held reacher
slides into the one-block slot with eps clearance and seats to 0.6 mm with the
placed structure undisturbed (0.0001 L), and the shim seats to 0.02 to 0.16 mm
under a drive that peaks near one block weight, where the monolithic reacher
jammed at about 700x block weight (Route A). A held block need not stand, so the
insertion is decoupled from the statics.

Second, the shim seam is a slip and compliance path that lowers the clamp's
dynamic margin, and it compounds with a deep-thread insertion barrier, so the
buildable-and-standing overhang backs off below the monolith. The split at exact
poses topples where the monolith merely crept: at solref 0.002 the clamp split
first stands (6 s) at 25/24 and the n6 split at 7/6, below the monolith's
falsework-standing 29/24 and 4/3. But standing needs the reacher backed off, which
threads it deeper under the bridge, and the cantilevered reacher rams the bridge on
the way in: clamp 25/24 (thread 7/24) flips the bridge 90 deg and shoves the
structure 0.38 L, while clamp 31/24 (thread 1/24) and 26/24 (thread 6/24) insert
cleanly. The n6 4/3 base cube is a knife-edge on the pedestal rim that the reacher
slide tips, so the n6 reacher stalls 38 mm short. The narrow window that both
inserts and stands is clamp 26/24 at eps = 0.02: it builds end to end (reacher
0.6 mm, shim 0.16 mm, shim push about one block weight) and stands (rotation 0.007
at 2 s and 6 s, solref 0.002). At eps = 0.01 the same build creep-topples (the
build disturbance tips the near-zero-margin split) and at eps = 0.04 the thicker
shim seats 4.7 mm off, so the working window is one eps wide. Verdict: hold-and-
shim makes a backed-off clamp buildable and standing without falsework (26/24 at
n=4, dx=1/24), one grid step below the falsework value and five below the
statically certified optimum; the full 31/24 optimum inserts cleanly but its split
creep-topples. The insertion blocker is removed; the shim seam pays for it in
overhang. Numbers and the standing-threshold scan: out/mujoco/mujoco_shim.json;
build movie out/mujoco/shim_movie_clamp_26_24_eps0.02.gif.

## Robotic-arm build: manipulation scope and sim enablers (2026-07-16)

examples/franka_build.py replaces the invisible-hand falsework driver with a
menagerie Franka Panda that picks each cube from a staging row and builds the
clamp 29/24 falsework design end to end at 1/20 scale (cube side 0.05 m,
0.25 kg; keystone margins are scale-invariant, re-certified in the run to
float precision, and the propped prefixes certify identically at both scales).
The build succeeds: placements 0.2 to 2.7 mm, props strike at under 0.7 N,
the structure stands after retraction (rotation 0.017 rad). Scope limits:

- Single arm. The falsework route is fully single-arm (top-grasp drops plus
  prop retraction on scene actuators). The hold-and-shim 26/24 build needs one
  hand to hold the short reacher while another drives the shim; it is out of
  scope here.
- Scripted waypoints, no motion planning. Collisions are avoided by
  construction: every approach is a vertical descent over a column the
  certified drop order keeps clear. Fingertip-pad contact with non-carried
  cubes is monitored and is zero in the clean run; the tightest measured
  margins are the counterweight and bridge descents, whose pads pass 12.6 mm
  above the block below (the pads grip the cube's top half and protrude
  12.4 mm below its center). Only the fingertip pads carry contact pairs with
  the cubes; a collision by any other arm link would be unmodeled. The planar
  designs guarantee the y faces of every cube stay free, which is what the
  fixed y-pinch needs.
- No grasp planning. Fixed y-pinch at the cube center, top grasp only.
- Cube-pose feedback (a vision stand-in; simulator poses) is used three
  times: hover correction, final alignment, and seat detection before
  release. This is load-bearing. Open-loop placement through the same arm
  misses by 3 to 27 mm: in-grasp slip along the unconstrained pinch axis,
  arm-servo gravity sag, and finger-opening drag (1.7 mm) accumulate, and
  the 27 mm bridge miss lands the bridge entirely on the counterweight, so
  the clamp never forms and the reacher falls at retraction. That ladder,
  27 mm open loop versus 0.5 mm closed loop on a 50 mm cube, is the measured
  cost of open-loop manipulation on this design.
- Sim enablers, both in-memory (the menagerie file on disk is untouched).
  The gripper tendon servo is stiffened from 100 to 1000 N/m: the stock
  servo pinches a 0.05 m cube at 2.8 N, below the 2.45 N cube weight at
  mu 1.0, and the cube slips out during the first lift. The steady pinch
  becomes 23 N with transient peaks to 72 N, at the real hand's 70 N
  continuous spec boundary. Contacts run elliptic cones with impratio 10:
  MuJoCo's regularized pyramidal friction lets the pinched cube creep down
  the fingers at a measured 2.0 mm/s, which ate the descent clearance and
  rammed the carried counterweight into the base; elliptic plus impratio 10
  cuts the creep to 0.05 mm/s. Structural contacts share the option, so the
  settle physics differs slightly from the pyramidal-cone falsework runs.
- MuJoCo dynamics are not scale-invariant (contact time constants, servo
  bandwidths), so the executed build is evidence at the 0.05 m scale
  specifically; the statics are scale-free by the property tests.

Numbers, per-block table, and renders: out/mujoco/franka_build.json,
franka_build.gif, franka_build.mp4.

## Ride-under: the full-height reacher is its own insertion tool (2026-07-16)

examples/mujoco_rideunder.py inserts the FULL-HEIGHT reacher of the clamp class
with no props and no shim, replacing the two-handed hold-and-shim with a
one-handed capped push. The reacher enters nose-first with a slight nose-down
tilt; its leading top corner slips under the bridge lip, the bridge lifts, and
leveling the reacher under the bridge lays the bridge back down. Geometry of the
trick: a unit reacher tilted nose-down by theta, riding on its leading bottom
corner at the base top, carries its leading top corner S*(1 - cos theta) below
the lip, so any positive tilt clears it while a flat reacher sits exactly at the
lip (the zero-clearance jam of the earlier entries).

Scale, stated first. The maneuver runs at the Franka scale (cube side 0.05 m,
0.25 kg) so push forces are robot-sized. keystone statics are scale-free (the P4
margins below reproduce the unit-scale clamp), but the MuJoCo settle is not: the
fixed solref time constant is relatively softer at the smaller scale. The
pre-reacher stack (base, counterweight, bridge) certifies feasible (margin
2.9e-12) and stands prop-free at every contact stiffness. The seated clamps all
certify feasible but settle differently by scale: 26/24 stands at the Franka
scale (rot 3e-4) and at unit scale; 29/24 stands at unit scale (rot 0.006) yet
creep-topples at the Franka scale (rot 1.8); 31/24 topples at both. So the
standing threshold in this simulator at the Franka scale is 26/24, one grid step
below the falsework-standing 29/24 and five below the certified 31/24.

Tilt table (29/24, drive cap 4 reacher weights, drive contacts solref 0.003).
Every tilt slips the reacher under (reach error 0.2 to 0.5 mm), but tilt sets the
force. A flat push rams: the reacher-bridge lift force is 5.8 N (about 2.4 bridge
weights) and the ballast rotates 1.8 deg. Nose-down tilt turns the ram into a
wedge; at 4 deg the lift force drops to 1.4 N and the ballast rotates 0.09 deg;
8 deg is similar. 4 deg is the clean optimum.

    tilt 0 deg: reacher-bridge lift 5.8 N, ballast rotation 1.82 deg
    tilt 1 deg: lift 5.2 N, ballast rotation 1.77 deg
    tilt 2 deg: lift 4.9 N, ballast rotation 1.72 deg
    tilt 4 deg: lift 1.4 N, ballast rotation 0.09 deg
    tilt 8 deg: lift 1.5 N, ballast rotation 0.13 deg

Minimum push force and where it goes. At tilt 4 the peak driver push is 1.0 N
(0.4 reacher weights) and the maneuver completes at every force cap down to the
0.5-reacher-weight floor tested. Decomposition against the analytic edge-lift
force for the 0.25 kg bridge (torque balance about the far support edge,
0.5 * W_bridge = 1.23 N): the measured reacher-bridge vertical lift is 1.42 N
(1.16 times the reference), and the 1.0 N push splits into a 0.09 N edge-lift
horizontal share (F_v * tan 4 deg) and a 0.92 N friction-drag share (measured
reacher-on-base drag 1.30 N). The push is mostly friction drag; the edge lift is
cheap.

Bridge return and per-design outcomes (tilt 4, cap 4).
- 26/24: slips under, reach 0.44 mm, push 0.93 N, bridge returns to 1.6 mm,
  clamp overlap 10.4 mm, ballast rotation 0.2 deg, seated P4 margin 1.4e-11, and
  it STANDS (2 s and 6 s stiff settle). The clean full-height prop-free build.
- 29/24: slips under just as cleanly, reach 0.42 mm, push 1.0 N, bridge returns
  to 0.4 mm, overlap 5.3 mm, seated P4 margin 1.7e-11, but the seated knife-edge
  creep-topples at the Franka scale (the scale effect above). Certified feasible;
  stands at unit scale.
- 31/24: the reacher seats under the driver (0.55 mm) but slides out on release
  and comes to rest beside the bridge (overlap negative). The 1/24 thread is too
  shallow to clamp; the released state is a shorter, stable, non-clamp rest.

The bridge pivot is small, and that is a finding. A clean reseat lifts the bridge
under 1 deg: only the reacher's low leading region passes under the bridge, its
tall excess-span middle never reaches the lip, and progressive leveling lays the
bridge back before it can walk. A large pivot appears only when the reseat fails
(the fully tilted reacher, not leveled, drags the bridge off its seat). The
user-facing visual is the tilted reacher sliding under, not a bridge flap.

Intermediate arm-free certification (P4 margin versus push versus time, host
pipeline, oriented 2D boxes with the y tilt). Frozen at snapshots, the
reacher-held states read as external help needed: start and engagement are
infeasible with margin 0.158 (the dangling reacher needs the hold), the tilted
transient overlaps the bridge and trips the interpenetration guard (margin
undefined), and the seated held state is feasible (margin 1.4e-11 for 26/24,
1.7e-11 for 29/24). The help the state needs without the arm falls from 0.158 to
near zero as the reacher seats.

Phase 2, Franka execution: finger clearance passes, arm compliance is the
blocker. The menagerie Franka picks the reacher and pushes it under the bridge
(props retracted; the pre-stack stands prop-free). The clearance the maneuver
needs is geometric and wide: the reacher is gripped near its trailing (+x) end,
the bridge overlaps only the leading tail, so the pads sit 37.5 mm clear of the
bridge (26/24) and the trailing end cantilevers 33 mm past the base over empty
space. The sim confirms it: pad-bridge contact is 0.0 N through the whole run,
and the carry (lift straight up, translate high, descend on the open side) leaves
the bridge within 0.01 mm. But the push fails: the position-controlled arm,
rigidly gripping the reacher, drags the bridge off its seat as it levels (bridge
displaced 237 mm, ballast dropped 50 mm, structure collapses), where the Phase 1
force-capped impedance driver reseated it to 1.6 mm. The ride-under needs the
compliance of a capped push; stiff joint-position servos ram. Cartesian impedance
or force control at the arm is the missing piece, not finger room.

Verdict: ride-under makes the counterweighted clamp buildable prop-free and
shim-free with a single capped push, and buildable-and-standing at the Franka
scale at overhang 26/24 (n=4, dx=1/24), one grid step below the falsework 29/24
and five below the statically certified 31/24. The 29/24 optimum inserts just as
cleanly and is certified feasible, but its seated knife-edge is scale-marginal in
MuJoCo (it stands at unit scale, creep-topples at the Franka scale). A real
position-controlled arm clears the fingers by a wide margin yet rams the bridge:
the open problem is arm compliance, not clearance. Numbers:
out/mujoco/mujoco_rideunder.json; movie out/mujoco/rideunder_clamp_26_24.mp4 (and
.gif); Franka attempt out/mujoco/rideunder_franka_14.mp4.

## Ride-under phase 2 fix: side approach plus admittance stops the prying (2026-07-16)

The phase-2 blocker above (the stiff position-controlled arm drags the bridge
237 mm while leveling) is fixed by two changes in examples/mujoco_rideunder.py
phase2_execution, both required. First, a side approach: franka_scene.compose_scene
grows an arm_base_pos/arm_base_yaw kwarg (default None keeps the front build
bit-for-bit) and the arm is rebased beside the build plane in -y, yawed to face
the structure, so the ride-under push (along -x) is perpendicular to the approach
and no arm link arches over the structure. Second, the contact thread runs a
force-aware outer loop (admittance, PLAN.md option b): each control step it
measures the reacher-on-structure reaction (horizontal push drag plus vertical
bridge lift), advances the commanded thread only while both stay under the phase-1
caps (push 4 reacher-weights, a 2.5 N bridge-lift cap standing in for the leveling
torque), and recedes otherwise. Free-space moves (pick, carry) keep the stiff
servo for accuracy; only the thread is admittance.

Result on 26/24, pre-stack teleported to certified poses (phase-1 protocol, so
the push is compared like for like). Bridge disturbance drops from 237 mm to
3.0 mm return (3.2 mm peak during the push), a 74x cut; the ballast moves 0.5 mm
and rotates 0.37 deg (was 50 mm dropped and a collapse). The reacher-on-structure
push peaks at 5.3 N (2.2 reacher-weights, under the 9.8 N cap) and the bridge lift
at 2.7 N, both near the phase-1 driver profile (which used about 1.0 N because a
6-DOF compliant driver needs less). The reacher seats flat (tilt 0.02 deg) with an
11.8 mm clamp overlap. So the documented blocker is real and the admittance push
removes it: the arm no longer rams.

What the arm still cannot do, and why. The seated state is settle-marginal: the
stiff-contact settle flags it unstable, but on the displacement channel only
(rotation stays 0.009 rad, 0.5 deg, under the 0.01 threshold, so it does not
topple), because the pushed reacher lands about 4.7 mm off the ideal seat where
phase-1's exact placement lands 0. A two-finger gripper has no pitch authority
about the pinch axis (measured: commanding 4 deg nose-down, the grip holds -2 to
+1 deg, wandering 5 deg under MuJoCo's regularized pad contact), so the reacher
tilt is set by the base contact and the grasp height rather than the wrist, and
the deep thread that clamps best also leaves the largest residual offset. The
ride-under has no closed-loop seat correction (the block is pushed, not placed),
where franka_build's drops reach 0.5 mm with a vision-in-the-loop press; adding
that here is future work. The arm executes the push and seats the clamp with the
prying fixed, but does not reproduce phase-1's clean stand.

Two side findings, both recorded because they are negative results. The 180 deg
end base (arm beyond the overhang in +x, pushing -x away from itself) has the
best manipulability on paper, and Krishna proposed pushing the trailing face with
closed fingertips (no grasp during the push, which would remove the pitch and
finger-clearance questions). It does not work as a released push: a free reacher
tilted on its leading corner tips over the instant the grip opens (measured 110
deg flop, no continuous support), and with the reachable top-grasp orientation the
end base fouls the counterweight during the carry (the horizontal -x push
orientation the idea needs is unreachable at that low height, 48 mm IK residual;
only tool tilts up to 35 deg from vertical are reachable). The side base is the
one that executes. Second, the full one-handed build (three arm-drops then the
push) fails at the drops, not the push: the base drops to 4.6 mm but the
knife-edge counterweight cantilever (center of mass on the base edge, caught by
its falsework prop) lands 75 mm off and topples, because the ride-under drop
routine lacks the closed-loop alignment and press that franka_build spends its
BuildDriver on. The pre-stack therefore stands only when teleported to certified
poses; the arm-drop of the propped knife-edge is the remaining gap for a true
one-handed build. 29/24 rams under the arm (bridge 186 mm) exactly as its seated
knife-edge is scale-marginal in phase 1. Numbers and both-base comparison:
out/mujoco/mujoco_rideunder.json (phase2.executions); movie
out/mujoco/rideunder_franka_14.mp4.
