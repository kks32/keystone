# KNOWN_LIMITS

Recorded gaps and sharp edges. Each entry states the limit, the impact, and the
planned resolution. Dates are when the entry was recorded.

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
