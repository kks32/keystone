# CHANGELOG

## 0.0.1 (2026-07-15, unreleased) - certified verdicts and bound semantics

Solver-layer review fixes. Verdicts are now verified against recomputed
residuals and validated certificates, never read off a solver flag.

- Verified verdicts. `margin_core` recomputes the equilibrium residual
  r = A f + w by matvec and reports margin = ||r|| / ||w||, the cone
  violation max(0, max(G f)), and the primal-dual gap, alongside the raw
  qpax flags. Its return grew to (margin, f, r, viol, gap, converged,
  iters). FEASIBLE now requires a primal certificate (margin <= tol_eq and
  viol <= tol_cone, all finite); INFEASIBLE requires a validated Farkas
  ray (load power > tol_power and A^T y in the dual cone, found as z >= 0
  with A^T y + G^T z = 0 to residual <= tol_dual); everything else is
  NO_CONVERGE. Band-edge verdicts that the old margin-primary rule called
  infeasible can now report NO_CONVERGE and escalate.
- Tri-state bisection. P2 and P3 escalate a NO_CONVERGE midpoint to the
  exact LP oracle rather than treating it as infeasible; if that also
  fails they stop and record info["uncertified_band"]. Iteration counts
  stay deterministic. P2 and P3 now attach the certified-feasible-side
  force state to Result.forces.
- Bound semantics. Field `lambda_upper_assoc` renamed to `lambda_assoc`
  (pre-release, no alias). New structured Result fields cone_model,
  constitutive_model ("associative"), and physical_bound_direction
  ("upper" for a linear 2D associative load factor, "unknown" for an
  inscribed pyramid, "lower" for mu_critical_assoc). info["bound"] carries
  a machine-readable tag. MATH_REFERENCE.md Section 4 Farkas sign
  corrected (A^T y in the dual cone K*, nonnegative pairing) and a
  bound-directions paragraph added.
- Exact oracle. Infeasible P0/P2 return a validated Farkas ray from an
  explicit normalized certificate LP (max y.w s.t. A^T y + G^T z = 0,
  z >= 0, -1 <= y <= 1), numerically validated before return. HiGHS
  feasibility options wired from tol.tol_feas. A lambda saturating lam_hi
  stays FEASIBLE and sets info["censored"] = True.
- Tolerances gained five certification fields: tol_eq, tol_cone (default
  tol_feas), tol_dual (10 * tol_feas), tol_power (tol_feas), tol_gap
  (100 * tol_feas). Unset fields derive from tol_feas in __post_init__,
  which now rejects any non-positive or non-finite tolerance. Solver knobs
  (solver_tol, max_iter) moved to the new keystone.solve.SolverOptions
  dataclass; they are not tolerances.
- Input validation. cone_matrix_pyramid requires even k >= 4 (isotropic);
  odd k >= 3 is gated behind allow_asymmetric=True (experimental
  anisotropic). Both cone builders reject non-finite or negative mu.
  assemble rejects dim outside (2, 3).
- margin_batch keeps its two-value return; the second value is now
  `certified` (per element: cone-admissible and finite) and margins are
  the recomputed residual margins. Added solve_p4_batch for a Result-typed
  batched P4.

## 0.0.1 (2026-07-15, unreleased) - tolerance change

- `Tolerances.eps_reg` default lowered from 1e-10 to 1e-12. The P4 feasible-side
  margin equals the Tikhonov bias, which is linear in eps_reg and proportional to
  the squared contact force norm. At 1e-10 the bias crossed tol_feas (1e-8) for
  2D towers of 8 or more blocks and made P2 bisection certify feasible states as
  infeasible (N=12 tower: lambda 0.0138 instead of the analytic 0.0833). At 1e-12
  the bias sits near 5.6e-10 on the worst measured case while infeasibility
  residuals are eps-independent (verified 1e-10 through 1e-14). Benchmark suite
  rerun after the change per charter Section 5; results in bench/RESULTS.md.

## 0.0.1 (2026-07-15, unreleased)

First working cube slice. 2D and 3D box stacks end to end in JAX.

- Tolerances dataclass established (charter Section 5) with two additions beyond
  the charter's geometric four: `eps_reg` (1e-10, P4 Tikhonov term on the force
  block) and `tol_feas` (1e-8, feasibility margin threshold). Both dimensionless,
  both new in this release, benchmark baseline recorded in bench/RESULTS.md.
- Sign and frame conventions of CLAUDE.md Sections 3 and 4 implemented and pinned
  by tests (tests/unit/test_assemble.py, tests/unit/test_geometry.py).
- Geometry: box-box interface detection, 2D segments and 3D clipped polygons
  (Sutherland-Hodgman, up to 8 vertices), ground plane as node 0.
- Mechanics: dense padded equilibrium map, nondimensionalized; linear 2D cones and
  inscribed pyramid cones (explicit at the call site, no fallback).
- Solve: P4 slack QP via qpax (jit/vmap), P0 with least-squares Farkas mechanism,
  P2 and P3 by bisection, batched margin entry point, scipy HiGHS oracles.
- Viz: 2D and 3D matplotlib renderers with force and mechanism overlays.
- Docs: MATH_REFERENCE.md (cube-slice draft), KNOWN_LIMITS.md.
