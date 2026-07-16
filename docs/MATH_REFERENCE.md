# MATH_REFERENCE

Derivations behind the keystone solvers. LaTeX is allowed in this file only.
Status: cube-slice draft (C0). The general mesh pipeline sections arrive with M2.

## 1. Unknowns and sign convention

For vertex $k$ on a patch joining nodes $(i, j)$, $i < j$, with unit frame
$(\hat n, \hat t_1, \hat t_2)$, $\hat n$ pointing from $i$ into $j$, the force
exerted by block $j$ on block $i$ is

$$F_k = -n_k \hat n + u_k \hat t_1 + v_k \hat t_2, \qquad n_k \ge 0 .$$

Compression is positive $n_k$: the blocks push each other apart, so block $i$
receives $-n_k \hat n$ (into $i$) and block $j$ receives $+n_k \hat n$. The
reaction on $j$ is $-F_k$. 2D drops $v$.

## 2. The equilibrium map A

Stack all vertex components into $f$. For each block $b$ with center of mass
$c_b$, force balance and moment balance about $c_b$:

$$\sum_{k \in \mathcal{K}(b)} \sigma_{bk} F_k + w_b = 0, \qquad
  \sum_{k \in \mathcal{K}(b)} (r_k - c_b) \times \sigma_{bk} F_k + \tau_b = 0,$$

where $\sigma_{bk} = +1$ if $b$ is the $i$ side of the patch of vertex $k$ and
$-1$ if it is the $j$ side, $r_k$ is the vertex position, and $(w_b, \tau_b)$
external wrench. Moments about each block's own com make self-weight torque
vanish. Rows per block: $[F_x, F_z, T_y]$ in 2D, all six in 3D. Ground
(node 0) contributes no rows. Columns of masked vertices are zero.

Nondimensionalization: lengths divide by $L$ (bounding-box diagonal), forces
by $W$ (total weight). Then $\lVert w_{dead} \rVert = O(1)$ and margins are
dimensionless.

## 3. Friction cones

Exact Coulomb per vertex: $K = \{ (n, u, v) : n \ge 0,\ \sqrt{u^2+v^2} \le \mu n \}$.
2D is exactly linear: $|u| \le \mu n$.

Inscribed pyramid (3D, k facets): polygon vertices at angles $2\pi j / k$ in
the $(u, v)$ plane at radius $\mu n$, facet rows

$$u \cos\theta_j + v \sin\theta_j \le \mu n \cos(\pi/k), \qquad
  \theta_j = (2j+1)\pi/k .$$

Every pyramid point satisfies the exact cone (conservative). Tangential
capacity along a polygon vertex direction is exact; along a facet-mid
direction it is $\mu n \cos(\pi/k)$, an underestimate of at most
$1 - \cos(\pi/k)$ of capacity ($\approx 7.6\%$ at $k = 8$). Standard
pyramids use even $k \ge 4$ so the cone is symmetric under $t \to -t$
(isotropic friction); odd $k$ is anisotropic and gated behind an explicit
flag.
Associativity: all cone relaxations here are associative and therefore upper
estimates of true Coulomb capacity (see Section 6). The inscribed pyramid
adds a conservative cone underestimate on top of the associative
overestimate, so the pyramid load factor is **not ordered** against true
Coulomb capacity (Section 4 bound directions).

## 4. Problems

- P0 feasibility: $\exists f \in K$ with $A f + w = 0$. Farkas alternative
  for the conic system: infeasible iff $\exists y$ with $y^\top w > 0$ and
  $A^\top y \in K^*$, the dual cone $K^* = \{ c : c^\top f \ge 0\ \forall f
  \in K \}$ (nonnegative pairing). Proof of the direction: if $A f = -w$,
  $f \in K$, then $y^\top w = -y^\top A f = -(A^\top y)^\top f \le 0$
  whenever $A^\top y \in K^*$; so $A^\top y \in K^*$ with $y^\top w > 0$
  forbids any feasible $f$. For the polyhedral cone $K = \{f : G f \le 0\}$,
  $A^\top y \in K^*$ iff $\exists z \ge 0$ with $A^\top y + G^\top z = 0$
  (Farkas), which is the certificate keystone validates. $y$ is a virtual
  twist per block: the mechanism.
- P4 elastic margin, slack form used by the JAX backend:

$$\min_{f \in K, s}\ \tfrac12 \lVert s \rVert^2 + \tfrac{\epsilon}{2}\lVert f \rVert^2
  \quad \text{s.t.} \quad A f - s = -w .$$

  Reported margin $= \lVert s^* \rVert / \lVert w \rVert$. With $\epsilon = 0$
  this is the distance from $-w$ to the cone $A K$; feasibility iff margin 0.
  $\epsilon > 0$ (Tolerances.eps_reg) makes the QP strictly convex; the
  perturbation of the margin is $O(\epsilon \lVert f^* \rVert^2 / \lVert s^* \rVert)$
  and is far below tol_feas for the defaults. At an infeasible point the
  residual direction $y = s^*/\lVert s^*\rVert$ is the least-squares Farkas
  certificate. At $\epsilon = 0$ the projection optimality gives
  $A^\top s^* \in K^*$ (so $A^\top y \in K^*$, nonnegative pairing with the
  cone) and $y^\top w = \lVert s^* \rVert > 0$. For $\epsilon > 0$ the dual
  membership holds up to $O(\epsilon \lVert f^* \rVert)$, which the
  certificate check absorbs through tol_dual.
- P2 load factor: $\lambda^* = \max\{ \lambda : A f + w_{dead} + \lambda w_{live} = 0,
  f \in K \}$. The feasible $\lambda$ set is an interval $[0, \lambda^*]$
  (convexity of $K$ and linearity in $\lambda$; $\lambda = 0$ assumed
  feasible), so bisection with the P4 feasibility test converges linearly and
  60 steps resolve $\lambda^*$ to $\lambda_{hi} 2^{-60}$. The LP dual at
  optimum is the collapse mechanism; complementary slackness is checked
  numerically.
- P3 critical friction: feasibility is monotone nondecreasing in $\mu$
  (cones nest: $K_{\mu_1} \subseteq K_{\mu_2}$ for $\mu_1 \le \mu_2$), so
  $\mu^* = \inf\{\mu : \text{P0 feasible}\}$ bisects.
- P5 non-associative bracket: Gilbert-Casapulla fixed point. Solve
  associative, freeze $n$, box-bound $|u|, |v| \le \mu n_{frozen}$, resolve,
  iterate (cap 20). Report $(\lambda_{assoc}, \lambda_{nonassoc})$ with
  $\lambda_{nonassoc} \le \lambda_{assoc}$ asserted.

Bound directions (reported as structured fields on `Result`). Every solver
is associative. `physical_bound_direction` is populated only when a capacity
factor is reported: P2 sets it for `lambda_assoc`, P3 for
`mu_critical_assoc`. P0 and P4 report no capacity factor and set it `None`.
The reported factor relates to true Coulomb capacity as:

- P2 `linear2d` (2D, exact associative), uncensored: `lambda_assoc` is an
  **upper** estimate of true Coulomb capacity. Associative capacity bounds
  true capacity from above (Section 6). `physical_bound_direction = "upper"`.
- P2 `pyramid` (inscribed $k$-gon): `lambda_assoc` is a **lower** bound of
  the exact-associative capacity (the cone is inscribed, hence conservative)
  but has **no ordering** against true Coulomb capacity: an associative
  overestimate combines with a cone underestimate and the net sign is not
  fixed. `physical_bound_direction = "unknown"`.
- P2 **censored** (any cone): when $\lambda$ saturates $\lambda_{hi}$ while
  still certified feasible, or when the hi endpoint is uncertified and the
  reported factor is the last certified-feasible $\lambda$, the value is
  only a **lower** bound on associative capacity. It is unordered vs true
  even in 2D. `physical_bound_direction = "unknown"`.
- P3 `linear2d` (`mu_critical_assoc`): a **lower** estimate of the true
  required friction. The associative relaxation can certify feasibility at a
  $\mu$ below what non-associative Coulomb friction needs, so the true
  critical friction can be higher. `physical_bound_direction = "lower"`.
- P3 `pyramid`: an inscribed pyramid needs **more** friction than exact
  associative, and true Coulomb also needs **more** than exact associative,
  so the two effects are unordered and the reported critical friction has no
  fixed sign vs true. `physical_bound_direction = "unknown"`.

## 5. Analytic gates (2D, unit depth, gravity g)

- Single block, width $b$, height $h$, horizontal live load $\lambda W$:
  overturning about the toe: $\lambda W h/2 = W b/2 \Rightarrow \lambda = b/h$.
  Sliding: $\lambda = \mu$. $\lambda^* = \min(b/h, \mu)$; verdict switch at
  $\mu = b/h$.
- Two stacked blocks, top offset $e$: feasible iff the top block's weight
  line passes through the contact patch: $e \le b/2$.
- Harmonic corbel over a finite support edge, $m$ blocks with consecutive
  shifts $b/(2k)$, $k = 1..m$ from the top: each substack's com sits exactly
  on the supporting edge below; total overhang $\sum_{k=1}^{m} b/(2k)$; for
  $m = 4$: $\tfrac{b}{2}(1 + \tfrac12 + \tfrac13 + \tfrac14) = \tfrac{25}{24} b$.
  Feasible for scale $c < 1$ of the shifts, infeasible for $c > 1$.
- Cuboid on a plane tilted by $\theta$ (3D tilt gate): topple at
  $\tan\theta = b/h$, slide at $\tan\theta = \mu$; equivalent to the P2 gate
  by rotating gravity.

## 6. Associative overestimate

The LP/SOCP relaxation takes the friction constraint as a set constraint on
forces with no flow rule, which is equivalent to associative (dilatant) flow
in the kinematic dual. Associative capacity bounds true Coulomb capacity from
above (Drucker; Gilbert-Casapulla). The API therefore labels these results
`*_assoc` upper estimates and pairs them with the P5 lower estimate. No
keystone function reports the associative verdict as exact.

## 7. Duality and mechanism structure

The four fundamental subspaces of $A$ organize everything: self-stress states
(force indeterminacy) live in $\mathrm{null}(A)$; rigid-body compatible
virtual velocities live in $\mathrm{null}(A^\top)$ restricted by contact
admissibility. The P2 LP dual variables are block twists $y$; complementary
slackness pairs positive contact forces with zero relative velocity and
opening/sliding contacts with zero force. See Mitchell, Baker, McRobie,
Mazurek 2016 (graphic statics part I) and Pellegrino, Calladine 1986 for the
subspace framing.
