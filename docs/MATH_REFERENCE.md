# Mathematical Reference

A masonry structure is a collection of rigid blocks held together by gravity
and friction alone; the joints carry no tension. Whether such a structure
stands is a question about forces: does a set of contact forces exist,
compressive and within the friction limits, that balances every block? This
document develops that question into the linear algebra keystone implements,
derives the solvers built on it, and states the proofs behind the archived
results.

The scope is what runs today: box assemblies in 2D and 3D, the exact 2D
friction cone, the inscribed 3D pyramid, the feasibility, load-factor,
critical-friction, and margin solvers with the checks behind each verdict,
the lateral-reserve screen, and the lattice search. General meshes, the exact
3D cone, and the non-associative iteration are planned and not covered.

## 1. Unknowns, sign convention, and the matrix A

The unknowns are the contact forces at interface vertices. Take one vertex $k$
on a patch that joins two blocks. Give the patch a right-handed frame
$(\hat n, \hat t_1, \hat t_2)$ with $\hat n$ the unit normal and
$\hat t_1, \hat t_2$ two unit tangents. The patch joins node $i$ and node $j$
with $i < j$, and $\hat n$ points from $i$ into $j$.

The force that block $j$ exerts on block $i$ at vertex $k$ is

$$F_k = -\,n_k\,\hat n + u_k\,\hat t_1 + v_k\,\hat t_2, \qquad n_k \ge 0. \tag{1}$$

Here $n_k$ is the normal force, $u_k$ and $v_k$ the two tangential components.
Compression is positive $n_k$: the two blocks push apart, so block $i$ feels
$-n_k \hat n$ (into $i$) and block $j$ feels the reaction $-F_k$. In 2D the
frame drops $\hat t_2$ and $v_k$, and the plane is $xz$ with gravity along
$-z$.

Stack every $(n_k, u_k, v_k)$ into one vector $f$. The equilibrium of the
assembly is a single linear map. For each block $b$ with center of mass $c_b$,
sum forces and moments about $c_b$:

$$\sum_{k \in \mathcal K(b)} \sigma_{bk}\,F_k + w_b = 0, \qquad
  \sum_{k \in \mathcal K(b)} (r_k - c_b)\times \sigma_{bk}\,F_k + \tau_b = 0.
  \tag{2}$$

$\mathcal K(b)$ is the set of vertices on patches touching block $b$. The sign
$\sigma_{bk}$ is $+1$ when $b$ is the $i$ side of that patch and $-1$ when $b$
is the $j$ side, so one block reads $F_k$ and its neighbor reads $-F_k$. The
position $r_k$ is the vertex location; $(w_b, \tau_b)$ is any external wrench.
Taking moments about each block's own center of mass makes the self-weight
torque vanish. Collecting the left sides into a matrix gives

$$A f + w = 0,$$

with three rows per block in 2D (order $[F_x, F_z, T_y]$) and six in 3D. Node 0
is the ground and contributes no rows. Masked (padded) vertices contribute zero
columns.

### A worked single block

Take a rectangular block of width $b$ and height $h$ in 2D, center of mass at
the origin, resting on the ground. The ground is node 0, the block is node 1,
so the block is the $j$ side and $\sigma = -1$. The one contact patch is the
bottom face at $z = -h/2$ with two vertices,

$$r_0 = (-b/2,\, -h/2), \qquad r_1 = (+b/2,\, -h/2).$$

The normal points from ground into block, $\hat n = +\hat z$, and the tangent
is $\hat t_1 = +\hat x$. The force on the block at vertex $k$ is the reaction
$-F_k = (-u_k,\; n_k)$ in $(x, z)$: a positive $n_k$ pushes the block up, as it
should.

Write $f = (n_0, u_0, n_1, u_1)$. The three rows of $A$ are the force-$x$,
force-$z$, and moment-about-$y$ balances. The moment of a planar force
$(F_x, F_z)$ applied at offset $(d_x, d_z) = r_k - c_b$ is
$T_y = d_z F_x - d_x F_z$. Substituting $-F_k$ and $r_k$:

$$A = \begin{bmatrix}
 0 & -1 & 0 & -1 \\[2pt]
 1 & 0 & 1 & 0 \\[2pt]
 \tfrac{b}{2} & \tfrac{h}{2} & -\tfrac{b}{2} & \tfrac{h}{2}
\end{bmatrix},
\qquad
w = \begin{bmatrix} 0 \\ -mg \\ 0 \end{bmatrix}.$$

$m$ is the block mass and $g$ the gravitational acceleration. Read the balances
off the rows. Row 1 gives $u_0 + u_1 = 0$. Row 2 gives $n_0 + n_1 = mg$: the two
normals carry the weight. Row 3 with row 1 gives $n_0 = n_1 = mg/2$: a symmetric
block loads its two corners equally.

Now add a horizontal live load $\lambda\,mg$ at the center of mass, the
pseudo-static lateral load used by P2. It enters on the force-$x$ row, so
$w_{\text{live}} = (mg, 0, 0)$ and the balances become

$$u_0 + u_1 = \lambda\,mg, \qquad n_0 + n_1 = mg, \qquad
  n_0 - n_1 = -\lambda\,mg\,\frac{h}{b}.$$

The windward normal $n_0 = \tfrac{mg}{2}\,(1 - \lambda h/b)$ falls to zero at
$\lambda = b/h$. That is the overturning load factor. The sliding limit comes
from friction (Section 2) at $\lambda = \mu$. This one matrix reproduces both
analytic gates.

### Nondimensionalization

Lengths divide by $L$, the bounding-box diagonal of the active vertices and
centers of mass. Forces divide by $W$, the total weight. Then
$\lVert w \rVert = O(1)$ and every margin below is dimensionless. Results are
reported back in SI.

## 2. Friction cones

Coulomb friction bounds the tangential force by $\mu$ times the normal force.
Per vertex the admissible set is the cone

$$K = \{\, (n, u, v) : n \ge 0,\ \sqrt{u^2 + v^2} \le \mu\,n \,\},$$

and the assembly cone is the product of these over all vertices. keystone
writes each cone as linear rows $G f \le 0$ and passes $G$ to the solvers.

### 2D is exact

In 2D the cone is the wedge $|u| \le \mu\,n$, which is already linear. Three
rows per vertex encode it:

$$-n \le 0, \qquad u - \mu\,n \le 0, \qquad -u - \mu\,n \le 0.$$

No approximation enters. The 2D verdicts are exact for the associative model of
Section 2.4.

### 3D uses an inscribed pyramid

The 3D cone $\sqrt{u^2+v^2} \le \mu n$ is not polyhedral. keystone inscribes a
$k$-sided pyramid. Its facets are

$$u\cos\theta_j + v\sin\theta_j \le \mu\,n\cos(\pi/k), \qquad
  \theta_j = \frac{(2j+1)\pi}{k}, \quad j = 0,\dots,k-1, \tag{3}$$

together with $n \ge 0$. Every point of the pyramid satisfies the true cone, so
the pyramid is conservative on force capacity. Along a polygon-vertex direction
the tangential capacity is exact; along a facet-mid direction it is
$\mu n \cos(\pi/k)$, short of the true $\mu n$. The shortfall is at most
$1 - \cos(\pi/k)$ of capacity, about $7.6\%$ at $k = 8$. $k$ is even and at
least 4 so the pyramid is symmetric under $t \to -t$ (isotropic friction); odd
$k$ is anisotropic and is gated behind an explicit flag.

### The dual cone

The verdict machinery needs the dual cone $K^* = \{ c : c^\top f \ge 0
\ \text{for all}\ f \in K \}$. For a polyhedral cone written $\{ f : G f \le
0 \}$, a vector lies in $K^*$ exactly when it is a nonnegative combination of
the rows of $G$: membership of $A^\top y$ in $K^*$ means there is $z \ge 0$ with
$A^\top y + G^\top z = 0$. For the 2D wedge this has the closed form: writing
$A^\top y$ per vertex as $(z_n, z_u)$, membership is $z_n \ge \mu\,|z_u|$.

### The associative lift and its consequence

The cone constraint bounds the force but says nothing about how a contact moves
when it yields. Limit analysis supplies the missing rule through duality. The
kinematic dual assigns each contact a relative velocity that is normal to the
active cone facet, the associated (normality) flow rule. For the Coulomb cone
the outward normal of the lateral facet has a tangential part of unit length
(sliding) and a normal part of size $\mu$ (opening). So associative sliding
comes with opening, or dilation, at rate $\mu$ times the slide.

True Coulomb friction slides without opening: its dilation angle is zero, which
makes it non-associative. Allowing the extra opening enlarges the set of
admissible collapse motions, which can only lower the collapse load. Therefore
the associative collapse load is an upper estimate of the true collapse load
(Drucker; Gilbert and Casapulla). Every solver in keystone is associative, and
every field carrying such a quantity says `assoc` in its name.

The estimate is one-sided, and the two verdicts are not symmetric. Let
$\lambda_{\text{assoc}}$ be the associative load factor and
$\lambda_{\text{true}}$ the true one, with
$\lambda_{\text{true}} \le \lambda_{\text{assoc}}$. If at a load the associative
model reports collapse (a falls-verdict), the true model, which has less
capacity, also collapses: a falls-verdict transfers to the true model. If the
associative model reports that the structure stands, the true model may still
collapse: a stands-verdict does not transfer. The 2D linear cone keeps this
ordering. The inscribed 3D pyramid combines the associative overestimate with
the cone underestimate of (3), so its load factor has no fixed ordering against
the true model; the code marks it `unknown`.

## 3. The four solvers

### The elastic margin (P4)

P4 measures how far an assembly is from equilibrium inside the cone. The JAX
backend solves the slack quadratic program

$$\min_{f,\,s}\ \tfrac12\lVert s\rVert^2 + \tfrac{\varepsilon}{2}\lVert f\rVert^2
  \quad \text{s.t.}\quad A f - s = -w,\ \ G f \le 0. \tag{4}$$

The variables are the force $f$ and a slack $s$ with one component per
equilibrium row. The slack absorbs the residual: at the optimum $s = A f + w$.
The objective drives $s$ toward zero while staying in the cone. The reported
margin is recomputed from the returned force by one matrix-vector product,

$$\text{margin} = \frac{\lVert A f + w\rVert}{\lVert w\rVert}, \tag{5}$$

not read from the internal slack. The quadratic weight matrix is block diagonal,
$\varepsilon I$ on $f$ and $I$ on $s$, which keeps the program well conditioned
and avoids squaring the condition number of $A$.

The term $\tfrac{\varepsilon}{2}\lVert f\rVert^2$ is a small Tikhonov bias that
makes the program strictly convex. It has a measured cost. At a feasible state
the optimum does not reach $s = 0$; it stops at a bias linear in $\varepsilon$
and proportional to the squared contact-force norm. On a 12-block 2D tower near
collapse the bias was $\text{margin} = 560\,\varepsilon$, stable across
$\varepsilon$ from $10^{-10}$ down to $10^{-14}$. The default
$\varepsilon = 10^{-12}$ puts the bias near $5.6\times 10^{-10}$, well below the
feasibility threshold $\text{tol\_feas} = 10^{-8}$. Infeasibility residuals are
independent of $\varepsilon$, so raising or lowering it does not move a
falls-verdict.

### The feasibility verdict (P0)

P0 asks the yes/no question: does an $f \in K$ with $A f + w = 0$ exist. The
answer is one of three values, and each is checked before it is returned rather
than read off a solver flag.

FEASIBLE requires a force state that passes two checks: the recomputed margin
(5) is at most $\text{tol\_eq}$, and the cone violation

$$\text{viol} = \max\big(0,\ \max(G f)\big)$$

is at most $\text{tol\_cone}$, with all quantities finite.

INFEASIBLE requires a checked collapse mechanism. The candidate is the
normalized residual direction $y = (A f + w)/\lVert A f + w\rVert$. It passes
when the load power $y^\top w$ exceeds $\text{tol\_power}$ and $A^\top y$ lies
in the dual cone, the latter tested by solving for $z \ge 0$ with
$A^\top y + G^\top z = 0$ (nonnegative least squares) and requiring the residual
below $\text{tol\_dual}$. This $y$ is a Farkas ray for the conic system (see
Section 4). Reshaped to one virtual twist per block it is the collapse
mechanism.

Anything that passes neither check is NO_CONVERGE. The solver abstains rather
than guess.

Alongside the verdict, keystone reports whether the P4 iterate is optimal for
(4), beyond being feasible. Optimality is a full Karush-Kuhn-Tucker check on
the slack program: four residuals, each an infinity norm, must sit below
$\text{tol\_gap}$.

$$
\begin{aligned}
r_{\text{stat}} &= \lVert Q x + q + A_{\text{eq}}^\top y_{\text{eq}}
   + G_{\text{ineq}}^\top z\rVert_\infty &&\text{(stationarity)}\\
r_{\text{eq}}   &= \lVert A_{\text{eq}} x - b_{\text{eq}}\rVert_\infty
   &&\text{(equality)}\\
r_{\text{ineq}} &= \max\big(0,\ \max(G_{\text{ineq}} x - h_{\text{ineq}})\big)
   &&\text{(inequality)}\\
r_{\text{comp}} &= \lVert s_{\text{ineq}} \odot z\rVert_\infty
   &&\text{(complementarity)}
\end{aligned}
$$

Here $x = [f, s]$ and $A_{\text{eq}}, G_{\text{ineq}}, Q, q$ are the program (4)
in standard form, $y_{\text{eq}}$ and $z$ the equality and inequality
multipliers, and $s_{\text{ineq}}$ the inequality slacks. These four mirror the
interior-point solver's own stopping test. A cone-admissible iterate always
upper-bounds the optimal margin through (5); this optimality flag says when it
also equals it.

The tolerances all derive from $\text{tol\_feas} = 10^{-8}$: $\text{tol\_eq}$
and $\text{tol\_cone}$ equal it, $\text{tol\_power}$ equals it,
$\text{tol\_dual} = 10\,\text{tol\_feas}$, and $\text{tol\_gap} =
100\,\text{tol\_feas}$. They live in one `Tolerances` dataclass and are passed
explicitly.

### The load factor (P2)

P2 finds the largest multiple of a live load the assembly carries:

$$\lambda^* = \max\{\lambda : A f + w_{\text{dead}} + \lambda\,w_{\text{live}} = 0,
  \ f \in K\}.$$

The default $w_{\text{live}}$ is horizontal pseudo-static gravity, so $\lambda^*$
is the tilt and pseudo-seismic margin. Feasibility is monotone in $\lambda$:
because $K$ is convex and the load is affine in $\lambda$, the feasible set is an
interval $[0, \lambda^*]$ with $\lambda = 0$ feasible. Bisection on the P0
verdict then converges linearly, and 60 steps resolve $\lambda^*$ to
$\lambda_{\text{hi}}\,2^{-60}$. A NO_CONVERGE midpoint does not read as
infeasible; it escalates to the exact LP oracle, and if that also abstains the
bisection stops and marks the band undecided. The reported factor is the
largest verified-feasible bracket bound. In 2D with the linear cone it is an
upper estimate of the true Coulomb capacity (Section 2.4).

### The critical friction (P3)

P3 finds the least friction that keeps the assembly standing:

$$\mu^* = \inf\{\mu : \text{P0 is feasible}\}.$$

Feasibility is monotone nondecreasing in $\mu$, because the cones nest:
$K_{\mu_1} \subseteq K_{\mu_2}$ for $\mu_1 \le \mu_2$, so a feasible force at
$\mu_1$ stays feasible at $\mu_2$. The feasible set is $[\mu^*, \infty)$ and
$\mu^*$ bisects. In 2D $\mu^*$ is a lower estimate of the true required
friction: the associative relaxation can stand at a friction below what
non-associative Coulomb needs.

## 4. Verified force states and checked mechanisms

The four solvers share one discipline: a verdict that reaches a caller has
passed a check on recomputed quantities, and a verdict that cannot be checked is
withheld. This section states the checks as equations and marks what is standard
and what is not.

A verified feasible force state is an $f$ with

$$\frac{\lVert A f + w\rVert}{\lVert w\rVert} \le \text{tol\_eq}, \qquad
  \max(0,\ \max(G f)) \le \text{tol\_cone}.$$

Both quantities are recomputed by matrix-vector product from the returned $f$,
so the check does not depend on the solver's internal state. A feasible force
state is a witness anyone can recheck. This is not new. A zero-tension
rigid-block-equilibrium (RBE) solution is the same kind of recheckable force
witness, and keystone claims nothing novel on the feasible side.

A checked collapse mechanism is the infeasible-side object. It is a vector $y$,
one virtual twist per block, with

$$y^\top w > \text{tol\_power}, \qquad
  A^\top y + G^\top z = 0 \ \text{for some}\ z \ge 0
  \ \text{to residual} \le \text{tol\_dual}. \tag{6}$$

The first inequality says the load does positive work on the motion $y$; the
second says $A^\top y$ lies in the dual cone. Together they are a Farkas
certificate: by Farkas' lemma exactly one of "a force $f \in K$ balances $-w$"
and "a $y$ satisfying (6) exists" holds, so a $y$ meeting (6) proves no
admissible force state exists and names the motion the assembly collapses
through. The infeasible side is where keystone reports something the RBE
lineage does not: those tools return a solver status, and keystone returns the
mechanism and rechecks it.

The three-valued discipline (FEASIBLE, INFEASIBLE, NO_CONVERGE) is verify or
abstain. It matters at scale. A single optimality proof by the branch-and-bound
of Section 7 consumes on the order of $10^5$ verdicts. If NO_CONVERGE were
folded into either decided value, a fraction of those verdicts would be wrong in
one direction, and the proof would inherit the error. Abstaining, escalating to
the exact oracle, and only then deciding keeps the proof sound.

## 5. Lateral reserve

Static feasibility at zero load does not say how close an assembly is to
collapse. Every feasible box design has a near-zero P4 margin at rest, so the
resting margin cannot separate a knife edge from a design with room to spare.
The separating quantity is the lateral reserve: the largest pseudo-static
lateral load, as a fraction of self-weight, the assembly carries while staying
verified feasible.

Solve P4 at $w_{\text{dead}} + \lambda\,w_{\text{live}}$ and at
$w_{\text{dead}} - \lambda\,w_{\text{live}}$, bisecting $\lambda$ in each
direction to the feasibility boundary. The symmetric reserve is

$$\rho = \min\big(|\lambda_+|,\ |\lambda_-|\big).$$

A state passes the reserve screen at threshold $\lambda_{\min}$ when its P4
margin verifies feasible under both $+\lambda_{\min}$ and $-\lambda_{\min}$, that
is when $\rho \ge \lambda_{\min}$.

The threshold is calibrated against MuJoCo compliant-physics settle outcomes on
a hand set of ten structures. The reserve separates the two classes cleanly:

    structure              reserve   physics
    corbel c=0.98          0.0050    toppled
    clamp 31/24            0.0069    toppled
    n6 17/12 pipeline      0.0069    collapsed
    n6 4/3                 0.0088    toppled
    clamp 5/4 (n4)         0.0139    toppled
    n6 back-1 5/4          0.0175    stood
    clamp 29/24            0.0208    stood
    clamp 26/24            0.0417    stood
    pair e=0.45            0.1000    stood
    tower 5-block          0.2000    stood

Every structure that toppled sits at or below $0.0139$, every one that stood at
or above $0.0175$. The default threshold
`keystone.search.lattice.LAM_MIN` $= 0.015$ sits in that gap and classifies all
ten correctly (10 of 10). The two 5/4 designs bracket the boundary from both
sides, so a run near it should be read as marginal. The screen predicts survival
at the recorded MuJoCo settings only; a different contact model or scale shifts
the boundary, so it is a screening heuristic, not a physical guarantee. Numbers:
`out/mujoco/mujoco_validate.json` and the pipeline stiff-contact driver runs.

Reserve costs overhang. On the branch-and-bound run at $n = 4$, grid step
$dx = 1/12$, friction $\mu = 0.7$, the price is

    reserve threshold      proved optimum
    none (static)          5/4  = 1.2500
    lam_min 0.01 - 0.05    7/6  = 1.1667
    lam_min 0.10           1    = 1.0000

Source: `examples/robust_sweep.py`, `out/robust/reserve_sweep_n4.json`.

## 6. The lattice environment

The search runs on a fixed 2D scene: a pedestal 6 wide and 1 tall with its right
edge at $x = 0$, and unit cubes stacked above it on a grid of step $dx$. A cube
at layer $L$ and grid index $j$ has center $x = j\,dx$ and center height
$z = 1.5 + L$.

State. A state is a set of placed cells $(L, j)$. Keying on the set, not on the
order it was built in, is what makes the search dedup sound (Section 7).

Actions. The action grid is all layers crossed with all positions,
$M = n_{\text{layers}}\, n_{\text{pos}}$ cells, enumerated in a fixed
layer-major order.

Legality. Placing a cube at $(L, j)$ on state $S$ is legal when every one of
these holds. The set has room, $|S| < n_{\max}$. The layer is in range and rests
on the frontier, $0 \le L \le n_{\max} - 1$ and $L \le 1 + \max_{(\ell,\cdot) \in
S} \ell$. There is support: for $L = 0$ the footprint $[x - \tfrac12, x +
\tfrac12]$ overlaps the pedestal top $[-6, 0]$ by at least $2\,dx$; for
$L \ge 1$ it overlaps some layer-$(L-1)$ cube by at least $2\,dx$. Same-layer
cubes clear each other, their centers more than one block width apart. The index
is in bounds. On the grid every overlap is a multiple of $dx$, so the code's
$> 1.5\,dx$ test is exactly the $\ge 2\,dx$ rule.

Reachability. Legality above is static; it checks the resting set, not the
motion that placed the block. Three optional modes add a motion predicate,
evaluated against the state before the placement, so the same final set can be
reachable in one build order and not another. Write $I(L,j)$ for the open
$x$-interval of the target cube and $I(L',j')$ for a placed cube.

- `drop` is legal when the column above the target is clear: no placed cube on a
  layer $L' > L$ has $I(L',j') \cap I(L,j) \ne \varnothing$ (open overlap;
  touching faces do not block).
- `slide` is legal when the drop column is clear, or a straight lateral corridor
  is clear at the target layer on either side: no placed layer-$L$ cube meets the
  interval the target sweeps out to that side. A layer-$(L+1)$ bridge clears the
  slide by construction, because layer heights are exact.
- `slide_clear` is `slide` with the under-bridge pass forbidden: a
  layer-$(L+1)$ cube over any part of the swept interval, target cell included,
  also blocks the corridor.

As action sets these nest: every drop-legal placement is slide_clear-legal and
every slide_clear-legal placement is slide-legal. The idealized `slide`
under-bridge pass is a zero-clearance press fit that rigid-body simulation shows
jamming, which is why `slide_clear` exists. The clearances are the exact
unit-cell gaps, motions are single-axis and straight, and no gripper or tool
volume is modeled; those limits are recorded in `docs/KNOWN_LIMITS.md`.

## 7. The optimality proof

For small $n$ the search proves the true grid optimum of maximum overhang, not
just a good stack. It runs best-first branch and bound over prefix-feasible
lattice states with an admissible upper bound.

Overhang is the rightmost cube edge measured from the pedestal right edge at
$x = 0$. A cube at grid index $j$ has right edge $j\,dx + \tfrac12$.

### The one-step lemma

A new cube rests on a support whose right edge is $S$: the pedestal top for
layer 0, a lower cube for layer $L \ge 1$. Legality requires the footprint
overlap with that support to reach $2\,dx$. Pushing the cube as far right as
that allows, its center $c$ satisfies $S - (c - \tfrac12) \ge 2\,dx$, so
$c \le S + \tfrac12 - 2\,dx$ and the new right edge obeys

$$c + \tfrac12 \le S + 1 - 2\,dx. \tag{7}$$

One placement extends the reach by at most $1 - 2\,dx$.

### The bound

Let $E$ be the largest right edge over the pedestal (edge 0) and every placed
cube, and let $r = n - |S|$ be the number of cubes still to place. Any support
has right edge at most $E$, so by (7) any next cube has right edge at most
$E + (1 - 2\,dx)$, and $r$ placements raise $E$ by at most $r\,(1 - 2\,dx)$. The
domain cap says no cube center exceeds $x_{\text{hi}}$. Hence

$$\text{bound}(S) = \min\!\big(E + r\,(1 - 2\,dx),\ x_{\text{hi}} + \tfrac12\big)
  \tag{8}$$

is an upper bound on the overhang of every legal completion of $S$, feasible or
not. It is admissible.

The bound reads geometry only. It never looks at density or friction, so it
stays admissible under heterogeneous materials: materials change which
completions are feasible, decided by the P0 verdict, never the reach ceiling.

### Monotonicity and optimality

Adding a cube lowers $r$ by one, which subtracts $1 - 2\,dx$ from the first term
of (8), and raises $E$ by at most $1 - 2\,dx$. The two moves cancel at best, so a
child bound never exceeds its parent bound. A monotone admissible bound makes
best-first branch and bound exact: the frontier is ordered by bound descending,
and the first time the frontier maximum bound drops to the incumbent overhang,
no unexpanded node can beat the incumbent, so the incumbent is the true optimum.
Bounds and overhangs all lie on the grid $\{k\,dx + \tfrac12\}$, so pruning
compares with a half-step margin $\tfrac12 dx$ that separates "no improvement
possible" from "at least one grid step better" exactly. No feasibility tolerance
enters the prune.

### Transposition soundness

A state is a set of cells, and the same set is reached by many placement orders.
Static feasibility of a set is a property of the assembly, not of the order that
built it. So the feasible completions of a set are the same whichever feasible
order first reached it. A closed list can drop repeat expansions of a set without
losing any reachable completion: one feasible arrival is enough to enumerate the
set's future. Heterogeneous materials keep this, because each cube's density and
friction is assigned by sorted-cell position of the set, not by build order, so
the assembly of a set is still a function of the set alone. The reachability
modes keep it too, because a drop or slide check reads the placed set and the
candidate cell, never the order.

The bound's admissibility survives the reachability modes: a mode only removes
actions, so every completion under a mode is also a static completion and the
bound never undercounts what a mode can reach. Feasibility for pruning uses the
verified qpax path at the full iteration cap, and the reported optimum's build
order is re-verified prefix by prefix through the host pipeline before it is
emitted. When a node or time budget stops the run early, the result is the
proved interval $[\text{incumbent},\ \text{best remaining bound}]$ and
optimality is not claimed.

## 8. Learning

A small network shapes where the tree search looks first. It never decides
feasibility; the verified qpax kernel does that. The network supplies a prior
over placements and a value for a partial stack.

### Feature map

A state maps to a vector of length $M + 3$. The first $M$ entries are the
occupancy grid: entry $L\,n_{\text{pos}} + (j - j_{\text{lo}})$ is 1 when cell
$(L, j)$ is filled. The three scalars carry the horizon $n$: the placed fraction
$\text{count}/n$, the current rightmost edge over two, and the remaining
fraction $(n - \text{count})/n$. Carrying $n$ in the features lets one network
read stacks of any size through the same input.

### Loss

Training minimizes a sum of three terms:

$$\mathcal L = \underbrace{-\textstyle\sum_a p_a \log \pi_a}_{\text{policy CE}}
  \; + \; \underbrace{(\hat v - v)^2}_{\text{value MSE}}
  \; + \; w_m\,\underbrace{(\hat m - m)^2}_{\text{margin MSE}}. \tag{9}$$

The policy cross-entropy is over legal actions only; illegal logits are masked
to $-\infty$ before the softmax. The value head is a sigmoid in $[0, 1]$. The
margin head is an auxiliary sigmoid, and its term is masked to rows that carry a
verified margin, so states the search never solved contribute nothing. The
weight $w_m$ defaults to $0.5$.

The margin target normalizes the verified P4 margin through a log. A margin
$m$ maps to

$$\hat m = \operatorname{clip}\!\left(
  \frac{\log_{10}(m + 10^{-12}) - (-12)}{0 - (-12)},\ 0,\ 1\right).$$

A near-zero margin (near equilibrium) maps near 0; a margin near 1 (clearly
infeasible) maps near 1. The window $[-12, 0]$ in $\log_{10}$ brackets the
observed range with headroom.

### Value target

The value target is the best overhang reachable from a state, normalized by
$2\,H_n$ where $H_n = \sum_{k=1}^{n} \tfrac{1}{2k}$ is the simple-stack overhang,
then clipped to $[0, 1]$. For an imitation prefix along a build order it is the
suffix-max overhang of the order; for a self-play node it is the subtree-max over
the node and its descendants. Each state is credited with what it can reach, not
with one constant episode return shared by the whole trajectory. A reserve
variant subtracts a small penalty from the value of any node whose best path runs
through a knife-edge state.

### What was measured

The material sweep at $n = 4$, $dx = 1/12$ records how the proved optimum
moves with friction and with a fixed mass budget spent on different cubes
(`out/search/material/summary.txt`):

    friction mu     proved optimum
    0.30 - 0.50     1.1667
    0.60 - 1.00     1.2500

    inventory (mu=0.7)     proved optimum
    uniform                1.2500
    heavy_low_4_4_1_1      1.3333
    heavy_high_1_1_4_4     1.0000

Putting the heavy cubes low lifts the optimum to 4/3; putting them high drops it
to 1. The reserve price table of Section 5 comes from the companion reserve
sweep.

## 9. Results registry

Each claim above is re-executable. The numbers are pinned by tests, which are
the proofs of record; do not quote a number from memory.

- Sign and frame conventions, the matrix $A$: `tests/unit/test_assemble.py`,
  `tests/unit/test_geometry.py`.
- The verdict checks and the mechanism (feasible witness, checked Farkas
  mechanism, the three-valued rule): `tests/unit/test_certificates.py`,
  `tests/unit/test_solve_synthetic.py`.
- 2D analytic gates (single-block $\lambda = b/h$ and $\lambda = \mu$, the
  verdict switch at $\mu = b/h$, the offset pair boundary $e = b/2$, the
  harmonic corbel, sliding versus toppling mechanisms):
  `tests/analytic/test_gates_2d.py`.
- 3D tilt gates, the pyramid facet-mid conservatism, the 2D-to-3D extrusion
  agreement, and oracle agreement: `tests/analytic/test_gates_3d.py`.
- Branch-and-bound optima and their prefix feasibility, the $n = 3$ value $5/6$,
  the drop and slide_clear price at $n = 4$: `tests/analytic/test_bnb_optima.py`.
- The $n = 6$ overhang $5/4$ regression and the greedy-order counterexample:
  `tests/analytic/test_overhang_regression.py`.
- Rigid-transform, scaling, and reordering invariances:
  `tests/property/test_invariants.py`.
- The lattice environment, its legality and reachability modes, and the reserve
  kernel: `tests/unit/test_search_lattice.py`,
  `tests/unit/test_search_lattice3d.py`.
- The learning feature map, loss, and targets: `tests/unit/test_az.py`.

## References

Heyman 1969 and Ochsendorf 2002 for the masonry arch limit state. Livesley 1978
for the linear-programming formulation of rigid-block limit analysis. Gilbert
and Casapulla, and Portioli and coworkers, for the associative relaxation and
the non-associative iteration. Drucker for the associative upper bound.
Pellegrino and Calladine 1986 for the four-subspace framing of $A$. keystone
claims no novelty in the equilibrium formulation itself.
