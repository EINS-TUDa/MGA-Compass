# Inner and outer approximations (ORACLE)

G-ORACLE builds on ORACLE (Turan, Moret and Bardow, 2026), which is so far the only MGA
method that guarantees convergence to the [near-optimal set](near-optimal-set.md)
$\mathcal{Z}$ within a chosen tolerance. This page explains ORACLE's idea; [](g-oracle.md)
explains what G-ORACLE changes.

## Two approximations that close in on each other

ORACLE maintains two polytopes with

$$
\mathcal{I} \subseteq \mathcal{Z} \subseteq \mathcal{O}.
$$

- The **inner approximation** $\mathcal{I}$ is the convex hull of the near-optimal points
  found so far, $\mathcal{I} = \operatorname{conv}(\mathcal{V}_I)$. Every point found is a
  real near-optimal design, so $\mathcal{I}$ only ever contains feasible designs.
- The **outer approximation** $\mathcal{O}$ is an intersection of halfspaces ("cuts"), each
  of which contains all of $\mathcal{Z}$. Everything outside $\mathcal{O}$ is provably not
  near-optimal.

Each iteration grows $\mathcal{I}$ by one point and shrinks $\mathcal{O}$ by a cut. The
method stops once the **directed Hausdorff distance** from $\mathcal{O}$ to $\mathcal{I}$,
that is, how far the point of $\mathcal{O}$ furthest from $\mathcal{I}$ is from it, falls
below a tolerance $\delta$. Since $\mathcal{Z}$ is sandwiched between the two, both are then
within $\delta$ of $\mathcal{Z}$: that is the convergence guarantee.

Distances are measured in a fixed norm $\|\cdot\|$. The distance of a point $z$ to the inner
approximation is

$$
f(z) = \min_{y \in \mathcal{I}} \|z - y\|,
$$

a convex projection problem whose dual vector $\pi$ (of the constraint $z - y = s$) G-ORACLE
reuses later.

## The four steps

1. **Initialization.** Maximize and minimize each PoI over $\mathcal{Z}$ (the direction
   problem with $w = \pm e_i$). The $2n_z$ resulting points and the cost-optimal point
   $z^* = Dx^*$ form the initial $\mathcal{V}_I$; the PoI ranges they reveal form the initial
   box $\mathcal{O}$.
2. **Trial point.** Find the point of $\mathcal{O}$ furthest from $\mathcal{I}$,
   $z_O \in \arg\max_{z \in \mathcal{O}} f(z)$, and its distance $d = f(z_O)$. Stop if
   $d \le \delta$.
3. **Nearest feasible point.** Solve the model once more, now for the near-optimal point
   $z_I = Dx_I$ *nearest* to $z_O$:

   $$
   \min_{x,\,r} \; \|r\| \quad \text{s.t.} \quad Ax \le b,\; c^\top x \le (1+\epsilon)\,\nu^*,\; z_O - Dx = r \quad (\sigma).
   $$

4. **Update.** Add $z_I$ to $\mathcal{V}_I$. If $z_I \ne z_O$, the trial point was not
   near-optimal, and the **projection cut**
   $\{ z \mid \sigma^\top z \le \sigma^\top z_I \}$, built from the dual $\sigma$ of the
   residual constraint, contains $\mathcal{Z}$ but cuts off $z_O$. Intersect $\mathcal{O}$
   with it.

### The cost cut

A second, optional cut uses the value function

$$
\nu(z) = \min_x \; c^\top x \quad \text{s.t.} \quad Ax \le b,\; z - Dx = 0 \quad (\mu),
$$

the cheapest system cost for given PoI values. $\nu$ is convex, so the dual $\mu$ at $z_I$ is
a subgradient and $\nu(z) \ge \nu(z_I) + \mu^\top (z - z_I)$ for all $z$. Every
$z \in \mathcal{Z}$ has $\nu(z) \le (1+\epsilon)\,\nu^*$, so the **cost cut**

$$
\{ z \mid \nu(z_I) + \mu^\top (z - z_I) \le (1+\epsilon)\,\nu^* \}
$$

also contains $\mathcal{Z}$. It is not needed for convergence, but it removes parts of
$\mathcal{O}$ that violate the cost bound and so speeds convergence up.

## ORACLE's bottleneck

Step 2 is a **bilevel problem**: the upper level maximizes $f(z)$ over $\mathcal{O}$, and
evaluating $f$ is itself a minimization. ORACLE rewrites it as a mixed-integer linear program
by replacing the lower level with its KKT conditions and linearizing them with big-M
constraints. This requires the $\ell_1$ or $\ell_\infty$ norm and adds binary variables for
every inner point, so the program grows with every iteration. On a PyPSA-Eur model with eight
PoIs, its solve time grew by about 12% per iteration and exceeded 8 hours at iteration 94,
far more than the model solves themselves. Removing this bottleneck is what
[G-ORACLE](g-oracle.md) does.

## In the code

| Step | Where |
|---|---|
| 1. Initialization | {py:func}`~mga_surveyor.explore.drive` calls `Method.bootstrap` with the box points and the optimum |
| 2. Trial point | `Method.find_trial_point`, implemented by each [method](methods.md) |
| 3. Nearest feasible point | {py:func}`~mga_surveyor.pypsa_ext.make_closest_point_fn` |
| 4. Update | `Method.add_point` and `Method.add_cut` |
| Cost cut | {py:func}`~mga_surveyor.common.linearized_objective_cut`, enabled by `use_objective_cuts=True` in {py:func}`~mga_surveyor.pypsa_ext.generate_mga_files` |

Two practical details differ from the idealized description:

- A cut is added only if $z_I$ is more than `sep_tol` away from $z_O$ (by default a fixed
  fraction of the tolerance), rather than whenever $z_I \ne z_O$; see
  {py:func}`~mga_surveyor.common.separates`.
- As in the paper's experiments, a cost cut is also added at the cost-optimal point $z^*$,
  but not at the box points. A box point's cost cut is parallel to its own box face, and
  nearly coincident halfspaces are a numerical risk for vertex enumeration.
