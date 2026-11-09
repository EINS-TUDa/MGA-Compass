# Traverse

Once navigation has found a target solution, the traverse view shows how the system changes
along the path from the current solution to it: not just the PoIs, but the detailed design, and
an estimate of what the intermediate solutions cost.

## Interpolating between two solutions

Let $\alpha^{\mathrm{s}}$ and $\alpha^{\mathrm{t}}$ be the weights of the start and target
solutions. A slider $\beta \in [0, 1]$ moves along

$$
\alpha(\beta) = (1 - \beta)\,\alpha^{\mathrm{s}} + \beta\,\alpha^{\mathrm{t}},
\qquad
z(\beta) = S\,\alpha(\beta),
$$

and any detailed output of the model can be shown along the path by mixing the samples'
designs with the same weights, $x(\beta) = \sum_i \alpha_i(\beta)\, x^{(i)}$.

Showing a path through the *designs* is less obvious than it looks, because many designs map
to the same PoI values. Mixing the samples gives a consistent path through the designs that
the PoIs alone do not determine.

## Breakpoints: a better cost estimate

The mixed cost $v^\top \alpha(\beta)$ is an upper bound on the true cost $\nu(z(\beta))$, but
straight-line mixing is not the cheapest way to reach the intermediate points. The
interpolation refines it recursively:

1. At the midpoint of an interval, solve the [convex combination problem](preparation.md)
   for $z(\beta)$, which finds the cheapest mixture with the same PoIs.
2. If that mixture is cheaper than the straight-line one by more than a tolerance, keep it as
   a **breakpoint** and repeat on both halves.
3. Stop when no interval improves by more than the tolerance.

Because $\nu$ is convex, the resulting piecewise-linear cost curve is still an upper bound on
the true cost along the path, and a tighter one than the straight line. The result is an
approximately cost-optimal path from start to target.

## A lower bound on the cost

The duals collected in the preparation phase give a matching lower bound. Since $\nu$ is
convex and $\mu^{(i)}$ is a subgradient of $\nu$ at the sample $z^{(i)}$,

$$
\nu(z) \;\ge\; \nu^{(i)} + \left(\mu^{(i)}\right)^\top \left(z - z^{(i)}\right)
\quad \text{for all } z.
$$

Along the path, each such bound is a straight line in $\beta$. One more line comes for free:
no design is cheaper than the optimum, so $\nu(z) \ge \nu^*$ everywhere. The largest of these
lines is a piecewise-linear lower bound on the cost. Only the samples that appear with non-zero
weight at some breakpoint are used. If the problems are solved to extreme points, each
breakpoint has at most $n_z + 2$ of those, so the number of lines stays small.

Together, the two bounds bracket the true cost of the solutions along the path.

## In the code

- {py:func}`~mga_compass.core.interpolate` computes the breakpoints. Its `threshold` is
  relative to the cost at each midpoint (default `1e-3`, i.e. 0.1%) rather than an absolute
  amount.
- {py:func}`~mga_compass.near_optimal.plot_data` mixes a detailed output for a list of weight
  vectors and arranges it for one of the set's plots.
- {py:func}`~mga_compass.core.linear_lower_bounds` builds the lower-bound lines from the
  breakpoints and the duals, plus the constant line $\nu^*$ when given the optimal cost, and
  {py:func}`~mga_compass.core.upper_envelope` takes their maximum.
- From Python, {py:meth}`~mga_compass.space.Space.traverse` (also part of every
  {py:meth}`~mga_compass.space.Space.navigate` result) gives both bounds along the path as a
  {py:class}`~mga_compass.space.Traverse`.

```{admonition} To be implemented
:class: warning

The web platform does not yet show the lower bound: every near-optimal set carries its
duals, but the server does not pass them on yet.
```
