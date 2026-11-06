# G-ORACLE

G-ORACLE keeps ORACLE's [inner and outer approximations](oracle.md) and its convergence
guarantee, and replaces only Step 2, the bilevel search for the trial point. It rests on one
observation: **the trial point is always a vertex of the outer approximation.**

## The trial point is a vertex

The distance $f(z) = \min_{y \in \mathcal{I}} \|z - y\|$ to a convex set is a convex function
of $z$, for any norm. A convex function on a polytope attains its maximum at a vertex: every
point of $\mathcal{O}$ is a convex combination of its vertices, and by Jensen's inequality
$f$ there is at most the largest value of $f$ at those vertices. Hence

$$
\max_{z \in \mathcal{O}} f(z) = \max_{v \in \operatorname{vert}(\mathcal{O})} f(v).
$$

So instead of solving a bilevel problem, G-ORACLE

1. keeps the vertex set $\mathcal{V}_O$ of $\mathcal{O}$ up to date,
2. evaluates $f(v)$ at each vertex, an independent single-level convex projection (a linear
   program for $\ell_1$ and $\ell_\infty$, a quadratic program for $\ell_2$), and
3. takes the vertex with the largest distance as the trial point.

Steps 1, 3 and 4 of ORACLE stay as they are, so the convergence guarantee carries over
unchanged. Unlike ORACLE's reformulation, this works for the $\ell_2$ norm too.

## Keeping the vertex set up to date

$\mathcal{O}$ only ever changes by adding a cut, so its vertices can be updated incrementally
rather than recomputed. There are two classical ways to do this.

**Double description.** When a halfspace is added, split the current vertices into those
inside it, on its boundary and outside it. Drop the outside ones, and create a new vertex
wherever the boundary crosses an edge joining an inside vertex to an outside one.

**Beneath-and-beyond on the dual polytope.** Shift $\mathcal{O}$ so that an interior point is
at the origin and take its *polar dual* $\mathcal{O}^*$. Each facet of $\mathcal{O}$ becomes
a vertex of $\mathcal{O}^*$, and each vertex of $\mathcal{O}$ becomes a facet of
$\mathcal{O}^*$. Adding a cut to $\mathcal{O}$ is then the same as adding one point to
$\mathcal{O}^*$, which an incremental convex-hull method such as beneath-and-beyond handles
directly: the facets it removes are exactly the vertices the cut removes, and the facets it
creates are the new vertices. A cut that leaves no facet visible is redundant.

In theory the two are the same algorithm seen from opposite sides. In practice they behave
differently; see [](methods.md).

The cost of each update, and the number of projections per iteration, grows with the number
of vertices of $\mathcal{O}$. In the worst case that is $O(m^{\lfloor n_z/2 \rfloor})$ for
$m$ cuts (the upper bound theorem), but in practice it stays far below: in the PyPSA-Eur
experiment with eight PoIs and 509 halfspaces, $\mathcal{O}$ had about 139,000 vertices,
against a worst-case bound of about $2.7 \times 10^9$.

## Reusing distances: the supporting-hyperplane check

Most vertices survive from one iteration to the next, and $\mathcal{I}$ grows by only one
point at a time, so recomputing $f(v)$ for every vertex in every iteration is mostly wasted
work. The projection's dual vector $\pi$ tells cheaply when a cached distance is still valid.

If $y^*$ is the nearest point of $\mathcal{I}$ to $z$ and $\pi$ the matching optimal dual
vector, then the hyperplane $\{ y \mid \pi^\top y = \pi^\top y^* \}$ supports $\mathcal{I}$
at $y^*$: all of $\mathcal{I}$ lies on the side $\pi^\top y \le \pi^\top y^*$. When a new
inner point $z_I$ is added,

$$
\pi^\top z_I \le \pi^\top y^*
$$

is enough to guarantee that $y^*$, $\pi$ and the distance $f(z)$ all remain optimal for the
enlarged inner approximation. The check costs one inner product per vertex instead of a
re-solve. It is a sufficient condition, so it never accepts a stale distance. For norms that
are differentiable away from the origin, such as $\ell_2$, it is also necessary. For $\ell_1$
and $\ell_\infty$ it can fail even though the distance did not change, which costs an
unnecessary re-solve but never a wrong result.

In the PyPSA-Eur experiment, 98.7% of the checks on surviving vertices passed, and only 0.06%
of the re-solves triggered by a failed check turned out to be unnecessary.

## The algorithm

Each iteration of G-ORACLE:

1. Update $\mathcal{V}_O$ with the cut added in the previous iteration.
2. Solve the projection for every **new** vertex, and cache its distance, $y^*$ and $\pi$.
3. For every **surviving** vertex, apply the supporting-hyperplane check with the inner point
   added in the previous iteration, and re-solve only if it fails.
4. Take the vertex with the largest distance as the trial point $z_O$, with $d = f(z_O)$.
   Stop if $d \le \delta$.
5. Continue with ORACLE's Steps 3 and 4: find the nearest near-optimal point $z_I$, add it to
   $\mathcal{I}$, and add the cut(s) to $\mathcal{O}$.

## Results in brief

On an electricity-only PyPSA-Eur model (34 nodes, 3-hourly resolution, about 2.7 million
variables) with eight capacity PoIs and a 5% cost slack, G-ORACLE with beneath-and-beyond
reached a tolerance of 4% of each PoI's range in 247 iterations. Its trial-point search took
at most 7 seconds per iteration and 333 seconds in total. The two model solves per iteration
took a median of 232 and 159 seconds, so the search no longer matters. Wherever both were
computed, ORACLE and G-ORACLE returned the same distance, as the vertex theorem predicts.

For the full experiments and proofs, see the paper
([arXiv:2610.32759](https://arxiv.org/abs/2610.32759)).

## In the code

- {py:class}`~mga_surveyor.methods.bb.BBMethod` and
  {py:class}`~mga_surveyor.methods.cdd.CddMethod` implement this vertex-based trial-point
  search, with beneath-and-beyond and double description respectively.
- The supporting-hyperplane check runs whenever an inner point is added (`add_point`): only
  the vertices whose cached normal fails the check are re-projected.
- The projection $f(v)$ and its dual are computed by
  {py:func}`~mga_surveyor.gurobi_projection.closest_point_on_hull` or
  {py:func}`~mga_surveyor.highs_projection.closest_point_on_hull`.
