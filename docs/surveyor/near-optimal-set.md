# The near-optimal set

## Why look beyond the optimum

An energy system optimization model (ESOM) finds the cheapest design that meets its
constraints. But the model leaves out much of what decides whether a design is actually
built: political feasibility, social acceptance, local environmental impact. Its inputs are
uncertain too. A design that costs a few percent more than the optimum can therefore be the
better choice in practice.

*Modeling to Generate Alternatives* (MGA) makes this systematic: instead of a single optimum,
it describes the whole set of designs whose cost is within a small slack of the optimum, so
that decision-makers can weigh them against criteria the model does not capture.

## Formulation

Consider a linear ESOM

$$
\nu^* = \min_{x \in \mathbb{R}^{n_x}} \; c^\top x \quad \text{s.t.} \quad Ax \le b,
$$

where $x$ collects the planning and operational decisions, $c$ the costs and $A, b$ the
technical and economic constraints. Let $x^*$ be an optimal solution, with cost
$\nu^* = c^\top x^* > 0$.

For a cost slack $\epsilon \ge 0$, the near-optimal solutions form the polyhedron

$$
\mathcal{X}^\epsilon = \{ x \mid Ax \le b,\; c^\top x \le (1+\epsilon)\,\nu^* \}.
$$

A model has far more variables than a decision-maker cares about, so MGA works with a small
number $n_z$ of **properties of interest (PoIs)**, defined by a linear map $z = Dx$ with
$D \in \mathbb{R}^{n_z \times n_x}$ and $n_z \ll n_x$, for example the total installed
capacity of each technology. The **near-optimal set** is the image of
$\mathcal{X}^\epsilon$ in PoI space:

$$
\mathcal{Z} = \{ z \mid z = Dx,\; x \in \mathcal{X}^\epsilon \}.
$$

Because $\mathcal{X}^\epsilon$ is a polyhedron and $D$ is linear, $\mathcal{Z}$ is a
polyhedron too; we assume it is bounded, so it is a polytope.

## Exploring it one direction at a time

Computing $\mathcal{Z}$ exactly means projecting a polyhedron with millions of variables
onto a handful of dimensions, which is intractable for realistic models. MGA methods instead
probe it by solving the model with the cost bound added, maximizing the PoIs along a search
direction $w$:

$$
\max_{x} \; w^\top D x \quad \text{s.t.} \quad Ax \le b,\; c^\top x \le (1+\epsilon)\,\nu^*.
$$

Each solve returns the near-optimal point furthest in direction $w$, and each is as hard as
the original model, so the number of solves dominates the cost of MGA.

Existing methods differ in how they choose the directions, and most cannot say how much of
$\mathcal{Z}$ is still unexplored:

| Method | Choice of the next search | Stopping criterion |
|---|---|---|
| Heuristics (Variable Min/Max, Random Vector, Hop-Skip-Jump, Manhattan MGA) | Ad-hoc rules | An iteration budget set by the user |
| Modeling All Alternatives (MAA) | Facet normals of the hull of points found so far | Volume growth below a threshold, with no error bound; limited to about 10 PoIs |
| ORACLE | The point of an outer approximation furthest from the inner one | Distance between the approximations, which bounds the error |
| **G-ORACLE** (this package) | The same as ORACLE, found by vertex enumeration | The same guaranteed bound as ORACLE |

ORACLE and G-ORACLE are explained in [](oracle.md) and [](g-oracle.md).

## In the code

- The PoIs (the rows of $D$) are the `dimensions` argument of
  {py:func}`~mga_surveyor.pypsa_ext.generate_mga_files`, and the slacks $\epsilon$ are its
  `slack_levels`.
- {py:func}`~mga_surveyor.pypsa_ext.make_optimality_fn` solves the model for its optimum
  $x^*$ and $\nu^*$.
- {py:func}`~mga_surveyor.pypsa_ext.make_furthest_point_fn` solves the direction problem
  above.
