# Preparation phase

The preparation phase turns the full model into a compact description that the exploration
phase can work with in real time. It is computationally heavy, but runs only once per model.

## Sample points

Start from the [near-optimal set](../surveyor/near-optimal-set.md) $\mathcal{Z}$: the
property-of-interest (PoI) values $z = Dx$ of all designs within a cost slack $\epsilon$ of
the optimum. The preparation phase collects $n$ sample points
$z^{(1)}, \dots, z^{(n)} \in \mathcal{Z}$:

- **vertices** of $\mathcal{Z}$, found by an MGA method, and
- optionally **interior points**, for example the vertices found for a smaller slack, which
  improve the approximations used later.

The sample points are collected as the columns of a matrix
$S = [z^{(1)}, \dots, z^{(n)}] \in \mathbb{R}^{n_z \times n}$. In this repository the samples
come from [MGA Surveyor](../surveyor/index.md), whose inner approximation consists of exactly
such near-optimal points.

## The cheapest design behind each sample

Many designs map to the same PoI values, so each sample point has many designs behind it. Of
these, the cheapest is the interesting one. It is found by fixing the PoIs and minimizing
cost:

$$
\nu(z) = \min_x \; c^\top x \quad \text{s.t.} \quad Ax \le b,\; Dx = z \quad (\mu).
$$

For each sample $z^{(i)}$ this gives

- its cost $\nu^{(i)} = \nu(z^{(i)})$, collected in the vector $v = [\nu^{(1)}, \dots, \nu^{(n)}]$,
- the full design $x^{(i)}$, used later to show detailed results such as dispatch, and
- the duals $\mu^{(i)}$ of the fixing constraint, used later for a
  [cost lower bound](traverse.md#a-lower-bound-on-the-cost).

## Approximating the cost anywhere

With the samples and their costs, the cost of any point $z$ in their convex hull can be
approximated, without the full model, by the **convex combination problem**

$$
\operatorname{CCP}(z): \quad \min_{\alpha} \; v^\top \alpha
\quad \text{s.t.} \quad S\alpha = z,\; \mathbf{1}^\top \alpha = 1,\; \alpha \ge 0.
$$

The weights $\alpha$ describe $z$ as a mixture of samples. Because $\nu$ is convex, mixing
samples never underestimates the cost: $\nu(z) \le v^\top \alpha$ for any feasible $\alpha$.
CCP finds the mixture with the tightest such upper bound.

This is the key to real-time exploration: everything the exploration phase does is a small
linear program over the $n$ weights $\alpha$, whatever the size of the original model.

## In the code

- A **dataset bundle** holds what the preparation phase produces: the sample points with
  their costs as a `Points` table (one row per sample, one column per PoI, plus the cost
  column named by `obj_label`), the detailed outputs per sample, and a manifest; see
  {py:mod}`mga_compass.dataset` and {py:mod}`mga_compass.manifest`.
- {py:func}`~mga_compass.core.point_2_alpha` solves CCP: it fixes every PoI column and
  minimizes the cost column.
- {py:func}`~mga_compass.core.aggregate_by_alpha` mixes the detailed outputs with weights
  $\alpha$, giving $x = \sum_i \alpha_i x^{(i)}$.
- The scripts in `mga-compass/scripts` build a bundle from a PyPSA network using MGA
  Surveyor.
