# Navigation

Navigation answers the question: *starting from the solution I am looking at, which
near-optimal solution best matches how I want things to change?*

## Saying what you want, not how to weigh it

Many MGA tools ask users for weights on an objective function. Weights are hard to
interpret: the PoIs have different scales, there is no natural unit, and a small change in
weights can have a large and unpredictable effect. MGA Compass instead lets the user state
preferences directly in terms of the properties they care about.

The navigable quantities are the PoIs plus the total cost, together called the
*augmented properties of interest*. For each of them the user sets:

- **Priority.** The order in which the preferences are satisfied, most important first.
- **Direction.** Whether the value should become *greater*, become *less* or stay *equal*.
- **Magnitude** $\delta_i \ge 0$. For greater or less, the minimum change wanted; for equal,
  how far the value may move from its current value.

For example: "keep the cost within 1%, increase wind capacity by at least 50 GW, then reduce
gas capacity as much as possible".

## The navigation algorithm

All navigation happens over the weights $\alpha$ of the sample points (see
[](preparation.md)): the cost and the PoIs of a candidate solution are linear in $\alpha$,
with the cost treated as one more row of the sample matrix.

**First loop: make the preferences feasible.** In priority order, for each quantity $i$:

1. Find how far it can move in the requested direction, given the constraints already added
   for higher priorities.
2. If the requested change is possible, add a constraint enforcing it. If only part of it is
   possible, reduce $\delta_i$ to what is achievable. If the quantity cannot move in that
   direction at all, change the direction to *equal* and set $\delta_i$ to the smallest
   achievable deviation.

So if the user asks for an increase of 2 units: if 2 is achievable, it is enforced; if at most
1 is, the target becomes an increase of 1; if the best possible is a *decrease* of 1, the
preference becomes "stay within 1 of the current value".

**Second loop: push further.** Again in priority order, move each quantity as far as possible
in its direction (or as close as possible to its current value, for *equal*), then fix it at
that value before moving on.

**Finally**, among all solutions that satisfy the accumulated constraints, take the cheapest.

The adjusted directions and magnitudes are returned to the user. This tells them which
preferences could not be met as stated and by how much, and helps them refine their request.

Each step is a linear program with at most $n + n_z + 1$ variables and $3(n_z + 1)$
constraints: its size depends only on the number of samples and PoIs, never on the size of
the energy system model.

## In the code

- {py:func}`~mga_compass.core.navigate` implements the algorithm. Its `constraints` argument
  is a {py:class}`~mga_compass.schemes.Constraints` table with one row per quantity, in
  priority order, with columns:
  - `direction`: `">="` (greater), `"<="` (less), `"=="` (equal), or `""` to leave the
    quantity free;
  - `value`: the current value;
  - `delta`: the magnitude $\delta_i$.

  It returns the adjusted constraints and the new point.
- {py:func}`~mga_compass.core.describe_changes` compares the requested and the adjusted
  constraints, for showing the user what was changed.
- {py:func}`~mga_compass.core.navigate_outer_approximation` runs the same algorithm over a
  polytope given by halfspaces, built with
  {py:func}`~mga_compass.core.build_outer_approximation`, instead of over the convex hull of
  the samples.

```{admonition} Known limitation: to be improved
:class: warning

Navigation and interpolation build their small linear programs directly from the values in
`points`. If the columns differ by many orders of magnitude, for example cost in euros
(around $10^9$) next to capacities in MW (around $10^4$), these programs can become
numerically ill-conditioned and fail as "infeasible", even though a solution exists. Until
the library rescales the columns itself, keep them at similar magnitudes, for example cost in
million euros and capacities in GW, and pass `default_solver_options("highs")` as the server
does.
```

```{admonition} Known bug
:class: danger

{py:func}`~mga_compass.core.navigate_outer_approximation` does not follow the algorithm above
for quantities with direction `""` (no preference). Its second loop does not skip them, as
{py:func}`~mga_compass.core.navigate` does, so each one is fixed at whatever value the
previous solve happened to return. The final minimum-cost step can then no longer choose it,
and the result may cost more than necessary. If no solve has run yet, the constraint refers to
a solution that does not exist.

Its second loop also adds the constraint fixing a quantity at its value twice for every quantity
with direction `>=`. The duplicate does not change the result, but it adds a redundant row to
the model.
```
