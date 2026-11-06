# Normalization and the tolerance

## Why normalize

PoIs come in different units and scales: in the PyPSA-Eur experiment, solar capacity ranged
over about 1,100 GW while hydrogen storage ranged over 164 TWh. Measured in raw units, the
distance between the approximations would be dominated by the PoIs with the largest ranges,
and the search would chase errors in those while neglecting the rest.

So each PoI is mapped affinely onto $[-1, 1]$ using its achievable range, found in the
initialization step. With $z_{lb}$ and $z_{ub}$ the bounds of the initial box and
$S = \operatorname{diag}(z_{ub} - z_{lb})$,

$$
\tilde z = 2 S^{-1} (z - z_{lb}) - \mathbf{1},
\qquad
z = \tfrac{1}{2} S \tilde z + \bar z,
\quad \bar z = \tfrac{1}{2}(z_{lb} + z_{ub}).
$$

The map is affine and invertible, so it maps polytopes to polytopes and the algorithm runs
unchanged in the normalized coordinates.

## What changes, and what doesn't

- The **model is still solved in real units.** The nearest-point problem receives the trial
  point mapped back with the inverse map, and measures the residual with the weighted norm
  $\|2 S^{-1} r\|$, which equals the distance in normalized coordinates.
- **Cuts are derived in real units, then transformed.** A halfspace $a^\top z \le \gamma$
  becomes

  $$
  \left(\tfrac{1}{2} S a\right)^\top \tilde z \le \gamma - a^\top \bar z,
  $$

  so each component of the normal is scaled by half its PoI's range, and the right-hand side
  is shifted by the value of $a^\top z$ at the centre of the box.
- **Everything written to disk stays in real units.** Only what the `Method` sees is
  normalized.

## What this means for the tolerance

Normalization also gives the tolerance a natural unit. In the normalized coordinates every
PoI spans the same width, so a single tolerance means the same thing for every PoI: a
percentage of its achievable range, whatever its physical unit. That is why, with
normalization on, `tol` is given as a percentage; see [](../reference/surveyor-parameters.md#tolerance)
for exactly what it guarantees under each norm.

## In the code

- {py:func}`~mga_surveyor.normalize.compute_affine_map` builds the map from the box points;
  it fails if a PoI has no range at the given slack.
- {py:func}`~mga_surveyor.normalize.normalize_oracle_fns` wraps the three oracle functions so
  that {py:func}`~mga_surveyor.explore.drive` sees only normalized coordinates.
