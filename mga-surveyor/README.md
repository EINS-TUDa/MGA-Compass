# mga-surveyor

Finds near-optimal solutions (Modeling to Generate Alternatives) for [PyPSA](https://pypsa.org)
energy planning models. Meant to be installed alongside PyPSA and used in place of its built-in
MGA implementation (`pypsa.optimization.mga`).

Currently `optimize_mga_in_direction` / `optimize_mga_in_multiple_directions` wrap PyPSA's own
hyperplane-direction MGA (see [`pypsa_ext.py`](src/mga_surveyor/pypsa_ext.py) for the exact
behavioral differences), pending a better-suited solution-space-finding algorithm.

This is a sibling package to [mga-compass](../mga-compass), which contains the graphical
platform logic (constraint navigation, interpolation, plotting) that runs on top of the points
`mga-surveyor` finds. `mga-surveyor` has no dependency on `mga-compass` and no GUI/server
dependencies of its own.
