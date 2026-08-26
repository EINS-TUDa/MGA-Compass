# mga-surveyor

Finds near-optimal solutions (Modeling to Generate Alternatives) for [PyPSA](https://pypsa.org)
energy planning models. Meant to be installed alongside PyPSA and used in place of its built-in
MGA implementation (`pypsa.optimization.mga`).

This is a sibling package to [mga-compass](../mga-compass), which contains the graphical
platform logic (constraint navigation, interpolation, plotting) that runs on top of the points
`mga-surveyor` finds. `mga-surveyor` has no dependency on `mga-compass` and no GUI/server
dependencies of its own.

## The engine

The core of the package is a generic **inner/outer polytope approximation** for near-optimal-
space exploration — a more rigorous alternative to PyPSA's built-in MGA, which only samples a
fixed set of hyperplane directions.

- An **inner approximation** `I`: the convex hull of feasible points found so far.
- An **outer approximation** `O`: the intersection of a growing set of supporting halfspaces
  ("cuts"), starting from a bounding box and shrinking as cuts are added.
- The main loop (`run`, in [`runner.py`](src/mga_surveyor/runner.py)): find the vertex of `O`
  farthest from `I` (the convergence gap) → ask the true region for the closest feasible point to
  it → grow `I` with that point, and if the trial point itself was infeasible, shrink `O` with a
  new cut → repeat until the gap closes below tolerance.
- Three interchangeable **worst-point methods**, all implementing the same `Method` interface
  ([`methods/base.py`](src/mga_surveyor/methods/base.py)):
  - [`VertexEnumMethod`](src/mga_surveyor/methods/vertex_enum.py) — incrementally maintains `O`'s
    vertex set and exploits convexity (the worst point is always a vertex); no MILP needed.
  - [`CddMethod`](src/mga_surveyor/methods/cdd.py) — recomputes `O`'s vertex set from scratch via
    `cdd`'s Double Description method every call; simplest, strongest correctness guarantee, also
    usable as a periodic ground-truth checkpoint (`VertexEnumMethod.resync`).
  - [`BilevelMethod`](src/mga_surveyor/methods/bilevel.py) — the original KKT+Big-M MILP
    reformulation (Turán, Moret & Bardow's "ORACLE" method). `linf` only, Gurobi-only.
- Each method supports **3 norms** (`l1`, `l2`, `linf`) for what "farthest/closest" means, backed
  by a `(point, points, norm) -> (closest_point, normal)` convex-hull projection with two
  interchangeable solver backends: [`gurobi_projection.py`](src/mga_surveyor/gurobi_projection.py)
  (all 3 norms, needs a Gurobi license) and
  [`highs_projection.py`](src/mga_surveyor/highs_projection.py) (open-source, `l1`/`linf` only —
  HiGHS has no quadratic-objective support).

None of this — `common.py`, `methods/`, `runner.py`, the two projection backends — depends on
PyPSA or any specific system model. `add_point`/`add_cut` are pure state updates, and the "ask
the true region" oracle (`FurthestPointFn`/`ClosestFeasiblePointFn` in `runner.py`) is a callback
the caller supplies. [`pypsa_ext.py`](src/mga_surveyor/pypsa_ext.py) is meant to become that
oracle backed by a real PyPSA network's own solve — **still under construction**; for now it
still just wraps PyPSA's own built-in MGA (`optimize_mga_in_direction`/
`optimize_mga_in_multiple_directions`).

The [`tests/`](src/mga_surveyor/tests/) subpackage (shipped as part of the installed package, not
just pytest-only) provides `lp_region.py` — a reference oracle implementation against an explicit
small LP region, for testing without a real system model — plus fixed 3D test regions built from
known polyhedra (a truncated icosahedron, a disdyakis triacontahedron) and a synthetic
perturbed-ball generator, so convergence can be checked against an exact, independently-known
vertex/facet count.

## Optional extras

`gurobipy`, `highspy`, and `pycddlib` are all imported lazily, so installing `mga-surveyor` alone
never forces any of them on you. Install what you need explicitly:

| Extra | Package | Needed for |
|---|---|---|
| `gurobi` | `gurobipy` | `gurobi_projection`, `BilevelMethod`, and `mga_surveyor.tests.lp_region` |
| `highs` | `highspy` | `highs_projection` (`l1`/`linf` only) |
| `cdd` | `pycddlib` | `CddMethod`, `VertexEnumMethod.resync` |

`pycddlib` has no prebuilt wheel for Python 3.14 on Windows as of this writing and fails building
from source there — check availability for your platform/Python before depending on it.
