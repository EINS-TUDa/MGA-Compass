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
- The main loop (`drive`, in [`explore.py`](src/mga_surveyor/explore.py)): find the vertex of `O`
  farthest from `I` (the convergence gap) → ask the true region for the closest feasible point to
  it → grow `I` with that point, and if the trial point itself was infeasible, shrink `O` with a
  new cut → repeat until the gap closes below tolerance.
- Two interchangeable **worst-point methods**, both implementing the same `Method` interface
  ([`methods/base.py`](src/mga_surveyor/methods/base.py)):
  - [`CddMethod`](src/mga_surveyor/methods/cdd.py) — maintains `O`'s vertex set via `cdd`'s Double
    Description method: from scratch on every call against a stock `pycddlib` install, or
    incrementally (folding in only the newest halfspaces) when the installed `cdd`/`cdd.gmp` has
    the local incremental-append patch (see [`vendor/cdd/README.md`](vendor/cdd/README.md)). Two
    arithmetic backends (`backend='float'`, the default, or `'gmp'` for exact rational arithmetic,
    immune to float precision limits on degenerate/near-degenerate halfspaces) and, separately, a
    choice of solver for its own inner-hull projection (`projection_backend`, below).
  - [`BilevelMethod`](src/mga_surveyor/methods/bilevel.py) — the original KKT+Big-M MILP
    reformulation (Turán, Moret & Bardow's "ORACLE" method). `linf` only, Gurobi-only.
- Each method supports **3 norms** (`l1`, `l2`, `linf`) for what "farthest/closest" means, backed
  by a `(point, points, norm) -> (closest_point, normal)` convex-hull projection with two
  interchangeable solver backends: [`gurobi_projection.py`](src/mga_surveyor/gurobi_projection.py)
  (all 3 norms, needs a Gurobi license) and
  [`highs_projection.py`](src/mga_surveyor/highs_projection.py) (open-source, `l1`/`linf` only —
  HiGHS has no quadratic-objective support).

None of this — `common.py`, `methods/`, `explore.py`, the two projection backends — depends on
PyPSA or any specific system model. `add_point`/`add_cut` are pure state updates, and the "ask
the true region" oracle (`FurthestPointFn`/`ClosestPointFn`/`OptimalPointFn` in `explore.py`) is a
callback the caller supplies. [`pypsa_ext.py`](src/mga_surveyor/pypsa_ext.py) is that oracle
backed by a real PyPSA network's own solve: `make_furthest_point_fn` and `make_closest_point_fn`
build batched, cacheable versions of the two queries against `n`'s own near-optimal feasible
region (replacing PyPSA's own built-in MGA, not just wrapping it), plus `make_optimality_fn` for
solving and caching a network's own optimum as the precondition both assume. `make_oracle_fns`
builds all three at once, wired straight into `explore.drive`'s parameter order.

[`normalize.py`](src/mga_surveyor/normalize.py) wraps any three `explore.drive` oracle functions
so the polytope a `Method` explores lives in a `[-1, 1]^k`-normalized cube instead of raw dimension
units — useful whenever those units differ wildly in scale (e.g. MW vs MWh), since an unweighted
polytope over such dimensions lets the large-range ones dominate. `explore.drive`/`Method` need no
changes at all for this — it's just oracle functions that happen to speak normalized coordinates.
`generate_mga_files`'s own `normalize` (`True` by default) wires this in for a whole run; everything
written to disk (and `consolidate_mga_results`'s output) stays in real units either
way, since normalization only changes what `Method` itself sees, not what the oracle functions
solve or cache.

`common.linearized_objective_cut` turns a single solved point's own `duals` (already computed,
otherwise unused, whenever `resolve_main_objective=True`) into an extra supporting halfspace of the
true near-optimal region — provably valid, not a heuristic: the achievable-cost-at-x value function
is convex, so the tangent at any solved point is a global underestimator, not just a local
approximation. Free tightening, no new solves. `generate_mga_files`'s own `use_objective_cuts`
(`True` by default; measured ~5x fewer iterations to converge at `tol=2%`, pass `False` to opt out)
uses this for every point the main loop finds — live, and for any
existing result read back on a restart — plus one free extra per slack level from the network's own
optimum (added unconditionally: it has no same-run geometric normal to check it against, and there's
no structural reason for its gradient to align with any axis the way a box-direction point's does) —
composing with `normalize=True` if that's also on (every cut is derived in real units first, then
transformed the same way any other halfspace is). Deliberately *not* derived from any box-direction
point, even though those also have `duals` sitting unused: a box-direction point's own objective cut
is, in practice, *always* exactly parallel to its own box halfspace (a direct consequence of the LP's
structure there, not a coincidence), and nearly-coincident hyperplanes are a real degeneracy risk
regardless of which method processes them — confirmed in practice against the incremental
edge-walking construction this project used before `CddMethod` became the production method, which
reliably exploded into an out-of-memory crash when unfiltered; kept excluded here as a precaution,
not re-validated against `CddMethod`'s own construction. Closest-point and optimum cuts don't share
that structural alignment (their angles to any same-run geometric normal come out scattered, not
clustered near 0°/180° the way box cuts are), so they're never filtered for near-parallelism — an
earlier version did check each closest-point cut against its own same-solve geometric normal first,
but on the real test network that check turned out to discard cuts that still carried real
tightening value, costing extra main-loop iterations for no compensating safety benefit, so it was
dropped; cddlib's own Double Description construction already discards any genuinely
duplicate/redundant cut for free, regardless. One remaining check
before any cut from the main loop (or a restart's existing files) is added: skipped if the point it
came from was already feasible (same test `explore.drive` uses for its own geometric cut).

The [`tests/`](src/mga_surveyor/tests/) subpackage (shipped as part of the installed package, not
just pytest-only) provides `lp_region.py` — a reference oracle implementation against an explicit
small LP region, for testing without a real system model — plus fixed 3D test regions built from
known polyhedra (a truncated icosahedron, a disdyakis triacontahedron) and a synthetic
perturbed-ball generator, so convergence can be checked against an exact, independently-known
vertex/facet count.

## Optional extras

`gurobipy` and `highspy` are on the default install (nearly every real use needs at least one
solver backend), but both are still imported lazily in code — picking one backend never requires
the other to actually be used. `pycddlib` is the one true optional extra, needed only for
`CddMethod` itself:

| Extra | Package | Needed for |
|---|---|---|
| `cdd` | `pycddlib` | `CddMethod` |

`gurobi_projection`/`BilevelMethod`/`mga_surveyor.tests.lp_region` need `gurobipy` specifically
(a Gurobi license); `highs_projection` needs `highspy` (open-source, `l1`/`linf` only — HiGHS has
no quadratic-objective support).

`pycddlib` has no prebuilt wheel for Python 3.14 on Windows as of this writing and fails building
from source there — check availability for your platform/Python before depending on it.
