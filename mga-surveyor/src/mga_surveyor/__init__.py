"""mga_surveyor: reusable inner/outer polytope approximation for
near-optimal-space exploration, with three interchangeable worst-point /
convergence-metric methods:

- `VertexEnumMethod` ("my solution"): replaces the bilevel-MILP worst-point
  search with vertex enumeration of the outer approximation, exploiting the
  fact that a convex function's max over a polytope is attained at a
  vertex -- so no MILP is needed at all. Maintains the outer polytope's
  vertex set incrementally.
- `BilevelMethod` ("bilevel solution"): the original KKT + Big-M MILP
  reformulation. L_infinity only.
- `CddMethod` ("cdd solution"): recomputes the outer polytope's vertex set
  from scratch via cdd's Double Description method on every call. Simpler
  and more independently-verified than the incremental path, at the cost
  of redoing more work each time; also useful as a periodic ground-truth
  checkpoint on top of `VertexEnumMethod` (see `VertexEnumMethod.resync`).

All three share the same `Method` interface (`methods/base.py`) and
accept `norm` (`'l1'`, `'l2'`, or `'linf'`) as a constructor argument
(`BilevelMethod` is restricted to `'linf'`; see its docstring for why).

None of this package depends on any specific test example or system model:
`add_point`/`add_cut` are pure state updates on whatever point/direction
you hand them, and `find_trial_point` only ever looks at the state already
inside the method. Use `runner.bootstrap_box` to seed a method and
`runner.run` to drive it against an external callback (a real system
model, or `mga_surveyor.tests.lp_region` against an explicit LP for
testing), or write your own bootstrap/loop.

`pypsa_ext` is the PyPSA-side adapter supplying the two oracle queries
`mga_surveyor.runner` needs against a real PyPSA network's own solve, so the
engine above can run against an actual system model instead of only a test
example. `make_furthest_point_fn` supplies a batched support-function query
(solved in parallel via `optimize_mga_in_multiple_directions`, not
`runner.FurthestPointFn`'s single-direction contract) for `runner.bootstrap_box`-
style seeding. `make_closest_point_fn` supplies a `runner.ClosestFeasiblePointFn`
for `run`'s main loop instead -- that one can't be batched the same way
(each iteration depends on the previous one's result), and currently only
supports `norm in ('l1', 'linf')`, not `'l2'` (would need a quadratic
objective, Gurobi-only). `pypsa_ext` also still wraps PyPSA's own built-in
MGA (`optimize_mga_in_direction`/`optimize_mga_in_multiple_directions`),
independently of the engine above.
"""
from .common import VALID_NORMS, gaps
from .gurobi_projection import closest_point_on_hull
# Both backend modules name their projection function `closest_point_on_hull`
# (module-qualified, e.g. `gurobi_projection.closest_point_on_hull` vs
# `highs_projection.closest_point_on_hull` -- they're never used side by side
# under one name, so there's no reason for either to carry a backend
# suffix). This flat top-level namespace re-exports both at once, though,
# so the highs one needs an alias here to avoid shadowing the gurobi one.
from .highs_projection import closest_point_on_hull as closest_point_on_hull_highs, VALID_NORMS_HIGHS
from .methods import (
    Method, VertexEnumMethod, BilevelMethod, solve_farthest_outer_point_milp,
    CddMethod, cdd_vertices_of,
)
from .runner import run, bootstrap_box
from .pypsa_ext import (
    make_furthest_point_fn,
    make_closest_point_fn,
    optimize_mga_in_direction,
    optimize_mga_in_multiple_directions,
)

__all__ = [
    "VALID_NORMS", "closest_point_on_hull", "gaps",
    "closest_point_on_hull_highs", "VALID_NORMS_HIGHS",
    "Method", "VertexEnumMethod", "BilevelMethod", "solve_farthest_outer_point_milp",
    "CddMethod", "cdd_vertices_of", "run", "bootstrap_box",
    "make_furthest_point_fn", "make_closest_point_fn",
    "optimize_mga_in_direction", "optimize_mga_in_multiple_directions",
]
